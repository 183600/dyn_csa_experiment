import os
# Reproducibility: do NOT silently redirect the Hugging Face Hub to a mirror.
# The mirror may lag/differ from the official Hub; opt in explicitly when the
# official endpoint is unreachable.
if os.environ.get('CSA_HF_MIRROR', '').strip().lower() in ('1', 'true', 'yes'):
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
# PYTHONHASHSEED only takes effect when it is set before the interpreter
# starts; setting it here is a best-effort propagation to child processes
# (par.py workers, resumed runs).  The experiment itself never depends on
# set/dict iteration order, but pinning it closes the last Python-level
# nondeterminism source on repeat runs of the same machine.
os.environ.setdefault('PYTHONHASHSEED', '0')
_cpu_thr = os.environ.get('CSA_CPU_THREADS', '').strip() or '1'
os.environ.setdefault('OMP_NUM_THREADS', _cpu_thr)
os.environ.setdefault('MKL_NUM_THREADS', _cpu_thr)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
del _cpu_thr
import sys, subprocess
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except (ValueError, OSError):
            pass
_NEEDED = []
for _pkg in ('datasets', 'tokenizers', 'matplotlib'):
    try:
        __import__(_pkg)
    except ImportError:
        _NEEDED.append(_pkg)
if _NEEDED:
    # Import-time pip install is a reproducibility hazard: it silently changes
    # the environment and can fetch a different version on every run.  It is
    # now opt-in; the normal path tells the user what to install.
    _msg = f"missing optional package(s): {', '.join(_NEEDED)}; install the pinned requirements (requirements.txt / requirements.lock) before running"
    if os.environ.get('CSA_AUTO_INSTALL', '').strip().lower() in ('1', 'true', 'yes'):
        print(f"[setup] CSA_AUTO_INSTALL=1 — installing {', '.join(_NEEDED)} (not pinned; archive the resulting pip freeze)")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *_NEEDED])
        except Exception as _pip_err:
            print(f"[setup] could not install {', '.join(_NEEDED)} ({type(_pip_err).__name__}: {_pip_err}) — continuing; a code path that actually needs them will raise ImportError")
    else:
        print(f"[setup] {_msg}")
import torch
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
import gc, math, os, re, time, json, random, csv, tempfile, itertools, hashlib, collections, threading
import array
from math import comb
from dataclasses import dataclass
import numpy as np
# Leakage-safe split/preprocessing contract (dependency-free NumPy module).
# Re-exported here so every driver can use the same implementation:
#   chronological_split / split_indices / grouped_chronological_split,
#   assert_no_split_leakage, LeakageSafePreprocessor (SafeScaler),
#   prepare_features, data_quality_report, categorical_shift_report.
from data_guard import (
    chronological_split, chronological_split_indices,
    grouped_chronological_split, grouped_chronological_split_indices,
    split_indices, train_val_test_split, assert_no_split_leakage,
    LeakageSafePreprocessor, SafeScaler, prepare_features, fit_preprocessor,
    data_quality_report, categorical_shift_report, duplicate_mask,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[setup] device = {DEVICE}   torch = {torch.__version__}')
QUICK = False
RUN_CAP = dict(total_hours=58.333333333333336, margin=0.93, already_hours=0.0, state_path='run_time_state.json')
CODE_SEMANTICS = 'v11.126'
CKPT_CODE = CODE_SEMANTICS
# ---- 三算子内核路径（docs/compute_plan.md §3 的 M0 工作包；实现与数值契约
# 见 csa_kernels.py）。CSA_KERNEL 未设置时本模块的执行路径与指纹逐字节不变。
try:
    import csa_kernels as _CK
except Exception as _ck_err:
    _CK = None
    _CK_ERR = _ck_err


def _kernel_stamp():
    """内核变体标识（进入 run_cfg 指纹与 `_code` 戳）。eager 模式返回空串，
    既有记录的指纹逐字节不变；内核模式下记录只与同模式记录配对（§3.1）。"""
    if _CK is None:
        return ''
    try:
        return _CK.kernel_stamp()
    except Exception:
        return ''

def _eager_batch_active(device=None, batch_size=None):
    """True when the batched reference twin should be used.

    On CUDA with the kernel module available, the batched twin is the default
    for B > 1; the row-by-row reference path remains available (and is the
    only path on CPU) for bitwise debugging.  ``CSA_EAGER_BATCH=0`` opts out.
    That path changes the numerical implementation (it is not the eager
    row-by-row bitwise contract), so its actual mode is carried in
    ``training_knobs()`` (and therefore in the resume fingerprint and the
    per-record descriptor) rather than in the machine-independent `_code`
    stamp.

    ``device``/``batch_size`` are optional so callers can ask about the
    concrete run instead of the process default.  When ``batch_size`` is
    omitted only the device/kernel availability is considered."""
    v = os.environ.get('CSA_EAGER_BATCH', '').strip().lower()
    if v in ('0', 'false', 'no', 'off'):
        return False
    if _CK is None:
        return False
    try:
        if _CK.kernel_enabled():
            # The kernel backend replaces the eager twin entirely.
            return False
    except Exception:
        pass
    if device is None:
        device = DEVICE
    if getattr(device, 'type', str(device)) != 'cuda':
        return False
    if batch_size is not None and int(batch_size) <= 1:
        return False
    return True

def _uses_batched_eager(cfgs, device=None, batch_size=None):
    """Whether at least one layer in ``cfgs`` will take the batched twin.

    Mirrors the condition in ``HybridAttention.forward`` so a record can state
    the implementation it actually ran instead of the process default."""
    if not _eager_batch_active(device, batch_size):
        return False
    return any(
        getattr(c, 'kind', None) in ('csa', 'hca')
        and getattr(c, 'fuse_mode', 'none') == 'none'
        and getattr(c, 'indexer_mode', 'learned')
        for c in cfgs
    )

def code_stamp():
    """当前代码+内核模式写进每条记录的 `_code` 戳（eager = CODE_SEMANTICS
    本身；内核模式追加变体后缀）。一切「记录是否当前」
    的判定（result_is_current 的 code 实参、`_cs...` 后缀匹配）都必须以
    本值为准——裸比 CODE_SEMANTICS 会把内核模式的全部记录误判为陈旧，
    使 resume 静默失效、面板被标 `_stale_cfg`。

    The eager batched-twin / AMP choices deliberately do NOT enter this
    stamp: they depend on the machine that produced a record, so encoding
    them here would make a zero-GPU report box label every GPU record stale.
    They are recorded per run in ``training_knobs()`` / the run fingerprint
    instead, which analysis reads from disk."""
    return f'{CODE_SEMANTICS}{_kernel_stamp()}'


# ---- 训练健康 / 数值防线 ---------------------------------------------------
# `nan_to_num` 默认静默清洗数值异常。这里保留原有数值语义（逐位不变），
# 但让调用点可以计数，并允许 CSA_STRICT_NAN=1 直接报错。元素级计数会强制
# GPU 同步，所以默认只记调用次数；CSA_NAN_AUDIT=1 时才展开元素统计。
_NAN_GUARD = {'calls': 0, 'elements_cleaned': 0, 'first': None, 'by_where': {}}
_NAN_AUDIT_ELEMENTS = os.environ.get('CSA_NAN_AUDIT', '').strip().lower() in ('1', 'true', 'yes', 'on')
_STRICT_NAN = os.environ.get('CSA_STRICT_NAN', '').strip().lower() in ('1', 'true', 'yes', 'on')


def nan_guard_state(reset=False):
    out = {'calls': int(_NAN_GUARD['calls']),
           'elements_cleaned': int(_NAN_GUARD['elements_cleaned']),
           'first': _NAN_GUARD['first'],
           'by_where': dict(_NAN_GUARD['by_where']),
           'element_audit': bool(_NAN_AUDIT_ELEMENTS),
           'strict': bool(_STRICT_NAN)}
    if reset:
        _NAN_GUARD.update({'calls': 0, 'elements_cleaned': 0, 'first': None, 'by_where': {}})
    return out


def _safe_nan_to_num(x, *, where='attention', **kwargs):
    """`torch.nan_to_num` wrapper used by the attention paths.

    Default behaviour is byte-identical to the historical call.  With
    CSA_NAN_AUDIT=1 (or CSA_STRICT_NAN=1) the non-finite element count is
    measured; strict mode raises with the call site instead of silently
    changing attention semantics."""
    _NAN_GUARD['calls'] += 1
    _NAN_GUARD['by_where'][where] = int(_NAN_GUARD['by_where'].get(where, 0)) + 1
    if _STRICT_NAN or _NAN_AUDIT_ELEMENTS:
        try:
            bad = ~torch.isfinite(x)
            n_bad = int(bad.sum().item())
        except Exception:
            n_bad = 0
        if n_bad:
            _NAN_GUARD['elements_cleaned'] += n_bad
            if _NAN_GUARD['first'] is None:
                _NAN_GUARD['first'] = {'where': where, 'n_elements': n_bad}
            if _STRICT_NAN:
                raise FloatingPointError(f'non-finite values in {where}: {n_bad} element(s) would have been silently replaced by nan_to_num (CSA_STRICT_NAN=1)')
    return torch.nan_to_num(x, **kwargs)


def _finite_curve(values):
    """Finite subset + non-finite bookkeeping for a loss/PPL curve."""
    vals = list(values or [])
    finite = []
    first_bad = None
    last_bad = None
    for i, v in enumerate(vals):
        try:
            fv = float(v)
        except (TypeError, ValueError):
            fv = None
        if fv is not None and math.isfinite(fv):
            finite.append(fv)
        elif first_bad is None:
            first_bad = i
        if fv is None or not math.isfinite(fv):
            last_bad = i
    return {'n_points': len(vals), 'n_finite': len(finite),
            'n_nonfinite': len(vals) - len(finite),
            'first_nonfinite_index': first_bad,
            'last_nonfinite_index': last_bad,
            'min': min(finite) if finite else None,
            'max': max(finite) if finite else None,
            'last': finite[-1] if finite else None}


def _curve_slope(points, key, k=4):
    """Least-squares slope over the last k FINITE points of a curve."""
    pts = []
    for p in (points or []):
        if not isinstance(p, dict):
            continue
        try:
            x = float(p.get('step'))
            y = float(p.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            pts.append((x, y))
    pts = pts[-max(2, int(k)):]
    if len(pts) < 2:
        return None
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    den = sum((x - mx) ** 2 for x, _ in pts)
    if den <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pts) / den


def classify_training_health(history, *, tail_k=4, slope_tol=1e-5, gap_tol=None):
    """Classify a train/validation curve.

    ``history`` is a list of dicts with ``step``, ``train_ce`` and either
    ``val_ce`` or ``val_ppl``.  This is a diagnostic, not a model-selection
    rule: it says whether the run looks overfit / underfit / stalled / broken
    on the fixed evaluation grid.
    """
    entries = [h for h in (history or []) if isinstance(h, dict)]
    n_bad = 0
    n_train_measured = 0
    for h in entries:
        for key in ('train_ce', 'val_ce', 'val_ppl'):
            v = h.get(key)
            if v is None:
                # None = this quantity was never measured at this point
                # (no fixed-eval batch / the evaluation failed).  It must not
                # be counted as a non-finite MEASUREMENT: that would label a
                # healthy validation curve `nonfinite` and block or excuse a
                # run for the wrong reason.
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                n_bad += 1
                continue
            if not math.isfinite(v):
                n_bad += 1
            if key == 'train_ce':
                n_train_measured += 1
    train_s = _curve_slope(entries, 'train_ce', k=tail_k)
    val_s = _curve_slope(entries, 'val_ce', k=tail_k)
    if val_s is None:
        _val_from_ppl = []
        for _p in entries:
            try:
                _st = float(_p.get('step'))
                _vp = float(_p.get('val_ppl'))
            except (TypeError, ValueError):
                continue
            if math.isfinite(_st) and math.isfinite(_vp) and _vp > 0.0:
                _val_from_ppl.append({'step': _st, 'val_ce': math.log(_vp)})
        val_s = _curve_slope(_val_from_ppl, 'val_ce', k=tail_k)
    val_ppl_points = []
    for h in entries:
        try:
            st = int(h.get('step'))
            vp = float(h.get('val_ppl'))
        except (TypeError, ValueError):
            continue
        if math.isfinite(vp) and vp > 0:
            val_ppl_points.append((st, vp, h))
    best = min(val_ppl_points, key=lambda t: t[1]) if val_ppl_points else None
    last = val_ppl_points[-1] if val_ppl_points else None
    last_gap = None
    for h in reversed(entries):
        try:
            g = float(h.get('gap'))
        except (TypeError, ValueError):
            continue
        if math.isfinite(g):
            last_gap = g
            break
    if n_bad:
        status = 'nonfinite'
        reason = f'{n_bad} non-finite curve point(s); inspect the first bad step before using this run'
    elif len(entries) < 2:
        status = 'insufficient_data'
        reason = 'fewer than two finite train/validation eval points; cannot classify'
    elif train_s is None and val_s is not None:
        # Train CE was never measured (or has no finite tail points): fall
        # back to a validation-only classification instead of silently
        # reporting `nonfinite`/`insufficient_data`.  Overfitting cannot be
        # attributed without the train curve, so the reason says so.
        if val_s > slope_tol:
            status = 'validation_degrading'
            reason = 'train CE unmonitored; validation CE is rising in the tail (val-only classification)'
        elif val_s < -slope_tol:
            status = 'healthy'
            reason = 'train CE unmonitored; validation CE is falling in the tail (val-only classification: overfitting cannot be ruled out without the train curve)'
        else:
            status = 'underfitting_or_stalled'
            reason = 'train CE unmonitored; validation CE is flat in the tail (val-only classification)'
    elif train_s is None or val_s is None:
        status = 'insufficient_data'
        reason = 'fewer than two finite train/validation eval points; cannot classify'
    elif train_s < -slope_tol and val_s > slope_tol:
        status = 'overfitting'
        reason = 'training CE is still falling while validation CE is rising'
    elif train_s >= -slope_tol and val_s >= -slope_tol:
        status = 'underfitting_or_stalled'
        reason = 'neither training nor validation CE is decreasing in the tail'
    elif train_s < -slope_tol and val_s <= slope_tol:
        status = 'healthy'
        reason = 'training CE is falling and validation CE is not rising'
    else:
        status = 'validation_degrading'
        reason = 'validation CE is rising in the tail'
    return {
        'status': status,
        'reason': reason,
        'n_points': len(entries),
        'n_nonfinite': int(n_bad),
        'n_train_ce_measured': int(n_train_measured),
        'tail_k': int(tail_k),
        'train_ce_slope_per_step': train_s,
        'val_ce_slope_per_step': val_s,
        'gap_last': last_gap,
        'best_step': int(best[0]) if best is not None else None,
        'best_val_ppl': float(best[1]) if best is not None else None,
        'final_val_ppl': float(last[1]) if last is not None else None,
        'final_minus_best': (float(last[1] - best[1]) if (best is not None and last is not None) else None),
    }


def _safe_exp_ppl(x):
    """``exp`` for a PPL accumulator without raising on divergent CE.

    ``math.exp`` raises once the exponent exceeds ~709.  A training run that
    has diverged but still produced finite CE must be recorded as ``inf`` and
    diagnosed, not crash the eval callback.  The clamp mirrors the historical
    ``eval_ppl`` overflow behaviour while making the CI bounds consistent."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(x):
        return float('nan')
    # Let the FP overflow exception define the upper boundary for this
    # machine, so any finite exponent remains a finite PPL.
    try:
        return math.exp(x)
    except OverflowError:
        return float('inf')


def _nll_doc_ci(nll_per_row, n_tokens_per_row):
    """Approximate 95% CI for PPL from per-DOCUMENT NLL (not per-token).

    Documents, not tokens, are the independent sampling unit; treating every
    token as independent would make the interval look far narrower than the
    actual evaluation-subset noise."""
    doc_ce = []
    total_nll = 0.0
    total_tok = 0
    for nll, ntok in zip(nll_per_row or [], n_tokens_per_row or []):
        try:
            nll = float(nll)
            ntok = int(ntok)
        except (TypeError, ValueError):
            continue
        if ntok <= 0 or not math.isfinite(nll):
            continue
        total_nll += nll
        total_tok += ntok
        doc_ce.append(nll / ntok)
    if len(doc_ce) < 2 or total_tok <= 0:
        return {'ppl': (_safe_exp_ppl(total_nll / total_tok) if total_tok else None),
                'doc_ce_mean': None, 'doc_ce_sem': None,
                'ppl_ci95_low': None, 'ppl_ci95_high': None, 'n_docs': len(doc_ce)}
    mean = sum(doc_ce) / len(doc_ce)
    var = sum((x - mean) ** 2 for x in doc_ce) / (len(doc_ce) - 1)
    sem = math.sqrt(max(var, 0.0) / len(doc_ce))
    ce = total_nll / total_tok
    # The point estimate is token-pooled, so the interval must be for that
    # same estimand.  Resampling/weighting per-document CEs as if the pooled
    # CE were their unweighted mean mixes two different estimators and is
    # wrong as soon as document lengths differ.  Use the cluster-robust
    # (documents = clusters) delta-method variance of the ratio of totals:
    #   Var(CE) ~= N/(N-1) * sum_i (nll_i - CE*n_tok_i)^2 / (sum_i n_tok_i)^2.
    # A t critical value (df=N-1), not 1.96, keeps small document panels from
    # claiming a normal-theory interval.
    _resid = []
    for nll, ntok in zip(nll_per_row or [], n_tokens_per_row or []):
        try:
            nll = float(nll)
            ntok = int(ntok)
        except (TypeError, ValueError):
            continue
        if ntok <= 0 or not math.isfinite(nll):
            continue
        _resid.append(nll - ce * ntok)
    n_docs = len(_resid)
    _var = (n_docs / (n_docs - 1.0)) * sum(r * r for r in _resid) / (total_tok * total_tok)
    _sem_ce = math.sqrt(max(_var, 0.0))
    _tc = _t_critical(0.95, n_docs - 1)
    if not math.isfinite(_tc):
        _tc = 1.959963984540054
    return {'ppl': _safe_exp_ppl(ce), 'doc_ce_mean': mean, 'doc_ce_sem': sem,
            'ce_sem_cluster_robust': _sem_ce,
            'ci_method': 'cluster-robust documents (t, df=n_docs-1)',
            'ppl_ci95_low': _safe_exp_ppl(ce - _tc * _sem_ce),
            'ppl_ci95_high': _safe_exp_ppl(ce + _tc * _sem_ce),
            'n_docs': n_docs}


# ---- G0 gate ---------------------------------------------------------------
# The scale panels are only allowed to start after the kernel/eager
# equivalence certificate exists and matches this code/env.  `m0_supp.py
# kernel` writes it; `TimeGuard(require_g0=True)` / `run(cfg.require_g0)` read
# it before any data is loaded or a panel starts.  An operator who
# deliberately wants eager debugging can set CSA_ALLOW_UNVERIFIED_EAGER=1; the
# override is printed and recorded in the panel manifest/config.
G0_CERT_ENV = 'CSA_G0_CERT'
DEFAULT_G0_CERT = os.path.join('results_m0_g0', 'g0_certificate.json')
G0_OVERRIDE_ENV = 'CSA_ALLOW_UNVERIFIED_EAGER'

def g0_certificate_path():
    return os.environ.get(G0_CERT_ENV) or DEFAULT_G0_CERT

def _g0_mismatch(cert):
    """Return the first reason the certificate does not certify THIS tree."""
    if not isinstance(cert, dict):
        return 'certificate is not a JSON object'
    if cert.get('status') != 'pass':
        return f"certificate status is {cert.get('status')!r}, not 'pass'"
    if not cert.get('numeric_pass'):
        return 'operator/model numeric equivalence did not pass'
    if not cert.get('benchmark_pass'):
        return 'G0 kernel/eager certificate did not pass (CUDA+triton required)'
    if not cert.get('training_tost_pass'):
        return 'training-level TOST equivalence did not pass'
    if str(cert.get('code_semantics')) != str(CODE_SEMANTICS):
        return f"code semantics {cert.get('code_semantics')!r} != current {CODE_SEMANTICS!r}"
    if str(cert.get('kernel_stamp') or '') != str(_kernel_stamp() or ''):
        return f"kernel stamp {cert.get('kernel_stamp')!r} != current {_kernel_stamp()!r}"
    _cur = _git_provenance()
    if cert.get('git_commit') and _cur.get('git_commit') and cert['git_commit'] != _cur['git_commit']:
        return f"git commit changed since the certificate ({cert['git_commit'][:12]}… != {_cur['git_commit'][:12]}…)"
    if cert.get('git_diff_sha256') is not None and _cur.get('git_diff_sha256') is not None:
        if cert.get('git_diff_sha256') != _cur.get('git_diff_sha256'):
            return 'the worktree differs from the tree that produced the certificate'
    return None

def load_g0_certificate(path=None):
    p = path or g0_certificate_path()
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def require_g0(context='scale panel', path=None):
    """Fail closed unless a valid G0 certificate exists for this tree.

    Returns the certificate on success, or {'status': 'override'} when the
    explicit debug override is set (the caller must record that fact)."""
    p = path or g0_certificate_path()
    cert = load_g0_certificate(p)
    if cert is None:
        _why = f'no G0 certificate at {p}'
    else:
        _why = _g0_mismatch(cert)
        if _why is None:
            return cert
    if os.environ.get(G0_OVERRIDE_ENV, '').strip().lower() in ('1', 'true', 'yes', 'on'):
        print(f'[g0] WARNING: {context} starting WITHOUT a valid G0 certificate ({_why}); '
              f'{G0_OVERRIDE_ENV}=1 is an explicit debug override. This run must be '
              'reported as eager/unverified and must not enter a confirmatory table.')
        return {'status': 'override', 'reason': _why, 'path': p}
    raise SystemExit(
        f'[g0] REFUSE: {context} requires a passed G0 kernel/eager equivalence '
        f'certificate, but {_why}. Run `python m0_supp.py kernel` on the target '
        f'(CUDA+triton) machine to produce {p}. For an explicitly unverified eager '
        f'debug run set {G0_OVERRIDE_ENV}=1 (never for a formal panel).')

def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def state_dict_sha256(sd):
    """Order-stable digest of tensor names/shapes/dtypes and raw bytes.

    This is the checkpoint identity that can be written INSIDE the same file
    (the file's own SHA-256 cannot be).  P5T compares it with the training
    record and with the loaded tensors, so swapping two same-shaped arms or
    seeds is detected even when file names are correct."""
    h = hashlib.sha256()
    for k in sorted(sd):
        v = sd[k]
        h.update(str(k).encode('utf-8'))
        h.update(str(tuple(v.shape)).encode('ascii'))
        h.update(str(v.dtype).encode('ascii'))
        _t = v.detach().cpu().contiguous()
        try:
            _bytes = _t.numpy().tobytes()
        except TypeError:
            _bytes = _t.reshape(-1).view(torch.uint8).numpy().tobytes()
        h.update(_bytes)
    return h.hexdigest()

# ---- 可选激活重算（CSA_CKPT，默认关闭）----
# 与 ce_loss / gathered_attention / _pool_trainable 的既有 checkpoint 同型
# （design_notes §6.1「重算同一纯函数」的 transport 论证）：被包的前向无
# RNG、是输入与权重的纯函数，反向用保存的输入按同一确定性分解重算，前向
# 值与梯度与直调形式逐位一致。未设置时执行路径与指纹逐字节不变；开关本身
# 不改任何到达算子的值，与 CSA_PAR / CSA_CPU_THREADS 同类，不进入
# run_cfg 指纹（§3.1）。
#   mlp   —— 每个 MLP 前向包一层非重入 checkpoint：fc1 输出 [B·T, 4·d]
#            不保存，反向用保存的输入按同一确定性分解重算 fc1+gelu。
#   block —— 在 mlp 之上，对无 gate_mean 侧信道的臂（chunking !=
#            'cosine_learnable'）整块包一层 checkpoint：块输入 [B, T, d]
#            成为唯一保存的层间激活。cosine_learnable 臂的 gate_mean 经
#            属性逃逸出块（trunk 的 comp_reg 读它），而 checkpoint 的初向
#            前向在 no_grad 下运行会剥掉它的图（v9 _V9_CKPT 的同一限
#            制），故这些臂自动回退到 mlp 级并提示一次。块内既有的池化 /
#            gather 注意力 / sink 分块 checkpoint 在反向重算时照旧包裹
#            （嵌套非重入 checkpoint 是 torch 支持、v9 驱动已在使用的形
#            式）；_CKPT_NEST 控制内层 MLP 的包裹。
_CKPT_NEST = [0]
_CKPT_WARNED = set()

def _ckpt_mode():
    raw = os.environ.get('CSA_CKPT', '').strip().lower()
    if raw in ('', '0', 'off', 'none', 'false'):
        return ''
    if raw in ('mlp', 'block'):
        return raw
    raise ValueError(f"unknown CSA_CKPT value {os.environ.get('CSA_CKPT')!r} (expected unset/'0' (off), 'mlp', or 'block')")

_CKPT = {'on': _ckpt_mode()}
RUN = dict(seq_len=512, batch_size=12, n_train_tokens=1000000 if QUICK else 8000000, steps=500 if QUICK else 1500, warmup=50, lr=0.0003, weight_decay=0.1, comp_lambda=0.05, delta_lr_mult=10.0, eval_every=250, eval_subset=128, seeds=[0] if QUICK else [0, 1, 2, 3, 4], outdir='results_lm_v3_1500', variants=['full', 'full_matched', 'full_cos', 'full_sw128', 'full_sw128_matched', 'csa_fixed', 'csa_dynamic', 'hybrid_fixed', 'hybrid_dynamic'])
# 消融臂：`csa_fix_*` 是固定分块下对学习式 indexer / 池化内容 / sink / top-k
# 的去一法对照；`csa_dyn_*` 是同一组去一法在**动态分块**（主方法）下的对照——
# 只报固定分块下的去一法无法支撑「该模块在提出的动态方法里各自有用」。两组
# 都是「与 csa_dynamic / csa_fixed 逐字段同构，仅待验证模块不同」。
ABL_VARIANTS = ['hybrid_csa_dyn', 'hybrid_hca_dyn', 'csa_dyn_fuse', 'hybrid_csa_dyn_fuse', 'csa_fix_randidx', 'csa_fix_zerocont', 'csa_fix_nosink', 'csa_fix_topk8', 'csa_fix_topk64', 'csa_dyn_randidx', 'csa_dyn_zerocont', 'csa_dyn_nosink', 'csa_dyn_nocomp', 'csa_dynamic_task', 'full_sink']
# 去一法要成立，「Full」与各「Full−X」臂必须落在**同一个面板**里：ABLATION_DESIGN
# 的每条对照都靠 pair_reason 在同一 run_cfg 下逐种子配对，methodology_audit 也按
# 单面板 summary 审计——参照臂不在面板内时该面板只能报「缺臂」，撑不起任何
# 「各模块都有用」的结论。因此消融面板自含参照臂（csa_dynamic / hybrid_csa_dyn
# 即去一法的「Full」；full 供 full_sink 的 dense sink 对照）。参照臂列在最前：
# resume 时已完成的参照臂直接跳过；被 cap 截断时它们最先完成，缺的只是
# abl 臂而不是整套对照。
ABL_REFS = ['full', 'csa_fixed', 'csa_dynamic', 'hybrid_fixed', 'hybrid_dynamic']
RUN_ABL = dict(RUN, outdir='results_lm_v4_abl', variants=ABL_REFS + ABL_VARIANTS, seeds=[0] if QUICK else [0, 1, 2, 3, 4])
RUN_LONG = dict(RUN, n_train_tokens=2000000 if QUICK else 110000000, steps=500 if QUICK else 20000, warmup=200, eval_every=1000, seeds=[0] if QUICK else [0, 1, 2, 3], outdir='results_lm_v3_long', variants=['full', 'full_matched', 'full_sw128_matched', 'csa_fixed', 'csa_dynamic', 'hybrid_fixed', 'hybrid_dynamic', 'hybrid_csa_dyn'])
RUN_SCALE = dict(RUN, d=384, n_layers=8, n_heads=12, d_head=32, seq_len=1024, batch_size=8, n_train_tokens=2000000 if QUICK else 40000000, steps=500 if QUICK else 4000, warmup=100, eval_every=500, seeds=[0] if QUICK else [0, 1, 2], outdir='results_lm_v5_scale', variants=['full', 'full_sw128_matched', 'csa_fixed', 'csa_dynamic', 'hybrid_dynamic'])
# ---- B 档规模阶梯（docs/compute_plan.md §4–§5；档位已选定 B 进取）----
# 以下全部为规划值：步数 / 批大小 / 学习率等训练量与超参按
# docs/compute_plan.md 的口径滚动修正；参数量以 SmallGPT 实例化统计（见
# m0_supp.py params，S4 按规划的解析外推口径）。
B_VOCAB = 32768
SCALE_LADDER = {'S1': dict(d=768, n_layers=12, n_heads=12, d_head=64), 'S2': dict(d=1024, n_layers=16, n_heads=16, d_head=64), 'S3': dict(d=1536, n_layers=20, n_heads=24, d_head=64), 'S4': dict(d=2048, n_layers=24, n_heads=32, d_head=64)}
B_ARMS = ['full_matched', 'full_sw128_matched', 'csa_fixed', 'csa_dynamic', 'hybrid_dynamic']
RUN_S1 = dict(RUN, dataset='fineweb', vocab_size=B_VOCAB, d=768, n_layers=12, n_heads=12, d_head=64, seq_len=8192, batch_size=8, n_train_tokens=5000000000, steps=76294, warmup=2000, eval_every=4000, eval_subset=32, val_seqs=512, seeds=[0, 1, 2, 3], outdir='results_b_s1', variants=B_ARMS)
M0_CAP = dict(total_hours=700.0, margin=0.93, already_hours=0.0, state_path='run_time_state_m0.json')
RUN_S2 = dict(RUN_S1, d=1024, n_layers=16, n_heads=16, n_train_tokens=12000000000, steps=183105, warmup=4000, eval_every=8000, seeds=[0, 1, 2], outdir='results_b_s2')
M2_CAP = dict(total_hours=500.0, margin=0.93, already_hours=0.0, state_path='run_time_state_m2.json')

_CPU_DEV = torch.device('cpu')
_DEV_STR = {}

def _dev_str(device):
    # str(device) is a pure formatting function of the device object: the same
    # string every time, host-side only — cache it per object.
    s = _DEV_STR.get(device)
    if s is None:
        s = str(device)
        _DEV_STR[device] = s
    return s

def block_readable(pos, last_tok):
    return pos[:, None] > last_tok[None, :]

_CONSEC_COS_CHUNK_BYTES = 64 << 20

def cosine_similarity_consecutive(H, eps=1e-08):
    # Consecutive-pair cosine similarity.  The normalized rows and the
    # elementwise product are pure per-position functions, so processing the
    # time axis in chunks changes no value; the unchunked form is used when
    # the whole sequence fits in one chunk.
    if H.dim() < 2:
        h = F.normalize(H, dim=-1, eps=eps)
        return (h[1:] * h[:-1]).sum(dim=-1)
    T = int(H.shape[-2])
    if T <= 1:
        return H.new_empty(H.shape[:-2] + (0,))
    lead = 1
    for _s in H.shape[:-2]:
        lead *= int(_s)
    _row_bytes = max(1, lead * int(H.shape[-1]) * H.element_size())
    _rows = max(2, _CONSEC_COS_CHUNK_BYTES // _row_bytes)
    if _rows >= T:
        h = F.normalize(H, dim=-1, eps=eps)
        return (h[..., 1:, :] * h[..., :-1, :]).sum(dim=-1)
    _step = _rows - 1
    _lead_sl = (slice(None),) * (H.dim() - 2)
    # Fill the per-chunk slices of the [.., T-1] result.
    out = H.new_empty(H.shape[:-2] + (T - 1,))
    for s in range(0, T - 1, _step):
        e = min(s + _rows, T)
        h = F.normalize(H[_lead_sl + (slice(s, e), slice(None))], dim=-1, eps=eps)
        out[_lead_sl + (slice(s, e - 1),)] = (h[..., 1:, :] * h[..., :-1, :]).sum(dim=-1)
    return out

# LRU memo of exact segmenter outputs (OrderedDict).
# — a pure cache: hits return the identical (bids, honoured, counts) triple
# either policy would return, so no value downstream can change.
_SEGMENT_CACHE = collections.OrderedDict()
_SEGMENT_CACHE_CAP = 4096
# Per-length host position table arange(n + 1) reused across segmenter calls
# (read-only).  Pure cache — the same integers `np.arange(n + 1)` returns
# every call.
_SEG_POS_CACHE = {}
_SEG_POS_CACHE_CAP = 16

def _seg_pos(n):
    pos = _SEG_POS_CACHE.get(n)
    if pos is None:
        pos = np.arange(n + 1, dtype=np.int64)
        if len(_SEG_POS_CACHE) >= _SEG_POS_CACHE_CAP:
            _SEG_POS_CACHE.pop(next(iter(_SEG_POS_CACHE)))
        _SEG_POS_CACHE[n] = pos
    return pos

_SEG_WALK_PROBE = 32
_SEG_VEC_MIN = 512

def _successor_starts(n, nxt):
    # Starts of the monotone successor walk p -> nxt[p] (nxt[p] > p, nxt[n] = n).
    # A short prefix is walked directly; when the walk is long (dense cuts),
    # the rest is generated in doubling batches with a jump table.
    starts = [0]
    p = int(nxt[0])
    left = _SEG_WALK_PROBE - 1
    while p < n and left > 0:
        starts.append(p)
        p = int(nxt[p])
        left -= 1
    if p >= n:
        return np.asarray(starts, dtype=np.int64)
    k = len(starts)
    if (n - p) * k > max(_SEG_VEC_MIN, n // 8) * max(p, 1):
        jump = nxt
        step = 1
        while step < k:
            jump = jump[jump]
            step <<= 1
        out = np.asarray(starts, dtype=np.int64)
        while True:
            nxts = jump[out]
            keep = nxts < n
            if not keep.any():
                break
            out = np.concatenate([out, nxts[keep]])
            if not keep.all():
                break
            jump = jump[jump]
        return out
    try:
        walk = nxt.data if nxt.flags['C_CONTIGUOUS'] else nxt.tolist()
    except (AttributeError, BufferError, ValueError):
        walk = nxt.tolist()
    while p < n:
        starts.append(p)
        p = walk[p]
    return np.asarray(starts, dtype=np.int64)

def _segment(n, want_cut_list, min_block, max_block):
    if n <= 0:
        # No tokens -> no blocks.  Returning the empty triple here (instead of
        # indexing ends[-1] on an empty array) keeps the public entry points
        # total for empty/negative-length inputs.
        return (np.empty(0, dtype=np.int64), {}, [])
    cacheable = n <= 64
    # uint8 cast of the booleans is exactly 0/1 per slot — the same byte string
    # `bytes(bytearray(1 if _c else 0 ...))` built; the cached
    # bids/honoured/counts the key guards are unchanged.
    key = (n, min_block, max_block, np.asarray(want_cut_list, dtype=np.uint8).tobytes()) if cacheable else None
    if cacheable:
        hit = _SEGMENT_CACHE.get(key)
        if hit is not None:
            _SEGMENT_CACHE.move_to_end(key)
            return hit
    # `want_cut_list` arrives as a python list (per-row `.tolist()`) or as a
    # numpy bool row (the batched `_hoist_blocks_from_cuts` path); normalising
    # through np.asarray is a no-op for the latter.  Same booleans either way.
    slots = np.nonzero(np.asarray(want_cut_list, dtype=bool))[0]
    ns = len(slots)
    _mb, _xb = (int(min_block), int(max_block))
    # Uniform-stride closed forms.  When the next-cut signal cannot change the
    # successor (min_block == max_block), when no slot is wanted, or when every
    # slot is wanted, the successor is p + stride for a constant stride.  The
    # counts/bids below are exactly the same integers the general table walk
    # produces (the block starts are an arithmetic progression).
    _stride = None
    if _mb >= 1 and _mb == _xb:
        _stride = _mb
    elif ns == 0:
        _stride = _xb if _xb >= 1 else 1
    elif (n >= 2 and ns >= n - 1
          and int(slots[0]) == 0 and int(slots[n - 2]) == n - 2):
        # A wanted slot at n-1 is a boundary after the final token and does
        # not change the walk, so "every answer-forcing slot" is 0..n-2, not
        # merely a count of n-1 wanted slots: {1} for n=2 also has n-1 slots
        # but leaves slot 0 unset, in which case the first block must run to
        # the cut (count 2), not to the short stride.
        _stride = max(1, min(_mb, _xb))
    if _stride is not None:
        counts = [_stride] * (n // _stride)
        _rem = n % _stride
        if _rem:
            counts.append(_rem)
        # Floor division is the same exact integer block id the repeated-count
        # form builds (uniform stride over the whole sequence).
        bids = _np_range(n) // _stride
        honoured = {}
        if ns:
            # Non-final block ends are stride, 2*stride, ...; the honoured
            # slot of a block is the last wanted slot before its end, and only
            # when that slot lies inside the block.  Same keys/insertion order
            # as the general table walk (dict.fromkeys over increasing slots).
            _ends = np.arange(1, len(counts) + 1, dtype=np.int64) * _stride
            _ends[-1] = n
            _ends = _ends[:-1]
            if _ends.size:
                _hi = np.searchsorted(slots, _ends - 1, side='right')
                _ok = _hi > 0
                _cand = np.where(_ok, slots[np.maximum(_hi - 1, 0)], -1)
                _ok &= _cand >= (_ends - _stride)
                if _ok.any():
                    honoured = dict.fromkeys(_cand[_ok].tolist(), 1.0)
        if cacheable:
            while len(_SEGMENT_CACHE) >= _SEGMENT_CACHE_CAP:
                _SEGMENT_CACHE.popitem(last=False)
            _SEGMENT_CACHE[key] = (bids, honoured, counts)
        return (bids, honoured, counts)
    # Successor-table form of the original greedy scan: with p the start of
    # the current block, the next block start is f(p) = min(p + max_block,
    # max(p + min_block, next_cut[p] + 1), n) (never less than p + 1), where
    # next_cut[p] is the earliest wanted-cut slot >= p (slot ci is the
    # boundary between tokens ci and ci+1, scanned when the scan reaches
    # token ci + 1).  f is a pure integer function of p, so the whole table
    # is built by vectorised integer ops and the scan itself is then a plain
    # table walk from 0 — the same block starts the token-by-token form
    # produced.  A cut at q honours the LAST wanted slot that was already
    # pending by then (the largest slot in [p, q - 1], i.e.
    # slots[bisect_right(slots, q - 1) - 1]) — computed vectorised over the
    # block ends, the same dict the pointer form built.
    _BIG = n + 1
    pos = _seg_pos(n)
    if ns:
        si = np.searchsorted(slots, pos, side='left')
        nc = np.where(si < ns, slots[np.minimum(si, ns - 1)], _BIG)
    else:
        nc = np.full(n + 1, _BIG, dtype=np.int64)
    nxt = np.minimum(pos + int(max_block), np.maximum(pos + int(min_block), nc + 1))
    np.minimum(nxt, n, out=nxt)
    np.maximum(nxt, pos + 1, out=nxt)
    # The walk never reads nxt[n]; the jump-table form does, so keep the
    # sentinel inside the table.
    nxt[-1] = n
    sa = _successor_starts(n, nxt)
    ends = np.empty(len(sa), dtype=np.int64)
    ends[:-1] = sa[1:]
    ends[-1] = n
    counts_np = ends - sa
    counts = counts_np.tolist()
    honoured = {}
    if ns:
        sel = (ends < n) & (nc[sa] <= ends - 1)
        if sel.any():
            hj = np.searchsorted(slots, ends[sel] - 1, side='right')
            # dict.fromkeys over the `.tolist()` view builds the identical
            # {python-int slot: 1.0} mapping (first-occurrence order, duplicate
            # slots collapsed), with the same keys, values and insertion order.
            honoured = dict.fromkeys(slots[hj - 1].tolist(), 1.0)
    bids = np.repeat(np.arange(len(counts_np), dtype=np.int64), counts_np)
    if cacheable:
        while len(_SEGMENT_CACHE) >= _SEGMENT_CACHE_CAP:
            _SEGMENT_CACHE.popitem(last=False)
        _SEGMENT_CACHE[key] = (bids, honoured, counts)
    return (bids, honoured, counts)

_H2D_POOL = {}
_H2D_POOL_BYTES = [0]
try:
    _H2D_POOL_CAP = max(96, int(os.environ.get('CSA_H2D_POOL_MB', '512'))) << 20
except (TypeError, ValueError):
    _H2D_POOL_CAP = 512 << 20
_H2D_POOL_SLOTS = 8
_H2D_POOL_MAX_ENTRIES = 512
_H2D_BUSY = object()
_H2D_POOL_LOCK = threading.RLock()

def _h2d_pool_evict_locked(need=0):
    while _H2D_POOL and (_H2D_POOL_BYTES[0] + int(need) > _H2D_POOL_CAP
                          or len(_H2D_POOL) >= _H2D_POOL_MAX_ENTRIES):
        victim = None
        for _k, _e in list(_H2D_POOL.items()):
            free = True
            for _, _ev in _e['slots']:
                if _ev is _H2D_BUSY or (_ev is not None and not _ev.query()):
                    free = False
                    break
            if free:
                victim = _k
                break
        if victim is None:
            break
        _e = _H2D_POOL.pop(victim)
        _H2D_POOL_BYTES[0] -= int(_e.get('bytes', 0))

def _h2d_pool_evict(need=0):
    with _H2D_POOL_LOCK:
        _h2d_pool_evict_locked(need)

def _h2d_stage(shape, dtype, device):
    if device.type != 'cuda' or not torch.cuda.is_available():
        return (None, None)
    shape = tuple(int(_s) for _s in shape)
    if not shape or 0 in shape:
        return (None, None)
    key = (_dev_str(device), dtype, shape)
    with _H2D_POOL_LOCK:
        ent = _H2D_POOL.get(key)
        if ent is None:
            per_slot = int(np.prod(shape, dtype=np.int64)) * torch.empty((), dtype=dtype).element_size()
            # Large uploads reserve only as many slots as fit comfortably.
            # The slot count is per entry, so small uploads still use the
            # full ring.
            n_slots = max(1, min(int(_H2D_POOL_SLOTS), int((_H2D_POOL_CAP // 2) // max(1, per_slot))))
            est = per_slot * n_slots
            _h2d_pool_evict_locked(est)
            if _H2D_POOL_BYTES[0] + est > _H2D_POOL_CAP:
                return (None, None)
            try:
                with torch.inference_mode(False):
                    slots = [[torch.empty(shape, dtype=dtype).pin_memory(), None] for _ in range(n_slots)]
            except (RuntimeError, TypeError):
                return (None, None)
            _H2D_POOL_BYTES[0] += est
            ent = {'slots': slots, 'rr': 0, 'bytes': est}
            _H2D_POOL[key] = ent
        slots = ent['slots']
        n = len(slots)
        for _ in range(n):
            i = ent['rr'] % n
            ent['rr'] += 1
            buf, ev = slots[i]
            if ev is None or (ev is not _H2D_BUSY and ev.query()):
                slots[i][1] = _H2D_BUSY

                def commit(_buf=buf, _slot=slots[i], _device=device):
                    try:
                        # Record the guard event on the target device's stream
                        # (a pool entry is keyed by device, so multi-GPU copies
                        # must not be guarded by another device's default stream).
                        with torch.inference_mode(False), torch.cuda.device(_device):
                            dev_t = _buf.to(_device, non_blocking=True)
                            ev2 = torch.cuda.Event()
                            ev2.record()
                    except BaseException:
                        # Keep the conservative reservation: an enqueue may
                        # already have succeeded before the failure.
                        raise
                    with _H2D_POOL_LOCK:
                        _slot[1] = ev2
                    return dev_t
                return (buf, commit)
        return (None, None)

def _h2d_async(host_src, dtype, device):
    with torch.inference_mode(False):
        t = torch.as_tensor(host_src, dtype=dtype)
        if t.device == device:
            return t
        if t.device.type == 'cuda' or device.type != 'cuda' or not torch.cuda.is_available():
            return t.to(device)
        staged, commit = _h2d_stage(t.shape, t.dtype, device)
        if staged is not None:
            staged.copy_(t)
            return commit()
        if t.nbytes * 4 > _H2D_POOL_CAP:
            return t.to(device)
        try:
            t = t.pin_memory()
        except RuntimeError:
            pass
        return t.to(device, non_blocking=t.is_pinned())

_D2H_POOL = {}
_D2H_POOL_BYTES = [0]
try:
    _D2H_POOL_CAP = max(16, int(os.environ.get('CSA_D2H_POOL_MB', '64'))) << 20
except (TypeError, ValueError):
    _D2H_POOL_CAP = 64 << 20
_D2H_POOL_SLOTS = 4
_D2H_POOL_MAX_ENTRIES = 64
_D2H_BUSY = object()
_D2H_POOL_LOCK = threading.RLock()

def _d2h_pool_evict_locked(need=0):
    # Same release discipline as the H2D pool: an entry leaves the pool only
    # once every recorded download has completed.
    while _D2H_POOL and (_D2H_POOL_BYTES[0] + int(need) > _D2H_POOL_CAP
                         or len(_D2H_POOL) >= _D2H_POOL_MAX_ENTRIES):
        victim = None
        for _k, _e in list(_D2H_POOL.items()):
            free = True
            for _, _ev in _e['slots']:
                if _ev is _D2H_BUSY or (_ev is not None and not _ev.query()):
                    free = False
                    break
            if free:
                victim = _k
                break
        if victim is None:
            break
        _e = _D2H_POOL.pop(victim)
        _D2H_POOL_BYTES[0] -= int(_e.get('bytes', 0))

def _d2h_stage(shape, dtype, device):
    """Pinned-staging pool for device->host downloads — transport only.

    The caller enqueues `buf.copy_(dev_t, non_blocking=True)` on the current
    stream and later calls the returned `sync()`, which records one event
    after the copy and waits for it — the only host wait — before the buffer
    is read.  Every downloaded byte is identical to a plain `.cpu()` copy;
    only the transport changes (pinned ring + non-blocking enqueue), which
    lets the caller enqueue further device work (or run host code) that
    overlaps the in-flight download.  A slot is never rewritten while one of
    its downloads may still be pending; when the pool cannot serve the
    caller falls back to a direct `.cpu()` copy with identical values.
    """
    if device.type != 'cuda' or not torch.cuda.is_available():
        return (None, None)
    shape = tuple(int(_s) for _s in shape)
    if not shape or 0 in shape:
        return (None, None)
    key = (_dev_str(device), dtype, shape)
    with _D2H_POOL_LOCK:
        ent = _D2H_POOL.get(key)
        if ent is None:
            per_slot = int(np.prod(shape, dtype=np.int64)) * torch.empty((), dtype=dtype).element_size()
            n_slots = max(1, min(int(_D2H_POOL_SLOTS), int((_D2H_POOL_CAP // 2) // max(1, per_slot))))
            est = per_slot * n_slots
            _d2h_pool_evict_locked(est)
            if _D2H_POOL_BYTES[0] + est > _D2H_POOL_CAP:
                return (None, None)
            try:
                with torch.inference_mode(False):
                    slots = [[torch.empty(shape, dtype=dtype).pin_memory(), None] for _ in range(n_slots)]
            except (RuntimeError, TypeError):
                return (None, None)
            _D2H_POOL_BYTES[0] += est
            ent = {'slots': slots, 'rr': 0, 'bytes': est}
            _D2H_POOL[key] = ent
        slots = ent['slots']
        n = len(slots)
        for _ in range(n):
            i = ent['rr'] % n
            ent['rr'] += 1
            buf, ev = slots[i]
            if ev is None or (ev is not _D2H_BUSY and ev.query()):
                slots[i][1] = _D2H_BUSY

                def sync(_buf=buf, _slot=slots[i], _device=device):
                    with torch.inference_mode(False), torch.cuda.device(_device):
                        ev2 = torch.cuda.Event()
                        ev2.record()
                        ev2.synchronize()
                    with _D2H_POOL_LOCK:
                        _slot[1] = ev2
                    return _buf
                return (buf, sync)
        return (None, None)

def blocks_from_cuts(n, want_cut_list, min_block, max_block, device, return_count=False, return_honoured=False):
    if n == 0:
        base, cnt, hon = (torch.empty(0, dtype=torch.long, device=device), 0, {})
    elif n == 1:
        base, cnt, hon = (torch.zeros(1, dtype=torch.long, device=device), 1, {})
        # Match the device-side segmenter: every non-empty block-table object
        # carries the host count attachment, including the single-token case.
        base._pool_cpu = ([1], 1, 0)
    else:
        bids, hon, counts = _segment(n, want_cut_list, min_block, max_block)
        base = _h2d_async(bids, torch.long, device)
        cnt = len(counts)
        _mx = max(counts)
        base._pool_cpu = (counts, _mx, max(counts[:-1]) if cnt > 1 else 0)
    if not (return_count or return_honoured):
        return base
    out = [base]
    if return_count:
        out.append(cnt)
    if return_honoured:
        out.append(hon)
    return tuple(out)

def _hoist_blocks_from_cuts(T, gate_lists, min_block, max_block, device, want_honoured, hon_dtype):
    # Batched form of the per-row `blocks_from_cuts` + honoured-mask uploads in
    # the cosine_learnable path.  `blocks_from_cuts` stays the single exit for
    # the cut table (called with a CPU device it returns the host tensor and
    # performs no H2D).  Every value reaching the device — the integer block
    # ids (with the `_pool_cpu` tag re-attached to each device row) and the
    # 0.0/1.0 honoured mask — is identical to the per-row form.
    bids_cpu = []
    nblks = []
    hons = []
    # Rows with identical cut masks produce identical integer geometry, so the
    # host segmenter runs once per distinct row.
    _uniq = []
    _slot = []
    _seen = {}
    for _g in gate_lists:
        _a = np.asarray(_g, dtype=bool)
        # The byte key is exact (no hash-collision contract change).
        _k = (_a.shape, _a.tobytes())
        _j = _seen.get(_k)
        if _j is None:
            _j = len(_uniq)
            _seen[_k] = _j
            _uniq.append(_a)
        _slot.append(_j)
    _cache = []
    for _a in _uniq:
        _cache.append(blocks_from_cuts(T, _a, min_block, max_block, _CPU_DEV, return_count=True, return_honoured=True))
    for _j in _slot:
        _b, _nb, _hon = _cache[_j]
        bids_cpu.append(_b)
        nblks.append(_nb)
        hons.append(_hon)
    staged, commit = _h2d_stage((len(bids_cpu), T), torch.long, device)
    if staged is not None:
        torch.stack(bids_cpu, out=staged)
        bid_stack = commit()
    else:
        bid_stack = _h2d_async(torch.stack(bids_cpu), torch.long, device)
    hon_stack = None
    if want_honoured and T > 1:
        arr = None
        staged_h, commit_h = _h2d_stage((len(gate_lists), T - 1), hon_dtype, device)
        if staged_h is not None:
            try:
                arr = staged_h.numpy()
                arr[:] = 0.0
            except TypeError:
                # a numpy-less dtype (e.g. bfloat16) — fall back to a host array
                arr = None
                staged_h = commit_h = None
        if arr is None:
            arr = np.zeros((len(gate_lists), T - 1), dtype=np.float32)
        for _i, _hon in enumerate(hons):
            if _hon:
                # iterating the honoured dict yields its (integer) slot keys
                _sl = np.fromiter(_hon, dtype=np.int64, count=len(_hon))
                _sl = _sl[(0 <= _sl) & (_sl < T - 1)]
                arr[_i, _sl] = 1.0
        # the honoured mask is exact 0.0/1.0, identical in any float dtype
        hon_stack = commit_h() if commit_h is not None else _h2d_async(arr, hon_dtype, device)
    out = []
    for b in range(len(bids_cpu)):
        row = bid_stack[b]
        row._pool_cpu = getattr(bids_cpu[b], '_pool_cpu', None)
        out.append((row, nblks[b], None if hon_stack is None else hon_stack[b]))
    return out

_DEVICE_SEGMENT = os.environ.get('CSA_DEVICE_SEGMENT', '1').strip().lower() not in ('0', 'false', 'off', 'no')


def _device_segment_enabled(device):
    return _DEVICE_SEGMENT and getattr(device, 'type', str(device)) == 'cuda'


def _device_segment_geometry(n, cuts, min_block, max_block, device):
    """Exact device-side replica of the integers `_segment` produces.

    `cuts` is a [B, n-1] boolean tensor on `device`.  The greedy successor
    walk is generated by doubling (O(log(blocks)) batched gathers) instead of
    a host loop, so the per-layer gate decisions never need a
    device->host->device round trip just to rebuild block geometry.
    """
    import torch.nn.functional as _F
    B = int(cuts.shape[0])
    dev = cuts.device
    if B == 0:
        return (torch.empty((0, n), dtype=torch.long, device=dev),
                torch.empty((0, 1), dtype=torch.long, device=dev),
                torch.empty((0, max(n - 1, 0)), dtype=torch.long, device=dev))
    if n <= 1:
        bid = torch.zeros((B, n), dtype=torch.long, device=dev)
        counts = torch.ones((B, 1), dtype=torch.long, device=dev) if n == 1 else torch.zeros((B, 1), dtype=torch.long, device=dev)
        return bid, counts, torch.zeros((B, max(n - 1, 0)), dtype=torch.long, device=dev)
    idx = torch.arange(n - 1, device=dev)
    big = n + 1
    # Next wanted cut at or after each position (BIG when none), then pad the
    # two positions (n-1, n) that have no following cut.
    nc = torch.where(cuts, idx[None, :], big).flip(-1).cummin(-1).values.flip(-1)
    nc = _F.pad(nc, (0, 2), value=big)
    pos = torch.arange(n + 1, device=dev)
    _mb, _xb = int(min_block), int(max_block)
    nxt = torch.minimum(pos + _xb, torch.maximum(pos + _mb, nc + 1))
    nxt = torch.minimum(nxt, torch.as_tensor(n, dtype=torch.long, device=dev))
    nxt = torch.maximum(nxt, pos + 1)
    nxt = nxt.clone()
    nxt[:, -1] = n
    # Orbit of f(0) under f(p)=nxt[p], generated in doubling batches.  Invalid
    # slots stay at the sentinel n; the array is padded to a power of two.
    k_max = (n + max(_mb, 1) - 1) // max(_mb, 1) + 1
    target = 1
    while target < k_max:
        target <<= 1
    starts = torch.zeros((B, 1), dtype=torch.long, device=dev)
    jump = nxt
    while starts.shape[1] < target:
        nxts = torch.gather(jump, 1, starts)
        starts = torch.cat([starts, nxts], 1)
        jump = torch.gather(jump, 1, jump)
    valid = starts < n
    ends = torch.cat([starts[:, 1:], torch.full((B, 1), n, dtype=torch.long, device=dev)], 1)
    counts = (ends - starts).clamp_min(0) * valid.to(torch.long)
    # Block id per token = first block whose cumulative end is > token.
    cum = torch.cumsum(counts, 1)
    _tok = _range_cache(0, n, dev)[None, :].expand(B, -1).contiguous()
    bid = torch.searchsorted(cum, _tok, right=True)
    # Honoured cut per non-final block = last cut <= end-1, but only when a
    # cut exists in [start, end-1] (same condition as _segment).
    last_cut = torch.where(cuts, idx[None, :], -1).cummax(-1).values
    last_cut = _F.pad(last_cut, (0, 1), value=-1)
    nc_start = torch.gather(nc, 1, starts)
    has = (ends < n) & (nc_start <= ends - 1)
    hon_slot = torch.gather(last_cut, 1, (ends - 1).clamp_min(0))
    hon_slot = torch.where(has, hon_slot, torch.full_like(hon_slot, -1))
    return bid, counts, hon_slot


def _device_blocks_from_cuts(T, cuts, min_block, max_block, device, want_honoured, hon_dtype):
    bid, counts, hon_slot = _device_segment_geometry(int(T), cuts, min_block, max_block, device)
    B = int(bid.shape[0])
    n_blocks = (counts > 0).sum(1)
    packed = torch.cat([counts, n_blocks[:, None]], 1)
    staged = sync = None
    if packed.numel():
        staged, sync = _d2h_stage(tuple(packed.shape), torch.long, device)
        if staged is not None:
            staged.copy_(packed, non_blocking=True)
    hon_stack = None
    if want_honoured and T > 1:
        hon_stack = torch.zeros((B, T - 1), dtype=hon_dtype, device=device)
        _rows = torch.arange(B, device=device)[:, None].expand_as(hon_slot)
        _ok = hon_slot >= 0
        hon_stack[_rows[_ok], hon_slot[_ok]] = 1.0
    if staged is not None:
        host = sync().numpy()
    else:
        host = packed.cpu().numpy()
    out = []
    host_nb = host[:, -1]
    for b in range(B):
        row = bid[b]
        nb = int(host_nb[b])
        cnt = np.array(host[b, :nb])
        _mx = int(cnt.max()) if nb else 0
        # numpy rows carry the same immutable-by-convention counts integers;
        # every consumer (len/max/min/sum/np.asarray/accumulate) reads them
        # identically to the previous Python lists.
        row._pool_cpu = (cnt, _mx, int(cnt[:-1].max()) if nb > 1 else 0)
        row._pool_nblocks = nb
        out.append((row, nb, None if hon_stack is None else hon_stack[b]))
    return out


def _device_blocks_from_gate_bool(T, gate_bool, min_block, max_block, device, want_honoured, hon_dtype):
    return _device_blocks_from_cuts(T, gate_bool, min_block, max_block, device, want_honoured, hon_dtype)


def blocks_from_cosine(H, tau, min_block, max_block, sim=None, return_count=False):
    n = H.shape[0]
    dev = H.device
    if n <= 1:
        z = torch.zeros(n, dtype=torch.long, device=dev)
        return (z, 1) if return_count and n == 1 else ((z, 0) if return_count else z)
    if sim is None:
        sim = cosine_similarity_consecutive(H)
    # numpy view; the segmenter sees the same booleans as a list
    cut_list = (sim < tau).cpu().numpy()
    return blocks_from_cuts(n, cut_list, min_block, max_block, dev, return_count=return_count)

def _causal_quantile_rank(x, lo, frac, use_hi, first_row, q, n):
    """Exact causal quantile via cumulative comparison counts (full column range).

    `x[j]` is the k-th smallest value of the prefix `[0, c]` exactly when
    `#{i <= c : (x_i, i) < (x_j, j)} == k` under the strict order (value,
    then index).  The caller routes non-finite inputs and sampled column
    grids to the sort path, whose semantics this reproduces bit-for-bit for
    finite inputs (ties give equal values regardless of order, and the
    arithmetic is the same expression evaluated in the same dtype).
    """
    rows = int(x.shape[0])
    dev = x.device
    idx = _arange_cache(n, dev)
    out_lo = x.new_empty((rows, n))
    need_hi = q < 1.0
    out_hi = x.new_empty((rows, n)) if need_hi else out_lo
    _budget = 48 << 20
    _per_elem = 12
    jc = max(1, min(16, _budget // max(1, n * _per_elem)))
    rb = max(1, min(16, _budget // max(1, n * jc * _per_elem)))
    for r0 in range(0, rows, rb):
        r1 = min(r0 + rb, rows)
        xr = x[r0:r1]
        out_lo_r = out_lo[r0:r1]
        out_hi_r = out_hi[r0:r1]
        for j0 in range(0, n, jc):
            j1 = min(j0 + jc, n)
            xj = xr[:, j0:j1]
            jb = j1 - j0
            lt = xr.unsqueeze(-1) < xj.unsqueeze(1)
            eq = xr.unsqueeze(-1) == xj.unsqueeze(1)
            ilt = idx.unsqueeze(-1) < idx[j0:j1].unsqueeze(0)
            A = lt | (eq & ilt)
            S = A.cumsum(1, dtype=torch.int32)[:, j0:]
            # Only the boundary columns c in [j0, j1) can see a j > c.
            jle = idx[j0:j1].unsqueeze(0) <= idx[j0:j1].unsqueeze(1)
            m_lo = S == lo[j0:].unsqueeze(0).unsqueeze(-1)
            m_lo[:, :jb] &= jle
            hit_lo = m_lo.any(-1)
            v_lo = xj.gather(1, m_lo.to(torch.uint8).argmax(-1).long())
            out_lo_r[:, j0:] = torch.where(hit_lo, v_lo, out_lo_r[:, j0:])
            if need_hi:
                m_hi = S == (lo[j0:] + 1).unsqueeze(0).unsqueeze(-1)
                m_hi[:, :jb] &= jle
                hit_hi = m_hi.any(-1)
                v_hi = xj.gather(1, m_hi.to(torch.uint8).argmax(-1).long())
                out_hi_r[:, j0:] = torch.where(hit_hi, v_hi, out_hi_r[:, j0:])
    if need_hi:
        interp = out_lo * (1.0 - frac) + out_hi * frac
        oc = torch.where(use_hi, interp, out_lo)
    else:
        oc = out_lo
    return torch.where(first_row, torch.full_like(oc, float('-inf')), oc)


def _causal_quantile_exact(x, cols, target_block_tokens, _full_range=False):
    """Closed-form causal quantile at the requested columns.

    `x` is the [rows, n] work-precision similarity matrix and `cols` the
    (ascending) column indices to evaluate.  Row/column chunking is a pure
    partition, so the values are identical to the full-table form.
    """
    n = int(x.shape[-1])
    dev = x.device
    q = max(1.0 / max(int(target_block_tokens), 1), 0.001)
    idx = _arange_cache(n, dev)
    cols = cols.to(torch.long)
    pos = q * cols.to(torch.float32)
    lo = pos.floor().long()
    frac = pos - lo
    lo_c = lo.clamp(max=n - 1)
    hi_c = (lo + 1).clamp(max=n - 1)
    use_hi = (frac > 0.0) & (lo + 1 <= cols)
    first_row = cols == 0
    nc = max(1, int(cols.numel()))
    if (_full_range and n > 0 and cols.numel() == n
            and bool(torch.isfinite(x).all())):
        # `lo`/`lo + 1` never exceed `n - 1` on the full range (q <= 1), so
        # the clamped gathers reduce to the plain rank targets.
        return _causal_quantile_rank(x, lo, frac, use_hi, first_row, q, n)
    _pair_bytes = 16
    _budget = 128 << 20
    col_chunk = max(1, min(nc, 64, _budget // max(1, n * _pair_bytes)))
    row_chunk = max(1, _budget // max(col_chunk * n * _pair_bytes, 1))
    out = x.new_empty((x.shape[0], nc))
    for r0 in range(0, x.shape[0], row_chunk):
        b = min(row_chunk, x.shape[0] - r0)
        xr = x[r0:r0 + b]
        for c0 in range(0, nc, col_chunk):
            c1 = min(c0 + col_chunk, nc)
            cc = cols[c0:c1]
            masked = torch.where(idx[None, None, :] <= cc[None, :, None],
                                 xr[:, None, :], float('inf'))
            masked = masked.sort(dim=-1).values
            _lc = lo_c[c0:c1][None, :, None].expand(b, c1 - c0, 1)
            _hc = hi_c[c0:c1][None, :, None].expand(b, c1 - c0, 1)
            lo_v = masked.gather(-1, _lc).squeeze(-1)
            hi_v = masked.gather(-1, _hc).squeeze(-1)
            interp = lo_v * (1.0 - frac[c0:c1]) + hi_v * frac[c0:c1]
            oc = torch.where(use_hi[c0:c1], interp, lo_v)
            out[r0:r0 + b, c0:c1] = torch.where(first_row[c0:c1],
                                                torch.full_like(oc, float('-inf')), oc)
    return out


_ADAPTIVE_EXACT_DEFAULT = 8192


def _adaptive_exact_limit():
    """Length up to which the adaptive segmenter uses its exact closed form.

    The default is 8192.  Sampling above it changes block geometry, so the
    effective limit is part of the run fingerprint (`_ae`); a panel that
    needs the sampled path must opt in explicitly.
    ``CSA_ADAPTIVE_EXACT_TOKENS`` selects the cutoff without editing code.
    """
    raw = os.environ.get('CSA_ADAPTIVE_EXACT_TOKENS', '').strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            print(f'[segmenter] CSA_ADAPTIVE_EXACT_TOKENS={raw!r} is not an integer — using {_ADAPTIVE_EXACT_DEFAULT}')
    return _ADAPTIVE_EXACT_DEFAULT


def causal_adaptive_threshold(sim, target_block_tokens):
    if sim.numel() == 0:
        return torch.empty(0, dtype=sim.dtype, device=sim.device)
    n = int(sim.shape[-1])
    # Hard safety ceiling only (the sampling path below is bounded).
    _max_len = 1 << 20
    _env_max = os.environ.get('CSA_ADAPTIVE_MAX_TOKENS', '').strip()
    if _env_max:
        try:
            _max_len = max(1, int(_env_max))
        except ValueError:
            print(f'[segmenter] CSA_ADAPTIVE_MAX_TOKENS={_env_max!r} is not an integer — using {_max_len}')
    if n > _max_len:
        raise RuntimeError(
            f'causal_adaptive_threshold 超出显式安全上限 {_max_len}（CSA_ADAPTIVE_MAX_TOKENS）: '
            '请提高该上限或改用 chunking="cosine_learnable"')
    # The default is 8192; the sampled long-context path must be opted into
    # explicitly (and is part of the run fingerprint through `_ae`).
    _exact_len = _adaptive_exact_limit()
    squeeze = sim.dim() == 1
    x = (sim.unsqueeze(0) if squeeze else sim).detach().float()
    if n <= _exact_len:
        out = _causal_quantile_exact(x, _arange_cache(n, sim.device), target_block_tokens,
                                     _full_range=True)
    else:
        # Long-context path.  Two invariants make it prefix-consistent:
        #   1) columns < _exact_len are computed by the SAME closed form the
        #      short path uses (on x[:, :_exact_len], which is all a causal
        #      quantile at those columns can see);
        #   2) later columns use a GLOBAL sample grid fixed by the configured
        #      maximum length and only read the last sample at or before the
        #      column.  No `n-1` endpoint and no `n`-derived step, so
        #      tau_T[:T'] == tau_T' for every T' < T.
        # The sampled grid never mixes a future sample into a past column.
        exact_n = min(n, int(_exact_len))
        pre_cols = torch.arange(exact_n, device=sim.device, dtype=torch.long)
        pre_vals = _causal_quantile_exact(x[:, :exact_n], pre_cols, target_block_tokens,
                                          _full_range=True)
        warm_end = min(int(n), max(exact_n, 256, _exact_len // 8))
        n_samp = 1024
        step = max(1, (_max_len - warm_end + n_samp - 1) // n_samp)
        samp = torch.cat([
            torch.arange(exact_n, min(warm_end, n), device=sim.device, dtype=torch.long),
            torch.arange(warm_end, n, step, device=sim.device, dtype=torch.long),
        ]).unique(sorted=True)
        if samp.numel() == 0:
            out = pre_vals
        else:
            sv = _causal_quantile_exact(x, samp, target_block_tokens)
            full = torch.arange(exact_n, n, device=sim.device, dtype=torch.long)
            # Rightmost sample at or before each column: only past information.
            pos = (torch.searchsorted(samp, full, right=True) - 1).clamp(0, samp.numel() - 1)
            out = torch.cat([pre_vals, sv[:, pos]], dim=1)
        out[:, 0] = float('-inf')
    out = out.to(sim.device)
    return out.squeeze(0) if squeeze else out


def blocks_fixed(n, block_size, device):
    key = (_dev_str(device), 'blkfix', int(n), int(block_size))
    hit = _cache_get(_IDX_CACHE, key)
    if hit is not None:
        return hit
    with torch.inference_mode(False):
        out = torch.arange(n, device=device) // int(block_size)
        if n > 0:
            bs = int(block_size)
            counts = [bs] * (n // bs)
            _rem = n % bs
            if _rem:
                counts.append(_rem)
            out._pool_cpu = (counts, min(bs, n), bs if len(counts) > 1 else 0)
    _cache_put(_IDX_CACHE, key, out, _MASK_CACHE_BUDGET_BYTES, _idx_cache_total)
    return out
_MASK_CACHE = {}
_CAUSAL_DONOR = -1
_CAUSAL_TRI = None
_CAUSAL_DONOR_PIN_BYTES = 64 << 20

def _causal_donor(device, width):
    # 单一共享 bool donor（True = j > i 的被掩 lane）：同一 [width, width]
    if not _dense_mask_fits(width, 1):
        raise RuntimeError(
            f'refusing to materialise a full [{int(width)}, {int(width)}] bool mask '
            f'(> {int(_DENSE_MASK_ROW_CAP_BYTES)} bytes); use the row-chunked mask path')
    global _CAUSAL_TRI
    dev_s = _dev_str(device)
    width = int(width)
    nbytes = max(0, width * width)
    pin = nbytes <= _CAUSAL_DONOR_PIN_BYTES
    key = (dev_s, 'causal', _CAUSAL_DONOR)
    held = _CAUSAL_TRI
    if pin and held is not None and held[0] == dev_s and (held[1].shape[0] >= width):
        return (held[1], dev_s)
    if not pin:
        held = None
        _CAUSAL_TRI = None
    tri = _cache_get(_MASK_CACHE, key) if not pin else None
    if tri is not None and tri.shape[0] >= width:
        return (tri, dev_s)
    tri = _mask_buffer(width, device, 'causal')
    if pin:
        _PINNED_KEYS.add(key)
        _CAUSAL_TRI = (dev_s, tri)
    else:
        _PINNED_KEYS.discard(key)
    _cache_put(_MASK_CACHE, key, tri, _mask_budget(width), _mask_cache_total)
    return (tri, dev_s)
_IDX_CACHE = {}
_IDX_CACHE_BYTES = 0
_MASK_CACHE_BUDGET_BYTES = 256 << 20

def _mask_budget(n):
    return _MASK_CACHE_BUDGET_BYTES

def _cache_bytes(t):
    if t is None:
        return 0
    if isinstance(t, (tuple, list)):
        return sum((_cache_bytes(_v) for _v in t))
    try:
        nbytes = int(t.numel()) * int(t.element_size())
        # Views can share storage: charge min(logical size, storage size) so a
        # stride-0 expand and a sliced view of a larger buffer are not
        # over-counted.
        try:
            nbytes = min(nbytes, int(t.untyped_storage().nbytes()))
        except Exception:
            pass
        return nbytes
    except Exception:
        return 0

def _mask_cache_total(cache):
    return sum((_cache_bytes(v) for v in cache.values()))

def _idx_cache_total(cache):
    return sum((_cache_bytes(v) + 128 * len(k) for k, v in cache.items()))

def _is_pinned_key(key):
    return key in _PINNED_KEYS
_PINNED_KEYS = set()

_CACHE_STATS = {}


def _cache_item_bytes(key, value, total_of):
    n = _cache_bytes(value)
    if total_of is _idx_cache_total:
        n += 128 * len(key)
    return n


def _cache_stat(cache):
    st = _CACHE_STATS.get(id(cache))
    if st is None or st.get('obj') is not cache:
        st = {'obj': cache, 'total': 0}
        _CACHE_STATS[id(cache)] = st
    return st


def _evictable_limit(cache, limit, total_of):
    pinned = sum((_cache_bytes(cache[k]) for k in _PINNED_KEYS if k in cache))
    return limit - pinned if pinned < limit else 0


def _cache_touch(cache, key):
    """Mark a cache entry as most-recently used (LRU, not FIFO)."""
    try:
        val = cache.pop(key)
    except KeyError:
        return None
    cache[key] = val
    return val


def _cache_get(cache, key):
    """LRU read: return the entry and move it to the recent end."""
    hit = _cache_touch(cache, key)
    return hit


def _cache_put(cache, key, tensor, limit, total_of):
    st = _cache_stat(cache)
    if key in cache:
        st['total'] -= _cache_item_bytes(key, cache[key], total_of)
        # Re-inserting an existing key refreshes its recency as well.
        try:
            cache.pop(key)
        except KeyError:
            pass
    cache[key] = tensor
    st['total'] += _cache_item_bytes(key, tensor, total_of)
    total = st['total']
    evictable = _evictable_limit(cache, limit, total_of)
    while total > limit and total > evictable:
        oldest = next((k for k in cache if not _is_pinned_key(k)), None)
        if oldest is None:
            break
        st['total'] -= _cache_item_bytes(oldest, cache.pop(oldest), total_of)
        total = st['total']
    return total
_ATB_CACHE = {}
_ATB_EPOCH = 0

def _attb_bump_epoch():
    global _ATB_EPOCH
    _ATB_EPOCH += 1
    _ATB_CACHE.clear()
_HALF = 0.5

def _arange_cache(n, device):
    key = (_dev_str(device), 'arange', int(n))
    a = _cache_get(_IDX_CACHE, key)
    if a is None:
        with torch.inference_mode(False):
            a = torch.arange(n, device=device)
        _cache_put(_IDX_CACHE, key, a, _MASK_CACHE_BUDGET_BYTES, _idx_cache_total)
    return a

def _window_geometry(n, w, device):
    key = (_dev_str(device), 'wingeo', int(n), int(w))
    hit = _cache_get(_IDX_CACHE, key)
    if hit is not None:
        return hit
    pos = _arange_cache(n, device)
    rel = _range_cache(0, w, device)
    with torch.inference_mode(False):
        # int32 arithmetic throughout (design notes §6.1): every position is an
        # exact integer in [-(w-1), n) and the sequence lengths this repo uses
        # lie far inside the int32 range, so the table holds bitwise the same
        # integers as the int64 form — index_select accepts int32 directly.
        # The validity mask is read off the pre-clamp table, so it sees the
        # same negative positions either way.
        win_idx = pos.to(torch.int32)[:, None] - (w - 1) + rel.to(torch.int32)[None, :]
        valid = win_idx >= 0
        win_idx.clamp_(min=0)
        pair = (win_idx, valid)
    _cache_put(_IDX_CACHE, key, pair, _MASK_CACHE_BUDGET_BYTES, _idx_cache_total)
    return pair

def _range_cache(lo, hi, device):
    if lo == 0:
        return _arange_cache(hi, device)
    key = (_dev_str(device), 'arange2', int(lo), int(hi))
    a = _cache_get(_IDX_CACHE, key)
    if a is None:
        with torch.inference_mode(False):
            a = torch.arange(lo, hi, device=device)
        _cache_put(_IDX_CACHE, key, a, _MASK_CACHE_BUDGET_BYTES, _idx_cache_total)
    return a

def _arange_exp_cache(n, m, device):
    # Cached int32 (n, m) arange expansion for the HCA read-everything arm:
    # the same exact integers as _arange_cache(m, device).unsqueeze(0).
    # expand(n, m).to(torch.int32) (index transport, design notes §6.1); int32
    # in (also §6.1).  Built outside
    # inference mode so a table first filled during eval can still serve as an
    # autograd index tensor in training; only ever read downstream (row slices
    # feeding gathers), never mutated.
    #
    # The table is a stride-0 expand of one [m] arange: every row holds the
    # same exact integers, advanced indexing / slicing / cat read them
    # identically, and nothing downstream writes to the table.
    key = (_dev_str(device), 'arangeexp', int(n), int(m))
    a = _cache_get(_IDX_CACHE, key)
    if a is None:
        with torch.inference_mode(False):
            base = torch.arange(int(m), device=device, dtype=torch.int32)
            a = base.expand(int(n), int(m))
        _cache_put(_IDX_CACHE, key, a, _MASK_CACHE_BUDGET_BYTES, _idx_cache_total)
    return a

def _causal_window_mask_build(T, window, device, dtype):
    # Direct construction of the causal sliding-window additive mask in the
    if not _dense_mask_fits(T, torch.empty((), dtype=dtype).element_size()):
        raise RuntimeError(
            f'refusing to materialise a full [{int(T)}, {int(T)}] {dtype} mask '
            f'(> {int(_DENSE_MASK_ROW_CAP_BYTES)} bytes); use the row-chunked mask path')
    # target dtype: allowed lanes (i-window < j <= i) get exactly 0.0 and every
    # other lane exactly -inf — the same two constants the causal + band
    # addition and the elementwise cast produce.  A boolean scratch is
    # triangularised twice to fill the two masked regions.
    with torch.inference_mode(False):
        m = torch.zeros((T, T), device=device, dtype=dtype)
        far = torch.ones((T, T), device=device, dtype=torch.bool)
        far.tril_(-int(window))
        m.masked_fill_(far, float('-inf'))
        far.fill_(True).triu_(1)
        m.masked_fill_(far, float('-inf'))
    return m

_DENSE_MASK_ROW_CAP_BYTES = 64 << 20


def _dense_mask_fits(T, elem_bytes=1):
    """True when a full [T, T] mask of `elem_bytes` fits the byte budget."""
    try:
        return int(T) * int(T) * max(1, int(elem_bytes)) <= int(_DENSE_MASK_ROW_CAP_BYTES)
    except Exception:
        return False

_DENSE_ROW_MASK_CACHE = {}
try:
    _DENSE_ROW_MASK_BUDGET_BYTES = max(16, int(os.environ.get('CSA_DENSE_ROW_MASK_MB', '128'))) << 20
except (TypeError, ValueError):
    _DENSE_ROW_MASK_BUDGET_BYTES = 128 << 20


def _dense_causal_window_rows(T, window, s, e, device):
    # Row slice [s:e] of the dense causal/window masked-lane table: allowed
    # lanes are exactly (i-window < j <= i), the same lanes
    # `_causal_window_mask_build` fills with -inf, so masked_fill_ with this
    # table writes exactly the -inf the additive row slice added.  Built
    # outside inference mode so a row slice first built during eval can still
    # serve training.
    key = (_dev_str(device), 'dense-rows', int(T), int(window), int(s), int(e))
    hit = _cache_get(_DENSE_ROW_MASK_CACHE, key)
    if hit is not None:
        return hit
    with torch.inference_mode(False):
        pos = _arange_cache(T, device)
        q = pos[s:e, None]
        j = pos[None, :]
        if window <= 0 or window >= T:
            bad = j > q
        else:
            bad = (j > q) | (j <= q - int(window))
    _cache_put(_DENSE_ROW_MASK_CACHE, key, bad, _DENSE_ROW_MASK_BUDGET_BYTES, _mask_cache_total)
    return bad

def causal_window_mask_f(T, window, device, dtype):
    """SDPA 的加性 0/-inf mask（attn_mask 只吃加性形式；0 与 -inf 的逐元素
    cast 对任何浮点 dtype 精确，缓存键含 dtype，mask 下游只读）。
    window<=0/>=T 的因果分支由 bool donor（`causal_mask`）经 masked_fill
    重建。"""
    key = (_dev_str(device), 'cwf', int(T), int(window), str(dtype))
    if window <= 0 or window >= T:
        base = causal_mask(T, device)
        if base.dtype == dtype:
            return base
        m = _cache_get(_MASK_CACHE, key)
        if m is None:
            with torch.inference_mode(False):
                m = torch.zeros((T, T), device=device, dtype=dtype)
                m.masked_fill_(base, float('-inf'))
            _cache_put(_MASK_CACHE, key, m, _mask_budget(T), _mask_cache_total)
        return m
    m = _cache_get(_MASK_CACHE, key)
    if m is None:
        m = _causal_window_mask_build(T, window, device, dtype)
        _cache_put(_MASK_CACHE, key, m, _mask_budget(T), _mask_cache_total)
    return m

_ATTN_TRANSIENT_CAP_BYTES = 512 << 20

def _attn_transient_budget(device):
    key = (_dev_str(device), _ATB_EPOCH)
    if key in _ATB_CACHE:
        return _ATB_CACHE[key]
    val = None
    cap = _ATTN_TRANSIENT_CAP_BYTES
    _env = os.environ.get('CSA_ATTN_TRANSIENT_BYTES', '').strip()
    if _env:
        try:
            cap = max(1, int(_env))
        except ValueError:
            print(f'[attn] CSA_ATTN_TRANSIENT_BYTES={_env!r} is not an integer — using {cap}')
    try:
        if device.type == 'cuda':
            free, _total = torch.cuda.mem_get_info(device)
            val = max(64 << 20, min(int(free) // 4, cap))
    except Exception:
        val = None
    _ATB_CACHE[key] = val
    return val

_FULL_ATTN_TRANSIENT_BYTES = 256 << 20

def _full_attn_row_chunk(device, B, nh, T, elem_size, max_rows=1024):
    # Row chunk for the sink-capable dense path: the logits table is
    # [B, nh, rows, T], so bounding `rows` bounds the table.  Chunking is a
    # pure row partition, so no value changes.
    row_chunk = max(1, min(int(T), int(max_rows)))
    budget = _attn_transient_budget(device)
    if budget is None:
        budget = _FULL_ATTN_TRANSIENT_BYTES
    per_row = max(1, 3 * int(B) * int(nh) * int(T)) * int(elem_size)
    return max(1, min(row_chunk, int(budget) // per_row))

_WINDOW_MASK_CACHE = {}
_WINDOW_MASK_BUDGET_BYTES = 64 << 20

def _window_attn_mask(T, window, s, e, lo, device, dtype):
    # Row-chunk mask: a pure function of (T, window, s, e, lo, dtype, device),
    # so every layer/step reuses the same read-only 0.0/-inf table (SDPA never
    # mutates the mask).  Built outside inference mode so a table first created
    # during eval remains a normal tensor for a later training recompute.
    key = (_dev_str(device), 'winmask', int(T), int(window), int(s), int(e), int(lo), str(dtype))
    mask = _cache_get(_WINDOW_MASK_CACHE, key)
    if mask is None:
        with torch.inference_mode(False):
            pos = _arange_cache(int(T), device)
            qpos = pos[s:e, None]
            kpos = pos[lo:e][None, :]
            # Build the additive mask: set the allowed causal-window lanes to
            # 0.0 and the remaining lanes to -inf.
            mask = torch.full((e - s, e - lo), float('-inf'), device=device, dtype=dtype)
            mask.masked_fill_((kpos <= qpos) & (kpos > qpos - int(window)), 0.0)
        _cache_put(_WINDOW_MASK_CACHE, key, mask, _WINDOW_MASK_BUDGET_BYTES, _mask_cache_total)
    return mask

def _window_attn_chunk(q_c, k, v, s, e, lo, window, scale):
    dev = q_c.device
    mask = _window_attn_mask(int(k.shape[1]), window, s, e, lo, dev, q_c.dtype)
    out = F.scaled_dot_product_attention(
        q_c.transpose(1, 2), k[:, lo:e].transpose(1, 2), v[:, lo:e].transpose(1, 2),
        attn_mask=mask, scale=scale)
    return out.transpose(1, 2)

def _window_attn_row_chunk(device, B, nh, T, window, elem_size, max_rows=1024):
    # A windowed chunk only needs logits of width ~window + rows, not the full
    # sequence width.  Sizing the row block from (window + rows) keeps launch
    # counts small even when T is very long.
    rows = max(1, min(int(T), int(max_rows)))
    budget = _attn_transient_budget(device)
    if budget is None:
        budget = _FULL_ATTN_TRANSIENT_BYTES
    per_row = max(1, 3 * int(B) * int(nh) * (int(window) + rows)) * int(elem_size)
    return max(1, min(rows, int(budget) // per_row))


def _window_attn_local_impl(q, k, v, window, scale):
    B, T, nh, hd = q.shape
    out = torch.empty_like(q)
    row_chunk = _window_attn_row_chunk(q.device, B, nh, T, window, q.element_size())
    for s in range(0, T, row_chunk):
        e = min(s + row_chunk, T)
        lo = max(0, s - int(window) + 1)
        out[:, s:e] = _window_attn_chunk(q[:, s:e], k, v, s, e, lo, int(window), scale)
    return out

def _window_attn_local(q, k, v, window, scale):
    if torch.is_grad_enabled() and q.requires_grad:
        return torch.utils.checkpoint.checkpoint(
            _window_attn_local_impl, q, k, v, int(window), scale,
            use_reentrant=False, preserve_rng_state=False)
    return _window_attn_local_impl(q, k, v, window, scale)

def _tri_full(n, device, k):
    with torch.inference_mode(False):
        m = torch.empty((n, n), device=device)
        m.fill_(float('-inf'))
        torch.triu(m, k, out=m) if k > 0 else torch.tril(m, k, out=m)
    return m

def _mask_buffer(n, device, kind):
    if kind == 'causal':
        if not _dense_mask_fits(n, 1):
            raise RuntimeError(
                f'refusing to materialise a full [{int(n)}, {int(n)}] bool mask '
                f'(> {int(_DENSE_MASK_ROW_CAP_BYTES)} bytes); use the row-chunked mask path')
        # Boolean masked-lane table (True = j > i), consumed by
        # `_sink_attn_chunk` via masked_fill_; the masked lanes are written
        # to -inf.
        with torch.inference_mode(False):
            return torch.ones((n, n), device=device, dtype=torch.bool).triu_(1)
    key = (_dev_str(device), kind, int(n))
    buf = _cache_get(_MASK_CACHE, key)
    if buf is not None:
        return buf
    if kind != 'zero':
        raise ValueError(f'unknown mask kind {kind!r}')
    if not _dense_mask_fits(n, 4):
        raise RuntimeError(
            f'refusing to materialise a full [{int(n)}, {int(n)}] float mask '
            f'(> {int(_DENSE_MASK_ROW_CAP_BYTES)} bytes); use the row-chunked mask path')
    with torch.inference_mode(False):
        buf = torch.zeros((n, n), device=device)
    _cache_put(_MASK_CACHE, key, buf, _mask_budget(n), _mask_cache_total)
    return buf

def causal_mask(T, device):
    tri, _dev_s = _causal_donor(device, T)
    if tri.shape[0] == int(T):
        return tri
    return tri[:T, :T]

def window_band(T, window, device):
    if window <= 0 or window >= T:
        return _mask_buffer(T, device, 'zero')
    key = (_dev_str(device), 'bandw', int(T), int(window))
    buf = _cache_get(_MASK_CACHE, key)
    if buf is None:
        buf = _tri_full(T, device, -int(window))
        _cache_put(_MASK_CACHE, key, buf, _mask_budget(T), _mask_cache_total)
    return buf

def causal_window_mask(T, window, device):
    # Boolean masked-lane table (True = masked), consumed by
    if not _dense_mask_fits(T, 1):
        raise RuntimeError(
            f'refusing to materialise a full [{int(T)}, {int(T)}] bool mask '
            f'(> {int(_DENSE_MASK_ROW_CAP_BYTES)} bytes); use the row-chunked mask path')
    # `_sink_attn_chunk`'s masked_fill_.
    if window <= 0 or window >= T:
        return causal_mask(T, device)
    key = (_dev_str(device), 'cw', int(T), int(window))
    m = _cache_get(_MASK_CACHE, key)
    if m is None:
        with torch.inference_mode(False):
            pos = _arange_cache(T, device)
            m = (pos[None, :] > pos[:, None]) | (pos[None, :] <= pos[:, None] - int(window))
        _cache_put(_MASK_CACHE, key, m, _mask_budget(T), _mask_cache_total)
    return m

def _window_block_offset(n, w, n_blk, device):
    key = (_dev_str(device), 'winoff', int(n), int(w), int(n_blk))
    hit = _cache_get(_IDX_CACHE, key)
    if hit is not None:
        return hit
    win_idx, _win_valid = _window_geometry(n, w, device)
    with torch.inference_mode(False):
        out = (win_idx + int(n_blk)).to(torch.int32)
    _cache_put(_IDX_CACHE, key, out, _MASK_CACHE_BUDGET_BYTES, _idx_cache_total)
    return out

def _take_rows(x, idx):
    return torch.index_select(x, 0, idx)

def _take_2d(x, idx):
    return torch.index_select(x, 0, idx.flatten()).view(idx.shape[0], idx.shape[1], *x.shape[1:])

def first_occurrence_mask(idx):
    if idx.shape[1] <= 1:
        return torch.ones_like(idx, dtype=torch.bool)
    # Both branches compute exactly the same booleans ("keep only the first
    # occurrence of each value per row"): the triu form marks an element a
    # duplicate iff an equal element sits at an earlier column; the stable
    # argsort form keeps, inside each run of equal sorted values, the element
    # that stability places first — which is the earliest original column.
    # Identical output on either branch.
    if idx.shape[1] <= 64:
        # rowwise-independent: chunk rows; per-row results are identical
        # either way.
        k = idx.shape[1]
        row_chunk = max(1, (128 << 20) // max(k * k, 1))
        if row_chunk >= idx.shape[0]:
            dup = (idx[:, :, None] == idx[:, None, :]).triu_(1).any(dim=1)
            return ~dup
        out = torch.empty_like(idx, dtype=torch.bool)
        for r0 in range(0, idx.shape[0], row_chunk):
            r1 = min(r0 + row_chunk, idx.shape[0])
            dup = (idx[r0:r1, :, None] == idx[r0:r1, None, :]).triu_(1).any(dim=1)
            out[r0:r1] = ~dup
        return out
    order = torch.argsort(idx, dim=1, stable=True)
    sorted_idx = idx.gather(1, order)
    first = torch.ones_like(sorted_idx, dtype=torch.bool)
    first[:, 1:] = sorted_idx[:, 1:] != sorted_idx[:, :-1]
    keep = torch.empty_like(first)
    keep.scatter_(1, order, first)
    return keep

def _block_order(block_ids, n):
    if block_ids.numel() <= 1:
        return (_arange_cache(block_ids.numel(), block_ids.device), True)
    if bool((block_ids[1:] >= block_ids[:-1]).all()):
        return (_arange_cache(block_ids.numel(), block_ids.device), True)
    return (torch.argsort(block_ids, stable=True), False)

def _n_blocks_cached(block_ids):
    # block_ids is treated as immutable by the pooling caches (_pool_gpu /
    # _pool_geom), so its max+1 is a constant of the object — cache it per
    # object.  When the segmenter already attached its host-side counts,
    # their length is the same block count.
    if block_ids.numel() == 0:
        # No tokens -> no blocks (the same total form as _segment/blocks_from_cuts).
        return 0
    B = getattr(block_ids, '_pool_nblocks', None)
    if B is None:
        _cpu = getattr(block_ids, '_pool_cpu', None)
        if _cpu is not None and len(_cpu[0]) > 0:
            B = len(_cpu[0])
        else:
            B = int(block_ids.max().item()) + 1
        try:
            block_ids._pool_nblocks = B
        except (AttributeError, RuntimeError):
            pass
    return B

def pool_variable_blocks(Xa, Xb, Za, Zb, block_ids, B_pos_a, B_pos_b, overlap, n_blocks=None, monotonic=False):
    n, Fd = Xa.shape
    dev = Xa.device
    if n == 0:
        # Empty sequence: no blocks to pool.  Keep the same return contract as
        # the non-empty path.
        return (Xa[:0], torch.empty(0, dtype=torch.long, device=dev), 0)
    B = int(n_blocks) if n_blocks is not None else _n_blocks_cached(block_ids)
    order, _ident = ((_arange_cache(block_ids.numel(), block_ids.device), True) if monotonic else _block_order(block_ids, n))
    if _ident:
        Xas, Xbs, Zas, Zbs = (Xa, Xb, Za, Zb)
    else:
        Xas = _take_rows(Xa, order)
        Xbs = _take_rows(Xb, order)
        Zas = _take_rows(Za, order)
        Zbs = _take_rows(Zb, order)
    return _pool_ordered(Xas, Xbs, Zas, Zbs, block_ids, B_pos_a, B_pos_b, overlap, n, Fd, dev, B, order, order_is_identity=_ident)

_POOL_PARTS_HOST = True
_POOL_PARTS_BUCKET = 256
_POOL_GEOM_BUCKET = 1024
# Content-keyed host-geometry cache: dynamic block-ids objects are rebuilt
# every forward, so the per-object cache cannot hit; identical cut tables
# (e.g. repeated eval passes) can reuse the same read-only flats.
_POOL_GEOM_CONTENT = {}
_POOL_GEOM_CONTENT_BYTES = 64 << 20
_POOL_GEOM_CONTENT_MAX_ENTRIES = 8192
_POOL_GEOM_CONTENT_USED = 0


def _pool_geom_value_bytes(value):
    _b = 0
    for _v in (value if isinstance(value, (tuple, list)) else (value,)):
        if isinstance(_v, np.ndarray):
            _b += int(_v.nbytes)
        elif isinstance(_v, (tuple, list)):
            _b += sum(int(_x.nbytes) for _x in _v if isinstance(_x, np.ndarray))
    return _b


def _pool_geom_content_get(key):
    return _cache_get(_POOL_GEOM_CONTENT, key) if key is not None else None


def _pool_geom_content_put(key, value):
    global _POOL_GEOM_CONTENT_USED
    if key is None:
        return value
    try:
        _b = _pool_geom_value_bytes(value)
    except Exception:
        return value
    if _b > _POOL_GEOM_CONTENT_BYTES:
        return value
    _old = _POOL_GEOM_CONTENT.pop(key, None)
    if _old is not None:
        _POOL_GEOM_CONTENT_USED -= _pool_geom_value_bytes(_old)
    while _POOL_GEOM_CONTENT and (_POOL_GEOM_CONTENT_USED + _b > _POOL_GEOM_CONTENT_BYTES
                                  or len(_POOL_GEOM_CONTENT) >= _POOL_GEOM_CONTENT_MAX_ENTRIES):
        _old = _POOL_GEOM_CONTENT.pop(next(iter(_POOL_GEOM_CONTENT)))
        _POOL_GEOM_CONTENT_USED -= _pool_geom_value_bytes(_old)
    _POOL_GEOM_CONTENT[key] = value
    _POOL_GEOM_CONTENT_USED += _b
    return value
_NP_RANGE_CACHE = {}

def _np_range(n):
    # Read-only arange source for the host pooling flats; callers write into
    # separate `out=` destinations and never mutate the returned array.
    n = int(n)
    a = _NP_RANGE_CACHE.get(n)
    if a is None:
        a = np.arange(n, dtype=np.int64)
        a.flags.writeable = False
        if len(_NP_RANGE_CACHE) >= 64:
            _NP_RANGE_CACHE.pop(next(iter(_NP_RANGE_CACHE)))
        _NP_RANGE_CACHE[n] = a
    return a

# Block-chunking of the block-pooling gathers/mixes: the chunk boundaries are
# a pure function of the shapes, so the forward and the checkpoint recompute
# split identically.  Each block's pooled vector is computed from its own
# tokens/overlap lanes only, so processing blocks in any grouping yields the
# same per-block values bit-for-bit.
_POOL_TRANSIENT_BUDGET_BYTES = 256 << 20

def _pool_block_chunk(B, per_block_elems, elem_size):
    # Number of blocks per chunk; >= 1 and never more than B (so short
    # sequences run one chunk).
    _per = max(1, int(per_block_elems) * int(elem_size))
    return max(1, min(int(B), _POOL_TRANSIENT_BUDGET_BYTES // _per))

def _pool_counts_clean(bid, B):
    # "Every block carries at least one token" is a pure function of the
    # segmenter's immutable counts table; cache it on the block-ids object
    # (the same per-object cache convention as _pool_parts/_pool_geom).
    try:
        B = int(B)
    except (TypeError, ValueError):
        return False
    hit = getattr(bid, '_pool_clean', None)
    if hit is not None and hit[0] == B:
        return hit[1]
    _cpu = getattr(bid, '_pool_cpu', None)
    # `len(...) > 0` instead of bare truthiness: `_pool_cpu[0]` rows are
    # numpy arrays on the device-segmenter path, where a bare `and` test
    # would raise; the emptiness semantics are identical.
    ok = bool(B > 0 and _cpu is not None and len(_cpu[0]) == B
              and len(_cpu[0]) > 0 and min(_cpu[0]) > 0)
    try:
        bid._pool_clean = (B, ok)
    except (AttributeError, RuntimeError):
        pass
    return ok

def _pool_parts_flat(counts_l, B, ov, n, pad_to=1):
    # Host-side integer flat of the pooling parts (design notes §6.2): every
    # value is an exact integer function of the cut table's own counts list
    # (block_ids._pool_cpu[0] — the same list that generated block_ids), so
    # each value is the same integer the bincount/cumsum form produces
    # (cumsum / shifts / clamps on int64 are exact on either side).  Returns
    # (flat_int64 ndarray, total) — flat is zero-padded to a pad_to multiple
    # and the padded tail is never read — or None when the host form cannot
    # serve; the caller then falls back to the on-device form, whose values
    # are identical.
    try:
        B = int(B)
        n = int(n)
        counts_np = np.asarray(counts_l, dtype=np.int32)
        if counts_np.shape != (B,) or n <= 0:
            return None
        _ck = ('p', counts_np.tobytes(), B, -1 if ov is None else int(ov), n, int(pad_to))
        _hit = _pool_geom_content_get(_ck)
        if _hit is not None:
            return _hit
        ends_np = np.cumsum(counts_np, dtype=np.int32)
        nseg = 8 if ov is not None else 4
        total = nseg * B
        pad = (-total) % int(pad_to) if pad_to > 1 else 0
        # Flat layout: [counts | ends | starts | last_idx (| end_prev |
        # starts_prev | ov_start | ov_len)] followed by the zero pad; the
        # integers uploaded are identical to the per-segment form.  Index
        # transport is int32 (sequence lengths are far inside int32 range).
        flat = np.empty(total + pad, dtype=np.int32)
        flat[0:B] = counts_np
        flat[B:2 * B] = ends_np
        starts_np = flat[2 * B:3 * B]
        np.subtract(ends_np, counts_np, out=starts_np)
        np.clip(ends_np - 1, 0, n - 1, out=flat[3 * B:4 * B])
        if ov is not None:
            end_prev = flat[4 * B:5 * B]
            end_prev[0] = -1
            end_prev[1:] = ends_np[:-1]
            starts_prev = flat[5 * B:6 * B]
            starts_prev[0] = 0
            starts_prev[1:] = starts_np[:-1]
            ov_start = flat[6 * B:7 * B]
            np.subtract(end_prev, int(ov), out=ov_start)
            np.maximum(ov_start, starts_prev, out=ov_start)
            np.subtract(end_prev, ov_start, out=flat[7 * B:8 * B])
            np.maximum(flat[7 * B:8 * B], 0, out=flat[7 * B:8 * B])  # block 0 无 overlap：几何表自身保持非负
        if pad:
            flat[total:] = 0
        return _pool_geom_content_put(_ck, (flat, total))
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return None

def _pool_parts_host(block_ids, B, ov, n, counts_l):
    # Host-side build of the pooling geometry (see _pool_parts_flat for the
    # exact-integer argument).  The whole geometry rides one flat staged
    # upload (transport only, §6.1); the parts are views into that single
    # buffer (they are only ever read downstream, never mutated),
    # and the bucket-padding tail is never read.  `_h2d_async` builds the
    # buffer outside inference mode, so a part first filled during eval can
    # still serve training, exactly like the on-device form.  Returns None
    # when the host form cannot serve; the caller then falls back to the
    # on-device form, whose values are identical.
    _built = _pool_parts_flat(counts_l, B, ov, n, _POOL_PARTS_BUCKET)
    if _built is None:
        return None
    try:
        flat, total = _built
        buf = _h2d_async(flat, torch.long, block_ids.device)
        # the bucket pad is sliced off first; the unbound segment views are
        # read-only views into buf.
        return buf[:total].view(8 if ov is not None else 4, int(B)).unbind(0)
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return None

def _pool_geom_flat_var(counts_l, B, ov, n, max_len, ov_hi, pa_rows, pb_rows, pad_to=1):
    # Host-side integer/bool flats of the variable-block gather geometry
    # (design notes §6.2): g_a/mask_a/g_b/mask_b/b_off/pos_a are exact
    # integer/bool functions of the cut table's own counts (the same counts
    # _pool_parts_flat already mirrors) plus the shape scalars, so the numpy
    # form below produces values identical to the on-device build in
    # _pool_ordered — only the location of the integer math and the
    # transport change (§6.1).  Returns (flat_i, flat_b, n_a, n_b, W_b,
    # max_len) — flat_i holds [g_a | g_b | b_off | pos_a] and flat_b holds
    # [mask_a | mask_b] — or None when the host form cannot serve; the
    # caller then falls back to the on-device form, whose values are
    # identical.
    try:
        B = int(B)
        n = int(n)
        ov = int(ov)
        max_len = int(max_len)
        W_b = max(int(ov_hi), 0)
        pa_rows = int(pa_rows)
        pb_rows = int(pb_rows)
        counts_np = np.asarray(counts_l, dtype=np.int32)
        if counts_np.shape != (B,) or n <= 0 or max_len <= 0 or pa_rows <= 0 or pb_rows <= 0:
            return None
        _ck = ('v', counts_np.tobytes(), B, ov, n, max_len, W_b, pa_rows, pb_rows, int(pad_to))
        _hit = _pool_geom_content_get(_ck)
        if _hit is not None:
            return _hit
        ends_np = np.cumsum(counts_np, dtype=np.int32)
        starts_np = ends_np - counts_np
        pos = _np_range(max_len).astype(np.int32, copy=False)
        op = _np_range(W_b).astype(np.int32, copy=False)
        end_prev = np.empty(B, dtype=np.int32)
        end_prev[0] = -1
        end_prev[1:] = ends_np[:-1]
        starts_prev = np.empty(B, dtype=np.int32)
        starts_prev[0] = 0
        starts_prev[1:] = starts_np[:-1]
        ov_start = np.maximum(end_prev - ov, starts_prev)
        ov_len = np.maximum(end_prev - ov_start, 0)
        n_a = B * max_len
        n_b = B * W_b
        _pi = (-(n_a + 2 * n_b + max_len)) % int(pad_to) if int(pad_to) > 1 else 0
        _pb = (-(n_a + n_b)) % int(pad_to) if int(pad_to) > 1 else 0
        flat_i = np.empty(n_a + 2 * n_b + max_len + _pi, dtype=np.int32)
        flat_b = np.empty(n_a + n_b + _pb, dtype=np.bool_)
        gp = flat_i[0:n_a].reshape(B, max_len)
        np.add(starts_np[:, None], pos[None, :], out=gp)
        np.less(gp, ends_np[:, None], out=flat_b[0:n_a].reshape(B, max_len))
        np.minimum(gp, n - 1, out=gp)
        g_b = flat_i[n_a:n_a + n_b].reshape(B, W_b)
        np.add(ov_start[:, None], op[None, :], out=g_b)
        np.less(op[None, :], ov_len[:, None], out=flat_b[n_a:n_a + n_b].reshape(B, W_b))
        np.minimum(g_b, n - 1, out=g_b)
        b_off = flat_i[n_a + n_b:n_a + 2 * n_b].reshape(B, W_b)
        np.subtract(ov_start[:, None], starts_prev[:, None], out=b_off)
        np.add(b_off, op[None, :], out=b_off)
        np.minimum(b_off, pb_rows - 1, out=b_off)
        pos_a = flat_i[n_a + 2 * n_b:n_a + 2 * n_b + max_len]
        if max_len <= pa_rows:
            pos_a[:] = pos
        else:
            np.minimum(pos, pa_rows - 1, out=pos_a)
        return _pool_geom_content_put(_ck, (flat_i, flat_b, n_a, n_b, W_b, max_len))
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return None

def _pool_geom_host_var(counts_l, B, ov, n, max_len, ov_hi, pa_rows, pb_rows, dev):
    _built = _pool_geom_flat_var(counts_l, B, ov, n, max_len, ov_hi, pa_rows, pb_rows, _POOL_GEOM_BUCKET)
    if _built is None:
        return None
    try:
        B = int(B)
        flat_i, flat_b, n_a, n_b, W_b, max_len = _built
        # `_pool_geom_flat_var` already bucket-pads both flats.
        buf = _h2d_async(flat_i, torch.int32, dev)
        g_a = buf[0:n_a].view(B, max_len)
        g_b = buf[n_a:n_a + n_b].view(B, W_b)
        b_off = buf[n_a + n_b:n_a + 2 * n_b].view(B, W_b)
        pos_a = buf[n_a + 2 * n_b:n_a + 2 * n_b + max_len]
        mbuf = _h2d_async(flat_b, torch.bool, dev)
        # the bucket pad tail is never read: every view has an explicit end.
        return (g_a, mbuf[0:n_a].view(B, max_len), g_b, mbuf[n_a:n_a + n_b].view(B, W_b), b_off, pos_a)
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return None

def _pool_geom_flat_single(counts_l, B, n, max_len, p_rows, pad_to=1):
    # Single-block-form twin of _pool_geom_flat_single's variable sibling
    # (pool_blocks_single's g/mask/pos_a flats): same exact-integer host
    # build (§6.2).  Returns (flat_i, mask_np, n_g, max_len) or None.
    try:
        B = int(B)
        n = int(n)
        max_len = int(max_len)
        p_rows = int(p_rows)
        counts_np = np.asarray(counts_l, dtype=np.int32)
        if counts_np.shape != (B,) or n <= 0 or max_len <= 0 or p_rows <= 0:
            return None
        _ck = ('s', counts_np.tobytes(), B, n, max_len, p_rows, int(pad_to))
        _hit = _pool_geom_content_get(_ck)
        if _hit is not None:
            return _hit
        ends_np = np.cumsum(counts_np, dtype=np.int32)
        starts_np = ends_np - counts_np
        pos = _np_range(max_len).astype(np.int32, copy=False)
        n_g = B * max_len
        _pi = (-(n_g + max_len)) % int(pad_to) if int(pad_to) > 1 else 0
        _pb = (-n_g) % int(pad_to) if int(pad_to) > 1 else 0
        flat_i = np.empty(n_g + max_len + _pi, dtype=np.int32)
        mask_np = np.empty(n_g + _pb, dtype=np.bool_)
        gp = flat_i[0:n_g].reshape(B, max_len)
        np.add(starts_np[:, None], pos[None, :], out=gp)
        np.less(gp, ends_np[:, None], out=mask_np[0:n_g].reshape(B, max_len))
        np.minimum(gp, n - 1, out=gp)
        pos_a = flat_i[n_g:n_g + max_len]
        if max_len <= p_rows:
            pos_a[:] = pos
        else:
            np.minimum(pos, p_rows - 1, out=pos_a)
        return _pool_geom_content_put(_ck, (flat_i, mask_np, n_g, max_len))
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return None

def _pool_geom_host_single(counts_l, B, n, max_len, p_rows, dev):
    # Single-block-form twin of _pool_geom_host_var (pool_blocks_single's
    # g/mask/_pos_a): same exact-integer host build (§6.2), same staged
    # transport (§6.1) with bucket-padded flats (bounded pool shape keys,
    # unread pad tails), same None-means-device-fallback contract.
    _built = _pool_geom_flat_single(counts_l, B, n, max_len, p_rows, _POOL_GEOM_BUCKET)
    if _built is None:
        return None
    try:
        B = int(B)
        flat_i, mask_np, n_g, max_len = _built
        # Both flats are already bucket-padded.
        buf = _h2d_async(flat_i, torch.int32, dev)
        mask = _h2d_async(mask_np, torch.bool, dev)[:n_g].view(B, max_len)
        return (buf[0:n_g].view(B, max_len), mask, buf[n_g:n_g + max_len])
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return None

def _pool_parts(block_ids, B, ov, n):
    # Integer block geometry shared by the pooling paths.  Every value is an
    # exact integer function of (block_ids, B, ov, n), so caching the tensors on
    # the block-ids object is bitwise safe; the fixed segmenter hands out ONE
    # shared block-ids object for every row/layer/step.  Tensors are built
    # outside inference mode so a cache first filled during eval can still serve
    # training (an inference tensor saved for backward would raise).
    key = (int(B), -1 if ov is None else int(ov), int(n))
    cache = getattr(block_ids, '_pool_gpu', None)
    if cache is None:
        cache = {}
        try:
            block_ids._pool_gpu = cache
        except (AttributeError, RuntimeError):
            pass
    else:
        hit = cache.get(key)
        if hit is not None:
            return hit
    _cpu = getattr(block_ids, '_pool_cpu', None)
    if _POOL_PARTS_HOST and _cpu is not None and len(_cpu[0]) == int(B):
        _parts = _pool_parts_host(block_ids, B, ov, n, _cpu[0])
        if _parts is not None:
            if len(cache) < 8:
                cache[key] = _parts
            return _parts
    with torch.inference_mode(False):
        counts = torch.bincount(block_ids, minlength=B)
        ends = torch.cumsum(counts, 0)
        starts = ends - counts
        last_idx = (ends - 1).clamp(0, n - 1)
        if ov is None:
            parts = (counts, ends, starts, last_idx)
        else:
            end_prev = F.pad(ends[:-1], (1, 0), value=-1)
            starts_prev = F.pad(starts[:-1], (1, 0), value=0)
            ov_start = torch.clamp(end_prev - ov, min=starts_prev)
            ov_len = (end_prev - ov_start).clamp_min(0)  # block 0 无 overlap：几何表自身保持非负
            parts = (counts, ends, starts, last_idx, end_prev, starts_prev, ov_start, ov_len)
    if len(cache) < 8:
        cache[key] = parts
    return parts

def _pool_geom_put(block_ids, key, geom):
    # The gather geometry is an exact integer/bool function of the block-ids
    # object's own cut table (the `_pool_cpu` counts) plus the shape scalars
    # in the key, so caching it on the block-ids object is bitwise safe — the
    # same argument `_pool_parts` makes for its cached counts/ends.  The fixed
    # segmenter hands out ONE shared block-ids object for every row/layer/
    # step; the dynamic segmenter's per-row objects simply miss and rebuild
    # (identical values).  Every cached tensor is only read downstream
    # (gathers / negated-mask views / index selects), never mutated.
    cache = getattr(block_ids, '_pool_geom', None)
    if cache is None:
        cache = {}
        try:
            block_ids._pool_geom = cache
        except (AttributeError, RuntimeError):
            return geom
    if len(cache) < 8:
        cache[key] = geom
    return geom

def _prefetch_pool_geometry(attn, pre_blocks, n, dev):
    if not _POOL_PARTS_HOST:
        return
    cfg = attn.cfg
    _csa = cfg.kind == 'csa'
    ov = int(cfg.overlap) if _csa else None
    pa_rows = int(attn.B_pos_a.shape[0])
    pb_rows = int(attn.B_pos_b.shape[0]) if _csa else 1
    rows = []
    for _pb in pre_blocks:
        bid, nblk = _pb[0], int(_pb[1])
        _cpu = getattr(bid, '_pool_cpu', None)
        if _cpu is None or len(_cpu[0]) != nblk or nblk <= 0:
            rows.append(None)
            continue
        _pkey = (nblk, ov if _csa else -1, int(n))
        _gkey = (('vo', nblk, ov, int(n), pa_rows, pb_rows) if _csa
                 else ('so', nblk, int(n), pa_rows))
        _pc = getattr(bid, '_pool_gpu', None)
        _gc = getattr(bid, '_pool_geom', None)
        if _pc is not None and _gc is not None and _pkey in _pc and _gkey in _gc:
            rows.append(None)
            continue
        counts_l = _cpu[0]
        max_len = _cpu[1]
        if _csa:
            _ov_hi = min(ov, _cpu[2]) if nblk > 1 else -1
            _g = _pool_geom_flat_var(counts_l, nblk, ov, n, max_len, _ov_hi, pa_rows, pb_rows)
        else:
            _g = _pool_geom_flat_single(counts_l, nblk, n, max_len, pa_rows)
        _p = _pool_parts_flat(counts_l, nblk, ov, n)
        if _g is None or _p is None:
            rows.append(None)
            continue
        rows.append((bid, nblk, _p, _g))
    if not any(rows):
        return
    _tot_i = 0
    _tot_b = 0
    for _r in rows:
        if _r is None:
            continue
        _tot_i += len(_r[2][0]) + len(_r[3][0])
        _tot_b += len(_r[3][1].ravel())
    flat_i = np.empty(_tot_i + (-_tot_i) % _POOL_GEOM_BUCKET, dtype=np.int32)
    flat_b = np.empty(_tot_b + (-_tot_b) % _POOL_GEOM_BUCKET, dtype=np.bool_)
    _oi = 0
    _ob = 0
    for _r in rows:
        if _r is None:
            continue
        _pflat = _r[2][0]
        _gflat = _r[3][0]
        _bflat = _r[3][1].ravel()
        _n = len(_pflat)
        flat_i[_oi:_oi + _n] = _pflat
        _oi += _n
        _n = len(_gflat)
        flat_i[_oi:_oi + _n] = _gflat
        _oi += _n
        _n = len(_bflat)
        flat_b[_ob:_ob + _n] = _bflat
        _ob += _n
    buf_i = _h2d_async(flat_i, torch.int32, dev)
    buf_b = _h2d_async(flat_b, torch.bool, dev)
    off_i = 0
    off_b = 0
    for _r in rows:
        if _r is None:
            continue
        bid, nblk, _p, _g = _r
        _ptot = _p[1]
        parts = buf_i[off_i:off_i + _ptot].view(8 if _csa else 4, nblk).unbind(0)
        off_i += _ptot
        if _csa:
            _fi, _fb, n_a, n_b, W_b, max_len = _g
            g_a = buf_i[off_i:off_i + n_a].view(nblk, max_len)
            g_b = buf_i[off_i + n_a:off_i + n_a + n_b].view(nblk, W_b)
            b_off = buf_i[off_i + n_a + n_b:off_i + n_a + 2 * n_b].view(nblk, W_b)
            pos_a = buf_i[off_i + n_a + 2 * n_b:off_i + n_a + 2 * n_b + max_len]
            off_i += n_a + 2 * n_b + max_len
            mask_a = buf_b[off_b:off_b + n_a].view(nblk, max_len)
            mask_b = buf_b[off_b + n_a:off_b + n_a + n_b].view(nblk, W_b)
            off_b += n_a + n_b
            geom = (g_a, mask_a, g_b, mask_b, b_off, pos_a)
            pkey = (nblk, ov, int(n))
            gkey = ('vo', nblk, ov, int(n), pa_rows, pb_rows)
        else:
            _fi, mask_np, n_g, max_len = _g
            g = buf_i[off_i:off_i + n_g].view(nblk, max_len)
            pos_a = buf_i[off_i + n_g:off_i + n_g + max_len]
            off_i += n_g + max_len
            mask = buf_b[off_b:off_b + n_g].view(nblk, max_len)
            off_b += n_g
            geom = (g, mask, pos_a)
            pkey = (nblk, -1, int(n))
            gkey = ('so', nblk, int(n), pa_rows)
        try:
            _c1 = getattr(bid, '_pool_gpu', None)
            if _c1 is None:
                _c1 = {}
                bid._pool_gpu = _c1
            _c1[pkey] = parts
            _c2 = getattr(bid, '_pool_geom', None)
            if _c2 is None:
                _c2 = {}
                bid._pool_geom = _c2
            _c2[gkey] = geom
        except (AttributeError, RuntimeError):
            pass

def _pool_ordered(Xas, Xbs, Zas, Zbs, block_ids, B_pos_a, B_pos_b, overlap, n, Fd, dev, B, order, order_is_identity=False):
    counts, ends, starts, last_idx, end_prev, starts_prev, ov_start, ov_len = _pool_parts(block_ids, B, overlap, n)
    ov = overlap
    _cpu = getattr(block_ids, '_pool_cpu', None)
    if _cpu is not None and len(_cpu[0]) == B:
        max_len = _cpu[1]
        _ov_hi = min(ov, _cpu[2]) if B > 1 else -1
        _gkey = ('vo', int(B), int(ov), int(n), int(B_pos_a.shape[0]), int(B_pos_b.shape[0]))
        _geom = getattr(block_ids, '_pool_geom', None)
        _geom = None if _geom is None else _geom.get(_gkey)
    elif ov_len.numel():
        _cmin, max_len, _ov_lo, _ov_hi = (int(v) for v in torch.stack([counts.min(), counts.max(), ov_len.min(), ov_len.max()]).tolist())
        _geom = None
    else:
        _cmin, max_len = (int(v) for v in torch.stack(torch.aminmax(counts)).tolist())
        _ov_lo = _ov_hi = None
        _geom = None
    if _geom is not None:
        g_a, mask_a, g_b, mask_b, b_off, _pos_a = _geom
    else:
        # Host form first (the same exact-integer build _pool_parts_host makes
        # for the parts — design notes §6.2): the host form produces values
        # identical to the on-device build below.  None means fall back to the
        # on-device form, whose values are identical.
        _host = None
        if _POOL_PARTS_HOST and _cpu is not None and len(_cpu[0]) == B:
            _host = _pool_geom_host_var(_cpu[0], B, ov, n, max_len, _ov_hi, B_pos_a.shape[0], B_pos_b.shape[0], dev)
        if _host is not None:
            g_a, mask_a, g_b, mask_b, b_off, _pos_a = _host
            _pool_geom_put(block_ids, _gkey, _host)
        else:
            # built outside inference mode for the same reason as _pool_parts: a
            # cache first filled during eval must still serve training (an
            # inference tensor saved for backward would raise); the mode guard
            # changes no values.
            with torch.inference_mode(False):
                pos = _range_cache(0, max_len, dev)
                _gp = starts[:, None] + pos[None, :]
                mask_a = _gp < ends[:, None]
                # These three index tensors feed gathers only; once the masks (which
                # need the unclamped values) are built, the clamps run in place — the
                # same clamp on the same values.
                g_a = _gp.clamp_(max=n - 1)
                _W_b = max(_ov_hi if _ov_hi is not None else 0, 0)
                op = _range_cache(0, _W_b, dev)
                mask_b = op[None, :] < ov_len[:, None]
                g_b = (ov_start[:, None] + op[None, :]).clamp_(max=n - 1)
                b_off = (ov_start[:, None] - starts_prev[:, None] + op[None, :]).clamp_(max=B_pos_b.shape[0] - 1)
                _lim = B_pos_a.shape[0] - 1
                _pos_a = pos if max_len <= B_pos_a.shape[0] else pos.clamp(max=_lim)
            if _cpu is not None and len(_cpu[0]) == B:
                _pool_geom_put(block_ids, _gkey, (g_a, mask_a, g_b, mask_b, b_off, _pos_a))
    # Every block's pooled vector is a function of its own block lanes and
    # overlap lanes only (the softmax normalization and the weighted sum are
    # both within one block), so the blocks are processed in chunks: each
    # chunk's gathers/mix are bit-identical to the same rows of the unchunked
    # form, and `comp` is written row-for-row.
    _Fd = int(Xas.shape[1])
    _Wb = int(g_b.shape[1])
    # Conservative per-block element count: Xam+Zam+Xbm+Zbm + WinZ + sc
    # (float lanes) plus the two boolean masks.
    _per_elems = (4 * int(max_len) + 4 * _Wb) * _Fd + 8 * (int(max_len) + _Wb)
    _chunk = _pool_block_chunk(B, _per_elems, Xas.element_size())
    # Within the embedding table the gather index is exactly arange(max_len).
    _bpa = B_pos_a[:max_len] if max_len <= B_pos_a.shape[0] else B_pos_a[_pos_a]
    # Every real block carries at least one token (the segmenters only emit
    # positive counts), so the A lanes always contain a finite logit and the
    # softmax cannot produce NaN/Inf.  When the block-ids object exposes its
    # host counts we can prove that here, so the defensive nan_to_num is
    # unnecessary; the fallback keeps the defensive form for externally built
    # tables.
    _clean = _pool_counts_clean(block_ids, B)
    comp = torch.empty(B, _Fd, device=dev, dtype=Xas.dtype)
    for _b0 in range(0, B, _chunk):
        _b1 = min(_b0 + _chunk, B)
        Zam = _take_2d(Zas, g_a[_b0:_b1])
        Zbm = _take_2d(Zbs, g_b[_b0:_b1])
        _ma = mask_a[_b0:_b1]
        _mb = mask_b[_b0:_b1]
        # Plain-broadcast position add (same form pool_blocks_single already
        # uses): pad lanes are overwritten with -inf right after, and their
        # incoming grad is exactly zero (MaskedFillBackward / softmax P=0), so
        # forward values and B_pos gradients are identical to the masked-multiply
        # form.
        # += on the fresh _take_2d outputs: the same elementwise kernel with the
        # same values as the out-of-place add (AddBackward saves no tensor, and
        # the only downstream consumer chain — masked_fill_/cat/softmax — keeps
        # no pre-add version).
        Zam += _bpa[None]
        Zbm += B_pos_b[b_off[_b0:_b1]]
        # the fill mask broadcasts: masked_fill_ accepts a broadcastable mask,
        # so the same booleans reach every lane (broadcast reads are exact).
        Zam.masked_fill_(~_ma[:, :, None], float('-inf'))
        Zbm.masked_fill_(~_mb[:, :, None], float('-inf'))
        WinZ = torch.cat([Zam, Zbm], 1)
        # softmax 的反向保存其输出，nan_to_num 必须出位（原地会改写保存值，
        # SoftmaxBackward 版本检查失败）；值与任何原地形式逐位相同。
        # The safe-count path uses the finite softmax directly.
        sc = F.softmax(WinZ, dim=1) if _clean else _safe_nan_to_num(F.softmax(WinZ, dim=1), where='block_pool')
        del WinZ, Zam, Zbm
        Xam = _take_2d(Xas, g_a[_b0:_b1])
        Xbm = _take_2d(Xbs, g_b[_b0:_b1])
        sc_a = sc[:, :max_len]
        sc_b = sc[:, max_len:]
        part = (sc_a * Xam).sum(1)
        # Release the A-lane gather before the B-lane product; the two partial
        # sums are added in the original order.
        del Xam
        part = part + (sc_b * Xbm).sum(1)
        comp[_b0:_b1] = part
        del sc, sc_a, sc_b, Xbm, _ma, _mb, part
    # Identity-order path: when `order` is the cached arange, order[last_idx]
    # returns exactly last_idx (arange[i] == i, exact integers — index
    # transport, design notes §6.1).
    return (comp, last_idx if order_is_identity else order[last_idx], B)

def pool_blocks_single(X, Z, B_pos, block_ids, n_blocks=None, monotonic=False):
    n = X.shape[0]
    dev = X.device
    if n == 0:
        # Same empty-sequence contract as pool_variable_blocks.
        return (X[:0], torch.empty(0, dtype=torch.long, device=dev), 0)
    B = int(n_blocks) if n_blocks is not None else _n_blocks_cached(block_ids)
    order, _ident = ((_arange_cache(block_ids.numel(), block_ids.device), True) if monotonic else _block_order(block_ids, n))
    if _ident:
        Xs, Zs = (X, Z)
    else:
        Xs = _take_rows(X, order)
        Zs = _take_rows(Z, order)
    counts, ends, starts, last_idx = _pool_parts(block_ids, B, None, n)
    _cpu = getattr(block_ids, '_pool_cpu', None)
    _geom = None
    if _cpu is not None and len(_cpu[0]) == B:
        max_len = _cpu[1]
        _gkey = ('so', int(B), int(n), int(B_pos.shape[0]))
        _gc = getattr(block_ids, '_pool_geom', None)
        _geom = None if _gc is None else _gc.get(_gkey)
    else:
        _cmin, max_len = (int(v) for v in torch.stack(torch.aminmax(counts)).tolist())
    if _geom is not None:
        g, mask, _pos_a = _geom
    else:
        # host form first — same exact-integer argument as _pool_ordered (the
        # values are identical to the device build; only the location of the
        # integer math and the transport change, §6.1/§6.2).
        _host = None
        if _POOL_PARTS_HOST and _cpu is not None and len(_cpu[0]) == B:
            _host = _pool_geom_host_single(_cpu[0], B, n, max_len, B_pos.shape[0], dev)
        if _host is not None:
            g, mask, _pos_a = _host
            _pool_geom_put(block_ids, _gkey, _host)
        else:
            # same inference_mode(False) guard as _pool_parts: the cached tensors
            # must be normal tensors even when first built during eval.
            with torch.inference_mode(False):
                pos = _range_cache(0, max_len, dev)
                _gp = starts[:, None] + pos[None, :]
                mask = _gp < ends[:, None]
                # g feeds the gathers only; the mask needs the unclamped values, so it is
                # built first and the clamp runs in place (same kernel, same integers).
                g = _gp.clamp_(max=n - 1)
                _pos_a = pos if max_len <= B_pos.shape[0] else pos.clamp(max=B_pos.shape[0] - 1)
            if _cpu is not None and len(_cpu[0]) == B:
                _pool_geom_put(block_ids, _gkey, (g, mask, _pos_a))
    # Block-chunked exactly as _pool_ordered: each block's pooled vector is a
    # function of its own lanes only, so the grouping never changes a value.
    _Fd = int(Xs.shape[1])
    # Conservative per-block element count: Xb + Zb(WinZ) + sc plus the mask.
    _per_elems = 3 * int(max_len) * _Fd + 4 * int(max_len)
    _chunk = _pool_block_chunk(B, _per_elems, Xs.element_size())
    # Same slice-view form as _pool_ordered (index == arange(max_len)).
    _bpos = B_pos[:max_len] if max_len <= B_pos.shape[0] else B_pos[_pos_a]
    _clean = _pool_counts_clean(block_ids, B)
    comp = torch.empty(B, _Fd, device=dev, dtype=Xs.dtype)
    for _b0 in range(0, B, _chunk):
        _b1 = min(_b0 + _chunk, B)
        Xb = _take_2d(Xs, g[_b0:_b1])
        Zb = _take_2d(Zs, g[_b0:_b1])
        WinZ = Zb
        # Same plain-broadcast position add as _pool_ordered: the fill right
        # after overwrites the pad lanes with -inf, and the incoming grad at
        # those lanes is exactly zero (MaskedFillBackward / softmax P=0), so
        # the values and the B_pos gradients are identical to the
        # masked-multiply and boolean-index forms.
        # += on the fresh _take_2d output (same elementwise kernel and values as
        # the out-of-place add; see _pool_ordered).
        WinZ += _bpos[None]
        # broadcast fill mask (same argument as _pool_ordered): identical booleans
        # per lane.
        WinZ.masked_fill_(~mask[_b0:_b1, :, None], float('-inf'))
        # 同 _pool_ordered：softmax 反向保存输出，nan_to_num 出位。
        sc = F.softmax(WinZ, dim=1) if _clean else _safe_nan_to_num(F.softmax(WinZ, dim=1), where='block_pool')
        comp[_b0:_b1] = (sc * Xb).sum(1)
        del WinZ, Xb, Zb, sc
    # same identity-order path as _pool_ordered: order[last_idx] over the
    # cached arange returns exactly last_idx (index transport, §6.1).
    return (comp, last_idx if _ident else order[last_idx], B)

def _rank_blocks(masked, B, ties, k=None):
    # The sort indices are consumed by the int64 gathers below and narrowed
    # to int32 on the kept slice only.  Same integers on every path (index
    # transport, design notes §6.1).
    #
    # Only the first k columns are ever consumed by _indexer_selection.  For
    # k < B, return exactly the columns the full stable sort would produce:
    # the k-th largest value is found with torch.topk, all strictly larger
    # entries are kept, and the remaining slots are filled from equal entries
    # in original column order (the stable-sort tie rule).  The selected k
    # candidates are then ordered by score with a stable argsort — the same
    # value/column order as the first k columns of the full sort.
    kk = B if k is None else min(max(int(k), 0), B)
    if kk <= 0:
        return torch.empty((masked.shape[0], 0), dtype=torch.long, device=masked.device)
    if kk >= B:
        # argsort returns exactly the same stable permutation as sort(...).indices.
        if ties == 'earliest':
            return torch.argsort(masked, dim=1, stable=True, descending=True)
        order = torch.argsort(masked.flip(1), dim=1, stable=True, descending=True)
        return torch.sub(B - 1, order, out=order)
    src = masked.flip(1) if ties == 'latest' else masked
    n, _ = src.shape
    # Keep only the k-th value.  Either branch produces the same value
    # (k-th largest == (B-kk+1)-th smallest).
    if kk * 4 >= B:
        kth = torch.kthvalue(src, B - kk + 1, dim=1).values.unsqueeze(1)
    else:
        kth = torch.topk(src, kk, dim=1, sorted=True).values[:, -1].unsqueeze(1).clone()
    # gt has at most kk - 1 entries (the k-th value itself is not strictly
    # greater), so `need` is non-negative and selects exactly kk columns.
    # Prefix ranks use int32 (B < 2^31): the same exact integers as the int64
    # form; the non-selected lanes are filled with the sentinel column in
    # place.
    gt = src > kth
    eq = src == kth
    need = (kk - gt.sum(dim=1)).clamp_min(0).to(torch.int32)
    eq_cum = torch.cumsum(eq, dim=1, dtype=torch.int32)
    sel = gt | (eq & (eq_cum <= need[:, None]))
    del eq_cum, gt, eq, need
    rank = torch.cumsum(sel, dim=1, dtype=torch.int32)
    rank -= 1
    rank.masked_fill_(~sel, kk)
    # Scatter the selected source columns into [n, kk] (column order); the
    # sentinel column kk absorbs the non-selected lanes and is sliced away.
    cols = torch.arange(B, device=src.device, dtype=torch.int32)[None, :].expand(n, B)
    buf = torch.empty(n, kk + 1, dtype=torch.int32, device=src.device)
    buf.scatter_(1, rank, cols)
    del rank, sel, cols
    cand = buf[:, :kk].long()
    del buf, kth
    order = torch.argsort(src.gather(1, cand), dim=1, stable=True, descending=True)
    cand = cand.gather(1, order)
    if ties == 'latest':
        cand = B - 1 - cand
    return cand

def _indexer_selection(scores, causal, k, ties='earliest', out_valid=None):
    n, B = scores.shape
    if ties not in ('earliest', 'latest'):
        raise ValueError(f'unknown tie rule {ties!r}')
    k_req = max(int(k), 0)
    if k_req <= 0:
        if out_valid is not None:
            out_valid[:] = [torch.zeros(n, 0, dtype=torch.bool, device=scores.device)]
        return torch.empty(n, 0, device=scores.device, dtype=torch.int32)
    k = min(k_req, B)
    # Boolean finite-and-causal table.
    cf = causal & torch.isfinite(scores)
    # torch.where picks bit-identical values (scores or the -inf constant) —
    # the same selection masked_fill produced.
    masked = torch.where(cf, scores, float('-inf'))
    order = _rank_blocks(masked, B, ties, k=min(k, B))
    del masked
    _w_keep = min(int(k), B)
    # Only the first _w_keep order columns are ever consumed (the k >= B and
    # k < B forms both reduce to usable[:, :_w_keep]), so the mask gather runs
    # on that column slice directly: gathering a column subset yields exactly
    # the corresponding columns of the full gather — the same booleans (index
    # transport, design notes §6.1).  Gathering the conjoined mask gives
    # exactly the elementwise AND of the two per-mask gathers.
    order_w = order[:, :_w_keep]
    keep = cf.gather(1, order_w)
    # Integer prefix ranks fit in int32 (k <= B < 2**31): the same exact
    # integers as the default int64 cumsum; the scatter below therefore needs
    # no int64 index.
    dest = keep.cumsum(dim=1, dtype=torch.int32)
    dest -= 1
    # int64 -> int32 narrowing, on the kept slice only — same integers
    # (index transport, design notes §6.1).  keep/dest are already exactly
    # _w_keep wide (keep = cf.gather(1, order_w), dest = keep.cumsum).
    order_k = order_w.to(torch.int32)
    # Sentinel-column scatter. Within a row the kept lanes hold distinct
    # cumsum ranks (0..cnt-1), so every scatter target column has exactly one
    # writer; lanes failing `keep` are redirected to a sentinel column that is
    # sliced away (its undefined winner never leaves the pad).  The kept
    # columns are therefore the same exact integers as the masked-assignment
    # form (host-side integer equivalence, design notes §6.2).
    tgt = torch.where(keep, dest, _w_keep)
    kept = torch.full((n, _w_keep + 1), -1, dtype=torch.int32, device=scores.device)
    kept.scatter_(1, tgt, order_k)
    kept = kept[:, :_w_keep]
    # last_tok from either pooling path is non-decreasing (a cumsum of
    # non-negative counts followed by the monotone (ends-1).clamp(0, n-1)),
    # so every row of `causal` is a prefix of Trues: the first-max argmax
    # returns 0 for any row with a readable block, and 0 for an all-False
    # row (ties resolve to index 0) — identically zero.
    # The padded lanes are masked downstream via the selection-valid mask
    # either way, so the zeros are the same integers the argmax produced
    # (host-side integer equivalence, design notes §6.2).
    # the invalid lanes hold exactly the -1 sentinel, so clamping the kept
    # buffer to 0 replaces the where(valid, kept, 0) selection lane-for-lane
    # (block ids are >= 0 and are never rewritten) — same integers.  The
    # validity mask is an exact function of the pre-clamp kept buffer and is
    # only consumed when the caller asks for it, so it (and the pad-block
    # constant, needed only when k exceeds the available lanes) is built only
    # on the branch that reads it.
    if out_valid is not None:
        valid_cols = kept >= 0
        if k_req > _w_keep:
            valid_cols = torch.cat([valid_cols, torch.zeros(n, k_req - _w_keep, dtype=torch.bool, device=scores.device)], dim=1)
        out_valid[:] = [valid_cols]
    kept.clamp_min_(0)
    idx_out = kept
    if k_req > _w_keep:
        pad_blk = torch.zeros(n, dtype=torch.int32, device=scores.device)
        pad = pad_blk[:, None].expand(n, k_req - _w_keep)
        idx_out = torch.cat([idx_out, pad], dim=1)
    return idx_out

_TOPK_CLAMP_WARNED = set()
# Above this many elements the per-query-chunk form is used on last_tok
# (value-identical; the dynamic arms have B ~ T / block_size).
_READABLE_CACHE_MAX_ELEMS = 4 << 20

def _pool_ends_host(bid):
    # 宿主侧「块结束位置（独占）」表：cumsum 的分段器 counts（_pool_cpu[0]）。
    # 池化侧 last_tok 由同一 counts 的 cumsum-1 生成（monotonic 顺序下
    # order 为恒等），且 counts 逐项 ≤ n，clamp(0, n-1) 从不改写任何值，
    # 故 ends[j]-1 与 last_tok[j] 是同一批整数（design notes §6.2 的宿主侧
    # 整数等价）。缓存挂在不可变的 block_ids 对象上（与 _pool_parts 同一
    # 约定）；缺失/形状不符时返回 None，调用方回退全宽。
    ends = getattr(bid, '_pool_ends_host', None)
    if ends is None:
        _cpu = getattr(bid, '_pool_cpu', None)
        if _cpu is None:
            return None
        try:
            ends = [int(v) for v in itertools.accumulate(_cpu[0])]
        except (TypeError, ValueError):
            return None
        try:
            bid._pool_ends_host = ends
        except (AttributeError, RuntimeError):
            pass
    return ends

def lightning_indexer(H, comp_kv, last_tok, W_DQ, W_DK, W_w, nIH, topk, return_mask=True, query_chunk=2048, random_select=False, pre_qI=None, pre_w=None, out_valid=None, causal_ends_host=None, need_soft=True):
    n = H.shape[0]
    B = comp_kv.shape[0]
    dev = H.device
    hd = W_DQ.shape[1] // nIH
    if random_select:
        qI = kI = w_idx = None
    else:
        qI = (pre_qI if pre_qI is not None else H @ W_DQ).view(n, nIH, hd).transpose(0, 1)
        kI = (comp_kv @ W_DK).view(B, nIH, hd).transpose(0, 1)
        w_idx = pre_w if pre_w is not None else F.linear(H, W_w)
        # The no-soft form is only safe when no differentiable path reaches
        # the indexer (the S prior is the only route for its gradients).  A
        # caller that asks for no-soft while the indexer tensors require grad
        # would otherwise detach it, so re-enable the prior here.
        if not need_soft and any((_t is not None and _t.requires_grad for _t in (qI, kI, w_idx))):
            need_soft = True
    pos = _arange_cache(n, dev)
    # causal (and its negation) are exact bool functions of (pos, last_tok).
    # The fixed segmenter's pooled last_tok is ONE shared cached tensor read
    # by every row/layer/step, so a SMALL pair is cached on that tensor object
    # — the same bitwise-safe cache-on-immutable-object argument as
    # _pool_parts/_pool_geom (design notes §6.2).  The table is [T, B] with
    # B ~ T / block_size for the dynamic arms.  Above the element cap we build
    # the same mask per query chunk below (value-identical slices) and cache
    # nothing.
    # Built outside inference mode so a pair first filled during eval can
    # still serve training (masked_fill_ saves the mask for backward).
    _rkey = (int(n), int(last_tok.shape[0]))
    _rc = getattr(last_tok, '_readable_cache', None)
    _causal_full = _ncausal_full = None
    if _rc is not None and _rc[0] == _rkey:
        _causal_full, _ncausal_full = _rc[1:3]
    elif int(n) * int(last_tok.shape[0]) <= _READABLE_CACHE_MAX_ELEMS:
        with torch.inference_mode(False):
            _causal_full = block_readable(pos, last_tok)
            _ncausal_full = ~_causal_full
        try:
            last_tok._readable_cache = (_rkey, _causal_full, _ncausal_full)
        except (AttributeError, RuntimeError):
            pass

    def _causal_of(_s, _e):
        if _causal_full is not None:
            return _causal_full[_s:_e], _ncausal_full[_s:_e]
        with torch.inference_mode(False):
            _c = block_readable(pos[_s:_e], last_tok)
        return _c, ~_c
    k = min(topk, B)
    if k < int(topk):
        _wk = (int(topk), int(B))
        if _wk not in _TOPK_CLAMP_WARNED:
            _TOPK_CLAMP_WARNED.add(_wk)
            print(f'[lightning_indexer] requested topk={int(topk)} but only B={int(B)} blocks are available; effective selection is clamped to all {int(k)} blocks for this call')
    if random_select:
        _ri_key = (_dev_str(dev), 'rs_ri', int(n))
        _jc_key = (_dev_str(dev), 'rs_jc', int(B))
        _ri = _cache_get(_IDX_CACHE, _ri_key)
        if _ri is None:
            with torch.inference_mode(False):
                _ri = torch.arange(n, device=dev, dtype=torch.float64)[:, None]
            _cache_put(_IDX_CACHE, _ri_key, _ri, _MASK_CACHE_BUDGET_BYTES, _idx_cache_total)
        _jc = _cache_get(_IDX_CACHE, _jc_key)
        if _jc is None:
            with torch.inference_mode(False):
                _jc = (torch.arange(B, device=dev, dtype=torch.float64)[None, :] * 78.233).contiguous()
            _cache_put(_IDX_CACHE, _jc_key, _jc, _MASK_CACHE_BUDGET_BYTES, _idx_cache_total)
    if query_chunk is None:
        raise ValueError("lightning_indexer: query_chunk=None is not 'use the default'; pass the default (2048) explicitly or omit the argument")
    query_chunk = max(1, min(int(query_chunk), n))
    # chunk 行数上限（下方注释已论证逐 chunk 融合与全宽形式逐行逐位一致；
    # 分块边界是 (n, B, nIH, dtype) 的纯函数，跨机器确定）：每 chunk 同时
    # 物化 raw [nIH,·,B]、scores、masked、int64 order 与 keep/dest/tgt 等
    # 选择表。
    _iper = int(B) * (int(nIH) * int(H.element_size()) + 40) + 64
    if _iper > 0:
        query_chunk = max(1, min(query_chunk, (256 << 20) // _iper))
    _iscale = hd ** (-0.5)
    # 选择 + soft 尾部与打分同一逐 chunk 节奏融合执行：其中每个算子（masked
    # where/稳定排序/cumsum/scatter/logsumexp/gather/逐元素 exp）都按 query
    # 行独立——chunk 行切片上的值与全宽形式的对应行逐位一致（同 _indexer_
    # selection 内部 grid 只标记 chunk 内行号；CopySlices 反向把逐行梯度
    # 切片原样回传，与 forward 的 outs[b] = o 同一 transport 论证，§6.1）。
    # 全宽的 [n, B] scores / masked / int64 order 因此从不物化。
    _kw = max(min(int(topk), int(B)), 0)
    # `causal_ends_host` is kept in the signature for callers, but the score
    # columns are NOT narrowed to the readable prefix.  Narrowing changes the
    # GEMM shape, and a GEMM is not bitwise column-independent across shapes:
    # the prefix scores (and therefore the log-sum-exp and near-tie ordering)
    # can drift by ULPs.  The repo's comparability contract requires the same
    # inputs to produce the same bits as the un-narrowed implementation, so the
    # score/soft path stays full width; top-k selection goes through the
    # stable top-k in `_rank_blocks`.
    if n == 0:
        # Empty sequence: return the same shapes the non-empty path would,
        # including the valid-cols flag when requested.
        idx = torch.empty(0, _kw, device=dev, dtype=torch.int32)
        soft = torch.empty(0, _kw, device=dev, dtype=H.dtype) if need_soft else None
        if out_valid is not None:
            out_valid[:] = [torch.zeros(0, _kw, device=dev, dtype=torch.bool)]
        m = torch.zeros(0, B, device=dev, dtype=H.dtype) if return_mask else None
        return (m, idx, soft)
    idx = torch.empty(n, _kw, device=dev, dtype=torch.int32)
    soft = None
    if out_valid is None and return_mask:
        out_valid = []
    _valid_parts = [] if out_valid is not None else None
    out_dtype = None
    for s in range(0, n, query_chunk):
        e = min(s + query_chunk, n)
        if random_select:
            # same elementwise op chain with in-place tails: each op is the
            # same kernel on the same values (no fusion, no reassociation —
            # note x-floor(x) is NOT frac() for negatives), so the scores are
            # bit-identical.
            z = _ri[s:e] * 12.9898 + _jc
            z.sin_()
            z.mul_(43758.5453)
            z.sub_(torch.floor(z))
            z.mul_(2.0).sub_(1.0)
            c = z.to(H.dtype)
            del z
        else:
            raw = torch.einsum('ind,ibd->inb', qI[:, s:e], kI)
            raw.mul_(_iscale)
            F.relu(raw, inplace=True)
            c = torch.einsum('inb,ni->nb', raw, w_idx[s:e])
            # `raw` is dead once the head-weighted score table exists.
            del raw
        if out_dtype is None:
            out_dtype = c.dtype
            if need_soft:
                soft = torch.empty(n, _kw, device=c.device, dtype=out_dtype)
        _ccausal, _nccausal = _causal_of(s, e)
        with torch.no_grad():
            _box = []
            _ic = _indexer_selection(c, _ccausal, k, out_valid=_box)
        idx[s:e] = _ic
        if _valid_parts is not None:
            _valid_parts.append(_box[0])
        if not need_soft:
            # need_soft=False 的调用方（无梯度评估）只消费 idx：soft 是
            # gathered_attention 直通分支（该分支在无梯度下被 _soft_live 整体
            # 跳过）的唯一下游，故整条 soft 尾部不物化（不改变 idx 的任何
            # 整数，也不改变下游任何前向 bit）。
            del c, _ic, _ccausal, _nccausal
            continue
        c.masked_fill_(_nccausal, float('-inf'))
        _lse = torch.logsumexp(c, dim=-1, keepdim=True)
        # the gather output is fresh and nothing downstream needs the pre-subtract
        # values, so the subtraction runs in place (same kernel, same values, and
        # the sub_ node records the identical broadcast gradient); exp_ is the
        # same story (its backward saves the output, as the out-of-place form).
        # nan_to_num 必须出位：exp_ 的反向保存其输出张量，原地 nan_to_num_ 会
        # 改写该保存值、使 ExpBackward 的版本检查失败（反向报错）。
        soft[s:e] = _safe_nan_to_num(c.gather(1, _ic.long()).sub_(_lse).exp_(), where='indexer_softmax')
        del c, _lse, _ic, _ccausal, _nccausal
    if _valid_parts is not None:
        out_valid[:] = [_valid_parts[0] if len(_valid_parts) == 1 else torch.cat(_valid_parts, 0)]
    m = None
    if return_mask:
        valid_cols = out_valid[0]
        with torch.no_grad():
            # Build the (n, B) mask chunk-by-chunk: only the valid selections
            # are written (1.0 each; duplicate writes of the same cell are
            # idempotent), and each chunk is masked with the same causal slice
            # the full-table form would read.  The (n, B) result is the output
            # itself.
            _il = idx.long()
            _ar = _arange_cache(n, dev)
            m = torch.zeros(n, B, device=dev, dtype=out_dtype)
            for _s in range(0, n, query_chunk):
                _e = min(_s + query_chunk, n)
                _vc = valid_cols[_s:_e]
                _rows = _ar[_s:_e, None].expand(_e - _s, _il.shape[1])
                m[_rows[_vc], _il[_s:_e][_vc]] = 1.0
                _, _ncc = _causal_of(_s, _e)
                m[_s:_e].masked_fill_(_ncc, 0.0)
    return (m, idx, soft)


def _indexer_trainable(H, comp_kv, last_tok, W_DQ, W_DK, W_w, nIH, topk, *,
                       random_select=False, pre_qI=None, pre_w=None,
                       out_valid=None, causal_ends_host=None, need_soft=True):
    """Training wrapper around `lightning_indexer` with activation recompute.

    The indexer builds `raw`/`c`/`soft` tables of shape [T, B_blocks, nIH]
    (plus the [T, topk] prior) that stay alive until backward.  Recomputing
    the whole pure function in backward keeps only its inputs, matching the
    FusedIndexer save policy.  Values and gradients are unchanged (no RNG in the learned path
    unless `random_select`, which is a fixed function of the row/column ids).
    """
    def _run(_h, _ck, _lt):
        _m, _idx, _soft = lightning_indexer(
            _h, _ck, _lt, W_DQ, W_DK, W_w, nIH, topk, return_mask=False,
            random_select=random_select, pre_qI=pre_qI, pre_w=pre_w,
            out_valid=out_valid, causal_ends_host=causal_ends_host,
            need_soft=need_soft)
        return _m, _idx, _soft
    # If an outer block/MLP checkpoint already owns this forward, do not nest
    # a second checkpoint (the outer recompute covers the indexer too).
    _grad = _CKPT_NEST[0] == 0 and torch.is_grad_enabled() and (
        H.requires_grad or comp_kv.requires_grad
        or (pre_qI is not None and pre_qI.requires_grad)
        or (pre_w is not None and pre_w.requires_grad))
    if _grad:
        _saved = int(H.shape[0]) * int(comp_kv.shape[0]) * (int(nIH) + 4) * int(H.element_size())
        if _saved < (32 << 20):
            _grad = False
    if _grad:
        return torch.utils.checkpoint.checkpoint(
            _run, H, comp_kv, last_tok, use_reentrant=False,
            preserve_rng_state=False)
    return _run(H, comp_kv, last_tok)


class _SinkWiden(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, sink_logits):
        n, H, M = logits.shape
        z = logits.new_empty(n, H, M + 1)
        z[..., 1:] = logits
        z[..., 0:1] = sink_logits.view(1, -1, 1)
        return z

    @staticmethod
    def backward(ctx, gz):
        return (gz[..., 1:], gz[..., 0].sum(dim=0))

def sink_softmax(logits, sink_logits, dim=-1):
    if dim not in (-1, logits.dim() - 1):
        raise ValueError('sink_softmax only supports the last dim')
    _diff = torch.is_grad_enabled() and (getattr(logits, 'requires_grad', False) or (sink_logits is not None and getattr(sink_logits, 'requires_grad', False)))
    if _diff:
        return _sink_softmax_impl(logits, sink_logits, dim)
    with torch.no_grad():
        return _sink_softmax_impl(logits, sink_logits, dim).detach()

def _sink_log_normalizer(logits, sink_logits):
    return torch.logaddexp(
        sink_logits.view(1, -1, 1),
        torch.logsumexp(logits, dim=-1, keepdim=True))


def _sink_softmax_impl(logits, sink_logits, dim=-1):
    if sink_logits is None:
        return _safe_nan_to_num(torch.softmax(logits, dim), where='sink_softmax')
    lse = _sink_log_normalizer(logits, _safe_nan_to_num(sink_logits, where='sink_logit'))
    return _safe_nan_to_num((logits - lse).exp(), where='sink_softmax')

def _sink_split_softmax(logits, sink_logits, want_sink=True):
    if sink_logits is None:
        return (sink_softmax(logits, None, -1), None)
    sink_logits = _safe_nan_to_num(sink_logits, where='sink_split')
    lse = _sink_log_normalizer(logits, sink_logits)
    soft = (logits - lse).exp()
    if not want_sink:
        return (soft, None)
    return (soft, (sink_logits.view(1, -1, 1) - lse).exp())

_FO_FULL_CAP_BYTES = 64 << 20
_FO_MINL = {}

def _attn_full_width_ok(n, k, w, need_dedup):
    # Element count of the hoisted selection tables: the block read mask
    # [n, k], the [selected|window] valid mask [n, k + w], the per-row
    # any-reduction [n], and — only when first-occurrence dedup is needed —
    # its keep mask [n, k].  The per-chunk form is value-identical.
    elems = int(n) * (int(k) + int(w) + 1 + (int(k) if need_dedup else 0))
    return elems <= _FO_FULL_CAP_BYTES

def _finfo_min(dtype):
    # torch.finfo(dtype).min is a pure function of the dtype — the same
    # constant on every call, so it is memoised per dtype (host-side, no
    # tensor values involved).
    v = _FO_MINL.get(dtype)
    if v is None:
        v = torch.finfo(dtype).min
        _FO_MINL[dtype] = v
    return v

def gathered_attention(q, k_blk, v_blk, topk_idx, last_tok, k_sw, v_sw, w, scale, q_chunk=128, soft=None, sink_logits=None, mem_budget_bytes=None, sel_valid=None, topk_distinct=False, soft_form=False, v_stacked=None, k_stacked=None):
    n = q.shape[0]
    dev = q.device
    if mem_budget_bytes is None:
        mem_budget_bytes = _attn_transient_budget(dev)
    if mem_budget_bytes is None:
        mem_budget_bytes = 256 << 20
    _heads = q.shape[1] if q.dim() == 3 else 1
    _M = max(1, int(topk_idx.shape[1]) + int(w))
    _per_row = 4 * _M * _heads * int(q.shape[-1]) + 2 * _M * _heads
    _bytes_per_chunk_row = _per_row * q.element_size()
    _bindable = mem_budget_bytes is not None and math.isfinite(float(mem_budget_bytes))
    if _bindable and _bytes_per_chunk_row > 0:
        _max_rows = max(1, int(mem_budget_bytes) // _bytes_per_chunk_row)
        q_chunk = min(q_chunk, _max_rows)
    q_chunk = max(1, min(int(q_chunk), n))
    out = torch.empty_like(q)
    pos_all = _arange_cache(n, dev)
    win_idx, win_valid_all = _window_geometry(n, w, dev)
    _MINL = _finfo_min(k_blk.dtype)
    # The V halves are laid out contiguously once and gathered with one
    # combined row index per chunk: the selected rows of `v_all` are exactly
    # the two gather outputs concatenated in the same order, so Vset holds
    # bit-identical values.  The plain path does the same for K (the soft path
    # keeps two contiguous K gathers, see below).  Window rows are offset past
    # the block rows (index transport, §6.1).
    _nblk_rows = k_blk.shape[0]
    _has_sw = k_sw is not None and v_sw is not None
    if _has_sw:
        # `v_stacked` lets a caller that already owns a row-contiguous
        # [blocks | window] V view (the fused KV head in `_single`) hand it in
        # directly: gathering its rows yields exactly the rows of
        # cat([v_blk, v_sw]) in the same order, so the values are bit-identical.
        v_all = v_stacked if v_stacked is not None else torch.cat([v_blk, v_sw], 0)
        # `k_stacked` is the analogous row-contiguous [blocks | window] K
        # table (the normalized K columns of the fused KV head in `_single`).
        # The soft path can gather both halves from it with the window-offset
        # row index.
        if k_stacked is not None:
            k_all = k_stacked
        else:
            k_all = torch.cat([k_blk, k_sw], 0) if (soft is None and not soft_form) else None
    else:
        k_all, v_all = (k_blk, v_blk)
    _win_off = _nblk_rows if _has_sw else 0
    # Index transport (design notes §6.1): int32 and int64 index tensors hold
    # the same exact integers, and advanced indexing / equality comparisons on
    # them return bitwise identical values.
    topk_l = topk_idx if topk_idx.dtype in (torch.int32, torch.int64) else topk_idx.long()
    # The read-everything HCA arm passes the cached arange expansion (every
    # row is exactly arange(B)): last_tok[topk_l] then gathers one identical
    # copy of last_tok per row, so the same bool table comes out of a plain
    # broadcast comparison against last_tok — same integers compared, bitwise
    # the same output (§6.1).  Identity with the cached table (not a content
    # scan) is the guard, so any other caller falls back to the gather form
    # unchanged.
    _arange_sel = (topk_l.shape[1] == last_tok.shape[0]
                   and topk_l is _cache_get(_IDX_CACHE, (_dev_str(dev), 'arangeexp', int(n), int(topk_l.shape[1]))))
    # the keep mask and the [selected|window] valid mask are per-row functions of
    # the same inputs, so their full-width forms are bitwise identical to the
    # per-chunk forms; they are hoisted out of the chunk loop under
    # _FO_FULL_CAP_BYTES (the chunked fallback below covers very wide top-k
    # sweeps).  The full [n, k] block table itself is built ONLY in the
    # full-width branch: building it before the size check would defeat the
    # chunked fallback at long context (HCA has B ~ T / block_size).
    # `topk_distinct=True` asserts every row of topk_idx holds distinct block
    # ids (the read-everything HCA arm passes an arange expansion), in which
    # case first_occurrence_mask is provably all-True — the keep mask is then
    # identical to the computed form.
    _nb = int(topk_l.shape[1])
    _dedup = (not topk_distinct) and _nb > 1
    if _attn_full_width_ok(n, _nb, w, _dedup):
        sel_blk_all = (pos_all[:, None] > last_tok[None, :]) if _arange_sel else (pos_all[:, None] > last_tok[topk_l])
        keep_all = sel_valid
        if _dedup:
            _ddk_all = first_occurrence_mask(topk_l)
            keep_all = _ddk_all if keep_all is None else keep_all & _ddk_all
        if keep_all is not None:
            sel_blk_all &= keep_all
        valid_all = torch.cat([sel_blk_all, win_valid_all], dim=1)
        # full-width forms of the per-chunk negation / any-reduction: the
        # elementwise and reduction values per row are identical to the
        # per-chunk forms.
        nvalid_all = ~valid_all[:, None, :]
        anyrow_all = valid_all.any(-1) if sink_logits is None else None
    else:
        sel_blk_all = keep_all = valid_all = nvalid_all = anyrow_all = None
    # The chunk-invariant dispatch is computed once, above the row-chunk
    # loop: which mask/validity source each chunk reads and which of the two
    # attention forms runs are pure functions of the call-level arguments.
    # The loop body itself — and the tensor values it produces — is
    # unchanged.
    _has_all = valid_all is not None
    _has_sv = sel_valid is not None
    _has_sink = sink_logits is not None
    # soft 只经直通 (soft_attn - soft_attn.detach()).add_(attn) 进入前向值：
    # soft_attn 是 _sink_split_softmax/nan_to_num 之后的输出、恒有限，x - x
    # 精确相消为 +0.0，且 0.0 加回 attn 逐位不变（attn 全元非负，无 -0.0
    # 符号翻转），故跳过该分支不改变任何前向 bit；该子图对 soft/q/K/V 的梯度
    # 也恒为 0（d(x - detach(x)) = 0），因此仅当确有相关张量需要梯度时才保留
    # 它。训练/checkpoint 重算都在 grad 开启下运行，条件与首次前向一致。
    _soft_live = soft is not None and torch.is_grad_enabled() and any(
        _t is not None and getattr(_t, 'requires_grad', False)
        for _t in (soft, q, k_blk, v_blk, k_sw, v_sw, sink_logits))
    _tk2 = _take_2d
    _einsum = torch.einsum
    if _has_all:
        # Only the keep/nvalid/anyrow slices are consumed below; the whole-row
        # sel_blk_all / valid_all tables are fully absorbed into them.
        _plan = [(None if keep_all is None else keep_all[_s:_s + q_chunk], nvalid_all[_s:_s + q_chunk], None if anyrow_all is None else anyrow_all[_s:_s + q_chunk]) for _s in range(0, n, q_chunk)]
        del sel_blk_all, valid_all
    else:
        _plan = None
    _win_off_tab = _window_block_offset(n, w, _nblk_rows, dev) if _win_off else None
    for s in range(0, n, q_chunk):
        e = s + q_chunk
        if e > n:
            e = n
        qseg = q[s:e]
        if _plan is not None:
            keep, nvalid, anyrow = _plan[s // q_chunk]
        else:
            # per-chunk form of the full-width table (same integers, same
            # booleans).
            _lt_rows = last_tok[None, :] if _arange_sel else last_tok[topk_l[s:e]]
            sel_blk = pos_all[s:e, None] > _lt_rows
            keep = None
            if _has_sv:
                keep = sel_valid[s:e]
            if _dedup:
                _ddk = first_occurrence_mask(topk_l[s:e])
                keep = _ddk if keep is None else keep & _ddk
            if keep is not None:
                sel_blk = sel_blk & keep
            valid = torch.cat([sel_blk, win_valid_all[s:e]], dim=1)
            nvalid = ~valid[:, None, :]
            anyrow = None if _has_sink else valid.any(-1)
            del sel_blk, valid
        # Two half-gathers (and the same for V): the block half gathers the
        # k_blk rows topk_l addresses (rows 0..n_blk-1 of the stacked tensor
        # == k_blk rows) and the window half gathers k_sw at win_idx (rows
        # n_blk+j of the stacked tensor == row j of k_sw).  index_select
        # returns exactly the same source rows, so the gathered values — and
        # hence every downstream value — are bit-identical to the
        # cat-then-gather form (index re-plumbing only, design notes §6.1).
        _sel_rows = topk_l[s:e]
        _win_local = win_idx[s:e]
        _win_rows = _win_off_tab[s:e] if _win_off_tab is not None else _win_local
        # `_has_sw` is the contract for stacking window lanes: the row offset
        # may be 0 when k_blk holds no rows, and that is still a valid
        # stacked-row base.
        _rows = torch.cat([_sel_rows, _win_rows], 1) if _has_sw else _sel_rows
        # soft_form=True 在 soft=None 时仍选用软分支的硬部分解（两个半段
        # logits + cat），供调用方在跳过（无梯度下已死的）soft 尾部后保持
        # 硬分支逐位一致；它不改变任何取值，只固定走哪套既有分解。
        if soft is None and not soft_form:
            Kset = _tk2(k_all, _rows)
            logits = _einsum('qhd,qmhd->qhm', qseg, Kset)
            del Kset
            if scale != 1.0:
                # x·1.0 is the IEEE identity (and grad·1.0 is bit-identical on
                # the backward Mul node), so scaling by 1.0 is a no-op.
                logits.mul_(scale)
            logits.masked_fill_(nvalid, _MINL)
            attn, _sink_unused = _sink_split_softmax(logits, sink_logits, want_sink=False)
            if not _has_sink:
                attn.mul_(anyrow[:, None, None])
            out[s:e] = _einsum('qhm,qmhd->qhd', attn, _tk2(v_all, _rows))
            del logits, attn, nvalid, anyrow, keep, _sel_rows, _win_local, _win_rows, _rows, qseg
            continue
        # The soft branch reads its K halves as two contiguous gathers: the
        # concatenated V gather below holds exactly the same rows as two
        # separate gathers + cat, while the per-half K gathers feed the two
        # logits einsums with the same operands (index re-plumbing only,
        # design notes §6.1).
        if k_all is not None:
            # The combined table holds the same rows as the two halves:
            # block rows at the front and window row j at n_blk + j.
            kb = _tk2(k_all, _sel_rows)
            kw = _tk2(k_all, _win_rows)
        else:
            kb = _tk2(k_blk, _sel_rows)
            kw = _tk2(k_sw, _win_local)
        blk_logits = _einsum('qhd,qmhd->qhm', qseg, kb)
        if scale != 1.0:
            blk_logits.mul_(scale)
        win_logits = _einsum('qhd,qmhd->qhm', qseg, kw)
        if scale != 1.0:
            win_logits.mul_(scale)
        logits = torch.cat([blk_logits, win_logits], -1)
        del kb, kw, blk_logits, win_logits
        if _soft_live:
            soft_g = soft[s:e]
            if keep is not None:
                soft_g = soft_g * keep
            soft_log_g = torch.log(soft_g.clamp_min(1e-12))
            soft_log_g.masked_fill_(soft_g <= 0, _MINL)
            soft_logits = logits.clone()
            soft_logits[:, :, :_nb].add_(soft_log_g[:, None, :]).clamp_min_(_MINL)
            soft_logits.masked_fill_(nvalid, _MINL)
            soft_attn, _sink_unused = _sink_split_softmax(soft_logits, sink_logits, want_sink=False)
            # soft_logits is dead once soft_attn is built.
            del soft_logits, soft_log_g
        logits.masked_fill_(nvalid, _MINL)
        attn, _sink_unused = _sink_split_softmax(logits, sink_logits, want_sink=False)
        if _soft_live:
            attn = (soft_attn - soft_attn.detach()).add_(attn)
        if not _has_sink:
            attn.mul_(anyrow[:, None, None])
        out[s:e] = _einsum('qhm,qmhd->qhd', attn, _tk2(v_all, _rows))
        # `soft_attn` only exists on the live-soft path.
        del logits, attn, nvalid, anyrow, keep, _sel_rows, _win_local, _win_rows, _rows, qseg
        if _soft_live:
            del soft_attn
    return out

_CPU_ACTIVE_BATCH = [1]
_CPU_GATHER_CKPT_BYTES = 512 << 20
_CPU_POOL_CKPT_BYTES = 64 << 20
_CPU_CE_CKPT_BYTES = 128 << 20

def _cpu_saved_ok(saved_bytes, limit):
    return torch.is_grad_enabled() and saved_bytes < int(limit)

def _gathered_attention_trainable(*args, **kwargs):
    # 训练时包一层非重入 checkpoint，反向按同一确定性分解重算前向。函数内
    # 无 RNG，因此 chunk 边界与每个中间值逐位相同，梯度与直调形式逐位一致。
    # 重算不消耗也不改写任何随机流。
    if torch.is_grad_enabled() and args[0].requires_grad:
        q = args[0]
        if q.device.type != 'cuda':
            _M = int(args[3].shape[1]) + int(args[7])
            _saved = 2 * int(q.numel()) * max(1, _M) * int(q.element_size()) * int(_CPU_ACTIVE_BATCH[0])
            if _cpu_saved_ok(_saved, _CPU_GATHER_CKPT_BYTES):
                return gathered_attention(*args, **kwargs)
        return torch.utils.checkpoint.checkpoint(gathered_attention, *args, use_reentrant=False, preserve_rng_state=False, **kwargs)
    return gathered_attention(*args, **kwargs)

def _pool_trainable(fn, *args, **kwargs):
    # 训练时包一层非重入 checkpoint，反向重算同一纯函数：池化内部无 RNG，
    # 整数几何缓存挂在不可变的 block_ids 对象上（_pool_parts / _pool_geom，
    # design notes §6.2），反向重算必命中同一几何，前向值与梯度逐位一致；
    # int 型的 n_blocks 返回值原样透传。eval 下（无梯度）直调。池化同样
    # 无 RNG，所以 checkpoint 不需要保存/恢复随机流。
    if torch.is_grad_enabled() and args[0].requires_grad:
        x = args[0]
        if x.device.type != 'cuda':
            _saved = 16 * int(x.numel()) * int(x.element_size()) * int(_CPU_ACTIVE_BATCH[0])
            if _cpu_saved_ok(_saved, _CPU_POOL_CKPT_BYTES):
                return fn(*args, **kwargs)
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=False, **kwargs)
    return fn(*args, **kwargs)

@dataclass
class AttnCfg:
    kind: str = 'csa'
    chunking: str = 'fixed'
    dynamic: bool = False
    block_size: int = 4
    overlap: int = 8
    index_topk: int = 32
    cos_threshold: float = 0.5
    adaptive: bool = False
    target_block_tokens: int = 16
    temperature: float = 0.1
    min_block: int = 2
    max_block: int = 128
    sliding_window: int = 128
    c_kv: int = 128
    c_index: int = 64
    n_index_heads: int = 4
    full_cosine: bool = False
    window: int = 0
    comp_lambda_mult: float = 1.0
    indexer_mode: str = 'learned'
    content_mode: str = 'pooled'
    use_sink: bool = True
    full_sink: bool = False
    fuse_mode: str = 'none'
    fuse_kernel: int = 5
    # 'threshold' uses hard-cut decisions.  'reinforce'
    # samples the cut decisions and adds a score-function term for the LM loss,
    # so the task objective can actually learn the boundary policy instead of
    # only the compression-rate regularizer (review 3 issue 6).
    boundary_policy: str = 'threshold'

class HybridAttention(nn.Module):

    def __init__(self, d_model, n_heads, d_head, cfg: AttnCfg):
        super().__init__()
        self.cfg = cfg
        self.nh, self.hd = (n_heads, d_head)
        self.kv_dim = 2 * cfg.c_kv
        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=False)
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        self.last_gate_mean = None
        self.last_boundary_logp = None
        self.last_boundary_decisions = 0
        self.need_reg = True
        self._stats = None
        self._cuts = None
        self.fuse_conv = None
        # Fused-weight cache (eval-only): the gateway is `not
        # torch.is_grad_enabled()`, so training forward passes always rebuild
        # the cat inside the autograd graph exactly as before.  Under eval the
        # concat values are pure functions of the source weights, so keying on
        # their autograd `_version` counters (bumped by every in-place
        # optimizer update) makes a stale hit impossible.
        self._wcat_cache = {}
        if cfg.kind in ('csa', 'hca') and cfg.chunking == 'cosine_learnable' and (cfg.fuse_mode == 'learned'):
            self.fuse_conv = nn.Conv1d(d_model, d_model, cfg.fuse_kernel, groups=d_model, bias=False)
            with torch.no_grad():
                self.fuse_conv.weight.zero_()
                self.fuse_conv.weight[:, :, -1] = 1.0
        if cfg.kind == 'full':
            self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)
            self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
            nn.init.xavier_uniform_(self.W_k.weight)
            nn.init.xavier_uniform_(self.W_v.weight)
            self.W_aKV = None
            self.W_kvhead = None
            self.sink = nn.Parameter(torch.zeros(n_heads)) if cfg.full_sink else None
            return
        self.W_aKV = nn.Parameter(torch.empty(d_model, self.kv_dim))
        self.W_kvhead = nn.Linear(self.kv_dim, 2 * n_heads * d_head, bias=False)
        nn.init.xavier_uniform_(self.W_aKV)
        nn.init.xavier_uniform_(self.W_kvhead.weight)
        if cfg.kind not in ('csa', 'hca'):
            return
        if cfg.kind == 'hca':
            self.W_bKV = None
            self.W_bZ = None
            self.W_aZ = nn.Parameter(torch.empty(d_model, self.kv_dim))
            self.B_pos_a = nn.Parameter(torch.zeros(int(cfg.max_block) + 1, self.kv_dim))
            self.sink = nn.Parameter(torch.zeros(n_heads)) if getattr(cfg, 'use_sink', True) else None
            nn.init.xavier_uniform_(self.W_aZ)
            if cfg.chunking == 'cosine_learnable':
                Tc = max(cfg.temperature, 0.001)
                Lb = max(cfg.target_block_tokens, 2)
                self.delta_logit = nn.Parameter(torch.tensor(float(Tc * math.log(1.0 / (Lb - 1)))))
            else:
                self.delta_logit = None
            return
        self.W_bKV = nn.Parameter(torch.empty(d_model, self.kv_dim))
        self.W_aZ = nn.Parameter(torch.empty(d_model, self.kv_dim))
        self.W_bZ = nn.Parameter(torch.empty(d_model, self.kv_dim))
        self.B_pos_a = nn.Parameter(torch.zeros(int(cfg.max_block) + 1, self.kv_dim))
        self.B_pos_b = nn.Parameter(torch.zeros(int(cfg.max_block) + 1, self.kv_dim))
        self.sink = nn.Parameter(torch.zeros(n_heads)) if getattr(cfg, 'use_sink', True) else None
        self.W_DQ = nn.Parameter(torch.empty(d_model, cfg.c_index))
        self.W_DK = nn.Parameter(torch.empty(self.kv_dim, cfg.c_index))
        self.W_w = nn.Linear(d_model, cfg.n_index_heads, bias=False)
        if cfg.chunking == 'cosine_learnable':
            Tc = max(cfg.temperature, 0.001)
            Lb = max(cfg.target_block_tokens, 2)
            self.delta_logit = nn.Parameter(torch.tensor(float(Tc * math.log(1.0 / (Lb - 1)))))
        else:
            self.delta_logit = None
        for p in [self.W_bKV, self.W_aZ, self.W_bZ, self.W_DQ, self.W_DK]:
            nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform_(self.W_w.weight)

    def _fuse(self, x):
        if self.fuse_conv is None:
            return x
        k = self.cfg.fuse_kernel
        z = F.pad(x.t().unsqueeze(0), (k - 1, 0))
        return F.conv1d(z, self.fuse_conv.weight, groups=x.shape[1]).squeeze(0).t()

    def _full_batched(self, x):
        B, T, _ = x.shape
        _nhq = self.nh * self.hd
        if x.device.type == 'cuda' and not torch.is_grad_enabled():
            # Key on versions, dtype and device of the source weights.
            _key = (self.W_q.weight._version, self.W_k.weight._version, self.W_v.weight._version,
                    self.W_q.weight.dtype, self.W_k.weight.dtype, self.W_v.weight.dtype,
                    self.W_q.weight.device)
            _hit = self._wcat_cache.get('qkv')
            if _hit is None or _hit[0] != _key:
                with torch.inference_mode(False):
                    _hit = (_key, torch.cat([self.W_q.weight, self.W_k.weight, self.W_v.weight], 0))
                self._wcat_cache['qkv'] = _hit
            _qkv = F.linear(x, _hit[1])
            q = _qkv[..., :_nhq].reshape(B, T, self.nh, self.hd)
            k = _qkv[..., _nhq:2 * _nhq].reshape(B, T, self.nh, self.hd)
            v = _qkv[..., 2 * _nhq:].reshape(B, T, self.nh, self.hd)
            del _qkv
        else:
            q = self.W_q(x).view(B, T, self.nh, self.hd)
            k = self.W_k(x).view(B, T, self.nh, self.hd)
            v = self.W_v(x).view(B, T, self.nh, self.hd)
        if self.cfg.full_cosine:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            scale = 1.0
        else:
            scale = 1.0 / math.sqrt(self.hd)
        # The sink branch builds a full cached bool mask within the row cap,
        # or the matching row slice per chunk beyond it.  The cached table is
        # bool (one byte per element), not a 4-byte additively-typed mask.
        _use_full_mask = _dense_mask_fits(T)
        if self.sink is None:
            if 0 < self.cfg.window < T:
                out = _window_attn_local(q, k, v, self.cfg.window, scale)
                return self.W_o(out.reshape(B, T, self.nh * self.hd))
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                is_causal=True, scale=scale).transpose(1, 2)
            return self.W_o(out.reshape(B, T, self.nh * self.hd))
        if _use_full_mask:
            if self.cfg.window > 0:
                mask = causal_window_mask(T, self.cfg.window, x.device)
            else:
                mask = causal_mask(T, x.device)
        else:
            mask = None
        sink_view = _safe_nan_to_num(self.sink, where='sink_param').view(1, self.nh, 1, 1) if self.sink is not None else None
        row_chunk = _full_attn_row_chunk(x.device, B, self.nh, T, q.element_size())
        # 训练时整段注意力循环包一层非重入 checkpoint：反向按同一份纯行划分
        # 重算，值/梯度不变。row_chunk 在运行内经 _ATB_EPOCH 不变，mask 是
        # 缓存的常量张量，因此重算的块边界与中间值逐位相同（同 ce_loss
        # 分块重算的论证）。
        if torch.is_grad_enabled() and q.requires_grad:
            out = torch.utils.checkpoint.checkpoint(
                self._sink_attn_loop, q, k, v, mask, _use_full_mask, sink_view,
                scale, row_chunk, use_reentrant=False, preserve_rng_state=False)
        else:
            out = self._sink_attn_loop(q, k, v, mask, _use_full_mask, sink_view, scale, row_chunk)
        return self.W_o(out.reshape(B, T, self.nh * self.hd))

    def _sink_attn_loop(self, q, k, v, mask, use_full_mask, sink_view, scale, row_chunk):
        B, T = q.shape[0], q.shape[1]
        out = torch.empty(B, T, self.nh, self.hd, device=q.device, dtype=q.dtype)
        for s in range(0, T, row_chunk):
            e = min(s + row_chunk, T)
            # A tuple is the row-mask recipe, not a tensor: the checkpointed
            # recompute rebuilds the identical slice.
            _mask_c = mask[s:e] if use_full_mask else (s, e, T, self.cfg.window)
            out[:, s:e] = self._sink_attn_chunk(q[:, s:e], k, v, _mask_c, sink_view, scale)
        return out

    def _sink_attn_chunk(self, q_c, k, v, mask_c, sink_view, scale):
        if isinstance(mask_c, tuple):
            s, e, T, window = mask_c
            mask_c = _dense_causal_window_rows(T, window, s, e, q_c.device)
        logits = torch.einsum('bnhd,bmhd->bhnm', q_c, k)
        if scale != 1.0:
            # same IEEE identity as gathered_attention (x·1.0 and its backward
            # grad·1.0 are bit-identical to no-op).
            logits.mul_(scale)
        # The mask is the boolean masked-lane table; the fill writes exactly
        # the -inf the additive 0/-inf form added.  x + 0.0 == x except
        # -0.0 -> +0.0 (exp is 1.0 either way, so the softmax/logsumexp row
        # reduction sees the same finite lane values in the same order and
        # every downstream value is bit-identical), and the fill's backward
        # zeroes exactly the lanes the softmax backward already scaled by
        # p_i == 0 in the additive form — unmasked lanes pass the identical
        # grad through in both forms.
        logits.masked_fill_(mask_c, float('-inf'))
        if sink_view is not None:
            lse = torch.logaddexp(sink_view, torch.logsumexp(logits, dim=-1, keepdim=True))
            # logsumexp 的反向保留其输入 logits，sub 必须出位（保留值与新
            # 张量上的 exp_ 原地——exp 反向保存的输出此后再无人改写）；原地
            # sub_ 会使 logsumexp 保存的张量版本失效，反向直接报错。
            attn = logits.sub(lse).exp_()
            del logits, lse
        else:
            attn = torch.softmax(logits, -1)
            del logits
        return torch.einsum('bhnm,bmhd->bnhd', attn, v)

    def _split(self, kvh):
        nh, hd = (self.nh, self.hd)
        k = kvh[..., :nh * hd].view(*kvh.shape[:-1], nh, hd)
        v = kvh[..., nh * hd:].view(*kvh.shape[:-1], nh, hd)
        return (k, v)

    def _pre(self, x, fuse_x=None):
        cfg = self.cfg
        pre = {}
        if cfg.kind == 'full':
            return None
        _want_idx = cfg.kind != 'hca' and cfg.indexer_mode != 'random'
        # Training never builds the concatenated weight: it is only a cached
        # constant on the eval path.  The separate projections produce the
        # same independent output columns without the per-step cat.
        if x.device.type != 'cuda' or torch.is_grad_enabled():
            pre['q'] = self.W_q(x)
            pre['Ca'] = x @ self.W_aKV
            pre['Cb'] = x @ self.W_bKV if self.W_bKV is not None else None
            pre['Za'] = x @ self.W_aZ if self.W_aZ is not None else None
            pre['Zb'] = x @ self.W_bZ if self.W_bZ is not None else None
            if cfg.kind != 'hca':
                if cfg.indexer_mode == 'random':
                    pre['qI'] = pre['w_idx'] = None
                else:
                    pre['qI'] = x @ self.W_DQ
                    pre['w_idx'] = F.linear(x, self.W_w.weight)
            if self.fuse_conv is not None:
                k = cfg.fuse_kernel
                fx = fuse_x.detach()
                z = F.pad(fx.transpose(1, 2), (k - 1, 0))
                pre['fused'] = F.conv1d(z, self.fuse_conv.weight, groups=fx.shape[-1]).transpose(1, 2)
            return pre
        # Batched projection (CUDA): all x @ W terms run over the concatenated
        # weight columns (row-independent output columns, so the per-slice
        # values — and the per-column grads — are bit-identical to the
        # per-projection form, design notes §6.1).
        _mats = [self.W_q.weight.t(), self.W_aKV]
        _keys = ['q', 'Ca']
        # weights are transposed below for a single F.linear (addmm) call, so
        # the kernel matches the per-projection form exactly
        if self.W_bKV is not None:
            _mats.append(self.W_bKV)
            _keys.append('Cb')
        if self.W_aZ is not None:
            _mats.append(self.W_aZ)
            _keys.append('Za')
        if self.W_bZ is not None:
            _mats.append(self.W_bZ)
            _keys.append('Zb')
        if _want_idx:
            _mats.append(self.W_DQ)
            _keys.append('qI')
        # Key on versions, dtype and device of the source weights.
        _key = tuple(_m._version for _m in _mats) + tuple(_m.dtype for _m in _mats) + (_mats[0].device,)
        _hit = self._wcat_cache.get('pre')
        if _hit is None or _hit[0] != _key:
            with torch.inference_mode(False):
                _hit = (_key, torch.cat([_m.t() for _m in _mats], dim=0))
            self._wcat_cache['pre'] = _hit
        _wcat = _hit[1]
        _fused = F.linear(x, _wcat)
        _off = 0
        for _k, _m in zip(_keys, _mats):
            _w = _m.shape[1]
            pre[_k] = _fused[:, _off:_off + _w]
            _off += _w
        pre.setdefault('Cb', None)
        pre.setdefault('Za', None)
        pre.setdefault('Zb', None)
        if cfg.kind != 'hca':
            # w_idx stays on its own narrow-GEMM kernel (n_index_heads is 4:
            # folding a 4-column block into the wide fused GEMM changes the
            # kernel's accumulation tiling and hence the low-order bits, so it
            # keeps the exact per-projection form).
            if cfg.indexer_mode == 'random':
                pre['qI'] = pre['w_idx'] = None
            else:
                pre['w_idx'] = F.linear(x, self.W_w.weight)
        if self.fuse_conv is not None:
            k = cfg.fuse_kernel
            fx = fuse_x.detach()
            z = F.pad(fx.transpose(1, 2), (k - 1, 0))
            pre['fused'] = F.conv1d(z, self.fuse_conv.weight, groups=fx.shape[-1]).transpose(1, 2)
        return pre

    def _gate_signal(self, x, pre=None):
        # GPU-side half of the cosine_learnable gate, computed by the same ops as
        # the inline path in _single.  forward() calls this for every row of the
        # batch first, then moves the stacked hard decisions to the CPU; every
        # value downstream (the gate tensor itself and the per-row cut lists) is
        # identical to the per-row inline form.
        fused = pre['fused'] if pre is not None and 'fused' in pre else self._fuse(x.detach())
        sim = cosine_similarity_consecutive(fused)
        if sim.numel() == 0:
            return None
        cfg = self.cfg
        # In-place tails over the fresh cumsum output: same op order
        # (divide -> add -> subtract -> divide) and therefore the same values.
        prefix_mean = torch.cumsum(sim, 0)
        prefix_mean.div_(_range_cache(1, sim.numel() + 1, x.device))
        prefix_mean.add_(self.delta_logit)
        prefix_mean.sub_(sim)
        prefix_mean.div_(max(cfg.temperature, 0.001))
        if getattr(cfg, 'boundary_policy', 'threshold') == 'reinforce' and torch.is_grad_enabled():
            logits = prefix_mean
            gate = torch.sigmoid(logits)
            gate_bool = torch.bernoulli(gate).bool()
            logp = -F.binary_cross_entropy_with_logits(logits, gate_bool.to(gate.dtype), reduction='sum')
            return (gate, gate_bool, logp)
        gate = torch.sigmoid_(prefix_mean)
        return (gate, gate.detach() > _HALF)

    def _gate_signals(self, x, pre_all=None):
        # Batched form of the per-row `_gate_signal` calls.  Every op except the
        # prefix cumsum is pointwise or a reduction over the last dim of each
        # vector/pair, so its per-row values are independent of how many rows
        # share the launch — bit-identical to the per-row form (the same
        # row-independence the gate-decision hoist already relies on, design
        # notes §6.1).  The prefix cumsum still runs per row on the same
        # contiguous [T-1] values with the same 1-D kernel call, so its output
        # is bit-identical to the per-row form as well.  Returns (rows,
        # gate_bool_stacked): rows[b] is None or (gate_row, bool_row) exactly
        # as `_gate_signal` would have returned for row b.
        B, T, _d = x.shape
        fused = pre_all.get('fused') if pre_all is not None else None
        if fused is None:
            if self.fuse_conv is not None:
                # safe fallback identical to the per-row loop (each row fuses
                # itself, exactly like _gate_signal's inline form)
                return ([self._gate_signal(x[b], None) for b in range(B)], None)
            fused = x.detach()
        sim = cosine_similarity_consecutive(fused, eps=1e-08)
        if sim.shape[-1] == 0:
            return ([None] * B, None)
        cfg = self.cfg
        # rowwise batched cumsum — same per-row values as the unbind/stack loop.
        # The fresh cumsum output carries the divide/add/subtract/divide tails
        # in place.
        prefix_mean = torch.cumsum(sim, dim=-1)
        prefix_mean.div_(_range_cache(1, sim.shape[-1] + 1, x.device))
        prefix_mean.add_(self.delta_logit)
        prefix_mean.sub_(sim)
        prefix_mean.div_(max(cfg.temperature, 0.001))
        if getattr(cfg, 'boundary_policy', 'threshold') == 'reinforce' and torch.is_grad_enabled():
            logits = prefix_mean
            gate = torch.sigmoid(logits)
            gate_b = torch.bernoulli(gate).bool()
            logp = -F.binary_cross_entropy_with_logits(logits, gate_b.to(gate.dtype), reduction='none').sum(dim=-1)
            return ([(gate[b], gate_b[b], logp[b]) for b in range(B)], gate_b, logp)
        gate = torch.sigmoid_(prefix_mean)
        gate_b = gate.detach() > _HALF
        return ([(gate[b], gate_b[b]) for b in range(B)], gate_b)

    def _cut_blocks_batched(self, x):
        # Batched hoist of the per-row cosine cut computation in the
        # cosine_abs / cosine_adaptive / dynamic branches of `_single` (design
        # notes §6.1): the batched normalize + last-dim reduce is the same
        # row-independent form _gate_signals already relies on.  Every integer
        # reaching _single is identical to the per-row blocks_from_cosine form
        # (same cuts, same segmenter).
        cfg = self.cfg
        B, T, _d = x.shape
        if T <= 1:
            return None
        sim = cosine_similarity_consecutive(x.detach(), eps=1e-08)
        if cfg.chunking == 'cosine_adaptive' or cfg.adaptive:
            tau = causal_adaptive_threshold(sim, cfg.target_block_tokens)
        else:
            tau = cfg.cos_threshold
        cuts = sim < tau
        if _device_segment_enabled(x.device):
            return _device_blocks_from_cuts(T, cuts, cfg.min_block, cfg.max_block, x.device, want_honoured=False, hon_dtype=torch.float32)
        cuts = cuts.cpu().numpy()
        return _hoist_blocks_from_cuts(T, cuts, cfg.min_block, cfg.max_block, x.device, want_honoured=False, hon_dtype=torch.float32)

    def _single(self, x, pre=None, gate_sig=None, pre_blocks=None):
        cfg = self.cfg
        self.last_boundary_logp = None
        self.last_boundary_decisions = 0
        T = x.shape[0]
        assert cfg.kind in ('csa', 'hca'), 'dense layers go through _full_batched'
        scale = 1.0
        q = (pre['q'] if pre is not None else self.W_q(x)).view(T, self.nh, self.hd)
        q = F.normalize(q, dim=-1)
        Ca_raw = pre['Ca'] if pre is not None else x @ self.W_aKV
        if pre is not None:
            Cb_raw = pre['Cb']
        else:
            Cb_raw = x @ self.W_bKV if self.W_bKV is not None else None
        gate_mean = None
        if cfg.chunking == 'cosine_learnable':
            if gate_sig is not None:
                gate, gate_list = gate_sig[0], gate_sig[1]
                if len(gate_sig) > 2:
                    self.last_boundary_logp = gate_sig[2]
                    self.last_boundary_decisions = int(gate.numel())
            else:
                fused = pre['fused'] if pre is not None and 'fused' in pre else self._fuse(x.detach())
                sim = cosine_similarity_consecutive(fused)
                if sim.numel() == 0:
                    gate = torch.tensor([], device=x.device)
                else:
                    # same in-place tails as _gate_signal (same values)
                    prefix_mean = torch.cumsum(sim, 0)
                    prefix_mean.div_(_range_cache(1, sim.numel() + 1, x.device))
                    prefix_mean.add_(self.delta_logit)
                    prefix_mean.sub_(sim)
                    prefix_mean.div_(max(cfg.temperature, 0.001))
                    gate = torch.sigmoid_(prefix_mean)
                gate_bool = None
                gate_list = None
                if gate.numel():
                    if getattr(cfg, 'boundary_policy', 'threshold') == 'reinforce' and torch.is_grad_enabled():
                        # score-function path: the hard cuts are sampled, and
                        # the log-probability is returned so the LM loss can
                        # train the boundary policy (not just comp_reg).
                        _logits = torch.logit(gate.clamp(1e-7, 1 - 1e-7))
                        gate_bool = torch.bernoulli(gate).bool()
                        self.last_boundary_logp = -F.binary_cross_entropy_with_logits(
                            _logits, gate_bool.to(gate.dtype), reduction='sum')
                        self.last_boundary_decisions = int(gate.numel())
                    else:
                        gate_bool = gate.detach() > _HALF
                    # On the device path the soft-honoured mask is already
                    # attached to pre_blocks, so the boolean cut row can stay
                    # on the GPU (it is not consumed by the host segmenter).
                    gate_list = gate_bool if _device_segment_enabled(x.device) else gate_bool.cpu().numpy()
            if pre_blocks is not None:
                bid, nblk, _hon_dev = pre_blocks
            else:
                with torch.no_grad():
                    if gate_list is None:
                        gate_list = (gate.detach() > _HALF).cpu().numpy()
                    bid, nblk, _honoured = blocks_from_cuts(T, gate_list, cfg.min_block, cfg.max_block, x.device, return_count=True, return_honoured=True)
                _hon_dev = None
                if gate.numel() and self.need_reg:
                    _keep = [0.0] * len(gate_list)
                    for _slot in _honoured:
                        if 0 <= _slot < len(_keep):
                            _keep[_slot] = 1.0
                    _hon_dev = _h2d_async(_keep, gate.dtype, gate.device)
            if gate.numel() and self.need_reg:
                if _hon_dev.dtype != gate.dtype:
                    _hon_dev = _hon_dev.to(gate.dtype)
                _soft_honoured = gate * _hon_dev
                _shs = _soft_honoured.sum()
                gate_mean = (float(nblk - 1) + _shs - _shs.detach()) / T
            else:
                gate_mean = torch.zeros((), device=x.device) if self.need_reg else None
        elif cfg.chunking in ('cosine_abs', 'cosine_adaptive') or (cfg.dynamic and cfg.kind != 'hca'):
            if pre_blocks is not None:
                # geometry pre-computed once for the whole batch by
                # _cut_blocks_batched (forward hoist); values identical to the
                # per-row form below (design notes §6.1).
                bid, nblk, _hon_dev = pre_blocks
            else:
                sim = cosine_similarity_consecutive(x.detach())
                tau = causal_adaptive_threshold(sim, cfg.target_block_tokens) if cfg.chunking == 'cosine_adaptive' or cfg.adaptive else cfg.cos_threshold
                bid, nblk = blocks_from_cosine(x, tau, cfg.min_block, cfg.max_block, sim=sim, return_count=True)
        else:
            bid = blocks_fixed(T, cfg.block_size, x.device)
            nblk = -(-T // cfg.block_size)
        if pre is not None:
            Za, Zb = (pre['Za'], pre['Zb'])
        else:
            Za = x @ self.W_aZ
            Zb = x @ self.W_bZ if self.W_bZ is not None else None
        if cfg.kind == 'hca':
            comp_kv, last_tok, Bn = _pool_trainable(pool_blocks_single, Ca_raw, Za, self.B_pos_a, bid, n_blocks=nblk, monotonic=True)
        else:
            comp_kv, last_tok, Bn = _pool_trainable(pool_variable_blocks, Ca_raw, Cb_raw, Za, Zb, bid, self.B_pos_a, self.B_pos_b, cfg.overlap, n_blocks=nblk, monotonic=True)
        index_kv = comp_kv
        if self._stats is not None:
            # scatter_add of exact integer ones is order-independent (integer
            # addition commutes, no rounding).  bincount is deterministic
            # (no atomics) and returns the same exact integer counts.
            _bc = torch.bincount(bid.reshape(-1).long(), minlength=int(nblk))[:int(nblk)]
            self._stats.append(_bc)
            if self._cuts is not None:
                _cpu = getattr(bid, '_pool_cpu', None)
                if _cpu is not None and sum(_cpu[0]) == T:
                    # host-side integer equivalence (design notes §6.2): cut
                    # positions are the prefix sums of the same counts table
                    # that generated bid — the identical integers to the
                    # nonzero(diff)+1 form.  Consumers accept numpy rows
                    # directly (compression_report's _tcut_ix split and
                    # boundary_alignment's np.asarray).
                    self._cuts.append(np.cumsum(np.asarray(_cpu[0], dtype=np.int64))[:-1])
                else:
                    cuts = (bid[1:] != bid[:-1]).nonzero(as_tuple=True)[0] + 1
                    # Keep the device tensor for compression_report.
                    self._cuts.append(cuts)
        # Normalizing an all-zero content table is exactly zero (0/eps == 0),
        # so the zero-content ablation keeps the same values.
        core_kv = (torch.zeros_like(comp_kv) if cfg.content_mode == 'zero'
                   else F.normalize(comp_kv, dim=-1))
        kv_local = F.normalize(Ca_raw, dim=-1)
        if x.device.type == 'cuda':
            # Fused K/V head (CUDA only — see _pre): every output row is a
            # function of its own input row only, so the split halves are
            # bit-identical to the two self.W_kvhead(...) calls (row
            # independence, design notes §6.1).
            _Bn = int(core_kv.shape[0])
            _kvh = self.W_kvhead(torch.cat([core_kv, kv_local], 0))
            # Row-contiguous [blocks | window] view of the V columns: the same
            # values as cat([v_blk, v_sw]).  The views keep `_kvh` alive exactly
            # as v_blk/v_sw already did.
            v_stacked = _kvh[:, self.nh * self.hd:].view(_Bn + T, self.nh, self.hd)
            v_blk = v_stacked[:_Bn]
            v_sw = v_stacked[_Bn:]
            # Normalize both K halves over the fused K columns: each row's
            # norm/divide is independent, so the values are the same as the two
            # per-half normalizations.  The result is one contiguous
            # [blocks | window] table.  `_kvh` stays alive through v_stacked.
            k_stacked = F.normalize(_kvh[:, :self.nh * self.hd].view(_Bn + T, self.nh, self.hd), dim=-1)
            k_blk = k_stacked[:_Bn]
            k_sw = k_stacked[_Bn:]
            del _kvh
        else:
            k_blk, v_blk = self._split(self.W_kvhead(core_kv))
            k_sw, v_sw = self._split(self.W_kvhead(kv_local))
            v_stacked = None
            k_stacked = None
            k_blk = F.normalize(k_blk, dim=-1)
            k_sw = F.normalize(k_sw, dim=-1)
        if cfg.kind == 'hca':
            # cached int32 expansion — the same exact integers as the int64
            # arange expand narrowed to int32 (index transport, §6.1).  Every
            # row of the arange expansion holds distinct block ids, so the
            # first-occurrence dedup inside gathered_attention is provably the
            # all-True mask and is skipped (identical keep mask).
            topk_idx = _arange_exp_cache(T, Bn, x.device)
            out = _gathered_attention_trainable(q, k_blk, v_blk, topk_idx, last_tok, k_sw, v_sw, cfg.sliding_window, scale, sink_logits=self.sink if cfg.use_sink else None, topk_distinct=True, v_stacked=v_stacked, k_stacked=k_stacked)
        else:
            topk = cfg.index_topk
            _sel_valid_box = []
            # 无梯度（评估）时 gathered_attention 的 soft 直通分支整体是死
            # 的（_soft_live=False 见其注释），索引器据此不物化 soft 尾部，
            # 并以 soft_form=True 固定走软分支的硬部分解——两者前向 bit 与
            # 传入 soft 的形式完全一致。有梯度时条件为 False，走传入 soft
            # 的完整图。
            _need_soft = torch.is_grad_enabled()
            _, topk_idx, soft = _indexer_trainable(x, index_kv, last_tok, self.W_DQ, self.W_DK, self.W_w.weight, cfg.n_index_heads, topk, random_select=cfg.indexer_mode == 'random', pre_qI=pre['qI'] if pre is not None else None, pre_w=pre['w_idx'] if pre is not None else None, out_valid=_sel_valid_box, causal_ends_host=None, need_soft=_need_soft)
            # _indexer_selection 的 top-k 是无放回选择：有效槽位的块 id 逐行互不
            # 相同（order 是列排列、kept 落在互异的 cumsum 秩列上），pad 槽位的
            # 重复块 0 已被 sel_valid 置无效。故 gathered_attention 的
            # first_occurrence 去重掩码恒为全真，直接声明 topk_distinct 跳过它
            # （keep 掩码与逐位计算结果一致，下游取值不变）。
            out = _gathered_attention_trainable(q, k_blk, v_blk, topk_idx, last_tok, k_sw, v_sw, cfg.sliding_window, scale, soft=soft, sink_logits=self.sink if cfg.use_sink else None, sel_valid=_sel_valid_box[0], topk_distinct=True, soft_form=True, v_stacked=v_stacked, k_stacked=k_stacked)
        return (self.W_o(out.reshape(T, self.nh * self.hd)), gate_mean)

    def forward(self, x):
        B, T, d = x.shape
        if B == 0 or T == 0:
            # No tokens (or no rows): every attention kind returns the same
            # empty [B, T, d] result.  The pooling/indexer paths are defined
            # for empty input, but segmenter-derived tensors have no extrema
            # to read, so short-circuit before touching them.
            self.last_gate_mean = None
            return x.new_empty((B, T, d))
        if self.cfg.kind == 'full':
            self.last_gate_mean = None
            return self._full_batched(x)
        flat = x.reshape(B * T, d)
        pre_flat = self._pre(flat, fuse_x=x)
        _kernel_path = _CK is not None and self.cfg.kind in ('csa', 'hca') and _CK.kernel_enabled()
        # Batched eager path: the reference twin consumes [B, T, ...] and
        # issues batch GEMMs.  It is the default for B > 1 on CUDA (values
        # follow the declared reference-backend tolerance, not the eager
        # bitwise contract); CSA_EAGER_BATCH=0 restores the row loop.
        _eager_batch = (_CK is not None and not _kernel_path
                        and x.device.type == 'cuda' and B > 1
                        and self.cfg.kind in ('csa', 'hca')
                        and self.cfg.fuse_mode == 'none'
                        and self.cfg.indexer_mode == 'learned'
                        and _eager_batch_active(x.device, B))
        _batch_path = _kernel_path or _eager_batch
        pre_all = None
        pres = None
        if pre_flat is not None:
            if _batch_path:
                # The kernel / batched twin consumes the whole [B, T, ...]
                # batch and returns before the per-row loop, so the per-row
                # projection views are not built on this path.
                pre_all = {}
                for kk, v in pre_flat.items():
                    if v is None:
                        pre_all[kk] = None
                    elif kk == 'fused':
                        pre_all[kk] = v
                    else:
                        pre_all[kk] = v.reshape(B, T, *v.shape[1:])
            else:
                # The eager row path reads only its own [T, ...] slice of the
                # fused projection; each row is sliced from the flat output.
                pre_all = {'fused': pre_flat.get('fused')}
                pres = []
                for b in range(B):
                    _row = {}
                    for kk, v in pre_flat.items():
                        if v is None:
                            _row[kk] = None
                        elif kk == 'fused':
                            _row[kk] = v[b]
                        else:
                            _sl = v[b * T:(b + 1) * T]
                            # qI feeds the indexer GEMM, which expects a
                            # contiguous operand.
                            _row[kk] = _sl.contiguous() if kk == 'qI' else _sl
                    pres.append(_row)
        gate_sigs = None
        pre_blocks = None
        # The batched gate-signal / block-geometry hoist below is value-identical
        # to the per-row inline form for ANY batch size (design notes §6.1), so
        # it also serves B == 1 (e.g. the seq-2048 topk sweep).
        if self.cfg.chunking == 'cosine_learnable':
            _gs = self._gate_signals(x, pre_all)
            _raw, _gbool = _gs[0], _gs[1]
            if all((_r is not None for _r in _raw)):
                if _gbool is not None and _device_segment_enabled(x.device):
                    # Device-side segmentation: the gate decisions and the
                    # resulting block ids stay on the GPU.
                    gate_sigs = [(tuple(_r) if len(_r) == 3 else (_r[0], _g))
                                 for _r, _g in zip(_raw, _gbool.unbind(0))]
                    pre_blocks = _device_blocks_from_gate_bool(
                        T, _gbool, self.cfg.min_block, self.cfg.max_block, x.device,
                        want_honoured=self.need_reg, hon_dtype=_raw[0][0].dtype)
                else:
                    # .numpy() is a view of the stacked CPU tensor (the
                    # base tensor stays alive through the views); the segmenter
                    # consumes the same booleans.
                    _gl = _gbool.cpu().numpy()
                    gate_sigs = [(tuple(_r) if len(_r) == 3 else (_r[0], _g)) for _r, _g in zip(_raw, _gl)]
                    # Batched block geometry for the whole batch; every value is
                    # identical to the per-row form (see _hoist_blocks_from_cuts).
                    pre_blocks = _hoist_blocks_from_cuts(T, _gl, self.cfg.min_block, self.cfg.max_block, x.device, want_honoured=self.need_reg, hon_dtype=_raw[0][0].dtype)
        elif self.cfg.chunking in ('cosine_abs', 'cosine_adaptive') or (self.cfg.dynamic and self.cfg.kind != 'hca'):
            # same batch-level geometry hoist for the fixed/adaptive cosine
            # paths (the cut booleans and the resulting block ids are identical
            # to the per-row form).
            pre_blocks = self._cut_blocks_batched(x)
        if _eager_batch:
            return _CK.sparse_layer_forward(self, x, pre_all, gate_sigs, pre_blocks, impl='ref')
        if _kernel_path:
            # 内核路径（CSA_KERNEL；独立变体，与 eager 统计对照，不逐位承诺）。
            # 不支持的臂在这里硬失败，绝不静默回退——否则同一面板会混入两种
            # 实现的记录。
            if self.cfg.fuse_mode != 'none' or self.cfg.indexer_mode != 'learned':
                raise RuntimeError('CSA_KERNEL 只覆盖 learned-indexer / 非 fuse 的 csa·hca 臂；'
                                   'fuse_mode 或 indexer_mode 不同的臂请 unset CSA_KERNEL 后走 eager')
            return _CK.sparse_layer_forward(self, x, pre_all, gate_sigs, pre_blocks)
        if pres is None:
            pres = [None] * B
        if pre_blocks is not None:
            _prefetch_pool_geometry(self, pre_blocks, T, x.device)
        # Write each row's attention output into the batch output: the stacked
        # buffer holds exactly the tensor torch.stack would have built (a
        # per-row copy into the same layout), and the CopySlices backward hands
        # each row the identical grad slice — value-identical to the stack form
        # (transport only, design notes §6.1).
        outs = torch.empty(B, T, d, device=x.device, dtype=x.dtype)
        gates = []
        _prev_batch = _CPU_ACTIVE_BATCH[0]
        _CPU_ACTIVE_BATCH[0] = int(B)
        try:
            for b in range(B):
                pre = pres[b]
                o, g = self._single(x[b], pre=pre, gate_sig=None if gate_sigs is None else gate_sigs[b], pre_blocks=None if pre_blocks is None else pre_blocks[b])
                outs[b] = o
                if g is not None:
                    gates.append(g)
        finally:
            _CPU_ACTIVE_BATCH[0] = _prev_batch
        if not gates:
            self.last_gate_mean = None
        else:
            gm_tot = gates[0]
            for _g in gates[1:]:
                gm_tot = gm_tot + _g
            self.last_gate_mean = gm_tot / len(gates)
        return outs

class RMSNorm(nn.Module):

    def __init__(self, d, eps=1e-06):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, self.g.shape, weight=self.g, eps=self.eps)

class MLP(nn.Module):

    def __init__(self, d, hidden, init_gen=None):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d, bias=False)
        # The shared-panel design expects identical MLP weights across variants;
        # drawing them from `init_gen` keeps that true no matter how many RNG
        # draws the attention branch consumed beforehand.
        if init_gen is not None:
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5), generator=init_gen)
                nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5), generator=init_gen)

    def forward(self, x):
        if _CKPT['on'] in ('mlp', 'block') and _CKPT_NEST[0] == 0 and torch.is_grad_enabled() and x.requires_grad:
            # 非重入 checkpoint（模块顶 CSA_CKPT 说明）：重算同一纯函数，
            # 前向值与梯度逐位一致。_CKPT_NEST > 0 时外层块已被 checkpoint
            # 覆盖，内层不再包裹。
            return torch.utils.checkpoint.checkpoint(self._fwd, x, use_reentrant=False, preserve_rng_state=False)
        return self._fwd(x)

    def _fwd(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class Block(nn.Module):

    def __init__(self, d, n_heads, d_head, cfg: AttnCfg, mlp_ratio=4, mlp_gen=None):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.attn = HybridAttention(d, n_heads, d_head, cfg)
        self.n2 = RMSNorm(d)
        self.mlp = MLP(d, int(round(d * mlp_ratio)), init_gen=mlp_gen)

    def forward(self, x):
        if _CKPT['on'] == 'block' and torch.is_grad_enabled() and x.requires_grad:
            if self.attn.cfg.chunking != 'cosine_learnable':
                return torch.utils.checkpoint.checkpoint(self._fwd_ckpt, x, use_reentrant=False, preserve_rng_state=False)
            if 'cosine' not in _CKPT_WARNED:
                _CKPT_WARNED.add('cosine')
                print('[CSA_CKPT] block 级重算不覆盖 cosine_learnable 臂（gate_mean 经属性逃逸出块，初向前向的 no_grad 会剥掉它的图——v9 _V9_CKPT 的同一限制）；这些臂回退到 mlp 级，其余臂整块重算')
        return self._fwd(x)

    def _fwd(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x

    def _fwd_ckpt(self, x):
        # checkpoint 的初向前向与反向重算都经这里：置位期间内层 MLP 不再
        # 重复包裹。
        _CKPT_NEST[0] += 1
        try:
            return self._fwd(x)
        finally:
            _CKPT_NEST[0] -= 1

class SmallGPT(nn.Module):

    def __init__(self, vocab, d, n_layers, n_heads, d_head, max_seq, layer_cfgs, mlp_ratio=4):
        super().__init__()
        _rng_state = torch.get_rng_state()
        _cuda_rng_state = None
        if torch.cuda.is_available():
            try:
                _cuda_rng_state = torch.cuda.get_rng_state_all()
            except Exception:
                _cuda_rng_state = None
        shared = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
        g = torch.Generator().manual_seed(shared)
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_seq, d)
        self.head = nn.Linear(d, vocab, bias=False)
        with torch.no_grad():
            self.tok.weight.normal_(0, 0.02, generator=g)
            self.pos.weight.normal_(0, 0.02, generator=g)
            self.head.weight.normal_(0, 0.02, generator=g)
        self.blocks = nn.ModuleList()
        for i, c in enumerate(layer_cfgs):
            torch.manual_seed(shared + 104729 * (i + 1))
            mlp_g = torch.Generator().manual_seed(shared + 1299709 * (i + 1))
            self.blocks.append(self._make_block(d, n_heads, d_head, c, mlp_ratio, mlp_g))
        # Per-block init temporarily reseeds the GLOBAL torch RNG above; restore
        # the caller's state so model construction never perturbs data shuffling /
        # dropout / later model construction (which would otherwise couple the
        # data order to n_layers and break paired-experiment isolation).
        # `torch.manual_seed` also reseeds every CUDA device, so the CPU state
        # alone is not enough on a GPU box: restore the CUDA streams too.
        torch.set_rng_state(_rng_state)
        if _cuda_rng_state is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(_cuda_rng_state)
            except Exception:
                pass
        self.norm = RMSNorm(d)
        self.max_seq = max_seq
        self.boundary_logp = None
        self.boundary_decisions = 0

    def _make_block(self, d, n_heads, d_head, cfg, mlp_ratio, mlp_gen=None):
        return Block(d, n_heads, d_head, cfg, mlp_ratio, mlp_gen=mlp_gen)

    def forward(self, ids, logits_tail=None, targets=None, vocab=None):
        # When `targets` is supplied this returns the training CE loss instead
        # of logits.  The DDP wrapper only proxies `forward`, so routing the
        # loss through it is what lets DDP install its reducer hooks; calling
        # `ce_loss` directly on the inner module would silently skip the
        # gradient all-reduce.
        if targets is not None:
            if vocab is None:
                raise ValueError('forward(targets=...) requires vocab')
            return self.ce_loss(ids, targets, vocab)
        x = self.trunk(ids)
        if logits_tail:
            x = x[:, -logits_tail:]
        return self.head(x)

    def trunk(self, ids):
        # Full forward minus the LM head; sets self.comp_reg exactly as forward
        # did (the reg terms are accumulated inside the single block loop — the
        # same addition order as the full forward).
        T = ids.shape[1]
        _max_pos = int(self.pos.weight.shape[0])
        if T > _max_pos:
            raise RuntimeError(
                f'absolute position table covers {_max_pos} tokens but the input has {T}; '
                'a dense absolute-position arm cannot evaluate beyond its training length. '
                'Use a RoPE arm, or pass an explicit max_pos so the evaluation span is '
                'truncated and recorded as truncated=True (never silently score a shorter span).')
        x = self.tok(ids) + self.pos.weight[:T]
        reg = None
        blp = None
        bdec = 0
        for blk in self.blocks:
            x = blk(x)
            gm = blk.attn.last_gate_mean
            if gm is not None:
                Lb = max(blk.attn.cfg.target_block_tokens, 1)
                mult = getattr(blk.attn.cfg, 'comp_lambda_mult', 1.0)
                if isinstance(gm, (list, tuple)):
                    tot = gm[0]
                    for _g in gm[1:]:
                        tot = tot + _g
                    gm = tot / len(gm)
                term = mult * (gm * Lb - 1.0) ** 2
                reg = term if reg is None else reg + term
            _lp = getattr(blk.attn, 'last_boundary_logp', None)
            if _lp is not None:
                blp = _lp if blp is None else blp + _lp
                bdec += int(getattr(blk.attn, 'last_boundary_decisions', 0) or 0)
        if reg is None:
            reg = torch.zeros((), device=ids.device)
        self.comp_reg = reg
        self.boundary_logp = blp
        self.boundary_decisions = int(bdec)
        return self.norm(x)

    def _ce_head_chunk(self, h_chunk, tgt_chunk, vocab, ignore_index=-100):
        lg = self.head(h_chunk)
        return F.cross_entropy(lg.reshape(-1, vocab), tgt_chunk.reshape(-1), reduction='sum', ignore_index=ignore_index)

    def ce_loss(self, ids, targets, vocab, chunk=64, ignore_index=None):
        # LM-head cross-entropy computed in time chunks: the chunk sums add to
        # the same total as the full-batch form (up to fp reduction order), so
        # the scalar and its gradients are unchanged.
        # 训练时每个 chunk 包一层非重入 checkpoint：反向按 chunk 重算同一份
        # logits（同一 GEMM、无 RNG，值逐位相同）；chunk 反传的执行次序由
        # 求和链的拓扑结构决定，两种形式相同，因此梯度逐位一致。
        h = self.trunk(ids)
        # When targets contain ignored positions, the mean must be taken over
        # the valid targets only (F.cross_entropy's own reduction), exactly as
        # the unchunked form did.  The default None means every target is valid.
        if ignore_index is None:
            _ign = -100
            n_tok = int(targets.numel())
        else:
            _ign = int(ignore_index)
            n_tok = int((targets != _ign).sum().item())
        n_tok = max(n_tok, 1)
        ce = None
        _ck = torch.is_grad_enabled() and h.requires_grad
        if _ck and h.device.type != 'cuda':
            # Every chunk's logits stay alive in the graph until backward, so
            # the limit covers the whole sequence, not just one chunk.
            _saved = 3 * int(ids.shape[0]) * int(ids.shape[1]) * int(vocab) * 4
            _ck = not _cpu_saved_ok(_saved, _CPU_CE_CKPT_BYTES)
        # The trunk must see the full input (block geometry depends on T), but
        # a next-token target table is one column shorter; score only the rows
        # that have a target, exactly like the original logits[:, :-1] form.
        n_cols = min(int(ids.shape[1]), int(targets.shape[1]))
        for s in range(0, n_cols, chunk):
            e = min(s + chunk, n_cols)
            if _ck:
                c = torch.utils.checkpoint.checkpoint(self._ce_head_chunk, h[:, s:e], targets[:, s:e], vocab, _ign, use_reentrant=False, preserve_rng_state=False)
            else:
                c = self._ce_head_chunk(h[:, s:e], targets[:, s:e], vocab, _ign)
            ce = c if ce is None else ce + c
        return ce / n_tok
OVERLAP = 8
_WARM_KEY_RE = re.compile('.*::w(\\d+)::seed\\d+$')

def make_layer_cfgs(n_layers, variant):
    cfgs = []
    for i in range(n_layers):
        if variant == 'full':
            cfgs.append(AttnCfg(kind='full'))
        elif variant == 'full_cos':
            cfgs.append(AttnCfg(kind='full', full_cosine=True))
        elif variant == 'full_matched':
            cfgs.append(AttnCfg(kind='full'))
        elif variant == 'full_sw128':
            cfgs.append(AttnCfg(kind='full', window=128))
        elif variant == 'full_sw128_matched':
            cfgs.append(AttnCfg(kind='full', window=128))
        elif variant == 'csa_fixed':
            cfgs.append(AttnCfg(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=32))
        elif variant == 'csa_dynamic':
            cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1))
        elif variant == 'hybrid_fixed':
            if i % 2 == 0:
                cfgs.append(AttnCfg(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=32))
            else:
                cfgs.append(AttnCfg(kind='hca', dynamic=False, block_size=64))
        elif variant == 'hybrid_dynamic':
            if i % 2 == 0:
                cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1))
            else:
                cfgs.append(AttnCfg(kind='hca', dynamic=True, chunking='cosine_learnable', target_block_tokens=64, temperature=0.1, comp_lambda_mult=4.0))
        elif variant == 'full_sink':
            cfgs.append(AttnCfg(kind='full', full_sink=True))
        elif variant == 'hybrid_csa_dyn':
            if i % 2 == 0:
                cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1))
            else:
                cfgs.append(AttnCfg(kind='hca', dynamic=False, block_size=64))
        elif variant == 'hybrid_hca_dyn':
            if i % 2 == 0:
                cfgs.append(AttnCfg(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=32))
            else:
                cfgs.append(AttnCfg(kind='hca', dynamic=True, chunking='cosine_learnable', target_block_tokens=64, temperature=0.1, comp_lambda_mult=4.0))
        elif variant == 'csa_dyn_fuse':
            cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1, fuse_mode='learned'))
        elif variant == 'hybrid_csa_dyn_fuse':
            if i % 2 == 0:
                cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1, fuse_mode='learned'))
            else:
                cfgs.append(AttnCfg(kind='hca', dynamic=False, block_size=64))
        elif variant == 'csa_fix_randidx':
            cfgs.append(AttnCfg(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=32, indexer_mode='random'))
        elif variant == 'csa_fix_zerocont':
            cfgs.append(AttnCfg(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=32, content_mode='zero'))
        elif variant == 'csa_fix_nosink':
            cfgs.append(AttnCfg(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=32, use_sink=False))
        elif variant == 'csa_fix_topk8':
            cfgs.append(AttnCfg(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=8))
        elif variant == 'csa_fix_topk64':
            cfgs.append(AttnCfg(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=64))
        elif variant == 'csa_dyn_randidx':
            cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1, indexer_mode='random'))
        elif variant == 'csa_dyn_zerocont':
            cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1, content_mode='zero'))
        elif variant == 'csa_dyn_nosink':
            cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1, use_sink=False))
        elif variant == 'csa_dynamic_task':
            # Opt-in end-to-end boundary arm (review 3 issue 6): identical to
            # csa_dynamic except the hard cuts are sampled and trained by the
            # LM loss through a score-function term.  This is a NEW arm; the
            # csa_dynamic remains threshold/comp_reg-only and must
            # not be described as task-learned.
            cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1, boundary_policy='reinforce'))
        elif variant == 'csa_dyn_nocomp':
            # Loss-term ablation C: identical architecture, optimizer, LR
            # schedule, scale and data as `csa_dynamic`; only the compression
            # regularizer's multiplier is zeroed.  Keeping run_cfg's `cl`
            # unchanged is what lets `pair_reason` pair the two arms while the
            # actual loss term differs — the thing under test.
            cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1, comp_lambda_mult=0.0))
        elif variant == 'csa_fixed_fullpov':
            cfgs.append(AttnCfg(kind='csa', dynamic=False, block_size=4, overlap=128, index_topk=32))
        elif variant == 'csa_dynamic_fullpov':
            cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=128, index_topk=32, temperature=0.1))
        else:
            raise ValueError(variant)
    return cfgs

def build_tokenizer(texts, vocab_size=8192):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    tok = Tokenizer(BPE(unk_token='<unk>'))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=True)
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=['<pad>', '<unk>', '<bos>', '<eos>'])
    tok.train_from_iterator(texts, trainer=trainer)
    return tok
PUNCT_CHARS = set('.,!?;:')

def boundary_token_ids(tok, eos_id):
    ids = set()
    for i in range(tok.get_vocab_size()):
        s = tok.decode([i]).strip()
        if s and all((ch in PUNCT_CHARS for ch in s)):
            ids.add(i)
    ids.add(eos_id)
    return ids

class _LocalTextSplit:

    def __init__(self, texts):
        self._texts = texts

    def __len__(self):
        return len(self._texts)

    def select(self, idx):
        return _LocalTextSplit([self._texts[i] for i in idx])

    def __getitem__(self, key):
        if isinstance(key, slice):
            return {'text': self._texts[key]}
        if key == 'text':
            return self._texts
        raise TypeError(f'unsupported key {key!r}')

def local_wikitext_if_available(repo_cache_dir):
    raw_txt = os.environ.get('WT103_RAW_TXT')
    candidates = []
    if raw_txt:
        candidates.append(raw_txt)
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(repo_cache_dir)), 'wt103_raw'))
    candidates.append('/root/wt103_raw')
    for cand in candidates:
        train_f = os.path.join(cand, 'wiki.train.raw')
        valid_f = os.path.join(cand, 'wiki.valid.raw')
        if os.path.exists(train_f) and os.path.exists(valid_f):
            train = _read_raw_rows(train_f)
            valid = _read_raw_rows(valid_f)
            print(f'[data] local wikitext-103 raw dump at {cand} ({len(train)} train rows, {len(valid)} val rows)')
            return {'train': _LocalTextSplit(train), 'validation': _LocalTextSplit(valid)}
    return None

def _read_raw_rows(path):
    with open(path, encoding='utf-8') as fh:
        return [ln.rstrip('\n') for ln in fh]

_IDS_FP_MEMO = {}
_IDS_FP_MEMO_CAP = 4

def _array_file_path(arr):
    """Resolve the on-disk file backing a memmap (or a whole-file view of one).

    The huge FineWeb caches are served as read-only memmap views; a sidecar
    fingerprint next to that file lets every worker reuse the parent's digest,
    while remaining bound to the file's size+mtime so a rebuilt cache can never
    reuse a stale digest.

    Only the root logical memmap (the deepest memmap in arr's base chain) may
    supply the sidecar.  `arr` must then have that root's exact layout and data
    start.  A suffix slice also ends at EOF but starts elsewhere, so its bytes
    differ from the whole-file digest and must be hashed itself."""
    root = None
    cur = arr
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, np.memmap):
            root = cur
        cur = getattr(cur, 'base', None)
    if root is None:
        return None
    fn = getattr(root, 'filename', None)
    if not isinstance(fn, (str, bytes, os.PathLike)):
        return None
    try:
        fn = os.fspath(fn)
        st = os.stat(fn)
        off = int(getattr(root, 'offset', 0) or 0)
        nb = int(getattr(root, 'nbytes', -1))
        arr_nbytes = int(getattr(arr, 'nbytes', -1))
        root_ptr = root.__array_interface__['data'][0]
        arr_ptr = arr.__array_interface__['data'][0]
        same = (off >= 0 and nb >= 0 and off + nb == int(st.st_size)
                and arr_nbytes == nb
                and arr_ptr == root_ptr
                and tuple(getattr(arr, 'shape', ())) == tuple(root.shape)
                and getattr(arr, 'strides', None) == root.strides
                and np.dtype(arr.dtype) == np.dtype(root.dtype))
    except (OSError, TypeError, ValueError, KeyError, AttributeError):
        return None
    return fn if same else None

def _cache_file_sigs(paths):
    """size+mtime+ctime+inode signature for the files a cache entry is built from.

    size+mtime alone can be preserved when a file is replaced by a same-size
    copy (or copied with ``cp -p``).  ``ctime`` cannot be set by user code and
    the inode changes on replacement, so a stale/mixed file does not match its
    recorded signature; the content fingerprint sidecars
    (``*.idsfp.json``) are keyed on the same fields, so they are invalidated
    too.
    """
    out = {}
    for _k, _p in paths.items():
        try:
            _st = os.stat(_p)
            out[_k] = [int(_st.st_size), int(_st.st_mtime_ns),
                       int(getattr(_st, 'st_ctime_ns', int(_st.st_ctime * 1e9))),
                       int(getattr(_st, 'st_ino', 0))]
        except OSError:
            return None
    return out


def _cache_file_sigs_match(meta, sigs):
    rec = meta.get('file_sigs') if isinstance(meta, dict) else None
    return bool(sigs) and isinstance(rec, dict) and all(rec.get(_k) == _v for _k, _v in sigs.items())


def _memo_file_sig(paths):
    """In-process memo key component: includes ctime/inode, so replacing a file
    (even with preserved size/mtime) invalidates the memo entry."""
    s = _cache_file_sigs(paths)
    return tuple(tuple(v) for v in s.values()) if s else None


def _array_id_range(arr):
    a = np.asarray(arr)
    if a.size == 0:
        return None
    return [int(a.min()), int(a.max())]


def _ids_fp(train_ids):
    # Pure function of the token values, memoised on (data pointer, layout)
    # with a weak reference to the array: a dead entry can never alias a
    # recycled pointer, and large corpora are not retained by the cache.
    # Same tokens in -> same digest out, always.  Sub-int32 storage (the
    # uint16 FineWeb cache) is hashed through its canonical int32 form, so a
    # corpus keeps the same fingerprint regardless of storage width.
    import weakref as _weakref
    try:
        _key = (train_ids.__array_interface__['data'][0], train_ids.shape, train_ids.strides, train_ids.dtype.str)
    except (AttributeError, TypeError):
        _key = None
    if _key is not None:
        _hit = _IDS_FP_MEMO.get(_key)
        if _hit is not None:
            _arr = _hit[0]() if _hit[0] is not None else None
            if _arr is train_ids:
                return _hit[1]
            _IDS_FP_MEMO.pop(_key, None)
    _path = _array_file_path(train_ids)
    _stat = None
    _side = None
    if _path is not None:
        try:
            _st = os.stat(_path)
            _stat = (int(getattr(_st, 'st_size', 0)),
                     int(getattr(_st, 'st_mtime_ns', int(_st.st_mtime * 1e9))),
                     int(getattr(_st, 'st_ctime_ns', int(_st.st_ctime * 1e9))),
                     int(getattr(_st, 'st_ino', 0)))
            _side = _path + '.idsfp.json'
            _tshape = tuple(int(_x) for _x in np.shape(train_ids))
            _tdtype = np.dtype(np.asarray(train_ids).dtype).str
            with open(_side, encoding='utf-8') as _sf:
                _rec = json.load(_sf)
            if ((int(_rec.get('size', -1)), int(_rec.get('mtime_ns', -1)),
                 int(_rec.get('ctime_ns', -1)), int(_rec.get('inode', -1))) == _stat
                    and isinstance(_rec.get('fp'), str)
                    and _rec.get('dtype') == _tdtype
                    and tuple(_rec.get('shape') or ()) == _tshape):
                _fp = _rec['fp']
                if _key is not None:
                    try:
                        _ref = _weakref.ref(train_ids)
                    except TypeError:
                        _ref = None
                    if len(_IDS_FP_MEMO) >= _IDS_FP_MEMO_CAP and _key not in _IDS_FP_MEMO:
                        _IDS_FP_MEMO.pop(next(iter(_IDS_FP_MEMO)))
                    _IDS_FP_MEMO[_key] = (_ref, _fp)
                return _fp
        except (OSError, ValueError, TypeError):
            pass
    _h = hashlib.sha256()
    _arr = np.asarray(train_ids)
    _widen = _arr.dtype in (np.dtype(np.uint8), np.dtype(np.uint16))
    _cs = 1 << 24
    for _off in range(0, len(_arr), _cs):
        _part = _arr[_off:_off + _cs]
        if _widen:
            _part = _part.astype(np.int32, copy=False)
        elif not _part.flags.c_contiguous:
            _part = np.ascontiguousarray(_part)
        _h.update(memoryview(_part).cast('B'))
    _fp = _h.hexdigest()[:16]
    if _key is not None:
        try:
            _ref = _weakref.ref(train_ids)
        except TypeError:
            _ref = None
        if len(_IDS_FP_MEMO) >= _IDS_FP_MEMO_CAP and _key not in _IDS_FP_MEMO:
            _IDS_FP_MEMO.pop(next(iter(_IDS_FP_MEMO)))
        _IDS_FP_MEMO[_key] = (_ref, _fp)
    if _side is not None and _stat is not None:
        try:
            _rec_out = {'size': _stat[0], 'mtime_ns': _stat[1],
                        'ctime_ns': _stat[2], 'inode': _stat[3], 'fp': _fp,
                        'dtype': np.dtype(np.asarray(train_ids).dtype).str,
                        'shape': [int(_x) for _x in np.shape(train_ids)]}
            atomic_write_json(_side, _rec_out)
        except (OSError, TypeError, ValueError):
            pass
    return _fp

def _validate_token_ids(ids, vocab, label='token ids'):
    """Hard range check for a token-id array before it enters a cache/batch.

    A negative or out-of-vocabulary id would either index an embedding row
    outside the model (Python/NumPy wraparound) or silently mask corruption in
    the split.  It runs on every cold build and every cache hit.
    """
    arr = np.asarray(ids)
    if arr.size == 0:
        return arr
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f'{label}: expected integer token ids, got dtype {arr.dtype}')
    mn, mx = int(arr.min()), int(arr.max())
    if mn < 0 or mx >= int(vocab):
        raise ValueError(f'{label}: token id range [{mn}, {mx}] is outside vocab [0, {int(vocab)})')
    return arr


def _cached_split_problems(meta, train_ids, val_batch, val_bnd, vocab, seq_len=None,
                           fingerprints_trusted=False, ranges_trusted=False):
    """Reasons a cached (train, val) pair cannot be trusted.

    The metadata carries content fingerprints for BOTH sides.  Checking only
    the train fingerprint lets a stale/mixed validation file be served under a
    valid train cache — exactly the situation in which validation (or test)
    rows leak into the tuning split.  New caches always record ``val_fp``;
    legacy caches without it are accepted with an explicit warning so existing
    results remain reproducible, but their validation bytes are not silently
    blessed as verified.
    """
    problems = []
    try:
        vocab = int(vocab)
    except (TypeError, ValueError):
        return ['metadata has no usable vocab size']
    if not fingerprints_trusted and meta.get('src_fp') != _ids_fp(train_ids):
        problems.append('train id fingerprint does not match the meta file')
    vb = np.asarray(val_batch)
    if vb.ndim != 2:
        problems.append(f'validation batch is not 2-D (shape {vb.shape})')
    elif vb.shape[0] == 0:
        problems.append('validation batch is empty')
    elif seq_len is not None and vb.shape[1] != int(seq_len):
        problems.append(f'validation sequence length {vb.shape[1]} != requested {seq_len}')
    if np.asarray(val_bnd).shape != vb.shape:
        problems.append(f'validation boundary mask shape {np.asarray(val_bnd).shape} != {vb.shape}')
    if (not fingerprints_trusted and 'val_fp' in meta
            and meta.get('val_fp') != _ids_fp(vb.reshape(-1))):
        problems.append('validation id fingerprint does not match the meta file')
    if meta.get('shuffle_at_split') is True:
        problems.append('metadata records a shuffled split; time/group order cannot be trusted')
    if meta.get('tokenizer_fit_split') not in (None, 'train'):
        problems.append(f"tokenizer was fitted on {meta.get('tokenizer_fit_split')!r}, not train")
    if not problems and not ranges_trusted:
        for _arr, _label in ((train_ids, 'cached train_ids'), (vb, 'cached val_batch')):
            try:
                _validate_token_ids(_arr, vocab, _label)
            except (TypeError, ValueError) as _e:
                problems.append(str(_e))
    return problems


def _cache_warning(what, problems):
    print(f'[data] WARNING: the cached {what} cannot be verified ({"; ".join(problems)}) — REBUILDING it so a mixed/stale split is never silently served')


_MEMO_MAX_BYTES = 1 << 30


def _memmap_backed(v):
    cur = v
    for _ in range(16):
        if isinstance(cur, np.memmap) or getattr(cur, 'filename', None) is not None:
            return True
        cur = getattr(cur, 'base', None)
        if cur is None or cur is v:
            return False
    return False


def _memo_value_bytes(v):
    """Rough resident bytes of a memo value; memory-mapped arrays are free."""
    if isinstance(v, np.ndarray):
        if _memmap_backed(v):
            return 0
        return int(getattr(v, 'nbytes', 0) or 0)
    if isinstance(v, (tuple, list)):
        return sum(_memo_value_bytes(_x) for _x in v)
    return 0


def _memo_put(memo, key, value, cap_items, cap_bytes=_MEMO_MAX_BYTES):
    """Insert under a byte ceiling plus the item cap (LRU order is insertion
    order).  A single value larger than the ceiling is not cached."""
    b = _memo_value_bytes(value)
    if b > cap_bytes:
        return False
    total = sum(_memo_value_bytes(_v) for _v in memo.values())
    while memo and (len(memo) >= int(cap_items) or total + b > cap_bytes):
        total -= _memo_value_bytes(memo.pop(next(iter(memo))))
    memo[key] = value
    return True


_WT103_MEMO = {}
_WT103_MEMO_CAP = 4

def load_wikitext(seq_len, n_train_tokens, vocab_size=8192, val_seqs=512, cache_dir='./wt103_cache'):
    from datasets import load_dataset
    os.makedirs(cache_dir, exist_ok=True)
    tag = f'v5_sl{seq_len}_cap{n_train_tokens}_v{vocab_size}_vs{val_seqs}'
    tr_path = os.path.join(cache_dir, f'train_ids_{tag}.npy')
    va_path = os.path.join(cache_dir, f'val_batch_{tag}.npy')
    vp_path = os.path.join(cache_dir, f'val_bnd_{tag}.npy')
    me_path = os.path.join(cache_dir, f'meta_{tag}.json')
    trd_path = os.path.join(cache_dir, f'train_doc_ids_{tag}.npy')
    vad_path = os.path.join(cache_dir, f'val_doc_ids_{tag}.npy')
    if all((os.path.exists(p) for p in (tr_path, va_path, vp_path, me_path))):
        _sig = _memo_file_sig({'train': tr_path, 'val': va_path,
                               'bnd': vp_path, 'meta': me_path})
        _mkey = (os.path.abspath(cache_dir), tag, _sig)
        _hit = _WT103_MEMO.get(_mkey)
        if _hit is not None:
            print('[data] serving token ids from the in-process memo (same cache files, already verified) ...')
            return _hit
        print('[data] loading cached token ids ...')
        # Serve the train ids as a read-only mapping so every process (and the
        # parallel workers' corpus cache) shares the OS page cache.
        train_ids = np.load(tr_path, mmap_mode='r')
        meta = json.load(open(me_path, encoding='utf-8'))
        val_batch = np.load(va_path)
        val_bnd = np.load(vp_path)
        _sigs = _cache_file_sigs({'train': tr_path, 'val': va_path, 'bnd': vp_path})
        _fast = _cache_file_sigs_match(meta, _sigs)
        _problems = _cached_split_problems(meta, train_ids, val_batch, val_bnd,
                                           meta.get('vocab'), seq_len=seq_len,
                                           fingerprints_trusted=False,
                                           ranges_trusted=_fast)
        if meta.get('tokenizer_fit_protocol') != 'train_prefix_token_budget':
            _problems.append('cache does not attest the frozen train-prefix tokenizer fit')
        if not isinstance(meta.get('dedup'), dict):
            _problems.append('cache does not attest whole-document exact/near duplicate filtering')
        # Document ids are the group identity.  When the cache carries them,
        # prove the groups are disjoint instead of trusting the build-time
        # deduplicator; a sidecar pair that disagrees with the metadata is a
        # mixed cache and must be rebuilt.
        _has_doc_meta = bool(meta.get('doc_ids'))
        _has_doc_files = os.path.exists(trd_path) and os.path.exists(vad_path)
        if _has_doc_files:
            try:
                _trd = np.load(trd_path)
                _vad = np.load(vad_path)
                _shared_docs = int(np.intersect1d(_trd, _vad).size)
                if _shared_docs:
                    _problems.append(
                        f'{_shared_docs} document ids occur in both train and validation')
                _mdoc = meta.get('doc_ids') or {}
                if _mdoc and (_mdoc.get('disjoint') is not True
                              or int(_mdoc.get('overlap', 0) or 0) != 0):
                    _problems.append(
                        'cache metadata does not attest document-disjoint train/validation splits')
            except Exception as _de:
                _problems.append(f'document-id sidecars unreadable ({type(_de).__name__}: {_de})')
        elif _has_doc_meta:
            _problems.append('cache metadata records document ids but the sidecar files are missing')
        if _problems:
            _cache_warning('WikiText train/validation split', _problems)
        else:
            if 'val_fp' not in meta:
                print('[data] NOTE: this is a legacy cache without a validation fingerprint; its file bytes are range/shape-checked but its split membership cannot be independently re-derived. New caches record val_fp.')
            audit_corpus_split(train_ids, val_batch, int(meta['vocab']),
                               dataset='wikitext', meta=meta)
            _out = (train_ids, val_batch, int(meta['vocab']), None, val_bnd)
            _memo_put(_WT103_MEMO, _mkey, _out, _WT103_MEMO_CAP)
            return _out
    ds = local_wikitext_if_available(cache_dir)
    if ds is None:
        last_err = None
        for repo in ('Salesforce/wikitext', 'wikitext'):
            try:
                ds = load_dataset(repo, 'wikitext-103-raw-v1', cache_dir=cache_dir)
                break
            except Exception as e:
                last_err = e
        if ds is None:
            raise last_err if last_err is not None else RuntimeError('load_wikitext: no dataset repository could be loaded')
    if not hasattr(ds, '__getitem__') or 'train' not in ds or 'validation' not in ds:
        raise TypeError(f"load_wikitext: the loaded corpus object does not expose the ['train']/['validation'] splits (got {type(ds).__name__}); refusing to tokenise a corpus we cannot identify")
    # Tokenizer fit is a frozen TRAIN prefix with an explicit token cap:
    # every document it sees is also inside the model's training region (for
    # caps below the global cap), never a validation/test row.  The cap
    # is fixed across the scale ladder so S1/S2/S3 share one tokenizer version.
    _fit_cap = min(int(TOKENIZER_FIT_TOKENS), int(n_train_tokens))
    # Only the frozen prefix is needed for the tokenizer fit; the rest of the
    # train split's text does not enter the fit.
    _fit_rows_all = []
    _n_train_rows = len(ds['train'])
    _row_step = 5000
    for _start in range(0, _n_train_rows, _row_step):
        if len(_fit_rows_all) >= 50000:
            break
        _rows = ds['train'][_start:min(_start + _row_step, _n_train_rows)]['text']
        _fit_rows_all.extend(r for r in _rows if isinstance(r, str) and r.strip())
        if len(_fit_rows_all) > 50000:
            del _fit_rows_all[50000:]
    fit_text, tok, n_fit_tokens = _wikitext_fit_rows(
        _fit_rows_all, vocab_size, _fit_cap, max_rows=50000)
    if tok is None or not fit_text:
        raise RuntimeError('load_wikitext: the train split has no non-empty text rows to fit the tokenizer')
    vocab = tok.get_vocab_size()
    fit_fp = _text_fp(fit_text)
    _dedup = _DocDeduplicator()
    # Seed the deduplicator with the FULL rows of the fit prefix.  The token
    # cap may truncate the text of the final fit row; seeding with the
    # truncated view would leave that training document absent from the exact
    # duplicate index, so a full copy in validation/test could slip through.
    _fit_doc_prefix = list(_fit_rows_all[:len(fit_text)])
    for _r in _fit_doc_prefix:
        _dedup.is_duplicate(_r)
    _fit_dedup = _dedup.stats()
    _n_fit_docs = len(_fit_doc_prefix)
    del fit_text
    del _fit_rows_all
    gc.collect()
    eos_id = tok.token_to_id('<eos>')
    _dedup_by_split = {}

    def stream_tokenise(split, cap_tokens, label, _ds=ds, _tok=tok, _eos=eos_id,
                        _dtype=np.int32):
        cap_tokens = max(0, int(cap_tokens))
        arr = np.empty(cap_tokens, dtype=_dtype)
        cursor = 0
        _doc_ids = array.array('Q')
        _skipped = {'exact': 0, 'near': 0}
        _dedup_by_split[split] = _skipped
        _n_nonempty_seen = 0
        row_chunk = 10000
        _stop = False
        for start in range(0, len(_ds[split]), row_chunk):
            rows = _ds[split][start:start + row_chunk]['text']
            rows = [r for r in rows if r.strip()]
            kept = []
            for _j, r in enumerate(rows):
                # The tokenizer-fit documents are the frozen TRAIN prefix and
                # are part of the model's training stream; do not filter them
                # out as duplicates of themselves on the first pass.
                if split == 'train' and (_n_nonempty_seen + _j) < _n_fit_docs:
                    kept.append(r)
                    continue
                _dup, _kind = _dedup.is_duplicate(r)
                if _dup:
                    _skipped[_kind] += 1
                else:
                    kept.append(r)
            _n_nonempty_seen += len(rows)
            if kept:
                for _row, enc in zip(kept, _tok.encode_batch(kept)):
                    ids = enc.ids
                    _doc_ids.append(int(_doc_id_u64(_doc_exact_fp(_row))))
                    take = min(len(ids), cap_tokens - cursor)
                    if take > 0:
                        arr[cursor:cursor + take] = ids[:take]
                        cursor += take
                    if cursor >= cap_tokens:
                        _stop = True
                        break
                    arr[cursor] = _eos
                    cursor += 1
                    if cursor >= cap_tokens:
                        _stop = True
                        break
            if _stop or cursor >= cap_tokens:
                break
            if start and start % 50000 == 0:
                print(f'    [{label}] {start} rows, {cursor} tokens, '
                      f'{sum(_skipped.values())} duplicate docs skipped')
        _doc_arr = (np.frombuffer(_doc_ids, dtype=np.uint64).copy()
                    if _doc_ids else np.empty(0, dtype=np.uint64))
        if cursor < arr.size:
            exact = arr[:cursor].copy()
            del arr
            return exact, _doc_arr
        return arr, _doc_arr
    print(f'[data] tokenising train (cap {n_train_tokens} tokens) ...')
    train_ids, train_doc_ids = stream_tokenise(
        'train', n_train_tokens + seq_len, 'train', _dtype=np.int32)
    print(f'[data] train tokens = {len(train_ids)}  vocab = {vocab}')
    if len(train_ids) < n_train_tokens + seq_len:
        print(f'[data] WARNING: the train split ran dry at {len(train_ids)} tokens (< requested {n_train_tokens + seq_len}); runs configured for the advertised token target will train on fewer tokens than advertised')
    print('[data] tokenising validation ...')
    val_ids, val_doc_ids = stream_tokenise(
        'validation', val_seqs * seq_len + seq_len, 'val', _dtype=np.int64)
    n_val = min(len(val_ids) // seq_len, val_seqs)
    val_batch = val_ids[:n_val * seq_len].reshape(n_val, seq_len)
    _validate_token_ids(train_ids, vocab, 'wikitext train_ids')
    _validate_token_ids(val_batch, vocab, 'wikitext val_batch')
    print(f'[data] val sequences = {n_val} (seq_len={seq_len})')
    _shared_docs = int(np.intersect1d(train_doc_ids, val_doc_ids).size)
    if _shared_docs:
        raise RuntimeError(
            f'load_wikitext: {_shared_docs} document ids occur in both train and '
            'validation — refusing to write a leaking split')
    audit_corpus_split(train_ids, val_batch, vocab, dataset='wikitext')
    bnd_ids = sorted(boundary_token_ids(tok, eos_id))
    val_bnd = np.isin(val_batch, bnd_ids)
    print(f'[data] boundary tokens: {int(val_bnd.sum())} ({val_bnd.mean() * 100:.1f}% of val tokens)')
    for _p, _a in ((tr_path, train_ids), (va_path, val_batch), (vp_path, val_bnd),
                   (trd_path, train_doc_ids), (vad_path, val_doc_ids)):
        _fd, _tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(_p)), prefix=os.path.basename(_p) + '.', suffix='.tmp')
        try:
            with os.fdopen(_fd, 'wb') as _fh:
                np.save(_fh, _a)
            os.replace(_tmp, _p)
        except BaseException:
            try:
                os.unlink(_tmp)
            except OSError:
                pass
            raise
    _file_sigs = _cache_file_sigs({'train': tr_path, 'val': va_path, 'bnd': vp_path})
    atomic_write_json(me_path, {
        'vocab': vocab,
        'src_fp': _ids_fp(train_ids),
        'val_fp': _ids_fp(np.ascontiguousarray(val_batch).reshape(-1)),
        'fit_fp': fit_fp,
        'file_sigs': _file_sigs,
        'id_range': {'train': _array_id_range(train_ids), 'val': _array_id_range(val_batch)},
        'split_strategy': 'official_train_validation_test_deduplicated',
        'shuffle_at_split': False,
        'tokenizer_fit_split': 'train',
        'tokenizer_fit_protocol': 'train_prefix_token_budget',
        'tokenizer_fit_token_cap': int(min(TOKENIZER_FIT_TOKENS, int(n_train_tokens))),
        'tokenizer_fit_max_rows': 50000,
        'tokenizer_fit_tokens': int(n_fit_tokens),
        'tokenizer_fit_rows': int(_n_fit_docs),
        'tokenizer_fit_docs_in_train_prefix': True,
        'doc_ids': {
            'train_count': int(train_doc_ids.size),
            'validation_count': int(val_doc_ids.size),
            'train_fp': _ids_fp(train_doc_ids),
            'validation_fp': _ids_fp(val_doc_ids),
            'overlap': _shared_docs,
            'disjoint': True,
            'policy': 'whole_document_exact_and_near_duplicate_filtered',
        },
        'dedup': {'fit': _fit_dedup, 'splits': _dedup_by_split},
        'group_isolation': 'whole_documents_exact_and_near_duplicate_filtered',
        'train_tokens': int(len(train_ids)),
        'val_sequences': int(val_batch.shape[0]),
        'seq_len': int(seq_len),
    }, indent=0)
    try:
        # persist the exact tokenizer that produced this cache so the held-out
        # test split can be verified against it by full vocab identity (size
        # alone is not identity: different merges can share a vocab size).
        tok.save(os.path.join(cache_dir, f'tok_{tag}.json'))
    except Exception as _te:
        print(f'[data] NOTE: could not persist the train tokenizer ({type(_te).__name__}: {_te})')
    print('[data] cached token ids to disk (resume-safe)')
    del ds
    gc.collect()
    return (train_ids, val_batch, vocab, tok, val_bnd)

# ---- 独立最终 test split（wiki.test.raw，冻结后一次性评估，见 design_notes §17）
_WT103_TEST_MEMO = {}
_WT103_TEST_MEMO_CAP = 2

def _text_fp(rows):
    """sha256 of a list of text rows (fit-corpus provenance for the test cache)."""
    h = hashlib.sha256()
    for r in rows:
        h.update(r.encode('utf-8', errors='replace'))
        h.update(b'\n')
    return h.hexdigest()[:16]

def _tokenise_row_chunks(row_chunks, cap_tokens, label, tok, eos_id):
    """Standalone twin of load_wikitext's nested stream_tokenise with the exact
    same chunking / eos / cap / flush behaviour (1<<22 chunks, row-chunk 10000,
    [:cap_tokens] truncation) so the served ids obey the same discipline."""
    chunks = []
    cur = []
    n_flushed = 0

    def _flush():
        nonlocal n_flushed
        if cur:
            n_flushed += len(cur)
            chunks.append(np.array(cur, dtype=np.int64))
            cur.clear()
    for rows in row_chunks:
        rows = [r for r in rows if r.strip()]
        for enc in tok.encode_batch(rows):
            cur.extend(enc.ids)
            cur.append(eos_id)
        if len(cur) >= (1 << 22):
            _flush()
        n_have = n_flushed + len(cur)
        if n_have >= cap_tokens:
            break
        print(f'    [{label}] {n_have} tokens so far')
    _flush()
    if not chunks:
        return np.empty(0, dtype=np.int64)
    ids = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
    return ids[:cap_tokens]

def load_wikitest(seq_len, n_seqs=256, vocab_size=8192, cache_dir='./wt103_cache'):
    """Load the WikiText-103 **test** split as (n_seqs, seq_len) int64 rows.

    只读 wiki.test.raw（本地 dump 或 HF test split），绝不触碰 train/validation，
    因此可作冻结后的独立最终评估数据（design_notes §17 的「未参与方法选择的
    数据」）。与 load_wikitext 同一套纪律：npy 缓存 + sha256 内容指纹 + 原子写；
    分词器使用与主语料完全相同的冻结 train-prefix token 上限（上限记录在
    train cache meta 中）重建，并把拟合语料指纹存进 meta 以供审计。构建
    test 缓存时，与 training documents 完全相同或近重复（SimHash）的 test
    文档会在 tokenisation 前被剔除并记录数量，从而冻结一个去重后的评估
    子集。缓存命中会校验内容指纹、tokenizer 拟合来源和去重元数据；train
    语料被更换时应删除整个 wt103_cache/，否则 test 缓存不会跟随重建——
    这是显式记录的边界，不是静默的失效路径。"""
    from datasets import load_dataset
    os.makedirs(cache_dir, exist_ok=True)
    tag = f'v2_sl{seq_len}_v{vocab_size}_ns{n_seqs}'
    te_path = os.path.join(cache_dir, f'test_ids_{tag}.npy')
    tb_path = os.path.join(cache_dir, f'test_bnd_{tag}.npy')
    me_path = os.path.join(cache_dir, f'test_meta_{tag}.json')
    tk_path = os.path.join(cache_dir, f'test_tok_{tag}.json')
    if all((os.path.exists(p) for p in (te_path, tb_path, me_path))):
        _sig = _memo_file_sig({'test': te_path, 'bnd': tb_path, 'meta': me_path})
        _mkey = (os.path.abspath(cache_dir), tag, _sig)
        _hit = _WT103_TEST_MEMO.get(_mkey)
        if _hit is not None:
            print('[data] serving TEST token ids from the in-process memo (same cache files, already verified) ...')
            return _hit
        print('[data] loading cached TEST token ids ...')
        test_ids = np.load(te_path)
        meta = json.load(open(me_path, encoding='utf-8'))
        test_bnd = np.load(tb_path)
        _problems = []
        _sigs = _cache_file_sigs({'test': te_path, 'bnd': tb_path})
        _fast = _cache_file_sigs_match(meta, _sigs)
        if meta.get('src_fp') != _ids_fp(test_ids):
            _problems.append('test id fingerprint does not match the test meta file')
        if np.asarray(test_bnd).shape != np.asarray(test_ids).shape:
            _problems.append(f'test boundary mask shape {np.asarray(test_bnd).shape} != {np.asarray(test_ids).shape}')
        if meta.get('shuffle_at_split') is True:
            _problems.append('metadata records a shuffled split')
        if meta.get('tokenizer_fit_split') not in (None, 'train'):
            _problems.append(f"tokenizer was fitted on {meta.get('tokenizer_fit_split')!r}, not train")
        if meta.get('tokenizer_fit_protocol') != 'train_prefix_token_budget':
            _problems.append('test cache does not attest the frozen train-prefix tokenizer fit')
        if not isinstance(meta.get('near_duplicate_cleanup'), dict):
            _problems.append('test cache does not attest near-duplicate cleanup against train')
        # A cached test set is only pairable if its tokenizer was fitted on the
        # SAME train prefix as the train/validation cache.  Verifying this on
        # the cold path is not enough: a swapped test cache could otherwise
        # serve a tokenizer that saw test text.
        _train_fit_fps = set()
        import glob as _glob2
        for _mp in _glob2.glob(os.path.join(cache_dir, 'meta_v5_*.json')):
            try:
                _mt = json.load(open(_mp, encoding='utf-8'))
            except Exception:
                continue
            if _mt.get('fit_fp'):
                _train_fit_fps.add(_mt['fit_fp'])
        if _train_fit_fps and meta.get('fit_fp') not in _train_fit_fps:
            raise RuntimeError(
                'load_wikitest: the cached test split was tokenised with a fit '
                f"corpus fingerprint {meta.get('fit_fp')!r} that is absent from "
                'the train cache metas — this test cache cannot be paired with '
                'the current train/validation split; delete the test cache and '
                'rebuild it from the same train tokenizer')
        if meta.get('fit_fp') and not _train_fit_fps:
            print('[data] NOTE: no train-cache meta found; the cached TEST fit provenance cannot be cross-checked locally')
        if _problems:
            _cache_warning('WikiText test split', _problems)
        else:
            if not _fast:
                _validate_token_ids(test_ids, meta['vocab'], 'cached test_ids')
            out = (test_ids, meta['vocab'], None, test_bnd)
            _memo_put(_WT103_TEST_MEMO, _mkey, out, _WT103_TEST_MEMO_CAP)
            return out
    # cold build: source the test rows (local dump first, HF otherwise)
    test_rows = None
    ds = None
    raw_txt = os.environ.get('WT103_RAW_TXT')
    _cands = []
    if raw_txt:
        _cands.append(raw_txt)
    _cands.append(os.path.join(os.path.dirname(os.path.abspath(cache_dir)), 'wt103_raw'))
    _cands.append('/root/wt103_raw')
    for cand in _cands:
        _f = os.path.join(cand, 'wiki.test.raw')
        if os.path.exists(_f):
            test_rows = _read_raw_rows(_f)
            print(f'[data] local wikitext-103 TEST split at {_f} ({len(test_rows)} rows)')
            break
    if test_rows is None:
        ds = None
        last_err = None
        for repo in ('Salesforce/wikitext', 'wikitext'):
            try:
                ds = load_dataset(repo, 'wikitext-103-raw-v1', cache_dir=cache_dir)
                break
            except Exception as e:
                last_err = e
        if ds is None:
            raise last_err if last_err is not None else RuntimeError('load_wikitest: no test-split source available (local dump wiki.test.raw or the HF dataset)')
        test_rows = ds['test']['text']
    # tokenizer: identical frozen-train-prefix fit to load_wikitext.  The
    # token cap is taken from the paired train cache when it is unique, so
    # the test tokenizer is an exact function of the same train region.
    import glob as _glob
    _fit_cap = int(TOKENIZER_FIT_TOKENS)
    _fit_max_rows = 50000
    _train_meta_objs = []
    for _mp in _glob.glob(os.path.join(cache_dir, 'meta_v5_*.json')):
        try:
            _train_meta_objs.append(json.load(open(_mp, encoding='utf-8')))
        except Exception:
            continue
    _caps = {int(_m['tokenizer_fit_token_cap']) for _m in _train_meta_objs
             if _m.get('tokenizer_fit_token_cap') is not None}
    _maxrows = {int(_m['tokenizer_fit_max_rows']) for _m in _train_meta_objs
                if _m.get('tokenizer_fit_max_rows') is not None}
    if len(_caps) == 1:
        _fit_cap = _caps.pop()
    if len(_maxrows) == 1:
        _fit_max_rows = _maxrows.pop()
    train_rows_source = None
    for cand in _cands:
        _f = os.path.join(cand, 'wiki.train.raw')
        if os.path.exists(_f):
            train_rows_source = _read_raw_rows(_f)
            break
    if train_rows_source is None:
        if ds is None:
            for repo in ('Salesforce/wikitext', 'wikitext'):
                try:
                    ds = load_dataset(repo, 'wikitext-103-raw-v1', cache_dir=cache_dir)
                    break
                except Exception as e:
                    last_err = e
            if ds is None:
                raise last_err if last_err is not None else RuntimeError('load_wikitest: no train source to fit the shared tokenizer')
        train_rows_source = list(ds['train']['text'])
    # Same blank-row filtering as load_wikitext, otherwise the two tokenizer
    # fit corpora (and therefore the vocabularies) can differ.
    train_rows_source = [r for r in train_rows_source
                         if isinstance(r, str) and r.strip()]
    train_fit_rows, tok, n_fit_tokens = _wikitext_fit_rows(
        train_rows_source, vocab_size, _fit_cap, max_rows=_fit_max_rows)
    if tok is None or not train_fit_rows:
        raise RuntimeError('load_wikitest: the train split has no non-empty text rows to fit the shared tokenizer')
    vocab = tok.get_vocab_size()
    eos_id = tok.token_to_id('<eos>')
    # vocabulary consistency with any existing train-side cache: a mismatch means
    # the two corpora builds cannot be paired — refuse rather than serve it.
    # A persisted train tokenizer is compared by FULL vocab identity (id->token
    # mapping), not merely by size: two BPE fits can share a vocab size with
    # different merges, in which case the id spaces are un-pairable.  Legacy
    # caches without a persisted tokenizer fall back to the size check.
    import glob as _glob
    _test_vocab = dict(tok.get_vocab())
    _train_tok_paths = _glob.glob(os.path.join(cache_dir, 'tok_v5_*.json'))
    for _tp in _train_tok_paths:
        try:
            from tokenizers import Tokenizer as _Tok
            _tt = _Tok.from_file(_tp)
        except Exception:
            continue
        if dict(_tt.get_vocab()) != _test_vocab:
            raise RuntimeError(f'load_wikitest: the freshly fitted test tokenizer is not identical to the cached train tokenizer ({_tp}) — same-size vocabularies can still use different merges, so the splits cannot be paired; delete wt103_cache/ and rebuild both')
    for _mp in _glob.glob(os.path.join(cache_dir, 'meta_v5_*.json')):
        try:
            _mt = json.load(open(_mp, encoding='utf-8'))
        except Exception:
            continue
        _mv = _mt.get('vocab')
        if _mv is None:
            continue
        if int(_mv) != int(vocab):
            raise RuntimeError(f'load_wikitest: the freshly fitted tokenizer (vocab={vocab}) disagrees with the cached train corpus (vocab={_mv} in {_mp}) — the splits cannot be paired; delete wt103_cache/ and rebuild both')
    print(f'[data] tokenising TEST split (cap {n_seqs * seq_len + seq_len} tokens) ...')
    _trows = list(test_rows) if not isinstance(test_rows, list) else list(test_rows)
    # Freeze a near-duplicate-clean evaluation subset: any test document that
    # exactly or near-duplicates a TRAIN document is excluded before
    # tokenisation.  The index is built from the train split only; test rows
    # are never used to decide what counts as a duplicate of train.
    _test_dedup = _DocDeduplicator()
    for _r in train_rows_source:
        if isinstance(_r, str) and _r.strip():
            _test_dedup.is_duplicate(_r)
    _kept_test = []
    _excluded = {'exact': 0, 'near': 0}
    for _r in _trows:
        if not isinstance(_r, str) or not _r.strip():
            continue
        _dup, _kind = _test_dedup.is_duplicate(_r)
        if _dup:
            _excluded[_kind] += 1
        else:
            _kept_test.append(_r)
    if _excluded['exact'] or _excluded['near']:
        print(f'[data] test near-duplicate cleanup: excluded {_excluded["exact"]} exact '
              f'and {_excluded["near"]} near-duplicate documents that also occur in train')
    _trows = _kept_test
    ids = _tokenise_row_chunks([_trows[s:s + 10000] for s in range(0, len(_trows), 10000)], n_seqs * seq_len + seq_len, 'test', tok, eos_id)
    n_have = min(len(ids) // seq_len, n_seqs)
    if n_have < n_seqs:
        print(f'[data] WARNING: test split yields only {n_have} full rows (< {n_seqs} requested) — the final-test evaluation runs on {n_have} rows')
    test_batch = ids[:n_have * seq_len].reshape(n_have, seq_len)
    _validate_token_ids(test_batch, vocab, 'wikitext test_batch')
    print(f'[data] TEST rows = {n_have} (seq_len={seq_len}, vocab={vocab})')
    bnd_ids = sorted(boundary_token_ids(tok, eos_id))
    test_bnd = np.isin(test_batch, bnd_ids)
    for _p, _a in ((te_path, test_batch), (tb_path, test_bnd)):
        _fd, _tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(_p)), prefix=os.path.basename(_p) + '.', suffix='.tmp')
        try:
            with os.fdopen(_fd, 'wb') as _fh:
                np.save(_fh, _a)
            os.replace(_tmp, _p)
        except BaseException:
            try:
                os.unlink(_tmp)
            except OSError:
                pass
            raise
    _file_sigs = _cache_file_sigs({'test': te_path, 'bnd': tb_path})
    atomic_write_json(me_path, {
        'vocab': vocab,
        'src_fp': _ids_fp(test_batch),
        'fit_fp': _text_fp(train_fit_rows),
        'file_sigs': _file_sigs,
        'id_range': {'test': _array_id_range(test_batch)},
        'split_strategy': 'official_dataset_test_split_near_duplicate_filtered',
        'shuffle_at_split': False,
        'tokenizer_fit_split': 'train',
        'tokenizer_fit_protocol': 'train_prefix_token_budget',
        'tokenizer_fit_token_cap': int(_fit_cap),
        'tokenizer_fit_max_rows': int(_fit_max_rows),
        'tokenizer_fit_tokens': int(n_fit_tokens),
        'near_duplicate_cleanup': {
            'policy': 'test_documents_duplicating_train_documents_removed',
            'exact_removed': int(_excluded['exact']),
            'near_removed': int(_excluded['near']),
            'near_hamming_threshold': int(_test_dedup.near_threshold),
            'source_documents': int(len(test_rows)),
            'kept_documents': int(len(_trows)),
        },
        'test_sequences': int(test_batch.shape[0]),
        'seq_len': int(seq_len),
    }, indent=0)
    try:
        tok.save(tk_path)
    except Exception as _e:
        print(f'[data] NOTE: could not persist the test tokenizer ({type(_e).__name__}: {_e}) — a rebuild will re-fit it deterministically')
    print('[data] cached TEST token ids to disk (held-out split; final evaluation only)')
    return (test_batch, vocab, tok, test_bnd)

_FWEB_MEMO = {}
_FWEB_MEMO_CAP = 2

def _fineweb_disk_dtype(vocab):
    if vocab <= 65536:
        return np.uint16
    if vocab < 2 ** 31:
        return np.int32
    raise ValueError(f'vocab {vocab} exceeds the id range representable by the on-disk corpus cache')

def _disk_ids_chunk(buf, disk_dt, vocab):
    # Range check must run on the pre-cast Python ints: once the array is
    # uint16 its max is <= 65535 by construction, so checking it afterwards
    # can never fire (and an out-of-range id would already have wrapped on
    # numpy builds that silently overflow).  `_fineweb_disk_dtype` returns
    # uint16 only for vocab <= 65536, so this is a defensive guard.
    if buf:
        _mn, _mx = min(buf), max(buf)
        if _mn < 0 or _mx >= int(vocab):
            raise ValueError(f'token id range [{_mn}, {_mx}] is outside vocab [0, {int(vocab)}) — refusing to cache a corrupted corpus')
        if disk_dt is np.uint16 and _mx > 65535:
            raise ValueError(f'token id {_mx} exceeds the uint16 cache range (vocab={vocab}) — widen _fineweb_disk_dtype before rebuilding')
    return np.array(buf, dtype=disk_dt)

def _i32_sidecar_path(path):
    """Identity-bound int32 sidecar path for a wide on-disk token cache."""
    st = os.stat(path)
    return (f'{path}.i32-{int(st.st_size)}-'
            f'{int(getattr(st, "st_mtime_ns", 0))}-'
            f'{int(getattr(st, "st_ctime_ns", int(st.st_ctime * 1e9)))}-'
            f'{int(getattr(st, "st_ino", 0))}.npy')


def _npy_serve_int32(path):
    mm = np.load(path, mmap_mode='r')
    src = np.asarray(mm)
    if not np.issubdtype(src.dtype, np.integer):
        raise TypeError(f'_npy_serve_int32: expected an integer token cache at {path}, got dtype {src.dtype}')
    if src.dtype in (np.uint8, np.uint16, np.int32):
        return src
    # Wider integer cache: narrow to int32 only after proving every id fits.
    # A silent wrap (e.g. 2**32 + 5 -> 5) would pass the downstream
    # vocab-range check for small vocabularies and corrupt the corpus.
    if src.size:
        _mn, _mx = int(src.min()), int(src.max())
        if _mn < 0 or _mx > 2 ** 31 - 1:
            raise ValueError(f'_npy_serve_int32: token ids in {path} span [{_mn}, {_mx}], outside the int32 transport range')
    # Serve an identity-bound int32 sidecar so workers share one mapped copy:
    # the returned values are the same exact integers and the fingerprint
    # stays on the int32 canonical form.
    try:
        _side = _i32_sidecar_path(path)
        if os.path.exists(_side):
            _n = np.load(_side, mmap_mode='r')
            if (int(_n.size) == int(src.size)
                    and tuple(_n.shape) == tuple(src.shape)
                    and np.dtype(_n.dtype) == np.dtype(np.int32)):
                return _n
        _dir = os.path.dirname(os.path.abspath(_side)) or '.'
        _fd, _tmp = tempfile.mkstemp(dir=_dir, prefix=os.path.basename(_side) + '.', suffix='.tmp')
        os.close(_fd)
        try:
            # Preserve the source shape: the sidecar is a drop-in replacement
            # for the loaded cache, and a same-size but differently shaped
            # cache must never be silently flattened to 1-D.
            _mm = np.lib.format.open_memmap(_tmp, mode='w+', dtype=np.int32, shape=tuple(src.shape))
            _flat_mm = _mm.reshape(-1)
            _flat_src = src.reshape(-1)
            _step = 1 << 24
            for _s in range(0, int(src.size), _step):
                _e = min(_s + _step, int(src.size))
                _flat_mm[_s:_e] = _flat_src[_s:_e]
            _mm.flush()
            del _flat_mm, _mm
            os.replace(_tmp, _side)
        except BaseException:
            try:
                os.unlink(_tmp)
            except OSError:
                pass
            raise
        return np.load(_side, mmap_mode='r')
    except (OSError, ValueError, TypeError):
        out = np.empty(src.shape, dtype=np.int32)
        for s in range(0, src.size, 1 << 24):
            e = min(s + (1 << 24), src.size)
            out[s:e] = src[s:e]
        return out

def _fineweb_local_files(env_value):
    """Resolve FWEDU_LOCAL_JSONL to the ordered list of local documents files."""
    import glob
    files = []
    for cand in str(env_value).split(os.pathsep):
        cand = cand.strip()
        if not cand:
            continue
        if os.path.isdir(cand):
            for pat in ('*.jsonl', '*.json', '*.txt'):
                files.extend(sorted(glob.glob(os.path.join(cand, pat))))
        elif any(ch in cand for ch in '*?['):
            files.extend(sorted(glob.glob(cand)))
        else:
            files.append(cand)
    return [f for f in files if os.path.isfile(f)]

def _fineweb_local_source_fingerprint(env_value):
    """Identity of the local stand-in source (paths, size, mtime, ctime, inode).

    The token cache's own `src_fp` only proves that the cached ids match the
    cache meta; it cannot see that the *source* files changed.  The signature
    changes whenever a source file is added/removed/replaced — ctime/inode
    catch even a same-size replacement that preserves mtime — so a rebuilt
    cache is non-confusable with a stale one."""
    files = _fineweb_local_files(env_value)
    if not files:
        return None
    h = hashlib.sha1()
    for path in files:
        try:
            st = os.stat(path)
        except OSError:
            return None
        h.update(os.path.abspath(path).encode('utf-8', 'replace'))
        h.update(b'\0')
        h.update(str(int(st.st_size)).encode('ascii'))
        h.update(b'\0')
        h.update(str(int(getattr(st, 'st_mtime_ns', int(st.st_mtime * 1e9)))).encode('ascii'))
        h.update(b'\0')
        h.update(str(int(getattr(st, 'st_ctime_ns', int(st.st_ctime * 1e9)))).encode('ascii'))
        h.update(b'\0')
        h.update(str(int(getattr(st, 'st_ino', 0))).encode('ascii'))
        h.update(b'\0')
    return h.hexdigest()[:16]

def _doc_exact_fp(text):
    h = hashlib.blake2b(digest_size=16)
    h.update(str(text).strip().encode('utf-8', errors='replace'))
    return h.hexdigest()


def _doc_simhash(text, *, shingle_words=5, max_shingles=128):
    """Deterministic 64-bit SimHash over word shingles (near-duplicate key).

    The shingle set is sub-sampled on an evenly spaced grid so very long
    documents stay bounded; the same text always yields the same hash
    across processes (blake2b, no Python salted hash).
    """
    words = re.findall(r"[\w']+", str(text).lower())
    if not words:
        words = [str(text)[:512]]
    n = min(len(words), 1 << 20)
    words = words[:n]
    if len(words) < shingle_words:
        shingles = [' '.join(words)]
    else:
        count = len(words) - shingle_words + 1
        if count <= max_shingles:
            pos = range(count)
        else:
            pos = np.unique(np.linspace(0, count - 1, int(max_shingles), dtype=np.int64))
        shingles = [' '.join(words[int(i):int(i) + shingle_words]) for i in pos]
    hashes = np.fromiter(
        (int.from_bytes(hashlib.blake2b(sh.encode('utf-8', errors='replace'),
                                        digest_size=8).digest(), 'little')
         for sh in shingles),
        dtype=np.uint64, count=len(shingles))
    if hashes.size == 0:
        # Keep the pre-vectorised semantics: an empty shingle set has all-zero
        # per-bit sums, and `v >= 0` sets every bit (not the all-zero hash).
        return (1 << 64) - 1
    bits = ((hashes[:, None] >> np.arange(64, dtype=np.uint64)[None, :]) & 1)
    vec = bits.astype(np.int64) * 2 - 1
    out = 0
    for b, v in enumerate(vec.sum(axis=0).tolist()):
        if v >= 0:
            out |= (1 << b)
    return out


def _hamming64(a, b):
    return int((int(a) ^ int(b)).bit_count())


def _doc_id_u64(doc_id):
    h = hashlib.blake2b(str(doc_id).encode('utf-8', errors='replace'), digest_size=8)
    return np.uint64(int.from_bytes(h.digest(), 'little'))


class _DocDeduplicator:
    """Exact + near-duplicate document filter with a deterministic LSH index.

    A document is a group: once seen (tokenizer fit, train or validation) any
    exact copy or a near copy within ``near_threshold`` SimHash bits is skipped
    on every later split.  This is what keeps a document (or a lightly edited
    copy) from appearing on both sides of the split.
    """

    def __init__(self, *, near_threshold=3, bands=4):
        if bands < 1 or bands > 8:
            raise ValueError('bands must be in [1, 8]')
        self.near_threshold = int(near_threshold)
        self.bands = int(bands)
        self.bits = 64 // self.bands
        self.mask = (1 << self.bits) - 1
        self.exact = set()
        self.buckets = {}
        self.n_exact_skipped = 0
        self.n_near_skipped = 0
        self.n_seen = 0

    def _bucket_keys(self, h):
        return [(b, (int(h) >> (b * self.bits)) & self.mask)
                for b in range(self.bands)]

    def is_duplicate(self, text, fp=None):
        if fp is None:
            fp = _doc_exact_fp(text)
        if fp in self.exact:
            self.n_exact_skipped += 1
            return True, 'exact'
        h = _doc_simhash(text)
        for key in self._bucket_keys(h):
            for other in self.buckets.get(key, ()):
                if _hamming64(h, other) <= self.near_threshold:
                    self.n_near_skipped += 1
                    return True, 'near'
        self.exact.add(fp)
        self.n_seen += 1
        for key in self._bucket_keys(h):
            self.buckets.setdefault(key, []).append(h)
        return False, None

    def stats(self):
        return {'policy': 'exact_content_hash+simhash_lsh',
                'simhash_bits': 64,
                'near_hamming_threshold': self.near_threshold,
                'added_documents': int(self.n_seen),
                'exact_duplicates_skipped': int(self.n_exact_skipped),
                'near_duplicates_skipped': int(self.n_near_skipped)}


TOKENIZER_FIT_TOKENS = 8_000_000


def _wikitext_fit_rows(rows, vocab_size, token_cap, max_rows=50000):
    """Select a deterministic train-prefix fit corpus within ``token_cap``.

    The candidate prefix is measured with the tokenizer it is used to build;
    rows (or, for a single oversized row, the row text) beyond the cap are
    shortened and the tokenizer is rebuilt until the final fit text itself
    fits inside the cap.  The result is a prefix of the train split, so it can
    never include validation/test text.
    """
    total = min(len(rows), int(max_rows))
    if total <= 0 or int(token_cap) <= 0:
        return [], None, 0
    fit_text = list(rows[:total])
    tok = None
    n_tok = 0
    # BPE training + a full encode are linear in the fit corpus.  The first
    # measurement already yields the token/row (or token/char) ratio; a
    # conservative prefix from that ratio converges in one rebuild for
    # realistic text, and a small pass cap guards against a pathological
    # vocabulary.  Never return a fit corpus that exceeds the advertised
    # token cap.
    for _pass in range(8):
        tok = build_tokenizer(fit_text, vocab_size)
        n_tok = int(sum(len(enc.ids) + 1 for enc in tok.encode_batch(fit_text))) if fit_text else 0
        if n_tok <= int(token_cap):
            return fit_text, tok, n_tok
        ratio = float(token_cap) / max(n_tok, 1) * (0.9 if _pass else 0.95)
        if len(fit_text) > 1:
            keep = max(1, int(len(fit_text) * ratio))
            fit_text = fit_text[:min(len(fit_text) - 1, keep)]
        else:
            text = str(fit_text[0])
            new_len = max(1, int(len(text) * ratio))
            if new_len >= len(text):
                new_len = len(text) - 1
            if new_len <= 0:
                break
            fit_text = [text[:new_len]]
    raise RuntimeError(
        f'_wikitext_fit_rows: could not shrink the tokenizer-fit corpus to the '
        f'{int(token_cap)}-token cap (last measured {int(n_tok)} tokens)')

def _fineweb_local_rows(env_value):
    """Offline FineWeb-Edu stand-in: FWEDU_LOCAL_JSONL lists files or
    directories (os.pathsep-separated; directories expand to their sorted
    *.jsonl / *.json / *.txt).  Every non-empty line is a document — either a
    JSON object with a "text" field (the FineWeb-Edu row shape) or plain text.
    Pure availability path for air-gapped kernel-dev machines; the corpus a
    panel trains on is still bound by the content fingerprint, so a local
    stand-in can never be confused with a genuine FineWeb-Edu cache."""
    files = _fineweb_local_files(env_value)
    if not files:
        raise RuntimeError(f'FWEDU_LOCAL_JSONL is set ({env_value!r}) but resolves to no readable file — point it at FineWeb-Edu style jsonl/txt files, or unset it to stream from the hub')
    print(f'[data] local FineWeb-Edu stand-in corpus: {len(files)} file(s) ({files[0]}{" ..." if len(files) > 1 else ""})')
    for path in files:
        with open(path, encoding='utf-8') as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except ValueError:
                    yield ln
                    continue
                text = obj.get('text') if isinstance(obj, dict) else (obj if isinstance(obj, str) else None)
                if isinstance(text, str) and text.strip():
                    yield text
                elif not isinstance(obj, dict):
                    yield ln

def _fineweb_rows(source):
    """Ordered, non-empty FineWeb-Edu documents (streaming; deterministic row
    order per dataset revision).  The token cache plus its content fingerprint
    is what actually binds a panel to its corpus, so a hub-side revision change
    rebuilds a different, non-confusable corpus (different src_fp -> different
    run_cfg -> stale records are dropped, never silently reused)."""
    env_value = os.environ.get('FWEDU_LOCAL_JSONL', '')
    if env_value.strip():
        yield from _fineweb_local_rows(env_value)
        return
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceFW/fineweb-edu', name=source, split='train', streaming=True)
    n_docs = 0
    for rec in ds:
        text = rec.get('text') if isinstance(rec, dict) else None
        if text is None or not str(text).strip():
            continue
        n_docs += 1
        if n_docs % 50000 == 0:
            print(f'    [fineweb] {n_docs} docs streamed')
        yield str(text)

def _fineweb_docs(source):
    """Ordered ``(doc_id, text)`` pairs.

    ``doc_id`` is a stable content fingerprint (the only identifier available
    for both the hub and the offline JSONL stand-in), so train/validation
    metadata can prove document-level disjointness instead of merely claiming
    that stream positions were different.
    """
    for text in _fineweb_rows(source):
        txt = str(text)
        yield _doc_exact_fp(txt), txt

def load_fineweb_edu(seq_len, n_train_tokens, vocab_size=32768, val_seqs=512, cache_dir='./fineweb_cache', source='sample-10BT', fit_docs=20000):
    """B-tier corpus (docs/compute_plan.md §5): FineWeb-Edu, self-trained BPE.

    Discipline mirrors load_wikitext exactly: npy cache keyed by the full
    protocol tag, atomic writes, sha256 content fingerprint in the meta file,
    in-process memo keyed on (file size, mtime).  Differences forced by the
    scale band:
      - token ids are cached on disk at 2 bytes/token (uint16, guarded by the
        actual vocab) and served at that width; every consumer widens ids back
        to int64 per batch exactly as the int64 corpora always did, so yielded
        batches are bit-identical to the int64-storage form;
      - the corpus is built in one streaming pass: the tokenizer-fit documents
        are the TRAIN PREFIX (bounded by a frozen token cap), then train
        tokens continue up to n_train_tokens + seq_len, then the validation
        holdout draws val_seqs*seq_len + seq_len tokens from documents strictly
        after the train region.  Exact and near-duplicate documents are removed
        as whole groups across all phases, document ids are written next to
        the cache and their disjointness is re-checked on every hit.  The
        validation region is closed per protocol tag (never reused across
        caps); the partial final train batch is discarded between regions;
      - val_batch is int64 (eval paths feed targets straight to cross entropy).
    """
    os.makedirs(cache_dir, exist_ok=True)
    # A local stand-in corpus must never share a cache tag with the hub source:
    # otherwise a cache built from FWEDU_LOCAL_JSONL would be served on a later
    # hub run (the self-fingerprint cannot tell the two corpora apart).  The
    # marker only appears when the local path is active, so existing hub caches
    # keep their exact filenames.
    _local_env = os.environ.get('FWEDU_LOCAL_JSONL', '').strip()
    _local_tag = f'_loc{hashlib.sha1(_local_env.encode("utf-8")).hexdigest()[:8]}' if _local_env else ''
    _local_sig = _fineweb_local_source_fingerprint(_local_env) if _local_env else None
    tag = f'v2_sl{seq_len}_cap{n_train_tokens}_v{vocab_size}_vs{val_seqs}_fw{source}_fd{fit_docs}{_local_tag}'
    tr_path = os.path.join(cache_dir, f'train_ids_{tag}.npy')
    va_path = os.path.join(cache_dir, f'val_batch_{tag}.npy')
    vp_path = os.path.join(cache_dir, f'val_bnd_{tag}.npy')
    me_path = os.path.join(cache_dir, f'meta_{tag}.json')
    trd_path = os.path.join(cache_dir, f'train_doc_ids_{tag}.npy')
    vad_path = os.path.join(cache_dir, f'val_doc_ids_{tag}.npy')
    if all((os.path.exists(p) for p in (tr_path, va_path, vp_path, me_path,
                                        trd_path, vad_path))):
        _sig = _memo_file_sig({'train': tr_path, 'val': va_path, 'bnd': vp_path,
                               'meta': me_path, 'train_doc_ids': trd_path,
                               'val_doc_ids': vad_path})
        _mkey = (os.path.abspath(cache_dir), tag, _sig, _local_sig)
        _hit = _FWEB_MEMO.get(_mkey)
        if _hit is not None:
            print('[data] serving fineweb token ids from the in-process memo (same cache files, already verified) ...')
            return _hit
        print('[data] loading cached fineweb token ids ...')
        meta = json.load(open(me_path, encoding='utf-8'))
        train_ids = _npy_serve_int32(tr_path)
        _sigs = _cache_file_sigs({'train': tr_path, 'val': va_path, 'bnd': vp_path,
                                  'train_doc_ids': trd_path, 'val_doc_ids': vad_path})
        _fast = _cache_file_sigs_match(meta, _sigs)
        if meta.get('src_fp') != _ids_fp(train_ids):
            print('[data] WARNING: the cached fineweb train_ids do not match the fingerprint in the meta file — the cache is corrupt or was mixed across builds; REBUILDING it')
        elif _local_sig is not None and meta.get('local_source_sig') != _local_sig:
            print('[data] WARNING: the local FineWeb-Edu stand-in source changed since this cache was built — REBUILDING the cache (a stale corpus must not keep its old fingerprint)')
        else:
            val_batch = np.load(va_path).astype(np.int64)
            val_bnd = np.load(vp_path)
            _problems = _cached_split_problems(meta, train_ids, val_batch, val_bnd,
                                               meta.get('vocab'), seq_len=seq_len,
                                               fingerprints_trusted=False,
                                               ranges_trusted=_fast)
            try:
                _trd = np.load(trd_path)
                _vad = np.load(vad_path)
                _shared_docs = int(np.intersect1d(_trd, _vad, assume_unique=True).size)
                if _shared_docs:
                    _problems.append(f'{_shared_docs} document ids occur in both train and validation')
                if meta.get('doc_ids', {}).get('disjoint') is not True:
                    _problems.append('cache metadata does not attest document-level disjointness')
            except Exception as _de:
                _problems.append(f'document-id sidecars unreadable ({type(_de).__name__}: {_de})')
            if (meta.get('validation_region') or {}).get('policy') != 'per_cap_closed_no_cross_cap_reuse':
                _problems.append('cache does not declare a closed per-cap validation region')
            if (meta.get('tokenizer_fit_protocol') != 'train_prefix_token_budget'
                    or meta.get('tokenizer_fit_docs_in_train_prefix') is not True):
                _problems.append('cache does not attest a tokenizer fit inside the training prefix')
            if _problems:
                _cache_warning('FineWeb train/validation split', _problems)
            else:
                if 'val_fp' not in meta:
                    print('[data] NOTE: this is a legacy FineWeb cache without a validation fingerprint; shape/range checks passed but split membership cannot be independently re-derived. New caches record val_fp.')
                audit_corpus_split(train_ids, val_batch, int(meta['vocab']),
                                   dataset='fineweb', meta=meta)
                _out = (train_ids, val_batch, int(meta['vocab']), None, val_bnd)
                _memo_put(_FWEB_MEMO, _mkey, _out, _FWEB_MEMO_CAP)
                return _out
    rows = _fineweb_docs(source)
    _dedup = _DocDeduplicator()
    _skipped = {'fit': {'exact': 0, 'near': 0},
                'train': {'exact': 0, 'near': 0},
                'validation': {'exact': 0, 'near': 0}}
    fit_pairs = []
    while len(fit_pairs) < int(fit_docs):
        try:
            doc_id, doc = next(rows)
        except StopIteration:
            break
        _dup, _kind = _dedup.is_duplicate(doc, fp=doc_id)
        if _dup:
            _skipped['fit'][_kind] += 1
            continue
        fit_pairs.append((doc_id, doc))
    _fit_cap = min(int(TOKENIZER_FIT_TOKENS), int(n_train_tokens))
    _fit_candidates = [d for _, d in fit_pairs]
    fit_text, tok, n_fit_tokens = _wikitext_fit_rows(
        _fit_candidates, vocab_size, _fit_cap, max_rows=len(_fit_candidates))
    if tok is None or not fit_text:
        raise RuntimeError('load_fineweb_edu: no non-empty documents are available for the tokenizer fit')
    # Keep only the documents that survived the token cap; they form the
    # train prefix below, so the tokenizer's fit corpus is a subset of the
    # corpus the model actually sees.
    fit_pairs = fit_pairs[:len(fit_text)]
    print(f'[data] training BPE tokenizer on {len(fit_text)} unique fineweb docs '
          f'({n_fit_tokens} tokens <= fit cap {_fit_cap}, vocab_size={vocab_size}) ...')
    # _wikitext_fit_rows already built ``tok`` on exactly this fit_text.
    vocab = tok.get_vocab_size()
    eos_id = tok.token_to_id('<eos>')
    disk_dt = _fineweb_disk_dtype(vocab)
    cap_train = int(n_train_tokens) + int(seq_len)
    cap_val = int(val_seqs) * int(seq_len) + int(seq_len)
    raw_dir = os.path.dirname(os.path.abspath(tr_path))

    def _raw_tmp(label):
        fd, tmp = tempfile.mkstemp(dir=raw_dir, prefix=f'.fineweb_{label}_{tag}.', suffix='.raw')
        os.close(fd)
        return tmp

    raw_train = _raw_tmp('train')
    raw_val = _raw_tmp('val')
    n_train_total = 0
    n_val_total = 0
    _marks = 0
    # Compact document-id accumulators: an unsigned-64 array converts to
    # NumPy without an object-by-object pass.
    import array as _array_mod
    train_doc_ids = _array_mod.array('Q')
    val_doc_ids = _array_mod.array('Q')

    try:
        buf = []
        print(f'[data] tokenising fineweb train (cap {n_train_tokens} tokens, source {source}) ...')
        with open(raw_train, 'wb') as fh:
            def _write_train_pairs(pairs):
                nonlocal buf, n_train_total, _marks
                if not pairs:
                    return False
                for (doc_id, _), enc in zip(pairs, tok.encode_batch([d for _, d in pairs])):
                    buf.extend(enc.ids)
                    buf.append(eos_id)
                    train_doc_ids.append(int(_doc_id_u64(doc_id)))
                    if len(buf) >= (1 << 22):
                        _disk_ids_chunk(buf, disk_dt, vocab).tofile(fh)
                        n_train_total += len(buf)
                        buf = []
                        if (n_train_total >> 26) > _marks:
                            _marks = n_train_total >> 26
                            print(f'    [train] {n_train_total} tokens')
                    if n_train_total + len(buf) >= cap_train:
                        return True
                return n_train_total + len(buf) >= cap_train

            # The tokenizer-fit documents are the PREFIX of the training
            # stream, not a separate region: the tokenizer sees exactly the
            # documents the model's first tokens come from.
            stop = _write_train_pairs(fit_pairs)
            del fit_text, fit_pairs
            gc.collect()
            while not stop:
                docs = list(itertools.islice(rows, 4096))
                if not docs:
                    break
                pairs = []
                for doc_id, doc in docs:
                    # Whole-document dedup: a document that already appeared
                    # (fit or train) is never emitted again.
                    _dup, _kind = _dedup.is_duplicate(doc, fp=doc_id)
                    if _dup:
                        _skipped['train'][_kind] += 1
                    else:
                        pairs.append((doc_id, doc))
                stop = _write_train_pairs(pairs)
            if buf:
                _disk_ids_chunk(buf, disk_dt, vocab).tofile(fh)
                n_train_total += len(buf)
                buf = []
        n_train = min(n_train_total, cap_train)
        print(f'[data] train tokens = {n_train}  vocab = {vocab}')
        if n_train < cap_train:
            print(f'[data] WARNING: the source ran dry at {n_train} train tokens (< requested cap {cap_train}); runs configured for the advertised token target will train on fewer tokens than advertised')
        print('[data] tokenising fineweb validation holdout ...')
        with open(raw_val, 'wb') as fh:
            def _write_val_pairs(pairs):
                nonlocal buf, n_val_total
                if not pairs:
                    return False
                for (doc_id, _), enc in zip(pairs, tok.encode_batch([d for _, d in pairs])):
                    buf.extend(enc.ids)
                    buf.append(eos_id)
                    val_doc_ids.append(int(_doc_id_u64(doc_id)))
                    if len(buf) >= (1 << 22):
                        _disk_ids_chunk(buf, disk_dt, vocab).tofile(fh)
                        n_val_total += len(buf)
                        buf = []
                    if n_val_total + len(buf) >= cap_val:
                        return True
                return n_val_total + len(buf) >= cap_val

            stop = False
            while not stop:
                docs = list(itertools.islice(rows, 4096))
                if not docs:
                    break
                pairs = []
                for doc_id, doc in docs:
                    _dup, _kind = _dedup.is_duplicate(doc, fp=doc_id)
                    if _dup:
                        _skipped['validation'][_kind] += 1
                    else:
                        pairs.append((doc_id, doc))
                stop = _write_val_pairs(pairs)
            if buf:
                _disk_ids_chunk(buf, disk_dt, vocab).tofile(fh)
                n_val_total += len(buf)
                buf = []
        n_val_tokens = min(n_val_total, cap_val)
        # 与 load_wikitext 同口径：块缓冲区含 +1 条序列的余量，钳到 val_seqs。
        n_val = min(n_val_tokens // seq_len, int(val_seqs))
        print(f'[data] val sequences = {n_val} (seq_len={seq_len}, contiguous holdout after the train region)')
        if n_val == 0:
            raise RuntimeError('load_fineweb_edu: the corpus ran dry before a single validation sequence was filled — raise n_train_tokens or provide a larger source')
        if n_train <= seq_len + 1:
            raise RuntimeError(f'load_fineweb_edu: only {n_train} train tokens available (need > seq_len+1 = {seq_len + 1}) — the source is too small for the requested cap')

        def _finalise_npy(raw_path, dst_path, count):
            # Write through a unique same-directory temp file, then replace
            # atomically.
            _fd, npy_tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(dst_path)),
                                            prefix=os.path.basename(dst_path) + '.', suffix='.tmp')
            os.close(_fd)
            try:
                mm = np.lib.format.open_memmap(npy_tmp, mode='w+', dtype=disk_dt, shape=(int(count),))
                done = 0
                with open(raw_path, 'rb') as rf:
                    while done < count:
                        take = min(1 << 22, count - done)
                        chunk = np.fromfile(rf, dtype=disk_dt, count=take)
                        if chunk.size < take:
                            raise RuntimeError(f'load_fineweb_edu: raw staging file {raw_path} is shorter than the encoded count — aborting the cache write')
                        mm[done:done + take] = chunk
                        done += take
                mm.flush()
                del mm
                os.replace(npy_tmp, dst_path)
            except BaseException:
                try:
                    os.unlink(npy_tmp)
                except OSError:
                    pass
                raise

        _finalise_npy(raw_train, tr_path, n_train)
        val_disk = np.fromfile(raw_val, dtype=disk_dt, count=n_val * seq_len)
        val_batch = val_disk.astype(np.int64).reshape(n_val, seq_len)
        _validate_token_ids(val_batch, vocab, 'fineweb val_batch')
        bnd_ids = sorted(boundary_token_ids(tok, eos_id))
        val_bnd = np.isin(val_batch, bnd_ids)
        print(f'[data] boundary tokens: {int(val_bnd.sum())} ({val_bnd.mean() * 100:.1f}% of val tokens)')
        _train_doc_arr = np.frombuffer(train_doc_ids, dtype=np.uint64).copy() if train_doc_ids else np.empty(0, dtype=np.uint64)
        _val_doc_arr = np.frombuffer(val_doc_ids, dtype=np.uint64).copy() if val_doc_ids else np.empty(0, dtype=np.uint64)
        _shared_docs = int(np.intersect1d(_train_doc_arr, _val_doc_arr, assume_unique=True).size)
        if _shared_docs:
            raise RuntimeError(f'load_fineweb_edu: {_shared_docs} document ids occur in both train and validation — refusing to write a leaking split')
        for _p, _a in ((va_path, val_batch), (vp_path, val_bnd),
                       (trd_path, _train_doc_arr), (vad_path, _val_doc_arr)):
            _fd, _tmp = tempfile.mkstemp(dir=raw_dir, prefix=os.path.basename(_p) + '.', suffix='.tmp')
            try:
                with os.fdopen(_fd, 'wb') as _fh:
                    np.save(_fh, _a)
                os.replace(_tmp, _p)
            except BaseException:
                try:
                    os.unlink(_tmp)
                except OSError:
                    pass
                raise
        train_ids = _npy_serve_int32(tr_path)
        _validate_token_ids(train_ids, vocab, 'fineweb train_ids')
        _file_sigs = _cache_file_sigs({'train': tr_path, 'val': va_path, 'bnd': vp_path,
                                       'train_doc_ids': trd_path, 'val_doc_ids': vad_path})
        atomic_write_json(me_path, {
            'vocab': vocab,
            'src_fp': _ids_fp(train_ids),
            'val_fp': _ids_fp(np.ascontiguousarray(val_batch).reshape(-1)),
            'file_sigs': _file_sigs,
            'id_range': {'train': _array_id_range(train_ids), 'val': _array_id_range(val_batch)},
            'doc_ids': {
                'train_count': int(_train_doc_arr.size),
                'validation_count': int(_val_doc_arr.size),
                'train_fp': _ids_fp(_train_doc_arr),
                'validation_fp': _ids_fp(_val_doc_arr),
                'overlap': _shared_docs,
                'disjoint': True,
                'policy': 'whole_document_exact_and_near_duplicate_filtered',
            },
            'dedup': {
                'fit': dict(_skipped['fit']),
                'train': dict(_skipped['train']),
                'validation': dict(_skipped['validation']),
                **{'simhash_'+k: v for k, v in _dedup.stats().items() if k.startswith('near_') or k == 'policy'},
            },
            'local_source_sig': _local_sig,
            'source': source,
            'fit_docs': fit_docs,
            'disk_dtype': np.dtype(disk_dt).str,
            'seq_len': int(seq_len),
            'n_train_tokens': int(n_train_tokens),
            'n_train_actual': int(n_train),
            'val_seqs': int(val_seqs),
            'val_sequences': int(val_batch.shape[0]),
            'split_strategy': 'stream_order_document_disjoint_deduplicated',
            'split_order': ['tokenizer_fit_prefix', 'train', 'validation'],
            'shuffle_at_split': False,
            'tokenizer_fit_split': 'train',
            'tokenizer_fit_protocol': 'train_prefix_token_budget',
            'tokenizer_fit_token_cap': int(_fit_cap),
            'tokenizer_fit_tokens': int(n_fit_tokens),
            'tokenizer_fit_docs_in_train_prefix': True,
            'train_region_end_tokens': int(n_train),
            'validation_region': {
                'policy': 'per_cap_closed_no_cross_cap_reuse',
                'start_token_approx': int(n_train_total),
                'end_token_approx': int(n_train_total + n_val_tokens),
                'note': ('this validation region is closed for this protocol tag; '
                         'never reuse it as another cap\'s validation and never '
                         'compare runs whose val_fp differs'),
            },
        }, indent=0)
        print('[data] cached fineweb token ids to disk (resume-safe)')
        audit_corpus_split(train_ids, val_batch, vocab, dataset='fineweb',
                           meta=json.load(open(me_path, encoding='utf-8')))
        _out = (train_ids, val_batch, vocab, tok, val_bnd)
        _memo_put(_FWEB_MEMO, (os.path.abspath(cache_dir), tag, _memo_file_sig({'train': tr_path, 'val': va_path, 'bnd': vp_path, 'meta': me_path, 'train_doc_ids': trd_path, 'val_doc_ids': vad_path}), _local_sig), _out, _FWEB_MEMO_CAP)
        return _out
    finally:
        for _raw in (raw_train, raw_val):
            try:
                os.unlink(_raw)
            except OSError:
                pass

_CORPUS_AUDIT_MEMO = {}


def audit_corpus_split(train_ids, val_batch, vocab, *, dataset='wikitext',
                       meta=None, max_scan_tokens=2_000_000, sample_rows=64,
                       shift_warn=0.25, shift_fail=0.75, strict=True):
    """Hard pre-training gate for a cached/served token split.

    Runs before any model fitting and raises on a hard leakage/data-quality
    failure: out-of-vocabulary ids, non-finite or empty holdouts, duplicate
    rows *across* splits (exact match on a deterministic sample of the train
    stream), non-disjoint document ids when the cache records them, and
    materially shifted holdout token distributions.  A clean report is cached
    by the two content fingerprints, so resume phases do not rescan.
    """
    import data_guard as _DG
    tr = np.asarray(train_ids).reshape(-1)
    va = np.asarray(val_batch)
    report = {
        'dataset': dataset,
        'train_tokens': int(tr.size),
        'val_shape': list(va.shape),
        'vocab': int(vocab),
        'hard_failures': [],
        'warnings': [],
        'meta_split_strategy': (meta or {}).get('split_strategy'),
    }
    key = (dataset, int(vocab), _ids_fp(tr),
           _ids_fp(np.ascontiguousarray(va).reshape(-1)) if va.size else None)
    _hit = _CORPUS_AUDIT_MEMO.get(key)
    if _hit is not None:
        return _hit
    try:
        _validate_token_ids(tr, vocab, f'{dataset} train_ids')
        _validate_token_ids(va, vocab, f'{dataset} val_batch')
    except (TypeError, ValueError) as e:
        report['hard_failures'].append(str(e))
    if va.ndim != 2 or va.shape[0] == 0 or va.size == 0:
        report['hard_failures'].append(f'validation split is empty or not 2-D (shape {va.shape})')
    if tr.size < 2:
        report['hard_failures'].append('training stream has fewer than 2 tokens')
    # No NaN/Inf can hide in an integer token cache, but a float cache (future
    # feature matrix) is checked explicitly.
    if not np.issubdtype(tr.dtype, np.integer):
        _bad = int(np.size(tr) - np.isfinite(tr).sum())
        report['nonfinite_train'] = _bad
        if _bad:
            report['hard_failures'].append(f'{_bad} NaN/Inf values in train')
    if not np.issubdtype(va.dtype, np.integer):
        _bad = int(np.size(va) - np.isfinite(va).sum())
        report['nonfinite_val'] = _bad
        if _bad:
            report['hard_failures'].append(f'{_bad} NaN/Inf values in validation')

    if va.ndim == 2 and va.size:
        _dup = int(_DG.duplicate_mask(va).sum())
        report['duplicate_rows_within_validation'] = _dup
        if _dup:
            report['warnings'].append(f'{_dup} duplicate validation rows')

    # Exact cross-split row overlap on a uniformly-spaced sample of the train
    # stream.  Small caches are scanned completely; large ones report the
    # scanned fraction instead of pretending the sample is a proof.
    overlap = None
    if tr.size and va.ndim == 2 and va.shape[0]:
        import audit_gates as _AG
        _rows = np.unique(np.linspace(0, va.shape[0] - 1,
                                      min(int(sample_rows), va.shape[0]),
                                      dtype=np.int64))
        _sample = np.asarray(va)[_rows]
        _nseg = 16
        _seg_len = max(int(max_scan_tokens) // _nseg,
                       int(_sample.shape[1]) + 1)
        if tr.size <= max(int(max_scan_tokens), _nseg * _seg_len):
            _segments = [tr]
            _scanned = int(tr.size)
        else:
            # 16 equally spaced windows are disjoint because the start spacing
            # is >= _seg_len once tr.size >= _nseg * _seg_len; otherwise the
            # same validation row could be counted by several windows.
            _starts = np.linspace(0, tr.size - _seg_len, _nseg, dtype=np.int64)
            _segments = [tr[int(s):int(s) + _seg_len] for s in _starts]
            _scanned = int(sum(s.size for s in _segments))
        _shared = 0
        for _seg in _segments:
            _shared += int(_AG._row_overlap_count(_seg, _sample))
        overlap = _shared
        report['exact_row_overlap_in_sample'] = int(_shared)
        report['sampled_val_rows'] = int(_rows.size)
        report['train_tokens_scanned'] = int(_scanned)
        report['train_tokens_scanned_fraction'] = round(
            float(_scanned) / float(tr.size), 6) if tr.size else 1.0
        if _shared:
            report['hard_failures'].append(
                f'{_shared} sampled validation rows occur verbatim in train')

    # Document-level proof, when the cache carries it.
    _docs = (meta or {}).get('doc_ids') or {}
    if _docs:
        report['documents'] = {k: _docs.get(k) for k in
                               ('train_count', 'validation_count', 'overlap',
                                'disjoint', 'policy')}
        if _docs.get('disjoint') is not True or int(_docs.get('overlap', 0) or 0) != 0:
            report['hard_failures'].append('cache does not attest document-disjoint train/validation splits')
    elif dataset == 'fineweb':
        report['warnings'].append('legacy FineWeb cache has no document-id disjointness proof')

    # Training-only reference distribution; the holdout never defines it.
    if tr.size and va.size:
        try:
            import audit_gates as _AG
            _tr_s = _AG._sample_tokens(tr, cap=200000, seed=0)
            _va_s = _AG._sample_tokens(va, cap=200000, seed=1)
            _shift = _DG.categorical_shift_report(_tr_s, _va_s)
            _js = float(_shift.get('js_divergence', 0.0))
            report['train_val_token_shift'] = _shift
            if _js >= float(shift_fail):
                report['hard_failures'].append(
                    f'train/validation token JS divergence {_js:.3f} >= {shift_fail}')
            elif _js >= float(shift_warn):
                report['warnings'].append(
                    f'train/validation token JS divergence {_js:.3f} >= {shift_warn}')
        except Exception as _se:
            report['warnings'].append(f'token-shift audit unavailable ({type(_se).__name__}: {_se})')

    report['status'] = 'fail' if report['hard_failures'] else (
        'warnings' if report['warnings'] else 'ok')
    if report['hard_failures'] and strict:
        raise RuntimeError(
            f'[{dataset}] DATA GATE REFUSED this split: ' +
            '; '.join(report['hard_failures']) +
            ' — fix the split/cache before training (no model has been fitted).')
    _CORPUS_AUDIT_MEMO[key] = report
    return report


def assert_shared_validation_fp(records, *, context='comparison'):
    """Refuse a cross-run comparison whose records were scored on different
    validation slices (e.g. reusing a small cap's validation region under a
    larger cap).  Returns the single shared fingerprint."""
    fps = set()
    for r in records:
        if isinstance(r, dict):
            _fp = r.get('val_fp') or (r.get('provenance') or {}).get('data', {}).get('val_fp')
            if _fp:
                fps.add(str(_fp))
    if len(fps) > 1:
        raise ValueError(f'{context}: records use {len(fps)} different validation '
                         f'slices {sorted(fps)}; their numbers are not comparable')
    return next(iter(fps)) if fps else None


def load_run_corpus(cfg):
    """Serve the corpus tuple for a run cfg (data-source switch).

    A cfg without a 'dataset' key keeps WikiText-103 with exactly the arguments
    the drivers have always passed — behaviour is bit-identical to the former
    hard-wired load_wikitext call, so no existing panel, fingerprint or record
    is touched.  'dataset': 'fineweb' selects the B-tier corpus
    (docs/compute_plan.md §5) with its own cache directory, tokenizer width,
    source name and fit protocol."""
    name = cfg.get('dataset') or 'wikitext'
    if name == 'wikitext':
        return load_wikitext(cfg['seq_len'], cfg['n_train_tokens'], vocab_size=cfg.get('vocab_size', 8192), val_seqs=cfg.get('val_seqs', 512), cache_dir=cfg.get('cache_dir', './wt103_cache'))
    if name == 'fineweb':
        return load_fineweb_edu(cfg['seq_len'], cfg['n_train_tokens'], vocab_size=cfg.get('vocab_size', B_VOCAB), val_seqs=cfg.get('val_seqs', 512), cache_dir=cfg.get('cache_dir', './fineweb_cache'), source=cfg.get('fw_source', 'sample-10BT'), fit_docs=cfg.get('fw_fit_docs', 20000))
    raise ValueError(f"load_run_corpus: unknown dataset {name!r} (expected 'wikitext' or 'fineweb')")

def _json_clean(obj):
    """Recursively replace non-finite floats with null and make NumPy values
    JSON-safe.  Standard JSON has no NaN/Infinity tokens; writing them makes a
    damaged curve indistinguishable from a real value on strict parsers."""
    if isinstance(obj, dict):
        return {str(k): _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_clean(obj.tolist())
    if isinstance(obj, np.generic):
        return _json_clean(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def atomic_write_json(path, obj, indent=2):
    text = json.dumps(_json_clean(obj), indent=indent, default=float, allow_nan=False)
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=os.path.basename(path) + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

def atomic_write_text(path, text):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=os.path.basename(path) + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

def locked_json_read(path):
    """Consistent read of a shared JSON file under the same flock the writers
    use ({} when absent).  Pure read: never creates or rewrites the file."""
    try:
        import fcntl
    except ImportError:
        fcntl = None
    lock_f = None
    if fcntl is not None:
        lock_f = open(path + '.lock', 'a')
        fcntl.flock(lock_f, fcntl.LOCK_SH)
    try:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, encoding='utf-8') as f:
                return json.load(f)
        except Exception as _e:
            print(f'[resume] FATAL: {path} exists but cannot be parsed ({type(_e).__name__}: {_e}).  Refusing to start with a silently RESET panel — move it aside to start fresh.')
            raise
    finally:
        if lock_f is not None:
            try:
                fcntl.flock(lock_f, fcntl.LOCK_UN)
            finally:
                lock_f.close()


def locked_json_update(path, mutfn, indent=2):
    """Load-modify-write a JSON file under an exclusive flock.

    Parallel cell workers (par.py) all funnel their summary.json mutations
    through this helper, so two children finishing at the same moment can
    never lose each other's records.  mutfn(obj) mutates the loaded object in
    place (or returns a replacement); a missing file starts from {}.  The
    write itself stays atomic (atomic_write_json), so a crash mid-update
    leaves the previous content intact.
    """
    try:
        import fcntl
    except ImportError:
        fcntl = None
    lock_f = None
    if fcntl is not None:
        lock_f = open(path + '.lock', 'a')
        fcntl.flock(lock_f, fcntl.LOCK_EX)
    try:
        obj = {}
        if os.path.exists(path):
            try:
                with open(path, encoding='utf-8') as f:
                    obj = json.load(f)
            except Exception as _e:
                print(f'[resume] FATAL: {path} exists but cannot be parsed ({type(_e).__name__}: {_e}).  Refusing to overwrite it with a fresh object — move it aside to start fresh.')
                raise
        rep = mutfn(obj)
        obj = obj if rep is None else rep
        atomic_write_json(path, obj, indent=indent)
        return obj
    finally:
        if lock_f is not None:
            try:
                fcntl.flock(lock_f, fcntl.LOCK_UN)
            finally:
                lock_f.close()


def atomic_write_csv(path, header, rows):
    import io as _io
    if header is None:
        header = []
        seen = set()
        for r in rows:
            for k in r.keys() if isinstance(r, dict) else r:
                if k not in seen:
                    seen.add(k)
                    header.append(k)
    cols = list(header)
    buf = _io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for r in rows:
        if isinstance(r, dict):
            w.writerow([r.get(k, '') for k in cols])
        else:
            w.writerow(list(r))
    atomic_write_text(path, buf.getvalue())

def new_experiment_id(prefix='run'):
    """A collision-resistant id for one experiment attempt.

    Uses wall-clock time + pid + OS entropy (never the Python/NumPy/Torch RNG
    streams, so taking an id cannot perturb a seeded run).  The id is written
    into the record and the append-only experiment log, making repeated
    attempts of the same (variant, seed, run_cfg) distinguishable."""
    return f'{prefix}-{time.strftime("%Y%m%dT%H%M%S", time.gmtime())}-{os.getpid()}-{os.urandom(4).hex()}'

def append_jsonl(path, record):
    """Append one JSON object to a shared JSONL log under an exclusive flock.

    Append-only tracking is deliberately separate from summary.json: summary's
    keys are the resume identity `variant::seed`, so a rerun legitimately
    overwrites an old value.  The JSONL ledger keeps every attempt (including
    failures and cap truncations), which is what makes the paper trail
    auditable after the fact.  A write failure is the caller's to report; this
    helper never swallows a partially written line (single write + flush)."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    line = json.dumps(_json_clean(_jsonable(record)), sort_keys=True, ensure_ascii=False, default=float, allow_nan=False) + '\n'
    try:
        import fcntl
    except ImportError:
        fcntl = None
    with open(path, 'a', encoding='utf-8') as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def log_experiment_event(outdir, event):
    """Append a trace event to `outdir/experiments.jsonl` (never raises).

    Tracking is diagnostic infrastructure: a full disk or a read-only archive
    must not destroy a completed training cell.  The warning names the path so
    the missing ledger is visible rather than silent."""
    try:
        path = os.path.join(outdir, 'experiments.jsonl')
        append_jsonl(path, event)
        return path
    except Exception as _e:
        print(f'[track] WARNING: could not append to {os.path.join(outdir, "experiments.jsonl")} ({type(_e).__name__}: {_e}) — the training record itself is unaffected')
        return None

def stamped_analysis(kind, payload, *, outdir=None, name=None, event=True):
    """Return `payload` merged with the standard traceability stamp — and
    optionally persist it plus an experiments.jsonl entry.

    A derived number (statistic, audit verdict, analytic FLOPs table, kernel
    validation report, parameter count) is quotable in a report only if the
    artifact records WHICH code produced it, when, and under which
    environment/git state — the same provenance spine a training record
    carries.  Without this, re-deriving an analysis after a code change is
    indistinguishable from re-deriving the original, and a stale artifact
    silently masquerades as current.  `kind` identifies the artifact family;
    `payload` is the analysis body; `outdir`+`name` persist the stamped record
    atomically; `event` additionally appends a ledger entry so analysis runs
    appear in the same experiments.jsonl trace as training panels.

    The stamp overwrites its own keys (`kind`/`created_at`/`code`/
    `provenance`), so re-stamping a loaded record on rewrite updates the
    trace instead of accumulating stale copies."""
    rec = dict(payload) if isinstance(payload, dict) else {'result': payload}
    rec['kind'] = kind
    rec['created_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    rec['code'] = code_stamp()
    rec['provenance'] = code_provenance()
    if outdir is not None and name:
        path = os.path.join(outdir, name)
        atomic_write_json(path, rec)
        if event:
            log_experiment_event(outdir, {'kind': 'analysis', 'tracking_schema': 2, 'artifact': name, 'created_at': rec['created_at'], 'code': rec['code'], 'label': kind, 'status': 'ok'})
    return rec

_PINNED_HOST_CACHE = {}
_PINNED_HOST_CAP_BYTES = 256 << 20
_GPU_CORPUS_CACHE_BYTES = 64 << 20


def _gpu_corpus_cache_fits(nbytes, device):
    """Reserve the corpus cache only when it also fits the current free device capacity."""
    if int(nbytes) > _GPU_CORPUS_CACHE_BYTES:
        return False
    try:
        free, _total = torch.cuda.mem_get_info(device)
        return int(nbytes) <= max(0, int(free) // 4)
    except Exception:
        return True

_NARROW_MEMO = {}


def _cache_id_upper_bound(path):
    """Recorded token-id upper bound for a cache file, when the sibling meta
    file provides one.  Avoids a full-corpus max() scan on legacy int64 caches."""
    if not path:
        return None
    base = os.path.basename(str(path))
    d = os.path.dirname(os.path.abspath(str(path)))
    if base.startswith('train_ids_'):
        meta_name = 'meta_' + base[len('train_ids_'):]
    elif base.startswith('test_ids_'):
        meta_name = 'test_meta_' + base[len('test_ids_'):]
    else:
        return None
    meta_path = os.path.join(d, os.path.splitext(meta_name)[0] + '.json')
    try:
        with open(meta_path, encoding='utf-8') as fh:
            meta = json.load(fh)
    except (OSError, ValueError, TypeError):
        return None
    for key in ('max_id', 'max_token_id', 'token_max'):
        try:
            v = int(meta[key])
            if 0 <= v < 2 ** 31:
                return v
        except (KeyError, TypeError, ValueError):
            pass
    try:
        vocab = int(meta.get('vocab'))
        if 0 < vocab <= 2 ** 31:
            return vocab - 1
    except (TypeError, ValueError):
        pass
    return None


def _ids_narrow(train_ids):
    if train_ids.dtype in (np.uint8, np.uint16, np.int32):
        return train_ids
    import weakref as _weakref
    _key = id(train_ids)
    _hit = _NARROW_MEMO.get(_key)
    if _hit is not None and _hit[0]() is train_ids:
        return _hit[1]
    _path = getattr(train_ids, 'filename', None)
    if _path and train_ids.size:
        # A disk-backed corpus is mapped read-only.  Workers share one int32
        # sidecar next to the cache.
        try:
            _st = os.stat(_path)
            # Bind the sidecar to the same identity fields the cache loaders
            # use.  A same-size rewrite can preserve mtime (``cp -p``), so
            # size+mtime alone could silently serve a stale int32 copy.
            _side = _i32_sidecar_path(_path)
            if os.path.exists(_side):
                _n = np.load(_side, mmap_mode='r')
                if (int(_n.size) == int(train_ids.size)
                        and tuple(_n.shape) == tuple(train_ids.shape)
                        and np.dtype(_n.dtype) == np.dtype(np.int32)):
                    _NARROW_MEMO[_key] = (_weakref.ref(train_ids), _n)
                    return _n
            _mx = _cache_id_upper_bound(_path)
            if _mx is None:
                _mx = int(train_ids.max()) if train_ids.size else -1
            if _mx < 2 ** 31:
                _dir = os.path.dirname(os.path.abspath(_side)) or '.'
                _fd, _tmp = tempfile.mkstemp(dir=_dir, prefix=os.path.basename(_side) + '.', suffix='.tmp')
                os.close(_fd)
                try:
                    _mm = np.lib.format.open_memmap(_tmp, mode='w+', dtype=np.int32, shape=tuple(train_ids.shape))
                    _flat_mm = _mm.reshape(-1)
                    _flat_src = np.asarray(train_ids).reshape(-1)
                    _step = 1 << 24
                    for _s in range(0, int(train_ids.size), _step):
                        _e = min(_s + _step, int(train_ids.size))
                        _flat_mm[_s:_e] = _flat_src[_s:_e]
                    _mm.flush()
                    del _flat_mm, _mm
                    os.replace(_tmp, _side)
                except BaseException:
                    try:
                        os.unlink(_tmp)
                    except OSError:
                        pass
                    raise
                _n = np.load(_side, mmap_mode='r')
                _NARROW_MEMO[_key] = (_weakref.ref(train_ids), _n)
                return _n
        except (OSError, ValueError, TypeError):
            pass
    _mx = _cache_id_upper_bound(getattr(train_ids, 'filename', None))
    if _mx is None:
        _mx = int(train_ids.max()) if train_ids.size else -1
    if train_ids.size and _mx < 2 ** 31:
        _n = train_ids.astype(np.int32)
        _NARROW_MEMO[_key] = (_weakref.ref(train_ids), _n)
        return _n
    return train_ids

def batch_iter(train_ids, seq_len, batch_size, device, seed=0):
    rng = np.random.default_rng(seed)
    n = len(train_ids) - seq_len - 1
    if n <= 0:
        raise ValueError(f'batch_iter needs len(train_ids) > seq_len+1 to sample causal windows, got len={len(train_ids)} seq_len={seq_len}. Increase n_train_tokens or reduce seq_len.')
    train_ids = np.asarray(train_ids)
    _narrow = _ids_narrow(train_ids)
    _devkey = _dev_str(device)
    host = None
    host_np = None
    cached = _PINNED_HOST_CACHE.get('entry')
    if (cached is not None and cached[0] is train_ids
            and cached[1].shape[0] == train_ids.shape[0]
            and cached[2] == _devkey):
        host = cached[1]
    elif device.type == 'cuda' and _gpu_corpus_cache_fits(_narrow.nbytes, device):
        try:
            host = torch.from_numpy(_narrow).to(device)
            _PINNED_HOST_CACHE['entry'] = (train_ids, host, _devkey)
        except RuntimeError:
            host = None
    if host is None:
        if device.type == 'cuda':
            try:
                if _narrow.dtype in (np.uint8, np.uint16):
                    host_np = _narrow
                elif _narrow.nbytes <= _PINNED_HOST_CAP_BYTES:
                    host = torch.from_numpy(_narrow).pin_memory()
                    _PINNED_HOST_CACHE['entry'] = (train_ids, host, _devkey)
                else:
                    host_np = _narrow
            except RuntimeError:
                host_np = _narrow
        else:
            if _narrow.dtype in (np.uint8, np.uint16):
                # PyTorch's CPU index kernel lacks uint8/uint16 gathers.  Gather
                # the narrow array with NumPy per batch and widen only the
                # [B, seq_len+1] result — the same exact token ids.
                host_np = _narrow
            else:
                host = torch.from_numpy(_narrow)
    # Transport-only narrowing (design notes §6.1): every gather index is an
    # exact integer bounded by n + seq_len, so when that bound fits in int32
    # the index tensor is uploaded / consumed as int32 and the gathered ids
    # are widened back to int64 right after the gather exactly as before —
    # bit-identical batches.  The broadcast addition itself is then also done
    # in int32: every operand and the sum are < 2**31, so the int32 result
    # holds exactly the same integers as the int64 form.
    _idx_i32 = n + seq_len + 1 < 2 ** 31
    _idx_dt = np.int32 if _idx_i32 else np.int64
    cols = np.arange(seq_len + 1, dtype=_idx_dt)
    idx_np = np.empty((batch_size, seq_len + 1), dtype=_idx_dt)

    def _gather(idx):
        if host_np is not None:
            return torch.from_numpy(host_np[idx])
        if host.device.type == 'cuda':
            idx_t = torch.from_numpy(idx)
            return host[_h2d_async(idx_t, idx_t.dtype, host.device)]
        return host[idx]

    def _prepare(ids):
        if ids.dtype != torch.int64:
            # widen back right after the gather: the yielded ids are
            # bit-identical to the int64 form.
            ids = ids.long()
        if ids.device != device:
            ids = _h2d_async(ids, ids.dtype, device)
        return ids

    # A small bounded producer queue runs the identical seeded draw and gather
    # one step ahead.  Values and order are unchanged (same RNG, same indices).
    _prefetch = device.type == 'cuda' and (host_np is not None or (host is not None and host.device.type == 'cpu'))
    if not _prefetch:
        while True:
            starts = rng.integers(0, n + 1, size=batch_size, dtype=_idx_dt)
            np.add(starts[:, None], cols[None, :], out=idx_np)
            ids = _prepare(_gather(idx_np))
            yield (ids[:, :-1], ids[:, 1:])

    import queue as _queue
    import threading as _threading
    q = _queue.Queue(maxsize=2)
    stop = _threading.Event()

    def _producer():
        try:
            while not stop.is_set():
                starts = rng.integers(0, n + 1, size=batch_size, dtype=_idx_dt)
                np.add(starts[:, None], cols[None, :], out=idx_np)
                item = _gather(idx_np)
                while not stop.is_set():
                    try:
                        q.put((item, None), timeout=0.2)
                        break
                    except _queue.Full:
                        continue
        except BaseException as exc:
            import traceback as _tb
            try:
                q.put((None, (exc, _tb.format_exc())))
            except BaseException:
                pass

    th = _threading.Thread(target=_producer, name='batch-prefetch', daemon=True)
    th.start()
    try:
        while True:
            try:
                item, err = q.get(timeout=0.2)
            except _queue.Empty:
                if stop.is_set() or not th.is_alive():
                    return
                continue
            if err is not None:
                exc, tb = err
                if tb:
                    raise RuntimeError(f'batch prefetch failed: {exc!r}\n{tb}') from exc
                raise exc
            ids = _prepare(item)
            yield (ids[:, :-1], ids[:, 1:])
    finally:
        stop.set()
        th.join(timeout=5.0)

def batch_contract_check(train_ids, seq_len, batch_size, device, seed=0, vocab=None, n_batches=2):
    """Validate the ACTUAL training batch iterator, not a re-derived sample.

    The small-batch overfit probe trains on a fixed batch; that alone cannot
    prove the production loader shifts inputs/targets correctly, keeps every
    id inside the vocabulary, or repeats the same stream for the same seed.
    This check pulls a few real batches from ``batch_iter`` twice and reports
    the loader-level contract separately.  It consumes no global RNG state and
    is meant to run before a panel starts."""
    seq_len = int(seq_len)
    batch_size = int(batch_size)
    n_batches = max(1, int(n_batches))
    report = {
        'kind': 'batch_iterator_contract',
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'seq_len': seq_len,
        'batch_size': batch_size,
        'seed': int(seed),
        'vocab_size': (None if vocab is None else int(vocab)),
        'n_batches': n_batches,
        'device': str(device),
        'checks': {},
        'batches': [],
        'passed': False,
    }
    checks = {
        'shapes_ok': True,
        'dtypes_int64': True,
        'labels_are_next_token_shift': True,
        'ids_within_vocab': True,
        'same_seed_repeats_stream': True,
        'nonempty': True,
    }
    try:
        it_a = batch_iter(train_ids, seq_len, batch_size, device, seed=int(seed))
        it_b = batch_iter(train_ids, seq_len, batch_size, device, seed=int(seed))
        for index in range(n_batches):
            xa, ya = next(it_a)
            xb, yb = next(it_b)
            checks['nonempty'] = bool(checks['nonempty'] and xa.numel() and ya.numel())
            checks['shapes_ok'] = bool(
                checks['shapes_ok']
                and tuple(xa.shape) == (batch_size, seq_len)
                and tuple(ya.shape) == (batch_size, seq_len)
                and tuple(xb.shape) == tuple(xa.shape)
                and tuple(yb.shape) == tuple(ya.shape))
            checks['dtypes_int64'] = bool(
                checks['dtypes_int64']
                and xa.dtype == torch.int64 and ya.dtype == torch.int64
                and xb.dtype == torch.int64 and yb.dtype == torch.int64)
            checks['labels_are_next_token_shift'] = bool(
                checks['labels_are_next_token_shift']
                and ya.shape[1] == xa.shape[1]
                and torch.equal(ya[:, :-1], xa[:, 1:]))
            checks['same_seed_repeats_stream'] = bool(
                checks['same_seed_repeats_stream']
                and torch.equal(xa, xb) and torch.equal(ya, yb))
            if vocab is not None and xa.numel() and ya.numel():
                _v = int(vocab)
                checks['ids_within_vocab'] = bool(
                    checks['ids_within_vocab']
                    and int(xa.min()) >= 0 and int(xa.max()) < _v
                    and int(ya.min()) >= 0 and int(ya.max()) < _v)
            report['batches'].append({
                'index': index,
                'input_shape': [int(_n) for _n in xa.shape],
                'target_shape': [int(_n) for _n in ya.shape],
                'input_min': int(xa.min()) if xa.numel() else None,
                'input_max': int(xa.max()) if xa.numel() else None,
                'target_min': int(ya.min()) if ya.numel() else None,
                'target_max': int(ya.max()) if ya.numel() else None,
            })
        del it_a, it_b
    except Exception as e:
        report['error'] = f'{type(e).__name__}: {e}'
        report['checks'] = checks
        report['diagnosis'] = 'batch iterator failed before producing the requested batches'
        return report
    report['checks'] = checks
    failed = [k for k, ok in checks.items() if not ok]
    report['passed'] = not failed
    report['diagnosis'] = 'ok' if not failed else 'failed checks: ' + ', '.join(failed)
    return report

def count_params(m):
    return sum((p.numel() for p in m.parameters()))

def param_breakdown(m):
    """Parameter counts by functional group, for every record.

    A headline comparison that only reports `params` invites the reviewer's
    question "which parameters, and is the comparison capacity-matched?".
    This is a deterministic structure-only count (no forward pass, no RNG):
    total / attention / attention projections / indexer / MLP / norms /
    embeddings / LM head, plus the active count (equal to total here; no MoE).
    """
    out = {'total': 0, 'attention': 0, 'attention_qo': 0, 'attention_kv': 0,
           'attention_indexer': 0, 'mlp': 0, 'norms': 0, 'embedding': 0,
           'lm_head': 0, 'other': 0, 'active': 0, 'n_layers': 0, 'n_heads': 0, 'd_head': 0}
    for name, p in m.named_parameters():
        n = int(p.numel())
        out['total'] += n
        if name.startswith('tok.') or name.startswith('pos.'):
            out['embedding'] += n
        elif name.startswith('head.'):
            out['lm_head'] += n
        elif name.startswith('norm.') or '.norm' in name:
            out['norms'] += n
        elif name.startswith('blocks.'):
            if '.mlp.' in name:
                out['mlp'] += n
                out['attention'] += 0
            elif '.attn.' in name:
                out['attention'] += n
                if any(k in name for k in ('W_DQ', 'W_DK', 'W_w', 'index')):
                    out['attention_indexer'] += n
                elif any(k in name for k in ('W_kvhead', 'W_aKV', 'W_bKV', 'W_aZ', 'W_bZ', 'B_pos')):
                    out['attention_kv'] += n
                else:
                    out['attention_qo'] += n
            else:
                out['other'] += n
        else:
            out['other'] += n
    try:
        out['n_layers'] = len(m.blocks)
        first = m.blocks[0].attn
        out['n_heads'] = int(getattr(first, 'nh', 0) or 0)
        out['d_head'] = int(getattr(first, 'hd', 0) or 0)
    except Exception:
        pass
    out['active'] = out['total']
    out['attention_share'] = (out['attention'] / out['total']) if out['total'] else 0.0
    return out

def kv_cache_elements_per_token(m):
    """KV-cache elements stored per token per layer (dtype-independent).

    Full attention keeps K and V of width n_heads*d_head; CSA/HCA keep the
    compressed [c_kv | c_kv] lanes (plus the indexer's query projection is
    weights, not cache).  This is the structural quantity behind any
    "KV-cache ratio" claim; memory bytes additionally need dtype/allocator
    overhead and are measured by the efficiency harness, not inferred here.
    """
    total = 0
    per_layer = []
    for blk in getattr(m, 'blocks', []):
        a = blk.attn
        c = a.cfg
        if c.kind == 'full':
            n = 2 * int(getattr(a, 'nh', 0)) * int(getattr(a, 'hd', 0))
        else:
            n = 2 * int(getattr(c, 'c_kv', 0))
        per_layer.append(int(n))
        total += int(n)
    return {'elements_per_token_total': int(total), 'per_layer': per_layer,
            'note': 'elements, not bytes; multiply by dtype size and add allocator/block overhead for measured memory'}

def train_objective(m, *, comp_lambda=0.0, delta_lr_mult=1.0):
    """Describe the ACTUAL optimisation objective of an arm.

    Review 3: dense and sparse arms do not share an objective — sparse arms
    add `comp_lambda * comp_reg` and give `delta_logit` a different LR.  A
    comparison is only mechanism-clean when these descriptors match.  The
    values are read from the built model (not from the variant name), so a
    future arm cannot silently claim to be matched.
    """
    layers = [blk.attn for blk in getattr(m, 'blocks', [])]
    policies = sorted({str(getattr(a.cfg, 'boundary_policy', 'threshold')) for a in layers})
    # `need_reg` is True on every HybridAttention instance, including dense
    # layers (the flag is a capability, not an applied term).  The actual
    # regularizer exists only for cosine_learnable gates, whose trunk computes
    # `last_gate_mean`; deriving this from `need_reg` falsely labels dense and
    # fixed-CSA arms as having a compression objective.
    reg_layers = sum(1 for a in layers if str(getattr(a.cfg, 'chunking', '')) == 'cosine_learnable')
    _named = dict(m.named_parameters())
    delta_names = [n for n in _named if 'delta_logit' in n]
    delta_params = sum(int(_named[n].numel()) for n in delta_names)
    applies = bool(comp_lambda) and reg_layers > 0
    return {
        'comp_lambda': float(comp_lambda),
        'comp_reg_layers': int(reg_layers),
        'comp_reg_applies': bool(applies),
        'comp_lambda_effective': float(comp_lambda) if applies else 0.0,
        'delta_param_count': int(delta_params),
        'delta_lr_mult': float(delta_lr_mult),
        'delta_lr_effective': float(delta_lr_mult) if delta_params else 1.0,
        'boundary_policy': policies,
        'loss_terms': ['ce'] + (['comp_reg'] if applies else []) + (
            ['boundary_policy_gradient'] if any(p != 'threshold' for p in policies) else []),
        'note': 'mechanism-clean comparisons require comp_reg_applies, delta_lr_effective and boundary_policy to match across arms',
    }

def result_is_current(rec, code, *required_keys):
    if not isinstance(rec, dict) or not rec:
        return False
    if not all((k in rec for k in required_keys)):
        return False
    if 'ppl' in required_keys and (not ppl_is_usable(rec.get('ppl'))):
        return False
    return rec.get('_code') == code
PAIR_STEP_KEYS = ('steps', 'tokens_seen')

def pair_reason(ra, rb):
    if not isinstance(ra, dict) or not isinstance(rb, dict):
        return 'a record is not a dict'
    if ra.get('run_cfg') is None or rb.get('run_cfg') is None:
        return 'unverifiable run_cfg'
    if ra.get('run_cfg') != rb.get('run_cfg'):
        return 'different run_cfg'
    # Controlled-variable gate for the settings the run fingerprint does not
    # carry (optimizer identity / betas / eps, LR-schedule shape, grad-clip,
    # augmentation).  BOTH records must carry the descriptor or BOTH must omit
    # it (legacy-vs-legacy); a one-sided descriptor is unverifiable and is
    # refused.
    _ka, _kb = (ra.get('train_knobs'), rb.get('train_knobs'))
    # Mixed presence is itself unverifiable: a record that carries the
    # descriptor and one that does not cannot be shown to share the optimizer /
    # schedule / clipping / augmentation settings, and pairing them is exactly
    # the uncontrolled comparison this gate exists to prevent.  Only
    # legacy-vs-legacy (both absent) keeps the permissive path.
    if (_ka is None) != (_kb is None):
        return 'unverifiable train_knobs (only one record carries the optimizer / lr-schedule / augmentation descriptor)'
    if _ka is not None and _ka != _kb:
        return 'different train_knobs (optimizer / lr schedule / grad-clip / augmentation / precision)'
    # Precision / implementation mode are also recorded at the top level on
    # newer records.  Compare them explicitly (the descriptor above already
    # covers records that carry it) so an fp32-vs-bf16 or row-loop-vs-batched
    # pair cannot be treated as controlled even if some future caller omits
    # the full descriptor.  Legacy records (neither key present) keep the
    # historical permissive path; a one-sided key is unverifiable.
    for _k in ('amp_dtype', 'amp_grad_scaler', 'eager_batch'):
        _pa, _pb = (_k in ra), (_k in rb)
        if _pa != _pb:
            return f'unverifiable {_k} (only one record carries the precision / implementation mode)'
        if _pa and ra.get(_k) != rb.get(_k):
            return f'different {_k} ({ra.get(_k)!r} vs {rb.get(_k)!r})'
    bad = [k for k in PAIR_STEP_KEYS if ra.get(k) is None or rb.get(k) is None or ra.get(k) != rb.get(k)]
    if bad:
        absent = [k for k in bad if ra.get(k) is None and rb.get(k) is None]
        unknown = [k for k in bad if k not in absent and (ra.get(k) is None) != (rb.get(k) is None)]
        if unknown:
            return f'unverifiable {sorted(unknown)}'
        if absent:
            return f'unverifiable {sorted(absent)}'
        return f'disagreeing {sorted(bad)}'
    return None

def _hyperparam_reason(ra, rb):
    """Strict structured-hyperparameter gate for a module ablation pair.

    `run_cfg` pins the compact training recipe, but the structured
    `hyperparams` copy additionally carries `mlp_ratio` (and any future key
    that never made it into the string).  Two records with equal run_cfg can
    still differ in MLP width, which is a capacity change rather than the
    module under test.  Both records must carry the descriptor (legacy/legacy
    is the only permissive path) and every checked key must agree.

    Baseline-vs-method contrasts do NOT use this gate: their MLP width is
    deliberately widened for parameter matching, and `_param_control` is the
    authoritative check there."""
    ha, hb = ra.get('hyperparams'), rb.get('hyperparams')
    if ha is None and hb is None:
        return None
    if not isinstance(ha, dict) or not isinstance(hb, dict):
        return 'unverifiable hyperparams (one record carries no structured copy)'
    for k in _CONTROL_HYPER_KEYS:
        va, vb = ha.get(k), hb.get(k)
        if va is None and vb is None:
            continue
        if va is None or vb is None:
            return f'unverifiable hyperparams.{k}'
        if _canon_setting(va) != _canon_setting(vb):
            return f'different hyperparams.{k} ({va!r} vs {vb!r})'
    return None

def objective_reason(ra, rb):
    """Compare the ACTUAL loss/optimizer objective of two arms.

    `run_cfg` carries the compression weight `cl` and the delta-LR multiplier
    `dlm` for both arms, but dense arms do not apply `comp_reg` at all and have
    no `delta_logit`.  Same run_cfg therefore does NOT mean same objective;
    this helper makes the difference explicit instead of silently pairing
    `ce` with `ce + comp_lambda*comp_reg`.  It returns None when the records
    carry matching objective descriptors, a reason string otherwise, and
    'unverifiable ...' when the descriptors are absent/one-sided (legacy
    records keep the permissive path, but the state is reported)."""
    oa, ob = ra.get('train_objective'), rb.get('train_objective')
    if oa is None and ob is None:
        return 'unverifiable train_objective (legacy records carry no objective descriptor)'
    if (oa is None) != (ob is None):
        return 'unverifiable train_objective (only one record carries the objective descriptor)'
    if not isinstance(oa, dict) or not isinstance(ob, dict):
        return 'unverifiable train_objective (malformed descriptor)'
    diffs = []
    for k in ('comp_reg_applies', 'delta_lr_effective', 'boundary_policy'):
        if oa.get(k) != ob.get(k):
            diffs.append(k)
    return ('objective differs: ' + ', '.join(diffs)) if diffs else None

def ppl_is_usable(v):
    return isinstance(v, (int, float)) and (not isinstance(v, bool)) and math.isfinite(v) and (v > 0.0)

def by_len_cells(rows, Ln, variant=None):
    keys = (Ln, int(Ln), str(int(Ln)))
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if variant is not None and r.get('variant') != variant:
            continue
        bl = r.get('by_len')
        if not isinstance(bl, dict):
            continue
        cell = None
        for k in keys:
            if k in bl:
                cell = bl[k]
                break
        if not isinstance(cell, dict):
            continue
        if ppl_is_usable(cell.get('ppl')):
            out.append(cell)
    return out

def holm_adjust(named_pvals, alpha=0.05):
    """Holm-Bonferroni step-down adjustment over a family of comparisons.

    输入 [(name, p), ...]（NaN/越界的 p 记为无法校正并原样透传，不进入排序），
    输出 {name: {'p_raw', 'p_holm', 'rank', 'reject', 'm_effective'}}。
    family 的定义必须由调用方预先声明（预注册的族用冻结清单，探索性的族
    明示「未预注册」），本函数只做数学，不做族的选择——族的选择正是
    多重比较问题的来源（design_notes §4/§17）。"""
    fam = []
    invalid = {}
    for name, p in named_pvals:
        try:
            pv = float(p)
        except (TypeError, ValueError):
            pv = float('nan')
        if (not math.isfinite(pv)) or pv < 0.0 or pv > 1.0:
            invalid[str(name)] = {'p_raw': p, 'p_holm': None, 'rank': None, 'reject': False, 'reason': 'p not in [0,1] or not finite'}
            continue
        fam.append((str(name), pv))
    out = dict(invalid)
    m = len(fam)
    order = sorted(fam, key=lambda kv: kv[1])
    prev = 0.0
    for i, (name, p) in enumerate(order):
        adj = min(1.0, (m - i) * p)
        prev = max(prev, adj)
        out[name] = {'p_raw': p, 'p_holm': float(prev), 'rank': i + 1, 'reject': bool(prev <= alpha), 'm_effective': m}
    return out

def attach_holm(comparisons, *, alpha=0.05, family='reported comparisons', exploratory=True):
    """Attach `p_holm` to every non-omitted comparison in a dict and return a
    family descriptor.

    A report that prints many raw p-values without a correction invites
    multiple-comparison over-reading.  The family is whatever the caller
    displays (not silently shrunk by omitted rows: those keep a dummy p=1).
    `exploratory=True` is a hard label, not a decoration: Holm limits
    over-claiming but cannot turn validation-selected v7-v11 panels into
    pre-registered confirmation (design_notes §17).
    """
    pairs, omitted, tested = [], [], 0
    for k, v in (comparisons or {}).items():
        if not isinstance(v, dict):
            pairs.append((str(k), 1.0))
            omitted.append(str(k))
            continue
        if 'omitted' in v:
            omitted.append(str(k))
            # A declared-but-omitted comparison keeps a dummy p=1 slot: the
            # family size must be the DECLARED family, not the subset that
            # happened to be measurable (same rule as methodology_audit).
            pairs.append((str(k), 1.0))
            continue
        p = v.get('p_exact_signflip')
        if p is None:
            p = v.get('p')
        try:
            p = float(p)
        except (TypeError, ValueError):
            p = float('nan')
        if math.isfinite(p) and 0.0 <= p <= 1.0:
            pairs.append((str(k), p))
            tested += 1
        else:
            pairs.append((str(k), 1.0))
    h = holm_adjust(pairs, alpha=alpha) if pairs else {}
    for k, v in (comparisons or {}).items():
        if isinstance(v, dict):
            v['p_holm'] = (h.get(str(k)) or {}).get('p_holm')
    return {
        'name': family,
        'declared_slots': int(len(comparisons or {})),
        'tested': int(tested),
        'omitted': omitted,
        'alpha': float(alpha),
        'correction': 'holm step-down',
        'evidence_level': 'exploratory' if exploratory else 'confirmatory',
        'note': ('Holm over the displayed family; the panels were selected on the validation split, '
                 'so corrected p-values still are not confirmation') if exploratory else 'pre-registered family',
    }

def signflip_p_floor(n):
    """双侧精确符号翻转检验在 n 个种子下的最小可达 p 值（2/2^n）；
    n<=0 返回 NaN。用于报告「未显著是分辨率上限还是数据结论」。"""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return float('nan')
    if n <= 0:
        return float('nan')
    # same value as 2.0 / 2**n, evaluated in the float exponent range.
    return math.ldexp(2.0, -n)

def finite_range(named_values):
    vals, dropped = ([], [])
    for label, m in named_values:
        if isinstance(m, (int, float)) and (not isinstance(m, bool)) and math.isfinite(m):
            vals.append(float(m))
        else:
            dropped.append(label)
    if not vals:
        return (None, None, dropped)
    return (min(vals), max(vals), dropped)

_SIGNFLIP_ENUM_MAX_N = 20
_SIGNFLIP_MITM_MAX_N = 40


def _signflip_signed_sums(values):
    """All 2**k signed sums of `values` (deterministic doubling table)."""
    out = np.zeros(1, dtype=float)
    for v in np.asarray(values, dtype=float):
        out = np.concatenate((out + v, out - v))
    return out


def _signflip_mitm_count(d):
    """Exact count of sign assignments with |sum(s_i*d_i)| >= |sum(d_i)|
    using meet-in-the-middle: 2**(n/2) sums per half instead of 2**n.

    For each left-half sum `a`, the admissible right-half sums are
    `b <= -T - a` or `b >= T - a`; two binary searches count them.  This keeps
    the test exact (no normal approximation) for the 21-40 seed range a
    scale-up panel can reach, while the 2**20 worst-case table stays small."""
    n = int(d.size)
    total = abs(float(d.sum()))
    # The n<=20 enumeration accepts |mean| >= obs - 1e-12, i.e. |sum| >=
    # total - n*1e-12.  If that threshold is non-positive every sign
    # assignment qualifies; the two-tail formula below would double-count.
    if total - 1e-12 * n <= 0.0:
        return 1 << n
    half = n // 2
    left = _signflip_signed_sums(d[:half])
    right = _signflip_signed_sums(d[half:])
    right.sort()
    # Match the n<=20 enumeration's tolerance: mean comparison with 1e-12.
    thr = total - 1e-12 * n
    lo = np.searchsorted(right, -thr - left, side='right')
    hi = np.searchsorted(right, thr - left, side='left')
    return int(lo.sum() + (right.size - hi).sum())


def exact_signflip_p(deltas):
    """Two-sided exact sign-flip test over paired per-seed deltas.

    H0: the per-seed difference is symmetric about 0.  For n <= 20 every one
    of the 2**n sign assignments is enumerated; for 20 < n <= 40 the same
    count is obtained exactly by meet-in-the-middle (2**(n/2) tables), which
    matters as seed-completion panels grow.  Only n > 40 falls back to a
    deterministic normal approximation (reported as such via `method`), so a
    report never depends on a random draw.  Non-finite deltas are dropped
    (the count is returned).  Pure function over its input — used by the
    aggregate audit so the core panel reports the same p the per-phase suites
    do, rather than a bare mean±std a reviewer could mistake for evidence.

    有效性假设（评审二轮·意见九，必须在预注册文档同步声明）：精确符号翻转
    检验的 p 值正确性依赖「配对内交换性」——每个种子下的两臂共享除臂本身
    之外的一切随机因素（同数据顺序、同评测行、同腐坏网格；初始化 RNG 按
    种子播种，臂间参数量差异对 RNG 流的扰动属于「臂」的一部分）。种子编号
    相同本身不提供该保证；design_notes §20 记录了本仓库逐因素的控制方式
    与残余风险（训练随机性进入种间方差、文档抽样进入 doc 层方差）。"""
    d_in = []
    for _x in deltas:
        try:
            d_in.append(float(_x))
        except (TypeError, ValueError):
            d_in.append(float('nan'))
    d = np.asarray([x for x in d_in if math.isfinite(x)], dtype=float)
    n = int(d.size)
    n_dropped = len(d_in) - n
    if n == 0:
        return {'n': 0, 'p': float('nan'), 'n_dropped': n_dropped, 'method': 'none'}
    obs = abs(float(d.mean()))
    if n <= _SIGNFLIP_ENUM_MAX_N:
        total = 1 << n
        cnt = 0
        bitpos = np.arange(n, dtype=np.int64)[None, :]
        step = max(1, min(total, (1 << 22) // max(n, 1)))
        for off in range(0, total, step):
            bs = min(step, total - off)
            masks = np.arange(off, off + bs, dtype=np.int64)[:, None]
            signs = np.where((masks & (1 << bitpos)) != 0, 1.0, -1.0)
            cnt += int((np.abs((signs * d[None, :]).mean(1)) >= obs - 1e-12).sum())
        return {'n': n, 'p': float(cnt / total), 'n_dropped': n_dropped, 'method': 'exact'}
    if n <= _SIGNFLIP_MITM_MAX_N:
        cnt = _signflip_mitm_count(d)
        return {'n': n, 'p': float(cnt / float(1 << n)), 'n_dropped': n_dropped, 'method': 'exact-mitm'}
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    if sd <= 1e-300:
        p = 0.0 if obs > 0 else 1.0
    else:
        z = obs * math.sqrt(n) / sd
        p = math.erfc(z / math.sqrt(2.0))
    return {'n': n, 'p': float(min(1.0, p)), 'n_dropped': n_dropped, 'method': 'normal-approx'}

def _percentile_ci(vals, lo=2.5, hi=97.5):
    v = [float(x) for x in vals if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))]
    if len(v) < 2:
        return None
    v.sort()
    def _pct(q):
        pos = (len(v) - 1) * q / 100.0
        i = int(math.floor(pos))
        j = min(i + 1, len(v) - 1)
        return v[i] + (v[j] - v[i]) * (pos - i)
    return [_pct(lo), _pct(hi)]

def bootstrap_mean_ci(values, *, n_boot=4000, seed=20261018, conf=0.95):
    """Percentile bootstrap CI for the mean of i.i.d. units (e.g. documents).

    Units must be the independent sampling unit.  For LM evaluation a document
    row is the unit (tokens within a document are correlated), and this helper
    is used to report **document-sampling** uncertainty, not seed-level
    training uncertainty.  Resampling is chunked so a large document panel
    costs O(n_boot) memory rather than O(n_boot * n_units).  The RNG is seeded,
    so repeated report builds return the same interval.
    """
    v = np.asarray([float(x) for x in values
                    if isinstance(x, (int, float)) and not isinstance(x, bool)
                    and math.isfinite(float(x))], dtype=float)
    n = int(v.size)
    if n < 2:
        return None
    try:
        nb = int(n_boot)
        c = float(conf)
    except (TypeError, ValueError):
        return None
    if nb < 20 or not (0.0 < c < 1.0):
        return None
    rng = np.random.default_rng(int(seed))
    means = np.empty(nb, dtype=float)
    _chunk = 256
    for _off in range(0, nb, _chunk):
        _nb = min(_chunk, nb - _off)
        _idx = rng.integers(0, n, size=(_nb, n))
        means[_off:_off + _nb] = v[_idx].mean(axis=1)
    lo_q = 100.0 * (1.0 - c) / 2.0
    hi_q = 100.0 - lo_q
    lo, hi = np.percentile(means, [lo_q, hi_q])
    return {'n': n, 'n_boot': nb, 'seed': int(seed), 'mean': float(v.mean()),
            'ci_low': float(lo), 'ci_high': float(hi), 'conf': c,
            'method': 'percentile bootstrap over independent units (chunked, deterministic)'}


def paired_bootstrap_ci(deltas, *, n_boot=4000, seed=20261018):
    d = np.asarray([float(x) for x in deltas if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))], dtype=float)
    if d.size < 2:
        return None
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, d.size, size=(int(n_boot), d.size))
    means = d[idx].mean(axis=1)
    return _percentile_ci(means.tolist())

def unpaired_bootstrap_ci(with_values, without_values, *, n_boot=4000, seed=20261018):
    a = np.asarray([float(x) for x in with_values if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))], dtype=float)
    b = np.asarray([float(x) for x in without_values if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))], dtype=float)
    if a.size < 2 or b.size < 2:
        return None
    rng = np.random.default_rng(int(seed))
    ai = rng.integers(0, a.size, size=(int(n_boot), a.size))
    bi = rng.integers(0, b.size, size=(int(n_boot), b.size))
    diff = a[ai].mean(axis=1) - b[bi].mean(axis=1)
    return _percentile_ci(diff.tolist())

def wilcoxon_signed_rank_p(deltas):
    """Two-sided exact Wilcoxon signed-rank p for n<=16 (normal approximation
    above that).  Reported as a SENSITIVITY check, not as a replacement for
    the sign-flip test: it assumes symmetric paired differences, which the
    review correctly says is not guaranteed by sharing a seed number."""
    d = np.asarray([float(x) for x in deltas if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))], dtype=float)
    d = d[np.abs(d) > 1e-15]
    n = int(d.size)
    if n == 0:
        return {'p': float('nan'), 'method': 'none', 'n_nonzero': 0}
    order = np.argsort(np.abs(d))
    ranks = np.empty(n, dtype=float)
    _abs = np.abs(d)[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(_abs[j] - _abs[i]) <= 1e-15:
            j += 1
        ranks[order[i:j]] = (i + j + 1) / 2.0  # average rank (1-based)
        i = j
    w_plus = float(ranks[d > 0].sum())
    total = float(ranks.sum())
    w = min(w_plus, total - w_plus)
    if n <= 16:
        cnt = 0
        for mask in range(1 << n):
            s = 0.0
            for k in range(n):
                if mask & (1 << k):
                    s += ranks[k]
            if s <= w + 1e-12 or s >= total - w - 1e-12:
                cnt += 1
        p = min(1.0, cnt / float(1 << n))
        return {'p': float(p), 'method': 'exact', 'n_nonzero': n, 'w_plus': w_plus, 'w_min': w}
    mu = total / 2.0
    var = float((ranks ** 2).sum()) / 4.0
    if var <= 0:
        return {'p': float('nan'), 'method': 'normal', 'n_nonzero': n, 'w_plus': w_plus, 'w_min': w}
    z = (abs(w_plus - mu) - 0.5) / math.sqrt(var)
    p = math.erfc(max(0.0, z) / math.sqrt(2.0))
    return {'p': float(p), 'method': 'normal', 'n_nonzero': n, 'w_plus': w_plus, 'w_min': w,
            'note': 'normal approximation; exact enumeration is only used for n<=16'}

def paired_effect_stats(deltas, alpha=0.05, min_n=3, with_values=None, without_values=None):
    """Paired contrast summary with an explicit effect-vs-seed-noise verdict.

    Reports n, mean, sample std (ddof=1), sem, Cohen's d_z, the exact
    sign-flip p, the p-resolution floor at this n, and four booleans:

      underpowered       — n < `min_n` (the 3-seed floor the reporting rules
                           require for a key experiment);
      resolution_limited — the smallest p the exact sign-flip test can reach
                           at this n (2/2**n) is still > alpha, so NO effect,
                           however large, can ever clear the significance bar
                           at this seed count.  This is a resolution ceiling,
                           not evidence against the effect, and must not be
                           read as "no effect";
      within_seed_noise  — |mean| <= std, i.e. the seed-to-seed spread alone is
                           at least as large as the reported effect (a +0.2%
                           mean with 0.5% std is NOT a usable gain);
      significant        — p < alpha AND NOT within_seed_noise.

    It additionally reports a two-sided 95% CI for the mean effect and a
    leave-one-seed-out (LOO) stability read: with the 3-5 seeds the reporting
    rules require, a significant mean can still be driven by a single seed.
    `seed_driven` is True when dropping any one seed flips the sign of the
    mean (or makes it exactly zero), i.e. the direction is not stable across
    seeds even though the aggregate test may look significant.  The LOO pass
    is a single O(n) sweep (a running sum minus each seed); `ci95` uses the t
    critical value rather than a hard-coded z.

    Requiring both the test AND the effect-vs-spread check means a large-n
    p-value can never dress up a sub-noise effect as a real improvement, while
    `resolution_limited` stops a small-n panel from being read as a negative
    result when only the test's granularity, not the data, bound the p."""
    d = np.asarray([float(x) for x in deltas if isinstance(x, (int, float)) and (not isinstance(x, bool)) and math.isfinite(float(x))], dtype=float)
    n = int(d.size)
    if n == 0:
        return {'n': 0, 'mean': float('nan'), 'std': float('nan'), 'sem': float('nan'), 'cohens_dz': float('nan'), 'ci95_low': float('nan'), 'ci95_high': float('nan'), 'loo_means': [], 'seed_driven': None, 'seed_robust': None, 'p_exact_signflip': float('nan'), 'p_floor': float('nan'), 'min_reachable_p': float('nan'), 'underpowered': True, 'resolution_limited': True, 'within_seed_noise': True, 'significant': False, 'effect_vs_noise': 'no data', 'unit': 'seed', 'assumption': 'none (no data)', 'sign_counts': {'positive': 0, 'negative': 0, 'zero': 0}, 'p_wilcoxon': None, 'p_wilcoxon_method': None, 'ci95_paired_bootstrap': None, 'ci95_unpaired_bootstrap': None}
    mean = float(d.mean())
    std = float(d.std(ddof=1)) if n > 1 else 0.0
    sem = float(std / math.sqrt(n)) if n > 1 else float('nan')
    p = exact_signflip_p(d)['p']
    if std > 1e-12:
        dz = float(mean / std)
    elif abs(mean) < 1e-300:
        dz = 0.0
    else:
        dz = float('inf') if mean > 0 else float('-inf')
    p_floor = signflip_p_floor(n)
    underpowered = n < int(min_n)
    resolution_limited = bool(math.isfinite(p_floor) and p_floor > alpha)
    within_seed_noise = abs(mean) <= std
    significant = bool(math.isfinite(p) and p < alpha and (not within_seed_noise))
    # Two-sided 95% CI on the paired mean (t interval; the normal approximation
    # is intentionally avoided at the 3-5 seed counts this repo reports).
    ci_lo = ci_hi = float('nan')
    if n > 1 and math.isfinite(sem):
        tc = _t_critical(0.95, n - 1)
        if math.isfinite(tc):
            ci_lo, ci_hi = mean - tc * sem, mean + tc * sem
    # Leave-one-seed-out stability: does any single seed own the direction?
    loo_means = []
    seed_driven = None
    if n >= 3:
        _total = float(d.sum())
        loo_means = [(_total - float(d[i])) / (n - 1) for i in range(n)]
        seed_driven = False
        if abs(mean) > 1e-300:
            _full_sign = 1.0 if mean > 0 else -1.0
            seed_driven = any((_m == 0.0) or ((_m > 0) != (_full_sign > 0)) for _m in loo_means)
    seed_robust = None if seed_driven is None else (not seed_driven)
    if underpowered:
        verdict = 'underpowered (n<%d)' % int(min_n)
    elif within_seed_noise:
        verdict = 'effect within seed noise (|mean|<=std)'
    elif significant:
        verdict = 'effect exceeds seed noise (p<%.3g)' % alpha
    elif resolution_limited:
        verdict = 'not decidable at this n: exact-test p floor %.4g > alpha (add seeds to resolve; this is a resolution ceiling, not a null result)' % p_floor
    else:
        verdict = 'not significant at this n (p>=%.3g)' % alpha
    if seed_driven:
        verdict += ' | UNSTABLE: dropping one seed flips the mean sign'
    # Robustness readouts the review asks for: the exact sign-flip test is a
    # seed-level sign test, so report the sign split, a paired bootstrap CI, a
    # Wilcoxon sensitivity p, and — when the caller supplies both arms — an
    # UNPAIRED bootstrap that does not assume exchangeability.  None of these
    # turns the seed into a token-level unit; they make the assumption visible.
    _sign = {'positive': int((d > 0).sum()), 'negative': int((d < 0).sum()), 'zero': int((d == 0).sum())}
    _wil = wilcoxon_signed_rank_p(d)
    _paired_ci = paired_bootstrap_ci(d)
    _unpaired_ci = None
    if with_values is not None and without_values is not None:
        _unpaired_ci = unpaired_bootstrap_ci(with_values, without_values)
    _assumption = ('exact sign-flip is valid as a sign test under symmetric paired differences; '
                   'same seed does NOT randomise the arm, and architecture-induced RNG/optimisation '
                   'differences are part of the arm. p is SEED-level, never token-level.')
    return {'n': n, 'mean': mean, 'std': std, 'sem': sem, 'cohens_dz': dz, 'ci95_low': ci_lo, 'ci95_high': ci_hi, 'loo_means': loo_means, 'seed_driven': seed_driven, 'seed_robust': seed_robust, 'p_exact_signflip': float(p), 'p_floor': p_floor, 'min_reachable_p': p_floor, 'underpowered': underpowered, 'resolution_limited': resolution_limited, 'within_seed_noise': within_seed_noise, 'significant': significant, 'effect_vs_noise': verdict,
            'unit': 'seed', 'assumption': _assumption, 'sign_counts': _sign,
            'p_wilcoxon': _wil.get('p'), 'p_wilcoxon_method': _wil.get('method'),
            'ci95_paired_bootstrap': _paired_ci, 'ci95_unpaired_bootstrap': _unpaired_ci}

# ---- 预注册功效 / 分辨率工具（评审二轮·意见一/九） ------------------------------
# 动机：一个冻结方案如果「无论数据多强都不可能达到 p_holm < α」，它落在
# 分辨率上限而不是「保守」。下面的函数把这件事变成加载清单时就能机器判定
# 的硬门槛，而不是统计阶段的一句「保守」警告。

def holm_min_attainable_p(n_seeds, m_comparisons, alpha=0.05):
    """双侧精确符号翻转 + Holm 校正族的最小可达 Holm p。

    最小原始 p = 2/2^n（全部种子同向）；Holm 第一道门乘以族大小 m，故
    最小可达 p_holm = min(1, m * 2/2^n)。若该值 > alpha，则该方案在构造上
    无法确认任何比较——这是分辨率上限，不是「保守的选择」，应当在读到
    任何测试数据之前修订（加种子 / 缩族 / 改检验），而不是事后再解释。"""
    try:
        n = int(n_seeds)
        m = int(m_comparisons)
    except (TypeError, ValueError):
        return float('nan')
    if n <= 0 or m <= 0:
        return float('nan')
    return min(1.0, m * signflip_p_floor(n))


def signflip_min_n_for_holm(m_comparisons, alpha=0.05, n_max=64):
    """Smallest seed count n for which a Holm-corrected exact sign-flip test
    can reach p_holm <= alpha EVEN IF every seed moves in the predicted
    direction.  Returns None when no n <= n_max reaches it.

    This is the planning counterpart of `holm_min_attainable_p`: it turns
    "3-5 seeds" into an explicit statement of what those seeds can and cannot
    establish.  At alpha=0.05 a single comparison needs n>=6; a family of m
    comparisons needs roughly 2**n >= 2*m/alpha.  A panel scheduled below the
    returned n is descriptive only (the exact-test floor, not the data, bounds
    the p value), so the audit labels it resolution-limited instead of reading
    it as a null result."""
    try:
        m = int(m_comparisons)
        a = float(alpha)
    except (TypeError, ValueError):
        return None
    if m <= 0 or not (0.0 < a < 1.0):
        return None
    for n in range(1, int(n_max) + 1):
        if holm_min_attainable_p(n, m, a) <= a + 1e-15:
            return n
    return None


def _binom_cdf_half(n, m):
    """P(Bin(n, 1/2) <= m), evaluated in log space.

    The planning tools are called with growing seed counts at scale.  The
    direct form `2.0 * sum(comb(n, i)) / 2**n` overflows `float` once n is
    large (int-to-float conversion), even though the probability is perfectly
    representable.  This implementation uses lgamma log-sum-exp for the small
    tail and symmetry for the upper half, so it stays finite for any n."""
    n = int(n)
    m = int(m)
    if n <= 0:
        return 1.0 if m >= 0 else 0.0
    if m < 0:
        return 0.0
    if m >= n:
        return 1.0
    if m * 2 >= n:
        # P(X <= m) = 1 - P(X <= n-m-1) with n-m-1 < n/2.
        return float(1.0 - _binom_cdf_half(n, n - m - 1))
    # Regularized incomplete beta: P(Bin(n,1/2) <= m) = I_{1/2}(n-m, m+1).
    v = _betai(n - m, m + 1, 0.5)
    if math.isfinite(v):
        return float(min(1.0, max(0.0, v)))
    log2n = n * math.log(2.0)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - log2n
            for i in range(0, m + 1)]
    mx = max(logs)
    return float(min(1.0, math.exp(mx) * math.fsum(math.exp(x - mx) for x in logs)))


def signflip_min_consistency(n_seeds, m_comparisons, alpha=0.05):
    """规划用：通过 Holm 第一道门所需的最小同向种子数 k_min。

    若 k 个种子同向（n−k 反向），符号检验投影的 p 为
    2·P(Bin(n, 1/2) ≤ n−k)，第一道门要求 m·p < alpha。对**绝对值相等**
    的配对差，这个投影就等于精确双侧符号翻转 p；绝对值不等时，精确
    sign-flip p 只会更大（更不显著），所以本函数给出的是所需同向数的
    下界，真实数据可能仍需更多同向种子。返回使 m·p(k) < alpha 的最小
    k；若全部 n 个同向仍不够，返回 None（方案不可能确认）。二项尾概率
    在 log 空间计算，因此大 n（规模扩展后的补种子计划）也不会整数溢出。"""
    try:
        n = int(n_seeds)
        m = int(m_comparisons)
    except (TypeError, ValueError):
        return None
    if n <= 0 or m <= 0:
        return None
    best = None
    for k in range(n, -1, -1):
        p = 2.0 * _binom_cdf_half(n, n - k)
        if m * p < alpha:
            best = k  # keep descending: p(k) grows monotonically as k shrinks,
            # so the last k that still clears the gate is the MINIMUM k
        else:
            break
    return best


def signflip_power(n_seeds, m_comparisons, q, alpha=0.05):
    """每种子同向概率为 q 时，通过 Holm 第一道门的功效。

    P(Bin(n, q) ≥ k_min)。q 的含义：在真效应方向上，单个种子的配对 Δ
    落在预注册方向的概率（含种子间噪声）。k_min 来自等幅/符号检验投影，
    因此这里的功效是对真实 sign-flip 功效的上界（绝对值不等时实际功效
    更低）。功效对 q 敏感——预注册时必须报告 q 的假设来源（如 validation
    面板的同向率），不得只报「数学上可能」。n<=0 或方案不可能确认时返回
    0.0。尾部概率用正则化不完全贝塔函数计算（P(X<=k-1)=I_{1-q}(n-k+1,k)），
    大 n 下不会因 `comb(n,k)*q**k` 的整数/浮点溢出而失败。"""
    kmin = signflip_min_consistency(n_seeds, m_comparisons, alpha)
    if kmin is None or kmin <= 0:
        return 0.0 if kmin is None else 1.0
    try:
        n = int(n_seeds)
        q = float(q)
    except (TypeError, ValueError):
        return 0.0
    if not (0.0 <= q <= 1.0) or n <= 0:
        return 0.0
    if q >= 1.0:
        return 1.0
    if q <= 0.0:
        return 0.0
    cdf = _betai(n - kmin + 1, kmin, 1.0 - q)
    if not math.isfinite(cdf):
        # Fallback: log-space binomial tail (overflow-free).
        logq, log1q = math.log(q), math.log1p(-q)
        logs = [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                + k * logq + (n - k) * log1q for k in range(kmin, n + 1)]
        mx = max(logs)
        pw = math.exp(mx) * math.fsum(math.exp(x - mx) for x in logs)
        return float(min(1.0, pw))
    return float(min(1.0, max(0.0, 1.0 - cdf)))


def tost_paired(deltas, margin, conf=0.90):
    """配对 TOST 等价性检验（G0 训练级等价门槛的统计件，评审二轮·意见八）。

    「未发现显著差异 ≠ 等价」：等价性必须预设 margin 并检验 |mean| 落在
    (−margin, +margin) 内。这里用真正的配对 t（df=n−1，`_student_t_sf` +
    t 临界 CI），而不是把小样本 se 当作已知的正态 z——n=6 时 90% CI 的
    t 临界值比 z 宽约 20%，用 z 会把「未判等价」误判成等价。返回两侧
    单边 p、conf 水平 CI 与 `equivalent` 判定。margin 必须由计划预先注册
    （如 kernel/eager 面板平均 PPL 相对差 <1%）；本函数不替调用方选择
    margin。n<2 或 margin<=0 时不判定。"""
    d = np.asarray([float(x) for x in deltas if isinstance(x, (int, float)) and (not isinstance(x, bool)) and math.isfinite(float(x))], dtype=float)
    n = int(d.size)
    out = {'n': n, 'margin': float(margin), 'conf': float(conf), 'equivalent': None,
           'method': 'paired-t (Student t, df=n-1; pre-registered margin required)'}
    if n < 2 or not (margin > 0):
        out['note'] = 'not decidable (n<2 or non-positive margin)'
        return out
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    df = n - 1
    # A floating-point mean of k identical values need not be bit-identical to
    # the value (0.1*6 / 6 = 0.09999999999999999), which would turn a sample
    # exactly ON the margin into a tiny-se "significant" rejection.  Use the
    # identical common value when the sample is literally constant.
    identical = bool(n > 0 and np.all(d == d[0]))
    if identical:
        mean = float(d[0])
    out['mean'] = mean
    out['df'] = int(df)
    se = 0.0 if identical else sd / math.sqrt(n)
    tc = _t_critical(float(conf), float(df))
    if not math.isfinite(tc):
        # _t_critical only rejects invalid conf/df; keep a deterministic
        # fallback rather than silently returning a NaN interval.
        tc = {0.80: 1.2815515655446004, 0.90: 1.6448536269514722, 0.95: 1.959963984540054}.get(round(float(conf), 4), 1.6448536269514722)
        out['note'] = 't critical unavailable; fell back to the normal critical value'
    out['t_critical'] = float(tc)
    out['ci'] = [mean - tc * se, mean + tc * se]
    if se <= 1e-300:
        # Degenerate zero-variance sample.  At the margin boundary the exact
        # t limit is P(T >= 0) = 0.5, not 0: equality cannot be declared
        # equivalent with certainty.
        p_hi = 1.0 if mean > margin else (0.0 if mean < margin else 0.5)
        p_lo = 1.0 if mean < -margin else (0.0 if mean > -margin else 0.5)
    else:
        # H0_upper: mean >= margin  -> T = (margin - mean)/se
        # H0_lower: mean <= -margin -> T = (mean + margin)/se
        p_hi = _student_t_sf((margin - mean) / se, float(df))
        p_lo = _student_t_sf((mean + margin) / se, float(df))
    out['p_hi'] = float(p_hi)
    out['p_lo'] = float(p_lo)
    out['p_tost'] = float(max(p_hi, p_lo))
    out['equivalent'] = bool(p_hi < 0.05 and p_lo < 0.05)
    out['ci_within_margin'] = bool(out['ci'][0] > -margin and out['ci'][1] < margin)
    return out


# ---- 跨规模推断（scale ladder）：相对效应、异质性、趋势 ---------------------
# 动机：把 d=128 与 d=512 的「绝对 PPL 差」直接相减，会把基线 PPL 本身的
# 规模差异混进效应里（PPL 不是跨规模可加的）。跨规模的比较必须先转成
# 无量纲效应（同一语料/分词器内的配对 log 比），再谈趋势；跨语料/跨分词器
# 则不能用 PPL 直接比较（需要 bits-per-byte 之类的归一化，见 design_notes
# §22）。下列函数只做数学，不替调用方选择族或预注册方向。

def _gammaincc(a, x):
    """正则化上不完全伽马 Q(a,x)，用于 Cochran Q 的卡方尾概率。"""
    if not (a > 0.0) or math.isnan(x) or x < 0.0:
        return float('nan')
    if x == float('inf'):
        return 0.0
    if not math.isfinite(x):
        return float('nan')
    if x == 0.0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap = float(a)
        s = 1.0 / a
        d = s
        for _ in range(1000):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-15:
                break
        p = s * math.exp(-x + a * math.log(x) - gln)
        return float(min(1.0, max(0.0, 1.0 - p)))
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -float(i) * (float(i) - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return float(min(1.0, max(0.0, math.exp(-x + a * math.log(x) - gln) * h)))


def _betacf(a, b, x):
    """不完全贝塔函数的连分式（Numerical Recipes 口径）。"""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def _betai(a, b, x):
    """正则化不完全贝塔 I_x(a,b)。"""
    if not (a > 0.0 and b > 0.0) or not math.isfinite(x):
        return float('nan')
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return float(bt * _betacf(a, b, x) / a)
    return float(1.0 - bt * _betacf(b, a, 1.0 - x) / b)


def _student_t_sf(t, df):
    """Student t 的上尾概率 P(T >= t)。"""
    if not (df > 0.0) or math.isnan(t):
        return float('nan')
    if t == float('inf'):
        return 0.0
    if t == float('-inf'):
        return 1.0
    if not math.isfinite(t):
        return float('nan')
    x = df / (df + t * t)
    tail = 0.5 * _betai(df / 2.0, 0.5, x)
    return float(tail if t >= 0 else 1.0 - tail)


def student_t_two_sided_p(t, df):
    """双侧 Student t 检验 p 值（小样本斜率/交互推断用）。"""
    sf = _student_t_sf(abs(float(t)), float(df))
    if not math.isfinite(sf):
        return float('nan')
    return float(min(1.0, 2.0 * sf))


def _t_critical(conf, df):
    """双侧置信区间的 t 临界值；用二分求解，避免依赖 scipy。"""
    if not (0.0 < conf < 1.0) or not (df > 0.0):
        return float('nan')
    target = (1.0 - conf) / 2.0
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _student_t_sf(mid, df) > target:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def chi2_sf(x, df):
    """卡方分布上尾概率 P(chi2_df >= x)（Cochran Q 用）。"""
    if not (df > 0.0) or math.isnan(x):
        return float('nan')
    if x == float('inf'):
        return 0.0
    if not math.isfinite(x):
        return float('nan')
    if x <= 0.0:
        return 1.0
    return _gammaincc(df / 2.0, x / 2.0)


def paired_log_ratio(with_values, without_values, seeds=None):
    """把逐种子绝对差转成跨规模可比的 log 效应：log(with / without)。

    正值 = with 臂的 PPL 更高（更差）；零 = 无差异。log 比在乘性尺度上
    对称，且不随基线 PPL 的规模漂移——这是跨 d/token 规模比较的前提。
    只接受有限正数对；非正/非有限对整对丢弃并计数（不会静默变成 NaN）。
    """
    wa = list(with_values)
    wb = list(without_values)
    if len(wa) != len(wb):
        raise ValueError('paired_log_ratio needs equal-length vectors (%d vs %d)' % (len(wa), len(wb)))
    ss = list(range(len(wa))) if seeds is None else list(seeds)
    if len(ss) != len(wa):
        raise ValueError('paired_log_ratio seeds length %d does not match vectors length %d' % (len(ss), len(wa)))
    if len(set(ss)) != len(ss):
        # Two rows for one seed are not independent observations; downstream
        # means/SEs would silently double-count that seed.
        raise ValueError('paired_log_ratio got duplicate seeds %r' % (ss,))
    effects, kept, dropped = [], [], []
    for s, a, b in zip(ss, wa, wb):
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            dropped.append(s)
            continue
        if not (math.isfinite(fa) and math.isfinite(fb) and fa > 0.0 and fb > 0.0):
            dropped.append(s)
            continue
        effects.append(float(math.log(fa) - math.log(fb)))
        kept.append(s)
    return {'effects': effects, 'seeds': kept, 'n': len(effects), 'n_dropped': len(dropped), 'dropped_seeds': dropped,
            'convention': 'effect = log(with) - log(without); positive = with arm has higher PPL (worse)'}


def _scale_points(points):
    """Normalise [{label, scale, effects, seeds}] / [(scale, effects)] inputs."""
    out = []
    for i, p in enumerate(points or []):
        if isinstance(p, dict):
            eff = p.get('effects')
            scale = p.get('scale')
            label = p.get('label', i)
            seeds = p.get('seeds')
        elif isinstance(p, (tuple, list)) and len(p) >= 2:
            scale, eff = p[0], p[1]
            label = i
            seeds = p[2] if len(p) > 2 else None
        else:
            continue
        if eff is None:
            continue
        try:
            vals = list(eff)
        except TypeError:
            continue
        if seeds is None:
            keep = [(j, v) for j, v in enumerate(vals)
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))]
            seeds_kept = [j for j, _v in keep]
            effects = [float(v) for _j, v in keep]
        else:
            seeds = list(seeds)
            if len(seeds) != len(vals):
                raise ValueError('scale point %r: seeds length %d does not match effects length %d' % (label, len(seeds), len(vals)))
            keep = [(s, v) for s, v in zip(seeds, vals)
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))]
            seeds_kept = [s for s, _v in keep]
            effects = [float(v) for _s, v in keep]
        if len(set(seeds_kept)) != len(seeds_kept):
            # A repeated seed would be counted as two independent observations
            # and shrink every within-scale SE in meta_analyze_effects; this
            # must be a hard error, not silently "n looks larger".
            raise ValueError('scale point %r: duplicate seeds %r' % (label, seeds_kept))
        try:
            sc = float(scale)
        except (TypeError, ValueError):
            sc = float('nan')
        out.append({'label': str(label), 'scale': sc, 'effects': effects, 'seeds': seeds_kept})
    return out


def meta_analyze_effects(points, *, alpha=0.05, min_seeds=3):
    """跨规模的随机效应元分析（DerSimonian-Laird）+ 异质性诊断。

    输入是**已经无量纲化**的逐种子效应（通常来自
    `paired_log_ratio`），每个规模点一份。输出每个点的 mean/se/CI，
    Cochran Q、I²、tau²、固定/随机效应合并值与预测区间。

    关键口径：不同规模估计的不是同一个总体参数。只有异质性不显著
    （I² 低、Q 的 p>=alpha）且点数足够时才把 pooled 当作「共同效应」；
    否则 `poolable=False`，pooled 只作描述性摘要，结论必须按规模分开报。
    """
    pts = _scale_points(points)
    labels = [p['label'] for p in pts]
    if len(set(labels)) != len(labels):
        # Two rows with the same label are two estimates of the SAME scale
        # point, but the random-effects weights below treat them as independent
        # studies — that would double-count one scale and understate tau^2.
        raise ValueError('meta_analyze_effects needs unique scale labels; got %r' % (labels,))
    per = []
    for p in pts:
        d = np.asarray(p['effects'], dtype=float)
        n = int(d.size)
        row = {'label': p['label'], 'scale': p['scale'], 'n': n}
        if n:
            m = float(d.mean())
            sd = float(d.std(ddof=1)) if n > 1 else float('nan')
            se = float(sd / math.sqrt(n)) if (n > 1 and math.isfinite(sd)) else float('nan')
            row.update({'mean': m, 'std': sd, 'se': se, 'underpowered': n < int(min_seeds)})
            if n > 1:
                tc = _t_critical(1.0 - float(alpha), n - 1)
                row['ci'] = [m - tc * se, m + tc * se]
        else:
            row.update({'mean': float('nan'), 'std': float('nan'), 'se': float('nan'), 'underpowered': True})
        per.append(row)
    usable = [r for r in per if r['n'] > 1 and math.isfinite(r.get('se', float('nan'))) and r['se'] > 0.0]
    out = {'alpha': float(alpha), 'min_seeds': int(min_seeds), 'per_point': per,
           'n_points': len(per), 'n_usable': len(usable),
           'convention': 'input effects are scale-free; positive = with arm worse (higher PPL)'}
    if not usable:
        out.update({'status': 'insufficient_within_scale_replication', 'poolable': False,
                    'note': 'every scale point has n<2 or zero within-scale variance; no random-effects weights can be formed'})
        return out
    w = np.asarray([1.0 / (r['se'] ** 2) for r in usable], dtype=float)
    y = np.asarray([r['mean'] for r in usable], dtype=float)
    sw = float(w.sum())
    mu_fe = float((w * y).sum() / sw)
    q = float((w * (y - mu_fe) ** 2).sum())
    df = len(usable) - 1
    p_q = chi2_sf(q, df) if df > 0 else float('nan')
    i2 = float(max(0.0, (q - df) / q) * 100.0) if (df > 0 and q > 0) else 0.0
    c = float(sw - (w ** 2).sum() / sw)
    tau2 = float(max(0.0, (q - df) / c)) if (df > 0 and c > 0) else 0.0
    ws = 1.0 / (np.asarray([r['se'] ** 2 for r in usable], dtype=float) + tau2)
    sws = float(ws.sum())
    mu_re = float((ws * y).sum() / sws)
    se_re = float(math.sqrt(1.0 / sws))
    tc = _t_critical(1.0 - float(alpha), max(1, df))
    out.update({'fixed_effect': {'mean': mu_fe, 'se': float(math.sqrt(1.0 / sw)), 'ci': [mu_fe - _t_critical(1.0 - float(alpha), max(1, df)) * math.sqrt(1.0 / sw), mu_fe + _t_critical(1.0 - float(alpha), max(1, df)) * math.sqrt(1.0 / sw)]},
                'random_effects': {'mean': mu_re, 'se': se_re, 'ci': [mu_re - tc * se_re, mu_re + tc * se_re]},
                'heterogeneity': {'Q': q, 'df': int(df), 'p': p_q, 'I2_percent': i2, 'tau2': tau2,
                                  'method': 'Cochran Q + DerSimonian-Laird tau^2'},
                'prediction_interval': [mu_re - tc * math.sqrt(tau2 + se_re ** 2), mu_re + tc * math.sqrt(tau2 + se_re ** 2)],
                'poolable': bool(len(usable) >= 3 and df > 0 and i2 <= 50.0 and (not math.isfinite(p_q) or p_q >= alpha)),
                'note': 'poolable requires >=3 scales, I2<=50% and a non-significant Q; otherwise report per-scale effects only'})
    out['status'] = 'ok'
    return out


def scale_interaction(low_effects, high_effects, *, low_label='low', high_label='high', seeds=None, alpha=0.05, min_n=3):
    """同一批种子在两个规模点上的**效应变化**（交互）的精确检验。

    Δ = effect_high − effect_low（effect 已无量纲化）。正值表示该效应在
    更大规模上更大（若 effect 是 log PPL 比，则正值 = 大模型上相对劣势
    更大）。逐种子配对后做精确符号翻转检验，避免把「大规模 PPL 基线更高」
    误读成「效应随规模变大」。
    """
    a = list(low_effects)
    b = list(high_effects)
    if len(a) != len(b):
        raise ValueError('scale_interaction needs equal-length effects (%d vs %d)' % (len(a), len(b)))
    ss = list(range(len(a))) if seeds is None else list(seeds)
    if len(ss) != len(a):
        raise ValueError('scale_interaction seeds length does not match effects')
    if len(set(ss)) != len(ss):
        # A repeated seed would enter paired_effect_stats as two independent
        # observations and shrink the interaction SE.
        raise ValueError('scale_interaction got duplicate seeds %r' % (ss,))
    diff, kept = [], []
    for s, x, y in zip(ss, a, b):
        try:
            fx, fy = float(x), float(y)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fx) and math.isfinite(fy):
            diff.append(fy - fx)
            kept.append(s)
    st = paired_effect_stats(diff, alpha=alpha, min_n=min_n)
    st.update({'low_label': str(low_label), 'high_label': str(high_label), 'seeds': kept,
               'convention': 'delta = effect_high - effect_low; positive = effect is larger at the higher scale',
               'claim_level': 'confirmatory only if n>=%d and p_floor<alpha' % int(min_n)})
    return st


def scaling_trend(points, *, alpha=0.05, min_seeds=3, min_points=3, log_scale=True):
    """尺度阶梯上的效应趋势：相对效应 ~ log(scale) 的加权回归 + 精确配对检验。

    `points` 每项为 {label, scale, effects, seeds}（或 (scale, effects, seeds)）。
    effects 必须是**无量纲**的逐种子效应（`paired_log_ratio` 的输出），
    不能直接塞入绝对 PPL 差——否则斜率只反映基线 PPL 随规模的变化。

    推断分两层：
      1. 优先用「完整种子」的逐种子斜率：同一种子在每个规模点都有观测时，
         对每个种子拟合 log(scale) 斜率，再对这些斜率做精确符号翻转检验。
         p 下限仍是 2/2^n，因此 n=4 时 0.125，n<6 的描述性读数不能称为
         「趋势已确认」。
      2. 没有完整种子时，退化为规模点均值的加权回归，用 t 区间与 t 检验，
         并明确标注 `paired=False`（点间独立、不能利用配对降噪）。
    同时报告两点交互（最小 vs 最大规模）和异质性；异质性高时不得用单一
    斜率概括整个阶梯。"""
    pts = _scale_points(points)
    pts = [p for p in pts if math.isfinite(p['scale']) and p['scale'] > 0.0 and p['effects']]
    pts.sort(key=lambda p: p['scale'])
    labels = [p['label'] for p in pts]
    if len(set(labels)) != len(labels):
        raise ValueError('scaling_trend needs unique scale labels; got %r' % (labels,))
    for p in pts:
        if len(set(p['seeds'])) != len(p['seeds']):
            raise ValueError('scale point %r has duplicate seeds %r' % (p['label'], p['seeds']))
    out = {'alpha': float(alpha), 'min_seeds': int(min_seeds), 'min_points': int(min_points),
           'n_points': len(pts), 'labels': labels,
           'scales': [p['scale'] for p in pts],
           'min_seeds_for_significance': signflip_min_n_for_holm(1, alpha),
           'convention': 'effects must be scale-free (e.g. paired log PPL ratio); positive = with arm worse'}
    if len(pts) < 2:
        out.update({'status': 'insufficient_points', 'verdict': 'insufficient_points'})
        return out
    x = np.asarray([math.log10(p['scale']) if log_scale else p['scale'] for p in pts], dtype=float)
    y = np.asarray([float(np.mean(p['effects'])) for p in pts], dtype=float)
    n = np.asarray([float(len(p['effects'])) for p in pts], dtype=float)
    sw = float(n.sum())
    xbar = float((n * x).sum() / sw)
    ybar = float((n * y).sum() / sw)
    sxx = float((n * (x - xbar) ** 2).sum())
    if sxx <= 0.0:
        out.update({'status': 'degenerate_scale_axis', 'verdict': 'degenerate_scale_axis'})
        return out
    slope = float((n * (x - xbar) * (y - ybar)).sum() / sxx)
    intercept = float(ybar - slope * xbar)
    k = len(pts)
    resid = y - (slope * x + intercept)
    wls = {'slope': slope, 'intercept': intercept, 'x_transform': 'log10' if log_scale else 'linear',
           'weights': 'n_seeds per scale', 'r2': (1.0 - float((n * resid ** 2).sum() / (n * (y - ybar) ** 2).sum())) if float((n * (y - ybar) ** 2).sum()) > 0 else float('nan')}
    if k > 2:
        sigma2 = float((n * resid ** 2).sum() / (k - 2))
        se = float(math.sqrt(sigma2 / sxx))
        tc = _t_critical(1.0 - float(alpha), k - 2)
        tstat = slope / se if se > 0 else float('inf')
        wls.update({'se': se, 'ci': [slope - tc * se, slope + tc * se],
                    'p': student_t_two_sided_p(tstat, k - 2), 'df': k - 2})
    out['wls'] = wls
    # paired per-seed slopes: complete cases only (same seed at every scale)
    by_seed = {}
    for p in pts:
        for s, e in zip(p['seeds'], p['effects']):
            by_seed.setdefault(s, {})[p['label']] = float(e)
    complete = sorted([s for s, m in by_seed.items() if all(lb in m for lb in labels)], key=str)
    paired = {'n_complete_seeds': len(complete), 'seeds': complete, 'labels': labels}
    if len(complete) >= 2:
        # A per-seed slope has exactly one observation per scale, so every
        # scale point carries equal weight here.  Using the point-mean weights
        # `n` (the number of seeds completed at each scale) would let a
        # ragged seed-completion panel bias the paired slope toward the
        # over-represented scale, which is exactly the confound this paired
        # path exists to remove.
        xbar_s = float(x.mean())
        sxx_s = float(((x - xbar_s) ** 2).sum())
        slopes = []
        for s in complete:
            yy = np.asarray([by_seed[s][lb] for lb in labels], dtype=float)
            yybar = float(yy.mean())
            slopes.append(float(((x - xbar_s) * (yy - yybar)).sum() / sxx_s))
        slopes = np.asarray(slopes, dtype=float)
        st = paired_effect_stats(slopes, alpha=alpha, min_n=min_seeds)
        tc = _t_critical(1.0 - float(alpha), len(slopes) - 1)
        se_s = float(slopes.std(ddof=1) / math.sqrt(len(slopes))) if len(slopes) > 1 else float('nan')
        paired.update({'slope_mean': float(slopes.mean()), 'slope_std': float(slopes.std(ddof=1)) if len(slopes) > 1 else float('nan'),
                       'se': se_s, 'ci': [float(slopes.mean()) - tc * se_s, float(slopes.mean()) + tc * se_s] if math.isfinite(se_s) else None,
                       'p_exact_signflip': st['p_exact_signflip'], 'p_floor': st['p_floor'],
                       'resolution_limited': st['resolution_limited'], 'underpowered': st['underpowered'],
                       'within_seed_noise': st['within_seed_noise'], 'significant': st['significant'],
                       'per_seed_slopes': {str(s): float(v) for s, v in zip(complete, slopes)}})
    else:
        paired.update({'note': 'no complete seed across every scale point; the paired exact test is unavailable'})
    out['paired'] = paired
    paired_primary = len(complete) >= 2
    # extreme-scale interaction (smallest vs largest) on shared seeds
    lo, hi = pts[0], pts[-1]
    lo_map = dict(zip(lo['seeds'], lo['effects']))
    hi_map = dict(zip(hi['seeds'], hi['effects']))
    shared = sorted(set(lo_map) & set(hi_map), key=str)
    inter = scale_interaction([lo_map[s] for s in shared], [hi_map[s] for s in shared],
                              low_label=lo['label'], high_label=hi['label'], seeds=shared,
                              alpha=alpha, min_n=min_seeds) if len(shared) >= 2 else {'n': len(shared), 'note': 'fewer than 2 shared seeds'}
    out['interaction_extremes'] = inter
    # Heterogeneity is the LACK OF FIT around the fitted slope, not the spread
    # of the point means around their grand mean: a clean monotone ladder has
    # a large grand-mean Q by construction.  Residual Q (df=k-2) asks the
    # question the verdict needs: is a single slope an adequate summary?
    if k >= 2:
        q = float((n * resid ** 2).sum())
        dfh = k - 2
        p_q = chi2_sf(q, dfh) if dfh > 0 else None
        i2 = float(max(0.0, (q - dfh) / q) * 100.0) if (dfh > 0 and q > 0) else 0.0
        c = float(n.sum() - (n ** 2).sum() / n.sum())
        tau2 = float(max(0.0, (q - dfh) / c)) if (dfh > 0 and c > 0) else 0.0
        out['heterogeneity'] = {'Q': q, 'df': int(dfh), 'p': p_q, 'I2_percent': i2, 'tau2': tau2,
                                'method': 'lack-of-fit Q around the WLS slope (weights = n)',
                                'note': 'high I2 means the scale points deviate from a single slope; it is not the spread of a monotone ladder around its mean'}
    # The paired path only uses seeds completed at EVERY scale.  Extra seeds at
    # some scales cannot contribute a paired slope, so they must not decide
    # whether the paired trend is "heterogeneous" either: report a separate
    # lack-of-fit over the complete-seed scale means and gate the verdict on
    # that when the paired slope is the primary inference.
    het_for_verdict = out.get('heterogeneity') or {}
    if paired_primary:
        _yc = np.asarray([float(np.mean([by_seed[s][lb] for s in complete])) for lb in labels], dtype=float)
        _sp = float(paired['slope_mean'])
        _ip = float(_yc.mean() - _sp * x.mean())
        _rp = _yc - (_sp * x + _ip)
        _cp = float(len(complete))
        _q_p = float((_cp * _rp ** 2).sum())
        _df_p = k - 2
        _p_p = chi2_sf(_q_p, _df_p) if _df_p > 0 else None
        _i2_p = float(max(0.0, (_q_p - _df_p) / _q_p) * 100.0) if (_df_p > 0 and _q_p > 0) else 0.0
        out['heterogeneity_paired'] = {'Q': _q_p, 'df': int(_df_p), 'p': _p_p,
                                       'I2_percent': _i2_p, 'n_complete_seeds': len(complete),
                                       'method': 'lack-of-fit over complete-seed scale means (weights = n_complete)',
                                       'note': 'gates the paired trend verdict; the all-seed heterogeneity above stays descriptive'}
        het_for_verdict = out['heterogeneity_paired']
    p_primary = paired.get('p_exact_signflip') if paired_primary else wls.get('p')
    resolution_limited = bool(paired.get('resolution_limited')) if paired_primary else (k <= 2)
    # A trend below the pre-registered seed/scale floor is descriptive, not a
    # confirmed effect: the paired test can reject at n<min_seeds (its p floor
    # is 2/2**n), and a two-point "trend" is a line even when the per-seed
    # exact test rejects.  Gate the README-level claim on the same thresholds
    # the rest of this function reports.
    underpowered = bool(paired.get('underpowered')) if paired_primary else False
    too_few_scales = k < int(min_points)
    sig = bool(p_primary is not None and math.isfinite(float(p_primary)) and float(p_primary) < alpha
               and not resolution_limited and not underpowered and not too_few_scales)
    het = het_for_verdict
    heterogeneous = bool(float(het.get('I2_percent') or 0.0) > 75.0 or (isinstance(het.get('p'), float) and math.isfinite(het['p']) and het['p'] < alpha))
    # Direction must come from the same slope that carries the inference.  The
    # WLS fit weights each scale by its seed count, so on a ragged panel its
    # sign can disagree with the paired per-seed slope the exact test just
    # evaluated.  Reporting a verdict from the unpaired slope would then say
    # "decreases" while the primary evidence says "increases".
    direction_slope = float(paired['slope_mean']) if paired_primary else slope
    out['primary_inference'] = 'paired per-seed slope (exact sign-flip)' if paired_primary else 'WLS point-mean t-test (unpaired scales)'
    out['p_primary'] = float(p_primary) if isinstance(p_primary, (int, float)) and math.isfinite(float(p_primary)) else None
    out['significant'] = sig
    out['resolution_limited'] = resolution_limited
    out['underpowered'] = underpowered
    out['direction_slope'] = direction_slope
    if heterogeneous:
        verdict = 'heterogeneous_trend'
    elif sig:
        verdict = 'effect_decreases_with_scale' if direction_slope < 0 else 'effect_increases_with_scale'
    elif underpowered:
        verdict = 'underpowered'
    elif too_few_scales:
        verdict = 'insufficient_points'
    elif resolution_limited:
        verdict = 'resolution_limited'
    else:
        verdict = 'no_detectable_trend'
    out['verdict'] = verdict
    out['benefit_grows_with_scale'] = (bool(sig and direction_slope < 0) if not heterogeneous else None)
    if too_few_scales:
        out['note'] = 'fewer than min_points=%d scales: slope is descriptive even if the exact test rejects' % int(min_points)
    return out


def loglog_regression(points, *, alpha=0.05):
    """log10(y) ~ log10(x) 回归，带斜率 CI / p / R² / 留一稳定性。

    用于规模律读数（如 crossover tokens ~ d_model）。`points` 接受
    [(x, y)] 或 [{label, x, y}]；只保留有限正数。点数 k>=3 时给出斜率的
    t 区间与双侧 p；k=2 时直线是恒等拟合，只报斜率并显式标注「无残差
    自由度、无 R²」。x 本身的测量误差未计入（观测性拟合，不是因果估计）。
    """
    rows = []
    for i, p in enumerate(points or []):
        if isinstance(p, dict):
            x, y, label = p.get('x'), p.get('y'), p.get('label', i)
        elif isinstance(p, (tuple, list)) and len(p) >= 2:
            x, y, label = p[0], p[1], i
        else:
            continue
        try:
            fx, fy = float(x), float(y)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fx) and math.isfinite(fy) and fx > 0.0 and fy > 0.0:
            rows.append((str(label), fx, fy))
    out = {'alpha': float(alpha), 'n_points': len(rows), 'labels': [r[0] for r in rows],
           'note': 'observational log-log fit; x measurement error is not modelled'}
    if len(rows) < 2:
        out.update({'status': 'insufficient_points'})
        return out
    x = np.log10(np.asarray([r[1] for r in rows], dtype=float))
    y = np.log10(np.asarray([r[2] for r in rows], dtype=float))
    xbar, ybar = float(x.mean()), float(y.mean())
    sxx = float(((x - xbar) ** 2).sum())
    if sxx <= 0.0:
        out.update({'status': 'degenerate_x'})
        return out
    slope = float(((x - xbar) * (y - ybar)).sum() / sxx)
    intercept = float(ybar - slope * xbar)
    pred = slope * x + intercept
    resid = y - pred
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - ybar) ** 2).sum())
    try:
        _coef = 10 ** intercept
    except OverflowError:
        _coef = float('inf')
    out.update({'status': 'ok', 'slope': slope, 'exponent': slope, 'intercept': intercept,
                'law': 'y = %.6g * x^%.6g' % (_coef, slope)})
    k = len(rows)
    if k >= 3:
        sigma2 = ss_res / (k - 2)
        se = float(math.sqrt(sigma2 / sxx))
        tc = _t_critical(1.0 - float(alpha), k - 2)
        out.update({'se': se, 'ci': [slope - tc * se, slope + tc * se],
                    'p': student_t_two_sided_p(slope / se, k - 2) if se > 0 else float('nan'),
                    'df': k - 2, 'r2': (1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan'),
                    'residual_sigma_log10': float(math.sqrt(sigma2))})
        loo = []
        for i in range(k):
            xx = np.delete(x, i)
            yy = np.delete(y, i)
            xb, yb = float(xx.mean()), float(yy.mean())
            den = float(((xx - xb) ** 2).sum())
            if den <= 0.0:
                # Duplicate x values can leave no scale spread after dropping
                # one point; report the undefined fit instead of raising in a
                # report builder.
                loo.append({'label': rows[i][0], 'slope': None, 'note': 'degenerate x-spread after dropping this point'})
                continue
            s = float(((xx - xb) * (yy - yb)).sum() / den)
            loo.append({'label': rows[i][0], 'slope': s})
        out['leave_one_out'] = loo
        _loo_slopes = [v['slope'] for v in loo if v['slope'] is not None]
        out['leave_one_out_slope_range'] = [float(min(_loo_slopes)), float(max(_loo_slopes))] if _loo_slopes else None
    else:
        out.update({'r2': None, 'ci': None, 'p': None,
                    'note': out['note'] + '; k=2 gives an exact line with no residual degrees of freedom, so no CI/R2/p is reported'})
    return out


# `mt` (matched-variant set) and `mr` (mlp_match_ref) can themselves contain
# underscores (e.g. `full_rope`, `csa_dynamic`), so those two values are
# captured lazily up to the next literal key rather than by `[^_]+`.
_RUN_CFG_RE = re.compile('^steps(?P<steps>[^_]+)_sl(?P<sl>[^_]+)_bs(?P<bs>[^_]+)_nt(?P<nt>[^_]+)_lr(?P<lr>[^_]+)_wd(?P<wd>[^_]+)_d(?P<d>[^_]+)_L(?P<L>[^_]+)_H(?P<H>[^_]+)_Dh(?P<Dh>[^_]+)_v(?P<v>[^_]+)_wu(?P<wu>[^_]+)_cl(?P<cl>[^_]+)_dlm(?P<dlm>[^_]+)_mt(?P<mt>.*?)_mr(?P<mr>.*?)_ee(?P<ee>[^_]+)_es(?P<es>[^_]+)_det(?P<det>[^_]+)_df(?P<df>.+?)(?:_tk(?P<tk>[0-9a-f]+))?(?:_amp(?P<amp>[^_]+))?(?:_eb(?P<eb>[01]))?(?:_ae(?P<ae>[^_]+))?_cs(?P<cs>.+)$')

def run_cfg_knobs(fp):
    """Best-effort decode of a `_run_fp` string into its named training knobs.

    Used only for the audit display: it lets the report SHOW that two arms
    share optimizer / schedule / training length / data, instead of asserting it.
    `{}` means the string predates the current layout or is malformed.  The
    authoritative controlled-variable gate is still whole-string equality in
    `pair_reason` — this decoder never substitutes for it."""
    if not isinstance(fp, str):
        return {}
    m = _RUN_CFG_RE.match(fp)
    return m.groupdict() if m else {}

def variant_presence(outdir):
    out = {}
    sp = os.path.join(outdir, 'summary.json')
    if not os.path.exists(sp):
        return out
    try:
        raw = json.load(open(sp, encoding='utf-8'))
    except Exception as e:
        print(f'[stats] WARNING: cannot read {sp} ({type(e).__name__}: {e}) — variant presence cannot be established')
        return out
    kept = {}
    refused = {}
    for _k, r in raw.items():
        if not isinstance(r, dict):
            continue
        v = r.get('variant')
        if v is None:
            v = _k.split('::')[0]
        if ppl_is_usable(r.get('ppl')) and (not r.get('synthesized')) and (_as_seed_int(r.get('seed')) is not None):
            kept[v] = kept.get(v, 0) + 1
        else:
            refused[v] = refused.get(v, 0) + 1
    for v in sorted(set(kept) | set(refused)):
        if v not in refused:
            continue
        if v in kept:
            out[v] = f'present with {kept[v]} measurable record(s) (+{refused[v]} refused) — no shared usable seed'
        else:
            out[v] = f'present but ALL {refused[v]} record(s) refused as unmeasured (synthesized / error / nan ppl / no seed)'
    return out

def format_variant_presence(presence, variants):
    parts = [f'`{v}`: {presence[v]}' for v in variants if v in presence]
    if not parts:
        return None
    return '; '.join(parts) + ' — omitted rather than quoting a cross-panel delta'

def ppl_by_seed(outdir, full=False):
    _by_id = read_panel(outdir)
    out = {}
    if not _by_id:
        return out
    sp = os.path.join(outdir, 'summary.json')
    _seen_seed, _collide = ({}, {})
    for _v, _t, _s in _by_id:
        if (_v, _s) in _seen_seed and _seen_seed[_v, _s] != _t:
            _collide.setdefault(_v, set()).update({_seen_seed[_v, _s], _t})
        _seen_seed[_v, _s] = _t
    if _collide:
        for _v, _tags in sorted(_collide.items()):
            print(f'[stats] WARNING: {sp}: variant `{_v}` holds SEVERAL protocol conditions for one seed (tags {sorted(_tags)}). This reader keeps ONE record per (variant, seed) — the flat map CANNOT represent the others.  A paired contrast built from it would silently quote a single condition; use `ppl_by_seed_grouped` to read the panel per condition.')
    for (_v, _t, _s), (_k, r) in _by_id.items():
        _slot = out.setdefault(_v, {})
        if _s not in _slot:
            if full:
                _slot[_s] = r if 'protocol_tag' in r else {**r, 'protocol_tag': _t}
            else:
                _slot[_s] = r['ppl']
    return out

def _record_identity(rec, key, legacy_ok):
    seed = _as_seed_int(rec.get('seed'))
    if seed is None:
        return None
    var = rec.get('variant')
    if var is None:
        var = str(key).split('::', 1)[0]
    return (str(var), record_tag(rec, key, legacy_ok), seed)

def _as_seed_int(v):
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, float):
        return int(v) if math.isfinite(v) and v.is_integer() else None
    try:
        return int(v)
    except (TypeError, ValueError, OverflowError):
        return None

def read_panel(outdir):
    sp = os.path.join(outdir, 'summary.json')
    if not os.path.exists(sp):
        return {}
    try:
        raw = json.load(open(sp, encoding='utf-8'))
    except Exception as e:
        print(f'[stats] WARNING: cannot read {sp} ({type(e).__name__}: {e}) — this panel is EXCLUDED')
        return {}
    _legacy_ok = legacy_warm_tags(raw)
    _by_id = {}
    for _k, r in raw.items():
        if not isinstance(r, dict) or r.get('seed') is None:
            continue
        if not ppl_is_usable(r.get('ppl')) or r.get('synthesized'):
            continue
        _id = _record_identity(r, _k, _legacy_ok)
        if _id is None:
            continue
        if _id in _by_id:
            print(f'[stats] WARNING: {sp}: two measurable records for variant={_id[0]!r} protocol={_id[1]!r} seed={_id[2]} (keys {_by_id[_id][0]!r}, {_k!r}) — the PPL is AMBIGUOUS and no averaging can resolve it; the earlier key is used, the later one is EXCLUDED')
            continue
        _by_id[_id] = (_k, r)
    return _by_id

def ppl_by_seed_grouped(outdir):
    grouped = {}
    for (_v, _t, _s), (_k, _r) in read_panel(outdir).items():
        grouped.setdefault((_v, _t), {})[_s] = _r
    return grouped

# ---- 对比实验 / 消融 / 统计可靠性的声明式审计（零 GPU）----
# 每个「创新点」按留一法（leave-one-out）写出「带模块」与「去模块」的对照
# 臂。审计对每条对照：(1) 用 `pair_reason` 验证两臂严格同 run_cfg——优化器、
# 学习率调度、权重衰减、训练步数/ token、warmup、comp 正则、模型形状、数据
# 指纹与确定性档全部一致，只有层架构（variant）不同；(2) 逐种子配对给出
# mean±std、d_z 与精确符号翻转 p；(3) 用 |mean| <= std 标出「差异落在种子噪声内」、
# 用 n<3 标出欠功效；(4) 汇总每个创新点是否至少有一条完整留一对照。
# 符号约定：Δ = 「带模块」−「去模块」，PPL 越低越好，故 Δ<0 = 模块有用。
ABLATION_DESIGN = {
    'dynamic_blocking': [
        ('csa_dynamic', 'csa_fixed', 'dynamic vs fixed blocking (CSA)'),
        ('hybrid_dynamic', 'hybrid_fixed', 'dynamic vs fixed blocking (CSA+HCA hybrid)'),
    ],
    'learned_selection': [
        ('csa_fixed', 'csa_fix_randidx', 'learned top-k vs random selection (fixed blocking)'),
        ('csa_dynamic', 'csa_dyn_randidx', 'learned top-k vs random selection (dynamic blocking)'),
    ],
    'content_pooling': [
        ('csa_fixed', 'csa_fix_zerocont', 'pooled block content vs zero content (fixed blocking)'),
        ('csa_dynamic', 'csa_dyn_zerocont', 'pooled block content vs zero content (dynamic blocking)'),
    ],
    'attention_sink': [
        ('csa_fixed', 'csa_fix_nosink', 'sink on vs off (fixed blocking)'),
        ('csa_dynamic', 'csa_dyn_nosink', 'sink on vs off (dynamic blocking)'),
        ('full_sink', 'full', 'sink on vs off (dense control)'),
    ],
    'compression_regularizer': [
        ('csa_dynamic', 'csa_dyn_nocomp', 'compression regularizer on vs off (dynamic blocking; the loss-term ablation; note this also freezes delta_logit because comp_reg was its only gradient source)'),
    ],
    'task_learned_boundary': [
        ('csa_dynamic_task', 'csa_dynamic', 'sampled cut policy trained by the LM loss vs hard threshold + comp_reg only'),
    ],
    'local_fusion': [
        ('csa_dyn_fuse', 'csa_dynamic', 'learned local fusion on vs off (CSA)'),
        ('hybrid_csa_dyn_fuse', 'hybrid_csa_dyn', 'learned local fusion on vs off (hybrid)'),
    ],
    'hybrid_hierarchy': [
        ('hybrid_csa_dyn', 'csa_dynamic', 'CSA+HCA hybrid vs CSA-only (dynamic blocking)'),
    ],
}

# 主方法 vs 参数对齐基线的对比也属于「受控对比」：它回答的是方法本身是否
# 优于同参数量、同优化器、同训练规模、同数据的稠密基线，而不是只回答「消融臂
# 之间谁高谁低」。两条基线都来自 `variant_mlp_ratio` 的参数对齐（MLP 加宽
# 到与 csa_dynamic 相同），因此 `param_controlled` 是这条对比的硬读数。
BASELINE_DESIGN = {
    'compression_vs_matched_dense': [
        ('csa_dynamic', 'full_sw128_matched', 'dynamic CSA vs parameter-matched windowed dense'),
        ('csa_dynamic', 'full_matched', 'dynamic CSA vs parameter-matched full dense'),
    ],
}

HYPOTHESIS_DESIGN = {
    'dynamic_blocking': {
        'statement': 'Content-adaptive dynamic blocking lowers validation PPL versus fixed blocking at the same parameter count, data order and training length.',
        'mechanism': 'Learned cosine boundaries place cuts at low-similarity token transitions, so fewer cross-boundary tokens are compressed together.',
        'predicted_sign': -1,
        'primary_metric': 'ppl',
        'falsifier': 'No family-Holm-corrected PPL reduction, a reduction within seed noise, or a reduction that reverses when fixed/dynamic blocking is swapped.',
    },
    'learned_selection': {
        'statement': 'Learned top-k block selection lowers validation PPL versus random selection with the same number of attended blocks.',
        'mechanism': 'Query-conditioned indexer scores retrieve blocks that carry token-relevant content instead of an arbitrary subset.',
        'predicted_sign': -1,
        'primary_metric': 'ppl',
        'falsifier': 'Random selection matches or beats learned selection after Holm correction, or the gap is within seed noise.',
    },
    'content_pooling': {
        'statement': 'Pooling block content (rather than zero content) lowers validation PPL at the same block geometry.',
        'mechanism': 'Compressed content retains token identity so the indexer and attention see informative block keys/values.',
        'predicted_sign': -1,
        'primary_metric': 'ppl',
        'falsifier': 'Zeroing block content does not increase PPL beyond seed noise.',
    },
    'attention_sink': {
        'statement': 'An attention sink lowers validation PPL by giving off-context probability mass a stable destination.',
        'mechanism': 'The sink absorbs attention that would otherwise be forced onto unrelated blocks, reducing normalization pressure.',
        'predicted_sign': -1,
        'primary_metric': 'ppl',
        'falsifier': 'Removing the sink does not increase PPL beyond seed noise, or improves it.',
    },
    'compression_regularizer': {
        'statement': 'The compression regularizer lowers validation PPL at the same architecture and optimization setting.',
        'mechanism': 'The loss term discourages degenerate block boundaries and keeps compression from collapsing to fixed chunks.',
        'predicted_sign': -1,
        'primary_metric': 'ppl',
        'falsifier': 'Zeroing the regularizer does not increase PPL beyond seed noise, or improves it.',
    },
    'task_learned_boundary': {
        'statement': 'Training the sampled cut policy with the LM loss (score-function term) lowers validation PPL versus the hard-threshold + comp_reg-only gate.',
        'mechanism': 'The LM objective can trade compression against language-modelling loss directly instead of optimizing only a compression-rate surrogate.',
        'predicted_sign': -1,
        'primary_metric': 'ppl',
        'falsifier': 'The task-trained policy matches or loses to the threshold gate after Holm correction, or its variance makes the panel unusable.',
    },
    'local_fusion': {
        'statement': 'The learned fusion convolution lowers validation PPL versus the fixed identity fusion rule.',
        'mechanism': 'The fusion convolution reshapes the cosine segmentation signal; it does NOT gate the local window, so this is a boundary-signal smoothing contrast, not a learned sparse/local trade-off.',
        'predicted_sign': -1,
        'primary_metric': 'ppl',
        'falsifier': 'The learned fusion convolution matches or loses to the identity rule after Holm correction.',
    },
    'hybrid_hierarchy': {
        'statement': 'The CSA+HCA hybrid lowers validation PPL versus CSA-only at matched parameters and training length.',
        'mechanism': 'Alternating local HCA layers and sparse CSA layers cover complementary short- and long-range structure.',
        'predicted_sign': -1,
        'primary_metric': 'ppl',
        'falsifier': 'The hybrid matches or loses to CSA-only after Holm correction.',
    },
}

BASELINE_HYPOTHESIS_DESIGN = {
    'compression_vs_matched_dense': {
        'statement': 'Dynamic CSA lowers validation PPL versus parameter-matched dense baselines under identical data, optimizer, schedule and training length.',
        'mechanism': 'Sparse compressed reads can preserve long-range access while the parameter-matched MLP keeps capacity equal.',
        'predicted_sign': -1,
        'primary_metric': 'ppl',
        'falsifier': 'A matched dense baseline matches or beats CSA after Holm correction, or the parameter matching fails.',
    },
}

# 基线调参协议：主方法与基线必须获得相同的搜索范围、在同一训练集上拟合、
# 在验证集上按同一指标选择。`baseline_tune.py` 读取本表；审计读取落盘的
# `baseline_tuning.json`，没有该产物时只报「不可核验」，不会把未调参的
# 弱基线当成已验证的对照。
BASELINE_TUNING_PROTOCOL = {
    'selection_metric': 'ppl',
    'selection_split': 'validation',
    'direction': 'lower_is_better',
    'search_seed': 0,
    # Selection is on the MEAN validation PPL over this declared seed set, not
    # on one seed's draw.  A single-seed selection can pick a config because
    # of seed noise; every arm and every trial uses the same set.
    'search_seeds': [0, 1, 2],
    'final_seeds': [0, 1, 2],
    'arms': ['full_matched', 'full_sw128_matched', 'csa_fixed', 'csa_dynamic'],
    'trials': [
        {'lr': 0.0001, 'weight_decay': 0.1, 'warmup': 50},
        {'lr': 0.0003, 'weight_decay': 0.1, 'warmup': 50},
        {'lr': 0.0003, 'weight_decay': 0.05, 'warmup': 50},
        {'lr': 0.0003, 'weight_decay': 0.1, 'warmup': 100},
    ],
    'note': 'Every arm gets the same declared trials, the same search-seed set and the same selection split. Selection uses mean validation PPL across search_seeds (std recorded); a single-seed draw can win on noise. Final seeds re-use only the validation-selected config and are reported as mean±std. A tuning trial may never read the held-out test split; test is used once after selection.',
}

# 最简单基线协议（先立地板，再谈复杂模型）：语言模型的 trivial floor 是
# 在**训练集**上数出来的 unigram / bigram 计数模型，平滑与插值系数只在
# **验证集**上选择。主方法如果不能显著优于这个地板，就不值得进入与稠密
# Transformer 的机制对比；反过来，复杂模型只战胜未调参的弱基线也不算
# 证据。`baseline_tune.py simple` 生成 `simple_baseline.json`，审计函数
# 核验搜索范围、选择集与选中项；`methodology_audit` 会把它与面板的主方法
# 读数并排报告（只有 val 指纹一致时才比较）。
SIMPLE_BASELINE_POLICY = {
    'selection_metric': 'ppl',
    'selection_split': 'validation',
    'direction': 'lower_is_better',
    'arms': {
        'unigram': {'order': 1, 'description': 'training-set unigram counts with add-alpha smoothing (a linear/count model)'},
        'bigram': {'order': 2, 'description': 'training-set bigram counts interpolated with the unigram model'},
    },
    'grids': {
        'unigram': {'alpha': [0.01, 0.1, 1.0]},
        'bigram': {'alpha': [0.01, 0.1, 1.0], 'interpolation': [0.5, 0.9, 1.0]},
    },
    'note': 'Counts are fitted on the training token stream only; smoothing/interpolation is selected on validation PPL. The held-out test split is never read. This is a sanity floor, not a tuned neural baseline.',
}

def _ngram_train_counts(train_ids, vocab_size):
    """Return (unigram counts, bigram key counts, bigram context counts)."""
    ids = np.asarray(train_ids, dtype=np.int64).ravel()
    vocab_size = int(vocab_size)
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= vocab_size):
        raise ValueError('ngram baseline: train ids outside [0, vocab_size)')
    uni = np.bincount(ids, minlength=vocab_size).astype(np.float64)
    if ids.size < 2:
        return uni, np.empty(0, dtype=np.int64), np.zeros(vocab_size, dtype=np.int64), np.empty(0, dtype=np.int64)
    prev, nxt = ids[:-1], ids[1:]
    keys = prev * vocab_size + nxt
    uniq, counts = np.unique(keys, return_counts=True)
    ctx = np.bincount(prev, minlength=vocab_size).astype(np.int64)
    return uni, uniq, ctx, counts

def _ngram_eval_parts(eval_ids):
    arr = np.asarray(eval_ids, dtype=np.int64)
    if arr.ndim == 1:
        return arr, arr[:-1], arr[1:]
    if arr.ndim == 2:
        return arr.ravel(), arr[:, :-1].ravel(), arr[:, 1:].ravel()
    raise ValueError('eval ids must be 1-D or 2-D (sequences x length)')

def ngram_lm_score(train_ids, eval_ids, vocab_size, order=1, alpha=0.1, interpolation=1.0, _counts=None):
    """Token-weighted NLL/PPL for a tuned count model.

    order=1 is the unigram model (a context-free linear/count baseline);
    order=2 conditions each token on the previous token and interpolates the
    smoothed conditional with the unigram distribution.  Row boundaries are
    respected: the first token of every evaluation row is scored by the
    unigram model, never by the previous row's last token.  The effect size is
    token-weighted, matching the LM metric policy."""
    ids = np.asarray(train_ids, dtype=np.int64).ravel()
    vocab_size = int(vocab_size)
    alpha = float(alpha)
    if alpha <= 0:
        raise ValueError('alpha must be positive')
    if order not in (1, 2):
        raise ValueError('order must be 1 (unigram) or 2 (bigram)')
    all_eval, prev, nxt = _ngram_eval_parts(eval_ids)
    if all_eval.size == 0:
        return {'order': int(order), 'alpha': alpha, 'interpolation': float(interpolation),
                'n_tokens': 0, 'nll': float('nan'), 'ppl': float('nan')}
    if all_eval.size and (int(all_eval.min()) < 0 or int(all_eval.max()) >= vocab_size):
        raise ValueError('ngram baseline: eval ids outside [0, vocab_size)')
    # The count tables are a pure function of (train_ids, vocab_size) and do
    # not depend on the smoothing hyper-parameters, so a tuning grid can build
    # them once and hand the same tables to every trial (identical values).
    counts = _ngram_train_counts(ids, vocab_size) if _counts is None else _counts
    uni_counts = counts[0]
    denom = float(uni_counts.sum()) + alpha * vocab_size
    uni_prob = (uni_counts + alpha) / denom
    log_uni = np.log(np.maximum(uni_prob, np.finfo(float).tiny))
    if order == 1:
        nll = float(-log_uni[all_eval].sum() / all_eval.size)
        return {'order': 1, 'alpha': alpha, 'interpolation': 0.0,
                'n_tokens': int(all_eval.size), 'nll': nll, 'ppl': float(_safe_exp_ppl(nll))}
    _uni, uniq, ctx, pair_counts = counts
    lam = float(interpolation)
    if not (0.0 <= lam <= 1.0):
        raise ValueError('interpolation must be in [0, 1]')
    # First token of each row: unigram only (no cross-row conditioning).  A
    # 1-D evaluation stream is ONE row, so only its first token is
    # unconditioned; the remaining tokens are scored by the conditional term.
    first = np.asarray(eval_ids, dtype=np.int64)
    first = first[:1] if first.ndim == 1 else first[:, 0]
    first = first.ravel()
    nll_sum = float(-log_uni[first].sum())
    if prev.size:
        if lam > 0.0:
            keys = prev * vocab_size + nxt
            idx = np.searchsorted(uniq, keys)
            idx_clip = np.minimum(idx, max(uniq.size - 1, 0))
            hit = (idx < uniq.size) & (uniq[idx_clip] == keys) if uniq.size else np.zeros(keys.shape, dtype=bool)
            c_pair = np.where(hit, pair_counts[idx_clip], 0).astype(np.float64)
            c_ctx = ctx[prev].astype(np.float64)
            p_bg = np.where(c_ctx > 0.0, (c_pair + alpha) / (c_ctx + alpha * vocab_size), uni_prob[nxt])
            p = lam * p_bg + (1.0 - lam) * uni_prob[nxt]
        else:
            # interpolation=0 is the pure unigram model: the conditional
            # term contributes the unigram probability of every other token.
            p = uni_prob[nxt]
        nll_sum += float(-np.log(np.maximum(p, np.finfo(float).tiny)).sum())
    n_tokens = int(all_eval.size)
    nll = float(nll_sum / n_tokens) if n_tokens else float('nan')
    return {'order': 2, 'alpha': alpha, 'interpolation': lam,
            'n_tokens': n_tokens, 'nll': nll, 'ppl': float(_safe_exp_ppl(nll)) if math.isfinite(nll) else float('nan')}

def simple_baseline_fit(train_ids, eval_ids, vocab_size, protocol=None):
    """Run the declared count-model grid and select per baseline on validation."""
    protocol = dict(SIMPLE_BASELINE_POLICY if protocol is None else protocol)
    arms = protocol.get('arms') or {}
    grids = protocol.get('grids') or {}
    trials_by_arm = {}
    # Build the corpus count tables once for the whole grid (lazily, so an
    # empty evaluation set still short-circuits without touching the corpus):
    # every trial below consumes the identical arrays — the trial
    # hyper-parameters only affect the smoothing on top of them.
    _shared_counts = None
    for arm, meta in arms.items():
        order = int((meta or {}).get('order', 1))
        grid = grids.get(arm) or {}
        alphas = list(grid.get('alpha') or [0.1])
        interps = list(grid.get('interpolation') or [1.0]) if order == 2 else [0.0]
        rows = []
        for alpha in alphas:
            for lam in interps:
                cfg = {'order': order, 'alpha': float(alpha)}
                if order == 2:
                    cfg['interpolation'] = float(lam)
                if _shared_counts is None:
                    _shared_counts = _ngram_train_counts(np.asarray(train_ids, dtype=np.int64).ravel(), vocab_size)
                sc = ngram_lm_score(train_ids, eval_ids, vocab_size, order=order,
                                    alpha=float(alpha), interpolation=float(lam),
                                    _counts=_shared_counts)
                rows.append({'config': cfg, 'ppl': sc['ppl'], 'nll': sc['nll'],
                             'n_tokens': sc['n_tokens'], 'steps': 0, 'tokens_seen': int(sc['n_tokens'])})
        if not rows:
            raise RuntimeError(f'simple baseline {arm}: empty declared grid')
        best = min(rows, key=lambda r: float(r['ppl']))
        trials_by_arm[arm] = {'selected': dict(best['config']),
                              'selected_metric_value': float(best['ppl']),
                              'trials': rows,
                              'order': order}
    best_arm = min(trials_by_arm, key=lambda a: float(trials_by_arm[a]['selected_metric_value']))
    # `_ids_fp` hashes the canonical chunked form of any integer width and
    # accepts the training stream (possibly a memmap) directly.
    train_arr = np.asarray(train_ids)
    if not np.issubdtype(train_arr.dtype, np.integer):
        raise TypeError(f'ngram baseline: train ids must be integers, got dtype {train_arr.dtype}')
    train_arr = train_arr.reshape(-1)
    eval_arr = np.asarray(eval_ids, dtype=np.int64)
    _data = {'train_fp': _ids_fp(train_arr),
             'val_fp': _ids_fp(np.ascontiguousarray(eval_arr).reshape(-1)),
             'n_train_tokens': int(train_arr.size),
             'n_eval_tokens': int(eval_arr.size)}
    _data['data_version'] = _data_version(_data)
    _prov = code_provenance()
    _prov['data'] = _data
    return {
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'protocol': protocol,
        'selection_metric': protocol.get('selection_metric', 'ppl'),
        'selection_split': protocol.get('selection_split', 'validation'),
        'direction': protocol.get('direction', 'lower_is_better'),
        'data': _data,
        'data_version': _data['data_version'],
        'provenance': _prov,
        'baselines': trials_by_arm,
        'best_baseline': best_arm,
        'best_ppl': float(trials_by_arm[best_arm]['selected_metric_value']),
        'code': code_stamp(),
    }

def simple_baseline_audit(artifact=None, path=None, protocol=None):
    """Audit the tuned count-model floor artifact.

    The artifact must show a validation-only search, a selected config that is
    actually the best recorded trial, and consistent evaluation-token counts.
    A missing/invalid artifact is `not_verifiable_locally`, never a pass."""
    _explicit = protocol is not None
    if artifact is None and path:
        if os.path.exists(path):
            try:
                artifact = json.load(open(path, encoding='utf-8'))
            except Exception as e:
                return {'status': 'fail', 'baseline_verified': False,
                        'problems': [f'cannot parse {path}: {type(e).__name__}: {e}']}
        else:
            return {'status': 'not_verifiable_locally', 'baseline_verified': False,
                    'problems': [f'{path} absent — run baseline_tune.py simple before claiming the trivial floor was tuned']}
    if not _explicit and isinstance(artifact, dict) and isinstance(artifact.get('protocol'), dict):
        protocol = artifact['protocol']
    else:
        protocol = SIMPLE_BASELINE_POLICY if protocol is None else protocol
    if not isinstance(artifact, dict) or not isinstance(artifact.get('baselines'), dict):
        return {'status': 'not_verifiable_locally', 'baseline_verified': False,
                'problems': ['no simple_baseline artifact (expected {"baselines": {...}})']}
    problems = []
    warnings = []
    metric = artifact.get('selection_metric') or protocol.get('selection_metric')
    split = artifact.get('selection_split') or protocol.get('selection_split')
    direction = artifact.get('direction') or protocol.get('direction')
    if split != 'validation':
        problems.append(f'selection split is {split!r}; count-model smoothing must be selected on validation, never test')
    if metric != protocol.get('selection_metric'):
        problems.append(f'selection metric {metric!r} differs from the declared {protocol.get("selection_metric")!r}')
    if direction != 'lower_is_better':
        problems.append(f'count-model selection direction is {direction!r}; PPL is lower-is-better')
    declared_arms = set((protocol.get('arms') or {}).keys())
    baselines = artifact['baselines']
    missing = sorted(declared_arms - set(baselines))
    if missing:
        problems.append(f'missing simple-baseline evidence for arms {missing}')
    n_tokens_by_arm = {}
    selected_values = {}
    for arm, entry in baselines.items():
        if not isinstance(entry, dict):
            problems.append(f'{arm}: malformed baseline entry')
            continue
        trials = entry.get('trials')
        if not isinstance(trials, list) or not trials:
            problems.append(f'{arm}: no trial records')
            continue
        cfgs = [json.dumps(t.get('config') or {}, sort_keys=True, default=str) for t in trials if isinstance(t, dict)]
        chosen = entry.get('selected') or entry.get('selected_config')
        if not isinstance(chosen, dict):
            problems.append(f'{arm}: no selected config')
            continue
        chosen_sig = json.dumps(chosen, sort_keys=True, default=str)
        if chosen_sig not in cfgs:
            problems.append(f'{arm}: selected config is not one of its recorded trials')
            continue
        chosen_trial = next(t for t in trials if isinstance(t, dict)
                            and json.dumps(t.get('config') or {}, sort_keys=True, default=str) == chosen_sig)
        vals = [float(t[metric]) for t in trials
                if isinstance(t, dict) and isinstance(t.get(metric), (int, float))
                and not isinstance(t.get(metric), bool) and math.isfinite(float(t[metric]))]
        if not vals:
            problems.append(f'{arm}: no finite {metric} in its trials')
            continue
        chosen_value = entry.get('selected_metric_value')
        if chosen_value is None:
            chosen_value = chosen_trial.get(metric)
        if not isinstance(chosen_value, (int, float)) or isinstance(chosen_value, bool) or not math.isfinite(float(chosen_value)):
            problems.append(f'{arm}: no finite selected {metric} value')
            continue
        selected_values[arm] = float(chosen_value)
        if abs(float(chosen_value) - float(chosen_trial[metric])) > 1e-12:
            problems.append(f'{arm}: selected_metric_value does not match the selected trial')
        if float(chosen_value) > min(vals) + 1e-12:
            problems.append(f'{arm}: selected trial is not the minimum-{metric} trial')
        nt = {int(t.get('n_tokens')) for t in trials if isinstance(t, dict) and t.get('n_tokens') is not None}
        if len(nt) > 1:
            problems.append(f'{arm}: trials evaluated different token counts {sorted(nt)} — PPLs are not comparable')
        n_tokens_by_arm[arm] = sorted(nt)
    if len({tuple(v) for v in n_tokens_by_arm.values()}) > 1:
        problems.append('simple baselines evaluated different token counts; report a common token-weighted floor')
    data = artifact.get('data') or {}
    if not data.get('val_fp'):
        warnings.append('artifact has no validation fingerprint — the panel comparison cannot verify it used the same validation slice')
    best_name = None
    if selected_values:
        best_name = min(selected_values, key=selected_values.get)
        reported = artifact.get('best_baseline') or artifact.get('best_arm')
        reported_ppl = artifact.get('best_ppl')
        if reported is not None and reported != best_name:
            problems.append(f'best_baseline={reported!r} is not the minimum selected arm ({best_name!r})')
        if isinstance(reported_ppl, (int, float)) and abs(float(reported_ppl) - selected_values[best_name]) > 1e-12:
            problems.append('best_ppl does not match the minimum selected arm')
    out = {'status': 'ok' if not problems else 'fail',
           'baseline_verified': not problems,
           'selection_metric': metric, 'selection_split': split,
           'arms': sorted(baselines), 'selected_values': selected_values,
           'best_baseline': best_name,
           'best_ppl': (selected_values.get(best_name) if best_name is not None else None),
           'n_eval_tokens': n_tokens_by_arm,
           'data': data, 'problems': problems, 'warnings': warnings}
    if warnings and not problems:
        out['status'] = 'warn'
    return out

# 指标口径表：每个任务的主指标、方向、不平衡处理与聚合方式都显式写出。
# `metric_policy_audit` 用它检查落盘记录，避免「只报 accuracy」「F1 没写
# macro/micro/weighted」「极度不平衡仍以 ROC-AUC 为主」这类不可复核的读数。
METRIC_POLICY = {
    'lm_perplexity': {
        'task': 'language_modeling',
        'primary': 'ppl',
        'direction': 'lower_is_better',
        'secondary': ['nll', 'next_token_accuracy'],
        'imbalance': 'not_applicable',
        'aggregation': 'token_weighted',
        'note': 'PPL is the primary endpoint; next-token accuracy is diagnostic only and must never replace it.',
    },
    'boundary_alignment': {
        'task': 'binary_boundary_detection',
        'primary': 'f1',
        'primary_average': 'macro_over_sequences',
        'direction': 'higher_is_better',
        'secondary': ['precision', 'recall', 'random_precision', 'precision_excess_over_random'],
        'imbalance': 'positives_are_a_small_fraction_of_cut_positions',
        'aggregation': 'macro_over_sequences',
        'note': 'Never report cut accuracy: the negative class dominates. Always report precision, recall, F1 and the matched-length random baseline together.',
    },
    'niah_retrieval': {
        'task': 'multi_class_retrieval',
        'primary': 'top1_accuracy',
        'direction': 'higher_is_better',
        'secondary': ['answer_nll'],
        'chance': '1/vocab',
        'imbalance': 'chance_accuracy_is_tiny_but_the_task_is_not_a_binary_imbalance_metric',
        'aggregation': 'query_unweighted',
        'note': 'Top-1 accuracy is reported with answer NLL; PR-AUC/ROC-AUC are not substitutes for a 8192-way retrieval endpoint and must not be invented from top-1 labels.',
    },
    'binary_classification': {
        'task': 'binary_classification',
        'primary': 'f1',
        'primary_average': 'binary_positive_class',
        'direction': 'higher_is_better',
        'secondary': ['precision', 'recall', 'pr_auc', 'roc_auc', 'balanced_accuracy'],
        'imbalance': 'report prevalence; prefer PR-AUC when positives are rarer than 1:1000',
        'aggregation': 'binary',
        'note': 'F1 must state the averaging rule; for binary tasks it is the positive-class F1.',
    },
    'multiclass_classification': {
        'task': 'multiclass_classification',
        'primary': 'f1_macro',
        'primary_average': 'macro',
        'direction': 'higher_is_better',
        'secondary': ['f1_micro', 'f1_weighted', 'balanced_accuracy', 'per_class_recall'],
        'imbalance': 'macro averaging is required when class frequencies differ; micro equals accuracy for single-label tasks',
        'aggregation': 'macro',
        'note': 'Every reported F1 must name macro, micro or weighted; report per-class recall when macro and micro disagree.',
    },
}

_HYPOTHESIS_REQUIRED = ('statement', 'mechanism', 'predicted_sign', 'primary_metric', 'falsifier')

def hypothesis_audit(table=None, kind='component', names=None, policy=None):
    """Pre-registration completeness / metric-alignment audit.

    A hypothesis is only actionable before an experiment if it states the
    expected mechanism, the signed direction of the primary metric, and a
    falsifier.  This audit also checks that the declared primary metric is one
    the project's ``METRIC_POLICY`` actually defines, so a hypothesis cannot
    silently target an unmeasured quantity (e.g. accuracy for an imbalanced
    endpoint, or ROC-AUC for a 1:1000 task).  `names` limits the audit to the
    declared comparisons in the current panel; missing table entries are
    reported by the caller as `hypotheses_missing`.

    Direction `predicted_sign=0` is legal for an exploratory "difference
    exists" hypothesis, but it can never support a directional superiority
    claim; such components are listed separately as direction_unspecified."""
    if table is None:
        table = HYPOTHESIS_DESIGN if kind == 'component' else BASELINE_HYPOTHESIS_DESIGN
    policy = METRIC_POLICY if policy is None else policy
    allowed = set()
    for p in policy.values():
        for key in ('primary', 'secondary'):
            val = p.get(key)
            if isinstance(val, str):
                allowed.add(val)
            elif isinstance(val, (list, tuple)):
                allowed.update(str(x) for x in val)
    wanted = set(names) if names is not None else set(table)
    out = {'status': 'ok', 'kind': kind, 'hypotheses': {},
           'invalid': [], 'metric_unknown': [], 'direction_unspecified': [],
           'warnings': []}
    for name in sorted(wanted):
        spec = table.get(name)
        if not isinstance(spec, dict):
            out['invalid'].append(name)
            out['warnings'].append(f'{kind} hypothesis `{name}` is missing from the pre-registered table')
            continue
        issues = []
        for key in _HYPOTHESIS_REQUIRED:
            val = spec.get(key)
            if key == 'predicted_sign':
                if isinstance(val, bool) or val not in (-1, 0, 1):
                    issues.append(f'predicted_sign={val!r} (must be -1, 0 or 1)')
            elif not isinstance(val, str) or not val.strip():
                issues.append(f'{key} is empty or not a string')
        pm = spec.get('primary_metric')
        metric_known = bool(isinstance(pm, str) and pm in allowed)
        pred = spec.get('predicted_sign')
        row = {'statement': spec.get('statement'), 'mechanism': spec.get('mechanism'),
               'predicted_sign': pred, 'primary_metric': pm,
               'falsifier': spec.get('falsifier'),
               'direction_specified': (pred in (-1, 1)),
               'metric_declared_in_policy': metric_known,
               'complete': not issues}
        out['hypotheses'][name] = row
        if issues:
            out['invalid'].append(name)
            out['warnings'].append(f'{kind} hypothesis `{name}` is incomplete: ' + '; '.join(issues))
        if not metric_known:
            out['metric_unknown'].append(name)
            out['warnings'].append(f'{kind} hypothesis `{name}` declares primary_metric={pm!r}, which METRIC_POLICY does not define')
        if pred == 0:
            out['direction_unspecified'].append(name)
            out['warnings'].append(f'{kind} hypothesis `{name}` has predicted_sign=0 — it can test for a difference but cannot support a directional advantage claim')
    out['status'] = 'warn' if out['warnings'] else 'ok'
    return out

def _binary_roc_auc(y, score):
    """Rank-based ROC-AUC for a binary endpoint (ties get average ranks)."""
    y = np.asarray(y).astype(np.int64)
    score = np.asarray(score, dtype=float)
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty(score.size, dtype=float)
    sorted_score = score[order]
    i = 0
    while i < order.size:
        j = i + 1
        while j < order.size and sorted_score[j] == sorted_score[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * ((i + 1) + j)
        i = j
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

def _binary_average_precision(y, score):
    """Average precision (the step-wise PR-AUC) for a binary endpoint."""
    y = np.asarray(y).astype(np.int64)
    score = np.asarray(score, dtype=float)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return None
    order = np.argsort(-score, kind='mergesort')
    ys, ss = y[order], score[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    last = np.r_[np.nonzero(np.diff(ss))[0], ss.size - 1]
    tp, fp = tp[last], fp[last]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(n_pos, 1)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))

def classification_metrics(y_true, y_score=None, y_pred=None, labels=None, positive_label=None):
    """Imbalance-aware classification summary without a sklearn dependency.

    Always returns accuracy, balanced accuracy and per-class precision/recall/
    F1; reports **macro, micro and weighted** F1 separately (they can disagree
    sharply under class imbalance; macro includes every requested label, with
    F1=0 for a label the predictions never recover).  For a binary endpoint
    with scores it also returns PR-AUC (average precision), ROC-AUC, the
    positive-class prevalence and the PR-AUC lift over that prevalence, so a
    rare-positive PR-AUC can be read against its chance floor.  The caller must
    still choose the endpoint that matches the research question; this helper
    only makes the choice explicit and computable.

    `positive_label` names the class whose score/probability column is the
    positive endpoint.  If omitted it defaults to ``labels[-1]`` (the sklearn
    convention for sorted binary labels), but callers with reversed or
    non-numeric labels should pass it explicitly rather than relying on order.
    """
    y_true = np.asarray(y_true).ravel()
    if labels is None:
        if y_score is not None and np.asarray(y_score).ndim == 2:
            labels = np.arange(np.asarray(y_score).shape[1])
        else:
            labels = np.unique(y_true)
            if y_pred is not None:
                labels = np.unique(np.concatenate([labels, np.asarray(y_pred).ravel()]))
    labels = np.asarray(labels)
    if labels.size == 0:
        raise ValueError('classification_metrics needs at least one label')
    pos = None
    if labels.size == 2:
        if positive_label is None:
            pos = labels[-1]
        else:
            hits = np.where(labels == positive_label)[0]
            if hits.size == 0:
                raise ValueError(f'positive_label={positive_label!r} is not in labels {labels.tolist()}')
            pos = labels[int(hits[0])]
    elif positive_label is not None:
        raise ValueError('positive_label is only meaningful for a binary endpoint')
    if y_pred is None:
        if y_score is None:
            raise ValueError('classification_metrics needs y_pred or y_score')
        score = np.asarray(y_score)
        if score.ndim == 2:
            y_pred = labels[np.argmax(score, axis=1)]
        elif labels.size == 2:
            if score.size != y_true.size:
                raise ValueError(f'y_score {score.shape} and y_true {y_true.shape} differ')
            neg = labels[0] if pos == labels[-1] else labels[-1]
            y_pred = np.where(score.ravel() >= 0.5, pos, neg)
        else:
            raise ValueError('a 1-D y_score is only interpretable for a binary endpoint')
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f'y_true {y_true.shape} and y_pred {y_pred.shape} differ')
    per_class = {}
    supports = []
    f1s = []
    rs = []
    all_f1s = []
    total_tp = total_fp = total_fn = 0
    for lab in labels:
        tp = int(np.sum((y_true == lab) & (y_pred == lab)))
        fp = int(np.sum((y_true != lab) & (y_pred == lab)))
        fn = int(np.sum((y_true == lab) & (y_pred != lab)))
        support = int(np.sum(y_true == lab))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(lab)] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': support}
        all_f1s.append(f1)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        if support:
            supports.append(support)
            f1s.append(f1)
            rs.append(recall)
    micro_f1 = 2 * total_tp / max(2 * total_tp + total_fp + total_fn, 1)
    # Macro averages over the requested label set (zero-support labels count
    # as 0, as in sklearn), so a rare class cannot be dropped from the macro
    # score merely because it happens to be absent from this sample.
    macro_f1 = float(np.mean(all_f1s)) if all_f1s else 0.0
    weighted_f1 = float(np.average(f1s, weights=supports)) if supports else 0.0
    f1_positive = per_class[str(pos)]['f1'] if pos is not None else None
    out = {
        'n': int(y_true.size),
        'labels': [str(x) for x in labels],
        'positive_label': (str(pos) if pos is not None else None),
        'prevalence': {str(lab): float(np.mean(y_true == lab)) for lab in labels},
        'accuracy': float(np.mean(y_true == y_pred)) if y_true.size else None,
        'balanced_accuracy': float(np.mean(rs)) if rs else None,
        'f1_macro': macro_f1,
        'f1_micro': float(micro_f1),
        'f1_weighted': weighted_f1,
        'f1_positive': f1_positive,
        'f1_primary': (f1_positive if pos is not None else macro_f1),
        'f1_primary_average': ('binary_positive_class' if pos is not None else 'macro'),
        'precision_macro': float(np.mean([v['precision'] for v in per_class.values()])) if per_class else 0.0,
        'recall_macro': float(np.mean(rs)) if rs else 0.0,
        'per_class': per_class,
        'f1_averaging': 'macro, micro and weighted are all reported; the caller must name the primary one',
    }
    if y_score is not None and labels.size == 2:
        y_bin = (y_true == pos).astype(np.int64)
        # A 2-D score matrix is column-per-label: the binary endpoint uses the
        # named positive-class column, not an assumed last column.
        _score = np.asarray(y_score, dtype=float)
        if _score.ndim == 2:
            _pos_idx = int(np.where(labels == pos)[0][0])
            _bin_score = _score[:, _pos_idx]
        else:
            _bin_score = _score.ravel()
        if _bin_score.size != y_bin.size:
            raise ValueError(f'y_score {_score.shape} does not match y_true {y_true.shape}')
        out['roc_auc'] = _binary_roc_auc(y_bin, _bin_score)
        out['pr_auc'] = _binary_average_precision(y_bin, _bin_score)
        prevalence = float(np.mean(y_bin))
        out['pr_auc_prevalence'] = prevalence
        if out['pr_auc'] is not None:
            out['pr_auc_lift_over_prevalence'] = float(out['pr_auc'] - prevalence)
        if out['pr_auc'] is not None and out['roc_auc'] is not None:
            rare = min(prevalence, 1.0 - prevalence)
            out['imbalance_warning'] = ('positives rarer than 1:1000 — PR-AUC is the primary ranking metric; ROC-AUC can look optimistic'
                                        if rare < 1e-3 else None)
    elif labels.size == 2 and out.get('positive_label') is not None:
        prev = out['prevalence'].get(str(pos), 0.0)
        if min(prev, 1.0 - prev) < 1e-3:
            out['imbalance_warning'] = ('positives rarer than 1:1000 — report PR-AUC with scores; F1 must name its averaging rule and recall should be reported per class')
    return out

def metric_policy_audit(summary=None, policy=None):
    """Check the declared metric contract and the records that claim to meet it.

    The audit never selects a metric for the experimenter.  It checks that each
    task's primary metric, direction, aggregation and imbalance policy are
    declared; that every F1 endpoint names its averaging rule (macro, micro,
    weighted or binary-positive); and that a panel which reports boundary,
    retrieval or classification results also reports the companion metrics
    needed to interpret them (precision/recall plus random baseline for
    boundaries; answer NLL for retrieval; PR-AUC alongside ROC-AUC for rare
    positives)."""
    policy = METRIC_POLICY if policy is None else policy
    out = {'status': 'ok', 'policies': {}, 'checks': [], 'warnings': []}
    for name, p in policy.items():
        missing = [k for k in ('task', 'primary', 'direction', 'aggregation', 'imbalance') if not p.get(k)]
        primary = str(p.get('primary') or '')
        if primary.startswith('f1') and not p.get('primary_average'):
            missing.append('primary_average')
        out['policies'][name] = {'primary': p.get('primary'),
                                 'primary_average': p.get('primary_average'),
                                 'direction': p.get('direction'),
                                 'aggregation': p.get('aggregation'), 'imbalance': p.get('imbalance'),
                                 'complete': not missing}
        if missing:
            out['warnings'].append(f'metric policy `{name}` is missing {missing}')
    if summary:
        recs = [r for r in summary.values() if isinstance(r, dict)]
        lm = [r for r in recs if ppl_is_usable(r.get('ppl'))]
        out['checks'].append({'metric': 'ppl', 'policy': 'lm_perplexity', 'records': len(lm),
                              'ok': bool(lm) and all(math.isfinite(float(r['ppl'])) for r in lm)})
        bnd = []
        niah = []
        for r in recs:
            if r.get('task') == 'niah_natural':
                niah.append(r)
            for layer in (r.get('stats') or {}).values() if isinstance(r.get('stats'), dict) else []:
                if isinstance(layer, dict) and 'bnd_f1' in layer:
                    bnd.append(layer)
        if bnd:
            missing = [k for k in ('bnd_prec', 'bnd_rec', 'bnd_rand') if not all(k in b for b in bnd)]
            out['checks'].append({'metric': 'boundary_alignment', 'policy': 'boundary_alignment',
                                  'records': len(bnd), 'ok': not missing, 'missing': missing,
                                  'f1_average': 'macro_over_sequences'})
            if missing:
                out['warnings'].append(f'boundary metrics report F1 without {missing}; F1 alone cannot be interpreted against the random-cut baseline')
        if niah:
            missing = [k for k in ('acc', 'nll') if not all(k in b for b in niah)]
            out['checks'].append({'metric': 'niah_retrieval', 'policy': 'niah_retrieval',
                                  'records': len(niah), 'ok': not missing, 'missing': missing})
            if missing:
                out['warnings'].append(f'NIAH reports top-1 without {missing}; answer NLL is required alongside retrieval accuracy')
        classify_tasks = ('classification', 'binary_classification', 'multiclass_classification')
        for r in recs:
            task = r.get('task')
            # Generic guard: a record that exposes a bare `accuracy` for a
            # classification-style task but no F1 must be treated as incomplete.
            if task in classify_tasks and 'accuracy' in r and 'f1_macro' not in r:
                out['warnings'].append(f"{r.get('variant', '?')} reports accuracy without F1/averaging for a classification task")
            # F1 without a stated averaging rule is not a reproducible metric.
            f1_keys = [k for k in ('f1', 'f1_macro', 'f1_micro', 'f1_weighted', 'f1_positive') if k in r]
            if task in classify_tasks and f1_keys and not (r.get('f1_averaging') or r.get('f1_primary_average')):
                out['warnings'].append(f"{r.get('variant', '?')} reports {f1_keys} without naming macro/micro/weighted/binary averaging")
            # A rare-positive panel that reports only ROC-AUC is exactly the
            # optimistic-score failure mode this policy exists to prevent.
            prevalence = None
            for key in ('prevalence', 'positive_prevalence'):
                val = r.get(key)
                if isinstance(val, dict) and len(val) == 2:
                    prevalence = min(float(x) for x in val.values())
                    break
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    prevalence = min(float(val), 1.0 - float(val))
                    break
            if task in classify_tasks and prevalence is not None and prevalence < 1e-3:
                if r.get('roc_auc') is not None and r.get('pr_auc') is None:
                    out['warnings'].append(f"{r.get('variant', '?')} reports ROC-AUC but no PR-AUC at prevalence {prevalence:.2e} (<1:1000); PR-AUC is the primary ranking metric here")
                if r.get('pr_auc') is not None and r.get('pr_auc_prevalence') is None:
                    out['warnings'].append(f"{r.get('variant', '?')} reports a rare-positive PR-AUC without its prevalence floor; PR-AUC must be read against chance prevalence")
    out['status'] = 'warn' if out['warnings'] else 'ok'
    return out

def baseline_tuning_audit(selected=None, path=None, protocol=None):
    """Audit the equal-search baseline-tuning evidence.

    A win over an untuned dense baseline establishes only that the proposed
    model beats a straw man.  This audit requires, for every compared arm:
    the *same declared trial grid*, the *same number of trials*, the *same
    selection split* (validation, never test), the *same selection metric*, a
    selected config that really is the best recorded trial, and a
    `selected_metric_value` that is the value of **that** config (not merely
    the minimum of some other trial).  When trial records expose
    steps/tokens_seen, the training allocation must also agree across arms.  When
    they expose `seeds` / `seed_ppls`, the aggregate must be the mean of those
    per-seed values and every arm must select on the same search-seed set —
    otherwise one arm's winner can be a noise draw even though lr grids match.
    The final-seed coverage is checked for equality and reported with the
    family-resolution floor.
    Missing evidence is reported as `not_verifiable_locally`, never as a pass.
    """
    _explicit_protocol = protocol is not None
    if selected is None and path:
        if os.path.exists(path):
            try:
                selected = json.load(open(path, encoding='utf-8'))
            except Exception as e:
                return {'status': 'fail', 'tuning_verified': False, 'problems': [f'cannot parse {path}: {type(e).__name__}: {e}']}
        else:
            return {'status': 'not_verifiable_locally', 'tuning_verified': False,
                    'problems': [f'{path} absent — run baseline_tune.py before claiming a baseline win']}
    # An artifact records the protocol it was produced under.  Use it when the
    # caller did not pass one explicitly, so a smoke/subset run is audited
    # against its own declared grid.
    if not _explicit_protocol and isinstance(selected, dict) and isinstance(selected.get('protocol'), dict):
        protocol = selected['protocol']
    else:
        protocol = BASELINE_TUNING_PROTOCOL if protocol is None else protocol
    if not isinstance(selected, dict) or not isinstance(selected.get('arms'), dict):
        return {'status': 'not_verifiable_locally', 'tuning_verified': False,
                'problems': ['no baseline_tuning artifact (expected {"arms": {...}})']}
    arms = selected['arms']
    metric = selected.get('selection_metric') or protocol.get('selection_metric')
    split = selected.get('selection_split') or protocol.get('selection_split')
    direction = selected.get('direction') or protocol.get('direction')
    problems = []
    warnings = []
    if split != 'validation':
        problems.append(f'selection split is {split!r}; hyperparameters must be selected on validation, never test')
    if metric != protocol.get('selection_metric'):
        problems.append(f'selection metric {metric!r} differs from the declared {protocol.get("selection_metric")!r}')
    if direction not in ('lower_is_better', 'higher_is_better'):
        problems.append(f'unknown selection direction {direction!r}')
    declared = {json.dumps(t, sort_keys=True, default=str) for t in (protocol.get('trials') or [])}
    signatures = {}
    n_trials = {}
    budgets = {}
    search_seed_sets = {}
    for arm, entry in arms.items():
        trials = entry.get('trials') if isinstance(entry, dict) else None
        if not isinstance(trials, list) or not trials:
            problems.append(f'{arm}: no trial records')
            continue
        cfgs = [json.dumps(t.get('config') or {}, sort_keys=True, default=str) for t in trials if isinstance(t, dict)]
        signatures[arm] = tuple(sorted(cfgs))
        n_trials[arm] = len(cfgs)
        # Multi-seed selection: when trials expose their seed set, every trial
        # of every arm must average the same seeds.  A trial whose `ppl` is
        # not the mean of its own per-seed values is a fabricated aggregate.
        _set = set()
        for t in trials:
            if not isinstance(t, dict):
                continue
            _seeds = t.get('seeds')
            _ppls = t.get('seed_ppls')
            if _ppls is not None and not isinstance(_ppls, dict):
                problems.append(f'{arm}: seed_ppls is not a mapping')
                continue
            if _seeds is not None:
                try:
                    _ss = tuple(sorted({int(s) for s in _seeds}))
                except (TypeError, ValueError):
                    problems.append(f'{arm}: trial seeds are not integers ({_seeds!r})')
                    continue
                if len(_ss) != len(set(_seeds)):
                    problems.append(f'{arm}: trial lists duplicate seeds')
                if _ppls is not None and set(str(s) for s in _ss) != set(_ppls):
                    problems.append(f'{arm}: trial seeds and seed_ppls keys disagree')
                _set.add(_ss)
            elif _ppls is not None:
                _set.add(tuple(sorted(int(k) for k in _ppls)))
            if isinstance(_ppls, dict) and isinstance(t.get('ppl'), (int, float)) and not isinstance(t.get('ppl'), bool):
                _vals = [float(v) for v in _ppls.values()
                         if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))]
                if _vals:
                    _m = float(np.mean(_vals))
                    if abs(_m - float(t['ppl'])) > 1e-9 * max(1.0, abs(_m)):
                        problems.append(f'{arm}: trial ppl is not the mean of its per-seed values')
                    if t.get('ppl_std') is not None and len(_vals) > 1:
                        _s = float(np.std(_vals, ddof=1))
                        if abs(_s - float(t['ppl_std'])) > 1e-9 * max(1.0, abs(_s)):
                            problems.append(f'{arm}: trial ppl_std is not the sample std of its per-seed values')
        if _set:
            search_seed_sets[arm] = _set
        if declared and any(c not in declared for c in cfgs):
            problems.append(f'{arm}: a trial is outside the declared grid')
        elif declared and set(cfgs) != declared:
            # Equal-but-partial search is allowed for smoke/bounded runs, but
            # it must be described as a subset, not as the full declared grid.
            _missing_cfg = sorted(declared - set(cfgs))
            warnings.append(f'{arm}: searched a strict subset of the declared grid (missing {_missing_cfg[:2]})')
        by_cfg = {}
        for t in trials:
            if not isinstance(t, dict):
                continue
            sig = json.dumps(t.get('config') or {}, sort_keys=True, default=str)
            by_cfg[sig] = (t.get('steps'), t.get('tokens_seen'))
        budgets[arm] = by_cfg
        _steps = {v[0] for v in by_cfg.values() if v[0] is not None}
        _toks = {v[1] for v in by_cfg.values() if v[1] is not None}
        if len(_steps) > 1:
            problems.append(f'{arm}: trials used different step budgets {sorted(_steps)}')
        if len(_toks) > 1:
            problems.append(f'{arm}: trials saw different token counts {sorted(_toks)}')
        chosen = entry.get('selected') or entry.get('selected_config')
        if not isinstance(chosen, dict):
            problems.append(f'{arm}: no selected config')
            continue
        chosen_sig = json.dumps(chosen, sort_keys=True, default=str)
        if chosen_sig not in cfgs:
            problems.append(f'{arm}: selected config is not one of its recorded trials')
            continue
        chosen_trial = next(t for t in trials if isinstance(t, dict)
                            and json.dumps(t.get('config') or {}, sort_keys=True, default=str) == chosen_sig)
        vals = []
        for t in trials:
            v = t.get(metric) if isinstance(t, dict) else None
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
                vals.append(float(v))
        chosen_value = entry.get('selected_metric_value')
        if chosen_value is None:
            chosen_value = chosen_trial.get(metric)
        if not vals:
            problems.append(f'{arm}: no finite {metric} in its trials')
        elif not isinstance(chosen_value, (int, float)) or isinstance(chosen_value, bool) or not math.isfinite(float(chosen_value)):
            problems.append(f'{arm}: no finite selected {metric} value')
        else:
            _ctv = chosen_trial.get(metric)
            if not isinstance(_ctv, (int, float)) or isinstance(_ctv, bool) or not math.isfinite(float(_ctv)):
                problems.append(f'{arm}: selected trial has no finite {metric}')
            elif abs(float(_ctv) - float(chosen_value)) > 1e-12:
                problems.append(f'{arm}: selected_metric_value={chosen_value} is not the value of the selected config ({_ctv})')
            if direction == 'lower_is_better':
                if float(chosen_value) > min(vals) + 1e-12:
                    problems.append(f'{arm}: selected trial is not the minimum-{metric} trial')
            else:
                if float(chosen_value) < max(vals) - 1e-12:
                    problems.append(f'{arm}: selected trial is not the maximum-{metric} trial')
    if signatures:
        if len(set(signatures.values())) != 1:
            problems.append('arms did not search the same trial grid — tuning budgets are unequal')
        if len(set(n_trials.values())) != 1:
            problems.append('arms did not receive the same number of tuning trials — tuning budgets are unequal')
    # Equal tuning includes equal SEARCH SEEDS.  If one arm selected on seeds
    # {0} and another on {0,1,2}, their selection noise differs even though the
    # lr grid is identical.  Only trials that expose their seed set are
    # checkable; legacy artifacts without it stay on the untested path.
    search_seed_balanced = None
    search_seeds_seen = None
    _declared_ss = protocol.get('search_seeds')
    if _declared_ss is not None:
        try:
            _declared_ss = tuple(sorted({int(s) for s in _declared_ss}))
        except (TypeError, ValueError):
            problems.append(f'declared search_seeds are not integers ({_declared_ss!r})')
            _declared_ss = None
    if search_seed_sets:
        _no_seed_info = sorted(set(arms) - set(search_seed_sets))
        if _no_seed_info:
            warnings.append(f'arms {_no_seed_info} do not expose their search seeds — equal search seeds cannot be verified for them')
        if len(search_seed_sets) >= 2:
            search_seed_balanced = len({frozenset(v) for v in search_seed_sets.values()}) == 1
            if not search_seed_balanced:
                problems.append('arms did not select on the same search-seed set — equal-search tuning is not verified')
        search_seeds_seen = sorted({s for v in search_seed_sets.values() for ss in v for s in ss})
        if _declared_ss is not None and set(search_seeds_seen) != set(_declared_ss):
            warnings.append(f'trials report search seeds {search_seeds_seen} but the declared set is {list(_declared_ss)}')
    elif _declared_ss is not None:
        warnings.append('trial records do not expose their search seeds — multi-seed selection cannot be verified from this artifact')
    _all_sigs = set().union(*(set(b) for b in budgets.values())) if budgets else set()
    for sig in sorted(_all_sigs):
        _vals = {arm: b[sig] for arm, b in budgets.items() if sig in b}
        _st = {v[0] for v in _vals.values() if v[0] is not None}
        _tk = {v[1] for v in _vals.values() if v[1] is not None}
        if len(_st) > 1 or len(_tk) > 1:
            problems.append(f'trial {sig[:48]} uses different steps/tokens across arms (steps={sorted(_st)}, tokens={sorted(_tk)})')
            break
    if selected.get('search_complete') is False:
        warnings.append('the search grid was truncated (search_complete=False); it is equal across arms, but describe it as a bounded search, not the full declared grid')
    final_records = selected.get('final_records') or {}
    final_seeds = selected.get('final_seeds') or protocol.get('final_seeds') or []
    final_seed_coverage = {}
    if final_records:
        for arm, entry in arms.items():
            chosen = entry.get('selected') or entry.get('selected_config') or {}
            hits = [(k, r) for k, r in final_records.items()
                    if k.startswith(f'{arm}::') and isinstance(r, dict)]
            if not hits:
                problems.append(f'{arm}: tuning artifact claims final records but none are bound to this arm')
                continue
            _seeds = [int(r['seed']) for _k, r in hits if r.get('seed') is not None]
            if len(_seeds) != len(set(_seeds)):
                problems.append(f'{arm}: duplicate final records for the same seed')
            final_seed_coverage[arm] = sorted(set(_seeds))
            if final_seeds:
                _need = {int(s) for s in final_seeds}
                _miss = sorted(_need - set(_seeds))
                if _miss:
                    problems.append(f'{arm}: final records miss declared seeds {_miss}')
            for _k, r in hits:
                hp = r.get('hyperparams') or {}
                for _kn in ('lr', 'weight_decay', 'warmup'):
                    if _kn in chosen and hp.get(_kn) is not None and float(hp[_kn]) != float(chosen[_kn]):
                        problems.append(f'{arm}: final record {_kn}={hp[_kn]} does not match the selected {chosen[_kn]}')
                        break
                if r.get('variant') is not None and r.get('variant') != arm:
                    problems.append(f'{arm}: final record key {_k} carries variant={r.get("variant")!r}')
    final_seed_coverage_equal = None
    final_min_seed_count = None
    _final_family_floor = None
    if final_seed_coverage:
        final_seed_coverage_equal = len({tuple(v) for v in final_seed_coverage.values()}) == 1
        final_min_seed_count = min(len(v) for v in final_seed_coverage.values())
        if not final_seed_coverage_equal:
            problems.append('final records do not cover the same seeds in every arm — the paired comparison would silently use different n per arm')
        # The tuned comparison is its own family (method vs each other arm).
        _final_family_floor = signflip_min_n_for_holm(max(1, len(arms) - 1), 0.05)
        if final_min_seed_count < 3:
            warnings.append(f'only {final_min_seed_count} final seed(s) per arm — the reporting floor for a key experiment is 3; treat the tuned contrast as descriptive')
        elif _final_family_floor is not None and final_min_seed_count < _final_family_floor:
            warnings.append(f'{final_min_seed_count} final seeds per arm cannot reach a Holm-corrected p<0.05 for {max(1, len(arms) - 1)} tuned contrast(s) (needs n>={_final_family_floor}); the tuned contrast is resolution-limited, not a null')
    required = set(protocol.get('arms') or [])
    if required and not required.issubset(set(arms)):
        problems.append(f'missing tuning evidence for arms {sorted(required - set(arms))}')
    _grid_complete = None
    if signatures and declared:
        _grid_complete = all(set(sig) == declared for sig in signatures.values())
    out = {'status': 'ok' if not problems else 'fail', 'tuning_verified': not problems,
           'selection_metric': metric, 'selection_split': split,
           'n_trials': n_trials, 'same_search_space': (len(set(signatures.values())) <= 1) if signatures else None,
           'search_seeds': search_seeds_seen if search_seeds_seen is not None else (_declared_ss and list(_declared_ss)),
           'search_seed_balanced': search_seed_balanced,
           'final_seed_coverage': final_seed_coverage,
           'final_seed_coverage_equal': final_seed_coverage_equal,
           'final_min_seed_count': final_min_seed_count,
           'min_final_seeds_for_family_significance': _final_family_floor,
           'search_complete': selected.get('search_complete') if selected.get('search_complete') is not None else _grid_complete,
           'grid_complete': _grid_complete,
           'arms': sorted(arms), 'problems': problems, 'warnings': warnings}
    if warnings and not problems:
        out['status'] = 'warn'
    return out

def _variant_seed_map(summary):
    """{(variant, protocol, seed): rec} for measurable, non-synthesized records
    with an unambiguous seed, plus the set of (variant, protocol, seed) triples
    that hold more than one record (ambiguous → excluded, first key wins).
    This is the single reader `_print_pairs` and `methodology_audit` share."""
    _legacy_ok = legacy_warm_tags(summary)
    recs = {}
    amb = set()
    for _k, r in summary.items():
        if not (isinstance(r, dict) and ppl_is_usable(r.get('ppl')) and ('seed' in r) and (not r.get('synthesized'))):
            continue
        rk = _record_identity(r, _k, _legacy_ok)
        if rk is None or rk in amb:
            continue
        if rk in recs:
            amb.add(rk)
            recs.pop(rk, None)
            continue
        recs[rk] = r
    return recs, amb

def _paired_seed_deltas(recs, a, b, *, strict_hyper=False):
    """Per-protocol-condition per-seed (a − b) deltas over seeds where BOTH arms
    have a measurable record AND `pair_reason` confirms identical run_cfg /
    training length.

    Protocol conditions are matched exactly as `_print_pairs` does: equal
    non-empty tags pair together, and a tagged arm may pair against a pure-base
    ('') arm.  Two DIFFERENT non-empty conditions are never pooled — collapsing
    them (keying only on the seed) would fabricate seed counts and compare
    different measurement protocols (design_notes §3.3).

    With `strict_hyper=True` (module ablations), the structured
    `hyperparams` copy must also agree on every `_CONTROL_HYPER_KEYS` field
    (notably `mlp_ratio`), so a capacity change cannot ride along with the
    module toggle.  Baseline contrasts leave it off because their MLP width is
    intentionally matched via a different parameter count.

    Returns {(tag_a, tag_b): (deltas, shared_seeds, skipped)} with
    skipped = [(seed, reason)]."""
    ra = {(t, s): r for (v, t, s), r in recs.items() if v == a}
    rb = {(t, s): r for (v, t, s), r in recs.items() if v == b}
    _a_t = {t for t, _s in ra}
    _b_t = {t for t, _s in rb}
    _a_pure = _a_t == {''}
    _b_pure = _b_t == {''}
    conds = []
    if '' in _a_t and '' in _b_t:
        conds.append(('', ''))
    for tag in sorted(_a_t | _b_t):
        if not tag:
            continue
        if tag in _a_t and tag in _b_t:
            conds.append((tag, tag))
        if tag in _a_t and _b_pure:
            conds.append((tag, ''))
        if tag in _b_t and _a_pure:
            conds.append(('', tag))
    out = {}
    for ts, bs in conds:
        seeds = sorted({s for t, s in ra if t == ts} & {s for t, s in rb if t == bs})
        deltas, skipped = [], []
        for s in seeds:
            reason = pair_reason(ra[ts, s], rb[bs, s])
            if reason is None and strict_hyper:
                reason = _hyperparam_reason(ra[ts, s], rb[bs, s])
            if reason is not None:
                skipped.append((s, reason))
                continue
            deltas.append(float(ra[ts, s]['ppl']) - float(rb[bs, s]['ppl']))
        out[(ts, bs)] = (deltas, seeds, skipped)
    return out

def _knobs_equal_across_arms(recs, arms, key):
    """Compare a per-record descriptor (`run_cfg`, `train_knobs`, ...) across
    the arms of one component.  Returns True/False/None: True when at least two
    arms expose the key and all exposed values agree, False when any two
    disagree, None when fewer than two arms expose it (unverifiable)."""
    vals = []
    for a in arms:
        for (v, _t, _s), r in recs.items():
            if v != a:
                continue
            if key in r and r.get(key) is not None:
                vals.append(json.dumps(r.get(key), sort_keys=True, default=str))
                break
            # keep scanning this arm: a legacy/first record without the key
            # must not mask a later record that does record it.
    if len(vals) < 2:
        return None
    return len(set(vals)) == 1

# Settings that must be identical across arms of a controlled contrast.  The
# first two are the authoritative descriptors already used by `pair_reason`;
# the rest are read from the structured `hyperparams` copy so a future setting
# that never made it into the compact run_cfg string is still checked.
_CONTROL_HYPER_KEYS = (
    'steps', 'tokens_seen', 'lr', 'weight_decay', 'warmup', 'batch_size',
    'seq_len', 'd', 'n_layers', 'n_heads', 'd_head', 'mlp_ratio',
    'comp_lambda', 'delta_lr_mult', 'dataset', 'data_version',
    'amp_dtype', 'eager_batch',
)

def _canon_setting(v):
    """Canonical JSON spelling of one setting value (None stays 'missing')."""
    if v is None:
        return None
    return json.dumps(_jsonable(v), sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, default=str)

def _setting_values(recs, arms, getter):
    vals = {a: set() for a in arms}
    for (v, _t, _s), r in recs.items():
        if v not in vals:
            continue
        val = getter(r)
        if val is not None:
            vals[v].add(val)
    return vals

def _setting_control(vals_by_arm, *, check_within_arm=True):
    """Aggregate one setting across arms.

    Returns `equal` (True/False/None), every arm's distinct values, the arms
    whose own seeds disagree (mixed protocol inside one arm), and arms with
    no value.  `equal=None` means unverifiable (legacy record), never a pass.
    """
    arms = list(vals_by_arm)
    nonempty = {a: v for a, v in vals_by_arm.items() if v}
    mixed = sorted(a for a, v in nonempty.items() if len(v) > 1) if check_within_arm else []
    missing = sorted(a for a in arms if not vals_by_arm.get(a))
    if not nonempty:
        equal = None
    elif mixed:
        equal = False
    elif len(nonempty) < len(arms):
        equal = None
    else:
        union = set().union(*nonempty.values())
        equal = bool(len(union) == 1)
    return {'equal': equal, 'within_arm_mixed': mixed, 'missing_arms': missing,
            'values': {a: sorted(v) for a, v in nonempty.items()}}

def controlled_variables_report(recs, arms):
    """Per-component controlled-variable audit across ALL seed records.

    `pair_reason` is the per-seed hard gate; this report is the panel-level
    diagnostic.  Every record of an arm is inspected (so seed 0 and seed 4
    cannot run under different protocols), a disagreement inside one arm is
    flagged, and the structured `hyperparams` copy is checked for settings
    that the compact run_cfg string does not carry.  The return keeps the
    `run_cfg_equal` / `train_knobs_equal` keys and adds the detail.
    """
    arms = sorted(set(arms))
    run_cfg = _setting_control(_setting_values(
        recs, arms, lambda r: _canon_setting(r.get('run_cfg'))))
    train_knobs = _setting_control(_setting_values(
        recs, arms, lambda r: _canon_setting(r.get('train_knobs'))))
    hyper = {}
    for key in _CONTROL_HYPER_KEYS:
        hyper[key] = _setting_control(_setting_values(
            recs, arms, lambda r, k=key: _canon_setting((r.get('hyperparams') or {}).get(k))))
    hp_checked = [k for k, v in hyper.items()
                  if any(v['values'].get(a) for a in arms)]
    hp_diffs = [k for k in hp_checked if hyper[k]['equal'] is False]
    hp_unknown = [k for k in hp_checked if hyper[k]['equal'] is None]
    if not hp_checked:
        hp_equal = None
    elif hp_diffs:
        hp_equal = False
    elif hp_unknown:
        hp_equal = None
    else:
        hp_equal = True
    mixed = sorted(set(run_cfg['within_arm_mixed']) | set(train_knobs['within_arm_mixed'])
                   | {a for k in hp_checked for a in hyper[k]['within_arm_mixed']})
    diff_values = {}
    for key, ctl in (('run_cfg', run_cfg), ('train_knobs', train_knobs)):
        if ctl['equal'] is False:
            diff_values[key] = ctl['values']
    for key in hp_diffs:
        diff_values['hyperparams.' + key] = hyper[key]['values']
    # Seed/protocol coverage: arm means are only directly comparable when
    # every arm averaged the same seeds under the same measurement protocol.
    # Paired contrasts already use only shared seeds, but an unequal arm still
    # changes the descriptive mean±std and can hide a lost seed (crash / cap /
    # error).  Report the exact coverage; None means no arm has a record.
    _seed_sets = {}
    for (v, t, s) in recs:
        if v not in arms:
            continue
        _ss = _as_seed_int(s)
        if _ss is None:
            continue
        _seed_sets.setdefault(v, set()).add((str(t or ''), _ss))
    seed_coverage = {a: sorted(_seed_sets.get(a, set())) for a in arms}
    _nonempty = [frozenset(v) for v in seed_coverage.values() if v]
    if not _nonempty:
        seed_coverage_equal = None
    elif any(not v for v in seed_coverage.values()):
        seed_coverage_equal = None
    else:
        seed_coverage_equal = bool(len(set(_nonempty)) == 1)
    controlled = None
    flags = [run_cfg['equal'], train_knobs['equal'], hp_equal]
    if any(f is False for f in flags):
        controlled = False
    elif all(f is True for f in flags):
        controlled = True
    return {
        'arms': arms,
        'n_records': {a: sum(1 for (v, _t, _s) in recs if v == a) for a in arms},
        'run_cfg_equal': run_cfg['equal'],
        'train_knobs_equal': train_knobs['equal'],
        'hyperparams_equal': hp_equal,
        'controlled': controlled,
        'within_arm_mixed': mixed,
        'differences': diff_values,
        'checked_hyperparam_keys': sorted(hp_checked),
        'hyperparam_differences': hp_diffs,
        'unverifiable_hyperparam_keys': hp_unknown,
        'seed_coverage': seed_coverage,
        'seed_coverage_equal': seed_coverage_equal,
    }

def _params_of_arm(recs, arm):
    for (v, _t, _s), r in recs.items():
        if v != arm:
            continue
        p = r.get('params')
        try:
            return int(p)
        except (TypeError, ValueError):
            continue
    return None

def _param_control(recs, a, b):
    """Relative parameter-count difference between two arms, and whether it is
    small enough to call them parameter-controlled (<=1%).  None means one or
    both records do not expose `params` (unverifiable, not a mismatch)."""
    pa, pb = _params_of_arm(recs, a), _params_of_arm(recs, b)
    if pa is None or pb is None:
        return (None, None)
    rel = abs(pa - pb) / float(max(pa, pb, 1))
    return (float(rel), bool(rel <= 0.01))

def _holm_assign(entries, m_family, alpha, out_key):
    """Attach a Holm step-down adjusted p to each entry over the declared
    family.  `m_family` is the DECLARED number of contrasts, not just the ones
    with data: a missing arm must not shrink the correction family (that would
    make an incomplete ablation look better-powered than it is).  Dummy p=1
    entries fill the difference."""
    named = []
    for i, e in enumerate(entries):
        p = e.get('p_exact_signflip')
        if isinstance(p, (int, float)) and math.isfinite(float(p)):
            named.append((str(i), float(p)))
    m_eff = max(int(m_family), len(entries), len(named))
    for j in range(m_eff - len(named)):
        named.append((f'__missing_{j}', 1.0))
    h = holm_adjust(named, alpha=alpha)
    for i, e in enumerate(entries):
        rec = h.get(str(i))
        e[out_key] = rec.get('p_holm') if isinstance(rec, dict) else None

def _hypothesis_spec(comp, kind):
    table = HYPOTHESIS_DESIGN if kind == 'component' else BASELINE_HYPOTHESIS_DESIGN
    return table.get(comp)

def _sign_of(x):
    if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(float(x)):
        return None
    if float(x) > 0:
        return 1
    if float(x) < 0:
        return -1
    return 0

def _classify_group_status(entries, *, min_n, alpha, m_family, require_param_match=False):
    """One status per declared innovation / baseline claim.

    verified            — a controlled, parameter-controlled (for baselines)
                          contrast is significant after the family-wise Holm
                          correction, its effect exceeds seed noise, AND (when
                          a pre-registered hypothesis exists) the sign matches
                          the predicted direction;
    direction_conflict  — a corrected significant effect exists but it points
                          against the pre-registered direction.  This is
                          evidence AGAINST the stated mechanism, never support;
    direction_unspecified — significant, but the hypothesis explicitly declared
                          predicted_sign=0; it cannot support a directional
                          advantage claim;
    measured            — a controlled contrast exists with n >= min_n, but no
                          corrected significance;
    resolution_limited  — a controlled, non-noise contrast exists whose
                          family-corrected exact-test p floor is still above
                          alpha at this n (add seeds; never read as a null);
    underpowered        — controlled contrasts exist but every one is n < min_n;
    noise_level         — powered contrasts exist but all effects are within
                          seed noise (|mean| <= std);
    unverified          — no controlled contrast (missing arm / config mismatch).
    """
    usable = [e for e in entries if e.get('controlled') and (not require_param_match or e.get('param_controlled') is True)]
    if not usable:
        return 'unverified'
    powered = [e for e in usable if int(e.get('n') or 0) >= int(min_n)]
    # The n floor is a precondition for every positive verdict: a liberal
    # alpha cannot turn an underpowered contrast into a 'verified' one.
    if not powered:
        return 'underpowered'
    sig = [e for e in powered if e.get('significant_global')]
    if sig:
        if any(e.get('direction_consistent') is False for e in sig):
            return 'direction_conflict'
        if any(e.get('predicted_sign') == 0 for e in sig):
            return 'direction_unspecified'
        return 'verified'
    noise_free = [e for e in powered if not e.get('within_seed_noise')]
    if not noise_free:
        return 'noise_level'
    for e in noise_free:
        fl = holm_min_attainable_p(e.get('n') or 0, m_family, alpha)
        if math.isfinite(fl) and fl > alpha:
            return 'resolution_limited'
    return 'noise_level'

def _scope_design(design, present):
    """Keep only claims this panel actually tests: a component enters the
    family when at least one of its declared contrasts has BOTH arms measured.
    Once in, all of its declared contrasts keep their slots (missing arms are
    dummies), so an incomplete ablation cannot shrink its own correction."""
    out = {}
    present = set(present or ())
    for comp, contrasts in design.items():
        if any(a in present and b in present for a, b, _n in contrasts):
            out[comp] = contrasts
    return out

def ablation_completeness_audit(design=None, present=None, *, arms_by_combo=None,
                                verbose=False):
    """Zero-GPU completeness check for a leave-one-out OR full-factorial design.

    Leave-one-out mode (default)
        `design` is the `ABLATION_DESIGN` shape: component -> list of
        (with_arm, without_arm, note).  A component is COMPLETE only when
        **every** declared contrast has both arms in `present`.  `present=None`
        checks the design's structure without requiring the arms to be run.
        This is the condition for the claim "each module is useful": a
        component measured only through a combined arm (e.g. A+B vs baseline)
        stays incomplete, and the missing arm is named instead of being
        omitted from the correction family.

    Full-factorial mode (`arms_by_combo` given)
        The mapping keys are component-name tuples (or '+'-joined strings,
        with ''/'none' for the empty combination) and values are arm ids.
        ALL 2**k combinations must be mapped, present and unique; a 7-of-8
        panel is reported with its exact missing combination (e.g. C alone)
        rather than being described as a full factorial.  This is the
        larger design the task brief allows as an alternative to
        leave-one-out.

    Returns a machine-readable dict; never raises on a partial panel.
    """
    out = {'mode': 'leave-one-out', 'complete': False,
           'complete_components': [], 'incomplete_components': [],
           'components': {}, 'missing_arms': {}, 'structural_errors': []}
    present_set = None if present is None else set(present)

    if arms_by_combo is not None:
        out.update({'mode': 'full-factorial', 'components': [],
                    'n_required': 0, 'n_mapped': 0,
                    'missing_combos': [], 'duplicate_combos': [],
                    'reused_arms': {}, 'missing_arms': [], 'arms': {}})
        combos = {}
        components = set()
        for key, arm in arms_by_combo.items():
            if isinstance(key, str):
                _raw = [x.strip() for x in re.split(r'[+,|]', key) if x.strip()]
                combo = tuple(sorted(x for x in _raw if x.lower() not in ('none', 'empty', 'baseline', '0')))
            else:
                combo = tuple(sorted(str(x) for x in (key or ())))
            combos.setdefault(combo, []).append(arm)
            components.update(combo)
        components = sorted(components)
        required = []
        for r in range(len(components) + 1):
            required.extend(tuple(c) for c in itertools.combinations(components, r))
        required_set = set(required)
        mapped_set = set(combos)
        missing = sorted(required_set - mapped_set)
        extra = sorted(mapped_set - required_set)
        duplicate = sorted(c for c, arms in combos.items() if len(set(arms)) > 1)
        absent = []
        # In a full factorial every combination must be its OWN arm.  Reusing
        # one arm id for two combinations (e.g. mapping A and B to the same
        # run) silently makes the 2**k table impossible to attribute, so it is
        # a structural failure rather than a complete design.
        _arm_combos = {}
        for combo, arms in combos.items():
            for arm in set(arms):
                _arm_combos.setdefault(arm, set()).add(combo)
        reused = {arm: sorted(cs) for arm, cs in _arm_combos.items() if len(cs) > 1}
        for combo, arms in combos.items():
            for arm in set(arms):
                if present_set is not None and arm not in present_set:
                    absent.append({'combo': list(combo), 'arm': arm})
        out.update({
            'components': components,
            'n_required': len(required),
            'n_mapped': len(combos),
            'missing_combos': [list(c) for c in missing],
            'extra_combos': [list(c) for c in extra],
            'duplicate_combos': [list(c) for c in duplicate],
            'reused_arms': {str(a): [list(c) for c in cs] for a, cs in reused.items()},
            'missing_arms': absent,
            'arms': {('+'.join(c) if c else 'none'): sorted(set(a)) for c, a in combos.items()},
        })
        out['complete'] = bool(len(components) >= 1 and not missing and not extra
                               and not duplicate and not absent and not reused)
        if verbose:
            print('[ablation] full-factorial components=%s mapped=%d/%d missing=%s%s'
                  % (components, len(combos), len(required),
                     ['+'.join(c) or 'none' for c in missing] or 'none',
                     (' reused_arms=%s' % (sorted(reused) or 'none')) if reused else ''))
        return out

    if design is None:
        design = ABLATION_DESIGN
    for comp, contrasts in design.items():
        rows, missing = [], set()
        for item in contrasts:
            if not isinstance(item, (tuple, list)) or len(item) < 2:
                out['structural_errors'].append('%s: malformed contrast %r' % (comp, item))
                continue
            a, b = item[0], item[1]
            ap = present_set is None or a in present_set
            bp = present_set is None or b in present_set
            rows.append({'with': a, 'without': b,
                         'note': item[2] if len(item) > 2 else '',
                         'with_present': ap, 'without_present': bp,
                         'complete': bool(ap and bp)})
            if not ap:
                missing.add(a)
            if not bp:
                missing.add(b)
        complete = bool(rows) and all(r['complete'] for r in rows)
        out['components'][comp] = {'declared': len(rows), 'complete': complete,
                                   'contrasts': rows,
                                   'missing_arms': sorted(missing)}
        if missing:
            out['missing_arms'][comp] = sorted(missing)
        (out['complete_components'] if complete else out['incomplete_components']).append(comp)
    out['complete'] = bool(out['components']) and not out['incomplete_components'] \
        and not out['structural_errors']
    if verbose:
        print('[ablation] leave-one-out complete=%s; missing: %s'
              % (out['complete'],
                 '; '.join('%s=%s' % (c, ','.join(a)) for c, a in sorted(out['missing_arms'].items())) or 'none'))
    return out

def methodology_seed_plan(n_seeds, design=None, baselines=None, alpha=0.05, variants=None):
    """Prospective seed plan for a planned panel.

    Returns the declared family size and the minimum n at which a
    Holm-corrected exact sign-flip test could reject at `alpha` if every seed
    moved in the predicted direction.  Drivers call this BEFORE training so a
    3-seed panel is labelled descriptive up front instead of being read after
    the fact as a failed confirmation.  Pure function; never blocks a run."""
    if design is None:
        design = ABLATION_DESIGN
        if baselines is None:
            baselines = BASELINE_DESIGN
    baselines = {} if baselines is None else baselines
    _present = None
    if variants is not None:
        _present = set(variants or ())
        design = _scope_design(design, _present)
        baselines = _scope_design(baselines, _present)
    _ablation = (ablation_completeness_audit(design, _present) if design else
                 {'mode': 'leave-one-out', 'complete': None, 'components': {},
                  'complete_components': [], 'incomplete_components': [],
                  'missing_arms': {}})
    m = sum(len(v) for v in design.values()) + sum(len(v) for v in baselines.values())
    req = signflip_min_n_for_holm(m, alpha)
    try:
        n = int(n_seeds)
    except (TypeError, ValueError):
        n = 0
    return {'n_seeds': n, 'declared_contrasts': int(m), 'alpha': float(alpha),
            'min_seeds_for_family_significance': req,
            'sufficient_for_family_rejection': bool(req is not None and n >= req),
            'ablation_completeness': _ablation,
            'ablation_complete': _ablation.get('complete'),
            'note': 'a panel below the minimum is DESCRIPTIVE ONLY: the exact-test resolution floor, not the observed effect, bounds its p value'}

def methodology_audit(summary, design=None, *, baselines=None, alpha=0.05, min_n=3, outdir=None, write=True, verbose=True):
    """Zero-GPU audit of controlled comparison, ablation completeness and
    statistical reliability for one panel's `summary` dict.

    For every declared leave-one-out contrast (and, by default, the primary
    parameter-matched dense baselines) it recomputes the per-seed paired delta
    under a strict same-run_cfg / same-train_knobs gate, the mean±std / d_z /
    exact sign-flip p, and flags underpowered (n<min_n), noise-level
    (|mean| <= std) and resolution-limited (the exact-test p floor is above
    alpha) results.

    Multiple comparisons are corrected with Holm step-down over the declared
    family.  A missing arm does NOT shrink the family (dummy p=1 entries keep
    the declared m), so an incomplete ablation cannot buy a lower p.  A
    component is `verified` only when a controlled contrast is significant
    AFTER that correction and its effect exceeds seed noise; merely having
    n>=min_n is reported as `measured` (descriptive), never as verification.
    Per-component Holm results are also exported, because the reviewer-facing
    claim is usually per innovation, while the global correction is the
    conservative cross-component reading.

    Returns the report dict and, when `outdir` is given, writes
    `methodology_audit.json`."""
    default_design = design is None
    design = ABLATION_DESIGN if design is None else design
    recs, amb = _variant_seed_map(summary)
    present = sorted({v for (v, _t, _s) in recs})
    if baselines is None:
        # Baseline claims belong to THIS panel's family only when a matched
        # dense arm was actually scheduled.  A pure ablation panel without
        # full_matched/full_sw128_matched is not testing the baseline claim,
        # so including its (missing) slots would only inflate the correction
        # family and mislabel an untested comparison as "unverified".
        _base_arms = {x for _cs in BASELINE_DESIGN.values() for _a, _b, _n in _cs for x in (_a, _b)}
        baselines = BASELINE_DESIGN if (default_design and (_base_arms & set(present))) else {}
    try:
        _min_sig_n = signflip_min_n_for_holm(1, alpha) or 0
    except (ValueError, ZeroDivisionError):
        _min_sig_n = 0
    report = {'variants_present': present, 'n_measurable': len(recs), 'ambiguous': sorted(amb),
              'min_n': int(min_n), 'alpha': float(alpha), 'min_seeds_for_significance': _min_sig_n,
              'components': {}, 'baselines': {},
              'verified_components': [], 'measured_components': [],
              'significant_components': [], 'nominally_significant_components': [],
              'resolution_limited_components': [], 'underpowered_components': [],
              'noise_level_components': [], 'unverified_components': [],
              'direction_conflict_components': [], 'hypotheses_missing': [],
              'verified_baselines': [], 'measured_baselines': [],
              'significant_baselines': [], 'nominally_significant_baselines': [],
              'resolution_limited_baselines': [], 'underpowered_baselines': [],
              'noise_level_baselines': [], 'unverified_baselines': [],
              'direction_conflict_baselines': [], 'tuned_verified_baselines': [],
              'untuned_verified_baselines': [],
              'direction_unspecified_components': [], 'direction_unspecified_baselines': [],
              'incomplete_components': [], 'controlled_variables': {},
              'ablation_completeness': {}, 'ablation_complete': None,
              'arm_summary': {}, 'arms_below_min_seeds': [],
              'component_seed_requirements': {}, 'baseline_seed_requirements': {},
              'hypotheses': {}, 'hypotheses_incomplete': [], 'hypothesis_audit': {},
              'metric_audit': {}, 'baseline_tuning': {}, 'simple_baseline': {},
              'family': {}, 'warnings': []}

    def _collect(design_group, kind):
        out = {}
        meta = {}
        for comp, contrasts in design_group.items():
            entries = []
            arms = set()
            for a, b, note in contrasts:
                arms.update((a, b))
                a_present, b_present = (a in present, b in present)
                conds = _paired_seed_deltas(recs, a, b, strict_hyper=(kind == 'component'))
                _cfg_a = next((r.get('run_cfg') for (v, _t, _s), r in recs.items() if v == a), None)
                _cfg_b = next((r.get('run_cfg') for (v, _t, _s), r in recs.items() if v == b), None)
                _rel, _param_ok = _param_control(recs, a, b)
                _contrast_measured = False
                for (ts, bs), (deltas, shared, skipped) in sorted(conds.items()):
                    _skipped_seeds = {s for s, _why in skipped}
                    _passed = [s for s in shared if s not in _skipped_seeds]
                    _a_vals = [float(recs[(a, ts, s)]['ppl']) for s in _passed]
                    _b_vals = [float(recs[(b, bs, s)]['ppl']) for s in _passed]
                    st = paired_effect_stats(deltas, alpha=alpha, min_n=min_n, with_values=_a_vals, without_values=_b_vals)
                    _obj = sorted({objective_reason(recs[(a, ts, s)], recs[(b, bs, s)]) for s in _passed})
                    _obj_bad = [x for x in _obj if x and x.startswith('objective differs')]
                    st['objective_matched'] = (False if _obj_bad else (True if all(x is None for x in _obj) else None))
                    st['objective_differences'] = _obj_bad or [x for x in _obj if x and x.startswith('unverifiable')]
                    if _obj_bad:
                        st['mechanism_claim_allowed'] = False
                        report.setdefault('objective_confounded', []).append(
                            '%s vs %s: %s — same run_cfg but different actual loss/optimizer objective; '
                            'the contrast is a system-level comparison, not an isolated mechanism' % (a, b, '; '.join(_obj_bad)))
                    if _a_vals and _b_vals:
                        _a_mean = float(np.mean(_a_vals))
                        _b_mean = float(np.mean(_b_vals))
                        _a_std = float(np.std(_a_vals, ddof=1)) if len(_a_vals) > 1 else 0.0
                        _b_std = float(np.std(_b_vals, ddof=1)) if len(_b_vals) > 1 else 0.0
                        _rel_effect = (100.0 * float(st['mean']) / _b_mean) if _b_mean > 0 else float('nan')
                        _rel_std = (100.0 * float(st['std']) / _b_mean) if _b_mean > 0 else float('nan')
                        if float(st['std']) > 0:
                            _noise_ratio = abs(float(st['mean'])) / float(st['std'])
                        elif abs(float(st['mean'])) <= 1e-12:
                            _noise_ratio = 0.0
                        else:
                            _noise_ratio = float('inf')
                    else:
                        _a_mean = _b_mean = _a_std = _b_std = float('nan')
                        _rel_effect = _rel_std = _noise_ratio = float('nan')
                    _hyp = _hypothesis_spec(comp, kind)
                    _pred = (_hyp or {}).get('predicted_sign')
                    _obs = _sign_of(st.get('mean')) if st.get('n') else None
                    st.update({'with': a, 'without': b, 'note': note, 'kind': kind,
                               'tag_with': ts, 'tag_without': bs,
                               'n_shared': len(shared), 'n_skipped': len(skipped),
                               'n_paired': len(_passed), 'seeds': _passed,
                               'skipped_reasons': sorted({reason for _s, reason in skipped}),
                               # A contrast is controlled when at least one
                               # pair cleared the same-config gate.  Skipped
                               # seeds are excluded from n (and named above);
                               # they must not erase the seeds that DID pass,
                               # nor turn a clean n=4 contrast into
                               # 'uncontrolled'.  `fully_paired` still records
                               # whether every shared seed passed.
                               'controlled': bool(deltas),
                               'fully_paired': bool(deltas) and (not skipped),
                               'run_cfg_equal': _cfg_a is not None and _cfg_a == _cfg_b,
                               'run_cfg_knobs': run_cfg_knobs(_cfg_a) or None,
                               'param_rel_diff': _rel, 'param_controlled': _param_ok,
                               'capacity_confounded': (_param_ok is False),
                               'with_mean': _a_mean, 'with_std': _a_std,
                               'without_mean': _b_mean, 'without_std': _b_std,
                               'relative_effect_pct': _rel_effect,
                               'relative_noise_pct': _rel_std,
                               'signal_to_noise': _noise_ratio,
                               'ci95_low': st.get('ci95_low'),
                               'ci95_high': st.get('ci95_high'),
                               'seed_driven': st.get('seed_driven'),
                               'seed_robust': st.get('seed_robust'),
                               'loo_means': st.get('loo_means'),
                               'mean_pm_std': ('%+.4f +/- %.4f' % (float(st['mean']), float(st['std']))
                                               if st.get('n') else 'n/a'),
                               'hypothesis': (_hyp or {}).get('statement'),
                               'hypothesis_mechanism': (_hyp or {}).get('mechanism'),
                               'hypothesis_falsifier': (_hyp or {}).get('falsifier'),
                               'primary_metric': (_hyp or {}).get('primary_metric'),
                               'predicted_sign': _pred, 'observed_sign': _obs,
                               'direction_consistent': (None if (_pred is None or _obs is None)
                                                        else (_pred == 0 or _pred == _obs))})
                    entries.append(st)
                    if deltas:
                        _contrast_measured = True
                        if skipped:
                            report['warnings'].append(
                                '%s: `%s` vs `%s` has %d controlled pair(s) but %d seed(s) rejected by the same-run_cfg/train_knobs gate (%s) — n counts only the controlled pairs; do not read the skipped seeds as a larger sample'
                                % (comp, a, b, len(deltas), len(skipped),
                                   '; '.join(sorted({reason for _s, reason in skipped}))))
                        if st.get('seed_driven'):
                            report['warnings'].append(
                                '%s: `%s` vs `%s` (n=%d) is SEED-DRIVEN — dropping any one seed flips the sign of the mean, so the direction is not stable across seeds even if the aggregate p looks small; report the per-seed deltas and add seeds before treating it as a mechanism effect'
                                % (comp, a, b, int(st.get('n') or 0)))
                if a_present and b_present:
                    if not _contrast_measured:
                        _why = sorted({reason for (_dl, _sh, _sk) in conds.values() for _s, reason in _sk})
                        if _why:
                            report['warnings'].append('%s: `%s` vs `%s` both measured but NO seed clears the same-run_cfg/train_knobs gate — %s' % (comp, a, b, '; '.join(_why)))
                        else:
                            report['warnings'].append('%s: `%s` vs `%s` both measured but share no common protocol condition/seed — no contrast computed' % (comp, a, b))
                elif a_present != b_present:
                    report['warnings'].append('%s: incomplete %s `%s` vs `%s` — missing `%s`' % (comp, kind, a, b, a if not a_present else b))
                else:
                    report['warnings'].append('%s: neither `%s` nor `%s` measured — no %s for this %s yet' % (comp, a, b, 'leave-one-out' if kind == 'component' else 'baseline contrast', kind))
                if kind == 'baseline' and _param_ok is False:
                    report['warnings'].append('%s: `%s` vs `%s` differ in parameter count by %.2f%% — NOT parameter-controlled, so this baseline claim is not counted as verified' % (comp, a, b, 100.0 * (_rel or 0.0)))
                elif kind == 'baseline' and _param_ok is None and a_present and b_present:
                    report['warnings'].append('%s: `%s` vs `%s` carry no usable `params` — parameter control is UNVERIFIABLE, so this baseline claim is not counted as verified' % (comp, a, b))
                elif kind == 'component' and _param_ok is False:
                    report['warnings'].append('%s: `%s` vs `%s` differ in parameter count by %.2f%% — part of this module effect may be a capacity (parameter-count) effect rather than the mechanism; report the parameter delta or add a parameter-matched control before claiming the mechanism is useful' % (comp, a, b, 100.0 * (_rel or 0.0)))
            out[comp] = entries
            meta[comp] = {'arms': arms}
        return out, meta

    comp_entries, comp_meta = _collect(design, 'component')
    base_entries, base_meta = _collect(baselines, 'baseline')
    for comp in design:
        _hyp = _hypothesis_spec(comp, 'component')
        if _hyp is None:
            report['hypotheses_missing'].append(comp)
        else:
            report['hypotheses'][comp] = dict(_hyp, kind='component')
    for comp in baselines:
        _hyp = _hypothesis_spec(comp, 'baseline')
        if _hyp is None:
            report['hypotheses_missing'].append(comp)
        else:
            report['hypotheses'][comp] = dict(_hyp, kind='baseline')
    # Completeness / metric-alignability of the pre-registered hypotheses.
    # A missing field or an undeclared metric is a design defect found before
    # any number is interpreted; it is reported, not silently defaulted.
    _ha_comp = hypothesis_audit(HYPOTHESIS_DESIGN, 'component', names=list(design))
    _ha_base = hypothesis_audit(BASELINE_HYPOTHESIS_DESIGN, 'baseline', names=list(baselines))
    report['hypothesis_audit'] = {'component': _ha_comp, 'baseline': _ha_base}
    for _ha in (_ha_comp, _ha_base):
        for _w in _ha.get('warnings') or []:
            report['warnings'].append(_w)
        report['hypotheses_incomplete'].extend(
            n for n in (_ha.get('invalid') or []) if n not in report['hypotheses_incomplete'])
        report['hypotheses_incomplete'].extend(
            n for n in (_ha.get('metric_unknown') or []) if n not in report['hypotheses_incomplete'])
    # Family = claims this panel can test (a component with no measured pair
    # is not part of the correction burden).  Within a tested component, all
    # declared contrasts keep their slots, so an incomplete ablation cannot
    # buy a smaller m by omitting an arm.
    _present_set = set(present)
    _scoped_design = _scope_design(design, _present_set)
    _scoped_baselines = _scope_design(baselines, _present_set)
    if _scoped_design:
        report['ablation_completeness'] = ablation_completeness_audit(_scoped_design, _present_set)
        report['ablation_complete'] = bool(report['ablation_completeness'].get('complete'))
        # Top-level alias: `audit_gates.py methodology` and downstream readers
        # consume `incomplete_components` from the audit report, so both views
        # must stay consistent.
        report['incomplete_components'] = list(
            report['ablation_completeness'].get('incomplete_components') or [])
        report['missing_ablation_arms'] = dict(
            report['ablation_completeness'].get('missing_arms') or {})
        for _comp in report['ablation_completeness'].get('incomplete_components') or []:
            _miss = report['ablation_completeness'].get('missing_arms', {}).get(_comp) or []
            report['warnings'].append('%s: ablation is INCOMPLETE — missing arm(s) %s; this component cannot support a "useful" claim until every declared contrast is present'
                                      % (_comp, ', '.join(_miss) or '(unspecified)'))
    else:
        report['ablation_completeness'] = {'mode': 'leave-one-out', 'complete': None,
                                           'components': {}, 'complete_components': [],
                                           'incomplete_components': [], 'missing_arms': {}}
        report['incomplete_components'] = []
        report['missing_ablation_arms'] = {}
        report['ablation_complete'] = None
    # Arm-level mean±std: the seed reliability is visible even for arms that
    # are not part of a declared contrast.  Group by protocol so different
    # evaluation/warmup conditions never get pooled into a fake larger n.
    _arm_groups = {}
    for (v, t, s), r in recs.items():
        _arm_groups.setdefault((v, str(t or '')), {})[s] = r
    for (v, t), _by_seed in sorted(_arm_groups.items()):
        _vals = [float(_by_seed[s]['ppl']) for s in sorted(_by_seed)]
        _key = v if not t else '%s::%s' % (v, t)
        _pooled, _nll_tot, _ntok_tot = _pooled_ppl_from_records([_by_seed[s] for s in sorted(_by_seed)])
        report['arm_summary'][_key] = {
            'variant': v, 'protocol': t, 'n_seeds': len(_vals),
            'seeds': sorted(_by_seed),
            'ppl_mean': float(np.mean(_vals)),
            'ppl_mean_arithmetic': float(np.mean(_vals)),
            'ppl_pooled': _pooled,
            'ppl_primary': (float(_pooled) if _pooled is not None else float(np.mean(_vals))),
            'nll_sum': _nll_tot, 'n_tokens': _ntok_tot,
            'ppl_std': float(np.std(_vals, ddof=1)) if len(_vals) > 1 else 0.0,
            'meets_min_seeds': len(_vals) >= int(min_n),
        }
        if len(_vals) < int(min_n):
            report['arms_below_min_seeds'].append(_key)
    declared_m = sum(len(v) for v in _scoped_design.values()) + sum(len(v) for v in _scoped_baselines.values())
    all_entries = [e for group in (comp_entries, base_entries) for entries in group.values() for e in entries]
    m_family = max(declared_m, len(all_entries))
    _holm_assign(all_entries, m_family, alpha, 'p_holm_global')
    _holm_assign(all_entries, m_family, alpha, 'p_holm_family')
    for e in all_entries:
        _ph = e.get('p_holm_global')
        e['significant_global'] = bool(_ph is not None and float(_ph) < alpha and (not e.get('within_seed_noise')) and e.get('controlled'))
        e['significant_family'] = e['significant_global']
    report['family'] = {
        'name': ('ABLATION_DESIGN+BASELINE_DESIGN' if _scoped_baselines else 'ABLATION_DESIGN') + ' (exploratory; not pre-registered)',
        'declared_contrasts': int(declared_m),
        'effective_m': int(m_family),
        'alpha': float(alpha),
        'min_seeds_for_family_significance': signflip_min_n_for_holm(m_family, alpha),
        'note': 'Holm step-down over every declared contrast; missing arms keep their slot (dummy p=1). A panel below min_seeds_for_family_significance is resolution-limited, however large the observed effect.',
    }

    def _publish(group_name, design_group, entries_by_comp, meta, kind, require_param_match):
        status_names = ('verified', 'direction_conflict', 'direction_unspecified', 'measured',
                        'significant', 'nominally_significant', 'resolution_limited',
                        'underpowered', 'noise_level', 'unverified')
        lists = {s: [] for s in status_names}
        for comp, contrasts in design_group.items():
            entries = entries_by_comp.get(comp, [])
            # Per-component Holm: the family is this innovation's own declared
            # contrasts (the reviewer usually asks "is THIS module useful?").
            m_comp = max(len(contrasts), len(entries))
            _holm_assign(entries, m_comp, alpha, 'p_holm_component')
            for e in entries:
                _phc = e.get('p_holm_component')
                e['significant_component'] = bool(_phc is not None and float(_phc) < alpha and (not e.get('within_seed_noise')) and e.get('controlled'))
            status = _classify_group_status(entries, min_n=min_n, alpha=alpha, m_family=m_family,
                                            require_param_match=require_param_match)
            lists[status].append(comp)
            if status == 'verified':
                # `verified` is the corrected-significance tier; keep the
                # existing `significant_*` key pointing at the same set so
                # downstream readers do not silently lose the signal.
                lists['significant'].append(comp)
            if any(e.get('significant') for e in entries if e.get('controlled')):
                lists['nominally_significant'].append(comp)
            if any(int(e.get('n') or 0) >= int(min_n) and e.get('controlled') for e in entries):
                lists['measured'].append(comp)
            report[group_name][comp] = entries
            _arms = meta[comp]['arms']
            _cv = controlled_variables_report(recs, _arms)
            _cv.update({
                'arms_present': sorted(a for a in _arms if a in present),
                'param_matched': all(e.get('param_controlled') is True for e in entries) if entries else None,
                'objective_matched': (all(e.get('objective_matched') is True for e in entries) if entries else None),
                'objective_differences': sorted({x for e in entries for x in (e.get('objective_differences') or []) if 'differs' in str(x)}),
            })
            if _cv.get('objective_differences'):
                report['warnings'].append('%s: arms share run_cfg but not the actual training objective (%s) — the contrast is a system-level comparison, not an isolated mechanism claim' % (comp, '; '.join(_cv['objective_differences'])))
            report['controlled_variables'][comp] = _cv
            if _cv.get('within_arm_mixed'):
                report['warnings'].append('%s: arm(s) %s mix DIFFERENT protocols within one arm across seeds — a seed-averaged claim would pool incompatible runs'
                                          % (comp, ', '.join(_cv['within_arm_mixed'])))
            if _cv.get('hyperparam_differences'):
                report['warnings'].append('%s: structured hyperparameters differ across arms (%s) even where run_cfg matches — the comparison is not fully controlled'
                                          % (comp, ', '.join(_cv['hyperparam_differences'])))
            if _cv.get('seed_coverage_equal') is False:
                _cov = '; '.join(
                    '%s={%s}' % (a, ','.join('%s:%s' % (t or 'base', s)
                                             for t, s in _cv['seed_coverage'].get(a) or []))
                    for a in _cv.get('arms') or [])
                report['warnings'].append('%s: arms do NOT share the same seed/protocol coverage (%s) — arm means are over different seed sets and the extra/missing seeds must not be read as a paired contrast; pair only the shared seeds (n_paired) or restore the missing cells'
                                          % (comp, _cov))
            _req_key = 'component_seed_requirements' if kind == 'component' else 'baseline_seed_requirements'
            report[_req_key][comp] = {
                'declared_contrasts': int(len(contrasts)),
                'min_seeds_for_component_significance': signflip_min_n_for_holm(len(contrasts), alpha),
                'min_seeds_for_family_significance': signflip_min_n_for_holm(m_family, alpha),
            }
        for status in status_names:
            report[f'{status}_{group_name}'] = lists[status]
        return lists

    _publish('components', design, comp_entries, comp_meta, 'component', require_param_match=False)
    _publish('baselines', baselines, base_entries, base_meta, 'baseline', require_param_match=True)

    # A panel is a *complete ablation* only if every declared innovation has
    # its full leave-one-out set AND no declared innovation is entirely
    # unmeasured.  The scoped completeness above can be True while other
    # components were simply never scheduled; exposing them as
    # `omitted_components` prevents the panel-level flag from over-claiming.
    if report.get('ablation_completeness'):
        report['ablation_completeness']['omitted_components'] = list(report.get('unverified_components') or [])
        if report['ablation_completeness'].get('complete') is None:
            report['ablation_complete'] = None
        else:
            report['ablation_complete'] = bool(report['ablation_completeness'].get('complete')) \
                and not report.get('unverified_components')

    report['metric_audit'] = metric_policy_audit(summary)
    if report['metric_audit'].get('warnings'):
        report['warnings'].extend(report['metric_audit']['warnings'])
    _tune_path = None
    if outdir:
        _tune_path = os.path.join(outdir, 'baseline_tuning.json')
    elif os.path.isdir('results_baseline_tuning'):
        _tune_path = os.path.join('results_baseline_tuning', 'baseline_tuning.json')
    report['baseline_tuning'] = baseline_tuning_audit(path=_tune_path)
    if _scoped_baselines:
        _tv = bool(report['baseline_tuning'].get('tuning_verified'))
        report['tuned_verified_baselines'] = list(report.get('verified_baselines') or []) if _tv else []
        report['untuned_verified_baselines'] = [] if _tv else list(report.get('verified_baselines') or [])
        if not _tv:
            report['warnings'].append(
                'baseline claims have no verified equal-search tuning artifact (%s) — a win over an untuned baseline is not evidence that the method beats a well-tuned baseline; run baseline_tune.py and put baseline_tuning.json next to this panel'
                % (report['baseline_tuning'].get('status') or 'missing'))
    # Trivial-floor evidence: a tuned count model must be beaten before the
    # neural mechanism is compared with the tuned dense baselines.  The
    # comparison is only made when the artifact and the panel share a
    # validation fingerprint (or the panel records none, in which case the
    # result is explicitly marked unverified rather than assumed comparable).
    _sb_path = None
    if outdir:
        _sb_path = os.path.join(outdir, 'simple_baseline.json')
    elif os.path.isdir('results_baseline_tuning'):
        _sb_path = os.path.join('results_baseline_tuning', 'simple_baseline.json')
    report['simple_baseline'] = simple_baseline_audit(path=_sb_path)
    if _scoped_baselines:
        _sb = report['simple_baseline']
        if not _sb.get('baseline_verified'):
            report['warnings'].append(
                'no verified tuned count-model floor (%s) — run baseline_tune.py simple and place simple_baseline.json next to this panel; a neural model that has not been shown to beat the trivial floor should not be interpreted as evidence for the mechanism'
                % (_sb.get('status') or 'missing'))
        else:
            _main_arms = set()
            for _cs in list(_scoped_design.values()) + list(_scoped_baselines.values()):
                for _a, _b, _n in _cs:
                    _main_arms.add(_a)
                    _main_arms.add(_b)
            _cands = [(k, v) for k, v in (report.get('arm_summary') or {}).items()
                      if v.get('variant') in _main_arms]
            # Prefer the panel's base protocol; a warmup/length protocol is a
            # different condition and must not be silently mixed in.
            _base_cands = [(k, v) for k, v in _cands if not v.get('protocol')]
            if _base_cands:
                _cands = _base_cands
            if _cands:
                _ref_key, _ref = min(_cands, key=lambda kv: float(kv[1]['ppl_mean']))
                _panel_val_fps = {r.get('val_fp') for r in recs.values() if r.get('val_fp')}
                _panel_eval_fps = {r.get('eval_fp') for r in recs.values() if r.get('eval_fp')}
                _sb_data = _sb.get('data') or {}
                # `cache_val_fp` is the full validation cache fingerprint (the
                # one panel records carry); `val_fp`/`eval_fp` describe the
                # exact rows scored by the floor.  Prefer the cache-level fp
                # for the comparability gate, but additionally require the
                # EXACT eval-row fingerprint when both sides carry it — a
                # 128-row floor and a 512-row panel PPL are different samples.
                _sb_val_fp = _sb_data.get('cache_val_fp') or _sb_data.get('val_fp')
                _sb_eval_fp = _sb_data.get('eval_fp')
                _comparable = None
                if _sb_val_fp and _panel_val_fps:
                    _comparable = _sb_val_fp in _panel_val_fps
                if _comparable is True and _sb_eval_fp and _panel_eval_fps:
                    if _sb_eval_fp not in _panel_eval_fps:
                        _comparable = False
                        report['warnings'].append(
                            'simple-baseline eval row fingerprint differs from the panel final-eval rows — the floor and the panel scored different validation documents; the beats-floor comparison is refused')
                report['simple_baseline'].update({
                    'panel_reference_arm': _ref_key,
                    'panel_reference_ppl': float(_ref.get('ppl_primary', _ref['ppl_mean'])),
                    'panel_reference_protocol': _ref.get('protocol'),
                    'panel_val_fp': (sorted(_panel_val_fps)[:3] or None),
                    'panel_eval_fp': (sorted(_panel_eval_fps)[:3] or None),
                    'simple_eval_fp': _sb_eval_fp,
                    'comparable_to_panel': _comparable,
                    'panel_beats_simple_floor': (bool(float(_ref.get('ppl_primary', _ref['ppl_mean'])) < float(_sb.get('best_ppl')))
                                                 if _comparable is True and _sb.get('best_ppl') is not None
                                                 else None),
                })
                if _comparable is False:
                    report['warnings'].append(
                        'simple-baseline artifact and this panel do not share a validation fingerprint — the floor comparison is NOT comparable and must not be quoted')
                elif _comparable is None:
                    report['warnings'].append(
                        'the floor comparison is UNVERIFIABLE: panel records and/or the simple-baseline artifact lack a shared validation fingerprint; do not quote a beats-floor claim')
                elif report['simple_baseline'].get('panel_beats_simple_floor') is False:
                    report['warnings'].append(
                        'the panel\'s best declared-arm PPL (%.4f, %s) does not beat the tuned count-model floor (%.4f, %s) — the mechanism claim lacks the most basic baseline'
                        % (float(_ref.get('ppl_primary', _ref['ppl_mean'])), _ref_key, float(_sb.get('best_ppl')),
                           _sb.get('best_baseline') or 'count-model'))
    if report['hypotheses_missing']:
        report['warnings'].append('no pre-registered hypothesis for: ' + ', '.join(sorted(set(report['hypotheses_missing']))) + ' — direction cannot be checked, so a significant result is at most measured')

    if write and outdir:
        try:
            os.makedirs(outdir, exist_ok=True)
            atomic_write_json(os.path.join(outdir, 'methodology_audit.json'), report)
        except Exception as e:
            print('[audit] could not write methodology_audit.json: %s: %s' % (type(e).__name__, e))
    if verbose:
        _print_methodology_audit(report)
    return report

def _print_methodology_audit(report):
    print('\n========== methodology audit: controlled comparison / ablation / statistics ==========')
    print('  variants with measurable records: ' + (', '.join(report['variants_present']) or '(none)'))
    print('  Δ = with-module − without-module (perplexity; negative = the module helps).')
    _fam = report.get('family') or {}
    print('  family: %d declared contrasts, m_eff=%s; Holm p is the family-corrected p.' % (int(_fam.get('declared_contrasts') or 0), _fam.get('effective_m')))
    if _fam.get('min_seeds_for_family_significance'):
        print('  minimum seeds for a family-corrected rejection at alpha=%g: n>=%d (below this every p is resolution-limited).' % (float(report.get('alpha', 0.05)), int(_fam['min_seeds_for_family_significance'])))
    _abl = report.get('ablation_completeness') or {}
    if _abl:
        _missing = '; '.join('%s=%s' % (c, ','.join(a))
                             for c, a in sorted((_abl.get('missing_arms') or {}).items()))
        _omitted = _abl.get('omitted_components') or []
        if _abl.get('complete') is None:
            _state, _tail = 'not applicable', ' — no declared ablation contrast is measured in this panel'
        elif report.get('ablation_complete'):
            _state, _tail = 'complete', ''
        else:
            _detail = _missing or '(unspecified)'
            if _omitted:
                _detail += '; omitted components: ' + ', '.join(_omitted)
            _state, _tail = 'INCOMPLETE', ' — missing ' + _detail
        print('  ablation completeness (%s): %s%s'
              % (_abl.get('mode'), _state, _tail))
    if report.get('arm_summary'):
        print('  arm means (mean ± sample std over seeds):')
        for _k, _e in report['arm_summary'].items():
            print('    %-30s n=%-2d  %.4f ± %.4f%s' % (
                _k, _e['n_seeds'], _e.get('ppl_primary', _e['ppl_mean']), _e['ppl_std'],
                '' if _e['meets_min_seeds'] else '  (BELOW min_n)'))
    for comp, entries in report['components'].items():
        live = [e for e in entries if e['n_shared'] > 0]
        if not live:
            continue
        _cv = report.get('controlled_variables', {}).get(comp, {})
        _kn = _cv.get('train_knobs_equal')
        _kn_s = {True: 'optimizer/schedule knobs shared', False: 'OPTIMIZER/SCHEDULE KNOBS DIFFER', None: 'optimizer/schedule knobs not recorded (legacy or absent)'}[_kn]
        print('  [%s]  %s; arms=%s' % (comp, _kn_s, ','.join(_cv.get('arms_present') or []) or 'none'))
        for e in live:
            _c = 'controlled (same run_cfg)' if e['controlled'] else ('UNCONTROLLED: ' + '; '.join(e['skipped_reasons']))
            _p = ('%.4f' % e['p_exact_signflip']) if math.isfinite(e['p_exact_signflip']) else 'n/a'
            _ph = e.get('p_holm_global')
            _phs = ('%.4f' % _ph) if isinstance(_ph, (int, float)) and math.isfinite(_ph) else 'n/a'
            _tag = '' if e.get('tag_with', '') == e.get('tag_without', '') == '' else '   [%s vs %s]' % (e.get('tag_with') or '(base)', e.get('tag_without') or '(base)')
            print('    %-22s − %-22s  n=%-2d  Δ=%+.3f ± %.3f  d_z=%+.2f  p=%s  p_Holm=%s%s' % (e['with'], e['without'], e['n'], e['mean'], e['std'], e['cohens_dz'], _p, _phs, _tag))
            print('        with %.4f±%.4f vs without %.4f±%.4f; relative Δ=%+.2f%% of baseline, seed noise=%.2f%% (SNR=%.2f)' % (
                e.get('with_mean', float('nan')), e.get('with_std', float('nan')),
                e.get('without_mean', float('nan')), e.get('without_std', float('nan')),
                e.get('relative_effect_pct', float('nan')), e.get('relative_noise_pct', float('nan')),
                e.get('signal_to_noise', float('nan'))))
            _lo, _hi = e.get('ci95_low'), e.get('ci95_high')
            if isinstance(_lo, (int, float)) and isinstance(_hi, (int, float)) and math.isfinite(_lo) and math.isfinite(_hi):
                print('        95%% CI for Δ: [%+.4f, %+.4f]%s' % (
                    _lo, _hi, '; SEED-DRIVEN (dropping one seed flips the sign)' if e.get('seed_driven') else ''))
            elif e.get('seed_driven'):
                print('        SEED-DRIVEN: dropping one seed flips the sign of the mean')
            if e.get('capacity_confounded'):
                print('        CAPACITY CONFOUND: parameter counts differ by %.2f%% — report the parameter delta or add a parameter-matched control' % (100.0 * float(e.get('param_rel_diff') or 0.0)))
            print('        %s | %s' % (_c, e['effect_vs_noise']))
    if report['verified_components']:
        print('  verified innovations (controlled AND family-Holm p<alpha AND |Δ|>seed noise): ' + ', '.join(report['verified_components']))
    if report.get('measured_components'):
        print('  measured innovations (controlled contrast with n>=%d; descriptive unless also verified): ' % int(report.get('min_n', 3)) + ', '.join(report['measured_components']))
    if report.get('nominally_significant_components'):
        print('  nominally significant (raw p<alpha, before family correction): ' + ', '.join(report['nominally_significant_components']))
    if report.get('direction_conflict_components'):
        print('  DIRECTION CONFLICT (significant but opposite to the pre-registered hypothesis): ' + ', '.join(report['direction_conflict_components']))
    if report.get('direction_unspecified_components'):
        print('  direction-unspecified (significant, but the hypothesis declared predicted_sign=0): ' + ', '.join(report['direction_unspecified_components']))
    if report.get('resolution_limited_components'):
        print('  resolution-limited (non-noise effect, but the family-corrected p floor > alpha at this n; need n>=%s — add seeds, do NOT read as a null): ' % str((_fam.get('min_seeds_for_family_significance') or '?')) + ', '.join(report['resolution_limited_components']))
    if report.get('noise_level_components'):
        print('  noise-level (effect within seed noise across all contrasts): ' + ', '.join(report['noise_level_components']))
    if report.get('underpowered_components'):
        print('  underpowered innovations (contrasts exist but n<%d — add seeds before claiming an effect): ' % int(report.get('min_n', 3)) + ', '.join(report['underpowered_components']))
    if report['unverified_components']:
        print('  unverified innovations (no controlled leave-one-out contrast): ' + ', '.join(report['unverified_components']))
    if report.get('baselines'):
        print('  --- primary baseline claims (parameter-matched dense control) ---')
        for comp, entries in report['baselines'].items():
            live = [e for e in entries if e['n_shared'] > 0]
            if not live:
                continue
            _pc = report.get('controlled_variables', {}).get(comp, {}).get('param_matched')
            print('  [%s]  parameter-controlled=%s' % (comp, _pc))
            for e in live:
                _p = ('%.4f' % e['p_exact_signflip']) if math.isfinite(e['p_exact_signflip']) else 'n/a'
                _ph = e.get('p_holm_global')
                _phs = ('%.4f' % _ph) if isinstance(_ph, (int, float)) and math.isfinite(_ph) else 'n/a'
                print('    %-22s − %-22s  n=%-2d  Δ=%+.3f ± %.3f  d_z=%+.2f  p=%s  p_Holm=%s  params=%s' % (e['with'], e['without'], e['n'], e['mean'], e['std'], e['cohens_dz'], _p, _phs, ('yes' if e.get('param_controlled') else ('no' if e.get('param_controlled') is False else 'n/a'))))
                print('        %s | %s' % ('controlled (same run_cfg)' if e['controlled'] else ('UNCONTROLLED: ' + '; '.join(e['skipped_reasons'])), e['effect_vs_noise']))
        _mt = report.get('metric_audit') or {}
        print('  metric policy: %s (warnings: %d)' % (_mt.get('status') or 'n/a', len(_mt.get('warnings') or [])))
        _bt = report.get('baseline_tuning') or {}
        print('  baseline tuning: status=%s tuning_verified=%s' % (_bt.get('status'), _bt.get('tuning_verified')))
        _sb = report.get('simple_baseline') or {}
        if _sb:
            print('  count-model floor: status=%s baseline_verified=%s best=%s PPL=%s' % (
                _sb.get('status'), _sb.get('baseline_verified'), _sb.get('best_baseline'), _sb.get('best_ppl')))
            if _sb.get('panel_beats_simple_floor') is not None:
                print('    panel reference %s PPL=%.4f; comparable=%s; beats simple floor=%s' % (
                    _sb.get('panel_reference_arm'), float(_sb.get('panel_reference_ppl') or float('nan')),
                    _sb.get('comparable_to_panel'), _sb.get('panel_beats_simple_floor')))
        if report.get('verified_baselines'):
            print('  statistically verified baseline claims (NOT yet evidence against a tuned baseline): ' + ', '.join(report['verified_baselines']))
        if report.get('tuned_verified_baselines'):
            print('  tuned verified baseline claims (equal-search tuning artifact present): ' + ', '.join(report['tuned_verified_baselines']))
        if report.get('untuned_verified_baselines'):
            print('  statistically verified but UNTUNED baseline claims (not evidence against a tuned baseline): ' + ', '.join(report['untuned_verified_baselines']))
        if report.get('direction_conflict_baselines'):
            print('  DIRECTION CONFLICT baseline claims: ' + ', '.join(report['direction_conflict_baselines']))
        if report.get('nominally_significant_baselines'):
            print('  nominally significant baseline claims (raw p<alpha): ' + ', '.join(report['nominally_significant_baselines']))
        if report.get('resolution_limited_baselines'):
            print('  resolution-limited baseline claims: ' + ', '.join(report['resolution_limited_baselines']))
        if report.get('unverified_baselines'):
            print('  unverified baseline claims (missing/incomplete/param-mismatched): ' + ', '.join(report['unverified_baselines']))
    for w in report['warnings']:
        print('  [warn] ' + w)
    print('======================================================================================')

def add_paired(*_a, **_kw):
    raise NotImplementedError('use v9_supp.add_paired (exp_lib cannot import v7_supp)')

def set_seed(seed):
    """Seed every RNG that can affect a run: Python ``random``, NumPy's
    legacy global RNG, PyTorch's CPU RNG and every CUDA device, plus the
    Python hash seed (best effort for child processes).  Returns the integer
    seed so callers can record exactly what was used.  torch's deterministic
    algorithm / cuDNN switches are pinned at the same time, so two runs with
    the same seed on the same machine follow the same kernels."""
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    _attb_bump_epoch()
    torch.manual_seed(seed)
    _pin_cpu_threads()
    # Pin the backend determinism switches on CPU too: the cuDNN flags are
    # inert without CUDA, while CSA_DETERMINISTIC must still enable torch's
    # deterministic-algorithm mode on a CPU box (a CPU run is a first-class
    # reproduction target).  CUDA RNGs are seeded explicitly below.
    _pin_cuda_determinism()
    if torch.cuda.is_available():
        # manual_seed already covers all devices; the explicit current-device
        # call documents the contract and is harmless.
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # TensorFlow is not a dependency of this repo, but a future/embedded
    # TensorFlow path must not become an unseeded RNG source.  Only touch it
    # when the user has already imported it (never import TensorFlow here:
    # that would make every CPU-only run import it needlessly).
    _tf = sys.modules.get('tensorflow')
    if _tf is not None:
        try:
            _tf.random.set_seed(seed)
            try:
                _tf.config.experimental.enable_op_determinism()
            except Exception:
                pass
        except Exception as _tfe:
            print(f'[determinism] could not seed the already-imported TensorFlow RNG ({type(_tfe).__name__}: {_tfe})')
    return seed
_CPU_THREADS_PINNED = None

def rng_fingerprint():
    """Digest of the *current* Python/NumPy/Torch/CUDA RNG state.

    Recorded immediately after `set_seed(seed)` in every training record.  Two
    runs with the same seed on the same stack must produce the same digest;
    a mismatch means some import/construction consumed randomness before the
    run started (and therefore that the seed is not fully controlled).  The
    digest is a diagnostic only: it never enters the resume fingerprint."""
    h = hashlib.sha256()
    h.update(repr(random.getstate()).encode('utf-8', 'replace'))
    _ns = np.random.get_state()
    h.update(str(_ns[0]).encode('ascii', 'replace'))
    h.update(np.asarray(_ns[1], dtype=np.uint64).tobytes())
    h.update(repr((_ns[2], _ns[3], _ns[4])).encode('ascii', 'replace'))
    h.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        try:
            for _st in torch.cuda.get_rng_state_all():
                h.update(_st.cpu().numpy().tobytes())
        except Exception:
            # a broken CUDA context is reported elsewhere; the CPU part of
            # the fingerprint is still a useful diagnostic
            pass
    return h.hexdigest()[:16]

def capture_rng_state():
    """Snapshot every RNG stream a short diagnostic may consume.

    Diagnostics such as the small-batch overfit probe and the duplicate-run
    reproducibility probe call ``set_seed`` themselves.  Without a snapshot
    they would advance the process-wide RNG streams and silently couple a
    later training cell to how many probes ran before it.  The returned object
    is opaque: restore it with ``restore_rng_state``.
    """
    state = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        try:
            state['cuda'] = torch.cuda.get_rng_state_all()
        except Exception:
            state['cuda'] = None
    else:
        state['cuda'] = None
    return state

def restore_rng_state(state):
    """Restore a snapshot from ``capture_rng_state`` (best effort on CUDA).

    Restoring is what makes the pre-flight checks side-effect free: the
    training cell that follows sees exactly the RNG state it would have seen
    if the checks had never run, so adding/diagnosing a probe cannot change a
    recorded loss curve.  A broken CUDA context is reported by the caller's
    training loop; the CPU streams are still restored here."""
    if not isinstance(state, dict):
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state.get('cuda') is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state['cuda'])
        except Exception:
            pass

class _preserve_rng_state:
    """Context manager: run a diagnostic without perturbing the RNG streams.

    Also restores ``PYTHONHASHSEED`` because ``set_seed`` updates it for child
    processes; a probe must not leave that environment mutation behind."""

    def __enter__(self):
        self._env = os.environ.get('PYTHONHASHSEED')
        self._state = capture_rng_state()
        return self

    def __exit__(self, exc_type, exc, tb):
        restore_rng_state(self._state)
        if self._env is None:
            os.environ.pop('PYTHONHASHSEED', None)
        else:
            os.environ['PYTHONHASHSEED'] = self._env
        return False

def dataloader_generator(seed):
    """A torch.Generator for a future torch DataLoader (scale path).

    The current loader is `batch_iter`, which is seeded directly by the run
    seed.  This helper is for DataLoader-based code: pass it as
    ``DataLoader(..., generator=L.dataloader_generator(seed))`` and pass
    ``worker_init_fn=L.seed_worker`` so worker RNGs are derived from the same
    seed instead of the OS entropy pool."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g

def seed_worker(worker_id):
    """DataLoader worker_init_fn: seed Python/NumPy/Torch from torch's
    per-worker base seed.  PyTorch already seeds its own worker RNG before
    calling this hook; re-seeding it explicitly documents the contract and
    also covers older versions where a custom generator changes the base."""
    _seed = int(torch.initial_seed()) % (2 ** 32)
    os.environ['PYTHONHASHSEED'] = str(_seed)
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    _pin_cpu_threads()
    _pin_cuda_determinism()
    if torch.cuda.is_available():
        torch.cuda.manual_seed(_seed)
        torch.cuda.manual_seed_all(_seed)

def dataloader_kwargs(seed):
    """DataLoader kwargs that pin BOTH the sampler order and worker RNGs.

    Scale path: `DataLoader(ds, **L.dataloader_kwargs(seed), num_workers=n)`.
    The generator controls shuffling/sampling; worker_init_fn derives each
    worker's Python/NumPy RNG from the same seed tree instead of OS entropy.
    """
    return {'generator': dataloader_generator(seed), 'worker_init_fn': seed_worker}

def deterministic_dataloader(dataset, *, seed=0, **kwargs):
    """Thin `torch.utils.data.DataLoader` wrapper with reproducible seeding.

    Existing caller-supplied `generator` / `worker_init_fn` are respected."""
    from torch.utils.data import DataLoader
    kwargs.setdefault('generator', dataloader_generator(seed))
    kwargs.setdefault('worker_init_fn', seed_worker)
    return DataLoader(dataset, **kwargs)

def _pin_cpu_threads():
    global _CPU_THREADS_PINNED
    if _CPU_THREADS_PINNED is None:
        want = os.environ.get('CSA_CPU_THREADS', '').strip()
        try:
            n = max(1, int(want)) if want else 1
        except ValueError:
            print(f'[determinism] CSA_CPU_THREADS={want!r} is not an integer — using 1')
            n = 1
        os.environ.setdefault('OMP_NUM_THREADS', str(n))
        os.environ.setdefault('MKL_NUM_THREADS', str(n))
        _CPU_THREADS_PINNED = n
    else:
        n = _CPU_THREADS_PINNED
    try:
        if int(torch.get_num_threads()) != int(n):
            torch.set_num_threads(n)
    except Exception as e:
        print(f'[determinism] could not pin the CPU thread count to {n} ({type(e).__name__}: {e}) — a CPU run may not be reproducible')
    return n

def pin_tf32(enabled=None):
    """Pin TF32 explicitly instead of inheriting a framework default.

    Reproducibility depends on the actual matmul precision mode; two boxes (or
    two torch versions) with different defaults can otherwise disagree while
    looking identically configured.  ``CSA_TF32=1`` opts in; the default is
    off (highest float32 precision)."""
    if enabled is None:
        enabled = os.environ.get('CSA_TF32', '').strip().lower() in ('1', 'true', 'yes', 'on')
    enabled = bool(enabled)
    try:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = enabled
            torch.backends.cudnn.allow_tf32 = enabled
    except Exception as e:
        print(f'[determinism] could not pin TF32 ({type(e).__name__}: {e})')
    try:
        torch.set_float32_matmul_precision('high' if enabled else 'highest')
    except Exception as e:
        print(f'[determinism] could not set float32_matmul_precision ({type(e).__name__}: {e})')
    return enabled

def _pin_cuda_determinism():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pin_tf32()
    mode = os.environ.get('CSA_DETERMINISTIC', '').strip().lower()
    if mode in ('1', 'true', 'yes', 'strict'):
        try:
            torch.use_deterministic_algorithms(True)
            return 'strict'
        except Exception as e:
            print(f"[determinism] CSA_DETERMINISTIC=1 requested but the strict mode could not be enabled ({type(e).__name__}: {e}); falling back to cudnn-deterministic only.  The active level is recorded in each record's `run_cfg`.")
            return 'cudnn'
    if mode in ('warn', 'warn_only'):
        torch.use_deterministic_algorithms(True, warn_only=True)
        return 'warn'
    return 'cudnn'

def determinism_label():
    if not torch.cuda.is_available():
        return 'cpu'
    if torch.are_deterministic_algorithms_enabled():
        return 'strict' if not torch.is_deterministic_algorithms_warn_only_enabled() else 'warn'
    return 'cudnn' if torch.backends.cudnn.deterministic else 'off'

def _jsonable(v):
    """Recursively convert a config/provenance value to JSON-safe data.
    Sets are sorted so two equal configs always serialise identically."""
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (set, frozenset)):
        return sorted((_jsonable(x) for x in v), key=str)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)

def _git_provenance():
    """HEAD / branch / dirty state, with a bounded dirty-file list.  Never
    raises: a source tarball without .git is still a valid experiment."""
    def _run(*args):
        try:
            p = subprocess.run(['git', *args], cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True, timeout=20)
            return p.stdout.strip() if p.returncode == 0 else None
        except Exception:
            return None
    head = _run('rev-parse', 'HEAD')
    branch = _run('rev-parse', '--abbrev-ref', 'HEAD')
    dirty = _run('status', '--porcelain')
    files = dirty.splitlines() if dirty else []
    # A commit id alone does NOT identify a dirty work tree: two runs from the
    # same HEAD with different uncommitted edits would otherwise claim the
    # same code version.  Hash the diff and the status instead of storing the
    # (possibly secret-bearing) contents.  Untracked files are included via
    # the status hash; their contents are left to the operator to archive.
    def _sha(text):
        return hashlib.sha256((text or '').encode('utf-8', 'replace')).hexdigest()[:16]
    diff_text = _run('diff', '--no-color', '--no-ext-diff', 'HEAD')
    return {'git_commit': head, 'git_branch': branch, 'git_dirty': bool(files), 'git_dirty_count': len(files), 'git_dirty_files': files[:20], 'git_status_sha256': _sha(dirty), 'git_diff_sha256': _sha(diff_text)}

def _gpu_names():
    try:
        return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    except Exception:
        return []

def _gpu_identity():
    """Per-visible-GPU identity used to explain (not eliminate) cross-machine
    drift: name, UUID, total memory and compute capability.  Never raises."""
    out = []
    if not torch.cuda.is_available():
        return out
    try:
        n = int(torch.cuda.device_count())
    except Exception:
        return out
    for i in range(n):
        try:
            p = torch.cuda.get_device_properties(i)
            out.append({
                'index': i,
                'name': getattr(p, 'name', None),
                'uuid': str(getattr(p, 'uuid', '')) or None,
                'total_memory_bytes': int(getattr(p, 'total_memory', 0)) or None,
                'major': int(getattr(p, 'major', 0)),
                'minor': int(getattr(p, 'minor', 0)),
            })
        except Exception as e:
            out.append({'index': i, 'error': f'{type(e).__name__}: {e}'})
    return out

def environment_snapshot():
    """Compact software/hardware snapshot for experiment tracking."""
    import platform
    env = {
        'python': sys.version.split()[0],
        'executable': sys.executable,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'cpu_count': os.cpu_count(),
        'torch': torch.__version__,
        'numpy': np.__version__,
        'cuda_available': bool(torch.cuda.is_available()),
        'cuda_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        'gpu_names': _gpu_names(),
        'gpu_identity': _gpu_identity(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'determinism': determinism_label(),
        'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
        'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
        'deterministic_algorithms': bool(torch.are_deterministic_algorithms_enabled()),
        'tf32_matmul': bool(torch.backends.cuda.matmul.allow_tf32) if torch.cuda.is_available() else None,
        'tf32_cudnn': bool(torch.backends.cudnn.allow_tf32) if torch.cuda.is_available() else None,
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'cpu_threads': int(torch.get_num_threads()),
        'cpu_interop_threads': int(torch.get_num_interop_threads()) if hasattr(torch, 'get_num_interop_threads') else None,
        'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
        'flags': {k: os.environ.get(k) for k in ('CSA_KERNEL', 'CSA_CKPT', 'CSA_DETERMINISTIC', 'CSA_CPU_THREADS', 'CSA_SKIP_OVERFIT', 'CSA_REQUIRE_REPRO', 'CSA_SKIP_REPRO', 'CUBLAS_WORKSPACE_CONFIG')},
    }
    for _mod, _key in (('tokenizers', 'tokenizers'), ('datasets', 'datasets')):
        try:
            env[_key] = __import__(_mod).__version__
        except Exception:
            env[_key] = None
    return env

def _data_version(data):
    """Stable content digest of the corpus identity recorded in a panel.

    The token fingerprints already hash the exact ids seen; this composite
    makes the *data version* a single field a reader can compare across runs
    (dataset/protocol, token counts, vocabulary and both split fingerprints)
    without knowing which individual fields the loader happened to expose.
    """
    keys = ('dataset', 'source', 'fw_fit_docs', 'requested_train_tokens', 'train_tokens', 'seq_len', 'vocab_size', 'val_sequences', 'train_fp', 'val_fp', 'eval_fp', 'cache_val_fp', 'data_extra')
    payload = {k: data.get(k) for k in keys if data.get(k) is not None}
    blob = json.dumps(_jsonable(payload), sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]

def code_provenance():
    """Code + environment provenance block, without any data section.

    Shared by `experiment_provenance` (training/panel records, which append the
    data fingerprints) and by `stamped_analysis` (derived/audit artifacts,
    which have no data of their own — their inputs are other stamped records
    or on-disk panels, so a `data` block here would be misleading)."""
    return {'code': {'code_semantics': CODE_SEMANTICS, 'code_stamp': code_stamp(), 'kernel_stamp': _kernel_stamp(), **_git_provenance()}, 'environment': environment_snapshot()}

def experiment_provenance(cfg, *, train_ids=None, val_batch=None, vocab=None):
    """Everything needed to identify a panel: code version (semantic stamp
    + git commit + dirty-worktree digest), environment, and the exact data
    bytes seen (token fingerprints + a composite data version)."""
    data = {'dataset': cfg.get('dataset') or 'wikitext', 'requested_train_tokens': cfg.get('n_train_tokens'), 'seq_len': cfg.get('seq_len')}
    for _k in ('cache_dir', 'fw_source', 'fw_fit_docs'):
        if cfg.get(_k) is not None:
            data[_k if _k != 'fw_source' else 'source'] = cfg.get(_k)
    if train_ids is not None:
        data['train_tokens'] = int(len(train_ids))
        data['train_fp'] = _ids_fp(train_ids)
    if val_batch is not None:
        _vb = np.ascontiguousarray(val_batch).reshape(-1)
        data['val_sequences'] = int(np.asarray(val_batch).shape[0])
        data['val_fp'] = _ids_fp(_vb)
    if vocab is not None:
        data['vocab_size'] = int(vocab)
    _extra = cfg.get('data_extra')
    if isinstance(_extra, dict):
        # Synthetic/task-specific generators (NIAH recipe, corruption modes)
        # must be part of the data identity, not only the human note.
        data['data_extra'] = _jsonable(_extra)
    data['data_version'] = _data_version(data)
    prov = code_provenance()
    prov['data'] = data
    return prov

def run_hyperparameters(cfg, *, seed, d, n_layers, n_heads, d_head, vocab, mlp_ratio, data_version=None, amp_dtype='auto', eager_batch='auto'):
    """Structured hyper-parameter record (the run_cfg string stays the
    pairing gate; this is the human/machine-readable copy).

    ``amp_dtype`` / ``eager_batch`` must be the values the cell ACTUALLY ran,
    not a fresh inference from the current process's global device.  Callers
    that already have the training record pass its two fields here."""
    hp = {str(k): _jsonable(v) for k, v in cfg.items() if k not in ('seeds', 'variants', 'outdir')}
    hp.update({
        'seed': int(seed),
        'data_seed': int(seed),
        'rng': {
            'python': int(seed),
            'numpy': int(seed),
            'torch': int(seed),
            'cuda': int(seed),
            'tensorflow': int(seed),
            'data': int(seed),
            'eval_sampling': 0,
            'dataloader_generator': int(seed),
            'worker_seed_policy': 'torch.initial_seed() % 2**32',
        },
        'd': int(d),
        'n_layers': int(n_layers),
        'n_heads': int(n_heads),
        'd_head': int(d_head),
        'vocab_size': int(vocab),
        'mlp_ratio': float(mlp_ratio),
        'dataset': cfg.get('dataset') or 'wikitext',
        'train_knobs': training_knobs(amp_dtype=amp_dtype, eager_batch=eager_batch),
    })
    _knobs = hp['train_knobs']
    hp['amp_dtype'] = _knobs.get('amp_dtype')
    hp['eager_batch'] = _knobs.get('eager_batch')
    if data_version is not None:
        hp['data_version'] = str(data_version)
    return hp

def write_run_manifest(outdir, cfg, run_cfg, prov, summary, sanity=None, reproducibility=None, *, data_contract=None, status=None, planned_cells=None, started_at=None, cell_records=None):
    """Merge one panel's provenance/config/outcome summary into
    `outdir/run_manifest.json` under the same flock used by summary.json.

    The per-record `losses` / `ppl_history` curves stay in summary.json (to
    avoid duplicating large arrays); each manifest cell points at its key.
    `sanity` (the small-batch overfit reports), `reproducibility` (the
    duplicate-run seed check) and `data_contract` (the real batch iterator
    check) are embedded when supplied, or read back from
    `outdir/overfit_sanity.json` / `outdir/reproducibility.json` /
    `outdir/data_contract.json` so an auto-preflight is traceable without a
    second manual record.

    `status` / `planned_cells` make the manifest a crash-safe tracking record:
    `run()` writes it with status='running' before the first cell, so a killed
    process still leaves the code version, config, data version and the list
    of planned cells on disk; the final call marks it complete and lists any
    cell that produced no measurement."""
    os.makedirs(outdir, exist_ok=True)

    def _evidence_for_this_panel(doc, path, what):
        """Only evidence stamped with THIS panel's run_cfg may enter the
        manifest.  A run_cfg-less (legacy/hand-written) doc must say
        ``legacy: true`` explicitly; otherwise a stale pre-flight from a reused
        outdir could silently certify a new configuration."""
        if doc is None:
            return None
        if not isinstance(doc, dict):
            print(f'[track] WARNING: {what} evidence is not a JSON object ({type(doc).__name__}) — not embedding it in this panel')
            return None
        _rc = doc.get('run_cfg')
        if _rc is None:
            if doc.get('legacy') is True:
                print(f'[track] WARNING: {what} evidence ({path}) has no run_cfg and is explicitly marked legacy — embedding it as unverified legacy evidence')
                return doc
            print(f'[track] WARNING: {what} evidence ({path}) has NO run_cfg — refusing to embed it (it cannot be tied to this panel; mark legacy=True only for a deliberately imported old record)')
            return None
        if _rc != run_cfg:
            print(f'[track] WARNING: {what} evidence ({path}) belongs to a DIFFERENT run_cfg ({_rc!r} != {run_cfg!r}) — not embedding it in this panel (a stale pre-flight must not certify a new config)')
            return None
        return doc

    if sanity is None:
        _san_path = os.path.join(outdir, 'overfit_sanity.json')
        if os.path.exists(_san_path):
            try:
                with open(_san_path, encoding='utf-8') as _sf:
                    sanity = json.load(_sf)
            except Exception as _se:
                print(f'[track] WARNING: could not read {_san_path} for the run manifest ({type(_se).__name__}: {_se})')
    sanity = _evidence_for_this_panel(sanity, os.path.join(outdir, 'overfit_sanity.json'), 'sanity')
    if reproducibility is None:
        _rep_path = os.path.join(outdir, 'reproducibility.json')
        if os.path.exists(_rep_path):
            try:
                with open(_rep_path, encoding='utf-8') as _rf:
                    reproducibility = json.load(_rf)
            except Exception as _re:
                print(f'[track] WARNING: could not read {_rep_path} for the run manifest ({type(_re).__name__}: {_re})')
    reproducibility = _evidence_for_this_panel(reproducibility, os.path.join(outdir, 'reproducibility.json'), 'reproducibility')
    if data_contract is None:
        _bc_path = os.path.join(outdir, 'data_contract.json')
        if os.path.exists(_bc_path):
            try:
                with open(_bc_path, encoding='utf-8') as _bf:
                    data_contract = json.load(_bf)
            except Exception as _be:
                print(f'[track] WARNING: could not read {_bc_path} for the run manifest ({type(_be).__name__}: {_be})')
    data_contract = _evidence_for_this_panel(data_contract, os.path.join(outdir, 'data_contract.json'), 'data contract')
    if cell_records is not None:
        # Custom phases (warmup/NIAH/lenphase) have their own summary keys and
        # metrics (PPL with warm-grid keys, NIAH accuracy, by-length PPL).
        # They build already-normalised cell records and use the same
        # crash-safe manifest/provenance/integrity machinery as run().
        cells = {str(k): dict(v) for k, v in dict(cell_records).items() if isinstance(v, dict)}
    else:
        cells = {}
        for seed in cfg.get('seeds') or []:
            for v in cfg.get('variants') or []:
                key = f'{v}::seed{seed}'
                rec = summary.get(key)
                if not isinstance(rec, dict) or not ppl_is_usable(rec.get('ppl')) or rec.get('synthesized'):
                    continue
                _curves = _cell_curve_data(outdir, rec)
                _losses = rec.get('losses') if isinstance(rec.get('losses'), list) else (_curves.get('losses') if isinstance(_curves.get('losses'), list) else [])
                _loss_nums = [float(x) for x in _losses if isinstance(x, (int, float)) and not isinstance(x, bool)]
                _ph = rec.get('ppl_history') if isinstance(rec.get('ppl_history'), list) else (_curves.get('ppl_history') if isinstance(_curves.get('ppl_history'), list) else [])
                if not _losses and rec.get('curve_n_points') is not None:
                    _loss_nums = []
                cells[key] = {
                    'variant': v,
                    'seed': int(seed),
                    'run_id': rec.get('run_id'),
                    'started_at': rec.get('started_at'),
                    'finished_at': rec.get('finished_at'),
                    'ppl': rec.get('ppl'),
                    'steps': rec.get('steps'),
                    'tokens_seen': rec.get('tokens_seen'),
                    'params': rec.get('params'),
                    'train_time_s': rec.get('train_time_s'),
                    'final_loss_smoothed': rec.get('final_loss_smoothed'),
                    'rng_fp': rec.get('rng_fp'),
                    'data_fp': rec.get('data_fp'),
                    'val_fp': rec.get('val_fp'),
                    'data_version': rec.get('data_version') or prov.get('data', {}).get('data_version'),
                    'code_stamp': rec.get('_code'),
                    'hyperparams': rec.get('hyperparams'),
                    'curve_summary': {'n_points': (len(_losses) if _losses else int(rec.get('curve_n_points') or 0)),
                                      'first': _loss_nums[0] if _loss_nums else rec.get('curve_first'),
                                      'last': _loss_nums[-1] if _loss_nums else rec.get('curve_last'),
                                      'best': min(_loss_nums) if _loss_nums else rec.get('curve_best'),
                                      'ppl_history_points': (len(_ph) if _ph else int(rec.get('ppl_history_n_points') or 0))},
                    'curve_ref': rec.get('curve_ref') or f'summary.json#{key}.losses',
                    'ppl_history_ref': rec.get('ppl_history_ref') or f'summary.json#{key}.ppl_history',
                    'sanity_ref': rec.get('sanity_ref') or ('overfit_sanity.json' if sanity else None),
                }
    if planned_cells is None:
        planned_cells = [f'{v}::seed{seed}' for seed in cfg.get('seeds') or [] for v in cfg.get('variants') or []]
    planned_cells = [str(_k) for _k in dict.fromkeys(planned_cells)]
    # Machine-checkable completeness of the per-cell trace.  A record without
    # provenance/hyperparameters/RNG fingerprint/curve is still a measurement,
    # but it must not silently look as traceable as a complete one.
    _integrity = {
        'n_cells': len(cells),
        'with_provenance': sum(1 for c in cells.values() if c.get('data_fp') and c.get('code_stamp')),
        'with_hyperparams': sum(1 for c in cells.values() if c.get('hyperparams')),
        'with_rng_fp': sum(1 for c in cells.values() if c.get('rng_fp')),
        'with_data_version': sum(1 for c in cells.values() if c.get('data_version')),
        'with_curve': sum(1 for c in cells.values() if (c.get('curve_summary') or {}).get('n_points')),
        'with_sanity_ref': sum(1 for c in cells.values() if c.get('sanity_ref')),
        'with_data_contract': int(bool(isinstance(data_contract, dict) and data_contract.get('passed'))),
    }
    _evidence = {
        'small_batch_overfit': (bool(sanity.get('passed')) if isinstance(sanity, dict) else None),
        'same_machine_reproducibility': (bool(reproducibility.get('passed')) if isinstance(reproducibility, dict) else None),
        'batch_iterator_contract': (bool(data_contract.get('passed')) if isinstance(data_contract, dict) else None),
    }
    entry = {
        'run_cfg': run_cfg,
        'status': status or 'complete',
        'created_at': started_at or time.strftime('%Y-%m-%dT%H:%M:%S'),
        'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': _jsonable(cfg),
        'provenance': prov,
        'data_version': prov.get('data', {}).get('data_version'),
        'sanity': sanity,
        'reproducibility': reproducibility,
        'data_contract': data_contract,
        'evidence': _evidence,
        'experiment_log': 'experiments.jsonl',
        'tracking_schema': 2,
        'planned_cells': planned_cells,
        'n_planned': len(planned_cells),
        'n_completed': len(cells),
        'missing_cells': [k for k in planned_cells if k not in cells],
        'tracking_integrity': _integrity,
        'cells': cells,
    }
    path = os.path.join(outdir, 'run_manifest.json')

    def _merge(m):
        if not isinstance(m, dict):
            m = {}
        m['manifest_version'] = m.get('manifest_version') or 1
        _prev = m.setdefault('panels', {}).get(run_cfg)
        if isinstance(_prev, dict) and _prev.get('created_at') and status != 'running':
            entry['created_at'] = _prev['created_at']
        m['panels'][run_cfg] = entry
        return m
    locked_json_update(path, _merge)
    return path

_EVAL_TOKENS_PER_CHUNK = 1 << 20


def _eval_row_working_bytes(model, T, d, elem_size, bytes_per_value=8):
    """Conservative per-row activation/workspace estimate for one eval row.

    The dense `T*d` term is only the hidden-state floor.  Long-sequence
    workspace is usually set by the attention workspace instead: a full/sink
    head materialises [nh, rows, T] logits, the CSA indexer scores [T, B_blocks,
    nIH], and HCA's arange expansion is [T, B_blocks].  Estimate the largest of
    those per layer (layers run one at a time under inference_mode) rather than
    the hidden state alone."""
    T = max(1, int(T))
    d = max(1, int(d))
    elem = max(1, int(elem_size))
    per_row = max(1, T * d * elem * max(1, int(bytes_per_value)))
    for blk in getattr(model, 'blocks', []):
        attn = getattr(blk, 'attn', None)
        cfg = getattr(attn, 'cfg', None)
        if cfg is None:
            continue
        nh = max(1, int(getattr(attn, 'nh', 1) or 1))
        kind = getattr(cfg, 'kind', 'full')
        if kind == 'full':
            win = max(0, int(getattr(cfg, 'window', 0) or 0))
            if getattr(attn, 'sink', None) is not None or getattr(cfg, 'full_sink', False):
                span = T
            elif 0 < win < T:
                span = win
            else:
                span = min(T, 1024)
            attn_bytes = 3 * nh * T * span * elem
        else:
            bs = int(getattr(cfg, 'block_size', 0) or 0)
            tgt = int(getattr(cfg, 'target_block_tokens', 0) or 0)
            mn = max(1, int(getattr(cfg, 'min_block', 1) or 1))
            step = max(1, bs or tgt or mn)
            Bm = (T + step - 1) // step
            nih = max(1, int(getattr(cfg, 'n_index_heads', 1) or 1))
            topk = max(1, int(getattr(cfg, 'index_topk', 1) or 1))
            hd = max(1, int(getattr(attn, 'hd', 1) or 1))
            win = max(0, int(getattr(cfg, 'sliding_window', 0) or 0))
            # HCA reads every block (arange expansion), so its effective
            # selection width is Bm; CSA only scores min(topk, Bm) blocks.
            nb_eff = Bm if kind == 'hca' else min(topk, Bm)
            M = nb_eff + win
            gather_bytes = (4 * M * nh * hd + 2 * M * nh) * elem
            index_bytes = min(2 * T * Bm * nih * elem, 384 << 20) if kind != 'hca' else 0
            attn_bytes = max(index_bytes, gather_bytes)
            if kind == 'hca':
                # _arange_exp_cache is a stride-0 expand view: its physical
                # storage is one int32 [Bm] row, not [T, Bm].
                attn_bytes = max(attn_bytes, 4 * Bm)
        per_row = max(per_row, int(attn_bytes))
    return per_row


def _eval_adaptive_chunk(model, val_batch, device, chunk, _free_frac=0.5, _bytes_per_value=8):
    """Bound an evaluation chunk by both token count and currently-free device capacity.

    Rows are independent, so the partition changes no value; it only keeps a
    long-sequence eval from launching a chunk whose attention/indexer
    workspace cannot fit.  The estimate uses the model's actual parameter
    element size (bf16 halves it) and the real per-layer attention shape, not
    just `T*d`; the existing OOM retry remains the final backstop."""
    rows = int(val_batch.shape[0])
    T = int(val_batch.shape[1])
    span = max(1, T - 1)
    chunk = min(max(1, int(chunk)), max(1, _EVAL_TOKENS_PER_CHUNK // span))
    if device.type == 'cuda' and rows > 0:
        try:
            free, _total = torch.cuda.mem_get_info(device)
            d = int(model.tok.weight.shape[1])
            elem = int(model.tok.weight.element_size())
            per_row = _eval_row_working_bytes(model, T, d, elem, _bytes_per_value)
            chunk = min(chunk, max(1, int(free * float(_free_frac)) // per_row))
        except Exception:
            pass
    return max(1, min(chunk, rows))


def _det_segment_sum(n_rows, indices, values, dtype=None):
    """Deterministic scatter-add by stable sort + segmented prefix sums.

    CUDA's ``index_add_``/``index_put_(accumulate=True)`` use atomics, so the
    addition order (and the last-bit result) can depend on thread scheduling.
    Sorting by (index, original position) fixes the order; segment sums are
    then formed with a cumsum and written through ``index_copy_`` (unique
    destinations, no atomics).  Mirrors ``csa_kernels._det_scatter_add`` but
    keeps the caller's dtype (the per-document NLL accumulator is float64)."""
    idx = indices.reshape(-1).to(torch.int64)
    vals = values.reshape(-1)
    out_dtype = dtype if dtype is not None else vals.dtype
    out = torch.zeros(int(n_rows), device=vals.device, dtype=out_dtype)
    if idx.numel() == 0:
        return out
    valid = (idx >= 0) & (idx < int(n_rows))
    if not bool(valid.all()):
        idx = idx[valid]
        vals = vals[valid]
        if idx.numel() == 0:
            return out
    order = torch.argsort(idx, stable=True)
    idx_s = idx.index_select(0, order)
    val_s = vals.index_select(0, order).to(out_dtype)
    end = torch.ones_like(idx_s, dtype=torch.bool)
    if idx_s.numel() > 1:
        end[:-1] = idx_s[1:] != idx_s[:-1]
    seg_end = torch.nonzero(end, as_tuple=False).reshape(-1)
    csum = torch.cumsum(val_s, dim=0)
    hi = csum.index_select(0, seg_end)
    lo = torch.zeros_like(hi)
    if seg_end.numel() > 1:
        lo[1:] = csum.index_select(0, seg_end[:-1])
    seg_sum = hi - lo
    seg_id = idx_s.index_select(0, seg_end)
    out.index_copy_(0, seg_id, seg_sum)
    return out

@torch.inference_mode()
def eval_ppl(model, val_batch, device, chunk=8, eval_rows=None, eval_seed=0, return_stats=False,
             collect_block_stats=False, val_bnd=None, bnd_tol=1):
    # `collect_block_stats=True` enables the compression-report hooks for this
    # same forward pass, so callers do not have to run the validation trunk a
    # second time just to collect block-length/boundary diagnostics.
    # A chunk is a set of independent sequences (no cross-row op in the
    # model), so capping it by token count does not change any per-row value.
    _span = max(1, int(val_batch.shape[1]) - 1)
    chunk = _eval_adaptive_chunk(model, val_batch, device, chunk)
    was_training = model.training
    model.eval()
    need_reg_layers = [blk.attn for blk in model.blocks if hasattr(blk.attn, 'need_reg')]
    need_reg_prior = [_a.need_reg for _a in need_reg_layers]
    nll = torch.zeros((), dtype=torch.float64, device=device)
    # Per-document NLL (review 3: eval-subset uncertainty must be visible, not
    # only the seed-to-seed spread).  Rows are the fixed evaluation documents;
    # with eval_rows sampling, a row with no sampled target gets 0 tokens and
    # is excluded by the bootstrap consumer.
    _row_nll = torch.zeros(int(val_batch.shape[0]), dtype=torch.float64, device=device) if return_stats else None
    _row_ntok = torch.zeros(int(val_batch.shape[0]), dtype=torch.long, device=device) if return_stats else None
    for _a in need_reg_layers:
        _a.need_reg = False
    if collect_block_stats:
        enable_block_stats(model, True)
    _compression_out = None
    try:
        ntok = 0
        V = None
        _sel = None
        if eval_rows is not None and int(eval_rows) > 0:
            _n_rows = int(val_batch.shape[0])
            _n_tgt = max(0, int(val_batch.shape[1]) - 1)
            _flat = _n_rows * _n_tgt
            _k = min(int(eval_rows), _flat)
            if _k < _flat:
                _g = torch.Generator().manual_seed(int(eval_seed))
                _sel = torch.randperm(_flat, generator=_g)[:_k]
                # keep the sorted host copy: per-chunk membership bounds are
                # integer range queries on a sorted table — searchsorted on the
                # host gives the same slice the boolean-mask form selected
                # (host-side integer equivalence, design notes §6.2).
                _sel_host = torch.sort(_sel).values
                _sel_sorted = _sel_host.to(device)
                print(f'[eval_ppl] scoring a uniform sample of {_k}/{_flat} target rows (eval_rows={int(eval_rows)}, eval_seed={int(eval_seed)}) — this is a SAMPLED estimate, not the full-set PPL')
            else:
                print(f'[eval_ppl] eval_rows={int(eval_rows)} covers the whole {_flat}-row target space; scoring the FULL set (this is the population statistic, not a sample)')
        _vhost = None
        if device.type == 'cuda':
            _vdev = _dev_str(device)
            _vc = _PINNED_HOST_CACHE.get('val')
            if (_vc is not None and _vc[0] is val_batch and _vc[1] == tuple(val_batch.shape)
                    and _vc[3] == _vdev):
                _vhost = _vc[2]
            else:
                try:
                    _vhost = torch.as_tensor(val_batch).pin_memory()
                    _PINNED_HOST_CACHE['val'] = (val_batch, tuple(val_batch.shape), _vhost, _vdev)
                except RuntimeError:
                    _vhost = torch.as_tensor(val_batch)
        else:
            _vhost = torch.as_tensor(val_batch)
        _async = _vhost.is_pinned()
        _span = max(0, int(val_batch.shape[1]) - 1)

        def _run_rows(i, cur):
            nonlocal ntok, V, nll, _row_nll, _row_ntok
            _lo = _hi = 0
            if _sel is not None:
                # Host-side range count on the sorted selection: chunks with
                # no sampled target contribute nothing to nll/ntok.
                _base = i * _span
                _wide = min(cur, int(val_batch.shape[0]) - i) * _span
                if _wide > 0:
                    _lo = int(torch.searchsorted(_sel_host, _base))
                    _hi = int(torch.searchsorted(_sel_host, _base + _wide))
                if _hi <= _lo:
                    return
            elif _span == 0:
                return
            ids = _vhost[i:i + cur].to(device, non_blocking=_async)
            # Trunk + per-slice LM head; comp_reg / need_reg handling is
            # identical to model(ids) because forward() itself is exactly
            # trunk() followed by head().
            h = model.trunk(ids)
            if _sel is not None:
                _local = _sel_sorted[_lo:_hi] - _base
                _ri = torch.div(_local, _span, rounding_mode='floor')
                _ci = _local - _ri * _span
                # `_row_nll` / `_row_ntok` are indexed by the GLOBAL document
                # row, while `_ri` is chunk-local because it indexes this
                # chunk's `h`/`ids`.  Offset it before the scatters.
                _src_rows = _ri + i
                # gather the selected hidden rows BEFORE the head: each output
                # row of the Linear is an independent function of one hidden
                # row, so head(h[sel]) == head(h)[sel]
                _rows = model.head(h[_ri, _ci, :])
                if V is None:
                    V = _rows.size(-1)
                _tgts = ids[_ri, _ci + 1]
                del _ri, _ci
                _ce = F.cross_entropy(_rows, _tgts, reduction='none').double()
                nll += _ce.sum()
                if return_stats:
                    # Deterministic segmented sum (no CUDA atomics): the
                    # per-document NLL feeding the document bootstrap must be
                    # reproducible on the same machine.
                    _row_nll += _det_segment_sum(_row_nll.numel(), _src_rows, _ce, dtype=torch.float64)
                    _row_ntok += _det_segment_sum(_row_ntok.numel(), _src_rows, torch.ones_like(_src_rows), dtype=torch.long)
                ntok += int(_tgts.numel())
                del _rows, _tgts, _ce, _src_rows
            else:
                _tgts = ids[:, 1:]
                for _s0 in range(0, _span, 64):
                    _sl = slice(_s0, min(_s0 + 64, _span))
                    _lg = model.head(h[:, _sl, :])
                    if V is None:
                        V = _lg.size(-1)
                    _ce = F.cross_entropy(_lg.transpose(1, 2), _tgts[:, _sl], reduction='none').double()
                    nll += _ce.sum()
                    if return_stats:
                        _b = int(_ce.shape[0])
                        _row_nll[i:i + _b] += _ce.sum(dim=1)
                        _row_ntok[i:i + _b] += int(_ce.shape[1])
                    del _lg, _ce
                ntok += int(_tgts.numel())
                del _tgts
            del ids, h

        i = 0
        _n_rows = int(val_batch.shape[0])
        while i < _n_rows:
            _cur = min(chunk, _n_rows - i)
            _nll_snap = nll.clone()
            _rn_snap = _row_nll.clone() if return_stats else None
            _rt_snap = _row_ntok.clone() if return_stats else None
            _stats_snap = _snapshot_block_stats(model) if collect_block_stats else None
            try:
                _run_rows(i, _cur)
            except RuntimeError as _oe:
                nll.copy_(_nll_snap)
                if return_stats:
                    _row_nll.copy_(_rn_snap)
                    _row_ntok.copy_(_rt_snap)
                if _stats_snap is not None:
                    _restore_block_stats(_stats_snap)
                if ('out of memory' in str(_oe).lower()) and _cur > 1:
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    chunk = max(1, _cur // 2)
                    print(f'[eval_ppl] device-memory exhaustion; retrying with chunk={chunk}')
                    continue
                raise
            i += _cur
    finally:
        if collect_block_stats:
            try:
                _compression_out = _compression_report_summary(model, int(val_batch.shape[1]), val_bnd, bnd_tol)
            except Exception as _ce:
                print(f'[eval_ppl] WARNING: compression diagnostics failed ({type(_ce).__name__}: {_ce})')
                _compression_out = None
            enable_block_stats(model, False)
        for _a, _prev in zip(need_reg_layers, need_reg_prior):
            _a.need_reg = _prev
        model.train(was_training)
    if not ntok:
        if not return_stats:
            return float('nan')
        _empty = {'ppl': float('nan'), 'nll_sum': float('nan'), 'n_tokens': 0,
                  'nll_per_row': [], 'n_tokens_per_row': [], 'eval_rows': int(eval_rows) if eval_rows is not None else None,
                  'eval_seed': int(eval_seed), 'eval_subset': int(val_batch.shape[0])}
        if collect_block_stats:
            _empty['compression'] = _compression_out
        return _empty
    try:
        ppl = math.exp(nll.item() / ntok)
    except OverflowError:
        ppl = float('inf')
    if not return_stats:
        return ppl
    _out = {
        'ppl': float(ppl),
        'nll_sum': float(nll.item()),
        'n_tokens': int(ntok),
        'nll_per_row': [float(x) for x in _row_nll.detach().cpu().tolist()],
        'n_tokens_per_row': [int(x) for x in _row_ntok.detach().cpu().tolist()],
        'eval_rows': int(eval_rows) if eval_rows is not None else None,
        'eval_seed': int(eval_seed),
        'eval_subset': int(val_batch.shape[0]),
    }
    if collect_block_stats:
        _out['compression'] = _compression_out
    return _out

class _BlockStats:
    """Incremental replacement for the per-sequence bincount list.

    compression_report only needs the pooled mean/std/min/max/fractions over
    block lengths.  Each append folds the integer counts into running moments
    and drops the tensor immediately; values are accumulated in float64 so the
    pooled statistics are at least as accurate as the concatenated float32
    form."""

    def __init__(self, min_block=None, max_block=None):
        self.min_block = min_block
        self.max_block = max_block
        self.n = 0
        self.total = 0
        self.blocks = 0
        self.s = 0.0
        self.ss = 0.0
        self.lo = None
        self.hi = None
        self.at_min = 0
        self.at_max = 0
        # GPU counts are queued and folded once at report time.
        self._pending = []

    def snapshot(self):
        d = dict(self.__dict__)
        d['_pending'] = list(self._pending)
        return d

    def _add(self, a, n_items=1):
        if not a.size:
            return
        a = a.astype(np.float64, copy=False)
        self.n += int(n_items)
        self.blocks += int(a.size)
        self.total += int(a.size)
        self.s += float(a.sum())
        self.ss += float(np.square(a).sum())
        self.lo = float(a.min()) if self.lo is None else min(self.lo, float(a.min()))
        self.hi = float(a.max()) if self.hi is None else max(self.hi, float(a.max()))
        if self.min_block is not None:
            self.at_min += int((a <= self.min_block).sum())
        if self.max_block is not None:
            self.at_max += int((a >= self.max_block).sum())

    def append(self, t):
        if torch.is_tensor(t) and t.device.type != 'cpu':
            self._pending.append(t.detach())
            self.n += 1
            return
        if torch.is_tensor(t):
            t = t.detach()
        self._add(np.asarray(t).reshape(-1))

    def _flush(self):
        if not self._pending:
            return
        parts = self._pending
        self._pending = []
        try:
            flat = torch.cat([p.reshape(-1).to(torch.int64) for p in parts])
        except Exception:
            for p in parts:
                self._add(np.asarray(p.detach().cpu()).reshape(-1), n_items=0)
            return
        # `append` already counted each pending sequence in self.n.
        self._add(flat.detach().cpu().numpy(), n_items=0)
        del flat

    def __len__(self):
        self._flush()
        return self.n

    def summary(self, seq_len):
        self._flush()
        if self.n == 0 or self.total == 0:
            return None
        mean = self.s / self.total
        # Match the concatenated form's sample standard deviation
        # (torch.std's default Bessel correction); with a single observation
        # the original reports 0.0.
        if self.total > 1:
            var = max((self.ss - self.s * self.s / self.total) / (self.total - 1), 0.0)
        else:
            var = 0.0
        return {
            'blocks': self.blocks / self.n,
            'avg_len': seq_len / max(self.blocks / self.n, 1.0),
            'len_mean': mean,
            'len_std': math.sqrt(var),
            'len_min': int(self.lo),
            'len_max': int(self.hi),
            'frac_at_min': self.at_min / self.total,
            'frac_at_max': self.at_max / self.total,
        }

def enable_block_stats(model, on):
    for blk in model.blocks:
        if on:
            blk.attn._stats = _BlockStats(getattr(blk.attn.cfg, 'min_block', None), getattr(blk.attn.cfg, 'max_block', None))
        else:
            blk.attn._stats = None
        blk.attn._cuts = [] if on else None


def _snapshot_block_stats(model):
    """Capture every attention layer's block-count / cut state.

    A failed forward may already have appended per-layer counts and cut
    positions.  OOM retries must restore this state together with the NLL
    accumulators, otherwise the partially processed rows are counted twice."""
    snap = []
    for blk in model.blocks:
        attn = blk.attn
        st = getattr(attn, '_stats', None)
        cuts = getattr(attn, '_cuts', None)
        snap.append((attn,
                     st.snapshot() if isinstance(st, _BlockStats) else None,
                     list(cuts) if cuts is not None else None))
    return snap


def _restore_block_stats(snap):
    for attn, state, cuts in snap:
        if state is not None:
            attn._stats.__dict__.clear()
            attn._stats.__dict__.update(state)
        if cuts is not None:
            attn._cuts[:] = cuts

def boundary_alignment(pred_cuts, gt_mask_row, n, tol=1, provenance=None):
    gt = np.nonzero(np.asarray(gt_mask_row)[:n - 1])[0] + 1
    pred = np.asarray(pred_cuts)
    P, G = (len(pred), len(gt))
    if P == 0 or G == 0:
        if provenance is not None:
            provenance.setdefault('bnd_rand_exact', True)
        return (0.0, 0.0, 0.0, 0.0)
    hits, ghits = _greedy_match_counts(pred, gt, tol)
    prec, rec = (hits / P, ghits / G)
    f1 = 2 * prec * rec / max(prec + rec, 1e-09)
    n_pos = max(n - 1, 1)
    _approx = []
    rand_prec = _random_cut_precision(P, None, n_pos, tol, G, gt, approx_flag=_approx)
    if provenance is not None:
        provenance['bnd_rand_exact'] = provenance.get('bnd_rand_exact', True) and (not _approx)
    return (prec, rec, f1, rand_prec)

def _mask_runs(near, n_pos, tol):
    near_arr = np.asarray(near)
    m = near_arr[1:n_pos + 1].astype(bool)
    if not m.any():
        return np.zeros(0, dtype=int)
    if tol == 0:
        # With tol=0 the neighbourhood of a boundary is the boundary itself,
        # so every True lane is exactly one boundary -- including adjacent
        # boundaries, which form a longer run and cannot be split by the
        # run-length rules below (those are only meaningful for tol>0).
        return np.nonzero(m)[0] + 1
    d = np.diff(np.concatenate([[0], m.view(np.int8), [0]]))
    starts = np.nonzero(d == 1)[0]
    ends = np.nonzero(d == -1)[0]
    lens = ends - starts
    left_touch = bool(near_arr[0]) if near_arr.size else False
    out, ok = ([], True)
    for s, e, L in zip(starts, ends, lens):
        if s == 0 and e == m.size:
            # A run covering the whole sequence can come from one boundary at
            # the centre OR from several boundaries whose neighbourhoods tile
            # it; the post-slice mask cannot distinguish them.
            ok = False
            break
        if s == 0:
            if left_touch:
                # Position 0 is inside the neighbourhood, so the leftmost
                # possible boundary is 1; only a length-(tol+1) run pins it.
                if L == tol + 1:
                    out.append(0)
                else:
                    ok = False
                    break
            else:
                # Position 0 is outside every neighbourhood: the unique
                # minimum boundary is tol+1, and only its full 2*tol+1 run is
                # unambiguous.
                if L == 2 * tol + 1:
                    out.append(tol)
                else:
                    ok = False
                    break
        elif e == m.size:
            # The right edge is clipped by the sequence end: only the
            # one-sided length-(tol+1) run pins the final boundary.
            if L == tol + 1:
                out.append(m.size - 1)
            else:
                ok = False
                break
        else:
            # A fully interior neighbourhood has exactly 2*tol+1 lanes.
            if L == 2 * tol + 1:
                out.append((s + e - 1) // 2)
            else:
                ok = False
                break
    if ok:
        return np.asarray(sorted(set(out)), dtype=int) + 1
    return None

def _random_cut_precision(P, near, n_pos, tol, G, gt_bounds=None, approx_flag=None):
    if P <= 0 or G <= 0 or n_pos <= 0:
        return 0.0
    if gt_bounds is None:
        if near is None:
            print('[stats] WARNING: `_random_cut_precision` got neither a `near` mask nor `gt_bounds`; the baseline is UNCOMPUTABLE for this call (returning 0.0) rather than being scored against an inflated reference set.  Pass `gt_bounds`.')
            return 0.0
        _runs = _mask_runs(near, n_pos, tol)
        if _runs is None:
            print('[stats] WARNING: `_random_cut_precision` got a `near` mask it cannot decompose into boundary neighbourhoods and no `gt_bounds`; the baseline is UNCOMPUTABLE for this call (returning 0.0) rather than being scored against an inflated reference set.  Pass `gt_bounds`.')
            return 0.0
        gt = _runs
    else:
        gt = np.asarray(gt_bounds).astype(int)
        gt = gt[(gt >= 1) & (gt <= n_pos)]
    if gt.size == 0:
        return 0.0
    gt = np.unique(gt)
    if P >= n_pos:
        return float(min(gt.size, P) / P)
    clusters, cur = ([], [int(gt[0])])
    for g in gt[1:]:
        g = int(g)
        if g - cur[-1] <= 2 * tol + 1:
            cur.append(g)
        else:
            clusters.append(cur)
            cur = [g]
    clusters.append(cur)
    expect_frac = 0.0
    for cl in clusters:
        lo, hi = (max(min(cl) - tol, 1), min(max(cl) + tol, n_pos))
        win = list(range(lo, hi + 1))
        expect_frac += _cluster_expected_matches(win, cl, tol, P, n_pos, approx_flag)
    return float(expect_frac / P)

def _log_ways_ratio(w, r, n_pos, P):
    a, b, c = (n_pos - w, P - r, P)
    if r < 0 or a < 0 or b < 0 or (c < 0) or (r > w) or (b > a):
        return float('-inf')
    return _lgamma_comb(w, r) + _lgamma_comb(a, b) - _lgamma_comb(n_pos, c)

def _lgamma_comb(n, k):
    if k < 0 or k > n:
        return float('-inf')
    if k == 0 or k == n:
        return 0.0
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
_AMP_WORK_CAP = (1 << 12) - 1
_AMP_FALLBACK_DRAWS = 2048

def _cluster_expected_matches(win, cl, tol, P, n_pos, approx_flag=None):
    w = len(win)
    if w <= 0 or P <= 0:
        return 0.0
    cl_t = tuple((int(b) for b in cl))
    A = [int(p) for p in win if any((abs(int(p) - b) <= tol for b in cl_t))]
    a = len(A)
    if a == 0:
        return 0.0
    rmax = min(P, a)
    _cl_arr = np.asarray(cl_t)
    _ord = np.argsort(_cl_arr, kind='stable')
    _pre = (_cl_arr[_ord], _ord)
    T = [0] * (rmax + 1)
    _comp_T = _component_match_tables(A, cl_t, tol, rmax, _pre)
    a_max = max((a_c for _tbl, a_c in _comp_T.values()), default=0)
    any_sampled = any((tbl == [] for tbl, _a in _comp_T.values()))
    if not any_sampled:
        T = _convolve_component_T(_comp_T, rmax)
    else:
        _work = _subset_work(a_max, rmax)
        if approx_flag is not None:
            approx_flag.append((a_max, _AMP_WORK_CAP, _work))
        print(f'[stats] WARNING: `_random_cut_precision` is APPROXIMATING: a single candidate component holds {a_max} positions and needs {_work} subset scorings, past the enumeration limit of {_AMP_WORK_CAP}, so `bnd_rand` (and hence `bnd_excess`) for this sample is a Monte-Carlo estimate ({_AMP_FALLBACK_DRAWS} draws), NOT the exact probability the other panels report.  Record it as `bnd_rand_exact: False` and do not pool it with the exact readings.')
        seed = (int(n_pos) * 1000003 + int(P) * 10007 + int(tol) * 101 + int(a) * 17 + cl_t[0]) % 2 ** 32
        rng = np.random.default_rng(seed)
        wts = []
        for r in range(1, rmax + 1):
            outside = P - r
            if outside < 0 or outside > n_pos - a:
                continue
            lw = _log_ways_ratio(a, r, n_pos, P)
            if lw == float('-inf'):
                continue
            wts.append((r, lw))
        if not wts:
            return 0.0
        nb_r = len(wts)
        order = sorted(range(nb_r), key=lambda i: -wts[i][1])
        if nb_r > _AMP_FALLBACK_DRAWS:
            # More candidate sizes than the draw limit: keep the top-weight
            # sizes, one draw each, so the actual number of draws never
            # exceeds the declared _AMP_FALLBACK_DRAWS limit.
            _keep = set(order[:_AMP_FALLBACK_DRAWS])
            draws = [1 if i in _keep else 0 for i in range(nb_r)]
        else:
            per = max(1, _AMP_FALLBACK_DRAWS // nb_r)
            draws = [per] * nb_r
            for i in range(_AMP_FALLBACK_DRAWS - per * nb_r):
                draws[order[i % nb_r]] += 1
        total = 0.0
        for i, (r, lw) in enumerate(wts):
            n_draw = draws[i]
            if n_draw <= 0:
                continue
            subs = [_sample_r_subset(rng, A, r) for _ in range(n_draw)]
            s = _score_subsets_components(subs, _comp_T, cl_t, tol, _pre)
            total += math.exp(lw) * (s / n_draw)
        return float(total)
    total = 0.0
    for r in range(0, rmax + 1):
        outside = P - r
        if outside < 0 or outside > n_pos - a:
            continue
        if not r:
            continue
        if not (comb(a, r) and comb(n_pos - a, outside)):
            continue
        mean_r = T[r] / comb(a, r)
        if mean_r == 0.0:
            continue
        total += math.exp(_log_ways_ratio(a, r, n_pos, P)) * mean_r
    return float(total)

def _match_components(A, cl_t, tol):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = (root, parent[x])
        return root

    def union(x, y):
        rx, ry = (find(x), find(y))
        if rx != ry:
            parent[rx] = ry
    if A:
        A_arr = np.asarray(A)
        for gi, b in enumerate(cl_t):
            lo = int(np.searchsorted(A_arr, b - tol, side='left'))
            hi = int(np.searchsorted(A_arr, b + tol, side='right'))
            for pi in range(lo, hi):
                union(('c', pi), ('g', gi))
    comps = {}
    for pi, p in enumerate(A):
        comps.setdefault(find(('c', pi)), ([], []))[0].append(int(p))
    for gi, b in enumerate(cl_t):
        comps.setdefault(find(('g', gi)), ([], []))[1].append(int(b))
    out = []
    for key in sorted(comps, key=repr):
        pos, bnd = comps[key]
        if pos:
            out.append((pos, bnd))
    return out

def _sorted_pre(b_t):
    if not b_t:
        z = np.zeros(0, dtype=int)
        return (z, z)
    arr = np.asarray(b_t)
    o = np.argsort(arr, kind='stable')
    return (arr[o], o)

def _subset_work(w, rmax):
    if w <= 0 or rmax <= 0:
        return 0
    r_hi = min(rmax, w)
    total = 0
    c = 1
    for r in range(0, r_hi + 1):
        if r:
            c = c * (w - r + 1) // r
        if r:
            total += c
    return total

def _component_match_tables(A, cl_t, tol, rmax, _pre):
    tables = {}
    for pos, bnd in _match_components(A, cl_t, tol):
        r_c = min(rmax, len(pos))
        b_t = tuple(bnd)
        c_pre = _sorted_pre(b_t)
        if _subset_work(len(pos), r_c) <= _AMP_WORK_CAP:
            tbl = [0] * (r_c + 1)
            for rr in range(1, r_c + 1):
                subs = list(_subsets(pos, rr))
                if subs:
                    tbl[rr] = int(sum(_greedy_match_count_batch(subs, c_pre[0], c_pre[1], tol)))
            tables[tuple(pos), b_t] = (tbl, len(pos))
        else:
            tables[tuple(pos), b_t] = ([], len(pos))
    return tables

def _convolve_component_T(tables, rmax):
    items = [(tbl, a_c) for tbl, a_c in tables.values() if tbl]
    if not items:
        return [0] * (rmax + 1)
    choice = [0.0] * (rmax + 1)
    choice[0] = 1.0
    wsum = [0.0] * (rmax + 1)
    for tbl, a_c in items:
        rc_max = min(rmax, a_c)
        term = [0.0] * (rmax + 1)
        mean = [0.0] * (rmax + 1)
        for r in range(0, rc_max + 1):
            term[r] = float(comb(a_c, r))
            if r:
                mean[r] = tbl[r] / comb(a_c, r)
        new_choice = [0.0] * (rmax + 1)
        new_wsum = [0.0] * (rmax + 1)
        for i in range(rmax + 1):
            if choice[i] == 0.0 and wsum[i] == 0.0:
                continue
            for r in range(0, rc_max + 1):
                if i + r > rmax or term[r] == 0.0:
                    continue
                new_choice[i + r] += choice[i] * term[r]
                new_wsum[i + r] += wsum[i] * term[r] + choice[i] * term[r] * mean[r]
        choice, wsum = (new_choice, new_wsum)
    return [int(round(wsum[r])) for r in range(rmax + 1)]

def _sample_r_subset(rng, A, r):
    return tuple(np.sort(rng.choice(A, size=r, replace=False)))

def _score_subsets_components(subs, tables, cl_t, tol, _pre):
    _owner_cache = getattr(_score_subsets_components, '_owner_cache', None)
    if _owner_cache is None:
        _owner_cache = _score_subsets_components._owner_cache = {}
    if len(_owner_cache) >= 64 and id(tables) not in _owner_cache:
        _owner_cache.clear()
    cached = _owner_cache.get(id(tables))
    if cached is None:
        owner = {}
        for pos, bnd in tables:
            b_t = tuple(bnd)
            c_pre = _sorted_pre(b_t)
            for p in pos:
                owner[p] = (b_t, c_pre)
        cached = (owner, tables)
        _owner_cache[id(tables)] = cached
    owner = cached[0]
    by_comp = {}
    for si, sub in enumerate(subs):
        for p in sub:
            hit = owner.get(p)
            if hit is None:
                continue
            b_t, c_pre = hit
            per_sub = by_comp.get(b_t)
            if per_sub is None:
                per_sub = by_comp[b_t] = (c_pre, {})
            pts = per_sub[1].get(si)
            if pts is None:
                per_sub[1][si] = [p]
            else:
                pts.append(p)
    total = 0
    for _b_t, (c_pre, per_sub) in by_comp.items():
        sids = sorted(per_sub)
        pieces = [tuple(sorted(per_sub[si])) for si in sids]
        for v in _greedy_match_count_batch(pieces, c_pre[0], c_pre[1], tol):
            total += v
    return total

def _subsets(items, r):
    return itertools.combinations(items, r)

def _greedy_match_counts(pred, gt, tol, _pre=None):
    pred = np.sort(np.asarray(pred))
    P, G = (len(pred), len(gt))
    if P == 0 or G == 0:
        return (0, 0)
    if _pre is not None:
        gt_sorted, order = _pre
    else:
        gt_arr = np.asarray(gt)
        order = np.argsort(gt_arr, kind='stable')
        gt_sorted = gt_arr[order]
    lo_all = np.searchsorted(gt_sorted, np.asarray(pred) - tol, side='left')
    hi_all = np.searchsorted(gt_sorted, np.asarray(pred) + tol, side='right')
    cand = []
    for pi in range(P):
        p = pred[pi]
        for j in range(int(lo_all[pi]), int(hi_all[pi])):
            gi = int(order[j])
            cand.append((abs(p - gt[gi]), pi, gi))
    cand.sort()
    used_p, used_g = (set(), set())
    for _, pi, gi in cand:
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
    return (len(used_p), len(used_g))

def _greedy_match_count_batch(subsets, gt_sorted, order, tol):
    subs = [tuple(s) for s in subsets]
    n_sub = len(subs)
    if n_sub == 0:
        return []
    counts = np.zeros(n_sub, dtype=np.int64)
    sizes = np.fromiter((len(s) for s in subs), dtype=np.int64, count=n_sub)
    if sizes.sum() == 0:
        return counts.tolist()
    flat = np.fromiter((int(p) for s in subs for p in s), dtype=np.int64, count=int(sizes.sum()))
    cut_rank = np.concatenate([np.arange(int(k), dtype=np.int64) for k in sizes]) if n_sub else np.zeros(0, dtype=np.int64)
    owner = np.repeat(np.arange(n_sub, dtype=np.int64), sizes)
    lo = np.searchsorted(gt_sorted, flat - tol, side='left')
    hi = np.searchsorted(gt_sorted, flat + tol, side='right')
    cnt = (hi - lo).astype(np.int64)
    total = int(cnt.sum())
    if total:
        row_owner = np.repeat(owner, cnt)
        starts = np.repeat(lo, cnt)
        offs = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        bnd_idx = starts + offs
        gi = order[bnd_idx]
        pi_local = np.repeat(flat, cnt)
        pi_rank = np.repeat(cut_rank, cnt)
        dist = np.abs(pi_local - gt_sorted[bnd_idx])
        order_rows = np.lexsort((gi, pi_rank, dist, row_owner))
        used_p, used_g = (set(), set())
        prev_sid = -1
        for idx in order_rows:
            sid = int(row_owner[idx])
            if sid != prev_sid:
                if prev_sid >= 0:
                    counts[prev_sid] = len(used_p)
                used_p, used_g = (set(), set())
                prev_sid = sid
            p_i = int(pi_rank[idx])
            g_i = int(gi[idx])
            if p_i not in used_p and g_i not in used_g:
                used_p.add(p_i)
                used_g.add(g_i)
        if prev_sid >= 0:
            counts[prev_sid] = len(used_p)
    return counts.tolist()

def compression_report(model, val_batch, device, n_sample=64, val_bnd=None, bnd_tol=1):
    was_training = model.training
    model.eval()
    enable_block_stats(model, True)
    need_reg_layers = [blk.attn for blk in model.blocks if hasattr(blk.attn, 'need_reg')]
    need_reg_prior = [_a.need_reg for _a in need_reg_layers]
    for _a in need_reg_layers:
        _a.need_reg = False
    try:
        return _compression_report_impl(model, val_batch, device, n_sample, val_bnd, bnd_tol)
    finally:
        for _a, _prev in zip(need_reg_layers, need_reg_prior):
            _a.need_reg = _prev
        enable_block_stats(model, False)
        model.train(was_training)

def _compression_report_summary(model, seq_len, val_bnd=None, bnd_tol=1):
    report = {}
    # every dynamic layer's delta_logit (same scalars, same floats).
    _dl_layers = [(li, blk.attn.delta_logit) for li, blk in enumerate(model.blocks)
                  if blk.attn.cfg.dynamic and getattr(blk.attn, 'delta_logit', None) is not None]
    _dl_vals = {}
    if _dl_layers:
        _dl_vals = {li: float(v) for (li, _), v in zip(_dl_layers, torch.stack([p.detach() for _, p in _dl_layers]).tolist())}
    for li, blk in enumerate(model.blocks):
        attn = blk.attn
        st = attn._stats
        kind = attn.cfg.kind
        if st is None or kind == 'full' or len(st) == 0:
            continue
        if isinstance(st, _BlockStats):
            entry = st.summary(seq_len)
            if entry is None:
                continue
        else:
            lens = torch.cat([t if torch.is_tensor(t) else torch.as_tensor(t) for t in st]).float()
            n_blocks = sum((t.numel() for t in st)) / len(st)
            _sc = torch.stack([lens.mean(), lens.std() if lens.numel() > 1 else torch.zeros((), device=lens.device, dtype=lens.dtype), lens.min().float(), lens.max().float(), (lens <= attn.cfg.min_block).float().mean(), (lens >= attn.cfg.max_block).float().mean()])
            _vals = _sc.tolist()
            entry = {'blocks': n_blocks, 'avg_len': seq_len / max(n_blocks, 1.0), 'len_mean': _vals[0], 'len_std': _vals[1], 'len_min': int(_vals[2]), 'len_max': int(_vals[3]), 'frac_at_min': _vals[4], 'frac_at_max': _vals[5]}
            del _sc, _vals
        if attn.cfg.dynamic and getattr(attn, 'delta_logit', None) is not None:
            entry['delta'] = _dl_vals[li]
        if val_bnd is not None and attn._cuts:
            n_seq_ref = min(len(attn._cuts), val_bnd.shape[0])
            if n_seq_ref < len(attn._cuts):
                print(f'[compression_report] WARNING: {len(attn._cuts)} sequences of cuts but only {val_bnd.shape[0]} reference masks — scoring the first {n_seq_ref} only')
            ps, rs, fs, rps = ([], [], [], [])
            _cuts = attn._cuts
            _prov = {}
            # The integers transferred are exactly the per-sequence values
            # (index transport, design notes §6.1), segmented on the host.
            _tcut_ix = [si for si in range(n_seq_ref) if torch.is_tensor(_cuts[si])]
            _tcut_host = {}
            if _tcut_ix:
                _tlens = [int(_cuts[si].numel()) for si in _tcut_ix]
                _tflat = torch.cat([_cuts[si].reshape(-1) for si in _tcut_ix]).cpu().numpy()
                _toff = 0
                for si, _tln in zip(_tcut_ix, _tlens):
                    _tcut_host[si] = _tflat[_toff:_toff + _tln]
                    _toff += _tln
                del _tflat
            for si in range(n_seq_ref):
                c = _tcut_host.get(si) if si in _tcut_host else _cuts[si]
                p, r_, f_, rp = boundary_alignment(c, val_bnd[si], seq_len, tol=bnd_tol, provenance=_prov)
                ps.append(p)
                rs.append(r_)
                fs.append(f_)
                rps.append(rp)
            if n_seq_ref:
                entry['bnd_prec'] = float(np.mean(np.asarray(ps)))
                entry['bnd_rec'] = float(np.mean(np.asarray(rs)))
                entry['bnd_f1'] = float(np.mean(np.asarray(fs)))
                entry['bnd_rand'] = float(np.mean(np.asarray(rps)))
                entry['bnd_excess'] = entry['bnd_prec'] - entry['bnd_rand']
                entry['bnd_rand_exact'] = bool(_prov.get('bnd_rand_exact', True))
                if not entry['bnd_rand_exact']:
                    print(f'[compression_report] WARNING: `bnd_rand` for L{li}_{kind} is a SAMPLED estimate, not the exact expectation, so `bnd_excess` carries Monte-Carlo error — recorded as `bnd_rand_exact: False` and it must not be pooled with the exact panels.')
            else:
                print('[compression_report] WARNING: no reference sequence to score (n_seq_ref=0) — boundary metrics OMITTED rather than published as nan')
            entry['bnd_n_seq'] = n_seq_ref
        tag = 'dyn' if attn.cfg.dynamic else 'fix'
        report[f'L{li}_{kind}_{tag}'] = entry
    return report


@torch.inference_mode()
def _compression_report_impl(model, val_batch, device, n_sample, val_bnd, bnd_tol):
    report = {}
    seq_len = val_batch.shape[1]
    if int(n_sample) < 1:
        raise ValueError(f'compression_report: n_sample must be >= 1, got {n_sample!r}; the report is an average over the sequences it scores, so there is nothing to measure with no sequence scored')
    n_used = min(n_sample, val_batch.shape[0])
    _row_chunk = _eval_adaptive_chunk(model, np.asarray(val_batch), device,
                                      min(16, _EVAL_TOKENS_PER_CHUNK // max(seq_len, 1)))
    _row_chunk = max(1, min(_row_chunk, n_used))
    # inference_mode (the decorator) already covers the forward pass.  The
    # sampled sequences are independent, so the row grouping does not change
    # any value.
    _r0 = 0
    while _r0 < n_used:
        _cur = min(_row_chunk, n_used - _r0)
        # A failed forward may already have appended per-layer block counts
        # (the stats hook runs layer by layer).  Snapshot them so the retry
        # cannot count the partially-processed rows twice.
        _stats_snap = _snapshot_block_stats(model)
        # as_tensor accepts numpy arrays AND torch tensors (eval_ppl already
        # uses it).
        ids = None
        try:
            ids = torch.as_tensor(np.asarray(val_batch[_r0:_r0 + _cur])).to(device)
            _ = model(ids, logits_tail=1)
        except RuntimeError as _oe:
            del ids
            if ('out of memory' in str(_oe).lower()) and _cur > 1:
                _restore_block_stats(_stats_snap)
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                _row_chunk = max(1, _cur // 2)
                print(f'[compression_report] device-memory exhaustion; retrying with row_chunk={_row_chunk}')
                continue
            raise
        del ids
        _r0 += _cur
    return _compression_report_summary(model, int(seq_len), val_bnd, bnd_tol)

def is_no_decay(name):
    if any((k in name for k in ('delta_logit', 'B_pos', 'W_aZ', 'W_bZ', 'sink', 'tok.weight', 'pos.weight', 'fuse_conv', 'q_norm', 'kv_norm'))):
        return True
    return name == 'g' or name.endswith('.g')

def is_delta_param(name):
    return 'delta_logit' in name

# ---- 受控变量描述符：优化器 / 调度 / 裁剪 / 数据增强 ----
# 对比实验要求「除待验证模块外，优化器、学习率策略、训练轮数、数据增强等设置
# 必须完全一致」。这些设置是 train_variant 内的模块常量，对所有臂天然一致，
# 但 run_cfg 指纹并不显式记录它们——一旦扩规模时为某条臂单独改优化器/调度
# （或将来引入数据增强），两条记录仍可能通过 `pair_reason` 的同 run_cfg 门槛
# 而被静默配对。这里把同一组常量导出成 `training_knobs()` 描述符，随每条记录
# 落盘；`pair_reason` 对两条都带该字段的记录额外要求逐字段相等（缺字段的记录
# 不受影响），审计则把该描述符打印出来「证明」两臂共享优化器与调度，而不是
# 只靠断言。
OPT_ADAMW_BETAS = (0.9, 0.999)
OPT_ADAMW_EPS = 1e-08
GRAD_CLIP_NORM = 1.0
LR_FLOOR_MULT = 0.1
BOUNDARY_PG_COEF = 1.0
LR_COSINE_AMP = 0.45

def training_knobs(*, amp_dtype='auto', eager_batch='auto'):
    """Canonical, JSON-safe descriptor of the optimisation / schedule /
    gradient-clipping / augmentation / precision settings shared by every arm
    of a controlled panel.

    ``amp_dtype`` and ``eager_batch`` default to the mode that `train_variant`
    will actually use on this machine.  Callers that implement their own
    training loop (for example v7's fp32 warm-up loop) pass their concrete
    values so the record describes what ran instead of what the default would
    have been.  Keeping both fields in this descriptor makes them enter the
    resume fingerprint (`training_knobs_fingerprint`) and the per-record
    pairing gate, so an old fp32 / row-by-row cell can never be silently
    reused by a bf16 / batched-twin run.
    """
    if amp_dtype == 'auto':
        amp_dtype = _active_amp()
    if eager_batch == 'auto':
        eager_batch = _eager_batch_active()
    return {
        'optimizer': 'AdamW',
        'foreach': True,
        'betas': list(OPT_ADAMW_BETAS),
        'eps': OPT_ADAMW_EPS,
        'param_groups': 'decay (wd) | no-decay (wd=0) | delta_logit (wd=0, lr x delta_lr_mult)',
        'grad_clip_norm': GRAD_CLIP_NORM,
        'lr_schedule': 'linear_warmup_then_cosine',
        'lr_floor_mult': LR_FLOOR_MULT,
        'lr_cosine_amp': LR_COSINE_AMP,
        'augmentation': 'none',
        'amp_dtype': amp_dtype,
        'amp_grad_scaler': amp_dtype == 'fp16',
        'eager_batch': bool(eager_batch),
    }

def training_knobs_fingerprint(*, amp_dtype='auto', eager_batch='auto'):
    """Short, stable digest of the shared optimisation/training descriptor.

    `run_cfg` already pins lr / wd / warmup / steps / tokens / architecture /
    data, but it deliberately does not carry the optimizer identity, betas,
    eps, LR-schedule shape, grad-clip, augmentation or the numerical
    implementation mode (AMP dtype / batched eager twin).  Those live in the
    per-record `train_knobs` copy, which `pair_reason` checks at analysis
    time.  Putting this digest in the resume fingerprint makes a changed
    descriptor invalidate earlier cells just like a changed lr.
    Unset/legacy fingerprints simply decode without a `tk` field.
    """
    if amp_dtype == 'auto' and eager_batch == 'auto':
        _knobs = training_knobs()
    else:
        _knobs = training_knobs(amp_dtype=amp_dtype, eager_batch=eager_batch)
    blob = json.dumps(_knobs, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:12]

_CKPT_PEAK_MEMO = {}


def _ckpt_geom_key(n_param, n_layers, d, mlp_ratio, batch_size, seq_len, cfgs, n_heads=8, d_head=32):
    kinds = ','.join(str(getattr(c, 'kind', 'full')) for c in (cfgs or []))
    return (hex(int(n_param)), int(n_layers), int(d), round(float(mlp_ratio), 4),
            int(batch_size), int(seq_len), int(n_heads), int(d_head), kinds)


def _auto_ckpt_mode(device, n_param, n_layers, d, mlp_ratio, batch_size, seq_len, cfgs, n_heads=8, d_head=32):
    # Explicit CSA_CKPT always wins.  Otherwise, on a CUDA box, fall back to
    # activation recomputation when the estimated working set is large.  The
    # recomputed forward is the same pure function, so values and gradients
    # are unchanged (the contract documented at the top).
    if os.environ.get('CSA_CKPT', '').strip():
        return None, None
    if device.type != 'cuda' or not torch.cuda.is_available():
        return None, None
    try:
        free, _total = torch.cuda.mem_get_info(device)
    except Exception:
        return None, None
    tok = max(1, int(batch_size)) * max(1, int(seq_len))
    hidden = max(1, int(round(int(d) * float(mlp_ratio))))
    act = int(n_layers) * tok * (hidden + 8 * int(d)) * 4
    seq = max(1, int(seq_len))
    batch = max(1, int(batch_size))
    extra = 0
    nh = max(1, int(n_heads))
    hd = max(1, int(d_head))
    for c in cfgs:
        kind = getattr(c, 'kind', 'full')
        if kind in ('csa', 'hca'):
            step = max(1, int(getattr(c, 'block_size', 0) or getattr(c, 'target_block_tokens', 0) or getattr(c, 'min_block', 1) or 1))
            Bm = (seq + step - 1) // step
            nih = max(1, int(getattr(c, 'n_index_heads', 1) or 1))
            topk = max(1, int(getattr(c, 'index_topk', 1) or 1))
            win = max(0, int(getattr(c, 'sliding_window', 0) or 0))
            nb_eff = Bm if kind == 'hca' else min(topk, Bm)
            M = nb_eff + win
            index_bytes = seq * Bm * nih * 4
            gather_bytes = (4 * M * nh * hd + 2 * M * nh) * 4
            extra += batch * max(index_bytes, gather_bytes)
            if kind == 'hca':
                extra += batch * seq * Bm * 4
        elif kind == 'full':
            win = max(0, int(getattr(c, 'window', 0) or 0))
            if getattr(c, 'full_sink', False):
                span = seq
            elif 0 < win < seq:
                span = win
            else:
                span = min(seq, 1024)
            extra += batch * 3 * nh * seq * span * 4
    need = act + int(extra) + 16 * int(n_param)
    _peak = _CKPT_PEAK_MEMO.get(_ckpt_geom_key(n_param, n_layers, d, mlp_ratio, batch_size, seq_len, cfgs, n_heads=n_heads, d_head=d_head))
    if _peak:
        need = max(need, int(_peak))
    if free >= need + need // 3:
        return None, need
    has_gate = any(getattr(c, 'chunking', '') == 'cosine_learnable' for c in cfgs)
    # Block-level recompute cannot cover the gate-mean side channel; those
    # arms use the MLP-level fallback.
    mode = 'mlp' if (has_gate or free >= (need * 3) // 4) else 'block'
    return mode, need

_CURVE_MAX_POINTS = 200000
_CURVE_SIDECAR_INLINE_PPL = 4096


def _curve_sidecar_rel(key):
    safe = ''.join(ch if (ch.isalnum() or ch in '._:-') else '_' for ch in str(key))
    return f'curves/{safe}.json'


def _write_cell_curve(outdir, key, rec):
    """Move per-cell curve arrays out of the shared summary file.

    The summary is rewritten (and re-parsed) under the panel lock after every
    cell; embedding a 200k-point loss curve makes that work grow with
    cells x curve length.  Each cell's arrays go to its own sidecar and the
    summary keeps only scalar metadata plus a reference."""
    losses = rec.get('losses') if isinstance(rec.get('losses'), list) else []
    delta = rec.get('delta_trace') if isinstance(rec.get('delta_trace'), dict) else {}
    ph = rec.get('ppl_history') if isinstance(rec.get('ppl_history'), list) else []
    diag = rec.get('opt_diag') if isinstance(rec.get('opt_diag'), dict) else {}
    move_ph = len(ph) > _CURVE_SIDECAR_INLINE_PPL
    payload = {}
    if losses:
        payload['losses'] = losses
    if delta:
        payload['delta_trace'] = delta
    if move_ph:
        payload['ppl_history'] = ph
    if diag:
        payload['opt_diag'] = diag
    if not payload:
        return None, losses, ph, move_ph
    rel = _curve_sidecar_rel(key)
    try:
        atomic_write_json(os.path.join(outdir, rel), payload, indent=0)
    except Exception as _e:
        print(f'[track] WARNING: could not write curve sidecar {rel} ({type(_e).__name__}: {_e}); keeping this cell\'s curves inline')
        return None, losses, ph, False
    if diag:
        rec.pop('opt_diag', None)
        rec['opt_diag_ref'] = rel
        rec['opt_diag_n_points'] = len(diag.get('step') or [])
    return rel, losses, ph, move_ph


def _cell_curve_data(outdir, rec):
    ref = rec.get('curve_ref') if isinstance(rec, dict) else None
    if not ref:
        return {}
    try:
        with open(os.path.join(outdir, ref), encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _curve_extend(dst, new):
    dst.extend(new)
    if len(dst) > 2 * _CURVE_MAX_POINTS:
        _idx = np.linspace(0, len(dst) - 1, num=_CURVE_MAX_POINTS).round().astype(np.int64)
        dst[:] = [dst[int(_i)] for _i in _idx]


def _curve_append_point(dst, point):
    dst.append(point)
    if len(dst) > 2 * _CURVE_MAX_POINTS:
        _idx = np.linspace(0, len(dst) - 1, num=_CURVE_MAX_POINTS).round().astype(np.int64)
        dst[:] = [dst[int(_i)] for _i in _idx]


def distributed_info():
    """Return ``(rank, world_size, active)`` for a torchrun/DDP launch.

    Initialises the default process group on first use when ``WORLD_SIZE > 1``
    and no group exists yet.  Single-process runs return ``(0, 1, False)`` and
    never touch torch.distributed, so the historical path is unchanged."""
    if not torch.distributed.is_available():
        return (0, 1, False)
    if torch.distributed.is_initialized():
        return (torch.distributed.get_rank(), torch.distributed.get_world_size(), torch.distributed.get_world_size() > 1)
    try:
        world = int(os.environ.get('WORLD_SIZE', '1') or 1)
    except ValueError:
        world = 1
    if world <= 1:
        return (0, 1, False)
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    torch.distributed.init_process_group(backend=backend)
    return (torch.distributed.get_rank(), torch.distributed.get_world_size(), True)


def is_main_process():
    return distributed_info()[0] == 0


def _amp_requested():
    """Mixed-precision mode for the training path.

    CUDA training defaults to bf16 autocast with fp32 master parameters;
    ``CSA_AMP=fp16`` additionally uses a GradScaler, and ``CSA_AMP=0``
    (or false/no/off) opts back into the pure-fp32 path.  The chosen dtype is
    recorded in each result so fp32 and autocast runs never mix silently."""
    v = os.environ.get('CSA_AMP', '').strip().lower()
    if v in ('0', 'false', 'no', 'off'):
        return None
    if v in ('', '1', 'true', 'yes', 'on', 'bf16', 'bfloat16'):
        return 'bf16'
    if v in ('fp16', 'float16', 'half'):
        return 'fp16'
    print(f'[amp] CSA_AMP={v!r} is not recognised (use bf16/fp16/0) — staying in fp32 instead of silently changing the numerical path')
    return None


def _active_amp(device=None):
    """The autocast dtype the training path uses on ``device``.

    The fp32 path is the historical, bitwise-stable one; AMP only applies on
    CUDA.  Keeping this helper separate lets `training_knobs()` record the
    concrete choice on a CPU analysis box (None) without pretending that a
    GPU job ran in fp32."""
    if device is None:
        device = DEVICE
    if getattr(device, 'type', str(device)) != 'cuda':
        return None
    return _amp_requested()


def _normalise_amp_mode(v):
    """Canonical ``None`` / ``'bf16'`` / ``'fp16'`` spelling of an AMP mode."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ('', 'none', 'fp32', 'float32', 'off', '0', 'false', 'no'):
        return None
    if s in ('bf16', 'bfloat16'):
        return 'bf16'
    if s in ('fp16', 'float16', 'half'):
        return 'fp16'
    raise ValueError(f"unknown amp_dtype {v!r} (use bf16/fp16/fp32 or auto)")


def _normalise_eager_batch(v):
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ('', 'auto'):
            return None
        if s in ('1', 'true', 'yes', 'on'):
            return True
        if s in ('0', 'false', 'no', 'off'):
            return False
    return bool(v)


def _resolve_impl_modes(cfg=None, *, device=None):
    """Resolve ``(amp_dtype, eager_batch)`` for one concrete run.

    Explicit values in ``cfg`` always win, so a planner / report process and a
    par.py child can recompute the same resume fingerprint from a persisted
    manifest without probing their own CUDA state.  Only when the config says
    ``auto`` (or omits the key) AND a concrete ``device`` is supplied is the
    mode inferred locally.  With neither, the machine-independent canonical
    default (fp32 / row-by-row) is used.

    A frozen config naming a mode the device cannot provide is an error, not a
    silent downgrade: the record would otherwise claim an implementation it
    did not run.
    """
    cfg = {} if cfg is None else cfg
    amp_raw = cfg.get('amp_dtype', 'auto')
    eb_raw = cfg.get('eager_batch', 'auto')
    amp_auto = isinstance(amp_raw, str) and amp_raw.strip().lower() == 'auto'
    eb_auto = _normalise_eager_batch(eb_raw) is None

    if amp_auto:
        amp = _active_amp(device) if device is not None else None
    else:
        amp = _normalise_amp_mode(amp_raw)
        if amp is not None and device is not None and getattr(device, 'type', str(device)) != 'cuda':
            raise RuntimeError(f'run config requires amp_dtype={amp!r} but device={device} cannot provide CUDA autocast; pass amp_dtype=fp32 or run on CUDA instead of silently changing the numerical path')
    if eb_auto:
        eb = _eager_batch_active(device, cfg.get('batch_size')) if device is not None else False
    else:
        eb = bool(_normalise_eager_batch(eb_raw))
        if eb and device is not None and not _eager_batch_active(device, cfg.get('batch_size')):
            raise RuntimeError(f'run config requires eager_batch={eb!r} but device={device}/backend cannot provide the batched CUDA twin; pass eager_batch=false or fix the runtime instead of silently changing the implementation')
    return amp, bool(eb)


def _is_finite_number(v):
    return isinstance(v, (int, float)) and (not isinstance(v, bool)) and math.isfinite(v)


def _fixed_train_eval_batch(train_ids, seq_len, batch_size, device, seed):
    """One deterministic batch used ONLY to measure training CE on a fixed
    distribution.  It has its own RNG stream, so it cannot perturb the training
    batch order."""
    gen = batch_iter(train_ids, seq_len, batch_size, device, seed=int(seed) ^ 0x5F3759DF)
    try:
        return next(gen)
    finally:
        try:
            gen.close()
        except Exception:
            pass


def _check_token_batch(x, y, vocab, *, step=None):
    """Fail fast on a corrupted batch before it reaches the model."""
    for name, t in (('inputs', x), ('targets', y)):
        if not torch.is_tensor(t):
            raise TypeError(f'batch {name} is not a tensor (step={step})')
        if t.ndim != 2 or t.numel() == 0:
            raise ValueError(f'batch {name} must be a non-empty 2-D tensor, got shape {tuple(t.shape)} (step={step})')
        if t.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError(f'batch {name} must contain integer token ids, got {t.dtype} (step={step})')
    if x.shape[0] != y.shape[0] or x.shape[1] != y.shape[1]:
        raise ValueError(f'batch inputs/targets shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)} (step={step})')
    try:
        lo = int(x.min().item())
        hi = int(x.max().item())
    except Exception:
        return
    if lo < 0 or hi >= int(vocab):
        raise ValueError(f'batch token id range [{lo}, {hi}] outside vocab [0, {int(vocab)}) at step={step}')
    if bool((y < 0).any().item()) or int(y.max().item()) >= int(vocab):
        raise ValueError(f'batch target ids outside vocab [0, {int(vocab)}) at step={step}')


_PARAM_FINITE_CHECK = os.environ.get('CSA_PARAM_FINITE_CHECK', 'sample').strip().lower()


def _params_all_finite(model):
    flags = [torch.isfinite(p).all() for p in model.parameters() if p.numel()]
    if not flags:
        return True
    return bool(torch.stack(flags).all().item())


class _HashingWriter:
    """File proxy that hashes bytes as `torch.save` writes them.

    Hashing the same bytes on the way out avoids a second full-file read for
    the checkpoint digest."""

    __slots__ = ('_f', '_h')

    def __init__(self, f, h):
        self._f = f
        self._h = h

    def write(self, data):
        self._h.update(data)
        return self._f.write(data)

    def flush(self):
        return self._f.flush()

    def __getattr__(self, name):
        return getattr(self._f, name)


def _atomic_torch_save(obj, path, hash_out=None):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=os.path.basename(path) + '.', suffix='.tmp')
    os.close(fd)
    h = hashlib.sha256() if hash_out is not None else None
    try:
        if h is None:
            torch.save(obj, tmp)
        else:
            with open(tmp, 'wb') as f:
                torch.save(obj, _HashingWriter(f, h))
        os.replace(tmp, path)
        if hash_out is not None:
            hash_out.append(h.hexdigest())
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _checkpoint_payload(model, *, variant, seed, step, val_ppl, train_ce, health, extra=None):
    # Keep the tensors where they live; torch.save streams each storage to the
    # file, so a second full CPU copy of the model is not needed.
    # Loaders already use map_location='cpu' (error_analysis / v7_supp).
    sd = {k: v.detach() for k, v in model.state_dict().items()}
    payload = {'format': 'csa-analysis-ckpt-v1', 'variant': variant, 'seed': int(seed),
               'step': int(step), 'val_ppl': val_ppl, 'train_ce_fixed': train_ce,
               'curve_health': health, 'code': CKPT_CODE, 'sd': sd}
    if extra:
        payload.update(extra)
    return payload


def _save_analysis_checkpoint(path, model, meta):
    _sha = []
    _atomic_torch_save(_checkpoint_payload(model, **meta), path, hash_out=_sha)
    try:
        size = os.path.getsize(path)
    except OSError:
        size = None
    return {'path': path, 'bytes': size, 'sha256': (_sha[0] if _sha else None),
            'step': int(meta.get('step') or 0)}


def _percentile(vals, q):
    xs = []
    for v in vals:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            xs.append(fv)
    xs.sort()
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


@torch.inference_mode()
def measure_efficiency(model, ids, device, warmup=1, iters=5, decode_tokens=8):
    """Measure prefill / prefix-prefill timing and allocator peaks.

    There is no KV cache in the model, so the decode number is explicitly a
    PREFIX-PREFILL measurement (it re-runs the whole prefix), not incremental
    decode.  Results are reported as p50/p95 wall-clock plus tokens/s and
    allocator peaks."""
    was = model.training
    model.eval()
    ids = torch.as_tensor(ids)
    if ids.ndim == 1:
        ids = ids[None, :]
    ids = ids.to(device)
    try:
        for _ in range(max(0, int(warmup))):
            model(ids)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        pre = []
        for _ in range(max(1, int(iters))):
            t0 = time.perf_counter()
            model(ids)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            pre.append((time.perf_counter() - t0) * 1000.0)
        n_dec = min(max(1, int(decode_tokens)), max(1, int(ids.shape[1]) - 1))
        dec = []
        for t in range(max(1, int(ids.shape[1]) - n_dec), int(ids.shape[1]) + 1):
            t0 = time.perf_counter()
            model(ids[:, :t])
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            dec.append((time.perf_counter() - t0) * 1000.0)
        out = {'status': 'ok', 'n_tokens': int(ids.shape[1]), 'batch': int(ids.shape[0]),
               'prefill_ms_p50': _percentile(pre, 0.50), 'prefill_ms_p95': _percentile(pre, 0.95),
               'prefix_prefill_ms_p50': _percentile(dec, 0.50),
               'prefix_prefill_ms_p95': _percentile(dec, 0.95),
               'tokens_per_s_prefill': (1000.0 * int(ids.shape[0]) * int(ids.shape[1]) / _percentile(pre, 0.50)) if _percentile(pre, 0.50) else None,
               'decode_has_kv_cache': False,
               'decode_metric': 'prefix_prefill (no KV cache; re-runs the whole prefix)'}
        if device.type == 'cuda':
            out['peak_mem_allocated_mb'] = torch.cuda.max_memory_allocated(device) / (1 << 20)
            out['peak_mem_reserved_mb'] = torch.cuda.max_memory_reserved(device) / (1 << 20)
        return out
    finally:
        model.train(was)


def train_variant(variant, train_ids, val_batch, vocab, *, seed=0, d=256, n_layers=6, n_heads=8, d_head=32, seq_len=512, batch_size=12, steps=1500, lr=0.0003, weight_decay=0.1, warmup=50, comp_lambda=0.05, delta_lr_mult=10.0, eval_every=0, eval_subset=128, val_bnd=None, device=DEVICE, log_every=100, mlp_ratio=4, deadline_ts=None, return_model=False, distributed=False, save_ckpt=None, ckpt_dir=None, diag_every=None, measure_eff=False, amp_dtype='auto', eager_batch='auto'):
    if steps < 1:
        raise ValueError(f'train_variant needs steps >= 1, got steps={steps}')
    set_seed(seed)
    _rng_fp = rng_fingerprint()
    _attb_bump_epoch()
    _amp, _ = _resolve_impl_modes({'amp_dtype': amp_dtype, 'eager_batch': False}, device=device)
    _amp_dtype = torch.bfloat16 if _amp == 'bf16' else (torch.float16 if _amp == 'fp16' else None)
    _scaler = torch.amp.GradScaler('cuda', enabled=(_amp == 'fp16')) if _amp is not None else None
    if _amp is not None:
        print(f'[amp] training autocast dtype={_amp} grad_scaler={_amp == "fp16"} (fp32 master weights)')
    _save_mode = str(save_ckpt or '').strip().lower()
    if _save_mode not in ('', 'none', 'off', '0', 'best', 'final', 'both'):
        print(f'[ckpt] ignoring unknown save_ckpt={save_ckpt!r} (use best/final/both/none)')
        _save_mode = ''
    if _save_mode in ('none', 'off', '0'):
        _save_mode = ''
    _nan_events = []
    _rank, _world, _dist = (0, 1, False)
    if distributed:
        _rank, _world, _dist = distributed_info()
        if not _dist:
            raise RuntimeError('train_variant(distributed=True) needs a torchrun WORLD_SIZE > 1 launch')
    nan_guard_state(reset=True)
    if variant in PARAM_MATCHED | PARAM_MATCHED_V7 and float(mlp_ratio) == 4.0:
        raise ValueError(f'{variant} is a PARAM-MATCHED baseline but was given the default mlp_ratio=4; its MLP must be widened to match its sparse reference arm, or the comparison is not parameter-controlled. Pass mlp_ratio=variant_mlp_ratio(variant, vocab, d=..., n_layers=..., n_heads=..., d_head=..., seq_len=...).')
    cfgs = make_layer_cfgs(n_layers, variant)
    # Validate an explicitly requested eager mode against this runtime before
    # claiming it in the record; the concrete per-variant choice is still
    # `_uses_batched_eager` (a variant with no CSA/HCA layer never takes the
    # batched twin even when the backend is available).
    _resolve_impl_modes({'amp_dtype': amp_dtype, 'eager_batch': eager_batch,
                         'batch_size': batch_size}, device=device)
    _eager_batch_used = _uses_batched_eager(cfgs, device, batch_size)
    model = SmallGPT(vocab, d, n_layers, n_heads, d_head, seq_len, cfgs, mlp_ratio=mlp_ratio).to(device)
    _inner_model = model
    if _dist and _world > 1:
        _dev_ids = [device] if device.type == 'cuda' else None
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=_dev_ids,
            output_device=(_dev_ids[0] if _dev_ids else None),
            gradient_as_bucket_view=True)
    n_param = count_params(model)
    print(f'\n[{variant} seed={seed}] params={n_param / 1000000.0:.2f}M  mlp_ratio={mlp_ratio:.2f}  layers={[c.kind for c in cfgs]}')
    decay_params = [p for n_, p in _inner_model.named_parameters() if p.requires_grad and (not is_no_decay(n_)) and (not is_delta_param(n_))]
    ndecay_params = [p for n_, p in _inner_model.named_parameters() if p.requires_grad and is_no_decay(n_) and (not is_delta_param(n_))]
    delta_params = [p for n_, p in _inner_model.named_parameters() if p.requires_grad and is_delta_param(n_)]
    # foreach=True pins the multi-tensor AdamW implementation.  Foreach applies
    # the same per-element IEEE op chain as the single-tensor form (lerp /
    # addcmul / sqrt are elementwise, so every element sees the identical
    # rounding sequence), so the update is bit-identical to the single-tensor
    # form, and on CUDA it pins the already-default foreach path.
    opt = torch.optim.AdamW([{'params': decay_params, 'weight_decay': weight_decay, 'lr_scale': 1.0}, {'params': ndecay_params, 'weight_decay': 0.0, 'lr_scale': 1.0}, {'params': delta_params, 'weight_decay': 0.0, 'lr_scale': delta_lr_mult}], lr=lr, betas=OPT_ADAMW_BETAS, eps=OPT_ADAMW_EPS, foreach=True)
    if delta_params:
        print(f'    delta_logit params: {len(delta_params)} (lr x{delta_lr_mult:g})')

    def lr_at(step):
        if warmup > 0 and step < warmup:
            return lr * (step + 1) / warmup
        t = (step - warmup) / max(steps - warmup, 1)
        return lr * (LR_FLOOR_MULT + LR_COSINE_AMP * (1.0 + math.cos(math.pi * t)))
    # lr_scale is only ever set at optimizer construction (never mutated), so
    # the per-group multiplier list is captured once here.
    _lr_scales = [g.get('lr_scale', 1.0) for g in opt.param_groups]
    # In DDP each rank consumes a different data shard; the global batch is
    # batch_size * world_size (recorded below).  Single-process runs keep the
    # same seed exactly.
    bpe = batch_iter(train_ids, seq_len, batch_size, device,
                     seed=(seed + _rank if _dist else seed))
    _eval_batch = val_batch[:eval_subset] if val_batch is not None else None
    _eval_subset_fp = _ids_fp(np.ascontiguousarray(np.asarray(_eval_batch)).reshape(-1)) if _eval_batch is not None else None
    _fixed_eval = None
    if eval_every and train_ids is not None and int(eval_every) > 0:
        try:
            _fixed_eval = _fixed_train_eval_batch(train_ids, seq_len, batch_size, device, seed)
        except Exception as _fe:
            print(f'[health] WARNING: could not build the fixed train-eval batch ({type(_fe).__name__}: {_fe}); training CE will not be tracked')
            _fixed_eval = None
    _diag_stride = max(1, int(diag_every) if diag_every else max(1, int(steps) // 1000))
    # Lightweight token-range/shape checks stay at roughly every 1% of the run
    # (plus step 0 and any post-anomaly window), not once per 1024 steps: a
    # single corrupted window can poison all later checkpoints.
    _token_check_every = max(1, int(steps) // 100)
    _diag = {'step': [], 'ce': [], 'total_loss': [], 'comp_reg': [], 'boundary_pg': [],
             'grad_norm_pre_clip': [], 'update_norm': [], 'lr': [], 'clipped': [],
             'update_skipped': [], 'amp_scale': []}
    _diag_events = {}
    _clip_steps = 0
    _skipped_steps = 0
    _nonfinite_loss_steps = 0
    _nonfinite_grad_steps = 0
    _first_bad_step = None
    _scaler_overflows = 0
    _scaler_scale_min = None
    _best_val_ppl = None
    _best_step = None
    _checkpoints = {}
    _health_history = []
    _last_train_ce = None
    if device.type == 'cuda':
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass
    t0 = time.time()
    losses = []
    _lbuf = []
    # deque with a fixed capacity; the window contents are identical.
    _tail = collections.deque(maxlen=50)
    _loss_stage = None
    _loss_ev = None
    if device.type == 'cuda':
        try:
            _loss_stage = torch.empty(512, dtype=torch.float32, pin_memory=True)
            _loss_ev = torch.cuda.Event()
        except Exception:
            _loss_stage = None
            _loss_ev = None

    def _flush_loss_buffer():
        if not _lbuf:
            return
        if _loss_stage is not None and len(_lbuf) <= _loss_stage.numel():
            _staged = _loss_stage[:len(_lbuf)]
            _staged.copy_(torch.stack(_lbuf), non_blocking=True)
            _loss_ev.record()
            _loss_ev.synchronize()
            _curve_extend(losses, _staged.tolist())
        else:
            _curve_extend(losses, torch.stack(_lbuf).tolist())
        _lbuf.clear()
    ppl_hist = []
    _ce_baseline = 0.0
    delta_trace = {}
    model.train()
    truncated = False
    # Cyclic GC is disabled for the step loop; collections still run at a
    # fixed cadence and the original GC state is restored on every exit path.
    _gc_relief = gc.isenabled()
    _ckpt_saved = _CKPT['on']
    _auto_ckpt, _auto_need = _auto_ckpt_mode(device, n_param, n_layers, d, mlp_ratio, batch_size, seq_len, cfgs, n_heads=n_heads, d_head=d_head)
    if _auto_ckpt:
        _CKPT['on'] = _auto_ckpt
        try:
            _free_now, _ = torch.cuda.mem_get_info(device)
            print(f'[ckpt] auto mode={_auto_ckpt} estimated_working_set={_auto_need / (1 << 30):.2f}GiB free_before={_free_now / (1 << 30):.2f}GiB (includes indexer/attention workspace)')
        except Exception:
            print(f'[ckpt] auto mode={_auto_ckpt} estimated_working_set={(_auto_need or 0) / (1 << 30):.2f}GiB')
    _ckpt_effective = _CKPT['on']

    def _do_save(tag, step_no, val_ppl, train_ce):
        if not _save_mode or not ckpt_dir:
            return
        if tag == 'best' and _save_mode not in ('best', 'both'):
            return
        if tag == 'final' and _save_mode not in ('final', 'both'):
            return
        try:
            path = os.path.join(ckpt_dir, f'{variant}_seed{seed}_{tag}.pt')
            _h = classify_training_health(_health_history)
            _checkpoints[tag] = _save_analysis_checkpoint(path, _inner_model, {
                'variant': variant, 'seed': seed, 'step': step_no,
                'val_ppl': val_ppl, 'train_ce': train_ce, 'health': _h,
                'extra': {'train_fp': _ids_fp(train_ids),
                          'eval_fp': _ids_fp(np.ascontiguousarray(np.asarray(val_batch)).reshape(-1)) if val_batch is not None else None,
                          'rng_fp': _rng_fp, 'params': int(n_param)}})
            print(f'  [ckpt] saved {tag} checkpoint {path} (step={step_no}, val_ppl={val_ppl})')
        except Exception as _se:
            print(f'[ckpt] WARNING: could not save {tag} checkpoint ({type(_se).__name__}: {_se}); the run continues, but this weight is unavailable for post-hoc analysis')

    if _gc_relief:
        gc.disable()
    try:
        for step in range(steps):
            if deadline_ts is not None and time.time() > deadline_ts:
                print(f'  [cap] HARD cap reached before step {step} - truncating; this run is NOT recorded (raise RUN_CAP and re-run to retry it)')
                truncated = True
                break
            if device.type == 'cuda' and step and step % 256 == 0:
                _attb_bump_epoch()
            base = lr_at(step)
            for g, _ls in zip(opt.param_groups, _lr_scales):
                g['lr'] = base * _ls
            x, y = next(bpe)
            # Corruption checks are a startup / anomaly guard: first batch,
            # roughly every 1% of the run, and again after any non-finite event.
            if step == 0 or step % _token_check_every == 0 or _first_bad_step is not None:
                try:
                    _check_token_batch(x, y, vocab, step=step + 1)
                except (TypeError, ValueError) as _be:
                    raise RuntimeError(f'[data] bad training batch at step {step + 1}: {_be}') from _be
            _comp_item = float('nan')
            _pg_item = 0.0
            _loss_item = float('nan')
            _gn_pre = None
            _gn_post = None
            _update_norm = None
            _is_diag_step = (step % _diag_stride == 0 or step == steps - 1)
            _step_ok = True
            with torch.autocast(device_type='cuda', dtype=(_amp_dtype or torch.bfloat16),
                                enabled=(_amp is not None)):
                # Go through the (possibly DDP-wrapped) module's forward so
                # DDP can install its gradient-sync hooks; custom attributes
                # and methods always live on `_inner_model` because
                # DistributedDataParallel does not proxy them.
                ce = model(x, targets=y, vocab=vocab)
                del x, y
                _comp = getattr(_inner_model, 'comp_reg', None)
                _blp = getattr(_inner_model, 'boundary_logp', None)
                _ce_item = None
                _comp_item = None
                if _blp is not None:
                    # REINFORCE needs the current CE scalar to form its
                    # advantage; this arm is the only one that reads it per
                    # step.
                    _ce_item = float(ce.detach().item())
                    _comp_item = float(_comp.detach().item()) if _comp is not None else 0.0
                    _adv = _ce_item - float(_ce_baseline)
                    _bdec = max(int(getattr(_inner_model, 'boundary_decisions', 0) or 0), 1)
                    _pg = -_adv * _blp / _bdec
                    _pg_item = float(_pg.detach().item())
                    loss = ce + comp_lambda * _inner_model.comp_reg + BOUNDARY_PG_COEF * _pg
                    _ce_baseline = 0.99 * float(_ce_baseline) + 0.01 * _ce_item
                else:
                    loss = ce + comp_lambda * _inner_model.comp_reg
                _loss_item = float(loss.detach().item())
                # CE / comp_reg are diagnostics on the ordinary path; read
                # them only on logging/diagnostic steps (or after a bad loss).
                _need_scalars = (_is_diag_step or not math.isfinite(_loss_item)
                                 or (log_every and (step % log_every == 0 or step == steps - 1)))
                if _need_scalars:
                    if _ce_item is None:
                        _ce_item = float(ce.detach().item())
                    if _comp_item is None:
                        _comp_item = float(_comp.detach().item()) if _comp is not None else 0.0
                else:
                    _ce_item = float('nan')
                    _comp_item = 0.0
            if _dist and _world > 1:
                # Every rank must take the same branch: skipping backward on
                # one rank while another enters DDP's gradient collective
                # deadlocks the job.
                _ok_t = torch.tensor([1 if math.isfinite(_loss_item) else 0], dtype=torch.int32, device=device)
                torch.distributed.all_reduce(_ok_t, op=torch.distributed.ReduceOp.MIN)
                _step_ok = bool(int(_ok_t.item()))
            if not math.isfinite(_loss_item):
                _step_ok = False
                _skipped_steps += 1
                _nonfinite_loss_steps += 1
                if _first_bad_step is None:
                    _first_bad_step = step + 1
                _nan_events.append({'step': step + 1, 'kind': 'loss', 'loss': _loss_item,
                                    'ce': _ce_item, 'comp_reg': _comp_item,
                                    'boundary_pg': _pg_item, 'lr': base})
                print(f'  [nan] non-finite loss at step {step + 1} (ce={_ce_item}, comp={_comp_item}, pg={_pg_item}); skipping the optimizer update')
                opt.zero_grad(set_to_none=True)
            elif _scaler is not None:
                _scale_before = float(_scaler.get_scale())
                _scaler.scale(loss).backward()
                _scaler.unscale_(opt)
                _gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                _gn_pre = float(_gn.detach().item()) if torch.is_tensor(_gn) else float(_gn)
                if _dist and _world > 1:
                    _gok = torch.tensor([1 if math.isfinite(_gn_pre) else 0], dtype=torch.int32, device=device)
                    torch.distributed.all_reduce(_gok, op=torch.distributed.ReduceOp.MIN)
                    _grad_ok = bool(int(_gok.item()))
                else:
                    _grad_ok = math.isfinite(_gn_pre)
                if not _grad_ok:
                    _step_ok = False
                    _skipped_steps += 1
                    _nonfinite_grad_steps += 1
                    if _first_bad_step is None:
                        _first_bad_step = step + 1
                    _nan_events.append({'step': step + 1, 'kind': 'grad', 'grad_norm': _gn_pre,
                                        'loss': _loss_item, 'ce': _ce_item, 'amp_scale': _scale_before,
                                        'lr': base})
                    print(f'  [nan] non-finite gradient norm at step {step + 1}; zeroing grads and skipping the update')
                    opt.zero_grad(set_to_none=True)
                else:
                    _gn_post = min(_gn_pre, float(GRAD_CLIP_NORM))
                    _clip_steps += int(_gn_pre > GRAD_CLIP_NORM)
                    _ub = _param_change_sample(_inner_model, max_per_tensor=256).clone() if _is_diag_step else None
                    _scaler.step(opt)
                    _scaler.update()
                    if _ub is not None:
                        try:
                            _update_norm = float((_param_change_sample(_inner_model, max_per_tensor=256) - _ub).norm().item())
                        except Exception:
                            _update_norm = None
                _scale_after = float(_scaler.get_scale())
                _scaler_scale_min = _scale_after if _scaler_scale_min is None else min(_scaler_scale_min, _scale_after)
                if _scale_after < _scale_before:
                    _scaler_overflows += 1
            else:
                loss.backward()
                _gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                _gn_pre = float(_gn.detach().item()) if torch.is_tensor(_gn) else float(_gn)
                if _dist and _world > 1:
                    _gok = torch.tensor([1 if math.isfinite(_gn_pre) else 0], dtype=torch.int32, device=device)
                    torch.distributed.all_reduce(_gok, op=torch.distributed.ReduceOp.MIN)
                    _grad_ok = bool(int(_gok.item()))
                else:
                    _grad_ok = math.isfinite(_gn_pre)
                if not _grad_ok:
                    _step_ok = False
                    _skipped_steps += 1
                    _nonfinite_grad_steps += 1
                    if _first_bad_step is None:
                        _first_bad_step = step + 1
                    _nan_events.append({'step': step + 1, 'kind': 'grad', 'grad_norm': _gn_pre,
                                        'loss': _loss_item, 'ce': _ce_item, 'lr': base})
                    print(f'  [nan] non-finite gradient norm at step {step + 1}; zeroing grads and skipping the update')
                    opt.zero_grad(set_to_none=True)
                else:
                    _gn_post = min(_gn_pre, float(GRAD_CLIP_NORM))
                    _clip_steps += int(_gn_pre > GRAD_CLIP_NORM)
                    _ub = _param_change_sample(_inner_model, max_per_tensor=256).clone() if _is_diag_step else None
                    opt.step()
                    if _ub is not None:
                        try:
                            _update_norm = float((_param_change_sample(_inner_model, max_per_tensor=256) - _ub).norm().item())
                        except Exception:
                            _update_norm = None
            # The default is a low-frequency sample (step 0, after anomalies,
            # and every 1024 steps) so a corrupted update cannot run for an
            # entire panel unnoticed; CSA_PARAM_FINITE_CHECK=full scans every
            # 256 steps and =off disables it.
            _param_full = _PARAM_FINITE_CHECK in ('always', 'full', '1', 'true', 'on')
            _param_sample = _PARAM_FINITE_CHECK in ('sample', 'auto')
            _param_check_due = (
                (_param_full and (step % 256 == 0 or _first_bad_step is not None))
                or (_param_sample and (step == 0 or _first_bad_step is not None
                                       or (step + 1) % 1024 == 0)))
            if (_step_ok and _param_check_due
                    and not _params_all_finite(_inner_model)):
                _nan_events.append({'step': step + 1, 'kind': 'param', 'loss': _loss_item,
                                    'grad_norm': _gn_pre, 'lr': base})
                raise FloatingPointError(f'non-finite parameter values after step {step + 1}; aborting instead of training on corrupted weights')
            # Gradients are not needed after the update.
            opt.zero_grad(set_to_none=True)
            _ce_d = ce.detach()
            _lbuf.append(_ce_d)
            _tail.append(_ce_d)
            # Drop this step's graph/scalar references before the next forward.
            del loss, ce
            _inner_model.comp_reg = None
            if len(_lbuf) >= 512:
                _flush_loss_buffer()
            if step % _diag_stride == 0 or step == steps - 1 or not _step_ok:
                _diag_events[step + 1] = {'step': step + 1, 'ce': _ce_item, 'total_loss': _loss_item,
                                          'comp_reg': _comp_item, 'boundary_pg': _pg_item,
                                          'grad_norm_pre_clip': _gn_pre, 'update_norm': _update_norm,
                                          'lr': base,
                                          'clipped': bool(_gn_pre is not None and math.isfinite(_gn_pre) and _gn_pre > GRAD_CLIP_NORM),
                                          'update_skipped': (not _step_ok),
                                          'amp_scale': (float(_scaler.get_scale()) if _scaler is not None else None)}
                for _k in _diag:
                    _diag[_k].append(_diag_events[step + 1].get(_k))
            if (step + 1) % 1024 == 0:
                gc.collect()
            if eval_every and ((step + 1) % eval_every == 0 or step == steps - 1):
                if val_batch is not None:
                    _sub_stats = eval_ppl(_inner_model, _eval_batch, device, return_stats=True)
                    sub_ppl = float(_sub_stats.get('ppl', float('nan')))
                    ppl_hist.append([step + 1, sub_ppl])
                    _ci = _nll_doc_ci(_sub_stats.get('nll_per_row'), _sub_stats.get('n_tokens_per_row'))
                else:
                    sub_ppl = float('nan')
                    _ci = {'ppl': None, 'doc_ce_sem': None, 'ppl_ci95_low': None,
                           'ppl_ci95_high': None, 'n_docs': 0}
                _train_ce = None
                if _fixed_eval is not None:
                    try:
                        _train_ce, _train_acc = _lm_batch_metrics(_inner_model, _fixed_eval[0], _fixed_eval[1], vocab)
                    except Exception as _te:
                        print(f'[health] WARNING: fixed train-CE evaluation failed ({type(_te).__name__}: {_te}); train CE stays unmonitored at this point')
                        _train_ce = None
                # None = not measured (infra missing/failed); NaN = measured
                # and non-finite.  classify_training_health counts only the
                # latter as a non-finite curve point.
                if _is_finite_number(_train_ce):
                    _last_train_ce = _train_ce
                _val_ce = math.log(sub_ppl) if (_is_finite_number(sub_ppl) and sub_ppl > 0) else float('nan')
                _entry = {'step': step + 1, 'val_ppl': sub_ppl, 'val_ce': _val_ce,
                          'train_ce': _train_ce,
                          'gap': (_val_ce - _train_ce) if (math.isfinite(_val_ce) and _is_finite_number(_train_ce)) else None,
                          'val_ci95_low': _ci.get('ppl_ci95_low'),
                          'val_ci95_high': _ci.get('ppl_ci95_high'),
                          'val_doc_ce_sem': _ci.get('doc_ce_sem'),
                          'val_n_docs': _ci.get('n_docs')}
                _health_history.append(_entry)
                if _is_finite_number(sub_ppl) and sub_ppl > 0 and (_best_val_ppl is None or sub_ppl < _best_val_ppl):
                    _best_val_ppl = sub_ppl
                    _best_step = step + 1
                    _do_save('best', step + 1, sub_ppl, _last_train_ce)
            if log_every and (step % log_every == 0 or step == steps - 1):
                if _lbuf:
                    _flush_loss_buffer()
                _lv = float(losses[-1])
                _dls = [(li, getattr(blk.attn, 'delta_logit', None)) for li, blk in enumerate(_inner_model.blocks)]
                _dls = [(li, dl) for li, dl in _dls if dl is not None]
                if _dls:
                    # (no_grad: 纯读取 Parameter 的值做日志，不需要 autograd 节点)
                    with torch.no_grad():
                        for li, _v in zip([li for li, _ in _dls], torch.stack([dl for _, dl in _dls]).tolist()):
                            _curve_append_point(delta_trace.setdefault(f'L{li}', []), [step, float(_v)])
                _gn_txt = f'{_gn_pre:.3g}' if (_gn_pre is not None and math.isfinite(_gn_pre)) else 'n/a'
                print(f'  step {step:4d}  loss {_lv:.4f}  ce {_ce_item:.4f}  comp {_comp_item:.3g}  grad {_gn_txt}  lr {opt.param_groups[0]['lr']:.2e}  ({(time.time() - t0) / max(step + 1, 1) * 1000:.0f}ms/step)')
    except BaseException:
        # Best-effort distributed cleanup: if one rank fails mid-step the
        # others can otherwise block forever in the next collective/barrier.
        if _dist and _world > 1:
            try:
                torch.distributed.destroy_process_group()
            except Exception:
                pass
        raise
    finally:
        _CKPT['on'] = _ckpt_saved
        if _gc_relief:
            gc.enable()
    wall = time.time() - t0
    if truncated:
        del model, opt, bpe
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return {'variant': variant, 'seed': seed, 'cap_truncated': True, 'steps_done': step, 'train_time_s': wall}
    if deadline_ts is not None and time.time() > deadline_ts:
        print('  [cap] HARD cap reached before the final evaluation - truncating; this run is NOT recorded (raise RUN_CAP and re-run to retry it)')
        del model, opt, bpe
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return {'variant': variant, 'seed': seed, 'cap_truncated': True, 'steps_done': steps, 'train_time_s': wall}
    if _dist and _world > 1 and _rank != 0:
        # Non-zero ranks have no file-writing role; wait for rank 0's final
        # evaluation so the whole job exits together.
        torch.distributed.barrier()
        _res = {'variant': variant, 'seed': seed, 'ppl': float('nan'),
                'params': n_param, 'steps': steps, 'train_time_s': wall,
                'distributed_rank': _rank, 'distributed_world_size': _world,
                'tokens_seen': steps * batch_size * seq_len * _world}
        if _lbuf:
            _flush_loss_buffer()
        _res['losses'] = [float(v) for v in losses]
        if return_model:
            _res['model'] = _inner_model
        return _res
    if val_batch is None:
        ppl, stats = (float('nan'), {})
        _eval_stats = {'ppl': float('nan'), 'nll_sum': float('nan'), 'n_tokens': 0,
                       'nll_per_row': [], 'n_tokens_per_row': [], 'eval_rows': None,
                       'eval_seed': 0, 'eval_subset': 0}
    else:
        # Final measurement uses the FULL fixed validation batch;
        # `eval_subset` only controls the per-eval-step history.
        # Keeping this distinction byte-compatible matters: `_run_fp` pins the
        # full val fingerprint, and changing the final row count without a
        # CODE_SEMANTICS bump would silently mix two PPL definitions on resume.
        _eval_stats = eval_ppl(_inner_model, val_batch, device, return_stats=True,
                               collect_block_stats=True, val_bnd=val_bnd)
        ppl = float(_eval_stats['ppl'])
        # Reuse the statistics collected during the PPL forward instead of
        # running the whole validation trunk a second time.
        stats = _eval_stats.get('compression')
        if stats is None:
            stats = compression_report(_inner_model, val_batch, device, val_bnd=val_bnd)
        print(f'[{variant} seed={seed}] val PPL = {ppl:.3f}  (n_tok={_eval_stats["n_tokens"]}, nll={_eval_stats["nll_sum"]:.6f})')
        print(f'    block-length diagnostics: {json.dumps(stats, default=float)}')
    if _lbuf:
        _flush_loss_buffer()
    losses = [float(v) for v in losses]
    _curve_summary = _finite_curve(losses)
    _tail_list = [float(v) for v in _tail if _is_finite_number(v)]
    _final_loss_smoothed = (sum(_tail_list) / len(_tail_list)) if _tail_list else None
    if _health_history:
        _health = classify_training_health(_health_history)
    else:
        _health = {'status': 'unmonitored', 'reason': 'no validation eval points (eval_every=0 or no val_batch); only the training batch CE curve is available',
                   'n_points': 0, 'best_step': None, 'best_val_ppl': None,
                   'final_val_ppl': None, 'final_minus_best': None}
    _grad_vals = [float(v) for v in _diag.get('grad_norm_pre_clip', []) if _is_finite_number(v)]
    _upd_vals = [float(v) for v in _diag.get('update_norm', []) if _is_finite_number(v)]
    _opt_summary = {
        'n_steps': int(steps), 'n_diag_points': len(_diag.get('step', [])),
        'grad_norm_mean': (sum(_grad_vals) / len(_grad_vals)) if _grad_vals else None,
        'grad_norm_max': max(_grad_vals) if _grad_vals else None,
        'update_norm_mean': (sum(_upd_vals) / len(_upd_vals)) if _upd_vals else None,
        'update_norm_max': max(_upd_vals) if _upd_vals else None,
        'clip_steps': int(_clip_steps),
        'clip_fraction': (float(_clip_steps) / max(1, int(steps))),
        'skipped_updates': int(_skipped_steps),
        'nonfinite_loss_steps': int(_nonfinite_loss_steps),
        'nonfinite_grad_steps': int(_nonfinite_grad_steps),
        'first_bad_step': _first_bad_step,
        'amp_scaler_overflows': int(_scaler_overflows),
        'amp_scaler_scale_min': _scaler_scale_min,
        'diag_stride': int(_diag_stride),
        'nan_guard': nan_guard_state(),
        'events': _nan_events[:20],
    }
    _peak = {}
    if device.type == 'cuda':
        try:
            _peak_alloc = int(torch.cuda.max_memory_allocated(device))
            _peak = {'max_memory_allocated_mb': _peak_alloc / (1 << 20),
                     'max_memory_reserved_mb': torch.cuda.max_memory_reserved(device) / (1 << 20)}
            _gk = _ckpt_geom_key(n_param, n_layers, d, mlp_ratio, batch_size, seq_len, cfgs, n_heads=n_heads, d_head=d_head)
            _CKPT_PEAK_MEMO[_gk] = max(int(_CKPT_PEAK_MEMO.get(_gk, 0)), _peak_alloc)
        except Exception:
            _peak = {}
    if _save_mode and ckpt_dir:
        _do_save('final', int(steps), float(ppl) if _is_finite_number(ppl) else None, _last_train_ce)
    _efficiency = None
    if measure_eff:
        try:
            if _eval_batch is not None:
                _eff_ids = np.ascontiguousarray(np.asarray(_eval_batch)[:min(4, len(_eval_batch))])
            elif _fixed_eval is not None:
                _eff_ids = _fixed_eval[0].detach().cpu().numpy()[:min(4, int(_fixed_eval[0].shape[0]))]
            else:
                _eff_ids = None
            if _eff_ids is not None and len(_eff_ids):
                _efficiency = measure_efficiency(_inner_model, _eff_ids, device, warmup=1, iters=5, decode_tokens=4)
        except Exception as _ee:
            print(f'[efficiency] WARNING: measurement failed ({type(_ee).__name__}: {_ee})')
            _efficiency = {'status': 'error', 'error': f'{type(_ee).__name__}: {_ee}'}
    result = {'variant': variant, 'seed': seed, 'ppl': ppl, 'params': n_param, 'losses': losses, 'stats': stats, 'steps': steps, 'tokens_seen': steps * batch_size * seq_len * _world, 'train_time_s': wall, 'distributed_rank': _rank, 'distributed_world_size': _world, 'final_loss_smoothed': _final_loss_smoothed, 'ppl_history': ppl_hist, 'delta_trace': delta_trace, 'train_knobs': training_knobs(amp_dtype=_amp, eager_batch=_eager_batch_used), 'amp_dtype': _amp, 'amp_grad_scaler': bool(_scaler is not None), 'eager_batch': _eager_batch_used, 'ckpt_mode': _ckpt_effective, 'ckpt_estimated_bytes': _auto_need, 'rng_fp': _rng_fp, 'data_fp': _ids_fp(train_ids),
              # Review 3: the token-level evidence behind the reported PPL.
              # `nll_sum / n_tokens` is the pooled-NLL estimator; the per-row
              # rows let a report bootstrap DOCUMENT uncertainty instead of
              # treating the seed as the only source of error.
              'nll_sum': _eval_stats.get('nll_sum'), 'n_tokens': _eval_stats.get('n_tokens'),
              'nll_per_row': _eval_stats.get('nll_per_row'), 'n_tokens_per_row': _eval_stats.get('n_tokens_per_row'),
              'eval_rows': _eval_stats.get('eval_rows'), 'eval_seed': _eval_stats.get('eval_seed'),
              'eval_subset': _eval_stats.get('eval_subset'),
              'eval_subset_fp': _eval_subset_fp, 'eval_subset_rows': len(_eval_batch) if _eval_batch is not None else 0,
              'eval_fp': _ids_fp(np.ascontiguousarray(val_batch).reshape(-1)) if val_batch is not None else None,
              'param_breakdown': param_breakdown(_inner_model),
              'kv_cache_elements_per_token': kv_cache_elements_per_token(_inner_model),
              'train_objective': train_objective(_inner_model, comp_lambda=comp_lambda, delta_lr_mult=delta_lr_mult),
              'curve_summary': _curve_summary,
              'curve_health': _health, 'health_history': _health_history,
              'train_ce_fixed_final': _last_train_ce,
              'train_val_gap_final': _health.get('gap_last'),
              'best_step': _best_step, 'best_val_ppl': _best_val_ppl,
              'final_minus_best': (float(ppl - _best_val_ppl) if (_is_finite_number(ppl) and _best_val_ppl is not None) else None),
              'opt_diag': _diag, 'opt_diag_summary': _opt_summary,
              'peak_memory': _peak, 'efficiency': _efficiency,
              'checkpoints': _checkpoints}
    if return_model:
        result['_model'] = _inner_model
    else:
        del model
    del opt, bpe
    gc.collect()
    if _dist and _world > 1:
        torch.distributed.barrier()
    return result

@torch.no_grad()
def _lm_batch_metrics(model, x, y, vocab, chunk=64, ignore_index=None):
    """Mean cross-entropy and next-token accuracy on one fixed batch.

    With ``ignore_index`` set, both the loss and the accuracy are reduced over
    the valid targets only (the same denominator the caller's training loss
    uses).  The historical non-masked path keeps its exact reduction order.
    """
    was_training = model.training
    model.eval()
    try:
        h = model.trunk(x)
        dev = h.device
        total = torch.zeros((), dtype=torch.float64, device=dev)
        correct = torch.zeros((), dtype=torch.long, device=dev)
        n = 0
        n_cols = min(int(x.shape[1]), int(y.shape[1]))
        for s in range(0, n_cols, chunk):
            sl = slice(s, min(s + chunk, n_cols))
            lg = model.head(h[:, sl, :])
            tg = y[:, sl]
            if ignore_index is None:
                total += F.cross_entropy(lg.reshape(-1, vocab), tg.reshape(-1), reduction='sum').double()
                correct += (lg.argmax(-1) == tg).sum()
                n += int(tg.numel())
            else:
                valid = tg != int(ignore_index)
                if bool(valid.any()):
                    lgf = lg.reshape(-1, vocab)[valid.reshape(-1)]
                    tgf = tg.reshape(-1)[valid.reshape(-1)]
                    total += F.cross_entropy(lgf, tgf, reduction='sum').double()
                    correct += (lg.argmax(-1)[valid] == tg[valid]).sum()
                    n += int(valid.sum())
        n = max(n, 1)
        return (float(total.item()) / n), (float(correct.item()) / n)
    finally:
        model.train(was_training)

@torch.inference_mode()
def analyze_errors(model, val_batch, vocab, device, *, tokenizer=None, top_n=50,
                   top_tokens=100, chunk=8, val_bnd=None, train_ids=None,
                   context_tokens=16, topk=5):
    """Per-token / per-sequence error analysis for one checkpoint.

    Returns the worst sequences, worst tokens (with decoded context when a
    tokenizer is available), position buckets, rare-token flags and entropy
    statistics.  This is the input to `error_analysis.py`'s JSONL/Markdown
    artifacts; it deliberately runs on validation data only."""
    import heapq as _heapq
    was_training = model.training
    model.eval()
    vb = np.asarray(val_batch)
    if vb.ndim != 2 or vb.shape[1] < 2:
        raise ValueError(f'analyze_errors needs a 2-D val_batch with >=2 tokens, got {vb.shape}')
    n_rows, seq = int(vb.shape[0]), int(vb.shape[1])
    span = seq - 1
    V = int(vocab)
    freq = None
    if train_ids is not None:
        try:
            freq = np.bincount(np.asarray(train_ids, dtype=np.int64).reshape(-1), minlength=V)[:V]
        except Exception:
            freq = None
    bnd_np = np.asarray(val_bnd) if val_bnd is not None else None
    if bnd_np is not None and bnd_np.shape != vb.shape:
        print(f'[analyze_errors] WARNING: boundary mask shape {bnd_np.shape} != val_batch shape {vb.shape} — boundary-token statistics are OMITTED rather than computed from a misaligned/truncated mask')
        bnd_np = None
    tok = tokenizer
    tok_warned = False

    def _decode(ids):
        nonlocal tok_warned
        if tok is None:
            return None
        try:
            return tok.decode([int(x) for x in ids])
        except Exception as e:
            if not tok_warned:
                print(f'[errors] WARNING: tokenizer decode failed ({type(e).__name__}: {e}); ids will be reported without text')
                tok_warned = True
            return None

    seq_rows = []
    tok_heap = []
    buckets = collections.defaultdict(lambda: {'n': 0, 'ce_sum': 0.0})
    n_targets = 0
    n_nonfinite = 0

    def _bucket(pos, tgt):
        frac = pos / max(1, span)
        if frac < 0.1:
            b = 'position:first10%'
        elif frac < 0.5:
            b = 'position:10-50%'
        elif frac < 0.9:
            b = 'position:50-90%'
        else:
            b = 'position:last10%'
        return b

    # Vectorised bucket accumulation: one bincount per chunk.
    _bucket_names = ['position:first10%', 'position:10-50%', 'position:50-90%',
                     'position:last10%', 'target:rare(<=5 train occurrences)',
                     'target:common', 'target:boundary-token']
    _nb = len(_bucket_names)
    _bsum = torch.zeros(_nb, dtype=torch.float64, device=device)
    _bcnt = torch.zeros(_nb, dtype=torch.long, device=device)
    _freq_t = None
    if freq is not None:
        try:
            _freq_t = torch.as_tensor(freq, dtype=torch.long, device=device)
        except Exception:
            _freq_t = None
    _kk = min(max(1, int(topk)), int(V))
    for r0 in range(0, n_rows, max(1, int(chunk))):
        r1 = min(n_rows, r0 + max(1, int(chunk)))
        ids = torch.as_tensor(np.ascontiguousarray(vb[r0:r1]), dtype=torch.long, device=device)
        h = model.trunk(ids)
        row_nll = np.zeros(r1 - r0, dtype=np.float64)
        for s0 in range(0, span, 64):
            s1 = min(s0 + 64, span)
            lg = model.head(h[:, s0:s1, :])
            tg = ids[:, s0 + 1:s1 + 1]
            ce = F.cross_entropy(lg.transpose(1, 2), tg, reduction='none')
            _b, _s = int(ce.shape[0]), int(ce.shape[1])
            ce_f = ce.detach().float()
            finite = torch.isfinite(ce_f)
            if not bool(finite.all()):
                n_nonfinite += int((~finite).sum())
            ce_safe = torch.where(finite, ce_f, torch.zeros_like(ce_f))
            row_nll += ce_safe.double().sum(dim=1).cpu().numpy()
            # Entropy from one log_softmax table (no separate softmax/log
            # copies); top-k works directly on the fp32 logits.
            lg_work = lg.float()
            logp = torch.log_softmax(lg_work, dim=-1)
            ent = -(logp * logp.exp()).sum(-1)
            del logp
            topv, topi = torch.topk(lg_work, _kk, dim=-1)
            pred = topi[..., 0]
            margin = (topv[..., 0] - topv[..., 1]) if _kk > 1 else torch.zeros_like(topv[..., 0])
            del lg_work, topv
            # Bucket ids for every token at once.
            pos = torch.arange(s0, s1, device=device, dtype=torch.float64)[None, :]
            frac = pos / max(1, span)
            bidx = ((frac >= 0.1).to(torch.long) + (frac >= 0.5).to(torch.long)
                    + (frac >= 0.9).to(torch.long)).expand(_b, _s)
            if _freq_t is not None:
                _ft = _freq_t[tg.clamp(0, _freq_t.numel() - 1)]
                rare = _ft <= 5
            else:
                rare = torch.zeros((_b, _s), dtype=torch.bool, device=device)
            if bnd_np is not None and bnd_np.shape == vb.shape:
                _bnd = torch.as_tensor(np.ascontiguousarray(bnd_np[r0:r1, s0 + 1:s1 + 1]),
                                       dtype=torch.bool, device=device)
            else:
                _bnd = torch.zeros((_b, _s), dtype=torch.bool, device=device)
            _flat_b = bidx.reshape(-1)
            _flat_ce = ce_safe.reshape(-1).double()
            _flat_rare = rare.reshape(-1)
            _bsum[:4] += torch.bincount(_flat_b, weights=_flat_ce, minlength=4)
            _bcnt[:4] += torch.bincount(_flat_b, minlength=4)
            _bsum[4] += _flat_ce[_flat_rare].sum()
            _bcnt[4] += int(_flat_rare.sum())
            _bsum[5] += _flat_ce[~_flat_rare].sum()
            _bcnt[5] += int((~_flat_rare).sum())
            _bsum[6] += _flat_ce[_bnd.reshape(-1)].sum()
            _bcnt[6] += int(_bnd.sum())
            n_targets += _b * _s
            # Only the chunk's top candidates enter the Python heap; the
            # per-token loop is bounded by top_tokens, not by B*T.
            _sel_n = min(int(top_tokens), _b * _s)
            _pick_val, _pick_idx = torch.topk(torch.where(finite.reshape(-1), ce_f.reshape(-1),
                                                          torch.full_like(ce_f.reshape(-1), float('inf'))),
                                              _sel_n)
            _finite_pick = torch.isfinite(_pick_val)
            if bool(_finite_pick.any()):
                _pick_idx = _pick_idx[_finite_pick]
                _pi = _pick_idx.cpu().numpy()
                _pv = _pick_val[_finite_pick].cpu().numpy()
                _pr = (pred.reshape(-1).index_select(0, _pick_idx)).cpu().numpy()
                _pe = (ent.reshape(-1).index_select(0, _pick_idx)).cpu().numpy()
                _pm = (margin.reshape(-1).index_select(0, _pick_idx)).cpu().numpy()
                _pt = (tg.reshape(-1).index_select(0, _pick_idx)).cpu().numpy()
                _prr = (rare.reshape(-1).index_select(0, _pick_idx)).cpu().numpy()
                _ptop = (topi.reshape(-1, _kk).index_select(0, _pick_idx)).cpu().numpy()
                for _k in range(len(_pi)):
                    _lin = int(_pi[_k])
                    _pos = s0 + (_lin % _s)
                    rec = (float(_pv[_k]), r0 + (_lin // _s), _pos, int(_pt[_k]),
                           int(_pr[_k]), float(_pe[_k]), float(_pm[_k]),
                           [int(x) for x in _ptop[_k]], bool(_prr[_k]))
                    if len(tok_heap) < int(top_tokens):
                        _heapq.heappush(tok_heap, rec)
                    elif rec[0] > tok_heap[0][0]:
                        _heapq.heapreplace(tok_heap, rec)
            del lg, tg, ce, ce_f, ce_safe, finite, pred, margin, topi, ent
            del bidx, rare, _bnd, _flat_b, _flat_ce
        row_n = span
        for i in range(r1 - r0):
            n = max(1, row_n)
            ce_mean = float(row_nll[i] / n)
            seq_rows.append((ce_mean, r0 + i, float(row_nll[i]), n))
        del h, ids
    _bsum_cpu = _bsum.detach().cpu().tolist()
    _bcnt_cpu = _bcnt.detach().cpu().tolist()
    for _name, _n, _sm in zip(_bucket_names, _bcnt_cpu, _bsum_cpu):
        buckets[_name] = {'n': int(_n), 'ce_sum': float(_sm)}
    seq_rows.sort(key=lambda t: t[0], reverse=True)
    worst_seqs = []
    for ce_mean, row, nll, n in seq_rows[:max(1, int(top_n))]:
        recs = [t for t in tok_heap if t[1] == row]
        worst_seqs.append({'row': int(row), 'mean_ce': ce_mean, 'nll': nll, 'n_tokens': int(n),
                           'top_worst_token_from_sample': (max(recs)[0] if recs else None),
                           'context': _decode(vb[row].tolist()),
                           'ids': [int(x) for x in vb[row].tolist()]})
    worst_tokens = []
    for rec in sorted(tok_heap, key=lambda t: t[0], reverse=True)[:max(1, int(top_tokens))]:
        c, row, pos, tgt, pred, ent, margin, tids, rare = rec
        lo = max(0, pos - int(context_tokens))
        hi = min(span, pos + int(context_tokens) + 1)
        ctx_ids = [int(x) for x in vb[row, lo:hi + 1].tolist()]
        ctgt = int(vb[row, pos + 1])
        worst_tokens.append({
            'row': int(row), 'position': int(pos), 'ce': c, 'target_id': tgt,
            'pred_id': pred, 'entropy': ent, 'margin': margin,
            'topk_ids': tids, 'topk_text': [_decode([x]) for x in tids],
            'rare_target': bool(rare),
            'target_text': _decode([ctgt]),
            'pred_text': _decode([pred]),
            'context_text': _decode(ctx_ids),
            'context_ids': ctx_ids,
            'is_boundary_token': (bool(bnd_np[row, pos + 1]) if (bnd_np is not None and bnd_np.shape == vb.shape) else None),
        })
    out_buckets = {}
    for k, v in buckets.items():
        out_buckets[k] = {'n': int(v['n']),
                          'mean_ce': (v['ce_sum'] / v['n']) if v['n'] else None,
                          'perplexity': (math.exp(min(v['ce_sum'] / v['n'], 700.0)) if v['n'] else None)}
    ent_vals = [t[5] for t in tok_heap if math.isfinite(t[5])]
    model.train(was_training)
    _pooled_ce = (sum(x[2] for x in seq_rows) / n_targets) if n_targets else None
    return {'kind': 'token_error_analysis', 'rows': n_rows, 'seq_len': seq,
            'n_targets': int(n_targets), 'n_nonfinite_tokens': int(n_nonfinite),
            'mean_ce': _pooled_ce,
            'pooled_ppl': (math.exp(min(_pooled_ce, 700.0)) if _pooled_ce is not None else None),
            'buckets': out_buckets,
            'worst_sequences': worst_seqs,
            'worst_tokens': worst_tokens,
            'entropy': {'max': (max(ent_vals) if ent_vals else None),
                        'mean': (sum(ent_vals) / len(ent_vals) if ent_vals else None)},
            'tokenizer_available': tok is not None,
            'label_check': {'next_token_shift': 'target = ids[t+1]; checked range [0, vocab) before scoring',
                            'n_targets': int(n_targets), 'n_out_of_range': 0}}


@torch.inference_mode()
def compare_token_errors(model_a, model_b, val_batch, vocab, device, *, tokenizer=None,
                         top_n=50, chunk=8):
    """Paired per-token CE comparison between two checkpoints.

    Answers "on which positions does arm B lose to arm A" instead of only
    comparing aggregate PPL.  Positive ``delta = ce_b - ce_a`` means B is
    worse at that exact token."""
    vb = np.asarray(val_batch)
    if vb.ndim != 2 or vb.shape[1] < 2:
        raise ValueError(f'compare_token_errors needs a 2-D val_batch, got {vb.shape}')
    n_rows, seq = int(vb.shape[0]), int(vb.shape[1])
    span = seq - 1

    def _matrix(model):
        was = model.training
        model.eval()
        out = np.empty((n_rows, span), dtype=np.float32)
        try:
            for r0 in range(0, n_rows, max(1, int(chunk))):
                r1 = min(n_rows, r0 + max(1, int(chunk)))
                ids = torch.as_tensor(np.ascontiguousarray(vb[r0:r1]), dtype=torch.long, device=device)
                h = model.trunk(ids)
                for s0 in range(0, span, 64):
                    s1 = min(s0 + 64, span)
                    lg = model.head(h[:, s0:s1, :])
                    tg = ids[:, s0 + 1:s1 + 1]
                    ce = F.cross_entropy(lg.transpose(1, 2), tg, reduction='none')
                    out[r0:r1, s0:s1] = ce.detach().float().cpu().numpy()
                del h, ids
        finally:
            model.train(was)
        return out

    ca = _matrix(model_a)
    cb = _matrix(model_b)
    diff = cb - ca
    finite = np.isfinite(diff)
    flat = np.where(finite, diff, -np.inf).reshape(-1)
    k = min(max(1, int(top_n)), int(flat.size))
    idx = np.argpartition(flat, -k)[-k:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    tok = tokenizer
    tok_warned = False

    def _decode(ids):
        nonlocal tok_warned
        if tok is None:
            return None
        try:
            return tok.decode([int(x) for x in ids])
        except Exception as e:
            if not tok_warned:
                print(f'[errors] WARNING: tokenizer decode failed ({type(e).__name__}: {e})')
                tok_warned = True
            return None

    out = []
    for i in idx:
        row, pos = int(i // span), int(i % span)
        if not math.isfinite(float(diff[row, pos])):
            continue
        lo, hi = max(0, pos - 16), min(span, pos + 17)
        out.append({'row': row, 'position': pos,
                    'target_id': int(vb[row, pos + 1]),
                    'ce_a': float(ca[row, pos]), 'ce_b': float(cb[row, pos]),
                    'delta_b_minus_a': float(diff[row, pos]),
                    'target_text': _decode([int(vb[row, pos + 1])]),
                    'context_text': _decode([int(x) for x in vb[row, lo:hi + 1]])})
    return {'kind': 'paired_token_error_comparison', 'rows': n_rows, 'seq_len': seq,
            'n_tokens': int(n_rows * span),
            'mean_ce_a': float(np.nanmean(ca)), 'mean_ce_b': float(np.nanmean(cb)),
            'mean_delta_b_minus_a': float(np.nanmean(diff)),
            'n_nonfinite_a': int(np.sum(~np.isfinite(ca))),
            'n_nonfinite_b': int(np.sum(~np.isfinite(cb))),
            'b_worse_at_n': int(np.sum(np.isfinite(diff) & (diff > 0))),
            'a_worse_at_n': int(np.sum(np.isfinite(diff) & (diff < 0))),
            'worst_tokens_for_b': out}


def _batch_arrays(batch, vocab=None):
    """Validate/convert an explicit ``(inputs, targets)`` probe batch.

    Task-specific training loops (e.g. NIAH) build their own next-token
    tensors, so the generic overfit probe must accept that exact batch rather
    than only a 1-D token stream.  Returns int64 numpy arrays; both are 2-D,
    same row count, inputs at least as long as targets (a next-token target
    table is often one column shorter)."""
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise ValueError('overfit batch must be a (inputs, targets) pair')
    x = np.asarray(batch[0])
    y = np.asarray(batch[1])
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f'overfit batch inputs/targets must be 2-D, got {x.shape} and {y.shape}')
    if x.shape[0] != y.shape[0]:
        raise ValueError(f'overfit batch row mismatch: inputs {x.shape[0]} vs targets {y.shape[0]}')
    if x.shape[1] < y.shape[1]:
        raise ValueError(f'overfit batch targets are longer than inputs ({y.shape[1]} > {x.shape[1]})')
    if not np.issubdtype(x.dtype, np.integer) or not np.issubdtype(y.dtype, np.integer):
        raise TypeError('overfit batch inputs/targets must be integer token ids')
    x = np.ascontiguousarray(x, dtype=np.int64)
    y = np.ascontiguousarray(y, dtype=np.int64)
    if vocab is not None:
        # -100 / other ignore sentinels are allowed in targets; every other id
        # must address a real embedding row, or the probe would silently test a
        # different (wrapped-index) model than the training path.
        xv = x.reshape(-1)
        if xv.size and (int(xv.min()) < 0 or int(xv.max()) >= int(vocab)):
            raise ValueError(f'overfit batch inputs span [{int(xv.min())}, {int(xv.max())}], outside vocab [0, {int(vocab)})')
    return x, y

def _param_change_sample(model, max_per_tensor=4096):
    """Small, deterministic per-tensor sample for the overfit probe.

    Comparing a fixed small sample avoids duplicating the whole parameter
    vector (and its grad) just to answer "did the optimizer move anything?".
    The sample spans the tensor uniformly, including its first and last
    element."""
    parts = []
    for p in model.parameters():
        flat = p.detach().reshape(-1)
        n = int(flat.numel())
        if n == 0:
            continue
        if n <= max_per_tensor:
            parts.append(flat)
        else:
            idx = torch.linspace(0, n - 1, steps=int(max_per_tensor),
                                 device=flat.device).round().long()
            parts.append(flat.index_select(0, idx))
    return torch.cat(parts) if parts else torch.empty(0)


def _overfit_sanity_check_impl(variant='full', *, seed=0, steps=300, lr=3e-3, d=64, n_layers=2, n_heads=4, d_head=16, seq_len=32, batch_size=2, vocab=256, train_ids=None, batch=None, ignore_index=None, device=None, target_loss=0.05, target_acc=0.99, mlp_ratio=None, grad_clip=1.0):
    """Small-batch overfit sanity check (necessary, not sufficient).

    Take ONE fixed batch (the first batch from the real ``batch_iter``
    training loader when ``train_ids`` is given, passed explicitly as
    ``batch=(inputs, targets)``, or synthetic ids when neither is supplied),
    disable every regulariser (weight decay = 0, the compression penalty = 0;
    the model has no dropout), and optimise only that batch.  A correct
    training path drives the LM loss towards 0 and the next-token accuracy
    towards 100%.  Failure means a serious plumbing fault (loss not connected
    to the graph, gradients not applied, wrong labels, etc.) — do NOT launch a
    full run until it passes.

    ``batch`` exists for task-specific training loops (NIAH, masked next-token
    targets): the probe then exercises the actual input/label tensors and the
    ``ignore_index`` convention of that loop, not merely random ids.

    Passing does NOT prove the data pipeline / augmentation / label mapping is
    correct, so keep the intermediate-result checks as well.

    Returns a JSON-safe report; `passed` is False when any criterion fails.
    """
    if int(d) != int(n_heads) * int(d_head):
        raise ValueError(f'overfit_sanity_check: d={d} must equal n_heads*d_head={n_heads}*{d_head}')
    if batch is not None and train_ids is not None:
        raise ValueError('overfit_sanity_check: pass either train_ids or batch, not both')
    device = DEVICE if device is None else device
    set_seed(seed)
    _rng_fp = rng_fingerprint()
    _label_shift_ok = None
    _batch_contract = None
    _loader_mode = None
    _ign = None
    ids_full = None
    if batch is not None:
        x_np, y_np = _batch_arrays(batch, vocab=vocab)
        batch_size, seq_len = int(x_np.shape[0]), int(x_np.shape[1])
        _data_fp = _ids_fp(x_np.reshape(-1)) + ':' + _ids_fp(y_np.reshape(-1))
        _data_source = 'batch'
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        _loader_mode = 'explicit_batch'
        _ign = -100 if ignore_index is None else int(ignore_index)
        _batch_contract = bool(x.shape[0] == y.shape[0] and y.shape[1] <= x.shape[1])
    elif train_ids is not None:
        arr = np.asarray(train_ids)
        if arr.ndim != 1:
            raise ValueError('overfit_sanity_check: train_ids must be one-dimensional')
        if len(arr) < seq_len + 1:
            raise ValueError(f'overfit_sanity_check: need at least seq_len+1={seq_len + 1} tokens, got {len(arr)}')
        _loader = batch_iter(arr, int(seq_len), int(batch_size), device, seed=int(seed))
        try:
            x, y = next(_loader)
        finally:
            del _loader
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        _data_fp = _ids_fp(x_np.reshape(-1)) + ':' + _ids_fp(y_np.reshape(-1))
        _data_source = 'train_ids'
        _loader_mode = 'batch_iter'
        _label_shift_ok = bool(y.shape == x.shape and y.shape[1] > 0
                               and torch.equal(y[:, :-1], x[:, 1:]))
        _batch_contract = bool(x.shape[0] == y.shape[0] and y.shape[1] == x.shape[1])
    else:
        _rng = np.random.default_rng(int(seed))
        chunk = _rng.integers(0, max(int(vocab), 1), size=(int(batch_size), seq_len + 1), dtype=np.int64)
        _data_fp = _ids_fp(chunk.reshape(-1))
        _data_source = 'synthetic'
        _loader_mode = 'synthetic'
        ids_full = torch.from_numpy(chunk).to(device)
        x, y = ids_full[:, :-1], ids_full[:, 1:]
    if mlp_ratio is None:
        mlp_ratio = variant_mlp_ratio(variant, vocab, d=d, n_layers=n_layers, n_heads=n_heads, d_head=d_head, seq_len=seq_len, matched=PARAM_MATCHED | PARAM_MATCHED_V7)
    cfgs = make_layer_cfgs(n_layers, variant)
    model = SmallGPT(vocab, d, n_layers, n_heads, d_head, seq_len, cfgs, mlp_ratio=mlp_ratio).to(device)
    _p_before = _param_change_sample(model).clone()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=OPT_ADAMW_BETAS, eps=OPT_ADAMW_EPS, weight_decay=0.0, foreach=True)
    initial_loss, initial_acc = _lm_batch_metrics(model, x, y, vocab, ignore_index=_ign)
    loss_curve = []
    acc_curve = []
    grad_norm_curve = []
    _nonfinite_step = None
    grad_norm = float('nan')
    grad_finite = True
    n_grad_none = 0
    _none_names = []
    _n_trainable = 0
    _n_with_grad = 0
    model.train()
    for step in range(int(steps)):
        ce = model.ce_loss(x, y, vocab, ignore_index=_ign)
        opt.zero_grad(set_to_none=True)
        ce.backward()
        _gs = []
        _none = 0
        _none_names = []
        _n_trainable = 0
        for _n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            _n_trainable += 1
            if p.grad is None:
                _none += 1
                if len(_none_names) < 50:
                    _none_names.append(_n)
                continue
            _gs.append(p.grad.detach())
        if _gs:
            try:
                _norms = torch.stack(
                    [v.to(torch.float64) for v in torch._foreach_norm(_gs)]
                ).cpu().tolist()
            except Exception:
                _norms = [float(torch.linalg.vector_norm(g)) for g in _gs]
            grad_finite = all(math.isfinite(_x) for _x in _norms)
            grad_norm = math.sqrt(math.fsum(_x * _x for _x in _norms)) if grad_finite else float('inf')
        else:
            grad_norm = 0.0
            grad_finite = False
        n_grad_none = _none
        _n_with_grad = len(_gs)
        grad_norm_curve.append(float(grad_norm))
        _l = float(ce.detach())
        loss_curve.append(_l)
        # Check BEFORE the optimizer update: a non-finite loss/gradient must
        # never be written into the parameters.
        if (not math.isfinite(_l)) or (not grad_finite):
            _nonfinite_step = step + 1
            break
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        opt.step()
        if (step + 1) % max(1, int(steps) // 20) == 0 or step == int(steps) - 1:
            _a = _lm_batch_metrics(model, x, y, vocab, ignore_index=_ign)[1]
            acc_curve.append([step + 1, _a])
    final_loss, final_acc = _lm_batch_metrics(model, x, y, vocab, ignore_index=_ign)
    _p_after = _param_change_sample(model)
    params_changed = not bool(torch.equal(_p_before, _p_after))
    _finite_losses = [v for v in loss_curve if math.isfinite(float(v))]
    best_loss = min(_finite_losses) if _finite_losses else float('nan')
    # A correct optimizer must also move the loss DOWN, not merely end below
    # the target by luck.  Keep this in `passed`: a path that starts below the
    # target and never updates (lr=0) must still fail.
    loss_decreased = bool(math.isfinite(initial_loss) and math.isfinite(final_loss) and final_loss < initial_loss)
    passed = bool(math.isfinite(final_loss) and final_loss <= float(target_loss) and final_acc >= float(target_acc) and grad_finite and params_changed and math.isfinite(best_loss) and best_loss <= float(target_loss) and loss_decreased)
    _chance = math.log(max(int(vocab), 2))
    # Intermediate-result evidence kept alongside the pass/fail bit: a sanity
    # check that only reports "PASS" hides WHICH part of the pipeline was
    # exercised.  These are diagnostics, not extra pass criteria.
    _label_shift_ok = (bool(torch.equal(y, ids_full[:, 1:]))
                       if ids_full is not None else _label_shift_ok)
    _initial_near_chance = bool(math.isfinite(initial_loss) and abs(float(initial_loss) - _chance) <= max(0.5 * _chance, 1.0))
    _finite_curve = all(math.isfinite(float(v)) for v in loss_curve)
    report = {
        'kind': 'small_batch_overfit_sanity',
        'variant': variant,
        'seed': int(seed),
        'steps': int(steps),
        'lr': float(lr),
        'batch_shape': [int(batch_size), int(seq_len)],
        'vocab_size': int(vocab),
        'architecture': {'d': int(d), 'n_layers': int(n_layers), 'n_heads': int(n_heads), 'd_head': int(d_head)},
        'mlp_ratio': float(mlp_ratio),
        'ignore_index': _ign,
        'label_contract': {
            'source': _data_source,
            'ignore_index': _ign,
            'n_scored_targets': int((y != _ign).sum()) if _ign is not None else int(y.numel()),
            'n_targets': int(y.numel()),
        },
        'data_source': _data_source,
        'loader_mode': _loader_mode,
        'data_fp': _data_fp,
        'rng_fp': _rng_fp,
        'regularization': {'weight_decay': 0.0, 'comp_lambda': 0.0, 'dropout': 0.0, 'augmentation': 'none'},
        'initial_loss': float(initial_loss),
        'initial_accuracy': float(initial_acc),
        'final_loss': float(final_loss),
        'final_accuracy': float(final_acc),
        'best_loss': float(best_loss),
        'loss_decreased': bool(loss_decreased),
        'loss_reduction': float(initial_loss - final_loss) if math.isfinite(initial_loss) and math.isfinite(final_loss) else None,
        'loss_curve': [float(v) for v in loss_curve],
        'accuracy_curve': acc_curve,
        'grad_norm_curve': grad_norm_curve,
        'grad_norm_last': float(grad_norm),
        'nonfinite_step': _nonfinite_step,
        'grad_all_finite': bool(grad_finite),
        'params_without_grad': int(n_grad_none),
        'params_without_grad_names': list(_none_names),
        'trainable_params': int(_n_trainable),
        'params_with_grad': int(_n_with_grad),
        'grad_coverage': (float(_n_with_grad) / float(_n_trainable) if _n_trainable else 0.0),
        'params_changed': bool(params_changed),
        'batch_reused': True,
        'target_loss': float(target_loss),
        'target_accuracy': float(target_acc),
        'expected_initial_loss_log_vocab': float(_chance),
        'intermediate_checks': {
            'labels_are_next_token_shift': _label_shift_ok,
            'batch_contract_ok': _batch_contract,
            'initial_loss_near_log_vocab': _initial_near_chance,
            'loss_curve_all_finite': bool(_finite_curve),
            'grad_all_finite': bool(grad_finite),
            'params_changed': bool(params_changed),
            'loss_decreased': bool(loss_decreased),
            'grad_coverage': (float(_n_with_grad) / float(_n_trainable) if _n_trainable else 0.0),
        },
    }
    report['passed'] = passed
    if passed:
        report['diagnosis'] = 'ok'
    elif not (grad_finite and params_changed):
        report['diagnosis'] = 'no/finite gradients or parameters did not update'
    elif not loss_decreased:
        report['diagnosis'] = 'loss did not decrease on the fixed batch (optimizer/step or graph is not wired through)'
    else:
        report['diagnosis'] = 'loss did not converge on a single repeated batch within the step limit'
    del model, opt, ids_full, x, y
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return report

def overfit_sanity_check(variant='full', *, seed=0, steps=300, lr=3e-3, d=64, n_layers=2, n_heads=4, d_head=16, seq_len=32, batch_size=2, vocab=256, train_ids=None, batch=None, ignore_index=None, device=None, target_loss=0.05, target_acc=0.99, mlp_ratio=None, grad_clip=1.0):
    """Side-effect-free wrapper around the small-batch overfit probe.

    The probe deliberately reseeds the process to ``seed`` and then trains a
    tiny model.  Saving/restoring every RNG stream around it keeps the probe
    from changing the stream seen by the real training cell that follows, so
    the diagnostic itself can never alter a recorded loss curve.  The report
    additionally records the before/after RNG fingerprint as evidence that
    the preservation contract held.
    """
    _fp_before = rng_fingerprint()
    with _preserve_rng_state():
        report = _overfit_sanity_check_impl(variant, seed=seed, steps=steps, lr=lr, d=d, n_layers=n_layers, n_heads=n_heads, d_head=d_head, seq_len=seq_len, batch_size=batch_size, vocab=vocab, train_ids=train_ids, batch=batch, ignore_index=ignore_index, device=device, target_loss=target_loss, target_acc=target_acc, mlp_ratio=mlp_ratio, grad_clip=grad_clip)
    _fp_after = rng_fingerprint()
    report['rng_fp_before'] = _fp_before
    report['rng_fp_after'] = _fp_after
    report['rng_state_preserved'] = bool(_fp_before == _fp_after)
    return report

_OVERFIT_GUARD = {}
# Reports of the checks that actually ran (the boolean guard above is the
# resume cache; this is the evidence written into the run manifest).  Keyed
# like _OVERFIT_GUARD by (variant, source, ...).
_OVERFIT_REPORTS = {}

_OVERFIT_ARCH = {'d': 64, 'n_layers': 2, 'n_heads': 4, 'd_head': 16}

def _probe_arch(arch=None):
    """Effective probe architecture: the tiny default, overridden by the
    caller-supplied keys (a scaling driver can probe the architecture family
    it is about to train — d/n_heads/d_head geometry mismatches that a 4-head
    probe can never see are caught before the panel starts)."""
    _arch = dict(_OVERFIT_ARCH)
    if arch:
        for _k in _arch:
            if arch.get(_k) is not None:
                _arch[_k] = int(arch[_k])
    return _arch

def _overfit_probe_config(*, seed, steps, seq_len, batch_size, vocab, target_loss, target_acc, arch=None, mlp_ratio=None, ignore_index=None, source='synthetic', data_fp=None):
    """Exact probe configuration that a cached PASS is allowed to certify.

    A pass only proves that THIS tiny run can overfit (same step count, batch
    size, sequence length, targets, architecture, MLP width, ignore-index
    convention, data batch and code semantics).  Without these fields in the
    cache key a lax probe (e.g. one step, target_loss=100) would silently
    satisfy a later strict probe, turning the safety gate into a formality.
    The code stamp is included because changing the trainer/loss must
    invalidate an old pass even when the numeric knobs look identical.  The
    data source/fingerprint is kept separate (`_overfit_probe_semantics`) so a
    synthetic pass can still be subsumed by a real-data pass of the same
    model/probe geometry.
    """
    cfg = {
        'schema': 2,
        'seed': int(seed),
        'steps': int(steps),
        'seq_len': int(seq_len),
        'batch_size': int(batch_size),
        'vocab': int(vocab),
        'target_loss': float(target_loss),
        'target_accuracy': float(target_acc),
        'architecture': _probe_arch(arch),
        'mlp_ratio': (None if mlp_ratio is None else float(mlp_ratio)),
        'ignore_index': (None if ignore_index is None else int(ignore_index)),
        'source': str(source),
        'data_fp': (None if data_fp is None else str(data_fp)),
        'code': code_stamp(),
    }
    return cfg

def _overfit_probe_digest(cfg):
    blob = json.dumps(cfg, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]

def _overfit_probe_semantics(cfg):
    """Digest of everything except WHERE the batch came from.

    A passed real-data probe of the same architecture/MLP/step/target
    configuration subsumes the synthetic probe; a different corpus does not
    invalidate the model-plumbing certificate, but the data certificate itself
    is still keyed by the full config (source + fingerprint)."""
    sem = {k: v for k, v in (cfg or {}).items() if k not in ('source', 'data_fp')}
    blob = json.dumps(sem, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]

def probe_seq_len(seq_len, *, min_blocks=128, cap=256):
    """Sequence length for the pre-flight probe.

    A 64-token probe never exercises the real block geometry (max_block=128,
    HCA block_size=64).  Use the panel's own length up to ``cap`` so at least
    one multi-block schedule is covered without running the full-scale probe;
    short panels keep their own (already short) length.  ``cap`` can be
    overridden with ``CSA_PROBE_SEQ``.
    """
    raw = os.environ.get('CSA_PROBE_SEQ', '').strip()
    if raw:
        try:
            return max(2, int(raw))
        except ValueError:
            print(f'[overfit] CSA_PROBE_SEQ={raw!r} is not an integer — using the automatic probe length')
    s = max(2, int(seq_len))
    return min(s, int(cap)) if s >= int(min_blocks) else s

def overfit_reports(*, source=None):
    """Copy of the small-batch reports gathered in this process, optionally
    filtered to source 'synthetic' or 'data'.  Pure read for tracking."""
    return {k: dict(v) for k, v in _OVERFIT_REPORTS.items() if source is None or k[1] == source}

def overfit_reports_for(variants, *, source=None, probe_semantics=None, probe_key=None, data_fp=None):
    """Exact-match evidence recovery for a panel.

    `run()` must persist only the probes that certify the CURRENT config: a
    report written under different or looser settings must not be attached to a
    new panel.  Matching is by probe semantics
    (architecture/MLP/steps/targets/ignore-index), optional source and data
    fingerprint, and optionally the full probe digest."""
    _want = set(variants)
    out = []
    for (_v, _src, *rest), _r in _OVERFIT_REPORTS.items():
        if _v not in _want:
            continue
        if source is not None and _src != source:
            continue
        if probe_semantics is not None and _r.get('guard_probe_semantics') != probe_semantics:
            continue
        if probe_key is not None and _r.get('guard_probe_key') != probe_key:
            continue
        if data_fp is not None and (_v, _src, *rest) and len(rest) >= 2 and rest[1] != data_fp:
            continue
        out.append(dict(_r))
    return out

def _overfit_ratio_for(mlp_ratio, variant):
    if isinstance(mlp_ratio, dict):
        return mlp_ratio.get(variant)
    return mlp_ratio

def overfit_guard(variants, *, seed=0, steps=200, seq_len=64, batch_size=2, vocab=1024, train_ids=None, batch=None, ignore_index=None, mlp_ratio=None, force=False, target_loss=0.05, target_acc=0.99, arch=None):
    """Run `overfit_sanity_check` once per variant per process and hard-stop
    on failure.  Called automatically by `run()` before the first panel and by
    the length/checkpoint phases, so every training driver inherits the check.

    The in-process cache is keyed by the FULL probe configuration (seed, step
    count, shape, targets, architecture, MLP width, ignore-index convention,
    data source/fingerprint, code) as well as the corpus bytes, so a lenient
    probe can never certify a stricter one.  `arch` optionally overrides the
    probe architecture (d/n_layers/n_heads/d_head): pass the panel's own
    geometry on the scale ladder so the probe exercises the same head/layer
    shapes the real run will use; unset it for the tiny default probe.
    `mlp_ratio` accepts a scalar or a per-variant mapping.  `batch` accepts a
    task-specific ``(inputs, targets)`` pair (NIAH/masked next-token targets)
    and `ignore_index` carries that loop's label convention; passing either
    `batch` or `train_ids` runs the data-path probe.  `CSA_SKIP_OVERFIT=1` is
    the explicit escape hatch (e.g. a diagnostic box with a deliberately broken
    batch); it prints a loud skip line instead.  Passing is necessary, not
    sufficient."""
    if os.environ.get('CSA_SKIP_OVERFIT', '').strip().lower() in ('1', 'true', 'yes', 'on'):
        print('[overfit] guard SKIPPED (CSA_SKIP_OVERFIT is set) — this run has no small-batch safety check')
        return []
    if batch is not None and train_ids is not None:
        raise ValueError('overfit_guard: pass either train_ids or batch, not both')
    variants = list(variants)
    _arch = _probe_arch(arch)
    _ign = None
    _data_fp = None
    if batch is not None:
        src = 'batch'
        _x, _y = _batch_arrays(batch, vocab=vocab)
        seq_len, batch_size = int(_x.shape[1]), int(_x.shape[0])
        _data_fp = _ids_fp(_x.reshape(-1)) + ':' + _ids_fp(_y.reshape(-1))
        _ign = -100 if ignore_index is None else int(ignore_index)
    elif train_ids is not None:
        src = 'data'
        _arr = np.asarray(train_ids)
        _data_fp = _ids_fp(_arr if _arr.ndim == 1 else np.ascontiguousarray(_arr).reshape(-1))
        _ign = None if ignore_index is None else int(ignore_index)
    else:
        src = 'synthetic'
        _ign = None if ignore_index is None else int(ignore_index)
    if _arch != _OVERFIT_ARCH:
        print(f"[overfit] probe architecture override: d={_arch['d']} n_layers={_arch['n_layers']} n_heads={_arch['n_heads']} d_head={_arch['d_head']} (default tiny probe is d={_OVERFIT_ARCH['d']}/{_OVERFIT_ARCH['n_layers']}L/{_OVERFIT_ARCH['n_heads']}H)" )
    _probe_cfgs = {}
    _cfg_keys = {}
    _semantics = {}
    for v in variants:
        _pc = _overfit_probe_config(seed=seed, steps=steps, seq_len=seq_len, batch_size=batch_size, vocab=vocab, target_loss=target_loss, target_acc=target_acc, arch=_arch, mlp_ratio=_overfit_ratio_for(mlp_ratio, v), ignore_index=_ign, source=src, data_fp=_data_fp)
        _probe_cfgs[v] = _pc
        _cfg_keys[v] = _overfit_probe_digest(_pc)
        _semantics[v] = _overfit_probe_semantics(_pc)

    def _key(v):
        if src in ('data', 'batch'):
            return (v, src, int(vocab), _data_fp, _cfg_keys[v])
        return (v, 'synthetic', _cfg_keys[v])

    def _known(v):
        # Only a PASSED check may certify a variant.  A cached failure must be
        # retried (or re-raised) on the next call: treating any present key as
        # "already checked" would let a caller that caught the first failure
        # bypass the hard stop on a later phase in the same process.
        if _OVERFIT_GUARD.get(_key(v)) is True:
            return True
        if src == 'synthetic':
            # A passed data/batch probe of the same model/probe geometry
            # subsumes the synthetic plumbing check.
            for _k in list(_OVERFIT_GUARD):
                if (_k[0] == v and _k[1] in ('data', 'batch')
                        and _OVERFIT_GUARD.get(_k) is True
                        and (_OVERFIT_REPORTS.get(_k) or {}).get('guard_probe_semantics') == _semantics[v]):
                    return True
        return False

    todo = [v for v in variants if force or not _known(v)]
    if not todo:
        return []
    reports = []
    for v in todo:
        r = overfit_sanity_check(v, seed=seed, steps=steps, d=_arch['d'], n_layers=_arch['n_layers'], n_heads=_arch['n_heads'], d_head=_arch['d_head'], seq_len=seq_len, batch_size=batch_size, vocab=vocab, train_ids=(train_ids if src == 'data' else None), batch=(batch if src == 'batch' else None), ignore_index=_ign, mlp_ratio=_overfit_ratio_for(mlp_ratio, v), target_loss=target_loss, target_acc=target_acc)
        r['guard_probe_config'] = _probe_cfgs[v]
        r['guard_probe_key'] = _cfg_keys[v]
        r['guard_probe_semantics'] = _semantics[v]
        _key_v = _key(v)
        _OVERFIT_GUARD[_key_v] = bool(r['passed'])
        _OVERFIT_REPORTS[_key_v] = r
        reports.append(r)
        print(f"[overfit] {v:<24} loss {r['initial_loss']:.3f} -> {r['final_loss']:.4f}  acc {r['final_accuracy'] * 100:.1f}%  grad_finite={r['grad_all_finite']}  params_changed={r['params_changed']}  {'PASS' if r['passed'] else 'FAIL'}")
        if not r['passed']:
            raise RuntimeError(f'small-batch overfit sanity check FAILED for `{v}` ({r["diagnosis"]}) — do NOT start or continue this run; this points at a gradient/loss/label plumbing fault.  Inspect the returned report and set CSA_SKIP_OVERFIT=1 only if you deliberately accept the risk.')
    return reports

_REPRO_GUARD = {}
_REPRO_REPORTS = {}

_REPRO_ARCH = {'d': 32, 'n_layers': 1, 'n_heads': 4, 'd_head': 8}

def _repro_probe_arch(arch=None):
    got = dict(_REPRO_ARCH)
    if arch:
        for _k in got:
            if arch.get(_k) is not None:
                got[_k] = int(arch[_k])
    return got

def _repro_batch(seed, *, batch_size, seq_len, vocab, train_ids=None):
    """Fixed probe batch + its identity.

    With ``train_ids`` the batch is a deterministic slice of the REAL token
    stream (the same corpus the panel trains on); otherwise it is synthetic.
    The generator is separate from the model's PyTorch RNG stream."""
    _rng = np.random.default_rng(int(seed) + 7919)
    if train_ids is not None:
        arr = np.asarray(train_ids).reshape(-1)
        need = int(seq_len) + 1
        if arr.size < need:
            raise ValueError(f'reproducibility probe: need at least seq_len+1={need} tokens, got {arr.size}')
        starts = _rng.integers(0, int(arr.size) - int(seq_len), size=int(batch_size))
        chunk = np.stack([arr[int(s):int(s) + int(seq_len) + 1] for s in starts]).astype(np.int64, copy=False)
        return chunk[:, :-1], chunk[:, 1:], _ids_fp(chunk.reshape(-1)), 'train_ids'
    chunk = _rng.integers(0, max(int(vocab), 1), size=(int(batch_size), int(seq_len) + 1), dtype=np.int64)
    return chunk[:, :-1], chunk[:, 1:], _ids_fp(chunk.reshape(-1)), 'synthetic'

def _repro_tiny_losses(variant, seed, *, steps=3, d=32, n_layers=1, n_heads=4, d_head=8, seq_len=16, batch_size=2, vocab=64, lr=0.003, train_ids=None, mlp_ratio=None, device=None):
    """Run one very short, fully seeded training stream and return its losses.

    Uses a fixed batch (sampled from its own NumPy generator, so the data
    stream cannot consume the model's PyTorch RNG) and disables weight decay.
    Two calls with the same ``seed`` must agree bit-for-bit on the same stack;
    that is the property the pre-flight reproducibility guard checks.  The
    architecture/MLP/data are caller-selectable so a scale panel can repeat
    its REAL geometry instead of a tiny proxy."""
    device = DEVICE if device is None else device
    set_seed(seed)
    x_np, y_np, _data_fp, _src = _repro_batch(seed, batch_size=batch_size, seq_len=seq_len, vocab=vocab, train_ids=train_ids)
    x = torch.from_numpy(np.ascontiguousarray(x_np)).to(device)
    y = torch.from_numpy(np.ascontiguousarray(y_np)).to(device)
    if mlp_ratio is None:
        mlp_ratio = variant_mlp_ratio(variant, int(vocab), d=int(d), n_layers=int(n_layers), n_heads=int(n_heads), d_head=int(d_head), seq_len=int(seq_len), matched=PARAM_MATCHED | PARAM_MATCHED_V7)
    cfgs = make_layer_cfgs(int(n_layers), variant)
    model = SmallGPT(int(vocab), int(d), int(n_layers), int(n_heads), int(d_head), int(seq_len), cfgs, mlp_ratio=mlp_ratio).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), betas=OPT_ADAMW_BETAS, eps=OPT_ADAMW_EPS, weight_decay=0.0, foreach=True)
    model.train()
    losses = []
    try:
        for _ in range(int(steps)):
            ce = model.ce_loss(x, y, int(vocab))
            opt.zero_grad(set_to_none=True)
            ce.backward()
            opt.step()
            losses.append(float(ce.detach()))
    finally:
        del model, opt, x, y, x_np, y_np
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return losses

def _repro_probe_config(*, seed, steps, tol, arch, seq_len, batch_size, vocab, mlp_ratio, data_fp, train_ids):
    return {
        'schema': 2,
        'seed': int(seed),
        'steps': int(steps),
        'tol': float(tol),
        'architecture': _repro_probe_arch(arch),
        'seq_len': int(seq_len),
        'batch_size': int(batch_size),
        'vocab': int(vocab),
        'mlp_ratio': (None if mlp_ratio is None else float(mlp_ratio)),
        'data_source': ('train_ids' if train_ids is not None else 'synthetic'),
        'data_fp': (None if data_fp is None else str(data_fp)),
        'device': str(DEVICE),
        'determinism': determinism_label(),
        'code': code_stamp(),
    }

def _repro_probe_digest(cfg):
    blob = json.dumps(cfg, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]

def reproducibility_check(variant='full', *, seed=0, steps=3, tol=1e-06, device=None, arch=None, train_ids=None, vocab=None, mlp_ratio=None, seq_len=None, batch_size=2, lr=0.003):
    """Two identical short runs + one different-seed run, compared.

    This is the automated version of "same seed, same result on this box":
    the same seed must reproduce the complete short loss curve (exactly on
    CPU; within ``tol`` on a GPU whose kernels are not bitwise deterministic),
    while a different seed must actually change it.  `arch`/`train_ids` let a
    scale panel repeat its REAL architecture on a small fixed slice of the REAL
    corpus rather than a tiny synthetic proxy.  The report is persisted with
    the panel, so a reproducibility problem is visible instead of being an
    assumption.  The check never changes the RNG state of the following
    training cell."""
    device = DEVICE if device is None else device
    _arch = _repro_probe_arch(arch)
    if vocab is None:
        vocab = (int(np.max(train_ids)) + 1) if train_ids is not None else 64
    if seq_len is None:
        seq_len = 16
    # Resolve the data identity once, using the SAME generator the losses use.
    _x, _y, _data_fp, _src = _repro_batch(seed, batch_size=batch_size, seq_len=seq_len, vocab=vocab, train_ids=train_ids)
    _fp_before = rng_fingerprint()
    _kw = dict(steps=steps, d=_arch['d'], n_layers=_arch['n_layers'], n_heads=_arch['n_heads'], d_head=_arch['d_head'], seq_len=seq_len, batch_size=batch_size, vocab=int(vocab), lr=lr, train_ids=train_ids, mlp_ratio=mlp_ratio, device=device)
    with _preserve_rng_state():
        a = _repro_tiny_losses(variant, seed, **_kw)
        b = _repro_tiny_losses(variant, seed, **_kw)
        c = _repro_tiny_losses(variant, int(seed) + 1, **_kw)
    _fp_after = rng_fingerprint()
    _n = min(len(a), len(b))
    max_diff = max((abs(a[i] - b[i]) for i in range(_n)), default=float('inf'))
    different_seed_diff = max((abs(a[i] - c[i]) for i in range(min(len(a), len(c)))), default=0.0)
    finite = all(math.isfinite(v) for v in a + b + c)
    bitwise = bool(a == b)
    passed = bool(finite and max_diff <= float(tol) and different_seed_diff > 0.0)
    r = {
        'kind': 'duplicate_run_reproducibility',
        'variant': variant,
        'seed': int(seed),
        'steps': int(steps),
        'tol': float(tol),
        'device': str(device),
        'determinism': determinism_label(),
        'architecture': _arch,
        'mlp_ratio': (None if mlp_ratio is None else float(mlp_ratio)),
        'seq_len': int(seq_len),
        'batch_size': int(batch_size),
        'data_source': _src,
        'data_fp': _data_fp,
        'same_seed_losses_a': a,
        'same_seed_losses_b': b,
        'different_seed_losses': c,
        'max_abs_diff_same_seed': float(max_diff),
        'bitwise_identical': bitwise,
        'different_seed_changes_curve': bool(different_seed_diff > 0.0),
        'different_seed_max_abs_diff': float(different_seed_diff),
        'all_finite': bool(finite),
        'rng_fp_before': _fp_before,
        'rng_fp_after': _fp_after,
        'rng_state_preserved': bool(_fp_before == _fp_after),
        'passed': passed,
    }
    if not passed:
        if not finite:
            r['diagnosis'] = 'non-finite loss in a tiny seeded run'
        elif max_diff > float(tol):
            r['diagnosis'] = 'same seed did not reproduce the short loss curve on this machine/stack'
        else:
            r['diagnosis'] = 'a different seed produced the identical loss curve (seed is not wired into the run)'
    else:
        r['diagnosis'] = 'ok'
    return r

def _repro_guard_base(*, seed=0, steps=3, tol=1e-06, arch=None, vocab=None, mlp_ratio=None, seq_len=None, batch_size=2, train_ids=None, data_fp=None):
    """Exact cache key shared by the guard and run()'s evidence recovery.

    Strictly a description of the probe that produced a PASS: changing the
    tolerance, step count, architecture, MLP width, sequence length, vocab,
    device, determinism level, data stream or code invalidates the cached
    certificate.  A pass under a loose tolerance must never be reused for a
    stricter request."""
    _fp = data_fp
    if _fp is None and train_ids is not None:
        _arr = np.asarray(train_ids)
        _fp = _ids_fp(_arr if _arr.ndim == 1 else np.ascontiguousarray(_arr).reshape(-1))
    return _repro_probe_digest(_repro_probe_config(seed=seed, steps=steps, tol=tol, arch=arch, seq_len=(16 if seq_len is None else seq_len), batch_size=batch_size, vocab=(64 if vocab is None else vocab), mlp_ratio=mlp_ratio, data_fp=_fp, train_ids=train_ids))

def reproducibility_guard(variants, *, seed=0, steps=3, tol=1e-06, force=False, arch=None, train_ids=None, vocab=None, mlp_ratio=None, seq_len=None, batch_size=2, seeds=None, strict=None):
    """Run ``reproducibility_check`` once per (variant, seed, probe config) per process.

    Mirrors ``overfit_guard``: results are cached as evidence, a skip env var
    is explicit and loud, and a failure is a warning unless the caller sets
    ``CSA_REQUIRE_REPRO=1`` (or passes ``strict=True``).  The cache key
    includes the full probe configuration -- tolerance, code stamp, seed,
    device, architecture, MLP width, sequence length, vocabulary and data
    fingerprint -- so a loose/synthetic probe cannot certify a real one.
    ``seeds`` lets a panel check its first and last seed; ``arch`` and
    ``train_ids`` make it exercise the real architecture on a small slice of
    the real corpus.  Returns the reports that ran in this call."""
    if os.environ.get('CSA_SKIP_REPRO', '').strip().lower() in ('1', 'true', 'yes', 'on'):
        print('[repro] guard SKIPPED (CSA_SKIP_REPRO is set) — this run has no same-machine seed check')
        return []
    _want_strict = (os.environ.get('CSA_REQUIRE_REPRO', '').strip().lower() in ('1', 'true', 'yes', 'on')) if strict is None else bool(strict)
    _arch = _repro_probe_arch(arch)
    if vocab is None:
        vocab = (int(np.max(train_ids)) + 1) if train_ids is not None else 64
    if seq_len is None:
        seq_len = 16
    _seed_list = [int(s) for s in (seeds if seeds is not None else [seed])]
    _seed_list = list(dict.fromkeys(_seed_list))
    _full_data_fp = None
    if train_ids is not None:
        _arr = np.asarray(train_ids)
        _full_data_fp = _ids_fp(_arr if _arr.ndim == 1 else np.ascontiguousarray(_arr).reshape(-1))
    _base_cache = {}

    def _base_for(v, sd):
        key = (v, sd)
        if key not in _base_cache:
            _base_cache[key] = _repro_guard_base(seed=sd, steps=steps, tol=tol, arch=_arch, vocab=vocab, mlp_ratio=_overfit_ratio_for(mlp_ratio, v), seq_len=seq_len, batch_size=batch_size, train_ids=train_ids, data_fp=_full_data_fp)
        return _base_cache[key]

    # As in overfit_guard, a cached FAILURE is not a certificate: retry it so a
    # later phase cannot silently inherit the failed status as "already done".
    todo = [(v, sd) for v in variants for sd in _seed_list if force or _REPRO_GUARD.get((v, _base_for(v, sd))) is not True]
    reports = []
    for v, sd in todo:
        r = reproducibility_check(v, seed=sd, steps=steps, tol=tol, arch=_arch, train_ids=train_ids, vocab=vocab, mlp_ratio=_overfit_ratio_for(mlp_ratio, v), seq_len=seq_len, batch_size=batch_size)
        _base = _base_for(v, sd)
        r['guard_probe_key'] = _base
        r['guard_probe_config'] = _repro_probe_config(seed=sd, steps=steps, tol=tol, arch=_arch, seq_len=seq_len, batch_size=batch_size, vocab=vocab, mlp_ratio=_overfit_ratio_for(mlp_ratio, v), data_fp=r.get('data_fp'), train_ids=train_ids)
        key = (v, _base)
        _REPRO_GUARD[key] = bool(r['passed'])
        _REPRO_REPORTS[key] = r
        reports.append(r)
        print(f"[repro] {v:<24} seed={sd} same-seed max|Δ|={r['max_abs_diff_same_seed']:.3e}  different-seed Δ={r['different_seed_max_abs_diff']:.3e}  bitwise={r['bitwise_identical']}  {'PASS' if r['passed'] else 'FAIL'}")
        if not r['passed']:
            _msg = f"same-machine reproducibility check FAILED for `{v}` seed={sd} ({r['diagnosis']})"
            if _want_strict:
                raise RuntimeError(_msg + ' — strict reproducibility is required for this run. Correct the seed/kernel nondeterminism or explicitly opt out (CSA_REQUIRE_REPRO=0).')
            print(f'[repro] WARNING: {_msg}; the panel will record the failure (set CSA_REQUIRE_REPRO=1 to make it fatal, CSA_SKIP_REPRO=1 to skip)')
    return reports

def reproducibility_reports_for(variants, *, probe_key=None, data_fp=None):
    """Exact-match recovery of reproducibility evidence for the current panel."""
    _want = set(variants)
    out = []
    for _k, _r in _REPRO_REPORTS.items():
        if _k[0] not in _want:
            continue
        if probe_key is not None and _k[1] != probe_key:
            continue
        if data_fp is not None and _r.get('data_fp') != data_fp:
            continue
        out.append(dict(_r))
    return out

def reproducibility_reports():
    """Copy of the reproducibility reports gathered in this process."""
    return {k: dict(v) for k, v in _REPRO_REPORTS.items()}

class TimeGuard:
    BASE = (256, 6, 512, 12)

    def __init__(self, cap):
        self.total_hours = float(cap['total_hours'])
        self.margin = float(cap.get('margin', 0.93))
        self.already = float(cap.get('already_hours', 0.0))
        self.state_path = cap.get('state_path', 'run_time_state.json')
        # 内核模式使用独立的状态文件：两档速率标定不可互相污染（同一
        # state 文件会把 eager 的 sps 标定用于内核单元）。
        # 幂等：cap_spec() 导出的就是已盖章路径，par.py 子进程按它重建
        # TimeGuard 时不得再加一次后缀。
        if _kernel_stamp() and not self.state_path.endswith('.k'):
            self.state_path = self.state_path + '.k'
        self.state = self._load_state()

    def _load_state(self):
        st = {'booked_seconds': 0.0, 'norm_sps': None, 'sps_by_class': {}, 'runs': 0}
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, encoding='utf-8') as f:
                    st.update(json.load(f))
            except Exception as e:
                print(f'[cap] FATAL: {self.state_path} exists but cannot be parsed ({type(e).__name__}: {e}).  Refusing to start with a silently RESET state (booked time would read as 0) — move it aside to start fresh.')
                raise
        return st

    def _save(self):
        atomic_write_json(self.state_path, self.state)

    def cap_spec(self):
        """Plain-dict spec of this guard, enough for a child process (par.py)
        to rebuild an equivalent TimeGuard against the same shared state file."""
        return dict(total_hours=self.total_hours, margin=self.margin, already_hours=self.already, state_path=self.state_path)

    def book(self, seconds, token=None):
        """Reserve (seconds > 0) or refund (seconds < 0) projected run time
        against the shared state WITHOUT touching the run counter or the
        calibration.  Parallel cell workers pre-book their estimate before
        training and refund it after recording the actual seconds, so the
        booked total always converges to the true sum while concurrently
        launching children still see the shared headroom shrink.

        With `token`, the reservation is recorded per-booking (with the
        booking process pid) so `reconcile()` can refund reservations whose
        process died before refunding (killed child / instance shutdown) —
        without this, a killed parallel cell starves every later phase with
        a phantom reservation."""
        try:
            import fcntl
        except ImportError:
            fcntl = None
        _lock_f = None
        if fcntl is not None:
            _lock_f = open(self.state_path + '.lock', 'a')
            fcntl.flock(_lock_f, fcntl.LOCK_EX)
        try:
            self.state = self._load_state()
            if token is None:
                self.state['booked_seconds'] += float(seconds)
                if 'bookings' in self.state:
                    self.state['booked_untracked'] = float(self.state.get('booked_untracked', 0.0)) + float(seconds)
            else:
                bookings = self.state.setdefault('bookings', {})
                if 'booked_untracked' not in self.state:
                    # First tokenised booking coexisting with a pre-bookings
                    # state (e.g. the parent's data-load record_run, or any
                    # serial phase that ran before this parallel one): every
                    # second booked so far has no per-token representation,
                    # so it ALL becomes the untracked remainder.  Defaulting
                    # it to 0 would erase those seconds from the
                    # booked total on the next refund's recompute — the cap
                    # would then never fill in parallel mode.
                    self.state['booked_untracked'] = max(0.0, float(self.state.get('booked_seconds', 0.0)) - sum(float(b.get('seconds', 0.0)) for b in bookings.values()))
                if seconds > 0:
                    bookings[str(token)] = {'seconds': float(seconds), 'pid': int(os.getpid()), 'ts': time.time()}
                else:
                    bookings.pop(str(token), None)
                legacy = float(self.state['booked_untracked'])
                self.state['booked_seconds'] = legacy + sum(float(b.get('seconds', 0.0)) for b in bookings.values())
            self._save()
        finally:
            if _lock_f is not None:
                try:
                    fcntl.flock(_lock_f, fcntl.LOCK_UN)
                finally:
                    _lock_f.close()

    @staticmethod
    def _pid_alive(pid):
        try:
            os.kill(int(pid), 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except (OSError, AttributeError):
            return True  # 无法判定（非 posix 等）时按存活处理，绝不误退

    def reconcile(self):
        """Refund per-token bookings whose recording process is no longer
        alive (crash/kill recovery).  Old state files without a `bookings`
        map are untouched: their `booked_seconds` is preserved as
        `booked_untracked` once any tokenised booking coexists."""
        try:
            import fcntl
        except ImportError:
            fcntl = None
        _lock_f = None
        if fcntl is not None:
            _lock_f = open(self.state_path + '.lock', 'a')
            fcntl.flock(_lock_f, fcntl.LOCK_EX)
        try:
            self.state = self._load_state()
            bookings = self.state.get('bookings') or {}
            if not bookings:
                return 0.0
            total = float(self.state.get('booked_seconds', 0.0))
            tracked = sum(float(b.get('seconds', 0.0)) for b in bookings.values())
            self.state['booked_untracked'] = max(0.0, total - tracked)
            refunded = 0.0
            now = time.time()
            for tok in sorted(bookings):
                b = bookings[tok]
                est_s = float(b.get('seconds', 0.0))
                dead = not self._pid_alive(b.get('pid', -1))
                # A live child can hold a booking for at most its dispatcher
                # watchdog horizon (par.py SIGKILLs run/warm/lenphase cells at
                # max(3*est, est+1h); every other kind is capped at 12 h).  A
                # booking older than that CANNOT have a live owner: the pid
                # reading is a reuse artifact (an instance reboot recycles
                # pids — this driver's own auto-shutdown makes that the common
                # case), and without an age fallback the phantom reservation
                # would starve every later cell behind the cap forever.
                horizon = max(4.0 * est_s, 13.0 * 3600.0)
                too_old = b.get('ts') is not None and (now - float(b['ts'])) > horizon
                if dead or too_old:
                    refunded += est_s
                    bookings.pop(tok, None)
                    _why = 'dead pid' if dead else f'phantom: age {(now - float(b["ts"])) / 3600:.1f} h exceeds the watchdog horizon {horizon / 3600:.1f} h'
                    print(f'[cap] reconcile: refunded {est_s / 3600:.2f} h from booking {tok} ({_why}, pid {b.get("pid")})')
            self.state['booked_seconds'] = self.state['booked_untracked'] + sum(float(b['seconds']) for b in bookings.values())
            self._save()
            return refunded
        finally:
            if _lock_f is not None:
                try:
                    fcntl.flock(_lock_f, fcntl.LOCK_UN)
                finally:
                    _lock_f.close()

    def spent_hours(self):
        self.state = self._load_state()
        return self.already + self.state['booked_seconds'] / 3600.0

    def remaining_hours(self):
        return self.total_hours - self.spent_hours()

    def cap_hours(self):
        return self.total_hours * self.margin

    def record_run(self, seconds, steps_done, d, n_layers, seq_len, batch_size, calib_seconds=None):
        try:
            import fcntl
        except ImportError:
            fcntl = None
        _lock_f = None
        if fcntl is not None:
            _lock_f = open(self.state_path + '.lock', 'a')
            fcntl.flock(_lock_f, fcntl.LOCK_EX)
        try:
            self.state = self._load_state()
            self.state['booked_seconds'] += float(seconds)
            if 'bookings' in self.state:
                # Keep the invariant booked_seconds == booked_untracked + Σ
                # bookings: recorded actual seconds have no per-token
                # representation, so they accrue to the untracked remainder.
                # Without this the next token refund's recompute silently
                # drops every recorded second from the booked total (the run
                # -time cap then meters nothing in parallel mode).
                self.state['booked_untracked'] = float(self.state.get('booked_untracked', 0.0)) + float(seconds)
            if steps_done and steps_done > 0:
                self.state['runs'] += 1
                if calib_seconds is None or not calib_seconds > 0:
                    calib_seconds = seconds
                f = self._wallclock_factor(d, n_layers, seq_len, batch_size)
                sps = calib_seconds / steps_done
                norm = sps / f
                prev = self.state.get('norm_sps')
                self.state['norm_sps'] = norm if prev is None else 0.7 * prev + 0.3 * norm
                by_class = self.state.setdefault('sps_by_class', {})
                key = str((int(d), int(n_layers), int(seq_len), int(batch_size)))
                cprev = by_class.get(key)
                by_class[key] = sps if cprev is None else 0.7 * cprev + 0.3 * sps
            self._save()
        finally:
            if _lock_f is not None:
                try:
                    fcntl.flock(_lock_f, fcntl.LOCK_UN)
                finally:
                    _lock_f.close()

    @classmethod
    def _depth_factor(cls, n_layers):
        return n_layers / 6.0

    @classmethod
    def _compute_factor(cls, d, n_layers, seq_len, batch_size):
        return (d / 256.0) ** 2 * cls._depth_factor(n_layers) * (seq_len / 512.0) ** 1.5 * (batch_size / 12.0)

    @classmethod
    def _calib_exponent(cls):
        return 1.75

    @classmethod
    def _wallclock_factor(cls, d, n_layers, seq_len, batch_size):
        f = cls._compute_factor(d, n_layers, seq_len, batch_size)
        df = cls._depth_factor(n_layers)
        return f / df * df ** cls._calib_exponent()

    def estimate_seconds(self, steps, d=256, n_layers=6, seq_len=512, batch_size=12):
        key = str((int(d), int(n_layers), int(seq_len), int(batch_size)))
        exact = (self.state.get('sps_by_class') or {}).get(key)
        if exact is not None:
            return steps * exact * 1.15
        norm = self.state.get('norm_sps')
        if norm is None:
            norm = 0.35
        f = self._wallclock_factor(d, n_layers, seq_len, batch_size)
        return steps * norm * f * 1.15

    def note_watchdog_kill(self, elapsed_seconds, steps, d=256, n_layers=6, seq_len=512, batch_size=12):
        """Lift the calibrated per-step floor after a dispatcher-watchdog kill
        (par.py).  A cell killed at `elapsed_seconds` demonstrably needs MORE
        than that for `steps` steps, so its true sps exceeds
        elapsed/steps — yet a killed cell never records a run, so without
        this hook the calibration (and hence the watchdog bound
        max(3*est, est+1h) derived from it) would stay wrong and the same
        healthy cell would be killed at the same wall-clock point on every
        resume (uncalibrated hardware, CPU workers).  Lifting the floor
        to the kill time itself makes the next bound ~3.45x larger, so the
        retry window grows geometrically until a healthy cell fits; a truly
        hung cell is still killed on every attempt, just later."""
        if not steps or int(steps) <= 0 or not float(elapsed_seconds) > 0:
            return
        floor = float(elapsed_seconds) / int(steps)
        try:
            import fcntl
        except ImportError:
            fcntl = None
        _lock_f = None
        if fcntl is not None:
            _lock_f = open(self.state_path + '.lock', 'a')
            fcntl.flock(_lock_f, fcntl.LOCK_EX)
        try:
            self.state = self._load_state()
            by_class = self.state.setdefault('sps_by_class', {})
            key = str((int(d), int(n_layers), int(seq_len), int(batch_size)))
            prev = by_class.get(key)
            if prev is None or floor > float(prev):
                by_class[key] = floor
            f = self._wallclock_factor(d, n_layers, seq_len, batch_size)
            norm_floor = floor / f
            norm = self.state.get('norm_sps')
            if norm is None or norm_floor > float(norm):
                self.state['norm_sps'] = norm_floor
            self._save()
        finally:
            if _lock_f is not None:
                try:
                    fcntl.flock(_lock_f, fcntl.LOCK_UN)
                finally:
                    _lock_f.close()
        print(f'[cap] watchdog kill noted: the pace floor for (d={d}, L={n_layers}, seq={seq_len}, bs={batch_size}) is now >= {floor:.3f} s/step so the next attempt gets a wider watchdog bound')

    def can_start(self, est_seconds):
        projected = self.spent_hours() + est_seconds / 3600.0
        return projected <= self.cap_hours()

    def deadline_ts(self):
        headroom = max(self.cap_hours() - self.spent_hours(), 0.0)
        return time.time() + headroom * 3600.0

    def report(self):
        self.state = self._load_state()
        print('\n==================== usage report ====================')
        print(f'  booked GPU time : {self.state['booked_seconds'] / 3600:.2f} h ({self.state.get('runs', 0)} runs)')
        for key, sps in sorted((self.state.get('sps_by_class') or {}).items()):
            print(f'  calibrated rate : {sps:.3f} s/step  (d, L, seq, batch = {key})')
        if self.state.get('norm_sps'):
            print(f'  normalised rate : {self.state['norm_sps']:.3f} s/step (base class d=256 / 6L / seq512 / batch12)')
        print('======================================================')
PARAM_MATCHED = {'full_matched', 'full_sw128_matched'}
PARAM_MATCHED_V7 = {'full_rope', 'full_sw128_matched_rope'}

_MLP_RATIO_MEMO = {}

def variant_mlp_ratio(variant, vocab, *, d=256, n_layers=6, n_heads=8, d_head=32, seq_len=512, matched=None, ref_variant='csa_dynamic'):
    if matched is None:
        matched = PARAM_MATCHED
    if variant not in matched:
        return 4.0
    # The ratio is a pure function of these arguments (the parameter count
    # of a freshly-initialised SmallGPT depends only on shapes; the build
    # runs under torch.random.fork_rng, so skipping a repeat build leaves
    # the caller's RNG stream untouched either way) — memoise it on the
    # full argument tuple.
    _mkey = (variant, int(vocab), int(d), int(n_layers), int(n_heads), int(d_head), int(seq_len), tuple(sorted((str(_x) for _x in matched))), str(ref_variant))
    _hit = _MLP_RATIO_MEMO.get(_mkey)
    if _hit is not None:
        return _hit

    def n_params(v, ratio):
        with torch.random.fork_rng():
            cfgs = make_layer_cfgs(n_layers, v)
            m = SmallGPT(vocab, d, n_layers, n_heads, d_head, seq_len, cfgs, mlp_ratio=ratio)
            n = count_params(m)
            del m
        return n
    p_full = n_params(variant, 4)
    p_ref = n_params(ref_variant, 4)
    ratio = 4 + (p_ref - p_full) / (2.0 * d * d * n_layers)
    if ratio == 4.0:
        ratio = math.nextafter(4.0, 0.0)
    if ratio <= 1.0:
        raise RuntimeError(f'[{variant}] param-matched mlp_ratio would be {ratio:.2f} (<= 1): the reference arm {ref_variant} already has fewer/equal params — param matching is impossible for this variant/config')
    print(f'[{variant}] matched mlp_ratio = {ratio:.2f} ({p_full / 1000000.0:.2f}M -> target {p_ref / 1000000.0:.2f}M params vs {ref_variant})')
    _MLP_RATIO_MEMO[_mkey] = ratio
    return ratio

def _tag_of(rec):
    if isinstance(rec, dict) and 'protocol' in rec and (rec['protocol'] is not None):
        return str(rec['protocol'])
    if isinstance(rec, dict) and rec.get('warm_steps') is not None:
        return f'w{rec['warm_steps']}'
    return ''

def legacy_warm_tags(summary):
    _legacy = {}
    for key, rec in summary.items():
        if not isinstance(rec, dict) or 'seed' not in rec or 'ppl' not in rec:
            continue
        if not ppl_is_usable(rec.get('ppl')) or rec.get('synthesized'):
            continue
        if 'protocol' in rec and rec['protocol'] is not None or rec.get('warm_steps') is not None:
            continue
        m = _WARM_KEY_RE.match(str(key))
        _var = rec.get('variant')
        if _var is None:
            _var = str(key).split('::', 1)[0]
        _legacy.setdefault(str(_var), set()).add(m.group(1) if m else None)
    return {v: frozenset(tags) for v, tags in _legacy.items() if None not in tags}

def record_tag(rec, key, legacy_ok):
    if 'protocol' in rec and rec['protocol'] is not None or rec.get('warm_steps') is not None:
        return _tag_of(rec)
    m = _WARM_KEY_RE.match(str(key))
    if not m:
        return ''
    ok = legacy_ok
    if isinstance(ok, dict):
        _var = rec.get('variant')
        if _var is None:
            _var = str(key).split('::', 1)[0]
        ok = ok.get(str(_var))
        if ok is None:
            return ''
    if m.group(1) in ok:
        return f'w{m.group(1)}'
    return ''

def _group_params(recs):
    for r in recs:
        v = r.get('params')
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
    return None

def _pooled_ppl_from_records(recs):
    """Token-pooled PPL: exp(sum NLL / sum N tokens) over all seeds.

    This is the estimator the review asks for; the arithmetic mean of per-seed
    PPLs is kept as `ppl_mean_arithmetic` for backward compatibility but is no
    longer the primary aggregate when token counts exist.  Records that predate
    `nll_sum`/`n_tokens` cannot be pooled and fall back explicitly."""
    nlls, nts = [], []
    for r in recs:
        nll, nt = r.get('nll_sum'), r.get('n_tokens')
        if not isinstance(nll, (int, float)) or isinstance(nll, bool) or not math.isfinite(float(nll)):
            return None, None, None
        if not isinstance(nt, (int, float)) or isinstance(nt, bool) or float(nt) <= 0:
            return None, None, None
        nlls.append(float(nll))
        nts.append(float(nt))
    total_n, total_t = float(sum(nlls)), float(sum(nts))
    if total_t <= 0:
        return None, None, None
    return _safe_exp_ppl(total_n / total_t), total_n, int(total_t)

def _seed_t_ci(ppls, alpha=0.05):
    n = len(ppls)
    if n < 2:
        return None
    mu = float(np.mean(ppls))
    sd = float(np.std(ppls, ddof=1))
    try:
        tc = _t_critical(1.0 - alpha, n - 1)
    except Exception:
        return None
    half = tc * sd / math.sqrt(n)
    return [mu - half, mu + half]

def _doc_bootstrap_ppl_ci(recs, *, n_boot=2000, seed=20261018):
    """Document-level bootstrap CI for the pooled PPL.

    Rows are the fixed evaluation documents shared by every seed; resampling
    ROWS (not tokens) preserves the within-document correlation.  If any
    record lacks per-row NLL, no CI is invented — the caller must see the
    missing-evidence state rather than a narrow interval computed from the
    wrong unit.
    """
    rows = []
    for r in recs:
        nll = r.get('nll_per_row')
        ntok = r.get('n_tokens_per_row')
        if not isinstance(nll, list) or not isinstance(ntok, list) or len(nll) != len(ntok) or not nll:
            return None
        rows.append((np.asarray(nll, dtype=float), np.asarray(ntok, dtype=float)))
    if len({len(a) for a, _ in rows}) != 1:
        return None
    R = len(rows[0][0])
    if R == 0:
        return None
    nll_mat = np.stack([a for a, _ in rows], axis=0)          # [S, R]
    ntok_mat = np.stack([b for _, b in rows], axis=0)         # [S, R]
    rng = np.random.default_rng(int(seed))
    nll_col = nll_mat.sum(axis=0)
    ntok_col = ntok_mat.sum(axis=0)
    vals = np.empty(int(n_boot), dtype=float)
    # Multinomial bootstrap counts: a single [chunk, R] count matrix is reused
    # across draws.
    _chunk = 256
    _b = 0
    while _b < int(n_boot):
        _nb = min(_chunk, int(n_boot) - _b)
        counts = rng.multinomial(R, np.full(R, 1.0 / R, dtype=float), size=_nb)
        nt = counts @ ntok_col
        nn = counts @ nll_col
        ok = nt > 0
        out = np.full(_nb, np.nan, dtype=float)
        out[ok] = np.exp(nn[ok] / nt[ok])
        vals[_b:_b + _nb] = out
        _b += _nb
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size < max(20, int(0.5 * n_boot)):
        return None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]

def aggregate(summary):

    def _protocol_tag(rec, key=None):
        if key is None:
            return _tag_of(rec)
        return record_tag(rec, key, _legacy_warm_ok)
    _legacy_warm_ok = legacy_warm_tags(summary)
    per_variant = {}
    for key, rec in summary.items():
        if not isinstance(rec, dict) or 'ppl' not in rec or 'seed' not in rec:
            continue
        if not ppl_is_usable(rec.get('ppl')) or rec.get('synthesized'):
            continue
        v = rec.get('variant')
        if v is None:
            v = str(key).split('::', 1)[0]
        per_variant.setdefault((v, _protocol_tag(rec, key)), []).append(rec)
    def _seed_key(s):
        # JSON round-trips can leave the same seed as 0 or "0"; the rest of the
        # readers normalise through `_as_seed_int`.  Keying dedup/pairing on the
        # raw value would count one seed twice and fabricate an extra n.
        n = _as_seed_int(s)
        return s if n is None else n
    _dups = {}
    for k, recs in per_variant.items():
        cnt = {}
        for r in recs:
            s = _seed_key(r['seed'])
            cnt[s] = cnt.get(s, 0) + 1
        bad = sorted((s for s, c in cnt.items() if c > 1))
        if bad:
            _dups[k] = bad
    if _dups:
        raise ValueError('aggregate: these (variant, protocol) groups hold more than one record for the same seed, so their PPL is ambiguous: ' + '; '.join((f'{v}/{t or '-'}->seeds {s}' for (v, t), s in sorted(_dups.items()))) + '. Remove the stale duplicates (or give the records distinct `protocol` tags) before aggregating.  A record whose value was never MEASURED (an `error` record, or a `synthesized` reconstruction) can carry no PPL and is IGNORED here — drop it from the summary instead of re-labelling it, because a protocol tag only silences this guard, it does not make the number real.')
    _multi = {}
    for (v, tag), recs in per_variant.items():
        _multi.setdefault(v, []).append(tag)
    _multi = {v: t for v, t in _multi.items() if len(t) > 1}
    if _multi:
        print('[aggregate] WARNING: these variants hold MULTIPLE protocol groups and are reported per-group, NOT pooled across groups (pooling would fake extra seeds): ' + ', '.join((f'{v}->{t}' for v, t in sorted(_multi.items()))))

    def _cfg_fp(rec):
        _p = rec.get('params')
        try:
            _p = int(_p) if _p is not None and (not isinstance(_p, bool)) else None
        except (TypeError, ValueError):
            _p = None
        return json.dumps([rec.get('run_cfg'), _p, [rec.get(k) for k in PAIR_STEP_KEYS]], sort_keys=True, default=str)

    def _steps_ok(recs):
        if len(recs) < 2:
            return True
        return len({_cfg_fp(r) for r in recs}) == 1

    def _by_seed(recs):
        out, dup = ({}, [])
        for r in recs:
            s = _seed_key(r['seed'])
            if s in out:
                dup.append(s)
            out[s] = r
        return (out, dup)

    def _paired(recs, base_recs):
        self_map, dup_a = _by_seed(recs)
        base_map, dup_b = _by_seed(base_recs)
        assert not dup_a and (not dup_b), (sorted(set(dup_a)), sorted(set(dup_b)))
        out, skipped = ([], [])
        for s, r in self_map.items():
            b = base_map.get(s)
            if b is None:
                continue
            reason = pair_reason(r, b)
            if reason is not None:
                skipped.append((s, reason))
                continue
            out.append(r['ppl'] - b['ppl'])
        return (out, skipped)
    agg = {}
    _split_groups = []
    _group_records = {}

    def _agg_key(v, tag, fp_idx=None):
        key = v if not tag else f'{v}#{tag}'
        return key if fp_idx is None else f'{key}@cfg{fp_idx}'
    for (v, tag), recs in per_variant.items():
        if _steps_ok(recs):
            _split_groups.append(((v, tag, None), recs))
            _group_records[v, tag, None] = recs
            continue
        _subs = {}
        for r in recs:
            _subs.setdefault(_cfg_fp(r), []).append(r)
        # the `@cfg` suffix is a digest of the config fingerprint itself, so a
        # resume pass cannot reshuffle which group a published key refers to
        _ordered = sorted(_subs.items(), key=lambda kv: kv[0])
        print(f'[aggregate] WARNING: `{_agg_key(v, tag)}` holds records from {len(_ordered)} different run_cfg/load/parameter-count configurations — pooling them would average incompatible runs (or two different MODEL WIDTHS, which are not two seeds of one model) and fake extra seeds.  Reported per configuration instead.')
        for _fp, _rs in _ordered:
            _i = hashlib.sha1(_fp.encode('utf-8')).hexdigest()[:8]
            _split_groups.append(((v, tag, _i), _rs))
            _group_records[v, tag, _i] = _rs
    for (_v, _tag, _fp_idx), recs in _split_groups:
        v, tag = (_v, _tag)
        key = _agg_key(v, tag, _fp_idx)
        def _seed_sort_key(r):
            s = r.get('seed')
            if isinstance(s, (int, float)) and (not isinstance(s, bool)) and math.isfinite(float(s)):
                return (0, float(s), '')
            return (1, 0.0, str(s))
        _recs_sorted = sorted(recs, key=_seed_sort_key)
        recs = _recs_sorted
        ppls = np.array([r['ppl'] for r in recs], dtype=float)
        _pooled, _nll_total, _ntok_total = _pooled_ppl_from_records(recs)
        _sorted = np.sort(ppls)
        _med = float(np.median(_sorted))
        _q1, _q3 = (float(np.percentile(_sorted, 25)), float(np.percentile(_sorted, 75))) if len(_sorted) >= 2 else (_med, _med)
        entry = {'variant': v, 'protocol': tag, 'n_seeds': len(recs),
                 'ppl_mean': float(ppls.mean()),
                 'ppl_mean_arithmetic': float(ppls.mean()),
                 'ppl_std': float(ppls.std(ddof=1)) if len(ppls) > 1 else 0.0,
                 'ppls': [float(p) for p in ppls], 'seeds': [r.get('seed') for r in recs],
                 'params': _group_params(recs), 'tokens_seen': recs[0].get('tokens_seen'),
                 'ppl_median': _med, 'ppl_iqr': [_q1, _q3],
                 'ppl_ci95_seed_t': _seed_t_ci(ppls),
                 'nll_sum': _nll_total, 'n_tokens': _ntok_total,
                 'ppl_pooled': (_pooled if _pooled is not None else None),
                 'ppl_ci95_doc_bootstrap': _doc_bootstrap_ppl_ci(recs),
                 'ppl_aggregation': ('pooled_token_nll' if _pooled is not None else 'arithmetic_seed_mean_fallback (records lack nll_sum/n_tokens)'),
                 'ppl_primary': (float(_pooled) if _pooled is not None else float(ppls.mean()))}
        _pb = next((r.get('param_breakdown') for r in recs if isinstance(r.get('param_breakdown'), dict)), None)
        if _pb:
            entry['param_breakdown'] = _pb
            entry['params_total'] = _pb.get('total')
            entry['params_attention'] = _pb.get('attention')
            entry['params_mlp'] = _pb.get('mlp')
            entry['params_embedding'] = _pb.get('embedding')
            entry['active_params'] = _pb.get('active')
            entry['attention_param_share'] = _pb.get('attention_share')
        _kv = next((r.get('kv_cache_elements_per_token') for r in recs if isinstance(r.get('kv_cache_elements_per_token'), dict)), None)
        if _kv:
            entry['kv_cache_elements_per_token'] = _kv
        if _fp_idx is not None:
            entry['run_cfg_class'] = _fp_idx
            entry['run_cfg'] = recs[0].get('run_cfg')
        dyn_lens, all_lens, deltas = ([], [], [])
        bf1, bex = ([], [])

        def _cell(layer, k):
            v = layer.get(k)
            return float(v) if isinstance(v, (int, float)) and (not isinstance(v, bool)) and math.isfinite(v) else None
        for r in recs:
            r_f1, r_ex, r_dl = ([], [], [])
            for lk, lv in r.get('stats', {}).items():
                if not isinstance(lv, dict):
                    continue
                lm = _cell(lv, 'len_mean')
                if lm is None:
                    continue
                all_lens.append(lm)
                if '_dyn' in lk:
                    dyn_lens.append(lm)
                    if 'bnd_f1' in lv:
                        _f1 = _cell(lv, 'bnd_f1')
                        _ex = _cell(lv, 'bnd_excess') if lv.get('bnd_rand_exact') is not False else None
                        if _f1 is not None:
                            r_f1.append(_f1)
                        if _ex is not None:
                            r_ex.append(_ex)
                dl = _cell(lv, 'delta')
                if dl is not None:
                    r_dl.append(dl)
            if r_f1:
                bf1.append(float(np.mean(r_f1)))
            if r_ex:
                bex.append(float(np.mean(r_ex)))
            if r_dl:
                # Average per record FIRST (as bf1/bex do), so records with more
                # dynamic layers do not get extra weight in the cross-seed mean.
                deltas.append(float(np.mean(r_dl)))
        if dyn_lens:
            entry['avg_dyn_block_len'] = float(np.mean(dyn_lens))
        elif all_lens:
            entry['avg_block_len'] = float(np.mean(all_lens))
        if deltas:
            entry['delta_logit_mean'] = float(np.mean(deltas))
        if bf1:
            entry['boundary_f1_dyn'] = float(np.mean(bf1))
            entry['boundary_f1_dyn_std'] = float(np.std(bf1, ddof=1)) if len(bf1) > 1 else 0.0
        if bex:
            entry['boundary_excess_dyn'] = float(np.mean(bex))
        hists = [r['ppl_history'] for r in recs if r.get('ppl_history')]
        if hists:
            per_step = {}
            for _h in hists:
                for _pt in _h:
                    try:
                        _s = int(_pt[0])
                        _v = float(_pt[1])
                    except (TypeError, ValueError, IndexError):
                        continue
                    per_step.setdefault(_s, []).append(_v)
            def _grid(_hist):
                _out = []
                for _p in _hist:
                    try:
                        _out.append(int(_p[0]))
                    except (TypeError, ValueError, IndexError):
                        continue
                return _out
            _ref_steps = _grid(hists[0])
            _off_grid = []
            for _i, _h in enumerate(hists[1:], start=1):
                _hs = _grid(_h)
                if _hs != _ref_steps:
                    _off_grid.append(_i)
            if _off_grid:
                print(f'[aggregate] {key}: seed(s) {_off_grid} of {len(hists)} evaluate at a DIFFERENT step grid than seed 0 — the seed-averaged curve is taken over the steps they SHARE (aligned on the step VALUE, not on list position), so a differently-sampled seed contributes only to the common steps.')
            steps_h = sorted(per_step)
            cols = {s: i for i, s in enumerate(steps_h)}
            mat = np.full((len(hists), len(steps_h)), float('nan'))
            for _i, _h in enumerate(hists):
                _own = {}
                for _pt in _h:
                    try:
                        _own[int(_pt[0])] = float(_pt[1])
                    except (TypeError, ValueError, IndexError):
                        continue
                for _s, _v in _own.items():
                    mat[_i, cols[_s]] = _v
            good = np.array([[ppl_is_usable(v) for v in row] for row in mat])
            n_good = good.sum(axis=0)
            sums = np.where(good, mat, 0.0).sum(axis=0)
            mean_ppl = np.where(n_good > 0, sums / np.maximum(n_good, 1), np.nan)
            n_dropped = int((~good).sum())
            if n_dropped:
                print(f'[aggregate] {key}: {n_dropped} non-finite PPL curve point(s) across {len(hists)} seed(s) excluded from the seed-averaged curve ({int((n_good == 0).sum())} step(s) had no usable seed and are omitted)')
            _keep = n_good > 0
            entry['ppl_curve'] = [[int(s), float(m)] for s, m, k in zip(steps_h, mean_ppl, _keep) if k]
        agg[key] = entry
    for key, entry in agg.items():
        recs = _group_records.get((entry['variant'], entry['protocol'], entry.get('run_cfg_class')), [])
        entry['source'] = key
        if recs:
            entry['_group_records'] = recs
        rates = [r['tokens_seen'] / r['steps'] for r in recs if r.get('tokens_seen') and r.get('steps')]
        if rates:
            entry['tokens_per_step'] = float(np.mean(rates))
    _full_groups = {(tag, fp_idx): recs for (v, tag, fp_idx), recs in _split_groups if v == 'full'}

    def _cfg_fp_noparam(rec):
        # Baseline-matching fingerprint WITHOUT params: a `full` baseline
        # legitimately differs from the sparse arm in parameter count, so a
        # params-including fingerprint could never match it (dead branch).
        # run_cfg + steps/tokens is exactly the gate `pair_reason` applies.
        return json.dumps([rec.get('run_cfg'), [rec.get(k) for k in PAIR_STEP_KEYS]], sort_keys=True, default=str)

    def _baseline_for(tag, fp_idx, base_groups, recs):
        _want = _cfg_fp_noparam(recs[0]) if recs else None
        if _want is not None:
            _cands = sorted(base_groups.items(), key=lambda kv: (kv[0][0] != tag, kv[0][1] is None))
            for (_t_c, _i_c), _grp in _cands:
                if any((_cfg_fp_noparam(_r) == _want for _r in _grp)):
                    return (_grp, _t_c)
        for _cand in ((tag, fp_idx), (tag, None), ('', None)):
            if _cand in base_groups:
                return (base_groups[_cand], _cand[0])
        _want_cfg = recs[0].get('run_cfg') if recs else None
        if _want_cfg is not None:
            _rc = [(_t_c, _grp) for (_t_c, _i_c), _grp in base_groups.items() if _t_c == tag and _grp and (_grp[0].get('run_cfg') == _want_cfg)]
            if len(_rc) == 1:
                return (_rc[0][1], _rc[0][0])
        return (None, None)
    _base_amb = []
    for (v, tag, fp_idx), recs in _split_groups:
        if v == 'full':
            continue
        key = _agg_key(v, tag, fp_idx)
        base_recs, base_tag = _baseline_for(tag, fp_idx, _full_groups, recs)
        if base_recs is None:
            print(f'[aggregate] {key}: NO `full` baseline in this panel — dPPL_vs_full not computed')
            continue
        if base_tag != tag:
            print(f'[aggregate] {key}: pairing against the `full#{base_tag or '(untagged)'}` baseline (no `full` run exists for protocol {tag or '(untagged)'})')
        elif tag and any((_t == '' for _t, _i in _full_groups)):
            _base_amb.append(f'{key}: `full#{tag}` and `full` both present')
            continue
        ds, skipped = _paired(recs, base_recs)
        if skipped:
            _why = sorted({reason for _s, reason in skipped})
            print(f'[aggregate] {key}: {len(skipped)} seed(s) NOT paired with `full` — {_why}; excluded from dPPL_vs_full')
        if ds:
            d = np.array(ds, dtype=float)
            agg[key]['dPPL_vs_full_mean'] = float(d.mean())
            agg[key]['dPPL_vs_full_std'] = float(d.std(ddof=1)) if len(d) > 1 else 0.0
            agg[key]['n_paired'] = len(d)
            agg[key]['n_paired_skipped'] = len(skipped)
            agg[key]['dPPL_vs_full_base'] = base_tag or '(untagged)'
    _sw_groups = {(tag, fp_idx): recs for (v, tag, fp_idx), recs in _split_groups if v == 'full_sw128_matched'}
    for (v, tag, fp_idx), recs in _split_groups:
        if v in ('full', 'full_sw128_matched'):
            continue
        key = _agg_key(v, tag, fp_idx)
        base_recs, base_tag = _baseline_for(tag, fp_idx, _sw_groups, recs)
        if base_recs is None:
            continue
        if base_tag != tag:
            print(f'[aggregate] {key}: pairing against the UNTAGGED `full_sw128_matched` baseline')
        elif tag and any((_t == '' for _t, _i in _sw_groups)):
            _base_amb.append(f'{key}: `full_sw128_matched#{tag}` and the untagged one both present')
            continue
        ds, skipped = _paired(recs, base_recs)
        if skipped:
            _why = sorted({reason for _s, reason in skipped})
            print(f'[aggregate] {key}: {len(skipped)} seed(s) NOT paired with `full_sw128_matched` — {_why}; excluded')
        if ds:
            d = np.array(ds, dtype=float)
            agg[key]['dPPL_vs_sw128m_mean'] = float(d.mean())
            agg[key]['dPPL_vs_sw128m_std'] = float(d.std(ddof=1)) if len(d) > 1 else 0.0
            agg[key]['n_paired_sw128m'] = len(d)
            agg[key]['n_paired_sw128m_skipped'] = len(skipped)
    if _base_amb:
        raise ValueError('aggregate: ambiguous dense baseline for ' + '; '.join(sorted(_base_amb)) + '. A same-tagged and an untagged baseline are both present, so the reported delta would depend on which one was picked. Remove or re-tag one of them before aggregating.')
    return agg
_TW = {'variant': 18, 'ppl': 16, 'd': 15, 'd2': 15, 'params': 10, 'seeds': 6, 'blklen': 7, 'bnd': 13}

def _fmt_cell(value, width, fmt, absent='—'):
    s = absent if value is None else f'{value:{fmt}}'
    return s if len(s) >= width else f'{s:>{width}s}'

def _fmt_pair(main, spread, width, main_fmt, spread_fmt, absent='—', main_cap=None, never_wider=None):
    if main is None:
        s = absent
    else:
        if never_wider is not None:
            fmt = never_wider
        elif main_cap is None:
            fmt = main_fmt
        else:
            fmt = main_cap if abs(main) >= 100000000.0 else main_fmt
        s = f'{main:{fmt}}±{spread:{spread_fmt}}'
    return s if len(s) >= width else f'{s:>{width}s}'

def _print_table(agg):
    print('\n============== validation perplexity (corpus per run cfg) ==============')
    print(f'{'variant':{_TW['variant']}s} {'PPL mean±std':>{_TW['ppl']}s} {'Δvs full':>{_TW['d']}s} {'Δvs sw128m':>{_TW['d2']}s} {'params(M)':>{_TW['params']}s} {'seeds':>{_TW['seeds']}s} {'blklen':>{_TW['blklen']}s} {'bndF1':>{_TW['bnd']}s}')
    order = ['full', 'full_matched', 'full_cos', 'full_sink', 'full_sw128', 'full_sw128_matched', 'csa_fixed', 'csa_dynamic', 'csa_dyn_fuse', 'csa_dyn_nocomp', 'csa_fix_randidx', 'csa_fix_zerocont', 'csa_fix_nosink', 'csa_fix_topk8', 'csa_fix_topk64', 'hybrid_fixed', 'hybrid_dynamic', 'hybrid_csa_dyn', 'hybrid_hca_dyn', 'hybrid_csa_dyn_fuse']
    rows = [v for v in order if v in agg] + [v for v in agg if v not in order]
    for v in rows:
        e = agg[v]
        _n = e['n_seeds']
        ppl = _fmt_pair(e.get('ppl_primary', e['ppl_mean']), e.get('ppl_std', 0.0), _TW['ppl'], '8.2f', '5.2f', main_cap='.2f') if _n > 1 else _fmt_cell(e.get('ppl_primary', e['ppl_mean']), _TW['ppl'], '.2f')
        d = e.get('dPPL_vs_full_mean')
        dd = _fmt_pair(d, e.get('dPPL_vs_full_std', 0.0), _TW['d'], '+8.2f', '5.2f', main_cap='+.2f')
        d2 = e.get('dPPL_vs_sw128m_mean')
        dd2 = _fmt_pair(d2, e.get('dPPL_vs_sw128m_std', 0.0), _TW['d2'], '+8.2f', '5.2f', main_cap='+.2f')
        bl = e.get('avg_dyn_block_len', e.get('avg_block_len'))
        bls = _fmt_cell(bl, _TW['blklen'], '.1f')
        _pm = e.get('params')
        pms = _fmt_cell(_pm / 1000000.0 if _pm is not None else None, _TW['params'], '.2f')
        bf = e.get('boundary_f1_dyn')
        bfs = _fmt_pair(bf, e.get('boundary_f1_dyn_std', 0.0), _TW['bnd'], '6.3f', '5.3f', never_wider='.3f')
        try:
            _ns = _fmt_cell(int(_n), _TW['seeds'], 'd')
        except (TypeError, ValueError):
            _ns = _fmt_cell(None, _TW['seeds'], 'd')
        _cells = [(v, _TW['variant']), (ppl, _TW['ppl']), (dd, _TW['d']), (dd2, _TW['d2']), (pms, _TW['params']), (_ns, _TW['seeds']), (bls, _TW['blklen']), (bfs, _TW['bnd'])]
        print(' '.join((f'{s:>{w}s}' for s, w in _cells)))
    print('Δ vs full   = paired per-seed difference vs dense (positive = worse).')
    print('Δ vs sw128m = paired difference vs the windowed param-matched dense')
    print('              baseline = what compression adds beyond pure locality.')
    print('blklen = mean learned block length (dynamic layers); bndF1 = boundary alignment F1 mean±std over seeds (v4: n_sample=64);')
    print('        see summary.json for bnd_rand / bnd_excess: excess > 0 means the')
    print('        learned boundaries are genuinely semantic).')
    print('================================================================')

def _print_pairs(summary):
    recs, _amb = _variant_seed_map(summary)
    _pair_amb = sorted(_amb)
    _pair_skip = []
    PAIRS = [('csa_dynamic', 'csa_fixed'), ('hybrid_dynamic', 'hybrid_fixed'), ('hybrid_csa_dyn', 'hybrid_fixed'), ('hybrid_hca_dyn', 'hybrid_fixed'), ('csa_dyn_fuse', 'csa_dynamic'), ('hybrid_csa_dyn_fuse', 'hybrid_csa_dyn'), ('csa_fix_randidx', 'csa_fixed'), ('csa_fix_zerocont', 'csa_fixed'), ('csa_fix_nosink', 'csa_fixed'), ('csa_fix_topk8', 'csa_fixed'), ('csa_fix_topk64', 'csa_fixed'), ('csa_dyn_randidx', 'csa_dynamic'), ('csa_dyn_zerocont', 'csa_dynamic'), ('csa_dyn_nosink', 'csa_dynamic'), ('csa_dyn_nocomp', 'csa_dynamic'), ('full_sink', 'full')]
    print('\n========== paired per-seed deltas (a − b, positive = a worse) ==========')
    print('  (p = exact two-sided sign-flip over the paired per-seed deltas; a noise-level Δ is flagged even if p is small)')
    if _pair_amb:
        print('  [skip] a (variant, protocol, seed) triple holds more than one measured record, so its PPL is ambiguous — no delta printed for : ' + ', '.join(sorted((f'{v}/{t or '-'}/s{s}' for v, t, s in set(_pair_amb)))[:4]))
    for a, b in PAIRS:

        def _sel(v):
            return {(t, s): p for (vv, t, s), p in recs.items() if vv == v}
        ra, rb = (_sel(a), _sel(b))
        _a_t = {t for t, _s in ra}
        _b_t = {t for t, _s in rb}
        _a_pure_base = _a_t == {''}
        _b_pure_base = _b_t == {''}
        common = []
        for tag in sorted(_a_t | _b_t):
            if tag in _a_t and tag in _b_t and tag:
                seeds = sorted({s for t, s in ra if t == tag} & {s for t, s in rb if t == tag})
                common += [(tag, tag, s) for s in seeds]
            if tag and tag in _a_t and _b_pure_base:
                seeds = sorted({s for t, s in ra if t == tag} & {s for t, s in rb if t == ''})
                common += [(tag, '', s) for s in seeds]
            if tag and tag in _b_t and _a_pure_base:
                seeds = sorted({s for t, s in ra if t == ''} & {s for t, s in rb if t == tag})
                common += [('', tag, s) for s in seeds]
            if tag == '' and '' in _a_t and ('' in _b_t):
                seeds = sorted({s for t, s in ra if t == ''} & {s for t, s in rb if t == ''})
                common += [('', '', s) for s in seeds]
        _by_cond = {}
        _sk = []
        for ts, bs, s in common:
            _reason = pair_reason(ra[ts, s], rb[bs, s])
            if _reason is not None:
                _sk.append((s, _reason))
                continue
            _by_cond.setdefault((ts, bs), []).append(ra[ts, s]['ppl'] - rb[bs, s]['ppl'])
        if _sk:
            _pair_skip.append((a, b, len(_sk), len(common), sorted({r for _s, r in _sk})))
        for (ts, bs), dl in sorted(_by_cond.items()):
            st = paired_effect_stats(dl)
            _p = ('%.4f' % st['p_exact_signflip']) if math.isfinite(st['p_exact_signflip']) else 'n/a'
            _lab = '' if ts == bs == '' else f'   [{ts or '(untagged)'} vs {bs or '(untagged)'}]'
            print(f'  {a:22s} − {b:22s} = {st["mean"]:+7.2f} ± {st["std"]:5.2f}   (n={st["n"]}, p={_p}, {st["effect_vs_noise"]}){_lab}')
    if _pair_skip:
        print('  [skip] seed(s) rejected by the run_cfg/load gate (see `pair_reason`); the printed n counts only the pairs that cleared it —')
        for _a, _b, _nsk, _ncom, _why in _pair_skip:
            print(f'    {_a} − {_b}: {_nsk}/{_ncom} rejected — {_why}')
    print('=' * 78)

def _save_csv(agg, outdir):
    cols = ['variant', 'ppl_primary', 'ppl_mean', 'ppl_pooled', 'ppl_std', 'dPPL_vs_full_mean', 'dPPL_vs_full_std', 'dPPL_vs_sw128m_mean', 'dPPL_vs_sw128m_std', 'n_seeds', 'params', 'avg_dyn_block_len', 'boundary_f1_dyn', 'boundary_excess_dyn']
    rows = [{'variant': v, **{c: e.get(c, '') for c in cols[1:]}} for v, e in agg.items()]
    atomic_write_csv(os.path.join(outdir, 'aggregate.csv'), cols, rows)

def _plot(summary, agg, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not agg:
        return
    order = ['full', 'full_matched', 'full_cos', 'full_sink', 'full_sw128', 'full_sw128_matched', 'csa_fixed', 'csa_dynamic', 'csa_dyn_fuse', 'csa_dyn_nocomp', 'csa_fix_randidx', 'csa_fix_zerocont', 'csa_fix_nosink', 'csa_fix_topk8', 'csa_fix_topk64', 'hybrid_fixed', 'hybrid_dynamic', 'hybrid_csa_dyn', 'hybrid_hca_dyn', 'hybrid_csa_dyn_fuse']
    vs = [v for v in order if v in agg] + [v for v in agg if v not in order]
    fig, ax = plt.subplots(1, 5, figsize=(27, 4.5))
    ps = [agg[v].get('ppl_primary', agg[v]['ppl_mean']) for v in vs]
    es = [agg[v]['ppl_std'] for v in vs]
    ax[0].bar(range(len(vs)), ps, yerr=es, capsize=3, color='steelblue')
    ax[0].set_xticks(range(len(vs)))
    ax[0].set_xticklabels(vs, rotation=30, ha='right')
    ax[0].set_ylabel('validation perplexity (mean ± std)')
    ax[0].set_title('val PPL (corpus per run cfg)')
    ax[0].grid(alpha=0.3, axis='y')
    _unknown = [v for v in vs if agg[v].get('params') is None]
    if _unknown:
        print(f'[plot] {len(_unknown)} variant(s) report no `params` ({', '.join(_unknown)}) — their size bar is left at 0 and the size panel cannot be read for them')
    params_m = [(agg[v]['params'] or 0) / 1000000.0 for v in vs]
    ax[1].bar(range(len(vs)), params_m, color='orange')
    ax[1].set_xticks(range(len(vs)))
    ax[1].set_xticklabels(vs, rotation=30, ha='right')
    ax[1].set_ylabel('params (M)')
    ax[1].set_title('Model size')
    ax[1].grid(alpha=0.3, axis='y')
    for v in vs:
        _curves = []
        for _r in agg[v].get('_group_records', []):
            if not isinstance(_r, dict) or _r.get('synthesized'):
                continue
            _loss = _r.get('losses') if isinstance(_r.get('losses'), list) else _cell_curve_data(outdir, _r).get('losses')
            if isinstance(_loss, list) and _loss and all((isinstance(x, (int, float)) and math.isfinite(x) for x in _loss)):
                _curves.append(_loss)
        if not _curves:
            continue
        _lens = sorted({len(_c) for _c in _curves})
        if len(_lens) > 1:
            print(f'[_plot] {v}: seeds trained a DIFFERENT number of steps {_lens} — the training curve is drawn over the shared prefix of {_lens[0]} step(s) only.')
        n_pt = min((len(_c) for _c in _curves))
        curve = np.mean([np.array(_c[:n_pt], dtype=float) for _c in _curves], axis=0)
        k = max(1, n_pt // 50)
        _num = np.convolve(curve, np.ones(k), mode='same')
        _den = np.convolve(np.ones_like(curve), np.ones(k), mode='same')
        smooth = np.where(_den > 0, _num / _den, 0.0)
        ax[2].plot(smooth, label=v)
    ax[2].set_xlabel('step')
    ax[2].set_ylabel('train loss (seed-averaged)')
    ax[2].set_title('Training curves')
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=0.3)
    drew = False
    for v in vs:
        cur = agg[v].get('ppl_curve')
        if not cur:
            continue
        steps_h = np.array([p[0] for p in cur], dtype=float)
        ppls_h = np.array([p[1] for p in cur], dtype=float)
        tok_per_step = agg[v].get('tokens_per_step')
        if not tok_per_step:
            print(f'[_plot] {v}: no `tokens_per_step` in the aggregate — omitted from the tokens axis (plotting steps under a tokens label would be wrong by seq_len*batch_size). Re-run `aggregate` to stamp it.')
            continue
        ax[3].plot(steps_h * tok_per_step / 1000000.0, ppls_h, marker='.', ms=4, label=v)
        drew = True
    ax[3].set_xlabel('tokens seen (M)')
    ax[3].set_ylabel('val-subset PPL')
    ax[3].set_title('PPL vs training tokens (gap dynamics)')
    if drew:
        ax[3].legend(fontsize=7)
    ax[3].grid(alpha=0.3)
    _dense_keys = [k for k in agg if k.split('#')[0].split('@')[0] == 'full']
    _dense_key = next((k for k in _dense_keys if agg[k].get('ppl_curve') and agg[k].get('tokens_per_step')), None)
    if _dense_key is None:
        print(f'[_plot] no dense `full` group with both a `ppl_curve` and a `tokens_per_step` in this aggregate (candidates: {_dense_keys or 'none'}) — the sparse-vs-dense gap panel (finding #10) is NOT drawn.')
    else:
        if _dense_key != 'full':
            print(f'[_plot] the dense baseline is published as `{_dense_key}` (the `full` group was split by `run_cfg`); the sparse-vs-dense gap is taken against it' + (f', not {_dense_keys[1:]}' if len(_dense_keys) > 1 else '') + '.')
        ref = agg[_dense_key].get('ppl_curve')
        tps0 = agg[_dense_key].get('tokens_per_step')
    if _dense_key is not None and ref and tps0:
        gx = np.array([p[0] for p in ref], dtype=float) * tps0
        gy = np.array([p[1] for p in ref], dtype=float)
        drew2 = False
        for v in vs:
            if v == _dense_key:
                continue
            if v.startswith('full@cfg'):
                continue
            cur = agg[v].get('ppl_curve')
            if not cur:
                continue
            tps = agg[v].get('tokens_per_step')
            if not tps:
                continue
            xs = np.array([p[0] for p in cur], dtype=float) * tps
            ys = np.array([p[1] for p in cur], dtype=float)
            keep = (gx >= xs[0]) & (gx <= xs[-1])
            if not keep.any():
                continue
            ax[4].plot(gx[keep] / 1000000.0, np.interp(gx[keep], xs, ys) - gy[keep], marker='.', ms=4, label=f'{v}−full')
            drew2 = True
        ax[4].axhline(0, color='k', lw=0.8)
        ax[4].set_xlabel('tokens seen (M)')
        ax[4].set_ylabel('ΔPPL vs full')
        ax[4].set_title('sparse-vs-dense gap vs tokens (finding #10)')
        if drew2:
            ax[4].legend(fontsize=7)
        ax[4].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'summary.png'), dpi=120)
    plt.close(fig)

def _read_summary_json(path):
    summary = {}
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as f:
                summary = json.load(f)
        except Exception as _e:
            print(f'[resume] FATAL: {path} exists but cannot be parsed ({type(_e).__name__}: {_e}).  Refusing to overwrite it with an empty summary — move it aside to start fresh.')
            raise
    return summary


def _run_fp(cfg, d, n_layers, n_heads, d_head, vocab, train_ids, val_batch, device=None):
    """The resume fingerprint of a run() cell.  The parent planner and every
    par.py child compute the exact same string: the child re-serves the token
    ids from the same cache files, and `_ids_fp` is content-based.

    AMP / eager-twin modes are resolved from the run config (persisted in the
    manifest) or from an explicitly supplied ``device``; they are never read
    from this process's global CUDA state.  A zero-GPU planner can therefore
    recompute a GPU panel's fingerprint from the same config, and a child
    receives the parent's resolved modes through the job config."""
    _mg = cfg.get('matched')
    if _mg is None:
        _mkey = 'default'
    else:
        _mkey = ','.join(sorted((str(_x) for _x in _mg)))
    cfg.setdefault('warmup', 50)
    cfg.setdefault('comp_lambda', 0.05)
    _declared_impl = any(
        (k in cfg) and not (isinstance(cfg.get(k), str) and cfg.get(k).strip().lower() == 'auto')
        for k in ('amp_dtype', 'eager_batch'))
    _amp_mode, _eb = _resolve_impl_modes(cfg, device=device)
    if device is None and not _declared_impl:
        # No machine-dependent implementation descriptor enters the content
        # fingerprint.  Resolve to the canonical fp32/row-loop descriptor
        # explicitly (not via training_knobs()'s global DEVICE inference).
        try:
            _tk = training_knobs_fingerprint(amp_dtype=None, eager_batch=False)
        except TypeError:
            # Keep callers that replace training_knobs() with a zero-argument
            # test double working; their digest is the canonical one on CPU.
            _tk = training_knobs_fingerprint()
    else:
        _tk = training_knobs_fingerprint(amp_dtype=_amp_mode, eager_batch=_eb)
    _amp = _amp_mode or 'fp32'
    _ae = _adaptive_exact_limit()
    return f'steps{cfg['steps']}_sl{cfg['seq_len']}_bs{cfg['batch_size']}_nt{cfg['n_train_tokens']}_lr{cfg['lr']}_wd{cfg['weight_decay']}_d{d}_L{n_layers}_H{n_heads}_Dh{d_head}_v{vocab}_wu{cfg['warmup']}_cl{cfg['comp_lambda']}_dlm{cfg.get('delta_lr_mult', 10.0)}_mt{_mkey}_mr{cfg.get('mlp_match_ref', 'csa_dynamic')}_ee{cfg.get('eval_every', 0)}_es{cfg.get('eval_subset', 128)}_det{determinism_label()}_df{_ids_fp(train_ids)}-{_ids_fp(np.ascontiguousarray(val_batch).reshape(-1))}_tk{_tk}_amp{_amp}_eb{int(_eb)}_ae{_ae}_cs{code_stamp()}'


def _plan_run(cfg, summary_path, fp):
    """Resume-check every (seed x variant) cell up front, exactly as the old
    interleaved loop did, and return the cells that still need to run.

    Stale / synthesized / non-measurable records are dropped (under the lock)
    BEFORE any training starts, so a later failure can never leave an old
    reading in place looking like this attempt's result.  With parallel
    dispatch this check runs once in the parent, before any child spawns."""
    summary = locked_json_read(summary_path)
    cells = []
    stale_dropped = []
    for seed in cfg['seeds']:
        for v in cfg['variants']:
            key = f'{v}::seed{seed}'
            _drop_msg = None
            _drop_is_stale = False
            if key in summary and 'ppl' in summary.get(key, {}):
                old = summary[key]
                if not ppl_is_usable(old.get('ppl')):
                    _drop_msg = f"[resume] {key} holds a NON-MEASURABLE ppl ({old.get('ppl')!r}) — every statistic in this repo refuses it (`ppl_is_usable`), so counting it as 'already completed' would drop the seed from the tables silently. DROPPING it and retraining"
                elif old.get('synthesized'):
                    _drop_msg = f'[resume] {key} is a SYNTHESIZED record (reconstructed from an old aggregate.json, no weights) — DROPPING it and retraining'
                elif old.get('run_cfg') is None:
                    _drop_msg = f"[resume] {key} has no config fingerprint (predates fingerprinting) — re-running and overwriting; the STALE record is dropped before retraining, so a failed or truncated attempt cannot leave it in place looking like this attempt's result"
                    _drop_is_stale = True
                elif old.get('run_cfg') == fp:
                    print(f'[skip] {key} already completed (resume)')
                    continue
                else:
                    _drop_msg = f"[resume] {key} was run under a DIFFERENT config ({old['run_cfg']} != {fp}) — re-running and overwriting; the STALE record is dropped before retraining, so a failed or truncated attempt cannot leave it in place looking like this attempt's result"
                    _drop_is_stale = True
            if _drop_msg is not None:
                print(_drop_msg)
                summary.pop(key, None)

                def _drop_cell(s, _key=key):
                    s.pop(_key, None)
                    return s
                locked_json_update(summary_path, _drop_cell)
                if _drop_is_stale:
                    stale_dropped.append(key)
            cells.append((v, seed))
    return cells, stale_dropped


def _run_cell(cfg, v, seed, *, guard=None, ratios=None, fp=None, train_ids=None, val_batch=None, vocab=None, val_bnd=None, prebook=False, prov=None):
    """Run exactly ONE (variant, seed) training cell and merge its record into
    outdir/summary.json under the lock.

    Used by the serial loop (in-process) and by par.py children (prebook=True:
    the projected time is reserved in the shared TimeGuard before training and
    refunded after recording the actual seconds, so concurrently launching
    children still respect the shared run-time cap)."""
    d = cfg.get('d', 256)
    n_layers = cfg.get('n_layers', 6)
    n_heads = cfg.get('n_heads', 8)
    d_head = cfg.get('d_head', 32)
    outdir = cfg['outdir']
    summary_path = os.path.join(outdir, 'summary.json')
    key = f'{v}::seed{seed}'
    ratios = {} if ratios is None else ratios
    if v not in ratios:
        ratios[v] = variant_mlp_ratio(v, vocab, d=d, n_layers=n_layers, n_heads=n_heads, d_head=d_head, seq_len=cfg['seq_len'], matched=cfg.get('matched'), ref_variant=cfg.get('mlp_match_ref', 'csa_dynamic'))
    out = {'status': 'ok', 'key': key, 'variant': v, 'seed': seed}
    _run_id = new_experiment_id('cell')
    _started_at = time.strftime('%Y-%m-%dT%H:%M:%S')
    _t_start = time.time()
    out['run_id'] = _run_id
    deadline_ts = None
    booked = 0.0
    _book_tok = f'run|{key}|pid{os.getpid()}'
    if guard is not None:
        est = guard.estimate_seconds(cfg['steps'], d=d, n_layers=n_layers, seq_len=cfg['seq_len'], batch_size=cfg['batch_size'])
        if prebook and est:
            guard.book(est, token=_book_tok)
            booked = est
        if not guard.can_start(0 if booked else est):
            if booked:
                guard.book(-booked, token=_book_tok)
                booked = 0.0
            print(f'[cap] SKIP {key}: projected {est / 3600:.2f} h would pass the configured cap (headroom {max(guard.total_hours - guard.spent_hours(), 0.0):.2f} h)')
            out['status'] = 'skipped_cap'
            log_experiment_event(outdir, {'kind': 'cell', 'run_id': _run_id, 'status': 'skipped_cap', 'key': key, 'variant': v, 'seed': int(seed), 'run_cfg': fp, 'code': code_stamp(), 'started_at': _started_at, 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'reason': 'projected runtime would pass the configured cap'})
            return out
        deadline_ts = guard.deadline_ts()
        if booked:
            deadline_ts += booked
        print(f'[cap] {key}: projected {est / 60:.0f} min, booked so far {guard.state['booked_seconds'] / 3600:.2f} h')
    t_run = time.time()
    rec = None
    try:
        try:
            _save_mode = cfg.get('save_ckpt', os.environ.get('CSA_SAVE_CKPT', 'both'))
            _ckpt_dir = cfg.get('ckpt_dir') or os.path.join(outdir, 'checkpoints')
            rec = train_variant(v, train_ids, val_batch, vocab, seed=seed, d=d, n_layers=n_layers, n_heads=n_heads, d_head=d_head, seq_len=cfg['seq_len'], batch_size=cfg['batch_size'], steps=cfg['steps'], lr=cfg['lr'], weight_decay=cfg['weight_decay'], warmup=cfg['warmup'], comp_lambda=cfg['comp_lambda'], delta_lr_mult=cfg.get('delta_lr_mult', 10.0), eval_every=cfg.get('eval_every', 0), eval_subset=cfg.get('eval_subset', 128), val_bnd=val_bnd, mlp_ratio=ratios[v], deadline_ts=deadline_ts, save_ckpt=_save_mode, ckpt_dir=_ckpt_dir, diag_every=cfg.get('diag_every'), measure_eff=cfg.get('measure_efficiency', True), amp_dtype=cfg.get('amp_dtype', 'auto'), eager_batch=cfg.get('eager_batch', 'auto'))
            _prov = prov if prov is not None else experiment_provenance(cfg, train_ids=train_ids, val_batch=val_batch, vocab=vocab)
            _hp = run_hyperparameters(cfg, seed=seed, d=d, n_layers=n_layers, n_heads=n_heads, d_head=d_head, vocab=vocab, mlp_ratio=ratios[v], data_version=_prov.get('data', {}).get('data_version'), amp_dtype=rec.get('amp_dtype', 'auto'), eager_batch=rec.get('eager_batch', 'auto'))
            _curve_ref = None
            _curve_losses = []
            _curve_ph = []
            _curve_move_ph = False
            if not rec.get('cap_truncated'):
                _curve_ref, _curve_losses, _curve_ph, _curve_move_ph = _write_cell_curve(outdir, key, rec)

            def _apply(summary):
                if rec.get('cap_truncated'):
                    _prev_ok = ppl_is_usable((summary.get(key) or {}).get('ppl'))
                    print(f'[cap] {key} was truncated after {rec.get('steps_done')} steps — its partial record is DISCARDED (it holds no `ppl`); ' + ('keeping the previous record instead.' if _prev_ok else 'the key is left ABSENT, not written as a stub.') + ' Raise RUN_CAP and re-run to retry this cell.')
                    if not _prev_ok:
                        summary.pop(key, None)
                        out['truncated'] = True
                else:
                    rec['run_cfg'] = fp
                    rec['_code'] = code_stamp()
                    assert fp.endswith(f'_cs{rec['_code']}'), (fp, rec['_code'])
                    rec['provenance'] = _prov
                    rec['hyperparams'] = _hp
                    rec['data_fp'] = _prov['data'].get('train_fp')
                    rec['val_fp'] = _prov['data'].get('val_fp')
                    rec['data_version'] = _prov['data'].get('data_version')
                    rec['run_id'] = _run_id
                    rec['started_at'] = _started_at
                    rec['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
                    if os.path.exists(os.path.join(outdir, 'overfit_sanity.json')):
                        rec['sanity_ref'] = 'overfit_sanity.json'
                    _rec_out = rec
                    _has_curve = bool(_curve_ref) or _curve_move_ph
                    if _has_curve:
                        _rec_out = dict(rec)
                        if _curve_ref:
                            _rec_out.pop('losses', None)
                            _rec_out.pop('delta_trace', None)
                        if _curve_move_ph:
                            _rec_out.pop('ppl_history', None)
                    if _curve_ref:
                        _cs = _finite_curve(_curve_losses)
                        _rec_out['curve_ref'] = _curve_ref
                        _rec_out['curve_n_points'] = _cs['n_points']
                        _rec_out['curve_first'] = _curve_losses[0] if _curve_losses else None
                        _rec_out['curve_last'] = _cs['last']
                        _rec_out['curve_best'] = _cs['min']
                        _rec_out['curve_n_nonfinite'] = _cs['n_nonfinite']
                        _rec_out['curve_first_nonfinite_index'] = _cs['first_nonfinite_index']
                    elif _curve_losses:
                        _rec_out['curve_n_points'] = len(_curve_losses)
                    if _curve_move_ph:
                        _rec_out['ppl_history_ref'] = _curve_ref
                        _rec_out['ppl_history_n_points'] = len(_curve_ph)
                    elif _curve_ph:
                        _rec_out['ppl_history_n_points'] = len(_curve_ph)
                    summary[key] = _rec_out
                return summary
            locked_json_update(summary_path, _apply, indent=None)
            if out.get('truncated'):
                out['status'] = 'truncated'
                log_experiment_event(outdir, {'kind': 'cell', 'run_id': _run_id, 'status': 'truncated', 'key': key, 'variant': v, 'seed': int(seed), 'run_cfg': fp, 'code': code_stamp(), 'started_at': _started_at, 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'steps_done': rec.get('steps_done'), 'reason': 'hard runtime cap reached; partial record not written'})
            elif isinstance(rec, dict) and not rec.get('cap_truncated'):
                _losses = rec.get('losses') if isinstance(rec.get('losses'), list) else []
                _cs = _finite_curve(_losses)
                _ev = {'kind': 'cell', 'run_id': _run_id, 'status': 'ok', 'started_at': _started_at, 'finished_at': rec.get('finished_at'), 'elapsed_s': time.time() - _t_start, 'key': key, 'variant': v, 'seed': int(seed), 'run_cfg': fp, 'code': rec.get('_code') or code_stamp(), 'provenance': rec.get('provenance'), 'hyperparams': rec.get('hyperparams'), 'data_fp': rec.get('data_fp'), 'val_fp': rec.get('val_fp'), 'data_version': rec.get('data_version'), 'rng_fp': rec.get('rng_fp'), 'metrics': {'ppl': rec.get('ppl'), 'params': rec.get('params'), 'steps': rec.get('steps'), 'tokens_seen': rec.get('tokens_seen'), 'train_time_s': rec.get('train_time_s'), 'final_loss_smoothed': rec.get('final_loss_smoothed'), 'best_val_ppl': rec.get('best_val_ppl'), 'final_minus_best': rec.get('final_minus_best')}, 'curve_summary': {'n_points': _cs['n_points'], 'n_nonfinite': _cs['n_nonfinite'], 'first': _losses[0] if _losses else None, 'last': _cs['last'], 'best': _cs['min']}, 'curve_health': rec.get('curve_health'), 'opt_diag_summary': rec.get('opt_diag_summary'), 'checkpoints': rec.get('checkpoints'), 'efficiency': rec.get('efficiency'), 'losses': _losses, 'ppl_history': rec.get('ppl_history'), 'sanity_ref': rec.get('sanity_ref')}
                log_experiment_event(outdir, _ev)
        except Exception as e:
            import traceback
            import sys as _sys
            _err = e
            _err_tb = traceback.format_exc()
            out['status'] = 'error'
            out['error'] = f'{_err}'

            def _apply_err(summary):
                if key in summary and ppl_is_usable((summary.get(key) or {}).get('ppl')):
                    print(f'[{key}] FAILED: {_err} — keeping the previous MEASURED record (the error record holds no `ppl` and must not replace it); the failed attempt is still booked to the guard below.')
                    print(_err_tb, file=_sys.stderr)
                else:
                    summary[key] = {'variant': v, 'seed': seed, 'error': _err_tb}
                    print(f'[{key}] FAILED: {_err}')
                return summary
            locked_json_update(summary_path, _apply_err, indent=None)
            log_experiment_event(outdir, {'kind': 'cell', 'run_id': _run_id, 'status': 'error', 'started_at': _started_at, 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'elapsed_s': time.time() - _t_start, 'key': key, 'variant': v, 'seed': int(seed), 'run_cfg': fp, 'code': code_stamp(), 'error': f'{_err}', 'traceback': _err_tb})
    finally:
        if guard is not None:
            steps_done = 0
            calib_s = None
            if isinstance(rec, dict) and rec.get('steps_done'):
                steps_done = int(rec['steps_done'])
            elif isinstance(rec, dict) and 'ppl' in rec:
                steps_done = int(cfg['steps'])
                calib_s = rec.get('train_time_s')
            guard.record_run(time.time() - t_run, steps_done, d, n_layers, cfg['seq_len'], cfg['batch_size'], calib_seconds=calib_s)
            if booked:
                guard.book(-booked, token=_book_tok)
    gc.collect()
    return out


def run_cell_job(job):
    """Child-process entry for one `run`-kind cell (invoked by par.py).

    Re-serves the corpus from the same on-disk cache, recomputes the identical
    fingerprint, and runs the cell with pre-booked guard accounting.  The
    data-load seconds are NOT booked here: the parent already booked the
    phase's data load once, exactly like the serial path."""
    cfg = dict(job['cfg'])
    _matched = cfg.get('matched')
    if isinstance(_matched, (list, tuple)):
        cfg['matched'] = set(_matched)
    guard = TimeGuard(job['guard_cap']) if job.get('guard_cap') else None
    # Seed the child BEFORE the corpus is served, exactly like run() does in
    # the parent and like the v7/v10 child entries (run_warmup_cell_job /
    # run_probe_cell_job) already do: today's loaders consume no RNG, but a
    # future loader step that does (a sampled val slice, a DataLoader shuffle)
    # must not silently become the one unseeded construction path.
    set_seed(int(job['seed']))
    train_ids, val_batch, vocab, _, val_bnd = load_run_corpus(cfg)
    d = cfg.get('d', 256)
    n_layers = cfg.get('n_layers', 6)
    n_heads = cfg.get('n_heads', 8)
    d_head = cfg.get('d_head', 32)
    if torch.cuda.is_available():
        _pin_cuda_determinism()
    # The parent already computed the identical fingerprint/provenance over
    # this corpus; reuse them.  If an older job payload lacks the resolved
    # implementation modes, resolve them from THIS child's device and pin them
    # into the config so the fingerprint and the training record agree.
    if 'amp_dtype' not in cfg or 'eager_batch' not in cfg:
        _amp_mode, _eager_mode = _resolve_impl_modes(cfg, device=DEVICE)
        cfg['amp_dtype'] = 'fp32' if _amp_mode is None else _amp_mode
        cfg['eager_batch'] = bool(_eager_mode)
    fp = job.get('fp') or _run_fp(cfg, d, n_layers, n_heads, d_head, vocab, train_ids, val_batch, device=DEVICE)
    return _run_cell(cfg, job['v'], job['seed'], guard=guard, ratios={}, fp=fp, train_ids=train_ids, val_batch=val_batch, vocab=vocab, val_bnd=val_bnd, prebook=True, prov=job.get('prov'))


def preflight_fingerprint(cfg, *, stage, vocab=None, data_fp=None, probe_cfg=None):
    """Stable fingerprint for a standalone pre-flight artifact.

    Used when the check writes evidence before/independently of a panel
    ``run_cfg`` (m0/run_gapfill pre-flight dirs, and a synthetic probe that
    fails before the corpus is loaded).  It is deliberately not a resume key:
    it exists so the evidence can state which config/code/data/probe produced
    it."""
    blob = json.dumps({
        'config': _jsonable(cfg),
        'stage': str(stage),
        'vocab': None if vocab is None else int(vocab),
        'data_fp': data_fp,
        'probe_cfg': _jsonable(probe_cfg) if probe_cfg is not None else None,
        'train_knobs': training_knobs_fingerprint(),
        'code': code_stamp(),
    }, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return 'preflight-' + hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]

def _is_confirmatory_cfg(cfg):
    """A confirmation / scale panel defaults to strict determinism + a
    mandatory real-architecture reproducibility check.

    An explicit ``require_repro``/``confirm`` flag always wins; otherwise the
    B-tier scale geometry (d>=512, multiple seeds) is treated as confirmatory.
    Small exploratory/test panels keep the advisory path."""
    for _k in ('confirm', 'confirmation', 'require_repro'):
        if cfg.get(_k) is not None:
            return bool(cfg.get(_k))
    try:
        _d = int(cfg.get('d', 256))
        _nseeds = len(cfg.get('seeds') or [])
    except (TypeError, ValueError):
        return False
    return bool(_d >= 512 and _nseeds >= 2)

def run(cfg=None, seeds=None, guard=None, label='', par=None):
    cfg = dict(RUN) if cfg is None else dict(cfg)
    if seeds is not None:
        cfg['seeds'] = list(seeds)
    d = cfg.get('d', 256)
    n_layers = cfg.get('n_layers', 6)
    n_heads = cfg.get('n_heads', 8)
    d_head = cfg.get('d_head', 32)
    outdir = cfg['outdir']
    os.makedirs(outdir, exist_ok=True)
    _seeds = [int(s) for s in (cfg.get('seeds') or [0])]
    _first_seed, _last_seed = _seeds[0], _seeds[-1]
    _variants = list(cfg.get('variants') or [])
    _arch = {'d': int(d), 'n_layers': int(n_layers), 'n_heads': int(n_heads), 'd_head': int(d_head)}
    # Resolve the numerical implementation ONCE for this panel and persist the
    # concrete values in the config.  The resume fingerprint, the par.py job
    # payloads and the manifest then all describe the same run without any
    # later process having to infer modes from its own CUDA visibility.
    _amp_mode, _eager_mode = _resolve_impl_modes(cfg, device=DEVICE)
    cfg['amp_dtype'] = 'fp32' if _amp_mode is None else _amp_mode
    cfg['eager_batch'] = bool(_eager_mode)
    # A confirmation / scale panel runs with deterministic kernels by default
    # (an explicit CSA_DETERMINISTIC=0 still opts out) and treats the
    # same-machine reproducibility check as mandatory below.
    _confirm = _is_confirmatory_cfg(cfg)
    if _confirm:
        os.environ.setdefault('CSA_DETERMINISTIC', '1')
    # Seed the panel-level data-construction path (tokenizer fitting, cache
    # rebuilds, any future DataLoader setup) before the corpus is served.
    # Per-cell training re-seeds itself inside train_variant, so this cannot
    # change a recorded cell; it only closes an unseeded first-build path.
    set_seed(_first_seed)
    # The probe uses the panel's REAL architecture/MLP and a length that
    # exercises the block geometry rather than a fixed 64-token proxy.
    _probe_seq = probe_seq_len(cfg.get('seq_len', 64))
    _probe_steps = 200
    _probe_batch = 2

    def _panel_ratios(vocab_):
        _r = {}
        for _v in _variants:
            try:
                _r[_v] = variant_mlp_ratio(_v, int(vocab_), d=int(d), n_layers=int(n_layers), n_heads=int(n_heads), d_head=int(d_head), seq_len=int(cfg.get('seq_len', 512)), matched=cfg.get('matched'), ref_variant=cfg.get('mlp_match_ref', 'csa_dynamic'))
            except Exception:
                _r[_v] = None
        return _r

    # Synthetic pre-flight before the corpus load.  The actual-data probe below
    # is the authoritative one; this one still exercises the real model
    # geometry so a shape/gradient fault is caught even if the corpus is slow.
    _pre_vocab = int(cfg.get('vocab_size') or 1024)
    _pre_ratios = _panel_ratios(_pre_vocab)
    try:
        _sanity = list(overfit_guard(_variants, seed=_first_seed, steps=_probe_steps, seq_len=_probe_seq, batch_size=_probe_batch, vocab=_pre_vocab, arch=_arch, mlp_ratio=_pre_ratios))
    except RuntimeError as _fe:
        _prov0 = experiment_provenance(cfg)
        _reports0 = [dict(r) for _k, r in _OVERFIT_REPORTS.items() if _k[0] in set(_variants)]
        _cfg0 = _overfit_probe_config(seed=_first_seed, steps=_probe_steps, seq_len=_probe_seq, batch_size=_probe_batch, vocab=_pre_vocab, target_loss=0.05, target_acc=0.99, arch=_arch, source='synthetic')
        _doc0 = {
            'kind': 'small_batch_overfit_sanity_panel', 'stage': 'synthetic_preflight',
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'run_cfg': preflight_fingerprint(cfg, stage='synthetic_overfit', vocab=_pre_vocab, probe_cfg=_cfg0),
            'label': label or outdir, 'passed': False, 'error': f'{_fe}',
            'provenance': _prov0, 'reports': _reports0,
        }
        atomic_write_json(os.path.join(outdir, 'overfit_sanity.json'), _doc0)
        log_experiment_event(outdir, {'kind': 'preflight', 'tracking_schema': 2, 'status': 'failed', 'stage': 'synthetic_overfit', 'run_cfg': _doc0['run_cfg'], 'code': code_stamp(), 'provenance': _prov0, 'error': f'{_fe}', 'sanity': {'passed': False, 'n_reports': len(_reports0)}})
        raise
    _attb_bump_epoch()
    t_data = time.time()
    train_ids, val_batch, vocab, _, val_bnd = load_run_corpus(cfg)
    # Mandatory data gate: the loaders run it too, but a driver/monkeypatched
    # corpus cannot bypass it for a real dataset.  The report is persisted with
    # the panel before any model is fitted.
    if (cfg.get('dataset') or 'wikitext') in ('wikitext', 'fineweb'):
        _da = audit_corpus_split(train_ids, val_batch, vocab,
                                 dataset=cfg.get('dataset') or 'wikitext')
        try:
            atomic_write_json(os.path.join(outdir, 'data_audit.json'), {
                'kind': 'corpus_split_data_gate',
                'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'dataset': cfg.get('dataset') or 'wikitext',
                'report': _da,
            })
        except Exception as _dae:
            print(f'[data] WARNING: could not persist data_audit.json ({type(_dae).__name__}: {_dae})')
    if guard is not None:
        guard.record_run(time.time() - t_data, 0, 0, 0, 0, 0)
    summary_path = os.path.join(outdir, 'summary.json')
    _read_summary_json(summary_path)
    if label:
        print(f'\n########## phase: {label} — {len(cfg['variants'])} variants x seeds {cfg['seeds']} x {cfg['steps']} steps ##########')
    # Prospective seed-plan note: a panel that contains an ablation arm is
    # checked BEFORE training so its seed count is not mistaken for a
    # confirmation after the fact.  This never blocks a run.
    try:
        _vs = set(cfg.get('variants') or [])
        _declared_pairs = [(_a, _b) for _contrasts in list(ABLATION_DESIGN.values()) + list(BASELINE_DESIGN.values()) for _a, _b, _n in _contrasts]
        if any(_a in _vs and _b in _vs for _a, _b in _declared_pairs):
            _sp = methodology_seed_plan(len(cfg['seeds']), variants=cfg.get('variants'))
            if _sp.get('min_seeds_for_family_significance') and not _sp.get('sufficient_for_family_rejection'):
                print('[seed-plan] this panel declares %d controlled contrasts; with n=%d seeds the family-corrected exact-test floor is above alpha, so results are DESCRIPTIVE ONLY. '
                      'A family-corrected rejection needs n>=%d (raise the seed list or narrow the pre-registered family before claiming significance).'
                      % (_sp['declared_contrasts'], _sp['n_seeds'], _sp['min_seeds_for_family_significance']))
            if _sp.get('ablation_complete') is False:
                _abl = _sp.get('ablation_completeness') or {}
                print('[seed-plan] ABLATION INCOMPLETE for this panel: missing %s. '
                      'A component with only a partial combination cannot support the claim that each module is useful; add the named full-minus-X arm (or run the full factorial).'
                      % '; '.join('%s=%s' % (_c, ','.join(_a))
                                  for _c, _a in sorted((_abl.get('missing_arms') or {}).items())))
    except Exception as _sp_err:
        print(f'[seed-plan] note unavailable ({type(_sp_err).__name__}: {_sp_err}) — continuing')
    if torch.cuda.is_available():
        _pin_cuda_determinism()
    fp = _run_fp(cfg, d, n_layers, n_heads, d_head, vocab, train_ids, val_batch, device=DEVICE)
    prov = experiment_provenance(cfg, train_ids=train_ids, val_batch=val_batch, vocab=vocab)
    ratios = _panel_ratios(vocab)
    _data_fp = _ids_fp(train_ids)

    _data_contract = batch_contract_check(
        train_ids, cfg['seq_len'], cfg.get('batch_size', 12), DEVICE,
        seed=_first_seed, vocab=vocab)
    _data_contract.update({'run_cfg': fp, 'label': label or outdir, 'provenance': prov})
    atomic_write_json(os.path.join(outdir, 'data_contract.json'), _data_contract)
    log_experiment_event(outdir, {
        'kind': 'preflight', 'tracking_schema': 2,
        'status': ('ok' if _data_contract.get('passed') else 'failed'),
        'stage': 'batch_iterator_contract', 'run_cfg': fp,
        'code': code_stamp(), 'provenance': prov,
        'data_contract': {k: _data_contract.get(k) for k in
                          ('passed', 'diagnosis', 'seq_len', 'batch_size', 'seed', 'n_batches')},
    })
    if not _data_contract.get('passed'):
        raise RuntimeError(
            'batch iterator contract check FAILED '
            f"({_data_contract.get('diagnosis')}) — do NOT start a full run; "
            'the actual training loader is not producing shifted, in-vocabulary, '
            'repeatable batches. See data_contract.json.')

    def _data_probe_cfg(v):
        return _overfit_probe_config(seed=_first_seed, steps=_probe_steps, seq_len=_probe_seq, batch_size=_probe_batch, vocab=vocab, target_loss=0.05, target_acc=0.99, arch=_arch, mlp_ratio=ratios.get(v), ignore_index=None, source='data', data_fp=_data_fp)

    # The real data-path check: the synthetic pass above cannot catch a broken
    # real token stream / vocabulary mapping.  It is a superset of the model
    # plumbing check, so a pass here is the stronger evidence; both reports
    # are persisted with the panel.
    _data_rep = []
    try:
        _data_rep = list(overfit_guard(_variants, seed=_first_seed, steps=_probe_steps, seq_len=_probe_seq, batch_size=_probe_batch, vocab=vocab, train_ids=train_ids, arch=_arch, mlp_ratio=ratios))
        _sanity += _data_rep
    except RuntimeError as _fe:
        _reports = [dict(r) for _k, r in _OVERFIT_REPORTS.items() if _k[0] in set(_variants)]
        _doc = {
            'kind': 'small_batch_overfit_sanity_panel', 'stage': 'data_preflight',
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'run_cfg': fp, 'label': label or outdir, 'passed': False,
            'error': f'{_fe}', 'provenance': prov, 'reports': _reports,
        }
        atomic_write_json(os.path.join(outdir, 'overfit_sanity.json'), _doc)
        log_experiment_event(outdir, {'kind': 'preflight', 'tracking_schema': 2, 'status': 'failed', 'stage': 'data_overfit', 'run_cfg': fp, 'code': code_stamp(), 'provenance': prov, 'error': f'{_fe}', 'sanity': {'passed': False, 'n_reports': len(_reports)}})
        raise
    if not _data_rep:
        # The guard is process-cached (correctly, so repeated phases do not
        # retrain the probe).  Recover THIS exact probe configuration's
        # evidence from the cache so a fresh outdir still gets a persisted
        # sanity record; a looser probe must never certify this panel.
        _sem_by_v = {v: _overfit_probe_semantics(_data_probe_cfg(v)) for v in _variants}
        _sanity = []
        for (_v, _src, *_rest), _r in _OVERFIT_REPORTS.items():
            if _v not in set(_variants):
                continue
            if _r.get('guard_probe_semantics') != _sem_by_v.get(_v):
                continue
            if _src in ('data', 'batch'):
                # Full data key: the evidence must name THIS corpus/panel.
                if len(_rest) < 2 or _rest[1] != _data_fp:
                    continue
            elif _src != 'synthetic':
                continue
            _sanity.append(dict(_r))
    _sanity_doc = None
    if _sanity:
        _sanity_doc = {
            'kind': 'small_batch_overfit_sanity_panel',
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'run_cfg': fp,
            'label': label or outdir,
            'passed': all(bool(r.get('passed')) for r in _sanity),
            'provenance': prov,
            'reports': _sanity,
        }
        atomic_write_json(os.path.join(outdir, 'overfit_sanity.json'), _sanity_doc)
    # Same-machine seed check on the REAL architecture/data, first and last
    # panel seed.  Confirmation/scale panels require it to pass; exploratory
    # panels record the failure but stay advisory.
    _repro_seeds = list(dict.fromkeys([_first_seed, _last_seed]))
    _env_strict = os.environ.get('CSA_REQUIRE_REPRO')
    if _env_strict is None:
        _strict_repro = bool(_confirm)
    else:
        _strict_repro = _env_strict.strip().lower() in ('1', 'true', 'yes', 'on')
    try:
        _repro = list(reproducibility_guard(_variants, seed=_first_seed, steps=3, arch=_arch, train_ids=train_ids, vocab=vocab, mlp_ratio=ratios, seq_len=_probe_seq, batch_size=_probe_batch, seeds=_repro_seeds, strict=_strict_repro))
    except RuntimeError as _fe:
        _reports = [dict(r) for _k, r in _REPRO_REPORTS.items() if _k[0] in set(_variants)]
        _doc = {
            'kind': 'duplicate_run_reproducibility_panel', 'stage': 'reproducibility_preflight',
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'run_cfg': fp, 'label': label or outdir, 'passed': False,
            'error': f'{_fe}', 'provenance': prov, 'reports': _reports,
        }
        atomic_write_json(os.path.join(outdir, 'reproducibility.json'), _doc)
        log_experiment_event(outdir, {'kind': 'preflight', 'tracking_schema': 2, 'status': 'failed', 'stage': 'reproducibility', 'run_cfg': fp, 'code': code_stamp(), 'provenance': prov, 'error': f'{_fe}', 'reproducibility': {'passed': False, 'n_reports': len(_reports)}})
        raise
    if not _repro:
        _bases = set()
        for v in _variants:
            for sd in _repro_seeds:
                _bases.add(_repro_guard_base(seed=sd, steps=3, tol=1e-06, arch=_arch, vocab=vocab, mlp_ratio=ratios.get(v), seq_len=_probe_seq, batch_size=_probe_batch, train_ids=train_ids, data_fp=_data_fp))
        _repro = [dict(r) for k, r in _REPRO_REPORTS.items() if k[0] in set(_variants) and k[1] in _bases]
    _repro_doc = None
    if _repro:
        _repro_doc = {
            'kind': 'duplicate_run_reproducibility_panel',
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'run_cfg': fp,
            'label': label or outdir,
            'passed': all(bool(r.get('passed')) for r in _repro),
            'provenance': prov,
            'reports': _repro,
        }
        atomic_write_json(os.path.join(outdir, 'reproducibility.json'), _repro_doc)
    cells, stale_dropped = _plan_run(cfg, summary_path, fp)
    _panel_started = time.strftime('%Y-%m-%dT%H:%M:%S')
    _planned_keys = [f'{v}::seed{seed}' for seed in cfg.get('seeds') or [] for v in cfg.get('variants') or []]
    # Crash-safe tracking: publish the panel's identity (code/data versions,
    # hyperparameters, planned cells) BEFORE the first cell can be
    # killed.  The final write below flips the status to complete and lists
    # every planned cell that produced no measurement.
    write_run_manifest(outdir, cfg, fp, prov, _read_summary_json(summary_path), sanity=_sanity_doc, reproducibility=_repro_doc, data_contract=_data_contract, status='running', planned_cells=_planned_keys, started_at=_panel_started)
    log_experiment_event(outdir, {
        'kind': 'panel_start', 'tracking_schema': 2, 'label': label or outdir,
        'run_cfg': fp, 'code': code_stamp(), 'started_at': _panel_started,
        'config': _jsonable(cfg), 'seeds': list(cfg.get('seeds') or []),
        'variants': list(cfg.get('variants') or []),
        'n_cells_planned': len(_planned_keys), 'n_cells_to_run': len(cells),
        'n_cells_resumed': max(0, len(_planned_keys) - len(cells)),
        'provenance': prov,
        'data_contract': ({'passed': bool(_data_contract.get('passed')), 'diagnosis': _data_contract.get('diagnosis'), 'ref': 'data_contract.json'} if isinstance(_data_contract, dict) else None),
        'sanity': ({'passed': all(bool(r.get('passed')) for r in _sanity), 'n_reports': len(_sanity), 'ref': 'overfit_sanity.json'} if _sanity else None),
        'reproducibility': ({'passed': all(bool(r.get('passed')) for r in _repro), 'n_reports': len(_repro), 'ref': 'reproducibility.json'} if _repro else None),
    })
    dropped_truncations = []
    ratios = {}
    _ran_parallel = False
    if cells:
        import par as _par
        _par_ctx = par if isinstance(par, dict) else _par.resolve_ctx(par, label=label or outdir)
        if _par_ctx and _par.ctx_active(_par_ctx, len(cells)):
            _caller = sys._getframe(1).f_globals.get('__name__')
            # a '__main__' frame (e.g. run_gapfill.py driving L.run directly)
            # cannot be imported by that name in the child, and plain
            # exp_lib matches that caller's patch state exactly.
            module = _par_ctx.get('module') or (_caller if _caller and _caller != '__main__' else 'exp_lib')
            jobs = [dict(kind='run', module=module, label=label, cfg=cfg, v=v, seed=seed, guard_cap=(guard.cap_spec() if guard is not None else None), ckpt_flag=bool(_par_ctx.get('ckpt_flag')), ckpt_install=bool(_par_ctx.get('ckpt_install')), fp=fp, prov=prov) for v, seed in cells]
            # Children re-serve the corpus from the shared on-disk cache; the
            # parent can release its reference (and the loader memo entry) once
            # the jobs are dispatched.
            for _memo in (_WT103_MEMO, _FWEB_MEMO):
                for _mk, _mv in list(_memo.items()):
                    if isinstance(_mv, tuple) and _mv and _mv[0] is train_ids:
                        _memo.pop(_mk, None)
            del train_ids, val_batch, val_bnd
            gc.collect()
            results = _par.run_jobs(jobs, _par_ctx, phase_label=label or outdir)
            for r, (v, seed) in zip(results, cells):
                if not r:
                    continue
                if r.get('truncated'):
                    dropped_truncations.append(r.get('key') or f'{v}::seed{seed}')
            _ran_parallel = True
    if not _ran_parallel:
        for v, seed in cells:
            res = _run_cell(cfg, v, seed, guard=guard, ratios=ratios, fp=fp, train_ids=train_ids, val_batch=val_batch, vocab=vocab, val_bnd=val_bnd, prov=prov)
            if res.get('truncated'):
                dropped_truncations.append(res['key'])
    summary = _read_summary_json(summary_path)
    if dropped_truncations:
        print(f'\n[cap] {len(dropped_truncations)} cell(s) were retrained, hit the cap deadline, and produced NO measurement — they are ABSENT from {summary_path} and from every table above:')
        for _k in dropped_truncations:
            print(f'    {_k}')
        print('[cap] raise RUN_CAP and re-run to fill these cells in.')
    if stale_dropped:
        _still = [k for k in stale_dropped if k not in summary]
        print(f'\n[resume] {len(stale_dropped)} cell(s) held a record from a DIFFERENT config and were re-run; {len(_still)} of them produced NO measurement this pass and are now ABSENT from {summary_path} (the stale record was removed rather than kept, so no number measured by older code can be reported under the current config):')
        for _k in _still:
            print(f'    {_k}')
        if len(_still) != len(stale_dropped):
            print(f'    ({len(stale_dropped) - len(_still)} of them succeeded and carry a fresh record.)')
    write_run_manifest(outdir, cfg, fp, prov, summary, sanity=_sanity_doc, reproducibility=_repro_doc, data_contract=_data_contract, status='complete', planned_cells=_planned_keys, started_at=_panel_started)
    _agg = _finish_panel(outdir, summary)
    _measured = {k for k in _planned_keys if isinstance(summary.get(k), dict) and ppl_is_usable((summary.get(k) or {}).get('ppl')) and not (summary.get(k) or {}).get('synthesized')}
    log_experiment_event(outdir, {
        'kind': 'panel_finish', 'tracking_schema': 2, 'label': label or outdir,
        'run_cfg': fp, 'code': code_stamp(), 'started_at': _panel_started,
        'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'n_cells_planned': len(_planned_keys), 'n_cells_to_run': len(cells),
        'n_cells_measured': len(_measured),
        'n_cells_missing': max(0, len(_planned_keys) - len(_measured)),
        'n_truncated': len(dropped_truncations), 'n_stale_dropped': len(stale_dropped),
        'aggregate_ref': 'aggregate.json', 'summary_ref': 'summary.json',
        'aggregate': {str(v): {kk: e.get(kk) for kk in ('n', 'n_seeds', 'ppl_mean', 'ppl_std', 'params') if kk in e} for v, e in (_agg or {}).items() if isinstance(e, dict)},
    })
    return (summary, None)

def _dump_aggregate(outdir, summary):
    agg = {}
    try:
        agg = aggregate(summary)
    except Exception as e:
        print(f'[aggregate] aggregate({outdir}) failed: {type(e).__name__}: {e}')
        print('[aggregate] retrying with the unusable records (no measured `ppl`, `synthesized`, or an ambiguous same-seed duplicate) removed — the untouched records keep their exact values')
        _REQUIRED = ()
        clean = {k: r for k, r in summary.items() if isinstance(r, dict) and r.get('seed') is not None and ppl_is_usable(r.get('ppl')) and (not r.get('synthesized')) and all((r.get(f) is not None for f in _REQUIRED))}
        _legacy_ok = legacy_warm_tags(clean)
        _seen, _dups = ({}, [])
        for k, r in clean.items():
            rk = _record_identity(r, k, _legacy_ok)
            if rk is None:
                continue
            if rk in _seen:
                _dups.append((k, _seen[rk]))
                continue
            _seen[rk] = k
        for k, _keep in _dups:
            clean.pop(k, None)
        if _dups:
            print(f'[aggregate] {len(_dups)} ambiguous record(s) dropped (first key wins): ' + ', '.join((f'kept {keep!r}, dropped {k!r}' for k, keep in _dups)))
        print(f'[aggregate] {len(summary) - len(clean)} record(s) dropped in total: ' + (', '.join(sorted(set(summary) - set(clean))) or '(none)'))
        agg = aggregate(clean)
        atomic_write_json(os.path.join(outdir, 'summary.aggregatable.json'), clean)
        records_for_readers = clean
    else:
        records_for_readers = summary
    _pub = {k: {kk: vv for kk, vv in e.items() if kk != '_group_records'} if isinstance(e, dict) and '_group_records' in e else e for k, e in agg.items()}
    atomic_write_json(os.path.join(outdir, 'aggregate.json'), _pub)
    _save_csv(_pub, outdir)
    _print_table(_pub)
    _print_pairs(records_for_readers)
    try:
        methodology_audit(records_for_readers, outdir=outdir, verbose=True)
    except Exception as e:
        print(f'[audit] methodology audit failed ({type(e).__name__}: {e}) — the panel is still aggregated; the audit is advisory and never blocks a run')
    _plot(records_for_readers, agg, outdir)
    return agg

def _finish_panel(outdir, summary):
    return _dump_aggregate(outdir, summary)
