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
#
# Capture whether the caller actually launched the interpreter with a pinned
# hash seed.  `os.environ.setdefault` below cannot change the CURRENT
# interpreter's hash randomisation, so a run that started without
# PYTHONHASHSEED is still hash-randomised even though every child process we
# spawn sees '0'.  Recording the startup value makes that residual gap visible
# in the provenance block instead of silently assuming it away.
_PYTHONHASHSEED_AT_STARTUP = os.environ.get('PYTHONHASHSEED')
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
import gc, copy, math, os, re, time, json, random, csv, tempfile, itertools, hashlib, collections, threading, numbers
import array
from math import comb
from dataclasses import dataclass
import dataclasses as _dc_mod
import numpy as np
# Leakage-safe split/preprocessing contract (dependency-free NumPy module).
# Re-exported here so every driver can use the same implementation:
#   chronological_split / split_indices / grouped_chronological_split,
#   assert_no_split_leakage, LeakageSafePreprocessor (SafeScaler),
#   prepare_features, data_quality_report, categorical_shift_report.
from data_guard import (
    _hashable_label,
    chronological_split, chronological_split_indices,
    grouped_chronological_split, grouped_chronological_split_indices,
    split_indices, split_and_prepare, split_content_fingerprints,
    train_val_test_split,
    assert_no_split_leakage, LeakageSafePreprocessor, SafeScaler,
    LeakageSafeCategoricalEncoder, LeakageSafeFrequencyEncoder,
    LeakageSafeTargetEncoder, clean_training_rows, prepare_features,
    prepare_frequency_features, prepare_target_features,
    fit_preprocessor, data_quality_report,
    prepare_categorical_features, fit_categorical_encoder,
    categorical_shift_report, duplicate_mask,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[setup] device = {DEVICE}   torch = {torch.__version__}')
QUICK = False
RUN_CAP = dict(total_hours=58.333333333333336, margin=0.93, already_hours=0.0, state_path='run_time_state.json')
CODE_SEMANTICS = 'v11.128'
CKPT_CODE = CODE_SEMANTICS
# NumPy's legacy global seeding API accepts only [0, 2**32 - 1]; a seed
# outside that range either crashes there (negative/too large) or must be
# silently remapped, which would make two different declared seeds share a
# stream.  Every entry point uses this one domain so Python/NumPy/Torch/CUDA/
# DataLoader seeding cannot disagree about what a valid seed is.
MAX_SEED = 2 ** 32 - 1
# ---- 三算子内核路径（M0 工作包；实现与数值契约
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
        # RoPE CSA/HCA layers ALWAYS take the per-row
        # `HybridAttentionRoPE._single_rope` loop (python -m explib.rope forward), and the
        # batched twin in csa_kernels has no RoPE/qk-norm implementation.  A
        # layer is batched only when it is NOT a rope layer.
        and not bool(getattr(c, 'rope', False))
        and getattr(c, 'fuse_mode', 'none') == 'none'
        and getattr(c, 'indexer_mode', 'learned')
        and getattr(c, 'open_block_mode', 'closed') == 'closed'
        and int(getattr(c, 'max_block', 128) or 0) == 128
        and int(getattr(c, 'position_capacity', lambda: 129)()) == 129
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

    Default behaviour is byte-identical to the plain call.  With
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
    vals = list(values) if values is not None else []
    finite = []
    first_bad = None
    last_bad = None
    for i, v in enumerate(vals):
        try:
            fv = float(v)
        except (TypeError, ValueError, OverflowError):
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


def _curve_slope_stats(points, key, k=4):
    """Least-squares slope + its standard error over the last k FINITE points.

    Returning the standard error is what makes the health thresholds
    noise-aware: with four equally spaced eval points and CE noise of order
    0.01-0.03, the slope's standard error is ~5e-5 per step, five times a
    fixed ``slope_tol=1e-5``.  A fixed tolerance therefore lets the
    *sign of the noise* decide ``healthy`` vs ``overfitting``.  Callers that
    need a scale-free criterion compare the slope with a multiple of this SE.
    """
    # Key by step so the tail is chronological even if the record was written
    # out of order, and so a repeated step (e.g. a resumed evaluation) is one
    # evaluation point (the last written value wins) rather than two.
    by_step = {}
    for p in (points if points is not None else []):
        if not isinstance(p, dict):
            continue
        # Use the same evaluation identity as best-step/early-stop selection:
        # Boolean and fractional steps are not two neighboring eval points.
        step = _as_seed_int(p.get('step'))
        if step is None:
            continue
        try:
            y = float(p.get(key))
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(y):
            by_step[float(step)] = y
    pts = sorted(by_step.items())[-max(2, int(k)):]
    if len(pts) < 2:
        return None, None
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    den = sum((x - mx) ** 2 for x, _ in pts)
    if den <= 0:
        return None, None
    slope = sum((x - mx) * (y - my) for x, y in pts) / den
    # Residual standard error of the slope.  With n=2 the residual dof is 0;
    # the slope is then interpolating the two points, so no noise estimate is
    # possible (None -> the caller falls back to the fixed floor).
    dof = n - 2
    if dof <= 0:
        return slope, None
    rss = sum((y - (my + slope * (x - mx))) ** 2 for x, y in pts)
    se = math.sqrt((rss / dof) / den) if den > 0 else None
    return slope, se


def _curve_slope(points, key, k=4):
    """Least-squares slope over the last k FINITE points of a curve."""
    return _curve_slope_stats(points, key, k=k)[0]


def classify_training_health(history, *, tail_k=4, slope_tol=None, gap_tol=None):
    """Classify a train/validation curve.

    ``history`` is a list of dicts with ``step``, ``train_ce`` and either
    ``val_ce`` or ``val_ppl``.  This is a diagnostic, not a model-selection
    rule: it says whether the run looks overfit / underfit / stalled / broken
    on the fixed evaluation grid.

    ``slope_tol=None`` (the default) makes the threshold
    noise-aware: each curve's tolerance is ``max(1e-5, 2 * SE(slope))``, where
    SE is the least-squares standard error of the fitted tail slope.  A slope
    must then exceed twice its own noise before it can flip a health status.
    Pass a number to use the fixed-threshold behaviour (tests
    and callers that deliberately use a synthetic noise-free curve do this).
    """
    entries = [h for h in (history if history is not None else []) if isinstance(h, dict)]
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
            except (TypeError, ValueError, OverflowError):
                n_bad += 1
                continue
            if not math.isfinite(v):
                n_bad += 1
            if key == 'train_ce':
                n_train_measured += 1
    train_s, train_se = _curve_slope_stats(entries, 'train_ce', k=tail_k)
    val_s, val_se = _curve_slope_stats(entries, 'val_ce', k=tail_k)
    if val_s is None:
        _val_from_ppl = []
        for _p in entries:
            try:
                _st = float(_p.get('step'))
                _vp = float(_p.get('val_ppl'))
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(_st) and math.isfinite(_vp) and _vp > 0.0:
                _val_from_ppl.append({'step': _st, 'val_ce': math.log(_vp)})
        val_s, val_se = _curve_slope_stats(_val_from_ppl, 'val_ce', k=tail_k)
    if slope_tol is None:
        # 2-sigma slope noise.  1e-5 is kept as the floor so a perfectly
        # linear synthetic curve still has a usable (tiny) tolerance.
        train_tol = max(1e-5, 2.0 * train_se) if train_se is not None else 1e-5
        val_tol = max(1e-5, 2.0 * val_se) if val_se is not None else 1e-5
    else:
        train_tol = val_tol = float(slope_tol)
    # The docstring promises ``val_ce`` OR ``val_ppl``.  Derive a PPL from CE
    # only when the PPL field is absent/None: an explicitly recorded NaN/inf
    # PPL is a measured failure and must not be papered over by the CE twin.
    def _entry_val_ppl(h):
        raw = h.get('val_ppl')
        if raw is not None:
            return _finite_scalar(raw), 'val_ppl'
        ce = _finite_scalar(h.get('val_ce'))
        if ce is None:
            return None, None
        ppl = _safe_exp_ppl(ce)
        if ppl is None or not math.isfinite(ppl) or ppl <= 0.0:
            return None, None
        return ppl, 'val_ce_derived'

    val_ppl_points = []
    _ppl_sources = set()
    for h in entries:
        # An exact integer step: int(1.5)/int(True) would merge two different
        # evaluations into one step and make best/last/steps-since-best lie.
        st = _as_seed_int(h.get('step'))
        vp, src = _entry_val_ppl(h)
        if st is not None and vp is not None and vp > 0:
            val_ppl_points.append((st, vp, h))
            _ppl_sources.add(src)
    # Same evaluation identity as the slope fit: chronological, one point per
    # step.  Without this, an out-of-order record made `last` (and therefore
    # final_minus_best/steps_since_best) disagree with the fitted tail.
    _by_step = {}
    for st, vp, h in val_ppl_points:
        _by_step[st] = (st, vp, h)
    val_ppl_points = [ _by_step[k] for k in sorted(_by_step) ]
    best = min(val_ppl_points, key=lambda t: (t[1], t[0])) if val_ppl_points else None
    last = val_ppl_points[-1] if val_ppl_points else None
    last_gap = None
    _gap_source = None
    for h in reversed(entries):
        try:
            g = float(h.get('gap'))
        except (TypeError, ValueError, OverflowError):
            g = None
        if g is not None and math.isfinite(g):
            last_gap = g
            _gap_source = 'recorded'
            break
    if last_gap is None:
        # A legacy/synthetic record may carry both curves but no explicit gap.
        # Derive it from the SAME validation quantity the rest of the
        # classification uses instead of reporting no gap evidence.
        for h in reversed(entries):
            _tc = _finite_scalar(h.get('train_ce'))
            _vc = _finite_scalar(h.get('val_ce'))
            if _vc is None:
                _vp, _src = _entry_val_ppl(h)
                _vc = math.log(_vp) if (_vp is not None and _vp > 0.0) else None
            if _tc is not None and _vc is not None:
                last_gap = _vc - _tc
                _gap_source = 'derived'
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
        if val_s > val_tol:
            status = 'validation_degrading'
            reason = 'train CE unmonitored; validation CE is rising in the tail (val-only classification)'
        elif val_s < -val_tol:
            status = 'healthy'
            reason = 'train CE unmonitored; validation CE is falling in the tail (val-only classification: overfitting cannot be ruled out without the train curve)'
        else:
            status = 'underfitting_or_stalled'
            reason = 'train CE unmonitored; validation CE is flat in the tail (val-only classification)'
    elif train_s is None or val_s is None:
        status = 'insufficient_data'
        reason = 'fewer than two finite train/validation eval points; cannot classify'
    elif train_s < -train_tol and val_s > val_tol:
        status = 'overfitting'
        reason = 'training CE is still falling while validation CE is rising'
    elif train_s >= -train_tol and val_s >= -val_tol:
        status = 'underfitting_or_stalled'
        reason = 'neither training nor validation CE is decreasing in the tail'
    elif train_s < -train_tol and val_s <= val_tol:
        status = 'healthy'
        reason = 'training CE is falling and validation CE is not rising'
    elif val_s < -val_tol:
        # Validation is IMPROVING while the fixed train-CE measurement is
        # flat or rising.  Calling this `validation_degrading` (the old else
        # branch) inverted the observed direction and sent the reader to the
        # wrong action.
        status = 'validation_improving_train_flat'
        reason = ('validation CE is falling while training CE is flat or rising; '
                  'audit the fixed train-eval batch/data order before interpreting this as a model effect')
    else:
        status = 'validation_degrading'
        reason = 'validation CE is rising in the tail'
    # The gap curve is what turns "both curves move" into direct overfitting
    # evidence.  Derive it from the two CE curves when a record predates the
    # explicit `gap` field, instead of reporting no gap trend at all.
    _gap_points = []
    _n_gap_rec = _n_gap_der = 0
    for h in entries:
        _raw_g = h.get('gap')
        _g = _finite_scalar(_raw_g) if _raw_g is not None else None
        if _g is not None:
            _gap_points.append({'step': h.get('step'), 'gap': _g})
            _n_gap_rec += 1
            continue
        _tc = _finite_scalar(h.get('train_ce'))
        _vc = _finite_scalar(h.get('val_ce'))
        if _vc is None:
            _vp, _src = _entry_val_ppl(h)
            _vc = math.log(_vp) if (_vp is not None and _vp > 0.0) else None
        if _tc is not None and _vc is not None:
            _gap_points.append({'step': h.get('step'), 'gap': _vc - _tc})
            _n_gap_der += 1
    if _n_gap_rec and _n_gap_der:
        _gap_slope_source = 'mixed'
    elif _n_gap_rec:
        _gap_slope_source = 'recorded'
    elif _n_gap_der:
        _gap_slope_source = 'derived'
    else:
        _gap_slope_source = None
    gap_s, gap_se = _curve_slope_stats(_gap_points, 'gap', k=tail_k)
    best_step = int(best[0]) if best is not None else None
    last_step = int(last[0]) if last is not None else None
    final_minus_best = (float(last[1] - best[1])
                        if (best is not None and last is not None) else None)
    steps_since_best = ((last_step - best_step)
                        if (best_step is not None and last_step is not None) else None)
    actions = []
    if status == 'overfitting':
        actions.append('add or strengthen a wired regularizer (weight decay; the model has no dropout layer) and/or increase unique training data')
        actions.append('use the best-validation checkpoint or enable early stopping (early_stop_patience) at best_step instead of reporting only the final step')
    elif status == 'underfitting_or_stalled':
        actions.append('verify the data pipeline (target shift, tokenizer/vocab mapping) before changing the model')
        actions.append('check the learning-rate schedule and warmup; try a different LR before increasing capacity')
        actions.append('if both curves are flat but low, train longer or increase capacity')
    elif status == 'validation_degrading':
        actions.append('inspect the first rising validation point and the accompanying train CE; if train CE is also flat, treat it as an optimization/data issue rather than overfitting')
    elif status == 'validation_improving_train_flat':
        actions.append('do not read this as overfitting: validation is improving; verify the fixed train-eval batch, label shift and data order before changing regularization')
        actions.append('if the train curve is genuinely flat while validation falls, check whether train CE is measured under augmentation/noise or on a stale fixed batch')
    elif status == 'nonfinite':
        actions.append('follow the NaN triage path: data NaN/Inf -> log(0)/div0/softmax -> fp16 overflow -> gradient explosion/LR')
    # A two-point curve has no residual degrees of freedom: the slope is an
    # interpolation and its SE is undefined.  Keep the label (it is what the
    # recorded curve shows) but make the resolution explicit so a two-eval
    # smoke run is not read as a mechanism-level overfitting result.
    _trend_statuses = {'overfitting', 'underfitting_or_stalled',
                       'validation_degrading', 'validation_improving_train_flat'}
    trend_reliable = len(val_ppl_points) >= 3
    if status in _trend_statuses and not trend_reliable:
        actions.append(f'only {len(val_ppl_points)} validation evaluation point(s) support this trend; '
                       'the tail slope is interpolated and cannot separate a trend from noise — '
                       'add evaluation points before changing the model or the data budget')
    train_rising = bool(train_s is not None and math.isfinite(train_s) and train_s > train_tol)
    if train_rising:
        actions.append('training CE itself is rising in the tail; lower the LR and check the gradient/AMP diagnostics before treating this as underfitting')
    if (final_minus_best is not None and final_minus_best > 0.0
            and steps_since_best is not None and steps_since_best > 0):
        actions.append(f'final PPL is above the best observed PPL by {final_minus_best:.4g}; the curve, not just the final value, must be reported')
    if (gap_s is not None and math.isfinite(gap_s) and gap_s > 0.0
            and gap_se is not None and gap_s > 2.0 * gap_se):
        actions.append('the train-validation gap is widening in the tail; this is direct overfitting evidence even if the validation slope is still small')
    return {
        'status': status,
        'reason': reason,
        'actions': actions,
        'n_points': len(entries),
        'n_eval_points': len(val_ppl_points),
        'n_nonfinite': int(n_bad),
        'n_train_ce_measured': int(n_train_measured),
        'val_ppl_source': (sorted(_ppl_sources)[0] if len(_ppl_sources) == 1
                           else ('mixed' if _ppl_sources else None)),
        'trend_reliable': bool(trend_reliable),
        'tail_k': int(tail_k),
        'train_ce_slope_per_step': train_s,
        'val_ce_slope_per_step': val_s,
        'train_ce_slope_se': train_se,
        'val_ce_slope_se': val_se,
        'train_slope_tol': train_tol,
        'val_slope_tol': val_tol,
        'slope_tol_mode': ('adaptive_2se' if slope_tol is None else 'fixed'),
        'gap_last': last_gap,
        'gap_source': _gap_source,
        'gap_slope_per_step': gap_s,
        'gap_slope_se': gap_se,
        'gap_slope_source': _gap_slope_source,
        'best_step': best_step,
        'best_val_ppl': float(best[1]) if best is not None else None,
        'final_val_ppl': float(last[1]) if last is not None else None,
        'final_minus_best': final_minus_best,
        'steps_since_best': steps_since_best,
        'train_ce_rising': train_rising,
    }


def classify_train_loss_curve(losses, *, min_points=8, tail_k=32,
                              spike_sigma=6.0, min_rel_change=0.01):
    """Classify the per-step training-loss curve without a validation grid.

    This is the fallback used when a panel has ``eval_every=0`` (or the
    validation batch is unavailable): it cannot diagnose overfitting, which
    needs held-out points, but it can still expose the failures a single final
    loss hides — a non-finite step, a rising tail (divergence / too-high LR),
    isolated finite spikes, and a plateau.  ``status`` is one of
    ``nonfinite``, ``insufficient_data``, ``diverging``,
    ``unstable_spikes``, ``stalled`` or ``descending``.
    """
    try:
        vals = list(losses) if losses is not None else []
    except TypeError:
        vals = []
    n_bad = 0
    first_bad = None
    finite = []
    for i, v in enumerate(vals):
        try:
            fv = float(v)
        except (TypeError, ValueError, OverflowError):
            fv = None
        if fv is not None and math.isfinite(fv):
            finite.append((i, fv))
        else:
            n_bad += 1
            if first_bad is None:
                first_bad = i
    _min = _exact_count(min_points, minimum=2)
    _tail = _exact_count(tail_k, minimum=2)
    _sig = _finite_scalar(spike_sigma)
    _rel = _finite_scalar(min_rel_change)
    if _min is None or _tail is None:
        raise ValueError('classify_train_loss_curve needs integer min_points>=2 and tail_k>=2')
    if _sig is None or _sig <= 0.0 or _rel is None or _rel < 0.0:
        raise ValueError('classify_train_loss_curve needs spike_sigma>0 and min_rel_change>=0')
    min_points, tail_k, spike_sigma, min_rel_change = int(_min), int(_tail), _sig, _rel
    n = len(vals)
    out = {
        'kind': 'train_loss_curve_health', 'mode': 'train_only',
        'n_points': n, 'n_nonfinite': int(n_bad),
        'first_nonfinite_index': first_bad,
        'n_finite': len(finite),
        'min': (min(v for _, v in finite) if finite else None),
        'max': (max(v for _, v in finite) if finite else None),
        'last': (finite[-1][1] if finite else None),
        'n_spikes': 0, 'first_spike_index': None,
        'tail_start_index': None, 'tail_slope_per_step': None,
        'tail_slope_se': None, 'tail_relative_change': None,
        'status': None, 'reason': None, 'actions': [],
        'note': ('training-side health only; overfitting requires held-out validation points'),
    }
    if n_bad:
        out['status'] = 'nonfinite'
        out['reason'] = f'{n_bad} non-finite training-loss point(s); first at step index {first_bad}'
        out['actions'].append('follow the NaN triage path: data NaN/Inf -> log(0)/div0/softmax -> fp16 overflow -> gradient explosion/LR')
    if len(finite) < min_points:
        if out['status'] is None:
            out['status'] = 'insufficient_data'
            out['reason'] = f'only {len(finite)} finite training-loss point(s); at least {min_points} are needed'
            out['actions'].append('record more training-loss points (lower log_every or raise the step budget) before reading the trend')
        return out
    window = max(tail_k, int(math.ceil(n * 0.1)))
    start = max(0, n - window)
    tail = [(i, v) for i, v in finite if i >= start]
    if len(tail) < min_points:
        tail = finite[-min_points:]
        start = tail[0][0]
    xs = [float(i) for i, _ in tail]
    ys = [v for _, v in tail]
    k = len(tail)
    mx = sum(xs) / k
    my = sum(ys) / k
    den = sum((x - mx) ** 2 for x in xs)
    slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den) if den > 0 else None
    se = None
    if slope is not None and k > 2:
        rss = sum((y - (my + slope * (x - mx))) ** 2 for x, y in zip(xs, ys))
        se = math.sqrt((rss / (k - 2)) / den) if den > 0 else None
    span = xs[-1] - xs[0]
    med = float(np.median(ys))
    rel = ((slope * span) / max(abs(med), 1e-12)) if slope is not None else None
    tol = max(1e-12, 2.0 * se) if se is not None else 1e-12
    deviations = [abs(v - med) for v in ys]
    mad = float(np.median(deviations))
    sigma = 1.4826 * mad
    spike_cut = max(spike_sigma * sigma, 0.5 * abs(med), 1e-12)
    spikes = [i for i, v in tail if v > med + spike_cut]
    out.update({
        'tail_start_index': int(start), 'tail_n': k,
        'tail_slope_per_step': slope, 'tail_slope_se': se,
        'tail_slope_tol': tol, 'tail_relative_change': rel,
        'tail_median': med, 'tail_robust_sigma': sigma,
        'n_spikes': len(spikes),
        'first_spike_index': (int(spikes[0]) if spikes else None),
    })
    if out['status'] is not None:
        return out
    if slope is not None and slope > tol and rel is not None and rel > min_rel_change:
        out['status'] = 'diverging'
        out['reason'] = (f'training loss is rising in the tail (slope={slope:.4g}/step, '
                         f'relative change={rel:+.3%})')
        out['actions'].append('lower the LR, extend warmup and inspect gradient/AMP diagnostics before adding capacity')
    elif spikes:
        out['status'] = 'unstable_spikes'
        out['reason'] = (f'{len(spikes)} finite training-loss spike(s) in the tail; '
                         f'first at step index {spikes[0]}')
        out['actions'].append('inspect the spike step for a corrupted batch, gradient clipping and the LR schedule')
    elif slope is not None and slope < -tol and rel is not None and rel < -min_rel_change:
        out['status'] = 'descending'
        out['reason'] = (f'training loss is still falling in the tail (slope={slope:.4g}/step, '
                         f'relative change={rel:+.3%})')
    else:
        out['status'] = 'stalled'
        out['reason'] = (f'training loss is flat in the tail (slope={slope!r}/step, '
                         f'relative change={rel!r})')
        out['actions'].append('verify the data pipeline and learning-rate schedule; if the loss plateau is high, train longer or change capacity')
    return out


def early_stop_decision(history, *, patience, min_delta=0.0):
    """Return whether a validation curve has stopped improving.

    ``patience`` is the number of consecutive finite validation points that may
    fail to improve on the best PPL before stopping.  ``min_delta`` is an
    absolute PPL margin: a point must be at least ``min_delta`` below the best
    to count as an improvement.  As in ``classify_training_health``, a history
    may carry ``val_ce`` instead of ``val_ppl``; the PPL is then derived from
    the finite CE.  This is a pure decision helper so the policy can be tested
    without training; the caller decides whether unequal token budgets are
    acceptable for its comparison.
    """
    if isinstance(patience, (bool, np.bool_)) or not isinstance(patience, (int, np.integer)):
        raise ValueError(f'early-stop patience must be a positive integer, got {patience!r}')
    patience = int(patience)
    if patience < 1:
        raise ValueError(f'early-stop patience must be >= 1, got {patience}')
    if isinstance(min_delta, (bool, np.bool_)):
        raise ValueError(f'early-stop min_delta must be a finite non-negative number, got {min_delta!r}')
    try:
        min_delta = float(min_delta)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'early-stop min_delta must be a finite non-negative number, got {min_delta!r}') from None
    if not math.isfinite(min_delta) or min_delta < 0.0:
        raise ValueError(f'early-stop min_delta must be a finite non-negative number, got {min_delta!r}')
    pts = []
    _sources = set()
    for p in (history if history is not None else []):
        if not isinstance(p, dict):
            continue
        # Steps are exact integers and Boolean/NaN PPLs are flags, not
        # measurements; truncating 1.5 to step 1 would merge two eval points.
        step = _as_seed_int(p.get('step'))
        raw = p.get('val_ppl')
        if raw is not None:
            ppl = _finite_scalar(raw)
            src = 'val_ppl'
        else:
            ce = _finite_scalar(p.get('val_ce'))
            ppl = _safe_exp_ppl(ce) if ce is not None else None
            src = 'val_ce_derived'
            if ppl is None or not math.isfinite(ppl):
                ppl = None
        if step is not None and ppl is not None and ppl > 0.0:
            pts.append((step, ppl))
            _sources.add(src)
    # One point per evaluation step, ordered chronologically (a duplicate
    # step from a resumed write is not two independent stale evaluations).
    _by_step = {}
    for step, ppl in pts:
        _by_step[step] = (step, ppl)
    pts = [_by_step[k] for k in sorted(_by_step)]
    best = None
    best_step = None
    stale = 0
    for step, ppl in pts:
        # `min_delta=0` keeps the historical strict-improvement contract
        # (equal PPL is a stale point).  For a positive margin the docstring's
        # "at least min_delta below" is inclusive: a point exactly on the
        # boundary is an improvement, not a stale evaluation.
        _threshold = None if best is None else best - min_delta
        _improved = (best is None or ppl < _threshold
                     or (min_delta > 0.0 and ppl == _threshold))
        if _improved:
            best = ppl
            best_step = step
            stale = 0
        else:
            stale += 1
    return {'should_stop': bool(pts and stale >= patience),
            'patience': patience, 'min_delta': min_delta,
            'n_points': len(pts), 'stale_evals': int(stale),
            'best_val_ppl': (float(best) if best is not None else None),
            'best_step': (int(best_step) if best_step is not None else None),
            'last_step': (int(pts[-1][0]) if pts else None),
            'last_val_ppl': (float(pts[-1][1]) if pts else None),
            'val_ppl_source': (sorted(_sources)[0] if len(_sources) == 1
                               else ('mixed' if _sources else None))}


def diagnose_training_instability(record, *, data_checked=None):
    """Triage a training record along the standard non-finite path.

    Order matters: data first, then the forward/loss and attention-pooling
    numerical guards, then fp16 gradient-scaler overflow, then gradient
    explosion / learning-rate.  Every check is reported as ``observed``,
    ``not_observed`` or ``unknown`` so an unmeasured cause is never presented
    as exonerated.
    """
    rec = record if isinstance(record, dict) else {}
    diag = rec.get('opt_diag_summary')
    if not isinstance(diag, dict):
        diag = rec
    events = [e for e in (diag.get('events') or []) if isinstance(e, dict)]
    malformed = []

    def _count_field(source, key):
        """Exact non-negative integer count, or flag a malformed field.

        Training-record fields come from JSON/checkpoints and may be edited or
        corrupted.  ``int("x")``/``int(nan)`` would crash the whole report; an
        invalid value must also never be silently read as zero, which would
        present a broken record as clean.
        """
        raw = source.get(key) if isinstance(source, dict) else None
        if raw is None:
            return 0
        if isinstance(raw, (bool, np.bool_)):
            malformed.append(f'{key}={raw!r}')
            return 0
        try:
            f = float(raw)
        except (TypeError, ValueError, OverflowError):
            malformed.append(f'{key}={raw!r}')
            return 0
        if not (math.isfinite(f) and f >= 0.0 and f == math.floor(f)):
            malformed.append(f'{key}={raw!r}')
            return 0
        return int(f)

    n_loss = _count_field(diag, 'nonfinite_loss_steps')
    n_grad = _count_field(diag, 'nonfinite_grad_steps')
    first_bad = diag.get('first_bad_step')
    _guard_raw = diag.get('nan_guard')
    if _guard_raw is not None and not isinstance(_guard_raw, dict):
        malformed.append(f'nan_guard={_guard_raw!r}')
    guard = _guard_raw if isinstance(_guard_raw, dict) else {}
    guard_cleaned = _count_field(guard, 'elements_cleaned')
    guard_first = guard.get('first') if isinstance(guard.get('first'), dict) else None
    overflows = _count_field(diag, 'amp_scaler_overflows')
    _scale_raw = diag.get('amp_scaler_scale_min')
    scale_min = _finite_scalar(_scale_raw)
    if _scale_raw is not None and scale_min is None:
        malformed.append(f'amp_scaler_scale_min={_scale_raw!r}')
    knobs = rec.get('train_knobs') if isinstance(rec.get('train_knobs'), dict) else {}
    _health_raw = rec.get('curve_health')
    _hist_raw = rec.get('health_history')
    health = _health_raw if isinstance(_health_raw, dict) else {}
    _health_source = 'record' if isinstance(_health_raw, dict) else 'missing'
    if _health_raw is not None and not isinstance(_health_raw, dict):
        malformed.append(f'curve_health={_health_raw!r}')
    if not health and isinstance(_hist_raw, list) and _hist_raw:
        # A legacy/edited record may carry the curve without the cached
        # summary.  Recompute it: an unsummarised non-finite point is still
        # evidence and must not be reported as a clean run.
        try:
            health = classify_training_health(_hist_raw)
            _health_source = 'derived_from_history'
        except Exception as _he:
            malformed.append(f'health_history could not be classified ({type(_he).__name__}: {_he})')
            health = {}
    if _hist_raw is not None and not isinstance(_hist_raw, list):
        malformed.append(f'health_history is not a list ({type(_hist_raw).__name__})')
    _train_raw = rec.get('train_curve_health')
    if _train_raw is not None and not isinstance(_train_raw, dict):
        malformed.append(f'train_curve_health={_train_raw!r}')
    train_health = _train_raw if isinstance(_train_raw, dict) else {}
    _train_source = 'record' if isinstance(_train_raw, dict) else 'missing'
    if not train_health:
        _losses_raw = rec.get('losses')
        if isinstance(_losses_raw, list) and _losses_raw:
            try:
                train_health = classify_train_loss_curve(_losses_raw)
                _train_source = 'derived_from_losses'
            except Exception as _tce:
                malformed.append(f'losses could not be classified ({type(_tce).__name__}: {_tce})')
        elif _losses_raw is not None and not isinstance(_losses_raw, list):
            malformed.append(f'losses is not a list ({type(_losses_raw).__name__})')
    health_nonfinite = str(health.get('status') or '') == 'nonfinite'
    # If both representations exist, the curve wins on a numerical failure:
    # a cached `healthy` summary must not hide a non-finite point in its own
    # history.
    if (_health_source == 'record' and isinstance(_hist_raw, list) and _hist_raw
            and not health_nonfinite):
        try:
            _hist_health = classify_training_health(_hist_raw)
        except Exception as _hhe:
            malformed.append(f'health_history could not be classified ({type(_hhe).__name__}: {_hhe})')
        else:
            if str(_hist_health.get('status') or '') == 'nonfinite':
                malformed.append('curve_health claims a finite curve but health_history '
                                 f'contains non-finite point(s) ({_hist_health.get("reason")})')
                health_nonfinite = True
    amp = rec.get('amp_dtype') or knobs.get('amp_dtype')
    _grad_raw = diag.get('grad_norm_max')
    grad_max = _grad_raw
    try:
        if isinstance(grad_max, (bool, np.bool_)):
            raise TypeError
        grad_max = float(grad_max)
    except (TypeError, ValueError, OverflowError):
        grad_max = None
        if _grad_raw is not None:
            malformed.append(f'grad_norm_max={_grad_raw!r}')
    lr = None
    for e in reversed(events):
        try:
            lr = float(e.get('lr'))
        except (TypeError, ValueError, OverflowError):
            continue
        break
    if lr is None:
        lr = rec.get('lr')
    checks = []

    def add(name, status, evidence, action):
        checks.append({'check': name, 'status': status, 'evidence': evidence, 'action': action})

    _audit_raw = rec.get('data_audit')
    if _audit_raw is not None and not isinstance(_audit_raw, dict):
        malformed.append(f'data_audit={_audit_raw!r}')
        _audit_raw = None
    _data_bad = False
    _label_shift_bad = False
    if isinstance(_audit_raw, dict):
        _sections = _audit_raw.get('sections') if isinstance(_audit_raw.get('sections'), dict) else {}
        _tr = _sections.get('train') if isinstance(_sections.get('train'), dict) else {}
        _va = _sections.get('validation') if isinstance(_sections.get('validation'), dict) else {}

        def _audit_count(sec, key):
            try:
                return int(sec.get(key) or 0)
            except (TypeError, ValueError, OverflowError):
                return 0
        _audit_status = str(_audit_raw.get('status') or '')
        _audit_bad = (_audit_status != 'ok'
                      or _audit_count(_tr, 'n_nonfinite') or _audit_count(_tr, 'n_out_of_range')
                      or _audit_count(_va, 'n_nonfinite') or _audit_count(_va, 'n_out_of_range'))
        _audit_bad = bool(_audit_bad)
        if _audit_bad:
            _data_bad = True
            add('data_nan_inf', 'observed',
                f'data_audit status={_audit_status!r}; train nonfinite/out-of-range='
                f'{_audit_count(_tr, "n_nonfinite")}/{_audit_count(_tr, "n_out_of_range")}, '
                f'validation={_audit_count(_va, "n_nonfinite")}/{_audit_count(_va, "n_out_of_range")}',
                'fix/replace the token cache or feature matrix before interpreting any training number')
        else:
            add('data_nan_inf', 'not_observed',
                f'data_audit checked {_audit_count(_tr, "n_checked"):,} training token(s) '
                f'({"sampled" if _tr.get("sampled") else "full"}) and '
                f'{_audit_count(_va, "n_checked"):,} validation token(s) (full); '
                'dtype/range/finiteness OK',
                'if a future run fails here, validate the token cache dtype/range and the label shift before touching the model')
        _ls = _audit_raw.get('label_shift')
        if isinstance(_ls, dict) and _ls.get('checked') is not None:
            if bool(_ls.get('checked')) and not bool(_ls.get('ok')):
                _label_shift_bad = True
                add('data_label_shift', 'observed',
                    f'label-shift check failed on {_ls.get("n_tokens")} token(s)',
                    'the loader is not producing (x, x_shifted) pairs; fix the target construction before training')
            else:
                add('data_label_shift', 'not_observed',
                    f'causal one-token target shift verified on {_ls.get("n_tokens")} token(s)',
                    'no action')
        else:
            add('data_label_shift', 'unknown',
                'no label-shift evidence in this record',
                'check the loader/target shift before treating a stalled curve as an optimisation problem')
    elif data_checked:
        add('data_nan_inf', 'not_observed',
            f'token batches validated at the configured cadence ({data_checked})',
            'if a future run fails here, validate the token cache dtype/range and the label shift before touching the model')
        add('data_label_shift', 'unknown',
            'legacy record has no explicit label-shift check',
            'check the loader/target shift before treating a stalled curve as an optimisation problem')
    else:
        add('data_nan_inf', 'unknown', 'no data-check evidence in this record',
            'validate token ids/float cache for NaN/Inf and the target shift')
        add('data_label_shift', 'unknown',
            'no label-shift evidence in this record',
            'check the loader/target shift before treating a stalled curve as an optimisation problem')

    bad_loss = [e for e in events if e.get('kind') == 'loss']
    bad_grad = [e for e in events if e.get('kind') == 'grad']
    if bad_loss:
        ce_bad = any(isinstance(e.get('ce'), (int, float)) and not math.isfinite(float(e.get('ce'))) for e in bad_loss)
        comp_bad = any(isinstance(e.get('comp_reg'), (int, float)) and not math.isfinite(float(e.get('comp_reg'))) for e in bad_loss)
        pg_bad = any(isinstance(e.get('boundary_pg'), (int, float)) and not math.isfinite(float(e.get('boundary_pg'))) for e in bad_loss)
        where = 'CE/logits path' if ce_bad else ('compression regularizer' if comp_bad else ('boundary policy term' if pg_bad else 'combined loss'))
        add('forward_loss_numerics', 'observed',
            f'{len(bad_loss)} non-finite loss step(s); first suspect: {where}',
            'check log(0)/division, log_softmax/softmax overflows, and the indexer/block-pool denominators at the first bad step')
    else:
        add('forward_loss_numerics', 'not_observed' if not n_loss else 'unknown',
            f'nonfinite_loss_steps={n_loss}', 'inspect the first non-finite loss step')

    if guard_cleaned:
        add('attention_pool_nan_to_num', 'observed',
            f'{guard_cleaned} element(s) cleaned, first at {guard_first}',
            'CSA_STRICT_NAN=1 reproduces the first non-finite attention/pooling site before it is silently replaced')
    else:
        add('attention_pool_nan_to_num', 'not_observed', 'no attention/pooling value was replaced', 'no action')

    if str(amp).lower() == 'fp16' and (overflows > 0 or (scale_min is not None and scale_min < 1.0)):
        add('fp16_overflow', 'observed',
            f'grad-scaler overflows={overflows}, min scale={scale_min}',
            'switch to bf16 or lower the LR/initial loss scale; confirm the first overflow step from opt_diag.events')
    elif amp:
        add('fp16_overflow', 'not_observed' if str(amp).lower() != 'fp16' else 'unknown',
            f'amp_dtype={amp}, overflows={overflows}', 'prefer bf16 when fp16 overflows repeatedly')
    else:
        add('fp16_overflow', 'not_observed', 'no mixed-precision autocast in this run', 'no action')

    # A non-finite recorded maximum is itself evidence; treating only finite
    # values as "huge" could report `not_observed` next to an inf/NaN grad max.
    huge_grad = (grad_max is not None
                 and ((not math.isfinite(grad_max))
                      or grad_max > 10.0 * GRAD_CLIP_NORM))
    if bad_grad or huge_grad:
        add('gradient_explosion_lr', 'observed',
            f'non-finite grad steps={n_grad}, max pre-clip grad={grad_max}, lr={lr}',
            'lower the LR and/or strengthen warmup/clipping; verify the loss scale before blaming the data')
    else:
        add('gradient_explosion_lr', 'not_observed',
            f'non-finite grad steps={n_grad}, max pre-clip grad={grad_max}', 'no action')

    _train_status = str(train_health.get('status') or '')
    _train_bad = _train_status in ('nonfinite', 'diverging', 'unstable_spikes')
    if _train_bad:
        add('train_loss_curve', 'observed',
            f'train curve status={_train_status} ({train_health.get("reason") or "no reason recorded"}; '
            f'source={_train_source})',
            'inspect the first bad training step: a corrupted batch, an LR/warmup fault or a gradient/AMP overflow precedes most divergences')
    elif _train_source == 'missing':
        add('train_loss_curve', 'unknown',
            'no per-step training-loss curve or train_curve_health was recorded',
            'record the training-loss curve (losses) even when eval_every=0; the final loss alone hides divergence and spikes')
    else:
        add('train_loss_curve', 'not_observed',
            f'train curve status={_train_status or "unknown"} (source={_train_source})',
            'no action')

    if health_nonfinite:
        add('validation_curve_numerics', 'observed',
            f'curve_health is nonfinite ({health.get("reason") or "no reason recorded"}; source={_health_source})',
            'inspect the first non-finite validation point; a training-side loss can be finite while eval_ppl/its log or the per-document rows overflow')
    elif _health_source == 'missing':
        add('validation_curve_numerics', 'unknown',
            'no curve_health or health_history was recorded',
            'record the validation curve (or at least its non-finite points) before treating this run as numerically clean')
    else:
        add('validation_curve_numerics', 'not_observed',
            f'no non-finite validation-curve point was recorded (source={_health_source})',
            'no action')

    if malformed:
        add('record_integrity', 'observed',
            'malformed/ambiguous training-diagnostic field(s): ' + ', '.join(malformed[:8]),
            'repair the record before treating its zero counts or health as evidence; '
            'an invalid count is unknown, not zero')

    # A finite but far-above-clip gradient is an observed instability even
    # when no step ever produced NaN/Inf.  Omitting it here let the function
    # report the run as `clean` while its own gradient_explosion_lr check said
    # `observed` — exactly the contradictory record the triage exists to
    # prevent.
    _has_event = bool(n_loss or n_grad or guard_cleaned or bad_loss or bad_grad
                      or huge_grad or health_nonfinite or _data_bad
                      or _label_shift_bad or _train_bad)
    if not _has_event and not malformed:
        status = 'clean'
        primary = None
        summary = 'no non-finite loss/gradient or guarded attention value was recorded'
    elif not _has_event:
        # A malformed count field could hide an event.  Report the record as
        # unverifiable (not clean) instead of reading the bad value as zero.
        status = 'unverifiable'
        primary = 'record_integrity'
        summary = ('training-diagnostic record is malformed; its event counts '
                   'cannot be read as zero')
    else:
        status = 'instability_observed'
        observed = [c for c in checks if c['status'] == 'observed']
        primary = (observed[0]['check'] if observed else 'unclassified')
        summary = ('first observed suspect: ' + primary) if observed else 'non-finite values were recorded but none of the standard causes was observed'
    return {'status': status, 'summary': summary, 'primary_suspect': primary,
            'first_bad_step': first_bad, 'curve_health_source': _health_source,
            'checks': checks,
            'recommended_order': ['data_nan_inf', 'data_label_shift',
                                  'forward_loss_numerics',
                                  'attention_pool_nan_to_num', 'validation_curve_numerics',
                                  'fp16_overflow', 'gradient_explosion_lr',
                                  'train_loss_curve']}


def _safe_exp_ppl(x):
    """``exp`` for a PPL accumulator without raising on divergent CE.

    ``math.exp`` raises once the exponent exceeds ~709.  A training run that
    has diverged but still produced finite CE must be recorded as ``inf`` and
    diagnosed, not crash the eval callback.  The clamp mirrors the historical
    ``eval_ppl`` overflow behaviour while making the CI bounds consistent."""
    try:
        x = float(x)
    except (TypeError, ValueError, OverflowError):
        return None
    if math.isnan(x):
        return float('nan')
    # Let the FP overflow exception define the upper boundary for this
    # machine, so any finite exponent remains a finite PPL.
    try:
        return math.exp(x)
    except OverflowError:
        return float('inf')


def _finite_scalar(v):
    """Finite float for a real numeric scalar, including NumPy scalars.

    Numeric helpers are routinely called with ``np.int64``/``np.float32``
    values from arrays; an ``isinstance(v, (int, float))`` filter would drop
    those and turn a valid panel into "no data".  Booleans and
    complex/non-numeric objects are not measurements.
    """
    if (isinstance(v, (bool, np.bool_))
            or isinstance(v, (complex, np.complexfloating))
            or not isinstance(v, numbers.Number)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        return None
    return f if math.isfinite(f) else None


def _contains_bool_value(x):
    """True when a scalar/sequence/array carries a boolean element.

    Booleans are flags, not measurements.  A mixed list such as
    ``[True, 2.0, 3.0]`` is coerced by ``np.asarray`` to a float array, so a
    dtype-only check would miss it; this recursive scan catches it before the
    numeric conversion on the small per-seed/per-document inputs these
    statistics helpers receive.
    """
    if isinstance(x, (bool, np.bool_)):
        return True
    if isinstance(x, np.ndarray):
        if x.dtype.kind == 'b':
            return True
        if x.dtype.kind == 'O':
            return any(isinstance(v, (bool, np.bool_))
                       for v in x.reshape(-1))
        return False
    if isinstance(x, (list, tuple)):
        return any(_contains_bool_value(v) for v in x)
    return False


def _scaled_sum_of_squares(values, multiplier):
    """``multiplier * sum(v**2)`` without overflowing representable results.

    Cluster-robust/jackknife variances square per-unit influences.  A finite
    influence can still be large enough that the naive square overflows to
    inf even though the final variance (and SE) is representable.  Scale by
    the largest deviation, compute the unitless sum, then combine in log
    space.  Returns None when the true result does not fit a float, so callers
    can mark the interval unavailable instead of emitting inf.
    """
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    scale = max((abs(v) for v in vals), default=0.0)
    if not math.isfinite(scale):
        return None
    if scale == 0.0:
        return 0.0
    try:
        ss = math.fsum((v / scale) ** 2 for v in vals)
    except (OverflowError, ValueError):
        return None
    if not (math.isfinite(ss) and ss > 0.0):
        return None
    try:
        logval = (math.log(float(multiplier)) + 2.0 * math.log(scale)
                  + math.log(ss))
    except (ValueError, OverflowError):
        return None
    if not math.isfinite(logval) or logval > math.log(sys.float_info.max):
        return None
    val = math.exp(logval)
    return val if math.isfinite(val) else None


def _scaled_mean_std_if_extreme(values, ddof=1):
    """Return ``(mean, std)`` for representable-but-extreme magnitudes.

    ``numpy.mean``/``numpy.std`` square deviations.  Finite values below
    ``sqrt(tiny)`` underflow those squares to zero, while finite values above
    ``sqrt(max_float)`` overflow them to inf, so a genuinely variable panel can
    report ``std=0`` (or ``inf``) and fabricate a zero-width interval or an
    infinite effect size.  Scale by the largest magnitude, compute a unitless
    mean/std, then rescale.

    Returns ``None`` when the ordinary NumPy path is safe, or when the input
    cannot be scaled.  The 1e-100/1e100 cutoffs are the same convention used by
    the exact sign-flip helpers.
    """
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return None
    scale = max(abs(v) for v in vals)
    if (not math.isfinite(scale) or scale == 0.0
            or 1e-100 <= scale <= 1e100):
        return None
    z = [v / scale for v in vals]
    mean_z = math.fsum(z) / n
    std = 0.0
    if n > ddof:
        ss = math.fsum((v - mean_z) ** 2 for v in z)
        std = scale * math.sqrt(ss / (n - ddof))
    return float(scale * mean_z), float(std)


def _nll_doc_ci(nll_per_row, n_tokens_per_row):
    """Approximate 95% CI for PPL from per-DOCUMENT NLL (not per-token).

    Documents, not tokens, are the independent sampling unit; treating every
    token as independent would make the interval look far narrower than the
    actual evaluation-subset noise."""
    # Materialise once: ``x or []`` raises for a multi-element NumPy array,
    # and a one-shot generator would be exhausted before the cluster-robust
    # residual pass below (silently reporting n_docs=0).  A bare scalar is a
    # single row, not an iterable protocol violation; strings/bytes are not
    # per-document measurements and are treated as empty rather than as
    # character sequences.
    def _rows(value):
        if value is None or isinstance(value, (str, bytes, bytearray)):
            return []
        try:
            return list(value)
        except TypeError:
            return [value]
    _nll = _rows(nll_per_row)
    _ntok = _rows(n_tokens_per_row)
    if len(_nll) != len(_ntok):
        # zip() below would silently drop the unmatched tail of the longer
        # list, reporting an interval for a different document sample than the
        # caller supplied (and hiding the malformed per-row accounting).
        raise ValueError(
            f'_nll_doc_ci: {len(_nll)} per-document NLL value(s) but '
            f'{len(_ntok)} token count(s); the accounting must be paired')

    def _parse_pair(nll, ntok):
        if isinstance(nll, (bool, np.bool_)) or isinstance(ntok, (bool, np.bool_)):
            return None
        try:
            nll = float(nll)
        except (TypeError, ValueError, OverflowError):
            return None
        # Token counts are exact integers.  A fractional count would silently
        # change the pooled denominator if int() truncated it.
        try:
            if isinstance(ntok, numbers.Integral):
                ntok = int(ntok)
            elif isinstance(ntok, numbers.Real):
                nf = float(ntok)
                if not math.isfinite(nf) or nf != math.floor(nf):
                    return None
                ntok = int(nf)
            else:
                # Objects outside numbers.Real (e.g. Decimal) can still
                # define __int__; check the numeric value before accepting it
                # so a fractional count cannot be silently truncated and
                # change the pooled denominator.
                nf = float(ntok)
                if not math.isfinite(nf) or nf != math.floor(nf):
                    return None
                ntok = int(nf)
        except (TypeError, ValueError, OverflowError):
            return None
        if ntok <= 0 or not math.isfinite(nll):
            return None
        return nll, ntok

    doc_ce = []
    total_nll = 0.0
    total_tok = 0
    for nll, ntok in zip(_nll, _ntok):
        pair = _parse_pair(nll, ntok)
        if pair is None:
            continue
        nll, ntok = pair
        total_nll += nll
        total_tok += ntok
        doc_ce.append(nll / ntok)
    if len(doc_ce) < 2 or total_tok <= 0:
        return {'ppl': (_safe_exp_ppl(total_nll / total_tok) if total_tok else None),
                'doc_ce_mean': None, 'doc_ce_sem': None,
                'ppl_ci95_low': None, 'ppl_ci95_high': None, 'n_docs': len(doc_ce)}
    mean = sum(doc_ce) / len(doc_ce)
    _doc_var = _scaled_sum_of_squares(
        [x - mean for x in doc_ce], 1.0 / (len(doc_ce) - 1))
    sem = (math.sqrt(max(_doc_var, 0.0) / len(doc_ce))
           if _doc_var is not None else None)
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
    for nll, ntok in zip(_nll, _ntok):
        pair = _parse_pair(nll, ntok)
        if pair is None:
            continue
        nll, ntok = pair
        _resid.append(nll - ce * ntok)
    n_docs = len(_resid)
    # Scale each residual by the total before squaring: r**2 and
    # total_tok**2 can each overflow while their ratio is representable.
    _var = _scaled_sum_of_squares(
        [r / total_tok for r in _resid], n_docs / (n_docs - 1.0))
    _sem_ce = (math.sqrt(max(_var, 0.0)) if _var is not None else None)
    _tc = _t_critical(0.95, n_docs - 1)
    if not math.isfinite(_tc):
        _tc = 1.959963984540054
    if _sem_ce is None:
        return {'ppl': _safe_exp_ppl(ce), 'doc_ce_mean': mean,
                'doc_ce_sem': sem, 'ce_sem_cluster_robust': None,
                'ci_method': ('point estimate only: the cluster-robust '
                              'variance exceeds the representable float range'),
                'ppl_ci95_low': None, 'ppl_ci95_high': None,
                'n_docs': n_docs}
    # Cross-entropy cannot be negative, so the corresponding PPL cannot be
    # below 1.  The symmetric t interval on CE can cross zero for a small,
    # high-variance document sample; exponentiating that raw lower endpoint
    # reported an impossible sub-1 perplexity confidence bound.
    _ce_lo = max(0.0, ce - _tc * _sem_ce)
    return {'ppl': _safe_exp_ppl(ce), 'doc_ce_mean': mean, 'doc_ce_sem': sem,
            'ce_sem_cluster_robust': _sem_ce,
            'ci_method': 'cluster-robust documents (t, df=n_docs-1)',
            'ppl_ci95_low': _safe_exp_ppl(_ce_lo),
            'ppl_ci95_high': _safe_exp_ppl(ce + _tc * _sem_ce),
            'n_docs': n_docs}

def _paired_nll_doc_ci(nll_a, nll_b, n_tokens, *, conf=0.95):
    """Cluster-robust CI for a PAIRED token-pooled PPL difference.

    ``nll_a`` / ``nll_b`` are ``[n_seeds, n_docs]`` per-document summed NLLs
    for the two arms; ``n_tokens`` is the shared ``[n_docs]`` token count (the
    two arms are evaluated on the same documents and the same positions).
    The point estimate is the frozen-estimand analogue

        mean_s [ exp(sum_i a_si / T) - exp(sum_i b_si / T) ],

    i.e. the mean over seeds of each seed's token-pooled PPL difference.

    The interval treats DOCUMENTS as clusters.  For each seed, the delta-method
    influence of its pooled CE difference on document ``i`` is

        exp(ce_a) * (a_i - ce_a * t_i) / T
      - exp(ce_b) * (b_i - ce_b * t_i) / T,

    and the influence of the seed-averaged difference is the average of those
    per-seed influences.  The variance is the usual cluster-robust
    ``D/(D-1) * sum_i psi_i^2`` with a t critical value (df=D-1).  This keeps
    the interval on the SAME token-weighted estimand as the pooled PPL, unlike
    a bootstrap over equal-weighted per-document PPLs (which answers a
    different question when document lengths differ).

    The interval conditions on the trained seeds: seed-to-seed training
    uncertainty is handled separately by the exact sign-flip test.  It is a
    document-sampling interval for the observed document panel, not a claim
    that the panel covers the language population.
    """
    def _matrix(x, name):
        try:
            raw = np.asarray(x)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f'{name}: values must be numeric') from None
        if _contains_bool_value(x) or raw.dtype.kind == 'b':
            # A boolean is a flag, not an NLL measurement (same scalar
            # contract as _finite_scalar/_nll_doc_ci).  Without this check
            # True/False would silently become 1.0/0.0.
            raise ValueError(f'{name}: booleans are not NLL measurements')
        try:
            arr = np.asarray(raw, dtype=float)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f'{name}: values must be numeric') from None
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
            raise ValueError(f'{name}: expected [n_seeds, n_docs], got shape {arr.shape}')
        if not np.all(np.isfinite(arr)):
            raise ValueError(f'{name}: all per-document NLLs must be finite')
        return arr
    A = _matrix(nll_a, '_paired_nll_doc_ci(nll_a)')
    B = _matrix(nll_b, '_paired_nll_doc_ci(nll_b)')
    if A.shape != B.shape:
        raise ValueError(
            f'_paired_nll_doc_ci: arm shapes differ ({A.shape} vs {B.shape}); '
            'a paired interval needs the same seeds and documents')
    try:
        _raw_t = np.asarray(n_tokens).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('_paired_nll_doc_ci: n_tokens must be numeric') from None
    if _contains_bool_value(n_tokens) or _raw_t.dtype.kind == 'b':
        # A mixed [True, 1.0] list coerces to a float array before
        # `_exact_count` sees it; reject the flag at the source.
        raise ValueError('_paired_nll_doc_ci: token counts must be integers, not booleans')
    # Token counts are exact integers: a fractional or boolean count must not
    # be silently truncated/coerced into a different pooled denominator (the
    # same contract _nll_doc_ci already enforces).
    _counts = [_exact_count(v, minimum=1) for v in _raw_t.tolist()]
    if any(c is None for c in _counts):
        raise ValueError('_paired_nll_doc_ci: every document needs a finite positive integer token count')
    T = np.asarray(_counts, dtype=float)
    if T.size != A.shape[1]:
        raise ValueError(
            f'_paired_nll_doc_ci: {T.size} token counts for {A.shape[1]} documents')
    if isinstance(conf, (bool, np.bool_)):
        raise ValueError('_paired_nll_doc_ci: conf must be a probability in (0, 1)')
    try:
        conf = float(conf)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('_paired_nll_doc_ci: conf must be a probability in (0, 1)') from None
    if not (0.0 < conf < 1.0):
        raise ValueError('_paired_nll_doc_ci: conf must be a probability in (0, 1)')

    S, D = A.shape
    total_tok = float(T.sum())
    if not math.isfinite(total_tok) or total_tok <= 0.0:
        raise ValueError('_paired_nll_doc_ci: the total token count must be positive')
    ce_a = A.sum(axis=1) / total_tok
    ce_b = B.sum(axis=1) / total_tok
    ea = np.asarray([_safe_exp_ppl(x) for x in ce_a], dtype=float)
    eb = np.asarray([_safe_exp_ppl(x) for x in ce_b], dtype=float)
    if not (np.all(np.isfinite(ea)) and np.all(np.isfinite(eb))):
        return None
    delta_ppl = float(np.mean(ea - eb))
    lr = float(np.mean(ce_a - ce_b))
    out = {'n_docs': int(D), 'n_seeds': int(S), 'conf': float(conf),
           'delta_ppl': delta_ppl, 'delta_log_ratio': lr,
           'delta_ppl_ci95': None, 'delta_log_ratio_ci95': None,
           'ratio_ci95': None, 'se_ppl': None, 'se_log_ratio': None,
           'df': (int(D - 1) if D > 1 else None),
           'method': ('cluster-robust documents (delta method, t df=n_docs-1; '
                      'token-pooled per seed then averaged over seeds; '
                      'conditions on the trained seeds)')}
    if D < 2:
        out['method'] = ('point estimate only: a cluster-robust interval needs '
                         'at least two documents')
        return out

    # Per-seed influence on the pooled CE difference, then average over seeds.
    psi = (ea[:, None] * (A - ce_a[:, None] * T[None, :]) / total_tok
           - eb[:, None] * (B - ce_b[:, None] * T[None, :]) / total_tok)
    bar = psi.mean(axis=0)
    var = _scaled_sum_of_squares(bar, D / (D - 1.0))
    se = (math.sqrt(max(var, 0.0)) if var is not None else None)
    psi_lr = ((A - ce_a[:, None] * T[None, :])
              - (B - ce_b[:, None] * T[None, :])) / total_tok
    bar_lr = psi_lr.mean(axis=0)
    var_lr = _scaled_sum_of_squares(bar_lr, D / (D - 1.0))
    se_lr = (math.sqrt(max(var_lr, 0.0)) if var_lr is not None else None)
    if se is None or se_lr is None:
        out['method'] = ('point estimate only: the cluster-robust variance '
                         'exceeds the representable float range')
        return out
    tc = _t_critical(conf, D - 1)
    if not math.isfinite(tc):
        return out
    _lo, _hi = delta_ppl - tc * se, delta_ppl + tc * se
    _lrl, _lrh = lr - tc * se_lr, lr + tc * se_lr
    out.update({
        'se_ppl': float(se),
        'se_log_ratio': float(se_lr),
        'delta_ppl_ci95': ([_lo, _hi]
                           if math.isfinite(_lo) and math.isfinite(_hi)
                           else None),
        'delta_log_ratio_ci95': ([_lrl, _lrh]
                                 if math.isfinite(_lrl) and math.isfinite(_lrh)
                                 else None),
        'ratio_ci95': ([_safe_exp_ppl(_lrl), _safe_exp_ppl(_lrh)]
                       if math.isfinite(_lrl) and math.isfinite(_lrh)
                       else None),
    })
    return out
