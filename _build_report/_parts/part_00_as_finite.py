#!/usr/bin/env python3
import collections
import json
import math
import numbers
import os

import numpy as np
import random
import re
import sys
import tempfile
import time
import traceback
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except (ValueError, OSError):
            pass
REPO = os.path.dirname(os.path.abspath(__file__))
PAIR_STEP_KEYS = ('steps', 'tokens_seen')
_LOADED = {}

def _as_finite(v):
    """Finite float for any real numeric scalar (NumPy scalars included).

    Report helpers are also called directly on in-memory records, not only on
    JSON-decoded ones.  An ``isinstance(v, (int, float))`` filter would miss
    ``np.int64``/``np.float32`` values, which could turn a non-empty panel into
    "no data" or make the local exact test disagree with ``exp_lib``.
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

def load(path, default=None):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except Exception as _e:
        # 「文件存在但损坏」与「实验未跑」是两种不同的事实：前者必须阻断
        # 发布（进入 main 的 failed 列表），而不是伪装成「（无结果）」。
        raise RuntimeError(f'{path} exists but cannot be parsed ({type(_e).__name__}: {_e}); refusing to build a report over corrupt data — repair or remove the file') from _e

def _panel_object(path, value, kind):
    """A panel artifact must be a cell -> record object.

    A valid-JSON list/string used to flow into ``.items()`` in later sections
    (per_seed_records, curve readers) and abort the whole report.  Excluding
    the malformed panel with a warning is the same contract as the exp_lib
    readers.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        print(f'[report] WARNING: {path} is a {type(value).__name__}, not a '
              f'JSON object — this {kind} is EXCLUDED')
        return {}
    return value

def agg(outdir):
    key = ('agg', outdir)
    if key not in _LOADED:
        path = os.path.join(REPO, outdir, 'aggregate.json')
        _LOADED[key] = _panel_object(path, load(path, {}), 'aggregate')
    return _LOADED[key]

def summary(outdir):
    key = ('summary', outdir)
    if key not in _LOADED:
        path = os.path.join(REPO, outdir, 'summary.json')
        _LOADED[key] = _panel_object(path, load(path, {}), 'panel')
    return _LOADED[key]

def fm(x, nd=2):
    _f = _as_finite(x)
    return f'{_f:.{nd}f}' if _f is not None else 'n/a'

def sp(x, nd=2):
    _f = _as_finite(x)
    return f'±{_f:.{nd}f}' if _f is not None else ''

def _ppl_value(entry):
    """Primary PPL aggregate: pooled token-level NLL when the records carry it,
    otherwise the historical arithmetic seed mean (explicitly labelled)."""
    if not isinstance(entry, dict):
        return None
    for k in ('ppl_primary', 'ppl_pooled', 'ppl_mean'):
        v = entry.get(k)
        _f = _as_finite(v)
        # PPL = exp(CE) is >= 1; a sub-1 value is a corrupt measurement.
        if _f is not None and _f >= 1.0:
            return _f
    return None

def _ppl_tag(entry):
    if isinstance(entry, dict) and entry.get('ppl_pooled') is not None:
        return 'pooled'
    return 'mean(PPL)' if isinstance(entry, dict) else 'n/a'

def _hist_points(r):
    for _pt in (r.get('ppl_history') or []) if isinstance(r, dict) else []:
        try:
            _st, _pv = _pt
            _st = int(_st)
        except (TypeError, ValueError, OverflowError):
            continue
        yield (_st, _pv)

def sample_std(vals):
    n = len(vals)
    if n < 2:
        return 0.0
    scale = max((abs(float(v)) for v in vals), default=0.0)
    if math.isfinite(scale) and scale > 0.0 and (scale < 1e-100 or scale > 1e100):
        z = [float(v) / scale for v in vals]
        mu = math.fsum(z) / n
        ss = math.fsum((v - mu) ** 2 for v in z)
        return scale * math.sqrt(ss / (n - 1))
    mu = sum(vals) / n
    return (sum(((v - mu) ** 2 for v in vals)) / (n - 1)) ** 0.5

def _signflip_mitm_count(deltas):
    """Exact sign-assignment count by meet-in-the-middle (NumPy).

    Mirrors `exp_lib.exact_signflip_p`: 2**(n/2) sums per half and two
    binary searches per left sum, so n=21..40 stays exact without importing
    torch or enumerating 2**n masks.  Threshold uses this module's existing
    convention (abs sum vs 1e-12) so the reported p is unchanged for n<=20."""
    import numpy as _np
    d = _np.asarray(deltas, dtype=float)
    n = int(d.size)
    total = abs(float(d.sum()))
    # Same tie tolerance as exp_lib._signflip_mitm_count: the compared means
    # differ by at most 1e-12, i.e. the sums by n*1e-12.
    thr = total - 1e-12 * n
    if thr <= 0.0:
        return 1 << n

    def _sums(vals):
        out = _np.zeros(1, dtype=float)
        for v in vals:
            out = _np.concatenate((out + v, out - v))
        return out

    half = n // 2
    left = _sums(d[:half])
    right = _sums(d[half:])
    right.sort()
    lo = _np.searchsorted(right, -thr - left, side='right')
    hi = _np.searchsorted(right, thr - left, side='left')
    return int(lo.sum() + (right.size - hi).sum())


def _signflip_saddlepoint_p(values):
    """Mirror of `exp_lib._signflip_saddlepoint_p` (no torch import here).

    Deterministic Lugannani-Rice tail for the sign-flip null at n > 40.
    Keeping the two implementations arithmetically identical is what makes
    the zero-GPU report agree with the audit instead of drawing a random
    Monte-Carlo p-value.
    """
    vals = [float(x) for x in values if float(x) != 0.0]
    m = len(vals)
    if m == 0:
        return 1.0, 'saddlepoint'
    amax = max(abs(x) for x in vals)
    if amax > 1e100 or amax < 1e-100:
        vals = [x / amax for x in vals]
    T = abs(math.fsum(vals))
    sabs = math.fsum(abs(x) for x in vals)
    floor = min(1.0, math.ldexp(2.0, -m))
    if T <= 0.0:
        return 1.0, 'saddlepoint'
    if T >= sabs * (1.0 - 1e-13):
        return floor, 'saddlepoint'
    inv = 1.0 / max(abs(x) for x in vals)

    def _kp(t):
        return math.fsum(x * math.tanh(t * x) for x in vals)

    def _k(t):
        return math.fsum(abs(t * x) + math.log1p(math.exp(-2.0 * abs(t * x))) - math.log(2.0)
                         for x in vals)

    def _kpp(t):
        # sech^2(z) = 1 - tanh^2(z) avoids a cosh overflow in the extreme
        # near-maximum-sum tail while staying algebraically identical.
        return math.fsum(x * x * (1.0 - math.tanh(t * x) ** 2) for x in vals)

    hi = inv
    for _ in range(60):
        if _kp(hi) >= T:
            break
        hi *= 2.0
    else:
        sd = math.sqrt(math.fsum(x * x for x in vals))
        p = math.erfc(T / (sd * math.sqrt(2.0))) if sd > 0 else floor
        return float(min(1.0, max(floor, p))), 'saddlepoint-normal-fallback'
    lo = 0.0
    for _ in range(160):
        mid = 0.5 * (lo + hi)
        if _kp(mid) < T:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    Kt = _k(t)
    K2 = _kpp(t)
    w = math.sqrt(max(0.0, 2.0 * (t * T - Kt)))
    u = t * math.sqrt(K2) if K2 > 0.0 else 0.0
    if w < 1e-8 or u <= 0.0:
        sd = math.sqrt(math.fsum(x * x for x in vals))
        p = math.erfc(T / (sd * math.sqrt(2.0))) if sd > 0 else floor
    else:
        phi = math.exp(-0.5 * w * w) / math.sqrt(2.0 * math.pi)
        tail = 0.5 * math.erfc(w / math.sqrt(2.0)) + phi * (1.0 / u - 1.0 / w)
        p = 2.0 * tail
    return float(min(1.0, max(floor, p))), 'saddlepoint'


def _signflip_floor(n):
    """Two-sided sign-flip resolution floor 2/2**n (independent of the branch
    used to compute p; exact, MITM and saddlepoint all obey it)."""
    try:
        ni = int(n)
    except (TypeError, ValueError, OverflowError):
        return float('nan')
    if isinstance(n, (bool, np.bool_)) or ni <= 0 or ni != n:
        return float('nan')
    return math.ldexp(2.0, -ni)


def exact_signflip(deltas, *, n_input=None):
    # Same scalar contract as exp_lib.exact_signflip_p: NumPy integers and
    # floats are measurements too, non-finite/boolean values are not.
    deltas = [f for f in (_as_finite(d) for d in deltas) if f is not None]
    n = len(deltas)
    if n == 0:
        return None
    _stat_scale = max((abs(x) for x in deltas), default=0.0)
    if _stat_scale > 1e100 or 0.0 < _stat_scale < 1e-100:
        _dn = [x / _stat_scale for x in deltas]
        _mu_n = sum(_dn) / n
        mu = _stat_scale * _mu_n
        std = (_stat_scale
               * ((sum((x - _mu_n) ** 2 for x in _dn) / (n - 1)) ** 0.5)
               if n > 1 else 0.0)
    else:
        mu = sum(deltas) / n
        std = (sum(((d - mu) ** 2 for d in deltas)) / (n - 1)) ** 0.5 if n > 1 else 0.0
    # Exact ties do not change p (their signs are arbitrary) but they do lower
    # the attainable resolution from 2/2**n to 2/2**(#nonzero).
    _n_nonzero = sum(1 for d in deltas if d != 0.0)
    _p_floor = (1.0 if _n_nonzero == 0
                else min(1.0, math.ldexp(2.0, -_n_nonzero)))
    _amax = max((abs(x) for x in deltas), default=0.0)
    _p_deltas = ([x / _amax for x in deltas]
                 if (_amax > 1e100 or 0.0 < _amax < 1e-100) else deltas)
    _obs_p = abs(sum(_p_deltas))
    # Panel attrition is part of the result: `deltas` may be the complete-case
    # subset after unpaired/non-finite seeds were dropped.  The caller can pass
    # the planned/paired count so the report gate can refuse a
    # complete-case-only significance claim.
    _n_in = _int_or(n_input, n)
    if _n_in is None or _n_in < n:
        _n_in = n
    _attrition = {'n_input': int(_n_in), 'n_dropped': int(_n_in - n)}
    # Leave-one-seed-out stability: a significant aggregate can still be owned
    # by a single seed.  Mirror exp_lib.paired_effect_stats so the report and
    # the methodology audit cannot disagree on this diagnostic.
    _loo = []
    _seed_driven = None
    if n >= 3:
        if _stat_scale > 1e100 or 0.0 < _stat_scale < 1e-100:
            _total_n = sum(_dn)
            _loo = [_stat_scale * ((_total_n - x) / (n - 1)) for x in _dn]
        else:
            _total = float(sum(deltas))
            _loo = [(_total - float(d)) / (n - 1) for d in deltas]
        _seed_driven = False
        if abs(mu) > 1e-300 * _stat_scale:
            _sign = 1.0 if mu > 0 else -1.0
            _seed_driven = any((m == 0.0) or ((m > 0) != (_sign > 0)) for m in _loo)
    _diag = {'n_nonzero': _n_nonzero, 'p_floor': _p_floor,
             'loo_means': _loo, 'seed_driven': _seed_driven,
             'seed_robust': (None if _seed_driven is None else (not _seed_driven)),
             **_attrition}
    if 20 < n <= 40:
        try:
            _cnt = _signflip_mitm_count(_p_deltas)
            return {'n': n, 'mean': mu, 'std': std,
                    'p_exact_signflip': _cnt / (1 << n),
                    'method': 'exact-mitm', **_diag}
        except ImportError:
            pass
    if n > 40:
        # Same deterministic tail as exp_lib: no random MC draw, and the
        # variance is the sign-flip null variance sum(d_i**2).
        _p, _method = _signflip_saddlepoint_p(_p_deltas)
        return {'n': n, 'mean': mu, 'std': std, 'p_exact_signflip': _p,
                'method': _method, **_diag}
    try:
        import numpy as _np
    except ImportError:
        _np = None
    if _np is not None:
        _total = 1 << n
        _thr = _obs_p - 1e-12 * n
        _darr = _np.asarray(_p_deltas, dtype=float)
        _ebnd = n * 2.220446049250313e-16 * float(_np.abs(_darr).sum())
        _bitpos = _np.arange(n, dtype=_np.int64)
        _step = max(1, min(_total, (1 << 24) // max(n, 1)))
        cnt = 0
        for _off in range(0, _total, _step):
            _masks = _np.arange(_off, min(_off + _step, _total), dtype=_np.int64)
            _signs = (((_masks[:, None] >> _bitpos) & 1) * 2 - 1).astype(float)
            _s = _np.abs((_signs * _darr[None, :]).sum(1))
            _near = _np.abs(_s - _thr) <= _ebnd
            cnt += int((_s[~_near] >= _thr).sum())
            for _m in _masks[_near].tolist():
                s = sum((d if _m >> i & 1 else -d for i, d in enumerate(_p_deltas)))
                if abs(s) >= _thr:
                    cnt += 1
    else:
        cnt = 0
        for mask in range(1 << n):
            s = sum((d if mask >> i & 1 else -d for i, d in enumerate(_p_deltas)))
            if abs(s) >= _obs_p - 1e-12 * n:
                cnt += 1
    return {'n': n, 'mean': mu, 'std': std,
            'p_exact_signflip': cnt / (1 << n), 'method': 'exact', **_diag}

def _measurable(r):
    if not isinstance(r, dict):
        return False
    v = r.get('ppl')
    return not r.get('synthesized') and _ppl_ok(v)

def _ppl_ok(v):
    f = _as_finite(v)
    return f is not None and f >= 1.0

def _nested_estimand(rec):
    return (isinstance(rec, dict)
            and rec.get('ppl_estimation_rows') == 'nested_validation_holdout')


def _primary_ppl(rec):
    """Per-seed PPL on the record's PRIMARY estimand.

    A nested tuning record keeps the full-validation value in ``ppl`` and the
    held-out-row estimate in ``ppl_holdout`` (or its holdout NLL accounting).
    A paired delta that reads ``ppl`` would silently use the rows that selected
    the hyper-parameters.  A record that declares the nested estimand but has
    no holdout accounting is unverifiable and returns None.
    """
    if not isinstance(rec, dict):
        return None
    if rec.get('ppl_estimation_rows') == 'nested_validation_holdout':
        v = _as_finite(rec.get('ppl_holdout'))
        if v is not None and v >= 1.0:
            return v
        nll = _as_finite(rec.get('nll_sum_holdout'))
        ntok = _int_or(rec.get('n_tokens_holdout'), None)
        if nll is not None and ntok is not None and ntok > 0:
            try:
                _v = math.exp(nll / ntok)
            except OverflowError:
                return None
            return _v if math.isfinite(_v) and _v >= 1.0 else None
        _nll = rec.get('nll_per_row_holdout')
        _ntok = rec.get('n_tokens_per_row_holdout')
        if isinstance(_nll, list) and isinstance(_ntok, list) and _nll and len(_nll) == len(_ntok):
            _v = [_as_finite(x) for x in _nll]
            _t = [_int_or(x, None) for x in _ntok]
            if all(x is not None for x in _v) and all(x is not None and x > 0 for x in _t):
                _tot = sum(_t)
                if _tot > 0:
                    try:
                        _v2 = math.exp(sum(_v) / _tot)
                    except OverflowError:
                        return None
                    return _v2 if math.isfinite(_v2) and _v2 >= 1.0 else None
        return None
    v = _as_finite(rec.get('ppl'))
    return v if v is not None and v >= 1.0 else None

def _finite_or_none(v):
    return _as_finite(v)

def _int_or(v, default):
    if v is None or isinstance(v, (bool, np.bool_)):
        return default
    if isinstance(v, numbers.Integral):
        return int(v)
    if isinstance(v, numbers.Real):
        f = float(v)
        return int(f) if math.isfinite(f) and f.is_integer() else default
    # Objects outside numbers.Real (e.g. Decimal) can still define __int__.
    # Check the numeric value before accepting it so a fractional value is
    # reported as the default instead of silently truncated to another count.
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        return default
    return int(f) if math.isfinite(f) and f.is_integer() else default

def _code_semantics():
    try:
        with open(os.path.join(REPO, 'exp_lib.py'), encoding='utf-8') as f:
            m = re.search("^CODE_SEMANTICS = '([^']+)'", f.read(), re.M)
            return m.group(1) if m else None
    except Exception:
        return None

def _kernel_stamp():
    """Mirror exp_lib.code_stamp()'s kernel suffix without importing torch.

    `_code` carries '-tk1:<backend>' under CSA_KERNEL; comparing it to the bare
    CODE_SEMANTICS labels every current kernel record stale (design_notes §14).
    For `auto` the real backend depends on CUDA/triton availability, so a
    report should run with the same explicit CSA_KERNEL value the panel used;
    an ambiguous `auto` produces a non-matching stamp and therefore a visible
    stale warning rather than a false 'current' claim.
    """
    mode = os.environ.get('CSA_KERNEL', '').strip().lower()
    if mode in ('', '0', 'off', 'none', 'false'):
        return ''
    if mode == 'ref':
        return '-tk1:ref'
    if mode in ('1', 'true', 'on', 'triton'):
        return '-tk1:triton'
    if mode == 'auto':
        return '-tk1:auto'
    return ''

_CUR_CS = (_code_semantics() or '') + _kernel_stamp()

def _finite_ppls(r):
    if not isinstance(r, dict):
        return []
    ppls = r.get('ppls') or r.get('ppl_list') or []
    if not isinstance(ppls, (list, tuple)):
        return []
    return [f for f in (_as_finite(x) for x in ppls) if f is not None and f >= 1.0]

_WARM_KEY_RE = re.compile('.*::w(\\d+)::seed\\d+$')

def _legacy_warm_tags(s):
    _legacy = {}
    for key, rec in s.items():
        if not isinstance(rec, dict) or 'seed' not in rec or 'ppl' not in rec:
            continue
        if not _ppl_ok(rec.get('ppl')) or rec.get('synthesized'):
            continue
        if ('protocol' in rec and rec['protocol'] is not None) or rec.get('warm_steps') is not None:
            continue
        m = _WARM_KEY_RE.match(str(key))
        _var = rec.get('variant')
        if _var is None:
            _var = str(key).split('::', 1)[0]
        _legacy.setdefault(str(_var), set()).add(m.group(1) if m else None)
    return {v: frozenset(tags) for v, tags in _legacy.items() if None not in tags}

def _legacy_warm_tags_cached(outdir):
    key = ('legacy', outdir)
    if key not in _LOADED:
        _LOADED[key] = _legacy_warm_tags(summary(outdir))
    return _LOADED[key]

def _tag_of(r, key=None, legacy_ok=None):
    if not isinstance(r, dict):
        return ''
    if r.get('protocol') is not None:
        return str(r['protocol'])
    if r.get('warm_steps') is not None:
        return f"w{r['warm_steps']}"
    if key is not None:
        m = _WARM_KEY_RE.match(str(key))
        if m:
            ok = legacy_ok
            if isinstance(ok, dict):
                _var = r.get('variant')
                if _var is None:
                    _var = str(key).split('::', 1)[0]
                ok = ok.get(str(_var))
                if ok is None:
                    return ''
            if ok is not None and m.group(1) in ok:
                return f'w{m.group(1)}'
    return ''

def _prefer_one_cfg(items, cfg_of, who=''):
    by = collections.defaultdict(list)
    for it in items:
        by[cfg_of(it)].append(it)
    per = collections.defaultdict(list)
    for (label, cfg), lst in by.items():
        per[label].append((cfg, lst))
    kept, dropped = ([], [])
    for label, gl in per.items():
        if len(gl) == 1:
            kept.extend(gl[0][1])
            continue
        gl.sort(key=lambda kv: (not (isinstance(kv[0], str) and _CUR_CS and kv[0].endswith(f'_cs{_CUR_CS}')), -len(kv[1])))
        kept.extend(gl[0][1])
        for _cfg, lst in gl[1:]:
            dropped.extend(lst)
    if dropped:
        print(f'[report] {who}: {len(dropped)} record(s) sit in a minority run_cfg group of their variant — pooling across configurations is not allowed, so one group per variant is kept (the current code stamp preferred) and the rest are dropped from this table')
    return kept
PARAM_MATCH_TOL = 0.01  # same 1% gate as exp_lib._param_control
PARAM_MATCH_SMALL = 0.3
PARAM_MATCH_ARCH = 0.06

def _exact_int_ratio(numerator, denominator):
    """Representable float of an exact non-negative integer ratio, or None.

    Parameter counts in an in-memory/JSON record can exceed float64's range.
    ``abs(pa - pb) / max(pa, pb)`` then raises ``OverflowError`` even though
    the ratio itself is in [0, 1].  Build the exact quotient first and round
    once; an underflowed nonzero ratio is unverifiable (None), never a silent
    zero that would look like a capacity match.
    """
    try:
        numerator, denominator = int(numerator), int(denominator)
    except (TypeError, ValueError, OverflowError):
        return None
    if denominator <= 0 or numerator < 0:
        return None
    try:
        value = numerator / denominator
    except OverflowError:
        from fractions import Fraction
        value = float(Fraction(numerator, denominator))
    if value == 0.0 and numerator != 0:
        return None
    return float(value)

def _exact_int_ratio_within(numerator, denominator, tolerance):
    """Exact integer ``numerator / denominator <= tolerance``, or None."""
    try:
        numerator, denominator = int(numerator), int(denominator)
        tolerance = float(tolerance)
    except (TypeError, ValueError, OverflowError):
        return None
    if denominator <= 0 or numerator < 0 or not math.isfinite(tolerance):
        return None
    try:
        from fractions import Fraction
        return bool(Fraction(numerator, denominator)
                    <= Fraction.from_float(tolerance))
    except (OverflowError, ValueError):
        return None

def param_gap(a, b, tol=PARAM_MATCH_TOL):
    if not isinstance(a, dict) or not isinstance(b, dict):
        return None
    # Parameter counts are exact integers (same contract as exp_lib); a
    # fractional/boolean value is not a capacity measurement.
    pa = _int_or(a.get('params'), None)
    pb = _int_or(b.get('params'), None)
    if pa is None or pb is None:
        return None
    if pa <= 0 or pb <= 0:
        return None
    return _exact_int_ratio(abs(pa - pb), max(pa, pb))

def is_param_matched(a, b, tol=PARAM_MATCH_TOL):
    g = param_gap(a, b, tol)
    if g is not None:
        return g <= tol
    # The relative gap may underflow float64 for enormous near-equal counts;
    # the tolerance gate itself is still exactly decidable.
    if not isinstance(a, dict) or not isinstance(b, dict):
        return None
    pa = _int_or(a.get('params'), None)
    pb = _int_or(b.get('params'), None)
    if pa is None or pb is None or pa <= 0 or pb <= 0:
        return None
    return _exact_int_ratio_within(abs(pa - pb), max(pa, pb), tol)

def gap_is_architecture(a, b, tol=PARAM_MATCH_TOL, arch=PARAM_MATCH_ARCH):
    g = param_gap(a, b, tol)
    if g is not None:
        return tol < g <= arch
    if not isinstance(a, dict) or not isinstance(b, dict):
        return None
    pa = _int_or(a.get('params'), None)
    pb = _int_or(b.get('params'), None)
    if pa is None or pb is None or pa <= 0 or pb <= 0:
        return None
    _num, _den = abs(pa - pb), max(pa, pb)
    _within_tol = _exact_int_ratio_within(_num, _den, tol)
    _within_arch = _exact_int_ratio_within(_num, _den, arch)
    if _within_tol is None or _within_arch is None:
        return None
    return (not _within_tol) and _within_arch

def unmatched_tag(a, b, tol=PARAM_MATCH_TOL, small=PARAM_MATCH_SMALL):
    g = param_gap(a, b, tol)
    if g is None:
        return ' (⚠ params 未知)'
    if g <= tol:
        return ''
    if g > small:
        return f' (⚠ params 差异过大 {100 * g:.1f}%)'
    if gap_is_architecture(a, b, tol):
        return f' (⚠ 参数差 {100 * g:.1f}%，属架构差异)'
    return f' (⚠ 参数不匹配 {100 * g:.1f}%)'

def per_seed_records(outdir):
    _ck = ('psr', outdir)
    if _ck in _LOADED:
        return _LOADED[_ck]
    s = summary(outdir)
    _legacy_ok = _legacy_warm_tags_cached(outdir)
    out = collections.defaultdict(dict)
    for _k, r in s.items():
        if isinstance(r, dict) and 'ppl' in r and ('variant' in r) and ('seed' in r) and _measurable(r):
            # An arm id is a dict/set key below and a grouping label in every
            # table; a non-string or empty id is a malformed record, not an
            # arm.  Exclude it explicitly instead of crashing on an
            # unhashable value (same contract as exp_lib.aggregate).
            _variant = r.get('variant')
            if not isinstance(_variant, str) or not _variant:
                print(f"[report] record {_k!r} in {outdir} carries a non-string/empty arm id ({_variant!r}) — skipped for pairing")
                continue
            _sv = r['seed']
            if isinstance(_sv, (bool, np.bool_)):
                print(f"[report] record {_k!r} in {outdir} carries a boolean seed — skipped for pairing")
                continue
            # Exact integer parsing: int(Decimal('1.5')) would silently
            # truncate to a different seed and pair the wrong run.  Numeric
            # strings keep the legacy integer spelling ("3", not "3.0").
            if isinstance(_sv, (str, bytes)):
                try:
                    _sd = int(_sv)
                except (TypeError, ValueError, OverflowError):
                    _sd = None
            else:
                _sd = _int_or(_sv, None)
            if _sd is None:
                print(f"[report] record {_k!r} in {outdir} carries a non-integer seed ({r.get('seed')!r}) — skipped for pairing")
                continue
            _slot = out[_variant]
            _tk = (_tag_of(r, _k, _legacy_ok), _sd)
            if _tk in _slot:
                print(f'[report] ambiguous (variant, protocol, seed) = ({_variant!r}, {_tk[0]!r}, {_sd}) in {outdir}: two measurable records found; keeping the FIRST (key {_k!r} ignored for pairing)')
                continue
            _slot[_tk] = r
    out = dict(out)
    _LOADED[_ck] = out
    return out

_TRAINING_KNOB_REQUIRED_FIELDS = (
    'optimizer', 'foreach', 'betas', 'eps', 'grad_clip_norm',
    'lr_schedule', 'lr_floor_mult', 'lr_cosine_amp', 'augmentation',
    'amp_dtype', 'amp_grad_scaler', 'eager_batch',
)

def _knob_number(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Number):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return f if math.isfinite(f) else None

def _training_knobs_problem(knobs, *, label='train_knobs'):
    """Report twin of exp_lib.training_knobs_problem (no exp_lib import)."""
    if not isinstance(knobs, dict):
        return f'{label} is not an object ({type(knobs).__name__})'
    missing = [k for k in _TRAINING_KNOB_REQUIRED_FIELDS if k not in knobs]
    if missing:
        return (f'{label} omits controlled field(s) {missing} — the descriptor '
                'does not cover the optimizer / schedule / augmentation recipe')
    if not isinstance(knobs.get('optimizer'), str) or not knobs['optimizer'].strip():
        return f'{label}.optimizer is missing/empty'
    if not isinstance(knobs.get('foreach'), (bool, np.bool_)):
        return f'{label}.foreach is not a boolean'
    betas = knobs.get('betas')
    if (not isinstance(betas, (list, tuple)) or len(betas) != 2
            or any(_knob_number(b) is None for b in betas)):
        return f'{label}.betas is not a pair of finite numbers'
    if any(not (0.0 <= float(b) < 1.0) for b in betas):
        return f'{label}.betas values must lie in [0,1)'
    eps = _knob_number(knobs.get('eps'))
    if eps is None or eps <= 0.0:
        return f'{label}.eps is not a finite positive number'
    clip = _knob_number(knobs.get('grad_clip_norm'))
    if clip is None or clip < 0.0:
        return f'{label}.grad_clip_norm is not a finite non-negative number'
    if (not isinstance(knobs.get('lr_schedule'), str)
            or not knobs['lr_schedule'].strip()):
        return f'{label}.lr_schedule is missing/empty'
    for key in ('lr_floor_mult', 'lr_cosine_amp'):
        val = _knob_number(knobs.get(key))
        if val is None or val < 0.0:
            return f'{label}.{key} is not a finite non-negative number'
    if (not isinstance(knobs.get('augmentation'), str)
            or not knobs['augmentation'].strip()):
        return f'{label}.augmentation is missing/empty'
    if knobs.get('amp_dtype') is not None and not isinstance(knobs.get('amp_dtype'), str):
        return f'{label}.amp_dtype is neither null nor a string'
    if not isinstance(knobs.get('amp_grad_scaler'), (bool, np.bool_)):
        return f'{label}.amp_grad_scaler is not a boolean'
    if not isinstance(knobs.get('eager_batch'), (bool, np.bool_)):
        return f'{label}.eager_batch is not a boolean'
    if 'param_groups' in knobs and knobs.get('param_groups') is not None:
        if (not isinstance(knobs['param_groups'], str)
                or not knobs['param_groups'].strip()):
            return f'{label}.param_groups is not a non-empty string'
    return None

def _training_knobs_reason(ka, kb):
    if (ka is None) != (kb is None):
        return ('unverifiable train_knobs (only one record carries the '
                'optimizer / lr-schedule / augmentation descriptor)')
    if ka is None:
        return None
    _pa = _training_knobs_problem(ka, label='train_knobs (arm a)')
    if _pa is not None:
        return _pa
    _pb = _training_knobs_problem(kb, label='train_knobs (arm b)')
    if _pb is not None:
        return _pb
    if ka != kb:
        return ('different train_knobs (optimizer / lr schedule / grad-clip / '
                'augmentation / precision)')
    return None

def _execution_environment_reason(ra, rb):
    """Mirror exp_lib.execution_environment_reason for the zero-GPU report.

    Same-seed replication is only controlled on the same hardware/software/
    precision class.  The statistics layer refuses a pair whose execution
    fingerprint disagrees; the report must not pair it either.  Legacy records
    without the digest keep the historical permissive path.
    """
    def _fp(rec):
        v = rec.get('exec_fp')
        if v is None and isinstance(rec.get('hyperparams'), dict):
            v = rec['hyperparams'].get('exec_fp')
        return v
    a, b = _fp(ra), _fp(rb)
    ha = a is not None and str(a) != ''
    hb = b is not None and str(b) != ''
    if ha and hb and str(a) != str(b):
        return (f'different execution environment ({str(a)[:12]}… vs '
                f'{str(b)[:12]}…): same-seed replication is not controlled '
                'across hardware/driver/precision classes')
    return None

def _pair_reason(ra, rb):
    if not isinstance(ra, dict) or not isinstance(rb, dict):
        return 'a record is not a dict'
    fa, fb = (ra.get('run_cfg'), rb.get('run_cfg'))
    if fa is None or fb is None:
        return 'unverifiable run_cfg'
    if fa != fb:
        return 'different run_cfg'
    # Mirror exp_lib.pair_reason's code-semantics and protocol grouping gates.
    _ca, _cb = ra.get('_code'), rb.get('_code')
    _ha = _ca is not None and str(_ca) != ''
    _hb = _cb is not None and str(_cb) != ''
    if _ha != _hb:
        return 'unverifiable _code (only one record carries the code-semantics stamp)'
    if _ha and str(_ca) != str(_cb):
        return f'different code semantics ({str(_ca)[:24]}… vs {str(_cb)[:24]}…)'
    _pa = ('protocol' in ra and ra.get('protocol') is not None) or ra.get('warm_steps') is not None
    _pb = ('protocol' in rb and rb.get('protocol') is not None) or rb.get('warm_steps') is not None
    if _pa != _pb:
        return 'unverifiable protocol (only one record carries a protocol tag)'
    if _pa and str(_tag_of(ra)) != str(_tag_of(rb)):
        return f'different protocol ({_tag_of(ra)!r} vs {_tag_of(rb)!r})'
    _env_reason = _execution_environment_reason(ra, rb)
    if _env_reason is not None:
        return _env_reason
    # The run_cfg string does not carry the optimizer identity / schedule
    # shape / clipping / augmentation or the AMP/eager implementation mode.
    # Mirror exp_lib.pair_reason so a zero-GPU report cannot pair cells that
    # ran under different numerical or optimisation settings.
    _ka, _kb = (ra.get('train_knobs'), rb.get('train_knobs'))
    _knob_reason = _training_knobs_reason(_ka, _kb)
    if _knob_reason is not None:
        return _knob_reason
    for _k in ('amp_dtype', 'amp_grad_scaler', 'eager_batch'):
        _pa, _pb = (_k in ra), (_k in rb)
        if _pa != _pb:
            return f'unverifiable {_k} (only one record carries the precision / implementation mode)'
        if _pa and ra.get(_k) != rb.get(_k):
            return f'different {_k} ({ra.get(_k)!r} vs {rb.get(_k)!r})'
    # Same corpus / evaluation slice, not just the same geometry and token
    # count.  Mirrors exp_lib.pair_reason (a mixed-cap cache rebuild must never
    # be averaged into one paired delta).
    for _k in ('data_fp', 'val_fp', 'eval_fp', 'eval_subset_fp',
               'holdout_row_fp'):
        _pa, _pb = ra.get(_k), rb.get(_k)
        _ha = _pa is not None and str(_pa) != ''
        _hb = _pb is not None and str(_pb) != ''
        if _ha != _hb:
            return f'unverifiable {_k} (only one record carries the corpus / evaluation-slice fingerprint)'
        if _ha and str(_pa) != str(_pb):
            return f'different {_k} ({str(_pa)[:12]}… vs {str(_pb)[:12]}…)'
    # Matched initialization: same seed does not mean the two architectures
    # consumed the same RNG before the shared trunk.  Mirror exp_lib.pair_reason
    # so the report cannot pair a mismatched-init contrast that the statistics
    # layer refuses.
    _ia, _ib = ra.get('init_core_fp'), rb.get('init_core_fp')
    _ha = _ia is not None and str(_ia) != ''
    _hb = _ib is not None and str(_ib) != ''
    if _ha != _hb:
        return 'unverifiable init_core_fp (only one record carries the matched-initialization digest)'
    if _ha and str(_ia) != str(_ib):
        return (f'different init_core_fp ({str(_ia)[:12]}… vs {str(_ib)[:12]}…): '
                'the two arms did not start from the same shared-trunk initialization '
                'despite the paired seed')
    bad = [k for k in PAIR_STEP_KEYS if ra.get(k) is None or rb.get(k) is None or ra.get(k) != rb.get(k)]
    if not bad:
        return None
    absent = [k for k in bad if ra.get(k) is None and rb.get(k) is None]
    unknown = [k for k in bad if k not in absent and (ra.get(k) is None) != (rb.get(k) is None)]
    if unknown:
        return f'unverifiable {sorted(unknown)}'
    if absent:
        return f'unverifiable {sorted(absent)}'
    return f'disagreeing {sorted(bad)}'

def paired_row(a_records, b_records):
    common = sorted(set(a_records) & set(b_records))
    if not common:
        return None
    _by_tag = {}
    for _tk in common:
        _by_tag.setdefault(_tk[0] if isinstance(_tk, tuple) else '', []).append(_tk)
    if len(_by_tag) > 1:
        _ranked = sorted(_by_tag.items(), key=lambda kv: (-len(kv[1]), str(kv[0])))
        _keep_tag, _kept_keys = _ranked[0]
        print(f"[report] paired_row: the two arms share {len(_by_tag)} protocol conditions ({sorted(_by_tag)}) — one sign-flip test cannot pool deltas across protocols (that would fake extra seeds), so only the largest shared condition `{_keep_tag or '(untagged)'}` ({len(_kept_keys)} seed(s)) is tested; the rest ({sorted((t for t in _by_tag if t != _keep_tag))}) is left to its own per-condition section")
        common = _kept_keys
    dl, kept, skipped = ([], [], [])
    for s in common:
        reason = _pair_reason(a_records[s], b_records[s])
        if reason:
            skipped.append(s)
            continue
        if _nested_estimand(a_records[s]) != _nested_estimand(b_records[s]):
            # The two arms' primary values would come from different row
            # samples (holdout vs full validation); their difference is on
            # neither estimand.
            skipped.append(s)
            continue
        _da, _db = _primary_ppl(a_records[s]), _primary_ppl(b_records[s])
        if _da is None or _db is None:
            skipped.append(s)
            continue
        dl.append(_da - _db)
        kept.append(s)
    if kept and len({_nested_estimand(a_records[s]) for s in kept}) > 1:
        print('[report] paired_row: the paired seeds mix nested-holdout and '
              'full-validation estimands — their deltas are not one sample; '
              'comparison omitted')
        return None
    if skipped:
        print(f'[report] {len(skipped)}/{len(common)} seed(s) NOT paired ({skipped}); excluded from the sign-flip test')
    if not dl:
        print('[report] no usable pair left after the run_cfg/steps checks — comparison omitted rather than quoting a cross-config delta')
        return None
    # `common` is the paired plan; `dl` is the complete-case subset after
    # the run_cfg/estimand gate.  Passing the planned count makes a
    # complete-case-only p-value visible (and refuses the significance wording
    # in `_sig_gate`) instead of silently shrinking the panel.
    st = exact_signflip(dl, n_input=len(common))
    return (kept, dl, st)

def sec_header():
    return ('# CSA / HCA 受控机制研究 — 补充实验报告 (v7)\n\n'
            f'> 生成时间：{time.strftime("%Y-%m-%d %H:%M:%S")}\n'
            '> 参考论文：arXiv:2606.19348。**范围声明**：本仓库是小规模、从头训练、WikiText-103、固定训练预算下的受控研究，不是 DeepSeek-V4 的全量复现；结论不外推到该模型的规模、语料、长上下文训练或生产系统。\n'
            '> 定位：**受控机制研究**（controlled mechanism study），非论文全量复现。\n'
            '> **证据等级：探索性（exploratory）**。本报告的面板/种子/比较是在同一 validation split 上选择后观察的；'
            'Holm 校正只限制过度宣称，不能把这些结果变成预注册确认。确认性证据见 `confirmation_frozen.json` / v12。\n'
            '> 说明：本报告全部数字由 `build_report.py` 从 `results_*/`、`analysis_v7/`\n'
            '> 的落盘产物计算得到，无手填数值。\n\n'
            '本报告汇总补充实验。所有新变体\n'
            '（RoPE+QK-norm、dense-warmup、topk 扫描、m=1、长度外推）都经由 **完全相同的\n'
            '训练/优化器/LR/评测/TimeGuard 代码路径**（对 `exp_lib` 的 `make_layer_cfgs`/\n'
            '`SmallGPT` 做 monkey-patch 注入），以保证可比性。长上下文证据采用\n'
            '**wikitext 长度外推探针**（train@512 → eval 512..4096）：这是**外推探针，不是长上下文训练**；\n'
            '块数/选择率、位置编码与截断随长度同时变化，报告须并列这些混淆项。\n\n')

_MODULE_NOTE = '本模块只读：从 `results_*/` 与 `analysis_v7/` 的落盘产物计算\n报告全文，不 import torch（也不 import exp_lib）。上面那段\n模块说明文字以 `_MODULE_NOTE` 命名保留，不参与报告正文的渲染。'

def sec_p0r():
    rope = agg('results_lm_v7_rope')
    ab = agg('results_lm_v3_1500')
    if not rope or not ab:
        return '## P0-3 位置编码对照 — （无结果）\n\n'
    pairs = [('full', 'full_rope'), ('csa_fixed', 'csa_fixed_rope'), ('csa_dynamic', 'csa_dynamic_rope'), ('hybrid_fixed', 'hybrid_fixed_rope')]
    lines = ['## P0-3 位置编码错配是否为混淆项？（RoPE + QK-RMSNorm 对照）', '', '**待检验论断**：论文用 partial RoPE（末 64 维）+ 输出侧反向 RoPE(pos=−i) + core attention 前 Q/KV RMSNorm；仓库用 `nn.Embedding(max_seq,d)` 学习式绝对位置编码、无 QK-norm。怀疑 PE 错配造成了 sparse 落后。', '', '**做法**：新增 `*_rope` 变体族，seed 0/1/2 × 1500 步 × seq 512，其余训练配置与 absPE 面板完全一致。', '', '| 变体 | absPE (v3_1500) | RoPE+QKnorm (v7) | Δ(RoPE−abs) | params(abs/rope) |', '|---|---|---|---|---|']
    for a, b in pairs:
        ea, eb = (_agg_entry(ab, a), _agg_entry(rope, b))
        if ea is not None and eb is not None:
            pa, pb = (_ppl_value(ea), _ppl_value(eb))
            if pa is None or pb is None:
                print(f'[report] sec_p0r: `{a}`/`{b}` lacks a finite PPL aggregate — row omitted rather than printed as n/a arithmetic')
                continue
            lines.append(f'| `{a}` | {fm(pa)} {sp(ea.get('ppl_std'))} | {fm(pb)} {sp(eb.get('ppl_std'))} | **{pb - pa:+.2f}** | {_params_m(ea)} / {_params_m(eb)} |')
        else:
            print(f'[report] sec_p0r: `{a}`/`{b}` not both present (config-split or absent) — row omitted')
    lines += ['', '> 注：表中 Δ(RoPE−abs) 为两臂**各自 seed 集合的均值之差（非配对）**——且跨面板（absPE vs RoPE）本身没有同 seed 配对的意义；显著性主张一律以下文配对检验的口径为准。', '', '**组间差距（决定该论断是否成立）**：', '']
    _cf, _fu = (_agg_entry(ab, 'csa_fixed'), _agg_entry(ab, 'full'))
    _cf_r, _fu_r = (_agg_entry(rope, 'csa_fixed_rope'), _agg_entry(rope, 'full_rope'))
    _mt = _agg_entry(ab, 'full_matched')
    if _cf is not None and _cf_r is not None and (_fu_r is not None) and (_fu is not None):
        _means = [_ppl_value(_e) for _e in (_cf, _fu, _cf_r, _fu_r)]
        if any((m is None for m in _means)):
            print('[report] sec_p0r: one of the four arm entries lacks a finite PPL aggregate — group-gap block omitted')
            return '\n'.join(lines) + '\n'
        g_abs = _means[0] - _means[1]
        _mtm = _ppl_value(_mt)
        g_abs_m = _means[0] - _mtm if _mtm is not None else float('nan')
        g_rope = _means[2] - _means[3]
        d_abs = _means[1] - _means[3]
        d_sp = _means[0] - _means[2]
        _has_mt = _mtm is not None
        rope_unmatched = unmatched_tag(_cf_r, _fu_r)
        _pgap_r = param_gap(_cf_r, _fu_r)
        # Only a *known* mismatch counts as a defect: when params are missing
        # (`is_param_matched` returns None) keep the "params 未知" tag but do
        # not assert the structural-defect block.
        rope_is_defect = is_param_matched(_cf_r, _fu_r) is False
        _w_abs = '领先' if g_abs < 0 else '落后'
        _w_absm = '领先' if g_abs_m < 0 else '落后'
        _w_rope = '领先' if g_rope < 0 else '落后'
        _d_abs_t = ('改善' if d_abs > 0 else '变差') + f' **{abs(d_abs):.2f}**'
        _d_sp_t = ('改善' if d_sp > 0 else '变差') + f' **{abs(d_sp):.2f}**'
        _gain_conf = [f'dense 臂{unmatched_tag(_fu, _fu_r)}' for _ in [1] if unmatched_tag(_fu, _fu_r)] + [f'sparse 臂{unmatched_tag(_cf, _cf_r)}' for _ in [1] if unmatched_tag(_cf, _cf_r)]
        _gain_note = f'（注：{'；'.join(_gain_conf)}，对应对比同时含容量效应）' if _gain_conf else ''
        lines += [f'- absPE 面板：`csa_fixed` {_w_abs} dense **{abs(g_abs):.2f}** PPL（参数对齐 `full_matched` 时{_w_absm} **{abs(g_abs_m):.2f}**；两者相差 **{g_abs - g_abs_m:+.2f} PPL**，即容量差异贡献的部分）{unmatched_tag(_cf, _mt)}。' if _has_mt else f'- absPE 面板：`csa_fixed` {_w_abs} dense **{abs(g_abs):.2f}** PPL。⚠ **本面板没有参数对齐的 dense 臂**（`full_matched` 缺失），该差距因此**含未剥离的容量效应**，不能作为机制性结论的依据。', f'- RoPE 面板：`csa_fixed_rope` {_w_rope} `full_rope` **{abs(g_rope):.2f}** PPL{rope_unmatched}。', f'- RoPE 使 dense {_d_abs_t}、sparse {_d_sp_t}{_gain_note}。', '']
        if rope_unmatched and rope_is_defect and _has_mt:
            lines += [f'> **本表的限制（务必先读）**：RoPE 面板里**没有**参数对齐的 dense 基线——落盘记录显示 `full_rope` 的 MLP 停在标准宽度（{_params_m(_fu_r)}），而 `csa_fixed_rope` 保持全宽（{_params_m(_cf_r)}），{f'相差 {100 * _pgap_r:.1f}%' if _pgap_r is not None else '参数差未知'}。对照 absPE 面板，同样的容量差异会贡献约 {g_abs - g_abs_m:.1f} PPL。因此**上面 RoPE 的组间差距不能与 absPE 的组间差距直接相减**，「差距几乎不变」的读法在当前产物上不成立。需**重跑 P0R 面板**（得到参数对齐的 dense 臂）才能给出该结论；在那之前，**P0-3 的证伪只由 absPE 面板的参数对齐数字支持**。', '']
        _arms_gain = d_abs > 0 and d_sp > 0
        _gain_txt = 'RoPE+QK-norm 对两臂都有真实增益，值得保留为新默认' if _arms_gain else f'RoPE+QK-norm 并未对两臂都带来增益（dense {_d_abs_t}、sparse {_d_sp_t}）'
        _pair_mt = None
        if _has_mt:
            _ps_ab = per_seed_records('results_lm_v3_1500')
            _pair_mt = paired_row(_ps_ab.get('csa_fixed', {}), _ps_ab.get('full_matched', {}))
        if _has_mt:
            if _pair_mt is not None:
                _st_mt = _pair_mt[2]
                # Single pre-specified contrast in this block: Holm over a
                # one-element family is the raw p; the n>=6 / noise gate in
                # `_sig_gate` still applies.
                _st_mt['p_holm'] = _st_mt.get('p_exact_signflip')
                _mt_stat = f'配对符号翻转 n={_st_mt['n']}、p={_st_mt['p_exact_signflip']:.3f}（Δ mean={_st_mt['mean']:+.2f}）'
                _mt_sig = _sig_gate(_st_mt, label='P0R csa_fixed vs full_matched (param-matched absPE)', family='P0R')
                _mt_gap = _st_mt['mean']
            else:
                _mt_stat = '本面板没有可通过配对门禁的同配置种子对，显著性无法检验'
                _mt_sig = False
                _mt_gap = g_abs_m
            if _mt_gap > 0:
                if _mt_sig:
                    _p0r_concl = f'**结论**：PE 错配 **不是** sparse 劣势的来源。（1）{_gain_txt}；（2）参数对齐的 absPE 数字（`csa_fixed` 对 `full_matched`）仍落后且通过「n≥6 + Holm + 效应>噪声」门槛（{_mt_stat}），**不支持 PE 混淆解释**；但这是该受控面板上的证据，不外推为普遍机制。'
                else:
                    _p0r_concl = f'**结论（方向性，未确认）**：PE 错配的证据方向与「sparse 劣势」一致。（1）{_gain_txt}；（2）参数对齐的 absPE 数字（`csa_fixed` 对 `full_matched`）方向仍为落后（{_mt_stat}），但未通过显著性/族校正门槛，因此只能说**不能据此排除** PE 混淆，不能写成「混淆假设被证伪」或「机制性落后」。'
            else:
                _p0r_concl = f'**结论**：{_gain_txt}；但参数对齐的 absPE 数字中 `csa_fixed` 不再落后 `full_matched`（Δ={_mt_gap:+.2f} PPL，{_mt_stat}）——与原论断方向相反，**P0-3 的混淆假设在本轮未被证伪**，PE 错配不能排除在 sparse 的差距之外。'
        else:
            _p0r_concl = f'**结论（受限）**：`csa_fixed` 在 absPE 面板{_w_abs}未匹配的 `full`，{_gain_txt}。但**参数对齐的 absPE 臂（`full_matched`）在本面板缺失**，所以「容量差贡献了多少」无法剥离，**P0-3 的证伪在本轮没有证据支持**——本条只作方向性表述，须待 `full_matched` 产出后方可作结论。'
        lines += [_p0r_concl, '']
    return '\n'.join(lines) + '\n'

def sec_p0w():
    s = summary('results_lm_v7_warmup')
    if not s:
        return '## P0-1 dense→sparse warmup — （无结果）\n\n'
    _legacy_ok = _legacy_warm_tags_cached('results_lm_v7_warmup')
    rows = []
    for k, r in s.items():
        if not isinstance(r, dict):
            continue
        _w = _int_or(r.get('warm_steps'), None)
        if _w is None:
            _t = _tag_of(r, k, _legacy_ok)
            _w = int(_t[1:]) if _t.startswith('w') and _t[1:].isdigit() else -1
        _sd = _int_or(r.get('seed'), -1)
        if _sd < 0 and _measurable({'ppl': r.get('ppl'), 'synthesized': r.get('synthesized', False)}):
            print(f"[report] sec_p0w: record {k!r} carries a non-integer seed — it is skipped, not counted")
            continue
        if _w < 0 and _measurable({'ppl': r.get('ppl'), 'synthesized': r.get('synthesized', False)}):
            print(f"[report] sec_p0w: record {k!r} carries no parseable warmup tag — it is skipped, not counted")
            continue
        # The table must show the same primary estimand as the paired test
        # below: a nested warmup record's full-validation `ppl` still contains
        # the rows that selected its hyper-parameters.
        rows.append((r.get('variant', '?'), _w, _sd, _primary_ppl(r),
                     r.get('ppl_at_switch'), r.get('synthesized', False), k))
    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    rows = [row for row in rows if _measurable({'ppl': row[3], 'synthesized': row[5]})]
    rows = _prefer_one_cfg(rows, lambda row: ((row[0], row[1]), (s.get(row[-1]) or {}).get('run_cfg')), who='sec_p0w')
    _kept_keys = {row[-1] for row in rows}
    rec_by_s = collections.defaultdict(dict)
    for _k, _r in s.items():
        if _k not in _kept_keys:
            continue
        if not (isinstance(_r, dict) and _measurable(_r)):
            continue
        _sd = _int_or(_r.get('seed'), -1)
        if _sd < 0:
            continue
        _w = _int_or(_r.get('warm_steps'), None)
        if _w is None:
            _t = _tag_of(_r, _k, _legacy_ok)
            _w = int(_t[1:]) if _t.startswith('w') and _t[1:].isdigit() else -1
        _slot = rec_by_s[_r.get('variant', '?'), _w]
        if _sd in _slot:
            print(f"[report] sec_p0w: ambiguous (variant, warm_steps, seed) = ({_r.get('variant', '?')!r}, {_w}, {_sd}): two measurable records found; keeping the FIRST (key {_k!r} ignored)")
            continue
        _slot[_sd] = _r
    grp = collections.defaultdict(list)
    for (v, w), _m in rec_by_s.items():
        for _sd, _r in sorted(_m.items()):
            grp[v, w].append((_primary_ppl(_r), _r.get('ppl_at_switch'), _sd))
    lines = ['## P0-1 dense→sparse warmup 是否能让 sparse 追平 dense？（待检验项）', '', '**待检验论断**：论文 §4.2.2 先训 1T token 的 dense 再在 seq 64K 切 sparse；仓库所有 sparse run 都是 from-scratch，故「渐近线更差」可能是没 warmup 的产物。不给这个实验，「渐近线反转」立不住。', '', '**做法**：`train_warmup` 用**相同全长度 cosine LR**，前 `warm_steps` 步走 dense 前向（复用各 csa/hca 层自己的 `W_kvhead` per-token KV 路径 + 同一 sink，不引入新参数），到点切换为 sparse 并记录切换瞬间 PPL。网格 `warm ∈ {0,5000,10000}`（每格 seed 数以下表 n 列为准），20k 步、seq 512。**两臂参数/步数/token 数/初始 PPL/LR 曲线完全相同**，唯一差异是切点。', '', '| variant | warm_steps | PPL (mean±std) | n | 切换时 PPL |', '|---|---|---|---|---|']
    for (v, w), vals in sorted(grp.items()):
        ppls = [p for p, _s, _sd in vals]
        mean = sum(ppls) / len(ppls)
        std = sample_std(ppls)
        sws = [s for _p, s, _sd in vals if _ppl_ok(s)]
        sw_s = f'{sum(sws) / len(sws):.2f}' if sws else '—'
        lines.append(f'| `{v}` | {w} | {mean:.2f} ± {std:.2f} | {len(ppls)} | {sw_s} |')
    base = grp.get(('csa_fixed', 0))
    if base and len(base) >= 1:
        b = sum((p for p, _s, _sd in base)) / len(base)
        lines += ['', f'**from-scratch 基线**（warm=0）：{b:.2f} PPL。相对基线的变化：', '']
        for (v, w), vals in sorted(grp.items()):
            if v != 'csa_fixed' or w == 0:
                continue
            m = sum((p for p, _s, _sd in vals)) / len(vals)
            lines.append(f'- `{v}` warm={w}：{m:.2f} PPL（{m - b:+.2f} vs 基线，n={len(vals)}）')
        curves = collections.defaultdict(lambda: collections.defaultdict(list))
        _dropped_pts = 0
        for (v, w), _m in rec_by_s.items():
            for _sd, r in sorted(_m.items()):
                for step, pv in _hist_points(r):
                    if not _ppl_ok(pv):
                        _dropped_pts += 1
                        continue
                    curves[v, w][int(step)].append(float(pv))
        if _dropped_pts:
            lines += ['', f'> **⚠ 轨迹表剔除了 {_dropped_pts} 个非有限/非正的 PPL 曲线点**（单次失败的 eval），它们不再参与 seed 平均。', '']
        if curves and any((len(c) > 1 for c in curves.values())) and any((_v == 'csa_fixed' for _v, _w in curves)):
            steps_all = sorted({st for c in curves.values() for st in c})
            lines += ['', '**验证 PPL 轨迹（seed 平均）**：', '', '| step | ' + ' | '.join((f'w={w}' for _v, w in sorted(curves) if _v == 'csa_fixed')) + ' |', '|' + '---|' * (1 + len([1 for _v, w in curves if _v == 'csa_fixed']))]
            for st in steps_all:
                cells = []
                for _v, w in sorted(curves):
                    if _v != 'csa_fixed':
                        continue
                    vals = curves[_v, w].get(st)
                    cells.append(f'{sum(vals) / len(vals):.1f}' if vals else '—')
                lines.append(f'| {st} | ' + ' | '.join(cells) + ' |')
        _conc = ['', '**结论**：']
        _parts = []
        _ds = []
        _ds_sig = []
        for v, w in sorted(grp):
            if v != 'csa_fixed' or w == 0:
                continue
            _wm = rec_by_s.get((v, w), {})
            _bm = rec_by_s.get(('csa_fixed', 0), {})
            _shared = sorted(set(_wm) & set(_bm))
            _dd = []
            _gated = 0
            _est = set()
            for sd in _shared:
                reason = _pair_reason(_wm[sd], _bm[sd])
                if reason:
                    _gated += 1
                    continue
                if _nested_estimand(_wm[sd]) != _nested_estimand(_bm[sd]):
                    _gated += 1
                    continue
                _da, _db = _primary_ppl(_wm[sd]), _primary_ppl(_bm[sd])
                if _da is None or _db is None:
                    _gated += 1
                    continue
                _dd.append(_da - _db)
                _est.add(_nested_estimand(_wm[sd]))
            if len(_est) > 1:
                print(f'[report] sec_p0w `{v}` warm={w}: seeds mix nested/full '
                      'estimands — the sign-flip test is omitted rather than '
                      'pooling two samples')
                continue
            if _gated:
                print(f'[report] sec_p0w `{v}` warm={w}: {_gated}/{len(_shared)} seed(s) NOT paired (run_cfg/steps gate); excluded from the sign-flip test')
            _st = exact_signflip(_dd, n_input=len(_shared)) if _dd else None
            if _st is None:
                continue
            _d = _st['mean']
            # The two-sided exact/saddlepoint floor is 2/2**n for EVERY n;
            # the old >20 branch showed a Monte-Carlo-like 1/200001 instead.
            _floor = _st.get('p_floor')
            if _floor is None:
                _floor = _signflip_floor(_st.get('n'))
            _same_dir = all((x != 0 and (x > 0) == (_d > 0) for x in _dd))
            _gate_ok = _sig_gate(_st, label=f'P0W csa_fixed warm={w} vs warm=0', family='P0W')
            if not _gate_ok:
                _verdict = f'n={_st['n']} 时 p={_st['p_exact_signflip']:.4f}（检验下限 {_floor:.3f}），**不构成显著/未过族校正**，' + ('各 seed 方向一致，仅作方向性读数' if _same_dir else '方向不一致')
            else:
                _verdict = f'n={_st['n']} 时 p={_st['p_exact_signflip']:.4f}、p_Holm={_st.get("p_holm", float("nan")):.4f}，**显著**'
            _parts.append(f'`{v}` warm={w}：{_d:+.2f} PPL vs from-scratch 基线（{_verdict}）')
            _ds.append(_d)
            _ds_sig.append(_gate_ok)
        if _parts:
            _conc.append('；'.join(_parts) + '。')
        else:
            _conc.append('（该面板无可配对的可测量记录，无法给出结论。）')
        if _ds and all((x > 0 for x in _ds)):
            if any(_ds_sig):
                _conc.append('即在该受控网格上 dense 预热未带来可检测的改善；这支持「from-scratch 对比不是靠预热缺失制造的优势」，但不足以写「证伪」或「负结果更稳固」（需预注册确认）。')
            else:
                _conc.append('各 warm 档方向一致（预热更差），dense 预热在方向上**未能**拯救 sparse 臂，但没有一档达到配对检验的显著性（分辨率见上）——**P0-1 的质疑按方向不成立，按显著性则既不能证伪也不能确认**，须补更多 seeds 后再判。')
        elif _ds and all((x < 0 for x in _ds)):
            if any(_ds_sig):
                _conc.append('即 dense 预热对 sparse 臂**有**真实帮助——**P0-1 的质疑成立**，from-scratch 对比不利于 sparse，相关负结果须按带 warmup 的设定重审。')
            else:
                _conc.append('各 warm 档方向一致（预热更好），dense 预热在方向上对 sparse 臂有帮助，但没有一档达到配对检验的显著性（分辨率见上）——**P0-1 质疑既不能证伪也不能确认**，须补更多 seeds 后再判。')
        elif _ds:
            _conc.append('各 warm 档的方向不一致，P0-1 质疑既不能证伪也不能确认，须补更多 seeds/档位后再判。')
        _conc.append('')
        lines += _conc
    return '\n'.join(lines) + '\n'

def sec_p0e():
    d = agg('results_lm_v7_long40')
    if not d:
        return '## P0-2 渐近线是否反演（40k 长跑）— （无结果）\n\n'
    s = summary('results_lm_v7_long40')
    # The headline token count is recomputed from the persisted records
    # (tokens_seen / steps), never hard-coded.
    _tok_hdr = 'token 数未知'
    for _k, _r in (s or {}).items():
        if isinstance(_r, dict) and _measurable(_r) and isinstance(_r.get('tokens_seen'), (int, float)) and _r['tokens_seen'] > 0:
            _tok_hdr = f'~{_r['tokens_seen'] / 1e6:.0f}M tokens'
            break
    lines = [f'## P0-2 「渐近线更差」是否成立？（延长到 40k 步 / {_tok_hdr}）', '', '**待检验论断**：20k 步时两臂都未平台化、差距在收窄。要么延长，要么改述为「等训练量下收敛更慢」。', '', '**做法**：`csa_fixed` + `full`（seed 数见下表），同 LONG 配置延到 40k 步（seq 512、bs 12、同一 110M token 池、同一 cosine LR 全长度；eval_every=2000 记录全程轨迹）。', '', '| variant | PPL@40k (pooled/mean; ±seed sd) | n | params |', '|---|---|---|---|']
    for v in sorted(d):
        r = d[v]
        lines.append(f'| `{v}` | {fm(_ppl_value(r))} {sp(r.get('ppl_std'))} | {r.get('n_seeds', r.get('n', '?'))} | {_params_m(r)} |')
    _s_items = _prefer_one_cfg([_kv for _kv in s.items() if isinstance(_kv[1], dict) and 'variant' in _kv[1] and _measurable(_kv[1])], lambda kv: (kv[1].get('variant'), kv[1].get('run_cfg')), who='sec_p0e')
    trajs = collections.defaultdict(lambda: collections.defaultdict(dict))
    _dropped_pts = 0
    for _k, r in _s_items:
        if not isinstance(r, dict) or 'variant' not in r:
            continue
        if not _measurable(r):
            continue
        _sdv = str(r.get('seed'))
        for step, pv in _hist_points(r):
            if not _ppl_ok(pv):
                _dropped_pts += 1
                continue
            _cell = trajs[r['variant']][int(step)]
            if _sdv in _cell:
                print(f"[report] sec_p0e: ambiguous (variant, step, seed) = ({r['variant']!r}, {int(step)}, {_sdv}): two measurable points found; keeping the FIRST")
                continue
            _cell[_sdv] = float(pv)
    if _dropped_pts:
        lines += ['', f'> **⚠ 轨迹表剔除了 {_dropped_pts} 个非有限/非正的 PPL 曲线点**（单次失败的 eval），它们不再参与 seed 平均、尾段斜率与下方结论的判定。', '']
    if 'csa_fixed' in trajs and 'full' in trajs:
        steps = sorted(set(trajs['csa_fixed']) & set(trajs['full']))
        if steps:
            lines += ['', '**验证 PPL 轨迹（seed 平均，每 2000 步）**：', '', '| step | csa_fixed | full | gap (csa−full) |', '|---|---|---|---|']
            gaps = {}
            for st in steps:
                c = sum(trajs['csa_fixed'][st].values()) / len(trajs['csa_fixed'][st])
                f_ = sum(trajs['full'][st].values()) / len(trajs['full'][st])
                gaps[st] = c - f_
                lines.append(f'| {st} | {c:.1f} | {f_:.1f} | **{gaps[st]:+.1f}** |')
            tail = steps[-5:] if len(steps) >= 5 else steps
            _tail_span_k = (steps[-1] - tail[0]) / 1000.0
            _paired_raw = sorted(set.intersection(*(set(trajs[v][st]) for v in ('csa_fixed', 'full') for st in tail))) if tail else []
            _all_seeds = sorted(set.union(*(set(trajs[v][st]) for v in ('csa_fixed', 'full') for st in tail))) if tail else []
            _incomplete = [sd for sd in _all_seeds if sd not in set(_paired_raw)]
            if _incomplete:
                print(f'[report] sec_p0e: seed(s) {_incomplete} miss at least one arm x tail-step cell and are excluded from the paired tail gap — pairing requires a measurement on EVERY tail step for BOTH arms')
            _rec_of = {}
            for _k, r in _s_items:
                if isinstance(r, dict):
                    _rec_of.setdefault((r.get('variant'), str(r.get('seed'))), r)
            _paired = []
            _pair_gated = []
            for _sd in _paired_raw:
                _ra = _rec_of.get(('csa_fixed', _sd))
                _rb = _rec_of.get(('full', _sd))
                _why = _pair_reason(_ra, _rb) if _ra is not None and _rb is not None else 'record missing'
                if _why is None:
                    _paired.append(_sd)
                else:
                    _pair_gated.append((_sd, _why))
            if _pair_gated:
                print(f'[report] sec_p0e: {len(_pair_gated)} shared seed(s) FAILED the run_cfg/steps gate ({sorted({w for _s, w in _pair_gated})}) and are excluded from the paired tail-gap — the pairing rules are the same as every other section')
            _unpaired_gap = not _paired
            if _unpaired_gap:
                print('[report] sec_p0e: the two arms share NO seed on the tail evaluation steps — the tail-gap slope and extrapolation below use the UNPAIRED per-step mean gap (each step averaged over whichever seeds measured it), not a paired per-seed difference; read it as indicative only')
            xs = [st / 1000.0 for st in tail]
            if _paired:
                ys = [sum((trajs['csa_fixed'][st][sd] - trajs['full'][st][sd] for sd in _paired)) / len(_paired) for st in tail]
            else:
                ys = [gaps[st] for st in tail]
            g_last = ys[-1]
            n_ = len(xs)
            mx, my = (sum(xs) / n_, sum(ys) / n_)
            slope = sum(((x - mx) * (y - my) for x, y in zip(xs, ys))) / sum(((x - mx) ** 2 for x in xs)) if n_ > 1 else 0.0

            def rate(var):
                if _paired:
                    yv = [sum((trajs[var][st][sd] for sd in _paired)) / len(_paired) for st in tail]
                else:
                    yv = [sum(trajs[var][st].values()) / len(trajs[var][st]) for st in tail]
                my2 = sum(yv) / n_
                return sum(((x - mx) * (y - my2) for x, y in zip(xs, yv))) / sum(((x - mx) ** 2 for x in xs)) if n_ > 1 else 0.0
            r_csa, r_full = (rate('csa_fixed'), rate('full'))

            def _gap_at(step):
                v = gaps.get(step)
                return f'**{v:+.1f}**' if isinstance(v, (int, float)) and math.isfinite(v) else '**未测量**（该面板无此步）'
            _g20 = _gap_at(20000)
            def _dir_word(r):
                return '下降' if r < 0 else ('上升' if r > 0 else '持平')
            _lastk = steps[-1] / 1000.0
            _tps = next((r['tokens_seen'] / r['steps'] for _k, r in _s_items if isinstance(r, dict) and isinstance(r.get('tokens_seen'), (int, float)) and isinstance(r.get('steps'), (int, float)) and r['steps']), None)
            _tok_m = f'{steps[-1] * _tps / 1e6:.0f}M' if _tps else 'token 数未知'
            if _unpaired_gap:
                _gap_calib = '非配对逐步均值差（两臂尾段无公共 seed，仅作方向性参考）'
            elif _pair_gated:
                _gap_calib = f'尾段斜率与终点差距为过门禁配对子集（{len(_paired)}/{len(_paired) + len(_pair_gated)} 个公共 seed；上表逐行仍是逐步全 seed 均值）'
            else:
                _gap_calib = 'seed 平均'
            lines += ['', f'**差距轨迹分析（尾段 {tail[0]:g}–{steps[-1]:g} 步，跨度 {_tail_span_k:g}k，{_gap_calib}）**：', '', f'- 20k 步差距：{_g20} PPL；{_lastk:g}k 步差距：**{g_last:+.1f}** PPL。', f'- 尾段差距斜率：**{slope:+.2f} PPL / 1k steps**（csa {_dir_word(r_csa)} {abs(r_csa):.2f}、dense {_dir_word(r_full)} {abs(r_full):.2f} PPL/1k）。']
            _n_paired_seeds = len(_paired)
            if _n_paired_seeds and _n_paired_seeds < 6:
                verdict = (f'**结论（描述性，n={_n_paired_seeds}）：本轮不对「渐近线是否反演/坐实」作判定。** '
                           f'通过配对门禁的公共 seed 只有 {_n_paired_seeds} 个（精确符号翻转在 n<6 时的最小可达 p 已 >0.05），'
                           f'且 40k 步的 cosine LR 已衰减到底、没有学习率续训或独立复现；'
                           f'因此只能报告「{_lastk:g}k 步窗口内终点差距 {g_last:+.1f} PPL、尾段斜率 {slope:+.2f} PPL/1k」，'
                           '不能写成「渐近线更差」或「渐近线反转」。')
            elif _unpaired_gap:
                verdict = f'**结论（受限）：本轮不对「渐近线是否反演」作判定。** 两臂在尾段评测步上没有任何通过配置/训练量配对门禁的公共 seed，上方尾段斜率（{slope:+.2f} PPL/1k）与终点差距（{g_last:+.1f} PPL）是**非配对逐步均值差**——每个评测步由各自测得的 seed 平均，seed 组成随步变化，组成变化本身即可伪造或掩盖交叉。需补齐两臂共享 seed 的长跑记录后再按同一判据重算。'
            elif n_ < 2:
                verdict = f'**结论：尾段差距统计量不可用**（两臂只在 {n_} 个公共评测步上有测量，尾段回归需要至少 2 个点），**本轮不给出渐近线读法**——这不是「差距不再收窄」，是**无法判定**。需补跑或加密尾段评测网格。'
            elif not (math.isfinite(g_last) and math.isfinite(slope)):
                verdict = f'**结论：尾段差距统计量不可用**（`g_last` 或 `slope` 非有限值：g_last={g_last!r}、slope={slope!r}），**本轮不给出渐近线读法**——这不是「差距不再收窄」，是**无法判定**。上表已剔除非有限的曲线点；若此处仍出现，说明该面板的尾段整体缺失，需补跑。'
            elif g_last <= 0:
                verdict = f'**结论：渐近线在 {_lastk:g}k 内反演**——`csa_fixed` 追平并超过 dense，P0-2 的质疑成立，此前「worse asymptote」的表述需撤回。'
            elif slope < -1e-06:
                x0 = steps[-1] - g_last * 1000.0 / slope
                if steps[-1] < x0 <= steps[-1] * 10:
                    _g20v = gaps.get(20000)
                    _g20c = f'由 20k 的 **{_g20v:+.1f}** ' if isinstance(_g20v, (int, float)) and math.isfinite(_g20v) else ''
                    verdict = f'- 按尾段斜率线性外推，差距将在 ~{x0 / 1000:.0f}k 步附近归零（外推仅供参考：学习率已 cosine 衰减到底，后期斜率通常进一步放缓）。\n\n**结论**：{_lastk:g}k 步内未发生反演，但差距仍在缓慢收窄。**保守表述**：「等 token 量下 CSA 收敛更慢，终点差距 {_g20c}收窄到 {_lastk:g}k 的 {g_last:+.1f}，未见交叉」。是否最终追平属外推，不属证据。'
                else:
                    verdict = f'- 线性外推的交叉点在 ~{x0 / 1000:.0f}k 步，超出可信外推范围。\n\n**结论**：{_lastk:g}k 步（~{_tok_m} tokens）仍未追平——终点差距 **{g_last:+.1f} PPL**。差距收窄速度在尾段为 {abs(slope):.2f} PPL/1k，即每多训 10k 步约收窄 {abs(slope) * 10:.0f} PPL。**在本轮 {_lastk:g}k 步窗口内差距没有消失**；这支持「等训练量下收敛更慢」，不足以判定收敛后的渐近线（缺学习率续训与独立复现）。'
            else:
                verdict = f'**结论**：尾段差距已不再收窄（斜率 {slope:+.2f} PPL/1k），**在本轮 {_lastk:g}k 步窗口内差距已不再收窄**——这是窗口内读数，不是收敛后的渐近线结论。'
            lines += ['', verdict, '']
    return '\n'.join(lines) + '\n'

def sec_p1t():
    d = agg('results_lm_v7_seq2k')
    if not d:
        return '## P1 选择率错配 / m=1 对照（seq 2048 topk 扫描）— （无结果）\n\n'
    order = ['csa_fix_m1', 'csa_fixed_topk8', 'csa_fixed_topk32', 'csa_fixed_topk128', 'csa_fixed_topk512', 'csa_fixed']
    lines = ['## P1 选择率错配 与 m=1(纯 DSA) 对照（seq 2048, topk 扫描）', '', '**待检验论断**：仓库 seq 512 时选择率 25%，论文在长序列约 0.2%。`topk` 是主控旋钮，需要扫描；且缺 m=1（纯 DSA，不压缩）对照。', '', '**做法**：seq 2048、1500 步（seed 数见下表），扫描 `topk ∈ {8,32,128,512}` 并加入 `csa_fix_m1`（block_size=1, overlap=0 → 逐 token 选择 = 纯 DSA）。', '', '| variant | PPL (mean±std) | n | 等效选择率 (topk / 实测块数) |', '|---|---|---|---|']
    # Selection rates are computed from the *measured* per-sequence block
    # counts persisted in each run's `stats` (never assumed): for a fixed
    # block size m the panel has n_blocks = seq_len / m blocks, and the
    # indexer's effective selection is min(topk, n_blocks) / n_blocks.
    # `topk` itself is parsed from the sweep variant names; the two
    # non-sweep rows use the drivers' configured constants
    # (csa_fix_m1 / csa_fixed both run with index_topk=32).
    _s1t = summary('results_lm_v7_seq2k') or {}

    _nblocks_cache = {}

    def _nblocks_measured(base_variant):
        # memoized: the summary table is not mutated inside sec_p1t, and this
        # function is a pure full-table scan per variant.
        if base_variant in _nblocks_cache:
            return _nblocks_cache[base_variant]
        _vals = []
        for _rid, _r in _s1t.items():
            if not isinstance(_r, dict) or _r.get('variant') != base_variant:
                continue
            if not _measurable(_r):
                continue
            for _e in (_r.get('stats') or {}).values():
                if isinstance(_e, dict) and isinstance(_e.get('blocks'), (int, float)) and _e['blocks'] > 0:
                    _vals.append(float(_e['blocks']))
        _out = (sum(_vals) / len(_vals)) if _vals else None
        _nblocks_cache[base_variant] = _out
        return _out

    def _sr(topk, base_variant, note=''):
        # 注意：选择率的分子 topk 是设计常量（不落盘于 run 记录），分母块数
        # 才是实测。若 exp_lib 中相应的 index_topk 默认值更改，这里必须同步。
        if note:
            print(f'[report] sec_p1t WARNING: selection rate for {base_variant!r} uses the DESIGN topk (not a measured value) — verify it matches exp_lib.make_layer_cfgs')
        _nb = _nblocks_measured(base_variant)
        if _nb is None:
            print(f'[report] sec_p1t: no measured block count for {base_variant!r} — selection-rate cell left blank rather than hard-coded')
            return '—'
        return f'{min(topk, _nb) / _nb * 100:.5g}%{note}'
    seen = set()
    for v in order + sorted(d):
        if v in seen or v not in d:
            continue
        seen.add(v)
        r = d[v]
        ppls = _finite_ppls(r)
        if ppls:
            mean = sum(ppls) / len(ppls)
            std = sample_std(ppls)
            n = len(ppls)
        elif r.get('ppls') or r.get('ppl_list'):
            mean, std, n = (None, None, 0)
        else:
            mean, std, n = (_ppl_value(r), r.get('ppl_std'), r.get('n_seeds') if r.get('n_seeds') is not None else r.get('n', 0))
        _bv = v.split('@')[0].split('#')[0]
        if _bv == 'csa_fix_m1':
            sr = _sr(32, _bv, ' (m=1, topk=32)')
        elif _bv.startswith('csa_fixed_topk'):
            _kv = _bv.replace('csa_fixed_topk', '')
            try:
                k = int(_kv)
            except ValueError:
                print(f'[report] sec_p1t: cannot parse topk from aggregate key {v!r} — row omitted')
                continue
            sr = _sr(k, _bv)
        elif _bv == 'csa_fixed':
            sr = _sr(32, _bv, ' (default topk=32)')
        else:
            sr = '—'
            print(f'[report] sec_p1t: aggregate key {v!r} is not one of the known sweep variants — its selection-rate cell is left blank rather than guessed')
        lines.append(f'| `{v}` | {fm(mean)} ± {fm(std)} | {n} | {sr} |')
    s = summary('results_lm_v7_seq2k')
    failed = {}
    _measured = set()
    for rid, r in s.items():
        if not isinstance(r, dict):
            continue
        if r.get('error'):
            v = r.get('variant', rid.split('::')[0])
            failed.setdefault(v, []).append(r.get('seed'))
        elif _measurable(r):
            _measured.add(r.get('variant', rid.split('::')[0]))
    _grid = ['csa_fix_m1', 'csa_fixed_topk8', 'csa_fixed_topk32', 'csa_fixed_topk128', 'csa_fixed_topk512']
    missing = [v for v in _grid if v not in _measured and v not in failed]
    for v in missing:
        print(f'[report] sec_p1t: sweep point `{v}` has NO record in results_lm_v7_seq2k (never ran, cap-skipped, or truncated) — it is absent from the table and is listed below')
    if failed:
        lines += ['', '**未完成的扫描点（如实记录）**：', '']
        for v in sorted(failed):
            seeds = ','.join((str(x) for x in sorted(failed[v], key=lambda x: (x is None, 0 if isinstance(x, (int, float)) and (not isinstance(x, bool)) else 1, x if isinstance(x, (int, float)) and (not isinstance(x, bool)) else str(x)))))
            lines.append(f'- `{v}`（seed {seeds}）：该扫描点在本配置与运行时长约束下未完成（记录为 error），扫描被截断。')
        lines.append('')
    if missing:
        lines += ['', '**无记录的扫描点（如实记录）**：', '']
        for v in missing:
            lines.append(f'- `{v}`：面板上没有任何记录（未运行、被时长门禁跳过或截断后未留下测量），扫描不完整。')
        lines.append('')
    ks = [8, 32, 128, 512]
    rows_k = []
    for k in ks:
        r = _agg_entry(d, f'csa_fixed_topk{k}')
        if r is not None:
            ppls = _finite_ppls(r)
            m = sum(ppls) / len(ppls) if ppls else _ppl_value(r)
            if isinstance(m, (int, float)):
                rows_k.append((k, m))
    if len(rows_k) >= 3:
        lines += ['', '**剂量-反应（topk → PPL）**：', '']
        mono = all((rows_k[i][1] >= rows_k[i + 1][1] - 1e-09 for i in range(len(rows_k) - 1)))
        for k, m in rows_k:
            _nb = _nblocks_measured(f'csa_fixed_topk{k}')
            _sr_txt = f'{min(k, _nb) / _nb * 100:.5g}%' if _nb is not None else 'n/a (未测得块数)'
            lines.append(f'- topk={k}（选择率 {_sr_txt}）：{m:.2f} PPL')
        lines.append('')
        if mono:
            lines.append('PPL 随选择率单调下降（选择越多越好），说明在 seq 2048 / 1500 步的受控训练量下未出现「少选反而优」的甜点——检索质量不足时，选择率是硬上限。')
        else:
            lines.append('存在非单调点：存在一个「少选反而更好/更差」的转折（见上表），提示 topk 存在训练量相关的最优值。')
        lines.append('')
    m1 = _agg_entry(d, 'csa_fix_m1')
    t8 = _agg_entry(d, 'csa_fixed_topk8')
    t32 = _agg_entry(d, 'csa_fixed_topk32')

    def _mean(r):
        if not r:
            return None
        ppls = _finite_ppls(r)
        return sum(ppls) / len(ppls) if ppls else _ppl_value(r)
    m1m, t8m, t32m = (_mean(m1), _mean(t8), _mean(t32))

    lines += ['', '**结论（基于已完成的扫描点）**：', '']
    _m1_ok = m1m is not None
    _t_ok = t8m is not None and t32m is not None
    if _m1_ok and _t_ok:
        _recs3 = per_seed_records('results_lm_v7_seq2k')
        _m1_r, _t8_r, _t32_r = (_recs3.get('csa_fix_m1', {}), _recs3.get('csa_fixed_topk8', {}), _recs3.get('csa_fixed_topk32', {}))
        _c3 = []
        _rej3 = 0
        _vals3 = {}
        _est3 = set()
        for sk in sorted(set(_m1_r) & set(_t8_r) & set(_t32_r)):
            if _pair_reason(_m1_r[sk], _t8_r[sk]) is not None or _pair_reason(_m1_r[sk], _t32_r[sk]) is not None:
                _rej3 += 1
                continue
            if len({_nested_estimand(_m1_r[sk]), _nested_estimand(_t8_r[sk]),
                    _nested_estimand(_t32_r[sk])}) > 1:
                _rej3 += 1
                continue
            _va, _vb, _vc = _primary_ppl(_m1_r[sk]), _primary_ppl(_t8_r[sk]), _primary_ppl(_t32_r[sk])
            if _va is None or _vb is None or _vc is None:
                _rej3 += 1
                continue
            _c3.append(sk)
            _vals3[sk] = (_va, _vb, _vc)
            _est3.add(_nested_estimand(_m1_r[sk]))
        if len(_est3) > 1:
            print('[report] sec_p1t: three-way seeds mix nested/full estimands — '
                  'the per-seed direction count is omitted')
            _c3 = []
            _vals3 = {}
            _rej3 += 1
        if _rej3:
            print(f'[report] sec_p1t: {_rej3} three-way shared seed(s) rejected by the run_cfg/steps/primary-estimand gate — excluded from the per-seed direction count')
        _m1_s = {sk: float(_vals3[sk][0]) for sk in _c3}
        _t8_s = {sk: float(_vals3[sk][1]) for sk in _c3}
        _t32_s = {sk: float(_vals3[sk][2]) for sk in _c3}
        if _c3:
            _nw = sum((1 for sk in _c3 if _m1_s[sk] > _t8_s[sk] and (_m1_s[sk] > _t32_s[sk])))
            _nb = sum((1 for sk in _c3 if _m1_s[sk] < _t8_s[sk] and (_m1_s[sk] < _t32_s[sk])))
            _dir3 = f'（逐 seed：{len(_c3)} 个通过门禁的三方公共 seed 中，{_nw} 个 m1 最差、{_nb} 个 m1 最好）'
        else:
            _dir3 = '（无通过门禁的三方公共 seed，逐 seed 方向不可核验）'
        if m1m > t8m and m1m > t32m:
            lines.append(f'1. **m=1（纯 DSA、不压缩）是三点中最差的**（{fm(m1m)} vs topk8 {fm(t8m)} / topk32 {fm(t32m)}{_dir3}）：去掉压缩并没有拯救 sparse 臂——在 seq 2048 / 1500 步的受控训练量下，m=1 的三点均值最差；但这是未配对、未做族校正的 3 点排序，只能支持「本轮压缩正则不是唯一瓶颈」的**方向性**表述，不能写成「压缩不是瓶颈」的机制结论。')
        elif m1m < t8m and m1m < t32m:
            lines.append(f'1. **m=1（纯 DSA、不压缩）是三点中最好的**（{fm(m1m)} vs topk8 {fm(t8m)} / topk32 {fm(t32m)}{_dir3}）：去掉压缩反而占优——在 seq 2048 / 1500 步的受控训练量下，这是支持「压缩是瓶颈之一」的方向性证据，但三点均值未做配对/族校正，不能单独支撑机制结论。')
        else:
            lines.append(f'1. m=1（{fm(m1m)}）介于 topk8（{fm(t8m)}）与 topk32（{fm(t32m)}）之间，三点排序非单调{_dir3}；「缺 m=1 对照」的质疑得到直接回答，但压缩是否瓶颈需结合显著性判断，此处只作方向性表述。')
    elif _m1_ok:
        lines.append(f'1. m=1 测得 {fm(m1m)}，但 topk8/topk32 至少其一缺失，三点排序不可比——本条只陈述事实，不作「压缩是否瓶颈」的读法。')
    else:
        lines.append('1. m=1 未测量（该扫描点缺失或失败），三点排序不可比——本条只陈述事实，不作「压缩是否瓶颈」的读法。')
    if _t_ok:
        if t8m <= t32m:
            _dir = f'topk8 {fm(t8m)} ≤ topk32 {fm(t32m)}：选得更少反而略好，与论文「长序列下低选择率足够」的设计方向一致'
        else:
            _dir = f'topk8 {fm(t8m)} > topk32 {fm(t32m)}：选得更少反而更差，整体为单调上升（选择越多越好）'
        _gap = abs(t32m - t8m)
        _recs = per_seed_records('results_lm_v7_seq2k')
        _pair = paired_row(_recs.get('csa_fixed_topk8', {}), _recs.get('csa_fixed_topk32', {}))
        if _pair is not None:
            _kept, _dl, _st = _pair
            _sig_txt = '显示该差距**达到 p≤0.05 且通过族校正/噪声门槛**，按显著性读法处理。' if _sig_gate(_st, label='P1T csa_fixed_topk8 vs topk32 (seq2048)', family='P1T') else '显示该差距未通过「n≥6 + 原始 p<0.05 + Holm + 效应大于 seed 噪声」门槛，**不构成显著性主张**，只作方向性参考。'
            lines.append(f'2. 在已完成的两个选择率点上：{_dir}；配对的精确符号翻转检验（n={_st['n']}，p={_st['p_exact_signflip']:.3f}，Δ(topk8−topk32) mean={_st['mean']:+.2f} PPL）{_sig_txt}')
        else:
            lines.append(f'2. 在已完成的两个选择率点上：{_dir}；面板上没有可通过 pairing 门禁的同配置种子对，topk8 与 topk32 之差（{_gap:.2f} PPL）无法配对检验，只作方向性参考。')
    else:
        lines.append('2. topk8/topk32 至少其一缺失，两个选择率点之间的对比不可算。')
    if failed or missing:
        _miss = '、'.join((f'`{v}`' for v in sorted(set(failed) | set(missing))))
        lines.append(f'3. 未完成的点（{_miss}，见上）需后续在完整扫描面板补齐，「甜点位置」的完整刻画以补齐后的面板为准。')
    else:
        lines.append('3. 全部扫描点均已完成，「甜点位置」以本面板数据为准。')
    return '\n'.join(lines) + '\n'
