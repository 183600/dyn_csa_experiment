

def paired_nll_unit_influence(nll_a, nll_b, n_tokens, *, unit_ids=None,
                              conf=0.95, top_k=5):
    """Leave-one-unit-out influence on the paired token-pooled PPL difference.

    ``_paired_nll_doc_ci`` answers "how wide is the document-sampling
    interval?" with a delta-method variance.  It does not answer the equally
    important result-analysis question: **is the reported effect carried by a
    single evaluation unit?**  A mean and an interval can both look stable
    while one article supplies most of the difference; removing that article
    would then reverse (or erase) the conclusion.  This function makes the
    influence explicit and deterministic:

      * exact leave-one-unit-out (LOO) recomputation of the SAME estimand
        (mean over seeds of the token-pooled PPL difference), not a
        re-weighting of equal per-unit means;
      * the delta-method influence vector already used by the cluster-robust
        variance, exposed for inspection;
      * a delete-one-unit jackknife SE/CI and its ratio to the cluster-robust
        SE (a large ratio is a high-leverage-unit warning);
      * direction stability over all LOO folds and a ``single_unit_dominates``
        flag when dropping one unit changes the log effect by at least 100%;
      * the ``top_k`` most influential units by absolute LOO change.

    ``unit_ids`` labels the columns (an article/document id); it is required to
    be the same length as the units and unique after string coercion.  When it
    is ``None`` deterministic ``unit_<i>`` labels are used.

    This is a **post-hoc descriptive** diagnostic.  It conditions on the
    trained seeds (seed uncertainty remains the exact sign-flip test), it does
    not turn units into a randomisation, and it never changes a p-value, Holm
    family or frozen decision.  Dropping a unit also changes the target slice
    to "the frozen slice minus that unit"; that is the point of the readout,
    not a claim about the population.
    """
    if isinstance(conf, (bool, np.bool_)):
        raise ValueError('paired_nll_unit_influence: conf must be a probability in (0, 1)')
    try:
        conf = float(conf)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('paired_nll_unit_influence: conf must be a probability in (0, 1)') from None
    if not (0.0 < conf < 1.0):
        raise ValueError('paired_nll_unit_influence: conf must be a probability in (0, 1)')
    _top_k = _as_seed_int(top_k)
    if _top_k is None or _top_k < 1:
        raise ValueError('paired_nll_unit_influence: top_k must be a positive integer')
    top_k = _top_k

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
            raise ValueError(f'{name}: expected [n_seeds, n_units], got shape {arr.shape}')
        if not np.all(np.isfinite(arr)):
            raise ValueError(f'{name}: all per-unit NLLs must be finite')
        return arr

    A = _matrix(nll_a, 'paired_nll_unit_influence(nll_a)')
    B = _matrix(nll_b, 'paired_nll_unit_influence(nll_b)')
    if A.shape != B.shape:
        raise ValueError(
            f'paired_nll_unit_influence: arm shapes differ ({A.shape} vs {B.shape}); '
            'a paired readout needs the same seeds and units')
    try:
        _raw_t = np.asarray(n_tokens).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('paired_nll_unit_influence: n_tokens must be numeric') from None
    if _contains_bool_value(n_tokens) or _raw_t.dtype.kind == 'b':
        # A mixed [True, 1.0] list coerces to a float array before
        # `_exact_count` sees it; reject the flag at the source.
        raise ValueError('paired_nll_unit_influence: token counts must be integers, not booleans')
    # Token counts are exact integers; a fractional or boolean count would
    # otherwise change the pooled denominator without any measurement meaning.
    _counts = [_exact_count(v, minimum=1) for v in _raw_t.tolist()]
    if any(c is None for c in _counts):
        raise ValueError('paired_nll_unit_influence: every unit needs a finite positive integer token count')
    T = np.asarray(_counts, dtype=float)
    if T.size != A.shape[1]:
        raise ValueError(
            f'paired_nll_unit_influence: {T.size} token counts for {A.shape[1]} units')

    S, D = A.shape
    if D < 3:
        return {'kind': 'paired_unit_influence', 'available': False,
                'n_units': int(D), 'n_seeds': int(S),
                'reason': ('a delete-one-unit jackknife/influence readout '
                           'needs at least three units')}
    if isinstance(unit_ids, (str, bytes, bytearray)):
        raise ValueError('paired_nll_unit_influence: unit_ids must be an iterable, not a string')
    if unit_ids is None:
        labels = [f'unit_{i}' for i in range(D)]
    else:
        try:
            labels = [str(x) for x in unit_ids]
        except TypeError:
            raise ValueError('paired_nll_unit_influence: unit_ids must be an iterable') from None
        if len(labels) != D:
            raise ValueError(
                f'paired_nll_unit_influence: {len(labels)} unit ids for {D} units')
        if len(set(labels)) != D:
            raise ValueError('paired_nll_unit_influence: unit ids must be unique')

    total_tok = float(T.sum())
    if not math.isfinite(total_tok) or total_tok <= 0.0:
        raise ValueError('paired_nll_unit_influence: the total token count must be positive')
    full = _paired_nll_doc_ci(A, B, T, conf=conf)
    if full is None:
        return {'kind': 'paired_unit_influence', 'available': False,
                'n_units': int(D), 'n_seeds': int(S),
                'reason': 'the full pooled PPL difference overflowed'}
    full_delta = float(full['delta_ppl'])
    full_lr = float(full['delta_log_ratio'])

    sum_a = A.sum(axis=1)
    sum_b = B.sum(axis=1)
    ce_a = sum_a / total_tok
    ce_b = sum_b / total_tok
    ea = np.asarray([_safe_exp_ppl(x) for x in ce_a], dtype=float)
    eb = np.asarray([_safe_exp_ppl(x) for x in ce_b], dtype=float)
    if not (np.all(np.isfinite(ea)) and np.all(np.isfinite(eb))):
        return {'kind': 'paired_unit_influence', 'available': False,
                'n_units': int(D), 'n_seeds': int(S),
                'reason': 'the per-seed pooled PPL overflowed'}

    # Exact delete-one-unit recomputation.  The full-fold point must be the
    # same estimand as the caller's pooled CI; otherwise the influence belongs
    # to a different comparison and is refused rather than silently reported.
    denom = total_tok - T
    usable_denom = denom > 0.0
    ce_a_loo = np.full((S, D), np.nan, dtype=float)
    ce_b_loo = np.full((S, D), np.nan, dtype=float)
    np.divide(sum_a[:, None] - A, denom[None, :], out=ce_a_loo,
              where=usable_denom[None, :])
    np.divide(sum_b[:, None] - B, denom[None, :], out=ce_b_loo,
              where=usable_denom[None, :])
    with np.errstate(over='ignore', invalid='ignore'):
        ea_loo = np.exp(ce_a_loo)
        eb_loo = np.exp(ce_b_loo)
    loo_delta = np.mean(ea_loo - eb_loo, axis=0)
    loo_lr = np.mean(ce_a_loo - ce_b_loo, axis=0)
    finite = np.isfinite(loo_delta) & np.isfinite(loo_lr)
    if not np.any(finite):
        return {'kind': 'paired_unit_influence', 'available': False,
                'n_units': int(D), 'n_seeds': int(S),
                'reason': ('no usable leave-one-unit-out fold (a unit may hold '
                           'all tokens, or the remaining pooled PPL overflowed)')}

    # Delta-method influence, exactly the ``bar`` term that the cluster-robust
    # variance squares.  Kept separate from the exact LOO fold so the report
    # can show local influence even when a fold overflows.
    psi = (ea[:, None] * (A - ce_a[:, None] * T[None, :]) / total_tok
           - eb[:, None] * (B - ce_b[:, None] * T[None, :]) / total_tok).mean(axis=0)
    psi_lr = ((A - ce_a[:, None] * T[None, :])
              - (B - ce_b[:, None] * T[None, :])).mean(axis=0) / total_tok

    change_ppl = loo_delta - full_delta
    change_lr = loo_lr - full_lr
    abs_change_ppl = np.where(finite, np.abs(change_ppl), -np.inf)
    abs_change_lr = np.where(finite, np.abs(change_lr), -np.inf)
    use_ppl = abs_change_ppl > -np.inf
    use_lr = abs_change_lr > -np.inf

    def _max_or_none(arr, mask):
        if not np.any(mask):
            return None
        v = float(np.max(arr[mask]))
        # A subtraction of two finite but opposite-sign large values can
        # overflow to inf; that is not a measured "maximum change".
        return v if math.isfinite(v) else None

    max_change_ppl = _max_or_none(abs_change_ppl, use_ppl)
    max_change_lr = _max_or_none(abs_change_lr, use_lr)

    def _relative(change, full_value):
        if change is None or full_value is None:
            return None
        try:
            full_value = float(full_value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(full_value) or abs(full_value) <= 1e-300:
            return None
        v = float(change / abs(full_value))
        return v if math.isfinite(v) else None

    rel_change_ppl = _relative(max_change_ppl, full_delta)
    rel_change_lr = _relative(max_change_lr, full_lr)

    def _sign_stable(values, ref, mask):
        if not np.any(mask):
            return None
        try:
            ref = float(ref)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(ref) or ref == 0.0:
            return None
        vals = values[mask]
        return bool(np.all((vals > 0.0) if ref > 0.0 else (vals < 0.0)))

    sign_stable_ppl = _sign_stable(loo_delta, full_delta, use_ppl)
    sign_stable_lr = _sign_stable(loo_lr, full_lr, use_lr)
    # A change that overflowed during subtraction still proves the fold is
    # influential even though its percentage cannot be represented.
    _change_overflow = bool(np.any(np.isinf(abs_change_lr[use_lr])))
    single_unit_dominates = bool(
        _change_overflow
        or (rel_change_lr is not None and rel_change_lr >= 1.0))
    if sign_stable_lr is None:
        fragile_to_one_unit = None
    else:
        fragile_to_one_unit = bool((not sign_stable_lr) or single_unit_dominates)

    n_flips_ppl = (int(np.sum((loo_delta[use_ppl] > 0.0) != (full_delta > 0.0)))
                   if sign_stable_ppl is not None else None)
    n_flips_lr = (int(np.sum((loo_lr[use_lr] > 0.0) != (full_lr > 0.0)))
                  if sign_stable_lr is not None else None)

    # Exact jackknife variance of the pooled estimate over the unit folds.
    # The LOO points themselves can be enormous while still finite (a unit
    # that carries almost all tokens leaves a tiny remaining pooled PPL); a
    # naive squaring of such deviations overflows to inf even when the final
    # SE would be representable.  Compute in a scaled/log space and explicitly
    # mark the jackknife unavailable when the variance does not fit a float.
    def _jackknife_se(values):
        n = int(len(values))
        if n < 2:
            return None, False
        try:
            center = math.fsum(float(v) for v in values) / n
        except (OverflowError, ValueError):
            return None, False
        if not math.isfinite(center):
            return None, False
        try:
            dev = [float(v) - center for v in values]
        except (TypeError, ValueError, OverflowError):
            return None, False
        scale = max((abs(d) for d in dev), default=0.0)
        if not math.isfinite(scale):
            return None, False
        if scale == 0.0:
            return 0.0, True
        try:
            ss = math.fsum((d / scale) ** 2 for d in dev)
        except (OverflowError, ValueError):
            return None, False
        factor = (n - 1.0) / n
        if not (math.isfinite(ss) and ss > 0.0):
            return None, False
        try:
            logvar = math.log(factor) + 2.0 * math.log(scale) + math.log(ss)
        except (ValueError, OverflowError):
            return None, False
        if logvar > math.log(sys.float_info.max):
            # The variance is mathematically beyond float, so neither the SE
            # nor a symmetric CI can be represented.  Do not emit inf as if
            # it were a measured interval.
            return None, False
        var = math.exp(logvar)
        if not math.isfinite(var):
            return None, False
        return math.sqrt(max(var, 0.0)), True
    _all_finite = bool(np.all(finite))
    jackknife_ok = False
    jack_se_ppl = jack_se_lr = None
    jack_ci_ppl = jack_ci_lr = None
    _jack_reason = None
    if _all_finite:
        jack_se_ppl, _jk_ppl_ok = _jackknife_se(loo_delta)
        jack_se_lr, _jk_lr_ok = _jackknife_se(loo_lr)
        jackknife_ok = bool(_jk_ppl_ok and _jk_lr_ok)
        if not jackknife_ok:
            _jack_reason = ('the delete-one-unit variance exceeds the '
                            'representable float range')
    else:
        _jack_reason = 'one or more leave-one-unit-out folds is unavailable'
    tc = _t_critical(conf, D - 1) if jackknife_ok else float('nan')
    if jackknife_ok and math.isfinite(tc):
        _lo, _hi = (full_delta - tc * jack_se_ppl,
                    full_delta + tc * jack_se_ppl)
        if math.isfinite(_lo) and math.isfinite(_hi):
            jack_ci_ppl = [_lo, _hi]
        _lo, _hi = (full_lr - tc * jack_se_lr,
                    full_lr + tc * jack_se_lr)
        if math.isfinite(_lo) and math.isfinite(_hi):
            jack_ci_lr = [_lo, _hi]
    cluster_se_ppl = full.get('se_ppl')
    cluster_se_lr = full.get('se_log_ratio')
    se_ratio_ppl = (float(jack_se_ppl / cluster_se_ppl)
                    if (jack_se_ppl is not None and cluster_se_ppl
                        and math.isfinite(cluster_se_ppl) and cluster_se_ppl > 0.0)
                    else None)
    if se_ratio_ppl is not None and not math.isfinite(se_ratio_ppl):
        se_ratio_ppl = None
    se_ratio_lr = (float(jack_se_lr / cluster_se_lr)
                   if (jack_se_lr is not None and cluster_se_lr
                       and math.isfinite(cluster_se_lr) and cluster_se_lr > 0.0)
                   else None)
    if se_ratio_lr is not None and not math.isfinite(se_ratio_lr):
        se_ratio_lr = None

    order = [int(i) for i in np.argsort(-np.nan_to_num(abs_change_lr, nan=-1.0),
                                        kind='stable')[:top_k] if abs_change_lr[i] > -np.inf]
    top_units = []
    for i in order:
        top_units.append({
            'unit_id': labels[i],
            'loo_delta_ppl': (None if not finite[i] else float(loo_delta[i])),
            'loo_change_ppl': (None if (not finite[i]
                                        or not math.isfinite(float(change_ppl[i])))
                               else float(change_ppl[i])),
            'loo_delta_log_ratio': (None if not finite[i] else float(loo_lr[i])),
            'loo_change_log_ratio': (None if (not finite[i]
                                              or not math.isfinite(float(change_lr[i])))
                                     else float(change_lr[i])),
        })
    best_idx = None
    tied = None
    if np.any(use_lr) and abs(full_lr) > 1e-300:
        _cand = np.where(use_lr, abs_change_lr, -np.inf)
        best_idx = int(np.argmax(_cand))
        best_val = float(_cand[best_idx])
        tied = int(np.sum(np.isclose(_cand, best_val, rtol=0.0, atol=1e-12)))

    def _vector_with_unavailable_as_none(values):
        # An unavailable fold is marked by `loo_available`; emitting inf/NaN
        # in the per-fold vectors would make the object non-JSON-compliant
        # until _json_clean rewrites it, and a consumer that reads the
        # in-memory result (or a strict writer) would see a fake point.  Use
        # JSON null for exactly the folds the flag marks unavailable, and for
        # a derived change that overflowed even though the fold itself is
        # finite.
        out = []
        for i in range(D):
            try:
                v = float(values[i])
            except (TypeError, ValueError, OverflowError):
                out.append(None)
                continue
            out.append(None if (not finite[i] or not math.isfinite(v)) else v)
        return out

    def _finite_vector(values):
        out = []
        for v in values:
            try:
                v = float(v)
            except (TypeError, ValueError, OverflowError):
                out.append(None)
                continue
            out.append(v if math.isfinite(v) else None)
        return out

    return {
        'kind': 'paired_unit_influence', 'available': True,
        'n_units': int(D), 'n_units_loo_available': int(np.sum(finite)),
        'n_seeds': int(S), 'conf': float(conf),
        'unit_ids': labels,
        'label_source': ('provided' if unit_ids is not None else 'index_fallback'),
        'full_delta_ppl': full_delta, 'full_delta_log_ratio': full_lr,
        'delta_method_influence_ppl': _finite_vector(psi),
        'delta_method_influence_log_ratio': _finite_vector(psi_lr),
        'loo_delta_ppl': _vector_with_unavailable_as_none(loo_delta),
        'loo_delta_log_ratio': _vector_with_unavailable_as_none(loo_lr),
        'loo_change_ppl': _vector_with_unavailable_as_none(change_ppl),
        'loo_change_log_ratio': _vector_with_unavailable_as_none(change_lr),
        'loo_available': finite.tolist(),
        'max_abs_change_ppl': max_change_ppl,
        'max_abs_change_log_ratio': max_change_lr,
        'max_abs_relative_change_ppl': rel_change_ppl,
        'max_abs_relative_change_log_ratio': rel_change_lr,
        'jackknife_available': jackknife_ok,
        'jackknife_unavailable_reason': _jack_reason,
        'jackknife_se_ppl': jack_se_ppl,
        'jackknife_se_log_ratio': jack_se_lr,
        'jackknife_ci95_ppl': jack_ci_ppl,
        'jackknife_ci95_log_ratio': jack_ci_lr,
        'cluster_robust_se_ppl': cluster_se_ppl,
        'cluster_robust_se_log_ratio': cluster_se_lr,
        'se_ratio_jackknife_to_cluster_ppl': se_ratio_ppl,
        'se_ratio_jackknife_to_cluster_log_ratio': se_ratio_lr,
        'sign_stable_ppl': sign_stable_ppl,
        'sign_stable_log_ratio': sign_stable_lr,
        'single_unit_dominates': single_unit_dominates,
        'fragile_to_one_unit': fragile_to_one_unit,
        'n_sign_flips_ppl': n_flips_ppl,
        'n_sign_flips_log_ratio': n_flips_lr,
        'most_influential_unit': (labels[best_idx] if best_idx is not None else None),
        'most_influential_index': best_idx,
        'most_influential_tied_count': tied,
        'top_units': top_units,
        'method': ('exact leave-one-unit-out on the token-pooled per-seed PPL '
                   'difference, averaged over seeds; delete-one-unit jackknife; '
                   'delta-method influence exposed alongside it'),
        'note': ('post-hoc descriptive robustness only: conditions on the trained '
                 'seeds, does not randomise units, does not change any p/Holm/'
                 'frozen decision, and describes the frozen slice minus one unit '
                 'rather than the language population'),
    }


# ---- G0 gate ---------------------------------------------------------------
# The scale panels are only allowed to start after the kernel/eager
# equivalence certificate exists and matches this code/env.  `explib.m0.py
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
    if cert.get('numeric_pass') is not True:
        return 'operator/model numeric equivalence did not pass'
    if cert.get('benchmark_pass') is not True:
        return 'G0 kernel/eager certificate did not pass (CUDA+triton required)'
    if cert.get('training_tost_pass') is not True:
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
        f'certificate, but {_why}. Run `python explib.m0.py kernel` on the target '
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
ABL_VARIANTS = ['hybrid_csa_dyn', 'hybrid_hca_dyn', 'csa_dyn_fuse', 'hybrid_csa_dyn_fuse', 'csa_fix_randidx', 'csa_fix_zerocont', 'csa_fix_shiftcont', 'csa_fix_nosink', 'csa_fix_topk8', 'csa_fix_topk64', 'csa_dyn_randidx', 'csa_dyn_zerocont', 'csa_dyn_shiftcont', 'csa_dyn_nosink', 'csa_dyn_nocomp', 'csa_dynamic_task', 'full_sink']
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
# ---- B 档规模阶梯（档位已选定 B 进取）----
# 以下全部为规划值：步数 / 批大小 / 学习率等训练量与超参以此为准；
# 实际参数量以 SmallGPT 实例化统计为准（见 explib.m0.py params，S4 按
# 规划的解析外推口径）。
B_VOCAB = 32768
SCALE_LADDER = {'S1': dict(d=768, n_layers=12, n_heads=12, d_head=64), 'S2': dict(d=1024, n_layers=16, n_heads=16, d_head=64), 'S3': dict(d=1536, n_layers=20, n_heads=24, d_head=64), 'S4': dict(d=2048, n_layers=24, n_heads=32, d_head=64)}
# Declared transfer assumptions for the S1→S4 ladder.  They are deliberately
# explicit: a scale comparison that changes width/depth also changes the
# optimal optimizer settings, so a constant LR is a DISCLOSED assumption, not
# a matched control.  `scale_ladder_audit` checks the token budget and makes
# the assumption visible before the panel spends GPU time.
SCALE_TRANSFER_POLICY = {
    'tokens_per_param_tolerance': 0.25,
    'lr_transfer': 'constant_declared_not_validated',
    'family_m': 4,
    'note': 'S1-S4 hold lr/wd/warmup fraction fixed across widths. No muP or LR-transfer calibration has been run; scale contrasts therefore carry an optimisation-transfer confound and the ladder is descriptive until a transfer check exists.',
}
B_ARMS = ['full_matched', 'full_sw128_matched', 'csa_fixed', 'csa_dynamic', 'hybrid_dynamic']
RUN_S1 = dict(RUN, dataset='fineweb', vocab_size=B_VOCAB, d=768, n_layers=12, n_heads=12, d_head=64, seq_len=8192, batch_size=8, n_train_tokens=5000000000, steps=76294, warmup=2000, eval_every=4000, eval_subset=32, val_seqs=512, seeds=[0, 1, 2, 3], outdir='results_b_s1', variants=B_ARMS, measure_efficiency_sweep=True)
M0_CAP = dict(total_hours=700.0, margin=0.93, already_hours=0.0, state_path='run_time_state_m0.json')
RUN_S2 = dict(RUN_S1, d=1024, n_layers=16, n_heads=16, n_train_tokens=12000000000, steps=183105, warmup=4000, eval_every=8000, seeds=[0, 1, 2], outdir='results_b_s2')
M2_CAP = dict(total_hours=500.0, margin=0.93, already_hours=0.0, state_path='run_time_state_m2.json')

# ---- Declared data protocol (scale-safe) -----------------------------------
# Every corpus/evaluation size a run may use is a FIELD of its config.  The
# defaults below are only defaults: once a run is prepared they are copied into
# its cfg (and therefore into run_cfg / provenance), so scaling up changes a
# value rather than a code path.  Loaders must never silently fall back to the
# historical 8192-token vocabulary or 512-token context.
DATA_PROTOCOL_FIELDS = ('dataset', 'seq_len', 'n_train_tokens', 'vocab_size',
                        'val_seqs', 'cache_dir')
DATA_PROTOCOL_DEFAULTS = {
    'wikitext': dict(dataset='wikitext', seq_len=512,
                     n_train_tokens=int(RUN['n_train_tokens']),
                     vocab_size=8192, val_seqs=int(RUN.get('val_seqs', 512)),
                     cache_dir='./wt103_cache'),
    'fineweb': dict(dataset='fineweb', seq_len=8192,
                    n_train_tokens=int(RUN_S1['n_train_tokens']),
                    vocab_size=int(B_VOCAB), val_seqs=int(RUN_S1.get('val_seqs', 512)),
                    cache_dir='./fineweb_cache'),
}


def _protocol_pos_int(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f'data protocol field {name} must be an integer, got {value!r}')
    value = int(value)
    if value < 1:
        raise ValueError(f'data protocol field {name} must be >= 1, got {value!r}')
    return value


def resolve_data_protocol(cfg, *, dataset=None, require=(), defaults=None,
                          allow_unknown=False):
    """Resolve the scale protocol from fields, never from a hard-coded shape.

    Returns a plain dict with ``dataset``, ``seq_len``, ``n_train_tokens``,
    ``vocab_size``, ``val_seqs`` and ``cache_dir``.  Required fields
    (``require``) must come from the caller's cfg, not from a default, so a
    confirmation plan can refuse an implicit current-scale fallback.  The
    input cfg is not mutated; :func:`run` copies the resolved values back
    before fingerprinting.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'resolve_data_protocol expects a config mapping, got {type(cfg).__name__}')
    name = str(dataset if dataset is not None else (cfg.get('dataset') or 'wikitext'))
    _known = (defaults or DATA_PROTOCOL_DEFAULTS)
    _has_defaults = name in _known
    if not _has_defaults and not allow_unknown:
        raise ValueError(
            f'unknown dataset {name!r}: declare its data protocol defaults or '
            'pass allow_unknown=True only when a caller supplies its own loader')
    base = dict(_known.get(name) or {})
    # The cfg is authoritative; a field that is present but malformed is an
    # error, not an invitation to use the default (that would let a typo in a
    # scale knob silently train the old size).  Unknown datasets (unit-test
    # loaders, future corpora) have no defaults: their fields must come from
    # the cfg, and absent optional fields are simply not injected.
    out = {'dataset': name}
    for key in DATA_PROTOCOL_FIELDS[1:]:
        raw = cfg.get(key, base.get(key))
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            if _has_defaults:
                raise ValueError(f'data protocol field {key!r} is missing for dataset {name!r}; pass it explicitly (scale must not be implicit)')
            continue
        if key == 'cache_dir':
            out[key] = str(raw)
        else:
            out[key] = _protocol_pos_int(key, raw)
    for key in require:
        if key not in cfg or cfg.get(key) is None:
            raise ValueError(f'data protocol field {key!r} must be declared explicitly for this panel (defaults are not allowed)')
    return out


def apply_data_protocol(cfg, *, dataset=None, require=(), defaults=None,
                        allow_unknown=False):
    """Return a copy of ``cfg`` with the resolved protocol fields filled in."""
    if not isinstance(cfg, dict):
        raise TypeError(f'apply_data_protocol expects a config mapping, got {type(cfg).__name__}')
    out = dict(cfg)
    out.update(resolve_data_protocol(out, dataset=dataset, require=require,
                                     defaults=defaults,
                                     allow_unknown=allow_unknown))
    return out


# Environment knobs for re-sizing a scale panel WITHOUT editing a driver.  The
# values are copied into the cfg (and therefore into run_cfg/provenance), so a
# resumed panel can never mix two geometries silently.
SCALE_GEOMETRY_ENV = {
    'B_SEQ_LEN': 'seq_len',
    'B_BATCH_SIZE': 'batch_size',
    'B_N_TRAIN_TOKENS': 'n_train_tokens',
    'B_STEPS': 'steps',
    'B_VAL_SEQS': 'val_seqs',
}


def apply_scale_geometry_overrides(cfg, env=None, *,
                                   preserve_draw_budget=True):
    """Return a copy of a scale-panel cfg with validated geometry overrides.

    The B-tier drivers used to read only the module-level ``RUN_S1``/``RUN_S2``
    dicts, so adapting a run to a different sequence length or to a GPU with
    less free memory meant editing code.  These knobs are deliberately
    explicit and are recorded in the returned cfg:

    * ``B_SEQ_LEN`` / ``B_BATCH_SIZE`` re-size the per-step geometry.  When
      ``B_STEPS`` is not supplied, the step count is recomputed (ceiling) so
      the number of sampled windows is preserved; a geometry change cannot
      quietly shorten the training budget.
    * ``B_N_TRAIN_TOKENS`` changes the declared token budget and, unless
      ``B_STEPS`` is explicit, recomputes steps from the new budget.
    * ``B_STEPS`` is an explicit override and is never recomputed.
    * ``B_VAL_SEQS`` changes the held-out row count (recorded in the data
      protocol).

    A malformed value is an error, not a silent fallback.  The function is
    pure with respect to ``cfg``; it only reads ``env`` (default os.environ).
    """
    if not isinstance(cfg, dict):
        raise TypeError('apply_scale_geometry_overrides expects a config dict')
    env = os.environ if env is None else env
    out = dict(cfg)

    def _read(env_key, field):
        raw = env.get(env_key)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return None
        if isinstance(raw, (bool, np.bool_)):
            raise ValueError(f'{env_key}={raw!r} is not a positive integer')
        try:
            val = int(str(raw).strip())
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f'{env_key}={raw!r} is not a positive integer') from None
        if val < 1:
            raise ValueError(f'{env_key}={raw!r} must be >= 1')
        return int(val)

    overrides = {}
    for env_key, field in SCALE_GEOMETRY_ENV.items():
        val = _read(env_key, field)
        if val is not None:
            overrides[field] = val
    if not overrides:
        return out
    _old = {k: int(out[k]) for k in ('seq_len', 'batch_size', 'steps')
            if out.get(k) is not None}
    out.update(overrides)
    _steps_from_env = 'steps' in overrides
    _geometry_changed = ('seq_len' in overrides or 'batch_size' in overrides)
    _tokens_changed = 'n_train_tokens' in overrides
    _steps_source = 'explicit_env' if _steps_from_env else 'unchanged'
    if preserve_draw_budget and not _steps_from_env and (_geometry_changed or _tokens_changed):
        try:
            _seq = int(out['seq_len'])
            _bs = int(out['batch_size'])
        except (KeyError, TypeError, ValueError, OverflowError):
            raise ValueError(
                'scale geometry override changed seq_len/batch_size/token '
                'budget but the cfg lacks a usable seq_len/batch_size; the '
                'step count cannot be recomputed (pass B_STEPS explicitly)') from None
        if _seq <= 0 or _bs <= 0:
            raise ValueError(
                f'scale geometry override needs positive seq_len/batch_size, '
                f'got {_seq}/{_bs}')
        if _tokens_changed:
            _per_step = _bs * _seq
            try:
                _budget = int(out['n_train_tokens'])
            except (KeyError, TypeError, ValueError, OverflowError):
                raise ValueError(
                    'scale geometry override changed n_train_tokens but the '
                    'value is not an integer') from None
            if _budget < 1:
                raise ValueError(
                    f'scale geometry override needs a positive token budget, '
                    f'got {_budget}')
            out['steps'] = max(1, (_budget + _per_step - 1) // _per_step)
            _steps_source = 'from_declared_token_budget'
        elif _geometry_changed:
            if not all(k in _old for k in ('steps', 'seq_len', 'batch_size')):
                raise ValueError(
                    'scale geometry override changed seq_len/batch_size but '
                    'the cfg has no existing steps/seq_len/batch_size to '
                    'preserve the sampled-window budget (pass B_STEPS explicitly)')
            _old_draws = _old['steps'] * _old['batch_size'] * _old['seq_len']
            _per_step = _bs * _seq
            out['steps'] = max(1, (_old_draws + _per_step - 1) // _per_step)
            _steps_source = 'preserve_sampled_windows'
    out['geometry_overrides'] = {
        'env': dict(sorted(overrides.items())),
        'old': _old,
        'steps_source': _steps_source,
    }
    return out


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
        # A 1-D input is a sequence of scalar features.  Treat it as [T, 1]
        # so the result has the same [T-1] shape as the general path; the old
        # scalar reduction made T=1 return 0.0 instead of an empty vector and
        # could not be consumed by the segmenter.
        h = F.normalize(H.reshape(-1, 1), dim=-1, eps=eps).reshape(-1)
        return h[1:] * h[:-1]
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


def chunk_similarity_source(cfg):
    """Effective similarity signal for a chunking configuration."""
    declared = getattr(cfg, 'cut_signal_source', 'auto')
    if declared not in CHUNK_SIGNAL_SOURCES:
        raise ValueError(f'unknown cut_signal_source {declared!r}')
    if declared != 'auto':
        return declared
    return 'fused' if getattr(cfg, 'chunking', 'fixed') == 'cosine_learnable' else 'context'


def chunk_similarity(H, cfg, *, fused=None, eps=1e-8):
    """One entry point for every dynamic-chunking similarity signal.

    All arms that claim to differ only in *chunking policy* must feed the same
    representation here.  `cut_signal_source='auto'` preserves the historical
    learned/fused vs context split; a panel that wants signal parity declares
    the source explicitly (e.g. all arms 'context') and the protocol descriptor
    records the effective choice.
    """
    source = chunk_similarity_source(cfg)
    if source == 'fused':
        if fused is None:
            raise RuntimeError(
                "chunk_similarity_source='fused' requires the fused projection; "
                'the caller would otherwise silently fall back to a different '
                'similarity signal')
        return cosine_similarity_consecutive(fused, eps=eps)
    return cosine_similarity_consecutive(H, eps=eps)


def chunking_protocol(cfg):
    """Machine-readable descriptor of the chunking DATA path.

    Recorded next to block statistics so a report can verify that arms being
    compared used the same similarity signal, the same strict-cut direction and
    the same geometry bounds before attributing a difference to the policy.
    """
    source = chunk_similarity_source(cfg)
    return {
        'chunking': str(getattr(cfg, 'chunking', 'fixed')),
        'dynamic': bool(getattr(cfg, 'dynamic', False)),
        'cut_signal_source': source,
        'cut_signal_declared': str(getattr(cfg, 'cut_signal_source', 'auto')),
        'cut_rule': 'sim < tau',
        'cut_rule_strict': True,
        'similarity_eps': 1e-08,
        'min_block': int(getattr(cfg, 'min_block', 1) or 1),
        'max_block': (None if getattr(cfg, 'max_block', None) is None
                      else int(getattr(cfg, 'max_block'))),
        'target_block_tokens': int(getattr(cfg, 'target_block_tokens', 1) or 1),
        'cos_threshold': (float(getattr(cfg, 'cos_threshold'))
                          if getattr(cfg, 'chunking', '') in ('cosine_abs', 'cosine_adaptive')
                          else None),
    }


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
_SEG_POS_CACHE_BYTES = [0]
try:
    # A count-only cap retains up to 16 full position tables: at a 10M-token
    # scale-up that is 1.3 GB of host memory even though each table is a pure
    # function of n.  Bound the retained bytes as well as the entry count.
    _SEG_POS_CACHE_BUDGET = max(1, int(os.environ.get('CSA_SEG_POS_CACHE_MB', '64'))) << 20
except (TypeError, ValueError, OverflowError):
    _SEG_POS_CACHE_BUDGET = 64 << 20

def _seg_pos(n):
    pos = _SEG_POS_CACHE.get(n)
    if pos is None:
        pos = np.arange(n + 1, dtype=np.int64)
        _nb = int(pos.nbytes)
        if _nb > _SEG_POS_CACHE_BUDGET:
            # Keep one oversized table (rebuilding it at every layer/step is
            # worse than retaining it) but make it the only entry.
            _SEG_POS_CACHE.clear()
            _SEG_POS_CACHE_BYTES[0] = 0
        else:
            while _SEG_POS_CACHE and (
                    len(_SEG_POS_CACHE) >= _SEG_POS_CACHE_CAP
                    or _SEG_POS_CACHE_BYTES[0] + _nb > _SEG_POS_CACHE_BUDGET):
                _old = _SEG_POS_CACHE.pop(next(iter(_SEG_POS_CACHE)))
                _SEG_POS_CACHE_BYTES[0] -= int(_old.nbytes)
        _SEG_POS_CACHE[n] = pos
        _SEG_POS_CACHE_BYTES[0] += _nb
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
    # `max_block=None` is "dynamic boundaries only".  At this entry point the
    # sequence length is known, so it becomes an explicit safe bound that can
    # never force a cut.  The integer arithmetic below is therefore total.
    if max_block is None:
        max_block = int(n)
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
        # The uniform shortcut must obey the same large-table contract as the
        # general walk below: small rows keep the historical Python-list form,
        # but a long sequence (e.g. the m=1 / block_size=1 arm, where stride is
        # one) must not materialise one Python object per block.  Build the
        # repeated row in NumPy first and convert only the <=64-token case.
        _nfull, _rem = divmod(n, _stride)
        _k = _nfull + (1 if _rem else 0)
        counts = np.full(_k, _stride, dtype=np.int64)
        if _rem:
            counts[-1] = _rem
        if n <= 64:
            counts = counts.tolist()
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
    # Small tables keep the Python-list contract (and the LRU cache above
    # stores exactly that form).  Larger rows stay as numpy arrays carrying the
    # identical integers; every consumer either calls len/max/np.asarray on
    # them or hands them to the host-geometry builders.
    counts = counts_np.tolist() if n <= 64 else counts_np
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
except (TypeError, ValueError, OverflowError):
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
            except (RuntimeError, TypeError, MemoryError):
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
        except (RuntimeError, MemoryError):
            pass
        return t.to(device, non_blocking=t.is_pinned())

_D2H_POOL = {}
_D2H_POOL_BYTES = [0]
try:
    _D2H_POOL_CAP = max(16, int(os.environ.get('CSA_D2H_POOL_MB', '64'))) << 20
except (TypeError, ValueError, OverflowError):
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
            except (RuntimeError, TypeError, MemoryError):
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
        if isinstance(counts, np.ndarray):
            # Large cut tables return their counts as a numpy row; use the
            # vectorised extrema instead of iterating Python ints over it.
            _mx = int(counts.max()) if counts.size else 0
            _mx_prev = int(counts[:-1].max()) if cnt > 1 else 0
        else:
            _mx = max(counts)
            _mx_prev = max(counts[:-1]) if cnt > 1 else 0
        base._pool_cpu = (counts, _mx, _mx_prev)
    if not (return_count or return_honoured):
        return base
    out = [base]
    if return_count:
        out.append(cnt)
    if return_honoured:
        out.append(hon)
    return tuple(out)


def segmentation_prefix_invariance(cut_table, n, min_block, max_block,
                                   prefixes=None, max_checks=64):
    """Check that the cut table induces a prefix-invariant segmentation.

    For every prefix ``T' <= n`` the block starts of ``x[:T']`` must be a
    prefix of the full block starts; otherwise a token belongs to different
    blocks at different evaluation lengths and cross-length panels are not
    comparable.  Returns a report (``ok`` plus the first mismatches) rather
    than raising, so it can be used as an audit/diagnostic on candidate cut
    tables.  Only the last (tail) block may legitimately be extended by the
    withheld suffix; its START is still checked.
    """
    cuts = np.asarray(cut_table, dtype=bool).reshape(-1)
    n = int(n)
    if n < 1:
        return {'ok': True, 'n': n, 'checked_prefixes': 0, 'mismatches': []}
    if cuts.size < n - 1:
        cuts = np.concatenate([cuts, np.zeros(n - 1 - cuts.size, dtype=bool)])
    cuts = cuts[:n - 1]

    def _starts_for(m):
        _, _, counts = _segment(m, cuts[:m - 1], min_block, max_block)
        # Block starts are the exclusive cumulative counts shifted by one.
        # `np.asarray` keeps a large segmenter row in NumPy: the old
        # `[0] + list(counts)` materialised every block as a Python int (and a
        # full extra list) even though only this arithmetic needs it.
        _counts = np.asarray(counts, dtype=np.int64)
        return np.cumsum(_counts) - _counts

    full = _starts_for(n)
    if prefixes is None:
        # Keep a small `max_checks` usable: the short-prefix block must never
        # exceed the caller's budget (a negative np.linspace count used to
        # raise instead of returning a partial report).
        max_checks = max(0, int(max_checks))
        if n <= max_checks:
            prefixes = range(1, n + 1)
        else:
            # Deterministic coverage of short prefixes (where geometry is
            # densest) plus evenly spaced longer ones.
            short = list(range(1, min(n, 32, max_checks) + 1))
            rest_n = min(max_checks - len(short), max(0, n - len(short)))
            rest = (np.linspace(len(short) + 1, n, num=rest_n,
                                dtype=np.int64).tolist() if rest_n > 0 else [])
            prefixes = sorted(set(short + [int(x) for x in rest]))
    elif isinstance(prefixes, numbers.Integral):
        # A single prefix is a legitimate convenience input; iterating an int
        # would otherwise raise an unrelated TypeError.
        prefixes = [int(prefixes)]
    mismatches = []
    checked = 0
    for t in sorted({int(x) for x in prefixes if 1 <= int(x) <= n}):
        checked += 1
        got = _starts_for(t)
        if got.size > full.size or not np.array_equal(got, full[:got.size]):
            mismatches.append({'prefix': int(t),
                               'prefix_starts': got.tolist()[:16],
                               'full_starts': full[:max(got.size, 1)].tolist()[:16]})
            if len(mismatches) >= 8:
                break
    return {'ok': not mismatches, 'n': n, 'checked_prefixes': checked,
            'mismatches': mismatches,
            'policy': 'prefix block starts must be a prefix of the full starts '
                      '(strictly causal cuts; the tail may only extend)'}
