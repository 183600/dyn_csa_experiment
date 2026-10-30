

def audit_confirmation_verdicts(path=None):
    """Result-analysis audit: the three-way directional verdict is coherent.

    ``explib.confirmation.finaltest_stats`` records ``supported`` / ``refuted`` /
    ``inconclusive`` / ``not_testable`` plus a per-comparison ``verdicts``
    block.  A Holm-significant result in the opposite direction is a
    refutation of the pre-registered directional claim, not "not confirmed";
    this gate independently replays the partition from the stored p-values,
    directions and validity gates.  It never recomputes a p-value and never
    changes the frozen decision.  A stored verdict that disagrees with the
    replay, a comparison that appears in two buckets, or a missing verdict is
    a ``fail``; an artifact produced before the three-way field existed is
    ``not_verifiable_locally`` rather than silently passing.
    """
    L = _import_exp_lib()
    p = path or os.path.join(REPO, 'analysis_v12', 'test_stats.json')
    out = {'kind': 'confirmation_verdict_audit', 'status': 'ok', 'path': p,
           'comparisons': {}, 'n_supported': 0, 'n_refuted': 0,
           'n_inconclusive': 0, 'n_not_testable': 0,
           'problems': [], 'warnings': [],
           'note': ('replays the three-way directional partition on the SAME '
                    'Holm family; it adds no test and never changes a frozen '
                    'p/Holm/confirmed verdict')}
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
    stored = ft.get('verdicts')
    if not isinstance(stored, dict) or not stored:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = ('the stats artifact predates the three-way verdict '
                         'block; regenerate it with the current v12 analysis')
        out['warnings'].append(
            'legacy artifact: a reverse-significant comparison cannot be '
            'distinguished from an unconfirmed one until the analysis is rerun')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    holm = ft.get('holm') if isinstance(ft.get('holm'), dict) else {}
    vin = {}
    for name, rec in comps.items():
        if not isinstance(rec, dict):
            out['problems'].append(
                f'{name}: comparison is a {type(rec).__name__}, not an object')
            continue
        cv = dict(rec)
        _h = holm.get(name) if isinstance(holm.get(name), dict) else {}
        cv['p_holm'] = _h.get('p_holm', cv.get('p_holm'))
        vin[str(name)] = cv
    _cov = ((ft.get('coverage_requirement') or {}).get('require_complete_coverage')
            if isinstance(ft.get('coverage_requirement'), dict) else False)
    _par = ((ft.get('parameter_binding') or {}).get('required_by_manifest')
            if isinstance(ft.get('parameter_binding'), dict) else False)
    _m = ft.get('manifest') if isinstance(ft.get('manifest'), dict) else {}
    _seeds = _m.get('seeds') if isinstance(_m.get('seeds'), list) else []
    _alpha = ft.get('alpha', 0.05)
    try:
        replay = L.directional_claim_verdicts(
            vin, names=list(vin), alpha=_alpha,
            require_complete_coverage=bool(_cov),
            require_param_match=bool(_par),
            default_min_seeds=(len(_seeds) if _seeds else None))
    except Exception as _e:
        out['status'] = 'fail'
        out['problems'].append(
            f'the stored comparison records cannot be replayed '
            f'({type(_e).__name__}: {_e})')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    expected = replay['verdicts']
    _allowed = {'supported', 'refuted', 'inconclusive',
                'significant_nondirectional', 'not_testable'}
    for name in vin:
        srec = stored.get(name)
        if not isinstance(srec, dict):
            out['problems'].append(
                f'{name}: stored verdict is missing or not an object')
            continue
        _v = srec.get('verdict')
        if _v not in _allowed:
            out['problems'].append(
                f'{name}: stored verdict {_v!r} is not one of {sorted(_allowed)}')
            continue
        want = expected[name]['verdict']
        if _v != want:
            out['problems'].append(
                f'{name}: stored verdict {_v!r} does not match the replay '
                f'{want!r} from p_holm/direction/validity gates')
        _sp, _wp = srec.get('p_holm'), expected[name].get('p_holm')
        if (_sp is None) != (_wp is None):
            out['problems'].append(
                f'{name}: stored p_holm presence does not match the replay')
        elif _sp is not None:
            try:
                if abs(float(_sp) - float(_wp)) > 1e-12:
                    out['problems'].append(
                        f'{name}: stored p_holm={_sp!r} does not match the '
                        f'replayed {_wp!r}')
            except (TypeError, ValueError, OverflowError):
                out['problems'].append(
                    f'{name}: stored p_holm is not numeric')
        out['comparisons'][name] = {
            'verdict': _v, 'expected': want,
            'expected_sign': expected[name].get('expected_sign'),
            'direction_ok': expected[name].get('direction_ok'),
            'claim_scope': expected[name].get('claim_scope'),
            'gate_failures': expected[name].get('gate_failures')}
    _extra = sorted(set(map(str, stored)) - set(vin))
    if _extra:
        out['problems'].append(
            f'stored verdicts contain undeclared comparisons: {_extra}')
    # ``significant_nondirectional`` predates the three-way block: the frozen
    # field includes every Holm rejection whose direction is unverifiable,
    # even when a validity gate fails.  Replay that legacy field exactly rather
    # than comparing it to the helper's stricter non-directional bucket.
    _legacy_nd = []
    for _name, _rec in vin.items():
        try:
            _p = float(_rec.get('p_holm'))
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(_p) and _p < float(replay['alpha']) and \
                _rec.get('direction_ok') is None:
            _legacy_nd.append(_name)
    _lists = {'confirmed': replay['supported'],
              'refuted': replay['refuted'],
              'inconclusive': replay['inconclusive'],
              'not_testable': replay['not_testable'],
              'significant_nondirectional': _legacy_nd,
              'opposite_nonsignificant': replay['opposite_nonsignificant'],
              'reverse_significant_unverified': replay['reverse_significant_unverified']}
    for key, want in _lists.items():
        got = ft.get(key)
        if got is None:
            if key in ('confirmed', 'refuted'):
                out['problems'].append(
                    f'top-level {key!r} list is missing from the stats artifact')
            continue
        if isinstance(got, (str, bytes)) or not isinstance(got, (list, tuple)):
            out['problems'].append(
                f'top-level {key!r} is not a list')
            continue
        got_s = sorted(str(x) for x in got)
        want_s = sorted(want)
        if len(got_s) != len(set(got_s)):
            out['problems'].append(
                f'top-level {key!r} contains duplicate names')
        if got_s != want_s:
            out['problems'].append(
                f'top-level {key!r}={got_s} does not match the replay {want_s}')
    _overlap = (set(replay['supported']) & set(replay['refuted']))
    if _overlap:
        out['problems'].append(
            f'comparisons cannot be both supported and refuted: {sorted(_overlap)}')
    out['n_supported'] = len(replay['supported'])
    out['n_refuted'] = len(replay['refuted'])
    out['n_inconclusive'] = len(replay['inconclusive'])
    out['n_not_testable'] = len(replay['not_testable'])
    if replay['refuted']:
        out['warnings'].append(
            'refuted comparisons are directional conflicts, not "not '
            'confirmed" or equivalence; the report must keep that wording')
    if replay['reverse_significant_unverified']:
        out['warnings'].append(
            'some comparisons have a significant reverse point estimate but '
            'failed a validity gate; they stay inconclusive, not refuted')
    if out['problems']:
        out['status'] = 'fail'
    elif out['warnings']:
        out['status'] = 'warnings'
    print(json.dumps({k: v for k, v in out.items() if k != 'comparisons'},
                     ensure_ascii=False, indent=2))
    return out


def _pvr_finite(value):
    """A finite float from a stored record, or None.

    A numeric string is not a measurement.  ``float('0.05')`` would otherwise
    let a hand-edited artifact turn a quoted p-value (or a quoted per-seed
    delta into a quoted headline field) into a number this gate accepts.  The
    same contract is already enforced by ``_finite_scalar`` elsewhere; the
    replay gate must not be the one reader that silently coerces text.
    """
    if isinstance(value, (bool, np.bool_, str, bytes, bytearray)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _pvr_int(value):
    """An exact integer from a stored record, or None.

    Counts and seed ids are integers, not approximate scalars: ``int(12.9)``
    would silently turn a corrupted count into a different valid one, and a
    boolean would become 0/1.  Integral floats (``12.0``) are accepted because
    JSON writers may emit them, but fractional values and strings are not.
    """
    if value is None or isinstance(value, (bool, np.bool_, str, bytes, bytearray)):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def _pvr_close(a, b, *, rel=1e-9, abs_=1e-12):
    """Numeric equality that also handles infinities and NaN deliberately."""
    fa, fb = _pvr_finite(a), _pvr_finite(b)
    if fa is None or fb is None:
        return False
    return math.isclose(fa, fb, rel_tol=rel, abs_tol=abs_)


def _pvr_replay_one(L, name, rec, *, where, problems, warnings):
    """Recompute one sign-flip p from its stored per-seed paired deltas.

    Returns ``{'kind', 'status', 'per_seed', 'p', ...}``.  A missing/invalid
    delta vector is reported as ``not_replayable`` rather than fabricated;
    a present vector whose stored p, n, seed set or p-floor disagrees is a
    hard problem because the artifact no longer reproduces its own headline
    statistic.
    """
    out = {'name': name, 'where': where, 'status': 'not_replayable',
           'p': None, 'n': None, 'seeds': None, 'per_seed': None}
    deltas = rec.get('per_seed')
    if deltas is None:
        warnings.append(
            f'{where}: no per_seed paired deltas recorded — the stored p '
            'cannot be replayed (legacy artifact, not a pass)')
        return out
    if isinstance(deltas, (str, bytes)) or not isinstance(deltas, (list, tuple)):
        problems.append(f'{where}: per_seed is a {type(deltas).__name__}, not a list of paired deltas')
        out['status'] = 'invalid'
        return out
    vals = []
    for i, x in enumerate(deltas):
        if isinstance(x, (bool, np.bool_, str, bytes, bytearray)):
            # A quoted number is not a measurement (same contract as
            # _finite_scalar); accepting "0.3" here would let a text-only edit
            # replay as the original numeric delta.
            problems.append(
                f'{where}: per_seed[{i}] is a {type(x).__name__}, not a '
                'numeric paired delta')
            out['status'] = 'invalid'
            return out
        try:
            v = float(x)
        except (TypeError, ValueError, OverflowError):
            problems.append(f'{where}: per_seed[{i}] is not numeric')
            out['status'] = 'invalid'
            return out
        if not math.isfinite(v):
            problems.append(f'{where}: per_seed[{i}] is not finite ({x!r})')
            out['status'] = 'invalid'
            return out
        vals.append(v)
    if len(vals) < 2:
        problems.append(
            f'{where}: a non-omitted comparison needs at least 2 paired '
            f'deltas, got {len(vals)}')
        out['status'] = 'invalid'
        return out
    seeds = rec.get('seeds')
    if seeds is not None:
        if isinstance(seeds, (str, bytes)) or not isinstance(seeds, (list, tuple)):
            problems.append(f'{where}: seeds is a {type(seeds).__name__}, not a list')
            out['status'] = 'invalid'
            return out
        seed_ints = []
        for i, s in enumerate(seeds):
            si = _pvr_int(s)
            if si is None or si < 0 or si > int(L.MAX_SEED):
                problems.append(
                    f'{where}: seeds[{i}]={s!r} is not an exact seed integer '
                    f'in [0, {int(L.MAX_SEED)}]')
                out['status'] = 'invalid'
                return out
            seed_ints.append(si)
        if len(seed_ints) != len(vals):
            problems.append(
                f'{where}: seeds has {len(seed_ints)} entries but per_seed has '
                f'{len(vals)} deltas')
            out['status'] = 'invalid'
            return out
        if len(set(seed_ints)) != len(seed_ints):
            problems.append(f'{where}: seeds contains duplicates {seed_ints}')
            out['status'] = 'invalid'
            return out
        out['seeds'] = seed_ints
    else:
        warnings.append(
            f'{where}: seeds is absent — the deltas cannot be tied to the '
            'frozen seed whitelist, so this is not a complete replay')
    rep = L.exact_signflip_p(vals)
    p = _pvr_finite(rep.get('p'))
    if p is None:
        problems.append(f'{where}: the recomputed exact sign-flip p is not finite')
        out['status'] = 'invalid'
        return out
    out.update({'per_seed': vals, 'p': float(p), 'n': int(rep.get('n') or 0),
                'n_nonzero': int(rep.get('n_nonzero') or 0),
                'p_floor': _pvr_finite(rep.get('p_floor')),
                'method': rep.get('method'), 'status': 'replayed'})
    # Every stored headline field must agree with the vector it claims to
    # summarise.  This is what turns `per_seed` from decoration into a
    # machine-checkable commitment.
    _stored_p = rec.get('p_exact_signflip')
    if _stored_p is None:
        problems.append(f'{where}: per_seed is recorded but p_exact_signflip is missing')
    elif not _pvr_close(_stored_p, p, rel=0.0, abs_=1e-12):
        problems.append(
            f'{where}: stored p_exact_signflip={_stored_p!r} does not match the '
            f'recomputed exact sign-flip p={p!r}')
    _n = rec.get('n_effective', rec.get('n'))
    if _n is not None:
        _n_int = _pvr_int(_n)
        if _n_int != len(vals):
            problems.append(
                f'{where}: stored n={_n!r} does not equal the number of paired '
                f'deltas ({len(vals)})')
    _nn = rec.get('n_nonzero')
    if _nn is not None:
        _nn_int = _pvr_int(_nn)
        if _nn_int != out['n_nonzero']:
            problems.append(
                f'{where}: stored n_nonzero={_nn!r} does not match the '
                f'recomputed {out["n_nonzero"]}')
    _pf = rec.get('p_floor')
    if _pf is not None and not _pvr_close(_pf, out['p_floor'], rel=0.0, abs_=1e-15):
        problems.append(
            f'{where}: stored p_floor={_pf!r} does not match the recomputed '
            f'{out["p_floor"]!r}')
    _mean = rec.get('mean')
    if _mean is not None:
        # np.mean sums first: finite 1e308-scale deltas overflow to inf even
        # when their mean is representable, which would falsely flag a valid
        # artifact as tampered.  Reuse the same guarded mean the panel uses.
        _mu = L._scaled_mean_or_raw(vals)
        if not _pvr_close(_mean, _mu, rel=1e-9, abs_=1e-12):
            problems.append(
                f'{where}: stored mean={_mean!r} does not match mean(per_seed)={_mu!r}')
    return out


def audit_confirmation_pvalues(path=None):
    """Recompute every stored confirmation p from its paired deltas.

    The three-way verdict audit (:func:`audit_confirmation_verdicts`) replays
    the decision partition from *stored* p-values and deliberately does not
    recompute them.  That leaves the central numbers of the frozen round
    trusted on faith: an edited p, a wrong family, a deleted comparison or a
    stale ``by_length`` curve could remain internally consistent.  This gate
    closes that gap.  For each comparison it takes the recorded per-seed
    paired deltas, reruns the same exact two-sided sign-flip test, and checks
    the stored p / n / non-zero count / resolution floor / mean.  It then
    rebuilds the Holm step-down over the declared family (omitted comparisons
    keep a p=1 slot) and compares every adjusted p.  Every by-length curve
    entry is replayed the same way, and the primary endpoint's delta vector
    must equal the primary curve's.

    A missing ``per_seed`` vector is ``not_replayable`` (a warning, never a
    pass); a present vector whose stored statistic disagrees is ``fail``.
    The audit is read-only: it does not retrain, recompute PPL, read the test
    split, or change any frozen verdict.
    """
    L = _import_exp_lib()
    p = path or os.path.join(REPO, 'analysis_v12', 'test_stats.json')
    out = {'kind': 'confirmation_pvalue_replay', 'status': 'ok', 'path': p,
           'alpha': None, 'n_comparisons': 0, 'n_replayed': 0,
           'n_omitted': 0, 'n_not_replayable': 0, 'n_curve_replayed': 0,
           'comparisons': {}, 'holm': {}, 'problems': [], 'warnings': [],
           'note': ('recomputes the exact two-sided sign-flip p and the Holm '
                    'family from the stored per-seed deltas; a missing vector '
                    'is not a pass and a disagreement is a fail')}
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
    _alpha = _pvr_finite(ft.get('alpha', 0.05))
    if _alpha is None or not (0.0 < _alpha < 1.0):
        out['problems'].append(f'alpha={ft.get("alpha")!r} is not in (0,1)')
        _alpha = 0.05
    out['alpha'] = float(_alpha)
    out['n_comparisons'] = len(comps)

    # The artifact's own declaration of the frozen family.  Without it a
    # deletion is invisible; with it the comparison set must be exact.
    _man = ft.get('manifest') if isinstance(ft.get('manifest'), dict) else {}
    _seeds = _man.get('seeds')
    _declared = _man.get('primary_comparison_names')
    # The frozen seed list is the whitelist every replayable per-seed vector
    # must be drawn from.  A comparison may be missing a seed (a dropped cell),
    # but it may never introduce a seed the plan did not pre-register: that is
    # how a hand-edited artifact would smuggle in a different observation while
    # keeping the same comparison name and family size.
    _seed_whitelist = None
    if isinstance(_seeds, list):
        _wl = []
        _wl_ok = True
        for i, s in enumerate(_seeds):
            si = _pvr_int(s)
            if si is None or si < 0 or si > int(L.MAX_SEED):
                out['problems'].append(
                    f'manifest.seeds[{i}]={s!r} is not an exact seed integer '
                    f'in [0, {int(L.MAX_SEED)}]')
                _wl_ok = False
                break
            _wl.append(si)
        if _wl_ok:
            if len(set(_wl)) != len(_wl):
                out['problems'].append(
                    f'manifest.seeds contains duplicates: {_wl}')
            else:
                _seed_whitelist = set(_wl)
    elif _seeds is not None:
        out['problems'].append(
            f'manifest.seeds is a {type(_seeds).__name__}, not a list')
    if isinstance(_declared, list):
        _decl = [str(x) for x in _declared]
        if len(set(_decl)) != len(_decl):
            out['problems'].append(
                f'manifest.primary_comparison_names has duplicates: {_decl}')
        if set(_decl) != set(map(str, comps)):
            out['problems'].append(
                'the comparison set does not equal the declared frozen family: '
                f'missing={sorted(set(_decl) - set(map(str, comps)))} '
                f'extra={sorted(set(map(str, comps)) - set(_decl))}')
        if _man.get('family_size') is not None and _man.get('family_size') != len(_decl):
            out['problems'].append(
                f'manifest.family_size={_man.get("family_size")!r} does not '
                f'equal the declared family length {len(_decl)}')
    elif _declared is None:
        out['warnings'].append(
            'the artifact does not record manifest.primary_comparison_names; '
            'a deleted comparison and its Holm slot cannot be ruled out')
    else:
        out['problems'].append(
            f'manifest.primary_comparison_names is a '
            f'{type(_declared).__name__}, not a list')

    # If the artifact says it was produced under the CURRENT frozen manifest,
    # cross-check its embedded family against that file.  This closes the
    # "edit both the comparison map and its embedded name list" case: the
    # artifact's manifest_sha256 still points at the tracked file, whose
    # primary_comparisons list is independent of the artifact.
    _mfp = os.path.join(REPO, 'confirmation_frozen.json')
    _stored_msha = _man.get('manifest_sha256')
    if os.path.isfile(_mfp):
        try:
            import hashlib
            with open(_mfp, 'rb') as _mfh:
                _mraw = _mfh.read()
            _msha = hashlib.sha256(_mraw).hexdigest()
            _mf = json.loads(_mraw.decode('utf-8'))
            _mnames = [str(c.get('name')) for c in (_mf.get('primary_comparisons') or [])]
            if _stored_msha is None:
                out['warnings'].append(
                    'the artifact does not record manifest_sha256; the frozen '
                    'family cannot be cross-checked against the tracked manifest')
            elif str(_stored_msha) != _msha:
                out['warnings'].append(
                    'the artifact was produced from a different manifest '
                    f'({str(_stored_msha)[:12]}… vs current {_msha[:12]}…); '
                    'its embedded family is used, not the current plan')
            else:
                if isinstance(_declared, list) and _mnames and _mnames != _decl:
                    out['problems'].append(
                        'manifest.primary_comparison_names does not equal the '
                        f'tracked manifest family: embedded={_decl} '
                        f'tracked={_mnames}')
                if set(_mnames) != set(map(str, comps)):
                    out['problems'].append(
                        'the comparison set does not equal the tracked frozen '
                        f'family: missing={sorted(set(_mnames) - set(map(str, comps)))} '
                        f'extra={sorted(set(map(str, comps)) - set(_mnames))}')
                if isinstance(_mf.get('seeds'), list) and _mf.get('seeds') != _seeds:
                    out['problems'].append(
                        'manifest.seeds does not equal the tracked manifest seeds')
        except Exception as _mfe:
            out['warnings'].append(
                f'the tracked manifest could not be cross-checked '
                f'({type(_mfe).__name__}: {_mfe})')
    else:
        out['warnings'].append(
            'confirmation_frozen.json is absent; the embedded family cannot be '
            'cross-checked against the tracked plan')

    family = []
    for name in sorted(map(str, comps)):
        rec = comps[name]
        where = f'comparison {name!r}'
        if not isinstance(rec, dict):
            out['problems'].append(f'{where}: record is a {type(rec).__name__}, not an object')
            family.append((name, 1.0))
            continue
        if 'omitted' in rec:
            out['n_omitted'] += 1
            if rec.get('p_exact_signflip') is not None:
                out['warnings'].append(
                    f'{where}: omitted comparison carries p_exact_signflip; '
                    'the Holm slot is still p=1')
            family.append((name, 1.0))
            out['comparisons'][name] = {'status': 'omitted', 'family_p': 1.0}
            continue
        rep = _pvr_replay_one(L, name, rec, where=where,
                              problems=out['problems'], warnings=out['warnings'])
        out['comparisons'][name] = rep
        if _seed_whitelist is not None and rep.get('seeds') is not None:
            _outside = sorted(set(rep['seeds']) - _seed_whitelist)
            if _outside:
                out['problems'].append(
                    f'{where}: seeds {_outside} are not in the frozen '
                    'manifest.seeds whitelist')
        if rep['status'] == 'replayed':
            out['n_replayed'] += 1
            family.append((name, float(rep['p'])))
        else:
            out['n_not_replayable'] += 1
            # Keep the stored p for an *indicative* family rebuild, but do not
            # compare Holm unless every slot is replayable (otherwise a legacy
            # or malformed record would manufacture a false mismatch).
            _sp = _pvr_finite(rec.get('p_exact_signflip'))
            family.append((name, _sp if _sp is not None else 1.0))
        # By-length curve: same exact vector replay for every descriptive
        # endpoint.  The primary curve must equal the primary record vector.
        _curve = rec.get('by_length_curve')
        if isinstance(_curve, dict):
            for ln, crec in sorted(_curve.items(), key=lambda kv: str(kv[0])):
                if not isinstance(crec, dict):
                    out['problems'].append(
                        f'{where} L{ln}: curve cell is a {type(crec).__name__}, not an object')
                    continue
                crep = _pvr_replay_one(
                    L, name, crec, where=f'{where} by_length L{ln}',
                    problems=out['problems'], warnings=out['warnings'])
                if crep['status'] == 'replayed':
                    out['n_curve_replayed'] += 1
                if _seed_whitelist is not None and crep.get('seeds') is not None:
                    _coutside = sorted(set(crep['seeds']) - _seed_whitelist)
                    if _coutside:
                        out['problems'].append(
                            f'{where} by_length L{ln}: seeds {_coutside} are '
                            'not in the frozen manifest.seeds whitelist')
                if (crep['status'] == 'replayed' and 'primary' in str(crec.get('role', ''))
                        and rep['status'] == 'replayed'
                        and len(crep['per_seed']) == len(rep['per_seed'])):
                    _a = np.asarray(crep['per_seed'], dtype=float)
                    _b = np.asarray(rep['per_seed'], dtype=float)
                    if not np.allclose(_a, _b, rtol=1e-12, atol=1e-12):
                        out['problems'].append(
                            f'{where}: the primary by-length curve per_seed '
                            'vector differs from the primary comparison vector')

    _holm = ft.get('holm')
    if not isinstance(_holm, dict):
        out['problems'].append('holm is missing or not a JSON object')
        _holm = {}
    else:
        _missing = sorted(set(map(str, comps)) - set(map(str, _holm)))
        _extra = sorted(set(map(str, _holm)) - set(map(str, comps)))
        if _missing:
            out['problems'].append(f'holm is missing declared comparisons: {_missing}')
        if _extra:
            out['problems'].append(f'holm contains undeclared comparisons: {_extra}')
    _rebuilt = L.holm_adjust(family, alpha=float(_alpha))
    out['holm'] = {'m': len(family),
                   'recomputed': {k: v.get('p_holm') for k, v in _rebuilt.items()},
                   'stored': {k: (v.get('p_holm') if isinstance(v, dict) else None)
                              for k, v in _holm.items()}}
    _fully_replayable = (out['n_replayed'] == len(comps) - out['n_omitted'])
    if _fully_replayable:
        for name in sorted(map(str, comps)):
            _srec = _holm.get(name)
            _want = (_rebuilt.get(name) or {}).get('p_holm')
            if not isinstance(_srec, dict):
                out['problems'].append(
                    f'holm[{name!r}] is missing or not an object')
                continue
            _got = _pvr_finite(_srec.get('p_holm'))
            if _got is None or _want is None or not _pvr_close(_got, _want, rel=0.0, abs_=1e-12):
                out['problems'].append(
                    f'holm[{name!r}]={_srec.get("p_holm")!r} does not match '
                    f'the recomputed {_want!r} over the declared family')
            _me = _srec.get('m_effective')
            if _me is not None and _me != len(family):
                out['problems'].append(
                    f'holm[{name!r}].m_effective={_me!r} does not match the '
                    f'declared family size {len(family)}')
    else:
        out['warnings'].append(
            'the Holm family was not rebuilt against the stored values because '
            f'{out["n_not_replayable"]} comparison(s) have no replayable delta vector')

    # The minimum attainable Holm p depends only on (n, m, alpha); it is a
    # useful tripwire for a silently edited seed list or family size.
    _seeds = _man.get('seeds')
    _m = len(_declared) if isinstance(_declared, list) else len(comps)
    if isinstance(_seeds, list) and _seeds:
        try:
            _n_seeds = len(_seeds)
            _want_min = L.holm_min_attainable_p(_n_seeds, int(_m), float(_alpha))
            _got_min = _pvr_finite(ft.get('min_attainable_p_holm'))
            if _got_min is not None and not _pvr_close(_got_min, _want_min, rel=0.0, abs_=1e-12):
                out['problems'].append(
                    f'min_attainable_p_holm={ft.get("min_attainable_p_holm")!r} '
                    f'does not match holm_min_attainable_p(n={_n_seeds}, '
                    f'm={int(_m)}, alpha={float(_alpha)})={_want_min!r}')
        except Exception:
            pass
    else:
        out['warnings'].append(
            'manifest.seeds is missing; the attainable-resolution tripwire '
            'could not run')

    if out['problems']:
        out['status'] = 'fail'
    elif out['n_replayed'] == 0:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = 'no comparison records a replayable per_seed vector'
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
                # Delta units are arbitrary; a stale look must be detected by
                # a scale-relative comparison against the current record, not
                # by an absolute 1e-9 margin that ignores a tiny panel.
                _look_scale = L._look_values_scale(
                    list(cur.values())
                    + [d for _lk in looks for d in _lk['deltas']])
                valid, stale = [], []
                for lk in looks:
                    _bad = [s for s, d in zip(lk['seeds'], lk['deltas'])
                            if s in cur
                            and not L._look_values_close(d, cur[s], _look_scale)]
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


def audit_error_case_inference(path=None):
    """Replay the split-half paired error-bucket inference from its artifact.

    The bad-case Markdown is a claim surface: it names a bucket as worse and
    reports a replication-half interval.  This gate re-derives the bucket
    means, the discovery-only selection, the replication sign/CI decision and
    the within-family Holm correction from the stored cluster-level units.  It
    reads a local artifact only (zero GPU) and starts no experiment.
    """
    L = _import_exp_lib()
    if path is None:
        path = os.path.join(REPO, 'analysis_errors', 'paired_errors.json')
    path = os.path.abspath(str(path))
    out = {'kind': 'error_case_inference', 'status': 'ok', 'path': path,
           'policy_version_expected': L.ERROR_BUCKET_POLICY['version'],
           'families': {}, 'problems': [], 'warnings': []}
    if not os.path.exists(path):
        out['status'] = 'not_verifiable'
        out['warnings'].append(f'no paired error artifact at {path}; run error_analysis.py --compare-ckpt first')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    try:
        with open(path, encoding='utf-8') as f:
            doc = json.load(f)
    except Exception as exc:
        out['status'] = 'fail'
        out['problems'].append(f'artifact is unreadable: {type(exc).__name__}: {exc}')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    fams = doc.get('bucket_inference') if isinstance(doc, dict) else None
    if not isinstance(fams, list) or not fams:
        out['status'] = 'not_verifiable'
        out['warnings'].append('artifact has no bucket_inference block (legacy or aborted analysis)')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out

    def _finite(x):
        if isinstance(x, (bool, np.bool_)):
            return None
        try:
            v = float(x)
        except (TypeError, ValueError, OverflowError):
            return None
        return v if math.isfinite(v) else None

    def _close(a, b, tol):
        return (a is not None and b is not None and abs(a - b) <= tol)

    def _close_optional(a, b, tol):
        if a is None and b is None:
            return True
        return _close(a, b, tol)

    def _mean(vals):
        return float(np.mean(vals)) if vals else None

    def _subset(units, keep):
        return [u for u in units if str(u.get('cluster')) in keep]

    for fi, fam in enumerate(fams):
        tag = f'family[{fi}]'
        if not isinstance(fam, dict):
            out['problems'].append(f'{tag}: not an object')
            continue
        fname = str(fam.get('family') or tag)
        tag = f'{fname}'
        rec = {'status': fam.get('status'), 'problems': []}
        out['families'][tag] = rec
        _p = rec['problems']
        if fam.get('policy_version') != L.ERROR_BUCKET_POLICY['version']:
            _p.append(f"policy version {fam.get('policy_version')!r} != "
                      f"{L.ERROR_BUCKET_POLICY['version']!r}")
        alpha = _finite(fam.get('alpha'))
        direction = fam.get('direction')
        min_clusters = _finite(fam.get('min_clusters'))
        min_tokens = _finite(fam.get('min_tokens'))
        frac = _finite(fam.get('discovery_frac'))
        seed = fam.get('split_seed')
        if alpha is None or not (0.0 < alpha < 1.0):
            _p.append('alpha missing/invalid')
        if direction not in ('positive', 'negative', 'two-sided'):
            _p.append(f'direction {direction!r} invalid')
        if min_clusters is None or min_clusters < 2 or min_clusters != int(min_clusters):
            _p.append('min_clusters missing/invalid')
        if min_tokens is None or min_tokens < 1 or min_tokens != int(min_tokens):
            _p.append('min_tokens missing/invalid')
        if frac is None or not (0.0 < frac < 1.0):
            _p.append('discovery_frac missing/invalid')
        if isinstance(seed, bool) or not isinstance(seed, int):
            _p.append('split_seed missing/invalid')
        if _p:
            rec['status'] = 'fail'
            continue
        min_clusters = int(min_clusters)
        min_tokens = int(min_tokens)
        buckets = fam.get('buckets') if isinstance(fam.get('buckets'), dict) else {}
        if fam.get('status') == 'no_data':
            if buckets:
                _p.append('status=no_data must not carry buckets')
            if fam.get('selected') is not None:
                _p.append('status=no_data must not carry a selected bucket')
            if fam.get('split') is not None:
                _p.append('status=no_data must not carry a split')
            if _finite(fam.get('n_tokens')) != 0.0:
                _p.append('status=no_data must record zero finite paired tokens')
            rec['n_clusters'] = 0
            rec['status'] = 'fail' if _p else 'ok'
            continue
        if not buckets:
            _p.append('no buckets recorded')
            rec['status'] = 'fail'
            continue
        units_by_bucket = {}
        all_clusters = set()
        ok_units = True
        for b, brec in sorted(buckets.items()):
            units = brec.get('cluster_units') if isinstance(brec, dict) else None
            if not isinstance(units, list):
                _p.append(f'bucket {b!r}: cluster_units missing (cannot replay)')
                ok_units = False
                continue
            seen = set()
            cleaned = []
            for u in units:
                if not isinstance(u, dict):
                    _p.append(f'bucket {b!r}: malformed cluster unit')
                    ok_units = False
                    continue
                c = u.get('cluster')
                m = _finite(u.get('mean_delta'))
                ntok = _finite(u.get('n_tokens'))
                if c is None or m is None or ntok is None or ntok < 1:
                    _p.append(f'bucket {b!r}: malformed cluster unit {u!r}')
                    ok_units = False
                    continue
                c = str(c)
                if c in seen:
                    _p.append(f'bucket {b!r}: duplicate cluster {c!r}')
                    ok_units = False
                seen.add(c)
                all_clusters.add(c)
                cleaned.append({'cluster': c, 'mean_delta': m, 'n_tokens': int(ntok)})
            units_by_bucket[b] = cleaned
            vals = [u['mean_delta'] for u in cleaned]
            ntok = sum(u['n_tokens'] for u in cleaned)
            if not _close(_mean(vals), _finite(brec.get('mean')), 1e-9):
                _p.append(f'bucket {b!r}: stored mean != replay')
            if int(brec.get('n_clusters') or 0) != len(cleaned):
                _p.append(f'bucket {b!r}: n_clusters != replay')
            if int(brec.get('n_tokens') or 0) != ntok:
                _p.append(f'bucket {b!r}: n_tokens != replay')
            p = L.exact_signflip_p(vals)
            pv = _finite(p.get('p'))
            if not _close(pv, _finite(brec.get('p_exact_signflip')), 1e-12):
                _p.append(f'bucket {b!r}: exact sign-flip p != replay')
        if not ok_units:
            rec['status'] = 'fail'
            continue
        # Recompute the within-family Holm correction from the replayed p-values.
        raw_pairs = []
        for b in sorted(buckets):
            vals = [u['mean_delta'] for u in units_by_bucket[b]]
            raw_pairs.append((b, _finite(L.exact_signflip_p(vals).get('p'))))
        holm = L.holm_adjust(raw_pairs, alpha=alpha)
        for b in sorted(buckets):
            stored = _finite(buckets[b].get('p_holm'))
            replay = _finite((holm.get(b) or {}).get('p_holm'))
            if stored is not None or replay is not None:
                if not _close(stored, replay, 1e-12):
                    _p.append(f'bucket {b!r}: Holm p != replay')
        split = fam.get('split')
        if fam.get('status') == 'insufficient_clusters':
            if split is not None:
                _p.append('insufficient_clusters status must not carry a split')
            if fam.get('selected') is not None:
                _p.append('insufficient_clusters status must not carry a selected bucket')
            if len(all_clusters) >= 2 * min_clusters:
                _p.append('insufficient_clusters but replay finds enough clusters for two halves')
            rec['n_clusters'] = len(all_clusters)
            rec['status'] = 'fail' if _p else 'ok'
            continue
        if not isinstance(split, dict):
            _p.append('split metadata missing')
            rec['status'] = 'fail'
            continue
        disc = split.get('discovery_clusters')
        repl = split.get('replication_clusters')
        order_raw = split.get('cluster_order')
        if not (isinstance(disc, list) and isinstance(repl, list)):
            _p.append('split cluster lists missing')
            rec['status'] = 'fail'
            continue
        disc_raw = [str(x) for x in disc]
        repl_raw = [str(x) for x in repl]
        disc = set(disc_raw)
        repl = set(repl_raw)
        if disc & repl:
            _p.append('discovery/replication clusters overlap')
        if disc | repl != all_clusters:
            _p.append('split does not partition the observed clusters')
        if len(disc_raw) != len(disc) or len(repl_raw) != len(repl):
            _p.append('split cluster lists contain duplicates')
        if split.get('seed') != seed:
            _p.append(f"split seed {split.get('seed')!r} != declared split_seed {seed!r}")
        if split.get('strategy') != 'sha256(seed, cluster) whole-cluster order; discovery then replication':
            _p.append(f"split strategy {split.get('strategy')!r} is not the audited SHA-256 rule")
        if not isinstance(order_raw, list):
            _p.append('split cluster_order missing (cannot verify the SHA-256 split)')
            rec['status'] = 'fail'
            continue
        ordered = [str(x) for x in order_raw]
        if len(ordered) != len(set(ordered)) or set(ordered) != all_clusters:
            _p.append('cluster_order is not a permutation of the observed clusters')
        expected_order = L._error_bucket_split_order(all_clusters, seed)
        if ordered != expected_order:
            _p.append('cluster_order != SHA-256(split_seed, cluster) order')
        if disc_raw != [c for c in ordered if c in disc]:
            _p.append('discovery cluster order was not preserved')
        if repl_raw != [c for c in ordered if c in repl]:
            _p.append('replication cluster order was not preserved')
        n_disc = int(round(frac * len(ordered)))
        n_disc = max(min_clusters, min(len(ordered) - min_clusters, n_disc))
        expected_disc = set(ordered[:n_disc])
        if disc != expected_disc:
            _p.append('discovery half is not the SHA-256 cluster split')
        if split.get('disjoint') is not True:
            _p.append('split metadata does not declare disjoint halves')
        if split.get('n_discovery') != len(disc_raw):
            _p.append('split n_discovery != replay')
        if split.get('n_replication') != len(repl_raw):
            _p.append('split n_replication != replay')
        # Replay each half's mean/tokens and the discovery-only selection.
        cand = []
        for b in sorted(buckets):
            for half, keep in (('discovery', disc), ('replication', repl)):
                sub = _subset(units_by_bucket[b], keep)
                vals = [u['mean_delta'] for u in sub]
                ntok = sum(u['n_tokens'] for u in sub)
                st = buckets[b].get(half) if isinstance(buckets[b].get(half), dict) else {}
                if not _close_optional(_mean(vals), _finite(st.get('mean')), 1e-9):
                    _p.append(f'bucket {b!r}: {half} mean != replay')
                if int(st.get('n_clusters') or 0) != len(sub):
                    _p.append(f'bucket {b!r}: {half} n_clusters != replay')
                if int(st.get('n_tokens') or 0) != ntok:
                    _p.append(f'bucket {b!r}: {half} n_tokens != replay')
            dsub = _subset(units_by_bucket[b], disc)
            dvals = [u['mean_delta'] for u in dsub]
            dm = _mean(dvals)
            if (len(dsub) >= min_clusters and sum(u['n_tokens'] for u in dsub) >= min_tokens
                    and dm is not None
                    and ((direction == 'positive' and dm > 0.0)
                         or (direction == 'negative' and dm < 0.0)
                         or (direction == 'two-sided' and dm != 0.0))):
                cand.append((b, dm))
        if direction == 'positive':
            exp_sel = sorted(cand, key=lambda x: (-x[1], x[0]))
        elif direction == 'negative':
            exp_sel = sorted(cand, key=lambda x: (x[1], x[0]))
        else:
            exp_sel = sorted(cand, key=lambda x: (-abs(x[1]), x[0]))
        selected = fam.get('selected') if isinstance(fam.get('selected'), dict) else None
        if fam.get('status') == 'no_candidate':
            if exp_sel:
                _p.append(f'status=no_candidate but replay finds candidate {exp_sel[0][0]!r}')
            if selected is not None:
                _p.append('status=no_candidate must not carry a selected bucket')
        elif fam.get('status') == 'ok':
            if not exp_sel:
                _p.append('status=ok but replay finds no eligible discovery bucket')
            elif selected is None:
                _p.append('status=ok but selected bucket is missing')
            else:
                sel_b = selected.get('bucket')
                if sel_b != exp_sel[0][0]:
                    _p.append(f'selected bucket {sel_b!r} != replay {exp_sel[0][0]!r}')
                if sel_b in units_by_bucket:
                    dsub = _subset(units_by_bucket[sel_b], disc)
                    dvals = [u['mean_delta'] for u in dsub]
                    dstat = L._error_cluster_stats(
                        dvals, [u['n_tokens'] for u in dsub])
                    rsub = _subset(units_by_bucket[sel_b], repl)
                    rvals = [u['mean_delta'] for u in rsub]
                    rntok = [u['n_tokens'] for u in rsub]
                    rst = L._error_cluster_stats(rvals, rntok)
                    if not _close(_finite(selected.get('discovery_mean')),
                                  _finite(dstat.get('mean')), 1e-9):
                        _p.append('selected discovery mean != replay')
                    if not _close_optional(_finite(selected.get('replication_mean')),
                                           _finite(rst.get('mean')), 1e-9):
                        _p.append('selected replication mean != replay')
                    _dci = selected.get('discovery_ci95')
                    if (isinstance(_dci, (list, tuple)) and len(_dci) == 2
                            and not (_close_optional(_finite(_dci[0]), _finite(dstat.get('ci95_low')), 1e-9)
                                     and _close_optional(_finite(_dci[1]), _finite(dstat.get('ci95_high')), 1e-9))):
                        _p.append('selected discovery CI != replay')
                    _rci = selected.get('replication_ci95')
                    if (isinstance(_rci, (list, tuple)) and len(_rci) == 2
                            and not (_close_optional(_finite(_rci[0]), _finite(rst.get('ci95_low')), 1e-9)
                                     and _close_optional(_finite(_rci[1]), _finite(rst.get('ci95_high')), 1e-9))):
                        _p.append('selected replication CI != replay')
                    if not _close_optional(_finite(selected.get('replication_p_exact_signflip')),
                                           _finite(rst.get('p_exact_signflip')), 1e-12):
                        _p.append('selected replication exact p != replay')
                    same = (rst.get('mean') is not None
                            and _finite(selected.get('discovery_mean')) is not None
                            and float(selected['discovery_mean']) * float(rst['mean']) > 0.0)
                    rep_ok = (rst.get('n_clusters', 0) >= min_clusters
                              and rst.get('n_tokens', 0) >= min_tokens)
                    replicated = bool(rep_ok and same and rst.get('ci_excludes_zero'))
                    if bool(selected.get('replicated')) != replicated:
                        _p.append('replicated verdict != replay')
                    if bool(selected.get('same_sign')) != same:
                        _p.append('same_sign != replay')
                    _ph = _finite((holm.get(sel_b) or {}).get('p_holm'))
                    if not _close(_finite(selected.get('full_sample_p_holm')), _ph, 1e-12):
                        _p.append('selected full-sample Holm p != replay')
        else:
            _p.append(f'unknown family status {fam.get("status")!r}')
        rec['n_clusters'] = len(all_clusters)
        rec['n_candidates_discovery'] = len(cand)
        rec['status'] = 'fail' if _p else 'ok'
    if any(isinstance(r, dict) and r.get('status') == 'fail' for r in out['families'].values()):
        out['status'] = 'fail'
    for k, r in out['families'].items():
        if isinstance(r, dict) and r.get('problems'):
            out['problems'].extend(f"{k}: {x}" for x in r['problems'])
    print(json.dumps(out, ensure_ascii=False, indent=2))
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


def audit_analysis_integrity(paths=None, *, exclude=()):
    """Verify stamped analysis artifacts against their content and inputs.

    A report number is reproducible only if the artifact that states it can be
    tied to the bytes it was derived from.  Every ``stamped_analysis`` now
    carries ``content_sha256`` (the result body, excluding volatile
    provenance) and, when the producer supplies them, ``inputs`` with the
    SHA-256 of each consumed file.  This gate replays both checks:

      * a changed statistic/verdict/curve fails the body digest;
      * a ``bylen.json``/manifest/summary edited after the analysis fails the
        input digest;
      * an input absent from this checkout is an explicit warning (an archived
        artifact remains valid evidence), never a silent pass;
      * a legacy artifact without the digest is ``not_verifiable_locally``,
        not accepted as verified.

    It reads local artifacts only, starts no experiment and changes no
    measurement, p-value, Holm family or frozen verdict.

    ``exclude`` lets ``run_report`` skip the aggregate report it is about to
    replace.  Auditing the previous report inside the new report made the new
    body depend on the previous run's freshness: after any evidence change the
    first rebuild recorded ``fail``, the next recorded ``ok``, so two rebuilds
    of the same evidence could not compare equal.  The standalone
    ``analysis_integrity`` command still audits ``audit_report.json``.
    """
    L = _import_exp_lib()
    import glob as _glob
    excluded = {os.path.abspath(os.fspath(x)) for x in exclude}
    if paths is None:
        roots = [os.path.join(REPO, 'audit_report.json')]
        for pattern in ('analysis_*/*.json', 'analysis_audit/*.json'):
            roots.extend(sorted(_glob.glob(os.path.join(REPO, pattern))))
        paths = roots
    elif isinstance(paths, (str, bytes, os.PathLike)):
        paths = [paths]
    files = []
    seen = set()
    for p in paths:
        p = os.fspath(p)
        if os.path.isdir(p):
            for q in sorted(_glob.glob(os.path.join(p, '*.json'))):
                aq = os.path.abspath(q)
                if aq in excluded or aq in seen:
                    continue
                seen.add(aq)
                files.append(aq)
        elif os.path.isfile(p) and p.endswith('.json'):
            ap = os.path.abspath(p)
            if ap in excluded or ap in seen:
                continue
            seen.add(ap)
            files.append(ap)
    out = {'kind': 'analysis_integrity_audit', 'status': 'ok',
           'artifacts': {}, 'n_checked': 0, 'n_ok': 0, 'n_fail': 0,
           'n_warnings': 0, 'n_legacy': 0, 'n_skipped': 0,
           'n_excluded': len(excluded),
           'problems': [], 'warnings': [],
           'note': ('content-addressed analysis provenance: the result body '
                    'and every locally available input are checked against '
                    'the hashes recorded when the artifact was written; this '
                    'never recomputes or changes a statistical result')}
    for f in sorted(files):
        rec = _load(f)
        if not isinstance(rec, dict):
            out['n_skipped'] += 1
            continue
        # Only stamped analysis artifacts are in scope.  Plain data/summary
        # files are inputs, not derivations; requiring a stamp on them would
        # misreport an untracked input as a failed analysis.
        if 'content_sha256' not in rec and not isinstance(rec.get('provenance'), dict):
            out['n_skipped'] += 1
            continue
        rep = L.verify_stamped_analysis(rec, base=REPO)
        out['n_checked'] += 1
        short = os.path.relpath(f, REPO) if f.startswith(REPO + os.sep) else f
        out['artifacts'][short] = rep
        if rep.get('status') == 'fail':
            out['n_fail'] += 1
            out['problems'].extend(f'{short}: {p}' for p in rep.get('problems') or [])
        elif rep.get('status') == 'warnings':
            out['n_warnings'] += 1
            out['warnings'].extend(f'{short}: {w}' for w in rep.get('warnings') or [])
        elif rep.get('status') == 'not_verifiable_locally':
            out['n_legacy'] += 1
            out['warnings'].append(
                f'{short}: {rep.get("note") or "legacy artifact without a content digest"}')
        else:
            out['n_ok'] += 1
    if out['n_checked'] == 0:
        out['status'] = 'not_verifiable_locally'
        if out['n_excluded']:
            out['reason'] = ('every candidate was the aggregate report being '
                             'replaced; run the standalone analysis_integrity '
                             'command to audit that file, or produce another '
                             'stamped analysis artifact first')
        else:
            out['reason'] = ('no stamped analysis artifact exists yet; run the '
                             'analysis with the current code to produce one')
    elif out['n_fail']:
        out['status'] = 'fail'
    elif out['n_legacy'] or out['n_warnings']:
        out['status'] = 'warnings'
    print(json.dumps({k: v for k, v in out.items() if k != 'artifacts'},
                     ensure_ascii=False, indent=2))
    return out


def audit_variance_components(path=None):
    """Replay the crossed seed x unit variance decomposition from its cells.

    The decomposition is descriptive, but it is still a quotable result: a
    component total that does not equal the sum of its parts, a share that is
    not ``component / total``, a mean variance that does not equal
    ``s/S + u/D + e/(S*D)``, an effective N that does not equal
    ``total / var(mean)``, or an allocation recommendation that does not match
    the finite-difference comparison is a tampered/internally inconsistent
    artifact, not a difference of opinion.  Legacy artifacts without the block
    are ``warnings`` (the analysis predates the field), never a pass.
    """
    L = _import_exp_lib()
    p = path or os.path.join(REPO, 'analysis_v12', 'test_stats.json')
    out = {'kind': 'variance_components_audit', 'status': 'ok', 'path': p,
           'comparisons': {}, 'n_checked': 0, 'n_ok': 0, 'n_legacy': 0,
           'n_not_estimable': 0, 'n_warnings': 0,
           'problems': [], 'warnings': [],
           'note': ('replays the descriptive crossed seed x unit decomposition; '
                    'it adds no test and never changes a frozen p/Holm/verdict')}
    if not os.path.exists(p):
        out['status'] = 'not_verifiable_locally'
        out['reason'] = 'no final-test stats artifact on disk'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    data = _load(p)
    if not isinstance(data, dict):
        out['status'] = 'fail'
        out['problems'].append(f'{p} exists but is not a readable JSON object')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    ft = data.get('finaltest') if isinstance(data.get('finaltest'), dict) else data
    comps = ft.get('comparisons') if isinstance(ft, dict) else None
    if not isinstance(comps, dict) or not comps:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = 'no per-comparison block in the final-test stats artifact'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out

    def _check(name, vc, dsm):
        probs, warns = [], []
        if not isinstance(vc, dict):
            return ['variance_components is not an object'], warns
        status = vc.get('status')
        if status not in ('ok', 'not_estimable'):
            probs.append(f'unknown status {status!r}')
            return probs, warns
        S = vc.get('n_seeds')
        D = vc.get('n_units')
        if (not isinstance(S, int) or isinstance(S, bool) or S < 2
                or not isinstance(D, int) or isinstance(D, bool) or D < 2):
            probs.append(f'n_seeds/n_units must be integers >=2, got {S!r}/{D!r}')
            return probs, warns
        if vc.get('n_cells') != S * D:
            probs.append(f"n_cells {vc.get('n_cells')!r} != n_seeds*n_units {S * D}")
        if isinstance(dsm, dict):
            if dsm.get('n_seeds') is not None and dsm.get('n_seeds') != S:
                probs.append(f"doc_sampling n_seeds {dsm.get('n_seeds')!r} != "
                             f'variance n_seeds {S}')
            if dsm.get('n_units') is not None and dsm.get('n_units') != D:
                probs.append(f"doc_sampling n_units {dsm.get('n_units')!r} != "
                             f'variance n_units {D}')
        if status != 'ok':
            warns.append(vc.get('warnings') or
                         'not estimable (constant panel or unusable scale)')
            return probs, warns

        def _fin(value):
            f = L._finite_scalar(value)
            return f if f is not None else None

        c = vc.get('components')
        if not isinstance(c, dict):
            probs.append('components block missing')
            return probs, warns
        est = {}
        raw = {}
        for key in ('seed', 'unit', 'residual'):
            rec = c.get(key)
            if not isinstance(rec, dict):
                probs.append(f'components[{key!r}] missing')
                continue
            e = _fin(rec.get('estimate'))
            r = _fin(rec.get('raw'))
            if e is None or e < 0.0:
                probs.append(f'components[{key!r}].estimate must be finite >=0, '
                             f'got {rec.get("estimate")!r}')
            else:
                est[key] = e
            if r is None:
                probs.append(f'components[{key!r}].raw must be finite, '
                             f'got {rec.get("raw")!r}')
            else:
                raw[key] = r
        if len(est) != 3:
            return probs, warns
        total = _fin(vc.get('total_variance'))
        total_raw = _fin(vc.get('total_variance_raw'))
        if total is None or total < 0.0:
            probs.append(f'total_variance must be finite >=0, got {vc.get("total_variance")!r}')
            return probs, warns
        if total_raw is None:
            probs.append(f'total_variance_raw must be finite, got {vc.get("total_variance_raw")!r}')
        if not _pvr_close(total, sum(est.values())):
            probs.append(f'total_variance {total!r} != sum(components) '
                         f'{sum(est.values())!r}')
        if total_raw is not None and not _pvr_close(total_raw, sum(raw.values())):
            probs.append(f'total_variance_raw {total_raw!r} != sum(raw) '
                         f'{sum(raw.values())!r}')
        if total <= 0.0:
            probs.append('status ok but total_variance is zero')
            return probs, warns
        for key in ('seed', 'unit', 'residual'):
            share = _fin((c.get(key) or {}).get('share'))
            want = est[key] / total
            if share is None or not _pvr_close(share, want):
                probs.append(f'components[{key!r}].share {share!r} != '
                             f'estimate/total {want!r}')
        icc = vc.get('icc')
        if not isinstance(icc, dict):
            probs.append('icc block missing')
        else:
            for key in ('seed', 'unit', 'residual'):
                got = _fin(icc.get(key))
                want = est[key] / total
                if got is None or not _pvr_close(got, want):
                    probs.append(f'icc[{key!r}] {got!r} != {want!r}')
        mv = vc.get('mean_variance')
        if not isinstance(mv, dict):
            probs.append('mean_variance block missing')
            return probs, warns
        want_mv = {
            'seed': est['seed'] / S,
            'unit': est['unit'] / D,
            'residual': est['residual'] / (S * D),
        }
        want_total = sum(want_mv.values())
        want_cond = want_mv['seed'] + want_mv['residual']
        for key, want in want_mv.items():
            got = _fin(mv.get(key))
            if got is None or not _pvr_close(got, want):
                probs.append(f'mean_variance[{key!r}] {got!r} != {want!r}')
        for key, want in (('total', want_total),
                          ('conditional_on_units', want_cond)):
            got = _fin(mv.get(key))
            if got is None or not _pvr_close(got, want):
                probs.append(f'mean_variance[{key!r}] {got!r} != {want!r}')
        se = _fin(mv.get('se'))
        if se is None or not _pvr_close(se, math.sqrt(want_total)):
            probs.append(f'mean_variance.se {se!r} != sqrt({want_total!r})')
        grand = _fin(vc.get('grand_mean'))
        if grand is None:
            probs.append(f'grand_mean is not finite ({vc.get("grand_mean")!r})')
            return probs, warns
        for label in ('mean_ci', 'mean_ci_conditional_on_units'):
            ci = vc.get(label)
            if not isinstance(ci, dict):
                probs.append(f'{label} missing')
                continue
            lo, hi = _fin(ci.get('low')), _fin(ci.get('high'))
            if lo is None or hi is None or lo > hi or not (lo <= grand <= hi):
                probs.append(f'{label} must be an ordered interval containing '
                             f'grand_mean, got {ci!r}')
            half = _fin(ci.get('halfwidth'))
            if half is not None and lo is not None and hi is not None:
                if not _pvr_close(half, 0.5 * (hi - lo)):
                    probs.append(f'{label}.halfwidth {half!r} != (hi-lo)/2')
        eff = vc.get('effective_n')
        if not isinstance(eff, dict):
            probs.append('effective_n block missing')
        else:
            if eff.get('iid_cells') != S * D:
                probs.append(f"effective_n.iid_cells {eff.get('iid_cells')!r} != {S * D}")
            eq = _fin(eff.get('equivalent'))
            de = _fin(eff.get('design_effect'))
            want_eq = total / want_total if want_total > 0.0 else None
            if want_eq is not None:
                if eq is None or not _pvr_close(eq, want_eq):
                    probs.append(f'effective_n.equivalent {eq!r} != total/var(mean) {want_eq!r}')
                if de is None or not _pvr_close(de, (S * D) / want_eq):
                    probs.append(f'effective_n.design_effect {de!r} != n_cells/equivalent '
                                 f'{(S * D) / want_eq!r}')
        preds = vc.get('prediction_intervals')
        if isinstance(preds, dict):
            for label, ci in preds.items():
                if not isinstance(ci, dict):
                    probs.append(f'prediction_intervals[{label!r}] is not an object')
                    continue
                lo, hi = _fin(ci.get('low')), _fin(ci.get('high'))
                if lo is None or hi is None or lo > hi or not (lo <= grand <= hi):
                    probs.append(f'prediction_intervals[{label!r}] must be an '
                                 f'ordered interval containing grand_mean')
        elif preds is not None:
            probs.append('prediction_intervals is not an object')
        alloc = vc.get('allocation')
        if not isinstance(alloc, dict):
            probs.append('allocation block missing for an estimable panel')
        elif alloc.get('status') == 'not_decidable':
            # This is the residual-only case: neither axis can reduce the
            # mean variance.  A component estimate above zero contradicts it.
            if est['seed'] > 0.0 or est['unit'] > 0.0:
                probs.append('allocation.status is not_decidable but a seed/unit '
                             'component is positive')
        elif alloc.get('status') == 'ok':
            _cur = alloc.get('current')
            if not isinstance(_cur, dict):
                probs.append('allocation.current is missing or not an object')
            else:
                for _key, _want in (('n_seeds', S), ('n_units', D),
                                    ('n_cells', S * D)):
                    _got = _pvr_int(_cur.get(_key))
                    if _got != _want:
                        probs.append(
                            f'allocation.current.{_key}={_cur.get(_key)!r} != '
                            f'{_want}')
                _sr = _fin(_cur.get('seed_to_unit_ratio'))
                if _sr is None or not _pvr_close(_sr, S / float(D)):
                    probs.append(
                        f'allocation.current.seed_to_unit_ratio='
                        f'{_cur.get("seed_to_unit_ratio")!r} != S/D '
                        f'{S / float(D)!r}')
            _s_est, _u_est = est['seed'], est['unit']
            _e_est = est['residual']
            # Fixed-cell finite differences: an extra cell on either axis also
            # reduces the residual term e/(S*D).
            _seed_gain = (_s_est + _e_est / D) / float(S * (S + 1))
            _unit_gain = (_u_est + _e_est / S) / float(D * (D + 1))
            _want_next = 'seed' if _seed_gain > _unit_gain else 'unit'
            if alloc.get('next_cell') != _want_next:
                probs.append(
                    f'allocation.next_cell={alloc.get("next_cell")!r} does '
                    f'not match the finite-difference comparison '
                    f'({_want_next!r})')
            ratio = alloc.get('recommended_seed_to_unit_ratio')
            if _u_est <= 0.0:
                if ratio is not None:
                    probs.append(
                        'allocation.recommended_seed_to_unit_ratio must be '
                        'null when the unit component is zero')
            else:
                want_ratio = (_s_est / _u_est) if _s_est > 0.0 else 0.0
                r = _fin(ratio)
                if r is None or not _pvr_close(r, want_ratio):
                    probs.append(f'allocation.recommended_seed_to_unit_ratio '
                                 f'{ratio!r} != seed/unit {want_ratio!r}')
        else:
            probs.append(
                f'allocation.status={alloc.get("status")!r} is not '
                "'ok' or 'not_decidable'")
        return probs, warns

    for name, comp in comps.items():
        if not isinstance(comp, dict):
            continue
        dsm = comp.get('doc_sampling')
        if not isinstance(dsm, dict):
            continue
        vc = dsm.get('variance_components')
        if vc is None and 'omitted' in comp:
            continue
        if vc is None:
            reason = dsm.get('variance_components_unavailable_reason')
            if reason:
                out['n_not_estimable'] += 1
                out['warnings'].append(f'{name}: {reason}')
            else:
                out['n_legacy'] += 1
                out['warnings'].append(
                    f'{name}: no variance_components block (artifact predates '
                    'the decomposition)')
            continue
        probs, warns = _check(name, vc, dsm)
        out['n_checked'] += 1
        if probs:
            out['comparisons'][name] = {'status': 'fail', 'problems': probs}
            out['problems'].extend(f'{name}: {x}' for x in probs)
        elif warns:
            out['n_warnings'] += 1
            out['comparisons'][name] = {'status': 'warnings', 'warnings': warns}
            out['warnings'].extend(f'{name}: {x}' for x in warns)
        else:
            out['n_ok'] += 1
            out['comparisons'][name] = {'status': 'ok'}
    if out['problems']:
        out['status'] = 'fail'
    elif out['n_checked'] == 0:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = ('no comparison carries a variance_components block; '
                         'run the analysis with the current code to produce one')
    elif out['warnings']:
        out['status'] = 'warnings'
    print(json.dumps({k: v for k, v in out.items() if k != 'comparisons'},
                     ensure_ascii=False, indent=2))
    return out


# Evidence files the aggregate report derives its verdicts from.  The report
# used to be stamped with an EMPTY input block, so editing a results_*/summary
# or analysis_*/test_stats file left `verify_stamped_analysis` green: the top
# level synthesis was content-addressed but not bound to the evidence it read.
# Text/JSON artifacts are hashed; multi-GB tensors/checkpoints are not (their
# identity belongs in the run records / release manifest, not in a report).
_AUDIT_INPUT_SUFFIXES = ('.json', '.jsonl', '.csv', '.tsv', '.md', '.txt',
                         '.yaml', '.yml')
_AUDIT_INPUT_SKIP_DIRS = frozenset({'analysis_audit'})
_AUDIT_INPUT_MAX_BYTES = 64 << 20
_AUDIT_INPUT_ROOT_FILES = ('confirmation_frozen.json',
                          'confirmation_frozen_dynamic.json',
                          'RELEASES.md')


def audit_evidence_inputs(root=None, *, max_file_bytes=_AUDIT_INPUT_MAX_BYTES):
    """Return ``(inputs, scope)`` for the local evidence an audit reads.

    ``inputs`` is a deterministic ``label -> path`` mapping suitable for
    ``exp_lib.stamped_analysis(inputs=...)``.  ``scope`` records the file
    count/bytes and anything deliberately skipped, so a reader can tell
    "bound to every evidence file" from "bound to the small text subset".

    ``analysis_audit/`` is excluded on purpose: it is the audit's own
    append-only trace, and binding a report to a log that the report appends
    to would make every rebuild look like changed evidence.  That ledger has
    its own hash-chain/anchor audit.
    """
    root = os.path.abspath(root or REPO)
    root_real = os.path.realpath(root)
    max_bytes = int(max_file_bytes)
    if max_bytes < 0:
        raise ValueError('max_file_bytes must be a non-negative integer')
    picked = {}
    sizes = {}
    skipped = {'large': [], 'outside_root': [], 'unreadable': []}

    def _inside(path):
        try:
            return os.path.commonpath(
                [root_real, os.path.realpath(path)]) == root_real
        except (OSError, ValueError):
            return False

    def _rel(path):
        try:
            text = os.path.relpath(path, root).replace(os.sep, '/')
        except (OSError, ValueError):
            return path
        return text

    def _add(path):
        path = os.path.abspath(path)
        rel = _rel(path)
        if not _inside(path):
            skipped['outside_root'].append(rel)
            return
        try:
            size = int(os.path.getsize(path))
        except OSError as exc:
            skipped['unreadable'].append(
                {'path': rel, 'error': f'{type(exc).__name__}: {exc}'})
            return
        if size > max_bytes:
            skipped['large'].append(rel)
            return
        picked[rel] = path
        sizes[rel] = size

    for name in _AUDIT_INPUT_ROOT_FILES:
        path = os.path.join(root, name)
        if os.path.isfile(path):
            _add(path)

    import glob as _glob
    for pattern in ('results_*', 'analysis_*'):
        for directory in sorted(_glob.glob(os.path.join(root, pattern))):
            if not os.path.isdir(directory):
                continue
            if os.path.basename(directory) in _AUDIT_INPUT_SKIP_DIRS:
                continue
            for cur, dirs, names in os.walk(directory, followlinks=False):
                dirs[:] = sorted(
                    d for d in dirs
                    if d not in _AUDIT_INPUT_SKIP_DIRS
                    and not d.startswith('.'))
                for name in sorted(names):
                    if not name.lower().endswith(_AUDIT_INPUT_SUFFIXES):
                        continue
                    _add(os.path.join(cur, name))

    inputs = {f'evidence:{rel}': picked[rel] for rel in sorted(picked)}
    scope = {
        'schema': 1,
        'roots': ['results_*', 'analysis_*', *_AUDIT_INPUT_ROOT_FILES],
        'extensions': list(_AUDIT_INPUT_SUFFIXES),
        'n_files': len(picked),
        'n_bytes': sum(sizes[rel] for rel in picked),
        'max_file_bytes': max_bytes,
        'excluded_dirs': sorted(_AUDIT_INPUT_SKIP_DIRS),
        'skipped_large': sorted(skipped['large']),
        'skipped_outside_root': sorted(skipped['outside_root']),
        'skipped_unreadable': sorted(
            skipped['unreadable'], key=lambda e: e.get('path', '')),
        'note': ('text/JSON evidence read by the aggregate audit; the '
                 'analysis_audit ledger is deliberately excluded because the '
                 'report appends to it and it has its own chain audit'),
    }
    return inputs, scope


def _report_timestamp():
    """SOURCE_DATE_EPOCH-aware stamp; invalid epochs fail closed.

    Only an import failure (a source tarball without the full library) falls
    back to the wall clock.  A malformed SOURCE_DATE_EPOCH is a user error and
    must not silently produce a non-reproducible report.
    """
    try:
        L = _import_exp_lib()
    except Exception:
        return time.strftime('%Y-%m-%dT%H:%M:%S')
    return L.reproducible_timestamp()


def run_report(out='audit_report.json'):
    res = {'generated_at': _report_timestamp(), 'source_tree': audit_source_integrity(), 'dependencies': audit_dependencies(), 'data_source': audit_data_source(), 'leakage': audit_leakage(), 'split_discipline': audit_split_discipline(), 'document_unit': audit_document_unit(), 'sample_design': audit_sample_design(), 'data_processing': audit_data_processing(), 'attribution': audit_attribution(), 'boundary': audit_boundary(), 'length': audit_length(), 'fairness': audit_fairness(), 'metrics': audit_metrics(), 'tuning': audit_tuning(), 'simple_baseline': audit_simple_baseline(), 'hypotheses': audit_hypotheses(), 'sesoi': audit_sesoi_declaration(), 'baseline_sesoi': audit_baseline_sesoi(), 'interaction_sesoi': audit_interaction_sesoi(), 'interaction_power': audit_interaction_power(), 'unit_influence': audit_unit_influence(), 'two_level_ci': audit_two_level_ci(), 'variance_components': audit_variance_components(), 'analysis_choices': audit_analysis_choices(), 'analysis_integrity': audit_analysis_integrity(exclude=(os.path.join(REPO, out),)), 'experiment_ledger': audit_experiment_ledger(), 'verdicts': audit_confirmation_verdicts(), 'pvalue_replay': audit_confirmation_pvalues(), 'looks': audit_look_history(), 'methodology': audit_methodology(), 'missingness': audit_missingness(), 'trajectory': audit_trajectory(), 'disposition': audit_run_disposition(), 'joint_correction': audit_joint_correction(), 'single_factor': audit_single_factor(), 'content_controls': audit_content_controls(), 'sink_controls': audit_sink_controls(), 'fusion_controls': audit_fusion_controls(), 'lesion': audit_lesion(), 'endpoint_selection': audit_endpoint_selection(), 'factorial': audit_factorial_design(), 'interaction_coverage': audit_interaction_coverage(verbose=False), 'factorial_anchor': audit_factorial_anchor(verbose=False), 'design_preflight': audit_design_preflight(), 'sensitivity': audit_design_sensitivity(), 'evaluation': audit_evaluation_design(), 'dose_response': audit_dose_response(verbose=False), 'failures': audit_failures(), 'budget': audit_budget(), 'training_health': audit_training_health(), 'efficiency_pairs': audit_efficiency_pairs(), 'error_case_inference': audit_error_case_inference(), 'release': audit_release_readiness(), 'selection': audit_selection(), 'run_order': audit_run_order(), 'run_allocation': audit_run_allocation(), 'evaluation_order': audit_evaluation_order(), 'reproduction': audit_reproduction(), 'calibration': audit_statistical_calibration()}
    L_atomic = None
    try:
        L = _import_exp_lib()
        # The audit verdicts are quotable claims: stamp code/git provenance and
        # append a ledger entry so an audit artifact regenerated after a code
        # change is distinguishable from the original (the ledger lives under
        # analysis_audit/; the artifact itself stays at the documented path).
        # Bind the report to the evidence files it actually read.  Without
        # this, editing a result/analysis input left the report's own digest
        # green and made a stale synthesis look current.
        _inputs, _scope = audit_evidence_inputs()
        res['audit_inputs'] = _scope
        res = L.stamped_analysis('audit_report', res, inputs=_inputs)
        L.atomic_write_json(os.path.join(REPO, out), res, indent=2)
        L.log_experiment_event('analysis_audit', {'kind': 'analysis', 'tracking_schema': 2, 'artifact': out, 'created_at': res.get('created_at'), 'code': res.get('code'), 'label': 'audit_report', 'status': 'ok', 'audits': sorted(k for k in res if k not in ('kind', 'created_at', 'code', 'provenance', 'generated_at'))})
        print(f'[audit] wrote {out}')
    except Exception as e:
        with open(os.path.join(REPO, out), 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f'[audit] wrote {out} (fallback writer: {type(e).__name__})')
    return res


def _run_cli():
    _cmds = {'leakage': audit_leakage, 'split': audit_split_discipline, 'split_discipline': audit_split_discipline, 'document_unit': audit_document_unit, 'unit': audit_document_unit, 'sample_design': audit_sample_design, 'sampling': audit_sample_design, 'sample': audit_sample_design, 'data_processing': audit_data_processing, 'dedup_protocol': audit_data_processing, 'dedup': audit_data_processing, 'cleaning': audit_data_processing, 'causality': audit_causality, 'attribution': audit_attribution, 'boundary': audit_boundary, 'length': audit_length, 'fairness': audit_fairness, 'metrics': audit_metrics, 'tuning': audit_tuning, 'simple': audit_simple_baseline, 'baseline_floor': audit_simple_baseline, 'baseline_plan': audit_baseline_design, 'hypothesis_coverage': audit_hypothesis_coverage, 'hypotheses': audit_hypotheses, 'sesoi': audit_sesoi_declaration, 'baseline_sesoi': audit_baseline_sesoi, 'baseline_sesoi_declaration': audit_baseline_sesoi, 'interaction_sesoi': audit_interaction_sesoi, 'interaction': audit_interaction_sesoi, 'interaction_power': audit_interaction_power, 'factorial_power': audit_interaction_power, 'power_2x2': audit_interaction_power, 'unit_influence': audit_unit_influence, 'influence': audit_unit_influence, 'two_level_ci': audit_two_level_ci, 'two_level': audit_two_level_ci, 'total_uncertainty': audit_two_level_ci, 'joint_ci': audit_two_level_ci, 'variance_components': audit_variance_components, 'variance': audit_variance_components, 'variance_decomposition': audit_variance_components, 'analysis_choices': audit_analysis_choices, 'specification_curve': audit_analysis_choices, 'analysis_integrity': audit_analysis_integrity, 'artifact_integrity': audit_analysis_integrity, 'content_integrity': audit_analysis_integrity, 'experiment_ledger': audit_experiment_ledger, 'event_ledger': audit_experiment_ledger, 'ledger': audit_experiment_ledger, 'verdicts': audit_confirmation_verdicts, 'verdict': audit_confirmation_verdicts, 'three_way': audit_confirmation_verdicts, 'pvalues': audit_confirmation_pvalues, 'pvalue_replay': audit_confirmation_pvalues, 'recompute_p': audit_confirmation_pvalues, 'joint_correction': audit_joint_correction, 'joint': audit_joint_correction, 'maxt': audit_joint_correction, 'max_t': audit_joint_correction, 'analysis_choice': audit_analysis_choices, 'looks': audit_look_history, 'look_history': audit_look_history, 'missingness': audit_missingness, 'missing': audit_missingness, 'missing_data': audit_missingness, 'attrition_sensitivity': audit_missingness, 'trajectory': audit_trajectory, 'traj': audit_trajectory, 'trajectory_stability': audit_trajectory,
                                       'disposition': audit_run_disposition,
                                       'attempts': audit_run_disposition,
                                       'retries': audit_run_disposition,
                                       'run_disposition': audit_run_disposition, 'methodology': audit_methodology, 'single_factor': audit_single_factor, 'content_controls': audit_content_controls, 'sink_controls': audit_sink_controls, 'sink_placebo': audit_sink_controls, 'fusion_controls': audit_fusion_controls, 'fusion_placebo': audit_fusion_controls, 'local_fusion': audit_fusion_controls, 'lesion': audit_lesion, 'inference_lesion': audit_lesion, 'lesion_panel': audit_lesion, 'component_lesion': audit_lesion, 'endpoint': audit_endpoint_selection, 'endpoint_selection': audit_endpoint_selection, 'checkpoint': audit_endpoint_selection, 'checkpoint_selection': audit_endpoint_selection, 'control_ladder': audit_content_controls, 'content_alignment': audit_content_controls, 'factorial': audit_factorial_design, 'factorial_design': audit_factorial_design, 'interaction_coverage': audit_interaction_coverage, 'interaction_scope': audit_interaction_coverage, 'factorial_anchor': audit_factorial_anchor, 'component_anchor': audit_factorial_anchor, 'design': audit_design_preflight, 'design_preflight': audit_design_preflight, 'sensitivity': audit_design_sensitivity, 'design_sensitivity': audit_design_sensitivity, 'precision': audit_design_sensitivity, 'evaluation': audit_evaluation_design, 'evaluation_design': audit_evaluation_design, 'unit_precision': audit_evaluation_design, 'dose_response': audit_dose_response, 'dose_response_design': audit_dose_response, 'failures': audit_failures, 'budget': audit_budget, 'efficiency': audit_efficiency_demo, 'efficiency_pairs': audit_efficiency_pairs, 'training_health': audit_training_health, 'selection': audit_selection, 'release': audit_release_readiness, 'run_order': audit_run_order, 'run_allocation': audit_run_allocation, 'allocation': audit_run_allocation, 'environment_balance': audit_run_allocation, 'evaluation_order': audit_evaluation_order, 'eval_order': audit_evaluation_order, 'measurement_order': audit_evaluation_order, 'reproduction': audit_reproduction, 'reproduce': audit_reproduction, 'replication': audit_reproduction, 'rerun_check': audit_reproduction, 'error_case_inference': audit_error_case_inference, 'error_buckets': audit_error_case_inference, 'calibration': audit_statistical_calibration, 'stats_calibration': audit_statistical_calibration, 'type_i': audit_statistical_calibration, 'source': audit_source_integrity, 'source_tree': audit_source_integrity, 'dependencies': audit_dependencies, 'dependency': audit_dependencies, 'data_source': audit_data_source, 'dataset_source': audit_data_source, 'report': run_report}
    _arg = sys.argv[1] if len(sys.argv) > 1 else 'report'
    if _arg in ('-h', '--help'):
        print(__doc__)
        raise SystemExit(0)
    if _arg not in _cmds:
        print(f'[audit] unknown subcommand {_arg!r}; expected one of {sorted(_cmds)}')
        raise SystemExit(2)
    _pos = sys.argv[2:]
    # Most gates take no positional arguments (they discover their artifacts).
    # The reproduction gate needs the two run directories or summary paths
    # explicitly and must never guess a comparison pair; the p-value replay
    # gate accepts one optional artifact path so a copy can be checked outside
    # the default analysis_v12/ location.
    if _arg in ('reproduction', 'reproduce', 'replication', 'rerun_check',
                'pvalues', 'pvalue_replay', 'recompute_p',
                'missingness', 'missing', 'missing_data', 'attrition_sensitivity',
                'trajectory', 'traj', 'trajectory_stability',
                'disposition', 'attempts', 'retries', 'run_disposition',
                'interaction_power', 'factorial_power', 'power_2x2',
                'lesion', 'inference_lesion', 'lesion_panel',
                'component_lesion', 'endpoint', 'endpoint_selection',
                'checkpoint', 'checkpoint_selection'):
        _r = _cmds[_arg](*_pos)
    elif _arg in ('joint_correction', 'joint', 'maxt', 'max_t'):
        _r = _cmds[_arg](list(_pos) or None)
    else:
        _r = _cmds[_arg]()
    if isinstance(_r, dict) and _r.get('status') == 'fail':
        raise SystemExit(1)
