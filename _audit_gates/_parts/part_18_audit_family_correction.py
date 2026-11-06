# The exact max-T block materialises 2**n_seeds sign patterns.  Its producer
# refuses n above the hard enumeration cap, so a larger declared count can only
# be a malformed artifact; never compute ``1 << n_seeds`` for it (a 1000-digit
# JSON integer would raise OverflowError while the gate is reporting the
# tamper).  This is a reporting guard, not the statistical enumeration cap.
_AFC_MAX_ENUM_SEEDS = 64


def _afc_repr(value, limit=160):
    """Bounded repr for untrusted artifact diagnostics.

    A finite JSON integer can have more digits than CPython's conversion
    limit, so ``repr`` itself raises while a validator is trying to report a
    malformed field.  Never let a diagnostic stringify an unbounded value.
    """
    try:
        if isinstance(value, int) and not isinstance(value, bool):
            bits = int(value.bit_length())
            if bits > 512:
                return f'<unrepresentable int bit_length={bits}>'
        text = repr(value)
    except Exception:
        return f'<unrepresentable {type(value).__name__}>'
    return text if len(text) <= limit else text[:limit - 3] + '...'


def audit_family_correction(path=None):
    """Result-analysis gate: replay the dependence-aware family correction.

    The confirmation report carries a descriptive correction-method block: an
    exact Westfall-Young max-T on the sign-invariant score statistic, the
    same-statistic Holm comparison, the studentised t twin (when available) and
    the family dependence diagnostics.  This gate rebuilds the whole block from
    the stored per-seed deltas and compares it field by field.  It never
    re-decides the frozen confirmation: a tampered p/interval/dependence value,
    a family that no longer matches the declared comparisons, or a block that
    smuggles a frozen decision field is a ``fail``; a legacy artifact without
    the block is ``not_verifiable_locally`` rather than a pass.
    """
    L = _import_exp_lib()
    p = path or os.path.join(REPO, 'analysis_v12', 'test_stats.json')
    out = {'kind': 'family_correction_audit', 'status': 'ok',
           'path': p, 'comparisons': {}, 'n_replayed': 0,
           'n_skipped': 0, 'problems': [], 'warnings': [],
           'note': ('descriptive correction-method/dependence replay; it never '
                    'changes the frozen p/Holm/confirmed verdict')}
    ft = _load_finaltest_stats(out, p)
    if ft is None:
        return out
    block = ft.get('family_correction') if isinstance(ft, dict) else None
    if not isinstance(block, dict):
        out['status'] = 'not_verifiable_locally'
        out['reason'] = 'the stats artifact carries no family_correction block'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    if block.get('kind') != 'confirmation_family_correction':
        out['problems'].append(
            f"family_correction.kind={_afc_repr(block.get('kind'))} is not "
            "'confirmation_family_correction'")
    forbidden = {'confirmed', 'p_holm', 'verified', 'verdict', 'decision'}
    bad_keys = sorted(k for k in block if k in forbidden)
    if bad_keys:
        out['problems'].append(
            'the descriptive family-correction block must not carry frozen '
            f'decision fields: {bad_keys}')
    status = block.get('status')
    if status not in ('ok', 'degenerate', 'not_estimable'):
        out['problems'].append(
            f'family_correction.status {_afc_repr(status)} is not a declared status')
    rows = block.get('comparisons')
    if status in ('not_estimable', 'degenerate') and not rows:
        out['status'] = 'not_applicable'
        out['reason'] = (f'family_correction.status={_afc_repr(status)} with no '
                         'comparison rows')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    names = block.get('family_names')
    if not isinstance(names, list) or not names:
        out['problems'].append('family_correction.family_names is missing/empty')
        names = []
    names = [str(x) for x in names]
    if len(set(names)) != len(names):
        out['problems'].append('family_correction.family_names has a duplicate')
    comps = ft.get('comparisons') if isinstance(ft.get('comparisons'), dict) else {}
    if names and set(names) != set(str(x) for x in comps):
        out['problems'].append(
            'family_correction.family_names does not equal the comparison set: '
            f'{sorted(names)!r} vs {sorted(str(x) for x in comps)!r}')
    manifest_names = ((ft.get('manifest') or {}).get('primary_comparison_names')
                      if isinstance(ft.get('manifest'), dict) else None)
    if isinstance(manifest_names, list) and names:
        if set(str(x) for x in manifest_names) != set(names):
            out['problems'].append(
                'family_correction.family_names does not equal the declared '
                f'manifest family: {sorted(names)!r} vs '
                f'{sorted(str(x) for x in manifest_names)!r}')
    if not isinstance(rows, dict) or not rows:
        out['problems'].append('family_correction carries no comparison rows')
        out['status'] = 'fail'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out

    # Rebuild the exact input the block claims to have consumed.
    replay_in = {}
    for name in names:
        rec = comps.get(name)
        if not isinstance(rec, dict):
            out['problems'].append(f'{name}: comparison record is not an object')
            continue
        if 'omitted' in rec or rec.get('omitted'):
            out['problems'].append(f'{name}: comparison is omitted; the family '
                                   'cannot be replayed as a complete panel')
            continue
        replay_in[name] = {
            'per_seed': rec.get('per_seed'),
            'seeds': rec.get('seeds'),
            'p_exact_signflip': rec.get('p_exact_signflip'),
        }
    if len(replay_in) != len(names):
        out['status'] = 'fail'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    try:
        rep = L.confirmation_family_correction(
            replay_in, family_names=names,
            frozen_holm=ft.get('holm') if isinstance(ft.get('holm'), dict) else None,
            alpha=block.get('alpha'))
    except Exception as e:
        out['problems'].append(f'replay raised {type(e).__name__}: {e}')
        out['status'] = 'fail'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    if rep.get('status') == 'not_estimable':
        out['problems'].append(
            'the stored block is estimable but the replay is not: '
            f'{rep.get("reason")}')
        out['status'] = 'fail'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out

    def _close(a, b, *, rel=1e-9, abs_=1e-12):
        x, y = _finite_number(a), _finite_number(b)
        if x is None or y is None:
            return x is None and y is None
        return math.isclose(x, y, rel_tol=rel, abs_tol=abs_)

    def _diff(path, stored, replayed, *, rel=1e-9, abs_=1e-12):
        if isinstance(stored, dict) and isinstance(replayed, dict):
            sk, rk = set(stored), set(replayed)
            if sk != rk:
                out['problems'].append(
                    f'{path}: keys {_afc_repr(sorted(sk))} != replayed '
                    f'{_afc_repr(sorted(rk))}')
            for k in sorted(sk & rk):
                _diff(f'{path}.{_afc_repr(k)}', stored[k], replayed[k],
                      rel=rel, abs_=abs_)
            return
        if isinstance(stored, list) and isinstance(replayed, list):
            if len(stored) != len(replayed):
                out['problems'].append(
                    f'{path}: length {len(stored)} != replayed {len(replayed)}')
                return
            for i, (a, b) in enumerate(zip(stored, replayed)):
                _diff(f'{path}[{i}]', a, b, rel=rel, abs_=abs_)
            return
        if isinstance(stored, bool) or isinstance(replayed, bool):
            if stored is not replayed:
                out['problems'].append(
                    f'{path}: {_afc_repr(stored)} != replayed '
                    f'{_afc_repr(replayed)}')
            return
        if isinstance(stored, (int, float)) and isinstance(replayed, (int, float)):
            if not _close(stored, replayed, rel=rel, abs_=abs_):
                out['problems'].append(
                    f'{path}: {_afc_repr(stored)} != replayed '
                    f'{_afc_repr(replayed)}')
            return
        if stored != replayed:
            out['problems'].append(
                f'{path}: {_afc_repr(stored)} != replayed {_afc_repr(replayed)}')

    for key in ('status', 'alpha', 'm_declared', 'm_used', 'n_seeds',
                'n_patterns', 'p_floor', 'holm_threshold',
                'holm_gate_attainable', 'max_n', 'seeds', 'sets', 'summary',
                'dependence', 'studentized', 'score_maxt', 'holm_critical_t'):
        if key not in block:
            out['problems'].append(f'family_correction is missing {key!r}')
            continue
        _diff(f'family_correction.{key}', block.get(key), rep.get(key))
    for name in names:
        if name not in rows:
            out['problems'].append(f'{name}: missing comparison row')
            continue
        if name not in (rep.get('comparisons') or {}):
            out['problems'].append(f'{name}: missing replayed comparison row')
            continue
        _diff(f'comparisons[{name!r}]', rows.get(name),
              (rep.get('comparisons') or {}).get(name))
        out['comparisons'][name] = {
            'p_raw_score': (rows.get(name) or {}).get('p_raw_score'),
            'p_maxt_score': (rows.get(name) or {}).get('p_maxt_score'),
            'reject_maxt_score': (rows.get(name) or {}).get('reject_maxt_score'),
        }
        out['n_replayed'] += 1

    # Structural invariants that a self-consistent tamper could otherwise
    # preserve: adjusted p cannot be below the marginal p, the rejection sets
    # must be exactly the rows that reject, and the family layout must agree
    # with the declared family.
    for name in names:
        r = rows.get(name) or {}
        ps = _finite_number(r.get('p_raw_score'))
        pm = _finite_number(r.get('p_maxt_score'))
        ph = _finite_number(r.get('p_holm_score'))
        # The mean columns must keep the raw/adjusted distinction: the raw
        # mean p is the comparison's p_exact_signflip, while the Holm(mean)
        # column must equal the already-adjusted frozen Holm value verbatim.
        # This is checked against the ORIGINAL inputs, not only the replay, so
        # a replay that shares the same bug cannot hide it.
        raw_mean = _finite_number(r.get('p_raw_mean'))
        comp_raw = _finite_number((comps.get(name) or {}).get('p_exact_signflip'))
        if raw_mean is not None and comp_raw is not None:
            if not _close(raw_mean, comp_raw):
                out['problems'].append(
                    f'{name}: p_raw_mean={raw_mean!r} != '
                    f'p_exact_signflip={comp_raw!r}')
        _stored_holm = ft.get('holm')
        if isinstance(_stored_holm, dict):
            _hp = _stored_holm.get(name)
            if isinstance(_hp, dict):
                _hp = _hp.get('p_holm')
            frozen_holm_p = _finite_number(_hp)
            holm_mean_p = _finite_number(r.get('p_holm_mean'))
            if frozen_holm_p is not None and holm_mean_p is not None:
                if not _close(holm_mean_p, frozen_holm_p):
                    out['problems'].append(
                        f'{name}: p_holm_mean={holm_mean_p!r} != frozen '
                        f'p_holm={frozen_holm_p!r}')
        if ps is not None and pm is not None and pm < ps - 1e-12:
            out['problems'].append(
                f'{name}: p_maxt_score={pm!r} < p_raw_score={ps!r}')
        if ps is not None and ph is not None and ph < ps - 1e-12:
            out['problems'].append(
                f'{name}: p_holm_score={ph!r} < p_raw_score={ps!r}')
        if r.get('reject_maxt_score') is not None:
            want = bool((pm if pm is not None else 1.0) < float(block.get('alpha', 0.05)))
            if bool(r.get('reject_maxt_score')) != want:
                out['problems'].append(
                    f'{name}: reject_maxt_score={r.get("reject_maxt_score")!r} '
                    f'disagrees with its p={pm!r}')
        ci = r.get('ci_simultaneous')
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            a0, b0 = _finite_number(ci[0]), _finite_number(ci[1])
            if a0 is None or b0 is None or a0 > b0:
                out['problems'].append(
                    f'{name}: simultaneous interval {_afc_repr(ci)} is not '
                    'ordered/finite')
    ns = block.get('n_seeds')
    if isinstance(ns, int) and not isinstance(ns, bool) and isinstance(
            block.get('seeds'), list):
        if ns < 0:
            out['problems'].append(
                f'n_seeds must be non-negative, got bit_length={ns.bit_length()}')
        elif ns > _AFC_MAX_ENUM_SEEDS:
            # Do not evaluate 1 << ns here: an untrusted artifact can carry a
            # finite integer far beyond the exact-enumeration cap.
            out['problems'].append(
                'n_seeds declares an integer with '
                f'bit_length={ns.bit_length()}, above the supported family '
                f'enumeration cap {_AFC_MAX_ENUM_SEEDS}')
        else:
            if ns != len(block['seeds']):
                out['problems'].append(
                    f'n_seeds={ns} != len(seeds)={len(block["seeds"])}')
            # The producer enumerates only seed columns that are non-zero
            # somewhere in the family; all-zero columns are sign-neutral and
            # are redundant patterns.  Replayed width is authoritative, and a
            # legacy block without the field still checks against n_seeds.
            _rep_n_eff = rep.get('n_seeds_effective')
            if (isinstance(_rep_n_eff, int)
                    and not isinstance(_rep_n_eff, bool)
                    and 0 <= _rep_n_eff <= _AFC_MAX_ENUM_SEEDS):
                n_patterns_seeds = _rep_n_eff
            else:
                n_patterns_seeds = ns
            want_patterns = 1 << n_patterns_seeds
            if n_patterns_seeds > 0 and block.get('n_patterns') != want_patterns:
                out['problems'].append(
                    'n_patterns=%s != 2**effective_n_seeds=%d'
                    % (_afc_repr(block.get('n_patterns')), want_patterns))
            if ('n_seeds_effective' in block
                    and block.get('n_seeds_effective') != _rep_n_eff):
                out['problems'].append(
                    'n_seeds_effective=%s != replayed %s'
                    % (_afc_repr(block.get('n_seeds_effective')),
                       _afc_repr(_rep_n_eff)))
    if block.get('m_declared') != len(names):
        out['problems'].append(
            'm_declared=%s != len(family_names)=%d'
            % (_afc_repr(block.get('m_declared')), len(names)))
    sets = block.get('sets') or {}
    want_maxt = sorted(n for n in names if (rows.get(n) or {}).get('reject_maxt_score') is True)
    want_holm = sorted(n for n in names if (rows.get(n) or {}).get('reject_holm_score') is True)
    if sorted(sets.get('maxt_score') or []) != want_maxt:
        out['problems'].append('sets.maxt_score does not match the rejecting rows')
    if sorted(sets.get('holm_score') or []) != want_holm:
        out['problems'].append('sets.holm_score does not match the rejecting rows')
    want_added = sorted(set(want_maxt) - set(want_holm))
    want_removed = sorted(set(want_holm) - set(want_maxt))
    if sorted(sets.get('maxt_score_added_vs_holm_score') or []) != want_added:
        out['problems'].append('sets.maxt_score_added_vs_holm_score is inconsistent')
    if sorted(sets.get('maxt_score_removed_vs_holm_score') or []) != want_removed:
        out['problems'].append('sets.maxt_score_removed_vs_holm_score is inconsistent')

    if out['problems']:
        out['status'] = 'fail'
    elif out['n_replayed'] == 0:
        out['status'] = 'not_applicable'
        out['reason'] = 'no comparison carried a replayable family-correction row'
    else:
        out['status'] = 'ok'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out
