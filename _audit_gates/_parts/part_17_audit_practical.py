def audit_practical_significance(path=None):
    """Result-analysis gate: replay the SESOI/TOST block from its own seeds.

    The confirmation report carries a descriptive practical-significance block
    (relative SESOI, TOST equivalence, beyond-margin p).  This gate does not
    re-decide the frozen confirmation: it independently rebuilds the block from
    the stored per-seed arm PPLs and checks the recorded classification,
    intervals and the two Holm families.  Tampering, a non-replayable block or
    a block that smuggles frozen decision fields is a ``fail``; a legacy
    artifact without the block is ``not_verifiable_locally`` rather than a
    pass.
    """
    L = _import_exp_lib()
    p = path or os.path.join(REPO, 'analysis_v12', 'test_stats.json')
    out = {'kind': 'practical_significance_audit', 'status': 'ok',
           'path': p, 'comparisons': {}, 'n_replayed': 0,
           'n_skipped': 0, 'n_missing': 0,
           'problems': [], 'warnings': [],
           'note': ('descriptive SESOI/TOST replay; it never changes the frozen '
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
    block = ft.get('practical_significance') if isinstance(ft, dict) else None
    if not isinstance(block, dict):
        out['status'] = 'not_verifiable_locally'
        out['reason'] = 'the stats artifact carries no practical_significance block'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    rows = block.get('comparisons')
    _declared_names = block.get('family_names')
    _comparison_names = set(ft.get('comparisons') or {}) if isinstance(ft, dict) else set()
    if isinstance(_declared_names, list) and _comparison_names:
        if set(str(x) for x in _declared_names) != _comparison_names:
            out['problems'].append(
                'practical_significance.family_names does not equal the '
                f'comparison set: declared={sorted(str(x) for x in _declared_names)!r}, '
                f'comparisons={sorted(_comparison_names)!r}; a missing slot would '
                'shrink the correction family')
    if not isinstance(rows, dict) or not rows:
        if block.get('status') in ('not_estimable', 'not_declared'):
            out['status'] = 'not_applicable'
            out['reason'] = f'practical block status={block.get("status")!r} with no rows'
        else:
            out['status'] = 'fail'
            out['problems'].append('practical_significance carries no comparison rows')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    forbidden = {'confirmed', 'p_holm', 'verified', 'verdict', 'decision'}
    # Only the block's own fields and each comparison row are checked; the
    # nested `holm` tables legitimately carry a `p_holm` key for the two
    # descriptive SESOI families.
    bad_keys = sorted({k for k in block if k in forbidden})
    for _row in (rows or {}).values():
        if isinstance(_row, dict):
            bad_keys.extend(k for k in _row if k in forbidden)
    bad_keys = sorted(set(bad_keys))
    if bad_keys:
        out['problems'].append(
            'the descriptive practical block must not carry frozen decision '
            f'fields: {bad_keys}')

    def _fin(x):
        if isinstance(x, (bool, np.bool_, str, bytes, bytearray)) or x is None:
            return None
        try:
            y = float(x)
        except (TypeError, ValueError, OverflowError):
            return None
        return y if math.isfinite(y) else None

    def _close(a, b, *, abs_tol=1e-12, rel_tol=1e-9):
        x, y = _fin(a), _fin(b)
        if x is None or y is None:
            return x is None and y is None
        return math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol)

    def _compare(name, recomputed, stored):
        problems = []
        for key in ('status', 'n', 'expected_sign', 'scale'):
            if recomputed.get(key) != stored.get(key):
                problems.append(f'{name}: {key}={stored.get(key)!r} != replayed {recomputed.get(key)!r}')
        for key in ('p_tost', 'p_beyond', 'p_lo', 'p_hi', 'mean',
                    'relative_effect_pct', 'relative_pct'):
            if not _close(recomputed.get(key), stored.get(key)):
                problems.append(f'{name}: {key}={stored.get(key)!r} != replayed {recomputed.get(key)!r}')
        for key in ('equivalent', 'beyond_sesoi'):
            rv, sv = recomputed.get(key), stored.get(key)
            if rv is not sv and not (rv is None and sv is None):
                problems.append(f'{name}: {key}={sv!r} != replayed {rv!r}')
        rci, sci = recomputed.get('ci'), stored.get('ci')
        if rci is None or sci is None:
            if not (rci is None and sci is None):
                problems.append(f'{name}: ci={sci!r} != replayed {rci!r}')
        elif (not isinstance(sci, (list, tuple)) or len(sci) != 2
              or not _close(rci[0], sci[0]) or not _close(rci[1], sci[1])):
            problems.append(f'{name}: ci={sci!r} != replayed {rci!r}')
        return problems

    comps = ft.get('comparisons') if isinstance(ft.get('comparisons'), dict) else {}
    for name in (block.get('family_names') or sorted(rows)):
        name = str(name)
        stored = rows.get(name)
        if not isinstance(stored, dict):
            out['problems'].append(f'{name}: missing practical row')
            continue
        # The aggregate row is compact; the full per-seed block lives next to
        # the frozen comparison.  Replay the full block when present and also
        # compare its summary fields to the aggregate row, so tampering with
        # either copy is caught.
        inner = (comps.get(name) or {}).get('practical_significance') \
            if isinstance(comps.get(name), dict) else None
        source = inner if isinstance(inner, dict) else stored
        if source.get('status') in ('not_declared',) or source.get('margin') is None:
            out['n_skipped'] += 1
            continue
        margin = _fin(source.get('margin'))
        if margin is None or margin <= 0.0:
            out['problems'].append(f'{name}: margin is not finite/positive')
            continue
        _pol = block.get('policy') if isinstance(block.get('policy'), dict) else {}
        _pm = ((_pol.get('margins') or {}).get(name) or {}).get('margin') \
            if isinstance(_pol.get('margins'), dict) else None
        if _pm is not None and not _close(_pm, margin):
            out['problems'].append(
                f'{name}: row margin={margin!r} != policy margin={_pm!r}')
        if _fin(_pol.get('alpha')) is not None and not _close(_pol.get('alpha'), block.get('alpha')):
            out['problems'].append(
                f'policy.alpha={_pol.get("alpha")!r} != family alpha={block.get("alpha")!r}')
        if _fin(_pol.get('tost_conf')) is not None and not _close(
                _pol.get('tost_conf'), block.get('tost_conf')):
            out['problems'].append(
                f'policy.tost_conf={_pol.get("tost_conf")!r} != family '
                f'tost_conf={block.get("tost_conf")!r}')
        alpha = _fin(source.get('alpha', block.get('alpha')))
        tost_conf = _fin(source.get('tost_conf', block.get('tost_conf')))
        scale = source.get('scale', block.get('scale', 'log_ratio'))
        seeds = source.get('seeds')
        if seeds is not None and not isinstance(seeds, list):
            out['problems'].append(f'{name}: seeds is not a list')
            continue
        pa = source.get('per_seed_ppl_a')
        pb = source.get('per_seed_ppl_b')
        stored_lr = source.get('per_seed_log_ratio')
        d = None
        if (isinstance(pa, list) and isinstance(pb, list) and pa and len(pa) == len(pb)
                and (seeds is None or len(seeds) == len(pa))):
            try:
                vals = [(float(a), float(b)) for a, b in zip(pa, pb)]
            except (TypeError, ValueError, OverflowError):
                vals = None
            if vals is not None and all(a > 0.0 and b > 0.0
                                        and math.isfinite(a) and math.isfinite(b)
                                        for a, b in vals):
                d = [math.log(a) - math.log(b) for a, b in vals]
                # The stored per-seed vector must equal the one rebuilt from
                # the two arms; otherwise the block and its cells disagree.
                if isinstance(stored_lr, list) and len(stored_lr) == len(d):
                    for i, (x, y) in enumerate(zip(d, stored_lr)):
                        if not _close(x, y):
                            out['problems'].append(
                                f'{name}: per_seed_log_ratio[{i}]={y!r} != rebuilt {x!r}')
                            break
                elif stored_lr is not None:
                    out['problems'].append(
                        f'{name}: per_seed_log_ratio is not a same-length list')
                raw_seed = (comps.get(name) or {}).get('per_seed')
                if isinstance(raw_seed, list) and len(raw_seed) == len(d):
                    for i, ((a, b), raw) in enumerate(zip(vals, raw_seed)):
                        if not _close(a - b, raw):
                            out['problems'].append(
                                f'{name}: frozen per_seed[{i}]={raw!r} != arm difference {a - b!r}')
                            break
        elif isinstance(stored_lr, list) and stored_lr:
            d = [_fin(x) for x in stored_lr]
            if any(x is None for x in d):
                d = None
        if not d:
            out['problems'].append(
                f'{name}: no replayable per-seed arm PPLs or log ratios')
            continue
        try:
            rep = L.confirmation_practical_significance(
                d, margin=margin, alpha=alpha, tost_conf=tost_conf,
                expected_sign=source.get('expected_sign'),
                seeds=seeds, ppl_a=pa, ppl_b=pb,
                policy_version=source.get('version'),
                relative_pct=source.get('relative_pct'), scale=scale)
        except Exception as e:
            out['problems'].append(
                f'{name}: replay raised {type(e).__name__}: {e}')
            continue
        out['problems'].extend(_compare(name, rep, inner if isinstance(inner, dict) else stored))
        for _k in ('status', 'p_tost', 'p_beyond', 'equivalent', 'beyond_sesoi'):
            if _k == 'status' or _k in ('equivalent', 'beyond_sesoi'):
                if stored.get(_k) != (inner if isinstance(inner, dict) else stored).get(_k):
                    out['problems'].append(
                        f'{name}: aggregate {_k}={stored.get(_k)!r} != per-comparison '
                        f'{(inner if isinstance(inner, dict) else stored).get(_k)!r}')
            elif not _close(stored.get(_k), (inner if isinstance(inner, dict) else stored).get(_k)):
                out['problems'].append(
                    f'{name}: aggregate {_k}={stored.get(_k)!r} != per-comparison '
                    f'{(inner if isinstance(inner, dict) else stored).get(_k)!r}')
        out['comparisons'][name] = {
            'status': rep.get('status'),
            'p_tost': rep.get('p_tost'),
            'p_beyond': rep.get('p_beyond'),
            'equivalent': rep.get('equivalent'),
            'beyond_sesoi': rep.get('beyond_sesoi'),
        }
        out['n_replayed'] += 1
    # Rebuild both Holm families from the stored rows and compare to the
    # recorded aggregate.  A tampered p without a matching family is caught
    # here even when the per-row vector itself replayed.
    try:
        fam = L.confirmation_practical_family(
            comps, confirmed=ft.get('confirmed') or [],
            policy=block.get('policy'), alpha=block.get('alpha'),
            family_names=block.get('family_names'))
        srow = block.get('summary') if isinstance(block.get('summary'), dict) else {}
        frow = fam.get('summary') or {}
        for key in ('practically_meaningful', 'equivalent_within_sesoi',
                    'direction_conflict', 'indeterminate', 'not_declared',
                    'not_estimable', 'frozen_confirmed_but_not_beyond_sesoi',
                    'frozen_confirmed_nondirectional'):
            if sorted(srow.get(key) or []) != sorted(frow.get(key) or []):
                out['problems'].append(
                    f'summary.{key}={srow.get(key)!r} != replayed {frow.get(key)!r}')
        stored_holm = block.get('holm') if isinstance(block.get('holm'), dict) else {}
        replay_holm = fam.get('holm') if isinstance(fam.get('holm'), dict) else {}
        for fam_name in ('beyond_sesoi', 'tost_equivalence'):
            s = stored_holm.get(fam_name) if isinstance(stored_holm.get(fam_name), dict) else {}
            r = replay_holm.get(fam_name) if isinstance(replay_holm.get(fam_name), dict) else {}
            if sorted(s) != sorted(r):
                out['problems'].append(
                    f'holm.{fam_name} names {sorted(s)} != replayed {sorted(r)}')
            for nm in set(s) & set(r):
                if not _close((s[nm] or {}).get('p_holm'), (r[nm] or {}).get('p_holm')):
                    out['problems'].append(
                        f'holm.{fam_name}[{nm}].p_holm={(s[nm] or {}).get("p_holm")!r} '
                        f'!= replayed {(r[nm] or {}).get("p_holm")!r}')
        out['holm'] = replay_holm
    except Exception as e:
        out['problems'].append(
            f'family replay raised {type(e).__name__}: {e}')
    if out['n_replayed'] == 0 and not out['problems']:
        out['status'] = 'not_applicable'
        out['reason'] = 'no comparison carried a replayable SESOI block'
    elif out['problems']:
        out['status'] = 'fail'
    else:
        out['status'] = 'ok'
        _sm = (block.get('summary') or {}) if isinstance(block.get('summary'), dict) else {}
        for _n in (_sm.get('frozen_confirmed_but_not_beyond_sesoi') or []):
            out['warnings'].append(
                f'{_n}: frozen confirmed but not beyond the declared SESOI; '
                'report the practical caveat next to the verdict')
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out
