def _lpa_close(got, want, rel=1e-9, abs_=1e-12):
    if got is None or want is None:
        return got is None and want is None
    try:
        a, b = float(got), float(want)
    except (TypeError, ValueError, OverflowError):
        return False
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return abs(a - b) <= max(abs_, rel * max(1.0, abs(a), abs(b)))


def _lpa_same(got, want, path, problems):
    """Strict structural comparison with a numeric tolerance."""
    if isinstance(want, dict):
        if not isinstance(got, dict):
            problems.append(f'{path}: stored value is not an object')
            return
        missing = sorted(set(want) - set(got))
        extra = sorted(set(got) - set(want))
        if missing:
            problems.append(f'{path}: missing keys {missing}')
        if extra:
            problems.append(f'{path}: unexpected keys {extra}')
        for key in sorted(set(want) & set(got), key=str):
            _lpa_same(got[key], want[key], f'{path}.{key}', problems)
        return
    if isinstance(want, list):
        if not isinstance(got, list) or len(got) != len(want):
            problems.append(f'{path}: list length/shape differs')
            return
        for i, (g, w) in enumerate(zip(got, want)):
            _lpa_same(g, w, f'{path}[{i}]', problems)
        return
    if isinstance(want, bool) or isinstance(got, bool):
        if got is not want:
            problems.append(f'{path}: {got!r} != replayed {want!r}')
        return
    if isinstance(want, int) and isinstance(got, int):
        if int(got) != int(want):
            problems.append(f'{path}: {got!r} != replayed {want!r}')
        return
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        if not _lpa_close(got, want):
            problems.append(f'{path}: {got!r} != replayed {want!r}')
        return
    if got != want:
        problems.append(f'{path}: {got!r} != replayed {want!r}')


def audit_length_position(path=None):
    """Replay the position-band decomposition of the by-length panel.

    ``finaltest_stats`` stores two things that make this a real gate rather
    than a prose check: per-seed ``nll_by_seed``/``n_tokens_by_seed`` on every
    by-length cell, and the derived band decomposition.  The gate rebuilds the
    latter from the former and compares every derived number.  It also checks
    the cells' own PPL identity (``ppl == exp(nll_sum/n_tok)``), so a hand edit
    that moves a band's claim without moving the cell is caught.

    A missing block is a legacy artifact (``not_verifiable_locally``), never a
    pass.  The audit is read-only and never changes a p-value, the Holm family
    or a frozen verdict.
    """
    L = _import_exp_lib()
    p = path or os.path.join(REPO, 'analysis_v12', 'test_stats.json')
    out = {'kind': 'length_position_audit', 'status': 'ok', 'path': p,
           'n_pairs': 0, 'n_replayed': 0, 'problems': [], 'warnings': [],
           'note': ('rebuilds the position-band decomposition from the stored '
                    'per-seed by-length NLL/token totals; a missing block is '
                    'not a pass and a disagreement is a fail; no frozen '
                    'p/Holm/verdict is changed')}
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
    lp = ft.get('length_position') if isinstance(ft, dict) else None
    bl = ft.get('by_length') if isinstance(ft, dict) else None
    if lp is None:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = ('no length_position block (legacy artifact); the '
                         'position decomposition is not replayable')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    if not isinstance(lp, dict):
        out['status'] = 'fail'
        out['problems'].append('length_position is not an object')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    if lp.get('kind') != 'length_position_decomposition':
        out['problems'].append(
            f"length_position.kind is {lp.get('kind')!r}, not "
            "'length_position_decomposition'")
    if lp.get('status') not in L.LENGTH_POSITION_STATUSES:
        out['problems'].append(
            f"length_position.status {lp.get('status')!r} is not a declared status")
    _confirm_fields = ('confirmed', 'p_holm', 'verified', 'verdict', 'decision')
    if any(f in lp for f in _confirm_fields):
        out['problems'].append(
            'length_position carries confirmatory field(s) '
            f'{sorted(set(lp) & set(_confirm_fields))}')
    if not isinstance(bl, dict) or not bl:
        out['status'] = 'fail'
        out['problems'].append(
            'length_position exists but by_length cells are missing; the '
            'decomposition cannot be rebuilt from its inputs')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    # Rebuild the exact input accepted by the pure function.
    replay_in = {}
    for arm, lengths in bl.items():
        if not isinstance(lengths, dict):
            out['problems'].append(f'{arm}: by_length is not an object')
            continue
        arm_cells = {}
        for Ln, cell in lengths.items():
            if not isinstance(cell, dict):
                out['problems'].append(f'{arm}@{Ln}: cell is not an object')
                continue
            ppl = cell.get('ppl_by_seed')
            nll = cell.get('nll_by_seed')
            ntok = cell.get('n_tokens_by_seed')
            if not isinstance(ppl, dict) or not isinstance(nll, dict) or not isinstance(ntok, dict):
                out['warnings'].append(
                    f'{arm}@{Ln}: no per-seed NLL/token accounting; this cell '
                    'is not replayable')
                continue
            if set(ppl) != set(nll) or set(ppl) != set(ntok):
                out['problems'].append(
                    f'{arm}@{Ln}: ppl/nll/n_tokens seed sets differ')
                continue
            for s in ppl:
                try:
                    pv, nv, tv = float(ppl[s]), float(nll[s]), int(ntok[s])
                except (TypeError, ValueError, OverflowError):
                    out['problems'].append(
                        f'{arm}@{Ln} seed {s}: non-numeric accounting')
                    continue
                if (not math.isfinite(pv) or pv <= 0.0
                        or not math.isfinite(nv) or nv < 0.0 or tv <= 0):
                    out['problems'].append(
                        f'{arm}@{Ln} seed {s}: invalid ppl/nll/n_tokens')
                    continue
                try:
                    tv_f = float(tv)
                except OverflowError:
                    out['problems'].append(
                        f'{arm}@{Ln} seed {s}: n_tokens is outside the '
                        'representable float range')
                    continue
                rate = nv / tv_f
                if not math.isfinite(rate) or abs(math.log(pv) - rate) > 1e-9 * max(1.0, abs(rate)):
                    out['problems'].append(
                        f'{arm}@{Ln} seed {s}: ppl is not exp(nll_sum/n_tok)')
            arm_cells[str(Ln)] = {'ppl_by_seed': ppl, 'nll_by_seed': nll,
                                  'n_tokens_by_seed': ntok}
        if arm_cells:
            replay_in[str(arm)] = arm_cells
    if out['problems']:
        out['status'] = 'fail'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    if not replay_in:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = ('no by-length cell carries per-seed NLL/token '
                         'accounting; the band claim is not replayable')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    pairs = lp.get('pairs')
    if not isinstance(pairs, dict) or not pairs:
        if lp.get('status') == 'not_estimable':
            out['status'] = 'not_applicable'
            out['warnings'].append(
                'length_position is a bare not_estimable block with no pair '
                'readouts; nothing to replay')
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return out
        out['problems'].append('length_position.pairs is missing or empty')
        out['status'] = 'fail'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    out['n_pairs'] = len(pairs)
    decls = []
    for name, entry in pairs.items():
        if not isinstance(entry, dict):
            out['problems'].append(f'pair {name!r} is not an object')
            continue
        a, b = entry.get('a'), entry.get('b')
        if not isinstance(a, str) or not isinstance(b, str):
            out['problems'].append(f'pair {name!r} does not name two arms')
            continue
        decls.append({'name': str(name), 'a': a, 'b': b})
    if out['problems']:
        out['status'] = 'fail'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    try:
        replay = L.length_position_decomposition(
            replay_in, pairs=decls, alpha=lp.get('alpha'),
            min_seeds=(lp.get('policy') or {}).get('min_seeds'))
    except Exception as exc:
        out['status'] = 'fail'
        out['problems'].append(f'replay raised {type(exc).__name__}: {exc}')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    if replay.get('status') != lp.get('status'):
        out['problems'].append(
            f"status={lp.get('status')!r} does not replay "
            f"(got {replay.get('status')!r})")
    for key in ('alpha', 'lengths', 'arms', 'pairs', 'policy'):
        _lpa_same(lp.get(key), replay.get(key), key, out['problems'])
    if out['problems']:
        out['status'] = 'fail'
    elif lp.get('status') == 'not_estimable':
        # The decomposition ran but no pair had the required coverage; there
        # is no estimate to verify.  Calling that "ok" would turn a missing
        # estimate into a passed gate.
        out['status'] = 'not_applicable'
        out['reason'] = ('the stored decomposition is not_estimable; the '
                         'block structure matches its inputs but carries no '
                         'pair estimate to verify')
        out['warnings'].append(str(lp.get('reason') or 'not_estimable'))
    else:
        out['n_replayed'] = len(pairs)
        if out['n_replayed'] == 0:
            out['status'] = 'not_verifiable_locally'
            out['reason'] = 'no pair block could be replayed'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out
