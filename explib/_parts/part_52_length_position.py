"""Position-band decomposition of the by-length PPL panel.

The frozen by-length endpoint evaluates the *same* fixed rows at several
sequence lengths (512/1024/2048/4096).  A causal LM predicts token ``t`` from
tokens ``< t`` only, so the NLL of the first ``L0`` tokens is identical in a
length-``L0`` cell and in a length-``L1 > L0`` cell of the same row (the
dynamic segmenter is prefix-invariant by contract, and the batched evaluator
computes every shorter length from the same cumulative row).  The difference
between the two cell totals is therefore the NLL of the newly covered target
positions, not a different sample.

That identity turns the by-length table into an exact position decomposition:
sort the evaluated lengths ``L0 < L1 < ...`` and define band ``j`` as the
target positions ``L_{j-1} .. L_j - 1``.  Band NLL = ``nll_sum(L_j) -
nll_sum(L_{j-1})`` and band tokens = ``n_tok(L_j) - n_tok(L_{j-1})``.  Comparing
two arms *inside a band* holds the position range fixed, so an apparent
"advantage that grows with length" is separated from the changing mixture of
early and late positions in the aggregate PPL.

This module is descriptive: it adds no p-value to the frozen Holm family and
never changes a confirmation verdict.  The exact sign-flip readings on band
deltas are reported with the same seed-level assumption as every other
paired readout in the repository.
"""

import math

import numpy as np

LENGTH_POSITION_POLICY = {
    'version': 'length-position-bands-1',
    'role': ('descriptive position-matched decomposition of the frozen '
             'by-length cells; not a confirmatory test'),
    'min_seeds': 2,
    'min_lengths': 2,
    'alpha': 0.05,
    'monotone_rtol': 1e-9,
    'band_rule': ('band j covers target positions L_{j-1} .. L_j-1; the total '
                  'at length L is the causal prefix total, so the difference '
                  'of two length totals is the newly covered band NLL'),
    'identity': ('causal LM: a position only reads its prefix, and the '
                 'segmenter is prefix-invariant, so shorter-length cells are '
                 'prefixes of the longer cell on the same row'),
    'assumption': ('exact sign-flip on band deltas keeps the repository '
                   'seed-level exchangeability assumption; band positions are '
                   'not independent samples and are never treated as such'),
}
LENGTH_POSITION_STATUSES = ('ok', 'not_estimable')


def _lpd_int(value, name, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer, got {value!r}')
    out = int(value)
    if minimum is not None and out < minimum:
        raise ValueError(f'{name} must be >= {minimum}, got {out!r}')
    if maximum is not None and out > maximum:
        raise ValueError(f'{name} must be <= {maximum}, got {out!r}')
    return out


def _lpd_key_int(value, name, *, minimum=None, maximum=None):
    """Integer key from JSON: ``"512"`` and ``512`` are the same length/seed."""
    if isinstance(value, str):
        if not value.isdigit():
            raise ValueError(f'{name} must be an integer key, got {value!r}')
        value = int(value)
    return _lpd_int(value, name, minimum=minimum, maximum=maximum)


def _lpd_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name} must be a number, got {value!r}')
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f'{name} must be finite, got {value!r}')
    return out


def _lpd_seed_map(value, name):
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, dict) or not value:
        raise ValueError(f'{name} must be a non-empty seed -> value mapping')
    out = {}
    for raw, val in value.items():
        seed = _lpd_key_int(raw, f'{name} key', minimum=0,
                            maximum=int(MAX_SEED))
        if seed in out:
            raise ValueError(f'{name} contains seed {seed} more than once '
                             '(JSON "0" and 0 are the same seed)')
        out[seed] = val
    return out


def _lpd_parse_cell(cell, label):
    if not isinstance(cell, dict):
        raise ValueError(f'{label} must be an object')
    ppl = _lpd_seed_map(cell.get('ppl_by_seed'), f'{label}.ppl_by_seed')
    nll = _lpd_seed_map(cell.get('nll_by_seed'), f'{label}.nll_by_seed')
    ntok = _lpd_seed_map(cell.get('n_tokens_by_seed'),
                         f'{label}.n_tokens_by_seed')
    if set(ppl) != set(nll) or set(ppl) != set(ntok):
        raise ValueError(f'{label}: ppl/nll/n_tokens seed sets differ')
    rows = {}
    for seed in sorted(ppl):
        p = _lpd_number(ppl[seed], f'{label}.ppl_by_seed[{seed}]')
        nll_s = _lpd_number(nll[seed], f'{label}.nll_by_seed[{seed}]')
        n_tok = _lpd_int(ntok[seed], f'{label}.n_tokens_by_seed[{seed}]',
                         minimum=1)
        if p <= 0.0:
            raise ValueError(f'{label}.ppl_by_seed[{seed}] must be > 0')
        if nll_s < 0.0:
            raise ValueError(f'{label}.nll_by_seed[{seed}] must be >= 0')
        implied = nll_s / float(n_tok)
        # The record's PPL is exp(nll_sum/n_tok); a cell whose own three
        # numbers do not satisfy that identity is not a measurement we can
        # decompose (it may be a different metric or a hand-edited row).
        tol = 1e-9 * max(1.0, abs(implied))
        if not math.isfinite(implied) or abs(math.log(p) - implied) > tol:
            raise ValueError(
                f'{label} seed {seed}: ppl={p!r} is not exp(nll_sum/n_tok) '
                f'for nll_sum={nll_s!r}, n_tok={n_tok}')
        rows[seed] = {'ppl': p, 'nll': nll_s, 'n_tokens': n_tok}
    return rows


def _lpd_parse(by_length):
    if not isinstance(by_length, dict) or not by_length:
        raise ValueError('by_length must be a non-empty arm -> length mapping')
    panels = {}
    for arm, lengths in by_length.items():
        if not isinstance(arm, str) or not arm:
            raise ValueError(f'arm name must be a non-empty string, got {arm!r}')
        if not isinstance(lengths, dict) or not lengths:
            raise ValueError(f'arm {arm!r}: lengths must be a non-empty mapping')
        cells = {}
        for raw_len, cell in lengths.items():
            Ln = _lpd_key_int(raw_len, f'{arm} length', minimum=2)
            if Ln in cells:
                raise ValueError(f'arm {arm!r} repeats length {Ln}')
            cells[Ln] = _lpd_parse_cell(cell, f'{arm}@{Ln}')
        panels[arm] = cells
    return panels


def _lpd_stats(values, *, alpha, min_seeds, seeds):
    """Small deterministic paired readout; no bootstrap is needed here."""
    if len(seeds) != len(values):
        raise ValueError(
            f'paired readout is misaligned: {len(values)} values but '
            f'{len(seeds)} seed labels')
    d = np.asarray([float(x) for x in values], dtype=float)
    n = int(d.size)
    if n == 0:
        return {'n': 0, 'mean': None, 'std': None, 'ci95': None,
                'p_exact_signflip': None, 'p_floor': None, 'direction': None,
                'significant': False, 'status': 'not_estimable'}
    sf = exact_signflip_p(d)
    _ms = _scaled_mean_std_if_extreme(d, ddof=1) if n > 1 else None
    if _ms is not None:
        mean, sd = _ms
    else:
        mean = float(d.mean())
        sd = float(d.std(ddof=1)) if n > 1 else None
    ci = None
    if n > 1 and sd is not None and math.isfinite(sd) and sd > 0.0:
        sem = sd / math.sqrt(n)
        tc = _t_critical(0.95, n - 1)
        if math.isfinite(tc) and math.isfinite(sem):
            ci = [mean - tc * sem, mean + tc * sem]
    direction = None if mean == 0.0 else ('negative' if mean < 0.0 else 'positive')
    p = sf.get('p')
    p_floor = sf.get('p_floor')
    within_noise = (sd is not None and abs(mean) <= sd)
    significant = bool(
        p is not None and math.isfinite(p) and p < alpha
        and n >= min_seeds and not within_noise
        and p_floor is not None and p_floor <= alpha)
    return {'n': n, 'mean': mean, 'std': sd, 'ci95': ci,
            'p_exact_signflip': p, 'p_floor': p_floor,
            'direction': direction, 'within_seed_noise': within_noise,
            'significant': significant, 'status': 'ok',
            'seeds': [int(s) for s in seeds]}


def _lpd_json_safe(value):
    if isinstance(value, dict):
        return {str(k): _lpd_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_lpd_json_safe(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return v if math.isfinite(v) else None
    return value


def _lpd_max_band_signflip(band_rows, *, alpha):
    """Exact score max-T across bands, with no zero-variance rejection.

    ``maxt_signflip_adjust`` refuses a contrast whose seed SD is zero even
    when its mean is non-zero.  That guard is right for its t-style interval,
    but the *score* statistic ``|sum d| / sqrt(sum d^2)`` is finite for a
    constant non-zero band.  This fallback uses that sign-invariant statistic
    for the family scan, so a perfectly consistent band is not silently
    removed from the multiplicity readout.  It is descriptive and only used
    when the primary max-T helper refuses the family.
    """
    if len(band_rows) < 2:
        return {'status': 'not_estimable',
                'reason': 'fewer than two bands',
                'n_bands': len(band_rows)}
    labels = [row['label'] for row in band_rows]
    seed_sets = [set(item['seed'] for item in row['per_seed'])
                 for row in band_rows]
    seeds = sorted(set.intersection(*seed_sets))
    n = len(seeds)
    if n < 2:
        return {'status': 'not_estimable',
                'reason': 'fewer than two common seeds across bands',
                'n_bands': len(band_rows)}
    if n > 16:
        return {'status': 'not_estimable',
                'reason': f'n={n} exceeds the exact fallback enumeration cap 16',
                'n_bands': len(band_rows)}
    D = np.asarray([[float(next(item['log_ratio'] for item in row['per_seed']
                              if item['seed'] == s)) for s in seeds]
                    for row in band_rows], dtype=float)
    scale = np.sqrt((D * D).sum(axis=1))
    obs = np.zeros(len(labels), dtype=float)
    nz = scale > 0.0
    obs[nz] = np.abs(D[nz].sum(axis=1)) / scale[nz]
    total = 1 << n
    bitpos = np.arange(n, dtype=np.int64)[None, :]
    step = max(1, min(total, (1 << 20) // max(len(labels), 1)))
    max_null = []
    for off in range(0, total, step):
        bs = min(step, total - off)
        masks = np.arange(off, off + bs, dtype=np.int64)[:, None]
        signs = np.where((masks & (1 << bitpos)) != 0, 1.0, -1.0)
        T = np.abs(signs @ D.T)
        T[:, nz] = T[:, nz] / scale[nz]
        max_null.append(T.max(axis=1))
    max_null = np.concatenate(max_null)
    tol = 1e-12 * max(1.0, float(np.max(obs)) if obs.size else 1.0)
    p_single = [float((max_null >= float(o) - tol).mean()) for o in obs]
    p_family = float((max_null >= float(obs.max()) - tol).mean())
    return {
        'status': 'ok',
        'method': ('exact sign-flip score max-T across position bands '
                   '(constant-band fallback)'),
        'n_bands': len(labels),
        'p_family': p_family,
        'critical_value': float(np.quantile(max_null, 1.0 - alpha)),
        'ci_tail_level': None,
        'contrasts': {
            label: {'p_maxt': float(p_single[i]),
                    'p_maxt_single': float(p_single[i]),
                    'rank': None,
                    'reject': bool(p_single[i] < alpha)}
            for i, label in enumerate(labels)},
        'note': ('descriptive family scan; the constant-band fallback uses the '
                 'sign-invariant score statistic and does not enter the frozen '
                 'Holm family'),
    }


def _lpd_seed_intersection(*maps):
    sets = [set(m) for m in maps]
    return sorted(set.intersection(*sets)) if sets else []


def _lpd_band_row(panel, lower, upper, seed):
    lo = panel[lower][seed]
    hi = panel[upper][seed]
    n_tok = int(hi['n_tokens']) - int(lo['n_tokens'])
    if n_tok <= 0:
        raise ValueError(
            f'length cells must add tokens: {lower}->{upper} gives n_tok '
            f'{lo["n_tokens"]}->{hi["n_tokens"]}')
    nll = float(hi['nll']) - float(lo['nll'])
    scale = max(1.0, abs(float(hi['nll'])), abs(float(lo['nll'])))
    if nll < -float(LENGTH_POSITION_POLICY['monotone_rtol']) * scale:
        raise ValueError(
            f'non-monotone NLL total at {lower}->{upper}: difference {nll!r}; '
            'a causal prefix cannot lose NLL by adding later positions')
    nll = max(0.0, nll)
    rate = nll / float(n_tok)
    return {'seed': int(seed), 'nll': nll, 'n_tokens': n_tok,
            'rate': rate, 'ppl': _safe_exp_ppl(rate)}


def _lpd_arm(panel):
    lengths = sorted(panel)
    # One full-coverage seed set per arm: a band curve that silently changes
    # its seed composition between consecutive bands is not a curve.
    arm_seeds = _lpd_seed_intersection(*[panel[Ln] for Ln in lengths])
    bands = []
    for lower, upper in zip(lengths, lengths[1:]):
        seeds = list(arm_seeds)
        rows = [_lpd_band_row(panel, lower, upper, s) for s in seeds]
        if not rows:
            continue
        ppls = [r['ppl'] for r in rows]
        rates = [r['rate'] for r in rows]
        bands.append({
            'lower_length': int(lower), 'upper_length': int(upper),
            'position_lo': int(lower), 'position_hi': int(upper - 1),
            'label': f'{lower}-{upper - 1}',
            'n': len(rows),
            'seeds': seeds,
            'n_tokens_per_seed': sorted({int(r['n_tokens']) for r in rows}),
            'mean_band_ppl': float(np.mean(ppls)),
            'mean_band_nll_per_token': float(np.mean(rates)),
            'per_seed': [{'seed': r['seed'], 'band_ppl': r['ppl'],
                          'band_nll': r['nll'],
                          'n_tokens': r['n_tokens']} for r in rows],
        })
    return {'lengths': lengths, 'bands': bands}


def length_position_decomposition(by_length, *, pairs=None, alpha=None,
                                  min_seeds=None):
    """Decompose by-length PPL cells into position bands.

    ``by_length[arm][length]`` must carry ``ppl_by_seed``,
    ``nll_by_seed`` and ``n_tokens_by_seed`` (the fields
    ``finaltest_stats`` now records from the frozen by-length cells).  ``pairs``
    is an optional iterable of ``{name, a, b}`` declarations; when omitted all
    unordered arm pairs are described.

    The return object contains the per-arm band PPL curve plus per-pair
    band contrasts and their exact decomposition of the total log PPL ratio.
    Malformed inputs raise ``ValueError`` (they are not a research result);
    insufficient but well-formed input returns ``status='not_estimable'``.
    """
    alpha = _validated_alpha(
        LENGTH_POSITION_POLICY['alpha'] if alpha is None else alpha)
    min_seeds = _lpd_int(
        LENGTH_POSITION_POLICY['min_seeds'] if min_seeds is None else min_seeds,
        'min_seeds', minimum=2)
    panels = _lpd_parse(by_length)
    arms = sorted(panels)
    lengths = sorted({Ln for c in panels.values() for Ln in c})
    out = {
        'kind': 'length_position_decomposition',
        'version': LENGTH_POSITION_POLICY['version'],
        'status': 'not_estimable',
        'role': LENGTH_POSITION_POLICY['role'],
        'alpha': float(alpha),
        'lengths': lengths,
        'policy': {
            'min_seeds': int(min_seeds),
            'band_rule': LENGTH_POSITION_POLICY['band_rule'],
            'identity': LENGTH_POSITION_POLICY['identity'],
            'assumption': LENGTH_POSITION_POLICY['assumption'],
        },
        'arms': {},
        'pairs': {},
        'warnings': [],
        'boundary': (
            'Descriptive position decomposition of the same by-length cells; '
            'it creates no new p-value in the frozen Holm family, does not '
            'change p_exact_signflip/p_holm/confirmed, and does not make the '
            'positions independent evaluation units.  It separates a length '
            'effect caused by newly covered positions from one already present '
            'on the common prefix, but it cannot repair a non-causal design or '
            'an external-validity limit.'),
    }
    if len(arms) < 2:
        out['reason'] = 'fewer than two arms have usable by-length cells'
        return _lpd_json_safe(out)
    arm_ok = 0
    for arm in arms:
        desc = _lpd_arm(panels[arm])
        out['arms'][arm] = desc
        if len(desc['bands']) > 0:
            arm_ok += 1
    if arm_ok < 2:
        out['reason'] = 'fewer than two arms have at least two consecutive lengths'
        return _lpd_json_safe(out)
    if pairs is None:
        pairs = [{'name': f'{a} vs {b}', 'a': a, 'b': b}
                 for i, a in enumerate(arms) for b in arms[i + 1:]]
    if isinstance(pairs, (str, bytes, bytearray)) or not isinstance(pairs, (list, tuple)):
        raise ValueError('pairs must be a sequence of {name, a, b} objects')
    seen_names = set()
    pair_out = {}
    for entry in pairs:
        if not isinstance(entry, dict):
            raise ValueError(f'pair declaration must be an object, got {entry!r}')
        name = entry.get('name')
        a = entry.get('a')
        b = entry.get('b')
        if not isinstance(name, str) or not name:
            raise ValueError(f'pair name must be a non-empty string, got {name!r}')
        if name in seen_names:
            raise ValueError(f'pair name {name!r} is declared more than once')
        seen_names.add(name)
        if a not in panels or b not in panels:
            raise ValueError(f'pair {name!r} names an arm that has no by-length cells')
        if a == b:
            raise ValueError(f'pair {name!r} compares an arm with itself')
        common_lengths = sorted(set(panels[a]) & set(panels[b]))
        if len(common_lengths) < 2:
            pair_out[name] = {'a': a, 'b': b, 'status': 'not_estimable',
                              'reason': 'fewer than two common lengths'}
            continue
        # One seed set for the whole pair: every band uses the seeds that
        # have ALL common lengths on BOTH arms.  A seed missing from one
        # length cannot contribute to a cumulative decomposition, and using
        # different seed sets per band would make the contributions silently
        # incomparable.
        pair_seeds = _lpd_seed_intersection(
            *[panels[a][Ln] for Ln in common_lengths],
            *[panels[b][Ln] for Ln in common_lengths])
        if len(pair_seeds) < min_seeds:
            pair_out[name] = {
                'a': a, 'b': b, 'status': 'not_estimable',
                'reason': (f'only {len(pair_seeds)} seed(s) cover every common '
                           f'length on both arms; min_seeds={min_seeds}'),
                'seeds': pair_seeds,
            }
            continue
        band_rows = []
        errors = []
        for lower, upper in zip(common_lengths, common_lengths[1:]):
            seeds = list(pair_seeds)
            d_ppl, d_log, contrib, band_nlls = [], [], [], []
            band_tokens = []
            max_len = int(common_lengths[-1])
            for seed in seeds:
                ra = _lpd_band_row(panels[a], lower, upper, seed)
                rb = _lpd_band_row(panels[b], lower, upper, seed)
                if ra['n_tokens'] != rb['n_tokens']:
                    errors.append(
                        f'arms {a}/{b} have different band token counts at '
                        f'{lower}->{upper} (seed {seed})')
                    continue
                da = ra['ppl'] - rb['ppl']
                dl = ra['rate'] - rb['rate']
                d_ppl.append(da)
                d_log.append(dl)
                band_tokens.append(ra['n_tokens'])
                band_nlls.append(ra['nll'] - rb['nll'])
                # Every band uses the SAME denominator (the maximum-length
                # token count), so the contributions sum exactly to the total
                # log PPL ratio at that length.
                total_tok = int(panels[a][max_len][seed]['n_tokens'])
                contrib.append((ra['nll'] - rb['nll']) / float(total_tok))
            if not d_ppl:
                continue
            # If a token mismatch skipped a seed, the seed list must follow.
            seed_list = []
            for seed in seeds:
                if len(seed_list) >= len(d_ppl):
                    break
                ra = _lpd_band_row(panels[a], lower, upper, seed)
                rb = _lpd_band_row(panels[b], lower, upper, seed)
                if ra['n_tokens'] == rb['n_tokens']:
                    seed_list.append(seed)
            stats_ppl = _lpd_stats(d_ppl, alpha=alpha,
                                   min_seeds=min_seeds, seeds=seed_list)
            stats_log = _lpd_stats(d_log, alpha=alpha,
                                   min_seeds=min_seeds, seeds=seed_list)
            band_rows.append({
                'lower_length': int(lower), 'upper_length': int(upper),
                'position_lo': int(lower), 'position_hi': int(upper - 1),
                'label': f'{lower}-{upper - 1}',
                'n': len(seed_list), 'seeds': seed_list,
                'n_tokens_per_seed': sorted(set(band_tokens)),
                'band_ppl_delta': stats_ppl,
                'band_log_ratio': stats_log,
                'mean_band_nll_delta': float(np.mean(band_nlls)),
                'mean_contribution_to_total_log_ratio_at_max': float(np.mean(contrib)),
                'per_seed': [{'seed': s, 'ppl_delta': dp,
                              'log_ratio': dl, 'band_nll_delta': bn,
                              'contribution_at_max': c}
                             for s, dp, dl, bn, c in zip(
                                 seed_list, d_ppl, d_log, band_nlls, contrib)],
            })
        if not band_rows:
            pair_out[name] = {'a': a, 'b': b, 'status': 'not_estimable',
                              'reason': 'no band has a common seed set',
                              'problems': errors}
            continue
        # Is the arm difference larger on later positions than on the first
        # (common-prefix) band?  A paired difference-in-differences on the
        # log-PPL ratio; the first band is the reference.  This is the direct
        # test of "the advantage only appears after the training length".
        first = band_rows[0]
        first_map = {item['seed']: float(item['log_ratio'])
                     for item in first['per_seed']}
        for row in band_rows[1:]:
            common = sorted(set(first_map) &
                            {item['seed'] for item in row['per_seed']})
            row_map = {item['seed']: float(item['log_ratio'])
                       for item in row['per_seed']}
            diffs = [row_map[s] - first_map[s] for s in common]
            row['interaction_vs_first_band'] = _lpd_stats(
                diffs, alpha=alpha, min_seeds=min_seeds, seeds=common)
            row['interaction_reference'] = first['label']
        # Check that the bands reconstruct the cumulative log ratio at every
        # common length.  This is the causal-prefix identity used as a
        # machine-checked invariant, not an assumption hidden in prose.
        reconstruction = {}
        for Ln in common_lengths:
            # The same full-coverage seed set as the bands: a cumulative
            # decomposition cannot mix seeds that are present at one length
            # but absent at another.
            seeds = []
            vals = []
            for seed in pair_seeds:
                ca = panels[a][Ln][seed]
                cb = panels[b][Ln][seed]
                if int(ca['n_tokens']) != int(cb['n_tokens']):
                    errors.append(
                        f'arms {a}/{b} have different n_tokens at L={Ln} '
                        f'(seed {seed})')
                    continue
                seeds.append(seed)
                vals.append(ca['nll'] / float(ca['n_tokens'])
                            - cb['nll'] / float(cb['n_tokens']))
            reconstruction[Ln] = _lpd_stats(
                vals, alpha=alpha, min_seeds=min_seeds, seeds=seeds)
        # Sum the band contributions up to each length.
        max_err = 0.0
        for Ln in common_lengths:
            for seed in reconstruction[Ln].get('seeds') or []:
                band_sum = 0.0
                for row in band_rows:
                    if int(row['upper_length']) <= int(Ln):
                        for item in row['per_seed']:
                            if item['seed'] == seed:
                                band_sum += float(item['band_nll_delta'])
                                break
                ca = panels[a][Ln][seed]
                cb = panels[b][Ln][seed]
                total = ca['nll'] / float(ca['n_tokens']) - cb['nll'] / float(cb['n_tokens'])
                n_tok = int(ca['n_tokens'])
                if n_tok <= 0:
                    errors.append(f'non-positive token count at L={Ln}')
                    continue
                max_err = max(max_err, abs(band_sum / float(n_tok) - total))
        if max_err > 1e-8:
            errors.append(
                f'band contributions do not reconstruct the total log ratio '
                f'(max error {max_err:.3g})')
        max_len = common_lengths[-1]
        contribs = [(row['label'],
                     abs(float(row['mean_contribution_to_total_log_ratio_at_max'])))
                    for row in band_rows if int(row['upper_length']) <= int(max_len)]
        dominant = max(contribs, key=lambda x: x[1])[0] if contribs else None
        # Family-level scan readout: selecting the most extreme band and
        # quoting its single-band p is a forking path.  The exact sign-flip
        # max-T over the band vector asks "does ANY band differ" while keeping
        # the cross-band correlation; it is still a descriptive readout and
        # does not enter the frozen Holm family.
        if len(band_rows) >= 2:
            try:
                _bmax = maxt_signflip_adjust(
                    {row['label']: [float(item['log_ratio'])
                                    for item in row['per_seed']]
                     for row in band_rows}, alpha=alpha)
                _bmax = {
                    'status': 'ok',
                    'method': 'exact sign-flip max-T across position bands',
                    'n_bands': len(band_rows),
                    'p_family': min(float(c['p_maxt'])
                                    for c in _bmax['contrasts'].values()),
                    'critical_value': _bmax.get('critical_value'),
                    'ci_tail_level': _bmax.get('ci_tail_level'),
                    'contrasts': {
                        k: {'p_maxt': v.get('p_maxt'),
                            'p_maxt_single': v.get('p_maxt_single'),
                            'rank': v.get('rank'),
                            'reject': v.get('reject')}
                        for k, v in _bmax['contrasts'].items()},
                    'note': ('family-wise scan readout over the declared bands; '
                             'descriptive, not part of the frozen Holm family'),
                }
            except Exception as _bexc:
                _bmax = _lpd_max_band_signflip(band_rows, alpha=alpha)
                if _bmax.get('status') != 'ok':
                    _bmax = {'status': 'not_estimable',
                             'reason': (f'{type(_bexc).__name__}: {_bexc}; '
                                        f'fallback: {_bmax.get("reason")}'),
                             'n_bands': len(band_rows)}
                else:
                    _bmax['primary_error'] = f'{type(_bexc).__name__}: {_bexc}'
        else:
            _bmax = {'status': 'not_estimable',
                     'reason': 'fewer than two bands'}
        pair_out[name] = {
            'a': a, 'b': b,
            'status': 'ok' if not errors else 'not_estimable',
            'reason': '; '.join(errors) if errors else None,
            'lengths': [int(x) for x in common_lengths],
            'seeds': list(pair_seeds),
            'n': len(pair_seeds),
            'bands': {row['label']: row for row in band_rows},
            'band_family_max_t': _bmax,
            'total_by_length': {str(Ln): reconstruction[Ln] for Ln in common_lengths},
            'dominant_band_at_max_length': dominant,
            'max_reconstruction_error': float(max_err),
            'decomposition': (
                'log(PPL_a/PPL_b) at length L equals the token-weighted sum of '
                'the band log ratios up to L; each band holds the position '
                'range fixed across arms'),
        }
    out['pairs'] = pair_out
    out['status'] = 'ok' if any(
        isinstance(v, dict) and v.get('status') == 'ok'
        for v in pair_out.values()) else 'not_estimable'
    if not pair_out:
        out['reason'] = 'no pair declaration could be evaluated'
    if out['status'] != 'ok':
        out['reason'] = out.get('reason') or 'no pair has two common lengths and a shared seed set'
    return _lpd_json_safe(out)
