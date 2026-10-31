"""Validation-to-test transfer of the pre-test (validation) effect.

The frozen confirmation decision uses the *sign* of the exploratory validation
estimate as one of its gates: a comparison is supported only when the held-out
test effect points in the same direction.  That rule makes the validation
estimate part of the design, but the report has never shown how large the test
effect actually is relative to it.  A pre-registered effect that shrinks to a
small fraction of its validation value, or one whose seed-level ordering does
not transfer at all, is a different scientific result from a replicated effect
of the same size even when both pass the same sign gate.

This fragment adds a descriptive **transportability / shrinkage** readout for
each frozen comparison.  For every seed it forms

    v_s = log(PPL_a[validation]) - log(PPL_b[validation])
    t_s = log(PPL_a[test])       - log(PPL_b[test])
    d_s = t_s - v_s

on the *same trained model* (same seed, same checkpoint) and reports the two
means, their ratio, the paired difference ``d_s`` with an exact sign-flip p and
a t interval, and a Holm correction over the declared comparison family.  The
paired construction is what makes this more than "two significance verdicts":
both effects are measured on one model, so their difference is a within-seed
split effect rather than an unpaired corpus comparison.

The result is deliberately **not** part of the frozen family.  Validation and
test are different corpora, so a nonzero mean difference is not by itself
"bias": it is the observed transport of the effect to the held-out corpus, and
the pre-test estimate is itself selected (the validation split informed the
frozen direction).  The readout adds no confirmatory p-value, changes no
``p_exact_signflip`` / ``p_holm`` / ``verified`` / ``confirmed`` field, reads no
raw test text (it consumes already-computed per-seed PPL cells), and never
turns a shrinkage estimate into a new decision rule.

Sufficient statistics (seeds and both per-seed log-ratio vectors) are stored so
``audit_gates.py validation_transfer`` can replay every derived number without
the raw artifacts.  A missing validation cell is named and the family keeps its
slot (p=1) rather than silently shrinking the correction family.
"""

import math

VALIDATION_TRANSFER_KIND = 'validation_transfer'
VALIDATION_TRANSFER_SCHEMA = 1
VALIDATION_TRANSFER_POLICY = {
    'version': 'validation-transfer-1',
    'scale': 'log_ppl_ratio: log(PPL_a) - log(PPL_b); positive = a is worse',
    'primary_quantity': ('test mean log effect minus validation mean log '
                         'effect, paired by training seed'),
    'min_pairs': 6,
    'classification_rule': (
        'same_sign_not_distinguishable unless the paired difference CI '
        'excludes zero AND the pointwise exact sign-flip p < alpha; then '
        'same_sign_attenuated / same_sign_amplified when |test| is smaller / '
        'larger than |validation|; opposite_sign when the two point '
        'estimates disagree in sign; no_validation_effect when the '
        'validation mean is exactly zero'),
    'role': (
        'descriptive transportability/shrinkage readout of the frozen '
        'comparisons; it adds no confirmatory test and changes no frozen '
        'p, Holm, verified or confirmed field'),
    'boundary': (
        'validation and test are different corpora, so the difference is a '
        'split/transport effect, not a measurement-error estimate of one '
        'fixed effect; the validation estimate is selected (it informed the '
        'frozen direction), so shrinkage is expected even when the method '
        'transfers. The paired test conditions on the trained seeds and is '
        'not a population claim about unseen seeds or corpora. The Holm '
        'correction over the comparison family is descriptive and does not '
        'redefine the frozen family.'),
}



def _vt_num(value):
    """Finite real scalar, or None (booleans/strings are not measurements)."""
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _vt_positive(value):
    out = _vt_num(value)
    return out if out is not None and out > 0.0 else None


def _vt_seed(value):
    """Integral seed label without accepting bool or a fractional float.

    JSON object keys are strings, so a canonical decimal string is accepted
    here; a non-canonical spelling (``'01'``, ``'1.0'``) is not a second way
    to name the same seed.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip()
        try:
            out = int(text)
        except (TypeError, ValueError, OverflowError):
            return None
        return out if str(out) == text else None
    try:
        out = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        if value != out:
            return None
    except (TypeError, ValueError):
        return None
    return out


def _vt_close(a, b, rel=1e-9, abs_=1e-12):
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return False
    if not (math.isfinite(float(a)) and math.isfinite(float(b))):
        return False
    return abs(float(a) - float(b)) <= abs_ + rel * max(
        1.0, abs(float(a)), abs(float(b)))


def _vt_list(value):
    if isinstance(value, (str, bytes, bytearray, dict)) or value is None:
        return None
    try:
        return list(value)
    except TypeError:
        return None


def _vt_endpoint_length(comp):
    endpoint = comp.get('endpoint')
    if not isinstance(endpoint, str) or '@' not in endpoint:
        return None
    try:
        length = int(endpoint.rsplit('@', 1)[-1])
    except (TypeError, ValueError):
        return None
    return length if length > 0 else None


def _vt_power_pairs(power, name):
    """Extract the validation per-seed cells for one comparison.

    Returns ``(pairs, reason)`` where ``pairs`` is a dict with ``seeds``,
    ``ppl_a`` and ``ppl_b``.  The artifact is produced by
    ``explib.confirmation run_power``; older artifacts lack the PPL vectors
    and are refused rather than reconstructed from a possibly different
    record set.
    """
    comps = power.get('comparisons') if isinstance(power, dict) else None
    if not isinstance(comps, dict):
        return None, 'power_validation artifact has no comparisons object'
    cell = comps.get(name)
    if not isinstance(cell, dict):
        return None, f'no validation cell for comparison {name!r}'
    seeds_raw = _vt_list(cell.get('pilot_seeds'))
    pa_raw = _vt_list(cell.get('pilot_ppl_a'))
    pb_raw = _vt_list(cell.get('pilot_ppl_b'))
    if seeds_raw is None or pa_raw is None or pb_raw is None:
        return None, ('validation cell lacks pilot_seeds/pilot_ppl_a/'
                      'pilot_ppl_b; regenerate power_validation with this '
                      'driver version')
    if not (len(seeds_raw) == len(pa_raw) == len(pb_raw)):
        return None, ('validation cell vectors have different lengths '
                      f'({len(seeds_raw)}/{len(pa_raw)}/{len(pb_raw)})')
    seeds, pa, pb = [], [], []
    for i, (s, a, b) in enumerate(zip(seeds_raw, pa_raw, pb_raw)):
        si = _vt_seed(s)
        ai = _vt_positive(a)
        bi = _vt_positive(b)
        if si is None or ai is None or bi is None:
            return None, (f'validation cell entry {i} is not a finite '
                          'positive PPL / integral seed')
        seeds.append(si)
        pa.append(ai)
        pb.append(bi)
    if len(set(seeds)) != len(seeds):
        return None, 'validation cell has duplicate seeds'
    deltas_raw = _vt_list(cell.get('pilot_deltas'))
    if deltas_raw is not None:
        if len(deltas_raw) != len(seeds):
            return None, 'validation pilot_deltas length does not match seeds'
        for i, (d, a, b) in enumerate(zip(deltas_raw, pa, pb)):
            di = _vt_num(d)
            if di is None or not _vt_close(di, a - b):
                return None, (f'validation pilot_deltas[{i}] does not equal '
                              'pilot_ppl_a - pilot_ppl_b')
    return {'seeds': seeds, 'ppl_a': pa, 'ppl_b': pb}, None


def _vt_test_pairs(finaltest, comp):
    """Extract the test per-seed PPL cells for one comparison.

    The per-seed PPLs come from ``finaltest.by_length[arm][L].ppl_by_seed``;
    their difference is cross-checked against the stored primary ``per_seed``
    vector so the transfer block cannot be built from two different record
    sets.
    """
    a, b = comp.get('a'), comp.get('b')
    name = comp.get('name')
    Ln = _vt_endpoint_length(comp)
    if not isinstance(a, str) or not isinstance(b, str) or not a or not b:
        return None, 'comparison has no usable arm names'
    if Ln is None:
        return None, 'comparison endpoint is not of the form <metric>@<length>'
    comps = finaltest.get('comparisons') if isinstance(finaltest, dict) else None
    if not isinstance(comps, dict) or not isinstance(comps.get(name), dict):
        return None, f'no test comparison {name!r}'
    tcomp = comps[name]
    seeds_raw = _vt_list(tcomp.get('seeds'))
    deltas_raw = _vt_list(tcomp.get('per_seed'))
    if seeds_raw is None or deltas_raw is None:
        return None, 'test comparison lacks the primary per_seed vector'
    if len(seeds_raw) != len(deltas_raw):
        return None, ('test comparison seeds and per_seed vectors have '
                      'different lengths')
    by_len = finaltest.get('by_length') if isinstance(finaltest, dict) else None
    if not isinstance(by_len, dict):
        return None, 'final-test stats carry no by_length cells'
    ppl = {}
    for arm in (a, b):
        cell = ((by_len.get(arm) or {}).get(str(Ln)) or {})
        raw = cell.get('ppl_by_seed')
        if not isinstance(raw, dict):
            return None, f'arm {arm!r} has no per-seed PPL cell at L{Ln}'
        vals = {}
        for key, value in raw.items():
            seed = _vt_seed(key)
            ppl_v = _vt_positive(value)
            if seed is None or ppl_v is None:
                return None, (f'arm {arm!r} L{Ln} has a non-finite or '
                              f'non-integral ppl_by_seed entry {key!r}')
            if seed in vals:
                return None, f'arm {arm!r} L{Ln} has duplicate seed {seed}'
            vals[seed] = ppl_v
        ppl[arm] = vals
    seeds, pa, pb = [], [], []
    for i, (s, delta) in enumerate(zip(seeds_raw, deltas_raw)):
        si = _vt_seed(s)
        di = _vt_num(delta)
        if si is None or di is None:
            return None, f'test per_seed entry {i} is not a finite pair'
        if si not in ppl[a] or si not in ppl[b]:
            return None, f'test comparison seed {si} is missing a PPL cell'
        if not _vt_close(di, ppl[a][si] - ppl[b][si]):
            return None, (f'test per_seed[{i}] does not equal the by_length '
                          'PPL difference of the same seed')
        seeds.append(si)
        pa.append(ppl[a][si])
        pb.append(ppl[b][si])
    if len(set(seeds)) != len(seeds):
        return None, 'test comparison has duplicate seeds'
    # The arm-level by_length cell may carry seeds that this comparison
    # dropped (e.g. an initialization mismatch).  Those seeds are not part of
    # the comparison vector; only the stored per_seed seeds define it, and
    # each of them was cross-checked against its PPL cell above.
    return {'seeds': seeds, 'ppl_a': pa, 'ppl_b': pb}, None


def _vt_classify(validation_mean, test_mean, ci_low, ci_high, p_value,
                 alpha=0.05):
    if validation_mean is None or test_mean is None:
        return 'not_estimable'
    if not (math.isfinite(validation_mean) and math.isfinite(test_mean)):
        return 'not_estimable'
    if validation_mean == 0.0:
        return 'no_validation_effect'
    if test_mean == 0.0:
        return 'test_zero'
    if (validation_mean > 0.0) != (test_mean > 0.0):
        return 'opposite_sign'
    if ci_low is None or ci_high is None:
        return 'not_estimable'
    # The label is a claim that the split difference is distinguishable, so it
    # requires BOTH the t interval and the exact sign-flip p to clear alpha.
    # Using the interval alone would let a skewed 12-seed panel print
    # "attenuated" when the exact test did not reject.
    excludes_zero = bool(ci_low > 0.0 or ci_high < 0.0)
    exact_ok = (p_value is not None and math.isfinite(p_value)
                and p_value < alpha)
    if not (excludes_zero and exact_ok):
        return 'same_sign_not_distinguishable'
    if abs(test_mean) < abs(validation_mean):
        return 'same_sign_attenuated'
    if abs(test_mean) > abs(validation_mean):
        return 'same_sign_amplified'
    return 'same_sign_shifted'


def _vt_record(name, comp, validation_mean, test_mean, diff_stats, seeds,
               validation_log, test_log, validation_ppl, test_ppl, alpha=0.05):
    """Assemble one JSON-safe comparison block from already-computed parts."""
    if isinstance(diff_stats, dict):
        ci_low = _vt_num(diff_stats.get('ci95_low'))
        ci_high = _vt_num(diff_stats.get('ci95_high'))
        p = _vt_num(diff_stats.get('p_exact_signflip'))
        mean = _vt_num(diff_stats.get('mean'))
        std = _vt_num(diff_stats.get('std'))
        dz = _vt_num(diff_stats.get('cohens_dz'))
    else:
        ci_low = ci_high = p = mean = std = dz = None
    ratio = None
    if (validation_mean is not None and test_mean is not None
            and validation_mean != 0.0):
        try:
            ratio = float(test_mean / validation_mean)
        except (OverflowError, ZeroDivisionError):
            ratio = None
        if ratio is not None and not math.isfinite(ratio):
            ratio = None
    shrinkage_pct = None
    if ratio is not None:
        try:
            shrinkage_pct = float((1.0 - ratio) * 100.0)
        except OverflowError:
            shrinkage_pct = None
    rec = {
        'name': name,
        'a': comp.get('a'),
        'b': comp.get('b'),
        'endpoint': comp.get('endpoint'),
        'status': 'ok',
        'n': len(seeds),
        'seeds': [int(s) for s in seeds],
        'validation_per_seed_log_ratio': [float(x) for x in validation_log],
        'test_per_seed_log_ratio': [float(x) for x in test_log],
        'difference_per_seed': [float(t - v) for t, v in zip(test_log, validation_log)],
        'validation_ppl_a': [float(x) for x in validation_ppl[0]],
        'validation_ppl_b': [float(x) for x in validation_ppl[1]],
        'test_ppl_a': [float(x) for x in test_ppl[0]],
        'test_ppl_b': [float(x) for x in test_ppl[1]],
        'validation_mean_log_ratio': validation_mean,
        'test_mean_log_ratio': test_mean,
        'test_over_validation_ratio': ratio,
        'shrinkage_pct': shrinkage_pct,
        'difference_mean': mean,
        'difference_std': std,
        'difference_cohens_dz': dz,
        'difference_ci95': ([ci_low, ci_high]
                            if ci_low is not None and ci_high is not None
                            else None),
        'difference_p_exact_signflip': p,
        'difference_p_holm': None,
        'classification': _vt_classify(validation_mean, test_mean,
                                       ci_low, ci_high, p, alpha),
        'direction_agreement': (None if validation_mean is None
                                or test_mean is None
                                or validation_mean == 0.0
                                else bool((validation_mean > 0.0) == (test_mean > 0.0))),
        'role': ('descriptive validation-to-test transfer; not in the frozen '
                 'Holm family and never changes a frozen verdict'),
        'evidence_level': 'descriptive',
        'frozen_decision_changed': False,
    }
    return rec


def _vt_finish(out, family, alpha):
    """Attach the declared-family Holm block, counts and summary.

    Kept in one place so a ``not_estimable`` early return carries the same
    family slots and summary shape as a successful build; otherwise the replay
    gate could not distinguish "no evidence" from a malformed artifact.
    """
    named_p = []
    for name in family:
        rec = out['comparisons'].get(name) or {}
        p = (rec.get('difference_p_exact_signflip')
             if rec.get('status') == 'ok' else None)
        named_p.append((name, p if p is not None else 1.0))
    out['holm'] = holm_adjust(named_p, alpha=alpha)
    n_ok = 0
    for name in family:
        rec = out['comparisons'].setdefault(name, {
            'name': name, 'status': 'not_estimable',
            'reason': out.get('reason') or 'comparison block was not built',
            'n': 0, 'seeds': [],
            'role': ('descriptive validation-to-test transfer; not in the '
                     'frozen Holm family and never changes a frozen verdict'),
            'evidence_level': 'descriptive',
            'frozen_decision_changed': False,
        })
        h = out['holm'].get(name) or {}
        rec['difference_p_holm'] = h.get('p_holm')
        rec['holm_reject'] = bool(h.get('reject'))
        if rec.get('status') == 'ok':
            n_ok += 1
    out['n_ok'] = n_ok
    out['n_not_estimable'] = len(family) - n_ok
    if n_ok == len(family):
        out['status'] = 'ok'
    elif n_ok == 0:
        out['status'] = 'not_estimable'
        out.setdefault('reason',
                       'no comparison has a complete validation and test cell')
    else:
        out['status'] = 'partial'
        out.setdefault('reason',
                       f'{n_ok}/{len(family)} comparisons have a complete '
                       'validation and test cell')
    classes = {}
    for name in family:
        cls = (out['comparisons'][name] or {}).get('classification') or 'not_estimable'
        classes[cls] = classes.get(cls, 0) + 1
    out['summary'] = {
        'n_comparisons': len(family),
        'n_ok': n_ok,
        'n_not_estimable': len(family) - n_ok,
        'class_counts': dict(sorted(classes.items())),
        'n_opposite_sign': classes.get('opposite_sign', 0),
        'n_attenuated': classes.get('same_sign_attenuated', 0),
        'n_amplified': classes.get('same_sign_amplified', 0),
        'n_not_distinguishable': classes.get('same_sign_not_distinguishable', 0),
        'n_incomplete_coverage': sum(
            1 for name in family
            if (out['comparisons'].get(name) or {}).get('coverage_ok') is False),
        'family_holm_rejections': sorted(
            name for name in family
            if (out['holm'].get(name) or {}).get('reject')),
        'note': ('classifications are descriptive labels computed from the '
                 'paired split difference; they are not new hypothesis tests '
                 'and never enter the frozen family'),
    }
    return out


def validation_transfer(power, test_stats, manifest, *, alpha=None,
                        min_pairs=6):
    """Build the validation-to-test transfer block for the frozen family.

    ``power`` is the ``analysis_v12/power_validation.json`` object (or an
    equivalent mapping) and ``test_stats`` is the whole
    ``analysis_v12/stats.json`` object (a bare ``finaltest`` object is also
    accepted).  The function only reads already-computed per-seed PPL cells;
    it never opens a data split.

    A comparison whose validation or test cell is missing, incomplete or
    internally inconsistent is named with ``status='not_estimable'`` and keeps
    its slot in the declared family.  The function fails closed on malformed
    top-level arguments with ``ValueError``; artifact-level problems are
    reported, not raised, so one missing panel cannot abort a report build.
    """
    if not isinstance(manifest, dict):
        raise ValueError('validation_transfer: manifest must be a JSON object')
    if not isinstance(test_stats, dict):
        raise ValueError('validation_transfer: test_stats must be a JSON object')
    stat = manifest.get('statistics') if isinstance(manifest.get('statistics'), dict) else {}
    if alpha is None:
        alpha = stat.get('alpha', 0.05)
    try:
        alpha = float(alpha)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'validation_transfer: alpha must be numeric, got {alpha!r}') from None
    if not (math.isfinite(alpha) and 0.0 < alpha < 1.0):
        raise ValueError(f'validation_transfer: alpha must be in (0,1), got {alpha!r}')
    try:
        min_pairs = int(min_pairs)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('validation_transfer: min_pairs must be an integer') from None
    if min_pairs < 2:
        raise ValueError('validation_transfer: min_pairs must be >= 2')
    if isinstance(power, dict) and power.get('kind') == VALIDATION_TRANSFER_KIND:
        # A stored transfer block is not a power artifact; accepting it here
        # would let a replayed block be re-analysed as if it were raw input.
        raise ValueError('validation_transfer: power argument is a transfer block, not a power_validation artifact')
    finaltest = test_stats.get('finaltest') if isinstance(test_stats.get('finaltest'), dict) else test_stats
    comps = finaltest.get('comparisons') if isinstance(finaltest, dict) else None
    raw_comps = manifest.get('primary_comparisons')
    if not isinstance(raw_comps, list) or not raw_comps:
        raise ValueError('validation_transfer: manifest declares no primary comparisons')
    family = []
    for c in raw_comps:
        if not isinstance(c, dict) or not isinstance(c.get('name'), str):
            raise ValueError('validation_transfer: every primary comparison needs a name')
        family.append(c['name'])
    if len(set(family)) != len(family):
        raise ValueError('validation_transfer: duplicate comparison names in the family')

    out = {
        'kind': VALIDATION_TRANSFER_KIND,
        'schema': VALIDATION_TRANSFER_SCHEMA,
        'policy': VALIDATION_TRANSFER_POLICY,
        'alpha': alpha,
        'min_pairs': min_pairs,
        'family': family,
        'family_size': len(family),
        'manifest_sha256': manifest.get('manifest_sha256'),
        'upstream': ('analysis_v12/power_validation.json (validation per-seed '
                     'PPL) + analysis_v12/stats.json finaltest (test per-seed '
                     'PPL); no raw split is read'),
        'status': 'not_estimable',
        'n_ok': 0,
        'n_not_estimable': 0,
        'comparisons': {},
        'holm': {},
        'summary': {},
        'frozen_decision_changed': False,
        'evidence_level': 'descriptive',
        'role': VALIDATION_TRANSFER_POLICY['role'],
        'boundary': VALIDATION_TRANSFER_POLICY['boundary'],
    }
    if not isinstance(comps, dict) or not comps:
        out['reason'] = 'test stats carry no finaltest comparisons'
        return _vt_finish(out, family, alpha)
    # The validation artifact is part of the frozen evidence chain.  A
    # power_validation file produced for a different manifest/alpha/family
    # must not be mixed with this test round just because its per-seed PPL
    # vectors happen to parse.
    _pv_problem = None
    if isinstance(power, dict):
        _pv_sha = power.get('manifest_sha256')
        _mf_sha = manifest.get('manifest_sha256')
        if (_pv_sha and _mf_sha and str(_pv_sha) != str(_mf_sha)):
            _pv_problem = (
                'power_validation manifest_sha256 does not match the frozen '
                f'manifest ({str(_pv_sha)[:12]}… vs {str(_mf_sha)[:12]}…)')
        _pv_m = power.get('m_family')
        if _pv_problem is None and _pv_m is not None:
            try:
                _pv_m = int(_pv_m)
            except (TypeError, ValueError, OverflowError):
                _pv_m = None
            if _pv_m is not None and _pv_m != len(family):
                _pv_problem = (f'power_validation family size {_pv_m} does '
                               f'not match {len(family)} frozen comparisons')
        _pv_alpha = _vt_num(power.get('alpha'))
        if (_pv_problem is None and _pv_alpha is not None
                and not _vt_close(_pv_alpha, alpha)):
            _pv_problem = (f'power_validation alpha {_pv_alpha!r} does not '
                           f'match the frozen alpha {alpha!r}')
    if _pv_problem is not None:
        out['status'] = 'not_estimable'
        out['reason'] = _pv_problem
        out['n_not_estimable'] = len(family)
        for comp in raw_comps:
            out['comparisons'][comp['name']] = {
                'name': comp['name'], 'a': comp.get('a'), 'b': comp.get('b'),
                'endpoint': comp.get('endpoint'), 'status': 'not_estimable',
                'reason': _pv_problem, 'n': 0, 'seeds': [],
                'role': ('descriptive validation-to-test transfer; not in '
                         'the frozen Holm family and never changes a frozen '
                         'verdict'),
                'evidence_level': 'descriptive',
                'frozen_decision_changed': False,
            }
        return _vt_finish(out, family, alpha)

    for comp in raw_comps:
        name = comp['name']
        rec = {
            'name': name,
            'a': comp.get('a'),
            'b': comp.get('b'),
            'endpoint': comp.get('endpoint'),
            'status': 'not_estimable',
            'reason': None,
            'n': 0,
            'seeds': [],
            'role': ('descriptive validation-to-test transfer; not in the '
                     'frozen Holm family and never changes a frozen verdict'),
            'evidence_level': 'descriptive',
            'frozen_decision_changed': False,
        }
        vpairs, vreason = _vt_power_pairs(power, name)
        tpairs, treason = _vt_test_pairs(finaltest, comp)
        if vpairs is None or tpairs is None:
            rec['reason'] = vreason or treason
            out['comparisons'][name] = rec
            continue
        vseeds = vpairs['seeds']
        tseeds = tpairs['seeds']
        common = sorted(set(vseeds) & set(tseeds))
        missing_v = sorted(set(tseeds) - set(vseeds))
        missing_t = sorted(set(vseeds) - set(tseeds))
        if len(common) < min_pairs:
            rec['reason'] = (f'only {len(common)} seeds are present on both '
                             f'sides; min_pairs={min_pairs} '
                             f'(missing validation={missing_v}, missing '
                             f'test={missing_t})')
            out['comparisons'][name] = rec
            continue
        vi = {s: i for i, s in enumerate(vseeds)}
        ti = {s: i for i, s in enumerate(tseeds)}
        order = common
        vlog = [math.log(vpairs['ppl_a'][vi[s]]) - math.log(vpairs['ppl_b'][vi[s]])
                for s in order]
        tlog = [math.log(tpairs['ppl_a'][ti[s]]) - math.log(tpairs['ppl_b'][ti[s]])
                for s in order]
        if not all(math.isfinite(x) for x in vlog + tlog):
            rec['reason'] = 'a per-seed log ratio is not representable'
            out['comparisons'][name] = rec
            continue
        vmean = _scaled_mean_or_raw(vlog)
        tmean = _scaled_mean_or_raw(tlog)
        d = [t - v for t, v in zip(tlog, vlog)]
        try:
            dstats = paired_effect_stats(d, alpha=alpha, min_n=min_pairs,
                                         seeds=order)
        except Exception as exc:  # pragma: no cover - defensive
            rec['reason'] = f'{type(exc).__name__}: {exc}'
            out['comparisons'][name] = rec
            continue
        rec = _vt_record(
            name, comp, vmean, tmean, dstats, order, vlog, tlog,
            (vpairs['ppl_a'], vpairs['ppl_b']),
            (tpairs['ppl_a'], tpairs['ppl_b']), alpha=alpha)
        rec['coverage_ok'] = not missing_v and not missing_t
        rec['missing_validation_seeds'] = missing_v
        rec['missing_test_seeds'] = missing_t
        if not rec['coverage_ok']:
            rec['coverage_note'] = (
                'the paired transfer uses the seeds present on both sides; '
                'the missing seeds are named and the comparison is not a '
                'complete frozen panel')
        out['comparisons'][name] = rec

    # Holm over the DECLARED family.  A missing comparison keeps its slot
    # (p=1.0) so an incomplete validation artifact cannot shrink the family
    # and make the remaining split-difference tests look sharper.
    return _vt_finish(out, family, alpha)


def replay_validation_transfer(block, manifest=None):
    """Recompute a stored transfer block from its sufficient statistics.

    Used by the audit gate.  A mismatch in any stored mean, interval, exact
    p, Holm value, classification or family slot is a ``fail``: the artifact
    can no longer support the claim that the transfer readout was computed by
    the registered analysis.  The function does not need the raw validation or
    test artifacts.
    """
    problems = []
    warnings = []
    if not isinstance(block, dict):
        return {'status': 'fail', 'problems': ['block is not a JSON object'],
                'warnings': warnings}
    if block.get('kind') != VALIDATION_TRANSFER_KIND:
        problems.append(f"kind is {block.get('kind')!r}, expected "
                        f"{VALIDATION_TRANSFER_KIND!r}")
    if block.get('schema') != VALIDATION_TRANSFER_SCHEMA:
        problems.append(f"schema is {block.get('schema')!r}, expected "
                        f"{VALIDATION_TRANSFER_SCHEMA}")
    if block.get('frozen_decision_changed') is not False:
        problems.append('frozen_decision_changed must be false')
    if block.get('evidence_level') != 'descriptive':
        problems.append('evidence_level must be "descriptive"')
    family = _vt_list(block.get('family'))
    if family is None or not family or len(set(family)) != len(family):
        problems.append('family is missing, empty or has duplicate names')
        family = []
    if block.get('family_size') != len(family):
        problems.append(f'family_size={block.get("family_size")!r}, expected '
                        f'{len(family)}')
    comps = block.get('comparisons')
    if not isinstance(comps, dict):
        problems.append('comparisons is not an object')
        comps = {}
    alpha = _vt_num(block.get('alpha'))
    if alpha is None or not (0.0 < alpha < 1.0):
        problems.append(f'alpha is not a probability in (0,1): {block.get("alpha")!r}')
        alpha = 0.05
    min_pairs = _vt_seed(block.get('min_pairs'))
    if min_pairs is None or min_pairs < 2:
        problems.append(f'min_pairs is not an integer >= 2: {block.get("min_pairs")!r}')
        min_pairs = 2
    if manifest is not None:
        want = [c.get('name') for c in (manifest.get('primary_comparisons') or [])
                if isinstance(c, dict)]
        if family and want and list(family) != list(want):
            problems.append(f'family {family} does not match the manifest '
                            f'comparison names {want}')
    n_ok = 0
    named_p = []
    for name in family:
        rec = comps.get(name)
        if not isinstance(rec, dict):
            problems.append(f'{name}: comparison block is missing')
            named_p.append((name, 1.0))
            continue
        status = rec.get('status')
        if status == 'not_estimable':
            if not rec.get('reason'):
                problems.append(f'{name}: not_estimable without a reason')
            named_p.append((name, 1.0))
            continue
        if status != 'ok':
            problems.append(f'{name}: unknown status {status!r}')
            named_p.append((name, 1.0))
            continue
        n_ok += 1
        seeds = _vt_list(rec.get('seeds'))
        vlog = _vt_list(rec.get('validation_per_seed_log_ratio'))
        tlog = _vt_list(rec.get('test_per_seed_log_ratio'))
        dvec = _vt_list(rec.get('difference_per_seed'))
        if seeds is None or vlog is None or tlog is None or dvec is None:
            problems.append(f'{name}: sufficient-statistics vectors are missing')
            named_p.append((name, 1.0))
            continue
        n = len(seeds)
        if not (n == len(vlog) == len(tlog) == len(dvec)):
            problems.append(f'{name}: vector lengths differ')
            named_p.append((name, 1.0))
            continue
        si = [_vt_seed(s) for s in seeds]
        if any(s is None for s in si) or len(set(si)) != n:
            problems.append(f'{name}: seeds are not unique integers')
            named_p.append((name, 1.0))
            continue
        vv = [_vt_num(x) for x in vlog]
        tt = [_vt_num(x) for x in tlog]
        dd = [_vt_num(x) for x in dvec]
        if any(x is None for x in vv + tt + dd):
            problems.append(f'{name}: a per-seed log value is not finite')
            named_p.append((name, 1.0))
            continue
        for i in range(n):
            if not _vt_close(dd[i], tt[i] - vv[i]):
                problems.append(f'{name}: difference_per_seed[{i}] does not '
                                'equal test - validation')
        mv = _vt_list(rec.get('missing_validation_seeds'))
        mt = _vt_list(rec.get('missing_test_seeds'))
        if mv is None or mt is None:
            problems.append(f'{name}: missing-seed lists are not arrays')
        else:
            mvi = [_vt_seed(x) for x in mv]
            mti = [_vt_seed(x) for x in mt]
            if any(x is None for x in mvi + mti):
                problems.append(f'{name}: missing-seed list has a non-integer')
            else:
                if mvi != sorted(set(mvi)) or mti != sorted(set(mti)):
                    problems.append(f'{name}: missing-seed lists are not '
                                    'sorted unique integers')
                if rec.get('coverage_ok') is not (not mvi and not mti):
                    problems.append(f'{name}: coverage_ok is inconsistent '
                                    'with the named missing seeds')
        if n < min_pairs:
            problems.append(f'{name}: n={n} below min_pairs={min_pairs}')
        try:
            dstats = paired_effect_stats(dd, alpha=alpha, min_n=min_pairs,
                                         seeds=si)
        except Exception as exc:
            problems.append(f'{name}: replay statistics failed '
                            f'({type(exc).__name__}: {exc})')
            named_p.append((name, 1.0))
            continue
        vmean = _scaled_mean_or_raw(vv)
        tmean = _scaled_mean_or_raw(tt)
        for key, got in (
                ('validation_mean_log_ratio', vmean),
                ('test_mean_log_ratio', tmean),
                ('difference_mean', dstats.get('mean')),
                ('difference_std', dstats.get('std')),
                ('difference_cohens_dz', dstats.get('cohens_dz')),
                ('difference_p_exact_signflip', dstats.get('p_exact_signflip'))):
            if not _vt_close(rec.get(key), got):
                problems.append(f'{name}: {key}={rec.get(key)!r}, '
                                f'recomputed {got!r}')
        ci = _vt_list(rec.get('difference_ci95'))
        if ci is None or len(ci) != 2:
            problems.append(f'{name}: difference_ci95 is not a two-element list')
        else:
            if not _vt_close(ci[0], dstats.get('ci95_low')):
                problems.append(f'{name}: difference_ci95[0]={ci[0]!r}, '
                                f'recomputed {dstats.get("ci95_low")!r}')
            if not _vt_close(ci[1], dstats.get('ci95_high')):
                problems.append(f'{name}: difference_ci95[1]={ci[1]!r}, '
                                f'recomputed {dstats.get("ci95_high")!r}')
        ci_low = dstats.get('ci95_low')
        ci_high = dstats.get('ci95_high')
        want_cls = _vt_classify(vmean, tmean, ci_low, ci_high,
                                dstats.get('p_exact_signflip'), alpha)
        if rec.get('classification') != want_cls:
            problems.append(f'{name}: classification={rec.get("classification")!r}, '
                            f'recomputed {want_cls!r}')
        ratio = None
        if vmean != 0.0:
            try:
                ratio = float(tmean / vmean)
            except (OverflowError, ZeroDivisionError):
                ratio = None
            if ratio is not None and not math.isfinite(ratio):
                ratio = None
        if not _vt_close(rec.get('test_over_validation_ratio'), ratio):
            problems.append(f'{name}: test_over_validation_ratio='
                            f'{rec.get("test_over_validation_ratio")!r}, '
                            f'recomputed {ratio!r}')
        if rec.get('frozen_decision_changed') is not False:
            problems.append(f'{name}: frozen_decision_changed must be false')
        if rec.get('evidence_level') != 'descriptive':
            problems.append(f'{name}: evidence_level must be "descriptive"')
        named_p.append((name, dstats.get('p_exact_signflip')))
    holm = holm_adjust(named_p, alpha=alpha)
    stored = block.get('holm')
    if not isinstance(stored, dict):
        problems.append('holm block is missing')
    else:
        for name, _p in named_p:
            got = (stored.get(name) or {}).get('p_holm')
            want = (holm.get(name) or {}).get('p_holm')
            if want is None:
                continue
            if got is None or not _vt_close(got, want):
                problems.append(f'{name}: holm p={got!r}, recomputed {want!r}')
    if block.get('n_ok') != n_ok:
        problems.append(f'n_ok={block.get("n_ok")!r}, recomputed {n_ok}')
    want_status = ('ok' if n_ok == len(family) and family
                   else 'not_estimable' if n_ok == 0 else 'partial')
    if family and block.get('status') != want_status:
        problems.append(f'status={block.get("status")!r}, recomputed {want_status!r}')
    classes = {}
    n_incomplete = 0
    for name in family:
        rec = comps.get(name) or {}
        cls = (rec.get('classification')
               if rec.get('status') == 'ok' else 'not_estimable')
        classes[cls] = classes.get(cls, 0) + 1
        if rec.get('coverage_ok') is False:
            n_incomplete += 1
    want_reject = sorted(name for name in family
                         if (holm.get(name) or {}).get('reject'))
    summary = block.get('summary')
    if not isinstance(summary, dict):
        problems.append('summary is not an object')
    else:
        if summary.get('n_ok') != n_ok:
            problems.append(f'summary.n_ok={summary.get("n_ok")!r}, '
                            f'recomputed {n_ok}')
        if summary.get('n_comparisons') != len(family):
            problems.append(f'summary.n_comparisons={summary.get("n_comparisons")!r}, '
                            f'expected {len(family)}')
        if summary.get('n_not_estimable') != len(family) - n_ok:
            problems.append('summary.n_not_estimable='
                            f'{summary.get("n_not_estimable")!r}, '
                            f'recomputed {len(family) - n_ok}')
        if summary.get('class_counts') != dict(sorted(classes.items())):
            problems.append('summary.class_counts='
                            f'{summary.get("class_counts")!r}, '
                            f'recomputed {dict(sorted(classes.items()))!r}')
        for key, want in (
                ('n_opposite_sign', classes.get('opposite_sign', 0)),
                ('n_attenuated', classes.get('same_sign_attenuated', 0)),
                ('n_amplified', classes.get('same_sign_amplified', 0)),
                ('n_not_distinguishable',
                 classes.get('same_sign_not_distinguishable', 0)),
                ('n_incomplete_coverage', n_incomplete)):
            if summary.get(key) != want:
                problems.append(f'summary.{key}={summary.get(key)!r}, '
                                f'recomputed {want}')
        if list(summary.get('family_holm_rejections') or []) != want_reject:
            problems.append('summary.family_holm_rejections='
                            f'{summary.get("family_holm_rejections")!r}, '
                            f'recomputed {want_reject!r}')
    if n_ok < len(family) and family:
        warnings.append(f'{len(family) - n_ok}/{len(family)} comparisons have '
                        'no complete validation/test cell')
    return {'status': 'fail' if problems else 'ok',
            'problems': problems, 'warnings': warnings}
