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
# CPython refuses decimal conversion of a 4,301+ digit integer by default, so
# such a value cannot be written to the JSON report at all.  A declared seed or
# count beyond this bound is refused as unverifiable instead of producing an
# artifact that crashes the strict-JSON writer.
_VT_MAX_REPORTABLE_INT = 10 ** 4000


def _vt_repr(value, limit=240):
    """Bounded, exception-safe repr for untrusted replay diagnostics.

    ``repr(10**10000)`` raises CPython's integer-to-decimal limit.  A
    malformed artifact must produce a named failure, not replace the verdict
    with that formatting exception.  Ordinary values keep their historical
    ``repr`` spelling, so existing diagnostics do not change.
    """
    try:
        text = repr(value)
    except Exception:
        return f'<unrepresentable {type(value).__name__}>'
    return text if len(text) <= limit else text[:limit - 3] + '...'


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
    """Finite real scalar, or None (booleans/strings are not measurements).

    A stored JSON artifact may contain a numeric-looking string such as
    ``"1.0"``; that is a serialisation artefact, not a measured PPL or log
    ratio.  Accepting it here used to let a hand-edited vector replay as
    data, so strings/bytes and booleans are refused before ``float``.
    """
    if isinstance(value, (bool, np.bool_, str, bytes, bytearray)):
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
    if isinstance(value, (bool, np.bool_, bytes, bytearray)):
        return None
    if isinstance(value, str):
        text = value.strip()
        try:
            out = int(text)
        except (TypeError, ValueError, OverflowError):
            return None
        if str(out) != text or abs(out) > _VT_MAX_REPORTABLE_INT:
            return None
        return out
    try:
        out = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        if value != out:
            return None
    except (TypeError, ValueError):
        return None
    if abs(out) > _VT_MAX_REPORTABLE_INT:
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


def _vt_int_arg(value, name, *, minimum=None):
    """Strict integer argument; strings and fractional numbers are refused.

    ``int('6')`` and ``int(6.9)`` silently accept values a caller did not
    mean to declare.  Only an exact integral value (including NumPy scalars
    and an ordinary ``6.0`` spelling) is a count.
    """
    if isinstance(value, (bool, np.bool_, str, bytes, bytearray)):
        raise ValueError(f'{name} must be an integer, got {_vt_repr(value)}')
    try:
        out = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name} must be an integer, got {_vt_repr(value)}') from None
    try:
        exact = bool(value == out)
    except (TypeError, ValueError):
        exact = False
    if not exact:
        raise ValueError(f'{name} must be an integer, got {_vt_repr(value)}')
    if minimum is not None and out < int(minimum):
        raise ValueError(f'{name} must be >= {int(minimum)}, got {out}')
    if abs(out) > _VT_MAX_REPORTABLE_INT:
        raise ValueError(
            f'{name} exceeds the reportable integer bound 10**4000; the '
            'block cannot be serialised as exact JSON')
    return out


def _vt_names(values):
    """Unique string family names, or ``None`` for a malformed list.

    ``set(family)`` used to raise ``TypeError`` for a hand-edited block whose
    entries were lists/objects.  Names are identities in this readout, so only
    non-empty strings are accepted, and duplicates fail closed.
    """
    if not isinstance(values, (list, tuple)):
        return None
    out = []
    for value in values:
        if not isinstance(value, str) or not value:
            return None
        out.append(value)
    return out if len(set(out)) == len(out) else None


def _vt_same_number(a, b):
    """Numeric agreement that also treats two unavailable values as equal."""
    if a is None or b is None:
        return a is None and b is None
    return _vt_close(a, b)


def _vt_ratio_and_shrinkage(validation_mean, test_mean):
    """Derived ``test/validation`` ratio and percent shrinkage, JSON safe.

    Both quantities are only defined for a nonzero validation mean.  A
    representable ratio can still make ``(1 - ratio) * 100`` overflow; an
    infinite percentage is not a measurement and must not enter a strict-JSON
    report.
    """
    if (validation_mean is None or test_mean is None
            or not (math.isfinite(validation_mean)
                    and math.isfinite(test_mean))
            or validation_mean == 0.0):
        return None, None
    try:
        ratio = float(test_mean / validation_mean)
    except (OverflowError, ZeroDivisionError, TypeError, ValueError):
        return None, None
    if not math.isfinite(ratio):
        return None, None
    try:
        shrinkage = float((1.0 - ratio) * 100.0)
    except (OverflowError, TypeError, ValueError):
        shrinkage = None
    if shrinkage is not None and not math.isfinite(shrinkage):
        shrinkage = None
    return ratio, shrinkage


def _vt_direction_agreement(validation_mean, test_mean):
    if (validation_mean is None or test_mean is None
            or not (math.isfinite(validation_mean)
                    and math.isfinite(test_mean))
            or validation_mean == 0.0):
        return None
    return bool((validation_mean > 0.0) == (test_mean > 0.0))


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
        return None, f'no validation cell for comparison {_vt_repr(name)}'
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
    # ``run_power`` also stores the log-ratio vector that the transfer block
    # actually consumes.  When present it is an independent redundancy check:
    # a hand-edited PPL vector must not disagree with the recorded log ratio.
    if 'pilot_log_ratio' in cell:
        log_raw = _vt_list(cell.get('pilot_log_ratio'))
        if log_raw is None:
            return None, 'validation pilot_log_ratio is not an array'
        if len(log_raw) != len(seeds):
            return None, 'validation pilot_log_ratio length does not match seeds'
        for i, (value, a, b) in enumerate(zip(log_raw, pa, pb)):
            li = _vt_num(value)
            want = math.log(a) - math.log(b)
            if li is None or not _vt_close(li, want):
                return None, (f'validation pilot_log_ratio[{i}] does not equal '
                              'log(pilot_ppl_a) - log(pilot_ppl_b)')
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
        return None, f'no test comparison {_vt_repr(name)}'
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
        arm_cell = by_len.get(arm)
        if not isinstance(arm_cell, dict):
            return None, f'arm {_vt_repr(arm)} has no by_length cells'
        cell = arm_cell.get(str(Ln))
        if not isinstance(cell, dict):
            return None, f'arm {_vt_repr(arm)} has no per-seed PPL cell at L{Ln}'
        raw = cell.get('ppl_by_seed')
        if not isinstance(raw, dict):
            return None, f'arm {_vt_repr(arm)} has no per-seed PPL cell at L{Ln}'
        vals = {}
        for key, value in raw.items():
            seed = _vt_seed(key)
            ppl_v = _vt_positive(value)
            if seed is None or ppl_v is None:
                return None, (f'arm {_vt_repr(arm)} L{Ln} has a non-finite or '
                              f'non-integral ppl_by_seed entry {_vt_repr(key)}')
            if seed in vals:
                return None, f'arm {_vt_repr(arm)} L{Ln} has duplicate seed {seed}'
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
    ratio, shrinkage_pct = _vt_ratio_and_shrinkage(validation_mean, test_mean)
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
        'direction_agreement': _vt_direction_agreement(validation_mean,
                                                        test_mean),
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
        raise ValueError(f'validation_transfer: alpha must be numeric, got {_vt_repr(alpha)}') from None
    if not (math.isfinite(alpha) and 0.0 < alpha < 1.0):
        raise ValueError(f'validation_transfer: alpha must be in (0,1), got {_vt_repr(alpha)}')
    min_pairs = _vt_int_arg(min_pairs, 'validation_transfer: min_pairs',
                            minimum=2)
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
        # Keep every declared family slot with its manifest arm/endpoint
        # binding, even when no test cell can be read.  Otherwise the replay
        # gate cannot tell an absent test panel from a block whose slots point
        # at different arms.
        for comp in raw_comps:
            out['comparisons'][comp['name']] = {
                'name': comp['name'], 'a': comp.get('a'), 'b': comp.get('b'),
                'endpoint': comp.get('endpoint'), 'status': 'not_estimable',
                'reason': out['reason'], 'n': 0, 'seeds': [],
                'role': ('descriptive validation-to-test transfer; not in '
                         'the frozen Holm family and never changes a frozen '
                         'verdict'),
                'evidence_level': 'descriptive',
                'frozen_decision_changed': False,
            }
        return _vt_finish(out, family, alpha)
    # The validation artifact is part of the frozen evidence chain.  A
    # power_validation file produced for a different manifest/alpha/family
    # must not be mixed with this test round just because its per-seed PPL
    # vectors happen to parse.
    _pv_problem = None
    if isinstance(power, dict):
        _pv_sha = power.get('manifest_sha256')
        _mf_sha = manifest.get('manifest_sha256')
        if _mf_sha not in (None, ''):
            # When the manifest declares a source identity, the power
            # artifact must carry the same one.  A missing/empty claimed hash
            # is not a match just because the comparison is falsy.
            if _pv_sha in (None, ''):
                _pv_problem = (
                    'power_validation has no manifest_sha256; it cannot be '
                    'bound to the frozen manifest')
            elif str(_pv_sha) != str(_mf_sha):
                _pv_problem = (
                    'power_validation manifest_sha256 does not match the '
                    f'frozen manifest ({str(_pv_sha)[:12]}… vs '
                    f'{str(_mf_sha)[:12]}…)')
        _pv_m = power.get('m_family')
        if _pv_problem is None and _pv_m is not None:
            try:
                _pv_m = _vt_int_arg(_pv_m, 'power_validation m_family',
                                    minimum=1)
            except ValueError as _mexc:
                _pv_problem = str(_mexc)
            else:
                if _pv_m != len(family):
                    _pv_problem = (
                        f'power_validation family size {_pv_m} does not match '
                        f'{len(family)} frozen comparisons')
        if _pv_problem is None and power.get('alpha') is not None:
            _pv_alpha = _vt_num(power.get('alpha'))
            if _pv_alpha is None:
                _pv_problem = ('power_validation alpha is not a finite '
                               f'number: {_vt_repr(power.get("alpha"))}')
            elif not _vt_close(_pv_alpha, alpha):
                _pv_problem = (f'power_validation alpha {_vt_repr(_pv_alpha)} does '
                               f'not match the frozen alpha {_vt_repr(alpha)}')
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
    p, Holm value, classification, count, shrinkage/ratio label, per-seed PPL
    cell or family slot is a ``fail``: the artifact can no longer support the
    claim that the transfer readout was computed by the registered analysis.
    The function does not need the raw validation or test artifacts; it only
    replays the values the block itself stores.

    ``manifest`` is optional.  When supplied, the declared family order and
    each comparison's arm/endpoint binding must match it.  The function never
    raises on malformed JSON: every structural problem is a named failure.
    """
    problems = []
    warnings = []
    if not isinstance(block, dict):
        return {'status': 'fail', 'problems': ['block is not a JSON object'],
                'warnings': warnings}
    if block.get('kind') != VALIDATION_TRANSFER_KIND:
        problems.append(f"kind is {_vt_repr(block.get('kind'))}, expected "
                        f"{_vt_repr(VALIDATION_TRANSFER_KIND)}")
    if block.get('schema') != VALIDATION_TRANSFER_SCHEMA:
        problems.append(f"schema is {_vt_repr(block.get('schema'))}, expected "
                        f"{VALIDATION_TRANSFER_SCHEMA}")
    if block.get('frozen_decision_changed') is not False:
        problems.append('frozen_decision_changed must be false')
    if block.get('evidence_level') != 'descriptive':
        problems.append('evidence_level must be "descriptive"')
    # The policy/role/boundary strings are the declared interpretation of the
    # block.  A replay that ignores them lets a tampered artifact relabel a
    # transport readout as confirmatory evidence while every number still
    # agrees.
    if block.get('policy') != VALIDATION_TRANSFER_POLICY:
        problems.append('policy does not match the registered transfer policy')
    if block.get('role') != VALIDATION_TRANSFER_POLICY['role']:
        problems.append('role does not match the declared descriptive role')
    if block.get('boundary') != VALIDATION_TRANSFER_POLICY['boundary']:
        problems.append('boundary does not match the declared boundary')

    raw_family = block.get('family')
    family = _vt_names(raw_family)
    if family is None:
        problems.append('family is missing, empty, non-string or has '
                        'duplicate names')
        family = []
    if block.get('family_size') != len(family):
        problems.append(f'family_size={_vt_repr(block.get("family_size"))}, expected '
                        f'{len(family)}')
    family_set = set(family)

    # Manifest binding: names/order plus the arm/endpoint identity of every
    # comparison.  A stored block could otherwise replay perfectly while
    # pointing at different arms or a different length endpoint.
    want_by_name = {}
    if manifest is not None:
        if not isinstance(manifest, dict):
            problems.append('manifest is not a JSON object')
        else:
            _mf_sha = manifest.get('manifest_sha256')
            _blk_sha = block.get('manifest_sha256')
            if _mf_sha not in (None, ''):
                if _blk_sha in (None, ''):
                    problems.append('block has no manifest_sha256; it cannot '
                                    'be bound to the frozen manifest')
                elif str(_blk_sha) != str(_mf_sha):
                    problems.append('manifest_sha256 does not match the frozen '
                                    'manifest')
            raw_want = manifest.get('primary_comparisons')
            if raw_want is not None:
                if not isinstance(raw_want, list):
                    problems.append('manifest primary_comparisons is not a list')
                else:
                    want = []
                    for comp in raw_want:
                        if not isinstance(comp, dict) or not isinstance(
                                comp.get('name'), str):
                            problems.append('manifest has a malformed primary '
                                            'comparison')
                            continue
                        want.append(comp['name'])
                        want_by_name[comp['name']] = comp
                    if family and list(family) != want:
                        problems.append(f'family {family} does not match the '
                                        f'manifest comparison names {want}')

    comps = block.get('comparisons')
    if not isinstance(comps, dict):
        problems.append('comparisons is not an object')
        comps = {}
    else:
        for key in comps:
            if key not in family_set:
                problems.append(f'comparisons carries undeclared slot {_vt_repr(key)}')

    alpha = _vt_num(block.get('alpha'))
    if alpha is None or not (0.0 < alpha < 1.0):
        problems.append(f'alpha is not a probability in (0,1): '
                        f'{_vt_repr(block.get("alpha"))}')
        alpha = 0.05
    try:
        min_pairs = _vt_int_arg(block.get('min_pairs'), 'min_pairs',
                                minimum=2)
    except ValueError as exc:
        problems.append(str(exc))
        min_pairs = 2

    n_ok = 0
    named_p = []
    for name in family:
        rec = comps.get(name)
        if not isinstance(rec, dict):
            problems.append(f'{name}: comparison block is missing')
            named_p.append((name, 1.0))
            continue
        if rec.get('name') != name:
            problems.append(f'{name}: name={_vt_repr(rec.get("name"))} does not '
                            'match its family slot')
        want_comp = want_by_name.get(name)
        if isinstance(want_comp, dict):
            for key in ('a', 'b', 'endpoint'):
                if (key in want_comp
                        and rec.get(key) != want_comp.get(key)):
                    problems.append(
                        f'{name}: {key}={_vt_repr(rec.get(key))} does not match the '
                        f'manifest value {_vt_repr(want_comp.get(key))}')
        status = rec.get('status')
        if status == 'not_estimable':
            if not rec.get('reason'):
                problems.append(f'{name}: not_estimable without a reason')
            named_p.append((name, 1.0))
            continue
        if status != 'ok':
            problems.append(f'{name}: unknown status {_vt_repr(status)}')
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
        if rec.get('n') != n:
            problems.append(f'{name}: n={_vt_repr(rec.get("n"))}, expected {n}')
        si = [_vt_seed(s) for s in seeds]
        if any(s is None for s in si):
            problems.append(f'{name}: seeds are not integral')
            named_p.append((name, 1.0))
            continue
        if len(set(si)) != n:
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

        # The block stores the input PPL cells as well as the derived log
        # ratios.  They are not needed to replay the statistics, but when
        # present they must agree with the log ratios; otherwise the artifact
        # mixes two different record sets (and a tampered cell would other-
        # wise pass the audit untouched).
        _ppl_lens = {}
        for side, vec, logs in (
                ('validation', (rec.get('validation_ppl_a'),
                                rec.get('validation_ppl_b')), vv),
                ('test', (rec.get('test_ppl_a'), rec.get('test_ppl_b')), tt)):
            if vec[0] is None and vec[1] is None:
                continue
            if vec[0] is None or vec[1] is None:
                problems.append(f'{name}: {side} PPL cell has only one arm')
                continue
            _pa = _vt_list(vec[0])
            _pb = _vt_list(vec[1])
            if _pa is None or _pb is None or len(_pa) != len(_pb):
                problems.append(f'{name}: {side} PPL cell is not two '
                                'equal-length vectors')
                continue
            _ppl_lens[side] = len(_pa)
            _pav = [_vt_positive(x) for x in _pa]
            _pbv = [_vt_positive(x) for x in _pb]
            if any(x is None for x in _pav + _pbv):
                problems.append(f'{name}: {side} PPL cell has a non-finite or '
                                'non-positive value')
                continue
            # The stored PPL cell is the full side vector, while the log-ratio
            # vectors cover only the common seeds (a missing-seed comparison
            # has fewer paired entries than either side's full cell).  Check
            # set membership rather than positional alignment.
            _side_logs = [math.log(a) - math.log(b)
                          for a, b in zip(_pav, _pbv)]
            # The stored log values are recomputed from the same binary floats
            # and the same operation order, so they are normally bit-identical.
            # The set membership keeps replay linear in the seed count; the
            # tolerance fallback only runs for a value that is not exact.
            _side_log_set = set(_side_logs)
            for i, value in enumerate(logs):
                if value in _side_log_set:
                    continue
                if not any(_vt_close(value, x) for x in _side_logs):
                    problems.append(
                        f'{name}: {side} PPL cell does not reproduce stored '
                        f'log ratio {i}')
                    break

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
                # The stored PPL cells are the full side vectors.  A
                # validation cell longer than the paired n means validation
                # carried a seed the test side did not (missing_test_seeds),
                # and symmetrically for test.  A cell cannot be shorter than
                # the intersection.
                if 'validation' in _ppl_lens:
                    if _ppl_lens['validation'] < n:
                        problems.append(f'{name}: validation PPL cell is '
                                        'shorter than the paired seeds')
                    elif ((_ppl_lens['validation'] > n)
                          != bool(mti)):
                        problems.append(
                            f'{name}: missing_test_seeds is inconsistent with '
                            'the validation PPL cell length')
                if 'test' in _ppl_lens:
                    if _ppl_lens['test'] < n:
                        problems.append(f'{name}: test PPL cell is shorter '
                                        'than the paired seeds')
                    elif ((_ppl_lens['test'] > n) != bool(mvi)):
                        problems.append(
                            f'{name}: missing_validation_seeds is inconsistent '
                            'with the test PPL cell length')
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
            if not _vt_same_number(rec.get(key), got):
                problems.append(f'{name}: {key}={_vt_repr(rec.get(key))}, '
                                f'recomputed {_vt_repr(got)}')
        ci = _vt_list(rec.get('difference_ci95'))
        if ci is None or len(ci) != 2:
            problems.append(f'{name}: difference_ci95 is not a two-element list')
        else:
            if not _vt_same_number(ci[0], dstats.get('ci95_low')):
                problems.append(f'{name}: difference_ci95[0]={_vt_repr(ci[0])}, '
                                f'recomputed {_vt_repr(dstats.get("ci95_low"))}')
            if not _vt_same_number(ci[1], dstats.get('ci95_high')):
                problems.append(f'{name}: difference_ci95[1]={_vt_repr(ci[1])}, '
                                f'recomputed {_vt_repr(dstats.get("ci95_high"))}')
        ci_low = dstats.get('ci95_low')
        ci_high = dstats.get('ci95_high')
        want_cls = _vt_classify(vmean, tmean, ci_low, ci_high,
                                dstats.get('p_exact_signflip'), alpha)
        if rec.get('classification') != want_cls:
            problems.append(f'{name}: classification={_vt_repr(rec.get("classification"))}, '
                            f'recomputed {_vt_repr(want_cls)}')
        want_ratio, want_shrink = _vt_ratio_and_shrinkage(vmean, tmean)
        if not _vt_same_number(rec.get('test_over_validation_ratio'), want_ratio):
            problems.append(f'{name}: test_over_validation_ratio='
                            f'{_vt_repr(rec.get("test_over_validation_ratio"))}, '
                            f'recomputed {_vt_repr(want_ratio)}')
        if not _vt_same_number(rec.get('shrinkage_pct'), want_shrink):
            problems.append(f'{name}: shrinkage_pct='
                            f'{_vt_repr(rec.get("shrinkage_pct"))}, recomputed '
                            f'{_vt_repr(want_shrink)}')
        want_dir = _vt_direction_agreement(vmean, tmean)
        if rec.get('direction_agreement') is not want_dir:
            problems.append(f'{name}: direction_agreement='
                            f'{_vt_repr(rec.get("direction_agreement"))}, recomputed '
                            f'{_vt_repr(want_dir)}')
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
            got = stored.get(name)
            want = holm.get(name) or {}
            if not isinstance(got, dict):
                problems.append(f'{name}: holm entry is missing')
                continue
            if want.get('p_holm') is None:
                continue
            if not _vt_same_number(got.get('p_holm'), want.get('p_holm')):
                problems.append(f'{name}: holm p={_vt_repr(got.get("p_holm"))}, '
                                f'recomputed {_vt_repr(want.get("p_holm"))}')
            if bool(got.get('reject')) is not bool(want.get('reject')):
                problems.append(f'{name}: holm reject={_vt_repr(got.get("reject"))}, '
                                f'recomputed {_vt_repr(want.get("reject"))}')
        for key in stored:
            if key not in family_set:
                problems.append(f'holm carries undeclared slot {_vt_repr(key)}')
    # The per-comparison cached Holm fields must agree with the recomputed
    # family block as well, not merely with themselves.
    for name in family:
        rec = comps.get(name)
        if not isinstance(rec, dict) or rec.get('status') not in (
                'ok', 'not_estimable'):
            continue
        want = holm.get(name) or {}
        if not _vt_same_number(rec.get('difference_p_holm'), want.get('p_holm')):
            problems.append(f'{name}: difference_p_holm='
                            f'{_vt_repr(rec.get("difference_p_holm"))}, recomputed '
                            f'{_vt_repr(want.get("p_holm"))}')
        if bool(rec.get('holm_reject')) is not bool(want.get('reject')):
            problems.append(f'{name}: holm_reject={_vt_repr(rec.get("holm_reject"))}, '
                            f'recomputed {_vt_repr(want.get("reject"))}')

    if block.get('n_ok') != n_ok:
        problems.append(f'n_ok={_vt_repr(block.get("n_ok"))}, recomputed {n_ok}')
    if block.get('n_not_estimable') != len(family) - n_ok:
        problems.append('n_not_estimable='
                        f'{_vt_repr(block.get("n_not_estimable"))}, recomputed '
                        f'{len(family) - n_ok}')
    want_status = ('ok' if n_ok == len(family) and family
                   else 'not_estimable' if n_ok == 0 else 'partial')
    if family and block.get('status') != want_status:
        problems.append(f'status={_vt_repr(block.get("status"))}, recomputed {_vt_repr(want_status)}')
    classes = {}
    n_incomplete = 0
    for name in family:
        rec = comps.get(name) or {}
        cls = (rec.get('classification')
               if rec.get('status') == 'ok' else 'not_estimable')
        if not isinstance(cls, str):
            # A malformed label cannot be a dictionary key; it is already a
            # replay failure when compared with the recomputed classification.
            cls = 'invalid_classification'
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
            problems.append(f'summary.n_ok={_vt_repr(summary.get("n_ok"))}, '
                            f'recomputed {n_ok}')
        if summary.get('n_comparisons') != len(family):
            problems.append(f'summary.n_comparisons={_vt_repr(summary.get("n_comparisons"))}, '
                            f'expected {len(family)}')
        if summary.get('n_not_estimable') != len(family) - n_ok:
            problems.append('summary.n_not_estimable='
                            f'{_vt_repr(summary.get("n_not_estimable"))}, '
                            f'recomputed {len(family) - n_ok}')
        if summary.get('class_counts') != dict(sorted(classes.items())):
            problems.append('summary.class_counts='
                            f'{_vt_repr(summary.get("class_counts"))}, '
                            f'recomputed {_vt_repr(dict(sorted(classes.items())))}')
        for key, want in (
                ('n_opposite_sign', classes.get('opposite_sign', 0)),
                ('n_attenuated', classes.get('same_sign_attenuated', 0)),
                ('n_amplified', classes.get('same_sign_amplified', 0)),
                ('n_not_distinguishable',
                 classes.get('same_sign_not_distinguishable', 0)),
                ('n_incomplete_coverage', n_incomplete)):
            if summary.get(key) != want:
                problems.append(f'summary.{key}={_vt_repr(summary.get(key))}, '
                                f'recomputed {want}')
        _stored_rejections = summary.get('family_holm_rejections')
        if not isinstance(_stored_rejections, (list, tuple)):
            problems.append('summary.family_holm_rejections is not an array')
        elif list(_stored_rejections) != want_reject:
            problems.append('summary.family_holm_rejections='
                            f'{_vt_repr(_stored_rejections)}, '
                            f'recomputed {_vt_repr(want_reject)}')
    if n_ok < len(family) and family:
        warnings.append(f'{len(family) - n_ok}/{len(family)} comparisons have '
                        'no complete validation/test cell')
    return {'status': 'fail' if problems else 'ok',
            'problems': problems, 'warnings': warnings}
