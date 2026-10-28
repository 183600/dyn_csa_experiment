

def panel_look_snapshot(recs, *, design=None, baselines=None, factorials=None):
    """Measured declared contrasts with their exact per-seed deltas.

    This is the evidence a look needs: the seed set that was actually paired
    at aggregation time and the signed ``with - without`` primary-endpoint
    delta for each seed.  Only contrasts with at least one controlled pair are
    emitted; missing arms are the job of the completeness audit, not of the
    look history.

    ``factorials`` additionally records each controlled 2x2 interaction as a
    per-seed look series (default: ``FACTORIAL_DESIGN``; pass ``{}`` to omit).
    A declared interaction family is an analysis family too, so a seed top-up
    must not silently buy an interaction or an additivity claim.
    """
    design = ABLATION_DESIGN if design is None else design
    baselines = BASELINE_DESIGN if baselines is None else baselines
    factorials = FACTORIAL_DESIGN if factorials is None else factorials
    # The aggregation path hands this function the raw summary mapping, whose
    # keys are strings/legacy ids; the unit-test path hands it the normalized
    # {(variant, protocol, seed): rec} map.  Normalize the former exactly like
    # methodology_audit does, or the first raw key is unpacked as a triple and
    # the look is silently lost ("too many values to unpack").
    if not all(isinstance(k, tuple) and len(k) == 3 for k in recs):
        recs, _amb = _variant_seed_map(recs)
    out = {}
    groups = (('component', design), ('baseline', baselines))
    for kind, group in groups:
        if not isinstance(group, dict):
            continue
        for comp, contrasts in group.items():
            if isinstance(contrasts, (str, bytes)):
                continue
            try:
                items = list(contrasts or ())
            except TypeError:
                continue
            for item in items:
                if not isinstance(item, (tuple, list)) or len(item) < 2:
                    continue
                a, b = item[0], item[1]
                if not (isinstance(a, str) and isinstance(b, str)
                        and a.strip() and b.strip()):
                    continue
                conds = _paired_seed_deltas(
                    recs, a, b, strict_hyper=(kind == 'component'))
                for (ts, bs), (deltas, shared, skipped) in sorted(conds.items()):
                    _skipped = {s for s, _why in skipped}
                    seeds = [s for s in shared if s not in _skipped]
                    if not deltas or not seeds:
                        continue
                    key = _contrast_look_key(kind, comp, a, b, ts, bs)
                    sf = exact_signflip_p(deltas)
                    out[key] = {
                        'kind': kind, 'component': comp,
                        'with': a, 'without': b,
                        'tag_with': ts or '', 'tag_without': bs or '',
                        'seeds': [int(s) for s in seeds],
                        'deltas': [float(d) for d in deltas],
                        'n': int(len(deltas)),
                        'mean': float(np.mean(deltas)),
                        'p_exact_signflip': float(sf.get('p', float('nan'))),
                        'n_skipped_pairs': int(len(skipped)),
                    }
    out.update(factorial_look_snapshot(recs, design=factorials))
    return out


def load_look_history(outdir):
    """Read ``<outdir>/look_history.json``; a missing file is an empty history.

    A malformed file is returned as ``{'schema': ..., 'looks': [],
    'invalid': reason}`` instead of being silently treated as "no looks": a
    corrupted look record must not erase the correction it implies.
    """
    path = os.path.join(os.fspath(outdir), 'look_history.json')
    if not os.path.exists(path):
        return {'schema': LOOK_HISTORY_SCHEMA, 'looks': []}
    try:
        with open(path, encoding='utf-8') as f:
            obj = json.load(f)
    except Exception as e:
        return {'schema': LOOK_HISTORY_SCHEMA, 'looks': [],
                'invalid': f'{type(e).__name__}: {e}'}
    if not isinstance(obj, dict):
        return {'schema': LOOK_HISTORY_SCHEMA, 'looks': [],
                'invalid': f'look_history.json is a {type(obj).__name__}, not an object'}
    obj.setdefault('schema', LOOK_HISTORY_SCHEMA)
    if not isinstance(obj.get('looks'), list):
        obj['looks'] = []
        obj['invalid'] = 'looks is not a list'
    return obj


def record_panel_look(outdir, recs, *, design=None, baselines=None,
                      factorials=None, now=None):
    """Append a look snapshot when the measured contrast set changed.

    Re-aggregating an unchanged panel does NOT add a look (the same seed set
    is the same analysis).  A new seed, a newly measured arm or a re-run delta
    does add one.  The file is bounded by :data:`LOOK_HISTORY_MAX_LOOKS`; the
    oldest looks are dropped with an explicit ``truncated`` flag so the record
    never silently becomes a different kind of evidence.  ``factorials`` is
    forwarded to :func:`panel_look_snapshot` so declared 2x2 interactions are
    covered by the same optional-stopping correction as ablations.
    """
    snapshot = panel_look_snapshot(recs, design=design, baselines=baselines,
                                   factorials=factorials)
    hist = load_look_history(outdir)
    if not snapshot:
        # No declared contrast is measured in this panel (e.g. a pure
        # confirmation or kernel run): do not create an empty look file.
        return hist
    if hist.get('invalid'):
        # Keep the reason visible; a fresh record cannot prove what was lost.
        hist.setdefault('recovered_after_invalid', hist['invalid'])
    looks = hist.get('looks') if isinstance(hist.get('looks'), list) else []
    marker = json.dumps(snapshot, sort_keys=True, separators=(',', ':'),
                        ensure_ascii=True, default=str)
    last = looks[-1] if looks and isinstance(looks[-1], dict) else {}
    if last:
        last_marker = json.dumps(last.get('contrasts'), sort_keys=True,
                                 separators=(',', ':'), ensure_ascii=True,
                                 default=str)
        if last_marker == marker:
            return hist
    look_no = int(last.get('look') or len(looks)) + 1 if looks else 1
    entry = {'look': look_no,
             'recorded_at': now or time.strftime('%Y-%m-%dT%H:%M:%S'),
             'n_contrasts': len(snapshot),
             'contrasts': snapshot}
    looks.append(entry)
    truncated = 0
    if len(looks) > LOOK_HISTORY_MAX_LOOKS:
        truncated = len(looks) - LOOK_HISTORY_MAX_LOOKS
        looks = looks[-LOOK_HISTORY_MAX_LOOKS:]
    hist.update({'schema': LOOK_HISTORY_SCHEMA, 'panel': os.fspath(outdir),
                 'looks': looks, 'n_looks': len(looks)})
    if truncated:
        hist['truncated_oldest_looks'] = int(
            hist.get('truncated_oldest_looks') or 0) + int(truncated)
    atomic_write_json(os.path.join(os.fspath(outdir), 'look_history.json'), hist)
    return hist


def looks_by_contrast(history):
    """Flatten a look history into ``{contrast_key: [look, ...]}``."""
    out = {}
    if not isinstance(history, dict):
        return out
    for look in history.get('looks') or []:
        if not isinstance(look, dict):
            continue
        contrasts = look.get('contrasts')
        if not isinstance(contrasts, dict):
            continue
        for key, c in contrasts.items():
            if not (isinstance(c, dict) and isinstance(c.get('seeds'), list)
                    and isinstance(c.get('deltas'), list)):
                continue
            # An empty look is not a look at this contrast; it must not make the
            # whole sequential call fail (the caller would lose the panel's
            # correction entirely).  The public primitive still rejects an
            # empty input contract.
            if not c['seeds'] or not c['deltas']:
                continue
            entry = {
                'look': look.get('look'), 'recorded_at': look.get('recorded_at'),
                'seeds': list(c['seeds']), 'deltas': list(c['deltas']),
                'p_exact_signflip': c.get('p_exact_signflip'),
            }
            # Interaction looks also bind the cell arms and the protocol tag of
            # each seed.  Keep those descriptors here so the interaction reader
            # can reject a look whose estimand changed even when its numeric
            # deltas happen to be unchanged.  Ablation readers simply ignore
            # the extra keys.
            for _meta in ('cells', 'tags'):
                if isinstance(c.get(_meta), dict):
                    entry[_meta] = dict(c[_meta])
            out.setdefault(str(key), []).append(entry)
    return out


def _knobs_equal_across_arms(recs, arms, key):
    """Compare a per-record descriptor (`run_cfg`, `train_knobs`, ...) across
    the arms of one component.  Returns True/False/None: True when at least two
    arms expose the key and all exposed values agree, False when any two
    disagree, None when fewer than two arms expose it (unverifiable)."""
    vals = []
    for a in arms:
        for (v, _t, _s), r in recs.items():
            if v != a:
                continue
            if key in r and r.get(key) is not None:
                vals.append(json.dumps(r.get(key), sort_keys=True, default=str))
                break
            # keep scanning this arm: a legacy/first record without the key
            # must not mask a later record that does record it.
    if len(vals) < 2:
        return None
    return len(set(vals)) == 1

# Settings that must be identical across arms of a controlled contrast.  The
# first two are the authoritative descriptors already used by `pair_reason`;
# the rest are read from the structured `hyperparams` copy so a future setting
# that never made it into the compact run_cfg string is still checked.
_CONTROL_HYPER_KEYS = (
    'steps', 'tokens_seen', 'lr', 'weight_decay', 'warmup', 'batch_size',
    'seq_len', 'd', 'n_layers', 'n_heads', 'd_head', 'mlp_ratio',
    'comp_lambda', 'delta_lr_mult', 'dataset', 'data_version',
    'amp_dtype', 'eager_batch',
)

def _canon_setting(v):
    """Canonical JSON spelling of one setting value (None stays 'missing')."""
    if v is None:
        return None
    return json.dumps(_jsonable(v), sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, default=str)

def _setting_values(recs, arms, getter):
    """Per-arm distinct values of one setting, plus how many records were
    inspected and how many omitted the setting.

    The counts are what let `_setting_control` distinguish a legacy arm (all
    records omit the descriptor) from an arm whose seeds only PARTIALLY carry
    it: the latter is a mixed protocol inside one arm, not an equality proof.
    """
    vals = {a: set() for a in arms}
    n_records = {a: 0 for a in arms}
    n_missing = {a: 0 for a in arms}
    for (v, _t, _s), r in recs.items():
        if v not in vals:
            continue
        n_records[v] += 1
        val = getter(r)
        if val is None:
            n_missing[v] += 1
        else:
            vals[v].add(val)
    return vals, n_records, n_missing

def _setting_control(vals_by_arm, *, check_within_arm=True, n_records=None,
                     n_missing=None):
    """Aggregate one setting across arms.

    Returns `equal` (True/False/None), every arm's distinct values, the arms
    whose own seeds disagree (mixed protocol inside one arm), arms whose
    records only partially carry the descriptor, and arms with no value.
    `equal=None` means unverifiable (legacy record), never a pass.

    A partially recorded arm (some seeds carry the setting, others omit it)
    is a FAILURE, not "unverifiable": the panel cannot show that every seed
    shared the setting, and silently trusting the records that do carry it is
    exactly how a changed optimizer/schedule can hide in one seed.
    """
    arms = list(vals_by_arm)
    nonempty = {a: v for a, v in vals_by_arm.items() if v}
    mixed = sorted(a for a, v in nonempty.items() if len(v) > 1) if check_within_arm else []
    _nr = n_records or {}
    _nm = n_missing or {}
    partial = sorted(a for a in arms
                     if _nr.get(a, 0) > 0 and 0 < _nm.get(a, 0) < _nr.get(a, 0)
                     ) if check_within_arm else []
    missing = sorted(a for a in arms if not vals_by_arm.get(a))
    if not nonempty:
        equal = None
    elif mixed or partial:
        equal = False
    elif len(nonempty) < len(arms):
        equal = None
    else:
        union = set().union(*nonempty.values())
        equal = bool(len(union) == 1)
    return {'equal': equal, 'within_arm_mixed': mixed,
            'partially_recorded': partial, 'missing_arms': missing,
            'n_records': {a: int(_nr.get(a, 0)) for a in arms},
            'n_missing': {a: int(_nm.get(a, 0)) for a in arms},
            'values': {a: sorted(v) for a, v in nonempty.items()}}

def controlled_variables_report(recs, arms):
    """Per-component controlled-variable audit across ALL seed records.

    `pair_reason` is the per-seed hard gate; this report is the panel-level
    diagnostic.  Every record of an arm is inspected (so seed 0 and seed 4
    cannot run under different protocols), a disagreement inside one arm is
    flagged, and the structured `hyperparams` copy is checked for settings
    that the compact run_cfg string does not carry.  The return keeps the
    `run_cfg_equal` / `train_knobs_equal` keys and adds the detail.
    """
    arms = sorted(set(arms))

    def _control(getter):
        _vals, _nr, _nm = _setting_values(recs, arms, getter)
        return _setting_control(_vals, n_records=_nr, n_missing=_nm)

    run_cfg = _control(lambda r: _canon_setting(r.get('run_cfg')))
    train_knobs = _control(lambda r: _canon_setting(r.get('train_knobs')))
    # The code-semantics stamp is a measurement identity, not decoration.  A
    # run_cfg normally embeds it, but legacy records may omit the suffix; an
    # arm whose seeds were measured under different stamps is a mixed protocol,
    # even when the compact run_cfg strings happen to agree.  Presence is not
    # required for old panels (both-absent stays the legacy path), but a
    # present-one-sided or differing stamp fails the control.
    def _code_setting(r):
        _c = r.get('_code') if isinstance(r, dict) else None
        return None if _c is None or str(_c) == '' else _canon_setting(str(_c))

    code = _control(_code_setting)
    # Equality of two partial descriptors is not controlled-recipe evidence.
    # Collect every present-but-incomplete descriptor per arm so the panel
    # report can fail closed instead of treating it as a legacy omission.
    _knob_defects = {a: [] for a in arms}
    for (v, t, s), r in recs.items():
        if v not in _knob_defects or not isinstance(r, dict):
            continue
        _kn = r.get('train_knobs')
        if _kn is None:
            continue
        _why = training_knobs_problem(_kn)
        if _why is not None:
            _knob_defects[v].append({'protocol': str(t or ''), 'seed': s,
                                     'reason': _why})
    _knob_defects = {a: sorted(v, key=lambda x: (x['protocol'], str(x['seed'])))
                     for a, v in _knob_defects.items() if v}
    objective = _control(lambda r: _canon_setting(r.get('train_objective')))
    hyper = {}
    for key in _CONTROL_HYPER_KEYS:
        hyper[key] = _control(
            lambda r, k=key: _canon_setting((r.get('hyperparams') or {}).get(k)))
    hp_checked = [k for k, v in hyper.items()
                  if any(v['values'].get(a) for a in arms)]
    hp_diffs = [k for k in hp_checked if hyper[k]['equal'] is False]
    hp_unknown = [k for k in hp_checked if hyper[k]['equal'] is None]
    if not hp_checked:
        hp_equal = None
    elif hp_diffs:
        hp_equal = False
    elif hp_unknown:
        hp_equal = None
    else:
        hp_equal = True
    mixed = sorted(set(run_cfg['within_arm_mixed']) | set(train_knobs['within_arm_mixed'])
                   | set(objective['within_arm_mixed'])
                   | set(code['within_arm_mixed'])
                   | {a for k in hp_checked for a in hyper[k]['within_arm_mixed']})
    partial = sorted(set(run_cfg['partially_recorded']) | set(train_knobs['partially_recorded'])
                     | set(objective['partially_recorded'])
                     | set(code['partially_recorded'])
                     | {a for k in hp_checked for a in hyper[k]['partially_recorded']})
    diff_values = {}
    for key, ctl in (('run_cfg', run_cfg), ('train_knobs', train_knobs),
                     ('train_objective', objective), ('_code', code)):
        if ctl['equal'] is False:
            diff_values[key] = ctl['values']
    for key in hp_diffs:
        diff_values['hyperparams.' + key] = hyper[key]['values']
    # Seed/protocol coverage: arm means are only directly comparable when
    # every arm averaged the same seeds under the same measurement protocol.
    # Paired contrasts already use only shared seeds, but an unequal arm still
    # changes the descriptive mean±std and can hide a lost seed (crash / cap /
    # error).  Report the exact coverage; None means no arm has a record.
    _seed_sets = {}
    for (v, t, s) in recs:
        if v not in arms:
            continue
        _ss = _as_seed_int(s)
        if _ss is None:
            continue
        _seed_sets.setdefault(v, set()).add((str(t or ''), _ss))
    seed_coverage = {a: sorted(_seed_sets.get(a, set())) for a in arms}
    _nonempty = [frozenset(v) for v in seed_coverage.values() if v]
    if not _nonempty:
        seed_coverage_equal = None
    elif any(not v for v in seed_coverage.values()):
        seed_coverage_equal = None
    else:
        seed_coverage_equal = bool(len(set(_nonempty)) == 1)
    controlled = None
    flags = [run_cfg['equal'], train_knobs['equal'], hp_equal]
    if any(f is False for f in flags):
        controlled = False
    elif all(f is True for f in flags):
        controlled = True
    # A cross-arm objective difference is a system-level disclosure, not
    # necessarily an uncontrolled recipe (a declared loss-term ablation is the
    # intervention).  But an arm whose OWN seeds mix different objectives, or
    # carry the descriptor only on some seeds, cannot attest any single
    # objective for its arm mean.
    if (objective['equal'] is False
            and (objective['within_arm_mixed'] or objective['partially_recorded'])):
        controlled = False
    # Same rule for the code-semantics stamp: a differing or partially
    # recorded stamp means the arm's seeds were not all measured by one
    # implementation.  All-absent remains the legacy path (None, not a pass).
    if (code['equal'] is False
            or code['within_arm_mixed'] or code['partially_recorded']):
        controlled = False
    # A present-but-partial descriptor cannot attest any single recipe, even
    # when every record carries the same incomplete dictionary.
    if _knob_defects:
        controlled = False
    return {
        'arms': arms,
        'n_records': {a: sum(1 for (v, _t, _s) in recs if v == a) for a in arms},
        'run_cfg_equal': run_cfg['equal'],
        'train_knobs_equal': train_knobs['equal'],
        'train_knobs_defects': _knob_defects,
        'objective_equal': objective['equal'],
        'objective_values': objective['values'],
        'code_equal': code['equal'],
        'code_values': code['values'],
        'hyperparams_equal': hp_equal,
        'controlled': controlled,
        'within_arm_mixed': mixed,
        'partially_recorded': partial,
        'setting_coverage': {
            'run_cfg': {'n_records': run_cfg['n_records'], 'n_missing': run_cfg['n_missing']},
            'train_knobs': {'n_records': train_knobs['n_records'], 'n_missing': train_knobs['n_missing']},
            'objective': {'n_records': objective['n_records'], 'n_missing': objective['n_missing']},
            '_code': {'n_records': code['n_records'], 'n_missing': code['n_missing']},
            'hyperparams': {k: {'n_records': hyper[k]['n_records'], 'n_missing': hyper[k]['n_missing']}
                            for k in hp_checked},
        },
        'differences': diff_values,
        'checked_hyperparam_keys': sorted(hp_checked),
        'hyperparam_differences': hp_diffs,
        'unverifiable_hyperparam_keys': hp_unknown,
        'seed_coverage': seed_coverage,
        'seed_coverage_equal': seed_coverage_equal,
    }

def _params_of_arm(recs, arm):
    """The arm's parameter count, or None when it is not a single verified value.

    Every measurable record of the arm must expose the SAME exact positive
    count.  Returning the first record that happens to carry a matching value
    would let one consistent seed hide a corrupted/different record elsewhere
    in the same arm (the same fail-open shape as a one-sided breakdown).
    """
    vals = []
    for (v, _t, _s), r in recs.items():
        if v != arm:
            continue
        # Parameter counts are exact integers.  int(1.5) would silently turn a
        # corrupted count into a different, apparently valid one, and bool is
        # not a count.
        p = _as_seed_int(r.get('params')) if isinstance(r, dict) else None
        if p is None or p <= 0:
            return None
        vals.append(p)
    if not vals or len(set(vals)) != 1:
        return None
    return vals[0]

def _param_control(recs, a, b):
    """Relative parameter-count difference between two arms, and whether it is
    small enough to call them parameter-controlled (<=1%).  None means one or
    both records do not expose `params` (unverifiable, not a mismatch).

    This is only the TOTAL count.  `_param_control_detail` additionally
    compares the recorded `param_breakdown` (non-embedding / active), because
    two arms can match in total while one carries dead or reallocated capacity
    (the random-indexer control has ~2e5 dead projection parameters, and the
    parameter-matched dense arm reallocates MLP width).  The detail is
    reported, never used to silently pass a total mismatch.
    """
    pa, pb = _params_of_arm(recs, a), _params_of_arm(recs, b)
    if pa is None or pb is None:
        return (None, None)
    rel = abs(pa - pb) / float(max(pa, pb, 1))
    return (float(rel), bool(rel <= 0.01))

def param_pair_reason(ra, rb, *, tolerance=0.01, require_breakdown=False):
    """Fail-closed parameter control for a single arm pair.

    A missing/zero parameter count is UNVERIFIABLE, not a pass: the historical
    tuned-baseline rule caught `TypeError` from `int(None)` and silently
    continued, so a tuned contrast could be quoted without any capacity check.
    When both records carry `param_breakdown`, a >`tolerance` disagreement in
    the non-embedding or active count is also a failure (totals can match while
    MLP width is reallocated or indexer projections are dead).  Returns None
    when the pair passes, else a human-readable reason.
    """
    # A tolerance is a gate parameter: True -> 1.0 (100%) or a negative/NaN
    # value would silently disable the capacity check.
    if isinstance(tolerance, (bool, np.bool_)):
        raise ValueError(f'tolerance must be a finite non-negative number, got {tolerance!r}')
    try:
        tol = float(tolerance)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'tolerance must be a finite non-negative number, got {tolerance!r}') from None
    if not math.isfinite(tol) or tol < 0.0:
        raise ValueError(f'tolerance must be a finite non-negative number, got {tolerance!r}')
    pa, pb = _as_seed_int(ra.get('params')), _as_seed_int(rb.get('params'))
    if pa is None or pb is None or pa <= 0 or pb <= 0:
        return 'unverifiable params (at least one record omits a usable parameter count)'
    rel = abs(pa - pb) / float(max(pa, pb, 1))
    if rel > tol:
        return f'parameter mismatch ({pa} vs {pb}, {100.0 * rel:.2f}% > {100.0 * tol:.2f}%)'
    ba, bb = ra.get('param_breakdown'), rb.get('param_breakdown')
    _ba = isinstance(ba, dict) and bool(ba)
    _bb = isinstance(bb, dict) and bool(bb)
    if _ba != _bb:
        return 'unverifiable param_breakdown (only one record carries it)'
    if _ba and _bb:
        for key, label in (('non_embedding', 'non-embedding'), ('active', 'active')):
            va, vb = _finite_scalar(ba.get(key)), _finite_scalar(bb.get(key))
            if va is None or vb is None:
                if require_breakdown:
                    return f'unverifiable param_breakdown.{key}'
                continue
            _rel = abs(va - vb) / max(abs(va), abs(vb), 1.0)
            if _rel > tol:
                return (f'parameter breakdown mismatch ({label}: {va:g} vs {vb:g}, '
                        f'{100.0 * _rel:.2f}% > {100.0 * tol:.2f}%)')
    elif require_breakdown:
        return 'unverifiable param_breakdown (at least one record omits it)'
    return None

def _param_breakdown_of_arm(recs, arm):
    """One arm-level breakdown, or None if the arm's records disagree / omit it.

    A missing breakdown on any measurable record makes the capacity claim
    unverifiable; selecting the first non-empty dict would silently ignore the
    rest of the arm.
    """
    vals = []
    for (v, _t, _s), r in recs.items():
        if v != arm:
            continue
        pb = r.get('param_breakdown') if isinstance(r, dict) else None
        if not isinstance(pb, dict) or not pb:
            return None
        vals.append(pb)
    if not vals:
        return None
    first = vals[0]
    if any(pb != first for pb in vals[1:]):
        return None
    return first

def _param_control_detail(recs, a, b):
    """Capacity breakdown for a pair: total as well as non-embedding and
    active parameters.  Returns `{}` when neither record exposes the detail,
    so legacy records are not silently labelled controlled."""
    out = {'total_rel_diff': None, 'total_controlled': None,
           'non_embedding_rel_diff': None, 'non_embedding_controlled': None,
           'active_rel_diff': None, 'active_controlled': None,
           'breakdown_one_sided': False, 'unavailable': True}
    pa, pb = _params_of_arm(recs, a), _params_of_arm(recs, b)
    if pa is not None and pb is not None:
        rel = abs(pa - pb) / float(max(pa, pb, 1))
        out.update({'total_rel_diff': float(rel), 'total_controlled': bool(rel <= 0.01), 'unavailable': False})
    ba, bb = _param_breakdown_of_arm(recs, a), _param_breakdown_of_arm(recs, b)
    _ba = isinstance(ba, dict) and bool(ba)
    _bb = isinstance(bb, dict) and bool(bb)
    out['breakdown_one_sided'] = (_ba != _bb)
    if not (_ba and _bb):
        return out

    def _num(d, key):
        return _finite_scalar(d.get(key))
    for key, label in (('non_embedding', 'non_embedding'), ('active', 'active')):
        va, vb = _num(ba, key), _num(bb, key)
        if va is None or vb is None:
            continue
        rel = abs(va - vb) / max(va, vb, 1.0)
        out[f'{label}_rel_diff'] = float(rel)
        out[f'{label}_controlled'] = bool(rel <= 0.01)
    out['unavailable'] = False
    return out

def _param_capacity_controlled(detail):
    """Collapse `_param_control_detail` into one tri-state capacity verdict.

    Total parameter equality is necessary but not sufficient: the
    parameter-matched dense arm reallocates MLP width and the random-indexer
    control keeps dead indexer projections, so the *active* or
    *non-embedding* counts can differ while totals match.  A recorded
    disagreement on either breakdown key is a capacity confound (False);
    unrecorded breakdown keys fall back to the total verdict; no usable
    total is unverifiable (None), never a silent pass.
    """
    if not isinstance(detail, dict):
        return None
    if detail.get('breakdown_one_sided') is True:
        # One arm records the breakdown and the other does not: the capacity
        # match cannot be verified, so this is unverifiable (None), never a
        # pass that silently trusts the total count.
        return None
    total = detail.get('total_controlled')
    if total is None:
        return None
    if total is False:
        return False
    for key in ('non_embedding_controlled', 'active_controlled'):
        if detail.get(key) is False:
            return False
    return True

def _compute_control_detail(recs, a, b, *, tolerance=0.25):
    """Measured compute budget of a pair, not just its step count.

    Equal `steps`/`tokens_seen` is enforced by `pair_reason`, but that is not
    equal compute: the parameter-matched dense arm widens its MLP (+69%) and
    reads every token, while a sparse arm reads a fraction of the blocks.  The
    mismatch must be *shown*, and its DIRECTION matters: if the arm with more
    measured compute also wins, the
    comparison is confounded in favour of the claim; if it loses, the claim is
    conservative.

    Sources, in order of preference:
      * measured wall-clock per token (`train_time_s / tokens_seen`), median
        over the arm's seeds;
      * the 6ND proxy (`params * tokens_seen`) when no timing is recorded.
    Returns `{}`-shaped fields with None when neither record can support the
    comparison; never guesses.
    """
    def _series(arm):
        tpt, prox, par = [], [], []
        for (v, _t, _s), r in recs.items():
            if v != arm or not isinstance(r, dict):
                continue
            tt = _finite_scalar(r.get('train_time_s'))
            tok = _exact_count(r.get('tokens_seen'), minimum=1)
            if tt is not None and tt > 0.0 and tok is not None:
                tpt.append(tt / float(tok))
            p = _as_seed_int(r.get('params'))
            if p is not None and p > 0:
                par.append(float(p))
                if tok is not None:
                    prox.append(float(p) * float(tok))
        med = (lambda xs: float(np.median(xs)) if xs else None)
        return med(tpt), med(par), med(prox)

    # Same gate-parameter contract as param_pair_reason.
    if isinstance(tolerance, (bool, np.bool_)):
        raise ValueError(f'tolerance must be a finite non-negative number, got {tolerance!r}')
    try:
        tolerance = float(tolerance)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'tolerance must be a finite non-negative number, got {tolerance!r}') from None
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(f'tolerance must be a finite non-negative number, got {tolerance!r}')
    ta, pa, xa = _series(a)
    tb, pb, xb = _series(b)
    out = {'arm_a': a, 'arm_b': b, 'tolerance': float(tolerance),
           'time_per_token_s': {a: ta, b: tb},
           'params': {a: pa, b: pb},
           'flops_proxy_6nd': {a: xa, b: xb},
           'time_ratio_a_over_b': None, 'proxy_ratio_a_over_b': None,
           'compute_matched': None, 'compute_matched_source': None,
           'more_compute_arm': None, 'note': None}
    ratio = None
    if ta and tb and ta > 0 and tb > 0:
        ratio = ta / tb
        out['time_ratio_a_over_b'] = float(ratio)
        out['compute_matched_source'] = 'measured wall-clock per token'
    elif xa and xb and xa > 0 and xb > 0:
        ratio = xa / xb
        out['proxy_ratio_a_over_b'] = float(ratio)
        out['compute_matched_source'] = '6ND proxy (params x tokens); not a measurement'
    if ratio is None:
        out['note'] = 'no timing or 6ND inputs on the records: compute matching is unverifiable for this pair'
        return out
    out['compute_matched'] = bool(abs(ratio - 1.0) <= float(tolerance))
    out['more_compute_arm'] = (a if ratio > 1.0 else b)
    if not out['compute_matched']:
        out['note'] = (f'{out["more_compute_arm"]} consumed about {max(ratio, 1.0 / ratio):.2f}x the compute of the other arm '
                       f'({out["compute_matched_source"]}); this is NOT an equal-compute comparison')
    return out

def _family_size_with_missing(design_group, entries_by_comp):
    """Declared family size, counting unmeasured contrasts as p=1 slots.

    ``len(entries) + missing declarations`` is not ``max(n_declared, n_entries)``
    when one contrast is measured under several protocol tags and a *different*
    declared contrast is missing: the extra tags would otherwise fill the
    missing contrast's slot and shrink the correction family.  This helper is
    the single place that accounting is defined.
    """
    total_entries = 0
    missing = 0
    _design = design_group if isinstance(design_group, dict) else {}
    _entries = entries_by_comp if isinstance(entries_by_comp, dict) else {}
    for comp, contrasts in _design.items():
        entries = _entries.get(comp) or []
        total_entries += len(entries)
        measured_pairs = set()
        for e in entries:
            if not isinstance(e, dict):
                continue
            a, b = e.get('with'), e.get('without')
            try:
                measured_pairs.add((a, b))
            except TypeError:
                # An unhashable arm id cannot identify a measured pair; it
                # must not crash the family accounting.
                continue
        if isinstance(contrasts, (str, bytes, bytearray)):
            _contrast_items = [contrasts]
        else:
            try:
                _contrast_items = list(contrasts or ())
            except TypeError:
                _contrast_items = [contrasts]
        for c in _contrast_items:
            if not isinstance(c, (tuple, list)) or len(c) < 2:
                missing += 1
                continue
            try:
                _pair = (c[0], c[1])
                _measured = _pair in measured_pairs
            except TypeError:
                # A malformed (unhashable) declaration still occupies a
                # declared family slot.
                _measured = False
            if not _measured:
                missing += 1
    return int(total_entries + missing)


def _holm_assign(entries, m_family, alpha, out_key, *, p_key='p_exact_signflip'):
    """Attach a Holm step-down adjusted p to each entry over the declared
    family.  `m_family` is the DECLARED number of contrasts, not just the ones
    with data: a missing arm must not shrink the correction family (that would
    make an incomplete ablation look better-powered than it is).  Dummy p=1
    entries fill the difference.

    ``p_key`` selects the per-contrast p fed into the family correction.  The
    default keeps the historical naive final-look p; the look-adjusted pass
    passes ``p_look_adjusted`` so optional stopping cannot be corrected away
    by a family that never saw the extra looks."""
    named = []
    for i, e in enumerate(entries):
        p = e.get(p_key)
        if isinstance(p, (int, float)) and math.isfinite(float(p)):
            named.append((str(i), float(p)))
    m_eff = max(int(m_family), len(entries), len(named))
    for j in range(m_eff - len(named)):
        named.append((f'__missing_{j}', 1.0))
    h = holm_adjust(named, alpha=alpha)
    for i, e in enumerate(entries):
        rec = h.get(str(i))
        e[out_key] = rec.get('p_holm') if isinstance(rec, dict) else None

def _hypothesis_spec(comp, kind):
    table = HYPOTHESIS_DESIGN if kind == 'component' else BASELINE_HYPOTHESIS_DESIGN
    return table.get(comp)

def _sign_of(x):
    f = _finite_scalar(x)
    if f is None:
        return None
    if f > 0:
        return 1
    if f < 0:
        return -1
    return 0

def _classify_group_status(entries, *, min_n, alpha, m_family, require_param_match=False,
                           n_declared=None, declared_pairs=None):
    """One status per declared innovation / baseline claim.

    verified            — a controlled, capacity-controlled (total plus the
                          recorded non-embedding/active breakdown for
                          baselines) contrast is significant after the family-wise Holm
                          correction, its effect exceeds seed noise, AND (when
                          a pre-registered hypothesis exists) the sign matches
                          the predicted direction;
    direction_conflict  — a corrected significant effect exists but it points
                          against the pre-registered direction.  This is
                          evidence AGAINST the stated mechanism, never support;
    direction_unspecified — significant, but the hypothesis explicitly declared
                          predicted_sign=0; it cannot support a directional
                          advantage claim;
    system_level_only   — significant after correction, but every significant
                          contrast is confounded by a different training
                          objective, by a compute budget that favours the
                          claimed winner, or by a capacity mismatch against a
                          hypothesis that declared matched capacity
                          (`mechanism_claim_allowed=False`).  This is a real
                          performance difference, not evidence for the
                          isolated mechanism;
    measured            — a controlled contrast exists with n >= min_n, but no
                          corrected significance;
    equivalent          — no corrected significance, BUT every SESOI-declared
                          powered contrast is TOST-equivalent inside its
                          pre-registered margin after family adjustment.  This
                          is positive evidence of practical removability, not a
                          failed significance test;
    sub_sesoi           — corrected significance exists, but the effect is
                          smaller than the pre-registered SESOI: detectable,
                          not practically meaningful;
    practically_indeterminate — significant (or seed-driven) effect whose
                          practical verdict is neither clearly above the SESOI
                          nor equivalent; do not call it verified;
    resolution_limited  — a controlled, non-noise contrast exists whose
                          family-corrected exact-test p floor is still above
                          alpha at this n (add seeds; never read as a null);
    underpowered        — controlled contrasts exist but every one is n < min_n;
    noise_level         — powered contrasts exist but all effects are within
                          seed noise (|mean| <= std);
    unverified          — no controlled contrast (missing arm / config mismatch).
    """
    _min_n = _as_seed_int(min_n)
    if _min_n is None or _min_n < 0:
        raise ValueError(f'min_n must be a non-negative integer, got {min_n!r}')
    min_n = _min_n

    def _capacity_ok(e):
        pc = e.get('param_capacity_controlled')
        return pc if pc is not None else e.get('param_controlled')
    usable = [e for e in entries if e.get('controlled')
              and (not require_param_match or _capacity_ok(e) is True)]
    if not usable:
        return 'unverified'
    powered = [e for e in usable if int(e.get('n') or 0) >= int(min_n)]
    # The n floor is a precondition for every positive verdict: a liberal
    # alpha cannot turn an underpowered contrast into a 'verified' one.
    if not powered:
        return 'underpowered'
    sig = [e for e in powered if e.get('significant_global')]
    if sig:
        if any(e.get('direction_consistent') is False for e in sig):
            return 'direction_conflict'
        if any(e.get('predicted_sign') == 0 for e in sig):
            return 'direction_unspecified'
        # A significant contrast whose objective/compute confound favours the
        # claimed winner is a system-level difference.  `verified` is reserved
        # for contrasts that pass `mechanism_claim_allowed`, so the flag the
        # audit already computes finally has a consumer.
        if all(e.get('mechanism_claim_allowed') is False for e in sig):
            return 'system_level_only'
        # A statistically significant effect can still be smaller than the
        # pre-registered SESOI, or be dominated by one seed.  `verified`
        # requires the *practical* verdict `superior` whenever a SESOI was
        # declared; without a declared margin the historical behaviour is kept.
        #
        # Include declared-but-untranslatable margins (`margin_ppl is None`,
        # decision `not_declared`) in this gate.  Treating them as "no SESOI"
        # would let a malformed/zero-baseline margin silently restore the
        # statistical-only `verified` verdict, which is exactly the fail-open
        # path the declaration gate is meant to prevent.
        _prac = [e for e in sig
                 if (e.get('sesoi') or {}).get('declared')]
        # Only tighten when EVERY significant contrast has a SESOI declaration;
        # a mixed custom design keeps the historical statistical-only verdict
        # rather than gating on an arbitrary subset.
        if _prac and len(_prac) == len(sig):
            _dec = [e.get('practical_decision') for e in _prac]
            if any(d in (None, 'not_declared') for d in _dec):
                return 'practically_indeterminate'
            if _dec and all(d == 'sub_sesoi' for d in _dec):
                return 'sub_sesoi'
            if any(d == 'inferior' for d in _dec):
                return 'direction_conflict'
            if any(d != 'superior' for d in _dec):
                return 'practically_indeterminate'
        return 'verified'
    _sesoi_powered = [e for e in powered
                      if (e.get('sesoi') or {}).get('margin_ppl') is not None]
    # "The component is removable" is a component-level claim: it needs the
    # whole declared ablation, not one convenient contrast.  Equivalence on a
    # partial panel stays descriptive instead of becoming `equivalent`.
    #
    # Count DISTINCT (with, without) pairs, not entries: a contrast measured
    # under two protocol tags has two entries, and an entry-count check could
    # then hide a missing second contrast.  Underpowered pairs are excluded
    # from `powered`, so a component with one n=8 equivalent contrast and one
    # n=2 contrast can no longer certify the whole component as removable.
    _powered_pairs = {(e.get('with'), e.get('without')) for e in powered}
    if declared_pairs is not None:
        _declared_pairs = set(declared_pairs)
        _all_declared_powered = bool(_declared_pairs) and _declared_pairs.issubset(
            _powered_pairs)
    elif n_declared is None:
        _all_declared_powered = True
    else:
        _all_declared_powered = len(_powered_pairs) >= int(n_declared)
    if (_sesoi_powered and len(_sesoi_powered) == len(powered)
            and _all_declared_powered
            and all(e.get('practical_decision') == 'equivalent'
                    for e in _sesoi_powered)):
        return 'equivalent'
    noise_free = [e for e in powered if not e.get('within_seed_noise')]
    if not noise_free:
        return 'noise_level'
    for e in noise_free:
        # Exact ties reduce the attainable p resolution; use the non-zero
        # pair count when the stats record carries it.
        _en = _as_seed_int(e.get('n'))
        _enn = _as_seed_int(e.get('n_nonzero'))
        if _enn is not None and _en is not None and 0 <= _enn <= _en:
            _en = _enn if _enn > 0 else 1
        fl = holm_min_attainable_p(_en or 0, m_family, alpha)
        if math.isfinite(fl) and fl > alpha:
            return 'resolution_limited'
    return 'noise_level'

def _apply_sesoi_holm(component_entries, component_design, baseline_entries,
                      baseline_design, specs, *, alpha, min_n):
    """Family-adjust TOST p-values and refresh every practical decision.

    The SESOI family is the set of *declared* contrasts belonging to components
    with a pre-registered margin.  Missing/unmeasured contrasts occupy a dummy
    p=1 slot, exactly like the superiority family, so an incomplete panel
    cannot buy a smaller correction burden.  The two one-sided TOST p-values
    are summarised by their max, and Holm is applied to that conservative
    p-value — adequate for intersection-union equivalence tests.
    """
    groups = (('component', component_entries, component_design),
              ('baseline', baseline_entries, baseline_design))
    named, by_name = [], {}
    declared = 0
    missing_declared = 0
    for kind, entries_by_comp, design_group in groups:
        for comp, contrasts in (design_group or {}).items():
            if comp not in specs:
                continue
            if isinstance(contrasts, (str, bytes, bytearray)):
                _contrast_list = [contrasts]
            else:
                try:
                    _contrast_list = list(contrasts or ())
                except TypeError:
                    _contrast_list = []
            declared += len(_contrast_list)
            _declared_pairs = []
            for _c in _contrast_list:
                if isinstance(_c, (tuple, list)) and len(_c) >= 2:
                    _declared_pairs.append((_c[0], _c[1]))
                else:
                    # A structural defect must still occupy a family slot
                    # (fail closed); it can never be silently dropped.
                    _declared_pairs.append(None)
            _measured_pairs = set()
            for j, e in enumerate((entries_by_comp or {}).get(comp) or []):
                s = e.get('sesoi') or {}
                if s.get('margin_ppl') is None:
                    continue
                name = f'{kind}::{comp}::{j}'
                _p_tost = s.get('tost_p_tost')
                _k_looks = int((e.get('look_analysis') or {}).get('n_looks') or 1)
                if (isinstance(_p_tost, (int, float)) and math.isfinite(float(_p_tost))
                        and _k_looks > 1):
                    # Equivalence is a positive claim and is equally exposed
                    # to optional stopping: a TOST run after each seed top-up
                    # can be "equivalent" at the look that happened to be
                    # tight.  Bonferroni over looks is valid and conservative.
                    _p_tost = min(1.0, float(_p_tost) * _k_looks)
                    s['p_tost_look_adjusted'] = _p_tost
                    s['n_looks'] = _k_looks
                named.append((name, _p_tost))
                by_name[name] = e
                _measured_pairs.add((e.get('with'), e.get('without')))
            # Every declared contrast that produced no TOST gets a dummy p=1
            # slot.  The count is per DECLARATION, not per entry: one
            # contrast measured under two protocol tags must not consume the
            # slot of a different, entirely unmeasured contrast.
            missing_declared += sum(
                1 for _p in _declared_pairs
                if _p is None or _p not in _measured_pairs)
    if not named:
        # No TOST is available, but the declared family size is still real:
        # reporting m=0 would falsely suggest that a partial panel has no
        # correction burden.
        m = max(int(declared), int(missing_declared))
        return {'m': int(m), 'declared_contrasts': int(declared), 'n_measured': 0,
                'min_attainable_p_holm': None,
                'note': ('no measured contrast carries a pre-registered SESOI; '
                         'declared contrasts keep their p=1 family slots')}
    m = int(len(named) + missing_declared)
    padded = list(named)
    padded.extend((f'__sesoi_declared_missing_{i}', 1.0)
                  for i in range(m - len(named)))
    holm = holm_adjust(padded, alpha=alpha)
    for name, e in by_name.items():
        s = e['sesoi']
        ph = (holm.get(name) or {}).get('p_holm')
        if ph is None:
            # An invalid/unavailable TOST p cannot certify equivalence; keep
            # the family slot at p=1 (fail closed) rather than skipping the
            # correction and letting a raw TOST pass.
            ph = 1.0
        stats2 = {
            'n': s.get('n'), 'mean': s.get('mean'),
            # Prefer the family-corrected verdict attached by the caller.
            'significant': (e.get('significant_global')
                            if e.get('significant_global') is not None
                            else s.get('statistically_significant')),
            'seed_driven': s.get('seed_driven'),
        }
        tost2 = {
            'p_tost': s.get('tost_p_tost'),
            'equivalent': s.get('tost_equivalent'),
            'ci': s.get('tost_ci'),
        }
        _min_for_pd = _as_seed_int(s.get('min_n'))
        if _min_for_pd is None:
            _min_for_pd = min_n
        _dir_for_pd = _as_seed_int(s.get('direction', -1))
        if _dir_for_pd not in (-1, 0, 1):
            _dir_for_pd = -1
        new = _practical_decision_from_stats(
            stats2, tost2, margin=s['margin'],
            direction=_dir_for_pd, alpha=alpha,
            min_n=_min_for_pd, p_holm_tost=ph)
        new['declared'] = True
        new['source'] = s.get('source')
        # Preserve the descriptive fields from the per-contrast decision; the
        # refreshed dict is the same object's replacement, not a lossy cast.
        for _k in ('label', 'baseline_ppl', 'relative_margin_pct',
                   'p_tost_look_adjusted', 'n_looks'):
            if _k in s:
                new[_k] = s[_k]
        new['p_floor'] = s.get('p_floor')
        new['p_floor_scope'] = s.get('p_floor_scope')
        new['n_nonzero'] = s.get('n_nonzero')
        new['statistically_significant_raw'] = s.get('statistically_significant')
        new['p_holm_global'] = e.get('p_holm_global')
        e['sesoi'] = new
        e['practical_decision'] = new['decision']
    return {
        'm': int(m),
        'declared_contrasts': int(declared),
        'n_measured': len(named),
        # TOST is a t-approximation: unlike the exact sign-flip test it has no
        # 2/2**n resolution floor, so reporting a sign-flip floor here would
        # wrongly suggest that equivalence is impossible at small n.  The
        # superiority floor remains in `report['family']`.
        'min_attainable_p_holm': None,
        'method': ('Holm step-down over the declared SESOI contrasts; TOST p is '
                   'max(one-sided p), missing declared contrasts keep p=1; '
                   'equivalence has no exact-sign-flip resolution floor'),
    }

def _materialise_contrast_design(design):
    """Return a design whose per-component contrast iterables are re-iterable.

    A caller may pass a one-pass generator of contrasts.  The audit currently
    walks each declaration twice (collection, then family scoping), and an
    exhausted iterator would silently drop every contrast on the second pass
    (or crash `len`).  Materialising once is value-preserving and makes both
    passes see the same declarations.  Non-iterable/malformed entries are
    kept as-is so the caller can report them rather than silently replacing
    them with an empty declaration.
    """
    if not isinstance(design, dict):
        return design
    out = {}
    for comp, contrasts in design.items():
        if isinstance(contrasts, (list, tuple, str, bytes, bytearray)):
            # Strings are a single malformed declaration, not an iterable of
            # per-character contrasts; preserving them lets the validators
            # count one bad slot instead of len(text) fake ones.
            out[comp] = contrasts
            continue
        try:
            out[comp] = list(contrasts)
        except TypeError:
            out[comp] = contrasts
    return out


def _declared_contrast_count(design):
    """Number of declared contrast slots, tolerating malformed declarations.

    A malformed/non-iterable declaration is still one declared slot (fail
    closed): dropping it would understate the correction family and make the
    plan look easier to confirm than it is.
    """
    if not isinstance(design, dict):
        return 0
    total = 0
    for contrasts in design.values():
        if isinstance(contrasts, (str, bytes, bytearray)):
            total += 1
            continue
        if isinstance(contrasts, (list, tuple)):
            total += len(contrasts)
            continue
        try:
            total += len(list(contrasts))
        except TypeError:
            total += 1
    return int(total)


def _scope_design(design, present):
    """Keep only claims this panel actually tests: a component enters the
    family when at least one of its declared contrasts has BOTH arms measured.
    Once in, all of its declared contrasts keep their slots (missing arms are
    dummies), so an incomplete ablation cannot shrink its own correction.

    A malformed design (non-mapping, non-list contrasts, short/unhashable
    entries) is tolerated here: callers report it separately, while this
    scoping pass must never crash the whole audit or fabricate a pair.
    """
    out = {}
    present = set(present or ())
    if not isinstance(design, dict):
        return out
    for comp, contrasts in design.items():
        if isinstance(contrasts, (str, bytes, bytearray)):
            continue
        try:
            _items = list(contrasts or ())
        except TypeError:
            continue
        _tested = False
        for item in _items:
            if not isinstance(item, (tuple, list)) or len(item) < 2:
                continue
            try:
                if item[0] in present and item[1] in present:
                    _tested = True
                    break
            except TypeError:
                # Unhashable arm id: not a testable contrast, but the
                # completeness/family audits still see the declaration.
                continue
        if _tested:
            # Store the materialised list: a one-pass iterable would already
            # be exhausted by the membership scan above, and downstream
            # readers need a re-iterable declaration.
            out[comp] = _items
    return out

def ablation_completeness_audit(design=None, present=None, *, arms_by_combo=None,
                                verbose=False):
    """Zero-GPU completeness check for a leave-one-out OR full-factorial design.

    Leave-one-out mode (default)
        `design` is the `ABLATION_DESIGN` shape: component -> list of
        (with_arm, without_arm, note).  A component is COMPLETE only when
        **every** declared contrast has both arms in `present`.  `present=None`
        checks the design's structure without requiring the arms to be run.
        This is the condition for the claim "each module is useful": a
        component measured only through a combined arm (e.g. A+B vs baseline)
        stays incomplete, and the missing arm is named instead of being
        omitted from the correction family.

    Full-factorial mode (`arms_by_combo` given)
        The mapping keys are component-name tuples (or '+'-joined strings,
        with ''/'none' for the empty combination) and values are arm ids.
        ALL 2**k combinations must be mapped, present and unique; a 7-of-8
        panel is reported with its exact missing combination (e.g. C alone)
        rather than being described as a full factorial.  This is the
        larger design the task brief allows as an alternative to
        leave-one-out.

    Returns a machine-readable dict; never raises on a partial panel.
    """
    out = {'mode': 'leave-one-out', 'complete': False,
           'complete_components': [], 'incomplete_components': [],
           'components': {}, 'missing_arms': {}, 'structural_errors': []}
    present_set = None if present is None else set(present)

    if arms_by_combo is not None:
        out.update({'mode': 'full-factorial', 'components': [],
                    'n_required': 0, 'n_mapped': 0,
                    'missing_combos': [], 'duplicate_combos': [],
                    'reused_arms': {}, 'missing_arms': [], 'arms': {}})
        combos = {}
        components = set()
        _empty = ('none', 'empty', 'baseline', '0')
        for key, arm in arms_by_combo.items():
            # Normalise the combination key.  A malformed key/arm must fail
            # closed (complete=False) instead of crashing or being silently
            # dropped, because a dropped cell would make a 7-of-8 panel look
            # complete.
            try:
                if isinstance(key, str):
                    _raw = [x.strip() for x in re.split(r'[+,|]', key) if x.strip()]
                elif isinstance(key, (tuple, list, set, frozenset)):
                    _raw = [str(x).strip() for x in key if str(x).strip()]
                elif key is None:
                    _raw = []
                else:
                    raise TypeError(f'unsupported combination key {key!r}')
                combo = tuple(sorted(x for x in _raw if x.lower() not in _empty))
            except (TypeError, ValueError) as e:
                out['structural_errors'].append(f'malformed combination key {key!r}: {e}')
                continue
            # One cell = one arm id.  An iterable is accepted only as a list
            # of ids; every id must be a non-empty string.  Two ids for one
            # cell (or the same id twice) is ambiguous, not a full factorial.
            _arms_raw = (list(arm) if isinstance(arm, (tuple, list, set, frozenset))
                         else [arm])
            if not _arms_raw:
                out['structural_errors'].append(
                    f'combination {combo or "(none)"} maps to no arm id')
                continue
            _ok_arms = True
            for _a in _arms_raw:
                if not isinstance(_a, str) or not _a.strip():
                    out['structural_errors'].append(
                        f'combination {combo or "(none)"} has an invalid arm id {_a!r}')
                    _ok_arms = False
                    break
                combos.setdefault(combo, []).append(_a.strip())
            if not _ok_arms:
                combos.pop(combo, None)
                continue
            components.update(combo)
        components = sorted(components)
        required = []
        for r in range(len(components) + 1):
            required.extend(tuple(c) for c in itertools.combinations(components, r))
        required_set = set(required)
        mapped_set = set(combos)
        missing = sorted(required_set - mapped_set)
        extra = sorted(mapped_set - required_set)
        duplicate = sorted(c for c, arms in combos.items() if len(arms) > 1)
        absent = []
        # In a full factorial every combination must be its OWN arm.  Reusing
        # one arm id for two combinations (e.g. mapping A and B to the same
        # run) silently makes the 2**k table impossible to attribute, so it is
        # a structural failure rather than a complete design.
        _arm_combos = {}
        for combo, arms in combos.items():
            for arm in set(arms):
                _arm_combos.setdefault(arm, set()).add(combo)
        reused = {arm: sorted(cs) for arm, cs in _arm_combos.items() if len(cs) > 1}
        for combo, arms in combos.items():
            for arm in set(arms):
                if present_set is not None and arm not in present_set:
                    absent.append({'combo': list(combo), 'arm': arm})
        out.update({
            'components': components,
            'n_required': len(required),
            'n_mapped': len(combos),
            'missing_combos': [list(c) for c in missing],
            'extra_combos': [list(c) for c in extra],
            'duplicate_combos': [list(c) for c in duplicate],
            'reused_arms': {str(a): [list(c) for c in cs] for a, cs in reused.items()},
            'missing_arms': absent,
            'arms': {('+'.join(c) if c else 'none'): sorted(set(a)) for c, a in combos.items()},
        })
        out['complete'] = bool(len(components) >= 1 and not missing and not extra
                               and not duplicate and not absent and not reused
                               and not out['structural_errors'])
        # Project-level coverage reads the same component lists in both modes.
        # A factorial is only a usable ablation as a whole: one missing cell
        # means no component's effect is separately identified.
        out['complete_components'] = list(components) if out['complete'] else []
        out['incomplete_components'] = [] if out['complete'] else list(components)
        if verbose:
            print('[ablation] full-factorial components=%s mapped=%d/%d missing=%s%s'
                  % (components, len(combos), len(required),
                     ['+'.join(c) or 'none' for c in missing] or 'none',
                     (' reused_arms=%s' % (sorted(reused) or 'none')) if reused else ''))
        return out

    if design is None:
        design = ABLATION_DESIGN
    if not isinstance(design, dict):
        out['structural_errors'].append(f'design must be a mapping, got {type(design).__name__}')
        design = {}
    for comp, contrasts in design.items():
        rows, missing = [], set()
        if not isinstance(contrasts, (tuple, list)):
            out['structural_errors'].append(
                '%s: contrasts must be a list of (with, without, note) tuples, got %r'
                % (comp, contrasts))
            out['components'][comp] = {'declared': 0, 'complete': False,
                                       'contrasts': [], 'missing_arms': []}
            out['incomplete_components'].append(comp)
            continue
        for item in contrasts:
            if not isinstance(item, (tuple, list)) or len(item) < 2:
                out['structural_errors'].append('%s: malformed contrast %r' % (comp, item))
                continue
            a, b = item[0], item[1]
            # Arm ids are non-empty strings; a contrast of an arm with itself
            # cannot measure the module, and an unhashable/empty id would make
            # the presence check meaningless (or crash).
            _bad_arm = [x for x in (a, b)
                        if not isinstance(x, str) or not x.strip()]
            if _bad_arm:
                out['structural_errors'].append(
                    '%s: invalid arm id in contrast %r' % (comp, item))
                continue
            if a == b:
                out['structural_errors'].append(
                    '%s: contrast %r compares an arm with itself' % (comp, item))
                continue
            ap = present_set is None or a in present_set
            bp = present_set is None or b in present_set
            rows.append({'with': a, 'without': b,
                         'note': item[2] if len(item) > 2 else '',
                         'with_present': ap, 'without_present': bp,
                         'complete': bool(ap and bp)})
            if not ap:
                missing.add(a)
            if not bp:
                missing.add(b)
        complete = bool(rows) and all(r['complete'] for r in rows)
        out['components'][comp] = {'declared': len(rows), 'complete': complete,
                                   'contrasts': rows,
                                   'missing_arms': sorted(missing)}
        if missing:
            out['missing_arms'][comp] = sorted(missing)
        (out['complete_components'] if complete else out['incomplete_components']).append(comp)
    out['complete'] = bool(out['components']) and not out['incomplete_components'] \
        and not out['structural_errors']
    if verbose:
        print('[ablation] leave-one-out complete=%s; missing: %s'
              % (out['complete'],
                 '; '.join('%s=%s' % (c, ','.join(a)) for c, a in sorted(out['missing_arms'].items())) or 'none'))
    return out
