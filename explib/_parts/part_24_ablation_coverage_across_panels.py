
def ablation_coverage_across_panels(reports, design=None, *, verbose=False):
    """Project-level view of ablation completeness.

    A single-panel audit answers "is this innovation complete HERE?".  The
    claim "each proposed module is useful" is a project-level claim: a component
    that is complete and verified in one panel is covered, while a component
    that no panel ever measures is NOT covered even if every individual panel
    looks internally consistent (because it is simply absent from each scoped
    family).  `reports` maps panel name -> `methodology_audit` report; the
    returned dict names, per declared component, the panels where it is
    complete / usable / measured / verified / unverified, its missing arms,
    and the components with no complete panel anywhere.  "Usable" additionally
    requires a controlled contrast with the required shared seeds (n>=min_n):
    arms can all be present yet never pair on a common seed, which is not a
    measured ablation even though the presence check passes.

    A component is project-level VERIFIED only when a panel both verifies the
    contrast and contains the component's complete declared ablation.  A panel
    that verified one contrast while another declared arm is missing can still
    prove that particular contrast, but it cannot support "the whole innovation
    is useful"; such panels are listed separately as
    `panels_verified_incomplete` instead of being silently promoted.

    This is a reporting gate, not a new statistical test: it never promotes a
    component; it only prevents a panel-by-panel reading from hiding an
    omitted innovation.

    Verification in one panel can still be contradicted elsewhere.  A
    correction-significant opposite direction (or a practically harmful
    removal contrast) in ANY reported panel closes the project-level
    "useful" gate; such panels are named in
    `direction_conflict_anywhere` / `practically_harmful_anywhere`.
    `replication_status` distinguishes `replicated` (verified in >=2 complete
    panels), `single_panel`, `contradicted`, and `not_verified`.  With the
    sampling identity written by :func:`methodology_audit`, the replicate
    label is further split: `replicated` requires at least one pair with
    disjoint training seeds AND at least one recorded eval/holdout fingerprint
    that is disjoint (with no recorded evaluation fingerprint overlapping);
    disjoint seeds on a shared evaluation slice are `replicated_new_seeds`; shared
    seeds are `replicated_shared_seeds` (repeated measurement, not independent
    replication); a mix with missing identities and no stronger pair is
    `replicated_unattested`.  Per-pair detail remains in
    `replication_independence`.
    These labels never change the p values or the project-level usefulness
    gate; they only stop repeated draws from being reported as independent
    replication.  A powered but non-significant opposite mean is reported as a
    warning, not counted as a corrected contradiction."""
    design = ABLATION_DESIGN if design is None else design
    out = {'status': 'ok', 'components': {}, 'complete_anywhere': [],
           'usable_anywhere': [], 'verified_anywhere': [], 'measured_anywhere': [],
           'missing_everywhere': [], 'complete_but_unverified': [],
           'verified_only_incomplete': [], 'replicated_anywhere': [],
           'independently_replicated_anywhere': [],
           'new_seed_replicated_anywhere': [],
           'shared_seed_replicated_anywhere': [],
           'unattested_replicated_anywhere': [],
           'single_panel_verified': [], 'direction_conflict_anywhere': [],
           'practically_harmful_anywhere': [], 'opposite_mean_anywhere': [],
           'warnings': []}
    if not isinstance(reports, dict):
        out['status'] = 'warn'
        out['warnings'].append('reports must be a {panel: methodology_audit} mapping')
        return out
    for comp in design:
        row = {'panels_complete': [], 'panels_measured': [],
               'panels_verified': [], 'panels_verified_raw': [],
               'panels_verified_incomplete': [], 'panels_unverified': [],
               'panels_measured_unverified': [],
               'panels_direction_conflict': [],
               'panels_practically_harmful': [],
               'panels_opposite_mean': [], 'panels_replicated': [],
               'panels_independently_replicated': [],
               'panels_new_seed_replicated': [],
               'panels_shared_seed_replicated': [],
               'replication_independence': {},
               'replication_status': 'not_measured',
               'missing_arms': {}, 'declared_in': []}
        for name in sorted(reports):
            rep = reports.get(name)
            if not isinstance(rep, dict):
                continue
            abl = rep.get('ablation_completeness') or {}
            if comp in (abl.get('components') or {}):
                row['declared_in'].append(name)
                _miss_all = abl.get('missing_arms') or {}
                _miss = None
                if isinstance(_miss_all, dict):
                    _miss = _miss_all.get(comp)
                elif isinstance(_miss_all, list):
                    # Full-factorial reports name the missing CELLS, not a
                    # per-component arm list.  Attribute each missing cell to
                    # every component it contains.
                    _miss = [f"{'+'.join(_m.get('combo') or []) or 'none'}:"
                             f"{_m.get('arm')!r}"
                             for _m in _miss_all if isinstance(_m, dict)
                             and comp in (_m.get('combo') or [])]
                if not _miss:
                    # A mapped-but-absent cell is not the only failure: an
                    # unmapped combination is reported in `missing_combos`.
                    _miss = [f"missing combination {'+'.join(_c) or 'none'}"
                             for _c in (abl.get('missing_combos') or [])
                             if isinstance(_c, (list, tuple)) and comp in _c]
                if _miss:
                    row['missing_arms'][name] = sorted(_miss)
            if comp in (abl.get('complete_components') or []):
                row['panels_complete'].append(name)
            if comp in (rep.get('measured_components') or []):
                row['panels_measured'].append(name)
            if comp in (rep.get('verified_components') or []):
                row['panels_verified_raw'].append(name)
                # A partial ablation can verify one contrast but cannot back
                # the project-level "every proposed module is useful" claim.
                if comp in (abl.get('complete_components') or []):
                    row['panels_verified'].append(name)
                else:
                    row['panels_verified_incomplete'].append(name)
            if comp in (rep.get('unverified_components') or []):
                row['panels_unverified'].append(name)
            # A single-panel result is not automatically replicated.  More
            # importantly, a panel that significantly points the OTHER way is
            # contradictory evidence, not an inconvenient panel to omit from
            # the project-level read.
            _conflicts = rep.get('direction_conflict_components') or []
            if isinstance(_conflicts, (list, tuple, set)) and comp in _conflicts:
                row['panels_direction_conflict'].append(name)
            _harmful = rep.get('practically_harmful_components') or []
            if isinstance(_harmful, (list, tuple, set)) and comp in _harmful:
                row['panels_practically_harmful'].append(name)
            _comp_map = rep.get('components')
            _entries = (_comp_map.get(comp)
                        if isinstance(_comp_map, dict) else None)
            if isinstance(_entries, (list, tuple)):
                _nmin = _as_seed_int(rep.get('min_n')) or 0
                for _e in _entries:
                    if not isinstance(_e, dict) or not _e.get('controlled'):
                        continue
                    _n = _as_seed_int(_e.get('n'))
                    if _n is None or _n < _nmin:
                        continue
                    if (_e.get('direction_consistent') is False
                            and _e.get('observed_sign') in (-1, 1)):
                        row['panels_opposite_mean'].append(name)
                        break
        row['panels_usable'] = sorted(set(row['panels_complete']) & set(row['panels_measured']))
        row['panels_measured_unverified'] = sorted(
            set(row['panels_measured']) - set(row['panels_verified_raw']))
        _verified_complete = sorted(set(row['panels_verified']))
        row['panels_replicated'] = (_verified_complete
                                    if len(_verified_complete) >= 2 else [])
        # Sampling identity: only panels that actually verified the component
        # can be replication evidence.  Missing identities are intentionally
        # kept in the audit as unattested rather than dropped, so two newly
        # identified panels cannot hide a third unlabelled duplicate panel.
        _draws, _draws_present = {}, False
        for _name in _verified_complete:
            _rp = reports.get(_name)
            _cd = ((_rp.get('component_draws') or {})
                   if isinstance(_rp, dict) else {})
            _draw = _cd.get(comp) if isinstance(_cd, dict) else None
            if isinstance(_draw, dict):
                _draws[_name] = _draw
                _draws_present = True
            else:
                _draws[_name] = {}
        _ind = replication_independence_audit(_draws)
        row['replication_independence'] = _ind
        if _ind.get('status') == 'independent_replication':
            row['panels_independently_replicated'] = list(
                _ind.get('panels_independent') or [])
        elif _ind.get('status') == 'new_seeds_only':
            row['panels_new_seed_replicated'] = list(
                _ind.get('panels_new_seeds') or [])
        elif _ind.get('status') == 'shared_training_draws':
            row['panels_shared_seed_replicated'] = list(
                _ind.get('panels_shared_seeds') or [])
        row['complete_anywhere'] = bool(row['panels_complete'])
        row['usable_anywhere'] = bool(row['panels_usable'])
        row['verified_anywhere'] = bool(row['panels_verified'])
        row['measured_anywhere'] = bool(row['panels_measured'])
        _contradicted = bool(row['panels_direction_conflict']
                             or row['panels_practically_harmful'])
        _repl_status = None
        if len(row['panels_replicated']) >= 2:
            _tier = _ind.get('status')
            if _tier == 'independent_replication':
                _repl_status = 'replicated'
            elif _tier == 'new_seeds_only':
                _repl_status = 'replicated_new_seeds'
            elif _tier == 'shared_training_draws':
                _repl_status = 'replicated_shared_seeds'
            elif _draws_present:
                _repl_status = 'replicated_unattested'
            else:
                # Legacy reports predate component_draws.  Keep the historical
                # label rather than retroactively calling them independent.
                _repl_status = 'replicated'
        row['replication_status'] = (
            'contradicted' if _contradicted else
            _repl_status if _repl_status else
            'single_panel' if row['panels_verified'] else
            'not_verified' if row['panels_measured'] else 'not_measured')
        out['components'][comp] = row
        if row['complete_anywhere']:
            out['complete_anywhere'].append(comp)
            if not row['usable_anywhere']:
                out['warnings'].append(
                    '%s: all declared arms are present in %s, but no panel has a controlled contrast with '
                    'the required shared seeds (n>=min_n) — arm presence alone is not a usable ablation'
                    % (comp, ', '.join(row['panels_complete'])))
        else:
            out['missing_everywhere'].append(comp)
            out['warnings'].append(
                '%s: no reported panel contains every declared leave-one-out arm%s — '
                'this innovation cannot support a "useful" claim at the project level'
                % (comp, (' (declared in %s but incomplete there)' % ', '.join(sorted(row['declared_in']))
                          if row['declared_in'] else ' (never declared in a measured panel)')))
        if row['verified_anywhere']:
            out['verified_anywhere'].append(comp)
        elif row['complete_anywhere']:
            out['complete_but_unverified'].append(comp)
        if row['panels_verified_incomplete']:
            out['verified_only_incomplete'].append(comp)
            out['warnings'].append(
                '%s: verified in %s only while its declared ablation is '
                'incomplete there — that is evidence for the measured '
                'contrast(s), not for the whole innovation; complete the '
                'missing arm(s) before calling this component verified'
                % (comp, ', '.join(row['panels_verified_incomplete'])))
        if row['panels_replicated']:
            out['replicated_anywhere'].append(comp)
        elif row['panels_verified']:
            out['single_panel_verified'].append(comp)
        if row['panels_independently_replicated']:
            out['independently_replicated_anywhere'].append(comp)
        elif row['panels_new_seed_replicated']:
            out['new_seed_replicated_anywhere'].append(comp)
            out['warnings'].append(
                '%s: verified in %s with disjoint training seeds, but the '
                'evaluation slice is shared or not attested as disjoint — '
                'report a new-seed replication, not an independent evaluation'
                % (comp, ', '.join(row['panels_new_seed_replicated'])))
        elif row['panels_shared_seed_replicated']:
            out['shared_seed_replicated_anywhere'].append(comp)
            out['warnings'].append(
                '%s: verified in %s but those panels share training seeds; '
                'this is repeated measurement of the same draws, not '
                'independent replication'
                % (comp, ', '.join(row['panels_shared_seed_replicated'])))
        elif row['replication_status'] == 'replicated_unattested':
            out['unattested_replicated_anywhere'].append(comp)
            out['warnings'].append(
                '%s: verified in >=2 complete panels, but at least one panel '
                'lacks a usable draw identity; replication independence is '
                'unattested' % comp)
        if row['panels_direction_conflict']:
            out['direction_conflict_anywhere'].append(comp)
            out['warnings'].append(
                '%s: the declared direction is contradicted by a correction-'
                'significant result in %s — a result in another panel is not '
                'a replication, it is evidence against a project-level '
                '"useful" claim'
                % (comp, ', '.join(row['panels_direction_conflict'])))
        if row['panels_practically_harmful']:
            out['practically_harmful_anywhere'].append(comp)
            if not row['panels_direction_conflict']:
                out['warnings'].append(
                    '%s: removal is practically harmful in %s (effect beyond '
                    'the pre-registered SESOI); do not average this panel away'
                    % (comp, ', '.join(row['panels_practically_harmful'])))
        if (row['panels_opposite_mean'] and not row['panels_direction_conflict']
                and not row['panels_practically_harmful']):
            out['opposite_mean_anywhere'].append(comp)
            out['warnings'].append(
                '%s: a complete, powered contrast has the opposite mean '
                'direction in %s; this is not a corrected contradiction, but '
                'it weakens any single-panel replication claim'
                % (comp, ', '.join(row['panels_opposite_mean'])))
        if row['usable_anywhere']:
            out['usable_anywhere'].append(comp)
        if row['measured_anywhere']:
            out['measured_anywhere'].append(comp)
    # One project-level answer to the claim the ablation exists to support:
    # "every proposed innovation is useful".  A component is not counted if
    # any panel measured it only through a partial ablation, and a component
    # that was never verified anywhere keeps the claim closed.
    _needed = set(design) if isinstance(design, dict) else set()
    _unverified = sorted(c for c in _needed if c not in out['verified_anywhere'])
    _reasons = []
    if not _needed:
        _reasons.append('the design declares no components')
    if _unverified:
        _reasons.append('component(s) not verified by a complete ablation: '
                        + ', '.join(_unverified))
    if out['verified_only_incomplete']:
        _reasons.append('verified only through an incomplete ablation: '
                        + ', '.join(sorted(set(out['verified_only_incomplete']))))
    if out['direction_conflict_anywhere'] or out['practically_harmful_anywhere']:
        _reasons.append('contradicted by another panel: '
                        + ', '.join(sorted(set(out['direction_conflict_anywhere'])
                                           | set(out['practically_harmful_anywhere']))))
    out['all_components_useful_allowed'] = bool(_needed) and not _reasons
    out['all_components_useful_reasons'] = _reasons
    out['unverified_components'] = _unverified
    out['status'] = 'warn' if out['warnings'] else 'ok'
    if verbose:
        print('[ablation-coverage] complete in >=1 panel: %s; missing everywhere: %s'
              % (', '.join(out['complete_anywhere']) or 'none',
                 ', '.join(out['missing_everywhere']) or 'none'))
    return out

_COMPONENT_DRAW_FP_FIELDS = ('val_fp', 'eval_fp', 'holdout_row_fp',
                             'data_fp', 'corpus_fp')


def _draw_value(v):
    """Stable string identity for a draw/fingerprint field.

    A string is already an identity and must not be JSON-quoted a second time
    when an exported report is re-audited; structured values are canonicalised
    with sorted keys so two equivalent mappings compare equal.
    """
    if isinstance(v, str):
        return v.strip()
    return _canon_setting(v)


def _identity_values(v):
    """Iterator for an identity field, treating a scalar as one value.

    Exported/older reports may carry a bare string or integer where the
    current writer emits a list.  Iterating a bare string would compare its
    characters (so "seed-1" and "seed-2" could share the character "e"),
    which is exactly the silent false match this gate must avoid.
    """
    if v is None:
        return ()
    if isinstance(v, (list, tuple, set, frozenset)):
        return v
    return (v,)


def component_draw_identity(recs, arms):
    """Describe the sampling draws behind one component's arm records.

    A cross-panel "replicated" label is only meaningful if the panels are not
    just the same training seeds measured twice.  `methodology_audit` stores
    this compact identity next to each component so the project-level gate can
    distinguish independent draws from repeated measurements.  Fingerprints
    are canonicalised, missing values stay missing (never an empty-string
    match), and malformed record keys are ignored instead of crashing.

    The identity is descriptive: it does not decide significance, does not
    compare arms, and does not read any result file.
    """
    if not isinstance(recs, dict):
        recs, arms = {}, []
    if isinstance(arms, str):
        arms = [arms]
    arm_set = {str(a) for a in (arms or ()) if isinstance(a, str) and a.strip()}
    seeds, fps, codes, knobs = set(), {k: set() for k in _COMPONENT_DRAW_FP_FIELDS}, set(), set()
    n_records = 0
    for key, rec in recs.items():
        if not isinstance(key, (tuple, list)) or len(key) != 3:
            continue
        variant = key[0]
        if variant not in arm_set:
            continue
        n_records += 1
        if not isinstance(rec, dict):
            continue
        seed = _as_seed_int(key[2])
        if seed is not None:
            seeds.add(seed)
        for field in _COMPONENT_DRAW_FP_FIELDS:
            val = rec.get(field)
            if val is None or (isinstance(val, str) and not val.strip()):
                continue
            fps[field].add(_draw_value(val))
        code = rec.get('_code')
        if code is not None and not (isinstance(code, str) and not code.strip()):
            codes.add(_draw_value(code))
        kn = rec.get('train_knobs')
        if kn is not None:
            knobs.add(_draw_value(kn))
    out = {'arms': sorted(arm_set), 'n_records': int(n_records),
           'seeds': sorted(seeds), 'code_values': sorted(codes),
           'train_knobs_values': sorted(knobs)}
    for field in _COMPONENT_DRAW_FP_FIELDS:
        out[field + 's'] = sorted(fps[field])
    return out


def _draws_overlap(a, b):
    """True/False/None for two canonical setting sets.

    None means at least one side has no recorded value, so overlap cannot be
    attested; it must never be read as "disjoint".
    """
    if not a or not b:
        return None
    return bool(set(a) & set(b))


def replication_independence_audit(panels):
    """Classify whether verified panels are independent replications.

    ``panels`` maps panel name -> a :func:`component_draw_identity` dict (or a
    compatible mapping with ``seeds`` and fingerprint lists).  Two panels are
    only an independent replication when their training seeds are disjoint,
    no recorded evaluation fingerprint overlaps, and at least one
    eval/holdout fingerprint is recorded as disjoint.  Disjoint seeds with a
    shared (or unverifiable) evaluation slice are recorded as new training
    draws, not as independent evaluation; overlapping seeds are repeated
    measurements of the same draws, however different the surrounding panel
    labels are.

    Returns per-pair detail plus one of ``independent_replication``,
    ``new_seeds_same_evaluation``, ``new_seeds_unverified_evaluation``,
    ``shared_training_draws``, ``unattested`` or ``insufficient_panels``.
    This is a labelling/diagnostic gate, not a new hypothesis test.
    """
    if not isinstance(panels, dict):
        return {'status': 'unattested', 'n_panels': 0, 'n_pairs': 0,
                'pairs': {}, 'independent_pairs': [], 'new_seed_pairs': [],
                'shared_seed_pairs': [], 'unattested_pairs': [],
                'panels_independent': [], 'panels_new_seeds': [],
                'panels_shared_seeds': [],
                'warnings': ['panels must be a mapping of panel -> draw identity']}
    norm = {}
    for name, ident in panels.items():
        if not isinstance(ident, dict):
            continue
        seeds = set()
        _bad_seed = False
        for s in _identity_values(ident.get('seeds')):
            _s = _as_seed_int(s)
            if _s is None:
                _bad_seed = True
            else:
                seeds.add(_s)
        fps = {f: {_draw_value(x) for x in _identity_values(ident.get(f + 's'))
                   if x is not None and not (isinstance(x, str) and not x.strip())}
               for f in _COMPONENT_DRAW_FP_FIELDS}
        _codes = {_draw_value(x) for x in _identity_values(ident.get('code_values'))
                  if x is not None and not (isinstance(x, str) and not x.strip())}
        _knobs = {_draw_value(x) for x in _identity_values(ident.get('train_knobs_values'))
                  if x is not None}
        norm[str(name)] = {'seeds': sorted(seeds) if seeds else None,
                           'bad_seed': _bad_seed, 'fps': fps,
                           'code_values': _codes,
                           'train_knobs_values': _knobs}
    pairs, independent, new_seed, shared, unattested = {}, [], [], [], []
    names = sorted(norm)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A, B = norm[a], norm[b]
            sa, sb = A['seeds'], B['seeds']
            seed_overlap = (sorted(set(sa) & set(sb))
                            if sa is not None and sb is not None else None)
            seed_disjoint = (not seed_overlap if seed_overlap is not None else False)
            ev_overlap = {}
            for field in ('val_fp', 'eval_fp', 'holdout_row_fp'):
                ev_overlap[field] = _draws_overlap(A['fps'][field], B['fps'][field])
            shared_fields = [f for f, v in ev_overlap.items() if v is True]
            attested = [f for f, v in ev_overlap.items() if v is not None]
            # Validation-only agreement is not enough to attest an independent
            # outcome slice: `eval_fp`/`holdout_row_fp` must also be recorded
            # as disjoint.  Otherwise a panel pair with two unrecorded
            # evaluation slices would be promoted to independence on the
            # strength of a validation fingerprint alone.
            _primary_attested = any(
                ev_overlap[f] is not None
                for f in ('eval_fp', 'holdout_row_fp'))
            evaluation_shared = (True if shared_fields else
                                 (False if _primary_attested else None))
            if A['bad_seed'] or B['bad_seed'] or sa is None or sb is None:
                tier = 'unattested'
            elif seed_overlap:
                tier = 'shared_training_draws'
            elif evaluation_shared is True:
                tier = 'new_seeds_same_evaluation'
            elif evaluation_shared is False:
                tier = 'independent_replication'
            else:
                tier = 'new_seeds_unverified_evaluation'
            pairs[f'{a}|{b}'] = {
                'panel_a': a, 'panel_b': b, 'tier': tier,
                'seed_overlap': seed_overlap,
                'seed_disjoint': bool(seed_disjoint),
                'evaluation_overlap': ev_overlap,
                'evaluation_shared': evaluation_shared,
                'shared_evaluation_fields': shared_fields,
                'evaluation_attested_fields': attested,
                'code_shared': _draws_overlap(A['code_values'],
                                               B['code_values']),
                'recipe_shared': _draws_overlap(A['train_knobs_values'],
                                                 B['train_knobs_values']),
            }
            if tier == 'independent_replication':
                independent.append(pairs[f'{a}|{b}'])
            elif tier in ('new_seeds_same_evaluation',
                          'new_seeds_unverified_evaluation'):
                new_seed.append(pairs[f'{a}|{b}'])
            elif tier == 'shared_training_draws':
                shared.append(pairs[f'{a}|{b}'])
            else:
                unattested.append(pairs[f'{a}|{b}'])
    if independent:
        status = 'independent_replication'
    elif new_seed:
        status = 'new_seeds_only'
    elif shared:
        status = 'shared_training_draws'
    elif len(norm) < 2:
        status = 'insufficient_panels'
    else:
        status = 'unattested'
    warnings = []
    if status == 'shared_training_draws':
        warnings.append(
            'verified panels share training seeds (%s); this is repeated '
            'measurement of the same draws, not independent replication'
            % ', '.join(map(str, sorted({s for p in shared for s in (p['seed_overlap'] or [])}))))
    elif status == 'new_seeds_only':
        warnings.append(
            'verified panels use disjoint training seeds but share or do not '
            'attest an independent evaluation slice; report this as a new-seed '
            'replication, not an independent evaluation')
    elif status == 'unattested':
        warnings.append('panel draw identities are missing or incomplete; '
                        'replication independence is unattested')
    return {
        'status': status, 'n_panels': len(norm), 'n_pairs': len(pairs),
        'pairs': pairs,
        'independent_pairs': [f'{p["panel_a"]}|{p["panel_b"]}' for p in independent],
        'new_seed_pairs': [f'{p["panel_a"]}|{p["panel_b"]}' for p in new_seed],
        'shared_seed_pairs': [f'{p["panel_a"]}|{p["panel_b"]}' for p in shared],
        'unattested_pairs': [f'{p["panel_a"]}|{p["panel_b"]}' for p in unattested],
        'panels_independent': sorted({p['panel_a'] for p in independent}
                                     | {p['panel_b'] for p in independent}),
        'panels_new_seeds': sorted({p['panel_a'] for p in new_seed}
                                   | {p['panel_b'] for p in new_seed}),
        'panels_shared_seeds': sorted({p['panel_a'] for p in shared}
                                      | {p['panel_b'] for p in shared}),
        'warnings': warnings,
    }


def _baseline_artifact_path(name, outdir=None):
    """Locate a baseline artifact next to a panel or its parent.

    A final panel usually lives in `<outdir>/final/`, while the tuning and
    count-floor artifacts are written to `<outdir>/`.  Looking only inside the
    final directory silently downgraded the equal-search and beats-floor gates
    to `not_verifiable_locally`.  Search the panel directory first, then its
    parent, then the conventional results_baseline_tuning location."""
    paths = []
    if outdir:
        paths.append(os.path.join(outdir, name))
        paths.append(os.path.join(os.path.dirname(os.path.abspath(outdir)), name))
    paths.append(os.path.join('results_baseline_tuning', name))
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]

def methodology_seed_plan(n_seeds, design=None, baselines=None, alpha=0.05, variants=None):
    """Prospective seed plan for a planned panel.

    Returns the declared family size and the minimum n at which a
    Holm-corrected exact sign-flip test could reject at `alpha` if every seed
    moved in the predicted direction.  Drivers call this BEFORE training so a
    3-seed panel is labelled descriptive up front instead of being read after
    the fact as a failed confirmation.  Pure function; never blocks a run."""
    alpha = _validated_alpha(alpha, context='methodology_seed_plan alpha')
    if design is None:
        design = ABLATION_DESIGN
        if baselines is None:
            baselines = BASELINE_DESIGN
    baselines = {} if baselines is None else baselines
    # A non-mapping design cannot declare any contrast; normalise it instead
    # of crashing on `.values()` / iterating a string below.  Malformed
    # per-component declarations are counted conservatively by
    # `_declared_contrast_count`.
    design = _materialise_contrast_design(design)
    baselines = _materialise_contrast_design(baselines)
    if not isinstance(design, dict):
        design = {}
    if not isinstance(baselines, dict):
        baselines = {}
    _present = None
    if variants is not None:
        _present = set(variants or ())
        design = _scope_design(design, _present)
        baselines = _scope_design(baselines, _present)
    _ablation = (ablation_completeness_audit(design, _present) if design else
                 {'mode': 'leave-one-out', 'complete': None, 'components': {},
                  'complete_components': [], 'incomplete_components': [],
                  'missing_arms': {}})
    m = (_declared_contrast_count(design)
         + _declared_contrast_count(baselines))
    req = signflip_min_n_for_holm(m, alpha)
    # A fractional/bool seed count is not a plan size; truncating 6.9 to 6
    # could label an invalid plan sufficient.  Treat it as zero (fail closed)
    # instead of silently coercing it.
    _n = _as_seed_int(n_seeds)
    n = _n if (_n is not None and _n >= 0) else 0
    return {'n_seeds': n, 'declared_contrasts': int(m), 'alpha': float(alpha),
            'min_seeds_for_family_significance': req,
            'sufficient_for_family_rejection': bool(req is not None and n >= req),
            'ablation_completeness': _ablation,
            'ablation_complete': _ablation.get('complete'),
            'note': 'a panel below the minimum is DESCRIPTIVE ONLY: the exact-test resolution floor, not the observed effect, bounds its p value'}

def _interaction_pair(a, b):
    """Order-insensitive component pair key; ``None`` if either id is invalid."""
    if not isinstance(a, str) or not isinstance(b, str):
        return None
    a, b = a.strip(), b.strip()
    if not a or not b or a == b:
        return None
    return tuple(sorted((a, b)))


def _factorial_factor_names(spec):
    """Return the two distinct factor names of a factorial declaration."""
    if not isinstance(spec, dict):
        return None
    factors = spec.get('factors')
    if (not isinstance(factors, (list, tuple)) or len(factors) != 2
            or not all(isinstance(f, str) and f.strip() for f in factors)):
        return None
    a, b = factors[0].strip(), factors[1].strip()
    if a == b:
        return None
    return a, b


def _scope_pairs(scope, components):
    """Normalise a declared interaction scope to a set of component pairs.

    The canonical form is ``{name: {'factors': (...), ...}}``.  A bare list of
    factor names is also accepted for a single scope.  Malformed declarations
    raise, so the public audit can fail closed instead of silently dropping a
    pair from the denominator.
    """
    if scope is None:
        return set()
    pairs = set()
    if isinstance(scope, (list, tuple)):
        # A bare list is a single scope.  Every element must be a usable factor
        # id: silently dropping a malformed entry would shrink the denominator
        # (and could make an incomplete scope look covered).
        if any(not isinstance(x, str) or not x.strip() for x in scope):
            raise ValueError('interaction scope list must contain non-empty factor names')
        factors = [x.strip() for x in scope]
        scope = {'scope': {'factors': factors}}
    if not isinstance(scope, dict):
        raise ValueError(f'interaction scope must be a mapping or list, got {type(scope).__name__}')
    for name, spec in scope.items():
        if isinstance(spec, (list, tuple)):
            factors = spec
        elif isinstance(spec, dict):
            factors = spec.get('factors')
        else:
            raise ValueError(f'interaction scope {name!r} must be a list or mapping')
        if not isinstance(factors, (list, tuple)) or not factors:
            raise ValueError(f'interaction scope {name!r} has no factor list')
        clean = []
        for f in factors:
            if not isinstance(f, str) or not f.strip():
                raise ValueError(f'interaction scope {name!r} has an invalid factor {f!r}')
            clean.append(f.strip())
        unique = sorted(set(clean))
        if len(unique) < 2:
            raise ValueError(f'interaction scope {name!r} needs at least two distinct factors')
        unknown = sorted(set(unique) - set(components))
        if unknown:
            raise ValueError(f'interaction scope {name!r} names undeclared component(s): '
                             + ', '.join(unknown))
        for a, b in itertools.combinations(unique, 2):
            pairs.add((a, b))
    return pairs


def _normalise_interaction_waivers(waivers, components):
    """Return ``{(a, b): reason}`` with structural validation.

    A waiver is a *scope* declaration, not evidence: it says why a pair is not
    part of this study's interaction claim.  It must name two declared
    components and carry a non-empty reason.  An empty reason is rejected
    because "not tested" without a reason is exactly the silent omission this
    audit exists to expose.
    """
    out = {}
    if waivers is None:
        return out
    if not isinstance(waivers, dict):
        raise ValueError(f'interaction waivers must be a mapping, got {type(waivers).__name__}')
    for key, reason in waivers.items():
        if not isinstance(key, (tuple, list)) or len(key) != 2:
            raise ValueError(f'interaction waiver key {key!r} must be a component pair')
        pair = _interaction_pair(key[0], key[1])
        if pair is None:
            raise ValueError(f'interaction waiver key {key!r} is not a distinct component pair')
        unknown = sorted(set(pair) - set(components))
        if unknown:
            raise ValueError(f'interaction waiver {key!r} names undeclared component(s): '
                             + ', '.join(unknown))
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f'interaction waiver {key!r} needs a non-empty reason')
        if pair in out:
            raise ValueError(f'interaction waiver {key!r} duplicates pair {pair!r}')
        out[pair] = reason.strip()
    return out


def interaction_coverage_audit(design=None, factorials=None, scope=None,
                               waivers=None):
    """Zero-GPU coverage gate for pairwise ablation interactions.

    Leave-one-out completeness answers a per-component question.  It does not
    tell the reader which *pairs* of components were ever manipulated
    factorially, so a panel can look complete while every interaction remains
    unmeasured.  This audit enumerates the component pairs declared in scope
    and classifies each one as:

    * ``tested``   - at least one 2x2 factorial names both components;
    * ``waived``   - the design explicitly excludes the pair, with a reason;
    * ``missing``  - neither tested nor waived.

    A missing interaction may not be reported as additivity, and a set of
    individually verified leave-one-out effects may not be added together as if
    the modules were independent.  The audit returns ``coverage_status='ok'``
    only when every in-scope pair is tested or waived; otherwise it is
    ``'partial'``.  This is deliberately separate from
    :func:`factorial_design_audit`: a valid 2x2 is not evidence that every
    relevant pair was considered.  Structural defects (unknown components,
    duplicate or reasonless waivers) fail closed; an untested pair is a
    *reported design gap*, not a fabricated failure of the pairwise tests that
    were actually run.
    """
    design = ABLATION_DESIGN if design is None else design
    factorials = FACTORIAL_DESIGN if factorials is None else factorials
    scope_spec = INTERACTION_SCOPE if scope is None else scope
    out = {
        'kind': 'interaction_coverage', 'status': 'ok',
        'coverage_status': 'not_applicable', 'components': [],
        'tested_pairs': [], 'waived_pairs': [], 'missing_pairs': [],
        'unexpected_pairs': [], 'waivers_outside_scope': [],
        'duplicate_pairs': {},
        'factorials': {}, 'scope': {}, 'warnings': [], 'problems': [],
        'full_pairwise_measured': False,
        'claim_policy': {
            'tested': ('only a measured 2x2 factorial may support an '
                       'interaction/sub-additivity claim'),
            'missing': ('a missing interaction is NOT evidence of additivity; '
                        'do not add leave-one-out effects as if modules were '
                        'independent'),
            'waived': ('a waiver is an explicit scope exclusion, not a null '
                       'result and not evidence about the pair'),
            'individual_claims': ('an already measured 2x2 remains usable even '
                                  'if another in-scope pair is missing; only '
                                  'the "all core interactions were measured" '
                                  'claim is closed'),
        },
    }
    if not isinstance(design, dict) or not design:
        out['status'] = 'fail'
        out['problems'].append(
            f'ablation design must be a non-empty mapping, got {type(design).__name__}')
        return out
    components = sorted(str(k) for k in design)
    out['components'] = components
    if not isinstance(factorials, dict):
        out['status'] = 'fail'
        out['problems'].append(
            f'factorial design must be a mapping, got {type(factorials).__name__}')
        return out
    try:
        expected = _scope_pairs(scope_spec, components)
        waived = _normalise_interaction_waivers(waivers, components)
    except ValueError as e:
        out['status'] = 'fail'
        out['problems'].append(str(e))
        return out
    out['scope'] = {
        'declared_pairs': [list(p) for p in sorted(expected)],
        'n_pairs': len(expected),
        'source': 'argument' if scope is not None else 'INTERACTION_SCOPE',
    }
    tested = {}
    for name, spec in factorials.items():
        sname = str(name)
        factors = _factorial_factor_names(spec)
        rec = {'factors': list(factors) if factors else None,
               'pair': None, 'status': 'invalid'}
        out['factorials'][sname] = rec
        if factors is None:
            out['problems'].append(
                f'{sname}: factorial has no two distinct factor names; it '
                'cannot cover an interaction pair')
            continue
        unknown = sorted(set(factors) - set(components))
        if unknown:
            out['status'] = 'fail'
            out['problems'].append(
                f'{sname}: factor(s) {unknown} are not declared components in '
                'the ablation design')
            continue
        pair = _interaction_pair(*factors)
        rec['pair'] = list(pair)
        rec['status'] = 'tested'
        if pair in tested:
            # Keep every factorial that declares this pair, including the
            # first one.  Appending `tested[pair]` on every duplicate used to
            # repeat the original name and lose the declaration order.
            names = out['duplicate_pairs'].setdefault(
                str(list(pair)), [tested[pair]])
            if sname not in names:
                names.append(sname)
            out['warnings'].append(
                f'{sname}: interaction pair {pair} is also declared by '
                f'{tested[pair]}; only one factorial should define that pair')
        else:
            tested[pair] = sname
    out['tested_pairs'] = [list(p) for p in sorted(tested)]
    out['waived_pairs'] = [
        {'pair': list(p), 'reason': r} for p, r in sorted(waived.items())]
    out['waivers_outside_scope'] = [
        {'pair': list(p), 'reason': waived[p]}
        for p in sorted(set(waived) - expected)]
    if out['waivers_outside_scope']:
        out['warnings'].append(
            'waiver(s) name pair(s) outside the declared interaction scope '
            'and do not change coverage: '
            + ', '.join('%s x %s' % tuple(x['pair'])
                        for x in out['waivers_outside_scope']))
    out['missing_pairs'] = [
        list(p) for p in sorted(expected - set(tested) - set(waived))]
    out['unexpected_pairs'] = [
        list(p) for p in sorted(set(tested) - expected)]
    # "Full pairwise measured" is a property of the declared scope.  A
    # factorial that lies outside the scope is reported separately; it does
    # not un-measure the in-scope pairs.
    out['full_pairwise_measured'] = bool(expected and not out['missing_pairs'])
    if out['missing_pairs']:
        out['coverage_status'] = 'partial'
        out['warnings'].append(
            'interaction(s) in the declared scope were not measured: '
            + ', '.join('%s x %s' % tuple(p) for p in out['missing_pairs'])
            + ' — report them as untested, not additive')
    elif expected:
        out['coverage_status'] = 'complete'
    else:
        out['coverage_status'] = 'not_applicable'
        out['warnings'].append('no interaction scope is declared; coverage is unverifiable')
    if out['unexpected_pairs']:
        out['warnings'].append(
            'factorial(s) measure pair(s) outside the declared interaction '
            'scope: ' + ', '.join('%s x %s' % tuple(p)
                                  for p in out['unexpected_pairs'])
            + ' — extend the scope declaration or move the factorial out of this family')
    _in_scope = set().union(*expected) if expected else set()
    outside = sorted(set(components) - _in_scope)
    out['components_outside_scope'] = outside
    if outside:
        out['warnings'].append(
            'component(s) outside the declared interaction scope: '
            + ', '.join(outside)
            + ' — this is a claim boundary, not a measured zero interaction')
    if out['problems']:
        out['status'] = 'fail'
    return out


def ablation_factorial_anchor_audit(design=None, factorials=None):
    """Check that each 2x2 factor is anchored to a declared leave-one-out.

    A factorial can pass the parallel-manipulation gate while comparing arms
    that were never declared as the ablation of either factor.  The cells then
    estimate an interaction between two *ad hoc* manipulations, not between the
    components the paper names.  For every factor this audit checks both
    simple-effect pairs used by the factorial against
    ``ABLATION_DESIGN[factor]`` (orientation-insensitive).  At least one level
    must reuse a declared contrast; the other level normally uses the parallel
    arm (for example ``csa_dyn_randidx`` vs ``csa_fix_randidx`` for dynamic
    blocking when selection is at its control level).  Zero matches is a hard
    violation.  The report records partial anchoring explicitly instead of
    pretending both levels are the same declared contrast.
    """
    design = ABLATION_DESIGN if design is None else design
    factorials = FACTORIAL_DESIGN if factorials is None else factorials
    out = {'kind': 'ablation_factorial_anchor', 'status': 'ok',
           'factorials': {}, 'violations': [], 'warnings': []}
    if not isinstance(design, dict) or not isinstance(factorials, dict):
        out['status'] = 'fail'
        out['violations'].append('design and factorials must both be mappings')
        return out
    declared = {}
    for comp, contrasts in design.items():
        pairs = set()
        if isinstance(contrasts, (list, tuple)):
            for item in contrasts:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    p = _interaction_pair(item[0], item[1])
                    if p is not None:
                        pairs.add(p)
        declared[str(comp)] = pairs
    for name, spec in factorials.items():
        sname = str(name)
        rec = {'factors': None, 'levels': {}, 'component_anchored': None,
               'fully_anchored': None}
        out['factorials'][sname] = rec
        factors = _factorial_factor_names(spec)
        cells = spec.get('cells') if isinstance(spec, dict) else None
        if factors is None:
            rec['component_anchored'] = False
            rec['fully_anchored'] = False
            out['violations'].append(
                f'{sname}: factorial has no two distinct factor names; '
                'anchoring is unverifiable')
            continue
        rec['factors'] = list(factors)
        if not isinstance(cells, dict):
            rec['component_anchored'] = False
            rec['fully_anchored'] = False
            out['violations'].append(
                f'{sname}: cells must be a mapping; anchoring is unverifiable')
            continue
        arms = {}
        for key in _FACTORIAL_CELL_KEYS:
            arm = cells.get(key)
            if not isinstance(arm, str) or not arm.strip():
                arms = None
                break
            arms[key] = arm.strip()
        if arms is None:
            rec['component_anchored'] = False
            rec['fully_anchored'] = False
            out['violations'].append(
                f'{sname}: all four 2x2 cells must name a non-empty arm; '
                'anchoring is unverifiable')
            continue
        level_pairs = {
            factors[0]: [
                (arms[('active', 'control')], arms[('control', 'control')]),
                (arms[('active', 'active')], arms[('control', 'active')]),
            ],
            factors[1]: [
                (arms[('control', 'active')], arms[('control', 'control')]),
                (arms[('active', 'active')], arms[('active', 'control')]),
            ],
        }
        for factor, pairs in level_pairs.items():
            if factor not in declared:
                out['warnings'].append(
                    f'{sname}: factor {factor!r} is not a declared ablation '
                    'component; anchoring is unverifiable')
                rec['levels'][factor] = {'declared_matches': None,
                                         'status': 'unverifiable'}
                continue
            norm = []
            for a, b in pairs:
                p = _interaction_pair(a, b)
                norm.append({'pair': list(p) if p else None,
                             'declared': bool(p is not None and p in declared[factor])})
            n_match = sum(1 for x in norm if x['declared'])
            rec['levels'][factor] = {
                'pairs': norm, 'declared_matches': n_match,
                'status': ('anchored' if n_match == 2 else
                           'partially_anchored' if n_match == 1 else
                           'unanchored')}
            if n_match == 0:
                out['violations'].append(
                    f'{sname}: factor {factor!r} uses no declared leave-one-out '
                    'contrast; the interaction is between ad hoc arms, not the '
                    'declared component')
        _levels = list(rec['levels'].values())
        rec['component_anchored'] = bool(_levels) and all(
            x.get('status') in ('anchored', 'partially_anchored')
            for x in _levels)
        rec['fully_anchored'] = bool(_levels) and all(
            x.get('status') == 'anchored' for x in _levels)
    if out['violations']:
        out['status'] = 'fail'
    return out


def factorial_design_audit(design=None, present=None, factory=None, n_layers=6):
    """Zero-GPU gate for the declared 2x2 ablation interactions.

    A factorial is only interpretable when the two factors are manipulated in
    a *parallel* way at both levels of the other factor.  This audit constructs
    every cell through the real arm factory and verifies:

      * all four (control/control, active/control, control/active,
        active/active) cells exist, use distinct arms, and are scheduled;
      * the ``control -> active`` AttnCfg diff of each factor is IDENTICAL at
        the other factor's control and active levels (otherwise the two cells
        compare different interventions and the interaction is not identified);
      * the factor diff stays inside the declared intervention family for that
        component and never swaps ``kind`` (an architecture change);
      * every factorial has a complete pre-registered interaction hypothesis.

    The full-factorial completeness mode of
    :func:`ablation_completeness_audit` is reused, so a missing or reused cell
    is reported with its exact combination.  Returns a report dict; ``status``
    is ``'ok'`` only when no violation was found.
    """
    _design_supplied = design is not None
    design = FACTORIAL_DESIGN if design is None else design
    factory = make_layer_cfgs if factory is None else factory
    _nl = _as_seed_int(n_layers)
    if _nl is None or _nl < 1:
        raise ValueError(
            f'factorial_design_audit: n_layers must be a positive integer, got {n_layers!r}')
    n_layers = int(_nl)
    report = {'kind': 'factorial_design', 'status': 'ok', 'n_layers': n_layers,
              'factorials': {}, 'violations': [], 'warnings': []}
    try:
        present_set = (set(RUN_ABL['variants']) if present is None
                       else set(present))
    except TypeError:
        report['violations'].append(
            'present arm ids must be a collection of hashable strings, got '
            f'{type(present).__name__}')
        report['status'] = 'fail'
        return report
    if not isinstance(design, dict) or not design:
        report['violations'].append(
            f'factorial design must be a non-empty mapping, got {type(design).__name__}')
        report['status'] = 'fail'
        return report
    for name, spec in design.items():
        rec = {'factors': None, 'cells': {}, 'complete': False,
               'missing_cells': [], 'extra_cells': [], 'missing_arms': [],
               'factor_diffs': {}, 'factor_diff_signatures': {},
               'hypothesis': None, 'hypothesis_audit': None,
               'full_factorial': None}
        report['factorials'][str(name)] = rec
        _ok = True

        def _bad(msg):
            nonlocal _ok
            _ok = False
            report['violations'].append(f'{name}: {msg}')

        if not isinstance(spec, dict):
            _bad(f'declaration must be a mapping, got {type(spec).__name__}')
            continue
        factors = spec.get('factors')
        cells = spec.get('cells')
        if (not isinstance(factors, (list, tuple)) or len(factors) != 2
                or not all(isinstance(f, str) and f.strip() for f in factors)
                or factors[0].strip() == factors[1].strip()):
            _bad('factors must be two distinct non-empty component names')
            continue
        factors = tuple(f.strip() for f in factors)
        rec['factors'] = list(factors)
        if not isinstance(cells, dict):
            _bad(f'cells must be a mapping, got {type(cells).__name__}')
            continue
        norm_cells = {}
        for key, arm in cells.items():
            if (not isinstance(key, (tuple, list)) or len(key) != 2
                    or any(k not in ('control', 'active') for k in key)):
                _bad(f'cell key {key!r} must be (control|active, control|active)')
                continue
            if not isinstance(arm, str) or not arm.strip():
                _bad(f'cell {tuple(key)!r} has an invalid arm id {arm!r}')
                continue
            _nkey = tuple(key)
            if _nkey in norm_cells:
                _bad(f'cell key {key!r} duplicates normalized cell {_nkey!r}; '
                     'the declaration is ambiguous')
                continue
            norm_cells[_nkey] = arm.strip()
        rec['cells'] = {f'{k[0]}|{k[1]}': v for k, v in norm_cells.items()}
        missing = [list(k) for k in _FACTORIAL_CELL_KEYS if k not in norm_cells]
        extra = [list(k) for k in norm_cells if k not in _FACTORIAL_CELL_KEYS]
        rec['missing_cells'] = missing
        rec['extra_cells'] = extra
        if missing:
            _bad('missing 2x2 cell(s): ' + json.dumps(missing))
        if extra:
            _bad('undeclared cell(s) outside the 2x2 design: ' + json.dumps(extra))
        if set(norm_cells) != set(_FACTORIAL_CELL_KEYS):
            continue
        arms = [norm_cells[k] for k in _FACTORIAL_CELL_KEYS]
        if len(set(arms)) != len(arms):
            _bad('the four cells must map to four distinct arm ids')
        missing_arms = sorted(set(arms) - present_set)
        rec['missing_arms'] = missing_arms
        if missing_arms:
            _bad('arm(s) not scheduled in the canonical ablation panel: '
                 + ', '.join(missing_arms))

        combos = {
            (): norm_cells[('control', 'control')],
            (factors[0],): norm_cells[('active', 'control')],
            (factors[1],): norm_cells[('control', 'active')],
            tuple(sorted(factors)): norm_cells[('active', 'active')],
        }
        abl = ablation_completeness_audit(arms_by_combo=combos, present=present_set)
        rec['full_factorial'] = {
            'complete': bool(abl.get('complete')),
            'n_mapped': int(abl.get('n_mapped') or 0),
            'missing_combos': abl.get('missing_combos') or [],
            'reused_arms': abl.get('reused_arms') or {},
        }
        if not abl.get('complete'):
            _bad('full-factorial completeness failed: '
                 + json.dumps({k: rec['full_factorial'][k] for k in
                               ('missing_combos', 'reused_arms')},
                              ensure_ascii=False))

        for fi, factor in enumerate(factors):
            if fi == 0:
                pairs = ((norm_cells[('active', 'control')],
                          norm_cells[('control', 'control')]),
                         (norm_cells[('active', 'active')],
                          norm_cells[('control', 'active')]))
            else:
                pairs = ((norm_cells[('control', 'active')],
                          norm_cells[('control', 'control')]),
                         (norm_cells[('active', 'active')],
                          norm_cells[('active', 'control')]))
            keys = []
            signatures = []
            for a, b in pairs:
                _k, _per = _design_pair_cfg_keys(a, b, factory, report['n_layers'])
                if isinstance(_k, Exception):
                    _bad(f'{factor}: cannot construct the {a} vs {b} contrast '
                         f'({type(_k).__name__}: {_k})')
                    keys.append(())
                    signatures.append(None)
                    continue
                keys.append(tuple(_k))
                signatures.append(_canon_setting(_per))
            rec['factor_diffs'][factor] = {
                'at_other_control': list(keys[0]),
                'at_other_active': list(keys[1]),
            }
            rec['factor_diff_signatures'][factor] = {
                'at_other_control': signatures[0],
                'at_other_active': signatures[1],
            }
            if keys[0] != keys[1]:
                _bad(f'{factor}: the intervention is not parallel across the '
                     f'other factor\'s levels ({list(keys[0])} vs {list(keys[1])}); '
                     'the interaction would compare two different manipulations')
            elif (signatures[0] is not None and signatures[1] is not None
                  and signatures[0] != signatures[1]):
                _bad(f'{factor}: the intervention changes the same key names but '
                     'with different values across the other factor\'s levels; '
                     'the interaction would compare two different manipulations')
            union = set(keys[0]) | set(keys[1])
            if not union:
                _bad(f'{factor}: no AttnCfg key differs between the declared '
                     'control and active cells; the intervention is absent')
            if 'kind' in union:
                _bad(f'{factor}: changing the attention-kind field makes this '
                     'an architecture swap, not a 2x2 factor')
            allowed = set()
            _decl = ABLATION_INTERVENTION_KEYS.get(factor, {})
            if isinstance(_decl, dict):
                for _kset in _decl.values():
                    allowed.update(_kset)
            extra_keys = sorted(union - allowed)
            if extra_keys:
                _bad(f'{factor}: undeclared AttnCfg key(s) move with the factor: '
                     + ', '.join(extra_keys))

        hname = str(spec.get('hypothesis') or name)
        rec['hypothesis'] = hname
        hyp = INTERACTION_HYPOTHESIS_DESIGN.get(hname)
        if not isinstance(hyp, dict):
            _bad(f'no pre-registered interaction hypothesis named {hname!r}')
        else:
            ha = hypothesis_audit({hname: hyp}, 'interaction', names=[hname])
            rec['hypothesis_audit'] = ha
            bad_h = (ha.get('invalid') or []) + (ha.get('metric_unknown') or []) + \
                    (ha.get('metric_misaligned') or [])
            if bad_h:
                _bad('interaction hypothesis is malformed or metric-misaligned: '
                     + ', '.join(str(x) for x in bad_h))
        rec['complete'] = bool(_ok and not missing and not extra
                               and not missing_arms)
    # Coverage and canonical anchoring are scope claims about the *declared*
    # factorial family.  A caller-supplied design may use custom factor names
    # or fixture arms; applying the canonical INTERACTION_SCOPE/ABLATION_DESIGN
    # to it would reject a valid custom design (and treat an unmeasured
    # canonical pair as that design's defect).  Derive a scope from the custom
    # factors and only run anchoring when all arms are canonical.
    _canonical_arms = set()
    for _spec in FACTORIAL_DESIGN.values():
        _cells = _spec.get('cells') if isinstance(_spec, dict) else None
        if isinstance(_cells, dict):
            _canonical_arms.update(
                str(_a).strip() for _a in _cells.values() if isinstance(_a, str))
    _factors, _arms, _valid = set(), set(), True
    for _spec in design.values():
        _fs = _factorial_factor_names(_spec)
        if _fs is None:
            _valid = False
            continue
        _factors.update(_fs)
        _cells = _spec.get('cells') if isinstance(_spec, dict) else None
        _declared_arms = (list(_cells.values()) if isinstance(_cells, dict)
                          else [])
        if not _declared_arms or any(
                not isinstance(_a, str) or not _a.strip()
                for _a in _declared_arms):
            _valid = False
            continue
        _arms.update(_a.strip() for _a in _declared_arms)
    _canonical = bool(_valid and _factors
                      and _factors <= set(ABLATION_DESIGN)
                      and _arms <= _canonical_arms)
    if _design_supplied:
        _flist = sorted(_factors)
        if len(_flist) >= 2:
            coverage = interaction_coverage_audit(
                design={f: [] for f in _flist}, factorials=design,
                scope={'design_factors': {'factors': _flist}})
        else:
            coverage = {
                'kind': 'interaction_coverage', 'status': 'ok',
                'coverage_status': 'not_applicable', 'components': _flist,
                'tested_pairs': [], 'waived_pairs': [], 'missing_pairs': [],
                'unexpected_pairs': [], 'waivers_outside_scope': [],
                'duplicate_pairs': {}, 'factorials': {}, 'scope': {},
                'warnings': ['custom factorial design has no pairwise scope; '
                             'coverage is unverifiable'], 'problems': [],
                'full_pairwise_measured': False,
            }
    else:
        coverage = interaction_coverage_audit(factorials=design)
    if _canonical:
        anchor = ablation_factorial_anchor_audit(factorials=design)
    else:
        anchor = {
            'kind': 'ablation_factorial_anchor', 'status': 'not_applicable',
            'factorials': {}, 'violations': [],
            'warnings': ['custom factorial design: canonical leave-one-out '
                         'anchoring is not evaluated'],
        }
    report['interaction_coverage'] = coverage
    report['warnings'].extend(coverage.get('warnings') or [])
    if coverage.get('status') == 'fail':
        report['violations'].extend(
            'interaction coverage: ' + str(p) for p in coverage.get('problems') or [])
    report['component_anchor'] = anchor
    report['warnings'].extend(anchor.get('warnings') or [])
    report['violations'].extend(anchor.get('violations') or [])
    if report['violations']:
        report['status'] = 'fail'
    return report


def interaction_sesoi_audit(design=None, sesoi=None):
    """Zero-GPU gate: every declared 2x2 interaction carries a valid SESOI.

    A non-significant interaction is not evidence of additivity; only a
    pre-registered margin + TOST can produce an ``additive`` verdict.  This
    gate therefore fails closed when a factorial has no margin, when the
    margin is malformed/ambiguous, or when the corresponding interaction
    hypothesis lacks a directional sign.  Extra factorial names are warnings
    (a typo also leaves the intended factorial unmargined, which fails).  It
    does not inspect results or choose a margin.
    """
    design = FACTORIAL_DESIGN if design is None else design
    spec = INTERACTION_SESOI if sesoi is None else sesoi
    out = {
        'kind': 'interaction_sesoi', 'status': 'ok',
        'factorials': {}, 'problems': [], 'warnings': [],
        'policy': {
            'scale': ('natural-log ratio-of-ratios; relative_pct p is '
                      'converted to log1p(p/100)'),
            'equivalence': ('paired TOST at alpha, Holm-adjusted across the '
                            'declared interaction family'),
            'non_significant_is_not_additivity': True,
            'sub_sesoi_is_not_zero': True,
        },
    }
    if not isinstance(design, dict):
        out['status'] = 'fail'
        out['problems'].append(
            f'factorial design must be a mapping, got {type(design).__name__}')
        return out
    if not design:
        out['status'] = 'not_applicable'
        out['note'] = 'no factorial design is declared'
        return out
    try:
        norm = _normalise_interaction_sesoi_design(spec)
    except Exception as e:
        out['status'] = 'fail'
        out['problems'].append(
            f'interaction SESOI declaration is malformed: '
            f'{type(e).__name__}: {e}')
        return out
    for name, decl in design.items():
        sname = str(name)
        rec = {'declared': sname in norm}
        out['factorials'][sname] = rec
        if not isinstance(decl, dict):
            out['problems'].append(
                f'{sname}: factorial declaration must be a mapping, got '
                f'{type(decl).__name__}')
            continue
        if sname not in norm:
            out['problems'].append(
                f'{sname}: no pre-registered interaction SESOI margin; a '
                'non-significant interaction cannot be called additive')
            continue
        rec['margin_spec'] = dict(norm[sname])
        rec['margin_log_ratio'] = norm[sname]['margin_log_ratio']
        rec['margin_ratio'] = (
            float(math.exp(norm[sname]['margin_log_ratio']))
            if norm[sname]['margin_log_ratio'] < 700.0 else None)
        if not str(norm[sname].get('rationale') or '').strip():
            out['warnings'].append(
                f'{sname}: interaction margin has no recorded rationale')
        if (norm[sname].get('relative_pct') is not None
                and norm[sname]['relative_pct'] > 50.0):
            out['warnings'].append(
                f'{sname}: relative interaction SESOI is >50% of the '
                'ratio-of-ratios; verify this is intentional')
        hname = str(decl.get('hypothesis') or sname)
        hyp = INTERACTION_HYPOTHESIS_DESIGN.get(hname)
        if not isinstance(hyp, dict):
            out['problems'].append(
                f'{sname}: no pre-registered interaction hypothesis named '
                f'{hname!r}')
            continue
        pred = hyp.get('predicted_sign')
        if isinstance(pred, bool) or pred not in (-1, 0, 1):
            out['problems'].append(
                f'{sname}: interaction hypothesis {hname!r} has no valid '
                f'predicted_sign (got {pred!r})')
        rec['hypothesis'] = hname
        rec['predicted_sign'] = pred
    extras = sorted(set(norm) - {str(k) for k in design})
    if extras:
        out['warnings'].append(
            'interaction SESOI declares factorial(s) not present in the '
            'design: ' + ', '.join(extras))
    if out['problems']:
        out['status'] = 'fail'
    return out


def _factorial_condition_rows(recs, cells, *, strict_hyper=True):
    """Per-seed 2x2 cells with the same controlled-pair gate as the LOO audit.

    All four cells must be present at one protocol tag and one seed; every cell
    is checked against the (control, control) cell with ``pair_reason`` (and
    the structured MLP-width gate), the primary estimand must be the same in
    all four records, and the interaction is formed on log PPL.  Missing or
    uncontrolled seeds are named, never silently averaged over a subset.
    """
    arms = [cells[k].strip() for k in _FACTORIAL_CELL_KEYS]
    by_arm = {a: {} for a in arms}
    for key, r in recs.items():
        if not (isinstance(key, tuple) and len(key) == 3):
            continue
        v, t, s = key
        if v in by_arm:
            by_arm[v].setdefault(s, {})[str(t or '')] = r
    if not all(by_arm.values()):
        return [], [], None
    all_seeds = set().union(*[set(d) for d in by_arm.values()])
    common = set.intersection(*[set(d) for d in by_arm.values()])
    rows, skipped = [], []
    for s in sorted(all_seeds - common):
        missing = [a for a in arms if s not in by_arm[a]]
        skipped.append({
            'seed': s,
            'reason': 'missing cell record(s) for arm(s): ' + ', '.join(missing),
        })
    for s in sorted(common):
        tagsets = [set(by_arm[a][s]) for a in arms]
        nonempty = set().union(*[ts - {''} for ts in tagsets])
        if len(nonempty) > 1:
            skipped.append({'seed': s, 'reason':
                            'ambiguous protocol tags across the four cells: '
                            + ', '.join(sorted(nonempty))})
            continue
        tag = next(iter(nonempty)) if nonempty else ''
        chosen, bad = {}, None
        for a, ts in zip(arms, tagsets):
            if tag and tag in ts:
                chosen[a] = by_arm[a][s][tag]
            elif '' in ts:
                chosen[a] = by_arm[a][s]['']
            else:
                bad = f'arm {a} has no record for protocol tag {tag!r}'
                break
        if bad is not None:
            skipped.append({'seed': s, 'reason': bad})
            continue
        ref = arms[0]
        reasons = []
        for a in arms[1:]:
            reason = pair_reason(chosen[ref], chosen[a])
            if reason is None and strict_hyper:
                reason = _hyperparam_reason(chosen[ref], chosen[a])
            if reason is not None:
                reasons.append(f'{a}: {reason}')
        if reasons:
            skipped.append({'seed': s, 'reason':
                            'uncontrolled cell(s): ' + '; '.join(reasons)})
            continue
        estimands = {chosen[a].get('ppl_estimation_rows') ==
                     'nested_validation_holdout' for a in arms}
        if len(estimands) > 1:
            skipped.append({'seed': s, 'reason':
                            'the four cells mix nested-holdout and '
                            'full-validation estimands'})
            continue
        ppls, bad = {}, None
        for a in arms:
            p = _record_primary_ppl(chosen[a])
            if p is None:
                bad = f'arm {a} has no usable primary PPL'
                break
            ppls[a] = float(p)
        if bad is not None:
            skipped.append({'seed': s, 'reason': bad})
            continue
        lp = {a: math.log(ppls[a]) for a in arms}
        interaction = ((lp[arms[3]] - lp[arms[2]])
                       - (lp[arms[1]] - lp[arms[0]]))
        simple = {
            'factor1_at_factor2_control': lp[arms[1]] - lp[arms[0]],
            'factor1_at_factor2_active': lp[arms[3]] - lp[arms[2]],
            'factor2_at_factor1_control': lp[arms[2]] - lp[arms[0]],
            'factor2_at_factor1_active': lp[arms[3]] - lp[arms[1]],
        }
        rows.append({'seed': s, 'tag': tag, 'ppl': ppls,
                     'log_ppl': lp, 'interaction_log_ppl': interaction,
                     'simple_log_effects': simple})
    ref = arms[0]
    param = {a: _param_control_detail(recs, a, ref) for a in arms[1:]}
    capacity = [_param_capacity_controlled(d) for d in param.values()]
    if any(v is False for v in capacity):
        capacity_controlled = False
    elif all(v is True for v in capacity):
        capacity_controlled = True
    else:
        capacity_controlled = None
    return rows, skipped, {'reference': ref,
                           'by_cell': param,
                           'capacity_controlled': capacity_controlled}


def factorial_look_snapshot(recs, *, design=None):
    """Per-seed 2x2 interaction values for the look history.

    Only seeds whose four cells clear the same controlled-pair gate as
    :func:`factorial_interaction_audit` are emitted.  The snapshot stores the
    exact per-seed interaction estimand, not a p-value decision, so the
    sequential max-over-looks test can be recomputed later without trusting a
    recorded verdict.  A partial or uncontrolled factorial emits nothing; the
    completeness audit, not the look file, is responsible for naming it.
    """
    design = FACTORIAL_DESIGN if design is None else design
    out = {}
    if not isinstance(recs, dict) or not isinstance(design, dict):
        return out
    if not all(isinstance(k, tuple) and len(k) == 3 for k in recs):
        recs, _amb = _variant_seed_map(recs)
    present = {k[0] for k in recs
               if isinstance(k, tuple) and len(k) == 3}
    for name, spec in design.items():
        if not isinstance(spec, dict):
            continue
        cells = spec.get('cells')
        if not isinstance(cells, dict):
            continue
        raw_arms = [cells.get(k) for k in _FACTORIAL_CELL_KEYS]
        if any(not isinstance(a, str) or not a.strip() for a in raw_arms):
            continue
        arms = [a.strip() for a in raw_arms]
        if len(set(arms)) != 4 or not set(arms).issubset(present):
            continue
        norm_cells = {k: arms[i] for i, k in enumerate(_FACTORIAL_CELL_KEYS)}
        rows, _skipped, _param = _factorial_condition_rows(recs, norm_cells)
        if not rows:
            continue
        seeds = [int(r['seed']) for r in rows]
        deltas = [float(r['interaction_log_ppl']) for r in rows]
        sf = exact_signflip_p(deltas)
        out[_interaction_look_key(str(name))] = {
            'kind': 'interaction',
            'factorial': str(name),
            'cells': {f'{k[0]}|{k[1]}': norm_cells[k]
                      for k in _FACTORIAL_CELL_KEYS},
            'tags': {str(r['seed']): str(r.get('tag') or '') for r in rows},
            'seeds': seeds,
            'deltas': deltas,
            'n': int(len(deltas)),
            'mean': float(np.mean(deltas)),
            'p_exact_signflip': float(sf.get('p', float('nan'))),
        }
    return out


def _interaction_look_analysis(look_series, key, final_seeds, final_deltas,
                               *, alpha=0.05, final_tags=None,
                               final_cells=None):
    """Sequential max-over-looks p for one interaction contrast.

    Mirrors the ablation look reader: a recorded look whose shared-seed delta
    disagrees with the current record is STALE (the seed was re-run), so it is
    excluded and named instead of being mixed into the joint null.  A look is
    also stale when the four cell arms or a shared seed's protocol tag changed,
    even if the numeric delta happens to be identical.  The current aggregation
    is appended when the history does not already contain it, so a manually
    edited summary still gets a two-look check.
    """
    if not final_seeds or not final_deltas:
        return None, [], 0
    cur = {}
    for s, d in zip(final_seeds, final_deltas):
        si = _as_seed_int(s)
        di = _finite_scalar(d)
        if si is None or di is None:
            raise ValueError(
                'final interaction look seeds/deltas must be finite numbers')
        cur[si] = float(di)
    cur_tags = {}
    if isinstance(final_tags, dict):
        for s, t in final_tags.items():
            si = _as_seed_int(s)
            if si is not None:
                cur_tags[si] = str(t or '')
    cur_cells = None
    if isinstance(final_cells, dict) and final_cells:
        cur_cells = {str(k): str(v) for k, v in final_cells.items()}

    def _rows(rec):
        raw_seeds, raw_deltas = rec.get('seeds'), rec.get('deltas')
        if raw_seeds is None or raw_deltas is None:
            return None
        if (isinstance(raw_seeds, (str, bytes, bytearray))
                or isinstance(raw_deltas, (str, bytes, bytearray))):
            return None
        seeds = [_as_seed_int(s) for s in raw_seeds]
        deltas = [_finite_scalar(d) for d in raw_deltas]
        if any(s is None for s in seeds) or any(d is None for d in deltas):
            return None
        if len(seeds) != len(deltas) or not seeds:
            return None
        # A duplicate seed makes the sign assignment ambiguous; treat the whole
        # look as invalid/stale instead of letting sequential_signflip_p raise
        # and take the interaction audit down with it.
        if len(set(seeds)) != len(seeds):
            return None
        pairs = sorted(zip(seeds, deltas), key=lambda t: t[0])
        return [int(s) for s, _d in pairs], [float(d) for _s, d in pairs]

    series, stale, seen = [], [], set()
    for lk in (look_series or {}).get(key) or []:
        if not isinstance(lk, dict):
            stale.append({'look': None,
                          'reason': 'look is not a mapping'})
            continue
        try:
            parsed = _rows(lk)
        except (TypeError, ValueError):
            parsed = None
        if parsed is None:
            stale.append({'look': lk.get('look'),
                          'reason': 'non-numeric/invalid seed-delta rows'})
            continue
        seeds, deltas = parsed
        look_cells = lk.get('cells')
        if (cur_cells is not None and isinstance(look_cells, dict)
                and look_cells):
            norm_cells = {str(k): str(v) for k, v in look_cells.items()}
            if norm_cells != cur_cells:
                stale.append({'look': lk.get('look'),
                              'reason': 'cell arms changed across looks'})
                continue
        look_tags = lk.get('tags')
        if cur_tags and isinstance(look_tags, dict) and look_tags:
            tags = {}
            for s, t in look_tags.items():
                si = _as_seed_int(s)
                if si is not None:
                    tags[si] = str(t or '')
            bad_tags = [s for s in seeds
                        if s in cur_tags and tags.get(s) != cur_tags[s]]
            if bad_tags:
                stale.append({'look': lk.get('look'),
                              'reason': 'protocol tag changed across looks',
                              'seeds': sorted(bad_tags)})
                continue
        dm = dict(zip(seeds, deltas))
        bad = [s for s in seeds if s in cur and abs(dm[s] - cur[s]) > 1e-9]
        if bad:
            stale.append({'look': lk.get('look'),
                          'reason': 'seed delta changed across looks',
                          'seeds': sorted(bad)})
            continue
        sig = (tuple(seeds), tuple(round(v, 12) for v in deltas))
        if sig in seen:
            continue
        seen.add(sig)
        series.append({'look': lk.get('look'), 'seeds': seeds,
                       'deltas': deltas})
    _order = sorted(cur)
    sig = (tuple(_order), tuple(round(cur[s], 12) for s in _order))
    if sig not in seen:
        series.append({
            'look': (series[-1].get('look', 0) + 1 if series else 1),
            'seeds': _order, 'deltas': [cur[s] for s in _order],
            'current': True,
        })
    if not series:
        return None, stale, 0
    seq = sequential_signflip_p(series, alpha=alpha)
    seq['contrast'] = key
    seq['n_stale_looks'] = len(stale)
    seq['stale_looks'] = stale
    seq['look_adjusted'] = bool(len(series) > 1)
    return seq, stale, len(series)
