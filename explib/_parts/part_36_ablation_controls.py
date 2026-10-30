# ---- ablation control ladders -------------------------------------------------
# A leave-one-out contrast answers "is the module useful".  It does not by
# itself show that the *claimed mechanism* is the reason: the control arm may
# differ in magnitude, parameter count or read budget as well as in the
# mechanism.  The strongest available control is a matched placebo that keeps
# every non-mechanism property fixed and breaks only the property under test.
#
# Content is the case where a missing placebo is material.  `csa_fix_zerocont` /
# `csa_dyn_zerocont` compare pooled content against NO content: zeroing changes
# both whether there is any content and whether it belongs to the block.  The
# shift arms keep the same non-zero pooled table (same shape, parameter count,
# read budget and, after normalisation, the same row norm) and move each block's
# content back one block.  Comparing aligned vs shifted therefore isolates
# *alignment*; comparing aligned vs zero isolates *presence*.  Both controls
# are declared here and checked against the real arm factory, so adding a
# content mode that silently changes a second knob (or forgetting one of the
# two controls) fails before any GPU time is spent.
CONTENT_CONTROL_DESIGN = {
    'component': 'content_alignment',
    'presence_component': 'content_pooling',
    'base_arms': {'fixed': 'csa_fixed', 'dynamic': 'csa_dynamic'},
    'levels': {
        'aligned': {
            'content_mode': 'pooled',
            'role': 'treatment',
            'arms': {'fixed': 'csa_fixed', 'dynamic': 'csa_dynamic'},
            'note': 'the block reads its own pooled content',
        },
        'misaligned': {
            'content_mode': 'shift',
            'role': 'alignment_placebo',
            'arms': {'fixed': 'csa_fix_shiftcont', 'dynamic': 'csa_dyn_shiftcont'},
            'note': ('same content table, moved back one block: matched shape/norm/'
                     'parameters/read budget, no block-content correspondence'),
        },
        'absent': {
            'content_mode': 'zero',
            'role': 'presence_control',
            'arms': {'fixed': 'csa_fix_zerocont', 'dynamic': 'csa_dyn_zerocont'},
            'note': 'no content at all; changes presence as well as alignment',
        },
    },
    'note': ('the presence control alone cannot support a content-information '
             'claim; the alignment placebo is the matched control for that claim'),
}


def _declared_arm_id(levels, level_name, base_label):
    """A hashable arm id from a (possibly malformed) level table, or None.

    Later consistency checks form ``(base_arm, control_arm)`` tuples and use
    them as dict keys.  A malformed ladder can put a list/dict where an arm id
    belongs; returning None keeps the audit fail-closed instead of raising
    ``TypeError: unhashable type``.
    """
    spec = levels.get(level_name) if isinstance(levels, dict) else None
    arms = spec.get('arms') if isinstance(spec, dict) else None
    arm = arms.get(base_label) if isinstance(arms, dict) else None
    return arm if isinstance(arm, str) and arm.strip() else None


def _declared_pairs(design, component):
    """String arm pairs declared for one ``ABLATION_DESIGN`` component.

    Returns ``None`` when the component is absent or is not a list/tuple, and
    otherwise a set of the well-formed ``(arm_a, arm_b)`` entries.  Malformed
    entries are skipped rather than raised: a broken declaration must make the
    consuming audit fail closed, not crash it with a ``TypeError``.
    """
    raw = design.get(component) if isinstance(design, dict) else None
    if not isinstance(raw, (list, tuple)):
        return None
    pairs = set()
    for item in raw:
        if (isinstance(item, (tuple, list)) and len(item) >= 2
                and isinstance(item[0], str) and isinstance(item[1], str)):
            pairs.add((item[0], item[1]))
    return pairs


def _control_cfg_field_values(factory, n_layers, arm, field):
    """Actual per-layer value of one config field for a constructed arm.

    The level tables are *declarations*.  Comparing a placebo's declared mode
    only against the field that happens to differ from the base cannot catch a
    treatment or absence level whose declaration silently contradicts the
    actual config (for example a treatment labelled ``random`` while the arm
    it names is the learned base).  Read the constructed value for every layer
    so the audit validates the declaration against the object, not the label.
    """
    cfgs = factory(int(n_layers), arm)
    if isinstance(cfgs, (str, bytes, bytearray)):
        raise TypeError('factory returned a string, not a sequence of layer configs')
    cfgs = list(cfgs)
    if len(cfgs) != int(n_layers):
        raise ValueError(
            f'arm {arm!r} has {len(cfgs)} layer config(s), expected {int(n_layers)}')
    out = {}
    for i, cfg in enumerate(cfgs):
        if not hasattr(cfg, field):
            raise TypeError(f'arm {arm!r} layer {i} has no {field!r} attribute')
        out[i] = getattr(cfg, field)
    return out


def content_control_audit(design=None, ladder=None, present=None, factory=None,
                          n_layers=2):
    """Zero-GPU gate for the content presence/alignment control ladder.

    For every declared base arm the audit constructs the real attention config
    through the arm factory and checks:

      * the ladder has exactly the declared levels (treatment, alignment
        placebo, presence control) and every level supplies one scheduled arm
        per base;
      * the arms are distinct across levels, so a level is not a relabelled
        copy of another;
      * the `aligned` arm is the base itself; every non-aligned level differs
        from the base in EXACTLY `content_mode` and that field has the declared
        value.  A mode change that also moves another knob is a confounded
        control, not a placebo;
      * `ABLATION_DESIGN` contains the two corresponding leave-one-out
        contrasts (`content_pooling` for absent, `content_alignment` for
        misaligned) and `ABLATION_INTERVENTION_KEYS` declares the same single
        key.  The two contrasts then share the same reference arm and differ
        only in which control they use.

    Missing/unconstructed arms and any structural mismatch are ``fail``
    (never a pass).  The audit never reads a result file.
    """
    design = ABLATION_DESIGN if design is None else design
    ladder = CONTENT_CONTROL_DESIGN if ladder is None else ladder
    factory = make_layer_cfgs if factory is None else factory
    _nl = _as_seed_int(n_layers)
    if _nl is None or _nl < 1:
        raise ValueError(
            f'content_control_audit: n_layers must be a positive integer, got {n_layers!r}')
    n_layers = int(_nl)
    out = {'kind': 'content_control_ladder', 'status': 'ok',
           'n_layers': n_layers, 'levels': {}, 'bases': {}, 'base_checks': {},
           'component_design': {}, 'violations': [], 'warnings': [],
           'treatment_component': 'content_alignment',
           'presence_component': 'content_pooling',
           'note': ('presence (zero) and alignment (shift) are different '
                    'controls; only the matched alignment placebo can support '
                    '"content must be aligned"')}
    if not isinstance(ladder, dict) or not ladder:
        out['status'] = 'fail'
        out['violations'].append(
            f'content control ladder must be a non-empty mapping, got '
            f'{type(ladder).__name__}')
        return out
    bases = ladder.get('base_arms')
    levels = ladder.get('levels')
    if (not isinstance(bases, dict) or not bases
            or not all(isinstance(k, str) and k.strip() and isinstance(v, str)
                       and v.strip() for k, v in bases.items())):
        out['status'] = 'fail'
        out['violations'].append('base_arms must map non-empty base labels to arm ids')
        return out
    if not isinstance(levels, dict) or not levels:
        out['status'] = 'fail'
        out['violations'].append('levels must be a non-empty mapping')
        return out
    out['bases'] = dict(bases)
    try:
        present_set = set(RUN_ABL['variants']) if present is None else set(present)
    except TypeError:
        out['status'] = 'fail'
        out['violations'].append('present arm ids must be a collection of strings')
        return out
    # The two roles are a package: a ladder with an alignment level but no
    # presence level (or vice versa) cannot separate the two claims.
    roles = {}
    for name, spec in levels.items():
        if not isinstance(spec, dict):
            out['violations'].append(f'{name}: level must be a mapping')
            continue
        roles[name] = str(spec.get('role') or '')
        _arms = spec.get('arms')
        if not isinstance(_arms, dict) or not _arms:
            out['violations'].append(
                f'{name}: arms must be a non-empty mapping base label -> arm id')
            _arms_norm = _arms
        else:
            _arms_norm = dict(_arms)
        out['levels'][str(name)] = {
            'role': spec.get('role'),
            'content_mode': spec.get('content_mode'),
            'arms': _arms_norm,
        }
    expected_roles = {'treatment', 'alignment_placebo', 'presence_control'}
    if set(roles.values()) != expected_roles:
        out['status'] = 'fail'
        out['violations'].append(
            'ladder roles must be exactly treatment/alignment_placebo/'
            f'presence_control, got {sorted(roles.values())}')
    elif len(roles) != len(expected_roles):
        # A second level claiming an already-used role passes the set
        # comparison, but `by_role` silently keeps only one of them: the
        # extra level is then never constructed or checked, so a malformed
        # ladder with (say) two presence controls could pass as complete.
        out['status'] = 'fail'
        out['violations'].append(
            'ladder levels must use each of treatment/alignment_placebo/'
            f'presence_control exactly once, got {sorted(roles.items())}')
    by_role = {v: k for k, v in roles.items()}
    treatment_name = by_role.get('treatment')
    absent_name = by_role.get('presence_control')
    misaligned_name = by_role.get('alignment_placebo')
    if treatment_name is None or absent_name is None or misaligned_name is None:
        out['status'] = 'fail'
        return out
    # Check the leave-one-out families that consume these controls.
    presence_pairs = _declared_pairs(design, 'content_pooling')
    alignment_pairs = _declared_pairs(design, 'content_alignment')
    if not presence_pairs:
        out['violations'].append(
            'ABLATION_DESIGN has no content_pooling contrast for the presence control')
    if not alignment_pairs:
        out['violations'].append(
            'ABLATION_DESIGN has no content_alignment contrast for the '
            'alignment placebo')
    out['component_design'] = {
        'content_pooling': sorted(presence_pairs or ()),
        'content_alignment': sorted(alignment_pairs or ()),
    }
    # Construct every level against its base and verify the exact diff.
    for base_label, base_arm in bases.items():
        base_rec = {'base_arm': base_arm}
        for level_name in (treatment_name, misaligned_name, absent_name):
            spec = levels[level_name]
            arms = spec.get('arms') if isinstance(spec, dict) else None
            arm = arms.get(base_label) if isinstance(arms, dict) else None
            rec = {'arm': arm, 'expected_mode': spec.get('content_mode')}
            base_rec[level_name] = rec
            if not isinstance(arm, str) or not arm.strip():
                out['violations'].append(
                    f'{level_name}/{base_label}: no single arm id declared')
                rec['status'] = 'invalid'
                continue
            if arm not in present_set:
                out['violations'].append(
                    f'{level_name}/{base_label}: arm {arm!r} is not scheduled in '
                    'the ablation panel')
                rec['status'] = 'missing'
                continue
            _keys, _per = _design_pair_cfg_keys(arm, base_arm, factory, n_layers)
            if isinstance(_keys, Exception):
                out['violations'].append(
                    f'{level_name}/{base_label}: cannot construct {arm!r} vs '
                    f'{base_arm!r} ({type(_keys).__name__}: {_keys})')
                rec['status'] = 'unconstructable'
                continue
            diff = sorted(_keys)
            rec['cfg_diff_keys'] = diff
            try:
                _actual_modes = _control_cfg_field_values(
                    factory, n_layers, arm, 'content_mode')
            except Exception as _me:
                out['violations'].append(
                    f'{level_name}/{base_label}: cannot read the constructed '
                    f'content_mode for {arm!r} ({type(_me).__name__}: {_me})')
                rec['status'] = 'unconstructable'
                continue
            rec['observed_modes'] = _actual_modes
            expected_mode = spec.get('content_mode')
            _bad_modes = sorted({str(v) for v in _actual_modes.values()
                                 if v != expected_mode})
            if level_name == treatment_name:
                bad = []
                if arm != base_arm:
                    bad.append(f'treatment arm {arm!r} must be the base arm '
                               f'{base_arm!r} (the aligned table is the reference)')
                if diff:
                    bad.append(f'treatment arm must not differ from the base '
                               f'config, got {diff}')
                if expected_mode not in CONTENT_MODES:
                    bad.append(f'declared content_mode {expected_mode!r} is not '
                               f'a declared mode {CONTENT_MODES}')
                if _bad_modes:
                    bad.append('constructed content_mode does not equal the '
                               f'declared {expected_mode!r}: {_bad_modes}')
                if bad:
                    out['violations'].append(
                        f'{level_name}/{base_label}: ' + '; '.join(bad))
                    rec['status'] = 'confounded'
                else:
                    rec['status'] = 'ok'
                continue
            bad = []
            if diff != ['content_mode']:
                bad.append(f'cfg diff is {diff}, not exactly ["content_mode"]')
            if expected_mode not in CONTENT_MODES:
                bad.append(f'expected content_mode {expected_mode!r} is not a '
                           f'declared mode {CONTENT_MODES}')
            if _bad_modes:
                bad.append(
                    'constructed content_mode does not equal the declared '
                    f'{expected_mode!r}: {_bad_modes}')
            if bad:
                out['violations'].append(
                    f'{level_name}/{base_label}: ' + '; '.join(bad))
                rec['status'] = 'confounded'
            else:
                rec['status'] = 'ok'
        out['base_checks'][base_label] = base_rec
    # Tie each control level to its declared leave-one-out family.
    for base_label, base_arm in bases.items():
        _mis = _declared_arm_id(levels, misaligned_name, base_label)
        _abs = _declared_arm_id(levels, absent_name, base_label)
        if presence_pairs is not None and (base_arm, _abs) not in presence_pairs:
            out['violations'].append(
                f'content_pooling does not declare {base_arm!r} vs {_abs!r} '
                'for the presence control')
        if alignment_pairs is not None and (base_arm, _mis) not in alignment_pairs:
            out['violations'].append(
                f'content_alignment does not declare {base_arm!r} vs {_mis!r} '
                'for the alignment placebo')
    # The declared intervention key set must be the same single field.
    for comp, pairs in (('content_pooling', presence_pairs),
                        ('content_alignment', alignment_pairs)):
        keys = ABLATION_INTERVENTION_KEYS.get(comp) if isinstance(
            ABLATION_INTERVENTION_KEYS, dict) else None
        if not isinstance(keys, dict) or not pairs:
            out['violations'].append(
                f'{comp}: no declared intervention-key mapping')
            continue
        for pair in pairs:
            kset = keys.get(pair)
            if kset != frozenset({'content_mode'}):
                out['violations'].append(
                    f'{comp}: contrast {pair} declares intervention keys '
                    f'{sorted(kset) if isinstance(kset, (set, frozenset)) else kset!r}, '
                    'expected exactly {"content_mode"}')
    # Distinct arm per level per base (a relabelled copy is not a control).
    for base_label in bases:
        seen = {}
        for level_name in (treatment_name, misaligned_name, absent_name):
            arm = _declared_arm_id(levels, level_name, base_label)
            if arm is not None:
                seen.setdefault(arm, []).append(level_name)
        for arm, names in seen.items():
            if len(names) > 1:
                out['violations'].append(
                    f'{base_label}: arm {arm!r} is reused by levels {names}; the '
                    'control is a relabelled copy, not an independent level')
    if out['violations']:
        out['status'] = 'fail'
    return out


# ---- attention-sink control ladder ------------------------------------------
# The on/off sink contrast changes two things at once: whether the sink is
# trained AND whether its parameter exists.  A module-usefulness claim from
# on/off alone therefore cannot distinguish "a learned sink redirects
# off-context mass" from "any extra per-head parameter helps".  The ladder
# below keeps the parameter shape/count and every other attention field fixed:
#
#   learned    -> treatment (the shippped arm, trainable sink)
#   randomized -> capacity placebo (same shape/count, frozen non-learned sink)
#   absent     -> absence control (no sink parameter; the old on/off control)
#
# `learned - randomized` isolates the learned component; `learned - absent`
# remains the total effect.  The random vector is derived from the layer's own
# initialized W_q without drawing from the global RNG, so the placebo does not
# perturb the initialization stream of any later tensor.
SINK_CONTROL_DESIGN = {
    'component': 'attention_sink_learning',
    'total_effect_component': 'attention_sink',
    'base_arms': {'fixed': 'csa_fixed', 'dynamic': 'csa_dynamic',
                  'dense': 'full_sink'},
    'absence_keys': {'fixed': frozenset({'use_sink'}),
                     'dynamic': frozenset({'use_sink'}),
                     'dense': frozenset({'full_sink'})},
    'levels': {
        'learned': {
            'role': 'treatment',
            'sink_mode': 'learned',
            'arms': {'fixed': 'csa_fixed', 'dynamic': 'csa_dynamic',
                     'dense': 'full_sink'},
            'note': 'the shipped trainable per-head sink',
        },
        'randomized': {
            'role': 'capacity_placebo',
            'sink_mode': 'random',
            'arms': {'fixed': 'csa_fix_randomsink',
                     'dynamic': 'csa_dyn_randomsink',
                     'dense': 'full_randomsink'},
            'note': ('identical sink shape/count/read budget, fixed non-learned '
                     'values, frozen: isolates learning from extra capacity'),
        },
        'absent': {
            'role': 'absence_control',
            'sink_mode': 'learned',
            'arms': {'fixed': 'csa_fix_nosink', 'dynamic': 'csa_dyn_nosink',
                     'dense': 'full'},
            'note': ('no sink parameter at all; changes capacity as well as the '
                     'mechanism, so it is the total-effect control only'),
        },
    },
    'note': ('the on/off control alone cannot support a *learned* sink claim; '
             'the frozen-random placebo is the capacity-matched control for it'),
}


def sink_control_audit(design=None, ladder=None, present=None, factory=None,
                       n_layers=2):
    """Zero-GPU gate for the attention-sink control ladder.

    For every declared base (fixed, dynamic, dense) the audit constructs the
    real arm configs, checks the exact field diff against the base, builds the
    tiny model to compare TOTAL and sink parameter counts, and checks that the
    capacity placebo's sink parameters are frozen.  The two leave-one-out
    families that consume the controls must be declared in ``ABLATION_DESIGN``
    with exactly the expected intervention keys.

    A missing level/arm, a relabelled copy, a second knob in the placebo diff,
    an unfrozen placebo, a parameter mismatch or an undeclared family is
    ``fail`` (never a pass).  The audit never reads a result file.
    """
    design = ABLATION_DESIGN if design is None else design
    ladder = SINK_CONTROL_DESIGN if ladder is None else ladder
    factory = make_layer_cfgs if factory is None else factory
    _nl = _as_seed_int(n_layers)
    if _nl is None or _nl < 1:
        raise ValueError(
            f'sink_control_audit: n_layers must be a positive integer, got {n_layers!r}')
    n_layers = int(_nl)
    out = {'kind': 'sink_control_ladder', 'status': 'ok',
           'n_layers': n_layers, 'levels': {}, 'bases': {},
           'base_checks': {}, 'component_design': {}, 'violations': [],
           'warnings': [],
           'treatment_component': 'attention_sink_learning',
           'total_effect_component': 'attention_sink',
           'note': ('learned - randomized isolates the learned component at '
                    'matched capacity; learned - absent is the total effect')}
    if not isinstance(ladder, dict) or not ladder:
        out['status'] = 'fail'
        out['violations'].append(
            f'sink control ladder must be a non-empty mapping, got '
            f'{type(ladder).__name__}')
        return out
    bases = ladder.get('base_arms')
    levels = ladder.get('levels')
    absence_keys = ladder.get('absence_keys')
    if (not isinstance(bases, dict) or not bases
            or not all(isinstance(k, str) and k.strip() and isinstance(v, str)
                       and v.strip() for k, v in bases.items())):
        out['status'] = 'fail'
        out['violations'].append('base_arms must map non-empty base labels to arm ids')
        return out
    if not isinstance(levels, dict) or not levels:
        out['status'] = 'fail'
        out['violations'].append('levels must be a non-empty mapping')
        return out
    if not isinstance(absence_keys, dict) or set(absence_keys) != set(bases):
        out['status'] = 'fail'
        out['violations'].append(
            'absence_keys must name exactly the declared base labels')
        return out
    _absence_norm = {}
    for _b, _ks in absence_keys.items():
        if not isinstance(_ks, (set, frozenset)) or not _ks:
            out['status'] = 'fail'
            out['violations'].append(
                f'absence_keys[{_b!r}] must be a non-empty set of cfg keys')
            _absence_norm[_b] = frozenset()
        elif not all(isinstance(_k, str) and _k.strip() for _k in _ks):
            out['status'] = 'fail'
            out['violations'].append(
                f'absence_keys[{_b!r}] must contain non-empty cfg field names')
            _absence_norm[_b] = frozenset()
        else:
            _absence_norm[_b] = frozenset(_ks)
    absence_keys = _absence_norm
    out['bases'] = dict(bases)
    try:
        present_set = set(RUN_ABL['variants']) if present is None else set(present)
    except TypeError:
        out['status'] = 'fail'
        out['violations'].append('present arm ids must be a collection of strings')
        return out
    roles = {}
    for name, spec in levels.items():
        if not isinstance(spec, dict):
            out['violations'].append(f'{name}: level must be a mapping')
            continue
        roles[name] = str(spec.get('role') or '')
        _arms = spec.get('arms')
        _arms_norm = dict(_arms) if isinstance(_arms, dict) and _arms else _arms
        if not isinstance(_arms, dict) or not _arms:
            out['violations'].append(
                f'{name}: arms must be a non-empty mapping base label -> arm id')
        out['levels'][str(name)] = {
            'role': spec.get('role'),
            'sink_mode': spec.get('sink_mode'),
            'arms': _arms_norm,
        }
    expected_roles = {'treatment', 'capacity_placebo', 'absence_control'}
    if set(roles.values()) != expected_roles:
        out['status'] = 'fail'
        out['violations'].append(
            'ladder roles must be exactly treatment/capacity_placebo/'
            f'absence_control, got {sorted(roles.values())}')
    elif len(roles) != len(expected_roles):
        out['status'] = 'fail'
        out['violations'].append(
            'ladder levels must use each of treatment/capacity_placebo/'
            f'absence_control exactly once, got {sorted(roles.items())}')
    by_role = {v: k for k, v in roles.items()}
    treatment_name = by_role.get('treatment')
    placebo_name = by_role.get('capacity_placebo')
    absent_name = by_role.get('absence_control')
    if treatment_name is None or placebo_name is None or absent_name is None:
        out['status'] = 'fail'
        return out

    treatment_pairs = _declared_pairs(design, 'attention_sink_learning')
    total_pairs = _declared_pairs(design, 'attention_sink')
    if not treatment_pairs:
        out['violations'].append(
            'ABLATION_DESIGN has no attention_sink_learning contrast for the '
            'capacity placebo')
    if not total_pairs:
        out['violations'].append(
            'ABLATION_DESIGN has no attention_sink contrast for the absence control')
    out['component_design'] = {
        'attention_sink_learning': sorted(treatment_pairs or ()),
        'attention_sink': sorted(total_pairs or ()),
    }

    def _model_counts(arm):
        # This is a zero-GPU design probe.  It must not perturb the caller's
        # RNG streams: a later training cell would otherwise start from a
        # different data order/dropout stream merely because an audit ran
        # first.  `_preserve_rng_state` restores Python/NumPy/Torch/CUDA (and
        # the hash-seed environment) even on error.
        with _preserve_rng_state():
            torch.manual_seed(1234)
            model = SmallGPT(32, 16, n_layers, 2, 8, 16, factory(n_layers, arm))
            named = dict(model.named_parameters())
            total = sum(int(p.numel()) for p in named.values())
            sink_names = [n for n in named if n == 'sink' or n.endswith('.sink')]
            sink = sum(int(named[n].numel()) for n in sink_names)
            frozen = bool(sink_names) and all(
                not bool(named[n].requires_grad) for n in sink_names)
            return {'total': total, 'sink_params': sink, 'sink_frozen': frozen,
                    'n_sink_tensors': len(sink_names)}

    # Construct every level against its base and verify the exact diff.
    for base_label, base_arm in bases.items():
        base_rec = {'base_arm': base_arm, 'expected_absence_keys':
                    sorted(absence_keys[base_label])}
        for level_name in (treatment_name, placebo_name, absent_name):
            spec = levels[level_name]
            arms = spec.get('arms') if isinstance(spec, dict) else None
            arm = arms.get(base_label) if isinstance(arms, dict) else None
            rec = {'arm': arm, 'expected_mode': spec.get('sink_mode')}
            base_rec[level_name] = rec
            if not isinstance(arm, str) or not arm.strip():
                out['violations'].append(
                    f'{level_name}/{base_label}: no single arm id declared')
                rec['status'] = 'invalid'
                continue
            if arm not in present_set:
                out['violations'].append(
                    f'{level_name}/{base_label}: arm {arm!r} is not scheduled in '
                    'the ablation panel')
                rec['status'] = 'missing'
                continue
            _keys, _per = _design_pair_cfg_keys(arm, base_arm, factory, n_layers)
            if isinstance(_keys, Exception):
                out['violations'].append(
                    f'{level_name}/{base_label}: cannot construct {arm!r} vs '
                    f'{base_arm!r} ({type(_keys).__name__}: {_keys})')
                rec['status'] = 'unconstructable'
                continue
            diff = sorted(_keys)
            rec['cfg_diff_keys'] = diff
            try:
                _actual_modes = _control_cfg_field_values(
                    factory, n_layers, arm, 'sink_mode')
            except Exception as _me:
                out['violations'].append(
                    f'{level_name}/{base_label}: cannot read the constructed '
                    f'sink_mode for {arm!r} ({type(_me).__name__}: {_me})')
                rec['status'] = 'unconstructable'
                continue
            rec['observed_sink_modes'] = _actual_modes
            expected_mode = spec.get('sink_mode')
            _bad_modes = sorted({str(v) for v in _actual_modes.values()
                                 if v != expected_mode})
            if level_name == treatment_name:
                bad = []
                if arm != base_arm:
                    bad.append(f'treatment arm {arm!r} must be the base arm '
                               f'{base_arm!r}')
                if diff:
                    bad.append(f'treatment arm must not differ from the base '
                               f'config, got {diff}')
                if expected_mode != 'learned':
                    bad.append(f'treatment must declare sink_mode="learned", '
                               f'got {expected_mode!r}')
                if _bad_modes:
                    bad.append('constructed sink_mode does not equal the '
                               f'declared {expected_mode!r}: {_bad_modes}')
                if bad:
                    out['violations'].append(
                        f'{level_name}/{base_label}: ' + '; '.join(bad))
                    rec['status'] = 'confounded'
                else:
                    rec['status'] = 'ok'
                continue
            if level_name == placebo_name:
                bad = []
                if diff != ['sink_mode']:
                    bad.append(f'cfg diff is {diff}, not exactly ["sink_mode"]')
                if expected_mode != 'random':
                    bad.append(
                        f'capacity placebo must declare sink_mode="random", got '
                        f'{expected_mode!r}')
                if _bad_modes:
                    bad.append('constructed sink_mode is not "random": '
                               f'{_bad_modes}')
                if bad:
                    out['violations'].append(
                        f'{level_name}/{base_label}: ' + '; '.join(bad))
                    rec['status'] = 'confounded'
                else:
                    rec['status'] = 'ok'
                continue
            # absence control
            bad = []
            expected = absence_keys[base_label]
            if set(diff) != set(expected):
                bad.append(f'cfg diff is {diff}, not the declared absence '
                           f'key(s) {sorted(expected)}')
            if expected_mode != 'learned':
                bad.append(f'absence control must declare '
                           f'sink_mode="learned", got {expected_mode!r}')
            if _bad_modes:
                bad.append('constructed sink_mode is not "learned": '
                           f'{_bad_modes}')
            if bad:
                out['violations'].append(
                    f'{level_name}/{base_label}: ' + '; '.join(bad))
                rec['status'] = 'confounded'
            else:
                rec['status'] = 'ok'
        # Parameter-count checks need the built models, not just the cfg diff.
        try:
            counts = {
                level_name: _model_counts(
                    levels[level_name]['arms'][base_label])
                for level_name in (treatment_name, placebo_name, absent_name)
            }
        except Exception as e:  # fail closed: an unbuildable arm is a defect
            out['violations'].append(
                f'{base_label}: cannot build the three ladder arms '
                f'({type(e).__name__}: {e})')
            out['base_checks'][base_label] = base_rec
            continue
        for level_name, c in counts.items():
            base_rec[level_name]['params'] = c['total']
            base_rec[level_name]['sink_params'] = c['sink_params']
            base_rec[level_name]['n_sink_tensors'] = c['n_sink_tensors']
        _t, _p, _a = (counts[treatment_name], counts[placebo_name],
                      counts[absent_name])
        base_rec['params_checked'] = True
        base_rec['treatment_placebo_matched'] = _t['total'] == _p['total']
        base_rec['treatment_absence_delta'] = _t['total'] - _a['total']
        base_rec['expected_absence_delta'] = _t['sink_params']
        if not base_rec['treatment_placebo_matched']:
            out['violations'].append(
                f'{base_label}: treatment and capacity placebo differ in total '
                f'parameters ({_t["total"]} vs {_p["total"]}); the placebo is '
                'not capacity-matched')
        if _t['sink_params'] != _p['sink_params']:
            out['violations'].append(
                f'{base_label}: treatment and capacity placebo differ in sink '
                f'parameters ({_t["sink_params"]} vs {_p["sink_params"]})')
        if _p['n_sink_tensors'] != _t['n_sink_tensors']:
            out['violations'].append(
                f'{base_label}: capacity placebo has a different number of sink '
                f'tensors ({_p["n_sink_tensors"]} vs {_t["n_sink_tensors"]})')
        if not _p['sink_frozen']:
            out['violations'].append(
                f'{base_label}: capacity placebo sink parameters are trainable; '
                'a learned placebo is not a placebo')
        if _t['n_sink_tensors'] and _t['sink_frozen']:
            out['violations'].append(
                f'{base_label}: treatment sink parameters are frozen; the '
                '"learned" level cannot be a treatment')
        if _t['total'] - _a['total'] != _t['sink_params']:
            out['violations'].append(
                f'{base_label}: absence control removes '
                f'{_t["total"] - _a["total"]} parameters but the sink accounts '
                f'for {_t["sink_params"]}; the on/off contrast changed more than '
                'the sink capacity')
        out['base_checks'][base_label] = base_rec

    # Tie each control level to its declared leave-one-out family.
    for base_label, base_arm in bases.items():
        _p = _declared_arm_id(levels, placebo_name, base_label)
        _a = _declared_arm_id(levels, absent_name, base_label)
        if treatment_pairs is not None and (base_arm, _p) not in treatment_pairs:
            out['violations'].append(
                f'attention_sink_learning does not declare {base_arm!r} vs {_p!r} '
                'for the capacity placebo')
        if total_pairs is not None and (base_arm, _a) not in total_pairs:
            out['violations'].append(
                f'attention_sink does not declare {base_arm!r} vs {_a!r} for the '
                'absence control')
    # Declared intervention keys must be the same single field for the
    # placebo contrast and exactly the absence key set for the on/off contrast.
    keys = ABLATION_INTERVENTION_KEYS if isinstance(
        ABLATION_INTERVENTION_KEYS, dict) else {}
    _pkeys = keys.get('attention_sink_learning')
    if not isinstance(_pkeys, dict):
        out['violations'].append(
            'attention_sink_learning: no declared intervention-key mapping')
    else:
        for base_label, base_arm in bases.items():
            _p = _declared_arm_id(levels, placebo_name, base_label)
            kset = _pkeys.get((base_arm, _p))
            if kset != frozenset({'sink_mode'}):
                out['violations'].append(
                    f'attention_sink_learning: contrast {(base_arm, _p)} declares '
                    f'intervention keys '
                    f'{sorted(kset) if isinstance(kset, (set, frozenset)) else kset!r}, '
                    'expected exactly {"sink_mode"}')
    _akeys = keys.get('attention_sink')
    if not isinstance(_akeys, dict):
        out['violations'].append(
            'attention_sink: no declared intervention-key mapping')
    else:
        for base_label, base_arm in bases.items():
            _a = _declared_arm_id(levels, absent_name, base_label)
            kset = _akeys.get((base_arm, _a))
            if kset != frozenset(absence_keys[base_label]):
                out['violations'].append(
                    f'attention_sink: contrast {(base_arm, _a)} declares '
                    f'intervention keys '
                    f'{sorted(kset) if isinstance(kset, (set, frozenset)) else kset!r}, '
                    f'expected {sorted(absence_keys[base_label])}')
    # Distinct arm per level per base (a relabelled copy is not a control).
    for base_label in bases:
        seen = {}
        for level_name in (treatment_name, placebo_name, absent_name):
            arm = _declared_arm_id(levels, level_name, base_label)
            if arm is not None:
                seen.setdefault(arm, []).append(level_name)
        for arm, names in seen.items():
            if len(names) > 1:
                out['violations'].append(
                    f'{base_label}: arm {arm!r} is reused by levels {names}; the '
                    'control is a relabelled copy, not an independent level')
    if out['violations']:
        out['status'] = 'fail'
    return out


# ---- boundary-gate control ladder ------------------------------------------
# The dynamic-vs-fixed contrast changes the cut geometry, adds the per-layer
# delta_logit offset and (through comp_reg) applies a different training
# objective.  It is the total effect, not a learned-gate single factor.  This
# ladder keeps the parameter, its initial value, the parameter count and the
# comp_reg term identical and freezes only its gradient:
#
#   learned -> treatment (the shipped trainable boundary offset)
#   frozen  -> capacity/objective placebo (same tensor, frozen at init)
#
# `learned - frozen` isolates the learned boundary-gate component at matched
# capacity and identical compression objective.  The frozen arm consumes no
# RNG, so every non-gate parameter is bit-identical at the same seed.
GATE_CONTROL_DESIGN = {
    'component': 'dynamic_blocking_learning',
    'total_effect_component': 'dynamic_blocking',
    'base_arms': {'csa': 'csa_dynamic', 'hybrid': 'hybrid_dynamic'},
    'levels': {
        'learned': {
            'role': 'treatment',
            'gate_mode': 'learned',
            'arms': {'csa': 'csa_dynamic', 'hybrid': 'hybrid_dynamic'},
            'note': 'the shipped trainable boundary offset and comp_reg term',
        },
        'frozen': {
            'role': 'capacity_placebo',
            'gate_mode': 'frozen',
            'arms': {'csa': 'csa_dyn_frozengate',
                     'hybrid': 'hybrid_dyn_frozengate'},
            'note': ('identical delta_logit parameter, initial value and '
                     'comp_reg term, frozen: isolates learning from capacity '
                     'and from the objective term'),
        },
    },
    'note': ('the dynamic on/off contrast alone cannot support a *learned* '
             'boundary claim; the frozen-offset placebo is the capacity- and '
             'objective-matched control for it'),
}


def gate_control_audit(design=None, ladder=None, present=None, factory=None,
                       n_layers=2, seed=1234):
    """Zero-GPU gate for the learned/frozen boundary-gate control ladder.

    For every declared base the audit constructs the real arm configs and
    checks that the placebo differs in exactly ``gate_mode``, that both arms
    are scheduled, that total and delta parameter counts match, that the
    delta_logit initial values match, that only the treatment's offset is
    trainable, and that the actual training objectives agree except for the
    declared trainability/delta-LR difference.  It also verifies that
    ``ABLATION_DESIGN['dynamic_blocking_learning']`` declares the same pair
    with exactly the ``gate_mode`` intervention key.

    A missing level/arm, a relabelled copy, a second knob in the placebo diff,
    an unfrozen placebo, a parameter/objective mismatch or an undeclared
    family is ``fail`` (never a pass).  The audit never reads a result file.
    """
    design = ABLATION_DESIGN if design is None else design
    ladder = GATE_CONTROL_DESIGN if ladder is None else ladder
    factory = make_layer_cfgs if factory is None else factory
    _nl = _as_seed_int(n_layers)
    if _nl is None or _nl < 1:
        raise ValueError(
            f'gate_control_audit: n_layers must be a positive integer, got {n_layers!r}')
    n_layers = int(_nl)
    _seed = _as_seed_int(seed)
    if _seed is None or _seed < 0:
        raise ValueError(
            f'gate_control_audit: seed must be a non-negative integer, got {seed!r}')
    seed = int(_seed)
    out = {'kind': 'gate_control_ladder', 'status': 'ok',
           'n_layers': n_layers, 'seed': seed, 'levels': {}, 'bases': {},
           'base_checks': {}, 'component_design': {}, 'violations': [],
           'warnings': [],
           'treatment_component': 'dynamic_blocking_learning',
           'total_effect_component': 'dynamic_blocking',
           'note': ('learned - frozen isolates the learned boundary-gate '
                    'component at matched capacity and identical comp_reg '
                    'objective; dynamic on/off is the total effect')}
    if not isinstance(ladder, dict) or not ladder:
        out['status'] = 'fail'
        out['violations'].append(
            f'gate control ladder must be a non-empty mapping, got '
            f'{type(ladder).__name__}')
        return out
    bases = ladder.get('base_arms')
    levels = ladder.get('levels')
    if (not isinstance(bases, dict) or not bases
            or not all(isinstance(k, str) and k.strip() and isinstance(v, str)
                       and v.strip() for k, v in bases.items())):
        out['status'] = 'fail'
        out['violations'].append('base_arms must map non-empty base labels to arm ids')
        return out
    if not isinstance(levels, dict) or not levels:
        out['status'] = 'fail'
        out['violations'].append('levels must be a non-empty mapping')
        return out
    out['bases'] = dict(bases)
    try:
        present_set = set(RUN_ABL['variants']) if present is None else set(present)
    except TypeError:
        out['status'] = 'fail'
        out['violations'].append('present arm ids must be a collection of strings')
        return out
    roles = {}
    for name, spec in levels.items():
        if not isinstance(spec, dict):
            out['violations'].append(f'{name}: level must be a mapping')
            continue
        roles[name] = str(spec.get('role') or '')
        _arms = spec.get('arms')
        _arms_norm = dict(_arms) if isinstance(_arms, dict) and _arms else _arms
        if not isinstance(_arms, dict) or not _arms:
            out['violations'].append(
                f'{name}: arms must be a non-empty mapping base label -> arm id')
        out['levels'][str(name)] = {
            'role': spec.get('role'),
            'gate_mode': spec.get('gate_mode'),
            'arms': _arms_norm,
        }
    expected_roles = {'treatment', 'capacity_placebo'}
    if set(roles.values()) != expected_roles:
        out['status'] = 'fail'
        out['violations'].append(
            'ladder roles must be exactly treatment/capacity_placebo, got '
            f'{sorted(roles.values())}')
    elif len(roles) != len(expected_roles):
        out['status'] = 'fail'
        out['violations'].append(
            'ladder levels must use each of treatment/capacity_placebo exactly '
            f'once, got {sorted(roles.items())}')
    by_role = {v: k for k, v in roles.items()}
    treatment_name = by_role.get('treatment')
    placebo_name = by_role.get('capacity_placebo')
    if treatment_name is None or placebo_name is None:
        out['status'] = 'fail'
        return out

    pairs = _declared_pairs(design, 'dynamic_blocking_learning')
    if not pairs:
        out['violations'].append(
            'ABLATION_DESIGN has no dynamic_blocking_learning contrast for the '
            'learned/frozen gate ladder')
    out['component_design'] = {
        'dynamic_blocking_learning': sorted(pairs or ()),
        'dynamic_blocking': sorted(_declared_pairs(design, 'dynamic_blocking') or ()),
    }

    def _model(arm):
        _prev = torch.get_rng_state()
        try:
            torch.manual_seed(seed)
            m = SmallGPT(32, 16, n_layers, 2, 8, 16, factory(n_layers, arm))
        finally:
            torch.set_rng_state(_prev)
        return m

    for base_label, base_arm in bases.items():
        base_rec = {'base_arm': base_arm}
        for level_name in (treatment_name, placebo_name):
            spec = levels[level_name]
            arms = spec.get('arms') if isinstance(spec, dict) else None
            arm = arms.get(base_label) if isinstance(arms, dict) else None
            expected_mode = spec.get('gate_mode') if isinstance(spec, dict) else None
            rec = {'arm': arm, 'expected_gate_mode': expected_mode}
            base_rec[level_name] = rec
            if not isinstance(arm, str) or not arm.strip():
                out['violations'].append(
                    f'{level_name}/{base_label}: no single arm id declared')
                rec['status'] = 'invalid'
                continue
            if arm not in present_set:
                out['violations'].append(
                    f'{level_name}/{base_label}: arm {arm!r} is not scheduled in '
                    'the ablation panel')
                rec['status'] = 'missing'
                continue
            _keys, _per = _design_pair_cfg_keys(arm, base_arm, factory, n_layers)
            if isinstance(_keys, Exception):
                out['violations'].append(
                    f'{level_name}/{base_label}: cannot construct {arm!r} vs '
                    f'{base_arm!r} ({type(_keys).__name__}: {_keys})')
                rec['status'] = 'unconstructable'
                continue
            diff = sorted(_keys)
            rec['cfg_diff_keys'] = diff
            try:
                _actual_modes = _control_cfg_field_values(
                    factory, n_layers, arm, 'gate_mode')
            except Exception as _me:
                out['violations'].append(
                    f'{level_name}/{base_label}: cannot read the constructed '
                    f'gate_mode for {arm!r} ({type(_me).__name__}: {_me})')
                rec['status'] = 'unconstructable'
                continue
            rec['observed_gate_modes'] = _actual_modes
            _bad_modes = sorted({str(v) for v in _actual_modes.values()
                                 if v != expected_mode})
            bad = []
            if expected_mode not in GATE_MODES:
                bad.append(f'declared gate_mode {expected_mode!r} is not a '
                           f'declared mode {GATE_MODES}')
            if _bad_modes:
                bad.append(
                    'constructed gate_mode does not equal the declared '
                    f'{expected_mode!r}: {_bad_modes}')
            if level_name == treatment_name:
                if arm != base_arm:
                    bad.append(f'treatment arm {arm!r} must be the base arm '
                               f'{base_arm!r}')
                if diff:
                    bad.append(f'treatment arm must not differ from the base '
                               f'config, got {diff}')
            else:
                if diff != ['gate_mode']:
                    bad.append(f'cfg diff is {diff}, not exactly ["gate_mode"]')
            if bad:
                out['violations'].append(
                    f'{level_name}/{base_label}: ' + '; '.join(bad))
                rec['status'] = 'confounded'
                continue
            rec['status'] = 'ok'
        # Construct both models once per base and check capacity, initialization
        # and objective identity.
        treat = base_rec.get(treatment_name) or {}
        plac = base_rec.get(placebo_name) or {}
        ta, pa = treat.get('arm'), plac.get('arm')
        if not (isinstance(ta, str) and isinstance(pa, str)):
            continue
        try:
            ma, mb = _model(ta), _model(pa)
        except Exception as _e:
            out['violations'].append(
                f'{base_label}: model construction failed for {ta!r}/{pa!r} '
                f'({type(_e).__name__}: {_e})')
            continue
        sa = dict(ma.named_parameters())
        sb = dict(mb.named_parameters())
        ta_params = sum(int(p.numel()) for p in sa.values())
        tb_params = sum(int(p.numel()) for p in sb.values())
        delta_a = sorted(n for n in sa if 'delta_logit' in n)
        delta_b = sorted(n for n in sb if 'delta_logit' in n)
        rec = {'a': ta, 'b': pa, 'a_params': ta_params, 'b_params': tb_params,
               'delta_params_a': sum(int(sa[n].numel()) for n in delta_a),
               'delta_params_b': sum(int(sb[n].numel()) for n in delta_b),
               'delta_tensors_a': delta_a, 'delta_tensors_b': delta_b}
        base_rec['capacity_check'] = rec
        bad = []
        if ta_params != tb_params:
            bad.append(f'total parameters differ ({ta_params} vs {tb_params})')
        if delta_a != delta_b:
            bad.append(f'delta_logit tensors differ ({delta_a} vs {delta_b})')
        if not delta_a:
            bad.append('treatment has no delta_logit parameter to freeze')
        if set(sa) != set(sb):
            bad.append('parameter names differ outside delta_logit')
        for n in set(sa) & set(sb):
            if n in delta_a:
                continue
            if not torch.equal(sa[n], sb[n]):
                bad.append(f'non-gate parameter {n!r} differs at the same seed')
                break
        for n in delta_b:
            if bool(sb[n].requires_grad):
                bad.append(f'placebo delta parameter {n!r} is still trainable')
                break
        for n in delta_a:
            if not bool(sa[n].requires_grad):
                bad.append(f'treatment delta parameter {n!r} is unexpectedly frozen')
                break
        if delta_a == delta_b and delta_a:
            for n in delta_a:
                if sa[n].shape != sb[n].shape:
                    bad.append(f'delta parameter {n!r} shape differs')
                    break
                if not torch.equal(sa[n], sb[n]):
                    bad.append(f'delta parameter {n!r} does not start at the '
                               'same frozen init value')
                    break
        # Objectives: capacity and the compression term must match; only the
        # declared trainability / effective delta LR may differ.
        try:
            oa = train_objective(ma, comp_lambda=0.05, delta_lr_mult=10.0)
            ob = train_objective(mb, comp_lambda=0.05, delta_lr_mult=10.0)
        except Exception as _oe:
            bad.append(f'objective descriptor failed ({type(_oe).__name__}: {_oe})')
            oa = ob = {}
        if oa and ob:
            _must_match = ('comp_reg_applies', 'comp_lambda', 'comp_reg_layers',
                           'comp_reg_configured_layers', 'comp_lambda_mults',
                           'comp_lambda_effective', 'delta_param_count',
                           'boundary_policy', 'loss_terms')
            for _k in _must_match:
                if oa.get(_k) != ob.get(_k):
                    bad.append(f'objective field {_k!r} differs '
                               f'({oa.get(_k)!r} vs {ob.get(_k)!r})')
            if oa.get('delta_trainable') is not True:
                bad.append('treatment objective does not report a trainable delta gate')
            if ob.get('delta_trainable') is not False:
                bad.append('placebo objective still reports a trainable delta gate')
            if oa.get('delta_lr_effective') == ob.get('delta_lr_effective'):
                bad.append('delta LR effective did not change with trainability')
        if bad:
            out['violations'].append(f'{base_label}: ' + '; '.join(bad))
            rec['status'] = 'confounded'
        else:
            rec['status'] = 'ok'
        out['base_checks'][base_label] = base_rec
    # Distinct arm per level per base (a relabelled copy is not a control).
    for base_label in bases:
        seen = {}
        for level_name in (treatment_name, placebo_name):
            _arms = (levels.get(level_name) or {}).get('arms') if isinstance(
                levels.get(level_name), dict) else None
            arm = _arms.get(base_label) if isinstance(_arms, dict) else None
            if isinstance(arm, str):
                seen.setdefault(arm, []).append(level_name)
        for arm, names in seen.items():
            if len(names) > 1:
                out['violations'].append(
                    f'{base_label}: arm {arm!r} is reused by levels {names}; the '
                    'placebo is a relabelled treatment, not an independent level')
    # Declared contrast and intervention-key contract.
    keys = ABLATION_INTERVENTION_KEYS.get('dynamic_blocking_learning') if isinstance(
        ABLATION_INTERVENTION_KEYS, dict) else None
    if not isinstance(keys, dict) or not pairs:
        out['violations'].append(
            'dynamic_blocking_learning: no declared intervention-key mapping')
    else:
        for base_label, base_arm in bases.items():
            arms_t = (levels.get(treatment_name) or {}).get('arms') if isinstance(
                levels.get(treatment_name), dict) else None
            arms_p = (levels.get(placebo_name) or {}).get('arms') if isinstance(
                levels.get(placebo_name), dict) else None
            ta = arms_t.get(base_label) if isinstance(arms_t, dict) else None
            pa = arms_p.get(base_label) if isinstance(arms_p, dict) else None
            if isinstance(ta, str) and isinstance(pa, str) and (ta, pa) not in pairs:
                out['violations'].append(
                    f'dynamic_blocking_learning does not declare {ta!r} vs '
                    f'{pa!r} for the frozen-gate control')
            kset = keys.get((ta, pa)) if isinstance(ta, str) and isinstance(pa, str) else None
            if kset != frozenset({'gate_mode'}):
                out['violations'].append(
                    f'dynamic_blocking_learning: contrast {(ta, pa)!r} declares '
                    f'intervention keys '
                    f'{sorted(kset) if isinstance(kset, (set, frozenset)) else kset!r}, '
                    'expected exactly {"gate_mode"}')
    if out['violations']:
        out['status'] = 'fail'
    return out


# ---- local-fusion control ladder -------------------------------------------
# The on/off fusion contrast changes two things at once: whether the depth-wise
# convolution is trained AND whether its parameters exist.  A "learned fusion
# helps" claim from on/off alone therefore cannot distinguish a learned kernel
# from "any extra capacity helps".  The ladder below keeps the conv shape/count,
# read budget and every other attention field fixed:
#
#   learned    -> treatment (the shipped arm, trainable fusion kernel)
#   randomized -> capacity placebo (same shape/count, frozen random kernel)
#   absent     -> absence control (identity fusion, no conv parameter)
#
# `learned - randomized` isolates the learned component; `learned - absent`
# remains the total effect.  Both the learned and the random Conv1d are built
# without consuming the global torch RNG, so all three arms start from the same
# weights for every non-fusion tensor.  Without that, merely adding the conv
# would shift the initialization of the later block parameters and the
# comparison would be multi-factor by construction.
FUSION_CONTROL_DESIGN = {
    'component': 'local_fusion_learning',
    'total_effect_component': 'local_fusion',
    'base_arms': {'csa': 'csa_dyn_fuse',
                  'hybrid': 'hybrid_csa_dyn_fuse'},
    'absence_keys': {'csa': frozenset({'fuse_mode'}),
                     'hybrid': frozenset({'fuse_mode'})},
    'levels': {
        'learned': {
            'role': 'treatment',
            'fuse_mode': 'learned',
            'arms': {'csa': 'csa_dyn_fuse',
                     'hybrid': 'hybrid_csa_dyn_fuse'},
            'note': 'the shipped trainable depth-wise fusion convolution',
        },
        'randomized': {
            'role': 'capacity_placebo',
            'fuse_mode': 'random',
            'arms': {'csa': 'csa_dyn_randfuse',
                     'hybrid': 'hybrid_csa_dyn_randfuse'},
            'note': ('identical conv shape/count/read budget, fixed non-learned '
                     'kernel, frozen: isolates learning from extra capacity'),
        },
        'absent': {
            'role': 'absence_control',
            'fuse_mode': 'none',
            'arms': {'csa': 'csa_dynamic', 'hybrid': 'hybrid_csa_dyn'},
            'note': ('identity fusion, no conv parameter; changes capacity as '
                     'well as the mechanism, so it is the total-effect control'),
        },
    },
    'note': ('the on/off control alone cannot support a *learned* fusion claim; '
             'the frozen-random placebo is the capacity-matched control for it'),
}


def fusion_control_audit(design=None, ladder=None, present=None, factory=None,
                         n_layers=2):
    """Zero-GPU gate for the local-fusion control ladder.

    For every declared base (pure CSA and hybrid) the audit constructs the real
    arm configs, checks that each non-absent level differs from the base in
    exactly ``fuse_mode`` (and that the field really takes the declared value
    on every fusable layer), builds tiny models to compare total and fusion
    parameter counts, verifies the capacity placebo's kernel is frozen, and
    checks that all non-fusion parameters start identically for the three
    levels at the same seed.  The two leave-one-out families that consume the
    controls must be declared in ``ABLATION_DESIGN`` with exactly the expected
    intervention keys.

    A missing level/arm, a relabelled copy, a second knob in the placebo diff,
    an unfrozen placebo, a parameter mismatch, a shifted non-fusion
    initialization or an undeclared family is ``fail`` (never a pass).  The
    audit never reads a result file.
    """
    design = ABLATION_DESIGN if design is None else design
    ladder = FUSION_CONTROL_DESIGN if ladder is None else ladder
    factory = make_layer_cfgs if factory is None else factory
    _nl = _as_seed_int(n_layers)
    if _nl is None or _nl < 1:
        raise ValueError(
            f'fusion_control_audit: n_layers must be a positive integer, got {n_layers!r}')
    n_layers = int(_nl)
    out = {'kind': 'fusion_control_ladder', 'status': 'ok',
           'n_layers': n_layers, 'levels': {}, 'bases': {},
           'base_checks': {}, 'component_design': {}, 'violations': [],
           'warnings': [],
           'treatment_component': 'local_fusion_learning',
           'total_effect_component': 'local_fusion',
           'note': ('learned - randomized isolates the learned component at '
                    'matched capacity; learned - absent is the total effect')}
    if not isinstance(ladder, dict) or not ladder:
        out['status'] = 'fail'
        out['violations'].append(
            f'fusion control ladder must be a non-empty mapping, got '
            f'{type(ladder).__name__}')
        return out
    bases = ladder.get('base_arms')
    levels = ladder.get('levels')
    absence_keys = ladder.get('absence_keys')
    if (not isinstance(bases, dict) or not bases
            or not all(isinstance(k, str) and k.strip() and isinstance(v, str)
                       and v.strip() for k, v in bases.items())):
        out['status'] = 'fail'
        out['violations'].append('base_arms must map non-empty base labels to arm ids')
        return out
    if not isinstance(levels, dict) or not levels:
        out['status'] = 'fail'
        out['violations'].append('levels must be a non-empty mapping')
        return out
    if not isinstance(absence_keys, dict) or set(absence_keys) != set(bases):
        out['status'] = 'fail'
        out['violations'].append(
            'absence_keys must name exactly the declared base labels')
        return out
    _absence_norm = {}
    for _b, _ks in absence_keys.items():
        if not isinstance(_ks, (set, frozenset)) or not _ks:
            out['status'] = 'fail'
            out['violations'].append(
                f'absence_keys[{_b!r}] must be a non-empty set of cfg keys')
            _absence_norm[_b] = frozenset()
        elif not all(isinstance(_k, str) and _k.strip() for _k in _ks):
            out['status'] = 'fail'
            out['violations'].append(
                f'absence_keys[{_b!r}] must contain non-empty cfg field names')
            _absence_norm[_b] = frozenset()
        else:
            _absence_norm[_b] = frozenset(_ks)
    absence_keys = _absence_norm
    out['bases'] = dict(bases)
    try:
        present_set = set(RUN_ABL['variants']) if present is None else set(present)
    except TypeError:
        out['status'] = 'fail'
        out['violations'].append('present arm ids must be a collection of strings')
        return out
    roles = {}
    for name, spec in levels.items():
        if not isinstance(spec, dict):
            out['violations'].append(f'{name}: level must be a mapping')
            continue
        roles[name] = str(spec.get('role') or '')
        _arms = spec.get('arms')
        _arms_norm = dict(_arms) if isinstance(_arms, dict) and _arms else _arms
        if not isinstance(_arms, dict) or not _arms:
            out['violations'].append(
                f'{name}: arms must be a non-empty mapping base label -> arm id')
        out['levels'][str(name)] = {
            'role': spec.get('role'),
            'fuse_mode': spec.get('fuse_mode'),
            'arms': _arms_norm,
        }
    expected_roles = {'treatment', 'capacity_placebo', 'absence_control'}
    if set(roles.values()) != expected_roles:
        out['status'] = 'fail'
        out['violations'].append(
            'ladder roles must be exactly treatment/capacity_placebo/'
            f'absence_control, got {sorted(roles.values())}')
    elif len(roles) != len(expected_roles):
        out['status'] = 'fail'
        out['violations'].append(
            'ladder levels must use each of treatment/capacity_placebo/'
            f'absence_control exactly once, got {sorted(roles.items())}')
    by_role = {v: k for k, v in roles.items()}
    treatment_name = by_role.get('treatment')
    placebo_name = by_role.get('capacity_placebo')
    absent_name = by_role.get('absence_control')
    if treatment_name is None or placebo_name is None or absent_name is None:
        out['status'] = 'fail'
        return out

    treatment_pairs = _declared_pairs(design, 'local_fusion_learning')
    total_pairs = _declared_pairs(design, 'local_fusion')
    if not treatment_pairs:
        out['violations'].append(
            'ABLATION_DESIGN has no local_fusion_learning contrast for the '
            'capacity placebo')
    if not total_pairs:
        out['violations'].append(
            'ABLATION_DESIGN has no local_fusion contrast for the absence control')
    out['component_design'] = {
        'local_fusion_learning': sorted(treatment_pairs or ()),
        'local_fusion': sorted(total_pairs or ()),
    }

    def _fuse_layers(factory_arm):
        cfgs = factory(n_layers, factory_arm)
        if isinstance(cfgs, (str, bytes, bytearray)):
            raise TypeError('factory returned a string, not a sequence of layer configs')
        cfgs = list(cfgs)
        if len(cfgs) != n_layers:
            raise ValueError(
                f'arm {factory_arm!r} has {len(cfgs)} layer config(s), expected {n_layers}')
        return cfgs

    def _model_bundle(arm):
        # Build under a restored RNG so the audit itself is neutral, then use
        # the SAME seed for every level so a shifted initialization is visible.
        with _preserve_rng_state():
            torch.manual_seed(1234)
            model = SmallGPT(32, 16, n_layers, 2, 8, 16, factory(n_layers, arm))
            named = dict(model.named_parameters())
            state = {n: p.detach().clone() for n, p in named.items()}
            fuse_names = [n for n in named if 'fuse_conv' in n]
            return {
                'total': sum(int(p.numel()) for p in named.values()),
                'fuse_params': sum(int(named[n].numel()) for n in fuse_names),
                'n_fuse_tensors': len(fuse_names),
                'fuse_frozen': bool(fuse_names) and all(
                    not bool(named[n].requires_grad) for n in fuse_names),
                'non_fuse_state': {n: p for n, p in state.items()
                                   if 'fuse_conv' not in n},
            }

    def _non_fuse_equal(sa, sb):
        if set(sa) != set(sb):
            return False
        return all(torch.equal(sa[n], sb[n]) for n in sa)

    # Construct every level against its base and verify the exact diff.
    for base_label, base_arm in bases.items():
        base_rec = {'base_arm': base_arm, 'expected_absence_keys':
                    sorted(absence_keys[base_label])}
        try:
            _base_cfgs = _fuse_layers(base_arm)
        except Exception as _be:
            out['violations'].append(
                f'{base_label}: cannot construct base arm {base_arm!r} '
                f'({type(_be).__name__}: {_be})')
            out['base_checks'][base_label] = base_rec
            continue
        for level_name in (treatment_name, placebo_name, absent_name):
            spec = levels[level_name]
            arms = spec.get('arms') if isinstance(spec, dict) else None
            arm = arms.get(base_label) if isinstance(arms, dict) else None
            rec = {'arm': arm, 'expected_mode': spec.get('fuse_mode')}
            base_rec[level_name] = rec
            if not isinstance(arm, str) or not arm.strip():
                out['violations'].append(
                    f'{level_name}/{base_label}: no single arm id declared')
                rec['status'] = 'invalid'
                continue
            if arm not in present_set:
                out['violations'].append(
                    f'{level_name}/{base_label}: arm {arm!r} is not scheduled in '
                    'the ablation panel')
                rec['status'] = 'missing'
                continue
            _keys, _per = _design_pair_cfg_keys(arm, base_arm, factory, n_layers)
            if isinstance(_keys, Exception):
                out['violations'].append(
                    f'{level_name}/{base_label}: cannot construct {arm!r} vs '
                    f'{base_arm!r} ({type(_keys).__name__}: {_keys})')
                rec['status'] = 'unconstructable'
                continue
            diff = sorted(_keys)
            rec['cfg_diff_keys'] = diff
            try:
                _cfgs = _fuse_layers(arm)
            except Exception as _me:
                out['violations'].append(
                    f'{level_name}/{base_label}: cannot construct {arm!r} '
                    f'({type(_me).__name__}: {_me})')
                rec['status'] = 'unconstructable'
                continue
            expected_mode = spec.get('fuse_mode')
            _actual_modes = {}
            _bad_modes = []
            for _i, (_c, _bc) in enumerate(zip(_cfgs, _base_cfgs)):
                _active = (getattr(_bc, 'kind', None) in ('csa', 'hca')
                           and getattr(_bc, 'chunking', None) == 'cosine_learnable')
                _expected_i = expected_mode if _active else 'none'
                _got = getattr(_c, 'fuse_mode', None)
                _actual_modes[_i] = _got
                if _got != _expected_i:
                    _bad_modes.append((_i, _got, _expected_i))
            rec['observed_fuse_modes'] = _actual_modes
            if level_name == treatment_name:
                bad = []
                if arm != base_arm:
                    bad.append(f'treatment arm {arm!r} must be the base arm '
                               f'{base_arm!r}')
                if diff:
                    bad.append(f'treatment arm must not differ from the base '
                               f'config, got {diff}')
                if expected_mode != 'learned':
                    bad.append(f'treatment must declare fuse_mode="learned", '
                               f'got {expected_mode!r}')
                if _bad_modes:
                    bad.append('constructed fuse_mode does not equal the '
                               f'declared per-layer mode: {_bad_modes}')
                if bad:
                    out['violations'].append(
                        f'{level_name}/{base_label}: ' + '; '.join(bad))
                    rec['status'] = 'confounded'
                else:
                    rec['status'] = 'ok'
                continue
            if level_name == placebo_name:
                bad = []
                if diff != ['fuse_mode']:
                    bad.append(f'cfg diff is {diff}, not exactly ["fuse_mode"]')
                if expected_mode != 'random':
                    bad.append(
                        f'capacity placebo must declare fuse_mode="random", got '
                        f'{expected_mode!r}')
                if _bad_modes:
                    bad.append('constructed fuse_mode is not the declared '
                               f'random/identity pattern: {_bad_modes}')
                if bad:
                    out['violations'].append(
                        f'{level_name}/{base_label}: ' + '; '.join(bad))
                    rec['status'] = 'confounded'
                else:
                    rec['status'] = 'ok'
                continue
            # absence control
            bad = []
            expected = absence_keys[base_label]
            if set(diff) != set(expected):
                bad.append(f'cfg diff is {diff}, not the declared absence '
                           f'key(s) {sorted(expected)}')
            if expected_mode != 'none':
                bad.append(f'absence control must declare fuse_mode="none", '
                           f'got {expected_mode!r}')
            if _bad_modes:
                bad.append('constructed fuse_mode is not the declared '
                           f'none/identity pattern: {_bad_modes}')
            if bad:
                out['violations'].append(
                    f'{level_name}/{base_label}: ' + '; '.join(bad))
                rec['status'] = 'confounded'
            else:
                rec['status'] = 'ok'
        # Parameter-count and non-fusion-initialization checks.
        try:
            counts = {
                level_name: _model_bundle(
                    levels[level_name]['arms'][base_label])
                for level_name in (treatment_name, placebo_name, absent_name)
            }
        except Exception as e:  # fail closed: an unbuildable arm is a defect
            out['violations'].append(
                f'{base_label}: cannot build the three ladder arms '
                f'({type(e).__name__}: {e})')
            out['base_checks'][base_label] = base_rec
            continue
        for level_name, c in counts.items():
            base_rec[level_name]['params'] = c['total']
            base_rec[level_name]['fuse_params'] = c['fuse_params']
            base_rec[level_name]['n_fuse_tensors'] = c['n_fuse_tensors']
        _t, _p, _a = (counts[treatment_name], counts[placebo_name],
                      counts[absent_name])
        base_rec['params_checked'] = True
        base_rec['treatment_placebo_matched'] = _t['total'] == _p['total']
        base_rec['treatment_absence_delta'] = _t['total'] - _a['total']
        base_rec['expected_absence_delta'] = _t['fuse_params']
        base_rec['shared_init_matched'] = (
            _non_fuse_equal(_t['non_fuse_state'], _p['non_fuse_state'])
            and _non_fuse_equal(_t['non_fuse_state'], _a['non_fuse_state']))
        if not base_rec['treatment_placebo_matched']:
            out['violations'].append(
                f'{base_label}: treatment and capacity placebo differ in total '
                f'parameters ({_t["total"]} vs {_p["total"]}); the placebo is '
                'not capacity-matched')
        if _t['fuse_params'] != _p['fuse_params']:
            out['violations'].append(
                f'{base_label}: treatment and capacity placebo differ in fusion '
                f'parameters ({_t["fuse_params"]} vs {_p["fuse_params"]})')
        if _p['n_fuse_tensors'] != _t['n_fuse_tensors']:
            out['violations'].append(
                f'{base_label}: capacity placebo has a different number of fusion '
                f'tensors ({_p["n_fuse_tensors"]} vs {_t["n_fuse_tensors"]})')
        if not _p['fuse_frozen']:
            out['violations'].append(
                f'{base_label}: capacity placebo fusion parameters are trainable; '
                'a learned placebo is not a placebo')
        if _t['n_fuse_tensors'] and _t['fuse_frozen']:
            out['violations'].append(
                f'{base_label}: treatment fusion parameters are frozen; the '
                '"learned" level cannot be a treatment')
        if _t['total'] - _a['total'] != _t['fuse_params']:
            out['violations'].append(
                f'{base_label}: absence control removes '
                f'{_t["total"] - _a["total"]} parameters but fusion accounts for '
                f'{_t["fuse_params"]}; the on/off contrast changed more than the '
                'fusion capacity')
        if not base_rec['shared_init_matched']:
            out['violations'].append(
                f'{base_label}: the three levels do not start from the same '
                'non-fusion parameter initialization; adding/removing the conv '
                'shifted the global RNG stream')
        out['base_checks'][base_label] = base_rec

    # Tie each control level to its declared leave-one-out family.
    for base_label, base_arm in bases.items():
        _p = _declared_arm_id(levels, placebo_name, base_label)
        _a = _declared_arm_id(levels, absent_name, base_label)
        if treatment_pairs is not None and (base_arm, _p) not in treatment_pairs:
            out['violations'].append(
                f'local_fusion_learning does not declare {base_arm!r} vs {_p!r} '
                'for the capacity placebo')
        if total_pairs is not None and (base_arm, _a) not in total_pairs:
            out['violations'].append(
                f'local_fusion does not declare {base_arm!r} vs {_a!r} for the '
                'absence control')
    keys = ABLATION_INTERVENTION_KEYS if isinstance(
        ABLATION_INTERVENTION_KEYS, dict) else {}
    _pkeys = keys.get('local_fusion_learning')
    if not isinstance(_pkeys, dict):
        out['violations'].append(
            'local_fusion_learning: no declared intervention-key mapping')
    else:
        for base_label, base_arm in bases.items():
            _p = _declared_arm_id(levels, placebo_name, base_label)
            kset = _pkeys.get((base_arm, _p))
            if kset != frozenset({'fuse_mode'}):
                out['violations'].append(
                    f'local_fusion_learning: contrast {(base_arm, _p)} declares '
                    f'intervention keys '
                    f'{sorted(kset) if isinstance(kset, (set, frozenset)) else kset!r}, '
                    'expected exactly {"fuse_mode"}')
    _akeys = keys.get('local_fusion')
    if not isinstance(_akeys, dict):
        out['violations'].append(
            'local_fusion: no declared intervention-key mapping')
    else:
        for base_label, base_arm in bases.items():
            _a = _declared_arm_id(levels, absent_name, base_label)
            kset = _akeys.get((base_arm, _a))
            if kset != frozenset(absence_keys[base_label]):
                out['violations'].append(
                    f'local_fusion: contrast {(base_arm, _a)} declares '
                    f'intervention keys '
                    f'{sorted(kset) if isinstance(kset, (set, frozenset)) else kset!r}, '
                    f'expected {sorted(absence_keys[base_label])}')
    # Distinct arm per level per base (a relabelled copy is not a control).
    for base_label in bases:
        seen = {}
        for level_name in (treatment_name, placebo_name, absent_name):
            arm = _declared_arm_id(levels, level_name, base_label)
            if arm is not None:
                seen.setdefault(arm, []).append(level_name)
        for arm, names in seen.items():
            if len(names) > 1:
                out['violations'].append(
                    f'{base_label}: arm {arm!r} is reused by levels {names}; the '
                    'control is a relabelled copy, not an independent level')
    if out['violations']:
        out['status'] = 'fail'
    return out
