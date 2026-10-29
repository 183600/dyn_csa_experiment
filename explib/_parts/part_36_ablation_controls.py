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
    by_role = {v: k for k, v in roles.items()}
    treatment_name = by_role.get('treatment')
    absent_name = by_role.get('presence_control')
    misaligned_name = by_role.get('alignment_placebo')
    if treatment_name is None or absent_name is None or misaligned_name is None:
        out['status'] = 'fail'
        return out
    # Check the leave-one-out families that consume these controls.
    def _pairs_for(comp):
        raw = design.get(comp) if isinstance(design, dict) else None
        if not isinstance(raw, (list, tuple)):
            return None
        pairs = set()
        for _item in raw:
            if (isinstance(_item, (tuple, list)) and len(_item) >= 2
                    and isinstance(_item[0], str) and isinstance(_item[1], str)):
                pairs.add((_item[0], _item[1]))
        return pairs
    presence_pairs = _pairs_for('content_pooling')
    alignment_pairs = _pairs_for('content_alignment')
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
            if level_name == treatment_name:
                if arm != base_arm:
                    out['violations'].append(
                        f'{level_name}/{base_label}: treatment arm {arm!r} must '
                        f'be the base arm {base_arm!r} (the aligned table is the '
                        'reference)')
                    rec['status'] = 'confounded'
                elif diff:
                    out['violations'].append(
                        f'{level_name}/{base_label}: treatment arm must not differ '
                        f'from the base config, got {diff}')
                    rec['status'] = 'confounded'
                else:
                    rec['status'] = 'ok'
                continue
            expected_mode = spec.get('content_mode')
            _values = {}
            if _per:
                for _layer, _fields in _per.items():
                    if isinstance(_fields, dict) and 'content_mode' in _fields:
                        _values[_layer] = _fields['content_mode'][0]
            rec['observed_modes'] = _values
            bad = []
            if diff != ['content_mode']:
                bad.append(f'cfg diff is {diff}, not exactly ["content_mode"]')
            if expected_mode not in CONTENT_MODES:
                bad.append(f'expected content_mode {expected_mode!r} is not a '
                           f'declared mode {CONTENT_MODES}')
            if _values and any(v != expected_mode for v in _values.values()):
                bad.append(
                    'constructed content_mode does not equal the declared '
                    f'{expected_mode!r}')
            if bad:
                out['violations'].append(
                    f'{level_name}/{base_label}: ' + '; '.join(bad))
                rec['status'] = 'confounded'
            else:
                rec['status'] = 'ok'
        out['base_checks'][base_label] = base_rec
    # Tie each control level to its declared leave-one-out family.
    for base_label, base_arm in bases.items():
        _mis_arms = levels[misaligned_name].get('arms')
        _abs_arms = levels[absent_name].get('arms')
        _mis = (_mis_arms.get(base_label) if isinstance(_mis_arms, dict) else None)
        _abs = (_abs_arms.get(base_label) if isinstance(_abs_arms, dict) else None)
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
            _level_arms = levels[level_name].get('arms')
            arm = (_level_arms.get(base_label)
                   if isinstance(_level_arms, dict) else None)
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
