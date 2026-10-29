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
    for _b, _ks in absence_keys.items():
        if not isinstance(_ks, (set, frozenset)) or not _ks:
            out['status'] = 'fail'
            out['violations'].append(
                f'absence_keys[{_b!r}] must be a non-empty set of cfg keys')
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

    treatment_pairs = _pairs_for('attention_sink_learning')
    total_pairs = _pairs_for('attention_sink')
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
            _values = {}
            if _per:
                for _layer, _fields in _per.items():
                    if isinstance(_fields, dict) and 'sink_mode' in _fields:
                        _values[_layer] = _fields['sink_mode'][0]
            rec['observed_sink_modes'] = _values
            if level_name == treatment_name:
                if arm != base_arm:
                    out['violations'].append(
                        f'{level_name}/{base_label}: treatment arm {arm!r} must '
                        f'be the base arm {base_arm!r}')
                    rec['status'] = 'confounded'
                elif diff:
                    out['violations'].append(
                        f'{level_name}/{base_label}: treatment arm must not differ '
                        f'from the base config, got {diff}')
                    rec['status'] = 'confounded'
                else:
                    rec['status'] = 'ok'
                continue
            if level_name == placebo_name:
                bad = []
                if diff != ['sink_mode']:
                    bad.append(f'cfg diff is {diff}, not exactly ["sink_mode"]')
                expected_mode = spec.get('sink_mode')
                if expected_mode != 'random':
                    bad.append(
                        f'capacity placebo must declare sink_mode="random", got '
                        f'{expected_mode!r}')
                if _values and any(v != 'random' for v in _values.values()):
                    bad.append('constructed sink_mode is not "random"')
                if bad:
                    out['violations'].append(
                        f'{level_name}/{base_label}: ' + '; '.join(bad))
                    rec['status'] = 'confounded'
                else:
                    rec['status'] = 'ok'
                continue
            # absence control
            expected = absence_keys[base_label]
            if set(diff) != set(expected):
                out['violations'].append(
                    f'{level_name}/{base_label}: cfg diff is {diff}, not the '
                    f'declared absence key(s) {sorted(expected)}')
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
        if _t['total'] - _a['total'] != _t['sink_params']:
            out['violations'].append(
                f'{base_label}: absence control removes '
                f'{_t["total"] - _a["total"]} parameters but the sink accounts '
                f'for {_t["sink_params"]}; the on/off contrast changed more than '
                'the sink capacity')
        out['base_checks'][base_label] = base_rec

    # Tie each control level to its declared leave-one-out family.
    for base_label, base_arm in bases.items():
        _p_arms = levels[placebo_name].get('arms')
        _a_arms = levels[absent_name].get('arms')
        _p = _p_arms.get(base_label) if isinstance(_p_arms, dict) else None
        _a = _a_arms.get(base_label) if isinstance(_a_arms, dict) else None
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
            _p_arms = levels[placebo_name].get('arms')
            _p = _p_arms.get(base_label) if isinstance(_p_arms, dict) else None
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
            _a_arms = levels[absent_name].get('arms')
            _a = _a_arms.get(base_label) if isinstance(_a_arms, dict) else None
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
