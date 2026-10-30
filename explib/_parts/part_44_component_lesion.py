# ---- inference-time lesion: ablation at evaluation, not only at training ----
# A leave-one-out arm retrains the model without a component.  A significant
# difference between that arm and the intact arm shows that *training with*
# the component changed the model.  It does not show that the trained model
# *uses* the component at inference: the component can be an optimisation
# scaffold / regulariser that is ignored at test time, and a component that
# learned to be an exact no-op (sink learned to zero, fusion learned identity)
# is observationally identical to "no effect" in a between-model table.
#
# This fragment adds the complementary, within-checkpoint measurement: keep
# the trained weights and evaluation rows fixed, disable one declared
# component only for the forward pass, and compare the outputs/PPL.  The
# comparison is paired at the checkpoint level, so it removes the between-run
# variance that dominates a 3-8 seed ablation panel.
#
# Two checks are deliberately separate:
#   * manipulation check: does the lesion change the forward output at all?
#     A no-op lesion cannot support any inference-time claim;
#   * functional effect: does the PPL move beyond the pre-registered SESOI?
#     A component can be engaged but redundant, or engaged and harmful.
#
# Nothing here retrains, changes a checkpoint, mutates a weight or reads the
# test split.  The lesion is a context manager around the existing forward
# path; it restores every config/module state even if the forward raises.

import contextlib as _lesion_contextlib

# Components that can be disabled at inference without changing the parameter
# count or re-instantiating the model.  The declaration is intentionally a
# whitelist: an undeclared component is an error, not a silent no-op.
LESION_COMPONENTS = ('attention_sink', 'local_fusion', 'content_pooling')

# `cfg_overrides` are applied to a *copy* of each layer's AttnCfg; the
# `module_overrides` replace a live submodule attribute and are restored to
# the original object afterwards.  For `local_fusion` the config is left
# untouched on purpose: `cfg.fuse_mode` participates in backend dispatch
# (`_eager_batch`), so changing it would compare the intact arm's row path
# with the lesion arm's batched path — a second factor hidden inside the
# lesion.  Removing `fuse_conv` is the surgical intervention.
LESION_DESIGN = {
    'attention_sink': {
        'source_component': 'attention_sink',
        'arms': ('csa_fixed', 'csa_dynamic', 'full_sink'),
        'cfg_overrides': {'use_sink': False, 'full_sink': False},
        'module_overrides': (),
        'mechanism_keys': frozenset({'use_sink', 'full_sink'}),
        'state_kind': 'sink',
        'note': ('the learned/frozen sink parameter stays in the state dict; '
                 'only its contribution to attention is removed'),
    },
    'local_fusion': {
        'source_component': 'local_fusion',
        'arms': ('csa_dyn_fuse', 'hybrid_csa_dyn_fuse'),
        'cfg_overrides': {},
        'module_overrides': (('fuse_conv', None),),
        'mechanism_keys': frozenset({'fuse_mode'}),
        'state_kind': 'fuse_conv',
        'note': ('the depth-wise convolution is removed from the forward '
                 'path; cfg.fuse_mode is kept so backend dispatch is '
                 'identical to the intact pass'),
    },
    'content_pooling': {
        'source_component': 'content_pooling',
        'arms': ('csa_fixed', 'csa_dynamic'),
        'cfg_overrides': {'content_mode': 'zero'},
        'module_overrides': (),
        'mechanism_keys': frozenset({'content_mode'}),
        'state_kind': None,
        'note': ('the compressed block content is replaced by zeros; this is '
                 'the content-presence lesion, not the content-alignment '
                 'placebo'),
    },
}

LESION_POLICY = {
    'version': 'lesion-1',
    'relative_pct': 1.0,
    'alpha': 0.05,
    'min_n': 3,
    'tost_conf': 0.90,
    'atol': 1e-6,
    'rtol': 1e-4,
    'note': ('within-checkpoint lesion only: the same checkpoint is '
             'evaluated with and without the component.  The manipulation '
             'check is a precondition for the PPL claim, not evidence that '
             'the component is the only mechanism.'),
}


def _lesion_design_entry(component):
    if not isinstance(component, str) or component not in LESION_DESIGN:
        raise ValueError(
            f'unknown lesion component {component!r}; expected one of '
            f'{sorted(LESION_DESIGN)}')
    return LESION_DESIGN[component]


def _lesion_cfg(cfg, overrides):
    """A copy of one layer config with only the declared fields changed."""
    new = copy.copy(cfg)
    for key, value in overrides.items():
        if not hasattr(new, key):
            raise ValueError(
                f'lesion override {key!r} is not an AttnCfg field')
        setattr(new, key, value)
    return new


def _lesion_layers(model, component):
    """Return (index, attention_module, cfg) for layers the lesion applies to.

    Applicability is read from the constructed module, not from the arm label:
    a sink lesion on an arm without a sink, or a fusion lesion on a layer
    without a convolution, fails closed instead of yielding an empty context
    that would look like a successful no-op.
    """
    blocks = getattr(model, 'blocks', None)
    if blocks is None:
        raise TypeError(
            f'component_lesion expects a model with .blocks, got '
            f'{type(model).__name__}')
    out = []
    for i, block in enumerate(blocks):
        attn = getattr(block, 'attn', None)
        cfg = getattr(attn, 'cfg', None) if attn is not None else None
        if attn is None or cfg is None:
            continue
        if component == 'attention_sink':
            active = (bool(getattr(cfg, 'full_sink', False))
                      if getattr(cfg, 'kind', None) == 'full'
                      else bool(getattr(cfg, 'use_sink', True)))
            if active:
                out.append((i, attn, cfg))
        elif component == 'local_fusion':
            if getattr(attn, 'fuse_conv', None) is not None:
                out.append((i, attn, cfg))
        elif component == 'content_pooling':
            if getattr(cfg, 'kind', None) in ('csa', 'hca'):
                out.append((i, attn, cfg))
        else:  # pragma: no cover - guarded by _lesion_design_entry
            raise ValueError(f'unsupported lesion component {component!r}')
    if not out:
        raise ValueError(
            f'component_lesion({component!r}) found no applicable layer in '
            f'{type(model).__name__}; the component is already absent')
    return out


@_lesion_contextlib.contextmanager
def component_lesion(model, component):
    """Temporarily disable one declared component for a forward pass.

    The context yields a small record (component, layer indices, overrides).
    Every touched ``AttnCfg`` and submodule attribute is restored in a
    ``finally`` block, including when the forward raises.  The model's
    ``training`` flag is not changed; callers that need inference semantics
    should use ``model.eval()`` (or let ``eval_ppl`` do it) and are
    responsible for restoring training mode.
    """
    entry = _lesion_design_entry(component)
    layers = _lesion_layers(model, component)
    snapshots = []
    try:
        for index, attn, cfg in layers:
            snap = {'index': index, 'attn': attn, 'cfg': cfg, 'modules': {}}
            snapshots.append(snap)
            attn.cfg = _lesion_cfg(cfg, entry['cfg_overrides'])
            for name, value in entry.get('module_overrides') or ():
                if not hasattr(attn, name):
                    raise ValueError(
                        f'component_lesion({component!r}): attention module '
                        f'has no attribute {name!r}')
                snap['modules'][name] = getattr(attn, name)
                setattr(attn, name, value)
        yield {
            'kind': 'component_lesion',
            'component': component,
            'layers': [s['index'] for s in snapshots],
            'cfg_overrides': dict(entry['cfg_overrides']),
            'module_overrides': [name for name, _value in
                                 (entry.get('module_overrides') or ())],
        }
    finally:
        for snap in reversed(snapshots):
            snap['attn'].cfg = snap['cfg']
            for name, value in snap['modules'].items():
                setattr(snap['attn'], name, value)


def _lesion_parameter_tensors(model, component):
    """Named parameter tensors that implement the declared component."""
    named = list(model.named_parameters())
    if component == 'attention_sink':
        return [(n, p) for n, p in named
                if n == 'sink' or n.endswith('.sink')]
    if component == 'local_fusion':
        return [(n, p) for n, p in named
                if n.endswith('.fuse_conv.weight') or n == 'fuse_conv.weight']
    if component == 'content_pooling':
        return []
    raise ValueError(f'unknown lesion component {component!r}')


def _fuse_identity_kernel(weight):
    """The identity depth-wise kernel used by the learned-fusion initialiser."""
    identity = torch.zeros_like(weight)
    identity[:, :, -1] = 1.0
    return identity


def component_lesion_state(model, component, *, atol=1e-8):
    """Report whether the trained component is present and non-trivial.

    A freshly initialised learned sink is exactly zero and a freshly
    initialised learned fusion is exactly identity.  Both are valid *training
    starting points* and exact *no-op* forward components.  A lesion applied
    there cannot change the output, so this state must be checked on the
    actual checkpoint before interpreting a null lesion as "unused".
    """
    entry = _lesion_design_entry(component)
    if isinstance(atol, (bool, np.bool_)):
        raise ValueError(f'atol must be a finite non-negative number, got {atol!r}')
    atol = float(atol)
    if not math.isfinite(atol) or atol < 0.0:
        raise ValueError(f'atol must be a finite non-negative number, got {atol!r}')
    kind = entry.get('state_kind')
    tensors = _lesion_parameter_tensors(model, component)
    out = {'kind': 'component_lesion_state', 'component': component,
           'state_kind': kind, 'n_tensors': len(tensors),
           'n_params': int(sum(int(p.numel()) for _n, p in tensors)),
           'status': 'unknown', 'trivial': None, 'n_nonzero': None,
           'zero_valued': None,
           'norm': None, 'distance_from_identity': None, 'atol': atol,
           'note': entry.get('note')}
    if kind is None:
        out['status'] = 'not_parameterized'
        out['note'] = ('this component is computed (content table), not a '
                       'parameter tensor; the forward manipulation check is '
                       'the only applicable test')
        return out
    if not tensors:
        out['status'] = 'absent'
        out['trivial'] = True
        return out
    with torch.no_grad():
        n_nonzero = int(sum(int((p.detach() != 0).sum()) for _n, p in tensors))
        norm = float(math.sqrt(math.fsum(
            float(p.detach().float().pow(2).sum()) for _n, p in tensors)))
    out['n_nonzero'] = n_nonzero
    out['norm'] = norm
    if kind == 'sink':
        # A zero sink is NOT a no-op: the sink slot still contributes
        # exp(0)=1 to the softmax denominator, so removing it changes the
        # normalisation even before the parameter has learned anything.
        # "All-zero" is a capacity/initialisation fact, not a functional one.
        out['zero_valued'] = bool(n_nonzero == 0)
        out['trivial'] = False
    elif kind == 'fuse_conv':
        with torch.no_grad():
            distance = max(
                float((p.detach() - _fuse_identity_kernel(p.detach())).abs().max())
                for _n, p in tensors)
        out['distance_from_identity'] = distance
        out['trivial'] = bool(distance <= atol)
    else:  # pragma: no cover - declaration guard
        raise ValueError(f'unknown lesion state kind {kind!r}')
    out['status'] = 'trivial' if out['trivial'] else 'present'
    if kind == 'sink' and out.get('zero_valued'):
        out['note'] = (
            'the sink parameter is all-zero, but a zero sink still adds a '
            'uniform sink slot to the softmax; it is functionally active and '
            'the lesion is not a no-op')
    return out


def _lesion_forward(model, ids, logits_tail=None):
    if logits_tail is None:
        return model(ids)
    return model(ids, logits_tail=logits_tail)


def lesion_sensitivity(model, ids, component, *, atol=LESION_POLICY['atol'],
                       rtol=LESION_POLICY['rtol'], check_weights=True,
                       logits_tail=None):
    """Forward-level manipulation check for one inference-time lesion.

    Returns the intact/lesion logit difference and whether the weights were
    byte-identical before and after.  ``status='changed'`` is the only state
    that supports an inference-use claim; ``status='unchanged'`` means the
    lesion is a no-op on *this input batch* at this checkpoint (the component
    may simply have learned to be inert, or the batch may not exercise it),
    not that the component is useless in training.  Callers should pass the
    same evaluation rows/batches they use for the PPL read.
    """
    if not torch.is_tensor(ids) or ids.ndim != 2:
        raise ValueError('lesion_sensitivity expects a 2-D token-id tensor')
    _lesion_design_entry(component)
    for _name, _value in (('atol', atol), ('rtol', rtol)):
        if isinstance(_value, (bool, np.bool_)):
            raise ValueError(f'{_name} must be a finite non-negative number, got {_value!r}')
        _f = float(_value)
        if not math.isfinite(_f) or _f < 0.0:
            raise ValueError(f'{_name} must be a finite non-negative number, got {_value!r}')
    atol, rtol = float(atol), float(rtol)
    was_training = bool(getattr(model, 'training', False))
    weights_before = state_dict_sha256(model.state_dict()) if check_weights else None
    try:
        model.eval()
        with torch.no_grad():
            intact = _lesion_forward(model, ids, logits_tail=logits_tail)
            with component_lesion(model, component) as rec:
                lesion = _lesion_forward(model, ids, logits_tail=logits_tail)
    finally:
        model.train(was_training)
    weights_after = state_dict_sha256(model.state_dict()) if check_weights else None
    _intact = intact[0] if isinstance(intact, tuple) else intact
    _lesion = lesion[0] if isinstance(lesion, tuple) else lesion
    if not (torch.is_tensor(_intact) and torch.is_tensor(_lesion)):
        raise TypeError('model(ids) must return a tensor (or a tuple with a tensor first)')
    if tuple(_intact.shape) != tuple(_lesion.shape):
        raise ValueError(
            f'lesion changed the output shape: {tuple(_intact.shape)} vs '
            f'{tuple(_lesion.shape)}')
    _i = _intact.detach().float()
    _l = _lesion.detach().float()
    diff = (_i - _l).abs()
    finite = bool(torch.isfinite(_i).all() and torch.isfinite(_l).all()
                  and torch.isfinite(diff).all())
    max_abs = float(diff.max()) if diff.numel() else 0.0
    mean_abs = float(diff.mean()) if diff.numel() else 0.0
    scale = float(_i.abs().max()) if _i.numel() else 0.0
    denom = float(_l.norm().item()) if _l.numel() else 0.0
    rel_l2 = (float(diff.norm().item()) / denom) if denom > 0.0 else None
    threshold = atol + rtol * scale
    changed = bool(finite and max_abs > threshold)
    if not finite:
        status = 'nonfinite'
    elif check_weights and weights_before != weights_after:
        status = 'weights_changed'
    elif changed:
        status = 'changed'
    else:
        status = 'unchanged'
    return {
        'kind': 'lesion_sensitivity',
        'component': component,
        'layers': list(rec.get('layers') or []),
        'cfg_overrides': rec.get('cfg_overrides') or {},
        'module_overrides': rec.get('module_overrides') or [],
        'status': status,
        'changed': changed,
        'max_abs_logit_delta': max_abs,
        'mean_abs_logit_delta': mean_abs,
        'relative_l2_delta': rel_l2,
        'threshold': threshold,
        'atol': atol,
        'rtol': rtol,
        'weights_unchanged': (None if weights_before is None
                              else bool(weights_before == weights_after)),
        'weights_sha256_before': weights_before,
        'weights_sha256_after': weights_after,
    }


def _lesion_ppl_values(values, name):
    if isinstance(values, (str, bytes)) or not hasattr(values, '__iter__'):
        raise ValueError(f'{name} must be an iterable of per-seed PPL values')
    out = []
    for i, value in enumerate(values):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f'{name}[{i}] must be a positive finite PPL, got {value!r}')
        f = _finite_scalar(value)
        if f is None or f <= 0.0:
            raise ValueError(f'{name}[{i}] must be a positive finite PPL, got {value!r}')
        out.append(float(f))
    return out


def lesion_effect(intact_ppl, lesion_ppl, *, seeds=None, component=None,
                  relative_pct=LESION_POLICY['relative_pct'],
                  alpha=LESION_POLICY['alpha'],
                  min_n=LESION_POLICY['min_n'],
                  sensitivity=None,
                  tost_conf=LESION_POLICY['tost_conf']):
    """Paired PPL effect of a within-checkpoint lesion.

    The estimand is ``log(PPL_lesion / PPL_intact)`` per seed: positive means
    the intact model is better, i.e. the component helps at inference.  The
    exact sign-flip test and a paired TOST use the *same* per-seed vector;
    the verdict distinguishes "engaged and necessary", "engaged but
    redundant", "engaged and harmful", "inert", and "not checked".

    A PPL claim is only allowed when a manipulation check is supplied and
    passed (``changed`` and ``weights_unchanged`` both true).  That is the
    point of the ladder: a null PPL difference from a no-op lesion is not
    evidence that the component is redundant.
    """
    if component is not None:
        _lesion_design_entry(component)
    for _name, _value in (('relative_pct', relative_pct),
                          ('alpha', alpha), ('min_n', min_n),
                          ('tost_conf', tost_conf)):
        if isinstance(_value, (bool, np.bool_)):
            raise ValueError(f'{_name} must be a number, got {_value!r}')
    relative_pct = float(relative_pct)
    alpha = _validated_alpha(alpha, context='lesion_effect alpha')
    _min_n = _as_seed_int(min_n)
    if _min_n is None or _min_n < 0:
        raise ValueError(f'min_n must be a non-negative integer, got {min_n!r}')
    min_n = int(_min_n)
    tost_conf = float(tost_conf)
    if not math.isfinite(relative_pct) or relative_pct <= 0.0:
        raise ValueError(
            f'relative_pct must be a finite positive number, got {relative_pct!r}')
    if not math.isfinite(tost_conf) or not (0.0 < tost_conf < 1.0):
        raise ValueError(f'tost_conf must be in (0,1), got {tost_conf!r}')
    intact = _lesion_ppl_values(intact_ppl, 'intact_ppl')
    lesion = _lesion_ppl_values(lesion_ppl, 'lesion_ppl')
    if len(intact) != len(lesion):
        raise ValueError(
            f'intact_ppl and lesion_ppl must be paired per seed, got '
            f'{len(intact)} and {len(lesion)}')
    if seeds is not None:
        seeds = list(seeds)
        if len(seeds) != len(intact):
            raise ValueError(
                f'seeds length {len(seeds)} does not match {len(intact)} PPL pairs')
    deltas = [math.log(l) - math.log(i) for i, l in zip(intact, lesion)]
    margin = math.log1p(relative_pct / 100.0)
    stats = paired_effect_stats(deltas, alpha=alpha, min_n=min_n, seeds=seeds)
    tost = tost_paired(deltas, margin, conf=tost_conf)
    mean = stats.get('mean')
    try:
        mean_pct = (math.expm1(mean) * 100.0
                    if isinstance(mean, (int, float)) and math.isfinite(mean)
                    else None)
    except OverflowError:
        mean_pct = None
    # Manipulation check state.  Strictly require both flags: a caller who
    # only recorded `changed=True` has not shown the weights were unchanged.
    if sensitivity is None:
        manipulation = 'not_checked'
    elif not isinstance(sensitivity, dict):
        raise ValueError('sensitivity must be a mapping or None')
    elif sensitivity.get('weights_unchanged') is False:
        manipulation = 'invalid'
    elif sensitivity.get('changed') is False:
        manipulation = 'failed'
    elif (sensitivity.get('changed') is True
          and sensitivity.get('weights_unchanged') is True):
        manipulation = 'passed'
    else:
        manipulation = 'not_checked'
    n = int(stats.get('n') or 0)
    exact_zero = bool(n and all(abs(x) <= 1e-15 for x in deltas))
    too_few = bool(stats.get('underpowered') or stats.get('incomplete_pairs'))
    if not n:
        verdict = 'not_verifiable'
    elif manipulation == 'invalid':
        verdict = 'invalid'
    elif manipulation == 'failed':
        verdict = 'inert'
    elif manipulation != 'passed':
        verdict = 'not_checked'
    elif exact_zero and not too_few:
        # No sampling variability: the same checkpoint produced the same PPL
        # under both conditions.  This is an exact zero, not an underpowered
        # non-rejection.  A single seed is still too few to generalise.
        verdict = 'redundant'
    elif too_few:
        verdict = 'underpowered'
    elif exact_zero:
        verdict = 'underpowered'
    elif tost.get('equivalent') is True:
        # Equivalence is a TOST claim with its own interval; it does not need
        # the exact sign-flip test to be able to reject zero.
        verdict = 'redundant'
    elif stats.get('resolution_limited'):
        verdict = 'underpowered'
    elif stats.get('significant') and mean > margin:
        verdict = 'necessary'
    elif stats.get('significant') and mean < -margin:
        verdict = 'harmful'
    elif stats.get('significant'):
        verdict = 'engaged_below_sesoi'
    else:
        verdict = 'inconclusive'
    if verdict == 'necessary':
        direction = 'lesion_worse'
    elif verdict == 'harmful':
        direction = 'lesion_better'
    elif verdict in ('redundant', 'inert'):
        direction = 'no_functional_change'
    else:
        direction = 'indeterminate'
    claims = {
        'necessary': ('the same trained checkpoint is worse when the '
                      'component is removed at inference; the component is '
                      'used and functionally necessary within the evaluated '
                      'scope'),
        'harmful': ('the same trained checkpoint improves when the component '
                    'is removed at inference; the component is engaged but '
                    'harmful at test time, so a between-model benefit cannot '
                    'be read as an inference-time mechanism benefit'),
        'redundant': ('the lesion changes the forward output but the PPL '
                      'effect is equivalent to zero within the declared '
                      'SESOI; the component is engaged but functionally '
                      'redundant at this checkpoint'),
        'engaged_below_sesoi': ('the lesion changes the output and the PPL '
                                'effect is statistically detectable but '
                                'smaller than the practical SESOI; do not '
                                'claim practical necessity'),
        'inert': ('the lesion does not change the forward output at this '
                  'checkpoint; a between-model on/off difference cannot be '
                  'attributed to inference-time use of the component'),
        'invalid': ('the lesion did not preserve the weights; this is not a '
                    'valid within-checkpoint measurement'),
        'not_checked': ('no passed forward manipulation check; the PPL '
                        'difference alone cannot establish inference-time use'),
        'underpowered': ('too few complete seeds (or the exact-test '
                         'resolution floor is above alpha) for a lesion '
                         'verdict; this is not evidence of no effect'),
        'inconclusive': ('neither a significant effect beyond the SESOI nor '
                         'equivalence within it; no lesion claim'),
        'not_verifiable': 'no complete paired PPL values',
    }
    claim_allowed = bool(verdict in ('necessary', 'harmful', 'redundant',
                                     'engaged_below_sesoi', 'inert'))
    _ci = [stats.get('ci95_low'), stats.get('ci95_high')]
    if not all(isinstance(v, (int, float)) and math.isfinite(float(v))
               for v in _ci):
        _ci = None

    def _finite_or_none(value):
        return (float(value) if isinstance(value, (int, float))
                and math.isfinite(float(value)) else None)

    return {
        'kind': 'lesion_effect',
        'component': component,
        'n': n,
        'seeds': stats.get('seeds'),
        'deltas': deltas,
        'mean_log_ratio': _finite_or_none(mean),
        'mean_pct': mean_pct,
        'sd_log_ratio': _finite_or_none(stats.get('std')),
        'p_exact_signflip': _finite_or_none(stats.get('p_exact_signflip')),
        'p_floor': _finite_or_none(stats.get('p_floor')),
        'ci95_mean': _ci,
        'significant': stats.get('significant'),
        'underpowered': stats.get('underpowered'),
        'resolution_limited': stats.get('resolution_limited'),
        'within_seed_noise': stats.get('within_seed_noise'),
        'sesoi_relative_pct': relative_pct,
        'sesoi_log_margin': margin,
        'tost': tost,
        'manipulation_check': manipulation,
        'sensitivity': sensitivity,
        'verdict': verdict,
        'direction': direction,
        'claim_allowed': claim_allowed,
        'claim': claims[verdict],
        'assumption': ('the sign-flip test is a seed-level sign test under '
                       'symmetric paired differences; the two PPLs come from '
                       'the same checkpoint and the same evaluation rows'),
    }


def lesion_panel_audit(records, *, relative_pct=LESION_POLICY['relative_pct'],
                       alpha=LESION_POLICY['alpha'],
                       min_n=LESION_POLICY['min_n'],
                       require_manipulation=True):
    """Validate and summarize a within-checkpoint lesion panel.

    ``records`` is an iterable of dicts (or a mapping with a ``records``
    key).  Each record pairs the intact and lesioned PPL for one
    ``(component, arm, seed)`` and must carry a checkpoint identity.  The
    audit fails closed on a missing/duplicate seed, a non-finite or
    non-positive PPL, a component/arm outside ``LESION_DESIGN``, or a record
    whose ``weights_unchanged`` is explicitly false.  It does not read a
    results directory; the gate feeds it the saved artifact.
    """
    if isinstance(records, dict):
        records = records.get('records')
    if records is None:
        records = []
    if isinstance(records, (str, bytes)) or not hasattr(records, '__iter__'):
        raise ValueError('lesion_panel_audit: records must be an iterable of mappings')
    if isinstance(relative_pct, (bool, np.bool_)):
        raise ValueError(f'relative_pct must be a finite positive number, got {relative_pct!r}')
    relative_pct = float(relative_pct)
    if not math.isfinite(relative_pct) or relative_pct <= 0.0:
        raise ValueError(f'relative_pct must be a finite positive number, got {relative_pct!r}')
    alpha = _validated_alpha(alpha, context='lesion_panel_audit alpha')
    _min_n = _as_seed_int(min_n)
    if _min_n is None or _min_n < 0:
        raise ValueError(f'min_n must be a non-negative integer, got {min_n!r}')
    min_n = int(_min_n)
    if not isinstance(require_manipulation, bool):
        raise ValueError(
            f'require_manipulation must be boolean, got {require_manipulation!r}')
    groups = {}
    problems = []
    warnings = []
    n_records = 0
    for pos, rec in enumerate(records):
        n_records += 1
        where = f'records[{pos}]'
        if not isinstance(rec, dict):
            problems.append(f'{where}: record must be a mapping')
            continue
        component = rec.get('component')
        if not isinstance(component, str) or component not in LESION_DESIGN:
            problems.append(f'{where}: unknown component {component!r}')
            continue
        arm = rec.get('arm')
        if not isinstance(arm, str) or not arm.strip():
            problems.append(f'{where}: arm must be a non-empty string')
            continue
        if arm not in LESION_DESIGN[component]['arms']:
            problems.append(
                f'{where}: arm {arm!r} is not declared for lesion component '
                f'{component!r}')
            continue
        seed = _as_seed_int(rec.get('seed'))
        if seed is None:
            problems.append(f'{where}: seed must be an integer, got {rec.get("seed")!r}')
            continue
        ckpt = rec.get('checkpoint_sha256')
        if not isinstance(ckpt, str) or not ckpt.strip():
            problems.append(f'{where}: checkpoint_sha256 is required for a within-checkpoint lesion')
            continue
        try:
            intact = _lesion_ppl_values([rec.get('intact_ppl')], f'{where}.intact_ppl')[0]
            lesion = _lesion_ppl_values([rec.get('lesion_ppl')], f'{where}.lesion_ppl')[0]
        except ValueError as exc:
            problems.append(str(exc))
            continue
        sens = rec.get('sensitivity')
        if sens is None:
            sens = {'changed': rec.get('lesion_changed_output'),
                    'weights_unchanged': rec.get('weights_unchanged')}
        elif not isinstance(sens, dict):
            problems.append(f'{where}: sensitivity must be a mapping or null')
            continue
        if sens.get('changed') is not None and not isinstance(sens.get('changed'), bool):
            problems.append(f'{where}: sensitivity.changed must be boolean or null')
            continue
        if (sens.get('weights_unchanged') is not None
                and not isinstance(sens.get('weights_unchanged'), bool)):
            problems.append(f'{where}: sensitivity.weights_unchanged must be boolean or null')
            continue
        if sens.get('weights_unchanged') is False:
            problems.append(f'{where}: lesion changed the model weights')
            continue
        if require_manipulation and sens.get('changed') is not True:
            warnings.append(
                f'{where}: no passed output manipulation check; the lesion '
                'PPL cannot support an inference-use claim')
        eval_fp = rec.get('eval_fp')
        if eval_fp is not None and (not isinstance(eval_fp, str) or not eval_fp.strip()):
            problems.append(f'{where}: eval_fp must be a non-empty string or null')
            continue
        key = (component, arm)
        g = groups.setdefault(key, {'component': component, 'arm': arm,
                                    'records': [], 'seeds': set(),
                                    'checkpoints': set(), 'eval_fps': set()})
        if seed in g['seeds']:
            problems.append(f'{where}: duplicate seed {seed} for {component}/{arm}')
            continue
        g['seeds'].add(seed)
        g['checkpoints'].add(ckpt)
        if eval_fp:
            g['eval_fps'].add(eval_fp)
        g['records'].append({'seed': seed, 'intact_ppl': intact,
                             'lesion_ppl': lesion,
                             'checkpoint_sha256': ckpt,
                             'eval_fp': eval_fp,
                             'sensitivity': sens})
    panels = {}
    for (component, arm), g in sorted(groups.items()):
        if len(g['eval_fps']) > 1:
            problems.append(
                f'{component}/{arm}: records use {len(g["eval_fps"])} distinct '
                'eval_fp values; a lesion panel must score the same rows')
        elif (0 < sum(1 for r in g['records'] if r['eval_fp'])
              < len(g['records'])):
            # Partial attestation is not a mixed panel, but "same rows" cannot
            # be verified for the records that omit the fingerprint.  Surface
            # it instead of treating the group as fully attested.
            warnings.append(
                f'{component}/{arm}: only '
                f'{sum(1 for r in g["records"] if r["eval_fp"])} of '
                f'{len(g["records"])} records carry eval_fp; the panel cannot '
                'verify that every seed scored the same rows')
        if len(g['records']) > 1 and len(g['checkpoints']) < len(g['records']):
            warnings.append(
                f'{component}/{arm}: {len(g["records"])} seeds share only '
                f'{len(g["checkpoints"])} distinct checkpoint(s); reusing one '
                'checkpoint across training seeds is pseudo-replication, not '
                'independent lesion evidence')
        rows = sorted(g['records'], key=lambda r: r['seed'])
        sens = [r['sensitivity'] for r in rows]
        # A group's manipulation check is the conjunction of its records.
        combined = {
            'changed': (all(s.get('changed') is True for s in sens)
                        if all(s.get('changed') is not None for s in sens) else None),
            'weights_unchanged': (all(s.get('weights_unchanged') is True for s in sens)
                                 if all(s.get('weights_unchanged') is not None for s in sens)
                                 else None),
            'component': component,
            'arm': arm,
            'records': len(rows),
        }
        eff = lesion_effect(
            [r['intact_ppl'] for r in rows],
            [r['lesion_ppl'] for r in rows],
            seeds=[r['seed'] for r in rows],
            component=component, relative_pct=relative_pct, alpha=alpha,
            min_n=min_n, sensitivity=combined)
        panels[f'{component}/{arm}'] = {
            **eff,
            'checkpoints': sorted(g['checkpoints']),
            'n_checkpoints': len(g['checkpoints']),
            'eval_fps': sorted(g['eval_fps']),
            'seeds': [r['seed'] for r in rows],
        }
    status = 'ok'
    if problems:
        status = 'fail'
    elif not panels:
        status = 'not_verifiable'
    elif warnings or any(p['verdict'] in ('not_checked', 'underpowered',
                                          'inconclusive', 'invalid')
                         for p in panels.values()):
        status = 'warn'
    counts = {}
    for p in panels.values():
        counts[p['verdict']] = counts.get(p['verdict'], 0) + 1
    return {
        'kind': 'lesion_panel',
        'status': status,
        'n_records': n_records,
        'n_panels': len(panels),
        'panels': panels,
        'verdict_counts': counts,
        'claim_allowed_panels': sorted(
            k for k, p in panels.items() if p.get('claim_allowed')),
        'necessary_panels': sorted(
            k for k, p in panels.items() if p.get('verdict') == 'necessary'),
        'redundant_panels': sorted(
            k for k, p in panels.items() if p.get('verdict') == 'redundant'),
        'inert_panels': sorted(
            k for k, p in panels.items() if p.get('verdict') == 'inert'),
        'harmful_panels': sorted(
            k for k, p in panels.items() if p.get('verdict') == 'harmful'),
        'problems': problems,
        'warnings': warnings,
        'policy': {'relative_pct': float(relative_pct),
                   'alpha': float(alpha), 'min_n': int(min_n),
                   'require_manipulation': bool(require_manipulation),
                   'unit': 'seed',
                   'estimand': 'log(PPL_lesion / PPL_intact), same checkpoint/rows'},
    }


def lesion_design_audit(design=None, *, factory=None, n_layers=2,
                        vocab=64, d=32, n_heads=2, d_head=16, seq_len=16,
                        present=None):
    """Design-time gate: every declared lesion is constructed and applies.

    It builds a tiny model for each declared arm through the real arm factory
    and checks that (a) the component is present, (b) ``component_lesion``
    changes a declared config field or submodule, (c) the arm is scheduled and
    belongs to a declared leave-one-out family, and (d) the mechanism key is
    covered by ``ABLATION_INTERVENTION_KEYS``.  A declaration that cannot be
    constructed is a failure, not an untested note.
    """
    design = LESION_DESIGN if design is None else design
    factory = make_layer_cfgs if factory is None else factory
    _nl = _as_seed_int(n_layers)
    if _nl is None or _nl < 1:
        raise ValueError(f'n_layers must be a positive integer, got {n_layers!r}')
    n_layers = int(_nl)
    for _name, _value in (('vocab', vocab), ('d', d), ('n_heads', n_heads),
                          ('d_head', d_head), ('seq_len', seq_len)):
        _v = _as_seed_int(_value)
        if _v is None or _v < 1:
            raise ValueError(f'{_name} must be a positive integer, got {_value!r}')
    out = {'kind': 'lesion_design', 'status': 'ok', 'components': {},
           'violations': [], 'warnings': [],
           'policy': dict(LESION_POLICY)}
    if not isinstance(design, dict) or not design:
        out['status'] = 'fail'
        out['violations'].append('LESION_DESIGN must be a non-empty mapping')
        return out
    if set(design) != set(LESION_COMPONENTS):
        out['violations'].append(
            f'LESION_DESIGN components must be exactly {sorted(LESION_COMPONENTS)}, '
            f'got {sorted(design)}')
    try:
        scheduled = set(RUN_ABL['variants']) if present is None else set(present)
    except TypeError:
        scheduled = set()
    for component, entry in sorted(design.items()):
        rec = {'status': 'ok', 'arms': {}, 'note': None}
        out['components'][component] = rec
        if component not in LESION_COMPONENTS:
            rec['status'] = 'fail'
            out['violations'].append(f'{component}: not a declared lesion component')
            continue
        if not isinstance(entry, dict):
            rec['status'] = 'fail'
            out['violations'].append(f'{component}: declaration must be a mapping')
            continue
        source = entry.get('source_component')
        if not isinstance(source, str) or source not in ABLATION_DESIGN:
            rec['status'] = 'fail'
            out['violations'].append(
                f'{component}: source_component {source!r} is not in ABLATION_DESIGN')
            # A list/dict source is unhashable: the later
            # ABLATION_INTERVENTION_KEYS.get(source) lookup would raise
            # TypeError instead of reporting a failed declaration.  Stop this
            # component here and keep auditing the other declarations.
            if not isinstance(source, str):
                continue
        arms = entry.get('arms')
        if (not isinstance(arms, (list, tuple)) or not arms
                or not all(isinstance(a, str) and a.strip() for a in arms)):
            rec['status'] = 'fail'
            out['violations'].append(f'{component}: arms must be a non-empty list of arm ids')
            continue
        keys = entry.get('mechanism_keys')
        if not isinstance(keys, (set, frozenset)) or not keys:
            rec['status'] = 'fail'
            out['violations'].append(f'{component}: mechanism_keys must be a non-empty set')
            continue
        declared_keys = set()
        for _pair, _keys in (ABLATION_INTERVENTION_KEYS.get(source) or {}).items():
            declared_keys |= set(_keys)
        if not set(keys).issubset(declared_keys):
            rec['status'] = 'fail'
            out['violations'].append(
                f'{component}: mechanism key(s) {sorted(set(keys) - declared_keys)} '
                f'are not declared in ABLATION_INTERVENTION_KEYS[{source!r}]')
        pair_arms = {a for a, _b, *_ in (ABLATION_DESIGN.get(source) or [])}
        pair_arms |= {b for _a, b, *_ in (ABLATION_DESIGN.get(source) or [])}
        for arm in arms:
            arec = {'scheduled': arm in scheduled,
                    'in_leave_one_out_family': arm in pair_arms,
                    'state': None, 'applied': False, 'changed_fields': []}
            rec['arms'][arm] = arec
            if arm not in scheduled:
                rec['status'] = 'fail'
                out['violations'].append(f'{component}/{arm}: arm is not scheduled in RUN_ABL')
            if arm not in pair_arms:
                rec['status'] = 'fail'
                out['violations'].append(
                    f'{component}/{arm}: arm is not in a declared {source!r} contrast')
            try:
                cfgs = list(factory(n_layers, arm))
            except Exception as exc:
                rec['status'] = 'fail'
                out['violations'].append(
                    f'{component}/{arm}: factory failed ({type(exc).__name__}: {exc})')
                continue
            if len(cfgs) != n_layers:
                rec['status'] = 'fail'
                out['violations'].append(
                    f'{component}/{arm}: factory returned {len(cfgs)} configs for {n_layers} layers')
                continue
            _rng = torch.get_rng_state()
            _cuda_rng = (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None)
            try:
                torch.manual_seed(1234)
                model = SmallGPT(vocab, d, n_layers, n_heads, d_head, seq_len, cfgs)
            except Exception as exc:
                rec['status'] = 'fail'
                out['violations'].append(
                    f'{component}/{arm}: model construction failed '
                    f'({type(exc).__name__}: {exc})')
                continue
            finally:
                torch.set_rng_state(_rng)
                if _cuda_rng is not None and torch.cuda.is_available():
                    try:
                        torch.cuda.set_rng_state_all(_cuda_rng)
                    except Exception:
                        pass
            state = component_lesion_state(model, component)
            arec['state'] = state
            try:
                live_layers = _lesion_layers(model, component)
                with component_lesion(model, component) as live:
                    arec['applied'] = True
                    arec['changed_fields'] = sorted(live.get('cfg_overrides') or {})
                    arec['module_overrides'] = list(live.get('module_overrides') or [])
                    # Validate the constructed intervention against the
                    # declaration, not just its label: a field that is named
                    # but not actually set would pass a name-only check.
                    for _i, _attn, _cfg in live_layers:
                        for _key, _want in (entry.get('cfg_overrides') or {}).items():
                            if getattr(_attn.cfg, _key, object()) != _want:
                                rec['status'] = 'fail'
                                out['violations'].append(
                                    f'{component}/{arm}: layer {_i} cfg.{_key} '
                                    f'is {getattr(_attn.cfg, _key, None)!r}, '
                                    f'expected {_want!r}')
                        for _name, _want in (entry.get('module_overrides') or ()):
                            if getattr(_attn, _name, object()) is not _want:
                                rec['status'] = 'fail'
                                out['violations'].append(
                                    f'{component}/{arm}: layer {_i} module '
                                    f'{_name!r} was not replaced by the declared value')
            except Exception as exc:
                rec['status'] = 'fail'
                out['violations'].append(
                    f'{component}/{arm}: component_lesion failed '
                    f'({type(exc).__name__}: {exc})')
                continue
            if state.get('status') == 'absent':
                rec['status'] = 'fail'
                out['violations'].append(
                    f'{component}/{arm}: the constructed arm has no {component} state')
            if state.get('status') == 'trivial':
                out['warnings'].append(
                    f'{component}/{arm}: component is at its no-op initialisation; '
                    'the forward manipulation check must be run on the trained checkpoint')
        rec['note'] = entry.get('note')
    if out['violations']:
        out['status'] = 'fail'
    elif any(c.get('status') == 'fail' for c in out['components'].values()):
        out['status'] = 'fail'
    return out
