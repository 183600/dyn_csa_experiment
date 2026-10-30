# ---- functional manipulation fidelity for comparative / ablation designs ----
# The design gates answer three different questions, and none of them alone is
# enough to interpret a null ablation result:
#
#   1. `single_factor_design_audit`   -- do the two arms differ in exactly the
#      declared AttnCfg keys?  (definition-site construct validity)
#   2. `single_factor_initialization_audit` -- do non-intervention parameters
#      stay bit-identical at the same seed?  (no hidden initialization factor)
#   3. THIS gate -- does the declared intervention actually change the model's
#      function or its training path?  A cfg field can be present and differ
#      while being inert on the real computation (a selector whose top-k
#      exceeds the number of available blocks, a content table that is never
#      read, a "frozen" parameter that still receives a gradient through a
#      second path).  In that case both arms compute the same function and a
#      null PPL contrast is not evidence that the module is unnecessary; it is
#      evidence that the manipulation did not manipulate anything.
#
# This is the ablation analogue of the within-checkpoint lesion manipulation
# check (design_notes section 97).  It is a zero-GPU design audit: it builds
# tiny models through the real arm factory, runs one fixed forward and one
# fixed backward, and records *which* evidence of an intervention exists:
#
#   forward-mode:  the two arms' logits differ at initialization;
#   training-mode: logits may be identical at initialization (identity fusion,
#                  a frozen boundary offset, a loss-term-only regularizer), but
#                  the trainable-parameter / gradient / objective descriptor
#                  differs.
#
# A declared `forward` contrast with identical logits is `inert`; a declared
# `training` contrast with identical trainability, gradient and objective is
# also `inert`.  Both fail closed.  The audit never reads a result file, never
# changes a p-value/Holm family/frozen verdict, and restores the caller's RNG
# state even when construction fails.
ABLATION_MANIPULATION_POLICY = {
    'version': 1,
    'n_layers': 2,
    'probe_tokens': 256,
    'vocab': 32,
    'd': 32,
    'n_heads': 2,
    'd_head': 16,
    'seed': 1234,
    'forward_seed': 20261030,
    'forward_rtol': 1e-6,
    'gradient_rtol': 1e-6,
    'objective_keys': (
        'comp_lambda', 'comp_lambda_effective', 'comp_reg_applies',
        'comp_reg_configured_layers', 'comp_reg_layers', 'boundary_policy',
        'delta_lr_effective', 'delta_trainable',
        'delta_trainable_param_count', 'gate_modes', 'loss_terms',
    ),
    'role': ('functional manipulation check only: it verifies that a declared '
             'ablation changes the forward function or the training path.  It '
             'does not test the outcome, estimate an effect, or replace the '
             'config/initialization gates.'),
}

# One declaration per ABLATION_DESIGN component.  `forward` is the expected
# case for an architectural presence/alignment/selection intervention;
# `training` is the expected case when the two arms are the same function at
# initialization and the intervention is trainability or the objective.
#
# `task_learned_boundary` is `training`: the REINFORCE policy does not change
# the initial hard-cut forward on the fixed probe, but it changes the objective
# and the gradient path.  `local_fusion` is `training` because the identity
# kernel makes the initial forward a no-op by construction; the added conv is
# still a real training intervention (it exists and receives gradients).
ABLATION_MANIPULATION_DESIGN = {
    'dynamic_blocking': {
        'mode': 'forward',
        'note': ('fixed vs adaptive cuts change the block geometry at '
                 'initialization; identical logits would mean the cut policy '
                 'never fired on the probe'),
    },
    'dynamic_blocking_learning': {
        'mode': 'training',
        'note': ('learned vs frozen boundary offset is the same forward '
                 'function at initialization; only trainability/objective may '
                 'differ'),
    },
    'learned_selection': {
        'mode': 'forward',
        'note': ('learned vs random top-k must select different blocks on a '
                 'probe with more blocks than top-k; if all blocks fit in the '
                 'read budget the selector is inert'),
    },
    'content_pooling': {
        'mode': 'forward',
        'note': ('pooled vs zero block content must change the read at '
                 'initialization; a zero projection/output would make the '
                 'presence control a no-op'),
    },
    'content_alignment': {
        'mode': 'forward',
        'note': ('aligned vs previous-block content must change the read at '
                 'initialization; if the content path is unused the alignment '
                 'placebo is inert'),
    },
    'attention_sink': {
        'mode': 'forward',
        'note': ('a sink slot changes the softmax denominator even when its '
                 'learned value is zero (design_notes section 97), so the on/off '
                 'forward must differ'),
    },
    'attention_sink_learning': {
        'mode': 'forward',
        'note': ('learned (zero-initialized) vs frozen random sink must change '
                 'the initial read; identical outputs mean the random values '
                 'were not installed or the sink is not read'),
    },
    'compression_regularizer': {
        'mode': 'training',
        'note': ('the loss-term ablation is the same forward function at '
                 'initialization; it must change the comp_reg objective and the '
                 'gate gradient path'),
    },
    'task_learned_boundary': {
        'mode': 'training',
        'note': ('the REINFORCE boundary policy may not change the initial '
                 'hard-cut forward on a fixed probe; it must change the '
                 'objective/loss terms (and normally the gradient path)'),
    },
    'local_fusion': {
        'mode': 'training',
        'note': ('the depthwise fusion kernel is initialized to identity, so '
                 'the initial forward is allowed to be identical; the optional '
                 'convolution must exist and receive gradients'),
    },
    'local_fusion_learning': {
        'mode': 'forward',
        'note': ('learned identity fusion vs a frozen random kernel must '
                 'change the initial read; identical outputs mean the random '
                 'kernel was not installed or the conv is not used'),
    },
    'hybrid_hierarchy': {
        'mode': 'forward',
        'note': ('a CSA->HCA layer swap changes the attention operator itself, '
                 'so the initial logits must differ'),
    },
}


def _manip_pair_items(design):
    """Well-formed ``(component, a, b)`` items and malformed-entry names."""
    items, bad = [], []
    if not isinstance(design, dict):
        return items, [f'design is a {type(design).__name__}, not a mapping']
    for comp, contrasts in design.items():
        if not isinstance(contrasts, (list, tuple)):
            bad.append(f'{comp}: contrasts must be a list/tuple')
            continue
        for i, item in enumerate(contrasts):
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                bad.append(f'{comp}[{i}]: expected a (treatment, control, ...) tuple')
                continue
            a, b = item[0], item[1]
            if not (isinstance(a, str) and a.strip()
                    and isinstance(b, str) and b.strip()):
                bad.append(f'{comp}[{i}]: treatment/control must be arm names')
                continue
            items.append((str(comp), a.strip(), b.strip()))
    return items, bad


def _manip_float(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'expected a finite number, got {value!r}')
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'expected a finite number, got {value!r}') from None
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f'expected a finite non-negative number, got {value!r}')
    return out


def _manip_model(factory, n_layers, arm, cfg):
    """Build one tiny arm through the real factory at a fixed seed."""
    with _preserve_rng_state():
        torch.manual_seed(int(cfg['seed']))
        cfgs = factory(int(n_layers), arm)
        model = SmallGPT(int(cfg['vocab']), int(cfg['d']), int(n_layers),
                         int(cfg['n_heads']), int(cfg['d_head']),
                         int(cfg['probe_tokens']), cfgs)
    model.eval()
    return model


def _manip_logits(model, ids, cfg):
    with torch.no_grad():
        torch.manual_seed(int(cfg['forward_seed']))
        return model(ids).detach()


def _manip_gradient_state(model, ids, targets, cfg):
    """One fixed backward pass on the declared training objective.

    The CE term is always present; the compression term is added only when it
    is part of the graph.  A REINFORCE boundary term is deliberately not
    invented here: this check is about the objective descriptor and the
    parameter/gradient path that the arm factory defines, not about a reward
    value that only the training loop can supply.
    """
    model.zero_grad(set_to_none=True)
    torch.manual_seed(int(cfg['forward_seed']))
    loss = model.ce_loss(ids, targets, int(cfg['vocab']))
    reg = getattr(model, 'comp_reg', None)
    if (isinstance(reg, torch.Tensor) and reg.numel() == 1
            and bool(reg.requires_grad)):
        loss = loss + 0.05 * reg
    loss.backward()
    state = {}
    for name, param in model.named_parameters():
        req = bool(param.requires_grad)
        grad = param.grad
        norm = 0.0
        finite = True
        if req and grad is not None:
            g = grad.detach().float()
            if g.numel():
                norm = float(torch.linalg.vector_norm(g))
                finite = bool(torch.isfinite(g).all())
        state[name] = {'requires_grad': req, 'grad_norm': norm,
                       'grad_finite': finite}
    model.zero_grad(set_to_none=True)
    return state


def _manip_objective(model):
    try:
        return train_objective(model, comp_lambda=0.05, delta_lr_mult=10.0)
    except Exception as exc:  # a probe must not crash the design audit
        return {'_error': f'{type(exc).__name__}: {exc}'}


def _manip_canon(value):
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError, OverflowError):
        return repr(value)


def _manip_compare(a, b, cfg, forward_tol, gradient_tol):
    """Compare two built arms and classify the evidence of an intervention."""
    ids = torch.randint(0, int(cfg['vocab']),
                        (1, int(cfg['probe_tokens'])),
                        generator=torch.Generator().manual_seed(11))
    targets = torch.randint(0, int(cfg['vocab']),
                            (1, int(cfg['probe_tokens'])),
                            generator=torch.Generator().manual_seed(12))
    la = _manip_logits(a, ids, cfg)
    lb = _manip_logits(b, ids, cfg)
    max_abs = float((la - lb).abs().max()) if la.numel() else 0.0
    scale = max(float(la.abs().max()) if la.numel() else 0.0,
                float(lb.abs().max()) if lb.numel() else 0.0, 1.0)
    max_rel = max_abs / scale
    forward_changed = bool(max_rel > forward_tol)

    ga = _manip_gradient_state(a, ids, targets, cfg)
    gb = _manip_gradient_state(b, ids, targets, cfg)
    names_a, names_b = set(ga), set(gb)
    only_a = sorted(names_a - names_b)
    only_b = sorted(names_b - names_a)
    common = sorted(names_a & names_b)
    trainability_diff = sorted(
        n for n in common
        if ga[n]['requires_grad'] != gb[n]['requires_grad'])
    grad_diff = []
    nonfinite = []
    for n in common:
        if not ga[n]['grad_finite'] or not gb[n]['grad_finite']:
            nonfinite.append(n)
            continue
        va, vb = ga[n]['grad_norm'], gb[n]['grad_norm']
        denom = max(abs(va), abs(vb), 1e-12)
        if abs(va - vb) / denom > gradient_tol:
            grad_diff.append(n)
    param_changed = bool(only_a or only_b or trainability_diff)

    oa = _manip_objective(a)
    ob = _manip_objective(b)
    objective_diff = sorted(
        k for k in cfg['objective_keys']
        if _manip_canon(oa.get(k)) != _manip_canon(ob.get(k)))
    objective_changed = bool(objective_diff)
    training_changed = bool(param_changed or grad_diff or objective_changed)

    evidence = ('forward' if forward_changed else
                'training' if training_changed else 'none')
    return {
        'forward': {
            'changed': forward_changed,
            'max_abs_diff': float(max_abs),
            'max_rel_diff': float(max_rel),
            'tolerance': float(forward_tol),
        },
        'parameter': {
            'changed': param_changed,
            'names_only_a': only_a[:8],
            'n_names_only_a': len(only_a),
            'names_only_b': only_b[:8],
            'n_names_only_b': len(only_b),
            'trainability_diff': trainability_diff[:8],
            'n_trainability_diff': len(trainability_diff),
        },
        'gradient': {
            'changed': bool(grad_diff),
            'names': grad_diff[:8],
            'n_names': len(grad_diff),
            'nonfinite': nonfinite[:8],
            'n_nonfinite': len(nonfinite),
            'tolerance': float(gradient_tol),
        },
        'objective': {
            'changed': objective_changed,
            'keys': objective_diff,
        },
        'training_changed': training_changed,
        'evidence': evidence,
    }


def ablation_manipulation_audit(design=None, declarations=None, *,
                                factory=None, n_layers=None, probe_tokens=None,
                                seed=None, forward_tol=None, gradient_tol=None):
    """Check that every declared ablation is a REAL functional intervention.

    For each ``ABLATION_DESIGN`` contrast this builds both arms through the
    real factory at the same seed and records:

      * whether their logits differ on a fixed probe (forward intervention);
      * whether their parameter names / ``requires_grad`` flags differ;
      * whether their gradients on a fixed CE(+comp_reg) backward differ;
      * whether their ``train_objective`` descriptor differs.

    A component declares whether the intervention is expected to be visible at
    initialization (``mode='forward'``) or only in the training path
    (``mode='training'``).  The declared mode is a hard gate: a `forward`
    contrast with identical logits, or a `training` contrast with identical
    parameters/gradients/objective, is ``inert`` and fails.  This is what makes
    a later null PPL contrast interpretable: the manipulation was shown to act
    on the model, so "no difference" is about the outcome, not about a dead
    flag.

    The audit is zero-GPU, never reads a result file, and preserves the
    caller's RNG state.  It does not estimate an effect or change any p-value,
    Holm family, ``verified`` verdict or frozen decision.
    """
    design = ABLATION_DESIGN if design is None else design
    declarations = (ABLATION_MANIPULATION_DESIGN if declarations is None
                    else declarations)
    factory = make_layer_cfgs if factory is None else factory
    cfg = dict(ABLATION_MANIPULATION_POLICY)
    if n_layers is not None:
        cfg['n_layers'] = n_layers
    if probe_tokens is not None:
        cfg['probe_tokens'] = probe_tokens
    if seed is not None:
        cfg['seed'] = seed
    if forward_tol is not None:
        cfg['forward_rtol'] = forward_tol
    if gradient_tol is not None:
        cfg['gradient_rtol'] = gradient_tol
    out = {
        'kind': 'ablation_manipulation',
        'version': int(cfg['version']),
        'status': 'ok',
        'policy': dict(cfg),
        'factory': getattr(factory, '__name__', repr(factory)),
        'contrasts': {},
        'inert': [],
        'not_verifiable': [],
        'violations': [],
        'warnings': [],
        'role': cfg['role'],
    }
    try:
        cfg['n_layers'] = int(_as_seed_int(cfg['n_layers']))
        cfg['probe_tokens'] = int(_as_seed_int(cfg['probe_tokens']))
        cfg['seed'] = int(_as_seed_int(cfg['seed']))
    except (TypeError, ValueError):
        out['status'] = 'fail'
        out['violations'].append(
            'n_layers/probe_tokens/seed must be non-negative integers')
        return out
    if cfg['n_layers'] < 1 or cfg['probe_tokens'] < 1 or cfg['seed'] < 0:
        out['status'] = 'fail'
        out['violations'].append(
            'n_layers/probe_tokens must be positive and seed non-negative')
        return out
    if not isinstance(design, dict) or not design:
        out['status'] = 'fail'
        out['violations'].append('ABLATION_DESIGN must be a non-empty mapping')
        return out
    if not isinstance(declarations, dict) or not declarations:
        out['status'] = 'fail'
        out['violations'].append(
            'ABLATION_MANIPULATION_DESIGN must be a non-empty mapping')
        return out
    try:
        cfg['forward_rtol'] = _manip_float(cfg['forward_rtol'])
        cfg['gradient_rtol'] = _manip_float(cfg['gradient_rtol'])
    except ValueError as exc:
        out['status'] = 'fail'
        out['violations'].append(str(exc))
        return out
    items, malformed = _manip_pair_items(design)
    out['violations'].extend(malformed)
    if not items:
        out['status'] = 'fail'
        out['violations'].append('no well-formed ablation contrast was declared')
        return out
    out['policy'] = dict(cfg)
    # The tiny arms are immutable during the probe (no optimizer step), so one
    # model per arm can be reused across its contrasts.  This also ensures a
    # single arm gets one initialization rather than a fresh draw per pair.
    models = {}
    seen = set()
    for comp, a, b in items:
        if (comp, a, b) in seen:
            out['violations'].append(
                f'{comp}: duplicate declared contrast {a} vs {b}')
        seen.add((comp, a, b))
        spec = declarations.get(comp)
        mode = spec.get('mode') if isinstance(spec, dict) else None
        if mode not in ('forward', 'training'):
            out['violations'].append(
                f'{comp}: no manipulation declaration with mode '
                f"'forward' or 'training'")
            continue
        if isinstance(spec.get('note'), str) and spec['note'].strip():
            note = spec['note'].strip()
        else:
            note = None
        label = f'{comp}: {a} vs {b}'
        try:
            with _preserve_rng_state():
                for arm in (a, b):
                    if arm not in models:
                        models[arm] = _manip_model(factory, cfg['n_layers'],
                                                   arm, cfg)
                cmp = _manip_compare(models[a], models[b], cfg,
                                     cfg['forward_rtol'], cfg['gradient_rtol'])
        except Exception as exc:
            out['not_verifiable'].append(label)
            out['violations'].append(
                f'{label}: manipulation probe could not be built '
                f'({type(exc).__name__}: {exc})')
            out['contrasts'][label] = {
                'component': comp, 'a': a, 'b': b, 'mode': mode,
                'note': note, 'status': 'not_verifiable',
                'reason': f'{type(exc).__name__}: {exc}',
            }
            continue
        if cmp['gradient']['n_nonfinite']:
            cmp['status'] = 'not_verifiable'
            cmp['reason'] = ('the probe produced non-finite gradients; the '
                             'training-path evidence cannot be interpreted')
            out['not_verifiable'].append(label)
            out['violations'].append(f'{label}: {cmp["reason"]}')
        else:
            declared_ok = (cmp['forward']['changed'] if mode == 'forward'
                           else cmp['training_changed'])
            if not declared_ok:
                cmp['status'] = 'fail'
                cmp['reason'] = (
                    f'declared {mode} intervention is inert: '
                    + ('the two arms produce identical logits'
                       if mode == 'forward'
                       else 'parameters, gradients and objective are identical'))
                out['inert'].append(label)
                out['violations'].append(f'{label}: {cmp["reason"]}')
            else:
                cmp['status'] = 'ok'
                cmp['reason'] = None
        cmp.update({'component': comp, 'a': a, 'b': b, 'mode': mode,
                    'note': note})
        out['contrasts'][label] = cmp
    # Declarations are a closed contract: a component in the manifest that is
    # no longer in ABLATION_DESIGN is stale, and every declared component must
    # be covered by at least one contrast.
    design_comps = {comp for comp, _a, _b in items}
    for comp in sorted(set(declarations) - design_comps):
        out['violations'].append(
            f'{comp}: manipulation declaration has no matching ABLATION_DESIGN '
            'contrast (stale declaration)')
    for comp in sorted(design_comps - set(declarations)):
        out['violations'].append(
            f'{comp}: ABLATION_DESIGN contrast has no manipulation declaration')
    out['n_contrasts'] = len(items)
    out['n_forward_mode'] = sum(
        1 for r in out['contrasts'].values() if r.get('mode') == 'forward')
    out['n_training_mode'] = sum(
        1 for r in out['contrasts'].values() if r.get('mode') == 'training')
    out['n_forward_evidence'] = sum(
        1 for r in out['contrasts'].values()
        if (r.get('forward') or {}).get('changed') is True)
    out['n_training_evidence'] = sum(
        1 for r in out['contrasts'].values()
        if r.get('training_changed') is True)
    if out['violations']:
        out['status'] = 'fail'
    elif not out['contrasts']:
        out['status'] = 'not_verifiable'
    else:
        out['status'] = 'ok'
    return out
