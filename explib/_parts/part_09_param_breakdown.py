
def param_breakdown(m):
    """Parameter counts by functional group, for every record.

    A headline comparison that only reports `params` invites a misleading
    question "which parameters, and is the comparison capacity-matched?".
    This is a deterministic structure-only count (no forward pass, no RNG):
    total / attention / attention projections / indexer / MLP / norms /
    embeddings / LM head, plus the active count (equal to total here; no MoE).
    """
    out = {'total': 0, 'attention': 0, 'attention_qo': 0, 'attention_kv': 0,
           'attention_indexer': 0, 'mlp': 0, 'norms': 0, 'embedding': 0,
           'lm_head': 0, 'other': 0, 'active': 0, 'n_layers': 0, 'n_heads': 0, 'd_head': 0}
    for name, p in m.named_parameters():
        n = int(p.numel())
        out['total'] += n
        if name.startswith('tok.') or name.startswith('pos.'):
            out['embedding'] += n
        elif name.startswith('head.'):
            out['lm_head'] += n
        elif name.startswith('norm.') or '.norm' in name:
            out['norms'] += n
        elif name.startswith('blocks.'):
            if '.mlp.' in name:
                out['mlp'] += n
                out['attention'] += 0
            elif '.attn.' in name:
                out['attention'] += n
                if any(k in name for k in ('W_DQ', 'W_DK', 'W_w', 'index')):
                    out['attention_indexer'] += n
                elif any(k in name for k in ('W_kvhead', 'W_aKV', 'W_bKV', 'W_aZ', 'W_bZ', 'B_pos')):
                    out['attention_kv'] += n
                else:
                    out['attention_qo'] += n
            else:
                out['other'] += n
        else:
            out['other'] += n
    try:
        out['n_layers'] = len(m.blocks)
        first = m.blocks[0].attn
        out['n_heads'] = int(getattr(first, 'nh', 0) or 0)
        out['d_head'] = int(getattr(first, 'hd', 0) or 0)
    except Exception:
        pass
    out['active'] = out['total']
    out['attention_share'] = (out['attention'] / out['total']) if out['total'] else 0.0
    return out

def kv_cache_elements_per_token(m):
    """KV-cache elements stored per token per layer (dtype-independent).

    Full attention keeps K and V of width n_heads*d_head; CSA/HCA keep the
    compressed [c_kv | c_kv] lanes (plus the indexer's query projection is
    weights, not cache).  This is the structural quantity behind any
    "KV-cache ratio" claim; storage bytes additionally need dtype/allocator
    overhead and are measured by the harness, not inferred here.
    """
    total = 0
    per_layer = []
    for blk in getattr(m, 'blocks', []):
        a = blk.attn
        c = a.cfg
        if c.kind == 'full':
            n = 2 * int(getattr(a, 'nh', 0)) * int(getattr(a, 'hd', 0))
        else:
            n = 2 * int(getattr(c, 'c_kv', 0))
        per_layer.append(int(n))
        total += int(n)
    return {'elements_per_token_total': int(total), 'per_layer': per_layer,
            'note': 'elements, not bytes; multiply by dtype size and add allocator/block overhead for measured memory'}

def open_block_config(m):
    """Machine-readable descriptor of every layer's read rule.

    This is part of the record metadata (and the checkpoint payload).  It
    pins the two experimental factors — maximum block length and open-block
    reading rule — together with the position-capacity policy that is held
    constant across arms, so a report cannot mix a record from one panel with
    another panel's settings.
    """
    rows = []
    for li, blk in enumerate(getattr(m, 'blocks', []) or []):
        cfg = getattr(blk, 'attn', None)
        cfg = getattr(cfg, 'cfg', None) if cfg is not None else None
        if cfg is None:
            continue
        rows.append({
            'layer': int(li),
            'kind': str(getattr(cfg, 'kind', '')),
            'max_block': getattr(cfg, 'max_block', None),
            'max_block_tokens': getattr(cfg, 'max_block', None),
            'pool_position_capacity': int(cfg.position_capacity()) if hasattr(cfg, 'position_capacity') else None,
            'pool_position_policy': getattr(cfg, 'pool_position_policy', None),
            'open_block_mode': getattr(cfg, 'open_block_mode', 'closed'),
            'open_block_scope': getattr(cfg, 'open_block_scope', 'full_prefix'),
            'selection_budget': getattr(cfg, 'selection_budget', 'shared_topk'),
            'sliding_window': int(getattr(cfg, 'sliding_window', 0) or 0),
            'index_topk': int(getattr(cfg, 'index_topk', 0) or 0),
            'overlap': int(getattr(cfg, 'overlap', 0) or 0),
        })
    blob = json.dumps(rows, sort_keys=True, separators=(',', ':'))
    return {'layers': rows,
            'all_closed_default': all(r['open_block_mode'] == 'closed'
                                      and r['max_block'] == 128
                                      and r['pool_position_capacity'] == 129
                                      for r in rows),
            'fingerprint': hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]}


def model_chunking_protocols(m):
    """Per-layer chunking DATA-path descriptor + one combined fingerprint.

    This is evidence about how each layer derived its cuts (similarity source,
    strict cut direction, geometry bounds), not a claim that every arm must
    have the same descriptor: dynamic-vs-fixed is itself a declared
    intervention.  It lets an audit compare arms that are supposed to differ
    only in policy and detect a preprocessing/signal confound.
    """
    rows = []
    for li, blk in enumerate(getattr(m, 'blocks', [])):
        cfg = getattr(getattr(blk, 'attn', None), 'cfg', None)
        if cfg is None or getattr(cfg, 'kind', 'full') == 'full':
            continue
        rows.append(dict({'layer': int(li)}, **chunking_protocol(cfg)))
    blob = json.dumps(rows, sort_keys=True, separators=(',', ':'))
    return {'layers': rows,
            'fingerprint': hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]}

def train_objective(m, *, comp_lambda=0.0, delta_lr_mult=1.0):
    """Describe the ACTUAL optimisation objective of an arm.

    Dense and sparse arms do not share an objective — sparse arms
    add `comp_lambda * comp_reg` and give `delta_logit` a different LR.  A
    comparison is only mechanism-clean when these descriptors match.  The
    values are read from the built model (not from the variant name), so a
    future arm cannot silently claim to be matched.
    """
    layers = [blk.attn for blk in getattr(m, 'blocks', [])]
    policies = sorted({str(getattr(a.cfg, 'boundary_policy', 'threshold')) for a in layers})
    # `need_reg` is True on every HybridAttention instance, including dense
    # layers (the flag is a capability, not an applied term).  The actual
    # regularizer exists only for cosine_learnable gates, whose trunk computes
    # `last_gate_mean`; deriving this from `need_reg` falsely labels dense and
    # fixed-CSA arms as having a compression objective.
    _reg_cfgs = [a.cfg for a in layers
                 if str(getattr(a.cfg, 'chunking', '')) == 'cosine_learnable']
    # `comp_lambda_mult` is a per-layer multiplier (HCA layers use 4x, and the
    # loss-term ablation zeroes it).  Counting the layer as regularised just
    # because its chunking is cosine_learnable mislabels `csa_dyn_nocomp` as
    # having the same objective as `csa_dynamic`.
    _mults = [float(getattr(c, 'comp_lambda_mult', 1.0)) for c in _reg_cfgs]
    _active = [m for m in _mults if float(comp_lambda) != 0.0 and m != 0.0]
    reg_layers = len(_active)
    _named = dict(m.named_parameters())
    delta_names = [n for n in _named if 'delta_logit' in n]
    delta_params = sum(int(_named[n].numel()) for n in delta_names)
    delta_trainable_params = sum(int(_named[n].numel()) for n in delta_names
                                 if bool(getattr(_named[n], 'requires_grad', False)))
    applies = bool(comp_lambda) and reg_layers > 0
    _weights = [float(comp_lambda) * m for m in _active]
    return {
        'comp_lambda': float(comp_lambda),
        'comp_reg_layers': int(reg_layers),
        'comp_reg_configured_layers': int(len(_reg_cfgs)),
        'comp_lambda_mults': sorted(set(_mults)),
        'comp_reg_applies': bool(applies),
        'comp_lambda_effective': (max(_weights) if _weights else 0.0),
        'delta_param_count': int(delta_params),
        'delta_trainable_param_count': int(delta_trainable_params),
        'delta_trainable': bool(delta_trainable_params),
        'delta_lr_mult': float(delta_lr_mult),
        'delta_lr_effective': (float(delta_lr_mult)
                               if delta_trainable_params else 1.0),
        'gate_modes': sorted({str(getattr(a.cfg, 'gate_mode', 'learned'))
                              for a in layers}),
        'boundary_policy': policies,
        'loss_terms': ['ce'] + (['comp_reg'] if applies else []) + (
            ['boundary_policy_gradient'] if any(p != 'threshold' for p in policies) else []),
        'note': 'mechanism-clean comparisons require comp_reg_applies, delta_lr_effective and boundary_policy to match across arms',
    }

def result_is_current(rec, code, *required_keys):
    if not isinstance(rec, dict) or not rec:
        return False
    if not all((k in rec for k in required_keys)):
        return False
    if 'ppl' in required_keys and (not ppl_is_usable(rec.get('ppl'))):
        return False
    return rec.get('_code') == code
PAIR_STEP_KEYS = ('steps', 'tokens_seen')

# A controlled comparison needs the FULL optimisation/schedule/augmentation
# descriptor, not merely two records that happen to carry the same partial
# dictionary.  Two legacy records that both omit the field remain the
# documented permissive path, but a present descriptor missing (say) the
# LR-schedule shape cannot prove that the schedule was shared and is refused.
TRAINING_KNOB_REQUIRED_FIELDS = (
    'optimizer', 'foreach', 'betas', 'eps', 'grad_clip_norm',
    'lr_schedule', 'lr_floor_mult', 'lr_cosine_amp', 'augmentation',
    'amp_dtype', 'amp_grad_scaler', 'eager_batch',
)

def _finite_knob(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Number):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return f if math.isfinite(f) else None

def training_knobs_problem(knobs, *, label='train_knobs'):
    """Why a present ``train_knobs`` descriptor cannot attest a controlled run.

    Returns ``None`` when the descriptor is complete and well typed, else a
    human-readable reason.  ``None``/absent descriptors are not checked here:
    callers decide whether legacy-vs-legacy is permitted.  A descriptor is a
    contract, so a partial one is a defect rather than "unverifiable".
    """
    if not isinstance(knobs, dict):
        return f'{label} is not an object ({type(knobs).__name__})'
    missing = [k for k in TRAINING_KNOB_REQUIRED_FIELDS if k not in knobs]
    if missing:
        return (f'{label} omits controlled field(s) {missing} — the descriptor '
                'does not cover the optimizer / schedule / augmentation recipe')
    if not isinstance(knobs.get('optimizer'), str) or not knobs['optimizer'].strip():
        return f'{label}.optimizer is missing/empty'
    if not isinstance(knobs.get('foreach'), (bool, np.bool_)):
        return f'{label}.foreach is not a boolean'
    betas = knobs.get('betas')
    if (not isinstance(betas, (list, tuple)) or len(betas) != 2
            or any(_finite_knob(b) is None for b in betas)):
        return f'{label}.betas is not a pair of finite numbers'
    if any(not (0.0 <= float(b) < 1.0) for b in betas):
        return f'{label}.betas values must lie in [0,1)'
    eps = _finite_knob(knobs.get('eps'))
    if eps is None or eps <= 0.0:
        return f'{label}.eps is not a finite positive number'
    clip = _finite_knob(knobs.get('grad_clip_norm'))
    if clip is None or clip < 0.0:
        return f'{label}.grad_clip_norm is not a finite non-negative number'
    if (not isinstance(knobs.get('lr_schedule'), str)
            or not knobs['lr_schedule'].strip()):
        return f'{label}.lr_schedule is missing/empty'
    for key in ('lr_floor_mult', 'lr_cosine_amp'):
        val = _finite_knob(knobs.get(key))
        if val is None or val < 0.0:
            return f'{label}.{key} is not a finite non-negative number'
    if (not isinstance(knobs.get('augmentation'), str)
            or not knobs['augmentation'].strip()):
        return f'{label}.augmentation is missing/empty'
    if knobs.get('amp_dtype') is not None and not isinstance(knobs.get('amp_dtype'), str):
        return f'{label}.amp_dtype is neither null nor a string'
    if not isinstance(knobs.get('amp_grad_scaler'), (bool, np.bool_)):
        return f'{label}.amp_grad_scaler is not a boolean'
    if not isinstance(knobs.get('eager_batch'), (bool, np.bool_)):
        return f'{label}.eager_batch is not a boolean'
    if 'param_groups' in knobs and knobs.get('param_groups') is not None:
        if (not isinstance(knobs['param_groups'], str)
                or not knobs['param_groups'].strip()):
            return f'{label}.param_groups is not a non-empty string'
    return None

def training_knobs_reason(ka, kb):
    """Controlled-variable gate for two records' training descriptors."""
    if (ka is None) != (kb is None):
        return ('unverifiable train_knobs (only one record carries the '
                'optimizer / lr-schedule / augmentation descriptor)')
    if ka is None:
        return None
    _problem_a = training_knobs_problem(ka, label='train_knobs (arm a)')
    if _problem_a is not None:
        return _problem_a
    _problem_b = training_knobs_problem(kb, label='train_knobs (arm b)')
    if _problem_b is not None:
        return _problem_b
    if ka != kb:
        return ('different train_knobs (optimizer / lr schedule / grad-clip / '
                'augmentation / precision)')
    return None

def pair_reason(ra, rb):
    if not isinstance(ra, dict) or not isinstance(rb, dict):
        return 'a record is not a dict'
    if ra.get('run_cfg') is None or rb.get('run_cfg') is None:
        return 'unverifiable run_cfg'
    if ra.get('run_cfg') != rb.get('run_cfg'):
        return 'different run_cfg'
    # Code semantics are a configuration component even when a legacy run_cfg
    # predates the `_cs` suffix.  A contrast across two semantic stamps would
    # silently mix preprocessing/kernel behaviour, so refuse it explicitly.
    _ca, _cb = ra.get('_code'), rb.get('_code')
    _ha = _ca is not None and str(_ca) != ''
    _hb = _cb is not None and str(_cb) != ''
    if _ha != _hb:
        return 'unverifiable _code (only one record carries the code-semantics stamp)'
    if _ha and str(_ca) != str(_cb):
        return f'different code semantics ({str(_ca)[:24]}… vs {str(_cb)[:24]}…)'
    # Protocol grouping: a warmup/early-stop/eval variant is its own protocol
    # and must never be paired with another one, even if its run_cfg happens
    # to match.  Both-absent is the untagged legacy path.
    _pa = ('protocol' in ra and ra.get('protocol') is not None) or ra.get('warm_steps') is not None
    _pb = ('protocol' in rb and rb.get('protocol') is not None) or rb.get('warm_steps') is not None
    if _pa != _pb:
        return 'unverifiable protocol (only one record carries a protocol tag)'
    if _pa and str(_tag_of(ra)) != str(_tag_of(rb)):
        return f'different protocol ({_tag_of(ra)!r} vs {_tag_of(rb)!r})'
    _env_reason = execution_environment_reason(ra, rb)
    if _env_reason is not None:
        return _env_reason
    # Controlled-variable gate for the settings the run fingerprint does not
    # carry (optimizer identity / betas / eps, LR-schedule shape, grad-clip,
    # augmentation).  BOTH records must carry the descriptor or BOTH must omit
    # it (legacy-vs-legacy); a one-sided descriptor is unverifiable and is
    # refused.
    _ka, _kb = (ra.get('train_knobs'), rb.get('train_knobs'))
    # Mixed presence is itself unverifiable: a record that carries the
    # descriptor and one that does not cannot be shown to share the optimizer /
    # schedule / clipping / augmentation settings, and pairing them is exactly
    # the uncontrolled comparison this gate exists to prevent.  Only
    # legacy-vs-legacy (both absent) keeps the permissive path.  A PRESENT but
    # partial descriptor is a defect, not a legacy record: equality of two
    # incomplete dictionaries cannot attest the controlled recipe.
    _knob_reason = training_knobs_reason(_ka, _kb)
    if _knob_reason is not None:
        return _knob_reason
    # Precision / implementation mode are also recorded at the top level on
    # newer records.  Compare them explicitly (the descriptor above already
    # covers records that carry it) so an fp32-vs-bf16 or row-loop-vs-batched
    # pair cannot be treated as controlled even if some future caller omits
    # the full descriptor.  Legacy records (neither key present) keep the
    # historical permissive path; a one-sided key is unverifiable.
    for _k in ('amp_dtype', 'amp_grad_scaler', 'eager_batch'):
        _pa, _pb = (_k in ra), (_k in rb)
        if _pa != _pb:
            return f'unverifiable {_k} (only one record carries the precision / implementation mode)'
        if _pa and ra.get(_k) != rb.get(_k):
            return f'different {_k} ({ra.get(_k)!r} vs {rb.get(_k)!r})'
    # Content identities: same run_cfg (steps / tokens / vocab / ...) does NOT
    # prove the same corpus or the same validation rows.  Scaling panels rebuild
    # caches at larger token caps, so pairing a new run with an old one on a
    # different val slice would silently compare two different evaluation sets.
    # Both-present equality is required; a one-sided fingerprint is
    # unverifiable, exactly like the optimizer-descriptor gate above.
    for _k in ('data_fp', 'val_fp', 'eval_fp', 'eval_subset_fp',
               'holdout_row_fp'):
        _pa, _pb = ra.get(_k), rb.get(_k)
        _ha = _pa is not None and str(_pa) != ''
        _hb = _pb is not None and str(_pb) != ''
        if _ha != _hb:
            return f'unverifiable {_k} (only one record carries the corpus / evaluation-slice fingerprint)'
        if _ha and str(_pa) != str(_pb):
            return f'different {_k} ({str(_pa)[:12]}… vs {str(_pb)[:12]}…)'
    # Matched initialization: same seed is not enough (architectures consume
    # different amounts of RNG), so compare the shared-trunk init digest.
    # Both-present equality is required; one-sided is unverifiable (legacy
    # records predate the field).  RNG-state equality is deliberately NOT
    # required: architecture-induced RNG consumption is part of the arm.
    _ia, _ib = ra.get('init_core_fp'), rb.get('init_core_fp')
    _ha = _ia is not None and str(_ia) != ''
    _hb = _ib is not None and str(_ib) != ''
    if _ha != _hb:
        return 'unverifiable init_core_fp (only one record carries the matched-initialization digest)'
    if _ha and str(_ia) != str(_ib):
        return f'different init_core_fp ({str(_ia)[:12]}… vs {str(_ib)[:12]}…): the two arms did not start from the same shared-trunk initialization despite the paired seed'
    bad = [k for k in PAIR_STEP_KEYS if ra.get(k) is None or rb.get(k) is None or ra.get(k) != rb.get(k)]
    if bad:
        absent = [k for k in bad if ra.get(k) is None and rb.get(k) is None]
        unknown = [k for k in bad if k not in absent and (ra.get(k) is None) != (rb.get(k) is None)]
        if unknown:
            return f'unverifiable {sorted(unknown)}'
        if absent:
            return f'unverifiable {sorted(absent)}'
        return f'disagreeing {sorted(bad)}'
    return None

def protocol_contrast_reason(ra, rb, declaration=None, *, factor=None):
    """Pair gate for a contrast whose ONLY permitted protocol difference is a
    declared protocol-level factor (currently ``warm_steps``).

    ``pair_reason`` deliberately refuses records from different protocol tags,
    because pooling a warmup/early-stop/eval variant with another protocol
    fabricates seeds.  A warmup panel is the exception: the warmup schedule IS
    its single factor, so comparing ``w=5000`` against ``w=0`` is the declared
    experiment, not pooling.  This gate removes only that declared axis and
    then runs the full ``pair_reason`` control check on the remaining record
    fields.  Anything else that differs still refuses.

    ``declaration`` must name ``protocol_factor='warm_steps'`` and the exact
    ``a_level``/``b_level`` (the same object used by the design audit), so an
    ad-hoc pair cannot opt into a different warmup schedule.
    """
    declaration = declaration or {}
    factor = factor if factor is not None else declaration.get('protocol_factor')
    if factor != 'warm_steps':
        return f'unsupported protocol contrast factor {factor!r}'
    if not isinstance(ra, dict) or not isinstance(rb, dict):
        return 'a record is not a dict'
    a_level = _as_seed_int(declaration.get('a_level'))
    b_level = _as_seed_int(declaration.get('b_level'))
    if a_level is None or b_level is None or a_level == b_level:
        return ('unverifiable protocol declaration (a_level/b_level must be '
                'distinct integer warmup cut points)')
    wa = _as_seed_int(ra.get('warm_steps'))
    wb = _as_seed_int(rb.get('warm_steps'))
    if wa is None or wb is None:
        return 'unverifiable warm_steps (only one record carries a warmup cut point)'
    if (wa, wb) != (a_level, b_level):
        return (f'recorded warm_steps ({wa}, {wb}) do not match the declared '
                f'protocol contrast ({a_level}, {b_level})')
    if ra.get('protocol') is not None or rb.get('protocol') is not None:
        return ('unverifiable protocol tag alongside warm_steps: the declared '
                'contrast is the warmup cut point, not another protocol axis')
    # Remove only the declared protocol axis; every other controlled field is
    # still checked by pair_reason (run_cfg, code stamp, knobs, content and
    # initialization fingerprints, steps/tokens, ...).
    ra_rest = {k: v for k, v in ra.items() if k != 'warm_steps'}
    rb_rest = {k: v for k, v in rb.items() if k != 'warm_steps'}
    return pair_reason(ra_rest, rb_rest)

def _hyperparam_reason(ra, rb):
    """Strict structured-hyperparameter gate for a module ablation pair.

    `run_cfg` pins the compact training recipe, but the structured
    `hyperparams` copy additionally carries `mlp_ratio` (and any future key
    that never made it into the string).  Two records with equal run_cfg can
    still differ in MLP width, which is a capacity change rather than the
    module under test.  Both records must carry the descriptor (legacy/legacy
    is the only permissive path) and every checked key must agree.

    Baseline-vs-method contrasts do NOT use this gate: their MLP width is
    deliberately widened for parameter matching, and `_param_control` is the
    authoritative check there."""
    ha, hb = ra.get('hyperparams'), rb.get('hyperparams')
    if ha is None and hb is None:
        return None
    if not isinstance(ha, dict) or not isinstance(hb, dict):
        return 'unverifiable hyperparams (one record carries no structured copy)'
    for k in _CONTROL_HYPER_KEYS:
        va, vb = ha.get(k), hb.get(k)
        if va is None and vb is None:
            continue
        if va is None or vb is None:
            return f'unverifiable hyperparams.{k}'
        if _canon_setting(va) != _canon_setting(vb):
            return f'different hyperparams.{k} ({va!r} vs {vb!r})'
    return None

_OBJECTIVE_COMPARE_KEYS = ('comp_reg_applies', 'comp_lambda', 'comp_lambda_effective',
                           'delta_lr_effective', 'boundary_policy', 'loss_terms')

# A component ablation may deliberately change an objective term.  For the
# loss-term and boundary-policy innovations the objective difference IS the
# intervention; treating it as an undeclared confound would make those
# declared components impossible to verify.  Every other objective difference
# remains a system-level confound.
OBJECTIVE_INTERVENTION_KEYS = {
    'compression_regularizer': frozenset({'comp_reg_applies', 'comp_lambda',
                                          'comp_lambda_effective',
                                          'loss_terms'}),
    'task_learned_boundary': frozenset({'boundary_policy', 'loss_terms'}),
    # Learned vs frozen boundary gate: the parameter, its initial value and
    # the comp_reg term are identical; only trainability (and therefore the
    # effective delta LR) changes.  `delta_lr_effective` is compared here;
    # the `requires_grad` state itself is verified directly by
    # `gate_control_audit`.
    'dynamic_blocking_learning': frozenset({'delta_trainable',
                                            'delta_lr_effective'}),
}

def _objective_diff(ra, rb):
    """Return (human reason or None, differing objective keys)."""
    oa, ob = ra.get('train_objective'), rb.get('train_objective')
    if oa is None and ob is None:
        return 'unverifiable train_objective (legacy records carry no objective descriptor)', []
    if (oa is None) != (ob is None):
        return 'unverifiable train_objective (only one record carries the objective descriptor)', []
    if not isinstance(oa, dict) or not isinstance(ob, dict):
        return 'unverifiable train_objective (malformed descriptor)', []
    # An empty/partial descriptor cannot attest equality: a `.get()` on a
    # missing key returns None on BOTH sides and would otherwise look like a
    # matched objective.  Old descriptors that only carry a subset are
    # unverifiable, not a pass.
    _miss_a = [k for k in _OBJECTIVE_COMPARE_KEYS if k not in oa]
    _miss_b = [k for k in _OBJECTIVE_COMPARE_KEYS if k not in ob]
    if _miss_a or _miss_b:
        _miss = sorted(set(_miss_a) | set(_miss_b))
        return ('unverifiable train_objective (descriptor misses ' + ', '.join(_miss) + ')', [])
    diffs = [k for k in _OBJECTIVE_COMPARE_KEYS if oa.get(k) != ob.get(k)]
    return (('objective differs: ' + ', '.join(diffs)) if diffs else None), diffs

def objective_reason(ra, rb):
    """Compare the ACTUAL loss/optimizer objective of two arms.

    `run_cfg` carries the compression weight `cl` and the delta-LR multiplier
    `dlm` for both arms, but dense arms do not apply `comp_reg` at all and have
    no `delta_logit`.  Same run_cfg therefore does NOT mean same objective;
    this helper makes the difference explicit instead of silently pairing
    `ce` with `ce + comp_lambda*comp_reg`.  It returns None when the records
    carry matching objective descriptors, a reason string otherwise, and
    'unverifiable ...' when the descriptors are absent/one-sided (legacy
    records keep the permissive path, but the state is reported)."""
    return _objective_diff(ra, rb)[0]

def ppl_is_usable(v):
    # PPL = exp(mean CE) with CE >= 0, so a valid value is >= 1.  A sub-1 value
    # is not a surprisingly strong model; it is an impossible/corrupt
    # measurement (negative CE, a raw CE accidentally stored as PPL, or a
    # rounded float underflow) and must not enter aggregates.
    f = _finite_scalar(v)
    return f is not None and f >= 1.0

def by_len_cells(rows, Ln, variant=None):
    keys = (Ln, int(Ln), str(int(Ln)))
    # Length-panel rows must all come from one validation slice: PPL@L is only
    # comparable when the underlying corpus is the same.  Records without a
    # fingerprint keep the historical permissive path; a mixed fingerprint set
    # raises instead of silently averaging two slices.
    assert_shared_validation_fp([r for r in rows if isinstance(r, dict)],
                                context=f'by_len_cells(L={Ln})')
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if variant is not None and r.get('variant') != variant:
            continue
        bl = r.get('by_len')
        if not isinstance(bl, dict):
            continue
        cell = None
        for k in keys:
            if k in bl:
                cell = bl[k]
                break
        if not isinstance(cell, dict):
            continue
        if ppl_is_usable(cell.get('ppl')):
            out.append(cell)
    return out

def _validated_alpha(alpha, *, context='alpha'):
    """A probability threshold must be a real number strictly inside (0, 1).

    Booleans are flags, not thresholds; ``alpha=1.5`` would make every p-value
    "significant" and ``alpha=0`` would make none reachable.  The claim and
    multiplicity helpers all share this check so an invalid threshold cannot
    silently change the wording of a result.
    """
    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError(f'{context} must be a real number in (0,1), got {alpha!r}')
    try:
        a = float(alpha)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{context} must be a real number in (0,1), got {alpha!r}') from None
    if not (0.0 < a < 1.0):
        raise ValueError(f'{context} must be in (0,1), got {alpha!r}')
    return a

def holm_adjust(named_pvals, alpha=0.05):
    """Holm-Bonferroni step-down adjustment over a family of comparisons.

    输入 [(name, p), ...]（NaN/越界的 p 记为无法校正并原样透传，不进入排序，
    但仍占用声明族的一个槽位——等价于按 p=1 处理，绝不因缺失 p 而缩小
    校正族），输出 {name: {'p_raw', 'p_holm', 'rank', 'reject', 'm_effective'}}。
    family 的定义必须由调用方预先声明（预注册的族用冻结清单，探索性的族
    明示「未预注册」），本函数只做数学，不做族的选择——族的选择正是
    多重比较问题的来源（design_notes §4/§17）。"""
    alpha = _validated_alpha(alpha)
    fam = []
    invalid = {}
    _seen_names = set()
    for name, p in (named_pvals if named_pvals is not None else []):
        sname = str(name)
        if sname in _seen_names:
            # A dict keyed by comparison name cannot represent two family
            # slots with the same name; silently keeping one would shrink or
            # corrupt the correction family.
            raise ValueError(f'holm_adjust: duplicate comparison name {sname!r}')
        _seen_names.add(sname)
        try:
            # A boolean is a flag, not a p-value: float(False) == 0 would
            # otherwise enter the family as a maximally significant result.
            pv = float('nan') if isinstance(p, (bool, np.bool_)) else float(p)
        except (TypeError, ValueError, OverflowError):
            pv = float('nan')
        if (not math.isfinite(pv)) or pv < 0.0 or pv > 1.0:
            invalid[sname] = {'p_raw': p, 'p_holm': None, 'rank': None, 'reject': False, 'reason': 'p not in [0,1] or not finite'}
            continue
        fam.append((sname, pv))
    out = dict(invalid)
    # Invalid p-values cannot be ranked, but they still occupy a slot in the
    # DECLARED family.  Treating them as p=1 (the same fail-closed convention
    # as `attach_holm`) keeps the multipliers conservative instead of silently
    # shrinking m and over-claiming the remaining comparisons.
    m = len(fam) + len(invalid)
    order = sorted(fam, key=lambda kv: kv[1])
    prev = 0.0
    for i, (name, p) in enumerate(order):
        adj = min(1.0, (m - i) * p)
        prev = max(prev, adj)
        out[name] = {'p_raw': p, 'p_holm': float(prev), 'rank': i + 1, 'reject': bool(prev <= alpha), 'm_effective': m}
    for name, rec in invalid.items():
        rec['m_effective'] = m
    return out

def attach_holm(comparisons, *, alpha=0.05, family='reported comparisons', exploratory=True):
    """Attach `p_holm` to every non-omitted comparison in a dict and return a
    family descriptor.

    A report that prints many raw p-values without a correction invites
    multiple-comparison over-reading.  The family is whatever the caller
    displays (not silently shrunk by omitted rows: those keep a dummy p=1).
    `exploratory=True` is a hard label, not a decoration: Holm limits
    over-claiming but cannot turn validation-selected v7-v11 panels into
    pre-registered confirmation (design_notes §17).
    """
    alpha = _validated_alpha(alpha)
    pairs, omitted, tested = [], [], 0
    _tested_ns = []

    def _resolution_n(v):
        # Exact ties do not contribute resolution.  When the comparison
        # records its non-zero pair count, the family ceiling is m*2/2**nn,
        # not the theoretical all-nonzero m*2/2**n.
        _n = _as_seed_int(v.get('n'))
        if _n is None or _n < 0:
            return None
        _raw_nn = v.get('n_nonzero')
        if _raw_nn is None:
            return _n
        _nn = _as_seed_int(_raw_nn)
        if _nn is None or _nn < 0 or _nn > _n:
            # A malformed tie count cannot buy the family a finer resolution.
            # Use the coarsest non-zero count (one) so the reported family floor
            # is conservative rather than silently reverting to all-nonzero n.
            return 1
        return _nn if _nn > 0 else 1

    for k, v in (comparisons or {}).items():
        if not isinstance(v, dict):
            pairs.append((str(k), 1.0))
            omitted.append(str(k))
            continue
        if 'omitted' in v:
            omitted.append(str(k))
            # A declared-but-omitted comparison keeps a dummy p=1 slot: the
            # family size must be the DECLARED family, not the subset that
            # happened to be measurable (same rule as methodology_audit).
            pairs.append((str(k), 1.0))
            continue
        p = v.get('p_exact_signflip')
        if p is None:
            p = v.get('p')
        try:
            p = (float('nan') if isinstance(p, (bool, np.bool_)) else float(p))
        except (TypeError, ValueError, OverflowError):
            p = float('nan')
        if math.isfinite(p) and 0.0 <= p <= 1.0:
            pairs.append((str(k), p))
            tested += 1
            _tn = _resolution_n(v)
            if _tn is not None and _tn > 0:
                _tested_ns.append(_tn)
        else:
            pairs.append((str(k), 1.0))
    h = holm_adjust(pairs, alpha=alpha) if pairs else {}
    for k, v in (comparisons or {}).items():
        if isinstance(v, dict):
            v['p_holm'] = (h.get(str(k)) or {}).get('p_holm')
    # Family-level resolution ceiling: with m declared slots and n seeds the
    # smallest attainable Holm p is m * 2 / 2^n.  The best chance any member
    # has is at the LARGEST n in the family (the first Holm gate only needs one
    # comparison), so using min(n) would falsely declare a mixed-n family
    # impossible even when its best-powered member can pass.
    _ns = []
    for v in (comparisons or {}).values():
        if not isinstance(v, dict):
            continue
        _n = _resolution_n(v)
        if _n is not None and _n > 0:
            _ns.append(_n)
    _m = int(len(comparisons or {}))
    # Only a tested, measurable comparison can pass the first Holm gate; an
    # omitted or invalid row must not make the family look better powered.
    # If there are tested rows but none carry a usable n, no family floor can
    # be certified; falling back to all rows would let an omitted high-n row
    # supply the resolution instead.
    _floor_ns = _tested_ns if tested else _ns
    _fam_floor = holm_min_attainable_p(max(_floor_ns), _m, alpha) if _floor_ns else None
    _fam_limited = bool(_fam_floor is not None and math.isfinite(_fam_floor) and _fam_floor > alpha)
    if _fam_limited:
        print(f'[stats] WARNING: family {family!r} has m={_m} declared comparisons and n_max={max(_floor_ns)} seeds — '
              f'the smallest attainable Holm p is {_fam_floor:.4g} > alpha={alpha}; NO comparison in this family can be significant. '
              'That is a resolution ceiling, not a null result; report it as such and add seeds before interpreting p_holm=1.0')
    return {
        'name': family,
        'declared_slots': _m,
        'tested': int(tested),
        'omitted': omitted,
        'alpha': float(alpha),
        'correction': 'holm step-down',
        'evidence_level': 'exploratory' if exploratory else 'confirmatory',
        'n_effective_min': (min(_ns) if _ns else None),
        'n_effective_max': (max(_ns) if _ns else None),
        'family_min_attainable_p_holm': _fam_floor,
        'family_resolution_limited': _fam_limited,
        'note': ('Holm over the displayed family; the panels were selected on the validation split, '
                 'so corrected p-values still are not confirmation') if exploratory else 'pre-registered family',
    }

# Minimum number of evaluation documents below which a claim about the
# language population (rather than about the frozen evaluation slice) is
# refused.  The confirmatory panel scores n_seqs=16; that is enough for a
# fixed-slice mechanism claim and not enough for a population claim.
MIN_DOCS_FOR_POPULATION_CLAIM = 64

def reportable_claim(stats, *, family_size=None, alpha=0.05, min_n=6,
                     require_holm=True, evidence_level=None):
    """One machine-checkable answer to: may this contrast be called significant?

    This is the shared gate behind report prose.  A contrast is reportable only
    when ALL of the following hold:

      * n >= min_n            (otherwise the exact sign-flip p floor is > alpha)
      * p_floor <= alpha      (resolution ceiling)
      * p_raw < alpha
      * p_holm < alpha        (when `require_holm`; an uncorrected p is exploratory)
      * |mean| > std          (effect larger than seed noise)
      * family_size is None or the family's minimum attainable Holm p <= alpha

    `evidence_level='confirmatory'` is only honoured when the caller passes it
    explicitly (a pre-registered family); otherwise a passing contrast is
    labelled `exploratory_significant`.  The function never turns a failure
    into a prohibition on *reporting* the number -- it only forbids the words
    "significant"/"confirmed" and records exactly why.
    """
    alpha = _validated_alpha(alpha)
    # min_n and family_size are gate parameters, not measurements: silently
    # truncating 5.9 to 5 (or accepting True as 1) would change which claims
    # the gate allows.  Treat an invalid value as unverifiable and refuse the
    # wording instead of coercing it.
    _min_n = _as_seed_int(min_n)
    out = {'allowed': False, 'level': 'unverifiable', 'reasons': [],
           'n': None, 'p_raw': None, 'p_holm': None, 'p_floor': None,
           'min_attainable_p_holm_family': None, 'min_n': _min_n,
           'alpha': alpha, 'prediction_interval_95': None,
           'prob_new_seed_same_sign': None}
    if _min_n is None or _min_n < 1:
        out['reasons'].append(
            f'min_n={min_n!r} is not a positive integer; the n floor is unverifiable')
    if not isinstance(stats, dict):
        out['reasons'].append('not a statistics dict')
        return out
    n = _as_seed_int(stats.get('n'))
    if n is None or n < 0:
        n = 0
    p = stats.get('p_exact_signflip')
    if p is None:
        p = stats.get('p')
    ph = stats.get('p_holm')
    mean, std = stats.get('mean'), stats.get('std')
    out['n'] = n
    try:
        out['p_raw'] = (None if p is None or isinstance(p, (bool, np.bool_))
                        else float(p))
    except (TypeError, ValueError, OverflowError):
        out['p_raw'] = None
    try:
        out['p_holm'] = (None if ph is None or isinstance(ph, (bool, np.bool_))
                         else float(ph))
    except (TypeError, ValueError, OverflowError):
        out['p_holm'] = None
    # Exact ties do not contribute resolution.  A caller that supplies the
    # non-zero count (paired_effect_stats/exact_signflip_p do) must not be
    # re-scored with the theoretical all-nonzero floor 2/2**n.
    _raw_nn = stats.get('n_nonzero')
    _nn = _as_seed_int(_raw_nn)
    _resolution_n = n
    if _raw_nn is not None and _nn is None:
        # A supplied tie count that is not an exact integer is corrupt.  Treat
        # it as an unverifiable resolution claim instead of silently reverting
        # to the (possibly finer) all-nonzero floor.
        out['n_nonzero'] = None
        out['p_floor'] = signflip_p_floor(n) if n > 0 else None
        out['reasons'].append(
            f'n_nonzero={_raw_nn!r} is not a non-negative integer; the exact-test '
            'resolution is unverifiable')
    elif _nn is not None and _nn >= 0:
        out['n_nonzero'] = _nn
        if _nn > n:
            # A non-zero count larger than the paired panel cannot describe
            # this test; it must not be used to claim a finer resolution.
            out['reasons'].append(
                f'n_nonzero={_nn} exceeds n={n}; the effective resolution is unverifiable')
        else:
            _resolution_n = _nn
        out['p_floor'] = (1.0 if _resolution_n == 0
                          else signflip_p_floor(_resolution_n))
    else:
        out['n_nonzero'] = _nn
        if _nn is not None:
            out['reasons'].append(
                f'n_nonzero={_nn} is negative; the exact-test resolution is '
                'unverifiable')
        out['p_floor'] = signflip_p_floor(n) if n > 0 else None
    out['n_effective_resolution'] = _resolution_n
    # A p-value is a probability.  p=-0.1 would satisfy every strict "< alpha"
    # comparison and could support a claim that no valid test produced.
    if out['p_raw'] is not None and not (0.0 <= out['p_raw'] <= 1.0):
        out['reasons'].append(f'raw p {out["p_raw"]!r} is outside [0,1]')
    if out['p_holm'] is not None and not (0.0 <= out['p_holm'] <= 1.0):
        out['reasons'].append(f'Holm p {out["p_holm"]!r} is outside [0,1]')
    # Complete-case significance is not the same as a complete paired panel.
    # If the caller's statistics record non-finite/missing pairs, the reported
    # n is the number that SURVIVED filtering; a claim over that subset can be
    # biased by which seeds/arms failed.  Keep the number reportable but refuse
    # the significance wording unless the caller supplies the complete panel.
    _raw_n_input = stats.get('n_input')
    _raw_n_dropped = stats.get('n_dropped')
    _n_input = _as_seed_int(_raw_n_input)
    _n_dropped = _as_seed_int(_raw_n_dropped)
    out['n_input'] = _n_input
    out['n_dropped'] = _n_dropped
    # A supplied accounting field that cannot be read as an exact integer is
    # corrupt, not absent.  Falling back to "no attrition" would let a
    # malformed n_dropped=0.5/True/'x' support a complete-panel claim.
    if _raw_n_input is not None and _n_input is None:
        out['reasons'].append(
            f'n_input={_raw_n_input!r} is not a non-negative integer; the paired '
            'panel size is unverifiable')
    elif _n_input is not None and _n_input < 0:
        out['reasons'].append(f'n_input={_n_input} is negative; the paired panel size is unverifiable')
    elif _n_input is not None and n > _n_input:
        out['reasons'].append(
            f'n={n} exceeds n_input={_n_input}; the paired panel accounting is inconsistent')
    if _raw_n_dropped is not None and _n_dropped is None:
        out['reasons'].append(
            f'n_dropped={_raw_n_dropped!r} is not a non-negative integer; the '
            'paired panel accounting is unverifiable')
    elif _n_dropped is not None and _n_dropped < 0:
        out['reasons'].append(f'n_dropped={_n_dropped} is negative; the paired panel size is unverifiable')
    elif (_n_dropped is not None and _n_input is not None
          and _n_dropped > _n_input):
        out['reasons'].append(
            f'n_dropped={_n_dropped} exceeds n_input={_n_input}; the paired '
            'panel accounting is inconsistent')
    elif _n_dropped is not None and _n_dropped > 0:
        out['reasons'].append(
            f'{_n_dropped} paired observation(s) were dropped as non-finite '
            f'(n_input={_n_input}); complete-case significance is not a '
            'complete-panel result')
    # The three fields describe one panel: n survivors + dropped inputs must
    # reconstruct n_input exactly.  A record that reports a smaller n with
    # n_dropped=0 has silently lost observations, and complete-case wording
    # over the survivors would be selection-biased.
    if _n_input is not None and _n_dropped is not None:
        if n != _n_input - _n_dropped:
            out['reasons'].append(
                f'n={n} != n_input={_n_input} - n_dropped={_n_dropped}; the '
                'paired panel accounting is inconsistent')
    elif _n_input is not None and _n_dropped is None and n != _n_input:
        out['reasons'].append(
            f'n={n} != n_input={_n_input} and no n_dropped was recorded; the '
            'paired panel accounting is incomplete')
    # Seeds excluded because the two arms failed the same-recipe controlled
    # comparison are ALSO an incomplete paired panel.  Keeping them out of the
    # test is correct, but the surviving subset was selected by that mismatch
    # and cannot support a significance claim.  Both field spellings are used
    # by the v9/v10/v11 contrast builders.
    _skipped = stats.get('n_skipped_config_mismatch')
    if _skipped is None:
        _skipped = stats.get('n_skipped')
    if _skipped is not None:
        _sk = _as_seed_int(_skipped)
        out['n_skipped_config_mismatch'] = _sk
        if _sk is None or _sk < 0:
            out['reasons'].append(
                f'n_skipped_config_mismatch={_skipped!r} is not a non-negative '
                'integer; paired-panel accounting is unverifiable')
        elif _sk > 0:
            out['reasons'].append(
                f'{_sk} seed(s) were excluded by the controlled-comparison '
                'gate; complete-case significance is not a complete paired '
                'panel result')
    if _min_n is not None and _min_n >= 1 and n < _min_n:
        out['reasons'].append(f'n={n} < min_n={_min_n}: the exact-test p floor is {out["p_floor"]:.3g} > alpha' if out['p_floor'] is not None else f'n={n} < min_n={_min_n}')
    elif out['p_floor'] is not None and math.isfinite(out['p_floor']) and out['p_floor'] > alpha:
        out['reasons'].append(f'p floor {out["p_floor"]:.3g} > alpha={alpha}: resolution ceiling, not a null result')
    try:
        if out['p_raw'] is None or not (out['p_raw'] < alpha):
            out['reasons'].append(f'raw p {out["p_raw"]!r} is not < alpha={alpha}')
    except TypeError:
        out['reasons'].append('raw p is not comparable')
    if require_holm:
        if out['p_holm'] is None:
            out['reasons'].append('no Holm-corrected p (family correction not attached)')
        elif not (out['p_holm'] < alpha):
            out['reasons'].append(f'Holm p {out["p_holm"]:.3g} is not < alpha={alpha}')
        if (out['p_holm'] is not None and out['p_raw'] is not None
                and math.isfinite(out['p_holm']) and math.isfinite(out['p_raw'])
                and out['p_holm'] < out['p_raw'] - 1e-12):
            out['reasons'].append(
                'Holm p is smaller than the raw p; a step-down correction '
                'cannot increase significance (mis-attached or fabricated '
                'family correction)')
    _mean, _std = _finite_scalar(mean), _finite_scalar(std)
    if _mean is None or _std is None:
        # Booleans and non-finite values are flags/corruption, not effect
        # measurements (same scalar contract as the p-value parsing above).
        out['reasons'].append('mean/std are not finite real measurements')
    elif abs(_mean) <= _std:
        out['reasons'].append('|mean| <= std: effect is within seed noise')
    # Leave-one-seed-out instability is a direct contradiction of a stable
    # directional effect: dropping one seed must not flip the sign.  Only
    # enforced when the caller supplied the diagnostic (old stats dicts do not
    # carry it); the verdict is never silently assumed robust.
    _seed_driven = stats.get('seed_driven')
    if isinstance(_seed_driven, (bool, np.bool_)) and bool(_seed_driven):
        out['reasons'].append('leave-one-seed-out stability fails: dropping one seed flips the mean sign')
    if family_size is not None:
        _fam_n = _as_seed_int(family_size)
        if _fam_n is None or _fam_n < 1:
            out['reasons'].append(
                f'family_size={family_size!r} is not a positive integer; '
                'the family resolution ceiling is unverifiable')
        else:
            _fl = holm_min_attainable_p(_resolution_n, _fam_n, alpha)
            out['min_attainable_p_holm_family'] = _fl
            if _fl is not None and math.isfinite(_fl) and _fl > alpha:
                out['reasons'].append(f'family of m={_fam_n} at n={n} has minimum attainable Holm p {_fl:.4g} > alpha')
    # Echo the prospective replication readout so a report can print the
    # predictive interval beside the significance verdict.  It is advisory:
    # a low sign probability never rewrites a p-value or the frozen verdict,
    # and a malformed field is dropped rather than converted into a gate.
    _pi = stats.get('prediction_interval_95')
    if isinstance(_pi, (list, tuple)) and len(_pi) == 2:
        _lo_pi, _hi_pi = _finite_scalar(_pi[0]), _finite_scalar(_pi[1])
        if (_lo_pi is not None and _hi_pi is not None and _lo_pi <= _hi_pi):
            out['prediction_interval_95'] = [float(_lo_pi), float(_hi_pi)]
    _ps = _finite_scalar(stats.get('prob_new_seed_same_sign'))
    if _ps is not None and 0.0 <= _ps <= 1.0:
        out['prob_new_seed_same_sign'] = float(_ps)
    out['allowed'] = not out['reasons']
    if out['allowed']:
        out['level'] = 'confirmatory' if str(evidence_level).lower() == 'confirmatory' else 'exploratory_significant'
    elif _min_n is not None and _min_n >= 1 and n < _min_n:
        # Underpowered is the actionable label; the resolution ceiling is the
        # reason (both are recorded in `reasons`).
        out['level'] = 'underpowered'
    elif out['p_floor'] is not None and math.isfinite(out['p_floor']) and out['p_floor'] > alpha:
        out['level'] = 'resolution_limited'
    else:
        out['level'] = 'not_significant'
    return out

def signflip_p_floor(n):
    """双侧精确符号翻转检验在 n 个种子下的最小可达 p 值（2/2^n）；
    n<=0 返回 NaN。用于报告「未显著是分辨率上限还是数据结论」。"""
    n = _as_seed_int(n)
    if n is None or n <= 0:
        return float('nan')
    # same value as 2.0 / 2**n, evaluated in the float exponent range.
    return math.ldexp(2.0, -n)

def finite_range(named_values):
    vals, dropped = ([], [])
    for label, m in (named_values if named_values is not None else []):
        f = _finite_scalar(m)
        if f is not None:
            vals.append(f)
        else:
            dropped.append(label)
    if not vals:
        return (None, None, dropped)
    return (min(vals), max(vals), dropped)

_SIGNFLIP_ENUM_MAX_N = 20
_SIGNFLIP_MITM_MAX_N = 40


def _signflip_signed_sums(values):
    """All 2**k signed sums of `values` (deterministic doubling table)."""
    out = np.zeros(1, dtype=float)
    for v in np.asarray(values, dtype=float):
        out = np.concatenate((out + v, out - v))
    return out


def _signflip_mitm_count(d):
    """Exact count of sign assignments with |sum(s_i*d_i)| >= |sum(d_i)|
    using meet-in-the-middle: 2**(n/2) sums per half instead of 2**n.

    For each left-half sum `a`, the admissible right-half sums are
    `b <= -T - a` or `b >= T - a`; two binary searches count them.  This keeps
    the test exact (no normal approximation) for the 21-40 seed range a
    scale-up panel can reach, while the 2**20 worst-case table stays small."""
    n = int(d.size)
    total = abs(float(d.sum()))
    # The n<=20 enumeration accepts |mean| >= obs - 1e-12, i.e. |sum| >=
    # total - n*1e-12.  If that threshold is non-positive every sign
    # assignment qualifies; the two-tail formula below would double-count.
    if total - 1e-12 * n <= 0.0:
        return 1 << n
    half = n // 2
    left = _signflip_signed_sums(d[:half])
    right = _signflip_signed_sums(d[half:])
    right.sort()
    # Match the n<=20 enumeration's tolerance: mean comparison with 1e-12.
    thr = total - 1e-12 * n
    lo = np.searchsorted(right, -thr - left, side='right')
    hi = np.searchsorted(right, thr - left, side='left')
    return int(lo.sum() + (right.size - hi).sum())


def _signflip_saddlepoint_p(values):
    """Deterministic Lugannani-Rice saddlepoint tail for n > 40.

    Under H0 each paired difference d_i gets an independent random sign, so
    the null distribution of ``S = sum(s_i * d_i)`` has cumulant generating
    function ``K(t) = sum(log cosh(t*d_i))``.  The exact variance of S is
    ``sum(d_i**2)`` (not ``n * sample_var``, which is centered on the observed
    mean and therefore shrinks as the effect grows).  Solving ``K'(t) = |S|``
    and applying the Lugannani-Rice formula gives a much more accurate tail
    than a normal approximation while staying deterministic.

    Returns ``(p, method)``.  Zeros are removed because they multiply the
    number of sign assignments without changing the null distribution; the
    resulting floor ``2 / 2**m`` is the same exact-test floor.
    """
    vals = [float(x) for x in values if float(x) != 0.0]
    m = len(vals)
    if m == 0:
        return 1.0, 'saddlepoint'
    amax = max(abs(x) for x in vals)
    if amax > 1e100 or amax < 1e-100:
        vals = [x / amax for x in vals]
    T = abs(math.fsum(vals))
    sabs = math.fsum(abs(x) for x in vals)
    floor = min(1.0, math.ldexp(2.0, -m))
    if T <= 0.0:
        return 1.0, 'saddlepoint'
    # |S| can only equal the maximum sum when every sign is the same.
    if T >= sabs * (1.0 - 1e-13):
        return floor, 'saddlepoint'
    inv = 1.0 / max(abs(x) for x in vals)

    def _kp(t):
        return math.fsum(x * math.tanh(t * x) for x in vals)

    def _k(t):
        # log cosh(z) = |z| + log1p(exp(-2|z|)) - log(2), stable for large |z|.
        return math.fsum(abs(t * x) + math.log1p(math.exp(-2.0 * abs(t * x))) - math.log(2.0)
                         for x in vals)

    def _kpp(t):
        # sech^2(z) = 1 - tanh^2(z) avoids a cosh overflow in the extreme
        # near-maximum-sum tail while staying algebraically identical.
        return math.fsum(x * x * (1.0 - math.tanh(t * x) ** 2) for x in vals)

    hi = inv
    for _ in range(60):
        if _kp(hi) >= T:
            break
        hi *= 2.0
    else:
        # Pathological near-maximum sum: the normal tail with the CORRECT
        # uncentered variance is the safe deterministic fallback.
        sd = math.sqrt(math.fsum(x * x for x in vals))
        p = math.erfc(T / (sd * math.sqrt(2.0))) if sd > 0 else floor
        return float(min(1.0, max(floor, p))), 'saddlepoint-normal-fallback'
    lo = 0.0
    for _ in range(160):
        mid = 0.5 * (lo + hi)
        if _kp(mid) < T:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    Kt = _k(t)
    K2 = _kpp(t)
    w = math.sqrt(max(0.0, 2.0 * (t * T - Kt)))
    u = t * math.sqrt(K2) if K2 > 0.0 else 0.0
    if w < 1e-8 or u <= 0.0:
        sd = math.sqrt(math.fsum(x * x for x in vals))
        p = math.erfc(T / (sd * math.sqrt(2.0))) if sd > 0 else floor
    else:
        phi = math.exp(-0.5 * w * w) / math.sqrt(2.0 * math.pi)
        tail = 0.5 * math.erfc(w / math.sqrt(2.0)) + phi * (1.0 / u - 1.0 / w)
        p = 2.0 * tail
    # A sign-flip p-value cannot be smaller than the two all-same-sign
    # assignments (the same floor the exact branch obeys).
    return float(min(1.0, max(floor, p))), 'saddlepoint'


def exact_signflip_p(deltas):
    """Two-sided exact sign-flip test over paired per-seed deltas.

    H0: the per-seed difference is symmetric about 0.  For n <= 20 every one
    of the 2**n sign assignments is enumerated; for 20 < n <= 40 the same
    count is obtained exactly by meet-in-the-middle (2**(n/2) tables), which
    matters as seed-completion panels grow.  For n > 40 the deterministic
    Lugannani-Rice saddlepoint of the sign-flip null is used (reported via
    `method`); it uses the null variance ``sum(d_i**2)`` and is therefore
    neither a Monte-Carlo draw nor the anti-conservative centered-SD normal
    approximation.  Non-finite deltas are dropped (the count is returned).
    Pure function over its input — used by the aggregate audit so the core
    panel reports the same p the per-phase suites do, rather than a bare
    mean±std that could be mistaken for evidence.

    有效性假设（必须在预注册文档同步声明）：精确符号翻转
    检验的 p 值正确性依赖「配对内交换性」——每个种子下的两臂共享除臂本身
    之外的一切随机因素（同数据顺序、同评测行、同腐坏网格；初始化 RNG 按
    种子播种，臂间参数量差异对 RNG 流的扰动属于「臂」的一部分）。种子编号
    相同本身不提供该保证；design_notes §20 记录了本仓库逐因素的控制方式
    与残余风险（训练随机性进入种间方差、文档抽样进入 doc 层方差）。"""
    d_in = []
    for _x in (deltas if deltas is not None else []):
        if isinstance(_x, (bool, np.bool_)):
            # Booleans are flags, not measurements (same scalar contract as
            # _finite_scalar and build_report.exact_signflip).
            d_in.append(float('nan'))
            continue
        try:
            d_in.append(float(_x))
        except (TypeError, ValueError, OverflowError):
            d_in.append(float('nan'))
    d = np.asarray([x for x in d_in if math.isfinite(x)], dtype=float)
    n = int(d.size)
    n_dropped = len(d_in) - n
    if n == 0:
        return {'n': 0, 'p': float('nan'), 'n_dropped': n_dropped,
                'n_nonzero': 0, 'p_floor': float('nan'), 'method': 'none'}
    # Exact zero deltas do not change the sign-flip statistic: their signs are
    # arbitrary, so each attainable count is multiplied by 2**(#zeros).  The
    # p-value is unchanged, but the *resolution floor* is 2 / 2**(#nonzero),
    # not 2 / 2**n.  Reporting the latter would claim a panel can resolve more
    # than its non-zero paired differences actually can.
    n_nonzero = int(np.count_nonzero(d))
    p_floor = (1.0 if n_nonzero == 0
               else min(1.0, math.ldexp(2.0, -n_nonzero)))
    _scale = float(np.max(np.abs(d))) if n_nonzero else 0.0
    # Work on a unit-scaled copy unconditionally.  The sign-flip statistic is
    # scale invariant, but the exact/MITM branches compare against an
    # absolute 1e-12 tie tolerance.  Leaving a small-but-representable panel
    # (e.g. deltas of 1e-13) unscaled makes that tolerance larger than the
    # observed effect, so every sign assignment qualifies and p becomes 1.
    # Scaling by the largest magnitude turns the tolerance into a relative
    # one without changing the comparison for ordinary O(1) panels.
    d_work = (d / _scale) if _scale > 0.0 else d
    obs = abs(float(d_work.mean()))
    if n <= _SIGNFLIP_ENUM_MAX_N:
        total = 1 << n
        cnt = 0
        bitpos = np.arange(n, dtype=np.int64)[None, :]
        step = max(1, min(total, (1 << 22) // max(n, 1)))
        for off in range(0, total, step):
            bs = min(step, total - off)
            masks = np.arange(off, off + bs, dtype=np.int64)[:, None]
            signs = np.where((masks & (1 << bitpos)) != 0, 1.0, -1.0)
            cnt += int((np.abs((signs * d_work[None, :]).mean(1))
                    >= obs - 1e-12).sum())
        return {'n': n, 'p': float(cnt / total), 'n_dropped': n_dropped,
                'n_nonzero': n_nonzero, 'p_floor': p_floor, 'method': 'exact'}
    if n <= _SIGNFLIP_MITM_MAX_N:
        cnt = _signflip_mitm_count(d_work)
        return {'n': n, 'p': float(cnt / float(1 << n)), 'n_dropped': n_dropped,
                'n_nonzero': n_nonzero, 'p_floor': p_floor, 'method': 'exact-mitm'}
    p, method = _signflip_saddlepoint_p(d_work.tolist())
    return {'n': n, 'p': float(p), 'n_dropped': n_dropped,
            'n_nonzero': n_nonzero, 'p_floor': p_floor, 'method': method}

# ---- 序列观测（look）修正：补种子不是免费的 --------------------------------
# 实验面板在补种子（n=3 -> 4 -> 6 ...）时会多次聚合/查看同一终点。按最终 n
# 只做一次检验会把「看了几次再决定加种子」的选择效应漏掉，这是典型的
# optional stopping。`sequential_signflip_p` 在**同一组逐种子配对差**上，
# 对每个 look 取「最终效应方向」的证据并取 max，再在零假设（每个种子的符号
# 独立等概率）下联合枚举所有 look 的符号指派，给出族级（跨 look）精确 p。
# 只做 post-hoc 稳健性修正，不改冻结确认清单里的 p/α/决策。
LOOK_SEQUENTIAL_ENUM_MAX_N = 16


def _normalise_look_series(looks, *, where='sequential_signflip_p'):
    """Normalise ``[{seeds, deltas}, ...]`` or ``[(seeds, deltas), ...]``.

    Each look is sorted by seed so the joint enumeration below has a stable,
    auditable column order.  Non-finite deltas are dropped (and counted), a
    duplicate seed inside one look is a hard error (it would make the sign
    assignment ambiguous), and an empty look is refused rather than silently
    skipped.
    """
    if isinstance(looks, (str, bytes, dict)):
        raise ValueError(f'{where}: looks must be an iterable of '
                         '(seeds, deltas) pairs or {seeds, deltas} mappings')
    # Never use ``x or []`` on a caller-supplied sequence: a multi-element
    # NumPy array / torch tensor raises "truth value ... is ambiguous", so a
    # perfectly valid array input would crash before the contract check.  Only
    # None means "empty" here.
    if looks is None:
        items = []
    else:
        try:
            items = list(looks)
        except TypeError:
            raise ValueError(f'{where}: looks must be an iterable') from None
    out = []
    for k, item in enumerate(items):
        if isinstance(item, dict):
            seeds, deltas = item.get('seeds'), item.get('deltas')
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            seeds, deltas = item
        else:
            raise ValueError(f'{where}: look {k} must be a (seeds, deltas) '
                             f'pair or a mapping, got {type(item).__name__}')
        if isinstance(seeds, (str, bytes)) or isinstance(deltas, (str, bytes)):
            raise ValueError(f'{where}: look {k} seeds/deltas must be '
                             'iterables of numbers, not strings')
        try:
            # Same array-truthiness guard as `looks` above.  A 0-d array is
            # not a sequence either, but `list()` raises TypeError for it and
            # the existing explicit error below is the right contract.
            s_in = [] if seeds is None else list(seeds)
            d_in = [] if deltas is None else list(deltas)
        except TypeError:
            raise ValueError(f'{where}: look {k} seeds/deltas must be iterables') from None
        if len(s_in) != len(d_in):
            raise ValueError(f'{where}: look {k} has {len(s_in)} seeds but '
                             f'{len(d_in)} deltas')
        if not s_in:
            raise ValueError(f'{where}: look {k} is empty')
        rows, n_dropped = [], 0
        for s, d in zip(s_in, d_in):
            si = _as_seed_int(s)
            if si is None:
                raise ValueError(f'{where}: look {k} seed {s!r} is not an integer')
            di = _finite_scalar(d)
            if di is None:
                n_dropped += 1
                continue
            rows.append((si, di))
        rows.sort(key=lambda t: t[0])
        _ss = [s for s, _d in rows]
        if len(set(_ss)) != len(_ss):
            raise ValueError(f'{where}: look {k} contains duplicate seeds {_ss!r}')
        if not rows:
            raise ValueError(f'{where}: look {k} has no finite delta')
        out.append({'seeds': _ss, 'deltas': [float(d) for _s, d in rows],
                    'n_dropped': int(n_dropped)})
    return out


# Look histories compare deltas from successive analyses of the same contrast.
# A delta's unit is arbitrary (a log-PPL difference can be rescaled by any
# positive constant), so both the stale check and the "same look" dedup
# signature must use one common scale.  An absolute tolerance (historically
# 1e-9) and ``round(v, 12)`` silently treat every tiny panel as unchanged and
# collapse distinct looks to one zero signature; a large panel can conversely
# flag a byte-identical value as stale because its last-bit differences exceed
# the absolute margin.  These helpers normalise once per contrast instead.
_LOOK_REL_TOL = 1e-9
_LOOK_SIG_DECIMALS = 12


def _look_values_scale(values):
    """Largest finite magnitude in a look/history sequence (0 if none)."""
    scale = 0.0
    try:
        iterator = iter(values)
    except TypeError:
        return 0.0
    for value in iterator:
        f = _finite_scalar(value)
        if f is not None:
            scale = max(scale, abs(f))
    return scale if math.isfinite(scale) else 0.0


def _look_values_close(a, b, scale, *, rel_tol=_LOOK_REL_TOL):
    """Scale-relative equality for two look deltas.

    ``scale`` is the common magnitude of the whole contrast history.  When it
    is zero both values must be exactly zero (the only finite value at that
    scale); otherwise the difference is measured in units of ``scale``.
    """
    fa = _finite_scalar(a)
    fb = _finite_scalar(b)
    if fa is None or fb is None:
        return False
    if scale and math.isfinite(scale):
        return abs(fa / scale - fb / scale) <= rel_tol
    return fa == fb


def _look_values_signature(values, scale):
    """Hashable identity of a look's deltas at the contrast's common scale."""
    if scale and math.isfinite(scale):
        return tuple(round(float(v) / scale, _LOOK_SIG_DECIMALS)
                     for v in values)
    return tuple(0.0 for _ in values)


def sequential_signflip_p(looks, *, alpha=0.05):
    """Family-wise sign-flip p over multiple looks at a growing seed set.

    ``looks`` is an ordered list of ``(seeds, deltas)`` / ``{seeds, deltas}``
    snapshots of the SAME paired contrast, earliest first.  The LAST look is
    the decisive one: its observed sign defines the direction of the claim.
    For a union of at most :data:`LOOK_SEQUENTIAL_ENUM_MAX_N` seeds the null is
    enumerated exactly:

      every seed gets an independent random sign; the same sign vector is
      projected onto each look's seed set; for every look the evidence in the
      FINAL direction is ``T_k = sign_final * sum_{i in look_k} sign_i d_i``.
      The sequential statistic is ``max_k T_k``, compared with the final
      observed ``T_final = |sum_{i in final} d_i|``.  ``p_seq`` is the exact
      two-sided p, ``min(1, 2 * P(max_k T_k >= T_final))``.

    This is deliberately NOT a min-p over the raw look p-values: an early look
    that happened to be more extreme (possibly in the opposite direction) must
    not turn a final null into a "significant" claim.  Including the final look
    in the maximum guarantees ``p_seq >= p_final``; the two-sided doubling over
    the max makes it a valid family-wise p for "the final effect, allowing for
    the analyses already spent".

    Larger unions use the valid but conservative Bonferroni bound over looks
    (``min(1, K * p_final)``).  Conflicting deltas for the same seed across
    looks (a re-run/edited record) also fall back to that bound and are named
    in ``inconsistent_seeds`` — a stale look is never silently combined.  A
    panel whose final significance is a product of topping up seeds is thus
    downgraded rather than reported at its final-n p.
    """
    alpha = _validated_alpha(alpha)
    series = _normalise_look_series(looks)
    k_looks = len(series)
    # Seed deltas were previously normalised only for extreme magnitudes
    # (<1e-100 / >1e100).  That left the absolute 1e-9 consistency tolerance
    # and the exact 1e-12 enumeration tolerance scale-dependent for ordinary
    # small panels: 1e-12-scaled deltas looked identical, and every sign
    # assignment could pass the 1e-12 cutoff.  Normalise the whole look
    # series by one common finite scale unconditionally (when non-zero); the
    # sign-flip p and the relative tolerances are then invariant under a
    # units-only rescaling.
    _delta_abs = [abs(float(d)) for lk in series for d in lk['deltas']]
    _delta_scale = max(_delta_abs, default=0.0)
    _scale_deltas = bool(math.isfinite(_delta_scale) and _delta_scale > 0.0)

    def _work_deltas(lk):
        if not _scale_deltas:
            return lk['deltas']
        return [float(d) / _delta_scale for d in lk['deltas']]

    look_stats = []
    for i, lk in enumerate(series, 1):
        sf = exact_signflip_p(lk['deltas'])
        work = _work_deltas(lk)
        if _scale_deltas:
            _ms = _scaled_mean_std_if_extreme(lk['deltas'], ddof=1)
            mean_raw = (_ms[0] if _ms is not None
                        else float(np.mean(lk['deltas'])))
            sum_raw = float(_delta_scale * math.fsum(work))
        else:
            mean_raw = float(np.mean(lk['deltas']))
            sum_raw = float(np.sum(lk['deltas']))
        look_stats.append({
            'look': int(i), 'n': int(sf.get('n') or 0),
            'seeds': list(lk['seeds']), 'p': float(sf.get('p', float('nan'))),
            'mean': mean_raw,
            'sum': sum_raw,
            'p_floor': sf.get('p_floor'), 'method': sf.get('method'),
            'n_dropped': int(lk.get('n_dropped') or 0),
        })
    if k_looks == 0:
        return {'n_looks': 0, 'p_sequential': float('nan'),
                'p_min_look': float('nan'), 'p_final_look': float('nan'),
                'looks': [], 'method': 'none', 'alpha': float(alpha),
                'n_union_seeds': 0, 'inconsistent_seeds': [],
                'note': 'no looks supplied'}
    _ps = [s['p'] for s in look_stats if math.isfinite(s['p'])]
    p_min = float(min(_ps)) if _ps else float('nan')
    p_final = look_stats[-1]['p']
    if k_looks == 1:
        return {'n_looks': 1, 'p_sequential': p_final, 'p_min_look': p_min,
                'p_final_look': p_final, 'looks': look_stats,
                'method': 'single-look', 'alpha': float(alpha),
                'n_union_seeds': len(series[0]['seeds']),
                'inconsistent_seeds': [],
                'note': ('one analysis look: no optional-stopping correction '
                         'is needed')}
    # Build the union of seed effects.  A seed must carry one delta across
    # looks; a changed value means the record was re-run and the old look is
    # not a look at the same estimand.
    seed_delta, inconsistent = {}, set()
    for lk in series:
        for s, d in zip(lk['seeds'], _work_deltas(lk)):
            if s in seed_delta:
                if abs(seed_delta[s] - d) > 1e-9:
                    inconsistent.add(s)
            else:
                seed_delta[s] = d
    union_seeds = sorted(seed_delta)
    base = {'n_looks': k_looks, 'p_min_look': p_min,
            'p_final_look': p_final, 'looks': look_stats,
            'alpha': float(alpha), 'n_union_seeds': len(union_seeds),
            'inconsistent_seeds': sorted(inconsistent)}
    _final_sum = float(math.fsum(_work_deltas(series[-1])))
    if abs(_final_sum) <= 1e-300:
        base.update({'p_sequential': 1.0,
                     'method': 'degenerate-final-zero',
                     'note': ('the final look has exactly zero total effect; '
                              'no directional sequential claim is possible')})
        return base
    direction = 1.0 if _final_sum > 0 else -1.0
    t_obs = abs(_final_sum)
    if inconsistent or len(union_seeds) > LOOK_SEQUENTIAL_ENUM_MAX_N:
        p_seq = (min(1.0, float(k_looks) * p_final)
                 if math.isfinite(p_final) else float('nan'))
        base.update({
            'p_sequential': p_seq,
            'method': ('bonferroni-over-looks' if not inconsistent
                       else 'bonferroni-over-looks-inconsistent-seeds'),
            'note': ('union too large for exact joint enumeration or seed '
                     'deltas changed across looks; Bonferroni over the FINAL '
                     'look p is valid but conservative')})
        return base
    # Exact joint enumeration of max_k T_k in the final direction.  Only the
    # column projection of each look is needed; the per-look null distribution
    # is implicit in the union sign enumeration.
    cols = []
    union_index = {s: i for i, s in enumerate(union_seeds)}
    for lk in series:
        cols.append(np.asarray([union_index[s] for s in lk['seeds']],
                               dtype=np.int64))
    d_union = np.asarray([seed_delta[s] for s in union_seeds], dtype=float)
    total = 1 << len(union_seeds)
    bitpos = np.arange(len(union_seeds), dtype=np.int64)[None, :]
    step = max(1, min(total, (1 << 18) // max(len(union_seeds), 1)))
    count = 0
    for off in range(0, total, step):
        bs = min(step, total - off)
        masks = np.arange(off, off + bs, dtype=np.int64)[:, None]
        signs = np.where((masks & (1 << bitpos)) != 0, 1.0, -1.0)
        t_max = None
        for cc in cols:
            # The same union sign vector restricted to this look's columns;
            # seeds outside the look contribute nothing.
            sub = direction * (signs[:, cc] @ d_union[cc])
            t_max = sub if t_max is None else np.maximum(t_max, sub)
        count += int((t_max >= t_obs - 1e-12).sum())
    p_one = count / float(total)
    p_seq = min(1.0, 2.0 * p_one)
    # The final look is inside the maximum, so p_seq >= p_final up to floating
    # rounding; clamp explicitly rather than report a numerically "better"
    # sequential p than the ordinary final test.
    if math.isfinite(p_final):
        p_seq = max(p_seq, float(p_final))
    base.update({'p_sequential': float(p_seq),
                 'method': 'exact-max-over-looks-final-direction',
                 'note': ('exact joint sign-flip null over all seeds used by '
                          'any look, maximised in the final effect direction; '
                          'p >= the final-look p')})
    return base


def _percentile_ci(vals, lo=2.5, hi=97.5):
    v = [f for f in (_finite_scalar(x) for x in vals) if f is not None]
    if len(v) < 2:
        return None
    v.sort()
    def _pct(q):
        pos = (len(v) - 1) * q / 100.0
        i = int(math.floor(pos))
        j = min(i + 1, len(v) - 1)
        return v[i] + (v[j] - v[i]) * (pos - i)
    lo_v, hi_v = _pct(lo), _pct(hi)
    if not (math.isfinite(lo_v) and math.isfinite(hi_v)):
        # Interpolation between two finite but opposite-signed extremes can
        # overflow the subtraction; that is an unrepresentable interval, not a
        # measured bound.
        return None
    return [lo_v, hi_v]

# Bootstrap draws are generated in row-chunks.  A fixed chunk of 256 draws
# needs 256 * n_units index/count elements; for a scale-up evaluation panel
# with millions of documents that is many GB per chunk.  Keep the historical
# chunk size for ordinary panels, and shrink it only when the per-draw unit
# count would otherwise exceed this element budget.
_BOOTSTRAP_CHUNK_DRAWS = 256
_BOOTSTRAP_CHUNK_ELEMS = 1 << 20


def _bootstrap_draw_chunk(n_units):
    """Maximum draws that keep one chunk under the element budget."""
    n = max(1, int(n_units))
    return max(1, min(_BOOTSTRAP_CHUNK_DRAWS, _BOOTSTRAP_CHUNK_ELEMS // n))


def _bootstrap_t_interval(values, *, n_boot, seed, conf):
    """Studentized (bootstrap-t) interval for a mean, with a safe fallback.

    The percentile bootstrap is first-order accurate and, at the 3-12 unit
    counts this repository reports, measurably under-covers a nominal 95%
    interval.  The bootstrap-t construction studentizes every resample with
    its own standard error and is second-order accurate, so its coverage is
    close to nominal at the same unit count.  Draws are generated in fixed
    chunks (the working set is O(n_boot), not O(n_boot * n_units)) and the RNG
    stream is local and seeded.

    A resample whose standard error is zero carries no scale information; if
    too few studentized draws are usable (e.g. a constant vector or a panel
    with only a handful of distinct values) the function falls back to the
    percentile interval and says so in ``method``.  The fallback is explicit,
    never a silent switch to a different estimand.
    """
    v = np.asarray(values, dtype=float)
    n = int(v.size)
    # Square-based sd under/overflows for extreme but finite magnitudes.  The
    # studentized statistic is scale invariant, so run the whole bootstrap on
    # a unit-scaled copy and map the interval endpoints back afterwards.
    _scale = (float(np.max(np.abs(v))) if n else 0.0)
    _scaled = bool(math.isfinite(_scale) and _scale > 0.0
                   and (_scale < 1e-100 or _scale > 1e100))
    if _scaled:
        v = v / _scale
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else float('nan')
    se = sd / math.sqrt(n) if n > 1 and math.isfinite(sd) else float('nan')
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(n_boot), dtype=float)
    tstats = np.empty(int(n_boot), dtype=float)
    chunk = _bootstrap_draw_chunk(n)
    for off in range(0, int(n_boot), chunk):
        m = min(chunk, int(n_boot) - off)
        idx = rng.integers(0, n, size=(m, n))
        X = v[idx]
        bm = X.mean(axis=1)
        bs = X.std(axis=1, ddof=1)
        means[off:off + m] = bm
        with np.errstate(divide='ignore', invalid='ignore'):
            tstats[off:off + m] = np.where(
                bs > 0, (bm - mean) / (bs / math.sqrt(n)), np.nan)
    lo_q = 100.0 * (1.0 - float(conf)) / 2.0
    hi_q = 100.0 - lo_q
    valid_t = np.isfinite(tstats)
    n_valid = int(valid_t.sum())
    if (n >= 3 and se > 0 and math.isfinite(se)
            and n_valid >= max(20, int(0.5 * int(n_boot)))):
        q_lo, q_hi = np.percentile(tstats[valid_t], [lo_q, hi_q])
        lo = mean - float(q_hi) * se
        hi = mean - float(q_lo) * se
        method = ('studentized bootstrap-t over independent units '
                  '(chunked, deterministic)')
        used = int(n_valid)
    else:
        lo, hi = np.percentile(means, [lo_q, hi_q])
        method = ('percentile bootstrap fallback (too few positive-variance '
                  'resamples for a studentized interval)')
        used = 0
    if _scaled:
        lo = float(lo) * _scale
        hi = float(hi) * _scale
    if not (math.isfinite(lo) and math.isfinite(hi)):
        # The true endpoint is beyond float64.  Do not hand an infinite bound
        # to a caller that writes JSON; the interval is unavailable.
        return {'ci_low': None, 'ci_high': None,
                'method': method + '; interval unavailable: an endpoint '
                          'exceeds the representable float64 range',
                'n_boot_studentized': used}
    return {'ci_low': float(lo), 'ci_high': float(hi), 'method': method,
            'n_boot_studentized': used}


def bootstrap_mean_ci(values, *, n_boot=4000, seed=20261018, conf=0.95):
    """Studentized bootstrap CI for the mean of i.i.d. units (e.g. documents).

    Units must be the independent sampling unit.  For LM evaluation a document
    row is the unit (tokens within a document are correlated), and this helper
    is used to report **document-sampling** uncertainty, not seed-level
    training uncertainty.  Resampling is chunked, so the working set does not
    grow with n_boot.  The RNG is seeded, so repeated report builds return the
    same interval.

    The interval is a bootstrap-t (studentized) interval rather than the raw
    percentile interval: the percentile construction under-covers at the small
    unit counts this repository uses (its coverage is calibrated and reported
    by ``statistical_calibration_suite``).  A constant or degenerate input
    falls back to the percentile interval, and the returned ``method`` names
    the fallback instead of hiding it.
    """
    v = np.asarray([f for f in (_finite_scalar(x) for x in (values if values is not None else []))
                    if f is not None], dtype=float)
    n = int(v.size)
    if n < 2:
        return None
    try:
        c = float(conf)
    except (TypeError, ValueError, OverflowError):
        return None
    nb = _as_seed_int(n_boot)
    sd = _as_seed_int(seed)
    if nb is None or sd is None or isinstance(conf, (bool, np.bool_)):
        return None
    if nb < 20 or not (0.0 < c < 1.0):
        return None
    _ci = _bootstrap_t_interval(v, n_boot=nb, seed=sd, conf=c)
    if _ci.get('ci_low') is None or _ci.get('ci_high') is None:
        return None
    _ms = _scaled_mean_std_if_extreme(v, ddof=1)
    _mean = _ms[0] if _ms is not None else float(v.mean())
    return {'n': n, 'n_boot': nb, 'seed': int(seed), 'mean': _mean,
            'ci_low': _ci['ci_low'], 'ci_high': _ci['ci_high'], 'conf': c,
            'method': _ci['method'],
            'n_boot_studentized': _ci['n_boot_studentized']}
