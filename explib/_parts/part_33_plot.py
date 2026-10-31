
def _plot(summary, agg, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not agg:
        return
    order = ['full', 'full_matched', 'full_cos', 'full_sink', 'full_sw128', 'full_sw128_matched', 'csa_fixed', 'csa_dynamic', 'csa_dyn_fuse', 'csa_dyn_nocomp', 'csa_fix_randidx', 'csa_fix_zerocont', 'csa_fix_shiftcont', 'csa_fix_nosink', 'csa_fix_topk8', 'csa_fix_topk64', 'hybrid_fixed', 'hybrid_dynamic', 'hybrid_csa_dyn', 'hybrid_hca_dyn', 'hybrid_csa_dyn_fuse']
    vs = [v for v in order if v in agg] + [v for v in agg if v not in order]
    fig, ax = plt.subplots(1, 5, figsize=(27, 4.5))
    ps = [agg[v].get('ppl_primary', agg[v].get('ppl_mean', 0.0))
          for v in vs]
    es = [agg[v].get('ppl_std') or 0.0 for v in vs]
    ax[0].bar(range(len(vs)), ps, yerr=es, capsize=3, color='steelblue')
    ax[0].set_xticks(range(len(vs)))
    ax[0].set_xticklabels(vs, rotation=30, ha='right')
    ax[0].set_ylabel('validation perplexity (mean ± std)')
    ax[0].set_title('val PPL (corpus per run cfg)')
    ax[0].grid(alpha=0.3, axis='y')
    _unknown = [v for v in vs if agg[v].get('params') is None]
    if _unknown:
        print(f'[plot] {len(_unknown)} variant(s) report no `params` ({', '.join(_unknown)}) — their size bar is left at 0 and the size panel cannot be read for them')
    params_m = [(agg[v].get('params') or 0) / 1000000.0 for v in vs]
    ax[1].bar(range(len(vs)), params_m, color='orange')
    ax[1].set_xticks(range(len(vs)))
    ax[1].set_xticklabels(vs, rotation=30, ha='right')
    ax[1].set_ylabel('params (M)')
    ax[1].set_title('Model size')
    ax[1].grid(alpha=0.3, axis='y')
    for v in vs:
        _curves = []
        for _r in agg[v].get('_group_records', []):
            if not isinstance(_r, dict) or _r.get('synthesized'):
                continue
            _loss = _r.get('losses') if isinstance(_r.get('losses'), list) else _cell_curve_data(outdir, _r).get('losses')
            if isinstance(_loss, list) and _loss and all((isinstance(x, (int, float)) and math.isfinite(x) for x in _loss)):
                _curves.append(_loss)
        if not _curves:
            continue
        _lens = sorted({len(_c) for _c in _curves})
        if len(_lens) > 1:
            print(f'[_plot] {v}: seeds trained a DIFFERENT number of steps {_lens} — the training curve is drawn over the shared prefix of {_lens[0]} step(s) only.')
        n_pt = min((len(_c) for _c in _curves))
        curve = np.mean([np.array(_c[:n_pt], dtype=float) for _c in _curves], axis=0)
        k = max(1, n_pt // 50)
        _num = np.convolve(curve, np.ones(k), mode='same')
        _den = np.convolve(np.ones_like(curve), np.ones(k), mode='same')
        smooth = np.where(_den > 0, _num / _den, 0.0)
        ax[2].plot(smooth, label=v)
    ax[2].set_xlabel('step')
    ax[2].set_ylabel('train loss (seed-averaged)')
    ax[2].set_title('Training curves')
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=0.3)
    drew = False
    for v in vs:
        cur = agg[v].get('ppl_curve')
        if not cur:
            continue
        steps_h = np.array([p[0] for p in cur], dtype=float)
        ppls_h = np.array([p[1] for p in cur], dtype=float)
        tok_per_step = agg[v].get('tokens_per_step')
        if not tok_per_step:
            print(f'[_plot] {v}: no `tokens_per_step` in the aggregate — omitted from the tokens axis (plotting steps under a tokens label would be wrong by seq_len*batch_size). Re-run `aggregate` to stamp it.')
            continue
        ax[3].plot(steps_h * tok_per_step / 1000000.0, ppls_h, marker='.', ms=4, label=v)
        drew = True
    ax[3].set_xlabel('tokens seen (M)')
    ax[3].set_ylabel('val-subset PPL')
    ax[3].set_title('PPL vs training tokens (gap dynamics)')
    if drew:
        ax[3].legend(fontsize=7)
    ax[3].grid(alpha=0.3)
    _dense_keys = [k for k in agg if k.split('#')[0].split('@')[0] == 'full']
    _dense_key = next((k for k in _dense_keys if agg[k].get('ppl_curve') and agg[k].get('tokens_per_step')), None)
    if _dense_key is None:
        print(f'[_plot] no dense `full` group with both a `ppl_curve` and a `tokens_per_step` in this aggregate (candidates: {_dense_keys or 'none'}) — the sparse-vs-dense gap panel (finding #10) is NOT drawn.')
    else:
        if _dense_key != 'full':
            print(f'[_plot] the dense baseline is published as `{_dense_key}` (the `full` group was split by `run_cfg`); the sparse-vs-dense gap is taken against it' + (f', not {_dense_keys[1:]}' if len(_dense_keys) > 1 else '') + '.')
        ref = agg[_dense_key].get('ppl_curve')
        tps0 = agg[_dense_key].get('tokens_per_step')
    if _dense_key is not None and ref and tps0:
        gx = np.array([p[0] for p in ref], dtype=float) * tps0
        gy = np.array([p[1] for p in ref], dtype=float)
        drew2 = False
        for v in vs:
            if v == _dense_key:
                continue
            if v.startswith('full@cfg'):
                continue
            cur = agg[v].get('ppl_curve')
            if not cur:
                continue
            tps = agg[v].get('tokens_per_step')
            if not tps:
                continue
            xs = np.array([p[0] for p in cur], dtype=float) * tps
            ys = np.array([p[1] for p in cur], dtype=float)
            keep = (gx >= xs[0]) & (gx <= xs[-1])
            if not keep.any():
                continue
            ax[4].plot(gx[keep] / 1000000.0, np.interp(gx[keep], xs, ys) - gy[keep], marker='.', ms=4, label=f'{v}−full')
            drew2 = True
        ax[4].axhline(0, color='k', lw=0.8)
        ax[4].set_xlabel('tokens seen (M)')
        ax[4].set_ylabel('ΔPPL vs full')
        ax[4].set_title('sparse-vs-dense gap vs tokens (finding #10)')
        if drew2:
            ax[4].legend(fontsize=7)
        ax[4].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'summary.png'), dpi=120)
    plt.close(fig)

def _read_summary_json(path):
    summary = {}
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as f:
                summary = json.load(f)
        except Exception as _e:
            print(f'[resume] FATAL: {path} exists but cannot be parsed ({type(_e).__name__}: {_e}).  Refusing to overwrite it with an empty summary — move it aside to start fresh.')
            raise
        if not isinstance(summary, dict):
            # Resume/update paths index the summary by cell key.  A parsed but
            # non-object file cannot carry any cell records, and overwriting it
            # would silently discard whatever the operator put there.
            raise ValueError(
                f'{path} parsed as a {type(summary).__name__}, not a cell -> '
                'record object; refusing to resume from/overwrite it')
    return summary


def _run_fp(cfg, d, n_layers, n_heads, d_head, vocab, train_ids, val_batch, device=None):
    """The resume fingerprint of a run() cell.  The parent planner and every
    par.py child compute the exact same string: the child re-serves the token
    ids from the same cache files, and `_ids_fp` is content-based.

    AMP / eager-twin modes are resolved from the run config (persisted in the
    manifest) or from an explicitly supplied ``device``; they are never read
    from this process's global CUDA state.  A zero-GPU planner can therefore
    recompute a GPU panel's fingerprint from the same config, and a child
    receives the parent's resolved modes through the job config."""
    _mg = cfg.get('matched')
    if _mg is None:
        _mkey = 'default'
    else:
        _mkey = ','.join(sorted((str(_x) for _x in _mg)))
    cfg.setdefault('warmup', 50)
    cfg.setdefault('comp_lambda', 0.05)
    _declared_impl = any(
        (k in cfg) and not (isinstance(cfg.get(k), str) and cfg.get(k).strip().lower() == 'auto')
        for k in ('amp_dtype', 'eager_batch'))
    _amp_mode, _eb = _resolve_impl_modes(cfg, device=device)
    if device is None and not _declared_impl:
        # No machine-dependent implementation descriptor enters the content
        # fingerprint.  Resolve to the canonical fp32/row-loop descriptor
        # explicitly (not via training_knobs()'s global DEVICE inference).
        try:
            _tk = training_knobs_fingerprint(amp_dtype=None, eager_batch=False)
        except TypeError:
            # Keep callers that replace training_knobs() with a zero-argument
            # test double working; their digest is the canonical one on CPU.
            _tk = training_knobs_fingerprint()
    else:
        _tk = training_knobs_fingerprint(amp_dtype=_amp_mode, eager_batch=_eb)
    _amp = _amp_mode or 'fp32'
    _ae = _adaptive_exact_limit()
    # The sampled long-context segmenter's grid depends on this ceiling too.
    # Keep the default token absent so existing default fingerprints stay
    # stable, but a non-default ceiling must invalidate resumed cells.
    _am = _adaptive_max_limit()
    _am_tok = '' if _am == (1 << 20) else f'_am{_am}'
    # The task/synthetic recipe (`data_extra`) is part of the data identity.
    # Keep the token absent when it is unset so legacy default fingerprints
    # stay stable, but a recipe change must invalidate resume.
    _es_pat = cfg.get('early_stop_patience')
    _es_tok = ''
    if _es_pat is not None:
        _es_tok = f'_esp{_es_pat}_esd{cfg.get("early_stop_min_delta", 0.0)}'
    _de = cfg.get('data_extra')
    _de_tok = ''
    if _de:
        try:
            _de_tok = '_de' + hashlib.sha256(
                json.dumps(_de, sort_keys=True, default=str).encode('utf-8')
            ).hexdigest()[:16]
        except (TypeError, ValueError):
            _de_tok = '_de' + str(_de)[:64]
    return f'steps{cfg['steps']}_sl{cfg['seq_len']}_bs{cfg['batch_size']}_nt{cfg['n_train_tokens']}_lr{cfg['lr']}_wd{cfg['weight_decay']}_d{d}_L{n_layers}_H{n_heads}_Dh{d_head}_v{vocab}_wu{cfg['warmup']}_cl{cfg['comp_lambda']}_dlm{cfg.get('delta_lr_mult', 10.0)}_mt{_mkey}_mr{cfg.get('mlp_match_ref', 'csa_dynamic')}_ee{cfg.get('eval_every', 0)}_es{cfg.get('eval_subset', 128)}_det{determinism_label()}_df{_ids_fp(train_ids)}-{_ids_fp(np.ascontiguousarray(val_batch).reshape(-1))}{_de_tok}{_es_tok}_tk{_tk}_amp{_amp}_eb{int(_eb)}_ae{_ae}{_am_tok}_cs{code_stamp()}'


# ---------------------------------------------------------------------------
# Run-order design: calendar time is a nuisance factor, not a treatment
# ---------------------------------------------------------------------------
# A panel that runs every seed of arm A and only then every seed of arm B has
# arm perfectly confounded with time: driver updates, thermal state, a noisy
# neighbour or a changed cluster image can all produce a between-arm gap that
# the paired seed test will read as an arm effect.  The design below makes the
# run order explicit: a deterministic randomized complete-block order with one
# block per seed (so both arms of every pair are trained close together), plus
# a position rotation that prevents one arm from always occupying the same
# slot inside a block.  It is a design choice, not a measurement change; the
# seed-level RNG and every recorded value are untouched.

RUN_ORDER_SCHEMA = 1
RUN_ORDER_DESIGNS = ('randomized_complete_block_balanced', 'inferred_no_plan')
EVALUATION_ORDER_SCHEMA = 1


def _run_order_u64(*parts):
    """Stable 64-bit draw: Python's hash() is randomized per process."""
    raw = '\x1f'.join(str(p) for p in parts).encode('utf-8')
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big')


def _run_order_default_label(cell, block_key):
    if isinstance(cell, (tuple, list)) and len(cell) >= 2:
        return f'{cell[0]}::seed{block_key(cell)}'
    return f'{cell}::seed{block_key(cell)}'


def blocked_run_order(cells, *, block_key, seed=0, balance_positions=True,
                      balance_key=None):
    """Return a deterministic randomized complete-block run order.

    ``block_key(cell)`` names the block (normally the seed).  Cells inside a
    block are run contiguously; the block order itself is sorted by key so a
    resumed or parallel pass sees the same plan.  ``balance_key(cell)`` names
    the within-block treatment (normally the arm), which is what position
    balance rotates.  It defaults to the first element of a ``(variant, seed)``
    tuple.

    When every block contains the same set of treatment keys,
    ``balance_positions`` uses a seeded base permutation plus a per-block
    rotation, so no arm always occupies the first slot (a fixed "first arm is
    always the baseline" order would leave a warm-up order effect confounded
    with arm).  Otherwise each block is shuffled independently with the same
    stable hash.

    The function is pure and deterministic: the same cells, block key and
    design seed always give the same order, independent of ``PYTHONHASHSEED``.
    It never changes a training value; each cell still seeds its own RNG.
    """
    if not callable(block_key):
        raise TypeError('block_key must be callable')
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise TypeError(f'run-order seed must be an integer, got {seed!r}')
    seed = int(seed)
    try:
        raw = [tuple(c) if isinstance(c, list) else c for c in cells]
    except TypeError as e:
        raise TypeError(f'cells must be an iterable of hashable cells: {e}') from e
    if not raw:
        return []
    groups = collections.OrderedDict()
    seen = set()
    for cell in raw:
        try:
            hash(cell)
        except TypeError as e:
            raise TypeError(
                f'run-order cells must be hashable tuples (got '
                f'{type(cell).__name__}); a dict/list cell cannot be a resume '
                f'identity') from e
        if cell in seen:
            raise ValueError(f'duplicate run-order cell {cell!r}; training it '
                             'twice would corrupt the (arm, seed) identity')
        seen.add(cell)
        key = block_key(cell)
        try:
            hash(key)
        except TypeError as e:
            raise TypeError(f'block key for {cell!r} must be hashable') from e
        groups.setdefault(key, []).append(cell)
    keys = list(groups)
    try:
        keys.sort()
    except TypeError:
        keys.sort(key=repr)

    def _treatment(cell):
        if balance_key is not None:
            return balance_key(cell)
        if isinstance(cell, (tuple, list)) and cell:
            return cell[0]
        return cell

    ref = max(groups.values(), key=len)
    ref_treatments = [_treatment(c) for c in ref]
    try:
        hash(tuple(ref_treatments))
    except TypeError as e:
        raise TypeError('balance_key values must be hashable') from e
    if len(set(ref_treatments)) != len(ref_treatments):
        raise ValueError(
            'balance_key is not unique inside a block; the position-rotation '
            'design cannot identify the cell to place')
    treatment_set = set(ref_treatments)
    same_sets = all(
        len(g) == len(ref)
        and {_treatment(c) for c in g} == treatment_set
        for g in groups.values())
    base = list(ref_treatments)
    for i in range(len(base) - 1, 0, -1):
        j = _run_order_u64(seed, 'base', i) % (i + 1)
        base[i], base[j] = base[j], base[i]
    out = []
    for bi, key in enumerate(keys):
        group = list(groups[key])
        if balance_positions and same_sets and len(base) > 1:
            by_treatment = {_treatment(c): c for c in group}
            if len(by_treatment) != len(group):
                raise ValueError(f'block {key!r} has duplicate treatment keys')
            offset = (seed + bi) % len(base)
            rotated = base[offset:] + base[:offset]
            group = [by_treatment[t] for t in rotated]
        elif len(group) > 1:
            for i in range(len(group) - 1, 0, -1):
                j = _run_order_u64(seed, repr(key), i) % (i + 1)
                group[i], group[j] = group[j], group[i]
        out.extend(group)
    return out


def _normalise_environment_slots(slots):
    """Validate the optional hardware slots of a run-order plan.

    Slots are labels (``gpu0``/``gpu1`` or the actual
    ``CUDA_VISIBLE_DEVICES`` tokens).  They are metadata only: the plan does
    not pin a physical device, it records the balanced assignment a scheduler
    should follow.  Duplicate/empty labels would make "arm on slot" ambiguous,
    so they fail closed.
    """
    if not isinstance(slots, (list, tuple)) or not slots:
        raise ValueError('environment_slots must be a non-empty list/tuple')
    out = []
    for slot in slots:
        if slot is None or isinstance(slot, (bool, np.bool_)):
            raise ValueError(f'invalid environment slot {slot!r}')
        text = str(slot).strip()
        if not text:
            raise ValueError('environment slots must be non-empty labels')
        if text in out:
            raise ValueError(f'duplicate environment slot {text!r}')
        out.append(text)
    return out


def build_run_order_plan(cells, *, block_key, seed=0, block_name='seed',
                         variant_key=None, label=None,
                         balance_positions=True, balance_key=None,
                         design_name='randomized_complete_block_balanced',
                         created_at=None, environment_slots=None):
    """Build the JSON-safe plan written next to a panel's summary.

    Only the plan is new metadata; the caller still trains the same cells with
    the same seeds and hyperparameters.  ``blocks`` records the within-block
    order and ``planned_cells`` the flat order, so a later audit can check both
    the design and the actual event order without re-deriving them.

    ``environment_slots`` optionally adds a balanced hardware/device
    assignment (``cell_environments``).  Within each complete block the slot
    rotates with the arm's position; because the position design already gives
    every arm every slot once across blocks, each arm then visits every
    environment equally often.  The assignment is a recommendation, not a
    measurement: a dynamic dispatcher may start a cell on a different device,
    which the allocation audit reports rather than silently accepting.
    """
    if design_name not in RUN_ORDER_DESIGNS:
        raise ValueError(f'unknown run-order design {design_name!r}')
    raw = [tuple(c) if isinstance(c, list) else c for c in cells]
    if not raw:
        raise ValueError('a run-order plan needs at least one cell')
    ordered = blocked_run_order(raw, block_key=block_key, seed=seed,
                                balance_positions=balance_positions,
                                balance_key=balance_key)
    slots = (None if environment_slots is None
             else _normalise_environment_slots(environment_slots))

    def _variant(cell):
        if variant_key is not None:
            return str(variant_key(cell))
        if isinstance(cell, (tuple, list)) and cell:
            return str(cell[0])
        return str(cell)

    def _label(cell):
        return _run_order_default_label(cell, block_key) if label is None else str(label(cell))

    labels = [_label(c) for c in ordered]
    if len(set(labels)) != len(labels):
        raise ValueError('run-order labels are not unique; a label is the '
                         'cell identity used by the audit')
    variants = []
    blocks = collections.OrderedDict()
    cell_blocks = {}
    cell_variants = {}
    for cell in ordered:
        v = _variant(cell)
        b = str(block_key(cell))
        if v not in variants:
            variants.append(v)
        blocks.setdefault(b, []).append(v)
        cell_blocks[_label(cell)] = b
        cell_variants[_label(cell)] = v
    _same_blocks = all(
        len(vs) == len(variants) and set(vs) == set(variants)
        for vs in blocks.values())
    _pos_policy = ('cyclic_base_rotation'
                   if balance_positions and _same_blocks
                   else 'independent_block_shuffle')
    plan = {
        'kind': 'run_order',
        'schema': RUN_ORDER_SCHEMA,
        'created_at': created_at or time.strftime('%Y-%m-%dT%H:%M:%S'),
        'design': {
            'name': design_name,
            'block': block_name,
            'design_seed': int(seed),
            'randomized': design_name != 'inferred_no_plan',
            'position_balance': _pos_policy,
            'execution': 'cells_inside_a_block_are_contiguous',
        },
        'variants': variants,
        'seeds': list(blocks),
        'blocks': {b: list(vs) for b, vs in blocks.items()},
        'planned_cells': labels,
        'cell_blocks': cell_blocks,
        'cell_variants': cell_variants,
    }
    if slots is not None:
        # The position design already gives every arm every slot once across
        # complete blocks.  Mapping position -> slot therefore balances arm
        # across environments too; rotating by the block index as well would
        # cancel that rotation (position = base - block-index) and put each
        # arm on a single slot.
        labels_by_block = collections.OrderedDict()
        for lab in labels:
            labels_by_block.setdefault(cell_blocks[lab], []).append(lab)
        cell_environments = {}
        for block_labels in labels_by_block.values():
            for pos, lab in enumerate(block_labels):
                cell_environments[lab] = slots[pos % len(slots)]
        if set(cell_environments) != set(labels):
            raise ValueError('environment assignment does not cover every planned cell')
        plan['environment_slots'] = list(slots)
        plan['cell_environments'] = cell_environments
        plan['design']['environment_balance'] = 'cyclic_slot_rotation'
    return plan


def _evaluation_cell_label(cell):
    if not isinstance(cell, (tuple, list)) or len(cell) < 2:
        raise TypeError(
            'evaluation-order cells must be (variant, seed) pairs, got '
            f'{cell!r}')
    s = cell[1]
    if isinstance(s, bool) or not isinstance(s, numbers.Integral):
        raise TypeError(f'evaluation-order seed must be an integer, got {s!r}')
    return f'{cell[0]}::s{int(s)}'


def build_evaluation_order_plan(cells, *, task, seed=0, created_at=None):
    """Build the precommitted execution order for one final-test task.

    The final-test panel has the same temporal-confounding hazard as training:
    evaluating every seed of arm A before arm B makes arm and measurement time
    perfectly confounded.  This wraps the training run-order design (seed
    blocks, randomized within-block arm order, cyclic position balance) and
    labels cells with the P5T event key ``variant::s{seed}`` so the audit can
    compare the plan with `experiments.jsonl` without guessing.

    The plan is metadata only: it never changes a checkpoint, a test row, a
    PPL, or the frozen comparison.  ``task`` names the endpoint family (for
    P5T: ``clean_by_length`` or ``distractor_probe``) so the clean confirmatory
    evaluation and the secondary probe are audited separately.
    """
    if not isinstance(task, str) or not task.strip():
        raise ValueError('an evaluation-order plan needs a non-empty task name')

    def _block_key(cell):
        if not isinstance(cell, (tuple, list)) or len(cell) < 2:
            raise TypeError(
                'evaluation-order cells must be (variant, seed) pairs, got '
                f'{cell!r}')
        s = cell[1]
        if isinstance(s, bool) or not isinstance(s, numbers.Integral):
            raise TypeError(
                f'evaluation-order seed must be an integer, got {s!r}')
        return int(s)

    plan = build_run_order_plan(
        cells, block_key=_block_key, seed=seed, block_name='seed',
        label=_evaluation_cell_label, created_at=created_at)
    plan['kind'] = 'evaluation_order'
    plan['schema'] = EVALUATION_ORDER_SCHEMA
    plan['task'] = task.strip()
    return plan


def evaluation_cells_in_plan_order(cells, plan):
    """Return ``cells`` in the exact order recorded by an evaluation plan.

    A plan that does not cover the current (arm, seed) grid exactly is
    rejected: silently evaluating an unplanned cell, or skipping a planned
    one, would make the recorded order describe a different measurement panel.
    """
    if not isinstance(plan, dict):
        raise ValueError('evaluation plan must be an object')
    labels = plan.get('planned_cells')
    if not isinstance(labels, list) or not labels:
        raise ValueError('evaluation plan has no planned_cells list')
    by_label = {}
    for cell in cells:
        lab = _evaluation_cell_label(cell)
        if lab in by_label:
            raise ValueError(f'duplicate evaluation cell {lab!r}')
        by_label[lab] = (cell[0], int(cell[1]))
    planned = [str(x) for x in labels]
    planned_set = set(planned)
    missing = [x for x in by_label if x not in planned_set]
    extra = [x for x in planned if x not in by_label]
    if missing or extra or len(planned) != len(by_label):
        raise ValueError(
            'evaluation plan does not cover the current cell grid exactly '
            f'(missing={missing[:5]}, extra={extra[:5]})')
    return [by_label[x] for x in planned]


def run_order_balance(plan, actual_order=None):
    """Audit a run-order plan and, optionally, the order actually observed.

    The plan-level contract is deliberately stronger than "the cells were
    shuffled": every block must contain every declared arm exactly once
    (complete blocks), and each arm's slot counts across blocks must differ by
    at most one (position balance).  An arm-major execution — every seed of
    one arm, then every seed of the next — is a FAIL, because arm and calendar
    time are then perfectly confounded and no seed-level p-value can repair it.

    ``actual_order`` is a list of cell labels in execution (start-time) order.
    Missing cells are allowed on a partial/resumed panel; a complete actual
    order that is arm-major is a failure, while a complete order that merely
    deviates from the planned permutation is a warning (parallel dispatch can
    legitimately reorder starts).  This audit is descriptive of the design and
    never changes a PPL, seed or decision rule.
    """
    out = {'kind': 'run_order_balance', 'status': 'ok', 'problems': [],
           'warnings': [], 'n_blocks': 0, 'n_cells': 0, 'n_variants': 0,
           'complete_blocks': 0, 'arm_major': False,
           'position_balanced': None, 'design_ok': False}
    if not isinstance(plan, dict):
        out['status'] = 'fail'
        out['problems'].append(f'run-order plan must be an object, got {type(plan).__name__}')
        return out
    design = plan.get('design') if isinstance(plan.get('design'), dict) else {}
    name = design.get('name')
    if name not in RUN_ORDER_DESIGNS:
        out['problems'].append(
            f'design.name={name!r} is not one of {list(RUN_ORDER_DESIGNS)}')
    elif name != 'inferred_no_plan':
        if design.get('randomized') is not True:
            out['problems'].append('a randomized run-order design must set randomized=true')
        _ds = design.get('design_seed')
        if isinstance(_ds, (bool, np.bool_)) or not isinstance(_ds, numbers.Integral):
            out['problems'].append(
                f'design.design_seed={_ds!r} must be an integer; a random '
                'sequence without a recorded seed cannot be reproduced')
    blocks = plan.get('blocks')
    variants = plan.get('variants')
    if not isinstance(blocks, dict) or not blocks:
        out['problems'].append('plan.blocks must be a non-empty object')
        blocks = {}
    if not isinstance(variants, list) or not variants:
        variants = list(dict.fromkeys(
            v for vs in blocks.values() if isinstance(vs, list) for v in vs))
        if variants:
            out['warnings'].append('plan.variants was missing; recovered it from blocks')
    variants = [str(v) for v in variants]
    if len(set(variants)) != len(variants):
        out['problems'].append('plan.variants contains duplicates')
    out['n_variants'] = len(variants)
    variant_set = set(variants)
    position_counts = {v: [0] * max(1, len(variants)) for v in variants}
    # When the design was inferred from events, an incomplete block is just a
    # partial observation, not proof of a bad plan; only arm-major is a hard
    # failure there.  A recorded plan is held to the full complete-block
    # contract.
    _issue = out['warnings'] if name == 'inferred_no_plan' else out['problems']
    complete = 0
    for b, vs in blocks.items():
        if not isinstance(vs, list) or not vs:
            _issue.append(f'block {b!r} must be a non-empty list')
            continue
        vs = [str(v) for v in vs]
        if len(vs) != len(set(vs)):
            _issue.append(f'block {b!r} contains a duplicate arm')
        if set(vs) == variant_set and len(vs) == len(variants):
            complete += 1
        else:
            _issue.append(
                f'block {b!r} is not a complete block: got {vs!r}, '
                f'expected each of {variants!r} exactly once')
        for i, v in enumerate(vs):
            if v in position_counts and i < len(position_counts[v]):
                position_counts[v][i] += 1
    out['n_blocks'] = len(blocks)
    out['complete_blocks'] = complete
    out['n_cells'] = sum(len(vs) for vs in blocks.values() if isinstance(vs, list))
    if not variants:
        out['problems'].append('plan declares no arms')
    if blocks and variants and complete != len(blocks):
        out['warnings'].append(
            f'{len(blocks) - complete} block(s) are incomplete; a partial '
            'panel can still be described, but it is not a complete-block design')
    balances = []
    for v, counts in position_counts.items():
        if counts:
            balances.append(max(counts) - min(counts))
    out['position_balanced'] = (max(balances) <= 1) if balances else None
    _pos_policy = design.get('position_balance')
    if out['position_balanced'] is False and _pos_policy != 'independent_block_shuffle':
        _issue.append(
            'arm positions are not balanced across blocks (an arm always '
            'occupies the same slot): within-block order is confounded with arm')
    out['arm_major'] = bool(
        len(variants) > 1 and len(blocks) > 1
        and all(isinstance(vs, list) and len(vs) == 1
                for vs in blocks.values()))
    if out['arm_major']:
        out['problems'].append(
            'every block contains only one arm: the panel is arm-major, so arm '
            'and calendar time are perfectly confounded')
    out['design_ok'] = bool(
        not out['problems'] and complete == len(blocks)
        and (out['position_balanced'] is not False
             or _pos_policy == 'independent_block_shuffle'))
    if name == 'inferred_no_plan':
        out['warnings'].append(
            'the run order was inferred from event timestamps because no '
            'run_order.json plan exists; the design itself cannot be verified')
        out['status'] = 'fail' if out['arm_major'] else 'not_verifiable_locally'
    elif out['problems']:
        out['status'] = 'fail'
    if actual_order is not None and not isinstance(actual_order, (list, tuple)):
        out['problems'].append('actual_order must be a list of cell labels')
    elif actual_order is not None:
        planned = plan.get('planned_cells')
        planned = [str(x) for x in planned] if isinstance(planned, list) else []
        label_block = plan.get('cell_blocks') if isinstance(plan.get('cell_blocks'), dict) else {}
        label_variant = plan.get('cell_variants') if isinstance(plan.get('cell_variants'), dict) else {}
        seen = set()
        actual_blocks = collections.OrderedDict()
        for lab in actual_order:
            lab = str(lab)
            if lab in seen:
                out['problems'].append(f'actual order repeats cell {lab!r}')
                continue
            seen.add(lab)
            b = label_block.get(lab)
            if b is None:
                out['problems'].append(
                    f'actual order names {lab!r}, which is not in the plan')
                continue
            actual_blocks.setdefault(str(b), []).append(lab)
        if planned:
            planned_set = set(planned)
            missing = [x for x in planned if x not in seen]
            extra = [x for x in seen if x not in planned_set]
            if extra:
                out['problems'].append(f'actual order contains unplanned cells: {extra[:5]}')
            out['actual'] = {
                'n_observed': len(seen),
                'n_planned': len(planned),
                'complete': not missing and not extra,
                'missing_cells': missing[:20],
            }
            if not missing and not extra:
                # `actual_blocks` groups by seed, so even an arm-major run has
                # every arm present in each seed group; it is the GLOBAL
                # sequence that reveals the confound.  Count maximal runs of
                # the same arm: a legacy arm-major run has one run per arm
                # (arm A seeds 0..n, then arm B seeds 0..n, ...), while a
                # seed-blocked run switches arm inside every block.
                seq_variants = []
                for lab in actual_order:
                    lab = str(lab)
                    v = label_variant.get(lab)
                    if v is None:
                        try:
                            v = lab.rsplit('::seed', 1)[0]
                        except ValueError:
                            v = lab
                    seq_variants.append(str(v))
                runs = 0
                for i, v in enumerate(seq_variants):
                    if i == 0 or v != seq_variants[i - 1]:
                        runs += 1
                variant_counts = collections.Counter(seq_variants)
                out['actual']['variant_runs'] = runs
                out['actual']['n_arms'] = len(variant_counts)
                actual_arm_major = bool(
                    len(actual_blocks) > 1
                    and len(variant_counts) > 1
                    and runs == len(variant_counts)
                    and all(n == len(actual_blocks)
                            for n in variant_counts.values()))
                if actual_arm_major:
                    out['arm_major'] = True
                    out['problems'].append(
                        'the completed run is arm-major even though the plan was '
                        'blocked; the driver did not honour the run-order plan')
        # Group the plan once instead of rescanning every planned cell for
        # every block: a future panel with many seeds would otherwise make the
        # audit quadratic in the number of cells.
        planned_by_block = collections.OrderedDict()
        for x in planned:
            planned_by_block.setdefault(str(label_block.get(x)), []).append(x)
        for b, seq in actual_blocks.items():
            plan_seq = planned_by_block.get(str(b), [])
            it = iter(plan_seq)
            if not all(any(x == y for y in it) for x in seq):
                out['warnings'].append(
                    f'block {b!r}: actual start order {seq!r} is not a '
                    f'subsequence of the planned order {plan_seq!r}')
    if out['problems'] and out['status'] != 'fail':
        out['status'] = 'fail'
    return out


def _same_run_order_plan(old, new):
    """True when ``new`` describes the same allocation design as ``old``.

    A resumed panel rebuilds the plan object from the same (arm, seed) grid.
    That rebuild has a fresh ``created_at``; overwriting the original file
    would turn a genuine precommitment into a post-hoc plan (and the allocation
    audit would correctly fail it).  Identity is the planned cells, the block
    mapping and the design seed — not the timestamp.
    """
    if not isinstance(old, dict) or not isinstance(new, dict):
        return False
    if old.get('kind') != 'run_order' or new.get('kind') != 'run_order':
        return False
    for key in ('planned_cells', 'blocks'):
        if old.get(key) != new.get(key):
            return False
    # Older plans may predate the per-cell maps.  A present map still has to
    # agree; an absent one is not evidence of a different design.
    for key in ('cell_blocks', 'cell_variants'):
        if key in old and old.get(key) != new.get(key):
            return False
    od = old.get('design') if isinstance(old.get('design'), dict) else {}
    nd = new.get('design') if isinstance(new.get('design'), dict) else {}
    return (od.get('name') == nd.get('name')
            and od.get('design_seed') == nd.get('design_seed'))


def write_run_order(outdir, plan):
    """Persist the run-order plan beside the panel's other provenance.

    The first write is the precommitment.  A resume that rebuilds the same
    design keeps that original file (including its ``created_at`` and any
    device assignment), so the allocation audit can still distinguish a
    prospective plan from one reconstructed after execution.
    """
    if not isinstance(plan, dict) or plan.get('kind') != 'run_order':
        raise ValueError('write_run_order expects a plan from build_run_order_plan')
    path = os.path.join(outdir, 'run_order.json')
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as fh:
                old = json.load(fh)
        except Exception:
            old = None
        if _same_run_order_plan(old, plan):
            return path
    atomic_write_json(path, plan)
    return path


def _plan_run(cfg, summary_path, fp):
    """Resume-check every (seed x variant) cell up front and return the cells
    that still need to run.

    Stale / synthesized / non-measurable records are dropped (under the lock)
    BEFORE any training starts, so a later failure can never leave a stale
    reading in place looking like this attempt's result.  With parallel
    dispatch this check runs once in the parent, before any child spawns."""
    summary = locked_json_read(summary_path)
    cells = []
    stale_dropped = []
    for seed in cfg['seeds']:
        for v in cfg['variants']:
            key = f'{v}::seed{seed}'
            _drop_msg = None
            _drop_is_stale = False
            if key in summary and 'ppl' in summary.get(key, {}):
                old = summary[key]
                if not ppl_is_usable(old.get('ppl')):
                    _drop_msg = f"[resume] {key} holds a NON-MEASURABLE ppl ({old.get('ppl')!r}) — every statistic in this repo refuses it (`ppl_is_usable`), so counting it as 'already completed' would drop the seed from the tables silently. DROPPING it and retraining"
                elif old.get('synthesized'):
                    _drop_msg = f'[resume] {key} is a SYNTHESIZED record (reconstructed from an old aggregate.json, no weights) — DROPPING it and retraining'
                elif old.get('run_cfg') is None:
                    _drop_msg = f"[resume] {key} has no config fingerprint (predates fingerprinting) — re-running and overwriting; the STALE record is dropped before retraining, so a failed or truncated attempt cannot leave it in place looking like this attempt's result"
                    _drop_is_stale = True
                elif old.get('run_cfg') == fp:
                    print(f'[skip] {key} already completed (resume)')
                    continue
                else:
                    _drop_msg = f"[resume] {key} was run under a DIFFERENT config ({old['run_cfg']} != {fp}) — re-running and overwriting; the STALE record is dropped before retraining, so a failed or truncated attempt cannot leave it in place looking like this attempt's result"
                    _drop_is_stale = True
            if _drop_msg is not None:
                print(_drop_msg)
                summary.pop(key, None)

                def _drop_cell(s, _key=key):
                    s.pop(_key, None)
                    return s
                locked_json_update(summary_path, _drop_cell)
                if _drop_is_stale:
                    stale_dropped.append(key)
            cells.append((v, seed))
    return cells, stale_dropped


def _run_cell(cfg, v, seed, *, guard=None, ratios=None, fp=None, train_ids=None, val_batch=None, vocab=None, val_bnd=None, prebook=False, prov=None):
    """Run exactly ONE (variant, seed) training cell and merge its record into
    outdir/summary.json under the lock.

    Used by the serial loop (in-process) and by par.py children (prebook=True:
    the projected time is reserved in the shared TimeGuard before training and
    refunded after recording the actual seconds, so concurrently launching
    children still respect the shared run-time cap)."""
    d = cfg.get('d', 256)
    n_layers = cfg.get('n_layers', 6)
    n_heads = cfg.get('n_heads', 8)
    d_head = cfg.get('d_head', 32)
    outdir = cfg['outdir']
    summary_path = os.path.join(outdir, 'summary.json')
    key = f'{v}::seed{seed}'
    ratios = {} if ratios is None else ratios
    if v not in ratios:
        ratios[v] = variant_mlp_ratio(v, vocab, d=d, n_layers=n_layers, n_heads=n_heads, d_head=d_head, seq_len=cfg['seq_len'], matched=cfg.get('matched'), ref_variant=cfg.get('mlp_match_ref', 'csa_dynamic'))
    out = {'status': 'ok', 'key': key, 'variant': v, 'seed': seed}
    _run_id = new_experiment_id('cell')
    _started_at = time.strftime('%Y-%m-%dT%H:%M:%S')
    _t_start = time.time()
    out['run_id'] = _run_id
    deadline_ts = None
    booked = 0.0
    _book_tok = f'run|{key}|pid{os.getpid()}'
    if guard is not None:
        est = guard.estimate_seconds(cfg['steps'], d=d, n_layers=n_layers, seq_len=cfg['seq_len'], batch_size=cfg['batch_size'], variant=v)
        if prebook and est:
            guard.book(est, token=_book_tok)
            booked = est
        if not guard.can_start(0 if booked else est):
            if booked:
                guard.book(-booked, token=_book_tok)
                booked = 0.0
            print(f'[cap] SKIP {key}: projected {est / 3600:.2f} h would pass the configured cap (headroom {max(guard.total_hours - guard.spent_hours(), 0.0):.2f} h)')
            out['status'] = 'skipped_cap'
            log_experiment_event(outdir, {'kind': 'cell', 'run_id': _run_id, 'status': 'skipped_cap', 'key': key, 'variant': v, 'seed': int(seed), 'run_cfg': fp, 'code': code_stamp(), 'started_at': _started_at, 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'reason': 'projected runtime would pass the configured cap'})
            return out
        deadline_ts = guard.deadline_ts()
        if booked:
            deadline_ts += booked
        print(f'[cap] {key}: projected {est / 60:.0f} min, booked so far {guard.state['booked_seconds'] / 3600:.2f} h')
    t_run = time.time()
    rec = None
    try:
        try:
            _save_mode = cfg.get('save_ckpt', os.environ.get('CSA_SAVE_CKPT', 'both'))
            _ckpt_dir = cfg.get('ckpt_dir') or os.path.join(outdir, 'checkpoints')
            rec = train_variant(v, train_ids, val_batch, vocab, seed=seed, d=d, n_layers=n_layers, n_heads=n_heads, d_head=d_head, seq_len=cfg['seq_len'], batch_size=cfg['batch_size'], steps=cfg['steps'], lr=cfg['lr'], weight_decay=cfg['weight_decay'], warmup=cfg['warmup'], comp_lambda=cfg['comp_lambda'], delta_lr_mult=cfg.get('delta_lr_mult', 10.0), eval_every=cfg.get('eval_every', 0), eval_subset=cfg.get('eval_subset', 128), val_bnd=val_bnd, mlp_ratio=ratios[v], deadline_ts=deadline_ts, save_ckpt=_save_mode, ckpt_dir=_ckpt_dir, diag_every=cfg.get('diag_every'), measure_eff=cfg.get('measure_efficiency', True), measure_eff_sweep=cfg.get('measure_efficiency_sweep', False), amp_dtype=cfg.get('amp_dtype', 'auto'), eager_batch=cfg.get('eager_batch', 'auto'), early_stop_patience=cfg.get('early_stop_patience'), early_stop_min_delta=cfg.get('early_stop_min_delta', 0.0))
            _prov = prov if prov is not None else experiment_provenance(cfg, train_ids=train_ids, val_batch=val_batch, vocab=vocab)
            _hp = run_hyperparameters(cfg, seed=seed, d=d, n_layers=n_layers, n_heads=n_heads, d_head=d_head, vocab=vocab, mlp_ratio=ratios[v], data_version=_prov.get('data', {}).get('data_version'), amp_dtype=rec.get('amp_dtype', 'auto'), eager_batch=rec.get('eager_batch', 'auto'))
            _curve_ref = None
            _curve_losses = []
            _curve_ph = []
            _curve_move_ph = False
            if not rec.get('cap_truncated'):
                _curve_ref, _curve_losses, _curve_ph, _curve_move_ph = _write_cell_curve(outdir, key, rec)
            _curve_sha = (_file_sha256(os.path.join(outdir, _curve_ref))
                          if _curve_ref else None)

            def _apply(summary):
                if rec.get('cap_truncated'):
                    _prev_ok = ppl_is_usable((summary.get(key) or {}).get('ppl'))
                    print(f'[cap] {key} was truncated after {rec.get('steps_done')} steps — its partial record is DISCARDED (it holds no `ppl`); ' + ('keeping the previous record instead.' if _prev_ok else 'the key is left ABSENT, not written as a stub.') + ' Raise RUN_CAP and re-run to retry this cell.')
                    if not _prev_ok:
                        summary.pop(key, None)
                        out['truncated'] = True
                else:
                    rec['run_cfg'] = fp
                    rec['_code'] = code_stamp()
                    assert fp.endswith(f'_cs{rec['_code']}'), (fp, rec['_code'])
                    rec['provenance'] = _prov
                    rec['hyperparams'] = _hp
                    rec['data_fp'] = _prov['data'].get('train_fp')
                    rec['val_fp'] = _prov['data'].get('val_fp')
                    rec['data_version'] = _prov['data'].get('data_version')
                    rec['corpus_fp'] = _prov['data'].get('corpus_fp')
                    rec['run_id'] = _run_id
                    rec['started_at'] = _started_at
                    rec['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
                    if os.path.exists(os.path.join(outdir, 'overfit_sanity.json')):
                        rec['sanity_ref'] = 'overfit_sanity.json'
                    _rec_out = rec
                    _has_curve = bool(_curve_ref) or _curve_move_ph
                    if _has_curve:
                        _rec_out = dict(rec)
                        if _curve_ref:
                            _rec_out.pop('losses', None)
                            _rec_out.pop('delta_trace', None)
                        if _curve_move_ph:
                            _rec_out.pop('ppl_history', None)
                    if _curve_ref:
                        _cs = _finite_curve(_curve_losses)
                        _rec_out['curve_ref'] = _curve_ref
                        _rec_out['curve_sha256'] = _curve_sha
                        _rec_out['curve_n_points'] = _cs['n_points']
                        _rec_out['curve_first'] = _curve_losses[0] if _curve_losses else None
                        _rec_out['curve_last'] = _cs['last']
                        _rec_out['curve_best'] = _cs['min']
                        _rec_out['curve_n_nonfinite'] = _cs['n_nonfinite']
                        _rec_out['curve_first_nonfinite_index'] = _cs['first_nonfinite_index']
                    elif _curve_losses:
                        _rec_out['curve_n_points'] = len(_curve_losses)
                    if _curve_move_ph:
                        _rec_out['ppl_history_ref'] = _curve_ref
                        _rec_out['ppl_history_sha256'] = _curve_sha
                        _rec_out['ppl_history_n_points'] = len(_curve_ph)
                    elif _curve_ph:
                        _rec_out['ppl_history_n_points'] = len(_curve_ph)
                    summary[key] = _rec_out
                return summary
            locked_json_update(summary_path, _apply, indent=None)
            if out.get('truncated'):
                out['status'] = 'truncated'
                log_experiment_event(outdir, {'kind': 'cell', 'run_id': _run_id, 'status': 'truncated', 'key': key, 'variant': v, 'seed': int(seed), 'run_cfg': fp, 'code': code_stamp(), 'started_at': _started_at, 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'steps_done': rec.get('steps_done'), 'reason': 'hard runtime cap reached; partial record not written'})
            elif isinstance(rec, dict) and not rec.get('cap_truncated'):
                _losses = rec.get('losses') if isinstance(rec.get('losses'), list) else []
                _cs = _finite_curve(_losses)
                _ledger_losses, _losses_n, _losses_down = _ledger_curve(_losses)
                _ledger_ph, _ph_n, _ph_down = _ledger_curve_points(rec.get('ppl_history'))
                _ev = {'kind': 'cell', 'run_id': _run_id, 'status': 'ok', 'started_at': _started_at, 'finished_at': rec.get('finished_at'), 'elapsed_s': time.time() - _t_start, 'key': key, 'variant': v, 'seed': int(seed), 'run_cfg': fp, 'code': rec.get('_code') or code_stamp(), 'provenance': rec.get('provenance'), 'hyperparams': rec.get('hyperparams'), 'data_fp': rec.get('data_fp'), 'val_fp': rec.get('val_fp'), 'data_version': rec.get('data_version'), 'rng_fp': rec.get('rng_fp'), 'exec_fp': rec.get('exec_fp'), 'metrics': {'ppl': rec.get('ppl'), 'params': rec.get('params'), 'steps': rec.get('steps'), 'steps_completed': rec.get('steps_completed'), 'early_stop': rec.get('early_stop'), 'tokens_seen': rec.get('tokens_seen'), 'train_time_s': rec.get('train_time_s'), 'final_loss_smoothed': rec.get('final_loss_smoothed'), 'best_val_ppl': rec.get('best_val_ppl'), 'final_minus_best': rec.get('final_minus_best')}, 'curve_summary': {'n_points': _cs['n_points'], 'n_nonfinite': _cs['n_nonfinite'], 'first': _losses[0] if _losses else None, 'last': _cs['last'], 'best': _cs['min']}, 'curve_health': rec.get('curve_health'), 'instability_diagnosis': rec.get('instability_diagnosis'), 'opt_diag_summary': rec.get('opt_diag_summary'), 'checkpoints': rec.get('checkpoints'), 'efficiency': rec.get('efficiency'), 'losses': _ledger_losses, 'losses_n_points': _losses_n, 'losses_downsampled': _losses_down, 'ppl_history': _ledger_ph, 'ppl_history_n_points': _ph_n, 'ppl_history_downsampled': _ph_down, 'curve_ref': _curve_ref or rec.get('curve_ref'), 'sanity_ref': rec.get('sanity_ref')}
                log_experiment_event(outdir, _ev)
        except Exception as e:
            import traceback
            import sys as _sys
            _err = e
            _err_tb = traceback.format_exc()
            out['status'] = 'error'
            out['error'] = f'{_err}'

            def _apply_err(summary):
                if key in summary and ppl_is_usable((summary.get(key) or {}).get('ppl')):
                    print(f'[{key}] FAILED: {_err} — keeping the previous MEASURED record (the error record holds no `ppl` and must not replace it); the failed attempt is still booked to the guard below.')
                    print(_err_tb, file=_sys.stderr)
                else:
                    summary[key] = {'variant': v, 'seed': seed, 'error': _err_tb}
                    print(f'[{key}] FAILED: {_err}')
                return summary
            locked_json_update(summary_path, _apply_err, indent=None)
            log_experiment_event(outdir, {'kind': 'cell', 'run_id': _run_id, 'status': 'error', 'started_at': _started_at, 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'elapsed_s': time.time() - _t_start, 'key': key, 'variant': v, 'seed': int(seed), 'run_cfg': fp, 'code': code_stamp(), 'error': f'{_err}', 'traceback': _err_tb})
    finally:
        if guard is not None:
            steps_done = 0
            calib_s = None
            if isinstance(rec, dict) and rec.get('steps_done'):
                steps_done = int(rec['steps_done'])
            elif isinstance(rec, dict) and 'ppl' in rec:
                steps_done = int(rec.get('steps_completed') or cfg['steps'])
                calib_s = rec.get('train_time_s')
            guard.record_run(time.time() - t_run, steps_done, d, n_layers, cfg['seq_len'], cfg['batch_size'], calib_seconds=calib_s, variant=v)
            if booked:
                guard.book(-booked, token=_book_tok)
    gc.collect()
    return out


def run_cell_job(job):
    """Child-process entry for one `run`-kind cell (invoked by par.py).

    Re-serves the corpus from the same on-disk cache, recomputes the identical
    fingerprint, and runs the cell with pre-booked guard accounting.  The
    data-load seconds are NOT booked here: the parent already booked the
    phase's data load once, exactly like the serial path."""
    cfg = dict(job['cfg'])
    _matched = cfg.get('matched')
    if isinstance(_matched, (list, tuple)):
        cfg['matched'] = set(_matched)
    guard = TimeGuard(job['guard_cap']) if job.get('guard_cap') else None
    # Seed the child BEFORE the corpus is served, exactly like run() does in
    # the parent and like the v7/v10 child entries (run_warmup_cell_job /
    # run_probe_cell_job) already do: today's loaders consume no RNG, but a
    # future loader step that does (a sampled val slice, a DataLoader shuffle)
    # must not silently become the one unseeded construction path.
    set_seed(int(job['seed']))
    train_ids, val_batch, vocab, _, val_bnd = load_run_corpus(cfg)
    d = cfg.get('d', 256)
    n_layers = cfg.get('n_layers', 6)
    n_heads = cfg.get('n_heads', 8)
    d_head = cfg.get('d_head', 32)
    if torch.cuda.is_available():
        _pin_cuda_determinism()
    # The parent already computed the identical fingerprint/provenance over
    # this corpus; reuse them.  If an older job payload lacks the resolved
    # implementation modes, resolve them from THIS child's device and pin them
    # into the config so the fingerprint and the training record agree.
    if 'amp_dtype' not in cfg or 'eager_batch' not in cfg:
        _amp_mode, _eager_mode = _resolve_impl_modes(cfg, device=DEVICE)
        cfg['amp_dtype'] = 'fp32' if _amp_mode is None else _amp_mode
        cfg['eager_batch'] = bool(_eager_mode)
    fp = job.get('fp') or _run_fp(cfg, d, n_layers, n_heads, d_head, vocab, train_ids, val_batch, device=DEVICE)
    return _run_cell(cfg, job['v'], job['seed'], guard=guard, ratios={}, fp=fp, train_ids=train_ids, val_batch=val_batch, vocab=vocab, val_bnd=val_bnd, prebook=True, prov=job.get('prov'))


def preflight_fingerprint(cfg, *, stage, vocab=None, data_fp=None, probe_cfg=None):
    """Stable fingerprint for a standalone pre-flight artifact.

    Used when the check writes evidence before/independently of a panel
    ``run_cfg`` (m0/run_gapfill pre-flight dirs, and a synthetic probe that
    fails before the corpus is loaded).  It is deliberately not a resume key:
    it exists so the evidence can state which config/code/data/probe produced
    it."""
    blob = json.dumps({
        'config': _jsonable(cfg),
        'stage': str(stage),
        'vocab': None if vocab is None else int(vocab),
        'data_fp': data_fp,
        'probe_cfg': _jsonable(probe_cfg) if probe_cfg is not None else None,
        'train_knobs': training_knobs_fingerprint(),
        'code': code_stamp(),
    }, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return 'preflight-' + hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]

def _is_confirmatory_cfg(cfg):
    """A confirmation / scale panel defaults to strict determinism + a
    mandatory real-architecture reproducibility check.

    An explicit ``require_repro``/``confirm`` flag always wins; otherwise the
    B-tier scale geometry (d>=512, multiple seeds) is treated as confirmatory.
    Small exploratory/test panels keep the advisory path."""
    for _k in ('confirm', 'confirmation', 'require_repro'):
        if cfg.get(_k) is not None:
            return bool(cfg.get(_k))
    try:
        _d = int(cfg.get('d', 256))
        _nseeds = len(cfg.get('seeds') or [])
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(_d >= 512 and _nseeds >= 2)

def _record_failed_panel(outdir, cfg, run_cfg, prov, *, stage, error, planned_cells=None):
    """Persist a failed pre-flight as a first-class panel record.

    Evidence files (`overfit_sanity.json`, `seed_contract.json`, ...) and the
    JSONL ledger are already written before a hard stop, but without a
    manifest a failed attempt had no single file carrying the code version,
    config, seed plan and failure stage.

    An existing entry for the SAME run_cfg is never erased: a completed or
    already-incomplete panel keeps its cells, and the failure is appended to
    the ledger.  A panel that was left in ``running`` state (the process was
    killed, or a later pre-flight refused) is flipped to ``failed`` so a run
    that never finished cannot keep looking live forever.  This is the
    run-state reconciliation, not a retry: no cell measurement is changed.
    """
    try:
        mpath = os.path.join(outdir, 'run_manifest.json')
        _existing_entry = None
        if os.path.exists(mpath):
            try:
                with open(mpath, encoding='utf-8') as _mf:
                    _existing = json.load(_mf)
                if isinstance(_existing, dict):
                    _existing_entry = (_existing.get('panels') or {}).get(run_cfg)
            except Exception:
                _existing_entry = None
        _prev_status = str((_existing_entry or {}).get('status') or '').lower()
        if _prev_status == 'complete':
            # A completed panel is evidence; a later failed pre-flight with
            # the same config must not rewrite it into a failed run.
            log_experiment_event(outdir, {
                'kind': 'panel_finish', 'tracking_schema': 2,
                'status': 'failed', 'stage': str(stage), 'run_cfg': run_cfg,
                'code': code_stamp(), 'provenance': prov, 'error': f'{error}',
                'planned_cells': list(planned_cells or []),
                'note': 'existing COMPLETE run_manifest entry preserved; '
                        'this failed attempt is recorded only in the ledger',
            })
            return mpath
        # Re-read the measured cells (if any) so the failed entry still carries
        # everything that DID complete.  A corrupt summary is not a reason to
        # overwrite it; in that case the previous cells are preserved through
        # write_run_manifest's merge and the failure is still recorded.
        try:
            _summary = _read_summary_json(os.path.join(outdir, 'summary.json'))
        except Exception:
            _summary = {}
        write_run_manifest(outdir, cfg, run_cfg, prov, _summary, status='failed',
                           planned_cells=planned_cells,
                           started_at=((_existing_entry or {}).get('created_at')
                                       or time.strftime('%Y-%m-%dT%H:%M:%S')))
        log_experiment_event(outdir, {
            'kind': 'panel_finish', 'tracking_schema': 2,
            'status': 'failed', 'stage': str(stage), 'run_cfg': run_cfg,
            'code': code_stamp(), 'provenance': prov, 'error': f'{error}',
            'planned_cells': list(planned_cells or []),
            'previous_status': (_prev_status or None),
            'note': ('running/incomplete panel reconciled to failed; measured '
                     'cells were preserved' if _existing_entry else
                     'first failed attempt for this run_cfg'),
        })
        return mpath
    except Exception as _mfe:
        print(f'[track] WARNING: could not persist the failed-panel manifest ({type(_mfe).__name__}: {_mfe})')
        return None


def run(cfg=None, seeds=None, guard=None, label='', par=None):
    cfg = dict(RUN) if cfg is None else dict(cfg)
    # Freeze the scale protocol into the cfg before any probe/fingerprint so
    # every downstream consumer (loader, run_cfg, data_version, resume check)
    # reads the SAME declared sizes instead of falling back to current code
    # defaults independently.
    cfg = apply_data_protocol(cfg, allow_unknown=True)
    if seeds is not None:
        cfg['seeds'] = list(seeds)
    d = cfg.get('d', 256)
    n_layers = cfg.get('n_layers', 6)
    n_heads = cfg.get('n_heads', 8)
    d_head = cfg.get('d_head', 32)
    outdir = cfg['outdir']
    os.makedirs(outdir, exist_ok=True)
    # Normalise the panel axes BEFORE any probe, fingerprint or manifest is
    # written.  Without this a duplicate seed plans/trains the same cell twice
    # and a cfg without `seeds` trains the fallback seed while the manifest
    # records an empty plan.
    _seeds = normalize_seed_list(cfg.get('seeds'))
    _variants = normalize_variant_list(cfg.get('variants'))
    cfg['seeds'] = list(_seeds)
    cfg['variants'] = list(_variants)
    _planned_keys = [f'{v}::seed{seed}' for seed in _seeds for v in _variants]
    _first_seed, _last_seed = _seeds[0], _seeds[-1]
    _arch = {'d': int(d), 'n_layers': int(n_layers), 'n_heads': int(n_heads), 'd_head': int(d_head)}
    # Resolve the numerical implementation ONCE for this panel and persist the
    # concrete values in the config.  The resume fingerprint, the par.py job
    # payloads and the manifest then all describe the same run without any
    # later process having to infer modes from its own CUDA visibility.
    _amp_mode, _eager_mode = _resolve_impl_modes(cfg, device=DEVICE)
    cfg['amp_dtype'] = 'fp32' if _amp_mode is None else _amp_mode
    cfg['eager_batch'] = bool(_eager_mode)
    # A confirmation / scale panel runs with deterministic kernels by default
    # (an explicit CSA_DETERMINISTIC=0 still opts out) and treats the
    # same-machine reproducibility check as mandatory below.
    _confirm = _is_confirmatory_cfg(cfg)
    if _confirm:
        os.environ.setdefault('CSA_DETERMINISTIC', '1')
    # Seed the panel-level data-construction path (tokenizer fitting, cache
    # rebuilds, any future DataLoader setup) before the corpus is served.
    # Per-cell training re-seeds itself inside train_variant, so this cannot
    # change a recorded cell; it only closes an unseeded first-build path.
    set_seed(_first_seed)
    # The probe uses the panel's REAL architecture/MLP and a length that
    # exercises the block geometry rather than a fixed 64-token proxy.
    _probe_seq = probe_seq_len(cfg.get('seq_len', 64))
    _probe_steps = 200
    _probe_batch = 2

    def _panel_ratios(vocab_):
        _r = {}
        for _v in _variants:
            try:
                _r[_v] = variant_mlp_ratio(_v, int(vocab_), d=int(d), n_layers=int(n_layers), n_heads=int(n_heads), d_head=int(d_head), seq_len=int(cfg.get('seq_len', 512)), matched=cfg.get('matched'), ref_variant=cfg.get('mlp_match_ref', 'csa_dynamic'))
            except Exception:
                _r[_v] = None
        return _r

    # Synthetic pre-flight before the corpus load.  The actual-data probe below
    # is the authoritative one; this one still exercises the real model
    # geometry so a shape/gradient fault is caught even if the corpus is slow.
    _pre_vocab = int(cfg.get('vocab_size') or 1024)
    _pre_ratios = _panel_ratios(_pre_vocab)
    try:
        _sanity = list(overfit_guard(_variants, seed=_first_seed, steps=_probe_steps, seq_len=_probe_seq, batch_size=_probe_batch, vocab=_pre_vocab, arch=_arch, mlp_ratio=_pre_ratios))
    except RuntimeError as _fe:
        _prov0 = experiment_provenance(cfg)
        _reports0 = [dict(r) for _k, r in _OVERFIT_REPORTS.items() if _k[0] in set(_variants)]
        _cfg0 = _overfit_probe_config(seed=_first_seed, steps=_probe_steps, seq_len=_probe_seq, batch_size=_probe_batch, vocab=_pre_vocab, target_loss=0.05, target_acc=0.99, arch=_arch, source='synthetic', device=DEVICE)
        _doc0 = {
            'kind': 'small_batch_overfit_sanity_panel', 'stage': 'synthetic_preflight',
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'run_cfg': preflight_fingerprint(cfg, stage='synthetic_overfit', vocab=_pre_vocab, probe_cfg=_cfg0),
            'label': label or outdir, 'passed': False, 'error': f'{_fe}',
            'provenance': _prov0, 'reports': _reports0,
        }
        atomic_write_json(os.path.join(outdir, 'overfit_sanity.json'), _doc0)
        log_experiment_event(outdir, {'kind': 'preflight', 'tracking_schema': 2, 'status': 'failed', 'stage': 'synthetic_overfit', 'run_cfg': _doc0['run_cfg'], 'code': code_stamp(), 'provenance': _prov0, 'error': f'{_fe}', 'sanity': {'passed': False, 'n_reports': len(_reports0)}})
        _record_failed_panel(outdir, cfg, _doc0['run_cfg'], _prov0, stage='synthetic_overfit', error=_fe, planned_cells=_planned_keys)
        raise
    _attb_bump_epoch()
    t_data = time.time()
    train_ids, val_batch, vocab, _, val_bnd = load_run_corpus(cfg)
    # Mandatory data gate: the loaders run it too, but a driver/monkeypatched
    # corpus cannot bypass it for a real dataset.  The report is persisted with
    # the panel before any model is fitted.
    if (cfg.get('dataset') or 'wikitext') in ('wikitext', 'fineweb'):
        _da = audit_corpus_split(train_ids, val_batch, vocab,
                                 dataset=cfg.get('dataset') or 'wikitext')
        try:
            atomic_write_json(os.path.join(outdir, 'data_audit.json'), {
                'kind': 'corpus_split_data_gate',
                'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'dataset': cfg.get('dataset') or 'wikitext',
                'report': _da,
            })
        except Exception as _dae:
            print(f'[data] WARNING: could not persist data_audit.json ({type(_dae).__name__}: {_dae})')
    if guard is not None:
        guard.record_run(time.time() - t_data, 0, 0, 0, 0, 0)
    summary_path = os.path.join(outdir, 'summary.json')
    _read_summary_json(summary_path)
    if label:
        print(f'\n########## phase: {label} — {len(cfg['variants'])} variants x seeds {cfg['seeds']} x {cfg['steps']} steps ##########')
    # Prospective seed-plan note: a panel that contains an ablation arm is
    # checked BEFORE training so its seed count is not mistaken for a
    # confirmation after the fact.  This never blocks a run.
    try:
        _vs = set(cfg.get('variants') or [])
        _declared_pairs = [(_a, _b) for _contrasts in list(ABLATION_DESIGN.values()) + list(BASELINE_DESIGN.values()) for _a, _b, _n in _contrasts]
        if any(_a in _vs and _b in _vs for _a, _b in _declared_pairs):
            _sp = methodology_seed_plan(len(cfg['seeds']), variants=cfg.get('variants'))
            if _sp.get('min_seeds_for_family_significance') and not _sp.get('sufficient_for_family_rejection'):
                print('[seed-plan] this panel declares %d controlled contrasts; with n=%d seeds the family-corrected exact-test floor is above alpha, so results are DESCRIPTIVE ONLY. '
                      'A family-corrected rejection needs n>=%d (raise the seed list or narrow the pre-registered family before claiming significance).'
                      % (_sp['declared_contrasts'], _sp['n_seeds'], _sp['min_seeds_for_family_significance']))
            if _sp.get('ablation_complete') is False:
                _abl = _sp.get('ablation_completeness') or {}
                print('[seed-plan] ABLATION INCOMPLETE for this panel: missing %s. '
                      'A component with only a partial combination cannot support the claim that each module is useful; add the named full-minus-X arm (or run the full factorial).'
                      % '; '.join('%s=%s' % (_c, ','.join(_a))
                                  for _c, _a in sorted((_abl.get('missing_arms') or {}).items())))
    except Exception as _sp_err:
        print(f'[seed-plan] note unavailable ({type(_sp_err).__name__}: {_sp_err}) — continuing')
    if torch.cuda.is_available():
        _pin_cuda_determinism()
    fp = _run_fp(cfg, d, n_layers, n_heads, d_head, vocab, train_ids, val_batch, device=DEVICE)
    prov = experiment_provenance(cfg, train_ids=train_ids, val_batch=val_batch, vocab=vocab)
    ratios = _panel_ratios(vocab)
    _data_fp = _ids_fp(train_ids)

    # Explicit seed contract: same seed reproduces every RNG source the
    # pipeline can consume (Python/NumPy/Torch/CUDA/TF/DataLoader generator
    # and worker hook) and a different seed actually changes each one.  This
    # is cheaper than the 3-step training check and pinpoints WHICH stream is
    # unseeded when a reproduction fails.
    _dl_workers = int(cfg.get('dataloader_workers') or 0)
    _seed_contract = seed_contract_check(
        seed=_first_seed,
        check_dataloader_workers=_dl_workers > 0,
        dataloader_workers=max(1, _dl_workers))
    _seed_contract.update({'run_cfg': fp, 'label': label or outdir, 'provenance': prov})
    for _w in _seed_contract.get('warnings') or []:
        print(f'[determinism] WARNING: {_w}')
    atomic_write_json(os.path.join(outdir, 'seed_contract.json'), _seed_contract)
    log_experiment_event(outdir, {
        'kind': 'preflight', 'tracking_schema': 2,
        'status': ('ok' if _seed_contract.get('passed') else 'failed'),
        'stage': 'rng_seed_contract', 'run_cfg': fp,
        'code': code_stamp(), 'provenance': prov,
        'seed_contract': {k: _seed_contract.get(k) for k in ('passed', 'seed', 'checks', 'errors')},
    })
    if not _seed_contract.get('passed'):
        _record_failed_panel(outdir, cfg, fp, prov, stage='rng_seed_contract',
                             error=_seed_contract.get('diagnosis'), planned_cells=_planned_keys)
        raise RuntimeError(
            'RNG seed contract check FAILED '
            f"({_seed_contract.get('diagnosis')}) — do NOT start training; "
            'at least one RNG stream is unseeded or unstable on this machine. '
            'See seed_contract.json.')

    _data_contract = batch_contract_check(
        train_ids, cfg['seq_len'], cfg.get('batch_size', 12), DEVICE,
        seed=_first_seed, vocab=vocab)
    _data_contract.update({'run_cfg': fp, 'label': label or outdir, 'provenance': prov})
    atomic_write_json(os.path.join(outdir, 'data_contract.json'), _data_contract)
    log_experiment_event(outdir, {
        'kind': 'preflight', 'tracking_schema': 2,
        'status': ('ok' if _data_contract.get('passed') else 'failed'),
        'stage': 'batch_iterator_contract', 'run_cfg': fp,
        'code': code_stamp(), 'provenance': prov,
        'data_contract': {k: _data_contract.get(k) for k in
                          ('passed', 'diagnosis', 'seq_len', 'batch_size', 'seed', 'n_batches')},
    })
    if not _data_contract.get('passed'):
        _record_failed_panel(outdir, cfg, fp, prov, stage='batch_iterator_contract',
                             error=_data_contract.get('diagnosis'), planned_cells=_planned_keys)
        raise RuntimeError(
            'batch iterator contract check FAILED '
            f"({_data_contract.get('diagnosis')}) — do NOT start a full run; "
            'the actual training loader is not producing shifted, in-vocabulary, '
            'repeatable batches. See data_contract.json.')

    def _data_probe_cfg(v):
        return _overfit_probe_config(seed=_first_seed, steps=_probe_steps, seq_len=_probe_seq, batch_size=_probe_batch, vocab=vocab, target_loss=0.05, target_acc=0.99, arch=_arch, mlp_ratio=ratios.get(v), ignore_index=None, source='data', data_fp=_data_fp, device=DEVICE)

    # The real data-path check: the synthetic pass above cannot catch a broken
    # real token stream / vocabulary mapping.  It is a superset of the model
    # plumbing check, so a pass here is the stronger evidence; both reports
    # are persisted with the panel.
    _data_rep = []
    try:
        _data_rep = list(overfit_guard(_variants, seed=_first_seed, steps=_probe_steps, seq_len=_probe_seq, batch_size=_probe_batch, vocab=vocab, train_ids=train_ids, arch=_arch, mlp_ratio=ratios))
        _sanity += _data_rep
    except RuntimeError as _fe:
        _reports = [dict(r) for _k, r in _OVERFIT_REPORTS.items() if _k[0] in set(_variants)]
        _doc = {
            'kind': 'small_batch_overfit_sanity_panel', 'stage': 'data_preflight',
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'run_cfg': fp, 'label': label or outdir, 'passed': False,
            'error': f'{_fe}', 'provenance': prov, 'reports': _reports,
        }
        atomic_write_json(os.path.join(outdir, 'overfit_sanity.json'), _doc)
        log_experiment_event(outdir, {'kind': 'preflight', 'tracking_schema': 2, 'status': 'failed', 'stage': 'data_overfit', 'run_cfg': fp, 'code': code_stamp(), 'provenance': prov, 'error': f'{_fe}', 'sanity': {'passed': False, 'n_reports': len(_reports)}})
        _record_failed_panel(outdir, cfg, fp, prov, stage='data_overfit', error=_fe, planned_cells=_planned_keys)
        raise
    if not _data_rep:
        # The guard is process-cached (correctly, so repeated phases do not
        # retrain the probe).  Recover THIS exact probe configuration's
        # evidence from the cache so a fresh outdir still gets a persisted
        # sanity record; a looser probe must never certify this panel.
        _sem_by_v = {v: _overfit_probe_semantics(_data_probe_cfg(v)) for v in _variants}
        _sanity = []
        for (_v, _src, *_rest), _r in _OVERFIT_REPORTS.items():
            if _v not in set(_variants):
                continue
            if _r.get('guard_probe_semantics') != _sem_by_v.get(_v):
                continue
            if _src in ('data', 'batch'):
                # Full data key: the evidence must name THIS corpus/panel.
                if len(_rest) < 2 or _rest[1] != _data_fp:
                    continue
            elif _src != 'synthetic':
                continue
            _sanity.append(dict(_r))
    _sanity_doc = None
    if _sanity:
        _sanity_doc = {
            'kind': 'small_batch_overfit_sanity_panel',
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'run_cfg': fp,
            'label': label or outdir,
            'passed': all(bool(r.get('passed')) for r in _sanity),
            'provenance': prov,
            'reports': _sanity,
        }
        atomic_write_json(os.path.join(outdir, 'overfit_sanity.json'), _sanity_doc)
    # The hand-rolled probe above certifies model/loss/label plumbing.  Re-run
    # one fixed window through the ACTUAL `train_variant` loop so a broken
    # optimizer/scheduler/loss assembly cannot pass a separately written loop.
    _prod_doc = None
    if _variants and os.environ.get('CSA_SKIP_OVERFIT', '').strip().lower() not in ('1', 'true', 'yes', 'on'):
        try:
            # The production trainer is shared by every arm; a dense sentinel
            # tests optimizer/scheduler/loss-assembly wiring without coupling
            # the gate to a stochastic task or boundary-policy arm.
            _prod = production_loop_sanity_check(
                'full', seed=_first_seed, train_ids=train_ids,
                vocab=vocab, device=DEVICE)
            _prod_error = None
        except Exception as _fe:
            _prod = {'kind': 'production_loop_overfit_sanity', 'variant': 'full',
                     'passed': False, 'diagnosis': f'{type(_fe).__name__}: {_fe}'}
            _prod_error = f'{_fe}'
        _prod_doc = {
            'kind': 'production_loop_overfit_sanity_panel',
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'run_cfg': fp, 'label': label or outdir,
            'passed': bool(_prod.get('passed')),
            'provenance': prov, 'reports': [dict(_prod)],
        }
        if _prod_error is not None:
            _prod_doc['error'] = _prod_error
        atomic_write_json(os.path.join(outdir, 'production_sanity.json'), _prod_doc)
        log_experiment_event(outdir, {
            'kind': 'preflight', 'tracking_schema': 2,
            'status': ('ok' if _prod_doc['passed'] else 'failed'),
            'stage': 'production_loop_overfit', 'run_cfg': fp,
            'code': code_stamp(), 'provenance': prov,
            'production_sanity': {'passed': _prod_doc['passed'],
                                  'ref': 'production_sanity.json',
                                  'variant': 'full'},
        })
        if _sanity_doc is not None:
            _sanity_doc['production_loop'] = _prod_doc
            if not _prod_doc['passed']:
                _sanity_doc['passed'] = False
            atomic_write_json(os.path.join(outdir, 'overfit_sanity.json'), _sanity_doc)
        if not _prod_doc['passed']:
            _record_failed_panel(outdir, cfg, fp, prov, stage='production_loop_overfit',
                                 error=_prod.get('diagnosis'), planned_cells=_planned_keys)
            raise RuntimeError(
                'production training-loop overfit check FAILED '
                f"({_prod.get('diagnosis')}) — do NOT start a full run; the "
                'actual train_variant loop is not applying updates to one fixed '
                'batch. See production_sanity.json.')
    # Same-machine seed check on the REAL architecture/data, first and last
    # panel seed.  Confirmation/scale panels require it to pass; exploratory
    # panels record the failure but stay advisory.
    _repro_seeds = list(dict.fromkeys([_first_seed, _last_seed]))
    _env_strict = os.environ.get('CSA_REQUIRE_REPRO')
    if _env_strict is None:
        _strict_repro = bool(_confirm)
    else:
        _strict_repro = _env_strict.strip().lower() in ('1', 'true', 'yes', 'on')
    try:
        _repro = list(reproducibility_guard(_variants, seed=_first_seed, steps=3, arch=_arch, train_ids=train_ids, vocab=vocab, mlp_ratio=ratios, seq_len=_probe_seq, batch_size=_probe_batch, seeds=_repro_seeds, strict=_strict_repro))
    except RuntimeError as _fe:
        _reports = [dict(r) for _k, r in _REPRO_REPORTS.items() if _k[0] in set(_variants)]
        _doc = {
            'kind': 'duplicate_run_reproducibility_panel', 'stage': 'reproducibility_preflight',
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'run_cfg': fp, 'label': label or outdir, 'passed': False,
            'error': f'{_fe}', 'provenance': prov, 'reports': _reports,
        }
        atomic_write_json(os.path.join(outdir, 'reproducibility.json'), _doc)
        log_experiment_event(outdir, {'kind': 'preflight', 'tracking_schema': 2, 'status': 'failed', 'stage': 'reproducibility', 'run_cfg': fp, 'code': code_stamp(), 'provenance': prov, 'error': f'{_fe}', 'reproducibility': {'passed': False, 'n_reports': len(_reports)}})
        _record_failed_panel(outdir, cfg, fp, prov, stage='reproducibility', error=_fe, planned_cells=_planned_keys)
        raise
    if not _repro:
        _bases = set()
        for v in _variants:
            for sd in _repro_seeds:
                _bases.add(_repro_guard_base(seed=sd, steps=3, tol=1e-06, arch=_arch, vocab=vocab, mlp_ratio=ratios.get(v), seq_len=_probe_seq, batch_size=_probe_batch, train_ids=train_ids, data_fp=_data_fp))
        _repro = [dict(r) for k, r in _REPRO_REPORTS.items() if k[0] in set(_variants) and k[1] in _bases]
    _repro_doc = None
    if _repro:
        _repro_doc = {
            'kind': 'duplicate_run_reproducibility_panel',
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'run_cfg': fp,
            'label': label or outdir,
            'passed': all(bool(r.get('passed')) for r in _repro),
            'provenance': prov,
            'reports': _repro,
        }
        atomic_write_json(os.path.join(outdir, 'reproducibility.json'), _repro_doc)
    cells, stale_dropped = _plan_run(cfg, summary_path, fp)
    # Plan the order over ALL declared cells, not only the pending ones, so a
    # resume cannot reshuffle a partially finished panel.  The same cells are
    # trained with the same seeds; only the calendar order changes.
    _all_order_cells = [(v, seed) for seed in cfg['seeds'] for v in cfg['variants']]
    _run_order_plan = build_run_order_plan(
        _all_order_cells, block_key=lambda c: int(c[1]), seed=_first_seed,
        block_name='seed')
    write_run_order(outdir, _run_order_plan)
    # Pre-register the failure/replacement rule before the first cell can fail.
    # The first write binds; a later, different plan is kept as a visible
    # conflict instead of silently rewriting the precommitment (the disposition
    # audit will report the resulting unplanned/missing cells).
    try:
        _dispo_plan = build_run_disposition_plan(
            _planned_keys,
            policy=(cfg.get('disposition_policy')
                    if isinstance(cfg.get('disposition_policy'), dict) else None),
            run_cfg=fp)
        _dispo = write_run_disposition(outdir, _dispo_plan, keep_existing=True)
    except Exception as _de:
        _dispo = {'written': False, 'conflict': True,
                  'reason': f'{type(_de).__name__}: {_de}',
                  'content_sha256': None}
    if _dispo.get('conflict'):
        print('[disposition] WARNING: the existing run_disposition.json does '
              f'not match this panel ({_dispo.get("reason")}); the original '
              'precommitment is kept and the deviation will be visible to '
              '`python audit_gates.py disposition`')
    _run_rank = {lab: i for i, lab in enumerate(_run_order_plan['planned_cells'])}
    cells.sort(key=lambda c: _run_rank.get(f'{c[0]}::seed{c[1]}', len(_run_rank)))
    _panel_started = time.strftime('%Y-%m-%dT%H:%M:%S')
    # Crash-safe tracking: publish the panel's identity (code/data versions,
    # hyperparameters, planned cells) BEFORE the first cell can be
    # killed.  The final write below flips the status to complete and lists
    # every planned cell that produced no measurement.
    write_run_manifest(outdir, cfg, fp, prov, _read_summary_json(summary_path), sanity=_sanity_doc, reproducibility=_repro_doc, data_contract=_data_contract, seed_contract=_seed_contract, status='running', planned_cells=_planned_keys, started_at=_panel_started)
    log_experiment_event(outdir, {
        'kind': 'panel_start', 'tracking_schema': 2, 'label': label or outdir,
        'run_cfg': fp, 'code': code_stamp(), 'started_at': _panel_started,
        'config': _jsonable(cfg), 'seeds': list(cfg.get('seeds') or []),
        'variants': list(cfg.get('variants') or []),
        'n_cells_planned': len(_planned_keys), 'n_cells_to_run': len(cells),
        'n_cells_resumed': max(0, len(_planned_keys) - len(cells)),
        'provenance': prov,
        'data_contract': ({'passed': bool(_data_contract.get('passed')), 'diagnosis': _data_contract.get('diagnosis'), 'ref': 'data_contract.json'} if isinstance(_data_contract, dict) else None),
        'sanity': ({'passed': all(bool(r.get('passed')) for r in _sanity), 'n_reports': len(_sanity), 'ref': 'overfit_sanity.json'} if _sanity else None),
        'reproducibility': ({'passed': all(bool(r.get('passed')) for r in _repro), 'n_reports': len(_repro), 'ref': 'reproducibility.json'} if _repro else None),
        'seed_contract': ({'passed': bool(_seed_contract.get('passed')), 'ref': 'seed_contract.json'} if isinstance(_seed_contract, dict) else None),
        'disposition_plan_sha256': _dispo.get('content_sha256'),
        'disposition_plan_revision': _dispo.get('revision'),
        'disposition_plan_conflict': bool(_dispo.get('conflict')),
    })
    dropped_truncations = []
    ratios = {}
    _ran_parallel = False
    if cells:
        import par as _par
        _par_ctx = par if isinstance(par, dict) else _par.resolve_ctx(par, label=label or outdir)
        if _par_ctx and _par.ctx_active(_par_ctx, len(cells)):
            _caller = sys._getframe(1).f_globals.get('__name__')
            # a '__main__' frame (e.g. run_gapfill.py driving L.run directly)
            # cannot be imported by that name in the child, and plain
            # exp_lib matches that caller's patch state exactly.
            module = _par_ctx.get('module') or (_caller if _caller and _caller != '__main__' else 'exp_lib')
            jobs = [dict(kind='run', module=module, label=label, cfg=cfg, v=v, seed=seed, guard_cap=(guard.cap_spec() if guard is not None else None), ckpt_flag=bool(_par_ctx.get('ckpt_flag')), ckpt_install=bool(_par_ctx.get('ckpt_install')), fp=fp, prov=prov) for v, seed in cells]
            # Children re-serve the corpus from the shared on-disk cache; the
            # parent can release its reference (and the loader memo entry) once
            # the jobs are dispatched.
            for _memo in (_WT103_MEMO, _FWEB_MEMO):
                for _mk, _mv in list(_memo.items()):
                    if isinstance(_mv, tuple) and _mv and _mv[0] is train_ids:
                        _memo.pop(_mk, None)
            del train_ids, val_batch, val_bnd
            gc.collect()
            results = _par.run_jobs(jobs, _par_ctx, phase_label=label or outdir)
            for r, (v, seed) in zip(results, cells):
                if not r:
                    continue
                if r.get('truncated'):
                    dropped_truncations.append(r.get('key') or f'{v}::seed{seed}')
            _ran_parallel = True
    if not _ran_parallel:
        for v, seed in cells:
            res = _run_cell(cfg, v, seed, guard=guard, ratios=ratios, fp=fp, train_ids=train_ids, val_batch=val_batch, vocab=vocab, val_bnd=val_bnd, prov=prov)
            if res.get('truncated'):
                dropped_truncations.append(res['key'])
    summary = _read_summary_json(summary_path)
    if dropped_truncations:
        print(f'\n[cap] {len(dropped_truncations)} cell(s) were retrained, hit the cap deadline, and produced NO measurement — they are ABSENT from {summary_path} and from every table above:')
        for _k in dropped_truncations:
            print(f'    {_k}')
        print('[cap] raise RUN_CAP and re-run to fill these cells in.')
    if stale_dropped:
        _still = [k for k in stale_dropped if k not in summary]
        print(f'\n[resume] {len(stale_dropped)} cell(s) held a record from a DIFFERENT config and were re-run; {len(_still)} of them produced NO measurement this pass and are now ABSENT from {summary_path} (the stale record was removed rather than kept, so no number measured by older code can be reported under the current config):')
        for _k in _still:
            print(f'    {_k}')
        if len(_still) != len(stale_dropped):
            print(f'    ({len(stale_dropped) - len(_still)} of them succeeded and carry a fresh record.)')
    _measured = {k for k in _planned_keys if isinstance(summary.get(k), dict) and ppl_is_usable((summary.get(k) or {}).get('ppl')) and not (summary.get(k) or {}).get('synthesized')}
    _missing = [k for k in _planned_keys if k not in _measured]
    _trace_bad = []
    for _k in _measured:
        _rec = summary.get(_k) or {}
        if _rec.get('run_cfg') not in (None, fp):
            _trace_bad.append(_k)
            continue
        for _ref_key in ('curve_ref', 'ppl_history_ref'):
            _ref = _rec.get(_ref_key)
            if _ref and '#' not in str(_ref) and not os.path.exists(os.path.join(outdir, str(_ref))):
                _trace_bad.append(_k)
                break
    if _trace_bad:
        print(f'\n[trace] {len(_trace_bad)} measured cell(s) reference a missing curve artifact or a foreign run_cfg: {_trace_bad[:10]} — the panel is marked incomplete.')
    # A finished process is not a finished panel.  Stamping a run with missing
    # measurements (or a broken trace reference) as `complete` is exactly how
    # a partially traced experiment gets quoted as if every planned cell
    # existed.
    _final_status = 'complete' if (not _missing and not _trace_bad) else 'incomplete'
    write_run_manifest(outdir, cfg, fp, prov, summary, sanity=_sanity_doc, reproducibility=_repro_doc, data_contract=_data_contract, seed_contract=_seed_contract, status=_final_status, planned_cells=_planned_keys, started_at=_panel_started)
    # Independent read-back of the trace that was just written.  A run whose
    # manifest/cells/curve digests do not agree is not "complete" even if the
    # process exited normally: that is how a partially traced panel would
    # otherwise be quoted as if every planned cell carried evidence.
    _tracking = verify_run_tracking(outdir, fp)
    atomic_write_json(os.path.join(outdir, 'tracking_verification.json'), _tracking)
    if not _tracking.get('passed'):
        print(f'[track] tracking verification FAILED: {_tracking.get("diagnosis")} — marking the panel incomplete')
        _final_status = 'incomplete'
    write_run_manifest(outdir, cfg, fp, prov, summary, sanity=_sanity_doc, reproducibility=_repro_doc, data_contract=_data_contract, seed_contract=_seed_contract, status=_final_status, planned_cells=_planned_keys, started_at=_panel_started, tracking_verification=_tracking)
    _agg = _finish_panel(outdir, summary)
    try:
        with open(os.path.join(outdir, 'aggregation_exclusions.json'),
                  encoding='utf-8') as _xf:
            _excl_doc = json.load(_xf)
    except Exception:
        _excl_doc = {}
    log_experiment_event(outdir, {
        'kind': 'panel_finish', 'tracking_schema': 2, 'label': label or outdir,
        'status': _final_status,
        'run_cfg': fp, 'code': code_stamp(), 'started_at': _panel_started,
        'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'provenance': prov, 'config': _jsonable(cfg),
        'seeds': list(cfg.get('seeds') or []),
        'variants': list(cfg.get('variants') or []),
        'data_version': prov.get('data', {}).get('data_version'),
        'n_cells_planned': len(_planned_keys), 'n_cells_to_run': len(cells),
        'n_cells_measured': len(_measured),
        'n_cells_missing': len(_missing), 'missing_cells': _missing[:20],
        'n_trace_bad': len(_trace_bad), 'trace_bad_cells': _trace_bad[:20],
        'n_truncated': len(dropped_truncations), 'n_stale_dropped': len(stale_dropped),
        'tracking_verification': {'passed': bool(_tracking.get('passed')), 'ref': 'tracking_verification.json', 'n_problems': len(_tracking.get('problems') or [])},
        'aggregate_ref': 'aggregate.json', 'summary_ref': 'summary.json',
        'aggregation_exclusions_ref': 'aggregation_exclusions.json',
        'n_records_quarantined': int(_excl_doc.get('n_excluded') or 0),
        'aggregation_exclusion_reasons': _excl_doc.get('reason_counts') or {},
        'aggregate': {str(v): {kk: e.get(kk) for kk in ('n', 'n_seeds', 'ppl_mean', 'ppl_std', 'params') if kk in e} for v, e in (_agg or {}).items() if isinstance(e, dict)},
    })
    return (summary, None)
