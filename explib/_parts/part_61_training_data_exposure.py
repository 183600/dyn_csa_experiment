# Exact training-data exposure for the with-replacement window sampler.
#
# ``batch_iter`` draws ``batch_size`` independent uniform start positions per
# step, with replacement, and then yields the causal window at each start.  A
# run's ``tokens_seen`` therefore counts *draws*, not distinct data: the same
# window can be drawn repeatedly and adjacent windows overlap by seq_len-1
# tokens.  The historical record carried a Poisson estimate of unique tokens
# (``tokens_seen_unique_estimate``) but no replayable measurement and no way
# to plan a budget by coverage rather than by draw count.
#
# ``training_window_exposure`` replays the exact seeded start stream and
# computes the union of the prediction-target token intervals.  Because every
# interval has the same length seq_len, the union length is
# ``seq_len + sum(min(seq_len, diff(sorted_unique_starts)))`` -- no per-token
# materialisation is needed even for a multi-billion-token corpus.  When the
# draw count exceeds the exact budget the block is explicitly marked as an
# analytic estimate (exact=False), never as a measurement.
#
# ``training_coverage_budget`` inverts the same model: given a desired
# unique-token coverage it returns the number of steps/draws needed under the
# current with-replacement sampler, so a panel can be budgeted by data
# exposure rather than by nominal token draws.
#
# Both are descriptive data-processing readouts.  They do not change the
# sampler, tokens, splits, training, checkpoints, PPL, p-values, Holm families,
# or any frozen verdict.  This fragment intentionally has no module docstring:
# the split loader executes fragments in the package globals, so one would
# overwrite exp_lib.__doc__.

import hashlib as _tx_hashlib
import math as _tx_math

import numpy as _tx_np

TRAINING_EXPOSURE_POLICY = {
    'version': 'training-data-exposure-1',
    'schema': 'training_window_coverage_v1',
    'draw_rule': (
        'per step, batch_size independent uniform start positions in '
        '[0, n_tokens-seq_len-1] are drawn with replacement; a DDP rank r '
        'uses data_seed_for_rank(seed, r), matching batch_iter'),
    'coverage_unit': 'prediction_target_token',
    'window_rule': (
        'a draw at start s predicts target positions [s+1, s+1+seq_len); '
        'all windows have length seq_len, so the union is computed from the '
        'sorted unique starts'),
    'exact_algorithm': (
        'unique_starts=sorted(unique(draws)); '
        'unique_tokens=seq_len+sum(min(seq_len, diff(unique_starts)))'),
    'estimate_algorithm': (
        'Poisson independent-window approximation used only when '
        'draws > max_exact_draws; the block is marked exact=False and '
        'coverage_quality="estimated"'),
    'statuses': ('ok', 'not_estimable'),
    'claim_boundary': (
        'coverage describes the observed seeded draw stream for this corpus '
        'slice, not a probability sample of the upstream corpus; a resumed '
        'record covers its own steps only, not the union over earlier '
        'segments; tokens_seen and unique_tokens are different quantities and '
        'must not be quoted interchangeably'),
}

TRAINING_EXPOSURE_STATUSES = ('ok', 'not_estimable')


def _tx_exact_pos_int(name, value, *, minimum=1):
    if isinstance(value, (bool, _tx_np.bool_)):
        raise ValueError(f'{name} must be an integer, not a boolean')
    if not isinstance(value, (int, _tx_np.integer)):
        raise ValueError(f'{name} must be an integer, got {type(value).__name__}')
    value = int(value)
    if value < int(minimum):
        raise ValueError(f'{name} must be >= {int(minimum)}, got {value}')
    return value


def _tx_nonneg_int(name, value, *, maximum=None):
    if isinstance(value, (bool, _tx_np.bool_)):
        raise ValueError(f'{name} must be an integer, not a boolean')
    if not isinstance(value, (int, _tx_np.integer)):
        raise ValueError(f'{name} must be an integer, got {type(value).__name__}')
    value = int(value)
    if value < 0:
        raise ValueError(f'{name} must be >= 0, got {value}')
    if maximum is not None and value > int(maximum):
        raise ValueError(f'{name} must be <= {int(maximum)}, got {value}')
    return value


def _tx_finite_fraction(name, value):
    if isinstance(value, (bool, _tx_np.bool_, str, bytes, bytearray)):
        raise ValueError(f'{name} must be a number, not {type(value).__name__}')
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name} must be a finite number, got {value!r}') from None
    if not _tx_math.isfinite(out):
        raise ValueError(f'{name} must be finite, got {value!r}')
    return out


def _tx_rank_seeds(seed, world_size, data_seeds=None):
    """Per-rank data seeds, exactly as the training loop derives them.

    ``batch_iter`` is called once per rank with
    ``data_seed_for_rank(seed, rank)``.  Replaying the union requires those
    concrete seeds.  The canonical derivation is used when available; the
    explicit ``data_seeds`` argument is preferred when a record already stores
    it, so an audit does not depend on a later helper remaining unchanged.
    """
    try:
        seed = _seed_value(seed)  # noqa: F821 - shared seed-domain validator
    except NameError:
        seed = _tx_nonneg_int('seed', seed, maximum=2 ** 63 - 1)
    world_size = _tx_exact_pos_int('world_size', world_size)
    if data_seeds is not None:
        try:
            vals = list(data_seeds)
        except TypeError:
            raise ValueError('data_seeds must be an iterable of integers') from None
        if len(vals) != world_size:
            raise ValueError(
                f'data_seeds has {len(vals)} entries but world_size={world_size}')
        seeds = [_tx_nonneg_int('data_seeds[]', v, maximum=2 ** 63 - 1)
                 for v in vals]
        # Rank 0 is the run seed by construction; keeping the two echoes in
        # sync stops a hand edit from changing ``inputs.seed`` while leaving
        # the replayed stream bound to a different rank-0 seed.
        if seeds and int(seeds[0]) != int(seed):
            raise ValueError(
                'data_seeds[0] must equal the run seed for rank 0, got '
                f'{seeds[0]} vs {seed}')
        return seeds
    if world_size == 1:
        return [seed]
    try:
        fn = data_seed_for_rank  # noqa: F821 - provided by the loaded module
    except NameError:
        fn = None
    if fn is not None:
        return [_tx_nonneg_int('data_seed_for_rank', fn(seed, rank),
                               maximum=2 ** 63 - 1)
                for rank in range(world_size)]
    # The canonical helper is always present in the assembled module; this
    # fallback keeps the primitive usable when the fragment is imported alone.
    out = []
    for rank in range(world_size):
        digest = _tx_hashlib.sha256(f'{int(seed)}:{int(rank)}'.encode('utf-8')).digest()
        out.append(int.from_bytes(digest[:8], 'big') % (2 ** 63))
    return out


def _tx_draw_starts(n_windows_minus_one, steps, batch_size, data_seeds,
                    idx_dtype):
    """Replay the exact starts consumed by ``batch_iter``.

    The loop mirrors ``batch_iter``: one ``Generator.integers`` call per step,
    with the same high bound and integer dtype.  Integer draws for int32 and
    int64 are not interchangeable in NumPy, so the dtype rule is copied too.
    """
    high = int(n_windows_minus_one) + 1
    chunks = []
    for seed in data_seeds:
        rng = _tx_np.random.default_rng(int(seed))
        parts = []
        for _ in range(int(steps)):
            parts.append(rng.integers(0, high, size=int(batch_size),
                                      dtype=idx_dtype))
        if parts:
            chunks.append(_tx_np.concatenate(parts))
    if not chunks:
        return _tx_np.empty(0, dtype=idx_dtype)
    return (chunks[0] if len(chunks) == 1 else _tx_np.concatenate(chunks))


def training_window_exposure(n_tokens, seq_len, batch_size, steps, seed=0, *,
                             world_size=1, data_seeds=None,
                             max_exact_draws=4_000_000):
    """Measure (or explicitly estimate) the unique data reached by a run.

    Parameters mirror the training stream: ``n_tokens`` is ``len(train_ids)``,
    ``seq_len``/``batch_size``/``steps`` are the loop geometry, and ``seed``
    is the run seed (per-rank streams are derived exactly like
    ``data_seed_for_rank``).  ``steps`` should be the completed step count for
    an observed run, or the planned count for a design readout.  When
    ``data_seeds`` is supplied it must contain one entry per rank and its first
    entry must equal ``seed`` (rank 0 is the run seed); this keeps the two
    echoes in a stored record from diverging.

    The return value is strict-JSON-friendly and includes the full input echo
    under ``inputs`` so an audit can replay it without reading any data file.
    """
    n_tokens = _tx_exact_pos_int('n_tokens', n_tokens)
    seq_len = _tx_exact_pos_int('seq_len', seq_len)
    batch_size = _tx_exact_pos_int('batch_size', batch_size)
    steps = _tx_exact_pos_int('steps', steps)
    world_size = _tx_exact_pos_int('world_size', world_size)
    max_exact_draws = _tx_nonneg_int('max_exact_draws', max_exact_draws)
    if n_tokens <= seq_len:
        raise ValueError(
            'training_window_exposure needs n_tokens >= seq_len+1 to form a '
            f'causal window, got n_tokens={n_tokens} seq_len={seq_len}')
    seeds = _tx_rank_seeds(seed, world_size, data_seeds=data_seeds)
    n_last_start = n_tokens - seq_len - 1
    window_positions = n_last_start + 1          # starts 0 .. n_last_start
    target_positions = n_tokens - 1              # positions 1 .. n_tokens-1
    draws = int(steps) * int(batch_size) * int(world_size)
    processed_tokens = draws * int(seq_len)
    idx_dtype = (_tx_np.int32
                 if (n_last_start + seq_len + 1) < 2 ** 31
                 else _tx_np.int64)

    exact = int(draws) <= int(max_exact_draws)
    if exact:
        starts = _tx_draw_starts(n_last_start, steps, batch_size, seeds,
                                 idx_dtype).astype(_tx_np.int64, copy=False)
        unique_starts_arr, counts = _tx_np.unique(starts, return_counts=True)
        unique_starts = int(unique_starts_arr.size)
        if unique_starts:
            steps_between = _tx_np.diff(unique_starts_arr)
            unique_tokens = int(seq_len) + int(
                _tx_np.minimum(int(seq_len), steps_between).sum(dtype=_tx_np.int64))
        else:  # unreachable for draws >= 1, kept for a total function
            unique_tokens = 0
        unique_tokens = min(int(unique_tokens), int(target_positions))
        max_start_multiplicity = int(counts.max()) if counts.size else 0
        coverage_quality = 'measured'
        estimate_reason = None
    else:
        # The exact union over millions of starts is still cheap, but an
        # unbounded array is not: mark the fallback explicitly rather than
        # silently returning a different estimand.
        p_token = min(float(seq_len) / float(target_positions), 1.0)
        frac_token = (1.0 if p_token >= 1.0
                      else 1.0 - _tx_math.exp(draws * _tx_math.log1p(-p_token)))
        unique_tokens = int(round(float(target_positions) * frac_token))
        unique_tokens = max(0, min(unique_tokens, int(target_positions)))
        p_start = min(1.0 / float(window_positions), 1.0)
        frac_start = (1.0 if p_start >= 1.0
                      else 1.0 - _tx_math.exp(draws * _tx_math.log1p(-p_start)))
        unique_starts = int(round(float(window_positions) * frac_start))
        unique_starts = max(0, min(unique_starts, int(window_positions), int(draws)))
        # Rounding must not create a block where more distinct targets were
        # visited than target positions were drawn.
        unique_tokens = min(int(unique_tokens), int(processed_tokens))
        max_start_multiplicity = None
        coverage_quality = 'estimated'
        estimate_reason = (
            f'draws={draws} exceeds max_exact_draws={int(max_exact_draws)}; '
            'values use the Poisson independent-window approximation')

    start_coverage = (float(unique_starts) / float(window_positions)
                      if window_positions else 0.0)
    token_coverage = (float(unique_tokens) / float(target_positions)
                      if target_positions else 0.0)
    unique_epochs = (float(unique_tokens) / float(target_positions)
                     if target_positions else 0.0)
    effective_epochs = (float(processed_tokens) / float(target_positions)
                        if target_positions else 0.0)
    reuse_factor = (float(processed_tokens) / float(unique_tokens)
                    if unique_tokens else None)
    duplicate_token_fraction = (1.0 - float(unique_tokens) / float(processed_tokens)
                                if processed_tokens else 0.0)
    duplicate_token_fraction = max(0.0, min(1.0, duplicate_token_fraction))
    start_collision_fraction = (1.0 - float(unique_starts) / float(draws)
                                if draws else 0.0)
    start_collision_fraction = max(0.0, min(1.0, start_collision_fraction))
    expected_draws_per_start = (float(draws) / float(window_positions)
                                if window_positions else None)

    warnings = []
    if unique_epochs < 1.0:
        warnings.append(
            'fewer than one full corpus pass in unique prediction targets; '
            'tokens_seen counts repeated draws, not distinct data')
    if token_coverage < 0.5:
        warnings.append(
            'less than half of the predictable token positions were visited '
            'at least once; the run is a partial-corpus exposure, not an epoch')
    if reuse_factor is not None and reuse_factor > 2.0:
        warnings.append(
            'on average each visited token position was drawn more than twice; '
            'effective data exposure is much smaller than tokens_seen')
    if not exact:
        warnings.append(estimate_reason)

    return {
        'kind': 'training_data_exposure',
        'schema': TRAINING_EXPOSURE_POLICY['schema'],
        'status': 'ok',
        'inputs': {
            'n_tokens': int(n_tokens),
            'seq_len': int(seq_len),
            'batch_size': int(batch_size),
            'steps': int(steps),
            'seed': int(seed),
            'world_size': int(world_size),
            'data_seeds': [int(s) for s in seeds],
            'max_exact_draws': int(max_exact_draws),
        },
        'draws': int(draws),
        'processed_tokens': int(processed_tokens),
        'window_positions': int(window_positions),
        'target_positions': int(target_positions),
        'unique_starts': int(unique_starts),
        'unique_tokens': int(unique_tokens),
        'start_coverage': float(start_coverage),
        'token_coverage': float(token_coverage),
        'unique_epochs': float(unique_epochs),
        'effective_epochs': float(effective_epochs),
        'reuse_factor': (None if reuse_factor is None else float(reuse_factor)),
        'duplicate_token_fraction': float(duplicate_token_fraction),
        'start_collision_fraction': float(start_collision_fraction),
        'expected_draws_per_start': (
            None if expected_draws_per_start is None
            else float(expected_draws_per_start)),
        'max_start_multiplicity': max_start_multiplicity,
        'exact': bool(exact),
        'coverage_quality': coverage_quality,
        'warnings': warnings,
        'policy': TRAINING_EXPOSURE_POLICY['version'],
        'boundary': TRAINING_EXPOSURE_POLICY['claim_boundary'],
    }


def training_coverage_budget(n_tokens, seq_len, batch_size, *, target_coverage,
                             world_size=1, max_steps=10_000_000):
    """Steps needed to reach a target unique-token coverage.

    This is the inverse of :func:`training_window_exposure` under the same
    with-replacement model.  It is a *design* readout: the realised coverage of
    a particular seed is measured by ``training_window_exposure``, while this
    function answers "how large should the draw budget be before the run has a
    chance to cover q of the predictable token positions?".

    ``target_coverage`` is a fraction in ``(0, 1)``.  The returned
    ``expected_steps`` is rounded up, and ``achievable`` is false when it
    exceeds ``max_steps``.  The Poisson model ignores finite-corpus edge
    effects; the measured block, not this plan, is the evidence.
    """
    n_tokens = _tx_exact_pos_int('n_tokens', n_tokens)
    seq_len = _tx_exact_pos_int('seq_len', seq_len)
    batch_size = _tx_exact_pos_int('batch_size', batch_size)
    world_size = _tx_exact_pos_int('world_size', world_size)
    max_steps = _tx_exact_pos_int('max_steps', max_steps)
    q = _tx_finite_fraction('target_coverage', target_coverage)
    if not (0.0 < q < 1.0):
        raise ValueError(f'target_coverage must be in (0, 1), got {q!r}')
    if n_tokens <= seq_len:
        raise ValueError(
            'training_coverage_budget needs n_tokens >= seq_len+1 to form a '
            f'causal window, got n_tokens={n_tokens} seq_len={seq_len}')
    target_positions = n_tokens - 1
    window = min(int(seq_len), int(target_positions))
    draws_needed = int(_tx_math.ceil(
        -_tx_math.log1p(-q) * float(target_positions) / float(window)))
    draws_needed = max(1, draws_needed)
    steps_needed = int(_tx_math.ceil(
        float(draws_needed) / float(int(batch_size) * int(world_size))))
    expected_processed = int(steps_needed) * int(batch_size) * int(world_size) * int(seq_len)
    expected_unique = int(_tx_math.ceil(float(target_positions) * (
        1.0 - _tx_math.exp(-float(steps_needed * batch_size * world_size)
                           * float(window) / float(target_positions)))))
    expected_unique = max(0, min(expected_unique, int(target_positions)))
    return {
        'kind': 'training_coverage_budget',
        'schema': TRAINING_EXPOSURE_POLICY['schema'],
        'status': 'ok',
        'target_coverage': float(q),
        'n_tokens': int(n_tokens),
        'seq_len': int(seq_len),
        'batch_size': int(batch_size),
        'world_size': int(world_size),
        'target_positions': int(target_positions),
        'expected_draws': int(steps_needed * batch_size * world_size),
        'expected_steps': int(steps_needed),
        'expected_processed_tokens': int(expected_processed),
        'expected_unique_tokens': int(expected_unique),
        'expected_unique_fraction': (float(expected_unique) / float(target_positions)),
        'achievable': bool(steps_needed <= int(max_steps)),
        'method': 'poisson_independent_windows',
        'exact': False,
        'policy': TRAINING_EXPOSURE_POLICY['version'],
        'boundary': (
            'design estimate under with-replacement sampling; use '
            'training_window_exposure on the actual seed for the measured '
            'coverage; edge effects and window overlap make this approximate'),
    }
