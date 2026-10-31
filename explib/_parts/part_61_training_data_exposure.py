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

import decimal as _tx_decimal
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

# A caller-supplied ``max_exact_draws`` is a policy threshold, not permission
# to allocate an unbounded array.  The exact path materialises one start per
# draw; the hard cap is therefore also an absolute memory guard for future
# scale (20M int64 starts is already ~160 MB before ``unique``/``counts``).
TRAINING_EXPOSURE_HARD_MAX_DRAWS = 20_000_000
_TX_INT64_MAX = 2 ** 63 - 1


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


def _tx_seed_int(name, value):
    """One seed in the same domain ``batch_iter`` accepts.

    The assembled module provides ``_seed_value`` (NumPy/PyTorch common seed
    range).  The fallback keeps this fragment importable on its own without
    silently widening the domain.
    """
    try:
        return _seed_value(value, where=name)  # noqa: F821
    except NameError:
        return _tx_nonneg_int(name, value, maximum=2 ** 63 - 1)


def _tx_ratio(num, den):
    """``num/den`` as a float, or ``None`` when it is not representable.

    Python's int/int true division scales operands first, so it does not
    overflow merely because one side is a huge exact count.  A quotient that
    genuinely exceeds float64 returns ``None`` instead of raising (or leaking
    ``inf`` into a strict-JSON block).
    """
    try:
        n, d = int(num), int(den)
    except (TypeError, ValueError, OverflowError):
        return None
    if d <= 0 or n < 0:
        return None
    try:
        return n / d
    except OverflowError:
        return None


def _tx_scaled_int(count, fraction, *, rounding):
    """``count * fraction`` rounded to an exact int, without float overflow.

    The ordinary small-count path preserves the historical round/ceil result.
    Huge exact counts are multiplied in ``Decimal`` so a 10**400-token corpus
    yields an integer coverage estimate instead of raising in ``float()``.
    """
    count = int(count)
    try:
        fraction = float(fraction)
    except (TypeError, ValueError, OverflowError):
        return None
    if count <= 0 or not _tx_math.isfinite(fraction) or fraction <= 0.0:
        return 0
    if fraction >= 1.0:
        return count
    if count <= 2 ** 53:
        product = count * fraction
        if rounding == _tx_decimal.ROUND_CEILING:
            return int(_tx_math.ceil(product))
        return int(round(product))
    try:
        with _tx_decimal.localcontext() as ctx:
            ctx.prec = 60
            product = _tx_decimal.Decimal(count) * _tx_decimal.Decimal(fraction)
            return int(product.to_integral_value(rounding=rounding))
    except (_tx_decimal.DecimalException, OverflowError, ValueError):
        return None


def _tx_ratio_fraction(num, den):
    """``1 - exp(-num/den)`` for non-negative exact counts."""
    try:
        n, d = int(num), int(den)
    except (TypeError, ValueError, OverflowError):
        return None
    if d <= 0 or n < 0:
        return None
    try:
        ratio = n / d
    except OverflowError:
        return 1.0
    if ratio <= 0.0:
        return 0.0
    # exp(-745) is below the smallest positive float64; for a larger ratio the
    # coverage is exactly one at this precision.
    if ratio > 745.0:
        return 1.0
    return -_tx_math.expm1(-ratio)


def _tx_ceil_neglog_scaled(q, num, den):
    """``ceil(-log1p(-q) * num / den)`` without float overflow."""
    try:
        return int(_tx_math.ceil(-_tx_math.log1p(-float(q))
                                * float(num) / float(den)))
    except (OverflowError, ValueError, ZeroDivisionError):
        try:
            with _tx_decimal.localcontext() as ctx:
                ctx.prec = 60
                factor = -(_tx_decimal.Decimal(1)
                           - _tx_decimal.Decimal(float(q))).ln()
                value = (factor * _tx_decimal.Decimal(int(num))
                         / _tx_decimal.Decimal(int(den)))
                return int(value.to_integral_value(
                    rounding=_tx_decimal.ROUND_CEILING))
        except (_tx_decimal.DecimalException, OverflowError, ValueError,
                ZeroDivisionError):
            return None


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
        seeds = [_tx_seed_int('data_seeds[]', v) for v in vals]
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
        return [_tx_seed_int('data_seed_for_rank', fn(seed, rank))
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
    batch_size = int(batch_size)
    steps = int(steps)
    total = int(len(data_seeds)) * steps * batch_size
    if total <= 0:
        return _tx_np.empty(0, dtype=idx_dtype)
    # Allocate the final array once and fill it in draw order.  Buffering one
    # array per step (the obvious list+concatenate form) makes the declared
    # draw cap a poor memory bound: with batch_size=1, 20M allowed draws would
    # put 20M tiny ndarrays in a Python list before the final 160MB array.
    # The RNG call sequence (one ``integers`` per step, same high/dtype) and
    # the concatenated order are unchanged.
    out = _tx_np.empty(total, dtype=idx_dtype)
    offset = 0
    for seed in data_seeds:
        rng = _tx_np.random.default_rng(int(seed))
        for _ in range(steps):
            out[offset:offset + batch_size] = rng.integers(
                0, high, size=batch_size, dtype=idx_dtype)
            offset += batch_size
    return out


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

    # A caller-controlled threshold cannot disable the allocation guard, and
    # an int64 replay cannot represent a corpus beyond 2**63.  Both are
    # explicit estimate fallbacks: the block stays total and strict JSON
    # instead of attempting an unbounded materialisation or raising deep in
    # NumPy.
    fallback_reasons = []
    if int(draws) > int(max_exact_draws):
        fallback_reasons.append(
            f'draws={draws} exceeds max_exact_draws={int(max_exact_draws)}')
    if int(draws) > TRAINING_EXPOSURE_HARD_MAX_DRAWS:
        fallback_reasons.append(
            f'draws={draws} exceeds the absolute exact-replay hard cap '
            f'{TRAINING_EXPOSURE_HARD_MAX_DRAWS}')
    if int(n_tokens) > _TX_INT64_MAX:
        fallback_reasons.append(
            f'n_tokens={n_tokens} exceeds the int64 window-start domain '
            f'({_TX_INT64_MAX})')
    exact = not fallback_reasons
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
        if int(window_positions) == 1:
            # Only one start exists, so every draw is that window; the exact
            # answer is known even when the general Poisson fallback would
            # understate a one-window (n_tokens=seq_len+1) corpus.
            unique_starts = 1
            unique_tokens = int(target_positions)
        else:
            frac_token = _tx_ratio_fraction(
                int(draws) * int(seq_len), int(target_positions))
            unique_tokens = _tx_scaled_int(
                int(target_positions), frac_token,
                rounding=_tx_decimal.ROUND_HALF_EVEN)
            if unique_tokens is None:
                unique_tokens = 0
            unique_tokens = max(0, min(int(unique_tokens), int(target_positions)))
            frac_start = _tx_ratio_fraction(int(draws), int(window_positions))
            unique_starts = _tx_scaled_int(
                int(window_positions), frac_start,
                rounding=_tx_decimal.ROUND_HALF_EVEN)
            if unique_starts is None:
                unique_starts = 0
            unique_starts = max(0, min(int(unique_starts), int(window_positions), int(draws)))
        # Rounding must not create a block where more distinct targets were
        # visited than target positions were drawn.
        unique_tokens = min(int(unique_tokens), int(processed_tokens))
        max_start_multiplicity = None
        coverage_quality = 'estimated'
        estimate_reason = (
            '; '.join(fallback_reasons)
            + '; values use the Poisson independent-window approximation')

    start_coverage = _tx_ratio(unique_starts, window_positions)
    token_coverage = _tx_ratio(unique_tokens, target_positions)
    unique_epochs = _tx_ratio(unique_tokens, target_positions)
    effective_epochs = _tx_ratio(processed_tokens, target_positions)
    reuse_factor = (_tx_ratio(processed_tokens, unique_tokens)
                    if unique_tokens else None)
    _duplicate = _tx_ratio(unique_tokens, processed_tokens)
    duplicate_token_fraction = (1.0 - _duplicate
                                if _duplicate is not None else 0.0)
    duplicate_token_fraction = max(0.0, min(1.0, duplicate_token_fraction))
    _collision = _tx_ratio(unique_starts, draws)
    start_collision_fraction = (1.0 - _collision
                                if _collision is not None else 0.0)
    start_collision_fraction = max(0.0, min(1.0, start_collision_fraction))
    expected_draws_per_start = _tx_ratio(draws, window_positions)

    warnings = []
    if unique_epochs is not None and unique_epochs < 1.0:
        warnings.append(
            'fewer than one full corpus pass in unique prediction targets; '
            'tokens_seen counts repeated draws, not distinct data')
    if token_coverage is not None and token_coverage < 0.5:
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
        'start_coverage': (0.0 if start_coverage is None
                           else float(start_coverage)),
        'token_coverage': (0.0 if token_coverage is None
                           else float(token_coverage)),
        'unique_epochs': (0.0 if unique_epochs is None
                          else float(unique_epochs)),
        'effective_epochs': (None if effective_epochs is None
                             else float(effective_epochs)),
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
    draws_needed = _tx_ceil_neglog_scaled(q, target_positions, window)
    if draws_needed is None:
        # The inverse is outside float64 and cannot even be expressed with
        # the exact Decimal path (an astronomically large corpus).  Report the
        # design as not achievable rather than raising or fabricating a plan.
        return {
            'kind': 'training_coverage_budget',
            'schema': TRAINING_EXPOSURE_POLICY['schema'],
            'status': 'not_estimable',
            'reason': ('the required draw budget is not representable for this '
                       'corpus size and target coverage'),
            'target_coverage': float(q),
            'n_tokens': int(n_tokens),
            'seq_len': int(seq_len),
            'batch_size': int(batch_size),
            'world_size': int(world_size),
            'target_positions': int(target_positions),
            'expected_draws': None,
            'expected_steps': None,
            'expected_processed_tokens': None,
            'expected_unique_tokens': None,
            'expected_unique_fraction': None,
            'achievable': False,
            'method': 'poisson_independent_windows',
            'exact': False,
            'policy': TRAINING_EXPOSURE_POLICY['version'],
            'boundary': (
                'design estimate under with-replacement sampling; the budget '
                'is beyond the representable range, so no plan is claimed'),
        }
    draws_needed = max(1, int(draws_needed))
    per_step = int(batch_size) * int(world_size)
    # Integer ceiling division: never route an exact step count through a
    # float, which would overflow for a very large planned panel.
    steps_needed = max(1, (int(draws_needed) + per_step - 1) // per_step)
    expected_processed = int(steps_needed) * per_step * int(seq_len)
    _frac = _tx_ratio_fraction(
        int(steps_needed) * per_step * int(window), int(target_positions))
    expected_unique = _tx_scaled_int(
        int(target_positions), _frac,
        rounding=_tx_decimal.ROUND_CEILING)
    if expected_unique is None:
        expected_unique = int(target_positions)
    expected_unique = max(0, min(int(expected_unique), int(target_positions)))
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
        'expected_unique_fraction': (0.0 if not target_positions
                                     else _tx_ratio(expected_unique,
                                                    target_positions)),
        'achievable': bool(steps_needed <= int(max_steps)),
        'method': 'poisson_independent_windows',
        'exact': False,
        'policy': TRAINING_EXPOSURE_POLICY['version'],
        'boundary': (
            'design estimate under with-replacement sampling; use '
            'training_window_exposure on the actual seed for the measured '
            'coverage; edge effects and window overlap make this approximate'),
    }
