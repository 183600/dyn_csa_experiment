"""Budget-truncation feasibility of a pre-registered cell panel.

The frozen design already fixes the arm x seed grid, the minimum number of
paired seeds (``min_seeds_for_confirmation``), the complete-coverage rule
(``require_complete_coverage``) and the retry policy for failed cells
(``run_disposition``).  Each of those pieces is audited.  What is *not*
audited is whether the campaign can actually finish the cells it registered:

* ``preparation_clearance`` checks that allocation, power, rehearsal and the
  held-out lock are in place, but it never relates the available compute/time
  budget to the number of *cell attempts* the design needs;
* the disposition policy allows retries, but a retry is only usable if budget
  remains **after** the last planned cell; a campaign whose budget exactly
  covers the planned grid cannot repair a single failed cell;
* under ``require_complete_coverage=true`` one missing cell in one block
  breaks every comparison's full-pair set, so the one-shot held-out read
  yields *no* confirmatory verdict even though ``min_seeds_for_confirmation``
  is 8.  The declared resolution floor is therefore not the operational
  floor, and a budget planned against the wrong floor is short by design.

This fragment adds a zero-GPU, no-split **budget-truncation feasibility**
analysis.  Given the registered execution order (or the canonical complete
block order when no plan is stored), the frozen comparison family, the retry
cap and an optional declared failure model, it reports:

* a deterministic **truncation frontier**: at every prefix of the execution
  order, how many blocks are complete, how many paired seeds each declared
  comparison has, and whether the minimum-n / complete-coverage rule is met;
* the **operational floor**: the shortest prefix that can still answer the
  pre-registered family, versus the declared ``min_seeds_for_confirmation``;
* the **completion-risk requirement**: for a declared per-attempt retryable /
  non-replaceable failure model, the smallest number of cell attempts ``B``
  for which ``P(the required prefix completes within B attempts)`` reaches
  ``1 - risk_level``, computed exactly from the truncated-geometric attempt
  law (no Monte Carlo, no fabricated number);
* a verdict on a declared ``cell_attempt`` budget: ``fail`` when the budget
  cannot complete the required prefix or misses the declared completion-risk
  target, ``warnings`` when the retry reserve is unquantified, and
  ``not_verifiable`` when no budget was declared.

It adds no run, reads no split, changes no arm, seed, endpoint, p-value,
Holm slot or verdict.  The budget unit is one training-cell attempt; a
wall-clock/GPU-hour budget is that count multiplied by the measured
per-attempt cost, which is an operational conversion, not a design decision.
"""

import collections
import hashlib
import json
import math
import numbers
import os
import time

import numpy as np


BUDGET_TRUNCATION_KIND = 'budget_truncation'
BUDGET_TRUNCATION_SCHEMA = 1
BUDGET_TRUNCATION_POLICY = {
    'version': 1,
    'default_risk_level': 0.05,
    'max_frontier_points': 4096,
    'max_attempts_sanity': 100,
    'max_exact_dp_states': 5000000,
    # The state cap alone bounds only the width of the DP table, not the
    # work: the convolution visits every cell x state x attempt.  A panel
    # just under the state cap could otherwise spend hours in Python before
    # the size guard fires.  Bound the actual inner-loop count as well.
    'max_exact_dp_operations': 10000000,
    'budget_unit': 'cell_attempt',
    'role': (
        'pre-run feasibility of the registered cell panel: relate the '
        'declared cell-attempt budget and the disposition retry cap to the '
        'shortest execution prefix that can still produce the frozen '
        'confirmatory family, and report the completion probability at a '
        'declared risk level.'),
    'rule': (
        'a declared budget fails when it cannot complete the required prefix '
        'once, or when the exact probability of completing that prefix '
        'within the budget is below 1 - risk_level under the declared '
        'failure model.  An undeclared budget is not_verifiable; an '
        'undeclared failure model leaves the retry reserve unquantified and '
        'is a warning, never a green pass.'),
    'boundary': (
        'a feasibility verdict is not a measurement, a power calculation, a '
        'replication or evidence about a model.  It does not reserve compute '
        'or enforce a scheduler.  A green result only says that the declared '
        'budget is consistent with the registered panel under the declared '
        'attempt-failure model; a red one means the campaign, as planned, '
        'cannot be expected to deliver the pre-registered analysis.'),
}


def _bt_is_bool(value):
    """True for Python and NumPy booleans (a flag is not a number/count)."""
    return isinstance(value, (bool, np.bool_))


def _bt_text(value):
    if value is None or _bt_is_bool(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _bt_int(value, *, minimum=None):
    if _bt_is_bool(value):
        return None
    if isinstance(value, numbers.Integral):
        out = int(value)
    elif isinstance(value, (float, np.floating)):
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(number) or not number.is_integer():
            return None
        out = int(number)
    else:
        return None
    if minimum is not None and out < minimum:
        return None
    return out


def _bt_number(value, *, minimum=None, maximum=None):
    if value is None or _bt_is_bool(value):
        return None
    if isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(out):
        return None
    if minimum is not None and out < minimum:
        return None
    if maximum is not None and out > maximum:
        return None
    return out


def _bt_dict(value):
    """A JSON-object view that never raises on a malformed artifact."""
    return value if isinstance(value, dict) else {}


def _bt_message_list(value):
    """A string list for problems/warnings, even if the artifact was edited.

    ``list(5)``, ``list(True)`` and a tuple of dicts used to abort the replay
    validator/audit gate instead of reporting the malformed artifact.
    """
    if not isinstance(value, (list, tuple)):
        return []
    return [item if isinstance(item, str) else str(item) for item in value]


def _bt_declared_list(values, label, problems):
    """Normalise a declared arm/seed list without accepting ambiguous shapes.

    ``None`` means "not declared" and is returned unchanged so the caller can
    derive the list from the cells.  A mapping/string/set is a declaration
    bug, not an iterable of names: iterating a string would silently turn
    ``"abc"`` into three arms.  Duplicate tokens are reported because they
    would double-count a planned block while the JSON keeps looking valid.
    """
    if values is None:
        return None
    if not isinstance(values, (list, tuple)):
        problems.append(f'{label} must be a list of names/seeds, got '
                        f'{type(values).__name__}')
        return []
    out = []
    seen = set()
    for index, value in enumerate(values):
        if _bt_is_bool(value):
            problems.append(
                f'{label}[{index}]={value!r} is not a string/integer '
                'identifier')
            continue
        if isinstance(value, numbers.Integral):
            token = str(int(value))
        elif isinstance(value, (float, np.floating)):
            parsed = _bt_int(value, minimum=0)
            if parsed is None:
                problems.append(
                    f'{label}[{index}]={value!r} is not a non-negative '
                    'integer identifier')
                continue
            token = str(parsed)
        elif isinstance(value, str):
            token = value.strip()
        else:
            problems.append(
                f'{label}[{index}]={value!r} is not a string/integer '
                'identifier')
            continue
        if not token:
            problems.append(f'{label}[{index}] is an empty identifier')
            continue
        if token in seen:
            problems.append(f'{label} contains duplicate identifier '
                            f'{token!r}')
            continue
        seen.add(token)
        out.append(token)
    if not out:
        problems.append(f'{label} must declare at least one identifier')
    return out


# Content identities share the one canonicalisation contract in part_00.
_bt_canonical = _canonical_json
_bt_sha_obj = _sha256_json


def _bt_sha256_text(value):
    text = _bt_text(value)
    if text is None:
        return None
    text = text.lower()
    if len(text) != 64 or any(ch not in '0123456789abcdef' for ch in text):
        return None
    return text


def _bt_manifest_canonical_sha(manifest):
    """Hash the declarations, not the loader's transient ``_path`` stamp."""
    if not isinstance(manifest, dict):
        return None
    body = {k: v for k, v in manifest.items()
            if k not in ('_path', 'manifest_sha256')}
    try:
        return _bt_sha_obj(body)
    except (TypeError, ValueError, RecursionError):
        return None


def _bt_parse_label(label):
    """Split an ``arm::seedN`` / ``arm::sN`` run-order label, or return None."""
    text = _bt_text(label)
    if text is None:
        return None, None
    for sep in ('::seed', '::s'):
        if sep in text:
            arm, _, block = text.rpartition(sep)
            if arm and block:
                return arm, block
    return None, None


def _bt_normalise_cells(cells):
    """Return ``[{label, variant, block}]`` plus named problems/warnings.

    Accepts the three shapes that occur in practice: normalized cell dicts
    (used on replay), ``(label, variant, block)`` triples, and a run-order
    plan's ``planned_cells`` + ``cell_variants``/``cell_blocks`` maps.  A cell
    whose identity cannot be recovered is a problem, not an inferred guess.
    """
    problems = []
    warnings = []
    if isinstance(cells, dict):
        labels = cells.get('planned_cells')
        if not isinstance(labels, (list, tuple)) or not labels:
            return [], ['plan.planned_cells must be a non-empty list'], []
        variants = cells.get('cell_variants')
        blocks = cells.get('cell_blocks')
        variants = variants if isinstance(variants, dict) else {}
        blocks = blocks if isinstance(blocks, dict) else {}
        raw = []
        for lab in labels:
            label = str(lab)
            variant = variants.get(label)
            block = blocks.get(label)
            if variant is None or block is None:
                parsed_v, parsed_b = _bt_parse_label(label)
                if variant is None:
                    variant = parsed_v
                if block is None:
                    block = parsed_b
            if variant is None or block is None:
                problems.append(
                    f'cell {label!r} has no variant/block identity; the '
                    'truncation frontier cannot attribute it')
                continue
            raw.append({'label': label, 'variant': str(variant),
                        'block': str(block)})
        labels = [c['label'] for c in raw]
        if len(set(labels)) != len(labels):
            dupes = sorted(label for label, count in
                           collections.Counter(labels).items() if count > 1)
            problems.append(f'cell labels are not unique: {dupes[:8]}')
        return raw, problems, warnings
    if not isinstance(cells, (list, tuple)):
        return [], [f'cells must be a list or a run-order plan, got '
                    f'{type(cells).__name__}'], []
    out = []
    for i, cell in enumerate(cells):
        if isinstance(cell, dict):
            label = _bt_text(cell.get('label')) or f'cell[{i}]'
            variant = _bt_text(cell.get('variant'))
            block = _bt_text(cell.get('block'))
        elif isinstance(cell, (list, tuple)) and len(cell) >= 3:
            label = _bt_text(cell[0]) or f'cell[{i}]'
            variant = _bt_text(cell[1])
            block = _bt_text(cell[2])
        else:
            problems.append(f'cells[{i}] is not a cell object/triple')
            continue
        if variant is None or block is None:
            problems.append(f'cells[{i}] ({label!r}) lacks a variant/block')
            continue
        out.append({'label': label, 'variant': variant, 'block': block})
    labels = [c['label'] for c in out]
    if len(set(labels)) != len(labels):
        dupes = sorted(label for label, count in
                       collections.Counter(labels).items() if count > 1)
        problems.append(f'cell labels are not unique: {dupes[:8]}')
    return out, problems, warnings


def _bt_normalise_comparisons(comparisons):
    out = []
    problems = []
    if not isinstance(comparisons, (list, tuple)) or not comparisons:
        return [], ['comparisons must be a non-empty list']
    seen = set()
    for i, comp in enumerate(comparisons):
        if not isinstance(comp, dict):
            problems.append(f'comparisons[{i}] is not an object')
            continue
        name = _bt_text(comp.get('name'))
        a = _bt_text(comp.get('a'))
        b = _bt_text(comp.get('b'))
        if name is None:
            name = f'{a} - {b}' if a and b else None
        if name is None or a is None or b is None:
            problems.append(f'comparisons[{i}] must name name/a/b')
            continue
        if name in seen:
            problems.append(f'comparison name {name!r} is duplicated')
            continue
        if a == b:
            problems.append(f'comparison {name!r} compares arm {a!r} with '
                            'itself')
            continue
        seen.add(name)
        out.append({'name': name, 'a': a, 'b': b})
    return out, problems


def truncation_frontier(cells, comparisons, *, min_n, complete_coverage,
                        max_points=None, required_blocks=None):
    """Deterministic per-prefix completeness of the registered cell panel.

    For every prefix of the execution order this records the number of
    *complete* blocks (blocks holding every arm a comparison needs), the
    paired-seed count of each comparison, and whether the declared
    ``min_n`` / complete-coverage rule is satisfied.  Nothing here depends
    on a failure model: it is the deterministic map of "what would we have
    if the campaign stopped after k cells".

    ``required_blocks`` is the declared seed grid (when the caller has one).
    Without it the block set is inferred from the cells, which silently makes
    a plan that forgot a frozen seed look complete.  With it, only a prefix
    holding every declared block can satisfy complete coverage, and a missing
    declared block keeps ``first_complete_cells`` at ``None``.
    """
    policy = BUDGET_TRUNCATION_POLICY
    max_points_problem = None
    if max_points is None:
        max_points = int(policy['max_frontier_points'])
    else:
        parsed = _bt_int(max_points, minimum=1)
        if parsed is None:
            max_points_problem = (
                f'max_points={max_points!r} must be an integer >= 1')
            max_points = int(policy['max_frontier_points'])
        else:
            max_points = parsed
    norm_cells, cell_problems, cell_warnings = _bt_normalise_cells(cells)
    comps, comp_problems = _bt_normalise_comparisons(comparisons)
    required = None
    required_problems = []
    if required_blocks is not None:
        required = _bt_declared_list(required_blocks, 'required_blocks',
                                     required_problems)
        if not required:
            required = None
    initial_problems = (list(cell_problems) + list(comp_problems)
                        + list(required_problems))
    if max_points_problem is not None:
        initial_problems.append(max_points_problem)
    out = {
        'kind': 'budget_truncation_frontier',
        'status': 'ok',
        'problems': initial_problems,
        'warnings': list(cell_warnings),
        'n_cells': len(norm_cells),
        'n_seeds': 0,
        'n_arms': 0,
        'points': [],
        'sampled': False,
        'first_min_n_cells': None,
        'first_complete_cells': None,
        'min_n': _bt_int(min_n, minimum=2),
        'complete_coverage': bool(complete_coverage),
    }
    if out['problems']:
        out['status'] = 'fail'
        return out
    blocks = []
    block_seen = set()
    for cell in norm_cells:
        if cell['block'] not in block_seen:
            block_seen.add(cell['block'])
            blocks.append(cell['block'])
    arms_needed = sorted({c['a'] for c in comps} | {c['b'] for c in comps})
    required_set = set(required) if required is not None else None
    seed_blocks = required if required is not None else blocks
    n_cells = len(norm_cells)
    n_seeds = len(seed_blocks)
    out['n_seeds'] = n_seeds
    observed_arm_set = {c['variant'] for c in norm_cells}
    out['n_arms'] = len(observed_arm_set)
    if out['min_n'] is None or out['min_n'] > n_seeds:
        out['problems'].append(
            f'min_n={min_n!r} is not an integer in [2, {n_seeds}]')
        out['status'] = 'fail'
        return out
    # A cell that carries an arm no comparison uses still consumes budget and
    # still belongs to a block; it simply never contributes a pair.  Record
    # that explicitly instead of silently dropping it from the counts.
    for arm in arms_needed:
        if arm not in observed_arm_set:
            out['problems'].append(
                f'comparison arm {arm!r} never appears in the cell panel')
    if out['problems']:
        out['status'] = 'fail'
        return out

    block_arms = {}
    block_pair = {}
    block_complete = set()
    arm_counts = collections.Counter()
    pairs = {comp['name']: 0 for comp in comps}
    complete_seeds = 0
    sample_every = max(1, int(math.ceil(n_cells / float(max_points))))
    # Select the rows once, before the scan.  Block boundaries are the most
    # informative rows, but a panel with more blocks than the point cap must
    # still respect the cap; downsampling keeps the thresholds exact because
    # they are tracked on every prefix below.
    emit = set()
    if n_cells:
        emit.update(range(sample_every, n_cells + 1, sample_every))
        emit.add(n_cells)
        for index, cell in enumerate(norm_cells, start=1):
            if index == n_cells or norm_cells[index]['block'] != cell['block']:
                emit.add(index)
        if len(emit) > max_points:
            ordered = sorted(emit)
            if max_points <= 1:
                emit = {n_cells}
            else:
                emit = {
                    ordered[round(slot * (len(ordered) - 1)
                                  / (max_points - 1))]
                    for slot in range(max_points)}
    out['sampled'] = len(emit) < n_cells
    last_block = None
    for index, cell in enumerate(norm_cells, start=1):
        arms = block_arms.setdefault(cell['block'], set())
        newly = cell['variant'] not in arms
        arms.add(cell['variant'])
        arm_counts[cell['variant']] += 1
        last_block = cell['block']
        in_seed_grid = (required_set is None
                        or cell['block'] in required_set)
        if newly and in_seed_grid:
            # Incremental pair accounting: a comparison gains a block exactly
            # when the arm that completes its pair arrives.  Re-scanning every
            # block on every cell is O(N * blocks * comparisons) and would
            # make a large arm grid or catalogue panel impractical.
            seen_pairs = block_pair.setdefault(cell['block'], set())
            for comp in comps:
                if (comp['name'] not in seen_pairs
                        and comp['a'] in arms and comp['b'] in arms):
                    seen_pairs.add(comp['name'])
                    pairs[comp['name']] += 1
            if (cell['block'] not in block_complete
                    and all(arm in arms for arm in arms_needed)):
                block_complete.add(cell['block'])
                complete_seeds += 1
        min_n_met = all(v >= out['min_n'] for v in pairs.values())
        complete_met = all(v >= n_seeds for v in pairs.values())
        # Thresholds are tracked on every prefix, whether or not the row is
        # emitted: sampling must not move the reported operational floor.
        if min_n_met and out['first_min_n_cells'] is None:
            out['first_min_n_cells'] = index
        if complete_met and out['first_complete_cells'] is None:
            out['first_complete_cells'] = index
        if index in emit:
            counts = {arm: int(arm_counts.get(arm, 0))
                      for arm in sorted(set(arm_counts) | set(arms_needed))}
            out['points'].append({
                'cells': index,
                'block': last_block,
                'complete_seeds': int(complete_seeds),
                'arm_counts': counts,
                'imbalance': (max(counts.values()) - min(counts.values())
                              if counts else 0),
                'pairs': dict(pairs),
                'min_n_met': bool(min_n_met),
                'complete_coverage_met': bool(complete_met),
            })
    return out


def truncation_frontier_point(cells, comparisons, k, *, min_n, n_seeds=None,
                              arms_needed=None, required_blocks=None):
    """Compute one frontier row at prefix ``k`` (used for exact thresholds)."""
    norm_cells, cell_problems, _warnings = _bt_normalise_cells(cells)
    comps, comp_problems = _bt_normalise_comparisons(comparisons)
    if cell_problems or comp_problems:
        raise ValueError('; '.join(cell_problems + comp_problems))
    raw_min_n = min_n
    min_n = _bt_int(min_n, minimum=2)
    if min_n is None:
        raise ValueError(f'min_n={raw_min_n!r} must be an integer >= 2')
    raw_k = k
    k = _bt_int(k, minimum=1)
    if k is None or k > len(norm_cells):
        raise ValueError(f'prefix k={raw_k!r} is outside [1, {len(norm_cells)}]')
    required_set = None
    if required_blocks is not None:
        required_problems = []
        required = _bt_declared_list(required_blocks, 'required_blocks',
                                     required_problems)
        if required_problems:
            raise ValueError('; '.join(required_problems))
        required_set = set(required)
        if n_seeds is None:
            n_seeds = len(required)
        else:
            parsed_seeds = _bt_int(n_seeds, minimum=1)
            if parsed_seeds != len(required):
                raise ValueError(
                    f'n_seeds={n_seeds!r} disagrees with the '
                    f'{len(required)} required block(s)')
            n_seeds = parsed_seeds
    if n_seeds is None:
        n_seeds = len({c['block'] for c in norm_cells})
    else:
        n_seeds = _bt_int(n_seeds, minimum=1)
        if n_seeds is None:
            raise ValueError('n_seeds must be an integer >= 1 or None')
    if arms_needed is None:
        arms_needed = sorted({c['a'] for c in comps} | {c['b'] for c in comps})
    elif not isinstance(arms_needed, (list, tuple)):
        raise ValueError('arms_needed must be a list/tuple or None')
    else:
        arms_needed = [str(arm) for arm in arms_needed]
    block_arms = {}
    arm_counts = collections.Counter()
    for cell in norm_cells[:k]:
        block_arms.setdefault(cell['block'], set()).add(cell['variant'])
        arm_counts[cell['variant']] += 1

    def _counts(block):
        return required_set is None or block in required_set

    pairs = {}
    for comp in comps:
        pairs[comp['name']] = sum(
            1 for block, arms in block_arms.items()
            if _counts(block) and comp['a'] in arms and comp['b'] in arms)
    complete_seeds = sum(1 for block, arms in block_arms.items()
                         if _counts(block)
                         and all(arm in arms for arm in arms_needed))
    counts = {arm: int(arm_counts.get(arm, 0))
              for arm in sorted(set(arm_counts) | set(arms_needed))}
    return {
        'cells': k,
        'block': norm_cells[k - 1]['block'],
        'complete_seeds': int(complete_seeds),
        'arm_counts': counts,
        'imbalance': (max(counts.values()) - min(counts.values())
                      if counts else 0),
        'pairs': pairs,
        'min_n_met': bool(all(v >= min_n for v in pairs.values())),
        'complete_coverage_met': bool(all(v >= n_seeds for v in pairs.values())),
    }


def _bt_cell_success_pmf(max_attempts, retryable, fatal):
    """Distribution of the attempt on which a cell succeeds.

    ``retryable`` failures consume an attempt and allow the next one;
    ``fatal`` failures and retry exhaustion lose the cell.  The returned
    values sum to the per-cell success probability.  The law is the
    truncated geometric ``P(success on k) = r**(k-1) * (1-r-f)``.
    """
    success = 1.0 - retryable - fatal
    return [(k, (retryable ** (k - 1)) * success)
            for k in range(1, max_attempts + 1)]


def _bt_expected_attempts(max_attempts, retryable):
    if retryable >= 1.0:
        return float(max_attempts)
    return (1.0 - retryable ** max_attempts) / (1.0 - retryable)


def _bt_prefix_completion_distribution(prefix_len, max_attempts, retryable,
                                       fatal, max_budget):
    """Exact attempt-count distribution of "all ``prefix_len`` cells succeed".

    Paths that lose a cell are dropped: they cannot produce the required
    prefix under any remaining budget.  The total attempt count therefore
    ranges over ``[prefix_len, prefix_len * max_attempts]``.
    """
    try:
        prefix_len = int(prefix_len)
        max_attempts = int(max_attempts)
        max_budget = int(max_budget)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('DP arguments must be integers') from None
    if prefix_len < 1 or max_attempts < 1 or max_budget < 0:
        raise ValueError(
            'DP arguments must satisfy prefix_len >= 1, max_attempts >= 1 '
            'and max_budget >= 0')
    work = _bt_dp_work(prefix_len, max_budget, max_attempts)
    if (max_budget > int(BUDGET_TRUNCATION_POLICY['max_exact_dp_states'])
            or work is None
            or work > int(
                BUDGET_TRUNCATION_POLICY['max_exact_dp_operations'])):
        raise ValueError(
            'exact completion-risk DP exceeds the absolute state/work cap; '
            'the caller must report the panel as not_verifiable instead of '
            'starting an unbounded convolution')
    pmf = _bt_cell_success_pmf(max_attempts, retryable, fatal)
    dist = [0.0] * (max_budget + 1)
    dist[0] = 1.0
    for _ in range(prefix_len):
        nxt = [0.0] * (max_budget + 1)
        for used in range(max_budget + 1):
            prob = dist[used]
            if prob == 0.0:
                continue
            for k, pk in pmf:
                if pk == 0.0:
                    continue
                total = used + k
                if total <= max_budget:
                    nxt[total] += prob * pk
        dist = nxt
    return dist


def _bt_required_attempts(dist, risk_level, *, minimum):
    """Smallest budget whose completion probability reaches ``1-risk``."""
    if not dist:
        return None
    cdf = 0.0
    for budget in range(len(dist)):
        cdf += dist[budget]
        if budget >= minimum and cdf >= 1.0 - risk_level - 1e-15:
            return budget
    return None


def _bt_dp_work(prefix_len, max_budget, attempts):
    """Inner-loop iterations of the exact DP, or ``None`` if not an integer."""
    try:
        return int(prefix_len) * (int(max_budget) + 1) * int(attempts)
    except (TypeError, ValueError, OverflowError):
        return None


def _bt_round(value, digits=12):
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError, OverflowError):
        return None


def _bt_normalise_failure_model(failure_model, *, max_attempts):
    out = {'problems': [], 'model': None}
    if failure_model is None:
        return out
    if not isinstance(failure_model, dict):
        out['problems'].append('failure_model must be an object')
        return out
    r = _bt_number(failure_model.get('replaceable_per_attempt'),
                   minimum=0.0, maximum=1.0 - 1e-15)
    f = _bt_number(failure_model.get('non_replaceable_per_attempt'),
                   minimum=0.0, maximum=1.0 - 1e-15)
    if r is None:
        out['problems'].append(
            'failure_model.replaceable_per_attempt must be a probability in '
            '[0, 1)')
    if f is None:
        out['problems'].append(
            'failure_model.non_replaceable_per_attempt must be a probability '
            'in [0, 1)')
    if out['problems']:
        return out
    if r + f >= 1.0:
        out['problems'].append(
            f'failure_model probabilities sum to {r + f} >= 1; every attempt '
            'would fail')
        return out
    attempts = max_attempts
    if ('max_attempts_per_cell' in failure_model
            and failure_model.get('max_attempts_per_cell') is not None):
        attempts = _bt_int(failure_model.get('max_attempts_per_cell'),
                           minimum=1)
        if attempts is None:
            out['problems'].append(
                'failure_model.max_attempts_per_cell must be an integer >= 1 '
                '(omit it to use the registered disposition cap)')
            return out
    out['model'] = {
        'replaceable_per_attempt': r,
        'non_replaceable_per_attempt': f,
        'max_attempts_per_cell': attempts,
    }
    return out


def build_budget_truncation(*, cells, comparisons, arms=None, seeds=None,
                            min_n, complete_coverage,
                            declared_n_seeds=None,
                            max_attempts_per_cell=1,
                            max_attempts_source=None, budget=None,
                            failure_model=None, risk_level=None,
                            plan_source=None, manifest_sha256=None,
                            manifest_canonical_sha256=None,
                            created_at=None):
    """Compile one budget-truncation feasibility artifact.

    Every value is copied from a declaration (the frozen manifest, the
    run-order plan, the disposition policy) or computed from those values.
    Missing declarations stay missing and are reported; the function never
    invents a convenient budget, failure rate or risk level.
    """
    norm_cells, cell_problems, cell_warnings = _bt_normalise_cells(cells)
    comps, comp_problems = _bt_normalise_comparisons(comparisons)
    problems = list(cell_problems) + list(comp_problems)
    warnings = list(cell_warnings)
    observed_arms = sorted({c['variant'] for c in norm_cells})
    observed_blocks = list(dict.fromkeys(c['block'] for c in norm_cells))
    declared_arms = _bt_declared_list(arms, 'arms', problems)
    declared_seeds = _bt_declared_list(seeds, 'seeds', problems)
    arms = observed_arms if declared_arms is None else declared_arms
    seeds = observed_blocks if declared_seeds is None else declared_seeds
    # The declared grid is the unit of "complete coverage": inferring the
    # block set from the cells would let a plan that forgot a frozen seed
    # look complete (66/72 cells reported as 11/11 complete seeds).
    missing_arms = []
    missing_seeds = []
    if declared_arms is not None:
        declared_arm_set = set(declared_arms)
        observed_arm_set = set(observed_arms)
        for arm in observed_arms:
            if arm not in declared_arm_set:
                problems.append(
                    f'cell panel uses undeclared arm {arm!r}')
        missing_arms = [arm for arm in declared_arms
                        if arm not in observed_arm_set]
    if declared_seeds is not None:
        declared_seed_set = set(declared_seeds)
        observed_seed_set = set(observed_blocks)
        for block in observed_blocks:
            if block not in declared_seed_set:
                problems.append(
                    f'cell panel uses undeclared seed/block {block!r}')
        missing_seeds = [block for block in declared_seeds
                         if block not in observed_seed_set]
    n_seeds = len(seeds)
    declared_seed_count = None
    if declared_n_seeds is not None:
        declared_seed_count = _bt_int(declared_n_seeds, minimum=1)
        if declared_seed_count is None:
            problems.append(
                'declared n_seeds must be an integer >= 1')
        elif declared_seed_count != n_seeds:
            problems.append(
                f'declared n_seeds={declared_seed_count} disagrees with the '
                f'{n_seeds} declared seed(s); the resolution floor and the '
                'planned grid must describe the same panel')
    min_n = _bt_int(min_n, minimum=2)
    if min_n is None:
        problems.append('min_n (min_seeds_for_confirmation) must be an '
                        'integer >= 2')
    elif min_n > n_seeds:
        problems.append(f'min_n={min_n} exceeds the {n_seeds} planned block(s)')
    if _bt_is_bool(complete_coverage):
        complete_coverage = bool(complete_coverage)
    else:
        problems.append('complete_coverage must be boolean')
        complete_coverage = None
    # A missing declared arm that is not part of any comparison consumes no
    # pair and is therefore a grid warning; a comparison arm missing entirely
    # is a structural problem and is reported by the family loop below.
    for arm in missing_arms:
        warnings.append(
            f'declared arm {arm!r} has no cell in the panel; if it belongs '
            'to the registered grid, the plan is incomplete')
    # Under complete coverage every declared seed is mandatory (the point of
    # this analysis); without it a shorter min-n panel can still answer the
    # family, so the same gap is only a warning.
    for block in missing_seeds:
        message = (f'declared seed/block {block!r} has no cell in the panel; '
                   'the registered arm x seed grid is incomplete')
        if complete_coverage:
            problems.append(message)
        else:
            warnings.append(message)
    max_attempts = _bt_int(max_attempts_per_cell, minimum=1)
    if max_attempts is None:
        problems.append('max_attempts_per_cell must be an integer >= 1')
    elif max_attempts > int(BUDGET_TRUNCATION_POLICY['max_attempts_sanity']):
        problems.append(
            f'max_attempts_per_cell={max_attempts} exceeds the '
            f'{BUDGET_TRUNCATION_POLICY["max_attempts_sanity"]}-attempt '
            'sanity cap; the retry policy is malformed')
    budget_attempts = None
    if budget is not None:
        budget_attempts = _bt_int(budget, minimum=1)
        if budget_attempts is None:
            problems.append('budget must be an integer >= 1 (cell attempts)')
    risk = (_bt_number(risk_level, minimum=1e-12, maximum=1.0 - 1e-12)
            if risk_level is not None
            else float(BUDGET_TRUNCATION_POLICY['default_risk_level']))
    if risk is None:
        problems.append('risk_level must be a probability in (0,1)')

    # Arms used by the family must exist in the panel at all; a missing arm is
    # a structural impossibility, not a budget question.
    arms_set = set(arms)
    observed_arm_set = set(observed_arms)
    for comp in comps:
        for key in ('a', 'b'):
            if comp[key] not in arms_set:
                problems.append(
                    f'comparison {comp["name"]!r} names undeclared arm '
                    f'{comp[key]!r}')
            elif comp[key] not in observed_arm_set:
                problems.append(
                    f'comparison {comp["name"]!r} arm {comp[key]!r} never '
                    'appears in the cell panel')

    frontier = truncation_frontier(
        norm_cells, comps, min_n=min_n or 2,
        complete_coverage=bool(complete_coverage),
        required_blocks=seeds if declared_seeds is not None else None)
    seen_problems = set(problems)
    for problem in frontier['problems']:
        if problem not in seen_problems:
            problems.append(problem)
            seen_problems.add(problem)
    warnings.extend(frontier['warnings'])
    required_cells = None
    if complete_coverage:
        required_cells = frontier['first_complete_cells']
        if required_cells is None:
            problems.append(
                'complete coverage is required but no prefix completes every '
                'declared comparison; the panel cannot reach the frozen rule')
            present_by_block = {}
            for cell in norm_cells:
                present_by_block.setdefault(cell['block'], set()).add(
                    cell['variant'])
            missing = []
            for comp in comps:
                for block, present in present_by_block.items():
                    if not ({comp['a'], comp['b']} <= present):
                        missing.append(
                            f'{comp["name"]}@{block} needs '
                            f'{sorted({comp["a"], comp["b"]} - present)}')
            if missing:
                problems.append('missing block arms: ' + '; '.join(missing[:6]))
    else:
        required_cells = frontier['first_min_n_cells']
        if required_cells is None:
            problems.append(
                f'no prefix reaches min_n={min_n} pairs for every comparison')

    family_arms = {comp['a'] for comp in comps} | {comp['b'] for comp in comps}
    single_cell_lethal = bool(
        complete_coverage and required_cells is not None
        and any(c['variant'] in family_arms for c in norm_cells))
    floor = {
        'declared_min_seeds': min_n,
        'planned_seeds': n_seeds,
        'effective_required_seeds': n_seeds if complete_coverage else min_n,
        'effective_required_cells': required_cells,
        'declared_floor_is_operational': bool(
            not complete_coverage or (min_n is not None and min_n == n_seeds)),
        'single_cell_lethal': single_cell_lethal,
        'complete_coverage_required': bool(complete_coverage),
        'shortest_min_n_prefix_cells': frontier['first_min_n_cells'],
        'shortest_complete_prefix_cells': frontier['first_complete_cells'],
    }
    if (complete_coverage and min_n is not None and min_n < n_seeds):
        warnings.append(
            f'the declared min_seeds_for_confirmation={min_n} is a resolution '
            f'floor, not the operational floor: require_complete_coverage=true '
            f'makes all {n_seeds} blocks mandatory, so the budget must cover '
            f'{required_cells if required_cells is not None else "all"} cells, '
            'not the shorter min-n prefix')
    if plan_source is not None and str(plan_source) != 'registered_run_order':
        warnings.append(
            f'the execution order is {plan_source!r}, not a stored registered '
            'run_order.json; block-level totals are exact but the mid-block '
            'arm counts are that canonical order, not a measured one')

    fm = _bt_normalise_failure_model(failure_model,
                                     max_attempts=max_attempts or 1)
    problems.extend(fm['problems'])
    model = fm['model']
    if model is not None:
        if (max_attempts is not None
                and model['max_attempts_per_cell'] > max_attempts):
            warnings.append(
                f'failure_model.max_attempts_per_cell='
                f'{model["max_attempts_per_cell"]} exceeds the registered '
                f'disposition cap {max_attempts}; the registered cap is used')
            model['max_attempts_per_cell'] = max_attempts

    cost = {
        'budget_unit': BUDGET_TRUNCATION_POLICY['budget_unit'],
        'planned_attempts': len(norm_cells),
        'max_attempts_per_cell': max_attempts,
        'max_attempts_source': max_attempts_source,
        'worst_case_attempts': (len(norm_cells) * max_attempts
                                if max_attempts else None),
        'expected_attempts_full_panel': None,
        'expected_attempts_required_prefix': None,
        'required_prefix_cells': required_cells,
        'required_attempts_for_risk': None,
        'p_required_prefix_unlimited': None,
        'p_full_panel_unlimited': None,
        'risk_level': risk,
        'failure_model': dict(model) if model is not None else None,
        'completion_risk_method': 'not_declared',
    }
    exact_dp = True
    if model is not None and required_cells is not None:
        r = model['replaceable_per_attempt']
        f = model['non_replaceable_per_attempt']
        a = model['max_attempts_per_cell']
        p_cell = sum(p for _k, p in _bt_cell_success_pmf(a, r, f))
        cost['p_required_prefix_unlimited'] = _bt_round(
            p_cell ** required_cells)
        if complete_coverage:
            cost['p_full_panel_unlimited'] = _bt_round(
                p_cell ** len(norm_cells))
        cost['expected_attempts_full_panel'] = _bt_round(
            len(norm_cells) * _bt_expected_attempts(a, r), 6)
        cost['expected_attempts_required_prefix'] = _bt_round(
            required_cells * _bt_expected_attempts(a, r), 6)
        max_budget = required_cells * a
        dp_work = _bt_dp_work(required_cells, max_budget, a)
        if (max_budget > int(BUDGET_TRUNCATION_POLICY['max_exact_dp_states'])
                or dp_work is None
                or dp_work > int(
                    BUDGET_TRUNCATION_POLICY['max_exact_dp_operations'])):
            exact_dp = False
            cost['completion_risk_method'] = 'skipped_panel_too_large'
            warnings.append(
                f'the exact completion-risk DP was skipped: '
                f'{required_cells} required cells x {a} attempts x '
                f'{max_budget + 1} states exceeds the '
                f'{BUDGET_TRUNCATION_POLICY["max_exact_dp_states"]}-state / '
                f'{BUDGET_TRUNCATION_POLICY["max_exact_dp_operations"]}-'
                'operation cap; only the deterministic floor was checked')
        else:
            cost['completion_risk_method'] = 'exact_dp'
            dist = _bt_prefix_completion_distribution(
                required_cells, a, r, f, max_budget)
            cost['required_attempts_for_risk'] = _bt_required_attempts(
                dist, risk, minimum=required_cells)

    budget_block = {
        'declared_attempts': budget_attempts,
        'affordable_planned_cells': None,
        'complete_seeds_at_budget': None,
        'pairs_at_budget': None,
        'min_n_met_at_budget': None,
        'complete_coverage_met_at_budget': None,
        'p_required_prefix_within_budget': None,
        'headroom_attempts': None,
        'worst_case_covered': None,
    }
    if budget_attempts is not None and required_cells is not None:
        affordable = min(budget_attempts, len(norm_cells))
        point = truncation_frontier_point(
            norm_cells, comps, affordable, min_n=min_n or 2, n_seeds=n_seeds,
            required_blocks=seeds if declared_seeds is not None else None)
        budget_block['affordable_planned_cells'] = affordable
        budget_block['complete_seeds_at_budget'] = point['complete_seeds']
        budget_block['pairs_at_budget'] = dict(point['pairs'])
        budget_block['min_n_met_at_budget'] = point['min_n_met']
        budget_block['complete_coverage_met_at_budget'] = (
            point['complete_coverage_met'])
        budget_block['headroom_attempts'] = budget_attempts - required_cells
        # The worst case is the *registered disposal cap*, not the failure
        # model's expected attempt count: a retry may be attempted even when
        # the model assigns it a tiny probability.
        budget_block['worst_case_covered'] = bool(
            max_attempts is not None
            and budget_attempts >= len(norm_cells) * max_attempts)
        if model is not None and exact_dp:
            # The per-cell law has at most ``model`` attempts, so the total
            # cannot exceed ``required_cells * model_attempts``.  Capping here
            # (rather than at the registered disposition cap) keeps the
            # budget row's DP no larger than the already-computed full DP.
            max_budget = min(budget_attempts,
                             required_cells * model['max_attempts_per_cell'])
            dist = _bt_prefix_completion_distribution(
                required_cells, model['max_attempts_per_cell'],
                model['replaceable_per_attempt'],
                model['non_replaceable_per_attempt'], max_budget)
            budget_block['p_required_prefix_within_budget'] = _bt_round(
                sum(dist[:budget_attempts + 1]))

    # ---- verdict ----------------------------------------------------------
    if problems:
        status = 'fail'
    elif budget_attempts is None:
        status = 'not_verifiable'
        warnings.append(
            'no cell-attempt budget is declared; the deterministic floor is '
            f'{required_cells} attempt(s) but the retry reserve cannot be '
            'checked')
    elif model is None:
        if budget_attempts < required_cells:
            problems.append(
                f'budget={budget_attempts} cell attempts cannot complete the '
                f'{required_cells}-cell required prefix once')
            status = 'fail'
        else:
            warnings.append(
                'no per-attempt failure model is declared; the budget covers '
                'the required prefix once but the retry reserve needed by the '
                'disposition policy is unquantified')
            status = 'warnings'
    else:
        p_required = cost['p_required_prefix_unlimited']
        need = cost['required_attempts_for_risk']
        if budget_attempts < required_cells:
            problems.append(
                f'budget={budget_attempts} cell attempts cannot complete the '
                f'{required_cells}-cell required prefix once')
            status = 'fail'
        elif p_required is None or p_required < 1.0 - risk - 1e-12:
            problems.append(
                'the declared failure model makes the required prefix '
                f'unlikely even with unlimited attempts '
                f'(P={p_required!r} < {1.0 - risk:.3f}); retries cannot '
                'repair non-replaceable failures')
            status = 'fail'
        elif not exact_dp:
            # The budget clears the deterministic floor but the panel is too
            # large for the exact risk DP: the question stays unanswered.
            status = 'not_verifiable'
        elif need is None or budget_attempts < need:
            problems.append(
                f'budget={budget_attempts} cell attempts is below the '
                f'{need!r} attempts needed for a {1.0 - risk:.0%} completion '
                'probability; the declared budget does not support the '
                'pre-registered complete panel')
            status = 'fail'
        else:
            status = 'warnings' if warnings else 'ok'

    artifact = {
        'kind': BUDGET_TRUNCATION_KIND,
        'schema': BUDGET_TRUNCATION_SCHEMA,
        'status': status,
        'problems': problems,
        'warnings': warnings,
        'study': {
            'manifest_sha256': _bt_sha256_text(manifest_sha256),
            'manifest_canonical_sha256': _bt_sha256_text(
                manifest_canonical_sha256),
            'plan_source': _bt_text(plan_source),
            'plan_sha256': _bt_sha_obj(norm_cells),
        },
        'design': {
            'n_cells': len(norm_cells),
            'n_blocks': n_seeds,
            'n_arms': len(arms),
            'arms': list(arms),
            'seeds': list(seeds),
            'comparisons': [dict(c) for c in comps],
            'min_n': min_n,
            'complete_coverage': complete_coverage,
        },
        'floor': floor,
        'cost': cost,
        'budget': budget_block,
        'frontier': {
            'n_points': len(frontier['points']),
            'sampled': frontier['sampled'],
            'first_min_n_cells': frontier['first_min_n_cells'],
            'first_complete_cells': frontier['first_complete_cells'],
            'points': frontier['points'],
        },
        'inputs': {
            'cells': norm_cells,
            'comparisons': [dict(c) for c in comps],
            'arms': list(arms),
            'seeds': list(seeds),
            'min_n': min_n,
            'complete_coverage': complete_coverage,
            'declared_n_seeds': declared_seed_count,
            'max_attempts_per_cell': max_attempts,
            'max_attempts_source': max_attempts_source,
            'budget': budget_attempts,
            'failure_model': (dict(model) if model is not None else None),
            'risk_level': risk,
            'plan_source': _bt_text(plan_source),
            'manifest_sha256': _bt_sha256_text(manifest_sha256),
            'manifest_canonical_sha256': _bt_sha256_text(
                manifest_canonical_sha256),
        },
        'policy': dict(BUDGET_TRUNCATION_POLICY),
        'created_at': created_at or time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    artifact['content_sha256'] = budget_truncation_content_sha(artifact)
    return artifact


def budget_truncation_content_sha(artifact):
    body = {k: v for k, v in artifact.items()
            if k not in ('created_at', 'content_sha256')}
    return _bt_sha_obj(body)


def budget_truncation_from_manifest(
        manifest, *, plan=None, cells=None, disposition=None, budget=None,
        failure_model=None, risk_level=None,
        plan_source=None, max_attempts_per_cell=None, created_at=None):
    """Build the analysis from the frozen manifest and an execution order.

    ``plan`` may be a run-order plan dict.  When it is absent the canonical
    complete-block order (seed-major, manifest arm order) is used and the
    artifact says so; the block-level frontier is identical for every
    complete-block order, only the mid-block arm counts differ.
    """
    if not isinstance(manifest, dict):
        raise TypeError('manifest must be an object')
    raw_arms = manifest.get('arms')
    raw_seeds = manifest.get('seeds')
    arms = list(raw_arms) if isinstance(raw_arms, (list, tuple)) else raw_arms
    seeds = list(raw_seeds) if isinstance(raw_seeds, (list, tuple)) \
        else raw_seeds
    stats = manifest.get('statistics')
    stats = stats if isinstance(stats, dict) else {}
    comps = []
    raw_comparisons = manifest.get('primary_comparisons')
    if not isinstance(raw_comparisons, (list, tuple)):
        raw_comparisons = []
    for raw in raw_comparisons:
        if not isinstance(raw, dict):
            continue
        comps.append({
            'name': _bt_text(raw.get('name')) or (
                f'{raw.get("a")} - {raw.get("b")}'),
            'a': _bt_text(raw.get('a')),
            'b': _bt_text(raw.get('b')),
        })
    if cells is None:
        if plan is None:
            if not (isinstance(arms, (list, tuple))
                    and isinstance(seeds, (list, tuple))):
                # The declaration is malformed; let build_budget_truncation
                # report the exact field instead of raising an opaque
                # TypeError from list()/int().
                cells = []
                if plan_source is None:
                    plan_source = 'canonical_complete_block_seed0'
            else:
                try:
                    raw_cells = [(str(arm), int(seed))
                                 for seed in seeds for arm in arms]
                    plan = build_run_order_plan(  # noqa: F821
                        raw_cells, block_key=lambda c: c[1], seed=0,
                        block_name='seed')
                except Exception as exc:  # pragma: no cover - defensive
                    raise ValueError(
                        'could not build the canonical run order: '
                        f'{type(exc).__name__}: {exc}') from None
                if plan_source is None:
                    plan_source = 'canonical_complete_block_seed0'
                cells = plan
        else:
            if plan_source is None:
                plan_source = 'registered_run_order'
            cells = plan
    if max_attempts_per_cell is None:
        policy = None
        explicit_policy = False
        if disposition is not None:
            # An explicitly supplied disposition is a declaration even when
            # malformed: falling through to the default would hide a broken
            # run_disposition artifact and silently re-open retries.
            explicit_policy = True
            if isinstance(disposition, dict):
                policy = (disposition.get('policy')
                          if disposition.get('kind') == 'run_disposition'
                          else disposition)
            else:
                policy = disposition
        elif stats.get('disposition_policy') is not None:
            explicit_policy = True
            policy = stats.get('disposition_policy')
        elif manifest.get('disposition_policy') is not None:
            explicit_policy = True
            policy = manifest.get('disposition_policy')
        if explicit_policy:
            if not isinstance(policy, dict):
                max_attempts_source = 'disposition_policy_invalid'
                max_attempts_per_cell = None
            else:
                try:
                    norm = validate_run_disposition_policy(policy)  # noqa: F821
                except Exception:
                    norm = None
                if norm is not None:
                    max_attempts_per_cell = norm.get('max_attempts_per_cell')
                    max_attempts_source = 'disposition_policy'
                else:
                    max_attempts_source = 'disposition_policy_invalid'
        else:
            max_attempts_per_cell = RUN_DISPOSITION_POLICY[  # noqa: F821
                'max_attempts_per_cell']
            max_attempts_source = 'default_run_disposition_policy'
    else:
        max_attempts_source = 'caller'
    if budget is None:
        disclosure = manifest.get('compute_budget_disclosure')
        if isinstance(disclosure, dict):
            budget = disclosure.get('cell_attempt_budget')
    if failure_model is None:
        for holder in (stats, manifest.get('compute_budget_disclosure')):
            if isinstance(holder, dict) and isinstance(
                    holder.get('failure_model'), dict):
                failure_model = holder['failure_model']
                break
    if not comps:
        raise ValueError('the manifest declares no primary comparisons')
    return build_budget_truncation(
        cells=cells, comparisons=comps, arms=arms, seeds=seeds,
        min_n=stats.get('min_seeds_for_confirmation'),
        # Pass the declaration through unchanged: ``is True`` would silently
        # turn 1 / "true" into "complete coverage not required", and a
        # missing field must be reported rather than read as a permissive
        # False.  Both cases fail the malformed freeze closed.
        complete_coverage=stats.get('require_complete_coverage'),
        declared_n_seeds=stats.get('n_seeds'),
        max_attempts_per_cell=max_attempts_per_cell,
        max_attempts_source=max_attempts_source, budget=budget,
        failure_model=failure_model, risk_level=risk_level,
        plan_source=plan_source,
        manifest_sha256=manifest.get('manifest_sha256'),
        manifest_canonical_sha256=_bt_manifest_canonical_sha(manifest),
        created_at=created_at)


def replay_budget_truncation(artifact):
    """Rebuild the artifact from its recorded inputs (no filesystem read)."""
    if not isinstance(artifact, dict):
        raise TypeError('artifact must be an object')
    inputs = artifact.get('inputs')
    if not isinstance(inputs, dict):
        raise ValueError('artifact.inputs is missing')
    return build_budget_truncation(
        cells=inputs.get('cells'),
        comparisons=inputs.get('comparisons'),
        arms=inputs.get('arms'),
        seeds=inputs.get('seeds'),
        min_n=inputs.get('min_n'),
        complete_coverage=inputs.get('complete_coverage'),
        declared_n_seeds=inputs.get('declared_n_seeds'),
        max_attempts_per_cell=inputs.get('max_attempts_per_cell'),
        max_attempts_source=inputs.get('max_attempts_source'),
        budget=inputs.get('budget'),
        failure_model=inputs.get('failure_model'),
        risk_level=inputs.get('risk_level'),
        plan_source=inputs.get('plan_source'),
        manifest_sha256=inputs.get('manifest_sha256'),
        manifest_canonical_sha256=inputs.get('manifest_canonical_sha256'),
        created_at=artifact.get('created_at'))


def validate_budget_truncation(artifact, *, manifest=None):
    """Validate structure, body hash and a full replay of every derived field."""
    out = {'kind': 'budget_truncation_validation', 'status': 'ok',
           'problems': [], 'warnings': [], 'replayed': False}
    if not isinstance(artifact, dict):
        out['problems'].append(
            f'artifact must be an object, got {type(artifact).__name__}')
        out['status'] = 'fail'
        return out
    if artifact.get('kind') != BUDGET_TRUNCATION_KIND:
        out['problems'].append(
            f'artifact.kind must be {BUDGET_TRUNCATION_KIND!r}')
    if artifact.get('schema') != BUDGET_TRUNCATION_SCHEMA:
        out['problems'].append(
            f'artifact.schema must be {BUDGET_TRUNCATION_SCHEMA}')
    if artifact.get('status') not in ('ok', 'warnings', 'not_verifiable',
                                      'fail'):
        out['problems'].append(
            f'artifact.status={artifact.get("status")!r} is not recognised')
    for field in ('problems', 'warnings'):
        value = artifact.get(field)
        if value is not None and not isinstance(value, (list, tuple)):
            out['problems'].append(
                f'artifact.{field} must be a list, got '
                f'{type(value).__name__}')
    if artifact.get('policy') != BUDGET_TRUNCATION_POLICY:
        out['problems'].append(
            'artifact.policy does not match this implementation; the '
            'artifact was produced by a different feasibility policy')
    recorded = _bt_sha256_text(artifact.get('content_sha256'))
    if recorded is None:
        out['problems'].append('content_sha256 is missing or not a digest')
    else:
        try:
            want = budget_truncation_content_sha(artifact)
        except (TypeError, ValueError, RecursionError) as exc:
            out['problems'].append(
                f'artifact body is not canonical JSON: '
                f'{type(exc).__name__}: {exc}')
        else:
            if recorded != want:
                out['problems'].append(
                    'content_sha256 does not reproduce from the artifact body')
    study = _bt_dict(artifact.get('study'))
    if manifest is not None:
        if not isinstance(manifest, dict):
            out['problems'].append('manifest argument must be an object')
        else:
            want = _bt_manifest_canonical_sha(manifest)
            got = _bt_sha256_text(study.get('manifest_canonical_sha256'))
            if want is not None:
                if got is None:
                    out['problems'].append(
                        'artifact carries no manifest body binding; a '
                        'manifest-checked artifact must record its canonical '
                        'hash')
                elif got != want:
                    out['problems'].append(
                        'artifact was built against a different manifest body')
            want_sha = _bt_sha256_text(manifest.get('manifest_sha256'))
            got_sha = _bt_sha256_text(study.get('manifest_sha256'))
            if want_sha is not None:
                if got_sha is None:
                    out['problems'].append(
                        'artifact carries no manifest_sha256 binding')
                elif got_sha != want_sha:
                    out['problems'].append(
                        'artifact manifest_sha256 does not match the manifest')
    # A full deterministic replay is the strongest tamper check: every count,
    # probability, threshold and verdict must reproduce from `inputs`.
    try:
        replayed = replay_budget_truncation(artifact)
    except Exception as exc:
        out['problems'].append(
            f'artifact cannot be replayed from its inputs: '
            f'{type(exc).__name__}: {exc}')
        out['status'] = 'fail'
        return out
    out['replayed'] = True
    for section in ('design', 'floor', 'cost', 'budget', 'inputs', 'study'):
        if artifact.get(section) != replayed.get(section):
            out['problems'].append(
                f'artifact.{section} does not reproduce from its inputs')
    kept = artifact.get('frontier') if isinstance(
        artifact.get('frontier'), dict) else {}
    fresh = replayed.get('frontier') if isinstance(
        replayed.get('frontier'), dict) else {}
    if kept.get('points') != fresh.get('points'):
        out['problems'].append(
            'artifact.frontier.points does not reproduce from the inputs')
    for key in ('n_points', 'sampled', 'first_min_n_cells',
                'first_complete_cells'):
        if kept.get(key) != fresh.get(key):
            out['problems'].append(
                f'artifact.frontier.{key} does not reproduce')
    if artifact.get('status') != replayed.get('status'):
        out['problems'].append(
            'artifact.status does not reproduce from the inputs')
    kept_problems = _bt_message_list(artifact.get('problems'))
    kept_warnings = _bt_message_list(artifact.get('warnings'))
    if kept_problems != _bt_message_list(replayed.get('problems')):
        out['problems'].append(
            'artifact.problems does not reproduce from the inputs')
    if kept_warnings != _bt_message_list(replayed.get('warnings')):
        out['problems'].append(
            'artifact.warnings does not reproduce from the inputs')
    if artifact.get('status') == 'ok' and (
            kept_problems or kept_warnings):
        out['problems'].append(
            'artifact is marked ok but carries problems/warnings')
    out['warnings'].extend(kept_warnings)
    if out['problems']:
        out['status'] = 'fail'
    return out


def write_budget_truncation(outdir, artifact, *, overwrite=False):
    """Atomically persist the artifact; refuse silent replacement."""
    if isinstance(outdir, os.PathLike):
        outdir = os.fspath(outdir)
    if not isinstance(outdir, str) or not outdir.strip():
        raise ValueError('outdir must be a non-empty path')
    verdict = validate_budget_truncation(artifact)
    path = os.path.join(outdir, 'budget_truncation.json')
    out = {'kind': 'write_budget_truncation', 'path': path, 'written': False,
           'unchanged': False, 'conflict': False, 'status': verdict['status'],
           'artifact_status': (artifact.get('status')
                               if isinstance(artifact, dict) else None),
           'problems': list(verdict['problems'])}
    if verdict['status'] != 'ok':
        return out
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as fh:
                old = json.load(fh)
        except Exception:
            old = None
        if (isinstance(old, dict)
                and old.get('content_sha256') == artifact.get('content_sha256')):
            out['unchanged'] = True
            return out
        if not overwrite:
            out['conflict'] = True
            out['problems'].append(
                'a different budget_truncation.json already exists; pass '
                'overwrite=True only after re-running the analysis')
            return out
    os.makedirs(outdir, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(artifact, fh, ensure_ascii=False, indent=1, allow_nan=False)
        fh.write('\n')
    os.replace(tmp, path)
    out['written'] = True
    return out


def budget_truncation_summary(artifact):
    """Compact block for preflight snapshots and clearance-style payloads."""
    if not isinstance(artifact, dict):
        return {'kind': 'budget_truncation_summary', 'status': 'missing',
                'ok': False}
    study = _bt_dict(artifact.get('study'))
    design = _bt_dict(artifact.get('design'))
    cost = _bt_dict(artifact.get('cost'))
    floor = _bt_dict(artifact.get('floor'))
    budget = _bt_dict(artifact.get('budget'))
    return {
        'kind': 'budget_truncation_summary',
        'status': artifact.get('status'),
        'ok': artifact.get('status') == 'ok',
        'content_sha256': artifact.get('content_sha256'),
        'manifest_canonical_sha256': study.get(
            'manifest_canonical_sha256'),
        'n_cells': design.get('n_cells'),
        'required_prefix_cells': cost.get('required_prefix_cells'),
        'effective_required_seeds': floor.get('effective_required_seeds'),
        'declared_min_seeds': floor.get('declared_min_seeds'),
        'budget_attempts': budget.get('declared_attempts'),
        'required_attempts_for_risk': cost.get('required_attempts_for_risk'),
        'p_required_prefix_within_budget': budget.get(
            'p_required_prefix_within_budget'),
    }
