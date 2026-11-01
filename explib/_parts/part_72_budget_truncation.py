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
import os
import time


BUDGET_TRUNCATION_KIND = 'budget_truncation'
BUDGET_TRUNCATION_SCHEMA = 1
BUDGET_TRUNCATION_POLICY = {
    'version': 1,
    'default_risk_level': 0.05,
    'max_frontier_points': 4096,
    'max_attempts_sanity': 100,
    'max_exact_dp_states': 5000000,
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


def _bt_text(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _bt_int(value, *, minimum=None):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        out = int(value)
    elif isinstance(value, float) and value.is_integer():
        out = int(value)
    else:
        return None
    if minimum is not None and out < minimum:
        return None
    return out


def _bt_number(value, *, minimum=None, maximum=None):
    if value is None or isinstance(value, (bool, str, bytes, bytearray)):
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


def _bt_canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False)


def _bt_sha_obj(obj):
    return hashlib.sha256(_bt_canonical(obj).encode('utf-8')).hexdigest()


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
        dupes = sorted({x for x in labels if labels.count(x) > 1})
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
                        max_points=None):
    """Deterministic per-prefix completeness of the registered cell panel.

    For every prefix of the execution order this records the number of
    *complete* blocks (blocks holding every arm a comparison needs), the
    paired-seed count of each comparison, and whether the declared
    ``min_n`` / complete-coverage rule is satisfied.  Nothing here depends
    on a failure model: it is the deterministic map of "what would we have
    if the campaign stopped after k cells".
    """
    policy = BUDGET_TRUNCATION_POLICY
    max_points = (_bt_int(max_points, minimum=1) if max_points is not None
                  else int(policy['max_frontier_points']))
    norm_cells, cell_problems, cell_warnings = _bt_normalise_cells(cells)
    comps, comp_problems = _bt_normalise_comparisons(comparisons)
    out = {
        'kind': 'budget_truncation_frontier',
        'status': 'ok',
        'problems': list(cell_problems) + list(comp_problems),
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
    for cell in norm_cells:
        if cell['block'] not in blocks:
            blocks.append(cell['block'])
    arms_needed = sorted({c['a'] for c in comps} | {c['b'] for c in comps})
    out['n_seeds'] = len(blocks)
    out['n_arms'] = len({c['variant'] for c in norm_cells})
    n_cells = len(norm_cells)
    n_seeds = len(blocks)
    if out['min_n'] is None or out['min_n'] > n_seeds:
        out['problems'].append(
            f'min_n={min_n!r} is not an integer in [2, {n_seeds}]')
        out['status'] = 'fail'
        return out
    # A cell that carries an arm no comparison uses still consumes budget and
    # still belongs to a block; it simply never contributes a pair.  Record
    # that explicitly instead of silently dropping it from the counts.
    for arm in arms_needed:
        if not any(c['variant'] == arm for c in norm_cells):
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
    if sample_every > 1:
        out['sampled'] = True
    last_block = None
    for index, cell in enumerate(norm_cells, start=1):
        arms = block_arms.setdefault(cell['block'], set())
        newly = cell['variant'] not in arms
        arms.add(cell['variant'])
        arm_counts[cell['variant']] += 1
        last_block = cell['block']
        if newly:
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
        is_block_end = (index == n_cells
                        or norm_cells[index]['block'] != cell['block'])
        min_n_met = all(v >= out['min_n'] for v in pairs.values())
        complete_met = all(v >= n_seeds for v in pairs.values())
        # Thresholds are tracked on every prefix, whether or not the row is
        # emitted: sampling must not move the reported operational floor.
        if min_n_met and out['first_min_n_cells'] is None:
            out['first_min_n_cells'] = index
        if complete_met and out['first_complete_cells'] is None:
            out['first_complete_cells'] = index
        if (not out['sampled']) or is_block_end or index % sample_every == 0:
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
                              arms_needed=None):
    """Compute one frontier row at prefix ``k`` (used for exact thresholds)."""
    norm_cells, cell_problems, _warnings = _bt_normalise_cells(cells)
    comps, comp_problems = _bt_normalise_comparisons(comparisons)
    if cell_problems or comp_problems:
        raise ValueError('; '.join(cell_problems + comp_problems))
    k = _bt_int(k, minimum=1)
    if k is None or k > len(norm_cells):
        raise ValueError(f'prefix k={k!r} is outside [1, {len(norm_cells)}]')
    if n_seeds is None:
        n_seeds = len({c['block'] for c in norm_cells})
    if arms_needed is None:
        arms_needed = sorted({c['a'] for c in comps} | {c['b'] for c in comps})
    block_arms = {}
    arm_counts = collections.Counter()
    for cell in norm_cells[:k]:
        block_arms.setdefault(cell['block'], set()).add(cell['variant'])
        arm_counts[cell['variant']] += 1
    pairs = {}
    for comp in comps:
        pairs[comp['name']] = sum(
            1 for arms in block_arms.values()
            if comp['a'] in arms and comp['b'] in arms)
    complete_seeds = sum(1 for arms in block_arms.values()
                         if all(arm in arms for arm in arms_needed))
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


def _bt_round(value, digits=12):
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
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
    attempts = _bt_int(failure_model.get('max_attempts_per_cell'), minimum=1)
    if attempts is None:
        attempts = max_attempts
    out['model'] = {
        'replaceable_per_attempt': r,
        'non_replaceable_per_attempt': f,
        'max_attempts_per_cell': attempts,
    }
    return out


def build_budget_truncation(*, cells, comparisons, arms=None, seeds=None,
                            min_n, complete_coverage,
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
    if arms is None:
        arms = sorted({c['variant'] for c in norm_cells})
    else:
        arms = [str(a) for a in arms]
    if len(set(arms)) != len(arms):
        problems.append('arms contains duplicates')
    if seeds is None:
        seeds = list(dict.fromkeys(c['block'] for c in norm_cells))
    else:
        seeds = [str(s) for s in seeds]
    n_seeds = len({c['block'] for c in norm_cells})
    min_n = _bt_int(min_n, minimum=2)
    if min_n is None:
        problems.append('min_n (min_seeds_for_confirmation) must be an '
                        'integer >= 2')
    elif min_n > n_seeds:
        problems.append(f'min_n={min_n} exceeds the {n_seeds} planned block(s)')
    if not isinstance(complete_coverage, bool):
        problems.append('complete_coverage must be boolean')
        complete_coverage = None
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
    for comp in comps:
        for key in ('a', 'b'):
            if comp[key] not in arms:
                problems.append(
                    f'comparison {comp["name"]!r} names undeclared arm '
                    f'{comp[key]!r}')
            elif not any(c['variant'] == comp[key] for c in norm_cells):
                problems.append(
                    f'comparison {comp["name"]!r} arm {comp[key]!r} never '
                    'appears in the cell panel')

    frontier = truncation_frontier(
        norm_cells, comps, min_n=min_n or 2,
        complete_coverage=bool(complete_coverage))
    problems.extend(p for p in frontier['problems'] if p not in problems)
    warnings.extend(frontier['warnings'])
    required_cells = None
    if complete_coverage:
        required_cells = frontier['first_complete_cells']
        if required_cells is None:
            problems.append(
                'complete coverage is required but no prefix completes every '
                'declared comparison; the panel cannot reach the frozen rule')
            missing = []
            for comp in comps:
                for block in dict.fromkeys(c['block'] for c in norm_cells):
                    present = {c['variant'] for c in norm_cells
                               if c['block'] == block}
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

    single_cell_lethal = bool(
        complete_coverage and required_cells is not None
        and any(c['variant'] in {comp['a'] for comp in comps}
                | {comp['b'] for comp in comps} for c in norm_cells))
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
        if max_budget > int(BUDGET_TRUNCATION_POLICY['max_exact_dp_states']):
            exact_dp = False
            cost['completion_risk_method'] = 'skipped_panel_too_large'
            warnings.append(
                f'the exact completion-risk DP was skipped: '
                f'{required_cells} required cells x {a} attempts exceeds the '
                f'{BUDGET_TRUNCATION_POLICY["max_exact_dp_states"]}-state '
                'cap; only the deterministic floor was checked')
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
            norm_cells, comps, affordable, min_n=min_n or 2, n_seeds=n_seeds)
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
            max_budget = min(budget_attempts, required_cells * max_attempts + 1)
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
    arms = list(manifest.get('arms') or [])
    seeds = list(manifest.get('seeds') or [])
    stats = manifest.get('statistics')
    stats = stats if isinstance(stats, dict) else {}
    comps = []
    for raw in (manifest.get('primary_comparisons') or []):
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
        else:
            if plan_source is None:
                plan_source = 'registered_run_order'
        cells = plan
    if max_attempts_per_cell is None:
        policy = None
        if isinstance(disposition, dict):
            policy = disposition
            if disposition.get('kind') == 'run_disposition':
                policy = disposition.get('policy')
        elif isinstance(stats.get('disposition_policy'), dict):
            policy = stats.get('disposition_policy')
        elif isinstance(manifest.get('disposition_policy'), dict):
            policy = manifest.get('disposition_policy')
        if policy is not None:
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
        complete_coverage=stats.get('require_complete_coverage') is True,
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
    recorded = _bt_sha256_text(artifact.get('content_sha256'))
    if recorded is None:
        out['problems'].append('content_sha256 is missing or not a digest')
    else:
        try:
            want = budget_truncation_content_sha(artifact)
        except (TypeError, ValueError) as exc:
            out['problems'].append(
                f'artifact body is not canonical JSON: '
                f'{type(exc).__name__}: {exc}')
        else:
            if recorded != want:
                out['problems'].append(
                    'content_sha256 does not reproduce from the artifact body')
    if manifest is not None:
        if not isinstance(manifest, dict):
            out['problems'].append('manifest argument must be an object')
        else:
            want = _bt_manifest_canonical_sha(manifest)
            got = _bt_sha256_text((artifact.get('study') or {}).get(
                'manifest_canonical_sha256'))
            if want is not None and got is not None and want != got:
                out['problems'].append(
                    'artifact was built against a different manifest body')
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
    if list(artifact.get('problems') or []) != list(
            replayed.get('problems') or []):
        out['problems'].append(
            'artifact.problems does not reproduce from the inputs')
    if list(artifact.get('warnings') or []) != list(
            replayed.get('warnings') or []):
        out['problems'].append(
            'artifact.warnings does not reproduce from the inputs')
    if artifact.get('status') == 'ok' and (
            artifact.get('problems') or artifact.get('warnings')):
        out['problems'].append(
            'artifact is marked ok but carries problems/warnings')
    out['warnings'].extend(artifact.get('warnings') or [])
    if out['problems']:
        out['status'] = 'fail'
    return out


def write_budget_truncation(outdir, artifact, *, overwrite=False):
    """Atomically persist the artifact; refuse silent replacement."""
    if not isinstance(outdir, str) or not outdir.strip():
        raise ValueError('outdir must be a non-empty path')
    verdict = validate_budget_truncation(artifact)
    path = os.path.join(outdir, 'budget_truncation.json')
    out = {'kind': 'write_budget_truncation', 'path': path, 'written': False,
           'unchanged': False, 'conflict': False, 'status': verdict['status'],
           'artifact_status': artifact.get('status'),
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
    return {
        'kind': 'budget_truncation_summary',
        'status': artifact.get('status'),
        'ok': artifact.get('status') == 'ok',
        'content_sha256': artifact.get('content_sha256'),
        'manifest_canonical_sha256': (artifact.get('study') or {}).get(
            'manifest_canonical_sha256'),
        'n_cells': (artifact.get('design') or {}).get('n_cells'),
        'required_prefix_cells': (artifact.get('cost') or {}).get(
            'required_prefix_cells'),
        'effective_required_seeds': (artifact.get('floor') or {}).get(
            'effective_required_seeds'),
        'declared_min_seeds': (artifact.get('floor') or {}).get(
            'declared_min_seeds'),
        'budget_attempts': (artifact.get('budget') or {}).get(
            'declared_attempts'),
        'required_attempts_for_risk': (artifact.get('cost') or {}).get(
            'required_attempts_for_risk'),
        'p_required_prefix_within_budget': (artifact.get('budget') or {}).get(
            'p_required_prefix_within_budget'),
    }
