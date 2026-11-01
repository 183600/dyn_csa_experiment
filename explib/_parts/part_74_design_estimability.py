"""Contrast estimability and design-rank audit (design preparation, no runs).

Every downstream statistical gate in this repository assumes that the
comparison it tests is *identified* by the cells that were actually planned.
That assumption is usually left implicit:

* a pairwise contrast is estimable only if its arms are connected through
  blocks (a seed/device/order block that contains both arms).  Two arms that
  never share a block are perfectly confounded with the block factor, no
  matter how many cells each has;
* a multi-arm contrast is estimable only if its weights sum to zero within
  every connected component of the block-arm incidence graph;
* a factorial main effect or interaction is estimable only if its effect
  columns lie in the column space of the realised design matrix.  Empty
  cells, a factor that never varies, or two factors with the same partition
  of arms can make an interaction or a main effect non-estimable even when
  every arm has many seeds.

Existing gates check *declarations* (arms/seeds exist, single-factor key
sets, factorial completeness) and *allocation balance* (arm not collinear
with one environment).  They do not build the design matrix and ask whether
the declared linear contrast is in its column space.  A design can therefore
pass every declaration gate, be fully "randomised" on paper, and still have
no estimate for the contrast it intends to report.

This fragment adds a zero-GPU, no-split design-preparation audit:

* normalise planned cells ``(arm, block, factors)`` from a run-order plan, a
  manifest arm x seed grid, or an explicit cell list;
* build the cell-mean model matrix ``X`` (intercept, block dummies, arm
  dummies) with exact rational arithmetic, so rank/deficiency is a property
  of the design and not of a floating-point tolerance;
* decompose the block-arm incidence graph into connected components and
  prove each declared contrast is (or is not) estimable, naming the
  component whose weight does not cancel;
* optionally analyse declared factors: main effects and pairwise
  interactions are coded as sum-to-zero effect columns, lifted to cells, and
  tested against the column space of ``X``; empty level combinations,
  aliased factors and redundant terms are reported;
* bind the result with ``content_sha256`` so a stored artifact can be
  replayed from its inputs.

The audit never reads a data split, never trains, never produces a p-value
and never changes an arm, seed, endpoint, alpha, Holm slot or frozen verdict.
A non-estimable contrast is a **design** failure: it is not a null result and
cannot be repaired by adding seeds within the same disconnected block.
"""

import collections
import copy
import hashlib
import itertools
import json
import math
import numbers
import os
import time
from fractions import Fraction


DESIGN_ESTIMABILITY_KIND = 'design_estimability'
DESIGN_ESTIMABILITY_SCHEMA = 1
DESIGN_ESTIMABILITY_POLICY = {
    'version': 1,
    'max_cells': 4096,
    'max_arms': 256,
    'max_blocks': 512,
    'max_factors': 8,
    'max_levels': 32,
    'max_comparisons': 256,
    'max_terms': 64,
    'default_block_key': 'block',
    'role': (
        'design-preparation audit of contrast estimability: build the exact '
        'cell-mean design matrix and the block-arm incidence graph, then ask '
        'whether every declared contrast and every declared factor term is '
        'in the column space of the planned design.'),
    'rule': (
        'a declared comparison is estimable only when its arm weights cancel '
        'inside every connected component of the block-arm incidence graph '
        '(equivalently, when the lifted contrast lies in the column space of '
        'the exact design matrix).  A non-estimable declared comparison is a '
        'design failure, not a null result.'),
    'boundary': (
        'the audit describes linear estimability under the declared '
        'cell-mean model.  It does not estimate variance, power, effect '
        'size or a p-value, does not choose a better design, and does not '
        'prove that an estimable contrast will be statistically resolvable.  '
        'It is a necessary condition, not a substitute for the power, '
        'attrition, budget and rehearsal gates.'),
}

# Rows x columns above which the exact rational elimination is replaced by the
# equivalent closed-form rank for the same reference-coded design.  This is a
# performance threshold, not part of the artifact policy/hash: the closed form
# is the same integer for every design the policy admits.
_DE_EXACT_RREF_WORK_BUDGET = 250000

_RESERVED_CELL_KEYS = frozenset({
    'arm', 'variant', 'a', 'block', 'seed', 'b', 'factors', 'name',
    'label', 'id', 'cell',
})


def _de_is_bool(value):
    return isinstance(value, bool)


def _de_text(value):
    if value is None or _de_is_bool(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, numbers.Integral):
        return str(int(value))
    return None


def _de_int(value, *, minimum=None):
    if _de_is_bool(value):
        return None
    if isinstance(value, numbers.Integral):
        out = int(value)
    elif isinstance(value, numbers.Real):
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


def _de_fraction(value, *, field='value'):
    if _de_is_bool(value):
        raise ValueError(f'{field} must be a finite number, got a boolean')
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f'{field} must be a finite number')
        try:
            return Fraction(text)
        except (ValueError, ZeroDivisionError):
            raise ValueError(
                f'{field} must be a finite number, got {value!r}') from None
    if isinstance(value, numbers.Integral):
        return Fraction(int(value))
    if isinstance(value, numbers.Real):
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f'{field} must be a finite number') from None
        if not math.isfinite(number):
            raise ValueError(f'{field} must be finite, got {value!r}')
        return Fraction(str(number))
    raise ValueError(f'{field} must be a finite number, got {value!r}')


def _de_canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False)


def _de_sha_obj(obj):
    return hashlib.sha256(_de_canonical(obj).encode('utf-8')).hexdigest()


def _de_manifest_canonical_sha(manifest):
    if not isinstance(manifest, dict):
        return None
    body = {k: v for k, v in manifest.items()
            if k not in ('_path', 'manifest_sha256')}
    try:
        return _de_sha_obj(body)
    except (TypeError, ValueError, RecursionError):
        return None


def _de_fraction_text(value):
    if value.denominator == 1:
        return str(value.numerator)
    return f'{value.numerator}/{value.denominator}'


def _de_float(value):
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None


# ---------------------------------------------------------------------------
# Exact rational linear algebra
# ---------------------------------------------------------------------------

def _de_rref(rows, n_cols):
    """Return ``(rref_rows, pivot_columns)`` for a small exact matrix."""
    work = [[Fraction(x) for x in row] for row in rows]
    if not work:
        return [], []
    width = max(n_cols, max(len(row) for row in work))
    for row in work:
        if len(row) < width:
            row.extend([Fraction(0)] * (width - len(row)))
    pivots = []
    rank = 0
    for col in range(width):
        pivot = None
        for i in range(rank, len(work)):
            if work[i][col] != 0:
                pivot = i
                break
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        pv = work[rank][col]
        if pv != 1:
            work[rank] = [x / pv for x in work[rank]]
        for i in range(len(work)):
            if i == rank:
                continue
            factor = work[i][col]
            if factor != 0:
                work[i] = [x - factor * y
                           for x, y in zip(work[i], work[rank])]
        pivots.append(col)
        rank += 1
        if rank == len(work):
            break
    return work, pivots


def _de_rows_from_columns(columns, n_rows):
    """Transpose a list of column vectors into a list of row vectors."""
    if not columns:
        return []
    return [[Fraction(col[i]) for col in columns] for i in range(n_rows)]


def _de_rank(columns, n_rows):
    if not columns:
        return 0
    rows = _de_rows_from_columns(columns, n_rows)
    _, pivots = _de_rref(rows, len(columns))
    return len(pivots)


def _de_in_column_space(columns, vector, *, base_rank=None):
    """Whether ``vector`` is in the span of ``columns`` (exact).

    ``base_rank`` lets a caller that already computed the rank of the same
    design matrix reuse it.  A contrast check otherwise repeats that exact
    elimination for every declared comparison, which is the dominant cost on
    a large arm x seed grid.
    """
    if not columns:
        return all(Fraction(x) == 0 for x in vector)
    n_rows = len(vector)
    rows = _de_rows_from_columns(columns, n_rows)
    aug = [list(row) + [Fraction(vector[i])] for i, row in enumerate(rows)]
    _, p_aug = _de_rref(aug, len(columns) + 1)
    rank_aug = len(p_aug)
    if base_rank is None:
        _, p_base = _de_rref(rows, len(columns))
        base_rank = len(p_base)
    return rank_aug == base_rank


def _de_columns_rank(columns, n_rows):
    return _de_rank(columns, n_rows)


def _de_span_rank_with(columns, extra, n_rows):
    return _de_columns_rank(list(columns) + list(extra), n_rows)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _de_normalise_cell(cell, index):
    if isinstance(cell, dict):
        arm = _de_text(cell.get('arm', cell.get('variant', cell.get('a'))))
        block = _de_text(cell.get('block', cell.get('seed', cell.get('b'))))
        raw_factors = cell.get('factors')
        if raw_factors is None:
            raw_factors = {
                key: value for key, value in cell.items()
                if key not in _RESERVED_CELL_KEYS
            }
        elif not isinstance(raw_factors, dict):
            raise ValueError(
                f'cell {index}: factors must be an object when present')
    elif isinstance(cell, (list, tuple)):
        if len(cell) < 2:
            raise ValueError(
                f'cell {index}: a cell needs at least (arm, block)')
        arm = _de_text(cell[0])
        block = _de_text(cell[1])
        raw_factors = cell[2] if len(cell) >= 3 else {}
        if raw_factors is None:
            raw_factors = {}
        if not isinstance(raw_factors, dict):
            raise ValueError(
                f'cell {index}: the third tuple element must be a factors '
                'object when present')
    else:
        raise ValueError(
            f'cell {index}: expected an object or (arm, block[, factors]), '
            f'got {type(cell).__name__}')
    if arm is None:
        raise ValueError(f'cell {index}: arm is missing/empty')
    if block is None:
        raise ValueError(f'cell {index}: block is missing/empty')
    factors = {}
    for key, value in raw_factors.items():
        name = _de_text(key)
        level = _de_text(value)
        if name is None:
            raise ValueError(f'cell {index}: factor name is missing/empty')
        if level is None:
            raise ValueError(
                f'cell {index}: factor {name!r} has an empty level')
        factors[name] = level
    return {'arm': arm, 'block': block, 'factors': factors}


def _de_cells_from_run_order(run_order):
    if not isinstance(run_order, dict):
        raise ValueError('run_order must be an object')
    planned = run_order.get('planned_cells')
    variants = run_order.get('cell_variants')
    blocks = run_order.get('cell_blocks')
    if not isinstance(planned, (list, tuple)) or not planned:
        raise ValueError('run_order.planned_cells must be a non-empty list')
    if len(planned) > DESIGN_ESTIMABILITY_POLICY['max_cells']:
        # Guard before materialising a cell object per label: a malformed or
        # hostile run order must be rejected in bounded memory.
        raise ValueError(
            f'run_order.planned_cells has {len(planned)} entries, above the '
            f'max_cells guard {DESIGN_ESTIMABILITY_POLICY["max_cells"]}')
    if not isinstance(variants, dict) or not isinstance(blocks, dict):
        raise ValueError(
            'run_order.cell_variants/cell_blocks must be objects')

    def _label_map(mapping, field):
        # JSON object keys are strings, but a direct Python caller may pass
        # integer labels in both ``planned_cells`` and the maps.  Normalise
        # them through one representation and refuse a collision rather than
        # silently letting one entry shadow another.
        out = {}
        for key, value in mapping.items():
            text = _de_text(key)
            if text is None:
                raise ValueError(f'run_order.{field} has an empty label key')
            if text in out:
                raise ValueError(
                    f'run_order.{field} repeats the label {text!r} after '
                    'normalisation')
            out[text] = value
        return out

    variants = _label_map(variants, 'cell_variants')
    blocks = _label_map(blocks, 'cell_blocks')
    cells = []
    for label in planned:
        label = _de_text(label)
        if label is None:
            raise ValueError('run_order.planned_cells contains an empty label')
        if label not in variants:
            raise ValueError(
                f'run_order.planned_cells contains {label!r} without a '
                'cell_variants entry')
        if label not in blocks:
            raise ValueError(
                f'run_order.planned_cells contains {label!r} without a '
                'cell_blocks entry')
        cells.append({'arm': variants[label], 'block': blocks[label]})
    return cells


def _de_normalise_cells(cells):
    raw = cells
    if isinstance(raw, dict) and 'planned_cells' in raw:
        raw = _de_cells_from_run_order(raw)
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError('cells must be a non-empty list/tuple or run-order '
                         'object')
    if len(raw) > DESIGN_ESTIMABILITY_POLICY['max_cells']:
        raise ValueError(
            f'cells has {len(raw)} entries, above the max_cells guard '
            f'{DESIGN_ESTIMABILITY_POLICY["max_cells"]}')
    out = []
    seen = set()
    for index, cell in enumerate(raw):
        norm = _de_normalise_cell(cell, index)
        key = (norm['arm'], norm['block'],
               tuple(sorted(norm['factors'].items())))
        if key in seen:
            raise ValueError(
                f'cell {index}: duplicate (arm={norm["arm"]!r}, '
                f'block={norm["block"]!r}, factors={norm["factors"]!r})')
        seen.add(key)
        out.append(norm)
    return out


def _de_normalise_comparison(raw, index):
    if not isinstance(raw, dict):
        raise ValueError(
            f'comparison {index}: expected an object, got '
            f'{type(raw).__name__}')
    name = _de_text(raw.get('name'))
    weights = raw.get('weights')
    a = _de_text(raw.get('a', raw.get('with', raw.get('arm_a'))))
    b = _de_text(raw.get('b', raw.get('without', raw.get('arm_b'))))
    if weights is None:
        if a is None or b is None:
            raise ValueError(
                f'comparison {index}: needs a and b, or an explicit weights '
                'mapping')
        if a == b:
            raise ValueError(
                f'comparison {index}: a and b must name two different arms')
        weights = {a: Fraction(1), b: Fraction(-1)}
        if name is None:
            name = f'{a} - {b}'
    else:
        if not isinstance(weights, dict) or not weights:
            raise ValueError(
                f'comparison {index}: weights must be a non-empty object')
        parsed = {}
        for key, value in weights.items():
            arm = _de_text(key)
            if arm is None:
                raise ValueError(
                    f'comparison {index}: weight arm is missing/empty')
            try:
                weight = _de_fraction(value, field=f'weight[{arm!r}]')
            except ValueError as exc:
                raise ValueError(
                    f'comparison {index}: {exc}') from None
            if weight == 0:
                continue
            parsed[arm] = weight
        if not parsed:
            raise ValueError(
                f'comparison {index}: every weight is zero')
        # ``a``/``b`` are the human-readable pair labels used for the
        # component diagnostics.  A weighted multi-arm contrast may omit
        # them, but a half-pair would make the artifact ambiguous.  If a label
        # is not one of the nonzero weights, keep it at zero so the build's
        # "arm not planned" check can name it instead of silently reporting an
        # unrelated pair as estimable.
        if (a is None) != (b is None):
            raise ValueError(
                f'comparison {index}: a and b must be given together, or '
                'both omitted for an explicit multi-arm weights mapping')
        if a is not None:
            if a == b:
                raise ValueError(
                    f'comparison {index}: a and b must name two different '
                    'arms')
            for label in (a, b):
                if label not in parsed:
                    parsed[label] = Fraction(0)
        weights = parsed
        if name is None:
            name = ' + '.join(
                f'{_de_fraction_text(w)}*{arm}'
                for arm, w in sorted(weights.items()) if w != 0)
    if name is None:
        raise ValueError(f'comparison {index}: name is missing/empty')
    total = sum(weights.values(), Fraction(0))
    return {
        'name': name,
        'a': a,
        'b': b,
        'weights': {arm: weights[arm] for arm in sorted(weights)},
        'sum_weights': total,
        'is_contrast': total == 0,
    }


def _de_normalise_comparisons(comparisons):
    if isinstance(comparisons, dict):
        if 'comparisons' in comparisons:
            comparisons = comparisons['comparisons']
        elif 'primary_comparisons' in comparisons:
            comparisons = comparisons['primary_comparisons']
    if not isinstance(comparisons, (list, tuple)) or not comparisons:
        raise ValueError('comparisons must be a non-empty list/tuple')
    if len(comparisons) > DESIGN_ESTIMABILITY_POLICY['max_comparisons']:
        raise ValueError(
            f'comparisons has {len(comparisons)} entries, above the '
            f'max_comparisons guard '
            f'{DESIGN_ESTIMABILITY_POLICY["max_comparisons"]}')
    out = []
    names = set()
    for index, raw in enumerate(comparisons):
        norm = _de_normalise_comparison(raw, index)
        if norm['name'] in names:
            raise ValueError(
                f'comparison {index}: duplicate name {norm["name"]!r}')
        names.add(norm['name'])
        out.append(norm)
    return out


def _de_normalise_factors(factors, cells, shape='auto'):
    """Return ``(arm_factors, warnings, resolved_shape)`` for a declaration."""
    warnings = []
    arm_factors = {}
    if shape not in ('auto', 'factor_to_arm', 'arm_to_factor'):
        raise ValueError(
            "factors_shape must be 'auto', 'factor_to_arm' or "
            "'arm_to_factor'")
    if isinstance(factors, dict) and not factors:
        # An explicit empty declaration means "no factor terms"; it must still
        # return the same three-field contract as every other path (the
        # caller unpacks arm_factors, warnings and the resolved shape).
        return {}, warnings, shape
    if factors is not None:
        if not isinstance(factors, dict):
            raise ValueError('factors must be an object mapping factor -> '
                             '{arm: level} or arm -> {factor: level}')
        # Two accepted shapes:
        #   {factor: {arm: level}}  (preferred for declarations)
        #   {arm: {factor: level}}  (canonical for the stored inputs)
        # ``shape`` disambiguates when an arm and a factor share a name.
        if shape not in ('auto', 'factor_to_arm', 'arm_to_factor'):
            raise ValueError(
                "factors_shape must be 'auto', 'factor_to_arm' or "
                "'arm_to_factor'")
        if shape == 'auto':
            arm_names = {cell['arm'] for cell in cells}
            factor_shape = all(
                isinstance(value, dict)
                and all(not isinstance(v, dict) for v in value.values())
                for value in factors.values())
            arm_shape = all(
                isinstance(value, dict)
                and all(isinstance(v, dict) for v in value.values())
                for value in factors.values())
            if factors and set(factors) <= arm_names and all(
                    isinstance(value, dict) for value in factors.values()):
                # Every outer key is an observed arm: the arm -> {factor:
                # level} shape.  This also resolves the structural ambiguity
                # when a factor happens to share a name with an arm.
                shape = 'arm_to_factor'
            elif factor_shape and arm_shape:
                shape = 'factor_to_arm'
            elif arm_shape:
                shape = 'arm_to_factor'
            elif factor_shape:
                shape = 'factor_to_arm'
            else:
                raise ValueError(
                    'factors must be EITHER factor -> {arm: level} OR '
                    'arm -> {factor: level}, not a mixture')
        if shape == 'factor_to_arm':
            declared_arms = {cell['arm'] for cell in cells}
            for factor, mapping in factors.items():
                fname = _de_text(factor)
                if fname is None:
                    raise ValueError('factor name is missing/empty')
                if not isinstance(mapping, dict) or not mapping:
                    raise ValueError(
                        f'factor {fname!r}: mapping must be a non-empty '
                        'object of arm -> level')
                for arm, level in mapping.items():
                    arm_name = _de_text(arm)
                    lvl = _de_text(level)
                    if arm_name is None or lvl is None:
                        raise ValueError(
                            f'factor {fname!r}: arm/level must be non-empty')
                    if arm_name not in declared_arms:
                        raise ValueError(
                            f'factor {fname!r}: arm {arm_name!r} is not a '
                            'planned cell arm')
                    prior = arm_factors.setdefault(arm_name, {}).get(fname)
                    if prior is not None and prior != lvl:
                        raise ValueError(
                            f'factor {fname!r}: arm {arm_name!r} has '
                            f'conflicting levels {prior!r} vs {lvl!r}')
                    arm_factors[arm_name][fname] = lvl
        elif shape == 'arm_to_factor':
            declared_arms = {cell['arm'] for cell in cells}
            for arm, mapping in factors.items():
                arm_name = _de_text(arm)
                if arm_name is None:
                    raise ValueError('factor mapping arm is missing/empty')
                if arm_name not in declared_arms:
                    raise ValueError(
                        f'factor mapping arm {arm_name!r} is not a planned '
                        'cell arm')
                if not isinstance(mapping, dict) or not mapping:
                    raise ValueError(
                        f'factor mapping for {arm_name!r} must be a '
                        'non-empty object of factor -> level')
                for factor, level in mapping.items():
                    fname = _de_text(factor)
                    lvl = _de_text(level)
                    if fname is None or lvl is None:
                        raise ValueError(
                            f'factor mapping for {arm_name!r}: factor/level '
                            'must be non-empty')
                    prior = arm_factors.setdefault(arm_name, {}).get(fname)
                    if prior is not None and prior != lvl:
                        raise ValueError(
                            f'factor mapping for {arm_name!r}: factor '
                            f'{fname!r} has conflicting levels {prior!r} vs '
                            f'{lvl!r}')
                    arm_factors[arm_name][fname] = lvl
        else:  # pragma: no cover - shape is validated above
            raise ValueError(
                'factors must be EITHER factor -> {arm: level} OR '
                'arm -> {factor: level}, not a mixture')
    else:
        # Derive whatever the cells carry.  A factor must be declared by every
        # arm that carries any factor at all; partial factor coverage would
        # silently create an empty level.
        for cell in cells:
            arm = cell['arm']
            for factor, level in cell['factors'].items():
                prior = arm_factors.setdefault(arm, {}).get(factor)
                if prior is not None and prior != level:
                    raise ValueError(
                        f'arm {arm!r} has conflicting levels for factor '
                        f'{factor!r}: {prior!r} vs {level!r}')
                arm_factors.setdefault(arm, {})[factor] = level
    if not arm_factors:
        return {}, warnings, shape
    factor_names = sorted({f for mapping in arm_factors.values()
                           for f in mapping})
    if len(factor_names) > DESIGN_ESTIMABILITY_POLICY['max_factors']:
        raise ValueError(
            f'{len(factor_names)} factors exceeds the max_factors guard '
            f'{DESIGN_ESTIMABILITY_POLICY["max_factors"]}')
    arms = sorted({cell['arm'] for cell in cells})
    for factor in factor_names:
        missing = [arm for arm in arms if factor not in arm_factors.get(arm, {})]
        if missing:
            raise ValueError(
                f'factor {factor!r} is not declared for arm(s): '
                + ', '.join(missing))
        levels = sorted({arm_factors[arm][factor] for arm in arms})
        if len(levels) > DESIGN_ESTIMABILITY_POLICY['max_levels']:
            raise ValueError(
                f'factor {factor!r} has {len(levels)} levels, above the '
                f'max_levels guard {DESIGN_ESTIMABILITY_POLICY["max_levels"]}')
        if len(levels) < 2:
            warnings.append(
                f'factor {factor!r} has only one level ({levels[0]!r}); its '
                'main effect is not defined and it cannot be tested')
    return arm_factors, warnings, shape


# ---------------------------------------------------------------------------
# Design matrix and graph
# ---------------------------------------------------------------------------

def _de_reference_columns(cells):
    """Build intercept + block dummies + arm dummies (reference coding)."""
    arms = sorted({cell['arm'] for cell in cells})
    blocks = sorted({cell['block'] for cell in cells})
    n_rows = len(cells)
    columns = []
    # Intercept.
    columns.append([Fraction(1)] * n_rows)
    # Block dummies, first block as reference.
    for block in blocks[1:]:
        columns.append([Fraction(1) if cell['block'] == block else Fraction(0)
                        for cell in cells])
    # Arm dummies, first arm as reference.
    for arm in arms[1:]:
        columns.append([Fraction(1) if cell['arm'] == arm else Fraction(0)
                        for cell in cells])
    return columns, arms, blocks


def _de_components(cells):
    """Connected components of the block-arm incidence graph."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    arms = sorted({cell['arm'] for cell in cells})
    for arm in arms:
        find(('arm', arm))
    by_block = collections.defaultdict(list)
    for cell in cells:
        by_block[cell['block']].append(cell['arm'])
    for block, block_arms in by_block.items():
        find(('block', block))
        first = block_arms[0]
        for arm in block_arms[1:]:
            union(('arm', first), ('arm', arm))
        union(('block', block), ('arm', first))
    groups = collections.defaultdict(list)
    for arm in arms:
        groups[find(('arm', arm))].append(arm)
    return [sorted(group) for group in sorted(groups.values(),
                                              key=lambda g: (len(g), sorted(g)))]


def _de_arm_cell_counts(cells):
    out = collections.Counter(cell['arm'] for cell in cells)
    return {arm: int(count) for arm, count in sorted(out.items())}


def _de_lift_arm_weights(weights, counts):
    """Spread an arm-level weight evenly over that arm's cells."""
    lifted = {}
    for arm, weight in weights.items():
        n = counts.get(arm, 0)
        if n <= 0:
            return None
        lifted[arm] = Fraction(weight) / n
    return lifted


def _de_lift_vector(vector, arms, counts):
    out = []
    for arm in arms:
        n = counts.get(arm, 0)
        if n <= 0:
            return None
        out.append(Fraction(vector.get(arm, 0)) / n)
    return out


def _de_lift_to_cells(arm_vector, cells, arms, counts):
    """Lift an arm-level vector to the cell space (each arm evenly split)."""
    out = []
    for cell in cells:
        n = counts.get(cell['arm'], 0)
        if n <= 0:
            return None
        out.append(Fraction(arm_vector.get(cell['arm'], 0)) / n)
    return out


# ---------------------------------------------------------------------------
# Factor terms
# ---------------------------------------------------------------------------

def _de_factor_contrast_vectors(arm_factors, arms, factor):
    levels = sorted({arm_factors[arm][factor] for arm in arms})
    if len(levels) < 2:
        return levels, []
    reference = levels[-1]
    columns = []
    for level in levels[:-1]:
        column = {}
        for arm in arms:
            value = arm_factors[arm][factor]
            if value == level:
                column[arm] = Fraction(1)
            elif value == reference:
                column[arm] = Fraction(-1)
            else:
                column[arm] = Fraction(0)
        columns.append(column)
    return levels, columns


def _de_term_columns(arm_factors, arms, term):
    """Effect-coded columns spanning the requested term's subspace."""
    by_factor = []
    for factor in term:
        levels, columns = _de_factor_contrast_vectors(
            arm_factors, arms, factor)
        by_factor.append((factor, levels, columns))
    if any(not columns for (_, _, columns) in by_factor):
        return []
    out = []
    for combo in itertools.product(*[entry[2] for entry in by_factor]):
        column = {}
        for arm in arms:
            value = Fraction(1)
            for part in combo:
                value *= part.get(arm, Fraction(0))
            column[arm] = value
        if any(value != 0 for value in column.values()):
            out.append(column)
    return out


def _de_term_label(term):
    return 'main:' + term[0] if len(term) == 1 else \
        'interaction:' + '|'.join(term)


def _de_base_columns(cells, model):
    """Intercept + block dummies; the treatment model starts from here."""
    columns, _, blocks = _de_reference_columns(cells)
    if model == 'cell_means':
        return [columns[0]]
    return columns[:1 + max(0, len(blocks) - 1)]


def _de_analyse_factors(cells, arms, counts, columns, arm_factors, model):
    """Test main effects and pairwise interactions in the treatment model.

    The saturated arm-dummy model answers "is this arm mean observed?"; it
    cannot answer "is this factor effect separately identified from the other
    declared terms?", because every treatment contrast lies in the arm-dummy
    span.  The treatment model here is therefore intercept + block dummies +
    effect-coded factor terms only.  A term is uniquely estimable when
    removing it lowers the rank, i.e. it contributes df that no other declared
    term supplies.
    """
    factor_names = sorted({f for mapping in arm_factors.values()
                           for f in mapping})
    if not factor_names:
        return {'factors': {}, 'terms': {}, 'redundant_pairs': [],
                'aliased_pairs': [], 'empty_combinations': {},
                'warnings': [], 'problems': [], 'treatment_model': {}}
    warnings = []
    problems = []
    levels = {factor: sorted({arm_factors[arm][factor] for arm in arms})
              for factor in factor_names}
    factors_out = {}
    for factor in factor_names:
        by_level = collections.defaultdict(list)
        for arm in arms:
            by_level[arm_factors[arm][factor]].append(arm)
        factors_out[factor] = {
            'levels': list(levels[factor]),
            'arms_by_level': {level: sorted(group)
                              for level, group in sorted(by_level.items())},
            'n_levels': len(levels[factor]),
        }
    # Pairwise empty combinations are the most interpretable sign that an
    # interaction is not identified.
    empty_pairs = {}
    for i, f1 in enumerate(factor_names):
        for f2 in factor_names[i + 1:]:
            present = {(arm_factors[arm][f1], arm_factors[arm][f2])
                       for arm in arms}
            expected = {(l1, l2) for l1 in levels[f1] for l2 in levels[f2]}
            missing = sorted(expected - present)
            empty_pairs[f'{f1}|{f2}'] = [
                {'levels': [a, b]} for a, b in missing
            ]
            if missing:
                warnings.append(
                    f'empty cell(s) for {f1!r} x {f2!r}: '
                    + ', '.join(f'{a}={b}' for a, b in missing))
    # Identical partitions of arms make two main effects perfectly aliased.
    signatures = {}
    for factor in factor_names:
        groups = frozenset(
            frozenset(group) for group in
            factors_out[factor]['arms_by_level'].values())
        signatures.setdefault(groups, []).append(factor)
    aliased_pairs = []
    for group in signatures.values():
        for f1, f2 in itertools.combinations(sorted(group), 2):
            aliased_pairs.append({'terms': [
                f'main:{f1}', f'main:{f2}'], 'kind': 'identical_partition'})
            warnings.append(
                f'factors {f1!r} and {f2!r} induce the same partition of '
                'arms; their main effects are not separately estimable')
    # Declared terms: all main effects and all pairwise interactions.
    term_specs = [(factor,) for factor in factor_names]
    for f1, f2 in itertools.combinations(factor_names, 2):
        term_specs.append((f1, f2))
    if len(term_specs) > DESIGN_ESTIMABILITY_POLICY['max_terms']:
        raise ValueError(
            f'{len(term_specs)} factor terms exceeds the max_terms guard '
            f'{DESIGN_ESTIMABILITY_POLICY["max_terms"]}')
    base_columns = _de_base_columns(cells, model)
    n_rows = len(cells)
    # The treatment model's base has a closed-form full rank: intercept plus
    # one dummy per non-reference block (each block has a cell), or just the
    # intercept for cell_means.  This avoids one exact elimination per run of
    # the factor audit.
    if model == 'cell_means':
        rank_base = 1
    else:
        rank_base = len({cell['block'] for cell in cells})
    term_columns = {}
    term_df = {}
    for term in term_specs:
        label = _de_term_label(term)
        raw_columns = _de_term_columns(arm_factors, arms, term)
        lifted = []
        for column in raw_columns:
            vector = _de_lift_to_cells(column, cells, arms, counts)
            if vector is None:
                vector = [Fraction(0)] * n_rows
            lifted.append(vector)
        term_columns[label] = lifted
        term_df[label] = len(lifted)
    all_columns = []
    for label in sorted(term_columns):
        all_columns.extend(term_columns[label])
    full_rank = _de_columns_rank(base_columns + all_columns, n_rows)
    full_columns = len(base_columns) + len(all_columns)
    treatment_deficiency = full_columns - full_rank
    if treatment_deficiency > 0:
        warnings.append(
            f'the declared treatment model has {treatment_deficiency} '
            'degree(s) of freedom less than its parameter count; at least '
            'one declared term is not separately estimable')
    terms_out = {}
    for term in term_specs:
        label = _de_term_label(term)
        cols_i = term_columns[label]
        df_i = term_df[label]
        if df_i == 0:
            terms_out[label] = {
                'term': list(term),
                'kind': 'main' if len(term) == 1 else 'interaction',
                'df': 0,
                'unique_df': 0,
                'own_df': 0,
                'rank': 0,
                'estimable_columns': 0,
                'estimable': False,
                'separately_estimable': False,
                'partial': False,
                'status': 'degenerate',
                'reason': 'a factor in this term has fewer than two levels',
                'levels': {factor: list(levels[factor]) for factor in term},
            }
            continue
        columns_without = []
        for other in sorted(term_columns):
            if other != label:
                columns_without.extend(term_columns[other])
        rank_without = _de_columns_rank(
            base_columns + columns_without, n_rows)
        unique_df = full_rank - rank_without
        own_rank = _de_columns_rank(base_columns + cols_i, n_rows) - rank_base
        estimable = own_rank == df_i
        separately_estimable = unique_df == df_i
        partial = (not estimable) and unique_df > 0
        if separately_estimable:
            status = 'estimable'
        elif estimable:
            status = 'aliased'
        elif partial:
            status = 'partial'
        else:
            status = 'not_estimable'
        if not separately_estimable:
            warnings.append(
                f'{label}: unique df={unique_df} of {df_i}; the full declared '
                'term is not separately identified from the other declared '
                'terms' + ('' if estimable else
                            ' (the term is not even in the design span)'))
        terms_out[label] = {
            'term': list(term),
            'kind': 'main' if len(term) == 1 else 'interaction',
            'df': df_i,
            'unique_df': int(unique_df),
            'own_df': int(own_rank),
            'rank': int(unique_df),
            'estimable_columns': int(own_rank),
            'estimable': bool(estimable),
            'separately_estimable': bool(separately_estimable),
            'partial': bool(partial),
            'status': status,
            'levels': {factor: list(levels[factor]) for factor in term},
        }
    # Symmetric redundancy: two terms whose combined span is smaller than
    # the sum of their individual spans.  This is the aliasing/overlap that
    # makes a pair of claims inseparable.
    redundant_pairs = []
    labels = sorted(terms_out)
    for i, l1 in enumerate(labels):
        for l2 in labels[i + 1:]:
            df1 = term_df.get(l1, 0)
            df2 = term_df.get(l2, 0)
            if df1 == 0 or df2 == 0:
                continue
            rank_1 = _de_columns_rank(
                base_columns + term_columns[l1], n_rows) - rank_base
            rank_2 = _de_columns_rank(
                base_columns + term_columns[l2], n_rows) - rank_base
            rank_pair = _de_columns_rank(
                base_columns + term_columns[l1] + term_columns[l2],
                n_rows) - rank_base
            overlap = rank_1 + rank_2 - rank_pair
            if overlap > 0:
                redundant_pairs.append({
                    'terms': [l1, l2],
                    'overlap_df': int(overlap),
                    'kind': 'shared_effect_space',
                })
                warnings.append(
                    f'{l1} and {l2} overlap by {overlap} df; the two claims '
                    'are not separately estimated by this design')
    return {
        'factors': factors_out,
        'terms': terms_out,
        'empty_combinations': empty_pairs,
        'aliased_pairs': aliased_pairs,
        'redundant_pairs': redundant_pairs,
        'warnings': warnings,
        'problems': problems,
        'treatment_model': {
            'n_columns': int(full_columns),
            'rank': int(full_rank),
            'deficiency': int(treatment_deficiency),
            'base_columns': int(len(base_columns)),
        },
    }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _de_normalise_model(model):
    text = _de_text(model) or 'block_effects'
    if text not in ('block_effects', 'cell_means'):
        raise ValueError(
            f'model must be "block_effects" or "cell_means", got {text!r}')
    return text


def build_design_estimability(cells, comparisons, *, factors=None,
                              factors_shape='auto', block_key=None,
                              model='block_effects', created_at=None,
                              notes=None):
    """Compile the exact estimability report for one planned design."""
    model = _de_normalise_model(model)
    norm_cells = _de_normalise_cells(cells)
    norm_comparisons = _de_normalise_comparisons(comparisons)
    arms = sorted({cell['arm'] for cell in norm_cells})
    blocks = sorted({cell['block'] for cell in norm_cells})
    counts = _de_arm_cell_counts(norm_cells)
    if len(arms) > DESIGN_ESTIMABILITY_POLICY['max_arms']:
        raise ValueError(
            f'{len(arms)} arms exceeds the max_arms guard '
            f'{DESIGN_ESTIMABILITY_POLICY["max_arms"]}')
    if len(blocks) > DESIGN_ESTIMABILITY_POLICY['max_blocks']:
        raise ValueError(
            f'{len(blocks)} blocks exceeds the max_blocks guard '
            f'{DESIGN_ESTIMABILITY_POLICY["max_blocks"]}')
    components = _de_components(norm_cells)
    # Reference coding: intercept + (blocks-1) block dummies + (arms-1) arm
    # dummies.  cell_means drops the block dummies.
    n_columns = (len(arms) if model == 'cell_means'
                 else len(arms) + len(blocks) - 1)
    # Exact rational elimination is cubic in the column count and holds a
    # dense Fraction row for every cell.  The reference coding has a
    # closed-form rank, so a large but legal grid does not have to materialise
    # that matrix or pay for the same rank twice (once here, once per
    # comparison).  Small designs keep the matrix path, which also
    # cross-checks the formula.
    exact_budget = _DE_EXACT_RREF_WORK_BUDGET
    exact_linear_algebra = (len(norm_cells) * max(1, n_columns)
                            <= exact_budget)
    columns = None
    if exact_linear_algebra:
        columns, _, _ = _de_reference_columns(norm_cells)
        if model == 'cell_means':
            # Keep only the intercept and the arm dummies; drop block dummies.
            columns = [columns[0]] + columns[len(blocks):]
        rank = _de_columns_rank(columns, len(norm_cells))
    elif model == 'cell_means':
        # intercept + all arm dummies, every arm observed
        rank = len(arms)
    else:
        # Rank of the two-way additive incidence matrix: every arm and every
        # block appears in a cell, and each disconnected component contributes
        # exactly one linear dependency.
        rank = len(arms) + len(blocks) - len(components)
    component_of = {}
    for index, component in enumerate(components):
        for arm in component:
            component_of[arm] = index
    # One pass over the cells instead of rescanning the panel per comparison
    # per block.  ``blocks_with_both`` is a diagnostic count, but computing it
    # naively is O(comparisons * blocks * cells) and becomes the dominant cost
    # for a large arm x seed grid.
    arms_by_block = collections.defaultdict(set)
    for cell in norm_cells:
        arms_by_block[cell['block']].add(cell['arm'])

    problems = []
    warnings = []
    arm_factors, factor_warnings, factors_shape_used = _de_normalise_factors(
        factors, norm_cells, factors_shape)
    warnings.extend(factor_warnings)

    # Pairwise and multi-arm comparisons.
    comparisons_out = []
    for comparison in norm_comparisons:
        name = comparison['name']
        weights = comparison['weights']
        missing_arms = sorted(set(weights) - set(arms))
        total = sum(weights.values(), Fraction(0))
        if not comparison['is_contrast']:
            warnings.append(
                f'{name}: weights sum to {_de_fraction_text(total)}, not zero; '
                'this is an arm-level mean combination, not a pure contrast')
        if missing_arms:
            comparisons_out.append({
                'name': name,
                'a': comparison['a'],
                'b': comparison['b'],
                'weights': {arm: _de_float(w)
                            for arm, w in comparison['weights'].items()},
                'sum_weights': _de_float(total),
                'is_contrast': bool(comparison['is_contrast']),
                'estimable': False,
                'reason': 'arm_not_planned',
                'missing_arms': missing_arms,
                'components': {},
                'n_blocks_both': 0,
            })
            problems.append(
                f'{name}: comparison names arm(s) that are not planned: '
                + ', '.join(missing_arms))
            continue
        # Component-wise cancellation is the exact criterion for a block
        # design; the column-space test is an independent cross-check.
        component_sums = {}
        component_totals = []
        for index, component in enumerate(components):
            subtotal = sum(
                (weights.get(arm, Fraction(0)) for arm in component),
                Fraction(0))
            component_sums[str(index)] = _de_float(subtotal)
            component_totals.append(subtotal)
        lift = _de_lift_arm_weights(weights, counts)
        if lift is None:
            comparisons_out.append({
                'name': name,
                'a': comparison['a'],
                'b': comparison['b'],
                'weights': {arm: _de_float(w)
                            for arm, w in comparison['weights'].items()},
                'sum_weights': _de_float(total),
                'is_contrast': bool(comparison['is_contrast']),
                'estimable': False,
                'reason': 'missing_cells',
                'components': component_sums,
                'n_blocks_both': 0,
            })
            problems.append(f'{name}: an arm has no planned cells')
            continue
        cell_vector = [Fraction(0)] * len(norm_cells)
        for index, cell in enumerate(norm_cells):
            cell_vector[index] = lift.get(cell['arm'], Fraction(0))
        if model == 'cell_means':
            graph_ok = True
        else:
            # A block design identifies within-component arm contrasts and one
            # common level.  A combination is estimable when every component
            # carries the same total weight: then the component-specific
            # intercepts cancel (pure contrasts have total zero in each
            # component; a connected design has one component).
            graph_ok = (len(set(component_totals)) <= 1)
        if exact_linear_algebra:
            span_ok = _de_in_column_space(columns, cell_vector, base_rank=rank)
        else:
            # Every comparison is lifted as one constant per arm; intercept +
            # block dummies + arm dummies spans all such vectors (and the
            # cell_means model's intercept + arm dummies does too).  The
            # component criterion above is what can fail on a disconnected
            # design, so the closed-form path does not change any verdict.
            span_ok = True
        estimable = bool(span_ok and graph_ok)
        if not graph_ok:
            if (comparison.get('a') and comparison.get('b')
                    and component_of.get(comparison['a'])
                    != component_of.get(comparison['b'])):
                reason = 'disconnected_components'
            else:
                reason = 'unequal_component_totals'
        elif not span_ok:
            reason = 'not_in_column_space'
        else:
            reason = 'ok'
        blocks_with_both = 0
        if comparison.get('a') and comparison.get('b'):
            pair_arms = {comparison['a'], comparison['b']}
            blocks_with_both = sum(
                1 for block in blocks
                if pair_arms <= arms_by_block.get(block, set()))
        if not estimable:
            if reason == 'disconnected_components':
                problems.append(
                    f'{name}: arms {comparison["a"]!r} and '
                    f'{comparison["b"]!r} never share a block '
                    f'(component_of={component_of.get(comparison["a"])} '
                    f'vs {component_of.get(comparison["b"])})')
            elif reason == 'unequal_component_totals':
                problems.append(
                    f'{name}: arm weights do not have the same total inside '
                    'every connected component of the block-arm graph')
            else:
                problems.append(
                    f'{name}: lifted contrast is not in the column space of '
                    'the planned design matrix')
        comparisons_out.append({
            'name': name,
            'a': comparison['a'],
            'b': comparison['b'],
            'weights': {arm: _de_float(w)
                        for arm, w in comparison['weights'].items()},
            'sum_weights': _de_float(total),
            'is_contrast': bool(comparison['is_contrast']),
            'estimable': estimable,
            'reason': reason,
            'components': component_sums,
            'n_blocks_both': int(blocks_with_both),
        })

    factor_out = _de_analyse_factors(
        norm_cells, arms, counts, columns, arm_factors, model) if arm_factors else {
            'factors': {}, 'terms': {}, 'empty_combinations': {},
            'aliased_pairs': [], 'redundant_pairs': [],
            'warnings': [], 'problems': [], 'treatment_model': {},
        }
    warnings.extend(factor_out.get('warnings') or [])
    problems.extend(factor_out.get('problems') or [])

    # Completeness diagnostics: a missing arm/block pair is a warning unless
    # it makes a declared comparison non-estimable (then it is already a
    # problem above).
    arm_block_pairs = [(cell['arm'], cell['block']) for cell in norm_cells]
    complete_grid = (len(set(arm_block_pairs)) == len(arms) * len(blocks))
    if not complete_grid and model == 'block_effects':
        warnings.append(
            'the planned cells are not a complete arm x block grid; '
            'estimability is checked directly rather than assumed')
    if model == 'cell_means':
        warnings.append(
            'model=cell_means ignores block/seed effects; a contrast that is '
            'estimable here may be confounded with the block factor')

    status = 'ok'
    if problems:
        status = 'fail'
    elif warnings:
        status = 'warnings'
    if not comparisons_out:
        status = 'not_estimable'

    artifact = {
        'kind': DESIGN_ESTIMABILITY_KIND,
        'schema': DESIGN_ESTIMABILITY_SCHEMA,
        'created_at': created_at or time.strftime('%Y-%m-%dT%H:%M:%S'),
        'policy': copy.deepcopy(DESIGN_ESTIMABILITY_POLICY),
        'notes': _de_text(notes),
        'inputs': {
            'cells': norm_cells,
            'comparisons': [
                {
                    'name': c['name'],
                    'a': c['a'],
                    'b': c['b'],
                    'weights': {arm: _de_fraction_text(w)
                                for arm, w in c['weights'].items()},
                }
                for c in norm_comparisons
            ],
            'factors': {arm: dict(mapping)
                        for arm, mapping in sorted(arm_factors.items())},
            # Stored inputs are canonical arm -> {factor: level}; the caller's
            # declaration shape is provenance, not the replay contract.
            'factors_shape': 'arm_to_factor',
            'declared_factors_shape': factors_shape_used,
            'model': model,
            'block_key': _de_text(block_key),
        },
        'design': {
            'n_cells': len(norm_cells),
            'n_arms': len(arms),
            'n_blocks': len(blocks),
            'arms': list(arms),
            'blocks': list(blocks),
            'cells_per_arm': counts,
            'complete_grid': bool(complete_grid),
            'n_columns': int(n_columns),
            'rank': int(rank),
            'deficiency': int(n_columns - rank),
            'components': [list(component) for component in components],
            'component_of': {arm: int(component_of[arm]) for arm in arms},
            'block_effects': model == 'block_effects',
            'treatment_model': factor_out.get('treatment_model') or {},
        },
        'comparisons': comparisons_out,
        'factors': factor_out.get('factors') or {},
        'terms': factor_out.get('terms') or {},
        'empty_combinations': factor_out.get('empty_combinations') or {},
        'aliased_pairs': factor_out.get('aliased_pairs') or [],
        'redundant_pairs': factor_out.get('redundant_pairs') or [],
        'status': status,
        'problems': problems,
        'warnings': warnings,
        'summary': {
            'n_comparisons': len(comparisons_out),
            'n_estimable': sum(1 for c in comparisons_out if c['estimable']),
            'n_not_estimable': sum(
                1 for c in comparisons_out if not c['estimable']),
            'n_terms': len(factor_out.get('terms') or {}),
            'n_terms_not_estimable': sum(
                1 for t in (factor_out.get('terms') or {}).values()
                if not t.get('estimable')),
            'n_terms_not_separately_estimable': sum(
                1 for t in (factor_out.get('terms') or {}).values()
                if not t.get('separately_estimable')),
            'n_components': len(components),
            'n_aliased_pairs': len(factor_out.get('aliased_pairs') or []),
            'n_redundant_pairs': len(factor_out.get('redundant_pairs') or []),
        },
    }
    artifact['content_sha256'] = design_estimability_content_sha(artifact)
    return artifact


def design_estimability_content_sha(artifact):
    body = {k: v for k, v in artifact.items()
            if k not in ('created_at', 'content_sha256')}
    return _de_sha_obj(body)


def replay_design_estimability(artifact):
    if not isinstance(artifact, dict):
        raise TypeError('artifact must be an object')
    inputs = artifact.get('inputs')
    if not isinstance(inputs, dict):
        raise ValueError('artifact.inputs is missing')
    factors = inputs.get('factors') or None
    factors_shape = inputs.get('factors_shape') or 'auto'
    if factors is None:
        # No factor columns: the declared shape is provenance, not a parser
        # instruction, so preserve the original value.
        factors_shape = inputs.get('declared_factors_shape') or 'auto'
    out = build_design_estimability(
        cells=inputs.get('cells'),
        comparisons=inputs.get('comparisons'),
        factors=factors,
        factors_shape=factors_shape,
        block_key=inputs.get('block_key'),
        model=inputs.get('model'),
        created_at=artifact.get('created_at'),
        notes=artifact.get('notes'),
    )
    # Preserve provenance fields that are not inputs to the linear algebra but
    # are part of the artifact identity (and therefore of its hash).
    for key in ('cell_source', 'manifest_sha256',
                'manifest_canonical_sha256', 'declared_factors_shape'):
        if key in inputs:
            out['inputs'][key] = inputs[key]
    out['content_sha256'] = design_estimability_content_sha(out)
    return out


def validate_design_estimability(artifact):
    """Validate structure, body hash and a full deterministic replay."""
    out = {'kind': 'design_estimability_validation', 'status': 'ok',
           'problems': [], 'warnings': [], 'replayed': False}
    if not isinstance(artifact, dict):
        out['problems'].append(
            f'artifact must be an object, got {type(artifact).__name__}')
        out['status'] = 'fail'
        return out
    if artifact.get('kind') != DESIGN_ESTIMABILITY_KIND:
        out['problems'].append(
            f'artifact.kind must be {DESIGN_ESTIMABILITY_KIND!r}')
    if artifact.get('schema') != DESIGN_ESTIMABILITY_SCHEMA:
        out['problems'].append(
            f'artifact.schema must be {DESIGN_ESTIMABILITY_SCHEMA}')
    if artifact.get('status') not in ('ok', 'warnings', 'not_estimable',
                                      'fail'):
        out['problems'].append(
            f'artifact.status={artifact.get("status")!r} is not recognised')
    for field in ('problems', 'warnings'):
        value = artifact.get(field)
        if value is not None and not isinstance(value, (list, tuple)):
            out['problems'].append(
                f'artifact.{field} must be a list, got '
                f'{type(value).__name__}')
    if out['problems']:
        out['status'] = 'fail'
        return out
    try:
        replayed = replay_design_estimability(artifact)
    except Exception as exc:
        out['status'] = 'fail'
        out['problems'].append(
            'replay failed: '
            f'{type(exc).__name__}: {exc}')
        return out
    out['replayed'] = True
    out['replayed_status'] = replayed.get('status')
    got = artifact.get('content_sha256')
    try:
        want = design_estimability_content_sha(artifact)
    except (TypeError, ValueError, RecursionError) as exc:
        # A non-canonical body (NaN/Inf/recursive value) is a failed artifact,
        # not an exception for the caller/CLI to crash on.
        out['status'] = 'fail'
        out['problems'].append(
            'artifact body is not canonical JSON: '
            f'{type(exc).__name__}: {exc}')
        return out
    if got != want:
        out['status'] = 'fail'
        out['problems'].append(
            'artifact.content_sha256 does not match the artifact body')
    # Compare every derived field.  created_at and the hash itself are
    # metadata; notes are provenance and are canonicalised above.
    keys = ('policy', 'design', 'comparisons', 'factors', 'terms',
            'empty_combinations', 'aliased_pairs', 'redundant_pairs',
            'status', 'problems', 'warnings', 'summary')
    for key in keys:
        try:
            same = (_de_canonical(artifact.get(key))
                    == _de_canonical(replayed.get(key)))
        except (TypeError, ValueError, RecursionError) as exc:
            out['status'] = 'fail'
            out['problems'].append(
                f'artifact.{key} is not canonical JSON: '
                f'{type(exc).__name__}: {exc}')
            continue
        if not same:
            out['status'] = 'fail'
            out['problems'].append(
                f'artifact.{key} does not match a replay from inputs')
    return out


def design_estimability_from_manifest(
        manifest, *, run_order=None, cells=None, factors=None,
        factors_shape='auto', block_key=None, model='block_effects',
        created_at=None, notes=None):
    """Compile the audit from a frozen manifest and an optional run order.

    When ``cells``/``run_order`` are absent the canonical complete arm x seed
    grid is used and the artifact records that choice.  The manifest is only
    read for the declared arms, seeds and primary comparisons; no data split
    is touched.
    """
    if not isinstance(manifest, dict):
        raise TypeError('manifest must be an object')
    raw_arms = manifest.get('arms')
    raw_seeds = manifest.get('seeds')
    arms = [str(a) for a in raw_arms] if isinstance(
        raw_arms, (list, tuple)) else []
    seeds = [str(s) for s in raw_seeds] if isinstance(
        raw_seeds, (list, tuple)) else []
    comparisons = []
    raw_comparisons = manifest.get('primary_comparisons')
    if not isinstance(raw_comparisons, (list, tuple)):
        raw_comparisons = []
    for index, raw in enumerate(raw_comparisons):
        if not isinstance(raw, dict):
            raise ValueError(
                f'primary_comparisons[{index}] must be an object, got '
                f'{type(raw).__name__}')
        a = _de_text(raw.get('a'))
        b = _de_text(raw.get('b'))
        weights = raw.get('weights')
        if weights is not None:
            if not isinstance(weights, dict) or not weights:
                raise ValueError(
                    f'primary_comparisons[{index}].weights must be a '
                    'non-empty object')
            comparisons.append({
                'name': _de_text(raw.get('name')) or 'weighted contrast',
                'a': a,
                'b': b,
                'weights': weights,
            })
        elif a is not None and b is not None:
            comparisons.append({
                'name': _de_text(raw.get('name')) or f'{a} - {b}',
                'a': a,
                'b': b,
            })
        else:
            raise ValueError(
                f'primary_comparisons[{index}] needs a and b, or an explicit '
                'weights mapping')
    if not comparisons:
        raise ValueError('the manifest declares no usable primary comparisons')
    source = 'explicit_cells'
    if cells is not None:
        resolved = cells
    elif run_order is not None:
        resolved = run_order
        source = 'registered_run_order'
    else:
        if not arms or not seeds:
            raise ValueError(
                'no cells/run_order were supplied and the manifest does not '
                'declare both arms and seeds')
        n_canonical = len(arms) * len(seeds)
        if n_canonical > DESIGN_ESTIMABILITY_POLICY['max_cells']:
            # Check the cardinality before materialising the grid: a manifest
            # with a huge seed list must not allocate one tuple per cell just
            # to be rejected by the same guard a moment later.
            raise ValueError(
                f'the canonical arm x seed grid has {n_canonical} cells, '
                f'above the max_cells guard '
                f'{DESIGN_ESTIMABILITY_POLICY["max_cells"]}')
        resolved = [(arm, seed) for seed in seeds for arm in arms]
        source = 'canonical_complete_arm_x_seed_grid'
    artifact = build_design_estimability(
        resolved, comparisons, factors=factors, factors_shape=factors_shape,
        block_key=block_key or ('seed' if source != 'explicit_cells' else
                                'block'),
        model=model, created_at=created_at, notes=notes)
    artifact['inputs']['cell_source'] = source
    artifact['inputs']['manifest_sha256'] = _de_text(
        manifest.get('manifest_sha256'))
    artifact['inputs']['manifest_canonical_sha256'] = (
        _de_manifest_canonical_sha(manifest))
    artifact['content_sha256'] = design_estimability_content_sha(artifact)
    return artifact


def design_estimability_summary(artifact):
    if not isinstance(artifact, dict):
        return {'status': 'not_verifiable',
                'note': 'artifact is not an object'}
    design = artifact.get('design') if isinstance(
        artifact.get('design'), dict) else {}
    summary = artifact.get('summary') if isinstance(
        artifact.get('summary'), dict) else {}
    return {
        'status': artifact.get('status'),
        'n_cells': design.get('n_cells'),
        'n_arms': design.get('n_arms'),
        'n_blocks': design.get('n_blocks'),
        'rank': design.get('rank'),
        'n_columns': design.get('n_columns'),
        'deficiency': design.get('deficiency'),
        'complete_grid': design.get('complete_grid'),
        'n_components': summary.get('n_components'),
        'n_comparisons': summary.get('n_comparisons'),
        'n_estimable': summary.get('n_estimable'),
        'n_not_estimable': summary.get('n_not_estimable'),
        'n_terms_not_estimable': summary.get('n_terms_not_estimable'),
        'n_terms_not_separately_estimable': summary.get(
            'n_terms_not_separately_estimable'),
        'content_sha256': artifact.get('content_sha256'),
    }


def write_design_estimability(outdir, artifact, *, keep_existing=True,
                               overwrite=None):
    """Atomically write ``design_estimability.json`` with conflict semantics.

    The artifact is validated before anything is written: a tampered or
    internally inconsistent report cannot be persisted as evidence.  An
    identical body is a no-op; a different body is a conflict unless
    ``keep_existing=False`` / ``overwrite=True`` is explicit.
    """
    if not isinstance(artifact, dict):
        raise TypeError('artifact must be an object')
    if overwrite is not None:
        keep_existing = not bool(overwrite)
    if isinstance(outdir, os.PathLike):
        outdir = os.fspath(outdir)
    if not isinstance(outdir, str) or not outdir.strip():
        raise ValueError('outdir must be a non-empty path')
    verdict = validate_design_estimability(artifact)
    path = os.path.join(outdir, 'design_estimability.json')
    out = {
        'kind': 'write_design_estimability',
        'path': path,
        'written': False,
        'noop': False,
        'unchanged': False,
        'conflict': False,
        'status': verdict.get('status'),
        'artifact_status': artifact.get('status'),
        'problems': list(verdict.get('problems') or []),
    }
    if verdict.get('status') != 'ok':
        return out
    existing = None
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as fh:
                existing = json.load(fh)
        except Exception:
            existing = None
        if (isinstance(existing, dict)
                and existing.get('content_sha256')
                == artifact.get('content_sha256')):
            out['noop'] = True
            out['unchanged'] = True
            out['status'] = 'ok'
            return out
        if keep_existing:
            out['conflict'] = True
            out['status'] = 'conflict'
            out['problems'].append(
                'a different design_estimability.json already exists; pass '
                'keep_existing=False / overwrite=True only after re-running '
                'the analysis')
            return out
    os.makedirs(outdir, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(artifact, fh, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        fh.write('\n')
    os.replace(tmp, path)
    out['written'] = True
    out['status'] = 'ok'
    return out
