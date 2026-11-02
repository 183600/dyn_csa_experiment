# ---- 跨 panel 的全局错误发现率：探索性读数的 BH / BY 校正 ----
#
# 冻结确认族用 Holm 控制 FWER，这是正确的。但 v6-v12 的大量**探索性**
# 比较分散在多个 panel 里：每个 panel 各自做 Holm 只能控制该 panel 的族，
# 不能回答「把整个探索性主张集合放在一起看，其中有多少是噪声」。把每个
# panel 的 p 值按 panel 内族校正后再逐 panel 报「显著」，会低估跨 panel 的
# 搜索量；完全不做跨 panel 校正，则探索性结论的过度宣称没有上限。
#
# 本片段补上这一层：对**显式声明**的探索性主张集合做全局 FDR 校正。默认
# 使用 Benjamini-Yekutieli（BY），因为它对任意依赖结构成立；panel 之间共享
# 训练种子、数据顺序或评测切片时，BH 的独立/正相关假设并不可信。BH 同时
# 计算并并列展示，但只有在调用方明确声明独立性时才作为 headline。
#
# SCOPE: 这是**探索性次级读数**。它不重算 PPL、不读取任何 split、不产生
# 新的确认 p 值、不进入冻结 Holm 族、不改变 `verified`/`confirmed`/
# `confirmed_effective` 或任何 panel 判定。确认性结论仍由冻结清单的
# exact sign-flip + Holm 决定。离散精确检验的 p 有下限，因此本片段同时
# 传递 `p_floor` 并给出该下限下仍可达到的最小 q；若最小 q 已高于 alpha，
# 报告必须说「分辨率不足」，而不是「没有效应」。
#
# 参考: Benjamini & Hochberg (1995); Benjamini & Yekutieli (2001);
#       Storey (2002) 对 q-value 的解释。

import math
import time


GLOBAL_FDR_POLICY = {
    'version': 'global-fdr-1',
    'methods': ['bh', 'by'],
    'default_method': 'by',
    'default_dependence': 'arbitrary (shared seeds/data/evaluation slices)',
    'invalid_p_policy': ('a non-finite/out-of-range p occupies one declared '
                         'family slot but is never rejected; valid q-values '
                         'use the full declared m (fail closed)'),
    'tier': 'exploratory_secondary',
    'role': ('exploratory multiple-comparison readout; it never changes a '
             'confirmatory p-value, the frozen Holm family, a panel `verified` '
             'flag or a `confirmed` verdict'),
}

GLOBAL_FDR_ARTIFACT_KIND = 'global_fdr_artifact'
GLOBAL_FDR_ARTIFACT_SCHEMA = 1

_GLOBAL_FDR_METHODS = ('bh', 'by')
_GLOBAL_FDR_METHOD_ALIASES = {
    'bh': 'bh', 'benjamini-hochberg': 'bh', 'benjamini_hochberg': 'bh',
    'fdr': 'bh', 'independent': 'bh',
    'by': 'by', 'benjamini-yekutieli': 'by', 'benjamini_yekutieli': 'by',
    'dependent': 'by', 'arbitrary': 'by',
}
_GLOBAL_FDR_TIERS = {
    'exploratory': 'exploratory',
    'exploratory_secondary': 'exploratory',
    'secondary_exploratory': 'exploratory',
    'exploratory_primary': 'exploratory',
}
_GLOBAL_FDR_FORBIDDEN_KEYS = frozenset({
    'confirmed', 'confirmed_effective', 'p_holm', 'verified', 'verdict',
    'decision', 'significant', 'significance', 'holm', 'frozen_decision',
})
_GLOBAL_FDR_INPUT_KEYS = ('claims', 'alpha', 'method', 'family')
_GLOBAL_FDR_CLAIM_INPUT_KEYS = (
    'name', 'p', 'p_floor', 'panel', 'contrast', 'tier', 'tier_declared',
    'n', 'n_nonzero',
)
_GLOBAL_FDR_SCALAR_FIELDS = (
    'kind', 'status', 'method', 'dependence', 'alpha', 'family',
    'm_declared', 'm_valid', 'm_invalid', 'n_reject', 'p_threshold',
    'false_discovery_budget', 'min_attainable_q_bh',
    'min_attainable_q_by', 'family_resolution_limited_bh',
    'family_resolution_limited_by', 'family_resolution_limited',
    'resolution_complete', 'note',
)
_GLOBAL_FDR_LIST_FIELDS = ('reject_names', 'invalid_claims', 'warnings')
_GLOBAL_FDR_CLAIM_FIELDS = (
    'p', 'p_floor', 'rank', 'q', 'q_bh', 'q_by', 'q_floor', 'q_floor_bh',
    'q_floor_by', 'reject', 'reject_bh', 'reject_by', 'at_floor',
    'resolution_limited', 'panel', 'contrast', 'tier', 'tier_declared',
    'n', 'n_nonzero', 'reason',
)
_GLOBAL_FDR_HARMONIC_CACHE = {}


def _gfdr_repr(value):
    """Short, exception-safe repr for error messages.

    Python 3.12 refuses to stringify an integer with more than ~4300 digits;
    calling ``repr`` on such a value inside a validation error would replace
    the intended ``ValueError`` with ``ValueError: Exceeds the limit ...``.
    """
    try:
        text = repr(value)
    except Exception:
        return f'<unrepresentable {type(value).__name__}>'
    return text if len(text) <= 120 else text[:117] + '...'


def _gfdr_method(value):
    """Canonical BH/BY method name; a string is a declaration, not a float."""
    if isinstance(value, (bool, np.bool_, bytes, bytearray)) or not isinstance(value, str):
        raise ValueError(
            f'global_fdr method must be a string, got {_gfdr_repr(value)}')
    key = value.strip().lower().replace(' ', '_')
    method = _GLOBAL_FDR_METHOD_ALIASES.get(key)
    if method is None:
        raise ValueError(
            f'global_fdr method must be one of {_GLOBAL_FDR_METHODS} '
            f'(or an accepted alias), got {_gfdr_repr(value)}')
    return method


def _gfdr_harmonic(m):
    """H_m = sum_{i=1..m} 1/i, the BY arbitrary-dependence factor."""
    if m in _GLOBAL_FDR_HARMONIC_CACHE:
        return _GLOBAL_FDR_HARMONIC_CACHE[m]
    h = math.fsum(1.0 / i for i in range(1, int(m) + 1))
    _GLOBAL_FDR_HARMONIC_CACHE[int(m)] = h
    return h


def _gfdr_close(a, b, *, rel=1e-9, abs_=1e-12):
    """Scale-aware equality used by replay; bool is never a measurement."""
    if isinstance(a, (bool,)) or isinstance(b, (bool,)):
        return a is b
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError, OverflowError):
        return False
    if not (math.isfinite(fa) and math.isfinite(fb)):
        return False
    return abs(fa - fb) <= max(abs_, rel * max(1.0, abs(fa), abs(fb)))


def _gfdr_exact(a, b):
    """Exact equality for non-numeric JSON fields (lists/dicts/strings)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if a is None or b is None:
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return _gfdr_close(a, b)
    return a == b


def _gfdr_text(value):
    """Optional display label; never let a dict/list become a JSON key."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode('utf-8')
        except UnicodeDecodeError:
            return None
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _gfdr_int_or_none(value, *, minimum=0, maximum=None):
    if value is None or isinstance(value, (bool, bytes, bytearray)):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if number != value and not (isinstance(value, float) and value.is_integer()):
        return None
    if number < minimum:
        return None
    if maximum is not None and number > maximum:
        return None
    return number


def _gfdr_p(value):
    """A valid p is a finite real in [0,1]; bool/NaN/Inf/non-numeric are not."""
    if isinstance(value, (bool, bytes, bytearray)):
        return None
    try:
        p = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return p if math.isfinite(p) and 0.0 <= p <= 1.0 else None


def _gfdr_floor(value, *, p=None, n_nonzero=None, where='p_floor'):
    """Optional attainable p floor from an explicit value or non-zero count.

    The exact sign-flip two-sided floor is ``2 / 2**n_nonzero`` (capped at
    1.0); a zero-only panel cannot resolve anything.  An explicit floor must
    be a finite number in (0,1].  A malformed floor is a hard error here: the
    artifact's hash must cover exactly the declared inputs, so silently
    changing it would make the replay disagree with the record.
    """
    if value is None and n_nonzero is not None:
        nn = _gfdr_int_or_none(n_nonzero, minimum=0, maximum=1_000_000)
        if nn is None:
            raise ValueError(f'{where}: n_nonzero must be a non-negative '
                             f'integer no larger than 1000000')
        return 1.0 if nn == 0 else min(1.0, math.ldexp(2.0, -nn + 1))
    if value is None:
        return None
    if isinstance(value, (bool, bytes, bytearray)):
        raise ValueError(f'{where}: p_floor must be a finite number in (0,1]')
    try:
        floor = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{where}: p_floor must be a finite number in (0,1]') from None
    if not (math.isfinite(floor) and 0.0 < floor <= 1.0):
        raise ValueError(f'{where}: p_floor must be a finite number in (0,1]')
    return floor


def _gfdr_normalise(claims):
    """Normalise a claim collection to a name-keyed list of JSON-safe records.

    ``claims`` may be a mapping ``name -> p`` / ``name -> record`` or an
    iterable of ``(name, p)`` pairs / records carrying ``name`` (or ``claim``/
    ``id``/``label``).  Duplicate names are refused: a mapping cannot hold two
    slots with one name, and silently keeping one would shrink the declared
    family and make the FDR readout anti-conservative.
    """
    if isinstance(claims, (str, bytes, bytearray)):
        raise ValueError('global_fdr claims must be a mapping or an iterable, '
                         'not a string')
    if claims is None:
        items = []
    elif isinstance(claims, dict):
        items = list(claims.items())
    else:
        try:
            items = list(claims)
        except TypeError:
            raise ValueError('global_fdr claims must be a mapping or an '
                             'iterable of pairs/records') from None
    out = []
    seen = set()
    for index, item in enumerate(items):
        record = None
        if isinstance(item, dict):
            name = item.get('name', item.get('claim', item.get('id',
                            item.get('label'))))
            if name is None and len(item) == 1:
                name, record = next(iter(item.items()))
            else:
                record = item
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            name, record = item
        else:
            raise ValueError(
                f'global_fdr claim {index} must be a (name, p) pair or a '
                f'record, got {type(item).__name__}')
        name = _gfdr_text(name)
        if name is None:
            raise ValueError(f'global_fdr claim {index} has no non-empty name')
        if name in seen:
            raise ValueError(f'global_fdr duplicate claim name {name!r}')
        seen.add(name)
        if not isinstance(record, dict):
            record = {'p': record}
        p = _gfdr_p(record.get('p', record.get('p_value')))
        tier_raw = record.get('tier', record.get('evidence_tier'))
        tier_flag = record.get('tier_declared')
        if tier_flag is None:
            tier_declared = tier_raw is not None
        elif isinstance(tier_flag, bool):
            tier_declared = tier_flag
        else:
            raise ValueError(
                f'global_fdr claim {name!r} tier_declared must be a boolean')
        if tier_raw is None:
            tier = 'exploratory'
        else:
            tier_key = _gfdr_text(tier_raw)
            tier_key = (tier_key.lower().replace('-', '_').replace(' ', '_')
                        if tier_key is not None else None)
            tier = _GLOBAL_FDR_TIERS.get(tier_key)
            if tier is None:
                raise ValueError(
                    f'global_fdr claim {name!r} declares tier '
                    f'{tier_raw!r}; only exploratory claims belong in this '
                    'readout (confirmatory claims stay in the frozen Holm '
                    'family)')
        if not tier_declared:
            # An artifact records the canonical tier plus whether the caller
            # actually declared it.  On replay the canonical string alone
            # would look like a declaration, so the explicit flag wins.
            tier = 'exploratory'
        n_raw = record.get('n')
        nn_raw = record.get('n_nonzero')
        n = _gfdr_int_or_none(n_raw, minimum=1, maximum=1_000_000_000)
        nn = _gfdr_int_or_none(nn_raw, minimum=0, maximum=1_000_000)
        warnings = []
        if n_raw is not None and n is None:
            warnings.append(
                f'claim {name!r}: ignored malformed n={_gfdr_repr(n_raw)}; '
                'it is not a positive integer')
        if nn_raw is not None and nn is None:
            warnings.append(
                f'claim {name!r}: ignored malformed '
                f'n_nonzero={_gfdr_repr(nn_raw)}; no discrete floor is derived')
        if n is not None and nn is not None and nn > n:
            warnings.append(
                f'claim {name!r}: n_nonzero={nn} exceeds n={n}; the floor '
                'count was clamped to n')
            nn = n
        floor = _gfdr_floor(record.get('p_floor'), p=p, n_nonzero=nn,
                            where=f'global_fdr claim {name!r}')
        if p is not None and floor is not None and floor > p and not _gfdr_close(
                floor, p):
            # A floor above the observed p is internally impossible; clamping
            # to p keeps the resolution bound a valid lower bound instead of
            # letting a typo silently buy a finer resolution claim.
            warnings.append(
                f'claim {name!r}: p_floor={floor!r} exceeds p={p!r}; the '
                'floor was clamped to p for the resolution bound')
            floor = p
        out.append({
            'name': name,
            'p': p,
            'p_floor': floor,
            'panel': _gfdr_text(record.get('panel')),
            'contrast': _gfdr_text(record.get('contrast')),
            'tier': tier,
            'tier_declared': bool(tier_declared),
            'n': n,
            'n_nonzero': nn,
            'reason': (None if p is not None else
                       'p is not a finite number in [0,1]'),
            'warnings': warnings,
        })
    out.sort(key=lambda row: row['name'])
    return out


def _gfdr_adjust(order, m_total, method, *, key='p'):
    """BH/BY adjusted q-values for an already p-sorted valid-claim list.

    ``q_i = min_{j>=i} m * p_(j) / j * H_m`` for BY (H_m=1 for BH).  The
    backward running minimum enforces monotonicity and clipping to 1.
    """
    factor = _gfdr_harmonic(m_total) if method == 'by' else 1.0
    q = [None] * len(order)
    running = 1.0
    for i in range(len(order) - 1, -1, -1):
        p = order[i][key]
        raw = (float(m_total) * float(p) / float(i + 1)) * factor
        if not math.isfinite(raw):
            raw = 1.0
        running = min(running, raw)
        q[i] = min(1.0, max(0.0, running))
    return q


def _gfdr_q_floor(order, m_total, method):
    """Lower bound on q_i from each claim's *attainable* p floor.

    If every ``p_(j) >= floor_j`` then ``q_i >= min_{j>=i} m floor_j / j *
    H_m``.  Using this bound (rather than the observed q) makes the statement
    "this family cannot reject at alpha" a resolution fact, not a data claim.
    Returns ``None`` when any floor in the trailing family is missing: a
    missing floor cannot be replaced by 1.0, which would overstate the bound.
    """
    factor = _gfdr_harmonic(m_total) if method == 'by' else 1.0
    floors = [row.get('p_floor') for row in order]
    q = [None] * len(order)
    running = 1.0
    blocked = False
    for i in range(len(order) - 1, -1, -1):
        floor = floors[i]
        if floor is None:
            # A missing floor in the trailing family means the min over
            # j>=i cannot be bounded from below; every earlier claim inherits
            # that missing information, so its bound is withheld too.
            blocked = True
            q[i] = None
            continue
        if blocked:
            q[i] = None
            continue
        raw = (float(m_total) * float(floor) / float(i + 1)) * factor
        if not math.isfinite(raw):
            raw = 1.0
        running = min(running, raw)
        q[i] = min(1.0, max(0.0, running))
    return q


def global_fdr(claims, *, alpha=0.05, method='by', family=None):
    """Global BH/BY false-discovery-rate readout over exploratory claims.

    Parameters
    ----------
    claims
        Mapping ``name -> p`` / ``name -> {p, p_floor?, panel?, contrast?,
        tier?, n?, n_nonzero?}`` or an iterable of ``(name, p)`` pairs /
        records with a ``name`` field.  A claim declaring a confirmatory tier
        is refused: those belong to the frozen Holm family.
    alpha
        FDR level in (0,1).
    method
        ``'by'`` (default, arbitrary dependence) or ``'bh'`` (independent /
        positive dependence).  Both q-vectors are always returned; ``q`` and
        ``reject`` follow the requested method.
    family
        Optional label for the declared family.  It is echoed, never inferred.

    Returns a strict-JSON-safe dict.  Invalid p-values occupy a family slot
    (m is the declared family size) but are never rejected; this is the same
    fail-closed convention as :func:`holm_adjust`.  A valid p at its exact
    discrete floor is marked ``resolution_limited`` when even that floor
    cannot reach ``alpha`` after correction.
    """
    if (isinstance(alpha, (str, bytes, bytearray))
            or isinstance(alpha, (bool, np.bool_))):
        raise ValueError('global_fdr alpha must be a real number in (0,1), '
                         f'got {_gfdr_repr(alpha)}')
    try:
        _alpha = float(alpha)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('global_fdr alpha must be a real number in (0,1), '
                         f'got {_gfdr_repr(alpha)}') from None
    if not (0.0 < _alpha < 1.0):
        raise ValueError('global_fdr alpha must be in (0,1), '
                         f'got {_gfdr_repr(alpha)}')
    alpha = _validated_alpha(_alpha, context='global_fdr alpha')  # noqa: F821
    method = _gfdr_method(method)
    if family is not None and not isinstance(family, (str, bytes, bytearray)):
        raise ValueError('global_fdr family must be a string or None')
    family = _gfdr_text(family)
    rows = _gfdr_normalise(claims)
    valid = [row for row in rows if row['p'] is not None]
    invalid = [row for row in rows if row['p'] is None]
    m_total = len(rows)
    warnings = []
    for row in rows:
        warnings.extend(row.get('warnings') or [])
    if any(not row['tier_declared'] for row in rows):
        warnings.append(
            'some claims did not declare a tier; they are treated as '
            'exploratory and never enter the frozen confirmatory family')
    out = {
        'kind': 'global_fdr',
        'status': 'ok',
        'policy': {key: (list(value) if isinstance(value, list) else value)
                   for key, value in GLOBAL_FDR_POLICY.items()},
        'method': method,
        'dependence': ('arbitrary' if method == 'by' else
                       'independent_or_positive'),
        'alpha': float(alpha),
        'family': family,
        'm_declared': int(m_total),
        'm_valid': int(len(valid)),
        'm_invalid': int(len(invalid)),
        'n_reject': 0,
        'reject_names': [],
        'p_threshold': None,
        'false_discovery_budget': 0.0,
        'min_attainable_q_bh': None,
        'min_attainable_q_by': None,
        'family_resolution_limited_bh': None,
        'family_resolution_limited_by': None,
        'family_resolution_limited': None,
        'resolution_complete': None,
        'invalid_claims': [row['name'] for row in invalid],
        'claims': {},
        'warnings': sorted(set(warnings)),
        'note': ('exploratory global FDR readout.  The frozen confirmatory '
                 'family remains exact sign-flip + Holm; this block never '
                 'creates a new confirmatory p-value or changes a verdict.'),
    }
    for row in rows:
        out['claims'][row['name']] = {
            'p': row['p'], 'p_floor': row['p_floor'], 'rank': None,
            'q': None, 'q_bh': None, 'q_by': None,
            'q_floor': None, 'q_floor_bh': None, 'q_floor_by': None,
            'reject': False, 'reject_bh': False, 'reject_by': False,
            'at_floor': None, 'resolution_limited': None,
            'panel': row['panel'], 'contrast': row['contrast'],
            'tier': row['tier'], 'tier_declared': bool(row['tier_declared']),
            'n': row['n'], 'n_nonzero': row['n_nonzero'],
            'reason': row['reason'],
        }
    if not valid:
        out['status'] = 'not_applicable'
        out['warnings'] = sorted(set(out['warnings'] + [
            'no claim carries a valid p in [0,1]; there is no FDR readout']))
        return out
    order = sorted(valid, key=lambda row: (row['p'], row['name']))
    q_bh = _gfdr_adjust(order, m_total, 'bh')
    q_by = _gfdr_adjust(order, m_total, 'by')
    qf_bh = _gfdr_q_floor(order, m_total, 'bh')
    qf_by = _gfdr_q_floor(order, m_total, 'by')
    chosen_q = q_by if method == 'by' else q_bh
    chosen_qf = qf_by if method == 'by' else qf_bh
    reject_all = []
    for i, row in enumerate(order):
        name = row['name']
        q_bh_i, q_by_i = q_bh[i], q_by[i]
        q_choice = chosen_q[i]
        row_out = out['claims'][name]
        row_out.update({
            'rank': i + 1,
            'q': q_choice,
            'q_bh': q_bh_i,
            'q_by': q_by_i,
            'q_floor': chosen_qf[i],
            'q_floor_bh': qf_bh[i],
            'q_floor_by': qf_by[i],
            'reject': bool(q_choice is not None and q_choice <= alpha),
            'reject_bh': bool(q_bh_i is not None and q_bh_i <= alpha),
            'reject_by': bool(q_by_i is not None and q_by_i <= alpha),
            'at_floor': (None if row['p_floor'] is None else
                         bool(_gfdr_close(row['p'], row['p_floor']))),
            'resolution_limited': (None if chosen_qf[i] is None else
                                   bool(chosen_qf[i] > alpha)),
        })
        if row_out['reject']:
            reject_all.append((row['p'], name))
    out['reject_names'] = sorted(name for _p, name in reject_all)
    out['n_reject'] = len(reject_all)
    out['p_threshold'] = (max(p for p, _name in reject_all)
                          if reject_all else None)
    out['false_discovery_budget'] = float(alpha * len(reject_all))
    out['min_attainable_q_bh'] = (min(v for v in qf_bh
                                     if v is not None)
                                  if qf_bh and all(v is not None for v in qf_bh)
                                  else None)
    out['min_attainable_q_by'] = (min(v for v in qf_by
                                     if v is not None)
                                  if qf_by and all(v is not None for v in qf_by)
                                  else None)
    out['family_resolution_limited_bh'] = (
        None if out['min_attainable_q_bh'] is None
        else bool(out['min_attainable_q_bh'] > alpha))
    out['family_resolution_limited_by'] = (
        None if out['min_attainable_q_by'] is None
        else bool(out['min_attainable_q_by'] > alpha))
    out['family_resolution_limited'] = (
        out['family_resolution_limited_by'] if method == 'by'
        else out['family_resolution_limited_bh'])
    out['resolution_complete'] = (out['min_attainable_q_bh'] is not None)
    return out


def _gfdr_input_claims(rows):
    """Canonical artifact inputs: the exact declared family, sorted by name."""
    return [{
        'name': row['name'],
        'p': row['p'],
        'p_floor': row['p_floor'],
        'panel': row['panel'],
        'contrast': row['contrast'],
        'tier': row['tier'],
        'tier_declared': bool(row['tier_declared']),
        'n': row['n'],
        'n_nonzero': row['n_nonzero'],
    } for row in rows]


def build_global_fdr(claims, *, alpha=0.05, method='by', family=None,
                     created_at=None):
    """Build a replayable global-FDR artifact.

    The artifact records the exact declared family in ``inputs`` and the
    derived q-values in ``result``.  An unusable family (no valid p) is not
    evidence and is refused rather than written as an empty claim.
    """
    rows = _gfdr_normalise(claims)
    # Reuse the already-normalised rows: a one-shot generator would otherwise
    # be consumed by the first pass and silently replay as an empty family.
    declared = _gfdr_input_claims(rows)
    result = global_fdr(declared, alpha=alpha, method=method, family=family)
    if result.get('status') != 'ok':
        raise ValueError(
            'global_fdr artifact needs at least one valid p-value; the '
            f'declared family status is {result.get("status")!r}')
    artifact = {
        'kind': GLOBAL_FDR_ARTIFACT_KIND,
        'schema': GLOBAL_FDR_ARTIFACT_SCHEMA,
        'created_at': (created_at if created_at is not None
                       else time.strftime('%Y-%m-%dT%H:%M:%S')),
        'inputs': {
            'claims': declared,
            'alpha': float(result['alpha']),
            'method': result['method'],
            'family': result['family'],
        },
        'result': result,
    }
    artifact['content_sha256'] = _sha256_json(  # noqa: F821
        {k: v for k, v in artifact.items()
         if k not in ('created_at', 'content_sha256')})
    return artifact


def _gfdr_forbidden_payload(obj):
    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, str) and key in _GLOBAL_FDR_FORBIDDEN_KEYS:
                found.append(key)
            found.extend(_gfdr_forbidden_payload(value))
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found.extend(_gfdr_forbidden_payload(value))
    return found


def _gfdr_compare_claim(stored, replay, name, problems):
    if not isinstance(stored, dict):
        problems.append(f'claims[{name!r}] is not an object')
        return
    extra = sorted(set(stored) - set(_GLOBAL_FDR_CLAIM_FIELDS))
    if extra:
        problems.append(
            f'claims[{name!r}] carries field(s) outside the replay schema: '
            + ', '.join(repr(key) for key in extra))
    missing = sorted(set(_GLOBAL_FDR_CLAIM_FIELDS) - set(stored))
    if missing:
        problems.append(f'claims[{name!r}] is missing field(s): {missing}')
    for key in _GLOBAL_FDR_CLAIM_FIELDS:
        if key not in stored:
            continue
        a, b = stored.get(key), replay.get(key)
        if not _gfdr_exact(a, b):
            problems.append(
                f'claims[{name!r}].{key}: stored={a!r}, replay={b!r}')


def validate_global_fdr(artifact):
    """Replay a stored global-FDR artifact and compare every derived field.

    Missing digest, a non-reproducing digest, a tampered q-value, an extra
    field or a smuggled frozen-decision key is ``fail``.  A structurally valid
    artifact whose replay has no valid p is ``not_applicable`` (the gate maps
    that status), never a silent pass.
    """
    out = {'kind': 'global_fdr_validation', 'schema': 1, 'status': 'fail',
           'problems': [], 'warnings': [], 'content_sha256': None,
           'result': None, 'result_status': None}
    if not isinstance(artifact, dict):
        out['problems'].append('artifact must be a JSON object')
        return out
    if artifact.get('kind') != GLOBAL_FDR_ARTIFACT_KIND:
        out['problems'].append(
            f'artifact.kind must be {GLOBAL_FDR_ARTIFACT_KIND!r}, got '
            f'{artifact.get("kind")!r}')
    schema = artifact.get('schema')
    if (isinstance(schema, bool) or not isinstance(schema, int)
            or schema != GLOBAL_FDR_ARTIFACT_SCHEMA):
        out['problems'].append(
            f'artifact.schema must be {GLOBAL_FDR_ARTIFACT_SCHEMA}, got '
            f'{schema!r}')
    stolen = sorted(set(_gfdr_forbidden_payload(artifact)))
    if stolen:
        out['problems'].append(
            'artifact carries frozen-decision field(s) that an exploratory '
            f'FDR readout must not contain: {stolen}')
    _allowed_top = {'kind', 'schema', 'created_at', 'inputs', 'result',
                    'content_sha256'}
    _extra_top = [key for key in artifact if key not in _allowed_top]
    if _extra_top:
        out['problems'].append(
            'artifact carries field(s) outside the replay schema: '
            + ', '.join(_gfdr_repr(key) for key in
                        sorted(_extra_top, key=_gfdr_repr)))
    inputs = artifact.get('inputs')
    if not isinstance(inputs, dict):
        out['problems'].append('artifact.inputs must be an object')
        inputs = {}
    missing = [k for k in _GLOBAL_FDR_INPUT_KEYS if k not in inputs]
    if missing:
        out['problems'].append(
            f'artifact.inputs is missing required key(s): {missing}')
    extra_inputs = sorted(set(inputs) - set(_GLOBAL_FDR_INPUT_KEYS))
    if extra_inputs:
        out['problems'].append(
            'artifact.inputs carries field(s) outside the replay schema: '
            + ', '.join(repr(key) for key in extra_inputs))
    recorded = artifact.get('content_sha256')
    out['content_sha256'] = recorded if isinstance(recorded, str) else None
    body = {k: v for k, v in artifact.items()
            if k not in ('created_at', 'content_sha256')}
    try:
        replay_sha = _sha256_json(body)  # noqa: F821
    except Exception as exc:
        out['problems'].append(f'artifact body is not canonical JSON: {exc}')
        replay_sha = None
    if not isinstance(recorded, str) or len(recorded) != 64:
        out['problems'].append(
            'artifact.content_sha256 is missing or not a 64-character digest')
    elif replay_sha is not None and replay_sha != recorded:
        out['problems'].append(
            'artifact.content_sha256 does not reproduce from the artifact body')
    claim_inputs = inputs.get('claims')
    if not isinstance(claim_inputs, list):
        out['problems'].append('artifact.inputs.claims must be a list')
        return out
    for i, row in enumerate(claim_inputs):
        if not isinstance(row, dict):
            out['problems'].append(
                f'artifact.inputs.claims[{i}] is not an object')
            continue
        extra = sorted(set(row) - set(_GLOBAL_FDR_CLAIM_INPUT_KEYS))
        if extra:
            out['problems'].append(
                f'artifact.inputs.claims[{i}] carries field(s) outside the '
                'replay schema: ' + ', '.join(repr(key) for key in extra))
        miss = sorted(set(_GLOBAL_FDR_CLAIM_INPUT_KEYS) - set(row))
        if miss:
            out['problems'].append(
                f'artifact.inputs.claims[{i}] is missing field(s): {miss}')
    if missing:
        return out
    try:
        replay = global_fdr(
            claim_inputs, alpha=inputs.get('alpha'),
            method=inputs.get('method'), family=inputs.get('family'))
    except Exception as exc:
        out['problems'].append(f'replay raised {type(exc).__name__}: {exc}')
        return out
    out['result'] = replay
    out['result_status'] = replay.get('status')
    stored = artifact.get('result')
    if not isinstance(stored, dict):
        out['problems'].append('artifact.result must be an object')
        return out
    extra = sorted(set(stored) - set(replay))
    if extra:
        out['problems'].append(
            'result carries field(s) outside the replay schema: '
            + ', '.join(repr(key) for key in extra))
    for key in _GLOBAL_FDR_SCALAR_FIELDS:
        if key not in stored:
            out['problems'].append(f'result.{key} is missing')
            continue
        if not _gfdr_exact(stored.get(key), replay.get(key)):
            out['problems'].append(
                f'result.{key}: stored={stored.get(key)!r}, '
                f'replay={replay.get(key)!r}')
    for key in _GLOBAL_FDR_LIST_FIELDS:
        if key not in stored:
            out['problems'].append(f'result.{key} is missing')
            continue
        if stored.get(key) != replay.get(key):
            out['problems'].append(
                f'result.{key}: stored={stored.get(key)!r}, '
                f'replay={replay.get(key)!r}')
    if stored.get('policy') != replay.get('policy'):
        out['problems'].append('result.policy differs from the replay')
    stored_claims = stored.get('claims')
    replay_claims = replay.get('claims') or {}
    if not isinstance(stored_claims, dict):
        out['problems'].append('result.claims must be an object')
    else:
        extra_claims = sorted(set(stored_claims) - set(replay_claims))
        missing_claims = sorted(set(replay_claims) - set(stored_claims))
        if extra_claims:
            out['problems'].append(
                'result.claims carries undeclared claim(s): '
                + ', '.join(repr(name) for name in extra_claims))
        if missing_claims:
            out['problems'].append(
                'result.claims is missing claim(s): '
                + ', '.join(repr(name) for name in missing_claims))
        for name in sorted(set(stored_claims) & set(replay_claims)):
            _gfdr_compare_claim(stored_claims[name], replay_claims[name],
                                name, out['problems'])
    if not out['problems']:
        out['status'] = 'ok'
    return out


def write_global_fdr(path, artifact, *, indent=2):
    """Atomically write a global-FDR artifact; returns the path."""
    atomic_write_json(path, artifact, indent=indent)  # noqa: F821
    return path
