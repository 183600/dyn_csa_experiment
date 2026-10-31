# Data-cleaning specification sensitivity: how much does the served corpus
# depend on the cleaning choices (and row order) that the protocol records?
#
# The cache metadata binds the exact/SimHash geometry, and the cleaning replay
# proves that the recorded geometry reproduces the served membership on the
# supplied raw rows.  Neither answers the next methodological question: is that
# membership *stable* under the nearby cleaning choices a reasonable analyst
# could have made, and under a different source-row order?  Near-duplicate
# filtering is greedy: the first member of a near-duplicate cluster is kept and
# later members are removed (and quarantined).  A small change in shingle width,
# LSH bands, the Hamming threshold, or simply the order of two source rows can
# therefore change which document represents a cluster, or even how many
# documents survive.
#
# This module makes that dependence measurable.  It replays the recorded
# decision sequence once as the baseline, then repeats it under valid
# alternative protocols and seeded row permutations, and reports the kept-set
# Jaccard / count change against the baseline.  The result is a data-cleaning
# specification curve: a descriptive statement of how much of the corpus is a
# function of the cleaning decision rather than of the source text.
#
# It changes no cache, token, split, training, PPL, seed or frozen decision.  It
# is an audit-side sensitivity readout, not a new dataset and not a replacement
# for the frozen protocol: the confirmatory analysis still uses exactly the
# recorded protocol and source order.

CLEANING_SENSITIVITY_SCHEMA = 'wikitext_cleaning_sensitivity_v1'
CLEANING_SENSITIVITY_MAX_ROWS = 1_000_000
# A variant is called materially different when it changes at least 5% of the
# kept-set members, or more than 1% of the kept-document count.  These are
# reporting thresholds, not a statistical test and not a cache verdict.
CLEANING_SENSITIVITY_JACCARD_FLOOR = 0.95
CLEANING_SENSITIVITY_COUNT_FRACTION = 0.01


def _cs_load_json(path):
    """Read a cache metadata object without importing the audit namespace."""
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError, TypeError):
        return None


CLEANING_SENSITIVITY_POLICY = {
    'version': 'cleaning-spec-1',
    'baseline': 'recorded dedup protocol applied to raw rows in source order',
    'axes': ['near_hamming_threshold', 'bands', 'shingle_words',
             'max_shingles', 'row_order'],
    'order_scope': ('the tokenizer-fit prefix stays fixed and in source order; '
                    'the remaining train rows and validation rows are permuted'),
    'stability_rule': ('material when kept-set Jaccard vs baseline < 0.95 or '
                       'kept-count change > 1%'),
    'role': ('descriptive data-cleaning sensitivity; it does not change the '
             'cache, split, tokens, training, PPL, seeds or any frozen decision'),
}


def _cs_int(value, name, *, minimum=0):
    """Exact integer argument; booleans and fractions are not counts."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f'{name} must be an integer, got {value!r}')
    value = int(value)
    if value < int(minimum):
        raise ValueError(f'{name} must be >= {int(minimum)}, got {value}')
    return value


def _cs_rows(rows, name):
    """Materialise a row source, type-check it, and drop blank rows.

    The loaders filter ``r.strip()`` before dedup, so blank rows never enter the
    decision sequence.  Returning the total as well lets the caller bind the
    replay to a declared ``source_rows`` count without silently ignoring blanks.
    """
    if isinstance(rows, (str, bytes)):
        raise TypeError(f'{name} must be an iterable of rows, not a single string')
    try:
        seq = list(rows)
    except TypeError:
        raise TypeError(f'{name} must be an iterable of strings, got '
                        f'{type(rows).__name__}') from None
    for i, row in enumerate(seq):
        if not isinstance(row, str):
            raise TypeError(f'{name}[{i}] must be a string, got '
                            f'{type(row).__name__}')
    return [row for row in seq if row.strip()], int(len(seq))


class _HashedDedup:
    """The production greedy deduper over precomputed exact/SimHash values.

    The sensitivity audit has to try several shingle geometries over the same
    rows; hashing every row once per geometry is the expensive part.  This
    small state machine reproduces ``_DocDeduplicator`` exactly (same exact-set
    membership, same LSH buckets, same quarantine of skipped near copies) while
    accepting the hashes as inputs.  The equivalence is exercised by the tests.
    """

    def __init__(self, protocol):
        self.near_threshold = int(protocol['near_hamming_threshold'])
        self.bands = int(protocol['bands'])
        self.bits = 64 // self.bands
        self.mask = (1 << self.bits) - 1
        self.exact = set()
        self.buckets = {}
        self.n_exact = 0
        self.n_near = 0
        self.n_seen = 0

    def _keys(self, h):
        return [(b, (int(h) >> (b * self.bits)) & self.mask)
                for b in range(self.bands)]

    def _match(self, fp_key, h):
        if fp_key in self.exact:
            return 'exact'
        for key in self._keys(h):
            for other in self.buckets.get(key, ()):
                if int(int(h) ^ int(other)).bit_count() <= self.near_threshold:
                    return 'near'
        return None

    def add(self, fp_key, h):
        """Return True when the row is an exact/near duplicate (not kept)."""
        kind = self._match(fp_key, h)
        if kind == 'exact':
            self.n_exact += 1
            return True
        if kind == 'near':
            self.n_near += 1
            # Match the production quarantining rule: a skipped near copy still
            # joins the LSH buckets, so a transitive chain cannot slip through.
            for key in self._keys(h):
                self.buckets.setdefault(key, []).append(int(h))
            return True
        self.exact.add(fp_key)
        self.n_seen += 1
        for key in self._keys(h):
            self.buckets.setdefault(key, []).append(int(h))
        return False


def _cs_prepare(rows):
    """Precompute exact-hash keys and document ids (compact id vector)."""
    exact = []
    ids = np.empty(len(rows), dtype=np.uint64)
    for i, row in enumerate(rows):
        fp = _doc_exact_fp(row)
        exact.append(int(fp, 16))
        ids[i] = np.uint64(_doc_id_u64(fp))
    return exact, ids


def _cs_hashes(rows, protocol, cache, *, tag):
    """SimHash values for one split and shingle geometry, computed once."""
    key = (str(tag), int(protocol['shingle_words']),
           int(protocol['max_shingles']))
    if key not in cache:
        # uint64 keeps a 1M-row audit inside the same order of memory as the
        # token caches instead of one Python int object per row per geometry.
        cache[key] = np.fromiter(
            (int(_doc_simhash(row, shingle_words=key[1],
                              max_shingles=key[2]))
             for row in rows),
            dtype=np.uint64, count=len(rows))
    return cache[key]


def _cs_sequence(state, exact, hashes, order, scanned, *, fit_rows=0):
    """Run one train/validation sequence and return kept original indices."""
    kept = []
    for pos in list(order)[:int(scanned)]:
        i = int(pos)
        if i < int(fit_rows):
            # The tokenizer-fit prefix is force-kept, exactly as stream_tokenise
            # does for its first _n_fit_docs non-empty train rows.
            kept.append(i)
            continue
        if not state.add(exact[i], hashes[i]):
            kept.append(i)
    return kept


def _cs_compare(expected, kept_ids):
    """Compare replayed kept ids with a cache sidecar (trailing tail allowed).

    The loader dedups a whole 10k-row chunk before stopping at the token cap,
    so the persisted sidecar may name a few kept rows beyond the tokenised
    prefix.  A shorter expected vector must match the replay prefix exactly;
    an expected vector longer than the replay is a mismatch.
    """
    if expected is None:
        return {'match': None, 'expected_count': None,
                'replayed_count': int(len(kept_ids)), 'trailing_replayed': None}
    try:
        exp = np.asarray(expected).reshape(-1)
    except Exception as exc:
        return {'match': False,
                'reason': f'expected ids are unreadable ({type(exc).__name__}: {exc})'}
    if exp.size and exp.dtype.kind not in 'iu':
        return {'match': False,
                'reason': f'expected ids have dtype {exp.dtype}, not an integer array'}
    exp = [int(x) for x in exp.tolist()]
    kept = [int(x) for x in kept_ids]
    if len(exp) > len(kept):
        return {'match': False, 'expected_count': len(exp),
                'replayed_count': len(kept), 'trailing_replayed': 0,
                'reason': ('the cache names more kept documents than the '
                           'replay produces')}
    match = exp == kept[:len(exp)]
    out = {'match': bool(match), 'expected_count': int(len(exp)),
           'replayed_count': int(len(kept)),
           'trailing_replayed': int(len(kept) - len(exp))}
    if not match:
        first = next((i for i, (a, b) in enumerate(zip(exp, kept)) if a != b),
                     None)
        out['first_mismatch'] = first
        out['reason'] = ('the replayed kept prefix does not match the cache '
                         'sidecar; the sensitivity baseline is not this cache')
    return out


def _cs_protocol_variants(protocol):
    """Nearby valid protocols along the declared cleaning axes.

    Only variants that pass the same ``dedup_protocol_problems`` validator as
    production are offered.  A variant that cannot be made valid is reported as
    skipped with its concrete reason instead of being silently dropped.
    """
    base = dict(protocol)
    variants = []

    def _add(name, axis, patch, rationale):
        cand = dict(base)
        cand.update(patch)
        problems = dedup_protocol_problems(cand)
        if problems:
            variants.append({'name': name, 'axis': axis, 'kind': 'protocol',
                             'status': 'skipped',
                             'skip_reason': '; '.join(problems),
                             'rationale': rationale})
            return
        if dedup_protocol_fp(cand) == dedup_protocol_fp(base):
            variants.append({'name': name, 'axis': axis, 'kind': 'protocol',
                             'status': 'skipped',
                             'skip_reason': 'variant is identical to the recorded protocol',
                             'rationale': rationale})
            return
        variants.append({'name': name, 'axis': axis, 'kind': 'protocol',
                         'status': 'planned', 'protocol': cand,
                         'protocol_fp': dedup_protocol_fp(cand),
                         'rationale': rationale})

    t = int(base['near_hamming_threshold'])
    bands = int(base['bands'])
    if t > 0:
        _add('near_threshold_strict', 'near_hamming_threshold',
             {'near_hamming_threshold': t - 1},
             'one bit stricter: fewer near copies removed')
    if t + 1 < bands:
        _add('near_threshold_loose', 'near_hamming_threshold',
             {'near_hamming_threshold': t + 1},
             'one bit looser: more near copies removed')
    elif bands != 8 and 64 % 8 == 0 and t + 1 < 8:
        # A looser threshold is not valid under the default 4-band layout
        # (threshold must stay below bands).  Eight bands preserves the
        # exhaustive-candidate guarantee and lets the threshold move up one
        # bit; the two fields are changed together because only that pair is a
        # valid protocol.
        _add('near_threshold_loose_bands8', 'near_hamming_threshold+bands',
             {'near_hamming_threshold': t + 1, 'bands': 8},
             'one bit looser under an 8-band exhaustive layout')
    if bands != 8 and 64 % 8 == 0 and t < 8:
        _add('more_lsh_bands', 'bands', {'bands': 8},
             'exhaustiveness control: a valid band layout must not change the '
             'true near-match set')
    for words in (4, 8):
        if words <= int(base['max_shingles']) and words != int(base['shingle_words']):
            _add(f'shingle_words_{words}', 'shingle_words',
                 {'shingle_words': words},
                 'a different word-shingle width for the same text')
    if int(base['max_shingles']) < 256:
        _add('max_shingles_256', 'max_shingles', {'max_shingles': 256},
             'subsample fewer shingles on long documents')
    return variants


def _cs_order_variants(n_order_permutations):
    out = [{'name': 'order_reverse', 'axis': 'row_order', 'kind': 'order',
            'status': 'planned', 'order_kind': 'reverse',
            'rationale': 'reverse the post-fit source order'}]
    for k in range(int(n_order_permutations)):
        out.append({'name': f'order_permutation_{k + 1}', 'axis': 'row_order',
                    'kind': 'order', 'status': 'planned',
                    'order_kind': 'permutation', 'permutation_index': int(k),
                    'rationale': 'seeded permutation of the post-fit source order'})
    return out


def _cs_order(n, fit_rows, kind, *, seed=0, permutation_index=0):
    """Order array that keeps the fit prefix first and permutes the rest."""
    n = int(n)
    fit_rows = int(fit_rows)
    prefix = list(range(min(fit_rows, n)))
    rest = np.arange(fit_rows, n, dtype=np.int64)
    if kind == 'reverse':
        rest = rest[::-1]
    elif kind == 'permutation':
        rng = np.random.default_rng(int(seed) + int(permutation_index))
        rest = rng.permutation(rest)
    else:
        raise ValueError(f'unknown order kind {kind!r}')
    return prefix + [int(i) for i in rest.tolist()]


def _cs_metrics(base_ids, variant_ids):
    base_list = [int(x) for x in base_ids]
    var_list = [int(x) for x in variant_ids]
    base = set(base_list)
    var = set(var_list)
    inter = len(base & var)
    union = len(base | var)
    jaccard = 1.0 if union == 0 else float(inter) / float(union)
    denom = max(1, len(base))
    return {
        'kept': int(len(var)),
        'kept_jaccard': round(float(jaccard), 8),
        'count_delta': int(len(var) - len(base)),
        'count_delta_fraction': round(float(abs(len(var) - len(base))) / float(denom), 8),
        'retained_fraction': round(float(inter) / float(denom), 8),
        'added_fraction': round(float(len(var - base)) / float(denom), 8),
        'membership_identical': bool(base == var),
        # The token stream is ordered: a permutation of the same kept set is a
        # different served sequence (and can change which rows the fixed
        # validation prefix scores), so set equality alone is not enough.
        'kept_sequence_identical': bool(base_list == var_list),
    }


def cleaning_sensitivity(train_rows, validation_rows=(), *, protocol,
                         fit_rows, train_scanned=None,
                         validation_scanned=None,
                         expected_train_doc_ids=None,
                         expected_validation_doc_ids=None,
                         expected_train_source_rows=None,
                         expected_validation_source_rows=None,
                         max_rows=CLEANING_SENSITIVITY_MAX_ROWS,
                         n_order_permutations=2, seed=0,
                         jaccard_floor=CLEANING_SENSITIVITY_JACCARD_FLOOR,
                         count_fraction_floor=CLEANING_SENSITIVITY_COUNT_FRACTION):
    """Measure kept-membership sensitivity to nearby cleaning choices/order.

    The baseline is the recorded protocol on the supplied raw rows in source
    order.  Every tested variant is a valid protocol on one declared axis or a
    seeded permutation of the post-fit rows (the fit prefix stays fixed, because
    a changed fit prefix would also change the tokenizer, a different stage).
    Results are descriptive: ``status='warnings'`` means at least one variant is
    materially different under the declared floors, not that the cache is
    invalid.  ``status='fail'`` is reserved for a baseline that contradicts the
    supplied cache sidecars or declared source row counts.

    ``train_scanned``/``validation_scanned`` default to all non-empty rows and
    are capped at the available length.  ``max_rows`` is a fail-closed guard:
    above it the result is ``not_verifiable`` rather than a silent sample; pass
    ``0`` to disable the guard deliberately (matching ``cleaning_replay``).
    """
    problems = []
    warnings = []
    if not isinstance(protocol, dict):
        raise ValueError('protocol must be an object')
    proto_problems = dedup_protocol_problems(protocol)
    if proto_problems:
        raise ValueError('invalid dedup protocol: ' + '; '.join(proto_problems))
    fit_rows = _cs_int(fit_rows, 'fit_rows')
    n_order_permutations = _cs_int(n_order_permutations, 'n_order_permutations')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral):
        raise ValueError(f'seed must be an integer, got {seed!r}')
    seed = int(seed)
    for name, value in (('jaccard_floor', jaccard_floor),
                        ('count_fraction_floor', count_fraction_floor)):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f'{name} must be a real number, not a boolean')
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f'{name} must be a finite number in [0, 1]') from None
        if not (math.isfinite(value) and 0.0 <= value <= 1.0):
            raise ValueError(f'{name} must be a finite number in [0, 1]')
    jaccard_floor = float(jaccard_floor)
    count_fraction_floor = float(count_fraction_floor)
    train_ne, train_total = _cs_rows(train_rows, 'train_rows')
    val_ne, val_total = _cs_rows(validation_rows, 'validation_rows')
    if fit_rows > len(train_ne):
        raise ValueError(f'fit_rows={fit_rows} exceeds the {len(train_ne)} '
                         'non-empty train rows')
    if train_scanned is None:
        train_scanned = len(train_ne)
    else:
        train_scanned = _cs_int(train_scanned, 'train_scanned')
    if validation_scanned is None:
        validation_scanned = len(val_ne)
    else:
        validation_scanned = _cs_int(validation_scanned, 'validation_scanned')
    if train_scanned > len(train_ne):
        raise ValueError(f'train_scanned={train_scanned} exceeds the '
                         f'{len(train_ne)} non-empty train rows')
    if validation_scanned > len(val_ne):
        raise ValueError(f'validation_scanned={validation_scanned} exceeds the '
                         f'{len(val_ne)} non-empty validation rows')
    if max_rows is not None:
        max_rows = _cs_int(max_rows, 'max_rows')
        over = []
        if max_rows:
            over = [(label, count) for label, count in
                    (('train rows', len(train_ne)),
                     ('validation rows', len(val_ne)),
                     ('fit rows', fit_rows))
                    if int(count) > int(max_rows)]
        if over:
            return {
                'kind': 'cleaning_sensitivity',
                'schema': CLEANING_SENSITIVITY_SCHEMA,
                'status': 'not_verifiable',
                'policy': dict(CLEANING_SENSITIVITY_POLICY),
                'protocol': dict(protocol),
                'protocol_fp': dedup_protocol_fp(protocol),
                'max_rows': int(max_rows),
                'problems': [f'{label} would be replayed above max_rows='
                             f'{int(max_rows)}; rerun explicitly with a larger '
                             'max_rows to measure this corpus'
                             for label, count in over],
                'warnings': [],
                'note': ('a sensitivity readout that skips rows is not the '
                         'corpus; raise max_rows deliberately'),
            }
    source_binding = {'train': None, 'validation': None}
    for split, total, expected in (
            ('train', train_total, expected_train_source_rows),
            ('validation', val_total, expected_validation_source_rows)):
        if expected is None:
            continue
        expected = _cs_int(expected, f'expected_{split}_source_rows')
        source_binding[split] = {'expected': expected, 'actual': int(total),
                                 'match': expected == int(total)}
        if expected != int(total):
            problems.append(
                f'{split}: raw source has {total} rows but the cache records '
                f'source_rows={expected}; refusing to describe a different dump')
    out = {
        'kind': 'cleaning_sensitivity',
        'schema': CLEANING_SENSITIVITY_SCHEMA,
        'status': 'ok',
        'policy': dict(CLEANING_SENSITIVITY_POLICY),
        'scope': ('train_and_validation_membership; fit prefix fixed for order '
                  'variants; test is never read'),
        'protocol': dict(protocol),
        'protocol_fp': dedup_protocol_fp(protocol),
        'fit_rows': int(fit_rows),
        'train_scanned': int(train_scanned),
        'validation_scanned': int(validation_scanned),
        'train_source_rows': int(train_total),
        'validation_source_rows': int(val_total),
        'source_binding': source_binding,
        'max_rows': (None if max_rows is None else int(max_rows)),
        'n_order_permutations': int(n_order_permutations),
        'seed': int(seed),
        'jaccard_floor': jaccard_floor,
        'count_fraction_floor': count_fraction_floor,
        'baseline': {},
        'baseline_match': {},
        'variants': [],
        'summary': {},
        'problems': problems,
        'warnings': warnings,
        'note': ('descriptive cleaning-specification curve: how much the served '
                 'membership depends on nearby valid cleaning choices and row '
                 'order; it does not change the frozen cache or any result'),
        'test_raw_read': False,
    }
    if problems:
        out['status'] = 'fail'
        return out
    # Precompute exact identity and document ids once; SimHash vectors are
    # cached per shingle geometry because that is the expensive step.
    tr_exact, tr_ids = _cs_prepare(train_ne)
    va_exact, va_ids = _cs_prepare(val_ne)
    hash_cache = {}

    def _run(protocol_variant, tr_order, va_order):
        state = _HashedDedup(protocol_variant)
        tr_h = _cs_hashes(train_ne, protocol_variant, hash_cache, tag='train')
        va_h = _cs_hashes(val_ne, protocol_variant, hash_cache, tag='validation')
        for i in range(min(int(fit_rows), len(train_ne))):
            state.add(tr_exact[i], tr_h[i])
        seed_counts = (state.n_exact, state.n_near)
        tr_kept = _cs_sequence(state, tr_exact, tr_h, tr_order,
                               train_scanned, fit_rows=fit_rows)
        tr_after = (state.n_exact, state.n_near)
        va_kept = _cs_sequence(state, va_exact, va_h, va_order,
                               validation_scanned, fit_rows=0)
        va_after = (state.n_exact, state.n_near)
        return {
            'train_kept_ids': [tr_ids[i] for i in tr_kept],
            'validation_kept_ids': [va_ids[i] for i in va_kept],
            'train_removed': {'exact': int(tr_after[0] - seed_counts[0]),
                              'near': int(tr_after[1] - seed_counts[1])},
            'validation_removed': {'exact': int(va_after[0] - tr_after[0]),
                                   'near': int(va_after[1] - tr_after[1])},
        }

    base_order_tr = list(range(len(train_ne)))
    base_order_va = list(range(len(val_ne)))
    baseline = _run(dict(protocol), base_order_tr, base_order_va)
    out['baseline'] = {
        'train': {'kept': int(len(baseline['train_kept_ids']))},
        'validation': {'kept': int(len(baseline['validation_kept_ids']))},
        'removed': {'train': baseline['train_removed'],
                    'validation': baseline['validation_removed']},
    }
    out['baseline_match']['train'] = _cs_compare(
        expected_train_doc_ids, baseline['train_kept_ids'])
    out['baseline_match']['validation'] = _cs_compare(
        expected_validation_doc_ids, baseline['validation_kept_ids'])
    for split in ('train', 'validation'):
        rec = out['baseline_match'][split]
        if rec.get('match') is False:
            problems.append(f'{split}: ' + str(rec.get('reason') or
                                                'replayed membership does not match the cache'))
        elif rec.get('match') is None:
            warnings.append(
                f'{split}: no cache document-id sidecar was supplied; the '
                'baseline is not independently tied to a frozen cache')
    if problems:
        out['status'] = 'fail'
        return out
    variants = _cs_protocol_variants(protocol)
    variants.extend(_cs_order_variants(n_order_permutations))
    tested = 0
    for variant in variants:
        name = variant.get('name')
        if variant.get('status') != 'planned':
            out['variants'].append(variant)
            continue
        axis = variant.get('axis')
        try:
            if variant.get('kind') == 'protocol':
                p = dict(variant['protocol'])
                tr_order = base_order_tr
                va_order = base_order_va
            else:
                p = dict(protocol)
                if len(train_ne) <= int(fit_rows) and len(val_ne) == 0:
                    variant['status'] = 'skipped'
                    variant['skip_reason'] = ('no post-fit row to permute')
                    out['variants'].append(variant)
                    continue
                if len(train_ne) > int(fit_rows):
                    tr_order = _cs_order(
                        len(train_ne), fit_rows,
                        variant.get('order_kind'),
                        seed=seed,
                        permutation_index=variant.get('permutation_index', 0))
                else:
                    tr_order = base_order_tr
                if len(val_ne) > 1:
                    va_order = _cs_order(
                        len(val_ne), 0, variant.get('order_kind'),
                        seed=seed + 1009,
                        permutation_index=variant.get('permutation_index', 0))
                else:
                    va_order = base_order_va
            run = _run(p, tr_order, va_order)
            rec = {
                'name': name,
                'axis': axis,
                'kind': variant.get('kind'),
                'status': 'tested',
                'rationale': variant.get('rationale'),
                'protocol_fp': dedup_protocol_fp(p),
                'train': dict(_cs_metrics(baseline['train_kept_ids'],
                                          run['train_kept_ids']),
                              removed=run['train_removed']),
                'validation': dict(_cs_metrics(baseline['validation_kept_ids'],
                                               run['validation_kept_ids']),
                                   removed=run['validation_removed']),
            }
            material = False
            for split in ('train', 'validation'):
                m = rec[split]
                if (float(m['kept_jaccard']) < jaccard_floor
                        or float(m['count_delta_fraction']) > count_fraction_floor):
                    material = True
            rec['sequence_changed'] = bool(
                not (rec['train']['kept_sequence_identical']
                     and rec['validation']['kept_sequence_identical']))
            rec['material'] = bool(material)
            out['variants'].append(rec)
            tested += 1
        except (TypeError, ValueError, OverflowError) as exc:
            variant = dict(variant)
            variant['status'] = 'error'
            variant['error'] = f'{type(exc).__name__}: {exc}'
            out['variants'].append(variant)
    tested_recs = [v for v in out['variants'] if v.get('status') == 'tested']
    material_names = [v['name'] for v in tested_recs if v.get('material')]
    membership_material = []
    sequence_changed = []
    for v in tested_recs:
        if v.get('material'):
            membership_material.append(v['name'])
        if v.get('sequence_changed'):
            sequence_changed.append(v['name'])
    tr_j = [float(v['train']['kept_jaccard']) for v in tested_recs]
    va_j = [float(v['validation']['kept_jaccard']) for v in tested_recs]
    tr_c = [float(v['train']['count_delta_fraction']) for v in tested_recs]
    va_c = [float(v['validation']['count_delta_fraction']) for v in tested_recs]
    all_j = tr_j + va_j
    all_c = tr_c + va_c
    out['summary'] = {
        'n_variants_planned': int(len(variants)),
        'n_variants_tested': int(tested),
        'n_variants_skipped': int(sum(1 for v in out['variants']
                                      if v.get('status') in ('skipped', 'error'))),
        'min_kept_jaccard': (min(all_j) if all_j else None),
        'min_kept_jaccard_train': (min(tr_j) if tr_j else None),
        'min_kept_jaccard_validation': (min(va_j) if va_j else None),
        'max_count_delta_fraction': (max(all_c) if all_c else None),
        'max_count_delta_fraction_train': (max(tr_c) if tr_c else None),
        'max_count_delta_fraction_validation': (max(va_c) if va_c else None),
        'n_sequence_changed_variants': int(len(sequence_changed)),
        'sequence_changed_variants': sequence_changed,
        'membership_material_variants': membership_material,
        'material_variants': material_names,
        'material_axes': sorted({str(v.get('axis')) for v in tested_recs
                                 if v.get('material')}),
        'membership_stable': bool(tested and not membership_material),
        'order_identical': bool(tested and not sequence_changed),
        'fully_identical': bool(tested and not material_names
                                and not sequence_changed),
        'stable': bool(tested and not membership_material),
    }
    if not tested:
        out['status'] = 'not_verifiable'
        warnings.append('no valid alternative cleaning specification could be '
                        'tested; sensitivity is unmeasured')
    elif material_names:
        out['status'] = 'warnings'
        warnings.append(
            f'{len(material_names)} of {tested} tested cleaning variants change '
            'the kept membership beyond the declared stability floors; report '
            'the data-cleaning specification curve together with the frozen '
            'split and do not describe the served set as protocol-neutral')
    if out['baseline_match']['train'].get('match') is None and out['status'] == 'ok':
        out['status'] = 'warnings'
    out['problems'] = problems
    out['warnings'] = warnings
    return out


def cleaning_sensitivity_audit(cache_dir=None, raw_dir=None, *,
                               max_rows=None, n_order_permutations=2,
                               train_seq_len=None, seed=0):
    """Zero-GPU wrapper: read one cache + its raw dump and run the curve.

    Only ``wiki.train.raw`` and ``wiki.valid.raw`` are opened.  ``wiki.test.raw``
    is never read: the sensitivity question concerns the training/selection
    corpus, and reading the held-out split would consume the one-shot access.
    Missing raw files, metadata, sidecars or protocol are ``not_verifiable``,
    never a clean sensitivity result.
    """
    import glob
    if cache_dir in (None, ''):
        cache_dir = os.environ.get('WT103_CACHE_DIR') or './wt103_cache'
    if raw_dir in (None, ''):
        # Match the loader/cleaning-replay discovery order instead of forcing
        # the caller to pass a directory that the loader already found.
        _cands = []
        _env_raw = os.environ.get('WT103_RAW_TXT')
        if _env_raw:
            _cands.append(_env_raw)
        _cands.extend((
            os.path.join(os.path.dirname(os.path.abspath(cache_dir)),
                         'wt103_raw'),
            './wt103_raw',
            '/root/wt103_raw',
        ))
        raw_dir = _cands[0]
        for _cand in _cands:
            if (os.path.exists(os.path.join(_cand, 'wiki.train.raw'))
                    and os.path.exists(os.path.join(_cand, 'wiki.valid.raw'))):
                raw_dir = _cand
                break
    out = {
        'kind': 'cleaning_sensitivity_audit',
        'status': 'ok',
        'cache_dir': os.path.abspath(cache_dir) if os.path.isdir(cache_dir) else str(cache_dir),
        'raw_dir': os.path.abspath(raw_dir) if os.path.isdir(raw_dir) else str(raw_dir),
        'test_raw_read': False,
        'problems': [],
        'warnings': [],
    }
    if not os.path.isdir(cache_dir):
        out['status'] = 'not_verifiable_locally'
        out['note'] = f'{cache_dir!r} does not exist; no cache to describe'
        return out
    metas = sorted(glob.glob(os.path.join(cache_dir, 'meta_v5_*.json')))
    if not metas:
        metas = sorted(p for p in glob.glob(os.path.join(cache_dir, 'meta_*.json'))
                       if not os.path.basename(p).startswith('test_meta_'))
    if train_seq_len is not None:
        want = _cs_int(train_seq_len, 'train_seq_len', minimum=1)
        keep = []
        for path in metas:
            meta = _cs_load_json(path)
            if isinstance(meta, dict) and meta.get('seq_len') is not None:
                try:
                    if int(meta['seq_len']) == want:
                        keep.append(path)
                except (TypeError, ValueError, OverflowError):
                    pass
        metas = keep
    if len(metas) != 1:
        out['status'] = 'not_verifiable_locally'
        out['note'] = (f'{len(metas)} train cache metadata file(s) matched; '
                       'refusing to guess which split defines the sensitivity '
                       'baseline' if metas else
                       'no train cache metadata found; build or copy the cache '
                       'before describing its cleaning sensitivity')
        return out
    meta_path = metas[0]
    meta = _cs_load_json(meta_path)
    if not isinstance(meta, dict):
        out['status'] = 'fail'
        out['problems'].append(f'{os.path.basename(meta_path)} is not a readable '
                               'JSON object')
        return out
    tag = os.path.basename(meta_path)[len('meta_'):-len('.json')]
    out['cache'] = {'meta': os.path.basename(meta_path), 'tag': tag}
    protocol = dedup_protocol_from_meta(meta)
    if protocol is None:
        out['status'] = 'not_verifiable_locally'
        out['warnings'].append(
            'the cache records no full dedup protocol; the recorded cleaning '
            'specification cannot be replayed')
        return out
    fit_rows = meta.get('tokenizer_fit_rows')
    if isinstance(fit_rows, bool) or not isinstance(fit_rows, numbers.Integral):
        out['status'] = 'not_verifiable_locally'
        out['warnings'].append(
            'the cache records no exact tokenizer_fit_rows; the decision '
            'sequence cannot be replayed')
        return out
    sampling = meta.get('sampling') if isinstance(meta.get('sampling'), dict) else {}
    train_block = sampling.get('train') if isinstance(sampling.get('train'), dict) else {}
    val_block = sampling.get('validation') if isinstance(sampling.get('validation'), dict) else {}
    train_scanned = train_block.get('nonempty_rows_scanned')
    val_scanned = val_block.get('nonempty_rows_scanned')
    if (isinstance(train_scanned, bool) or not isinstance(train_scanned, numbers.Integral)
            or isinstance(val_scanned, bool) or not isinstance(val_scanned, numbers.Integral)):
        out['status'] = 'not_verifiable_locally'
        out['warnings'].append(
            'the cache does not record exact train/validation nonempty row '
            'counts; a replay of a different row range would describe another '
            'corpus')
        return out
    if max_rows is None:
        max_rows = CLEANING_SENSITIVITY_MAX_ROWS
    try:
        max_rows = _cs_int(max_rows, 'max_rows')
    except ValueError as exc:
        out['status'] = 'fail'
        out['problems'].append(
            f'cannot replay the recorded cleaning protocol (ValueError: {exc})')
        return out
    if max_rows:
        # The full raw dump is materialised below, so a declared dump above
        # the row budget must be refused BEFORE reading it.  The direct helper
        # can only guard after materialisation; the audit wrapper has the
        # recorded source/scanned counts and can fail closed here.
        _over = []
        for _label, _count in (
                ('train source rows', train_block.get('source_rows')),
                ('validation source rows', val_block.get('source_rows')),
                ('train rows', train_scanned),
                ('validation rows', val_scanned),
                ('fit rows', fit_rows)):
            if (isinstance(_count, numbers.Integral)
                    and not isinstance(_count, (bool, np.bool_))
                    and int(_count) > int(max_rows)):
                _over.append((_label, int(_count)))
        if _over:
            out.update({
                'kind': 'cleaning_sensitivity_audit',
                'schema': CLEANING_SENSITIVITY_SCHEMA,
                'status': 'not_verifiable_locally',
                'policy': dict(CLEANING_SENSITIVITY_POLICY),
                'protocol': dict(protocol),
                'protocol_fp': dedup_protocol_fp(protocol),
                'max_rows': int(max_rows),
                'problems': [
                    f'{label} would be replayed above max_rows='
                    f'{int(max_rows)}; rerun explicitly with a larger '
                    'max_rows to measure this corpus'
                    for label, _count in _over],
                'warnings': list(out.get('warnings') or []),
                'note': ('a sensitivity readout that skips rows is not the '
                         'corpus; raise max_rows deliberately'),
                'cache': {'meta': os.path.basename(meta_path), 'tag': tag},
                'test_raw_read': False,
            })
            return out
    raw_train = os.path.join(raw_dir, 'wiki.train.raw')
    raw_val = os.path.join(raw_dir, 'wiki.valid.raw')
    if not (os.path.exists(raw_train) and os.path.exists(raw_val)):
        out['status'] = 'not_verifiable_locally'
        out['warnings'].append(
            f'raw train/validation dumps not found under {raw_dir!r}; the '
            'cleaning sensitivity cannot be replayed')
        return out
    try:
        train_rows = _read_raw_rows(raw_train)
        val_rows = _read_raw_rows(raw_val)
    except Exception as exc:
        out['status'] = 'not_verifiable_locally'
        out['warnings'].append(
            f'raw dumps are unreadable ({type(exc).__name__}: {exc})')
        return out
    train_ids_path = os.path.join(cache_dir, f'train_doc_ids_{tag}.npy')
    val_ids_path = os.path.join(cache_dir, f'val_doc_ids_{tag}.npy')
    expected_train = None
    expected_val = None
    if os.path.exists(train_ids_path):
        try:
            expected_train = np.load(train_ids_path)
            if np.asarray(expected_train).size == 0:
                out['warnings'].append(
                    'train document-id sidecar is empty; an empty served '
                    'training set cannot certify a replayed baseline')
                expected_train = None
        except Exception as exc:
            out['problems'].append(
                f'train document-id sidecar is unreadable '
                f'({type(exc).__name__}: {exc})')
    else:
        out['warnings'].append('train document-id sidecar is missing; the '
                               'baseline cannot be tied to the frozen cache')
    if os.path.exists(val_ids_path):
        try:
            expected_val = np.load(val_ids_path)
            if np.asarray(expected_val).size == 0 and any(
                    isinstance(r, str) and r.strip() for r in val_rows):
                out['warnings'].append(
                    'validation document-id sidecar is empty while the raw '
                    'validation source is not; the baseline cannot be tied to '
                    'the frozen selection split')
                expected_val = None
        except Exception as exc:
            out['problems'].append(
                f'validation document-id sidecar is unreadable '
                f'({type(exc).__name__}: {exc})')
    else:
        out['warnings'].append('validation document-id sidecar is missing; the '
                               'baseline cannot be tied to the frozen cache')
    if out['problems']:
        out['status'] = 'fail'
        return out
    try:
        result = cleaning_sensitivity(
            train_rows, val_rows, protocol=protocol, fit_rows=int(fit_rows),
            train_scanned=int(train_scanned), validation_scanned=int(val_scanned),
            expected_train_doc_ids=expected_train,
            expected_validation_doc_ids=expected_val,
            expected_train_source_rows=train_block.get('source_rows'),
            expected_validation_source_rows=val_block.get('source_rows'),
            max_rows=max_rows, n_order_permutations=n_order_permutations,
            seed=seed)
    except (TypeError, ValueError, OverflowError) as exc:
        out['status'] = 'fail'
        out['problems'].append(f'cannot replay the recorded cleaning protocol '
                               f'({type(exc).__name__}: {exc})')
        return out
    _pre_warnings = list(out.get('warnings') or [])
    out.update(result)
    out['warnings'] = list(dict.fromkeys(_pre_warnings
                                         + list(out.get('warnings') or [])))
    out['kind'] = 'cleaning_sensitivity_audit'
    out['cache'] = {'meta': os.path.basename(meta_path), 'tag': tag}
    out['test_raw_read'] = False
    if expected_train is None or expected_val is None:
        # A numeric curve against a baseline that is not tied to the frozen
        # cache is still useful, but it must not be reported as verified.  Do
        # not let a material warning (which is about the sensitivity result,
        # not about provenance) upgrade the missing sidecar to a pass.
        out['status'] = 'not_verifiable_locally'
    return out
