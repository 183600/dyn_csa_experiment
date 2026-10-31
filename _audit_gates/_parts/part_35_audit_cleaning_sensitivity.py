def audit_cleaning_sensitivity(cache_dir=None, raw_dir=None, *,
                               max_rows=None, n_order_permutations=2,
                               train_seq_len=None, seed=0):
    """Describe how sensitive the served corpus is to data-cleaning choices.

    The cache binds one exact/SimHash geometry and the cleaning replay proves
    that geometry reproduces the served membership.  This gate asks a different
    methodological question: how much of the kept membership is an artefact of
    the nearby cleaning choices (shingle width, LSH bands, Hamming threshold,
    max-shingle subsampling) or of the raw row order?  It replays the recorded
    decision sequence once as the baseline and then under each valid nearby
    protocol and a seeded permutation of the post-fit rows, reporting kept-set
    Jaccard and count changes.

    It reads only ``wiki.train.raw`` / ``wiki.valid.raw``.  ``wiki.test.raw``
    and the test cache are never read: this is a training/selection-corpus
    sensitivity readout, not a held-out access.  Missing metadata, protocol,
    exact scan counts or raw dumps are ``not_verifiable_locally``; a replayed
    baseline that contradicts the cache sidecars is ``fail``.

    The result is descriptive.  ``warnings`` means at least one tested variant
    changes at least 5% of the kept members or 1% of the kept count; it does not
    invalidate the frozen cache, change any p-value, or alter a split.  A kept
    set that is stable but reordered is reported separately by
    ``order_identical`` / ``fully_identical``.  ``max_rows=0`` disables the row
    guard deliberately, matching ``cleaning_replay``.
    """
    import json
    L = _import_exp_lib()
    if max_rows is not None:
        if isinstance(max_rows, (bool, np.bool_)):
            raise ValueError('max_rows must be a non-negative integer or None')
        if not isinstance(max_rows, numbers.Integral):
            raise ValueError('max_rows must be a non-negative integer or None')
        max_rows = int(max_rows)
        if max_rows < 0:
            raise ValueError('max_rows must be a non-negative integer or None')
    if train_seq_len is not None:
        train_seq_len = _audit_positive_int('train_seq_len', train_seq_len)
    if isinstance(n_order_permutations, (bool, np.bool_)):
        raise ValueError('n_order_permutations must be a non-negative integer')
    if not isinstance(n_order_permutations, numbers.Integral):
        raise ValueError('n_order_permutations must be a non-negative integer')
    n_order_permutations = int(n_order_permutations)
    if n_order_permutations < 0:
        raise ValueError('n_order_permutations must be a non-negative integer')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral):
        raise ValueError('seed must be an integer')
    out = L.cleaning_sensitivity_audit(
        cache_dir=cache_dir, raw_dir=raw_dir, max_rows=max_rows,
        n_order_permutations=n_order_permutations,
        train_seq_len=train_seq_len, seed=int(seed))
    out['kind'] = 'cleaning_sensitivity_audit'
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    return out
