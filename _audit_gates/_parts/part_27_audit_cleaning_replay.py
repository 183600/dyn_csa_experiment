# Audit a local WikiText cache by replaying its cleaning decisions.
#
# Counts and a protocol fingerprint are self-consistent claims: they show that a
# cache is *internally* coherent, not that its served documents are the ones the
# declared geometry actually keeps for the raw split.  This gate recomputes the
# entire decision sequence (fit-prefix seeding, train/validation dedup, and --
# only when explicitly requested -- the held-out test cleanup) from the raw rows
# and compares the kept document ids with the cache sidecars.
#
# Without a raw dump, a full protocol, a fit-row count or the document-id
# sidecars the result is ``not_verifiable``, never ``ok``.  ``wiki.test.raw`` is
# never opened unless ``include_test`` is requested (env
# ``CSA_REPLAY_INCLUDE_TEST=1`` or the second/third positional argument).
def audit_cleaning_replay(cache_dir=None, raw_dir=None, *,
                          train_seq_len=None, include_test=None, tag=None,
                          max_rows=None):
    """Zero-model audit: is the cache's cleaned membership reproducible?

    ``not_verifiable`` means an input needed for the replay is missing, not
    that the cache is clean.  ``fail`` means the replayed kept set disagrees
    with the served sidecar, a recorded count, or the validation-cleanup
    ledger, or that the supplied raw dump does not match the cache's declared
    source-row counts.

    Reading ``wiki.test.raw`` is a held-out access.  It is opt-in through
    ``include_test=True`` (or ``CSA_REPLAY_INCLUDE_TEST=1``) and is reported in
    ``test_raw_read`` so a reader can tell a train/validation replay from a
    held-out one.
    """
    import os
    L = _import_exp_lib()
    if include_test is None:
        include_test = str(os.environ.get('CSA_REPLAY_INCLUDE_TEST', '')
                           ).strip().lower() in ('1', 'true', 'yes', 'on')
    if max_rows is None:
        raw_max = os.environ.get('CSA_REPLAY_MAX_ROWS')
        if raw_max not in (None, ''):
            try:
                max_rows = int(raw_max)
            except (TypeError, ValueError):
                max_rows = L.CLEANING_REPLAY_MAX_ROWS
        else:
            max_rows = L.CLEANING_REPLAY_MAX_ROWS
    if tag is None:
        tag = os.environ.get('CSA_REPLAY_TAG') or None
    if train_seq_len is None:
        env_len = os.environ.get('CSA_REPLAY_TRAIN_SEQ_LEN')
        if env_len not in (None, ''):
            try:
                train_seq_len = int(env_len)
            except (TypeError, ValueError):
                raise SystemExit(
                    f'[audit] CSA_REPLAY_TRAIN_SEQ_LEN must be an integer, got {env_len!r}')
    if train_seq_len is not None:
        try:
            train_seq_len = int(train_seq_len)
        except (TypeError, ValueError):
            raise SystemExit(f'[audit] train_seq_len must be an integer, got {train_seq_len!r}')
    out = L.replay_wikitext_cleaning(
        cache_dir=cache_dir, raw_dir=raw_dir,
        train_seq_len=train_seq_len, include_test=bool(include_test),
        tag=tag, max_rows=max_rows)
    out['kind'] = 'cleaning_replay_audit'
    out['max_rows'] = max_rows
    # A cache is only certified when the train/validation membership matched.
    # The test block is reported separately because it is an explicit held-out
    # access; an include_test=False run must not imply test was checked.
    out['scope'] = ({'train_validation': True, 'heldout_test': True}
                    if include_test else
                    {'train_validation': True, 'heldout_test': False})
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    return out
