# ---------------------------------------------------------------------------
# 100. The held-out split is evaluated once: counted access, not a free rerun
#
# `_check_or_write_freeze_lock` already refuses a second P5T access unless both
# `V12_REFREEZE=1` and `V12_ALLOW_TEST_REREAD=1` are set, and it increments
# `test_reads` / appends a `reread` event when that happens.  The counter was,
# however, write-only: `finaltest_stats` calls the lock in `analysis` mode and
# never reads it.  A panel that has been evaluated twice (or whose plan/test
# slice was re-frozen after a first read) can therefore still be reported as
# "confirmed" with the nominal alpha.
#
# That is the held-out analogue of the exploratory look-history problem in
# section 58: re-reading the same test data with new checkpoints, code or an
# analysis variant is a selection opportunity.  This fragment makes the access
# budget a machine-checkable pre-registration (`statistics.test_access_policy`)
# and turns the recorded access pattern into an effective verdict.  It never
# recomputes a frozen p-value, changes a Holm family, or rewrites `confirmed`;
# it only decides whether the recorded access pattern can support a
# confirmatory claim and, when a repeated-look budget was declared BEFORE the
# first access, applies the declared alpha spending.
# ---------------------------------------------------------------------------

TEST_ACCESS_POLICY = {
    'schema': 1,
    'method': 'single_look',
    'max_looks': 1,
    'require_reason_for_reread': True,
    'methods': ('single_look', 'bonferroni'),
    'allowed_keys': (
        'schema', 'method', 'max_looks', 'require_reason_for_reread',
        'rationale', 'note', 'description', 'source'),
    'max_looks_limit': 1000,
    'role': ('pre-registered held-out evaluation budget.  Repeated reads of '
             'the frozen test slice are selection opportunities, not free '
             'reruns.  This policy adds no test to the frozen family; it only '
             'labels whether the recorded access pattern supports a '
             'confirmatory claim and, for a pre-declared multi-look design, '
             'applies the declared alpha spending.'),
}


def _ta_policy_int(name, value, *, minimum, maximum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, numbers.Integral):
        raise ValueError(f'test_access_policy.{name} must be an integer, '
                         f'got {value!r}')
    out = int(value)
    if out < minimum or out > maximum:
        raise ValueError(f'test_access_policy.{name} must be in '
                         f'[{minimum}, {maximum}], got {out}')
    return out


def test_access_policy(manifest=None):
    """Return the normalised held-out access policy bound by ``manifest``.

    The absence of ``statistics.test_access_policy`` means the historical
    single-look contract: exactly one evaluation of the frozen test slice is
    confirmatory, and any additional read is a protocol deviation.  A plan may
    pre-declare a repeated-look budget (``method='bonferroni'``, ``max_looks``)
    before its first read; the correction is then applied against the PLANNED
    look count, not the observed one, so stopping early cannot buy back alpha.

    A malformed policy raises ``ValueError`` rather than falling back to the
    permissive default.
    """
    raw = None
    source = None
    if isinstance(manifest, dict):
        stat = manifest.get('statistics')
        if isinstance(stat, dict) and 'test_access_policy' in stat:
            raw = stat.get('test_access_policy')
            source = 'statistics.test_access_policy'
        elif 'test_access_policy' in manifest:
            raw = manifest.get('test_access_policy')
            source = 'test_access_policy'
    if raw is None:
        out = {k: TEST_ACCESS_POLICY[k] for k in
               ('schema', 'method', 'max_looks', 'require_reason_for_reread')}
        out.update({'declared': False, 'source': None, 'rationale': None,
                    'role': TEST_ACCESS_POLICY['role']})
        return out
    if not isinstance(raw, dict):
        raise ValueError('test_access_policy must be a JSON object, got '
                         f'{type(raw).__name__}')
    unknown = sorted(set(raw) - set(TEST_ACCESS_POLICY['allowed_keys']))
    if unknown:
        # A typo such as `max_look` must not silently fall back to one look.
        raise ValueError(f'test_access_policy has unknown key(s) {unknown}')
    method = raw.get('method', 'single_look')
    if not isinstance(method, str) or method not in TEST_ACCESS_POLICY['methods']:
        raise ValueError('test_access_policy.method must be one of '
                         f'{TEST_ACCESS_POLICY["methods"]}, got {method!r}')
    max_looks = _ta_policy_int(
        'max_looks', raw.get('max_looks', 1), minimum=1,
        maximum=TEST_ACCESS_POLICY['max_looks_limit'])
    if method == 'single_look' and max_looks != 1:
        raise ValueError('method=single_look requires max_looks=1; declare '
                         'method=bonferroni for a repeated-look budget')
    require_reason = raw.get('require_reason_for_reread', True)
    if not isinstance(require_reason, (bool, np.bool_)):
        raise ValueError('test_access_policy.require_reason_for_reread must be '
                         f'a boolean, got {require_reason!r}')
    rationale = raw.get('rationale')
    if rationale is not None and not isinstance(rationale, str):
        raise ValueError('test_access_policy.rationale must be a string, got '
                         f'{type(rationale).__name__}')
    schema = raw.get('schema', 1)
    if isinstance(schema, (bool, np.bool_)) or not isinstance(
            schema, numbers.Integral):
        raise ValueError('test_access_policy.schema must be an integer, got '
                         f'{schema!r}')
    schema = int(schema)
    if schema != TEST_ACCESS_POLICY['schema']:
        raise ValueError(f'test_access_policy.schema {schema} is not supported '
                         f'(this analyzer implements {TEST_ACCESS_POLICY["schema"]})')
    return {
        'schema': schema,
        'method': method,
        'max_looks': max_looks,
        'require_reason_for_reread': bool(require_reason),
        'declared': True,
        'source': source,
        'rationale': rationale,
        'role': TEST_ACCESS_POLICY['role'],
    }


def _ta_load_lock(lock):
    """Return ``(obj, path, error)`` for a lock dict or JSON path."""
    if isinstance(lock, dict):
        return lock, None, None
    if isinstance(lock, (str, bytes, os.PathLike)):
        path = os.fsdecode(os.fspath(lock))
        if not os.path.exists(path):
            return None, path, 'lock_missing'
        try:
            with open(path, encoding='utf-8') as fh:
                obj = json.load(fh)
        except Exception as exc:
            return None, path, f'unreadable: {type(exc).__name__}: {exc}'
        if not isinstance(obj, dict):
            return None, path, f'not a JSON object ({type(obj).__name__})'
        return obj, path, None
    return None, None, f'unsupported lock type {type(lock).__name__}'


def _ta_number(name, value, *, minimum=None, maximum=None):
    if isinstance(value, (bool, np.bool_)) or isinstance(value, str) or value is None:
        raise ValueError(f'{name} must be a number, got {value!r}')
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name} must be a number, got {value!r}')
    if not math.isfinite(out):
        raise ValueError(f'{name} must be finite, got {value!r}')
    if minimum is not None and out <= minimum:
        raise ValueError(f'{name} must be > {minimum}, got {out}')
    if maximum is not None and out >= maximum:
        raise ValueError(f'{name} must be < {maximum}, got {out}')
    return out


def _ta_same(a, b):
    """Comparison that does not let bool/str stand in for a number."""
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return (isinstance(a, (bool, np.bool_))
                and isinstance(b, (bool, np.bool_))
                and bool(a) == bool(b))
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        # Exact integer comparison.  Converting an arbitrarily large JSON
        # integer to float raises OverflowError and would abort the replay
        # instead of reporting a mismatch.
        return int(a) == int(b)
    if isinstance(a, (int, float, np.integer, np.floating)) and isinstance(
            b, (int, float, np.integer, np.floating)):
        try:
            return abs(float(a) - float(b)) <= 1e-12 * max(
                1.0, abs(float(a)), abs(float(b)))
        except (OverflowError, ValueError):
            # An int/float mixed pair at an unrepresentable scale: Python's
            # exact equality is the only safe comparison left.
            return a == b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_ta_same(x, y) for x, y in zip(a, b))
    return a == b


def confirmatory_access_audit(lock, *, stats=None, manifest=None, alpha=None,
                              lock_label=None):
    """Replay the held-out access ledger and decide whether it can confirm.

    ``lock`` is a parsed freeze lock or its JSON path.  ``stats`` is an
    optional ``analysis_*/test_stats.json`` body: when supplied, the frozen
    p/Holm table is cross-checked against the lock's manifest/test-slice
    identity and the stored access block is replayed.  ``manifest`` supplies
    the pre-registered ``statistics.test_access_policy``; without it the
    historical single-look contract applies.

    The return value is an audit record, not a new test.  It adds no p-value to
    the frozen family and never changes ``confirmed``; it exports
    ``confirmed_effective`` (empty unless the access pattern is admissible)
    and, for a pre-declared Bonferroni look budget, the adjusted p/alpha.
    """
    obj, path, err = _ta_load_lock(lock)
    label = lock_label if lock_label is not None else path
    if isinstance(label, (bytes, bytearray)):
        label = os.fsdecode(bytes(label))
    out = {
        'kind': 'test_access_audit',
        'schema': 1,
        'status': 'not_verifiable_locally',
        'lock': label,
        'lock_present': False,
        'stage': None,
        'n_test_reads': None,
        'test_reads_inferred': False,
        'n_reread_events': 0,
        'n_rereads_without_reason': 0,
        'n_accessed_events': 0,
        'n_post_access_plan_changes': 0,
        'n_pre_access_plan_revisions': 0,
        'n_analysis_source_drift_events': 0,
        'test_slice_changed_after_access': False,
        'manifest_changed_since_lock': False,
        'policy': None,
        'alpha_nominal': None,
        'alpha_effective': None,
        'confirmatory_allowed': None,
        'descriptive_only': False,
        'downgrade_reasons': [],
        'confirmed_frozen': None,
        'confirmed_effective': None,
        'comparisons': {},
        'problems': [],
        'warnings': [],
        'note': ('held-out access accounting only: it does not recompute a '
                 'frozen p-value, change a Holm family or rewrite `confirmed`. '
                 'It decides whether the recorded access pattern can support a '
                 'confirmatory claim.'),
    }
    if err is not None:
        out['status'] = 'fail'
        out['confirmatory_allowed'] = False
        out['problems'].append(f'cannot read the freeze lock: {err}')
        return out
    out['lock_present'] = True
    if not any(k in obj for k in ('stage', 'test_reads', 'test_ids_fp',
                                  'manifest_sha256')):
        out['status'] = 'fail'
        out['confirmatory_allowed'] = False
        out['problems'].append(
            'the lock carries none of stage/test_reads/test_ids_fp/'
            'manifest_sha256; an empty object cannot attest any access')
        return out

    # ---- structural validation ------------------------------------------------
    for key in ('history', 'superseded', 'source_tree_drift'):
        val = obj.get(key)
        if val is None:
            obj[key] = []
        elif not isinstance(val, list):
            out['problems'].append(f'{key} must be a list, got '
                                   f'{type(val).__name__}')
            obj[key] = []
    history = obj['history']
    superseded = obj['superseded']
    drift = obj['source_tree_drift']
    for key, seq in (('history', history), ('superseded', superseded),
                     ('source_tree_drift', drift)):
        for i, item in enumerate(seq):
            if not isinstance(item, dict):
                out['problems'].append(
                    f'{key}[{i}] must be an object, got {type(item).__name__}')

    stage = obj.get('stage')
    if stage is not None and not isinstance(stage, str):
        out['problems'].append(f'stage must be a string, got {type(stage).__name__}')
        stage = None
    out['stage'] = stage

    test_fp = obj.get('test_ids_fp')
    if test_fp is not None and (not isinstance(test_fp, str) or not test_fp.strip()):
        out['problems'].append('test_ids_fp must be a non-empty string or null')
        test_fp = None

    reread_events = [h for h in history
                     if isinstance(h, dict) and h.get('stage') == 'reread']
    out['n_reread_events'] = len(reread_events)
    out['n_rereads_without_reason'] = sum(
        1 for h in reread_events
        if not (isinstance(h.get('reason'), str) and h['reason'].strip()))
    out['n_accessed_events'] = sum(
        1 for h in history if isinstance(h, dict) and h.get('stage') == 'accessed')
    out['n_pre_access_plan_revisions'] = sum(
        1 for p in superseded
        if isinstance(p, dict) and p.get('test_ids_fp') is None)

    raw_reads = obj.get('test_reads')
    if raw_reads is None:
        out['test_reads_inferred'] = True
        n_reads = (len(reread_events) + 1) if test_fp is not None else 0
    else:
        if isinstance(raw_reads, (bool, np.bool_)) or not isinstance(
                raw_reads, numbers.Integral):
            out['problems'].append(
                f'test_reads must be an integer, got {raw_reads!r}')
            n_reads = None
        else:
            n_reads = int(raw_reads)
            # Preserve the exact reported value for the replay, but do not
            # let an implausible counter enter the arithmetic below.  A lock
            # is untrusted JSON: ``10**400`` is a representable Python int
            # yet not a usable float/read count, and used to raise
            # OverflowError at the Bonferroni display multiplication.
            out['n_test_reads'] = n_reads
            if abs(n_reads) > TEST_ACCESS_POLICY['max_looks_limit']:
                # Do not echo/format the arbitrary-precision counter: a value
                # above CPython's 4,300-digit conversion limit cannot even be
                # turned into the problem string (or JSON) without raising.
                out['problems'].append(
                    'test_reads (bit_length='
                    f'{n_reads.bit_length()}) is outside the declared maximum '
                    f'{TEST_ACCESS_POLICY["max_looks_limit"]}; the access '
                    'counter is not plausible')
                out['n_test_reads'] = None
                out['n_test_reads_bit_length'] = int(n_reads.bit_length())
                n_reads = None
            elif n_reads < 0:
                out['problems'].append(f'test_reads must be >= 0, got {n_reads}')
    if 'n_test_reads' not in out:
        out['n_test_reads'] = n_reads

    if test_fp is not None and n_reads is not None and n_reads < 1:
        out['problems'].append(
            'test_reads is 0 but a test-slice fingerprint is bound; the lock '
            'is internally inconsistent')
    if test_fp is None and n_reads is not None and n_reads > 0:
        out['problems'].append(
            'test_reads is positive but no test-slice fingerprint is bound; '
            'the access cannot be tied to a specific held-out slice')
    if n_reads is not None and test_fp is not None:
        expected = max(0, n_reads - 1)
        if len(reread_events) != expected:
            out['problems'].append(
                f'test_reads={n_reads} implies {expected} reread event(s), '
                f'but history records {len(reread_events)}')
        if n_reads >= 1 and out['n_accessed_events'] == 0 and history:
            out['warnings'].append(
                'the lock has test_reads>=1 but no `accessed` event in its '
                'history; it predates the two-phase access record')
    if n_reads is None:
        out['status'] = 'fail'

    # Re-freeze events: a prior plan that had already read the test slice is a
    # post-access plan change.  A prior plan that only reached `preaccess` is
    # an ordinary pre-registration revision and is not penalised.
    for i, prior in enumerate(superseded):
        if not isinstance(prior, dict):
            continue
        prior_fp = prior.get('test_ids_fp')
        if prior_fp is not None:
            out['n_post_access_plan_changes'] += 1
            if not isinstance(prior_fp, str) or not prior_fp.strip():
                out['problems'].append(
                    f'superseded[{i}].test_ids_fp is malformed')
            elif test_fp is not None and prior_fp != test_fp:
                out['test_slice_changed_after_access'] = True
        if (isinstance(prior.get('manifest_sha256'), str)
                and isinstance(obj.get('manifest_sha256'), str)
                and prior['manifest_sha256'] == obj['manifest_sha256']):
            out['warnings'].append(
                f'superseded[{i}] carries the same manifest hash as the '
                'current lock; the re-freeze did not change the plan')
    if out['n_post_access_plan_changes']:
        out['warnings'].append(
            f'{out["n_post_access_plan_changes"]} prior plan(s) had already '
            'read the held-out slice before being superseded; the earlier '
            'round is exploratory and this plan is a new look')

    for i, ev in enumerate(drift):
        if not isinstance(ev, dict):
            continue
        if ev.get('stage') == 'analysis':
            out['n_analysis_source_drift_events'] += 1
    if out['n_analysis_source_drift_events']:
        out['warnings'].append(
            f'{out["n_analysis_source_drift_events"]} disclosed analysis-source '
            'drift event(s) after test access; the confirmatory analysis is '
            'not a byte-identical replay of the frozen tree (the frozen '
            'measurement remains bound by the lock)')

    # The repository's current manifest is not automatically the locked plan.
    # A hash mismatch is a warning here (the lock remains the attestation); a
    # stats record produced under a different manifest is a hard problem below.
    if isinstance(manifest, dict):
        cur_hash = manifest.get('manifest_sha256')
        lock_hash = obj.get('manifest_sha256')
        if isinstance(cur_hash, str) and isinstance(lock_hash, str) \
                and cur_hash != lock_hash:
            out['manifest_changed_since_lock'] = True
            out['warnings'].append(
                'the current manifest file hash differs from the hash bound at '
                'first test access; the lock, not the working-tree file, is '
                'the frozen plan')
    try:
        policy = test_access_policy(manifest)
    except ValueError as exc:
        out['problems'].append(str(exc))
        policy = None
    out['policy'] = policy
    if (policy is not None and policy.get('require_reason_for_reread')
            and out['n_rereads_without_reason']):
        out['warnings'].append(
            f'{out["n_rereads_without_reason"]} reread event(s) carry no '
            'V12_TEST_REREAD_REASON; the extra look is real but unexplained')

    # ---- alpha and effective verdict -----------------------------------------
    alpha_value = alpha
    if alpha_value is None and isinstance(stats, dict):
        alpha_value = stats.get('alpha')
    if alpha_value is None and isinstance(manifest, dict):
        stat = manifest.get('statistics')
        if isinstance(stat, dict):
            alpha_value = stat.get('alpha')
    if alpha_value is None:
        alpha_value = 0.05
    try:
        alpha_value = _ta_number('alpha', alpha_value, minimum=0.0,
                                 maximum=1.0)
    except ValueError as exc:
        out['problems'].append(str(exc))
        alpha_value = None
    out['alpha_nominal'] = alpha_value

    if policy is not None:
        out['alpha_effective'] = (
            alpha_value / policy['max_looks']
            if (alpha_value is not None and policy['method'] == 'bonferroni')
            else alpha_value)

    # A `descriptive_only` plan (smoke, dynamic/HCA draft, exploratory
    # replication) makes no confirmatory claim, so its access pattern cannot be
    # a confirmatory deviation.  It is still replayed for structural integrity,
    # but reported as not_applicable rather than counted as a confirmatory or
    # downgraded look.
    descriptive_only = False
    if isinstance(stats, dict):
        _sman = stats.get('manifest')
        _sst = _sman.get('statistics') if isinstance(_sman, dict) else None
        descriptive_only = (isinstance(_sst, dict)
                            and _sst.get('descriptive_only') is True)
    out['descriptive_only'] = bool(descriptive_only)

    reasons = []
    if out['n_post_access_plan_changes']:
        reasons.append('plan_changed_after_access')
    if out['test_slice_changed_after_access']:
        reasons.append('test_slice_changed_after_access')
    if policy is not None and n_reads is not None:
        if n_reads > 1 and policy['method'] == 'single_look':
            reasons.append('unplanned_reread')
        elif n_reads > policy['max_looks']:
            reasons.append('more_looks_than_declared')
    out['downgrade_reasons'] = reasons

    if out['problems']:
        allowed = False
    elif n_reads is None:
        allowed = False
    elif n_reads == 0:
        allowed = None
    else:
        allowed = not reasons
    if descriptive_only and not out['problems']:
        allowed = None
        reasons = []
        out['downgrade_reasons'] = []
    out['confirmatory_allowed'] = allowed

    # Replay the frozen p/Holm table.  `p_holm` is the frozen decision
    # statistic; `p_exact_signflip` is the fallback when a legacy artifact has
    # no Holm block.  The Bonferroni-adjusted p uses the PLANNED look budget,
    # not the observed count, so an early stop cannot undo the spending.
    comps = {}
    holm = {}
    confirmed_frozen = None
    if isinstance(stats, dict):
        # A present-but-malformed identity block is not "no identity".  The
        # old code silently turned e.g. manifest=[] into {}, skipped every
        # cross-check and could report the artifact as clean.
        for _key in ('manifest', 'admission'):
            if _key in stats and stats.get(_key) is not None \
                    and not isinstance(stats.get(_key), dict):
                out['problems'].append(
                    f'stats.{_key} must be an object, got '
                    f'{type(stats.get(_key)).__name__}')
        for _key in ('comparisons', 'holm'):
            if _key in stats and stats.get(_key) is not None \
                    and not isinstance(stats.get(_key), dict):
                out['problems'].append(
                    f'stats.{_key} must be an object, got '
                    f'{type(stats.get(_key)).__name__}')
        if isinstance(stats.get('comparisons'), dict):
            comps = stats['comparisons']
        if isinstance(stats.get('holm'), dict):
            holm = stats['holm']
        raw_conf = stats.get('confirmed')
        if isinstance(raw_conf, (list, tuple)):
            confirmed_frozen = [str(x) for x in raw_conf]
        elif raw_conf is None:
            confirmed_frozen = None
        else:
            out['problems'].append(
                f'confirmed must be a list, got {type(raw_conf).__name__}')
            confirmed_frozen = None
    out['confirmed_frozen'] = confirmed_frozen
    if n_reads == 0 and confirmed_frozen:
        out['problems'].append(
            'the stats artifact reports confirmed comparison(s) but the access '
            'ledger records no held-out test read')
    confirmed_set = set(confirmed_frozen or [])
    effective = []
    planned = (policy['max_looks'] if policy is not None else 1)
    for name in sorted(set(comps) | set(holm)):
        p = None
        src = None
        hv = holm.get(name)
        if isinstance(hv, dict) and hv.get('p_holm') is not None:
            p = hv.get('p_holm')
            src = 'holm.p_holm'
        if p is None and isinstance(comps.get(name), dict):
            p = comps[name].get('p_holm')
            src = 'comparisons.p_holm'
            if p is None:
                p = comps[name].get('p_exact_signflip')
                src = 'comparisons.p_exact_signflip'
        _raw_p = p
        try:
            p = _ta_number('p', p, minimum=-1e-12, maximum=1.000001)
            p = min(1.0, max(0.0, float(p)))
        except ValueError:
            if _raw_p is not None:
                out['problems'].append(
                    f'comparison {name!r} carries a non-numeric or '
                    f'out-of-range p ({_raw_p!r})')
            p = None
            src = None
        p_eff = (min(1.0, p * planned)
                 if (p is not None and policy is not None
                     and policy['method'] == 'bonferroni') else p)
        p_desc = (min(1.0, p * max(n_reads or 1, 1))
                  if p is not None else None)
        rec = {'p': p, 'p_source': src,
               'p_holm_effective': p_eff,
               'p_holm_bonferroni_descriptive': p_desc,
               'confirmed_frozen': name in confirmed_set}
        if (allowed is True and name in confirmed_set and p is not None
                and out['alpha_effective'] is not None
                and p < out['alpha_effective']):
            # Apply the look correction ONCE.  `p_holm_effective` already
            # multiplies by max_looks (for display), so comparing it against
            # the already-divided alpha would require p < alpha/L**2.
            effective.append(name)
        out['comparisons'][name] = rec
    # Every comparison the frozen table calls confirmed must have a replayable
    # p value.  A name that is absent from (or p-less in) both blocks cannot be
    # checked, so it must not silently vanish from `confirmed_effective`
    # while the audit still reports a clean status.
    for _name in sorted(confirmed_set):
        _rec = out['comparisons'].get(_name)
        if _rec is None or _rec.get('p') is None:
            out['problems'].append(
                f'confirmed comparison {_name!r} has no replayable p/Holm '
                'value in the stats artifact')
    if allowed is True:
        out['confirmed_effective'] = effective
    elif allowed is False:
        out['confirmed_effective'] = []
    else:
        out['confirmed_effective'] = None

    # ---- cross-checks against the stored artifact ----------------------------
    if isinstance(stats, dict):
        sman = stats.get('manifest') if isinstance(stats.get('manifest'), dict) else {}
        if (isinstance(sman.get('manifest_sha256'), str)
                and isinstance(obj.get('manifest_sha256'), str)
                and sman['manifest_sha256'] != obj['manifest_sha256']):
            out['problems'].append(
                'the stats artifact binds a different manifest than the freeze '
                'lock; the numbers and the access ledger are not from the same plan')
        sadm = stats.get('admission') if isinstance(stats.get('admission'), dict) else {}
        s_fp = sadm.get('test_slice_fp')
        if (isinstance(s_fp, str) and isinstance(test_fp, str)
                and s_fp != test_fp):
            out['problems'].append(
                'the stats artifact evaluated a different test slice than the '
                'one bound by the freeze lock')
        s_alpha = stats.get('alpha')
        if s_alpha is not None:
            try:
                if not _ta_same(_ta_number('stats.alpha', s_alpha,
                                           minimum=0.0, maximum=1.0),
                                alpha_value):
                    out['problems'].append(
                        'the stats artifact alpha differs from the lock/manifest alpha')
            except ValueError as exc:
                out['problems'].append(str(exc))
        stored = stats.get('test_access')
        if stored is not None:
            if not isinstance(stored, dict):
                out['problems'].append(
                    'the stats artifact test_access block is not an object')
            else:
                # Replay every stored field that is a deterministic function
                # of the lock + stats.  Checking only four keys let a tampered
                # `comparisons`, `policy` or `alpha_nominal` block survive.
                _keys = ['lock_present', 'stage', 'n_test_reads',
                         'test_reads_inferred', 'n_reread_events',
                         'n_rereads_without_reason', 'n_accessed_events',
                         'n_post_access_plan_changes',
                         'n_pre_access_plan_revisions',
                         'n_analysis_source_drift_events',
                         'test_slice_changed_after_access',
                         'confirmatory_allowed', 'descriptive_only',
                         'confirmed_frozen', 'confirmed_effective',
                         'downgrade_reasons']
                # These fields depend on the manifest/policy supplied *now*.
                # When the locked manifest is unavailable, the documented
                # fallback is the historical single-look rule plus a warning;
                # the replay must not turn that fallback into a false tamper
                # verdict merely because the stored block used a declared
                # multi-look policy.
                if isinstance(manifest, dict):
                    _keys.extend(('policy', 'alpha_nominal', 'alpha_effective',
                                  'manifest_changed_since_lock'))
                for key in _keys:
                    if key not in stored:
                        continue
                    if not _ta_same(stored.get(key), out.get(key)):
                        out['problems'].append(
                            f'the stored test_access.{key} does not replay '
                            f'(stored={stored.get(key)!r}, '
                            f'recomputed={out.get(key)!r})')
                _stored_comps = stored.get('comparisons')
                if _stored_comps is not None:
                    if not isinstance(_stored_comps, dict):
                        out['problems'].append(
                            'the stored test_access.comparisons is not an '
                            'object')
                    else:
                        _now_comps = out.get('comparisons') or {}
                        for _name in sorted(set(_stored_comps) | set(_now_comps)):
                            if _name not in _stored_comps or _name not in _now_comps:
                                out['problems'].append(
                                    f'the stored test_access.comparisons has a '
                                    f'different comparison set at {_name!r}')
                                continue
                            _cmp_keys = ('p', 'p_source', 'confirmed_frozen')
                            if isinstance(manifest, dict):
                                _cmp_keys += ('p_holm_effective',
                                              'p_holm_bonferroni_descriptive')
                            for _ck in _cmp_keys:
                                if not _ta_same(_stored_comps[_name].get(_ck)
                                                if isinstance(_stored_comps[_name], dict)
                                                else None,
                                                _now_comps[_name].get(_ck)
                                                if isinstance(_now_comps[_name], dict)
                                                else None):
                                    out['problems'].append(
                                        f'the stored test_access.comparisons'
                                        f'[{_name!r}].{_ck} does not replay')
        if 'confirmed_effective' in stats:
            if not _ta_same(stats.get('confirmed_effective'),
                            out.get('confirmed_effective')):
                out['problems'].append(
                    'the stored confirmed_effective set does not replay from '
                    'the access ledger and the frozen p/Holm table')
    if out['problems']:
        out['confirmatory_allowed'] = False
        if out['confirmed_effective']:
            out['confirmed_effective'] = []

    # ---- status ---------------------------------------------------------------
    if out['problems']:
        out['status'] = 'fail'
    elif allowed is None:
        out['status'] = 'not_applicable'
        out['note'] = (
            'the plan is descriptive_only; its held-out access is recorded but '
            'makes no confirmatory claim' if out['descriptive_only'] else
            'no held-out test evaluation is recorded; there is no confirmatory '
            'access to audit')
    elif allowed is False:
        out['status'] = 'fail'
        out['note'] = ('the recorded access pattern cannot support a '
                       'confirmatory claim: ' + ', '.join(out['downgrade_reasons']))
    elif out['warnings']:
        out['status'] = 'warn'
    else:
        out['status'] = 'ok'
    if policy is not None and policy['declared'] and allowed:
        out['note'] = ('the plan pre-declared a repeated-look budget; '
                       f'alpha_effective={out["alpha_effective"]!r} uses the '
                       f'planned max_looks={policy["max_looks"]}')
    return out
