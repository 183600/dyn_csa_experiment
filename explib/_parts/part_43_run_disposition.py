"""Pre-registered run disposition for failed ``(arm, seed)`` cells.

A planned cell is not the same thing as a measured cell.  Between the plan and
the final table a cell can fail (NaN/divergence, OOM, watchdog timeout, a
runtime cap, a bad configuration), be truncated, be skipped for budget, or
simply be re-run after a code change.  Every one of those events is a chance
to keep the attempt that looks best, and the chance is not equal across arms:
the expensive/dynamic arms are the ones that hit the cap and OOM most often.
"Run it again until it produces a number" is therefore a selection mechanism,
not an infrastructure detail.

This fragment makes the missing design decision explicit *before* the first
cell runs:

* a **disposition plan** records the planned cells and a policy that says
  which failure classes may be replaced, how many attempts a cell may receive,
  and which successful attempt is binding;
* an **attempt audit** replays ``experiments.jsonl`` and checks that the
  retained measurement is the pre-declared one (by default the *first*
  successful attempt), that no planned cell was silently replaced by another
  seed, that no non-replaceable failure was retried, and that retries are not
  concentrated in one arm.

The audit is a *pre-registration* check, not a new statistical test.  It adds
no p-value, no confidence interval and changes no PPL, seed, Holm family or
frozen verdict.  Differential retry concentration is reported descriptively:
with a handful of cells per arm it cannot support a formal test, and inventing
one would repeat the small-sample overreach the rest of the repository avoids.
"""


RUN_DISPOSITION_SCHEMA = 1

# Failure classes are deliberately coarse: the point is whether the failure is
# plausibly independent of the scientific hypothesis (infrastructure) or is a
# property of the arm itself (numerical/optimisation).  A class that is not
# declared here cannot be replaced.
RUN_DISPOSITION_POLICY = {
    'version': 'run-disposition-1',
    'retention': 'first_success',
    'replacement': 'same_cell',
    'max_attempts_per_cell': 2,
    'max_total_attempts': None,
    'allow_supersede': False,
    'require_plan_before_first_attempt': True,
    'replaceable_failure_classes': [
        'infrastructure', 'resource_exhausted', 'timeout', 'runtime_cap'],
    'non_replaceable_failure_classes': [
        'numerical', 'config', 'protocol'],
    'failure_class_patterns': {
        'infrastructure': [
            'connection', 'network', 'socket', 'ssh', 'node failure',
            'disk', 'i/o error', 'input/output error', 'broken pipe',
            'permission denied', 'no space left', 'killed', 'sigterm',
            'signal 15', 'host key', 'dns', 'http error', 'download'],
        'resource_exhausted': [
            'out of memory', 'oom', 'outofmemory', 'memoryerror',
            'cuda error: out of memory', 'resource exhausted',
            'too many open files', 'exit code 137'],
        'timeout': ['timeout', 'timed out', 'timeoutexpired', 'watchdog',
                    'deadline exceeded'],
        'runtime_cap': ['projected runtime', 'hard cap', 'run cap',
                        'cap_truncated', 'truncated by cap'],
        'numerical': ['nan', 'not finite', 'non-finite', 'inf detected',
                      'diverg', 'overflow', 'underflow', 'loss is inf'],
        'config': ['config', 'assert', 'assertionerror', 'keyerror',
                   'typeerror', 'valueerror', 'shape', 'mismatch',
                   'unknown variant', 'missing key', 'invalid argument'],
        'protocol': ['protocol', 'contract', 'integrity', 'verification',
                     'freeze lock', 'data_fp', 'checksum'],
    },
    'max_retry_rate_gap': 0.0,
    'role': ('pre-registered replacement/disposition rule for failed or '
             'truncated (arm, seed) cells; it does not change any measurement, '
             'p-value, Holm family or frozen verdict'),
}

_RETENTION_MODES = ('first_success', 'last_success', 'single_attempt')
_REPLACEMENT_MODES = ('same_cell',)
_SUCCESS_STATUSES = ('ok',)
_FAILURE_STATUSES = ('error', 'failed', 'truncated', 'skipped_cap')
_KNOWN_STATUSES = _SUCCESS_STATUSES + _FAILURE_STATUSES


def _dispo_int(name, value, *, minimum=None, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f'{name} must be an integer, got {value!r}')
    as_int = int(value)
    if minimum is not None and as_int < minimum:
        raise ValueError(f'{name} must be >= {minimum}, got {as_int}')
    if maximum is not None and as_int > maximum:
        raise ValueError(f'{name} must be <= {maximum}, got {as_int}')
    return as_int


def _dispo_bool(name, value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f'{name} must be a boolean, got {value!r}')


def _dispo_number(name, value, *, minimum=None, maximum=None):
    if isinstance(value, (bool, np.bool_)) or isinstance(value, str) or value is None:
        raise ValueError(f'{name} must be a number, got {value!r}')
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name} must be a number, got {value!r}')
    if not math.isfinite(out):
        raise ValueError(f'{name} must be finite, got {value!r}')
    if minimum is not None and out < minimum:
        raise ValueError(f'{name} must be >= {minimum}, got {out}')
    if maximum is not None and out > maximum:
        raise ValueError(f'{name} must be <= {maximum}, got {out}')
    return out


def _dispo_str_tuple(name, value, *, allow_empty=False):
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (list, tuple)):
        raise ValueError(f'{name} must be a list/tuple of strings, got {value!r}')
    out = []
    seen = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f'{name} entries must be non-empty strings, got {item!r}')
        text = item.strip()
        if text in seen:
            raise ValueError(f'{name} contains a duplicate entry {text!r}')
        seen.add(text)
        out.append(text)
    if not out and not allow_empty:
        raise ValueError(f'{name} must not be empty')
    return out


def _dispo_cells(cells, *, where='planned_cells'):
    """Validate a flat ``arm::seedN`` cell list and preserve its order."""
    if isinstance(cells, (str, bytes, bytearray)) or not isinstance(cells, (list, tuple)):
        raise ValueError(f'{where} must be a list/tuple of cell labels, got {cells!r}')
    out = []
    seen = set()
    for cell in cells:
        if not isinstance(cell, str) or not cell.strip():
            raise ValueError(f'{where} entries must be non-empty strings, got {cell!r}')
        text = cell.strip()
        arm, sep, seed_text = text.rpartition('::seed')
        if not sep or not arm:
            raise ValueError(f'{where} entry {text!r} is not of the form arm::seedN')
        try:
            seed = int(seed_text)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f'{where} entry {text!r} has a non-integer seed')
        if str(seed) != seed_text or seed < 0:
            raise ValueError(f'{where} entry {text!r} has an invalid non-negative seed')
        if text in seen:
            raise ValueError(f'{where} contains a duplicate cell {text!r}')
        seen.add(text)
        out.append(text)
    if not out:
        raise ValueError(f'{where} must not be empty')
    return out


def _dispo_policy(policy=None):
    """Normalise and validate a disposition policy.

    A partial mapping is merged over the defaults, so a caller can override one
    rule without restating the pattern table.  Unknown keys fail closed: a
    typo must not silently leave a rule at its permissive default.
    """
    base = {k: (list(v) if isinstance(v, list) else
                dict(v) if isinstance(v, dict) else v)
            for k, v in RUN_DISPOSITION_POLICY.items()}
    if policy is None:
        policy = {}
    if not isinstance(policy, dict):
        raise ValueError(f'policy must be an object, got {type(policy).__name__}')
    unknown = sorted(set(policy) - set(base))
    if unknown:
        raise ValueError(f'unknown disposition policy keys: {unknown}')
    merged = dict(base)
    merged.update(policy)

    retention = merged.get('retention')
    if retention not in _RETENTION_MODES:
        raise ValueError(f'policy.retention must be one of {list(_RETENTION_MODES)}, '
                         f'got {retention!r}')
    replacement = merged.get('replacement')
    if replacement not in _REPLACEMENT_MODES:
        raise ValueError(f'policy.replacement must be one of '
                         f'{list(_REPLACEMENT_MODES)}, got {replacement!r}')
    max_cell = _dispo_int('policy.max_attempts_per_cell',
                          merged.get('max_attempts_per_cell'), minimum=1)
    max_total = merged.get('max_total_attempts')
    if max_total is not None:
        max_total = _dispo_int('policy.max_total_attempts', max_total, minimum=1)
    merged['max_attempts_per_cell'] = max_cell
    merged['max_total_attempts'] = max_total
    merged['allow_supersede'] = _dispo_bool('policy.allow_supersede',
                                            merged.get('allow_supersede'))
    merged['require_plan_before_first_attempt'] = _dispo_bool(
        'policy.require_plan_before_first_attempt',
        merged.get('require_plan_before_first_attempt'))
    merged['max_retry_rate_gap'] = _dispo_number(
        'policy.max_retry_rate_gap', merged.get('max_retry_rate_gap'),
        minimum=0.0, maximum=1.0)

    replaceable = _dispo_str_tuple('policy.replaceable_failure_classes',
                                   merged.get('replaceable_failure_classes'),
                                   allow_empty=True)
    non_replaceable = _dispo_str_tuple(
        'policy.non_replaceable_failure_classes',
        merged.get('non_replaceable_failure_classes'), allow_empty=True)
    overlap = sorted(set(replaceable) & set(non_replaceable))
    if overlap:
        raise ValueError(f'failure classes cannot be both replaceable and '
                         f'non-replaceable: {overlap}')
    if not replaceable and not non_replaceable:
        raise ValueError('policy must declare at least one failure class')

    patterns = merged.get('failure_class_patterns')
    if not isinstance(patterns, dict):
        raise ValueError('policy.failure_class_patterns must be an object')
    known_classes = set(replaceable) | set(non_replaceable)
    clean_patterns = {}
    for cls, pats in patterns.items():
        if cls not in known_classes:
            raise ValueError(f'policy.failure_class_patterns has a class '
                             f'{cls!r} that is not declared replaceable or '
                             'non-replaceable')
        clean_patterns[cls] = _dispo_str_tuple(
            f'policy.failure_class_patterns[{cls!r}]', pats)
    # A declared class with no pattern is allowed (events may carry an explicit
    # `failure_class`); a pattern for an undeclared class is not.
    merged['replaceable_failure_classes'] = replaceable
    merged['non_replaceable_failure_classes'] = non_replaceable
    merged['failure_class_patterns'] = {k: clean_patterns[k]
                                        for k in sorted(clean_patterns)}
    return merged


def _dispo_canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)


def _dispo_timestamp(value):
    """Parse the drivers' ISO wall-clock stamp, or ``None`` when unparseable.

    The drivers write ``time.strftime('%Y-%m-%dT%H:%M:%S')``, but a hand-written
    or third-party ledger may use the equally valid ``...Z`` / fractional-second
    forms.  Treating those as "no timestamp" silently disabled the
    plan-before-first-attempt check on exactly the artifacts an auditor is most
    likely to inspect, so accept the ISO forms ``datetime`` already understands
    before falling back to the historical strptime list.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith('Z'):
        text = text[:-1] + '+00:00'
    try:
        return _dt.datetime.fromisoformat(text)
    except (TypeError, ValueError, OverflowError):
        pass
    for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M', '%Y-%m-%d'):
        try:
            return _dt.datetime.strptime(text, fmt)
        except (TypeError, ValueError, OverflowError):
            continue
    return None


def _dispo_epoch(value):
    """Comparable epoch seconds for a disposition timestamp.

    Naive stamps are interpreted as UTC: the drivers' ``strftime`` stamps carry
    no zone, and both the plan and the attempt are written by the same process,
    so only their order matters.  Using one fixed zone (instead of
    ``time.mktime``, which follows the host's local/DST rules) keeps the
    comparison stable when an artifact is audited on another machine.
    """
    stamp = _dispo_timestamp(value)
    if stamp is None:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=_dt.timezone.utc)
    try:
        return float(stamp.timestamp())
    except (OverflowError, OSError, ValueError):
        return None


def _dispo_content_sha(planned_cells, policy):
    return hashlib.sha256(_dispo_canonical({
        'planned_cells': list(planned_cells),
        'policy': policy,
    }).encode('utf-8')).hexdigest()


def validate_run_disposition_policy(policy=None):
    """Validate a declared disposition policy and return its normalized form.

    This is the design-time entry point: a manifest may declare
    ``statistics.disposition_policy`` and the design preflight can reject an
    invalid/contradictory declaration before any GPU run.
    """
    return _dispo_policy(policy)


def build_run_disposition_plan(planned_cells, *, policy=None, revision=0,
                               supersedes=None, reason=None, created_at=None,
                               run_cfg=None):
    """Build the JSON-safe precommitment for one panel.

    The plan's identity (``content_sha256``) covers only the planned cells and
    the policy.  Timestamps, the run cfg and the revision history are metadata:
    a resume may rebuild the same plan and must not change its identity, while
    a changed plan is a new revision carrying an explicit reason and the
    superseded content hash.
    """
    cells = _dispo_cells(planned_cells)
    pol = _dispo_policy(policy)
    rev = _dispo_int('revision', revision, minimum=0)
    if supersedes is None:
        supersedes = []
    sup = _dispo_str_tuple('supersedes', supersedes, allow_empty=True)
    if rev == 0 and sup:
        raise ValueError('a first plan (revision=0) cannot supersede anything')
    if rev > 0 and not sup:
        raise ValueError('a revised plan must list the superseded content hash')
    if rev > 0:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('a revised plan must carry a non-empty reason')
        reason = reason.strip()
    elif reason is not None:
        reason = str(reason).strip() or None
    if run_cfg is not None and not isinstance(run_cfg, str):
        raise ValueError('run_cfg must be a string or None')
    return {
        'kind': 'run_disposition',
        'schema': RUN_DISPOSITION_SCHEMA,
        'created_at': created_at or time.strftime('%Y-%m-%dT%H:%M:%S'),
        'revision': rev,
        'supersedes': sup,
        'reason': reason,
        'run_cfg': run_cfg,
        'planned_cells': cells,
        'policy': pol,
        'content_sha256': _dispo_content_sha(cells, pol),
    }


def _dispo_validate_plan(plan):
    if not isinstance(plan, dict):
        raise ValueError(f'disposition plan must be an object, got '
                         f'{type(plan).__name__}')
    if plan.get('kind') != 'run_disposition':
        raise ValueError(f"disposition plan kind must be 'run_disposition', "
                         f"got {plan.get('kind')!r}")
    if plan.get('schema') != RUN_DISPOSITION_SCHEMA:
        raise ValueError(f"disposition plan schema must be "
                         f"{RUN_DISPOSITION_SCHEMA}, got {plan.get('schema')!r}")
    cells = _dispo_cells(plan.get('planned_cells'))
    pol = _dispo_policy(plan.get('policy'))
    want = _dispo_content_sha(cells, pol)
    got = plan.get('content_sha256')
    if got != want:
        raise ValueError('disposition plan content_sha256 does not match its '
                         'planned_cells/policy')
    rev = _dispo_int('revision', plan.get('revision'), minimum=0)
    sup = _dispo_str_tuple('supersedes', plan.get('supersedes') or [],
                           allow_empty=True)
    if rev == 0 and sup:
        raise ValueError('a first plan cannot supersede anything')
    if rev > 0 and not sup:
        raise ValueError('a revised plan must list the superseded content hash')
    reason = plan.get('reason')
    if rev > 0 and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError('a revised plan must carry a non-empty reason')
    return {'planned_cells': cells, 'policy': pol, 'revision': rev,
            'supersedes': sup, 'content_sha256': want}


def write_run_disposition(outdir, plan, *, keep_existing=False):
    """Persist the plan before the first attempt; the first write binds.

    A later write with the same content is a no-op (a resume).  A different
    plan is only accepted when it is a declared revision that supersedes the
    existing content hash and carries a reason.  Otherwise the function either
    raises (default) or returns ``conflict=True`` with ``keep_existing=True``,
    leaving the original precommitment untouched — a changed plan must never
    silently overwrite the one that was registered first.
    """
    valid = _dispo_validate_plan(plan)
    path = os.path.join(os.fspath(outdir), 'run_disposition.json')
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        atomic_write_json(path, plan, indent=2)
        return {'path': path, 'written': True, 'conflict': False,
                'content_sha256': valid['content_sha256'], 'revision': valid['revision']}
    try:
        with open(path, encoding='utf-8') as fh:
            old = json.load(fh)
        old_valid = _dispo_validate_plan(old)
    except Exception as e:
        if keep_existing:
            return {'path': path, 'written': False, 'conflict': True,
                    'reason': f'existing plan is unreadable/invalid: '
                              f'{type(e).__name__}: {e}'}
        raise ValueError(f'existing disposition plan is unreadable/invalid: '
                         f'{type(e).__name__}: {e}') from e
    if old_valid['content_sha256'] == valid['content_sha256']:
        return {'path': path, 'written': False, 'conflict': False,
                'content_sha256': valid['content_sha256'],
                'revision': old_valid['revision']}
    is_revision = (valid['revision'] > old_valid['revision']
                   and old_valid['content_sha256'] in valid['supersedes']
                   and bool(plan.get('reason')))
    if is_revision:
        atomic_write_json(path, plan, indent=2)
        return {'path': path, 'written': True, 'conflict': False,
                'content_sha256': valid['content_sha256'],
                'revision': valid['revision'],
                'supersedes': old_valid['content_sha256']}
    msg = ('an existing run-disposition plan already covers this panel with '
           'different planned_cells/policy; a precommitment cannot be '
           'rewritten silently (build a revision with a reason and the old '
           'content_sha256 in supersedes, or keep the original)')
    if keep_existing:
        return {'path': path, 'written': False, 'conflict': True,
                'reason': msg, 'content_sha256': old_valid['content_sha256'],
                'revision': old_valid['revision']}
    raise ValueError(msg)


def _dispo_failure_class(event, policy):
    """Return ``(class, ambiguous, matched_terms)`` for a failed event."""
    explicit = event.get('failure_class')
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit.strip():
            return '__invalid__', False, []
        text = explicit.strip()
        known = set(policy['replaceable_failure_classes']) | set(
            policy['non_replaceable_failure_classes'])
        if text in known:
            return text, False, [text]
        return '__unknown__', False, [text]
    # The driver records a cap-related terminal state as its own status, so the
    # class is known even when the free-text reason does not contain one of the
    # pattern phrases.
    if event.get('status') in ('truncated', 'skipped_cap'):
        return 'runtime_cap', False, [str(event.get('status'))]
    haystack = ' '.join(str(event.get(k) or '') for k in ('error', 'reason',
                                                          'diagnosis',
                                                          'detail')).lower()
    if not haystack.strip():
        return 'unclassified', False, []
    matches = []
    for cls, needles in policy['failure_class_patterns'].items():
        for needle in needles:
            # Word-boundary matching: a bare 'oom' rule must not fire on
            # 'kaboom', and 'nan' must not fire on 'financial'.  Substring
            # containment would turn an unrelated message into a declared
            # (replaceable) failure class.
            pattern = r'(?<![a-z0-9])' + re.escape(needle.lower()) + r'(?![a-z0-9])'
            if re.search(pattern, haystack):
                matches.append(cls)
                break
    if not matches:
        return 'unclassified', False, []
    # A message that matches both a replaceable and a non-replaceable class is
    # ambiguous; fail closed by naming the first declared class and flagging it.
    unique = list(dict.fromkeys(matches))
    return unique[0], len(unique) > 1, unique


def _dispo_run_cfg(value):
    """Hashable canonical spelling of a cell's run_cfg descriptor.

    Events and retained records are untrusted JSON.  A mapping/list in the
    optional ``run_cfg`` field used to reach a dict key and raise TypeError;
    canonical JSON keeps the comparison deterministic while leaving a legacy
    string unchanged.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=True, default=str, allow_nan=False)
    except (TypeError, ValueError, OverflowError):
        return repr(value)


def _dispo_event_rows(events):
    """Parse canonical training-cell events; return ``(rows, ignored_keys)``.

    Some drivers log auxiliary ``kind='cell'`` events under their own key
    convention (``arm::s7`` for a final-test evaluation, or
    ``arm::seed3::niah_train`` for an auxiliary probe).  Those are not the
    training cells the disposition plan covers, so they are counted and
    skipped instead of being silently forced into a seed cell.  A canonical
    ``arm::seedN`` event with an inconsistent variant/seed/status still fails
    closed.
    """
    rows = []
    ignored = []
    for i, ev in enumerate(events):
        if not isinstance(ev, dict):
            raise ValueError(f'event #{i} is a {type(ev).__name__}, not an object')
        if ev.get('kind') != 'cell':
            continue
        key = ev.get('key')
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f'cell event #{i} has no non-empty key')
        key = key.strip()
        arm, sep, seed_text = key.rpartition('::seed')
        try:
            seed_from_key = int(seed_text) if sep and arm else None
        except (TypeError, ValueError, OverflowError):
            seed_from_key = None
        if seed_from_key is None or str(seed_from_key) != seed_text \
                or seed_from_key < 0:
            ignored.append(key)
            continue
        seed_int = _dispo_int(f'event #{i} seed', ev.get('seed'), minimum=0)
        if seed_int != seed_from_key:
            raise ValueError(f'cell event #{i} key {key!r} disagrees with '
                             f'seed={seed_int}')
        variant = ev.get('variant')
        if not isinstance(variant, str) or not variant:
            raise ValueError(f'cell event #{i} has no variant')
        if variant != arm:
            raise ValueError(f'cell event #{i} variant {variant!r} disagrees '
                             f'with key arm {arm!r}')
        status = ev.get('status')
        if not isinstance(status, str) or status not in _KNOWN_STATUSES:
            raise ValueError(f'cell event #{i} has unknown status {status!r}')
        run_id = ev.get('run_id')
        if not isinstance(run_id, str) or not run_id.strip():
            run_id = None
        rows.append({'index': i, 'key': key, 'variant': arm, 'seed': seed_int,
                     'status': status, 'run_id': run_id,
                     'run_cfg': _dispo_run_cfg(ev.get('run_cfg')),
                     'started_at': ev.get('started_at'),
                     'finished_at': ev.get('finished_at'),
                     'failure_class': None, 'ambiguous_class': False,
                     'matched_terms': [], 'success': status in _SUCCESS_STATUSES,
                     'failure': status in _FAILURE_STATUSES})
    return rows, ignored


def run_disposition_audit(events, *, planned_cells=None, plan=None,
                          retained=None, policy=None):
    """Audit one panel's attempt ledger against its pre-registered disposition.

    ``events`` are the parsed ``experiments.jsonl`` cell events (other kinds are
    ignored).  ``plan`` is a plan from :func:`build_run_disposition_plan`;
    ``planned_cells`` may be supplied instead for a panel whose plan file is
    unavailable (that frame is explicitly *not* a precommitment).  ``retained``
    maps a cell key to the measurement record that entered the analysis (only
    records with a usable PPL); its ``run_id`` is compared with the first
    successful attempt.

    Returns a JSON-safe report.  ``status`` is ``fail`` for a protocol
    violation, ``warn`` for a consistent but noteworthy panel, ``ok`` for a
    clean one, and ``not_verifiable_locally`` when no plan or no cell event
    exists.  No p-value, PPL, Holm family or frozen verdict is computed or
    changed here.
    """
    out = {
        'kind': 'run_disposition_audit',
        'status': 'ok',
        'frame': None,
        'n_planned': 0,
        'n_attempted': 0,
        'n_measured': 0,
        'n_attempts': 0,
        'n_cells_retried': 0,
        'n_multiple_success_cells': 0,
        'n_after_success_attempts': 0,
        'n_unplanned_measurements': 0,
        'n_unplanned_attempts': 0,
        'n_missing_planned': 0,
        'n_planned_without_success': 0,
        'n_retained_without_success_event': 0,
        'n_retained_second_success': 0,
        'n_retained_not_declared_success': 0,
        'n_failures': 0,
        'n_unclassified_failures': 0,
        'n_ambiguous_failures': 0,
        'n_ambiguous_retried': 0,
        'n_non_replaceable_retried': 0,
        'n_single_attempt_violations': 0,
        'n_attempt_budget_exceeded': 0,
        'n_superseded_cells': 0,
        'n_legacy_events': 0,
        'n_ignored_cell_events': 0,
        'plan_before_first_attempt': None,
        'first_attempt_at': None,
        'cells': {},
        'by_arm': {},
        'failure_classes': {},
        'differential_attempts': {},
        'problems': [],
        'warnings': [],
        'note': ('pre-registered disposition audit: it checks that the retained '
                 'measurement is the declared attempt and that retries are not '
                 'arm-selective; it does not recompute any statistic or change '
                 'a frozen verdict'),
    }

    plan_valid = None
    if plan is not None:
        plan_valid = _dispo_validate_plan(plan)
        if planned_cells is not None:
            supplied = _dispo_cells(planned_cells, where='planned_cells')
            if supplied != plan_valid['planned_cells']:
                raise ValueError('planned_cells disagrees with the plan file')
        planned = list(plan_valid['planned_cells'])
        pol = _dispo_policy(plan_valid['policy'])
        out['frame'] = 'precommitted'
        out['plan_revision'] = plan_valid['revision']
        out['plan_content_sha256'] = plan_valid['content_sha256']
        out['plan_created_at'] = plan.get('created_at')
    elif planned_cells is not None:
        planned = _dispo_cells(planned_cells, where='planned_cells')
        pol = _dispo_policy(policy)
        out['frame'] = 'caller_supplied'
        out['warnings'].append(
            'no run_disposition.json plan was supplied; the cell list is a '
            'caller-provided frame, not a precommitment, so a later cell '
            'addition cannot be ruled out')
    else:
        planned = []
        pol = _dispo_policy(policy)
        out['frame'] = 'observed_only'
    out['policy'] = pol
    out['retention'] = pol['retention']

    rows, ignored_keys = _dispo_event_rows(events)
    if ignored_keys:
        out['n_ignored_cell_events'] = len(ignored_keys)
        out['ignored_cell_key_examples'] = list(dict.fromkeys(ignored_keys))[:20]
        out['warnings'].append(
            f'{len(ignored_keys)} cell event(s) use a non-training key '
            '(e.g. an evaluation/probe cell) and are outside this plan; they '
            'are counted, not silently treated as training seeds')
    if not rows:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = ('no cell events in experiments.jsonl; run a panel '
                         'with the current code to produce an attempt ledger')
        return out
    out['n_attempts'] = len(rows)
    # A precommitment written after the first attempt is a reconstruction.  The
    # same rule the run-order audit applies to its plan applies here.
    if out['frame'] == 'precommitted':
        _plan_epoch = _dispo_epoch(out.get('plan_created_at'))
        # JSONL rows are appended on completion, so "the first row with a
        # timestamp" is the first FINISHED attempt, not the first STARTED one.
        # Take the minimum start time over every row.
        _timed_rows = [(_dispo_epoch(r['started_at']), r['started_at'])
                       for r in rows
                       if _dispo_epoch(r['started_at']) is not None]
        if _timed_rows:
            _first_epoch, first_started = min(_timed_rows,
                                              key=lambda pair: pair[0])
        else:
            _first_epoch = first_started = None
        out['first_attempt_at'] = first_started
        if _first_epoch is None:
            # No attempt timestamp (synthetic/legacy ledger): the plan-before-
            # attempt relation is simply not testable here.  The run-order
            # timing gate owns the general plan-timing check; do not invent a
            # warning from a ledger that carries no wall-clock stamps.
            out['plan_before_first_attempt'] = None
        elif _plan_epoch is None:
            out['plan_before_first_attempt'] = None
            out['warnings'].append(
                'the first attempt has a timestamp but the plan does not; '
                'whether the disposition was registered before execution '
                'cannot be established')
        else:
            out['plan_before_first_attempt'] = bool(_plan_epoch <= _first_epoch)
            if not out['plan_before_first_attempt']:
                _msg = ('run_disposition.json was written after the first cell '
                        f"started ({out.get('plan_created_at')!r} > "
                        f'{first_started!r}); it is a reconstruction, not a '
                        'precommitment')
                if pol['require_plan_before_first_attempt']:
                    out['problems'].append(_msg)
                else:
                    out['warnings'].append(
                        _msg + ' (the policy explicitly accepts a '
                        'reconstruction, so this is descriptive only)')
    out['n_legacy_events'] = sum(1 for r in rows if r['run_id'] is None)
    if out['n_legacy_events']:
        out['warnings'].append(
            f"{out['n_legacy_events']} cell event(s) carry no run_id; the "
            'retained attempt cannot be tied to a specific attempt for those '
            'cells (legacy ledger)')
    if out['frame'] == 'observed_only':
        planned = list(dict.fromkeys(r['key'] for r in rows))
        out['warnings'].append(
            'no run_disposition.json plan exists; the audit falls back to the '
            'observed cell set, which cannot detect a silently added cell')

    planned_set = set(planned)
    out['n_planned'] = len(planned)

    # run_id must identify an attempt, not repeat across attempts.
    seen_run_ids = {}
    for r in rows:
        rid = r['run_id']
        if rid is None:
            continue
        if rid in seen_run_ids:
            out['problems'].append(
                f"duplicate run_id {rid!r} appears in cell {seen_run_ids[rid]!r} "
                f"and {r['key']!r}; the ledger does not identify two distinct "
                'attempts')
        else:
            seen_run_ids[rid] = r['key']

    grouped = {}
    for r in rows:
        grouped.setdefault(r['key'], []).append(r)
        if r['key'] not in planned_set:
            out['n_unplanned_attempts'] += 1
            if r['success']:
                out['problems'].append(
                    f"cell {r['key']!r} has a successful attempt but is not in "
                    'the pre-registered plan (an unplanned seed/arm cannot be '
                    'added after the fact)')
        if r['failure']:
            out['n_failures'] += 1
            # Rows are built in order, so the source event is the row's index.
            cls, ambiguous, terms = _dispo_failure_class(events[r['index']], pol)
            r['failure_class'] = cls
            r['ambiguous_class'] = bool(ambiguous)
            r['matched_terms'] = list(terms)
            if cls in ('unclassified', '__unknown__', '__invalid__'):
                out['n_unclassified_failures'] += 1
            if r['ambiguous_class']:
                # A free-text failure that matches both a replaceable and a
                # non-replaceable class cannot be classified as infrastructure
                # noise.  The first class is kept for display, but the retry
                # check below treats the ambiguity as fail-closed.
                out['n_ambiguous_failures'] += 1
            key = f"{r['key']}::{r['status']}"
            rec = out['failure_classes'].setdefault(
                cls, {'class': cls, 'n_attempts': 0, 'n_cells': 0,
                      'n_retried': 0, 'n_ambiguous': 0,
                      'replaceable': cls in
                      set(pol['replaceable_failure_classes'])})
            rec['n_attempts'] += 1
            if r['ambiguous_class']:
                rec['n_ambiguous'] += 1
    # Count failed cells once per class (a cell can have two identical errors).
    _seen_class_cells = {}
    for key, cell_rows in grouped.items():
        for r in cell_rows:
            if not r['failure']:
                continue
            c = r['failure_class']
            _seen_class_cells.setdefault(c, set())
            if key not in _seen_class_cells[c]:
                out['failure_classes'][c]['n_cells'] += 1
                _seen_class_cells[c].add(key)

    out['n_attempted'] = len([k for k in planned if k in grouped])
    if out['n_unclassified_failures']:
        out['warnings'].append(
            f"{out['n_unclassified_failures']} failed attempt(s) have no "
            'declared failure class; the policy cannot say whether replacing '
            'them is infrastructure noise or an arm property')
    if out['n_ambiguous_failures']:
        out['warnings'].append(
            f"{out['n_ambiguous_failures']} failed attempt(s) match more than "
            'one declared failure class; an ambiguous failure is never treated '
            'as replaceable infrastructure noise')
    if out['n_unplanned_attempts'] and not out['n_unplanned_measurements']:
        out['warnings'].append(
            f"{out['n_unplanned_attempts']} attempt(s) belong to a cell outside "
            'the plan and produced no retained measurement; they are counted '
            'as a protocol deviation even though they did not enter a table')
    by_arm = {}
    for key in planned:
        arm = key.rpartition('::seed')[0]
        a = by_arm.setdefault(arm, {
            'n_cells': 0, 'n_attempted': 0, 'n_measured': 0,
            'n_attempts': 0, 'n_retried_cells': 0,
            'n_failed_attempts': 0, 'n_multiple_success_cells': 0,
            'retry_rate': 0.0})
        a['n_cells'] += 1
        cell_rows = grouped.get(key, [])
        if cell_rows:
            a['n_attempted'] += 1
        a['n_attempts'] += len(cell_rows)
        a['n_failed_attempts'] += sum(1 for r in cell_rows if r['failure'])
        if len(cell_rows) > 1:
            a['n_retried_cells'] += 1
        if sum(1 for r in cell_rows if r['success']) > 1:
            a['n_multiple_success_cells'] += 1
    out['by_arm'] = {k: by_arm[k] for k in sorted(by_arm)}

    # ---------------------------------------------------------------- per cell
    retained_valid = {}
    if retained is not None:
        if not isinstance(retained, dict):
            raise ValueError(f'retained must be an object, got '
                             f'{type(retained).__name__}')
        for key, rec in retained.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f'retained keys must be non-empty strings, '
                                 f'got {key!r}')
            key = key.strip()
            if not isinstance(rec, dict):
                raise ValueError(f'retained[{key!r}] must be an object')
            rid = rec.get('run_id')
            retained_valid[key] = {'run_id': (rid if isinstance(rid, str) and
                                              rid.strip() else None),
                                   'run_cfg': _dispo_run_cfg(rec.get('run_cfg')),
                                   'record': rec}
            if key not in planned_set:
                out['n_unplanned_measurements'] += 1
                out['problems'].append(
                    f'retained measurement {key!r} is not in the pre-registered '
                    'planned_cells (seed/arm shopping cannot be accepted as a '
                    'confirmation)')

    for key in planned:
        cell_rows = grouped.get(key, [])
        ok_rows = [r for r in cell_rows if r['success']]
        fail_rows = [r for r in cell_rows if r['failure']]
        first_ok = ok_rows[0] if ok_rows else None
        first_ok_idx = first_ok['index'] if first_ok else None
        last_ok = ok_rows[-1] if ok_rows else None
        rec = retained_valid.get(key)
        rec_rid = (rec or {}).get('run_id')
        first_ok_rid = (first_ok or {}).get('run_id')
        last_ok_rid = (last_ok or {}).get('run_id')
        later_attempts = ([r for r in cell_rows
                           if first_ok_idx is not None and r['index'] > first_ok_idx]
                          if first_ok_idx is not None else [])
        ok_cfgs = list(dict.fromkeys(
            [r['run_cfg'] for r in ok_rows if r['run_cfg'] is not None]))
        superseded = len(ok_cfgs) > 1
        cell = {
            'planned': True,
            'attempts': [],
            'n_attempts': len(cell_rows),
            'n_success': len(ok_rows),
            'n_failure': len(fail_rows),
            'first_success_run_id': first_ok_rid,
            'retained_run_id': rec_rid,
            'retained_is_first_success': (None if rec_rid is None or
                                          first_ok_rid is None
                                          else rec_rid == first_ok_rid),
            'retried_after_failure': False,
            'attempt_after_success': bool(later_attempts),
            'superseded_run_cfgs': ok_cfgs if superseded else [],
        }
        for r in cell_rows:
            cell['attempts'].append({
                'run_id': r['run_id'],
                'status': r['status'],
                'run_cfg': r['run_cfg'],
                'failure_class': r['failure_class'],
                'started_at': r['started_at'],
                'finished_at': r['finished_at'],
            })
        # A failure is "retried" when any later attempt exists for the cell.
        for r in fail_rows:
            if any(x['index'] > r['index'] for x in cell_rows):
                cell['retried_after_failure'] = True
                cls = r['failure_class']
                if cls in out['failure_classes']:
                    out['failure_classes'][cls]['n_retried'] += 1
                if r['ambiguous_class']:
                    out['n_ambiguous_retried'] += 1
                    out['problems'].append(
                        f'cell {key!r}: the failure text matches more than one '
                        f'declared class ({", ".join(r["matched_terms"])}); an '
                        'ambiguous failure cannot be treated as replaceable '
                        'infrastructure noise, so the retry is not permitted')
                elif cls in set(pol['non_replaceable_failure_classes']):
                    out['n_non_replaceable_retried'] += 1
                    out['problems'].append(
                        f'cell {key!r}: failure class {cls!r} is declared '
                        'non-replaceable but the cell was attempted again; the '
                        'replacement is not permitted by the policy')
                elif cls in ('unclassified', '__unknown__', '__invalid__'):
                    out['problems'].append(
                        f'cell {key!r}: a failure of unknown class ({cls!r}) was '
                        'retried; a replacement that the policy cannot classify '
                        'cannot be treated as infrastructure noise')
        if len(cell_rows) > 1:
            out['n_cells_retried'] += 1
        if len(ok_rows) > 1:
            out['n_multiple_success_cells'] += 1
        if later_attempts:
            out['n_after_success_attempts'] += len(later_attempts)
        if len(cell_rows) > pol['max_attempts_per_cell']:
            out['n_attempt_budget_exceeded'] += 1
            out['problems'].append(
                f'cell {key!r} has {len(cell_rows)} attempts but the declared '
                f'maximum is {pol["max_attempts_per_cell"]}')
        if superseded:
            out['n_superseded_cells'] += 1
            if pol['allow_supersede']:
                out['warnings'].append(
                    f'cell {key!r} was measured under {len(ok_cfgs)} different '
                    'run_cfg values and the policy allows supersession; the '
                    'retained measurement is the last declared one, not an '
                    'independent replicate')
            else:
                out['problems'].append(
                    f'cell {key!r} has successful attempts under '
                    f'{len(ok_cfgs)} different run_cfg values; the policy does '
                    'not allow superseding a measured cell after a code/config '
                    'change')
        if not cell_rows:
            out['n_missing_planned'] += 1
            cell['status'] = 'not_attempted'
        elif not ok_rows:
            out['n_planned_without_success'] += 1
            cell['status'] = 'no_successful_attempt'
            out['warnings'].append(
                f'cell {key!r} was attempted but produced no successful '
                'measurement; report it as missing, not as a replaced cell')
        else:
            cell['status'] = 'measured'
        # Selection checks (only meaningful when there is a retained record).
        if rec is None and ok_rows:
            out['n_retained_without_success_event'] += 1
            out['problems'].append(
                f'cell {key!r} has a successful attempt but no retained '
                'measurement record; the measured value was dropped from the '
                'summary')
        if rec is not None and not ok_rows:
            out['n_retained_without_success_event'] += 1
            out['problems'].append(
                f'cell {key!r} has a retained measurement but no successful '
                'attempt event (the measurement cannot be traced to an attempt)')
        if rec is not None and ok_rows:
            ok_rids = [r['run_id'] for r in ok_rows if r['run_id']]
            if rec_rid is None or first_ok_rid is None:
                out['warnings'].append(
                    f'cell {key!r}: the retained record or the first successful '
                    'event has no run_id, so the first-success rule is not '
                    'verifiable for this cell (legacy ledger)')
            elif rec_rid not in ok_rids:
                out['n_retained_without_success_event'] += 1
                out['problems'].append(
                    f'cell {key!r}: retained run_id {rec_rid!r} is not one of '
                    'the successful attempts in the ledger')
            elif rec_rid != first_ok_rid:
                out['n_retained_second_success'] += 1
                if pol['retention'] == 'first_success':
                    out['problems'].append(
                        f'cell {key!r}: the retained measurement is a later '
                        f'attempt ({rec_rid!r}) than the first success '
                        f'({first_ok_rid!r}); retaining a later run of the same '
                        'cell is post-hoc selection, not a pre-registered '
                        'measurement')
                elif pol['retention'] == 'last_success' \
                        and rec_rid == last_ok_rid:
                    out['warnings'].append(
                        f'cell {key!r}: the policy explicitly retains the last '
                        'success; the result is descriptive/exploratory, not a '
                        'confirmatory measurement')
            if (pol['retention'] == 'last_success'
                    and rec_rid is not None and rec_rid in ok_rids
                    and last_ok_rid is not None and rec_rid != last_ok_rid):
                # The declared rule is "last success"; keeping an earlier
                # success under that rule is selection in the opposite
                # direction, and it would otherwise pass silently because the
                # generic later-than-first check only covers first_success.
                out['n_retained_not_declared_success'] += 1
                out['problems'].append(
                    f'cell {key!r}: the policy declares retention=last_success '
                    f'but the retained attempt is {rec_rid!r}, not the last '
                    f'success ({last_ok_rid!r}); the declared rule was not '
                    'applied')
        if len(cell_rows) > 1 and pol['retention'] == 'single_attempt' \
                and len(ok_rows) <= 1:
            # A failure followed by a success is still two attempts, not the
            # single attempt this policy declares.
            out['n_single_attempt_violations'] += 1
            out['problems'].append(
                f'cell {key!r} has {len(cell_rows)} attempts but the policy '
                'requires a single attempt')
        if len(ok_rows) > 1 and pol['retention'] == 'first_success' \
                and rec_rid is not None and rec_rid == first_ok_rid:
            out['warnings'].append(
                f'cell {key!r} succeeded {len(ok_rows)} times; the first '
                'success was retained, so this is not a selection, but the '
                'repeated attempt should be explained in the decision log')
        if len(ok_rows) > 1 and pol['retention'] == 'single_attempt':
            # Count every single_attempt violation once.  The retained-record
            # branch above intentionally does not append its own duplicate:
            # two successes are a violation whether the retained attempt is
            # the first or the last one.
            out['n_single_attempt_violations'] += 1
            out['problems'].append(
                f'cell {key!r} succeeded {len(ok_rows)} times but the policy '
                'requires a single attempt')
        # A retained record under a run_cfg that never succeeded is untraceable.
        if rec is not None and rec.get('run_cfg') is not None and ok_rows \
                and rec['run_cfg'] not in ok_cfgs:
            out['problems'].append(
                f'cell {key!r}: the retained record carries run_cfg '
                f'{rec["run_cfg"]!r}, which no successful ledger attempt used')
        out['cells'][key] = cell

    # Unplanned retained keys are already a problem; record them for reading.
    for key, rec in retained_valid.items():
        if key not in planned_set:
            out['cells'].setdefault(key, {
                'planned': False, 'attempts': [], 'n_attempts': 0,
                'n_success': 0, 'n_failure': 0,
                'first_success_run_id': None,
                'retained_run_id': rec.get('run_id'),
                'retained_is_first_success': None,
                'retried_after_failure': False,
                'attempt_after_success': False,
                'superseded_run_cfgs': [],
                'status': 'unplanned_measurement',
            })
    out['n_measured'] = sum(1 for c in out['cells'].values()
                            if c.get('status') == 'measured')
    for key, cell in out['cells'].items():
        arm = key.rpartition('::seed')[0]
        if arm in out['by_arm'] and cell.get('status') == 'measured':
            out['by_arm'][arm]['n_measured'] += 1
    if out['n_missing_planned']:
        out['warnings'].append(
            f"{out['n_missing_planned']} planned cell(s) have no attempt at "
            'all; they are missing, not silently replaceable, and any '
            'complete-case table must say so')
    if out['n_cells_retried']:
        out['warnings'].append(
            f"{out['n_cells_retried']} cell(s) were attempted more than once; "
            'the attempt table and the retained run_id are part of the result, '
            'not an infrastructure footnote')

    # ------------------------------------------------------- arm level retries
    arms_with_cells = [a for a, rec in out['by_arm'].items() if rec['n_cells']]
    for a in arms_with_cells:
        rec = out['by_arm'][a]
        rec['retry_rate'] = (rec['n_retried_cells'] / rec['n_cells']
                             if rec['n_cells'] else 0.0)
    retry_arms = [a for a in arms_with_cells
                  if out['by_arm'][a]['n_retried_cells'] > 0]
    rates = [out['by_arm'][a]['retry_rate'] for a in arms_with_cells]
    spread = (max(rates) - min(rates)) if rates else 0.0
    # One pass over the cell table.  Re-scanning every cell for every arm
    # (``for a in arms: max(... for k, c in out['cells'].items() if ...)``)
    # is O(n_arms * n_cells): a sweep/ablation grid with thousands of arms
    # would spend minutes deriving a single descriptive map.  The per-arm
    # maximum is accumulated once and the declared arm order is preserved.
    _cell_max_by_arm = {}
    for _key, _cell in out['cells'].items():
        _arm = _key.rpartition('::seed')[0]
        _n = _cell.get('n_attempts') or 0
        if _n > _cell_max_by_arm.get(_arm, 0):
            _cell_max_by_arm[_arm] = _n
    diff = {
        'status': ('no_retries' if not retry_arms else
                   'concentrated' if len(retry_arms) < len(arms_with_cells)
                   else 'balanced'),
        'n_arms_with_cells': len(arms_with_cells),
        'n_arms_with_retries': len(retry_arms),
        'arms_with_retries': retry_arms,
        'retry_rate_spread': spread,
        'max_attempts_per_cell_by_arm': {
            a: _cell_max_by_arm.get(a, 0) for a in arms_with_cells},
        'role': ('descriptive concentration readout; with a few cells per arm '
                 'it is not a test of differential attrition and carries no '
                 'p-value'),
    }
    out['differential_attempts'] = diff
    if diff['status'] == 'concentrated':
        out['warnings'].append(
            'retries are concentrated in a strict subset of arms '
            f'({", ".join(retry_arms)}); the retained measurements are selected '
            'by a different number of attempts across arms, so a complete-case '
            'comparison is not exchangeable without accounting for it')
    if spread > pol['max_retry_rate_gap'] and diff['status'] != 'concentrated':
        out['warnings'].append(
            f'per-arm retry rates differ by {spread:.3f} (declared tolerance '
            f'{pol["max_retry_rate_gap"]:.3f}); inspect the attempt table before '
            'treating the panel as balanced')
    if any(a['n_measured'] == 0 for a in out['by_arm'].values() if a['n_cells']):
        out['warnings'].append(
            'at least one arm has no measured cell; report the panel as '
            'incomplete rather than comparing the surviving arms')

    if pol['max_total_attempts'] is not None and out['n_attempts'] > pol['max_total_attempts']:
        out['problems'].append(
            f'the panel made {out["n_attempts"]} attempts but the plan declared '
            f'at most {pol["max_total_attempts"]}')

    if out['n_unplanned_measurements']:
        out['status'] = 'fail'
    if out['problems']:
        out['status'] = 'fail'
    elif out['frame'] != 'precommitted':
        out['status'] = 'not_verifiable_locally' if out['frame'] == 'observed_only' \
            else 'warn'
    elif out['warnings']:
        out['status'] = 'warn'
    else:
        out['status'] = 'ok'
    return out
