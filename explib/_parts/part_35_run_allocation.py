"""Experiment-preparation audits: prospective plans and arm x environment balance.

The run-order design (``part_33_plot.py``) randomizes *when* a cell runs.  This
module covers the two allocation variables that are still confounded in a
multi-GPU campaign:

* *when the plan was written* — a ``run_order.json`` produced after the first
  cell is a reconstruction, not a precommitment;
* *where the cell ran* — a dynamic dispatcher can put one arm on GPU 0 and its
  paired arm on GPU 1 for every seed, so device/driver/thermal differences
  become part of the arm contrast even though ``exec_fp`` (which intentionally
  excludes GPU UUID and hostname) agrees.

Nothing here changes a measurement, seed, endpoint, p-value or the frozen
decision rule.  Missing environment metadata is reported as
``not_verifiable_locally`` rather than as a green pass.
"""

import collections
import datetime as _dt
import json


ENVIRONMENT_BALANCE_POLICY = {
    'version': 1,
    'min_records_per_arm': 2,
    'dominant_share_warn': 0.90,
    'identity_fields': (
        'hostname', 'cuda_available', 'cuda_visible_devices',
        'gpus', 'torch', 'cuda_version', 'cudnn_version', 'determinism',
        'tf32_matmul', 'float32_matmul_precision',
    ),
    'note': ('environment identity is descriptive; a balanced design cannot '
             'repair a confound that already occurred, and a missing identity '
             'is not evidence of balance'),
}


def _allocation_timestamp(value):
    """Parse the ISO-like timestamps used by the drivers, or return None."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith('Z'):
        text = text[:-1] + '+00:00'
    try:
        return _dt.datetime.fromisoformat(text)
    except (TypeError, ValueError):
        for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%dT%H:%M', '%Y-%m-%d'):
            try:
                return _dt.datetime.strptime(text, fmt)
            except (TypeError, ValueError):
                continue
    return None


def _comparable_timestamps(a, b):
    """True when two parsed timestamps can be ordered on one clock."""
    if a is None or b is None:
        return False
    return (a.tzinfo is None) == (b.tzinfo is None)


def environment_identity(environment):
    """Canonical hardware/software identity of one recorded environment.

    Returns ``None`` when the record carries no environment object at all.
    ``CUDA_VISIBLE_DEVICES`` is included verbatim: in a parallel child it is
    the single physical token the cell actually saw, which is exactly the
    variable that can become collinear with the arm.  Hostname and the CUDA
    stack are included because two GPUs of the same model on different hosts
    or driver builds are not interchangeable blockers.
    """
    if not isinstance(environment, dict):
        return None
    gpus = []
    raw_gpus = environment.get('gpu_identity')
    if isinstance(raw_gpus, list):
        for g in raw_gpus:
            if not isinstance(g, dict):
                continue
            gpus.append([
                g.get('name'), g.get('major'), g.get('minor'),
                g.get('total_memory_bytes'),
            ])
    elif isinstance(environment.get('gpu_names'), list):
        for name in environment.get('gpu_names'):
            gpus.append([name, None, None, None])
    ident = {
        'hostname': environment.get('hostname'),
        'cuda_available': environment.get('cuda_available'),
        'cuda_visible_devices': environment.get('cuda_visible_devices'),
        'gpus': gpus,
        'torch': environment.get('torch'),
        'cuda_version': environment.get('cuda_version'),
        'cudnn_version': environment.get('cudnn_version'),
        'determinism': environment.get('determinism'),
        'tf32_matmul': environment.get('tf32_matmul'),
        'float32_matmul_precision': environment.get('float32_matmul_precision'),
    }
    if (ident['hostname'] is None and ident['cuda_available'] is None
            and ident['cuda_visible_devices'] is None and not gpus
            and all(ident[k] is None for k in (
                'torch', 'cuda_version', 'cudnn_version', 'determinism',
                'tf32_matmul', 'float32_matmul_precision'))):
        return None
    return ident


def _environment_key(ident):
    if ident is None:
        return None
    return json.dumps(ident, sort_keys=True, separators=(',', ':'),
                      default=str, ensure_ascii=True)


def _environment_label(ident):
    if not isinstance(ident, dict):
        return None
    host = ident.get('hostname') or '?'
    device = ident.get('cuda_visible_devices')
    if device in (None, ''):
        device = 'cuda-all' if ident.get('cuda_available') else 'cpu'
    names = []
    for g in ident.get('gpus') or []:
        if isinstance(g, (list, tuple)) and g and g[0]:
            names.append(str(g[0]))
    gpu = ','.join(names) if names else '?'
    stack = '|'.join(str(ident.get(k)) for k in (
        'torch', 'cuda_version', 'cudnn_version', 'determinism'))
    return f'{host}|{device}|{gpu}|{stack}'


def _record_environment(record):
    if not isinstance(record, dict):
        return None
    env = record.get('environment')
    if isinstance(env, dict):
        return env
    prov = record.get('provenance')
    if isinstance(prov, dict) and isinstance(prov.get('environment'), dict):
        return prov.get('environment')
    return None


def _iter_allocation_records(records):
    """Yield ``(label, record)`` from a summary dict or a record list."""
    if isinstance(records, dict):
        items = list(records.items())
    elif isinstance(records, (list, tuple)):
        items = [(str(i), r) for i, r in enumerate(records)]
    else:
        return
    for label, rec in items:
        if not isinstance(rec, dict):
            continue
        if rec.get('variant') is None:
            continue
        yield str(label), rec


def environment_balance_audit(records, *, min_records_per_arm=None):
    """Audit whether measured arms are collinear with the execution environment.

    A *complete* confound is a pair of arms whose environment sets are
    disjoint: every observation of arm A came from one environment and every
    observation of arm B from another, so the arm effect and the environment
    effect cannot be separated at all.  That is a design failure (``fail``),
    not a null result.

    Partial imbalance (one arm mostly on one device) is a warning.  A single
    observation per arm is not enough to call a confound; that case is
    reported as ``not_verifiable_locally`` with the insufficient arms named.
    Missing environment identity is also ``not_verifiable_locally``, never a
    pass.
    """
    policy = ENVIRONMENT_BALANCE_POLICY
    try:
        min_per_arm = int(policy['min_records_per_arm'] if min_records_per_arm is None
                          else min_records_per_arm)
    except (TypeError, ValueError, OverflowError):
        min_per_arm = int(policy['min_records_per_arm'])
    if min_per_arm < 1:
        min_per_arm = 1
    out = {
        'kind': 'environment_balance_audit',
        'status': 'ok',
        'policy': dict(policy),
        'min_records_per_arm': min_per_arm,
        'n_records': 0,
        'n_arms': 0,
        'n_environments': 0,
        'n_missing_environment': 0,
        'environment_labels': {},
        'arm_environment_counts': {},
        'arm_dominant_share': {},
        'confounded_pairs': [],
        'insufficient_arms': [],
        'problems': [],
        'warnings': [],
        'note': ('a disjoint arm x environment pattern is complete confounding; '
                 'missing environment identity is not a green pass'),
    }
    counts = collections.OrderedDict()
    envs = {}
    for _label, rec in _iter_allocation_records(records):
        out['n_records'] += 1
        variant = str(rec.get('variant'))
        if variant not in counts:
            counts[variant] = collections.Counter()
        ident = environment_identity(_record_environment(rec))
        if ident is None:
            out['n_missing_environment'] += 1
            continue
        key = _environment_key(ident)
        envs[key] = ident
        counts[variant][key] += 1
    out['n_arms'] = len(counts)
    out['n_environments'] = len(envs)
    for key, ident in envs.items():
        out['environment_labels'][key] = _environment_label(ident)
    for variant, counter in counts.items():
        out['arm_environment_counts'][variant] = {
            out['environment_labels'].get(k, k): int(v) for k, v in counter.items()
        }
        total = sum(counter.values())
        out['arm_dominant_share'][variant] = (
            max(counter.values()) / total if total else None)
    if out['n_missing_environment']:
        out['warnings'].append(
            f"{out['n_missing_environment']} of {out['n_records']} record(s) "
            'carry no environment object; the environment balance is only '
            'partially observable')
    if not envs:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = ('no record carries a usable environment identity; '
                         'arm x device confounding cannot be checked')
        return out
    measured = [v for v, c in counts.items() if sum(c.values()) >= min_per_arm]
    out['insufficient_arms'] = [v for v, c in counts.items()
                                if sum(c.values()) < min_per_arm]
    if len(measured) < 2:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = (
            f'fewer than two arms have >= {min_per_arm} environment-bearing '
            'record(s); a per-arm confound cannot be distinguished from a '
            'single observation')
        return out
    # A complete confound is a pair of arms whose environment sets are
    # disjoint.  The naive form constructs/sets both sets for every arm pair,
    # which is O(n_arms**2) work and allocations; a large arm grid spends all
    # its time proving that the common case (every arm shares an environment)
    # has no pair.  Group arms with the same environment set: arms inside one
    # group always intersect, and only whole groups can be disjoint.  The
    # number of distinct environment sets is normally tiny (one or two
    # devices), so this is linear in the panel, and the emitted pair order is
    # still the historical (first-arm, second-arm) order.
    env_sets = {a: frozenset(counts[a]) for a in measured}
    groups = collections.OrderedDict()
    for a in measured:
        groups.setdefault(env_sets[a], []).append(a)
    _group_items = list(groups.items())
    _measured_index = {a: i for i, a in enumerate(measured)}
    _confounded = []
    for gi, (sa, arms_a) in enumerate(_group_items):
        if not sa:
            continue
        for sb, arms_b in _group_items[gi + 1:]:
            if not sb or not sa.isdisjoint(sb):
                continue
            for a in arms_a:
                ia = _measured_index[a]
                for b in arms_b:
                    ib = _measured_index[b]
                    _confounded.append((min(ia, ib), max(ia, ib), a, b)
                                       if ia < ib else
                                       (ib, ia, b, a))
    _confounded.sort(key=lambda row: (row[0], row[1]))
    out['confounded_pairs'] = [[a, b] for _i, _j, a, b in _confounded]
    threshold = float(policy['dominant_share_warn'])
    for variant, share in out['arm_dominant_share'].items():
        if share is not None and share > threshold and len(counts[variant]) > 1:
            out['warnings'].append(
                f'{variant}: {share:.0%} of its environment-bearing records '
                f'came from one environment (>{threshold:.0%}); the arm is '
                'nearly collinear with the device')
    if out['confounded_pairs']:
        out['status'] = 'fail'
        for a, b in out['confounded_pairs']:
            out['problems'].append(
                f'arms {a!r} and {b!r} have disjoint environment sets; arm and '
                'device/host are perfectly confounded, so no paired contrast '
                'can separate them')
    elif out['insufficient_arms']:
        # A comparison among the sufficiently sampled arms can still look
        # balanced, but the under-sampled arm(s) cannot be checked at all.
        # That is an explicit gap in the allocation audit, not a green pass
        # for the panel: every arm named in the design must be assessable.
        out['status'] = 'not_verifiable_locally'
        out['reason'] = (
            f'arm(s) {out["insufficient_arms"]} have fewer than {min_per_arm} '
            'environment-bearing record(s); their arm x environment balance '
            'cannot be distinguished from a single observation')
        out['warnings'].append(
            f"{len(out['insufficient_arms'])} arm(s) have fewer than "
            f'{min_per_arm} environment-bearing record(s) and cannot be '
            f'assessed: {out["insufficient_arms"]}')
    elif out['warnings']:
        out['status'] = 'warnings'
    else:
        out['status'] = 'ok'
    return out


def environment_plan_audit(plan):
    """Check that a run-order plan's optional device assignment is balanced.

    A plan without ``environment_slots`` is ``not_declared`` (a warning, not a
    failure): the historical drivers did not record a device design.  A plan
    that declares slots must cover every planned cell, use a declared slot, and
    give each arm every slot within one occurrence across complete blocks.
    """
    out = {
        'kind': 'environment_plan_audit',
        'status': 'not_declared',
        'n_slots': 0,
        'n_cells': 0,
        'slots': [],
        'arm_slot_counts': {},
        'slot_balanced': None,
        'problems': [],
        'warnings': [],
        'note': ('a device assignment is optional metadata; when declared it '
                 'must cover the plan and be position-balanced'),
    }
    if not isinstance(plan, dict):
        out['status'] = 'not_declared'
        out['warnings'].append('no run-order plan object; no device assignment to audit')
        return out
    if 'environment_slots' not in plan:
        if 'cell_environments' in plan:
            out['status'] = 'fail'
            out['problems'].append(
                'cell_environments is present but environment_slots is not '
                'declared; the assignment has no slot vocabulary to check')
            return out
        out['warnings'].append(
            'the run-order plan declares no environment slots; arm x device '
            'balance is not part of the recorded design')
        return out
    raw_slots = plan.get('environment_slots')
    try:
        slots = _normalise_environment_slots(raw_slots)
    except (TypeError, ValueError) as e:
        out['status'] = 'fail'
        out['problems'].append(f'invalid environment_slots: {e}')
        return out
    assignments = plan.get('cell_environments')
    planned = plan.get('planned_cells')
    if not isinstance(assignments, dict) or not assignments:
        out['status'] = 'fail'
        out['problems'].append('environment_slots is declared but cell_environments is missing/empty')
        return out
    if not isinstance(planned, list) or not planned:
        out['status'] = 'fail'
        out['problems'].append('environment_slots is declared but planned_cells is missing/empty')
        return out
    planned_labels = [str(x) for x in planned]
    if len(set(planned_labels)) != len(planned_labels):
        out['status'] = 'fail'
        out['problems'].append('planned_cells contains duplicate labels')
        return out
    missing = [x for x in planned_labels if x not in assignments]
    planned_set = set(planned_labels)
    extra = [k for k in assignments if str(k) not in planned_set]
    if missing:
        out['problems'].append(
            f'{len(missing)} planned cell(s) have no environment assignment: '
            f'{missing[:5]}')
    if extra:
        out['problems'].append(
            f'{len(extra)} environment assignment(s) name unplanned cells: '
            f'{extra[:5]}')
    slot_set = set(slots)
    cell_variants = plan.get('cell_variants')
    if not isinstance(cell_variants, dict):
        cell_variants = {}
    arm_counts = collections.OrderedDict()
    for label in planned_labels:
        if label not in assignments:
            continue
        slot = str(assignments.get(label))
        if slot not in slot_set:
            out['problems'].append(
                f'cell {label!r} is assigned to undeclared slot {slot!r}')
            continue
        variant = cell_variants.get(label)
        if variant is None:
            try:
                variant = label.rsplit('::seed', 1)[0]
            except ValueError:
                variant = label
        variant = str(variant)
        arm_counts.setdefault(variant, collections.Counter())[slot] += 1
    out['n_slots'] = len(slots)
    out['n_cells'] = len(planned_labels)
    out['slots'] = list(slots)
    out['arm_slot_counts'] = {
        v: {s: int(c.get(s, 0)) for s in slots} for v, c in arm_counts.items()
    }
    balanced = True
    for variant, counter in arm_counts.items():
        vals = [int(counter.get(s, 0)) for s in slots]
        if max(vals) - min(vals) > 1:
            balanced = False
            out['problems'].append(
                f'{variant}: slot counts differ by more than one ({vals}); '
                'the arm is not balanced across devices')
    out['slot_balanced'] = balanced
    if out['problems']:
        out['status'] = 'fail'
    else:
        out['status'] = 'ok'
    return out


def run_order_plan_timing(plan, events, *, kind='cell'):
    """Check that a run-order plan was written before the first matching cell.

    A plan reconstructed after execution can look perfectly randomized and
    still not have governed the run.  The comparison is intentionally simple
    and conservative: the plan's ``created_at`` must not be later than the
    earliest matching ``started_at``.  A missing/malformed timestamp is
    ``not_verifiable``/``fail`` rather than an implicit pass.
    """
    out = {
        'kind': 'run_order_plan_timing',
        'status': 'ok',
        'plan_created_at': None,
        'first_event_at': None,
        'n_events': 0,
        'event_kind': kind,
        'problems': [],
        'warnings': [],
        'note': ('a post-hoc run-order plan is not a precommitment even when '
                 'its permutation is complete and position-balanced'),
    }
    if not isinstance(plan, dict):
        out['status'] = 'not_verifiable_locally'
        out['reason'] = 'no run-order plan object'
        return out
    created_raw = plan.get('created_at')
    out['plan_created_at'] = created_raw
    created = _allocation_timestamp(created_raw)
    if created is None:
        out['status'] = ('not_verifiable_locally' if created_raw in (None, '')
                         else 'fail')
        if created_raw in (None, ''):
            out['warnings'].append('the run-order plan has no created_at timestamp')
        else:
            out['problems'].append(
                f'plan.created_at={created_raw!r} is not a parseable timestamp')
        return out
    kinds = kind if isinstance(kind, (list, tuple, set)) else (kind,)
    kinds = {str(k) for k in kinds}
    starts = []
    skipped = 0
    for event in events or []:
        if not isinstance(event, dict):
            continue
        if str(event.get('kind')) not in kinds:
            continue
        ts = _allocation_timestamp(event.get('started_at'))
        if ts is None:
            if event.get('started_at') not in (None, ''):
                skipped += 1
            continue
        starts.append(ts)
    out['n_events'] = len(starts)
    if skipped:
        out['warnings'].append(
            f'{skipped} matching event(s) have an unparseable started_at and '
            'were ignored')
    if not starts:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = 'no matching event carries a parseable started_at'
        return out
    first = min(starts)
    out['first_event_at'] = first.isoformat()
    if not _comparable_timestamps(created, first):
        out['status'] = 'not_verifiable_locally'
        out['reason'] = ('plan.created_at and the first event use different '
                         'timezone conventions; their order cannot be checked')
        return out
    if created > first:
        out['status'] = 'fail'
        out['problems'].append(
            f'plan.created_at={created.isoformat()} is later than the first '
            f'matching {sorted(kinds)} event ({first.isoformat()}); the plan '
            'was written after execution and is not a precommitment')
    return out


def run_allocation_audit(records=None, *, plan=None, events=None,
                         kind='cell', min_records_per_arm=None):
    """Combine prospective-plan timing, device design and observed balance."""
    timing = run_order_plan_timing(plan, events, kind=kind)
    plan_env = environment_plan_audit(plan)
    actual = environment_balance_audit(
        records if records is not None else {},
        min_records_per_arm=min_records_per_arm)
    out = {
        'kind': 'run_allocation_audit',
        'status': 'ok',
        'timing': timing,
        'environment_plan': plan_env,
        'environment_actual': actual,
        'problems': [],
        'warnings': [],
        'note': ('prospective allocation and arm x environment balance; '
                 'descriptive only, no measurement or frozen decision changes'),
    }
    for block in (timing, plan_env, actual):
        out['problems'].extend(block.get('problems') or [])
        out['warnings'].extend(block.get('warnings') or [])
    _statuses = {b.get('status') for b in (timing, plan_env, actual)}
    if 'fail' in _statuses:
        out['status'] = 'fail'
    elif (timing.get('status') == 'not_verifiable_locally'
          and actual.get('status') == 'not_verifiable_locally'
          and plan_env.get('status') in ('not_declared', 'not_verifiable_locally')):
        # Nothing to audit: no plan, no events, no environment-bearing records.
        out['status'] = 'not_verifiable_locally'
    elif all(s == 'not_verifiable_locally' for s in _statuses):
        out['status'] = 'not_verifiable_locally'
    elif 'warnings' in _statuses or 'not_declared' in _statuses:
        out['status'] = 'warnings'
    elif 'not_verifiable_locally' in _statuses:
        out['status'] = 'warnings'
    else:
        out['status'] = 'ok'
    return out
