"""Prospective preparation clearance: the launch gate between design and run.

``design_protocol`` records *what* the study intends to do.  It does not
prove that the machinery needed to do it was ready at the moment the first
cell started.  The individual preparation checks exist, but they are separate
objects checked by separate commands:

* ``run_order.json`` -- is the allocation a complete, position-balanced
  randomization of the registered cells?
* ``run_disposition.json`` -- is the failed-cell policy the one that was
  registered, over the same planned cells?
* the pre-test power artifact -- did the panel clear the registered power
  gate on validation deltas before the held-out split was touched?
* the dependency contract and analysis closure -- is the measured stack the
  registered one?
* the pre-run rehearsal (``confirmation plan``) -- did the frozen admission
  and decision path actually run end to end on synthetic records?
* the held-out lock -- has the test split *not* been read yet?

This fragment compiles those pieces into one hash-bound
``preparation_clearance``.  It is the object a launch should require: a
complete protocol plus a complete set of *passed* checks, all bound to the
same protocol hash and written before the first event.  Missing evidence is
never silently treated as a pass.  The clearance adds no test, p-value, Holm
slot or run and reads JSON only; it never calls a data loader.
"""

import hashlib
import json
import os
import time


PREPARATION_CLEARANCE_KIND = 'preparation_clearance'
PREPARATION_CLEARANCE_SCHEMA = 1
PREPARATION_CLEARANCE_CHECK_ORDER = (
    'protocol', 'source_tree', 'run_order', 'disposition', 'power',
    'rehearsal', 'test_access', 'dependency', 'environment', 'temporal_order',
)

PREPARATION_CLEARANCE_POLICY = {
    'version': 1,
    'strict_levels': ('confirmatory',),
    'levels': ('confirmatory', 'exploratory', 'draft'),
    'required_confirmatory': (
        'protocol', 'source_tree', 'run_order', 'disposition', 'power',
        'rehearsal', 'test_access', 'dependency',
    ),
    'role': (
        'one prospective, hash-bound clearance that the registered design is '
        'actually launchable: allocation, disposition, power, environment, '
        'rehearsal and held-out access all agree with the same protocol and '
        'were checked before the first cell.'),
    'rule': (
        'confirmatory: every required check must pass; a missing power '
        'artifact, rehearsal, held-out state or dependency contract is a '
        'failure, not a warning.  exploratory/draft: missing pieces are '
        'warnings and the clearance is not green.'),
    'timing_rule': (
        'protocol.created_at <= clearance.created_at <= first recorded cell '
        'event.  A clearance written after execution is a reconstruction.'),
    'boundary': (
        'a clearance is not a signature, a trusted timestamp or proof of '
        'execution.  It makes an unprepared or inconsistent launch visible; '
        'it does not stop someone who rewrites every input and the clearance '
        'together, and it does not replace the result-level audits.'),
}


def _prep_text(value):
    """Non-empty stripped text, or None; booleans are not names."""
    if value is None or isinstance(value, (bool,)):
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _prep_study_level(protocol):
    """Read ``protocol.study.level`` without assuming the study block exists."""
    study = protocol.get('study') if isinstance(protocol, dict) else None
    return _prep_text(study.get('level')) if isinstance(study, dict) else None


def _prep_int(value, *, minimum=None):
    """Exact integer (booleans rejected), or None."""
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


def _prep_number(value):
    """Finite float, or None; booleans/strings are not measurements."""
    if value is None or isinstance(value, (bool, str, bytes, bytearray)):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if out == out and out not in (float('inf'), float('-inf')) else None


def _prep_str_list(value):
    """String-normalise a declared list field, or return None if malformed.

    A protocol/plan is untrusted JSON: ``planned_cells`` is normally a list,
    but a hand-edited artifact can put a scalar or mapping there.  Returning
    None lets the caller record a named validation failure instead of raising
    ``TypeError: 'int' object is not iterable`` and aborting the whole launch
    gate before it can explain what is wrong.
    """
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        return None
    return [str(x) for x in value]


def _prep_detail_head(value, limit=5):
    """Render the first few diagnostic items of a malformed evidence field.

    Evidence objects are untrusted: ``mismatches``/``missing``/``problems``
    are normally lists, but a hand-written or third-party contract may make
    one a mapping, a bare string, or a scalar.  Slicing such a value directly
    raised ``TypeError``/``KeyError`` inside a gate whose whole contract is
    "fail closed, never crash"; this helper keeps the historical list/tuple
    rendering and turns every other shape into a readable short list.
    """
    if value is None:
        return []
    if isinstance(value, dict):
        return [f'{k}={v}' for k, v in list(value.items())[:limit]]
    if isinstance(value, (list, tuple)):
        return list(value[:limit])
    if isinstance(value, (set, frozenset)):
        return list(value)[:limit]
    return [value][:limit]


def _prep_canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False)


def _prep_sha_obj(obj):
    return hashlib.sha256(_prep_canonical(obj).encode('utf-8')).hexdigest()


def _prep_sha256_text(value):
    text = _prep_text(value)
    if text is None:
        return None
    text = text.lower()
    if len(text) != 64 or any(ch not in '0123456789abcdef' for ch in text):
        return None
    return text


def _prep_timestamp(value):
    """Parse the ISO-like timestamps used by the drivers, or return None."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith('Z'):
        text = text[:-1] + '+00:00'
    try:
        return _dp_dt.datetime.fromisoformat(text)  # noqa: F821
    except (TypeError, ValueError):
        for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%dT%H:%M', '%Y-%m-%d'):
            try:
                return _dp_dt.datetime.strptime(text, fmt)  # noqa: F821
            except (TypeError, ValueError):
                continue
    return None


def _prep_comparable(a, b):
    if a is None or b is None:
        return False
    return (a.tzinfo is None) == (b.tzinfo is None)


def _prep_check(name, *, required=False):
    return {'name': name, 'required': bool(required), 'status': 'ok',
            'problems': [], 'warnings': []}


def _prep_fail(check, message):
    check['problems'].append(str(message))
    check['status'] = 'fail'
    return check


def _prep_warn(check, message, *, unverifiable=False):
    check['warnings'].append(str(message))
    if check['status'] == 'ok':
        check['status'] = ('not_verifiable' if unverifiable else 'warnings')
    return check


def _prep_not_applicable(check, reason):
    check['status'] = 'not_applicable'
    check['reason'] = str(reason)
    return check


def _prep_require(check, strict, message):
    """Missing preparation evidence: hard for confirmatory, named warning else."""
    if strict:
        return _prep_fail(check, message)
    return _prep_warn(check, message, unverifiable=True)


def _prep_hash_into(check, key, obj):
    """Record a canonical hash, failing closed if the object is not JSON."""
    try:
        check[key] = _prep_sha_obj(obj)
    except (TypeError, ValueError, RecursionError) as exc:
        _prep_fail(check, f'{key} cannot be computed: '
                         f'{type(exc).__name__}: {exc}')
    return check


def _prep_protocol_check(protocol, *, power_supplied=False):
    out = _prep_check('protocol', required=True)
    if not isinstance(protocol, dict):
        return _prep_fail(out, 'protocol must be an object')
    out['design_protocol_sha256'] = _prep_text(protocol.get('content_sha256'))
    study = protocol.get('study') if isinstance(protocol.get('study'), dict) else {}
    out['level'] = _prep_text(study.get('level'))
    out['study_name'] = _prep_text(study.get('name'))
    out['manifest_sha256'] = _prep_text(study.get('manifest_sha256'))
    out['protocol_created_at'] = protocol.get('created_at')
    if protocol.get('kind') != DESIGN_PROTOCOL_KIND:  # noqa: F821
        _prep_fail(out, f'protocol.kind must be {DESIGN_PROTOCOL_KIND!r}')  # noqa: F821
    if protocol.get('schema') != DESIGN_PROTOCOL_SCHEMA:  # noqa: F821
        _prep_fail(out, 'protocol.schema is not the supported design-protocol schema')
    recorded = _prep_text(protocol.get('content_sha256'))
    if recorded is None:
        _prep_fail(out, 'protocol.content_sha256 is missing')
    else:
        try:
            want = design_protocol_content_sha(protocol)  # noqa: F821
        except (TypeError, ValueError, RecursionError) as exc:
            _prep_fail(out, f'protocol body is not canonical JSON: '
                            f'{type(exc).__name__}: {exc}')
        else:
            if recorded != want:
                _prep_fail(out, 'protocol.content_sha256 does not reproduce from '
                                'its body; the registered design was edited')
    try:
        validation = validate_design_protocol(protocol)  # noqa: F821
    except Exception as exc:  # a malformed body must not crash the gate
        _prep_fail(out, f'protocol validation raised '
                        f'{type(exc).__name__}: {exc}')
        return out
    out['validation_status'] = validation.get('status')
    if validation.get('status') == 'fail':
        _prep_fail(out, 'design protocol is invalid: '
                        + '; '.join((validation.get('problems') or [])[:5]))
    for warning in (validation.get('warnings') or [])[:12]:
        if power_supplied and 'power_method is a projection' in str(warning):
            # The protocol was compiled before the power artifact existed; a
            # supplied, matching artifact resolves exactly this warning.
            out.setdefault('resolved_warnings', []).append(str(warning))
            continue
        _prep_warn(out, f'design protocol warning: {warning}')
    return out


def _prep_source_tree_check(protocol, source_tree, *, strict=True):
    if not isinstance(protocol, dict):
        protocol = {}
    out = _prep_check('source_tree', required=True)
    impl = ((protocol.get('analysis') or {}).get('implementation')
            if isinstance(protocol.get('analysis'), dict) else None)
    if not isinstance(impl, dict):
        return _prep_require(
            out, strict, 'protocol does not bind an analysis implementation')
    out['registered_analysis_source_sha256'] = _prep_text(
        impl.get('analysis_source_sha256'))
    out['registered_full_source_tree_sha256'] = _prep_text(
        impl.get('full_source_tree_sha256'))
    if source_tree is None:
        return _prep_warn(out, 'no source tree was supplied; the analysis '
                               'closure cannot be re-hashed before launch',
                          unverifiable=True)
    try:
        current = _dp_implementation_binding(source_tree)  # noqa: F821
    except Exception as exc:
        return _prep_fail(out, f'analysis closure could not be hashed: '
                               f'{type(exc).__name__}: {exc}')
    if not isinstance(current, dict):
        return _prep_fail(out, 'source tree has no readable analysis closure')
    out['current_analysis_source_sha256'] = _prep_text(
        current.get('analysis_source_sha256'))
    out['current_n_files'] = _prep_int(current.get('analysis_n_files'),
                                       minimum=0)
    if out['current_analysis_source_sha256'] != out['registered_analysis_source_sha256']:
        _prep_fail(out, 'analysis implementation changed after registration '
                        '(analysis closure digest differs)')
    if out['current_n_files'] != _prep_int(impl.get('analysis_n_files'),
                                           minimum=0):
        _prep_fail(out, 'analysis closure file count differs from the '
                        'registered count')
    current_code = _prep_text(current.get('code_semantics'))
    if current_code is not None and current_code != _prep_text(
            impl.get('code_semantics')):
        _prep_fail(out, 'code semantics changed after registration')
    current_reg = _prep_text(current.get('split_registry_status'))
    registered_reg = _prep_text(impl.get('split_registry_status'))
    if registered_reg == 'ok' and current_reg != 'ok':
        # `None` means the supplied tree did not carry a registry at all; that
        # is not evidence that the previously registered order is intact.
        _prep_fail(out, 'executed split registry cannot be verified as ok '
                        f'(current status={current_reg!r})')
    return out


def _prep_run_order_check(protocol, run_order, *, strict=True):
    if not isinstance(protocol, dict):
        protocol = {}
    out = _prep_check('run_order', required=True)
    if not isinstance(run_order, dict):
        return _prep_require(out, strict, 'run_order plan is missing')
    try:
        balance = run_order_balance(run_order)  # noqa: F821
    except Exception as exc:
        return _prep_fail(out, f'run_order_balance raised '
                               f'{type(exc).__name__}: {exc}')
    out['design_ok'] = balance.get('design_ok')
    out['n_blocks'] = balance.get('n_blocks')
    out['n_cells'] = balance.get('n_cells')
    out['position_balanced'] = balance.get('position_balanced')
    if balance.get('status') != 'ok' or balance.get('design_ok') is not True:
        _prep_fail(out, 'run-order design is not a complete, position-balanced '
                        'randomization: '
                        + '; '.join((balance.get('problems') or [])[:5]))
    design = run_order.get('design') if isinstance(
        run_order.get('design'), dict) else {}
    rnd = protocol.get('randomization') if isinstance(
        protocol.get('randomization'), dict) else {}
    if not rnd:
        _prep_fail(out, 'protocol does not record a randomization plan')
        return out
    actual_cells = _prep_str_list(run_order.get('planned_cells'))
    if actual_cells is None:
        _prep_fail(out, 'run_order.planned_cells must be a list of cell labels')
        actual_cells = []
    registered_cells = _prep_str_list(rnd.get('planned_cells'))
    if registered_cells is None:
        _prep_fail(out, 'protocol randomization.planned_cells must be a list '
                        'of cell labels')
        registered_cells = []
    out['n_planned_cells'] = len(actual_cells)
    if actual_cells != registered_cells:
        first = next((i for i, (a, b) in enumerate(
            zip(actual_cells, registered_cells)) if a != b), None)
        detail = (f'; first difference at position {first}' if first is not None
                  else '; lengths %d vs %d' % (len(actual_cells),
                                               len(registered_cells)))
        _prep_fail(out, 'run_order.planned_cells is not the registered cell '
                        'sequence' + detail)
    if design.get('name') != rnd.get('design'):
        _prep_fail(out, f'run order design {design.get("name")!r} != '
                        f'registered design {rnd.get("design")!r}')
    if _prep_int(design.get('design_seed'), minimum=0) != _prep_int(
            rnd.get('design_seed'), minimum=0):
        _prep_fail(out, 'run-order design_seed differs from the registration')
    if _prep_text(design.get('block')) != _prep_text(rnd.get('block')):
        _prep_fail(out, 'run-order block key differs from the registration')
    if bool(design.get('position_balance')) != bool(rnd.get('position_balance')):
        _prep_fail(out, 'run-order position-balance declaration differs from '
                        'the registration')
    registered_slots = rnd.get('environment_slots')
    if registered_slots:
        try:
            env_plan = environment_plan_audit(run_order)  # noqa: F821
        except Exception as exc:
            _prep_fail(out, f'environment_plan_audit raised '
                            f'{type(exc).__name__}: {exc}')
        else:
            if env_plan.get('status') == 'fail':
                _prep_fail(out, 'declared environment slots are not balanced: '
                                + '; '.join((env_plan.get('problems') or [])[:3]))
            elif env_plan.get('status') != 'ok':
                _prep_warn(out, f'environment slots could not be verified '
                                f'({env_plan.get("status")})')
    return _prep_hash_into(out, 'run_order_sha256', run_order)


def _prep_disposition_check(protocol, disposition, *, strict=True):
    if not isinstance(protocol, dict):
        protocol = {}
    out = _prep_check('disposition', required=True)
    if not isinstance(disposition, dict):
        return _prep_require(out, strict, 'run-disposition plan is missing')
    try:
        validated = _dispo_validate_plan(disposition)  # noqa: F821
    except Exception as exc:
        return _prep_fail(out, f'run-disposition plan is invalid: '
                               f'{type(exc).__name__}: {exc}')
    out['disposition_sha256'] = _prep_text(validated.get('content_sha256'))
    out['n_planned_cells'] = len(validated.get('planned_cells') or [])
    ops = protocol.get('operations') if isinstance(
        protocol.get('operations'), dict) else {}
    registered = ops.get('disposition')
    if not isinstance(registered, dict):
        return _prep_fail(out, 'protocol does not record a disposition policy')
    try:
        current = _dp_normalise_disposition(None, disposition)  # noqa: F821
    except Exception as exc:
        return _prep_fail(out, f'disposition policy could not be normalised: '
                               f'{type(exc).__name__}: {exc}')
    if current != registered:
        different = sorted(set(current or {}) | set(registered or {}))
        pairs = [k for k in different if (current or {}).get(k) !=
                 (registered or {}).get(k)]
        _prep_fail(out, 'disposition policy differs from the registration: '
                        + ', '.join(pairs[:6]))
    rnd = protocol.get('randomization') if isinstance(
        protocol.get('randomization'), dict) else {}
    _want_list = _prep_str_list(rnd.get('planned_cells'))
    if _want_list is None:
        _prep_fail(out, 'protocol randomization.planned_cells must be a list '
                        'of cell labels')
        _want_list = []
    _got_list = _prep_str_list(validated.get('planned_cells'))
    if _got_list is None:
        _prep_fail(out, 'disposition plan planned_cells must be a list of '
                        'cell labels')
        _got_list = []
    want = set(_want_list)
    got = set(_got_list)
    if got != want:
        missing = sorted(want - got)[:5]
        extra = sorted(got - want)[:5]
        _prep_fail(out, 'disposition plan does not cover the registered cells'
                        f' (missing={missing}, extra={extra})')
    return out


def _prep_power_report(power):
    if not isinstance(power, dict):
        return None
    for key in ('power_validation', 'report', 'plan'):
        if isinstance(power.get(key), dict):
            return power[key]
    return power


def _prep_power_check(protocol, power, *, strict=True):
    if not isinstance(protocol, dict):
        protocol = {}
    out = _prep_check('power')
    ss = protocol.get('sample_size') if isinstance(
        protocol.get('sample_size'), dict) else {}
    study = protocol.get('study') if isinstance(protocol.get('study'), dict) else {}
    level = _prep_text(study.get('level'))
    explicit = ss.get('power_gate_required')
    if isinstance(explicit, bool):
        required = explicit
    else:
        required = (level == 'confirmatory'
                    and ss.get('target_power') is not None
                    and ss.get('power_q') is not None)
    out['required'] = bool(required)
    out['target_power'] = _prep_number(ss.get('target_power'))
    out['power_q'] = _prep_number(ss.get('power_q'))
    out['family_size'] = _prep_int(ss.get('family_size'), minimum=1)
    out['planned_seeds'] = _prep_int(ss.get('planned_seeds'), minimum=1)
    out['min_n'] = _prep_int(ss.get('min_n'), minimum=1)
    out['alpha'] = _prep_number(study.get('alpha'))
    if not required:
        return _prep_not_applicable(out, 'the protocol does not declare a power gate')
    report = _prep_power_report(power)
    if not isinstance(report, dict):
        return _prep_fail(out, 'the protocol requires a power gate but no '
                               'power-validation artifact was supplied')
    _prep_hash_into(out, 'power_sha256', power)
    verdict = report.get('status')
    all_pass = report.get('all_pass')
    # Both the status label and the aggregate flag are evidence.  A `fail`
    # status with `all_pass=True` (or a passing status with `all_pass=False`)
    # is internally inconsistent and must not pass merely because one of the
    # two fields happens to look green.  An absent flag is not a contradiction
    # when the explicit status already passes.
    if (verdict not in ('pass', 'passed', 'ok')
            or all_pass is False):
        _prep_fail(out, f'power-validation artifact did not pass '
                        f'(status={verdict!r}, all_pass={all_pass!r})')
    for key, want, label in (
            ('alpha', out['alpha'], 'alpha'),
            ('m_family', out['family_size'], 'family size'),
            ('power_q_for_test', out['power_q'], 'power q'),
            ('min_power_for_test', out['target_power'], 'target power')):
        got = _prep_number(report.get(key))
        if got is None or want is None:
            _prep_fail(out, f'power artifact does not record a comparable {label}')
        elif abs(got - want) > 1e-12:
            _prep_fail(out, f'power artifact {key}={got} != registered {label}={want}')
    comps = protocol.get('comparisons') if isinstance(
        protocol.get('comparisons'), list) else []
    cells = report.get('comparisons') if isinstance(
        report.get('comparisons'), dict) else {}
    names = {str(c.get('name')) for c in comps if isinstance(c, dict)
             and c.get('name') is not None}
    if not cells:
        _prep_fail(out, 'power artifact carries no per-comparison cells')
    missing = sorted(names - set(cells))
    extra = sorted(set(cells) - names)
    if missing:
        _prep_fail(out, f'power artifact omits registered comparisons: {missing[:6]}')
    if extra:
        _prep_warn(out, f'power artifact carries unregistered comparisons: {extra[:6]}')
    q = out['power_q']
    min_power = out['target_power']
    q_key = f'q={q:.2f}' if q is not None else None
    for comp in comps:
        if not isinstance(comp, dict) or comp.get('name') is None:
            continue
        name = str(comp['name'])
        cell = cells.get(name)
        if not isinstance(cell, dict):
            continue
        directional = (comp.get('predicted_sign') in (-1, 1)
                       and comp.get('non_directional') is not True)
        if not directional:
            continue
        ok = cell.get('power_ok')
        value = None
        table = cell.get('power')
        if isinstance(table, dict):
            power_map = table.get('power')
            if isinstance(power_map, dict) and q_key is not None:
                value = _prep_number(power_map.get(q_key))
        if value is None:
            _prep_fail(out, f'{name}: no numeric power at the registered '
                            f'{q_key}; a status label is not a power reading')
        elif min_power is not None and value + 1e-12 < min_power:
            _prep_fail(out, f'{name}: power at {q_key} is {value:.6f} < '
                            f'target {min_power}')
        if ok is False:
            _prep_fail(out, f'{name}: power_ok is false')
        n = _prep_int(cell.get('n'), minimum=0)
        if n is None:
            _prep_fail(out, f'{name}: power calibration does not record the '
                            'number of complete validation pairs')
        elif out['min_n'] is not None and n < out['min_n']:
            _prep_fail(out, f'{name}: power calibration has n={n} complete '
                            f'pairs, below the registered min_n={out["min_n"]}')
        elif out['planned_seeds'] is not None \
                and n != out['planned_seeds']:
            _prep_warn(out, f'{name}: power calibration uses n={n}, not the '
                            f'planned {out["planned_seeds"]} seeds')
    declared_manifest = _prep_text(report.get('manifest_sha256'))
    registered_manifest = _prep_text(study.get('manifest_sha256'))
    if declared_manifest and registered_manifest \
            and declared_manifest != registered_manifest:
        _prep_fail(out, 'power artifact names a different manifest')
    return out


def _prep_rehearsal_verdict(rehearsal, *, _depth=0):
    """Return ``(ok, detail, payload)`` for the supported rehearsal shapes."""
    if _depth > 32:
        # A self-referential/deeply nested object is a malformed artifact, not
        # a verdict; refuse it instead of exhausting the interpreter stack.
        return None, 'rehearsal nesting exceeds 32 levels', {}
    if isinstance(rehearsal, bool):
        return rehearsal, None, {}
    if not isinstance(rehearsal, dict):
        return None, f'unsupported rehearsal object {type(rehearsal).__name__}', {}
    if isinstance(rehearsal.get('plan'), dict):
        return _prep_rehearsal_verdict(rehearsal.get('plan'),
                                       _depth=_depth + 1)
    if isinstance(rehearsal.get('ok'), bool):
        errors = rehearsal.get('errors')
        if rehearsal.get('ok') is False and isinstance(errors, list) and errors:
            return False, '; '.join(str(x) for x in errors[:5]), rehearsal
        return rehearsal.get('ok'), None, rehearsal
    status = _prep_text(rehearsal.get('status'))
    if status is not None:
        return status.lower() in ('ok', 'pass', 'passed'), None, rehearsal
    errors = rehearsal.get('errors')
    if isinstance(errors, list):
        return (not errors), ('; '.join(str(x) for x in errors[:5]) or None), rehearsal
    return None, 'rehearsal object has no ok/status/errors verdict', rehearsal


def _prep_rehearsal_check(protocol, rehearsal):
    if not isinstance(protocol, dict):
        protocol = {}
    out = _prep_check('rehearsal', required=True)
    # A malformed/absent study block is a protocol problem that the protocol
    # check already reports; reading its level must not raise here and abort
    # the whole launch gate with an AttributeError.
    _study = protocol.get('study')
    level = _prep_text(_study.get('level')) if isinstance(_study, dict) else None
    if rehearsal is None:
        if level == 'confirmatory':
            return _prep_fail(out, 'no pre-run rehearsal of the frozen '
                                   'admission / decision path was supplied')
        return _prep_warn(out, 'no pre-run rehearsal was supplied',
                          unverifiable=True)
    verdict, detail, payload = _prep_rehearsal_verdict(rehearsal)
    if verdict is None:
        if level == 'confirmatory':
            return _prep_fail(out, 'no pre-run rehearsal of the frozen admission '
                                   '/ decision path was supplied'
                                   + (f': {detail}' if detail else ''))
        return _prep_warn(out, 'no readable pre-run rehearsal was supplied'
                               + (f': {detail}' if detail else ''),
                          unverifiable=True)
    out['rehearsal_ok'] = bool(verdict)
    if not verdict:
        return _prep_fail(out, 'pre-run rehearsal did not pass'
                               + (f': {detail}' if detail else ''))
    _analysis = (protocol.get('analysis')
                 if isinstance(protocol.get('analysis'), dict) else {})
    impl = (_analysis.get('implementation')
            if isinstance(_analysis.get('implementation'), dict) else {})
    registered = _prep_text(impl.get('analysis_source_sha256'))
    declared = _prep_text(payload.get('analysis_source_sha256')) if isinstance(
        payload, dict) else None
    if registered and declared and registered != declared:
        _prep_fail(out, 'rehearsal was run against a different analysis closure')
    # The plan rehearsal may carry a known-truth rehearsal of the decision
    # path.  Only its identity is copied into the clearance (status, hashes,
    # counts); the synthetic p-values stay in the rehearsal artifact, so a
    # clearance still cannot be read as containing a test result.
    _ar = payload.get('analysis_rehearsal') if isinstance(payload, dict) else None
    if isinstance(_ar, dict):
        try:
            _arv = validate_analysis_rehearsal(_ar)  # noqa: F821
        except Exception as exc:
            _arv = {'status': 'fail',
                    'problems': [f'{type(exc).__name__}: {exc}']}
        out['analysis_rehearsal'] = {
            'status': _arv.get('status'),
            'content_sha256': _ar.get('content_sha256'),
            'manifest_sha256': _ar.get('manifest_sha256'),
            'n_scenarios': _ar.get('n_scenarios'),
            'n_failures': _ar.get('n_failures'),
            'synthetic': _ar.get('synthetic'),
            'test_split_read': _ar.get('test_split_read'),
        }
        if _arv.get('status') != 'ok':
            _prep_fail(
                out, 'known-truth analysis rehearsal did not pass: '
                     + '; '.join(str(x) for x in
                                 (_arv.get('problems') or [])[:5]))
        else:
            _study = protocol.get('study') if isinstance(
                protocol.get('study'), dict) else {}
            _want = _prep_text(_study.get('manifest_sha256'))
            _candidates = {
                _prep_text(_ar.get(k)) for k in (
                    'manifest_sha256', 'manifest_canonical_sha256',
                    'manifest_raw_canonical_sha256')
            }
            _candidates.discard(None)
            if _want and not _candidates:
                _prep_fail(
                    out, 'known-truth analysis rehearsal carries no manifest '
                         'identity, so it cannot be tied to the registered '
                         'frozen plan')
            elif _want and _candidates and _want not in _candidates:
                _prep_fail(
                    out, 'known-truth analysis rehearsal names a different '
                         'frozen manifest')
    else:
        out['analysis_rehearsal'] = {
            'status': 'missing',
            'note': ('the rehearsal payload carries no known-truth analysis '
                     'rehearsal; run `python -m explib.confirmation plan` with '
                     'the current code to bind it'),
        }
        # Backward compatibility: a legacy plan snapshot or a bare
        # ``{'ok': true}`` rehearsal cannot be upgraded into the known-truth
        # control.  Record the gap without turning an otherwise complete
        # clearance red; ``confirmation plan`` now always embeds the artifact.
    return _prep_hash_into(out, 'rehearsal_sha256', rehearsal)


def _prep_test_read_from_path(path):
    """True when ``path`` is a freeze lock or a panel directory holding one."""
    if not isinstance(path, str) or not path:
        return None
    if os.path.isdir(path):
        return os.path.exists(os.path.join(path, '.freeze_lock.json'))
    return os.path.exists(path)


def _prep_test_access_check(protocol, state):
    out = _prep_check('test_access', required=True)
    level = _prep_study_level(protocol)
    read = None
    evidence = None
    if isinstance(state, bool):
        read = state
        evidence = 'boolean'
    elif isinstance(state, str):
        evidence = state
        read = _prep_test_read_from_path(state)
    elif isinstance(state, dict):
        evidence = state.get('evidence') or state.get('path') or 'object'
        path = state.get('path')
        if isinstance(path, str) and path:
            read = _prep_test_read_from_path(path)
        for key in ('test_read', 'read', 'accessed', 'test_accessed'):
            if isinstance(state.get(key), bool):
                read = state[key]
                break
        if read is None:
            for key in ('lock_present', 'freeze_lock_present'):
                if isinstance(state.get(key), bool):
                    read = state[key]
                    break
        reads = _prep_int(state.get('test_reads'), minimum=0)
        if reads is not None:
            read = reads > 0
    elif state is None:
        if level == 'confirmatory':
            return _prep_fail(out, 'held-out access state was not declared; a '
                                   'launch gate must show the test split is unread')
        return _prep_warn(out, 'held-out access state is not declared',
                          unverifiable=True)
    out['read'] = read
    out['evidence'] = evidence
    if read is None:
        return _prep_warn(out, 'held-out access state is not interpretable',
                          unverifiable=True)
    if read:
        return _prep_fail(out, 'the held-out split has already been read; a '
                               'preparation clearance is no longer prospective')
    return out


def _prep_dependency_check(dependency_contract, *, strict=True):
    out = _prep_check('dependency', required=True)
    if dependency_contract is None:
        return _prep_require(
            out, strict, 'no dependency contract was supplied; the '
            'measurement stack is not pinned to the registration')
    if not isinstance(dependency_contract, dict):
        return _prep_fail(out, 'dependency contract must be an object')
    out['dependency_status'] = _prep_text(dependency_contract.get('status'))
    out['lock_sha256'] = _prep_sha256_text(
        dependency_contract.get('lock_sha256'))
    if out['dependency_status'] != 'ok':
        detail = (dependency_contract.get('mismatches')
                  or dependency_contract.get('missing')
                  or dependency_contract.get('problems') or [])
        return _prep_fail(out, 'dependency contract is not ok'
                               + (f': {_prep_detail_head(detail)}'
                                  if detail else ''))
    # `status=ok` alone is a label; the contract must also identify the lock
    # bytes it actually compared.  A missing/malformed digest or a listed
    # mismatch means the environment is not pinned to the registration, even
    # if a hand-edited status says otherwise.
    detail = (dependency_contract.get('mismatches')
              or dependency_contract.get('missing')
              or dependency_contract.get('invalid')
              or dependency_contract.get('problems') or [])
    if detail:
        return _prep_fail(out, 'dependency contract lists unresolved '
                               f'problem(s): {_prep_detail_head(detail)}')
    if out['lock_sha256'] is None:
        return _prep_fail(out, 'dependency contract does not record the lock '
                               'bytes it compared')
    return out


def _prep_environment_check(environment):
    out = _prep_check('environment', required=False)
    if environment is None:
        return _prep_not_applicable(out, 'no environment identity supplied')
    try:
        ident = environment_identity(environment)  # noqa: F821
    except Exception as exc:
        return _prep_fail(out, f'environment identity could not be read: '
                               f'{type(exc).__name__}: {exc}')
    if ident is None:
        return _prep_fail(out, 'environment object carries no identity fields')
    out['environment_sha256'] = _prep_sha_obj(ident)
    return out


def _prep_temporal_check(protocol, created_at):
    if not isinstance(protocol, dict):
        protocol = {}
    out = _prep_check('temporal_order', required=True)
    proto = _prep_timestamp(protocol.get('created_at'))
    clear = _prep_timestamp(created_at)
    out['protocol_created_at'] = protocol.get('created_at')
    out['clearance_created_at'] = created_at
    if proto is None or clear is None:
        return _prep_warn(out, 'protocol/clearance timestamps are missing or '
                               'unparseable; their order cannot be checked',
                          unverifiable=True)
    if not _prep_comparable(proto, clear):
        return _prep_warn(out, 'protocol and clearance timestamps use different '
                               'timezone conventions', unverifiable=True)
    if clear < proto:
        return _prep_fail(out, 'clearance predates the design protocol it binds')
    return out


def build_preparation_clearance(
        protocol, *, run_order=None, disposition=None, power=None,
        environment=None, dependency_contract=None, source_tree=None,
        rehearsal=None, test_access_state=None, created_at=None, revision=0,
        supersedes=None, reason=None, notes=None):
    """Compile one hash-bound launch clearance from the existing evidence.

    Every value is copied from an existing object or check result.  The
    function does not invent a missing pass, a default power number or a
    fabricated held-out state; the strict confirmatory path turns absence into
    a failure and the exploratory/draft path into a named warning.
    """
    if not isinstance(protocol, dict):
        raise TypeError('protocol must be an object')
    created_at = created_at or time.strftime('%Y-%m-%dT%H:%M:%S')
    study = protocol.get('study') if isinstance(protocol.get('study'), dict) else {}
    level = _prep_text(study.get('level'))
    strict = level == 'confirmatory'

    checks = {
        'protocol': _prep_protocol_check(
            protocol, power_supplied=isinstance(power, dict) and bool(power)),
        'source_tree': _prep_source_tree_check(
            protocol, source_tree, strict=strict),
        'run_order': _prep_run_order_check(
            protocol, run_order, strict=strict),
        'disposition': _prep_disposition_check(
            protocol, disposition, strict=strict),
        'power': _prep_power_check(protocol, power, strict=strict),
        'rehearsal': _prep_rehearsal_check(protocol, rehearsal),
        'test_access': _prep_test_access_check(protocol, test_access_state),
        'dependency': _prep_dependency_check(
            dependency_contract, strict=strict),
        'environment': _prep_environment_check(environment),
        'temporal_order': _prep_temporal_check(protocol, created_at),
    }
    required = set(PREPARATION_CLEARANCE_POLICY['required_confirmatory'])
    if checks['power']['status'] == 'not_applicable':
        required.discard('power')
    if not strict:
        required = set()

    if checks['protocol']['status'] == 'fail':
        # Without a valid registration every downstream statement is moot.
        for name in PREPARATION_CLEARANCE_CHECK_ORDER:
            if name == 'protocol':
                continue
            check = checks[name]
            if check.get('status') == 'ok':
                check['status'] = 'not_verifiable'
                check['warnings'].append(
                    'not meaningful until the design protocol validates')

    problems = []
    warnings = []
    for name in PREPARATION_CLEARANCE_CHECK_ORDER:
        check = checks[name]
        check['required'] = name in required
        for message in check.get('problems') or []:
            problems.append(f'{name}: {message}')
        for message in check.get('warnings') or []:
            warnings.append(f'{name}: {message}')
    missing_required = [name for name in required
                        if checks[name].get('status') != 'ok']
    for name in missing_required:
        message = f'{name}: required preparation evidence is not a passed check'
        if message not in problems:
            problems.append(message)

    if problems:
        status = 'fail'
    elif warnings:
        status = 'warnings'
    else:
        status = 'ok'
    clearance = {
        'kind': PREPARATION_CLEARANCE_KIND,
        'schema': PREPARATION_CLEARANCE_SCHEMA,
        'study': {
            'name': _prep_text(study.get('name')),
            'level': level,
            'manifest_sha256': _prep_text(study.get('manifest_sha256')),
            'design_protocol_sha256': _prep_text(protocol.get('content_sha256')),
            'design_protocol_created_at': protocol.get('created_at'),
            'code_semantics': _prep_text(study.get('code_semantics')),
        },
        'required_checks': sorted(required),
        'checks': checks,
        'status': status,
        'launch_allowed': status == 'ok',
        'problems': problems,
        'warnings': warnings,
        'policy': dict(PREPARATION_CLEARANCE_POLICY),
        'created_at': created_at,
        'revision': _prep_int(revision, minimum=0) or 0,
        'supersedes': [str(x) for x in (supersedes or [])],
        'reason': _prep_text(reason),
        'notes': _prep_text(notes),
    }
    clearance['content_sha256'] = _prep_sha_obj(
        {k: v for k, v in clearance.items()
         if k not in ('created_at', 'revision', 'supersedes', 'reason',
                      'notes', 'content_sha256')})
    return clearance


def validate_preparation_clearance(clearance, *, protocol=None):
    """Check structure, identity and internal consistency of a clearance."""
    out = {'kind': 'preparation_clearance_validation', 'status': 'ok',
           'level': None, 'strict': None, 'missing': [], 'problems': [],
           'warnings': [], 'checks': {}}
    if not isinstance(clearance, dict):
        out['problems'].append(
            f'clearance must be an object, got {type(clearance).__name__}')
        out['status'] = 'fail'
        return out
    if clearance.get('kind') != PREPARATION_CLEARANCE_KIND:
        out['problems'].append(
            f'clearance.kind must be {PREPARATION_CLEARANCE_KIND!r}')
    if clearance.get('schema') != PREPARATION_CLEARANCE_SCHEMA:
        out['problems'].append(
            f'clearance.schema must be {PREPARATION_CLEARANCE_SCHEMA}')
    study = clearance.get('study') if isinstance(
        clearance.get('study'), dict) else {}
    level = _prep_text(study.get('level'))
    out['level'] = level
    if level not in PREPARATION_CLEARANCE_POLICY['levels']:
        out['problems'].append(f'study.level must be one of '
                               f'{PREPARATION_CLEARANCE_POLICY["levels"]}, '
                               f'got {level!r}')
    strict = level == 'confirmatory'
    out['strict'] = strict
    proto_sha = _prep_sha256_text(study.get('design_protocol_sha256'))
    if proto_sha is None:
        out['problems'].append(
            'study.design_protocol_sha256 must be a 64-character SHA-256 digest')
    manifest_sha = study.get('manifest_sha256')
    if manifest_sha is not None and _prep_sha256_text(manifest_sha) is None:
        out['problems'].append(
            'study.manifest_sha256 must be a 64-character SHA-256 digest or null')
    if _prep_text(study.get('name')) is None:
        out['problems'].append('study.name is missing')
    if protocol is not None:
        if not isinstance(protocol, dict):
            out['problems'].append('protocol argument must be an object')
        else:
            current = _prep_text(protocol.get('content_sha256'))
            if current != study.get('design_protocol_sha256'):
                out['problems'].append(
                    'clearance is bound to a different design protocol hash')
            proto_level = _prep_study_level(protocol)
            if proto_level != level:
                out['problems'].append(
                    'clearance level does not equal the protocol level')
    checks = clearance.get('checks')
    if not isinstance(checks, dict) or not checks:
        out['problems'].append('checks must be a non-empty object')
        checks = {}
    recorded_required = clearance.get('required_checks')
    if not isinstance(recorded_required, list) \
            or not all(isinstance(x, str) for x in recorded_required):
        out['problems'].append('required_checks must be a list of check names')
        recorded_required = []
    unknown_required = sorted(set(recorded_required)
                              - set(PREPARATION_CLEARANCE_CHECK_ORDER))
    if unknown_required:
        out['problems'].append(
            f'required_checks names unknown checks: {unknown_required}')
    if len(set(recorded_required)) != len(recorded_required):
        out['problems'].append('required_checks contains duplicates')
    # The check vocabulary is fixed.  An extra entry is not harmless metadata:
    # it would let a caller believe a check was recorded while the audit only
    # replays the ten known names (and it is part of the content hash).
    unknown_checks = sorted(set(checks) - set(PREPARATION_CLEARANCE_CHECK_ORDER))
    if unknown_checks:
        out['problems'].append(
            f'checks names unknown checks: {unknown_checks}')
    for name in PREPARATION_CLEARANCE_CHECK_ORDER:
        check = checks.get(name)
        if not isinstance(check, dict):
            if strict:
                out['problems'].append(f'checks.{name} is missing')
            else:
                out['warnings'].append(f'checks.{name} is missing')
            continue
        status = _prep_text(check.get('status'))
        if status not in ('ok', 'warnings', 'not_verifiable',
                          'not_applicable', 'fail'):
            out['problems'].append(f'checks.{name}.status={status!r} is not '
                                   'a recognised status')
        if (check.get('name') is not None
                and _prep_text(check.get('name')) != name):
            out['problems'].append(
                f'checks.{name}.name={check.get("name")!r} does not match its '
                'key')
        if 'required' in check and not isinstance(check.get('required'), bool):
            out['problems'].append(
                f'checks.{name}.required must be boolean')
        out['checks'][name] = status
        if status == 'fail':
            # ``checks`` is untrusted JSON: problems may be a mapping, a
            # scalar, or a list of non-strings.  Slicing/joining it directly
            # aborts the validator instead of returning the named failure.
            _detail = _prep_detail_head(check.get('problems'), limit=3)
            out['problems'].append(
                f'checks.{name} failed: '
                + '; '.join(str(x) for x in _detail))
        elif status in ('warnings', 'not_verifiable'):
            out['warnings'].append(f'checks.{name} is {status}')
    required = set(recorded_required)
    for name in PREPARATION_CLEARANCE_CHECK_ORDER:
        check = checks.get(name)
        if isinstance(check, dict) and 'required' in check:
            if check.get('required') is not (name in required):
                out['problems'].append(
                    f'checks.{name}.required={check.get("required")!r} does '
                    f'not match required_checks (name present: '
                    f'{name in required})')
    if strict:
        expected_required = set(
            PREPARATION_CLEARANCE_POLICY['required_confirmatory'])
        _power_entry = checks.get('power')
        if (isinstance(_power_entry, dict)
                and _power_entry.get('status') == 'not_applicable'):
            expected_required.discard('power')
        missing = sorted(expected_required - required)
        if missing:
            out['problems'].append(
                f'confirmatory clearance omits required checks: {missing}')
        extra = sorted(required - expected_required)
        if extra:
            out['problems'].append(
                f'confirmatory clearance marks non-required checks as '
                f'mandatory: {extra}')
        for name in sorted(expected_required):
            _entry = checks.get(name)
            if (not isinstance(_entry, dict)
                    or _entry.get('status') != 'ok'):
                out['problems'].append(
                    f'confirmatory check {name!r} is not ok')
    for name in required:
        if name not in checks:
            out['problems'].append(f'required check {name!r} is absent')
    if clearance.get('launch_allowed') is not (clearance.get('status') == 'ok'):
        out['problems'].append(
            'launch_allowed must equal (status == "ok")')
    if clearance.get('status') not in ('ok', 'warnings', 'fail'):
        out['problems'].append(
            f'clearance.status={clearance.get("status")!r} is not recognised')
    if clearance.get('status') == 'ok' and (out['problems'] or out['warnings']):
        out['problems'].append(
            'clearance is marked ok but its checks are not all clean')
    recorded = _prep_text(clearance.get('content_sha256'))
    if recorded is None:
        out['problems'].append('content_sha256 is missing')
    else:
        try:
            want = _prep_sha_obj({k: v for k, v in clearance.items()
                                  if k not in ('created_at', 'revision',
                                               'supersedes', 'reason', 'notes',
                                               'content_sha256')})
        except (TypeError, ValueError) as exc:
            out['problems'].append(f'clearance body is not canonical JSON: '
                                   f'{type(exc).__name__}: {exc}')
        else:
            if recorded != want:
                out['problems'].append(
                    'content_sha256 does not reproduce from the clearance body')
    rev = _prep_int(clearance.get('revision'), minimum=0)
    if rev is None:
        out['problems'].append('revision must be an integer >= 0')
        rev = 0
    sup = clearance.get('supersedes')
    if not isinstance(sup, list):
        out['problems'].append('supersedes must be a list')
    elif rev == 0 and sup:
        out['problems'].append('a first clearance (revision=0) cannot supersede anything')
    elif rev > 0:
        if not sup:
            out['problems'].append('a revised clearance must list the superseded hash')
        if _prep_text(clearance.get('reason')) is None:
            out['problems'].append('a revised clearance must carry a reason')
    if _prep_timestamp(clearance.get('created_at')) is None:
        out['problems'].append('created_at is missing or not a parseable timestamp')
    if out['problems']:
        out['status'] = 'fail'
    elif out['warnings']:
        out['status'] = 'warnings'
    return out


def preparation_clearance_timing(clearance, events):
    """Check that the clearance was written before the first recorded event."""
    out = {'kind': 'preparation_clearance_timing', 'status': 'not_verifiable',
           'clearance_created_at': None, 'first_event_at': None,
           'first_event_kind': None, 'n_events': 0, 'reconstruction': None,
           'problems': [], 'warnings': []}
    if not isinstance(clearance, dict):
        out['status'] = 'fail'
        out['problems'].append('clearance must be an object')
        return out
    created = _prep_timestamp(clearance.get('created_at'))
    out['clearance_created_at'] = clearance.get('created_at')
    if created is None:
        out['status'] = 'fail'
        out['problems'].append('clearance.created_at is missing or unparseable')
        return out
    if events is None:
        events_iter = iter(())
    elif isinstance(events, str):
        events_iter = _dp_iter_events(events=events, panel_dirs=[])  # noqa: F821
    else:
        try:
            events_iter = iter(events)
        except TypeError:
            events_iter = iter(())
    post_run = {'analysis', 'report', 'audit', 'aggregate', 'look',
                'synthesis', 'reproduction'}
    first = None
    first_kind = None
    n = 0
    for ev in events_iter:
        if not isinstance(ev, dict):
            continue
        if str(ev.get('kind') or '').lower() in post_run:
            continue
        n += 1
        stamp = ev.get('started_at') or ev.get('created_at')
        when = _prep_timestamp(stamp)
        if when is None or not _prep_comparable(created, when):
            continue
        if first is None or when < first:
            first = when
            first_kind = ev.get('kind')
    out['n_events'] = n
    if first is None:
        if n == 0:
            out['warnings'].append('no recorded events; the clearance could '
                                   'not be ordered against execution')
        else:
            out['warnings'].append('no event carried a comparable timestamp')
        return out
    out['first_event_at'] = first.isoformat()
    out['first_event_kind'] = first_kind
    if created > first:
        out['status'] = 'fail'
        out['reconstruction'] = True
        out['problems'].append(
            f'clearance.created_at={clearance.get("created_at")!r} is later '
            f'than the first recorded {first_kind or "event"} at '
            f'{first.isoformat()}; this is a reconstruction')
        return out
    out['status'] = 'ok'
    out['reconstruction'] = False
    return out


def _prep_load_json(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh, parse_constant=_dp_reject_json_constant)  # noqa: F821


def _prep_find_named(name, explicit=None):
    if explicit is not None:
        try:
            return os.path.abspath(os.fsdecode(os.fspath(explicit)))
        except (TypeError, ValueError, OSError):
            # A non-path object is not a stored artifact; report it as not
            # found instead of crashing the gate before it can explain why.
            return None
    root = globals().get('_REPO') or os.getcwd()
    candidates = [os.path.join(root, name)]
    try:
        import glob as _glob
        for pattern in ('analysis_*', 'results_*'):
            candidates.extend(sorted(_glob.glob(
                os.path.join(root, pattern, name))))
    except Exception:
        pass
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    return None


def preparation_clearance_audit(
        path=None, protocol_path=None, events=None, panel_dirs=None,
        run_order=None, disposition=None, power=None, source_tree=None,
        rehearsal=None, dependency_contract=None, test_access_state=None,
        verbose=False):
    """Replay a stored clearance against its protocol and, when supplied,
    against the plan artifacts it claims to have checked.

    The recorded held-out state is replayed from the clearance itself, not
    from the current filesystem: a clearance that was genuinely written before
    the single test read must stay valid after that read.  Timing against the
    event ledger is what makes the prospective claim checkable.
    """
    out = {'kind': 'preparation_clearance_audit', 'status': 'ok',
           'path': None, 'protocol_path': None, 'validation': None,
           'timing': None, 'replay': {}, 'problems': [], 'warnings': [],
           'note': ('prospective launch evidence; this gate creates no test, '
                    'p-value, Holm slot or confirmation verdict')}
    clearance_path = _prep_find_named('preparation_clearance.json', path)
    out['path'] = clearance_path
    if clearance_path is None or not os.path.isfile(clearance_path):
        out['status'] = 'not_verifiable_locally'
        out['reason'] = ('no preparation_clearance.json at the repository root, '
                         'in analysis_*/results_*, or at the supplied path')
        out['warnings'].append(out['reason'])
        return out
    if protocol_path is None:
        nearby = os.path.join(os.path.dirname(clearance_path),
                             'design_protocol.json')
        if os.path.isfile(nearby):
            protocol_path = nearby
    proto_path = _prep_find_named('design_protocol.json', protocol_path)
    out['protocol_path'] = proto_path
    if proto_path is None or not os.path.isfile(proto_path):
        out['status'] = 'not_verifiable_locally'
        out['reason'] = 'clearance exists but its design protocol cannot be found'
        out['warnings'].append(out['reason'])
        return out
    try:
        clearance = _prep_load_json(clearance_path)
    except Exception as exc:
        out['status'] = 'fail'
        out['problems'].append(f'clearance is not strict JSON: '
                               f'{type(exc).__name__}: {exc}')
        return out
    try:
        protocol = _prep_load_json(proto_path)
    except Exception as exc:
        out['status'] = 'fail'
        out['problems'].append(f'design protocol is not strict JSON: '
                               f'{type(exc).__name__}: {exc}')
        return out
    validation = validate_preparation_clearance(clearance, protocol=protocol)
    out['validation'] = validation
    out['problems'].extend(validation.get('problems') or [])
    out['warnings'].extend(validation.get('warnings') or [])
    # A stored artifact is untrusted JSON: a top-level list/scalar is a
    # malformed clearance/protocol, not an object the replay helpers may call
    # ``.get()`` on.  Validation already names the bad shape; stop before any
    # replay helper can raise AttributeError and abort the gate.
    if not isinstance(clearance, dict) or not isinstance(protocol, dict):
        out['status'] = 'fail'
        if not isinstance(protocol, dict):
            out['problems'].append(
                'design protocol must be a JSON object, got '
                f'{type(protocol).__name__}')
        return out
    if validation.get('status') == 'fail':
        # A structurally invalid/tampered clearance is a hard failure even when
        # no replay artifact happens to be present to expose it a second time.
        out['status'] = 'fail'
    # Build the event stream once.  A path is passed straight to the timing
    # helper; otherwise scan the explicit panel dirs (never the whole tree when
    # the caller supplied an empty list).
    if events is None:
        try:
            events = _dp_iter_events(events=None, panel_dirs=panel_dirs)  # noqa: F821
        except Exception:
            events = iter(())
    timing = preparation_clearance_timing(clearance, events)
    out['timing'] = timing
    out['problems'].extend(timing.get('problems') or [])
    out['warnings'].extend(timing.get('warnings') or [])
    if timing.get('status') == 'fail':
        out['status'] = 'fail'

    # Re-run only the checks whose source artifacts are available.  A missing
    # replay input is not evidence against the clearance: it is recorded as
    # not replayed.  The recorded held-out state is replayed, not the current
    # filesystem, so a genuinely prospective clearance stays valid after the
    # single test read.
    near = os.path.dirname(clearance_path)

    # Replay artifacts belong to the clearance being audited.  Searching the
    # repository root even when the clearance lives in an unrelated temporary
    # directory would import another panel's plan (and report a false hash
    # drift against a perfectly valid clearance).  Only include the repository
    # fallback when the audited clearance is actually inside it.
    _base_dirs = [near, os.path.dirname(proto_path)]
    _repo_base = globals().get('_REPO') or os.getcwd()
    try:
        _in_repo = (os.path.commonpath(
            [os.path.realpath(_repo_base), os.path.realpath(near)])
            == os.path.realpath(_repo_base))
    except (OSError, ValueError):
        _in_repo = False
    if _in_repo and _repo_base not in _base_dirs:
        _base_dirs.append(_repo_base)

    def _artifact(name, explicit):
        if explicit is not None:
            return explicit
        for base in _base_dirs:
            for candidate in (os.path.join(base, name),
                              os.path.join(base, 'analysis_v12', name)):
                if os.path.isfile(candidate):
                    return candidate
        return None

    if run_order is None:
        run_order = _artifact('run_order.json', None)
    if disposition is None:
        disposition = _artifact('run_disposition.json', None)
    if power is None:
        power = _artifact('power_validation.json', None)
    if source_tree is None:
        try:
            source_tree = source_tree_manifest()  # noqa: F821
        except Exception:
            source_tree = None
    if rehearsal is None:
        snapshot = _artifact('plan_snapshot.json', None)
        if snapshot is not None:
            try:
                loaded = _prep_load_json(snapshot)
            except Exception:
                loaded = None
            if isinstance(loaded, dict):
                rehearsal = loaded.get('plan') or loaded
    _auto_dependency = False
    if dependency_contract is None:
        # The parameter has the same name as the module-level
        # ``dependency_contract()`` helper, so a direct call here would recurse
        # into the (None) local and always fail.  Resolve the live helper
        # through the module globals instead; a monkeypatched helper is still
        # honoured, and an unavailable helper remains missing evidence.
        _live_contract = globals().get('dependency_contract')
        if callable(_live_contract):
            try:
                dependency_contract = _live_contract()
                _auto_dependency = dependency_contract is not None
            except Exception:
                dependency_contract = None
    recorded_checks = clearance.get('checks') if isinstance(
        clearance.get('checks'), dict) else {}
    recorded_access = recorded_checks.get('test_access')
    if test_access_state is None and isinstance(recorded_access, dict):
        read = recorded_access.get('read')
        if isinstance(read, bool):
            test_access_state = {'test_read': read,
                                 'evidence': recorded_access.get('evidence')}

    def _load_if_path(value):
        if isinstance(value, (str, os.PathLike)) and os.path.isfile(value):
            try:
                return _prep_load_json(value)
            except Exception:
                return value
        return value

    run_order = _load_if_path(run_order)
    disposition = _load_if_path(disposition)
    power = _load_if_path(power)
    strict = _prep_study_level(protocol) == 'confirmatory'
    # The protocol check resolves one warning only when a power artifact was
    # supplied.  On replay the artifact may have been archived/moved while the
    # clearance still records that it was checked.  In that case use the
    # recorded (hash-bound) power check as the evidence for the resolved
    # warning; otherwise a valid clearance would fail merely because replay
    # could not find the file it was originally built with.  A malformed
    # recorded power check is still caught by validation above.
    _recorded_power = recorded_checks.get('power')
    _power_supplied = isinstance(power, dict) and bool(power)
    if (not _power_supplied and isinstance(_recorded_power, dict)
            and _recorded_power.get('status') in ('ok', 'warnings')):
        _power_supplied = True
    checks = {'protocol': _prep_protocol_check(
        protocol, power_supplied=_power_supplied)}
    if source_tree is not None:
        checks['source_tree'] = _prep_source_tree_check(
            protocol, source_tree, strict=strict)
    if run_order is not None:
        checks['run_order'] = _prep_run_order_check(
            protocol, run_order, strict=strict)
    if disposition is not None:
        checks['disposition'] = _prep_disposition_check(
            protocol, disposition, strict=strict)
    if power is not None:
        checks['power'] = _prep_power_check(protocol, power, strict=strict)
    if rehearsal is not None:
        checks['rehearsal'] = _prep_rehearsal_check(protocol, rehearsal)
    if test_access_state is not None:
        checks['test_access'] = _prep_test_access_check(
            protocol, test_access_state)
    if dependency_contract is not None:
        checks['dependency'] = _prep_dependency_check(
            dependency_contract, strict=strict)
    out['replay']['checks'] = {
        name: check.get('status') for name, check in checks.items()}
    out['replay']['replayed'] = sorted(checks)
    # Keep the recorded launch decision in the replay block.  It is part of the
    # clearance identity and lets the compact preflight summary state it
    # without re-deriving `status == ok` from a partial replay.
    out['replay']['launch_allowed'] = clearance.get('launch_allowed')
    for name, current in checks.items():
        recorded = recorded_checks.get(name)
        if not isinstance(recorded, dict):
            out['warnings'].append(
                f'replay produced {name} but the clearance did not record it')
            continue
        if current.get('status') != recorded.get('status'):
            out['problems'].append(
                f'{name}: recorded status {recorded.get("status")!r} does '
                f'not reproduce ({current.get("status")!r})')
            out['status'] = 'fail'
        for key in ('design_protocol_sha256', 'run_order_sha256',
                    'disposition_sha256', 'power_sha256', 'rehearsal_sha256',
                    'lock_sha256', 'current_analysis_source_sha256'):
            if key in recorded and key in current \
                    and recorded.get(key) != current.get(key):
                if (_auto_dependency and name == 'dependency'
                        and key == 'lock_sha256'
                        and current.get('status') == recorded.get('status')
                        == 'ok'):
                    # The caller did not supply the environment artifact: the
                    # helper found THIS checkout's lock, which need not be the
                    # one a clearance copied from another panel records.  A
                    # healthy but different lock is not a failed replay; a
                    # live contract that is itself not ok is still replayed
                    # and fails through the status comparison above.
                    out['warnings'].append(
                        f'{name}.{key} differs between the clearance and the '
                        'auto-discovered local environment; the local lock is '
                        'not the artifact recorded by this clearance, so it '
                        'was not treated as a failing replay')
                    continue
                out['problems'].append(
                    f'{name}.{key} differs between the clearance and the '
                    'artifact supplied for replay')
                out['status'] = 'fail'
    out['replay']['status'] = out['status']
    if out['status'] == 'ok' and out['warnings']:
        out['status'] = 'warnings'
    if verbose:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def write_preparation_clearance(outdir, clearance, *, keep_existing=True):
    """Write a validated clearance without silently rewriting a commitment."""
    if not isinstance(outdir, str) or not outdir.strip():
        raise ValueError('outdir must be a non-empty path')
    val = validate_preparation_clearance(clearance)
    if val['status'] == 'fail':
        raise ValueError('refusing to write an invalid preparation clearance: '
                         + '; '.join(val['problems'][:5]))
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'preparation_clearance.json')
    if os.path.isfile(path):
        try:
            old = _prep_load_json(path)
        except Exception as exc:
            raise ValueError(f'existing preparation_clearance.json is '
                             f'unreadable: {type(exc).__name__}: {exc}') from None
        old_sha = old.get('content_sha256') if isinstance(old, dict) else None
        new_sha = clearance.get('content_sha256')
        if old_sha == new_sha:
            return {'written': False, 'path': path, 'conflict': False,
                    'noop': True, 'content_sha256': new_sha}
        if keep_existing:
            return {'written': False, 'path': path, 'conflict': True,
                    'existing_sha256': old_sha,
                    'content_sha256': new_sha,
                    'reason': ('a different clearance already exists; pass '
                               'keep_existing=False with a revision that '
                               'supersedes it to replace it')}
        old_rev = _prep_int((old or {}).get('revision'), minimum=0) or 0
        new_rev = _prep_int(clearance.get('revision'), minimum=0)
        if new_rev is None or new_rev <= old_rev or new_rev < 1:
            raise ValueError('a replacement clearance must advance the revision')
        sup = clearance.get('supersedes')
        if not isinstance(sup, list) or old_sha not in sup:
            raise ValueError('a replacement clearance must list the existing '
                             'content_sha256 in supersedes')
        if _prep_text(clearance.get('reason')) is None:
            raise ValueError('a replacement clearance must carry a reason')
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(clearance, fh, ensure_ascii=False, indent=1, allow_nan=False)
        fh.write('\n')
    os.replace(tmp, path)
    return {'written': True, 'path': path, 'conflict': False,
            'content_sha256': clearance.get('content_sha256')}


def preparation_clearance_preflight_summary(path=None, protocol_path=None):
    """Compact block for ``audit_design_preflight`` / ``run_report``."""
    rep = preparation_clearance_audit(
        path=path, protocol_path=protocol_path, events=[], panel_dirs=[])
    validation = rep.get('validation') or {}
    _launch = (rep.get('replay') or {}).get('launch_allowed')
    if rep.get('status') == 'fail':
        # The recorded bit is part of the (possibly tampered) body.  A failed
        # replay must never be summarised as launchable.
        _launch = False
    return {
        'status': rep.get('status'),
        'path': rep.get('path'),
        'protocol_path': rep.get('protocol_path'),
        'level': validation.get('level'),
        'launch_allowed': _launch,
        'timing': (rep.get('timing') or {}).get('status'),
        'n_problems': len(rep.get('problems') or []),
        'n_warnings': len(rep.get('warnings') or []),
        'problems': list(rep.get('problems') or [])[:12],
        'warnings': list(rep.get('warnings') or [])[:12],
    }
