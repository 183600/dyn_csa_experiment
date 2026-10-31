# ---------------------------------------------------------------------------
# 123. The same comparison name is not the same pre-registered question
#
# `audit_confirmation_pvalues` replays every stored p from its per-seed delta
# vector, and `audit_confirmation_verdicts` replays the three-way decision from
# the stored direction/gates.  Both read the endpoint, expected sign and
# mechanism-claim flag *as stored in the artifact*.  An artifact can therefore
# stay internally consistent while answering a different question:
#
#   * move the `role='primary'` label from L4096 to L512 and the by-length
#     curve still replays against its own deltas;
#   * flip `expected_sign` (and `direction_ok`) so a reverse-significant result
#     looks like it matched the hypothesis;
#   * flip `mechanism_claim_allowed` from false to true;
#   * change alpha or the declared family.
#
# The frozen manifest is the independent record of the question.  This gate
# binds each artifact comparison back to it: exact family, endpoint length,
# expected direction, mechanism-claim flag, alpha and Holm slots, plus the
# internal identity between the headline p/mean and the curve cell the artifact
# calls primary.  It never recomputes a p or changes a frozen verdict.
# ---------------------------------------------------------------------------

_PLAN_CONFORMANCE_SCHEMA = 'confirmation_plan_conformance_v1'
_PC_INVALID = object()


def _pc_text(value):
    """Non-empty text, or None.  Booleans are not names."""
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _pc_int(value):
    """Exact integer, or None.  No float truncation and no booleans."""
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        as_float = float(value)
        if math.isfinite(as_float) and as_float.is_integer():
            return int(as_float)
    return None


def _pc_num(value):
    """Finite float, or None.  ``True`` is a flag, not a measurement."""
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _pc_sign(value):
    """Normalise a frozen expected direction.

    Returns ``None``/``'positive'``/``'negative'``, or ``_PC_INVALID`` for a
    malformed declaration.  An unknown spelling must not silently become
    "non-directional" (which would make a directional claim untestable while
    still appearing to conform).
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return _PC_INVALID
    text = value.strip().lower()
    if text in ('', 'none', 'null'):
        return None
    if text in ('positive', 'negative'):
        return text
    return _PC_INVALID


def _pc_endpoint_length(endpoint):
    """The integer length in ``by_len.ppl@<int>``, or None."""
    if not isinstance(endpoint, str):
        return None
    match = re.fullmatch(r'by_len\.ppl@([0-9]+)', endpoint.strip())
    if match is None:
        return None
    return int(match.group(1))


def _pc_manifest(manifest_path, problems, warnings):
    """Load and validate the tracked frozen plan that the artifact must match."""
    path = os.path.abspath(manifest_path or os.path.join(
        REPO, 'confirmation_frozen.json'))
    out = {'path': path, 'sha256': None, 'alpha': None,
           'min_seeds_for_confirmation': None, 'correction': None,
           'n_comparisons': 0, 'comparisons': {}, 'names': []}
    if not os.path.isfile(path):
        problems.append(f'tracked frozen manifest is missing: {path}')
        return out
    try:
        with open(path, 'rb') as fh:
            raw = fh.read()
        manifest = json.loads(raw.decode('utf-8'))
    except Exception as exc:
        problems.append(
            f'tracked frozen manifest is unreadable ({type(exc).__name__}: {exc})')
        return out
    out['sha256'] = hashlib.sha256(raw).hexdigest()
    if not isinstance(manifest, dict):
        problems.append(
            f'tracked frozen manifest is a {type(manifest).__name__}, not an object')
        return out
    stats = manifest.get('statistics')
    if not isinstance(stats, dict):
        problems.append('frozen manifest statistics is missing or not an object')
        stats = {}
    alpha = _pc_num(stats.get('alpha') if 'alpha' in stats else 0.05)
    if alpha is None or not (0.0 < alpha < 1.0):
        problems.append(
            f'frozen manifest statistics.alpha={stats.get("alpha")!r} is not in (0,1)')
    else:
        out['alpha'] = alpha
    if 'min_seeds_for_confirmation' in stats:
        n_min = _pc_int(stats.get('min_seeds_for_confirmation'))
        if n_min is None or n_min < 2:
            problems.append(
                'frozen manifest statistics.min_seeds_for_confirmation is not '
                'an exact integer >= 2')
        else:
            out['min_seeds_for_confirmation'] = n_min
    correction = _pc_text(stats.get('correction'))
    out['correction'] = correction
    comps = manifest.get('primary_comparisons')
    if not isinstance(comps, list) or not comps:
        problems.append('frozen manifest primary_comparisons is missing/empty')
        return out
    seen = set()
    for index, comp in enumerate(comps):
        where = f'frozen manifest primary_comparisons[{index}]'
        if not isinstance(comp, dict):
            problems.append(f'{where} is not an object')
            continue
        name = _pc_text(comp.get('name'))
        if name is None:
            problems.append(f'{where}.name is missing or not usable text')
            continue
        if name in seen:
            problems.append(f'{where}.name is a duplicate: {name!r}')
            continue
        seen.add(name)
        endpoint = comp.get('endpoint')
        length = _pc_endpoint_length(endpoint)
        if length is None:
            problems.append(
                f'{where}.endpoint={endpoint!r} is not a by_len.ppl@<integer> '
                'endpoint; the primary length cannot be checked')
        sign = _pc_sign(comp.get('expected_sign'))
        if sign is _PC_INVALID:
            problems.append(
                f'{where}.expected_sign={comp.get("expected_sign")!r} is not '
                '"positive", "negative" or null')
            sign = None
        flag = comp.get('mechanism_claim_allowed')
        if flag is not None and not isinstance(flag, (bool, np.bool_)):
            problems.append(
                f'{where}.mechanism_claim_allowed={flag!r} is not a boolean')
            flag = None
        arm_a = _pc_text(comp.get('a'))
        arm_b = _pc_text(comp.get('b'))
        if arm_a is None or arm_b is None:
            problems.append(f'{where} does not name both arms a/b')
        out['comparisons'][name] = {
            'name': name, 'a': arm_a, 'b': arm_b,
            'endpoint': endpoint, 'length': length,
            'expected_sign': sign,
            'mechanism_claim_allowed': (bool(flag) if flag is not None else None),
        }
        out['names'].append(name)
    out['n_comparisons'] = len(out['names'])
    family_size = manifest.get('binding', {}).get('family_size') \
        if isinstance(manifest.get('binding'), dict) else None
    if family_size is not None:
        fs = _pc_int(family_size)
        if fs is not None and fs != out['n_comparisons']:
            warnings.append(
                f'frozen manifest binding.family_size={family_size!r} does not '
                f'equal its {out["n_comparisons"]} primary comparisons')
    return out


def _pc_primary_cells(record):
    """``[(length_or_None, cell), ...]`` for cells labelled primary."""
    curve = record.get('by_length_curve') if isinstance(record, dict) else None
    out = []
    if not isinstance(curve, dict):
        return out
    for key, cell in curve.items():
        if isinstance(cell, dict) and cell.get('role') == 'primary':
            out.append((_pc_int(key), cell))
    return out


def _pc_compare_record(name, record, plan_comp, out, strict):
    """Check one artifact comparison against its frozen declaration.

    ``strict`` is true only when the artifact records the same manifest digest
    as the tracked file.  Plan-dependent differences are failures in strict
    mode and are downgraded to warnings when the artifact claims a different
    plan (the current file cannot prove what that other plan said).  Internal
    inconsistencies (headline p/mean vs the cell it calls primary, missing
    role, bad direction flag) are always failures.
    """

    def _plan_problem(message):
        (out['problems'] if strict else out['warnings']).append(message)

    row = {'name': name, 'status': 'ok', 'strict': bool(strict),
           'endpoint': None, 'expected_sign': plan_comp.get('expected_sign'),
           'mechanism_claim_allowed': plan_comp.get('mechanism_claim_allowed'),
           'problems': [], 'warnings': []}
    out['comparisons'][name] = row
    if not isinstance(record, dict):
        out['problems'].append(
            f'comparison {name!r}: record is a {type(record).__name__}, not an object')
        row['status'] = 'fail'
        return
    if 'omitted' in record:
        row['status'] = 'omitted'
        art_sign = _pc_sign(record.get('expected_sign'))
        if art_sign is _PC_INVALID:
            out['problems'].append(
                f'comparison {name!r}: omitted record has a malformed expected_sign')
        elif art_sign != plan_comp.get('expected_sign'):
            _plan_problem(
                f'comparison {name!r}: omitted expected_sign={art_sign!r} does '
                f'not match the frozen {plan_comp.get("expected_sign")!r}')
        if _pc_int(record.get('n')) not in (None, 0):
            out['problems'].append(
                f'comparison {name!r}: omitted record carries n={record.get("n")!r}')
        return

    # Expected direction: a frozen directional claim and the artifact's
    # declaration must agree before the stored `direction_ok` can be read as
    # "the pre-registered direction was observed".
    art_sign = _pc_sign(record.get('expected_sign'))
    if art_sign is _PC_INVALID:
        out['problems'].append(
            f'comparison {name!r}: artifact expected_sign is malformed')
        art_sign = None
    elif art_sign != plan_comp.get('expected_sign'):
        _plan_problem(
            f'comparison {name!r}: expected_sign={art_sign!r} does not match '
            f'the frozen {plan_comp.get("expected_sign")!r}')

    # Exactly one cell may claim the frozen primary endpoint.  Without this,
    # moving the label to an easier length leaves every stored p replayable.
    primaries = _pc_primary_cells(record)
    headline_p = _pc_num(record.get('p_exact_signflip'))
    headline_mean = _pc_num(record.get('mean'))
    if headline_p is None:
        out['problems'].append(
            f'comparison {name!r}: headline p_exact_signflip is missing or non-finite')
    if headline_mean is None:
        out['problems'].append(
            f'comparison {name!r}: headline mean is missing or non-finite')
    if len(primaries) != 1:
        out['problems'].append(
            f'comparison {name!r}: expected exactly one by-length cell with '
            f'role="primary", found {len(primaries)}')
    else:
        length, cell = primaries[0]
        row['endpoint'] = length
        if length is None:
            out['problems'].append(
                f'comparison {name!r}: the primary by-length key is not an integer')
        elif length != plan_comp.get('length'):
            _plan_problem(
                f'comparison {name!r}: primary endpoint L{length} does not '
                f'match the frozen endpoint L{plan_comp.get("length")}')
        if cell.get('descriptive_only') is not False:
            out['problems'].append(
                f'comparison {name!r}: the primary cell carries '
                f'descriptive_only={cell.get("descriptive_only")!r}')
        cell_p = _pc_num(cell.get('p_exact_signflip'))
        cell_mean = _pc_num(cell.get('mean'))
        if cell_p is None:
            out['problems'].append(
                f'comparison {name!r}: primary cell p_exact_signflip is missing '
                'or non-finite')
        elif headline_p is not None and abs(cell_p - headline_p) > 1e-12:
            out['problems'].append(
                f'comparison {name!r}: headline p={headline_p!r} differs from '
                f'the primary cell p={cell_p!r}')
        if cell_mean is None:
            out['problems'].append(
                f'comparison {name!r}: primary cell mean is missing or non-finite')
        elif headline_mean is not None and abs(cell_mean - headline_mean) > 1e-12:
            out['problems'].append(
                f'comparison {name!r}: headline mean={headline_mean!r} differs '
                f'from the primary cell mean={cell_mean!r}')
        # The delta vector behind the headline p is the one that must be
        # labelled primary.  `audit_confirmation_pvalues` checks this when the
        # role is present; do it here as well so a role/label edit cannot
        # detach the headline from the endpoint it claims.
        headline_vec = record.get('per_seed')
        cell_vec = cell.get('per_seed')
        if isinstance(headline_vec, list) and isinstance(cell_vec, list):
            if len(headline_vec) != len(cell_vec):
                out['problems'].append(
                    f'comparison {name!r}: headline and primary cell per_seed '
                    'vectors have different lengths')
            else:
                for index, (left, right) in enumerate(zip(headline_vec, cell_vec)):
                    a, b = _pc_num(left), _pc_num(right)
                    if a is None or b is None or abs(a - b) > 1e-12:
                        out['problems'].append(
                            f'comparison {name!r}: headline and primary cell '
                            f'per_seed[{index}] disagree')
                        break
        cell_n = _pc_int(cell.get('n'))
        headline_n = _pc_int(record.get('n_effective', record.get('n')))
        if cell_n is not None and headline_n is not None and cell_n != headline_n:
            out['problems'].append(
                f'comparison {name!r}: headline n={headline_n} differs from '
                f'the primary cell n={cell_n}')

    # Direction consistency is recomputed from the headline mean.  The stored
    # flag alone is not evidence: a reverse-significant result with a flipped
    # flag would otherwise replay as "supported".
    direction_ok = record.get('direction_ok')
    frozen_sign = plan_comp.get('expected_sign')
    if frozen_sign in ('positive', 'negative'):
        if not isinstance(direction_ok, (bool, np.bool_)):
            out['problems'].append(
                f'comparison {name!r}: frozen directional claim but '
                f'direction_ok={direction_ok!r} is not a boolean')
        elif headline_mean is not None:
            expected_ok = (headline_mean > 0.0) == (frozen_sign == 'positive')
            if bool(direction_ok) != bool(expected_ok):
                out['problems'].append(
                    f'comparison {name!r}: direction_ok={bool(direction_ok)} '
                    f'does not match sign(mean)={headline_mean!r} and the '
                    f'frozen expected_sign={frozen_sign!r}')
    elif direction_ok is not None:
        out['problems'].append(
            f'comparison {name!r}: non-directional comparison carries '
            f'direction_ok={direction_ok!r}')

    # The mechanism-claim flag is part of the pre-registration: it decides
    # whether a result may be written as mechanism evidence at all.
    if 'mechanism_claim_allowed' in record:
        art_flag = record.get('mechanism_claim_allowed')
        if art_flag is not None and not isinstance(art_flag, (bool, np.bool_)):
            out['problems'].append(
                f'comparison {name!r}: mechanism_claim_allowed={art_flag!r} '
                'is not a boolean')
        elif art_flag is not None and bool(art_flag) != bool(
                plan_comp.get('mechanism_claim_allowed')):
            _plan_problem(
                f'comparison {name!r}: mechanism_claim_allowed={art_flag!r} '
                f'does not match the frozen '
                f'{plan_comp.get("mechanism_claim_allowed")!r}')
    else:
        _plan_problem(
            f'comparison {name!r}: mechanism_claim_allowed is missing; the '
            'artifact does not record whether a mechanism claim was allowed')

    delta_definition = _pc_text(record.get('delta_definition'))
    if delta_definition is None:
        row['warnings'].append('delta_definition is missing')
    else:
        expected_prefix = f'{plan_comp.get("a")} - {plan_comp.get("b")}'
        if not delta_definition.startswith(expected_prefix):
            _plan_problem(
                f'comparison {name!r}: delta_definition={delta_definition!r} '
                f'does not start with the frozen {expected_prefix!r}')
    if row['problems'] or out['problems'] and not strict:
        row['status'] = 'fail'


def audit_plan_conformance(path=None, manifest_path=None):
    """Bind a confirmation artifact to the tracked frozen plan.

    This is the missing link between two already-existing replays:
    ``pvalues`` proves the stored p follows from the stored deltas, and
    ``verdicts`` proves the stored decision follows from the stored p/gates.
    Neither proves the artifact asked the *pre-registered* question.  Here the
    artifact's family, primary endpoint, expected sign, mechanism-claim flag,
    alpha and Holm slots are compared with ``confirmation_frozen.json``.

    Status: ``ok`` (all checked), ``warnings`` (no failure, but the artifact
    records a different manifest digest or a non-fatal legacy omission),
    ``fail`` (a present artifact contradicts the tracked plan or itself), or
    ``not_verifiable_locally`` (no artifact / no executed primary comparison).
    It recomputes no PPL, reads no test split and changes no frozen verdict.
    """
    out = {'kind': 'confirmation_plan_conformance',
           'schema': _PLAN_CONFORMANCE_SCHEMA, 'status': 'ok',
           'artifact': None, 'manifest': None, 'strict_manifest_match': None,
           'alpha': None, 'n_comparisons': 0, 'n_checked': 0,
           'n_omitted': 0, 'n_primary': 0, 'comparisons': {},
           'problems': [], 'warnings': [],
           'note': ('checks that a confirmation artifact still answers the '
                    'pre-registered family/endpoint/direction/claim question; '
                    'it does not recompute a p-value or alter a verdict')}
    plan = _pc_manifest(manifest_path, out['problems'], out['warnings'])
    out['manifest'] = {k: plan[k] for k in
                       ('path', 'sha256', 'alpha', 'correction',
                        'min_seeds_for_confirmation', 'n_comparisons')}
    if out['problems']:
        out['status'] = 'fail'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    art_path = os.path.abspath(path or os.path.join(
        REPO, 'analysis_v12', 'test_stats.json'))
    out['artifact'] = art_path
    if not os.path.isfile(art_path):
        out['status'] = 'not_verifiable_locally'
        out['reason'] = 'no final-test stats artifact on disk'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    try:
        with open(art_path, 'rb') as fh:
            raw = fh.read()
        data = json.loads(raw.decode('utf-8'))
    except Exception as exc:
        out['status'] = 'fail'
        out['problems'].append(
            f'{art_path} is not readable JSON ({type(exc).__name__}: {exc})')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    if not isinstance(data, dict):
        out['status'] = 'fail'
        out['problems'].append(
            f'{art_path} is a {type(data).__name__}, not a JSON object')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    ft = data.get('finaltest') if isinstance(data.get('finaltest'), dict) else data
    comps = ft.get('comparisons') if isinstance(ft, dict) else None
    if not isinstance(comps, dict) or not comps:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = 'the stats artifact carries no comparisons'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    out['n_comparisons'] = len(comps)

    artifact_manifest = ft.get('manifest') if isinstance(ft.get('manifest'), dict) else {}
    stored_sha = _pc_text(artifact_manifest.get('manifest_sha256'))
    strict = stored_sha is not None and stored_sha == plan['sha256']
    out['strict_manifest_match'] = bool(strict)
    if stored_sha is None:
        out['warnings'].append(
            'the artifact does not record manifest_sha256; plan-dependent '
            'differences are reported as warnings, not failures')
    elif not strict:
        out['warnings'].append(
            'the artifact records a different manifest digest '
            f'({stored_sha[:12]}… vs tracked {plan["sha256"][:12]}…); '
            'plan-dependent differences are reported as warnings')

    # Family: names and Holm slots are exact.  A deleted comparison must not
    # shrink the correction family.
    artifact_names = artifact_manifest.get('primary_comparison_names')
    if isinstance(artifact_names, list):
        _art_names = []
        for value in artifact_names:
            text = _pc_text(value)
            if text is None:
                out['problems'].append(
                    'artifact manifest.primary_comparison_names contains a '
                    'non-text entry')
                continue
            _art_names.append(text)
        if len(set(_art_names)) != len(_art_names):
            out['problems'].append(
                'artifact manifest.primary_comparison_names contains duplicates')
        if set(_art_names) != set(plan['names']):
            _plan_problem(
                'artifact manifest family does not equal the tracked plan: '
                f'missing={sorted(set(plan["names"]) - set(_art_names))} '
                f'extra={sorted(set(_art_names) - set(plan["names"]))}')
    else:
        _plan_problem(
            'artifact manifest.primary_comparison_names is missing; a deleted '
            'comparison cannot be ruled out')
    _extra = sorted(set(map(str, comps)) - set(plan['names']))
    _missing = sorted(set(plan['names']) - set(map(str, comps)))
    if _missing:
        _plan_problem(f'artifact is missing frozen comparisons: {_missing}')
    if _extra:
        _plan_problem(f'artifact carries undeclared comparisons: {_extra}')

    holm = ft.get('holm')
    if not isinstance(holm, dict):
        out['problems'].append('artifact holm block is missing or not an object')
        holm = {}
    else:
        _h_missing = sorted(set(plan['names']) - set(map(str, holm)))
        _h_extra = sorted(set(map(str, holm)) - set(plan['names']))
        if _h_missing:
            _plan_problem(f'artifact holm block is missing slots: {_h_missing}')
        if _h_extra:
            _plan_problem(f'artifact holm block has undeclared slots: {_h_extra}')

    # Alpha and the correction family are part of the pre-registered decision
    # rule; a silently changed alpha changes every reject/accept boundary.
    art_alpha = _pc_num(ft.get('alpha'))
    if art_alpha is None or not (0.0 < art_alpha < 1.0):
        out['problems'].append(
            f'artifact alpha={ft.get("alpha")!r} is missing or not in (0,1)')
    else:
        out['alpha'] = art_alpha
        if plan['alpha'] is not None and abs(art_alpha - plan['alpha']) > 1e-12:
            _plan_problem(
                f'artifact alpha={art_alpha!r} does not match the frozen '
                f'{plan["alpha"]!r}')
    art_stats = artifact_manifest.get('statistics')
    if isinstance(art_stats, dict):
        art_corr = _pc_text(art_stats.get('correction'))
        if art_corr is not None and plan['correction'] is not None \
                and art_corr != plan['correction']:
            _plan_problem(
                f'artifact statistics.correction={art_corr!r} does not match '
                f'the frozen {plan["correction"]!r}')
        if 'min_seeds_for_confirmation' in art_stats \
                and plan['min_seeds_for_confirmation'] is not None:
            art_min = _pc_int(art_stats.get('min_seeds_for_confirmation'))
            if art_min != plan['min_seeds_for_confirmation']:
                _plan_problem(
                    'artifact statistics.min_seeds_for_confirmation='
                    f'{art_stats.get("min_seeds_for_confirmation")!r} does not '
                    f'match the frozen {plan["min_seeds_for_confirmation"]!r}')

    for name in plan['names']:
        record = comps.get(name)
        if record is None:
            _plan_problem(f'artifact is missing comparison {name!r}')
            continue
        _pc_compare_record(name, record, plan['comparisons'][name], out, strict)
        row = out['comparisons'].get(name) or {}
        if row.get('status') == 'omitted':
            out['n_omitted'] += 1
        elif row.get('endpoint') is not None:
            out['n_primary'] += 1
        if row.get('status') == 'fail':
            row['status'] = 'fail'
        elif row.get('warnings'):
            row['status'] = 'warnings'
    out['n_checked'] = len(out['comparisons'])

    if out['problems']:
        out['status'] = 'fail'
    elif out['n_primary'] == 0:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = 'the artifact carries no executed primary comparison'
    elif out['warnings']:
        out['status'] = 'warnings'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out
