# ---------------------------------------------------------------------------
# 99. Reproducible report timestamps and "same evidence, same report" replay
#
# Two independent gaps remained after the content-addressed analysis work:
#
#   1. `stamped_analysis` wrote a wall-clock `created_at` unconditionally, and
#      aggregate builders wrote a wall-clock `generated_at` INTO the hashed body.
#      Two identical audit runs therefore produced different `content_sha256`
#      values.  The artifact could detect an edit, but it could not recognise a
#      rebuild as the same result.  `content_schema` 2 makes `generated_at`
#      volatile and honours SOURCE_DATE_EPOCH for the stamp, while legacy
#      artifacts keep their historical hash.
#
#   2. `verify_stamped_analysis` proves that a body matches its own digest and
#      that recorded inputs still have the recorded bytes.  It does not answer
#      the reproduction question a reviewer actually asks: "I rebuilt the
#      report from the same evidence; are the numbers the same?"  The
#      comparison below separates a changed result body from changed inputs,
#      changed code and a changed execution environment, and reports the exact
#      JSON pointers that differ.
# ---------------------------------------------------------------------------

_MAX_SOURCE_DATE_EPOCH = 253402300799  # 9999-12-31T23:59:59Z


def source_date_epoch(value=None):
    """Parse SOURCE_DATE_EPOCH (or an explicit value) as a non-negative int.

    This mirrors ``archive_release._source_date_epoch`` so the analysis stamp
    and the release archive interpret the reproducible-builds convention the
    same way.  The value is a timestamp, not a measurement: floats, signs,
    dates and non-numeric strings are refused instead of truncated.
    """
    raw = os.environ.get('SOURCE_DATE_EPOCH') if value is None else value
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValueError('SOURCE_DATE_EPOCH must be a non-negative integer, not a boolean')
    text = str(raw).strip()
    if not text:
        return None
    if not re.fullmatch(r'[0-9]+', text):
        raise ValueError(
            f'SOURCE_DATE_EPOCH must be a non-negative integer, got {raw!r}')
    epoch = int(text)
    if epoch > _MAX_SOURCE_DATE_EPOCH:
        raise ValueError(
            f'SOURCE_DATE_EPOCH {epoch} is later than 9999-12-31T23:59:59Z; '
            'refusing a value that cannot be represented portably')
    return epoch


def reproducible_timestamp(value=None, fmt='%Y-%m-%dT%H:%M:%S'):
    """Wall-clock time, or the fixed UTC time declared by SOURCE_DATE_EPOCH.

    The output format deliberately matches the rest of the analysis records
    (naive ISO-like UTC) rather than the archive's trailing ``Z``.  A caller
    that needs the archive spelling should pass its own ``fmt``.
    """
    epoch = source_date_epoch(value)
    if epoch is None:
        return time.strftime(fmt)
    return time.strftime(fmt, time.gmtime(epoch))


def _report_arg_label(value):
    if isinstance(value, (str, bytes, os.PathLike)):
        return os.path.abspath(os.fsdecode(os.fspath(value)))
    return '<object>'


def _load_report_arg(value):
    """Load a report from a JSON path or accept an already-parsed dict."""
    if isinstance(value, dict):
        return _json_clean(_jsonable(value))
    if isinstance(value, (str, bytes, os.PathLike)):
        path = os.path.abspath(os.fsdecode(os.fspath(value)))
        with open(path, encoding='utf-8') as fh:
            obj = json.load(fh)
        if not isinstance(obj, dict):
            raise ValueError(f'{path} is a {type(obj).__name__}, not a JSON object')
        return obj
    raise TypeError(f'report must be a path or dict, got {type(value).__name__}')


def _pointer_parts(pointer):
    """Decode a JSON pointer (RFC 6901) into dict keys / list indexes.

    The empty string is the whole document.  A single slash is the pointer to
    the object member whose key is the empty string (``{"": ...}``), not the
    document root: conflating the two made it impossible to ignore a report
    block stored under the empty key.  The leading slash is optional only for
    the convenience of callers that pass ``experiment_ledger`` instead of
    ``/experiment_ledger``.
    """
    if pointer is None or pointer == '':
        return []
    if not isinstance(pointer, str):
        raise TypeError(f'JSON pointer must be a string, got {type(pointer).__name__}')
    text = pointer[1:] if pointer.startswith('/') else pointer
    return [p.replace('~1', '/').replace('~0', '~') for p in text.split('/')]


def _pointer_list_index(part):
    """Canonical non-negative list index for a JSON pointer segment.

    RFC 6901 array indices are decimal digits with no sign and no leading
    zero.  Python's ``int()`` would otherwise accept ``-1`` (silently removing
    the last element) and ``01``; a malformed pointer must be a no-op instead
    of deleting a different element than the caller named.
    """
    if not isinstance(part, str) or not part:
        return None
    if (not part.isascii() or not part.isdigit()
            or (len(part) > 1 and part[0] == '0')):
        return None
    try:
        return int(part)
    except (TypeError, ValueError, OverflowError):
        return None


def _pointer_escape(part):
    return str(part).replace('~', '~0').replace('/', '~1')


def _drop_pointer(obj, pointer):
    """Return a copy of ``obj`` without the value at ``pointer``.

    A missing path is a no-op: an older report that predates the block must
    not make the comparison crash.  Lists are copied so the caller's record is
    never mutated.
    """
    parts = _pointer_parts(pointer)
    if not parts:
        return obj
    root = copy.deepcopy(obj)
    cur = root
    for part in parts[:-1]:
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list):
            index = _pointer_list_index(part)
            if index is None or index >= len(cur):
                return root
            cur = cur[index]
        else:
            return root
    last = parts[-1]
    if isinstance(cur, dict):
        cur.pop(last, None)
    elif isinstance(cur, list):
        index = _pointer_list_index(last)
        if index is not None and index < len(cur):
            cur.pop(index)
    return root


def _diff_json(a, b, path='', out=None, limit=200):
    """Collect the first ``limit`` structural differences as JSON pointers."""
    if out is None:
        out = []
    if len(out) >= limit:
        return out
    if type(a) is not type(b):
        out.append({'pointer': path or '/', 'kind': 'type',
                    'original': type(a).__name__, 'rerun': type(b).__name__})
        return out
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if len(out) >= limit:
                break
            child = f'{path}/{_pointer_escape(key)}'
            if key not in a:
                out.append({'pointer': child, 'kind': 'only_rerun',
                            'original': None, 'rerun': b[key]})
            elif key not in b:
                out.append({'pointer': child, 'kind': 'only_original',
                            'original': a[key], 'rerun': None})
            else:
                _diff_json(a[key], b[key], child, out, limit)
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            out.append({'pointer': path or '/', 'kind': 'length',
                        'original': len(a), 'rerun': len(b)})
        for i, (va, vb) in enumerate(zip(a, b)):
            if len(out) >= limit:
                break
            _diff_json(va, vb, f'{path}/{i}', out, limit)
        return out
    if a != b:
        out.append({'pointer': path or '/', 'kind': 'value',
                    'original': a, 'rerun': b})
    return out


def _report_body_for_diff(record):
    """Result body without volatile stamps or the separate input binding.

    ``inputs_sha256`` is deliberately inside ``content_sha256`` (so it cannot
    be stripped), but it is not a result: a changed evidence file must be
    reported as ``inputs_changed``, not as a changed statistic.  It therefore
    has its own dimension below.

    ``content_schema`` names the digest/volatile-field convention, not a
    measured result.  It is compared separately so a schema migration of an
    otherwise identical body is reported as ``schema_changed`` instead of
    being indistinguishable from a changed statistic.
    """
    return {k: v for k, v in record.items()
            if k not in _ANALYSIS_VOLATILE_KEYS_V2
            and k not in ('inputs_sha256', 'content_schema')}


def _report_schema(record):
    """Declared content schema, normalised to the legacy convention.

    An absent/older schema deliberately reads as legacy (1) because
    ``_analysis_volatile_keys`` hashes it with the legacy volatile set.  A
    boolean or non-integral value is not a schema declaration; keep it as
    ``None`` so the caller follows that same legacy digest convention instead
    of inventing a version.
    """
    schema = record.get('content_schema') if isinstance(record, dict) else None
    if schema is None:
        return None
    if isinstance(schema, bool) or not isinstance(schema, numbers.Integral):
        return None
    return int(schema)


def _effective_schema(record):
    """Schema generation whose digest/volatile-field semantics apply."""
    schema = _report_schema(record)
    if schema is None or schema < ANALYSIS_CONTENT_SCHEMA:
        return ANALYSIS_CONTENT_SCHEMA - 1
    return schema


def _fingerprint_object(obj):
    text = json.dumps(_json_clean(_jsonable(obj)),
                      sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False, default=float)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _report_identity(record):
    """Extract the non-body provenance dimensions from one stamped record."""
    prov = record.get('provenance') if isinstance(record.get('provenance'), dict) else {}
    code = prov.get('code') if isinstance(prov.get('code'), dict) else {}
    tree = code.get('source_tree') if isinstance(code.get('source_tree'), dict) else {}
    env = prov.get('environment') if isinstance(prov.get('environment'), dict) else {}
    contract = env.get('dependency_contract') if isinstance(env.get('dependency_contract'), dict) else {}
    gpus = env.get('gpu_names')
    if not isinstance(gpus, list):
        gpus = None
    else:
        gpus = [str(g) for g in gpus]
    return {
        'source_sha256': tree.get('sha256'),
        'code_semantics': code.get('code_semantics'),
        'dependency_lock_sha256': contract.get('lock_sha256'),
        # The strict environment generations recorded by the snapshot: v2
        # covers the lock closure/OS identity, v3 adds the numerical backend
        # and unpinned runtime distributions.  A report rebuilt on a
        # different BLAS build is not the same analysis even when torch and
        # the lock agree.
        'env_fp_v2': env.get('env_fp_v2'),
        'env_fp_v3': env.get('env_fp_v3'),
        'torch': env.get('torch'),
        'cuda_version': env.get('cuda_version'),
        'determinism': env.get('determinism'),
        'gpu_names': gpus,
    }


def _dimension_status(a, b, *, required=False):
    """Match/mismatch/unverified for two optional identity values."""
    if a is None and b is None:
        return 'unverified' if required else 'match'
    if a is None or b is None:
        return 'unverified'
    return 'match' if a == b else 'mismatch'


def analysis_report_diff(original, rerun, *, ignore=(), max_differences=200):
    """Compare two stamped analysis reports as reproducible rebuilds.

    The result separates:

      * ``body`` — the result/verdict/curve body, excluding volatile stamps;
      * ``inputs`` — the recorded input-byte binding;
      * ``code`` — source-tree digest and semantic stamp;
      * ``environment`` — lock closure, v2/v3 environment fingerprints,
        torch/CUDA and determinism class.

    A bare "the JSON files differ" cannot tell a corrected statistic from a
    new timestamp, so the verdict is one of ``reproduced``, ``body_changed``,
    ``inputs_changed``, ``provenance_changed``, ``schema_changed`` or
    ``unverified``.  ``ignore`` accepts JSON pointers (for example
    ``/experiment_ledger``) for append-only audit traces whose own growth is
    not a change in the measured result.
    """
    if (isinstance(max_differences, bool)
            or not isinstance(max_differences, numbers.Integral)
            or int(max_differences) < 1):
        raise ValueError('max_differences must be a positive integer')
    max_differences = int(max_differences)
    a = _load_report_arg(original)
    b = _load_report_arg(rerun)
    ignore = [ignore] if isinstance(ignore, str) else list(ignore or [])
    problems = []
    dimensions = {}

    # Body identity: compare the recomputed schema-aware digest, then locate
    # the actual differing pointers for the reader.  `generated_at` is always
    # excluded here even for a legacy record, because a wall-clock field is
    # not a result difference.
    try:
        hash_a = analysis_content_sha256(a)
        hash_b = analysis_content_sha256(b)
    except (TypeError, ValueError, OverflowError) as exc:
        return {
            'kind': 'analysis_report_diff', 'schema': 1,
            'verdict': 'invalid', 'reproduced': False,
            'original': _report_arg_label(original),
            'rerun': _report_arg_label(rerun),
            'problems': [f'body cannot be canonicalised: {type(exc).__name__}: {exc}'],
            'warnings': [], 'dimensions': {},
            'n_differences': 0, 'differences': [], 'truncated': False,
        }

    stored = {}
    for tag, rec, digest in (('original', a, hash_a), ('rerun', b, hash_b)):
        recorded = rec.get('content_sha256')
        if recorded is None:
            stored[tag] = 'missing'
        elif not isinstance(recorded, str) or recorded.lower() != digest:
            stored[tag] = 'mismatch'
            problems.append(
                f'{tag} content_sha256 does not match its own body; it was '
                'edited after it was written or used a different schema')
        else:
            stored[tag] = 'match'

    body_a = _report_body_for_diff(a)
    body_b = _report_body_for_diff(b)
    for pointer in ignore:
        body_a = _drop_pointer(body_a, pointer)
        body_b = _drop_pointer(body_b, pointer)
    # Request one pointer past the display limit so "exactly N differences"
    # is not mislabeled as truncated.  `_diff_json` stops as soon as it has
    # enough, so this costs at most one extra comparison.
    _probe = _diff_json(body_a, body_b, limit=max_differences + 1)
    truncated = len(_probe) > max_differences
    differences = _probe[:max_differences]
    n_differences = len(differences)

    body_fp_a = _fingerprint_object(body_a)
    body_fp_b = _fingerprint_object(body_b)
    schema_a = _report_schema(a)
    schema_b = _report_schema(b)
    schema_changed = _effective_schema(a) != _effective_schema(b)
    if body_fp_a == body_fp_b and not differences:
        # A schema migration changes the digest convention, not the numbers.
        body_status = 'schema_changed' if schema_changed else 'match'
    elif not differences and body_fp_a != body_fp_b:
        # The only observable difference is in a hashed-but-not-displayed
        # field; this is a schema migration, not a changed number.
        body_status = 'schema_changed'
    else:
        body_status = 'mismatch'
    dimensions['body'] = {
        'status': body_status,
        'original_sha256': hash_a,
        'rerun_sha256': hash_b,
        'original_schema': schema_a,
        'rerun_schema': schema_b,
        'n_differences': n_differences,
    }

    inputs_a = a.get('inputs_sha256')
    inputs_b = b.get('inputs_sha256')
    if inputs_a is None and inputs_b is None:
        inputs_status = 'unverified'
    elif inputs_a is None or inputs_b is None:
        inputs_status = 'unverified'
    else:
        inputs_status = 'match' if inputs_a == inputs_b else 'mismatch'
    dimensions['inputs'] = {
        'status': inputs_status,
        'original_sha256': inputs_a,
        'rerun_sha256': inputs_b,
    }

    ident_a = _report_identity(a)
    ident_b = _report_identity(b)
    code_status = _dimension_status(ident_a['source_sha256'],
                                    ident_b['source_sha256'], required=True)
    semantics_status = _dimension_status(ident_a['code_semantics'],
                                         ident_b['code_semantics'])
    if 'mismatch' in (code_status, semantics_status):
        code_verdict = 'mismatch'
    elif 'unverified' in (code_status, semantics_status):
        code_verdict = 'unverified'
    else:
        code_verdict = 'match'
    dimensions['code'] = {
        'status': code_verdict,
        'source_sha256': code_status,
        'code_semantics': semantics_status,
    }

    env_fields = {}
    for key in ('dependency_lock_sha256', 'env_fp_v2', 'env_fp_v3', 'torch',
                'cuda_version', 'determinism', 'gpu_names'):
        env_fields[key] = _dimension_status(ident_a[key], ident_b[key])
    if 'mismatch' in env_fields.values():
        env_verdict = 'mismatch'
    elif 'unverified' in env_fields.values():
        env_verdict = 'unverified'
    else:
        env_verdict = 'match'
    dimensions['environment'] = {'status': env_verdict, 'fields': env_fields}

    if 'mismatch' in stored.values():
        # One side's stored digest does not match its own body.  That is a
        # tamper/integrity failure, not a legitimate result difference.
        verdict = 'invalid'
    elif body_status == 'mismatch':
        verdict = 'body_changed'
    elif body_status == 'schema_changed':
        verdict = 'schema_changed'
    elif inputs_status == 'mismatch':
        verdict = 'inputs_changed'
    elif code_verdict == 'mismatch' or env_verdict == 'mismatch':
        verdict = 'provenance_changed'
    elif 'unverified' in (inputs_status, code_verdict, env_verdict):
        verdict = 'unverified'
    else:
        verdict = 'reproduced'

    warnings = []
    if body_status == 'schema_changed':
        warnings.append(
            'bodies are identical but content_schema/digest semantics differ '
            f'(original={dimensions["body"]["original_schema"]!r}, '
            f'rerun={dimensions["body"]["rerun_schema"]!r}); the rebuild used '
            'a newer analysis schema')
    if inputs_status == 'unverified':
        warnings.append(
            'one or both reports do not record inputs_sha256; the same body '
            'cannot be tied to the same evidence bytes')
    if verdict == 'reproduced' and stored['original'] == 'missing':
        warnings.append('the original report predates content_sha256; only the '
                        'recomputed bodies were compared')

    return {
        'kind': 'analysis_report_diff', 'schema': 1,
        'verdict': verdict, 'reproduced': verdict == 'reproduced',
        'original': _report_arg_label(original),
        'rerun': _report_arg_label(rerun),
        'dimensions': dimensions,
        'stored_integrity': stored,
        'ignore': ignore,
        'n_differences': n_differences,
        'differences': differences,
        'truncated': bool(truncated),
        'problems': problems,
        'warnings': warnings,
        'note': ('reproduced means the result body and input-byte binding '
                 'match; a changed body is a changed result, while a changed '
                 'source/environment is a provenance mismatch even when the '
                 'numbers happen to agree'),
    }
