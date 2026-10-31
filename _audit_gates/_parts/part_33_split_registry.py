# ---------------------------------------------------------------------------
# 128. Executed split modules must be auditable as registered, not merely
#      present on disk.
#
# The library and drivers are assembled by executing `_parts/*.py` in `_PARTS`
# order.  A fragment can be present but unregistered (dead source) or the list
# can be reordered while every file hash stays the same.  The registry digest
# is derived from the ordered declaration, not the directory listing, so this
# gate is the place where that difference becomes visible.  It is a
# provenance/integrity check: it produces no p-value and changes no verdict.
# ---------------------------------------------------------------------------

_SPLIT_REGISTRY_MAX_FILES = 500


def _split_registry_find(value, path='', out=None, depth=0):
    """Collect recorded split-registry blocks from an artifact tree."""
    if out is None:
        out = []
    if depth > 10 or len(out) >= 200:
        return out
    if isinstance(value, dict):
        source_tree = value.get('source_tree')
        if isinstance(source_tree, dict) and 'split_registry' in source_tree:
            out.append({'pointer': path or '/',
                        'registry': source_tree.get('split_registry')})
        for key in sorted(value):
            child = f'{path}/{key}' if path else f'/{key}'
            _split_registry_find(value[key], child, out, depth + 1)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _split_registry_find(item, f'{path}/{i}', out, depth + 1)
    return out


def _split_registry_artifact_paths(panel_dirs):
    paths = []
    seen = set()

    def _add(path):
        path = os.path.abspath(path)
        if path not in seen:
            seen.add(path)
            paths.append(path)

    for item in panel_dirs:
        item = os.path.abspath(str(item))
        if os.path.isdir(item):
            preferred = [os.path.join(item, n) for n in
                         ('summary.json', 'plan_snapshot.json',
                          'design_protocol.json')]
            for path in preferred:
                if os.path.isfile(path):
                    _add(path)
            try:
                names = sorted(os.listdir(item))
            except OSError:
                names = []
            for name in names:
                path = os.path.join(item, name)
                if (name.endswith('.json') and os.path.isfile(path)
                        and path not in seen and len(paths) < _SPLIT_REGISTRY_MAX_FILES):
                    _add(path)
        else:
            _add(item)
    return paths


def audit_split_registry(panel_dirs=None):
    """Verify that the executed split-package registration is explicit.

    Local part (always): every `_parts/*.py` fragment is registered exactly
    once, every registered fragment exists, the digests and order recompute,
    and each package declares a resolvable `historical_path`.

    Recorded part (optional): scan JSON artifacts for
    `source_tree.split_registry` and compare their digest/status with the
    local tree.  A supplied artifact with no registry is `not_verifiable`, not
    a green match: absence of the field cannot attest the executed order.

    Status is `fail` for an invalid local registration, an unreadable/malformed
    supplied artifact, or a digest mismatch; `not_verifiable` when supplied
    artifacts contain no registry at all.  The gate never reads a test split,
    starts training, produces a p-value, or changes Holm/verified/confirmed.
    """
    L = _import_exp_lib()
    reg = L.split_registry_manifest()
    local_status = reg.get('status')
    problems = list(reg.get('problems') or [])
    result = {
        'kind': 'split_registry_audit',
        'status': 'ok' if local_status == 'ok' else 'fail',
        'local': {
            'sha256': reg.get('sha256'),
            'n_packages': reg.get('n_packages'),
            'n_parts': reg.get('n_parts'),
            'status': local_status,
            'packages': {name: {'n_parts': pkg.get('n_parts'),
                                'sha256': pkg.get('sha256'),
                                'historical_path': pkg.get('historical_path'),
                                'problems': list(pkg.get('problems') or [])}
                         for name, pkg in sorted((reg.get('packages') or {}).items())},
        },
        'problems': problems,
        'note': ('the registry binds the ordered _PARTS membership that the '
                 'loader actually executes; the plain source-tree digest only '
                 'binds the bytes present on disk'),
    }
    if panel_dirs is None:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    if isinstance(panel_dirs, (str, bytes, os.PathLike)):
        panel_dirs = [panel_dirs]
    paths = _split_registry_artifact_paths(list(panel_dirs))
    if not paths:
        result['status'] = 'not_verifiable'
        result['note'] += '; no local artifact path was supplied'
        return result
    n_records = n_match = n_mismatch = n_absent = 0
    panels = []
    for path in paths:
        try:
            with open(path, encoding='utf-8') as fh:
                obj = json.load(fh)
        except Exception as e:
            problems.append(
                f'{os.path.relpath(path, REPO)}: unreadable JSON '
                f'({type(e).__name__}: {e})')
            panels.append({'path': os.path.relpath(path, REPO),
                           'status': 'fail'})
            continue
        records = _split_registry_find(obj)
        if not records:
            n_absent += 1
            panels.append({'path': os.path.relpath(path, REPO),
                           'status': 'not_verifiable',
                           'detail': 'no source_tree.split_registry block'})
            continue
        for rec in records:
            n_records += 1
            got = rec['registry']
            if not isinstance(got, dict):
                n_mismatch += 1
                problems.append(
                    f'{os.path.relpath(path, REPO)}{rec["pointer"]}: '
                    'split_registry is not an object')
                continue
            got_sha = got.get('sha256')
            got_status = got.get('status')
            if got_sha == reg.get('sha256') and got_status == reg.get('status'):
                n_match += 1
            else:
                n_mismatch += 1
                problems.append(
                    f'{os.path.relpath(path, REPO)}{rec["pointer"]}: '
                    f'split registry {str(got_sha)[:12]}… ({got_status!r}) != '
                    f'local {str(reg.get("sha256"))[:12]}… ({reg.get("status")!r})')
        panels.append({'path': os.path.relpath(path, REPO),
                       'status': 'match' if n_mismatch == 0 else 'mismatch',
                       'n_records': len(records)})
    result.update({'n_artifacts': len(paths), 'n_records': n_records,
                   'n_match': n_match, 'n_mismatch': n_mismatch,
                   'n_absent': n_absent, 'artifacts': panels})
    if problems:
        result['status'] = 'fail'
    elif n_records == 0:
        result['status'] = 'not_verifiable'
    else:
        result['status'] = 'ok'
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
