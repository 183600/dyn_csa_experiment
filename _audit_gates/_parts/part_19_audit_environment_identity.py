# Strict declared-environment identity (zero GPU).
#
# `audit_dependencies` checks that the *current* process satisfies
# requirements.lock.  It does not check the environment recorded inside local
# run artifacts, and the legacy `exec_fp` only covers six direct packages.
# This gate answers the complementary question: do the panels on disk carry
# the same full-stack `env_fp_v2` digest as the machine reading them?
#
# A difference across machines is not a false result: a GPU evidence archive
# read on a CPU audit box should differ.  The gate therefore reports a
# `warning`, not a `fail`, unless the local stack itself violates the lock
# (which `audit_dependencies` already turns into a failure).  What it refuses
# to do is call a coarse six-package match "the same environment".


def _environment_records_from_doc(doc):
    """Yield ``(key, record)`` pairs that can carry an environment digest."""
    if not isinstance(doc, dict):
        return []
    out = []
    cells = doc.get('cells')
    if isinstance(cells, dict):
        for key, rec in cells.items():
            if isinstance(rec, dict) and not rec.get('synthesized'):
                out.append((str(key), rec))
    for key, rec in doc.items():
        # summary.json cells are keyed `variant::seedN`; non-cell aggregates
        # are ignored exactly like the reproduction comparator does, and a
        # synthesized record is display-only, not an execution to identify.
        if ('::seed' in str(key) and isinstance(rec, dict)
                and not rec.get('synthesized')):
            out.append((str(key), rec))
    return out


def _environment_fp_of(record):
    value = record.get('env_fp_v2')
    if value is None and isinstance(record.get('hyperparams'), dict):
        value = record['hyperparams'].get('env_fp_v2')
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def audit_environment_identity(results_dirs=None):
    """Compare local full-stack environment digests with recorded panels.

    ``env_fp_v2`` binds every lock-pinned distribution, the lock bytes, the
    interpreter/platform contract and the OS/kernel/libc/driver block.  The
    legacy ``exec_fp`` binds only six direct packages, so a transitive pin
    change could pass the old comparison.  This gate reports:

    * ``local.env_fp_v2`` and the local dependency-contract status;
    * every local panel cell that carries a digest, grouped by panel;
    * ``match`` / ``different_from_local`` / ``missing_v2`` counts;
    * ``status='ok'`` only when at least one digest exists and every one
      matches the local stack; a cross-machine difference is a ``warning``
      (it is not evidence that either run is wrong); no digest at all is
      ``not_verifiable_locally``.

    It never starts a run, reads the test split, or changes a p-value.
    """
    L = _import_exp_lib()
    try:
        local_env = L.environment_snapshot()
        local_fp = L.environment_fingerprint_v2(local_env)
    except Exception as e:
        local_env, local_fp = {}, None
        local_error = f'{type(e).__name__}: {e}'
    else:
        local_error = None
    contract = local_env.get('dependency_contract') if isinstance(local_env, dict) else None
    if not isinstance(contract, dict):
        contract = {}
    interpreter = contract.get('interpreter') if isinstance(contract.get('interpreter'), dict) else {}
    out = {
        'kind': 'environment_identity_audit',
        'status': 'ok',
        'local': {
            'env_fp_v2': local_fp,
            'error': local_error,
            'lock_path': contract.get('lock_path'),
            'lock_sha256': contract.get('lock_sha256'),
            'contract_status': contract.get('status'),
            'n_pinned': contract.get('n_pinned'),
            'n_missing_pins': len(contract.get('missing') or []),
            'n_mismatched_pins': len(contract.get('mismatches') or []),
            'interpreter_status': interpreter.get('status'),
            'interpreter_declared': interpreter.get('declared') or {},
            'interpreter_actual': interpreter.get('actual') or {},
            'os_identity': (local_env.get('os_identity') if isinstance(local_env, dict) else None),
        },
        'n_panels': 0,
        'n_records': 0,
        'n_with_v2': 0,
        'n_match': 0,
        'n_different_from_local': 0,
        'n_missing_v2': 0,
        'panels': {},
        'problems': [],
        'warnings': [],
    }
    if local_fp is None:
        out['status'] = 'not_verifiable_locally'
        out['note'] = ('could not compute the local environment fingerprint; '
                       'the installed stack cannot be identified')
        print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
        return out

    if results_dirs is None:
        dirs = [os.path.join(REPO, name) for name in _iter_result_dirs()]
    elif isinstance(results_dirs, (str, bytes, os.PathLike)):
        dirs = [_audit_artifact(results_dirs)]
    else:
        dirs = [_audit_artifact(d) for d in results_dirs]
    # A directory is part of one panel.  `run_manifest.json` and
    # `summary.json` usually describe the same cells, so merge by cell key
    # instead of counting a cell twice; a digest present on either copy wins
    # over a legacy copy, and two different non-null digests for one cell are
    # surfaced as a corruption/merge conflict rather than silently picked.
    seen = set()
    for panel_dir in dirs:
        panel_dir = os.path.abspath(panel_dir)
        if panel_dir in seen or not os.path.isdir(panel_dir):
            continue
        seen.add(panel_dir)
        panel_name = os.path.basename(panel_dir.rstrip(os.sep)) or panel_dir
        by_key = {}
        conflicts = []
        for filename in ('run_manifest.json', 'summary.json'):
            path = os.path.join(panel_dir, filename)
            doc = _load(path)
            if not isinstance(doc, dict):
                continue
            for key, rec in _environment_records_from_doc(doc):
                fp = _environment_fp_of(rec)
                if key not in by_key:
                    by_key[key] = fp
                    continue
                old = by_key[key]
                if old is None:
                    by_key[key] = fp
                elif fp is not None and fp != old:
                    conflicts.append({'record': key, 'digests': sorted({old, fp}),
                                      'source': filename})
        records = sorted(by_key.items())
        if not records:
            continue
        out['n_panels'] += 1
        panel = {'n_records': 0, 'n_with_v2': 0, 'n_match': 0,
                 'n_different_from_local': 0, 'n_missing_v2': 0,
                 'recorded': {}, 'different': [], 'missing_v2': [],
                 'conflicts': conflicts[:20]}
        for key, fp in records:
            panel['n_records'] += 1
            out['n_records'] += 1
            if fp is None:
                panel['n_missing_v2'] += 1
                out['n_missing_v2'] += 1
                panel['missing_v2'].append(key)
                continue
            panel['n_with_v2'] += 1
            out['n_with_v2'] += 1
            panel['recorded'][fp] = panel['recorded'].get(fp, 0) + 1
            if fp == local_fp:
                panel['n_match'] += 1
                out['n_match'] += 1
            else:
                panel['n_different_from_local'] += 1
                out['n_different_from_local'] += 1
                panel['different'].append({'record': key, 'env_fp_v2': fp})
        panel['different'] = panel['different'][:20]
        panel['missing_v2'] = panel['missing_v2'][:20]
        out['panels'][panel_name] = panel

    n_conflicts = sum(len(p.get('conflicts') or [])
                      for p in out['panels'].values())
    if n_conflicts:
        out['warnings'].append(
            f'{n_conflicts} cell(s) carry two different non-null env_fp_v2 '
            'digests across run_manifest.json/summary.json; the copies were '
            'not produced by one environment and the conflict is reported, '
            'not silently resolved')
    if out['n_records'] == 0:
        out['status'] = 'not_verifiable_locally'
        out['note'] = ('no local results_*/summary.json or run_manifest.json; '
                       'there is no recorded environment to compare')
    elif out['n_with_v2'] == 0:
        out['status'] = 'not_verifiable_locally'
        out['warnings'].append(
            f'{out["n_missing_v2"]} local record(s) predate env_fp_v2; only '
            'the coarse six-package exec_fp is available, so full-stack '
            'identity is unverified (not a match)')
        out['note'] = 'all local records are legacy (no env_fp_v2)'
    elif out['n_different_from_local']:
        out['status'] = 'warning'
        out['warnings'].append(
            f'{out["n_different_from_local"]}/{out["n_with_v2"]} recorded '
            'cell(s) were produced in a different declared environment than '
            'this audit process; this is expected for a CPU audit of GPU '
            'evidence and is not by itself a reproduction failure')
        out['note'] = ('some recorded environments differ from the local '
                       'stack; compare those panels on their original machine '
                       'before claiming same-environment reproduction')
    else:
        out['status'] = 'ok'
        out['note'] = ('every recorded cell with env_fp_v2 matches the local '
                       'lock-pinned stack, interpreter and OS/driver identity')
    if contract.get('status') in ('mismatch', 'invalid', 'missing_lock'):
        out['warnings'].append(
            f'local dependency contract is {contract.get("status")!r}; '
            'the local digest describes an environment that does not satisfy '
            'requirements.lock')
    print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
    return out
