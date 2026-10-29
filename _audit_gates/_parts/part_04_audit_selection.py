

def audit_selection(path='analysis_mechanism/selection_attribution.json'):
    """Gate for the DIRECT mechanism evidence.

    A claim that "the learned selector finds the relevant blocks" needs an
    artifact comparing the selected blocks with a dependency oracle, per
    query.  This audit defines the contract and validates one when it exists:

        { "cells": [ { "arm": ..., "eval_len": 4096, "k": 32, "n_blocks": 1024,
                       "n_queries": ..., "recall_at_k": ..., "mass_covered": ...,
                       "random_baseline": ...,
                       "per_seed": {"0": {...}, "1": {...}} } ] }

    `recall_at_k` is the fraction of the oracle's top-mass blocks captured by
    the learned top-k; `mass_covered` is the fraction of the oracle attention
    mass inside the selected blocks; `random_baseline` is k/n_blocks (or the
    matched-length random-cut baseline).  Without this artifact the repository
    has NO selection-level mechanism evidence, and wording such as "the
    selector found the relevant information" is unsupported.
    """
    import numpy as np
    out = {'status': 'not_verifiable_locally', 'path': path,
           'required_schema': ['arm', 'eval_len', 'k', 'n_blocks', 'n_queries',
                               'recall_at_k', 'mass_covered', 'random_baseline', 'per_seed'],
           'cells': [], 'problems': [],
           'note': ('no selection-attribution artifact: selection recall/precision versus a dependency oracle '
                    'has NOT been measured, so "the learned selector retrieves the relevant blocks" is an '
                    'unsupported mechanism claim. Run the mechanism probe, then re-run this audit.')}
    p = os.path.join(REPO, path)
    if not os.path.exists(p):
        print(json.dumps({k: v for k, v in out.items() if k != 'cells'}, ensure_ascii=False, indent=2))
        return out
    raw = _load(p)
    cells = raw.get('cells') if isinstance(raw, dict) else None
    if not isinstance(cells, list) or not cells:
        out.update({'status': 'fail', 'problems': ['artifact has no cells list']})
        return out
    def _exact_int(value):
        # Block/query counts are exact: int(1.5) would silently certify a
        # different geometry (and a boolean is a flag, not a count).
        if value is None or isinstance(value, (bool, np.bool_)):
            return None
        if isinstance(value, numbers.Integral):
            return int(value)
        try:
            f = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return int(f) if math.isfinite(f) and f == math.floor(f) else None

    def _exact_count(value):
        # A numeric string is not an exact count.  `_exact_int` stays
        # permissive for seed *keys* (JSON object keys are strings), but cell
        # geometry is a measurement and must arrive as a real number.
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            return None
        f = float(value)
        return int(f) if math.isfinite(f) and f == math.floor(f) else None

    def _measurement(value):
        # Numeric strings are not recall/mass readings; float("0.9") would
        # otherwise let a hand-edited artifact unlock the claim gate.
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            return None
        f = float(value)
        return f if math.isfinite(f) else None

    for i, c in enumerate(cells):
        if not isinstance(c, dict):
            out['problems'].append(f'cell {i}: not an object')
            continue
        _miss = [k for k in ('arm', 'eval_len', 'k', 'n_blocks', 'n_queries', 'recall_at_k', 'mass_covered', 'random_baseline', 'per_seed') if c.get(k) is None]
        if _miss:
            out['problems'].append(f'cell {i} ({c.get("arm")}): missing {_miss}')
            continue
        if not isinstance(c.get('per_seed'), dict):
            out['problems'].append(f'cell {i} ({c.get("arm")}): per_seed must be a seed -> measurement object')
            continue
        _rec = _measurement(c['recall_at_k'])
        _mass = _measurement(c['mass_covered'])
        _base = _measurement(c['random_baseline'])
        if _rec is None or _mass is None or _base is None:
            out['problems'].append(
                f'cell {i}: recall/mass/baseline must be finite real numbers '
                '(not strings, booleans, or non-finite values)')
            continue
        rec, mass, base = _rec, _mass, _base
        k, B = _exact_count(c['k']), _exact_count(c['n_blocks'])
        el, nq = _exact_count(c['eval_len']), _exact_count(c['n_queries'])
        if k is None or B is None or el is None or nq is None:
            out['problems'].append(
                f'cell {i}: k/n_blocks/eval_len/n_queries must be exact '
                'integers (counts, not strings, booleans, or fractions)')
            continue
        if not (0.0 <= rec <= 1.0 and 0.0 <= mass <= 1.0 and 0.0 <= base <= 1.0):
            out['problems'].append(f'cell {i}: recall/mass/baseline outside [0,1]')
        if k <= 0 or B <= 0 or k > B:
            out['problems'].append(f'cell {i}: require 0 < k <= n_blocks (got k={k}, B={B})')
        if nq <= 0:
            out['problems'].append(f'cell {i}: n_queries must be positive')
        _seeds = c.get('per_seed')
        _n_seeds = len(_seeds)
        _st = None
        # Only complete seed rows enter the paired test.  A malformed row must
        # not inflate the reported seed count or the claim gate: the exact test
        # is run on the valid deltas and its own n is what the n>=6 floor uses.
        _d, _bad_seed = [], []
        _seen_seed_keys = set()
        for _sk, v in _seeds.items():
            # A seed key is an exact run identity.  "not-a-seed" (or two keys
            # that normalise to the same seed) must not inflate n and unlock
            # the n>=6 claim gate.
            _seed_id = _exact_int(_sk)
            if _seed_id is None or _seed_id in _seen_seed_keys:
                _bad_seed.append(str(_sk))
                continue
            _seen_seed_keys.add(_seed_id)
            if not isinstance(v, dict) or v.get('mass_covered') is None or v.get('random_baseline') is None:
                _bad_seed.append(str(_sk))
                continue
            _mv, _bv = v.get('mass_covered'), v.get('random_baseline')
            if (any(isinstance(x, (bool, np.bool_)) for x in (_mv, _bv))
                    or not all(isinstance(x, numbers.Real) for x in (_mv, _bv))):
                _bad_seed.append(str(_sk))
                continue
            _mf, _bf = float(_mv), float(_bv)
            # A present-but-NaN/out-of-range pair would be silently dropped by
            # the exact test, leaving `n_seeds` larger than the sample the p
            # value was computed on.  Treat it as a malformed row.
            if (not (math.isfinite(_mf) and math.isfinite(_bf))
                    or not (0.0 <= _mf <= 1.0 and 0.0 <= _bf <= 1.0)):
                _bad_seed.append(str(_sk))
                continue
            _d.append(_mf - _bf)
        if _bad_seed:
            out['problems'].append(f'cell {i} ({c.get("arm")}): per_seed rows have an invalid/duplicate seed key or lack mass_covered/random_baseline: {_bad_seed[:5]}')
        _n_valid = len(_d)
        _mean_delta = (float(np.mean(_d)) if _n_valid else None)
        if _n_valid >= 2:
            try:
                L = _import_exp_lib()
                _r = L.exact_signflip_p(_d)
                _st = {'p_exact_signflip': float(_r.get('p')), 'n': int(_r.get('n', _n_valid)), 'method': _r.get('method'),
                       'mean_delta': _mean_delta}
            except Exception as e:
                out['problems'].append(f'cell {i}: could not run the seed test ({type(e).__name__}: {e})')
        out['cells'].append({'arm': c.get('arm'), 'eval_len': el, 'k': k, 'n_blocks': B,
                             'recall_at_k': rec, 'mass_covered': mass, 'random_baseline': base,
                             'mass_excess': mass - base, 'recall_excess': rec - base,
                             'n_queries': nq, 'n_seeds': _n_seeds, 'n_seeds_valid': _n_valid,
                             'seed_test': _st,
                             'claim_allowed': bool(mass > base and rec > base and nq > 0 and _st is not None
                                                   and _st.get('mean_delta') is not None
                                                   and float(_st['mean_delta']) > 0.0
                                                   and _st.get('p_exact_signflip', 1.0) < 0.05
                                                   and int(_st.get('n', 0)) >= 6)})
    out['status'] = 'fail' if out['problems'] else 'ok'
    out['note'] = ('cells pass the structural contract; `claim_allowed` additionally requires mass/recall above the '
                   'random baseline and a seed-level exact sign-flip p<0.05 with n>=6')
    print(json.dumps({k: v for k, v in out.items() if k != 'cells'}, ensure_ascii=False, indent=2)[:4000])
    return out


def _extract_source_tree(obj):
    """Find a recorded source-tree manifest in a snapshot/summary object."""
    if not isinstance(obj, dict):
        return None
    for key in ('source_tree',):
        val = obj.get(key)
        if isinstance(val, dict) and (val.get('sha256') or val.get('files')):
            return val
    for key in ('code', 'provenance'):
        val = obj.get(key)
        if isinstance(val, dict):
            got = _extract_source_tree(val)
            if got:
                return got
    return None


def _display_repo_path(path):
    """Relative when inside the repo, absolute when the caller passed a temp path."""
    ap = os.path.abspath(path)
    rel = os.path.relpath(ap, REPO)
    return ap if rel.startswith('..') else rel


def _iter_freeze_locks(results_dirs=None):
    """Freeze-lock artifacts, explicit dirs first then every results_* tree."""
    dirs = list(results_dirs) if results_dirs is not None else list(_iter_result_dirs())
    for d in dirs:
        p = _audit_artifact(os.path.join(d, '.freeze_lock.json'))
        if os.path.exists(p):
            yield p


def _declared_source_drift(lock_files):
    """All declared analysis-time source-drift events among ``lock_files``.

    A drift event is evidence that the analysis was NOT run from the bytes
    bound at first test access.  It must be read from the freeze lock even
    when the plan snapshot supplies the *expected* source tree: the snapshot
    records the approved plan, while the lock records what actually ran.
    """
    out, seen, sources = [], set(), []
    for lp in lock_files:
        if not lp:
            continue
        p = os.path.abspath(lp)
        if p in seen or not os.path.exists(p):
            continue
        seen.add(p)
        lock = _load(p)
        ev = lock.get('source_tree_drift') if isinstance(lock, dict) else None
        if not ev:
            continue
        if not isinstance(ev, list):
            # Malformed evidence still means "a drift was declared"; never
            # silently treat a non-list value as a clean lock.
            ev = [ev]
        sources.append(_display_repo_path(p))
        out.extend(dict(item) if isinstance(item, dict) else {'event': item}
                   for item in ev)
    return out, sources


def _apply_declared_source_drift(out, lock_files):
    """Fail an audit result when a relevant freeze lock declares drift."""
    drift, sources = _declared_source_drift(lock_files)
    if drift:
        out['declared_source_drift'] = drift
        out['declared_source_drift_locks'] = sources
        out['problems'].append(
            f'{len(drift)} declared analysis-time source-drift event(s) '
            'are recorded in the freeze lock; the analysis was not run '
            'from byte-identical code')
        out['note'] = ('the analysis declared source drift via '
                       'V12_ALLOW_SOURCE_DRIFT=1; even if the bytes now '
                       'match the original attestation, this round is not '
                       'a byte-identical replay')
        out['status'] = 'fail'
    return out


def audit_source_integrity(snapshot_path='analysis_v12/plan_snapshot.json',
                           results_dirs=None, lock_path=None):
    """Compare the current source bytes with the source bound by an artifact.

    The git commit + CODE_SEMANTICS stamp cannot detect a dirty edit that
    leaves the stamp unchanged (the exact hazard documented at the top of the
    continuous-loop workflow), and a source tarball has no commit at all.  The
    plan snapshot, the first-test-access freeze lock and panel provenance now
    carry a content digest; this gate recomputes it and reports which files
    changed when the full manifest was recorded.  Preference order is
    snapshot -> explicit freeze lock -> any discovered freeze lock -> panel
    provenance, because the freeze lock is the artifact written immediately
    before the one-shot test access.  No recorded digest is
    ``not_verifiable_locally`` — never "source is fine".
    """
    L = _import_exp_lib()
    cur = L.source_tree_manifest()
    expected = None
    source = None
    legacy_lock = None

    def _relevant_locks():
        # Explicit lock first; otherwise every discovered freeze lock.  The
        # lock that supplied `source` is included even when a plan snapshot
        # was preferred for the expected tree, because a drift event is
        # evidence about what actually ran, not about the approved plan.
        locks = ([_audit_artifact(lock_path)] if lock_path
                 else list(_iter_freeze_locks(results_dirs)))
        if isinstance(source, str) and source.endswith('.freeze_lock.json'):
            locks.append(_audit_artifact(source))
        out, seen = [], set()
        for lp in locks:
            if not lp:
                continue
            p = os.path.abspath(lp)
            if p in seen or not os.path.exists(p):
                continue
            seen.add(p)
            out.append(p)
        return out

    def _check_source_locks(out):
        """Require every source-bound freeze lock to agree with `expected`.

        A plan snapshot records the approved tree, while a freeze lock records
        the tree actually used at first test access.  The audit used to prefer
        the snapshot and ignore a conflicting lock unless that lock happened to
        declare a drift event; a stale/mis-copied snapshot could therefore
        report "byte-identical" while the lock attested a different
        measurement tree.  Compare the digests whenever both sides exist.
        """
        if not isinstance(expected, dict):
            return out
        want = expected.get('sha256')
        if not isinstance(want, str) or not want:
            return out
        checked, mismatches, malformed = [], [], []
        for lp in _relevant_locks():
            lock = _load(lp)
            if not isinstance(lock, dict):
                continue
            tree = _extract_source_tree(lock)
            tree_sha = tree.get('sha256') if isinstance(tree, dict) else None
            top_sha = lock.get('source_tree_sha256')
            # A lock from the pre-source-binding driver has neither field: it
            # is honest evidence of the plan/slice, not of the code bytes.
            if tree_sha is None and top_sha is None:
                continue
            rec = {'lock': _display_repo_path(lp)}
            if (isinstance(tree_sha, str) and tree_sha
                    and isinstance(top_sha, str) and top_sha
                    and tree_sha != top_sha):
                rec['error'] = ('source_tree.sha256 disagrees with '
                                'source_tree_sha256')
                malformed.append(rec)
            sha = (tree_sha if isinstance(tree_sha, str) and tree_sha
                   else top_sha if isinstance(top_sha, str) and top_sha
                   else None)
            if sha is None:
                rec['error'] = 'source tree has no usable sha256'
                malformed.append(rec)
                continue
            rec['source_sha256'] = sha
            checked.append(rec)
            if sha != want:
                mismatches.append(rec)
        if checked:
            out['source_locks'] = checked
        if malformed:
            out['malformed_source_locks'] = malformed
            out['problems'].append(
                f'{len(malformed)} freeze lock(s) carry an incomplete or '
                'contradictory source-tree binding')
        if mismatches:
            out['source_lock_mismatches'] = mismatches
            out['problems'].append(
                f'{len(mismatches)} freeze lock(s) bind a different source '
                'tree than the snapshot/provenance; the measurement code '
                'bytes are not established by the preferred artifact')
        if malformed or mismatches:
            out['status'] = 'fail'
        return out

    def _emit(out):
        _check_source_locks(out)
        _apply_declared_source_drift(out, _relevant_locks())
        print(json.dumps({k: v for k, v in out.items() if k != 'files'},
                         ensure_ascii=False, indent=2)[:4000])
        return out
    snap_file = _audit_artifact(snapshot_path)
    snap = _load(snap_file)
    if isinstance(snap, dict):
        expected = _extract_source_tree(snap)
        if expected:
            source = _display_repo_path(snap_file)
    if expected is None:
        locks = ([_audit_artifact(lock_path)] if lock_path
                 else list(_iter_freeze_locks(results_dirs)))
        for lp in locks:
            if not os.path.exists(lp):
                continue
            lock = _load(lp)
            got = _extract_source_tree(lock)
            if got:
                expected = got
                source = _display_repo_path(lp)
                break
            if isinstance(lock, dict) and 'manifest_sha256' in lock:
                # A lock from the pre-source-binding driver is honest evidence
                # of the plan/slice, but it cannot attest the code bytes.
                legacy_lock = _display_repo_path(lp)
    if expected is None and legacy_lock is None:
        dirs = list(results_dirs) if results_dirs is not None else list(_iter_result_dirs())
        for d in dirs:
            p = _audit_artifact(os.path.join(d, 'summary.json'))
            summ = _load(p)
            expected = _extract_source_tree(summ)
            if expected:
                source = _display_repo_path(p)
                break
    if expected is None and legacy_lock is not None:
        out = {
            'kind': 'source_tree_audit',
            'status': 'not_verifiable_locally',
            'snapshot': (_display_repo_path(snap_file)
                         if os.path.exists(snap_file) else None),
            'source': legacy_lock,
            'expected_sha256': None,
            'current_sha256': cur.get('sha256'),
            'n_expected_files': None,
            'n_current_files': cur.get('n_files'),
            'added': [], 'removed': [], 'changed': [],
            'problems': [],
            'note': ('the discovered freeze lock predates source-tree '
                     'binding and has no source digest; the lock attests the '
                     'plan/test slice but not the measurement code bytes'),
        }
        return _emit(out)
    out = {
        'kind': 'source_tree_audit',
        'status': 'not_verifiable_locally',
        'snapshot': (_display_repo_path(snap_file)
                     if os.path.exists(snap_file) else None),
        'source': source,
        'expected_sha256': None,
        'current_sha256': cur.get('sha256'),
        'n_expected_files': None,
        'n_current_files': cur.get('n_files'),
        'added': [], 'removed': [], 'changed': [],
        'problems': [],
        'note': ('no source-tree digest is present in a plan snapshot or panel '
                 'provenance; a git commit/CODE_SEMANTICS alone cannot prove '
                 'which source bytes produced an artifact'),
    }
    if not isinstance(expected, dict):
        return _emit(out)
    want_sha = expected.get('sha256')
    out['expected_sha256'] = want_sha
    if not isinstance(want_sha, str) or not want_sha:
        out['status'] = 'fail'
        out['problems'].append('the recorded source-tree entry has no sha256')
        return _emit(out)
    _raw_exp_files = expected.get('files')
    if _raw_exp_files is not None and not isinstance(_raw_exp_files, dict):
        out['problems'].append('the recorded source-tree files entry is not an object')
    exp_files = _raw_exp_files if isinstance(_raw_exp_files, dict) else {}
    _raw_cur_files = cur.get('files')
    cur_files = _raw_cur_files if isinstance(_raw_cur_files, dict) else {}
    out['n_expected_files'] = len(exp_files) if _raw_exp_files is not None else None

    def _entry_sha(entry):
        # A malformed per-file entry is not evidence of a match.  Returning
        # None makes it differ from every real digest instead of raising
        # AttributeError on an untrusted snapshot.
        if not isinstance(entry, dict):
            return None
        value = entry.get('sha256')
        return value if isinstance(value, str) and value else None

    if any(not isinstance(v, dict) for v in exp_files.values()):
        out['problems'].append('the recorded source-tree file table has malformed entries')
    if exp_files:
        _added = sorted(set(cur_files) - set(exp_files))
        _removed = sorted(set(exp_files) - set(cur_files))
        _changed = sorted(
            rel for rel in set(cur_files) & set(exp_files)
            if _entry_sha(cur_files.get(rel)) != _entry_sha(exp_files.get(rel)))
        out['added'], out['removed'], out['changed'] = _added, _removed, _changed
        if _added:
            out['problems'].append(f'{len(_added)} source file(s) added since the snapshot')
        if _removed:
            out['problems'].append(f'{len(_removed)} source file(s) removed since the snapshot')
        if _changed:
            out['problems'].append(f'{len(_changed)} source file(s) changed since the snapshot')
    if want_sha != cur.get('sha256'):
        out['problems'].append(
            f'source tree sha256 {str(cur.get("sha256"))[:12]}… does not match '
            f'the recorded {str(want_sha)[:12]}…')
    out['status'] = 'fail' if out['problems'] else 'ok'
    if out['status'] == 'ok':
        out['note'] = 'the current source tree is byte-identical to the recorded tree'
    else:
        out['note'] = ('the current source tree differs from the tree bound to '
                       'the snapshot/provenance; a git commit + CODE_SEMANTICS '
                       'stamp alone cannot establish source identity')
    # `_emit` also inspects declared analysis-time drift in the relevant
    # freeze lock(s), even when the expected tree came from the plan snapshot.
    return _emit(out)


def audit_dependencies(lock_path=None):
    """Verify the installed environment against requirements.lock."""
    L = _import_exp_lib()
    c = L.dependency_contract(lock_path)
    _interp = c.get('interpreter') if isinstance(c.get('interpreter'), dict) else {}
    out = {
        'kind': 'dependency_audit',
        'status': ('ok' if c.get('status') in ('ok', 'partial') else 'fail'),
        'contract_status': c.get('status'),
        'lock_path': c.get('lock_path'),
        'lock_sha256': c.get('lock_sha256'),
        'n_pinned': c.get('n_pinned'),
        'n_installed': c.get('n_installed'),
        'n_unpinned_installed': c.get('n_unpinned_installed'),
        'mismatches': c.get('mismatches') or [],
        'missing': c.get('missing') or [],
        'invalid': c.get('invalid') or [],
        'unpinned_installed': (c.get('unpinned_installed') or [])[:50],
        'interpreter_status': _interp.get('status'),
        'interpreter_declared': _interp.get('declared') or {},
        'interpreter_actual': _interp.get('actual') or {},
        'interpreter_problems': _interp.get('problems') or [],
        'problems': [],
        'warnings': [],
    }
    if c.get('status') not in ('ok', 'partial'):
        out['problems'].append(c.get('note') or f'dependency contract is {c.get("status")}')
        if _interp.get('status') == 'mismatch':
            out['problems'].extend(_interp.get('problems') or [])
    elif c.get('status') == 'partial':
        out['warnings'].append(c.get('note') or 'the lock does not pin every installed distribution')
    print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
    return out


def audit_data_source(cache_dir=None):
    """Verify the upstream corpus revision recorded in each cache metadata file.

    A token-content fingerprint proves that two runs used the same tokens; it
    does not prove that a *new* cache was built from the same upstream snapshot.
    WikiText-103 and FineWeb-Edu are mutable Hub repositories, so the loaders
    pin a full commit and record it in `dataset_source`.  This audit recomputes
    that contract independently of the loader:

      - every non-legacy cache must carry a schema-valid `dataset_source`;
      - a Hub source must name a full 40-hex commit and attest the pin;
      - all paired train/validation/test caches must name the same commit;
      - a legacy cache (no field) is reported as unverifiable, never as clean.

    Local raw/jsonl corpora are legitimate; their content is bound by the token
    fingerprints and the audit records that the upstream commit is not
    attested.
    """
    import glob as _glob
    L = _import_exp_lib()
    if cache_dir in (None, ''):
        cache_dir = os.environ.get('WT103_CACHE_DIR') or './wt103_cache'
    cache_dir = _audit_artifact(cache_dir)
    out = {
        'kind': 'dataset_source_audit',
        'status': 'ok',
        'cache_dir': cache_dir,
        'n_caches': 0,
        'n_pinned': 0,
        'n_legacy': 0,
        'n_local': 0,
        'revisions': {},
        'caches': {},
        'problems': [],
        'warnings': [],
    }
    if not os.path.isdir(cache_dir):
        out['status'] = 'not_verifiable_locally'
        out['note'] = f'{cache_dir} does not exist; build or copy the corpus cache to audit its upstream revision'
        print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
        return out
    metas = sorted(_glob.glob(os.path.join(cache_dir, 'meta_*.json')))
    metas += sorted(_glob.glob(os.path.join(cache_dir, 'test_meta_*.json')))
    # `meta_*.json` also matches test_meta_*.json on case-sensitive globs only
    # when the pattern is literal, but de-duplicate defensively.
    metas = sorted(set(metas))
    if not metas:
        out['status'] = 'not_verifiable_locally'
        out['note'] = f'{cache_dir} has no meta_*.json/test_meta_*.json; cannot audit the corpus revision'
        print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
        return out
    train_metas = []
    test_metas = []
    dataset_revisions = {}
    for path in metas:
        name = os.path.basename(path)
        meta = _load(path)
        if not isinstance(meta, dict):
            out['problems'].append(f'{name}: unreadable metadata')
            continue
        out['n_caches'] += 1
        # FineWeb metadata records `source`; WikiText metadata does not.
        try:
            if meta.get('source'):
                expected = L.fineweb_edu_dataset_source(str(meta.get('source')))
            else:
                expected = L.wikitext_dataset_source()
        except Exception as e:
            out['problems'].append(
                f'{name}: the pinned source cannot be resolved '
                f'({type(e).__name__}: {e})')
            expected = None
        probs, warns = L._dataset_source_problems(meta, expected, name)
        src = meta.get('dataset_source')
        if isinstance(src, dict):
            if src.get('provider') == 'huggingface':
                out['n_pinned'] += 1
                rev = str(src.get('revision') or '').lower()
                if rev:
                    out['revisions'].setdefault(rev, []).append(name)
                    dataset_revisions.setdefault(
                        'fineweb' if meta.get('source') else 'wikitext',
                        set()).add(rev)
            elif src.get('provider') == 'local':
                out['n_local'] += 1
        else:
            out['n_legacy'] += 1
        out['caches'][name] = {
            'dataset': 'fineweb' if meta.get('source') else 'wikitext',
            'provider': (src or {}).get('provider') if isinstance(src, dict) else None,
            'repo': (src or {}).get('repo') if isinstance(src, dict) else None,
            'revision': (src or {}).get('revision') if isinstance(src, dict) else None,
            'revision_pinned': (src or {}).get('revision_pinned') if isinstance(src, dict) else None,
            'problems': probs,
            'warnings': warns,
        }
        out['problems'].extend(probs)
        out['warnings'].extend(warns)
        if name.startswith('test_meta_'):
            test_metas.append((name, meta))
        else:
            train_metas.append(meta)
    # Cross-cache consistency: every test cache must name the same upstream
    # commit as its paired train caches.  A directory may legitimately hold
    # several corpora (e.g. WikiText and FineWeb), so pair only train caches
    # from the same upstream repository/config.  Match the loaders' tokenizer-
    # fit rule too: a cache built for a different fit corpus is a different
    # protocol, not a reference snapshot for this test slice.
    if test_metas:
        for test_name, tm in test_metas:
            ts = (tm.get('dataset_source')
                  if isinstance(tm, dict) else None)
            ts = ts if isinstance(ts, dict) else {}
            tfit = tm.get('fit_fp') if isinstance(tm, dict) else None
            paired = []
            for _m in train_metas:
                s = _m.get('dataset_source') if isinstance(_m, dict) else None
                s = s if isinstance(s, dict) else {}
                if (ts.get('provider') == 'huggingface'
                        and s.get('provider') == 'huggingface'
                        and (ts.get('repo') != s.get('repo')
                             or ts.get('config') != s.get('config'))):
                    continue
                mfit = _m.get('fit_fp') if isinstance(_m, dict) else None
                if tfit and mfit and tfit != mfit:
                    continue
                paired.append(_m)
            if ts and not paired:
                out['warnings'].append(
                    f'{test_name}: no train cache matches the test corpus and '
                    'tokenizer-fit provenance; the upstream revision cannot '
                    'be compared')
            probs, warns = L.dataset_source_pair_problems(tm, paired,
                                                          label='test/train')
            out['problems'].extend(probs)
            out['warnings'].extend(warns)
    # Two different pinned revisions *within one corpus* mean that corpus's
    # caches mix upstream snapshots.  Per-cache records may still be
    # internally consistent, so this is a warning at directory level; the
    # train/test pair check above turns the dangerous case into a failure.
    # Different corpora naturally have different commits and must not warn.
    for _dataset, _revs in sorted(dataset_revisions.items()):
        if len(_revs) > 1:
            out['warnings'].append(
                f'the {_dataset} caches contain more than one upstream '
                'revision: ' + ', '.join(sorted(_revs)))
    if out['problems']:
        out['status'] = 'fail'
    elif out['n_legacy'] or out['n_local'] or out['warnings']:
        if out['n_legacy']:
            # Any cache without the field means the directory as a whole is
            # not fully attested.  Say exactly that instead of letting the
            # verified subset stand in for the missing one.
            out['status'] = 'not_verifiable_locally'
            out['note'] = (
                f'{out["n_legacy"]} of {out["n_caches"]} cache metadata file(s) '
                'predate dataset_source; the upstream commit cannot be attested '
                'for those caches without rebuilding (they remain readable, but '
                'are not clean)')
        else:
            out['status'] = 'ok'
            out['note'] = ('present dataset-source records verify; local '
                           'caches and directory-level caveats are listed under '
                           'warnings and are not claimed as attestations')
    print(json.dumps({k: v for k, v in out.items() if k != 'caches'},
                     ensure_ascii=False, indent=2)[:4000])
    return out


def _read_experiment_events(outdir):
    path = os.path.join(outdir, 'experiments.jsonl')
    events = []
    if not os.path.exists(path):
        return events
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if isinstance(e, dict):
                    events.append(e)
    except OSError:
        return []
    return events


def _actual_run_order(events, *, kind='cell', label_fmt='{v}::seed{s}'):
    """Cell start order from the append-only event log (completed cells only).

    ``label_fmt`` must match the label convention of the plan being audited:
    training uses ``variant::seed{seed}``, while P5T evaluation/probe events use
    ``variant::s{seed}``.  Deriving the label from the event's variant/seed
    (rather than trusting an arbitrary ``key`` field) keeps a corrupted event
    from inventing a planned cell.
    """
    cells = []
    for e in events:
        if (e.get('kind') != kind or e.get('status') != 'ok'
                or e.get('variant') is None or e.get('seed') is None):
            continue
        try:
            label = label_fmt.format(v=e['variant'], s=int(e['seed']))
        except (TypeError, ValueError, OverflowError, KeyError):
            continue
        cells.append((str(e.get('started_at') or ''), label))
    # A retrained/resumed cell legitimately appears more than once; its LAST
    # start is the execution that produced the current measurement.
    latest = {}
    for started, lab in cells:
        if lab not in latest or started >= latest[lab]:
            latest[lab] = started
    return [lab for lab, _started in sorted(latest.items(), key=lambda kv: kv[1])]


def audit_run_order(paths=None):
    """Run-order / temporal-confounding audit (zero GPU).

    A paired seed test only compares arms within a seed.  If the driver runs
    all seeds of arm A and only then all seeds of arm B, the two arms of every
    pair were trained in different machine/time blocks; a driver update, a
    thermal change or a noisy neighbour can then masquerade as an arm effect.
    This audit checks the recorded design (`run_order.json`) and, when present,
    the executed order from `experiments.jsonl`:

      * complete blocks: every block (default: seed) contains every declared
        arm exactly once;
      * position balance: an arm does not always occupy the same slot;
      * actual order: a completed arm-major run is a FAIL even when the plan
        was blocked (the driver did not honour it), while a missing plan is
        reported as not verifiable rather than silently assumed.

    It is descriptive: it never changes a PPL, seed or decision rule.
    """
    L = _import_exp_lib()
    if paths is None:
        paths = [os.path.join(REPO, d) for d in _iter_result_dirs()]
    elif isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    out = {'kind': 'run_order_audit', 'status': 'ok', 'panels': {},
           'n_ok': 0, 'n_fail': 0, 'n_not_verifiable': 0,
           'problems': [], 'warnings': [],
           'note': ('blocked randomized run order controls calendar-time / '
                    'machine-drift confounding; arm-major execution is a '
                    'design failure, not a null result')}
    for d in paths:
        d = os.fspath(d)
        if not os.path.isdir(d):
            continue
        name = os.path.basename(os.path.normpath(d))
        plan_path = os.path.join(d, 'run_order.json')
        _plan_exists = os.path.exists(plan_path)
        plan = _load(plan_path) if _plan_exists else None
        if _plan_exists and plan is None:
            # A file that exists but cannot be parsed is not the same as a
            # pre-planning legacy panel: fail closed instead of inferring.
            rec = {'status': 'fail', 'plan_present': True,
                   'problems': ['run_order.json exists but is not a JSON object; '
                                'the recorded design cannot be verified']}
            out['n_fail'] += 1
            out['problems'].extend(f'{name}: {p}' for p in rec['problems'])
            out['panels'][name] = rec
            continue
        events = _read_experiment_events(d)
        actual = _actual_run_order(events)
        if plan is None:
            if not actual:
                rec = {'status': 'not_verifiable_locally',
                       'reason': 'no run_order.json and no completed cell events'}
                out['n_not_verifiable'] += 1
                out['panels'][name] = rec
                continue
            # Legacy/inferred order: classify it, but never claim the design
            # was randomized when no plan exists.
            blocks = {}
            for lab in actual:
                try:
                    v, s = lab.rsplit('::seed', 1)
                except ValueError:
                    continue
                blocks.setdefault(s, []).append(v)
            cell_blocks = {}
            cell_variants = {}
            for lab in actual:
                try:
                    v, s = lab.rsplit('::seed', 1)
                except ValueError:
                    continue
                cell_blocks[lab] = str(s)
                cell_variants[lab] = v
            inferred = {'design': {'name': 'inferred_no_plan', 'randomized': False,
                                   'block': 'seed', 'design_seed': None},
                        'variants': sorted({v for vs in blocks.values() for v in vs}),
                        'blocks': {b: vs for b, vs in blocks.items()},
                        'planned_cells': list(actual),
                        'cell_blocks': cell_blocks,
                        'cell_variants': cell_variants}
            # Passing the actual order lets the same arm-major detector run on
            # a legacy panel; the inferred design itself stays unverifiable.
            rec = L.run_order_balance(inferred, actual)
            rec['actual']['inferred_from_events'] = True
            rec['plan_present'] = False
        else:
            _cb = plan.get('cell_blocks')
            if isinstance(_cb, dict):
                labels_in_plan = set(_cb.keys())
                filtered = [x for x in actual if x in labels_in_plan]
                rec = L.run_order_balance(plan, filtered)
                if not filtered and rec.get('status') == 'ok':
                    rec['status'] = 'not_verifiable_locally'
                    rec.setdefault('warnings', []).append(
                        'the plan exists but no completed cell event matches it; '
                        'the recorded design is verified, the execution is not')
            else:
                # A plan without its cell->block mapping cannot be tied to the
                # event log; audit the plan itself and say the actual order was
                # not checkable rather than crashing or silently skipping it.
                rec = L.run_order_balance(plan)
                rec.setdefault('warnings', []).append(
                    'plan has no cell_blocks mapping; the actual execution '
                    'order could not be checked')
            rec['plan_present'] = True
            rec['actual'] = rec.get('actual') or {'n_observed': len(actual)}
        st = rec.get('status')
        out['n_ok'] += int(st == 'ok')
        out['n_fail'] += int(st == 'fail')
        out['n_not_verifiable'] += int(st == 'not_verifiable_locally')
        if st == 'fail':
            out['problems'].extend(f'{name}: {p}' for p in rec.get('problems') or [])
        for w in rec.get('warnings') or []:
            out['warnings'].append(f'{name}: {w}')
        out['panels'][name] = rec
    if out['n_fail']:
        out['status'] = 'fail'
    elif out['n_ok'] == 0:
        out['status'] = 'not_verifiable_locally'
    print(json.dumps({k: v for k, v in out.items() if k != 'panels'},
                     ensure_ascii=False, indent=2)[:4000])
    if out['panels']:
        for _n, _r in out['panels'].items():
            print(f"  [run-order] {_n}: {_r.get('status')} "
                  f"(blocks={_r.get('n_blocks')}, complete={_r.get('complete_blocks')}, "
                  f"arm_major={_r.get('arm_major')})")
    return out


def audit_evaluation_order(paths=None):
    """Final-test evaluation-order / measurement-time audit (zero GPU).

    Training is blocked by seed (``run_order.json``), but the one-shot P5T
    evaluation used to iterate arms outside seeds: every seed of arm A, then
    every seed of arm B.  That makes arm and measurement time perfectly
    confounded for the confirmatory endpoint too (a driver update, thermal
    drift or noisy neighbour can masquerade as an arm effect).  P5T now writes
    ``clean_by_length_order.json`` and ``distractor_probe_order.json`` before
    reading the test split.  This audit verifies:

      * the plan is a complete-block, position-balanced design;
      * the plan covers every planned cell exactly once;
      * the plan was created no later than the first matching measurement;
      * the executed order honours the plan (a completed arm-major run fails);
      * a clean P5T measurement with no precommitted plan fails closed; the
        secondary probe without a plan is reported as not-verifiable with a
        warning instead.

    It is descriptive and never changes a PPL, seed, endpoint or decision rule.
    """
    L = _import_exp_lib()
    if paths is None:
        paths = [os.path.join(REPO, d) for d in _iter_result_dirs()]
    elif isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    out = {'kind': 'evaluation_order_audit', 'status': 'ok', 'panels': {},
           'n_ok': 0, 'n_fail': 0, 'n_not_verifiable': 0,
           'problems': [], 'warnings': [],
           'note': ('the clean P5T evaluation must be seed-blocked and '
                    'position-balanced; arm-major execution is a measurement-'
                    'time confound, not a null result')}
    tasks = (
        ('clean_by_length', 'clean_by_length_order.json', 'eval', True),
        ('distractor_probe', 'distractor_probe_order.json', 'cell', False),
    )
    for d in paths:
        d = os.fspath(d)
        if not os.path.isdir(d):
            continue
        name = os.path.basename(os.path.normpath(d))
        if 'smoke' in name.lower():
            continue
        events = _read_experiment_events(d)
        # Only audit directories that actually carry P5T artifacts or clean
        # evaluation events; training panels have kind='cell' events too, but
        # no evaluation plan/records and no kind='eval' events.
        _p5t_marker = any(os.path.exists(os.path.join(d, f)) for f in (
            'bylen.json', 'distractor.json', 'clean_by_length_order.json',
            'distractor_probe_order.json', '.freeze_lock.json'))
        _eval_event = any(e.get('kind') == 'eval' for e in events)
        if not (_p5t_marker or _eval_event):
            continue
        panel = {'status': 'ok', 'tasks': {}, 'problems': [], 'warnings': []}
        clean_status = None
        for task, fname, kind, required in tasks:
            plan_path = os.path.join(d, fname)
            plan_exists = os.path.exists(plan_path)
            plan = _load(plan_path) if plan_exists else None
            actual = _actual_run_order(events, kind=kind,
                                       label_fmt='{v}::s{s}')
            rec = {'task': task, 'plan_present': plan_exists, 'kind': kind,
                   'n_observed': len(actual), 'status': 'ok',
                   'problems': [], 'warnings': []}
            _measured = bool(actual) or (
                required and os.path.exists(os.path.join(d, 'bylen.json')))
            if plan_exists and plan is None:
                rec['status'] = 'fail'
                rec['problems'].append(
                    f'{fname} exists but is not a readable JSON object; the '
                    'recorded design cannot be verified')
            elif plan is None:
                if required and _measured:
                    rec['status'] = 'fail'
                    rec['problems'].append(
                        f'{task}: {max(1, len(actual))} measurement(s) exist '
                        'but no precommitted evaluation-order plan was found; '
                        'the measurement-time order is confounded/unverifiable')
                else:
                    rec['status'] = 'not_verifiable_locally'
                    if actual:
                        rec['warnings'].append(
                            f'{task}: no precommitted plan; the observed order '
                            'is classified but the design is not claimed')
            else:
                if plan.get('kind') != 'evaluation_order':
                    rec['problems'].append(
                        f'plan.kind={plan.get("kind")!r}, expected '
                        "'evaluation_order'")
                if plan.get('task') != task:
                    rec['problems'].append(
                        f'plan.task={plan.get("task")!r}, expected {task!r}')
                planned = plan.get('planned_cells')
                cell_blocks = plan.get('cell_blocks')
                if not isinstance(planned, list) or not planned:
                    rec['problems'].append(
                        'plan.planned_cells must be a non-empty list')
                elif len(set(map(str, planned))) != len(planned):
                    rec['problems'].append(
                        'plan.planned_cells contains duplicate labels')
                if not isinstance(cell_blocks, dict) or not cell_blocks:
                    rec['problems'].append(
                        'plan.cell_blocks must be a non-empty object')
                elif isinstance(planned, list) and set(map(str, planned)) != set(map(str, cell_blocks)):
                    rec['problems'].append(
                        'plan.cell_blocks keys do not match planned_cells')
                rec2 = L.run_order_balance(plan, actual)
                for k in ('n_blocks', 'n_cells', 'n_variants',
                          'complete_blocks', 'arm_major', 'position_balanced',
                          'design_ok', 'actual'):
                    if k in rec2:
                        rec[k] = rec2[k]
                rec['problems'].extend(rec2.get('problems') or [])
                rec['warnings'].extend(rec2.get('warnings') or [])
                # A plan written after the first measurement is not a
                # precommitment, even if its permutation happens to match.
                _starts = [str(e.get('started_at') or '') for e in events
                           if e.get('kind') == kind and e.get('status') == 'ok'
                           and e.get('started_at')]
                _created = str(plan.get('created_at') or '')
                if _starts and _created and _created > min(_starts):
                    rec['problems'].append(
                        f'plan.created_at={_created!r} is later than the first '
                        f'{task} measurement ({min(_starts)!r}); the order was '
                        'not recorded before measurement')
                if not _created:
                    rec['warnings'].append('plan has no created_at timestamp')
                if not actual and not rec['problems']:
                    rec['status'] = 'not_verifiable_locally'
                    rec['warnings'].append(
                        'the plan exists but no completed measurement event '
                        'matches it; the design is verified, the execution is not')
                elif rec['problems']:
                    rec['status'] = 'fail'
            if rec['status'] == 'ok':
                out['n_ok'] += 1
            elif rec['status'] == 'fail':
                out['n_fail'] += 1
            else:
                out['n_not_verifiable'] += 1
            if task == 'clean_by_length':
                clean_status = rec['status']
            panel['tasks'][task] = rec
            panel['problems'].extend(rec['problems'])
            panel['warnings'].extend(rec['warnings'])
        if clean_status == 'fail' or any(
                r['status'] == 'fail' for r in panel['tasks'].values()):
            panel['status'] = 'fail'
        elif clean_status == 'ok':
            panel['status'] = 'ok'
        else:
            panel['status'] = 'not_verifiable_locally'
        out['problems'].extend(f'{name}: {p}' for p in panel['problems'])
        out['warnings'].extend(f'{name}: {w}' for w in panel['warnings'])
        out['panels'][name] = panel
    _statuses = [r.get('status') for r in out['panels'].values()]
    if any(s == 'fail' for s in _statuses):
        out['status'] = 'fail'
    elif any(s == 'ok' for s in _statuses):
        out['status'] = 'ok'
    else:
        out['status'] = 'not_verifiable_locally'
    print(json.dumps({k: v for k, v in out.items() if k != 'panels'},
                     ensure_ascii=False, indent=2)[:4000])
    for _n, _r in out['panels'].items():
        _clean = (_r.get('tasks') or {}).get('clean_by_length') or {}
        print(f"  [eval-order] {_n}: {_r.get('status')} "
              f"(clean={_clean.get('status')}, "
              f"blocks={_clean.get('n_blocks')}, "
              f"complete={_clean.get('complete_blocks')}, "
              f"arm_major={_clean.get('arm_major')})")
    return out


def audit_unit_influence(path=None):
    """Result-analysis audit: leave-one-unit-out influence is present and sound.

    A pooled ΔPPL can be statistically significant and still be carried by one
    evaluation article.  ``explib.confirmation._doc_level_analysis`` now records a
    ``doc_sampling.unit_influence`` block; this audit independently checks that
    the block is internally consistent with the reported pooled estimate and
    flags comparisons whose direction or magnitude depends on one unit.

    It is a **warning-level** audit by design: the frozen decision rule does
    not include a unit-drop threshold, so a fragile comparison must carry the
    caveat in the report but must not be silently turned into "not confirmed".
    A malformed block (point estimate disagrees with the pooled Δ, missing
    units, wrong vector lengths) is a ``fail`` because it cannot support any
    influence statement.
    """
    p = path or os.path.join(REPO, 'analysis_v12', 'test_stats.json')
    out = {'kind': 'unit_influence_audit', 'status': 'ok', 'path': p,
           'comparisons': {}, 'n_ok': 0, 'n_fragile': 0, 'n_missing': 0,
           'problems': [], 'warnings': [],
           'note': ('post-hoc descriptive robustness; a fragile comparison must '
                    'carry the single-unit caveat, but this audit never changes '
                    'a frozen p/Holm/confirmed verdict')}
    if not os.path.exists(p):
        out['status'] = 'not_verifiable_locally'
        out['reason'] = 'no final-test stats artifact on disk'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    data = _load(p)
    if data is None:
        out['status'] = 'fail'
        out['problems'].append(f'{p} exists but is not a readable JSON object')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    ft = data.get('finaltest') if isinstance(data.get('finaltest'), dict) else data
    comps = ft.get('comparisons') if isinstance(ft, dict) else None
    if not isinstance(comps, dict) or not comps:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = 'the stats artifact carries no comparisons'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out

    def _finite(x):
        # A numeric string (or bytes) is not a measurement: float("1.0")
        # would let a hand-edited/stale artifact replay as a finite reading.
        if (isinstance(x, (bool, np.bool_, str, bytes, bytearray))
                or x is None):
            return None
        try:
            y = float(x)
        except (TypeError, ValueError, OverflowError):
            return None
        return y if math.isfinite(y) else None

    for name, comp in comps.items():
        if not isinstance(comp, dict):
            # A non-object comparison cannot carry a measured influence block;
            # silently skipping it would hide a corrupt comparison.
            out['problems'].append(
                f'{name}: comparison is a {type(comp).__name__}, not an object')
            out['comparisons'][name] = {'present': False,
                                        'error': 'not_an_object'}
            continue
        ds = comp.get('doc_sampling')
        inf = ds.get('unit_influence') if isinstance(ds, dict) else None
        rec = {'present': isinstance(inf, dict)}
        if not isinstance(inf, dict):
            if isinstance(ds, dict) and ds.get('pooled_delta_ppl_ci95') is not None:
                out['n_missing'] += 1
                out['warnings'].append(
                    f'{name}: pooled CI exists but no unit_influence block '
                    '(stale/legacy analysis artifact); the single-unit '
                    'sensitivity is unverified')
            out['comparisons'][name] = rec
            continue
        _available = inf.get('available')
        if not isinstance(_available, (bool, np.bool_)):
            # The availability flag decides whether the block is a measurement
            # or an explicit omission; "false"/1 must not be silently treated
            # as available evidence.
            out['problems'].append(
                f'{name}: unit_influence.available is not a boolean '
                f'({_available!r})')
            rec['error'] = 'invalid_available'
            out['comparisons'][name] = rec
            continue
        rec['available'] = bool(_available)
        if not _available:
            out['warnings'].append(
                f'{name}: unit influence unavailable ({inf.get("reason")})')
            out['comparisons'][name] = rec
            continue
        n_units = inf.get('n_units')
        ids = inf.get('unit_ids')
        pooled = _finite(ds.get('pooled_delta_ppl'))
        full = _finite(inf.get('full_delta_ppl'))
        rec.update({'n_units': n_units, 'fragile_to_one_unit': inf.get('fragile_to_one_unit'),
                    'most_influential_unit': inf.get('most_influential_unit'),
                    'max_abs_relative_change_log_ratio':
                        inf.get('max_abs_relative_change_log_ratio')})
        if not isinstance(n_units, int) or n_units < 3:
            out['problems'].append(f'{name}: invalid n_units={n_units!r}')
        if not isinstance(ids, list) or len(ids) != n_units:
            out['problems'].append(
                f'{name}: unit_ids length does not match n_units={n_units!r}')
        elif len(set(map(str, ids))) != n_units:
            out['problems'].append(
                f'{name}: unit_ids are not unique; the folds cannot identify units')
        for _vec_key in ('loo_delta_ppl', 'loo_delta_log_ratio',
                         'loo_change_ppl', 'loo_change_log_ratio'):
            _vec = inf.get(_vec_key)
            if not isinstance(_vec, list) or len(_vec) != n_units:
                out['problems'].append(
                    f'{name}: {_vec_key} length does not match n_units={n_units!r}')
                continue
            if any(v is not None and _finite(v) is None for v in _vec):
                out['problems'].append(
                    f'{name}: {_vec_key} contains a non-finite, non-null value')
        _idx = inf.get('most_influential_index')
        if (_idx is not None
                and (not isinstance(_idx, int) or isinstance(_idx, bool)
                     or not (0 <= _idx < (n_units if isinstance(n_units, int) else 0)))):
            out['problems'].append(
                f'{name}: most_influential_index {_idx!r} is outside the unit vector')
        if (isinstance(_idx, int) and isinstance(ids, list)
                and 0 <= _idx < len(ids)
                and inf.get('most_influential_unit') is not None
                and str(ids[_idx]) != str(inf.get('most_influential_unit'))):
            out['problems'].append(
                f'{name}: most_influential_unit does not match unit_ids[index]')
        if pooled is None or full is None:
            out['problems'].append(
                f'{name}: pooled or full influence point estimate is missing/non-finite')
        elif abs(full - pooled) > 1e-6 * max(1.0, abs(pooled)):
            out['problems'].append(
                f'{name}: unit-influence full point {full!r} does not reproduce '
                f'the reported pooled Δ {pooled!r}')
        _mi = inf.get('most_influential_unit')
        if (_mi is not None and isinstance(ids, list) and _mi not in ids):
            out['problems'].append(
                f'{name}: most_influential_unit {_mi!r} is not one of unit_ids')
        _frag = inf.get('fragile_to_one_unit')
        if _frag is not None and not isinstance(_frag, bool):
            out['problems'].append(
                f'{name}: fragile_to_one_unit is not a boolean or null')
        elif _frag is True:
            out['n_fragile'] += 1
            out['warnings'].append(
                f'{name}: direction/magnitude depends on one '
                f'{inf.get("most_influential_unit")!r}; the report must carry '
                'the single-unit caveat')
        elif _frag is False:
            out['n_ok'] += 1
        out['comparisons'][name] = rec
    if out['problems']:
        out['status'] = 'fail'
    elif out['n_fragile'] or out['n_missing']:
        out['status'] = 'warnings'
    print(json.dumps({k: v for k, v in out.items() if k != 'comparisons'},
                     ensure_ascii=False, indent=2))
    return out


def audit_analysis_choices(path=None):
    """Result-analysis audit: multi-specification concordance is coherent.

    A single panel can be described by per-seed mean/median/trimmed summaries,
    a token-pooled difference, or equal-weighted evaluation units.  The report
    records ``doc_sampling.analysis_choice_concordance``; this gate
    independently replays that summary from the stored specification rows and
    checks it against the frozen per-seed mean.  Disagreement in direction or
    in zero-exclusion is a **warning** (a reporting caveat), not a change to
    the frozen decision.  A stale, internally inconsistent or non-reproducible
    block is a ``fail`` because it cannot support the claim that the
    sensitivity was actually checked.
    """
    L = _import_exp_lib()
    p = path or os.path.join(REPO, 'analysis_v12', 'test_stats.json')
    out = {'kind': 'analysis_choice_audit', 'status': 'ok', 'path': p,
           'comparisons': {}, 'n_ok': 0, 'n_fragile': 0,
           'n_missing': 0, 'problems': [], 'warnings': [],
           'note': ('post-hoc specification concordance; divergent directions or '
                    'zero-exclusion are reporting caveats, never changes to a '
                    'frozen p/Holm/confirmed verdict')}
    if not os.path.exists(p):
        out['status'] = 'not_verifiable_locally'
        out['reason'] = 'no final-test stats artifact on disk'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    data = _load(p)
    if data is None:
        out['status'] = 'fail'
        out['problems'].append(f'{p} exists but is not a readable JSON object')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    ft = data.get('finaltest') if isinstance(data.get('finaltest'), dict) else data
    comps = ft.get('comparisons') if isinstance(ft, dict) else None
    if not isinstance(comps, dict) or not comps:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = 'the stats artifact carries no comparisons'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out

    def _finite(x):
        if (isinstance(x, (bool, np.bool_, str, bytes, bytearray))
                or x is None):
            return None
        try:
            y = float(x)
        except (TypeError, ValueError, OverflowError):
            return None
        return y if math.isfinite(y) else None

    def _same_choice_value(a, b):
        # bool must not compare equal to 1/0, and a numeric string must not
        # replay as a float (the concordance helper deliberately rejects both).
        if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
            return (isinstance(a, (bool, np.bool_))
                    and isinstance(b, (bool, np.bool_))
                    and bool(a) == bool(b))
        if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            return (isinstance(a, (list, tuple))
                    and isinstance(b, (list, tuple))
                    and len(a) == len(b)
                    and all(_same_choice_value(x, y) for x, y in zip(a, b)))
        if isinstance(a, numbers.Number) and isinstance(b, numbers.Number):
            if (isinstance(a, (complex, np.complexfloating))
                    or isinstance(b, (complex, np.complexfloating))):
                return False
            _fa, _fb = _finite(a), _finite(b)
            return (_fa is not None and _fb is not None
                    and abs(_fa - _fb) <= 1e-12 * max(1.0, abs(_fa), abs(_fb)))
        return a == b

    keys = ('status', 'claim', 'n_specs', 'n_available', 'n_unavailable',
            'n_nonzero', 'n_positive', 'n_negative', 'n_zero',
            'n_same_as_primary', 'n_opposite_to_primary', 'direction_stable',
            'n_with_ci', 'n_excluding_zero', 'n_crossing_zero',
            'precision_concordant', 'excluding_zero_specs',
            'crossing_zero_specs', 'fragile', 'primary_sign',
            'primary_excludes_zero')
    for name, comp in comps.items():
        if not isinstance(comp, dict):
            # A non-object comparison cannot carry a replayable analysis-choice
            # block; silently skipping it would hide a corrupt comparison.
            out['problems'].append(
                f'{name}: comparison is a {type(comp).__name__}, not an object')
            out['comparisons'][name] = {'present': False,
                                        'error': 'not_an_object'}
            continue
        ds = comp.get('doc_sampling')
        _has_cc = isinstance(ds, dict) and 'analysis_choice_concordance' in ds
        cc = ds.get('analysis_choice_concordance') if _has_cc else None
        rec = {'present': isinstance(cc, dict)}
        if _has_cc and not isinstance(cc, dict):
            # Present-but-malformed is not the same as absent: a corrupt block
            # must not silently downgrade to a warning (or, worse, look like a
            # legacy artifact that simply never recorded the sensitivity).
            out['problems'].append(
                f'{name}: analysis_choice_concordance is a '
                f'{type(cc).__name__}, not a JSON object')
            rec['error'] = 'not_an_object'
            out['comparisons'][name] = rec
            continue
        if not _has_cc:
            if 'omitted' not in comp:
                out['n_missing'] += 1
                if (isinstance(ds, dict)
                        and (ds.get('per_doc_delta_mean') is not None
                             or ds.get('pooled_delta_ppl') is not None)):
                    _why = ('doc_sampling exists but no analysis-choice block '
                            '(stale/legacy analysis artifact)')
                else:
                    _why = 'no doc_sampling/analysis-choice block'
                out['warnings'].append(
                    f'{name}: {_why}; the multi-specification sensitivity is '
                    'unverified')
            out['comparisons'][name] = rec
            continue
        rec['status'] = cc.get('status')
        specs = cc.get('specifications')
        rec['n_specs'] = cc.get('n_specs')
        if cc.get('kind') != 'analysis_choice_concordance':
            out['problems'].append(
                f'{name}: unexpected kind {cc.get("kind")!r}')
        if cc.get('version') != L.ANALYSIS_CHOICE_POLICY['version']:
            out['problems'].append(
                f'{name}: analysis-choice version {cc.get("version")!r} does '
                f'not match {L.ANALYSIS_CHOICE_POLICY["version"]!r}')
        if not isinstance(specs, list) or len(specs) < 2:
            out['problems'].append(
                f'{name}: specifications must be a list with at least 2 rows')
            out['comparisons'][name] = rec
            continue
        if not _same_choice_value(cc.get('n_specs'), len(specs)):
            out['problems'].append(
                f'{name}: n_specs={cc.get("n_specs")!r} but {len(specs)} '
                'specification rows are stored')
        names = [s.get('spec') for s in specs if isinstance(s, dict)]
        if (len(names) != len(specs)
                or any(not isinstance(n, str) or not n.strip() for n in names)
                or len(set(names)) != len(names)):
            out['problems'].append(
                f'{name}: specification names are missing, non-string, '
                'duplicated or not all dicts')
            out['comparisons'][name] = rec
            continue
        pname = cc.get('primary_spec')
        primaries = [s for s in specs if s.get('spec') == pname]
        if len(primaries) != 1:
            out['problems'].append(
                f'{name}: primary_spec {pname!r} does not identify exactly one '
                'stored specification')
            out['comparisons'][name] = rec
            continue
        alts = [s for s in specs if s.get('spec') != pname]
        try:
            want = L.analysis_choice_concordance(primaries[0], alts)
        except (ValueError, TypeError, FloatingPointError, OverflowError) as e:
            out['problems'].append(
                f'{name}: stored specifications cannot be replayed '
                f'({type(e).__name__}: {e})')
            out['comparisons'][name] = rec
            continue
        for k in keys:
            if not _same_choice_value(cc.get(k), want.get(k)):
                out['problems'].append(
                    f'{name}: stored {k}={cc.get(k)!r} does not reproduce '
                    f'replayed {want.get(k)!r}')
        # The aggregate keys above do not see every derived field of every
        # row (``is_primary`` is the clearest example).  Replaying the rows
        # and comparing them also catches a row whose stored labels/counts
        # contradict its own point/CI instead of letting the aggregates hide it.
        _wspecs = want.get('specifications')
        if not isinstance(_wspecs, list) or len(_wspecs) != len(specs):
            out['problems'].append(
                f'{name}: specification rows do not reproduce the replayed '
                f'row count ({len(specs)} stored vs '
                f'{len(_wspecs) if isinstance(_wspecs, list) else "non-list"})')
        else:
            for _i, (_stored, _replayed) in enumerate(zip(specs, _wspecs)):
                if not isinstance(_stored, dict) or not isinstance(_replayed, dict):
                    out['problems'].append(
                        f'{name}: specification row {_i} is not an object')
                    continue
                # Compare the fields actually stored (a hand-written minimal
                # row may omit derived fields; the replay supplies them), but
                # reject fields the replayed row never produces.
                _bad = sorted(k for k in set(_stored) & set(_replayed)
                              if not _same_choice_value(_stored.get(k),
                                                        _replayed.get(k)))
                _extra = sorted(set(_stored) - set(_replayed))
                if _bad or _extra:
                    _what = []
                    if _bad:
                        _what.append(f'does not reproduce fields {_bad}')
                    if _extra:
                        _what.append(f'has unexpected fields {_extra}')
                    out['problems'].append(
                        f'{name}: specification row {_i} '
                        f'({_stored.get("spec")!r}) ' + ' and '.join(_what))
        _er = cc.get('effect_range_ppl')
        _wer = want.get('effect_range_ppl')
        if (_er is None) != (_wer is None):
            out['problems'].append(
                f'{name}: stored effect_range_ppl presence does not reproduce')
        elif _wer is not None and not isinstance(_er, dict):
            out['problems'].append(
                f'{name}: stored effect_range_ppl is a '
                f'{type(_er).__name__}, not an object')
        elif isinstance(_er, dict) and isinstance(_wer, dict):
            _extra = sorted(set(_er) - {'min', 'max', 'span',
                                        'relative_span', 'n'})
            if _extra:
                out['problems'].append(
                    f'{name}: stored effect_range_ppl has unexpected fields '
                    f'{_extra}')
            for k in ('min', 'max', 'span', 'relative_span', 'n'):
                _a, _b = _er.get(k), _wer.get(k)
                if k == 'n':
                    if not _same_choice_value(_a, _b):
                        out['problems'].append(
                            f'{name}: effect_range_ppl.{k}={_a!r} does not '
                            f'reproduce {_b!r}')
                    continue
                _fa, _fb = _finite(_a), _finite(_b)
                if (_fa is None) != (_fb is None):
                    out['problems'].append(
                        f'{name}: effect_range_ppl.{k} presence does not '
                        'reproduce')
                elif _fa is not None and abs(_fa - _fb) > 1e-9 * max(
                        1.0, abs(_fb)):
                    out['problems'].append(
                        f'{name}: effect_range_ppl.{k}={_fa!r} does not '
                        f'reproduce {_fb!r}')
        _pm = _finite(primaries[0].get('point'))
        _mean = _finite(comp.get('mean'))
        if _pm is None or _mean is None:
            out['problems'].append(
                f'{name}: primary analysis-choice point or frozen mean is '
                'missing/non-finite')
        elif abs(_pm - _mean) > 1e-6 * max(1.0, abs(_mean)):
            out['problems'].append(
                f'{name}: primary analysis-choice point {_pm!r} does not '
                f'reproduce the frozen per-seed mean {_mean!r}')
        if cc.get('status') in ('direction_unstable', 'precision_sensitive'):
            out['n_fragile'] += 1
            if cc.get('status') == 'direction_unstable':
                out['warnings'].append(
                    f'{name}: analysis specifications disagree on direction; '
                    'the report must show the specification table and must not '
                    'quote one aggregation as robust')
            else:
                out['warnings'].append(
                    f'{name}: direction is stable but zero-exclusion depends on '
                    'the analysis choice; interval claims must name the unit/'
                    'estimand')
        elif cc.get('status') in ('insufficient_specs', 'zero_effect'):
            out['warnings'].append(
                f'{name}: analysis-choice status is {cc.get("status")!r}; the '
                'multi-specification check is not informative')
        else:
            out['n_ok'] += 1
        out['comparisons'][name] = rec
    if out['problems']:
        out['status'] = 'fail'
    elif out['n_ok'] == 0 and not out['n_fragile']:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = ('no comparison carries a replayable analysis-choice '
                         'block')
    elif out['warnings']:
        out['status'] = 'warnings'
    print(json.dumps({k: v for k, v in out.items() if k != 'comparisons'},
                     ensure_ascii=False, indent=2))
    return out


def audit_experiment_ledger(paths=None):
    """Verify the hash-chained experiment ledgers used by the order audits.

    ``experiments.jsonl`` is the execution record from which run-order,
    evaluation-order and failure/selection audits reconstruct what actually
    happened.  It was append-only but not tamper-evident: deleting the one
    failed event, editing a ``started_at`` or dropping the tail could make an
    arm-major run look blocked without leaving a trace.  Every event written
    by the current code now carries ``_ledger`` = ``{schema, seq, prev,
    sha256}`` and an atomic ``*.anchor.json`` sidecar pins the count and last
    link.  This gate:

      * replays the chain and checks sequence, predecessor and body hashes;
      * checks the anchor count/last link/file size, so a truncated suffix is
        detected rather than accepted as "the run ended early";
      * reports a legacy ledger (all events before chaining) as
        ``not_verifiable_locally`` and a missing anchor as ``warnings`` — it
        never upgrades an unverifiable log to a green pass;
      * reads local files only, starts no training and changes no measurement,
        p-value, Holm family or frozen verdict.

    It is tamper-EVIDENT, not a signature: rewriting the log and regenerating
    its chain/anchor can still forge it.  Its job is to make accidental or
    one-line tampering visible, not to defend against a malicious author with
    write access to both files.
    """
    L = _import_exp_lib()
    import glob as _glob
    if paths is None:
        found = []
        for name in _iter_result_dirs():
            p = os.path.join(REPO, name, 'experiments.jsonl')
            if os.path.isfile(p):
                found.append(p)
        found.extend(sorted(_glob.glob(
            os.path.join(REPO, 'analysis_*', 'experiments.jsonl'))))
        paths = found
    elif isinstance(paths, (str, bytes, os.PathLike)):
        paths = [paths]
    out = {'kind': 'experiment_ledger_audit', 'status': 'ok',
           'ledgers': {}, 'n_ledgers': 0, 'n_ok': 0, 'n_fail': 0,
           'n_warnings': 0, 'n_not_verifiable': 0,
           'n_events': 0, 'n_chained': 0, 'n_legacy_events': 0,
           'problems': [], 'warnings': [],
           'note': ('hash-chained experiment ledger: an edited, deleted, '
                    'inserted or reordered event breaks the next link, and the '
                    'tail anchor exposes truncation; this does not change any '
                    'measurement or frozen decision')}
    seen = set()
    for raw in paths:
        p = os.fsdecode(os.path.abspath(os.fspath(raw)))
        if os.path.isdir(p):
            p = os.path.join(p, 'experiments.jsonl')
        if p in seen:
            continue
        seen.add(p)
        rep = L.verify_experiment_ledger(p)
        out['n_ledgers'] += 1
        out['n_events'] += int(rep.get('n_events') or 0)
        out['n_chained'] += int(rep.get('n_chained') or 0)
        out['n_legacy_events'] += int(rep.get('n_legacy') or 0)
        short = os.path.relpath(p, REPO) if p.startswith(REPO + os.sep) else p
        out['ledgers'][short] = rep
        status = rep.get('status')
        if status == 'fail':
            out['n_fail'] += 1
            out['problems'].extend(f'{short}: {x}' for x in rep.get('problems') or [])
        elif status == 'warnings':
            out['n_warnings'] += 1
            out['warnings'].extend(f'{short}: {x}' for x in rep.get('warnings') or [])
        elif status == 'not_verifiable_locally':
            out['n_not_verifiable'] += 1
            out['warnings'].append(
                f'{short}: {rep.get("note") or "ledger is not verifiable"}')
        else:
            out['n_ok'] += 1
    if out['n_ledgers'] == 0:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = ('no experiments.jsonl ledger exists yet; run a panel '
                         'with the current code to produce a chained one')
    elif out['n_fail']:
        out['status'] = 'fail'
    elif out['n_ok'] == 0 and out['n_warnings'] == 0:
        out['status'] = 'not_verifiable_locally'
    elif out['n_warnings'] or out['n_not_verifiable']:
        out['status'] = 'warnings'
    print(json.dumps({k: v for k, v in out.items() if k != 'ledgers'},
                     ensure_ascii=False, indent=2))
    return out
