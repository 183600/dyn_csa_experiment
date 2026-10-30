# Audit the held-out test-access ledger (one-shot evaluation accounting).
#
# The freeze lock is written immediately before the first P5T read and updated
# with the test-slice fingerprint immediately after it.  It also records every
# explicitly allowed re-read (``test_reads`` / ``history``) and every conscious
# re-freeze (``superseded``), but until this gate no downstream analysis read
# those counters.  A confirmatory report could therefore survive a second
# evaluation of the same frozen slice at the nominal alpha.
#
# This fragment reuses ``exp_lib.confirmatory_access_audit`` and only reports:
# it does not change a p-value, a Holm family, the frozen ``confirmed`` set or
# any training/evaluation measurement.  ``fail`` means the recorded access
# pattern cannot support a confirmatory claim; an unplanned re-read is a
# protocol deviation, not an arithmetic error.
#
# This fragment deliberately carries no module docstring: the audit CLI help
# text lives in part_00 and must remain the package __doc__.
import glob
import json
import os


def _ta_display(path):
    try:
        rel = os.path.relpath(os.path.abspath(path), REPO)
    except Exception:
        rel = str(path)
    return rel if not rel.startswith('..') else os.path.abspath(path)


def _ta_load(path):
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except Exception:
        return None


def _ta_collect_paths(paths):
    """Return ``(lock_paths, stats_paths)`` from explicit paths or results_*."""
    locks, stats = [], []
    if paths is None:
        lock_paths = list(_iter_freeze_locks())
        stats = sorted(glob.glob(os.path.join(REPO, 'analysis_*', 'test_stats.json')))
        return lock_paths, stats
    if isinstance(paths, (str, bytes, os.PathLike)):
        paths = [paths]
    for raw in paths:
        p = os.fsdecode(os.fspath(raw))
        if os.path.isdir(p):
            cand = os.path.join(p, '.freeze_lock.json')
            if os.path.isfile(cand):
                locks.append(cand)
            cand = os.path.join(p, 'test_stats.json')
            if os.path.isfile(cand):
                stats.append(cand)
            continue
        base = os.path.basename(p)
        if base == 'test_stats.json':
            stats.append(p)
        else:
            locks.append(p)
    return locks, stats


def _ta_match_stats(lock, stats_paths):
    """Pick the one stats artifact that belongs to this lock, if any.

    A match needs a non-empty manifest hash or test-slice fingerprint on both
    sides.  When both identities are present they must agree; a stats file
    that matches one but contradicts the other is ambiguous and is refused
    rather than silently attached.
    """
    if not stats_paths:
        return None, []
    lock_hash = lock.get('manifest_sha256') if isinstance(lock, dict) else None
    lock_fp = lock.get('test_ids_fp') if isinstance(lock, dict) else None
    hits = []
    for p in stats_paths:
        obj = _ta_load(p)
        if not isinstance(obj, dict):
            continue
        man = obj.get('manifest') if isinstance(obj.get('manifest'), dict) else {}
        adm = obj.get('admission') if isinstance(obj.get('admission'), dict) else {}
        s_hash = man.get('manifest_sha256')
        s_fp = adm.get('test_slice_fp')
        hash_ok = (isinstance(lock_hash, str) and isinstance(s_hash, str)
                   and lock_hash == s_hash)
        fp_ok = (isinstance(lock_fp, str) and isinstance(s_fp, str)
                 and lock_fp == s_fp)
        if hash_ok and fp_ok:
            hits.append((0, p, obj))
        elif hash_ok:
            hits.append((1, p, obj))
        elif fp_ok:
            # The slice agrees but the manifest does not.  Attach it at the
            # lowest priority: the primitive's cross-check will then fail the
            # audit visibly instead of hiding a plan/analysis mismatch behind
            # a "no stats matched" warning.
            hits.append((2, p, obj))
    hits.sort(key=lambda x: (x[0], x[1]))
    if not hits:
        return None, []
    return hits[0][2], [h[1] for h in hits]


def _ta_manifest_from_lock(lock):
    """Load the manifest the lock names, but only if its bytes still match.

    A lock records ``manifest_path`` and the manifest hash it was created
    under.  Preferring that file over the repository default keeps renamed
    plans (and the smoke manifest) attached to their own access policy while
    still refusing a changed file.
    """
    if not isinstance(lock, dict):
        return None
    path = lock.get('manifest_path')
    if not isinstance(path, str) or not os.path.isfile(path):
        return None
    obj = _ta_load(path)
    if not isinstance(obj, dict):
        return None
    try:
        obj['manifest_sha256'] = _import_exp_lib().file_sha256(path)
    except Exception:
        return None
    if obj['manifest_sha256'] != lock.get('manifest_sha256'):
        return None
    obj['_path'] = os.path.abspath(path)
    return obj


def _ta_current_manifest():
    path = os.path.join(REPO, 'confirmation_frozen.json')
    if not os.path.isfile(path):
        return None
    obj = _ta_load(path)
    if not isinstance(obj, dict):
        return None
    try:
        obj['manifest_sha256'] = _import_exp_lib().file_sha256(path)
    except Exception:
        return None
    obj['_path'] = os.path.abspath(path)
    return obj


def audit_test_access(paths=None):
    """Replay held-out access ledgers and classify their confirmatory status.

    ``paths=None`` scans every ``results_*/.freeze_lock.json`` and every
    ``analysis_*/test_stats.json``.  Explicit paths may be lock files, stats
    files or directories containing either.

    A lock that is internally inconsistent, a stats artifact bound to a
    different manifest/test slice, a stored ``test_access`` block that does not
    replay, and a second read under the single-look default are all ``fail``:
    the confirmatory claim is not supported.  A pre-declared Bonferroni look
    budget that is respected is ``ok`` (the effective alpha/p are exported).
    No lock at all is ``not_verifiable_locally``, never "clean".
    """
    L = _import_exp_lib()
    lock_paths, stats_paths = _ta_collect_paths(paths)
    # De-duplicate while preserving discovery order.
    lock_paths = list(dict.fromkeys(os.path.abspath(p) for p in lock_paths))
    stats_paths = list(dict.fromkeys(os.path.abspath(p) for p in stats_paths))
    out = {
        'kind': 'test_access_gate',
        'status': 'ok',
        'locks': {},
        'n_locks': 0,
        'n_confirmatory': 0,
        'n_downgraded': 0,
        'n_not_applicable': 0,
        'n_invalid': 0,
        'n_stats_matched': 0,
        'problems': [],
        'warnings': [],
        'note': ('held-out access accounting only: it does not change a frozen '
                 'p-value, a Holm family, the `confirmed` set or any '
                 'measurement.  A second read of the frozen test slice is a '
                 'protocol deviation; only a pre-declared repeated-look budget '
                 'can be confirmatory, using the planned look count.'),
    }
    if not lock_paths and not stats_paths:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = ('no .freeze_lock.json or analysis_*/test_stats.json '
                         'found — no held-out evaluation has been recorded here')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    if not lock_paths:
        out['status'] = 'fail'
        out['problems'].append(
            'a confirmation stats artifact exists but no freeze lock was found; '
            'the held-out access cannot be tied to a pre-registered plan/slice')
        out['stats_only'] = [_ta_display(p) for p in stats_paths]
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out

    current = _ta_current_manifest()
    for lp in lock_paths:
        lock_obj = _ta_load(lp)
        stats_obj, hits = _ta_match_stats(lock_obj, stats_paths)
        rec_path = _ta_display(lp)
        # Prefer the manifest named by the lock (renamed plans and the smoke
        # manifest keep their own policy), then the repository default if its
        # bytes still match.  A post-access policy edit cannot be laundered in
        # either case: the manifest hash must equal the locked hash.
        manifest_obj = _ta_manifest_from_lock(lock_obj)
        if manifest_obj is None and isinstance(lock_obj, dict) and isinstance(
                current, dict):
            locked_hash = lock_obj.get('manifest_sha256')
            cur_hash = current.get('manifest_sha256')
            if (isinstance(locked_hash, str) and isinstance(cur_hash, str)
                    and locked_hash == cur_hash):
                manifest_obj = current
        rec = L.confirmatory_access_audit(
            lp, stats=stats_obj, manifest=manifest_obj,
            lock_label=rec_path)
        if (manifest_obj is None and isinstance(lock_obj, dict)
                and isinstance(lock_obj.get('manifest_sha256'), str)):
            rec['manifest_changed_since_lock'] = True
            rec['warnings'].append(
                'the manifest named by the lock is absent or its bytes differ; '
                'the locked access policy is unavailable, so the historical '
                'single-look rule was applied')
            if rec['status'] == 'ok':
                rec['status'] = 'warn'
        if stats_obj is not None:
            out['n_stats_matched'] += 1
            rec['stats'] = _ta_display(hits[0]) if hits else None
            if len(hits) > 1:
                rec['warnings'].append(
                    f'{len(hits)} stats artifacts matched this lock; the first '
                    'was used and the others are listed as ambiguous')
                rec['ambiguous_stats'] = [_ta_display(p) for p in hits[1:]]
        else:
            rec['stats'] = None
        out['locks'][rec_path] = rec
        out['n_locks'] += 1
        if rec.get('confirmatory_allowed') is True:
            out['n_confirmatory'] += 1
        elif rec.get('confirmatory_allowed') is False:
            out['n_downgraded'] += 1
        else:
            out['n_not_applicable'] += 1
        if rec.get('status') == 'fail':
            out['n_invalid'] += 1
            why = (rec.get('problems') or rec.get('downgrade_reasons')
                   or ['access ledger cannot support confirmation'])
            out['problems'].append(
                f'{rec_path}: {"; ".join(str(x) for x in why)}')
        for w in rec.get('warnings') or []:
            out['warnings'].append(f'{rec_path}: {w}')

    if out['problems']:
        out['status'] = 'fail'
    elif out['n_downgraded']:
        out['status'] = 'warn'
    elif out['warnings']:
        out['status'] = 'warn'
    elif out['n_confirmatory']:
        out['status'] = 'ok'
    else:
        out['status'] = 'not_verifiable_locally'
    if out['n_downgraded']:
        out['note'] = ('at least one held-out evaluation ledger cannot support a '
                       'confirmatory claim (unplanned re-read, post-access plan '
                       'change, or an inconsistent lock); see per-lock reasons')
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out
