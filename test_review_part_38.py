"""CPU checks for one-shot held-out test access accounting.

The freeze lock already refuses a second P5T read unless two environment
variables are set, and it increments ``test_reads`` when that happens.  The
confirmatory analysis never read the counter, so an explicitly allowed re-read
could still be reported as "confirmed" at the nominal alpha.  These checks
cover the access policy, the lock/history/superseded replay, the effective
(alpha-spent) verdict, the audit gate, and the confirmation/report wiring.

No test split, checkpoint or GPU is read.
"""
import contextlib
import inspect
import io
import json
import os
import sys as _sys
import tempfile

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


def _lock(**kw):
    base = {
        'stage': 'accessed',
        'manifest_sha256': 'MANIFEST',
        'test_ids_fp': 'TEST_FP',
        'test_reads': 1,
        'history': [
            {'stage': 'preaccess'},
            {'stage': 'accessed', 'test_ids_fp': 'TEST_FP'},
        ],
        'superseded': [],
        'source_tree_drift': [],
    }
    base.update(kw)
    return base


def _reread_lock(n=2, **kw):
    history = [{'stage': 'preaccess'},
               {'stage': 'accessed', 'test_ids_fp': 'TEST_FP'}]
    for _ in range(n - 1):
        history.append({'stage': 'reread', 'reason': 'unit-test extra look'})
    return _lock(test_reads=n, history=history, **kw)


def _stats(p_holm=0.01, confirmed=('c1',), **kw):
    out = {
        'alpha': 0.05,
        'manifest': {'manifest_sha256': 'MANIFEST'},
        'admission': {'test_slice_fp': 'TEST_FP'},
        'comparisons': {'c1': {'p_holm': p_holm}},
        'holm': {'c1': {'p_holm': p_holm}},
        'confirmed': list(confirmed),
    }
    out.update(kw)
    return out


def _quiet(fn, *args, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kw)


def test_access_policy_defaults_and_normalisation():
    default = L.test_access_policy(None)
    assert default['method'] == 'single_look'
    assert default['max_looks'] == 1
    assert default['declared'] is False

    man = {'statistics': {'test_access_policy': {
        'method': 'bonferroni', 'max_looks': 3,
        'require_reason_for_reread': False}}}
    pol = L.test_access_policy(man)
    assert pol['method'] == 'bonferroni'
    assert pol['max_looks'] == 3
    assert pol['require_reason_for_reread'] is False
    assert pol['declared'] is True
    assert pol['source'] == 'statistics.test_access_policy'

    # Top-level declaration is accepted for older/simpler manifests.
    pol = L.test_access_policy({'test_access_policy': {'max_looks': 1}})
    assert pol['method'] == 'single_look' and pol['declared'] is True


def test_access_policy_fails_closed_on_malformed_declarations():
    bad = [
        'single_look',
        {'method': 'single_look', 'max_looks': 2},
        {'method': 'bonferroni', 'max_looks': 0},
        {'method': 'bonferroni', 'max_looks': True},
        {'method': 'bonferroni', 'max_looks': 1.5},
        {'method': 'pocock', 'max_looks': 2},
        {'method': 'bonferroni', 'max_looks': 2, 'max_look': 2},
        {'method': 'bonferroni', 'max_looks': 2,
         'require_reason_for_reread': 'yes'},
        {'method': 'bonferroni', 'max_looks': 2, 'schema': 2},
        {'method': 'bonferroni', 'max_looks': 2, 'rationale': 123},
    ]
    for raw in bad:
        try:
            L.test_access_policy({'statistics': {'test_access_policy': raw}})
        except ValueError:
            pass
        else:
            raise AssertionError(f'malformed access policy was accepted: {raw!r}')


def test_single_look_is_confirmatory_and_preserves_frozen_p():
    a = L.confirmatory_access_audit(_lock(), stats=_stats())
    assert a['status'] == 'ok', a
    assert a['confirmatory_allowed'] is True
    assert a['n_test_reads'] == 1 and a['n_reread_events'] == 0
    assert a['alpha_effective'] == 0.05
    assert a['confirmed_frozen'] == ['c1']
    assert a['confirmed_effective'] == ['c1']
    assert a['comparisons']['c1']['p_holm_effective'] == 0.01


def test_unplanned_reread_downgrades_and_keeps_descriptive_spending():
    a = L.confirmatory_access_audit(_reread_lock(2), stats=_stats(p_holm=0.01))
    assert a['status'] == 'fail', a
    assert a['confirmatory_allowed'] is False
    assert a['downgrade_reasons'] == ['unplanned_reread']
    assert a['confirmed_frozen'] == ['c1']
    assert a['confirmed_effective'] == []
    # The frozen p is untouched; a Bonferroni-over-observed-reads value is
    # exported as a descriptive sensitivity only.
    assert a['comparisons']['c1']['p'] == 0.01
    assert a['comparisons']['c1']['p_holm_effective'] == 0.01
    assert a['comparisons']['c1']['p_holm_bonferroni_descriptive'] == 0.02


def test_declared_bonferroni_look_budget_spends_against_planned_count():
    man = {'statistics': {'alpha': 0.05, 'test_access_policy': {
        'method': 'bonferroni', 'max_looks': 2}}}
    lock = _reread_lock(2)
    stats = {
        'alpha': 0.05,
        'manifest': {'manifest_sha256': 'MANIFEST'},
        'admission': {'test_slice_fp': 'TEST_FP'},
        'comparisons': {'c1': {'p_holm': 0.01}, 'c2': {'p_holm': 0.02}},
        'holm': {'c1': {'p_holm': 0.01}, 'c2': {'p_holm': 0.02}},
        'confirmed': ['c1', 'c2'],
    }
    a = L.confirmatory_access_audit(lock, stats=stats, manifest=man)
    assert a['status'] == 'ok', a
    assert a['confirmatory_allowed'] is True
    assert abs(a['alpha_effective'] - 0.025) < 1e-15
    # The look correction is applied once: alpha_effective = 0.05/2, so both
    # frozen p=0.01 and p=0.02 clear it (comparing p_eff=p*2 against
    # alpha_effective would require 0.04 < 0.025 and double-count the budget).
    assert a['confirmed_effective'] == ['c1', 'c2']
    assert a['comparisons']['c1']['p_holm_effective'] == 0.02
    assert a['comparisons']['c2']['p_holm_effective'] == 0.04

    # One observed look under a pre-declared two-look budget still spends the
    # planned alpha: stopping early cannot buy back the second look.
    one = L.confirmatory_access_audit(_lock(), stats=stats, manifest=man)
    assert one['confirmatory_allowed'] is True
    assert abs(one['alpha_effective'] - 0.025) < 1e-15
    assert one['confirmed_effective'] == ['c1', 'c2']


def test_more_looks_than_declared_is_a_failure():
    man = {'statistics': {'test_access_policy': {
        'method': 'bonferroni', 'max_looks': 2}}}
    a = L.confirmatory_access_audit(_reread_lock(3), stats=_stats(), manifest=man)
    assert a['status'] == 'fail'
    assert a['confirmatory_allowed'] is False
    assert a['downgrade_reasons'] == ['more_looks_than_declared']
    assert a['confirmed_effective'] == []


def test_post_access_plan_change_downgrades_but_pre_access_revision_is_ok():
    prior = {'manifest_sha256': 'OLD', 'test_ids_fp': 'TEST_FP'}
    a = L.confirmatory_access_audit(
        _lock(superseded=[prior], manifest_sha256='MANIFEST'),
        stats=_stats())
    assert a['n_post_access_plan_changes'] == 1
    assert a['confirmatory_allowed'] is False
    assert 'plan_changed_after_access' in a['downgrade_reasons']
    assert a['confirmed_effective'] == []

    # A revision that never reached a test read is an ordinary pre-registration
    # change, not a post-access look.
    b = L.confirmatory_access_audit(
        _lock(superseded=[{'manifest_sha256': 'OLD', 'test_ids_fp': None}]),
        stats=_stats())
    assert b['n_post_access_plan_changes'] == 0
    assert b['n_pre_access_plan_revisions'] == 1
    assert b['confirmatory_allowed'] is True

    # A changed test slice after access is independently fatal.
    c = L.confirmatory_access_audit(
        _lock(superseded=[{'manifest_sha256': 'OLD', 'test_ids_fp': 'OTHER_FP'}]),
        stats=_stats())
    assert c['test_slice_changed_after_access'] is True
    assert c['confirmatory_allowed'] is False
    assert 'test_slice_changed_after_access' in c['downgrade_reasons']


def test_malformed_lock_fails_closed():
    cases = [
        {},
        {'test_reads': True, 'test_ids_fp': 'TEST_FP'},
        {'test_reads': -1, 'test_ids_fp': 'TEST_FP'},
        {'test_reads': 0, 'test_ids_fp': 'TEST_FP'},
        {'test_reads': 1, 'test_ids_fp': None},
        {'test_reads': 2, 'test_ids_fp': 'TEST_FP',
         'history': [{'stage': 'accessed'}]},
        {'test_reads': 1, 'test_ids_fp': 'TEST_FP',
         'history': [{'stage': 'accessed'}, {'stage': 'reread'}]},
        {'test_reads': 1, 'test_ids_fp': 'TEST_FP', 'history': 'not-a-list'},
        {'test_reads': 1, 'test_ids_fp': 'TEST_FP',
         'history': [{'stage': 'accessed'}], 'superseded': 'x'},
        None,
        [1, 2],
    ]
    for lock in cases:
        a = L.confirmatory_access_audit(lock)
        assert a['status'] == 'fail', (lock, a)
        assert a['confirmatory_allowed'] is False, (lock, a)


def test_descriptive_only_plan_is_not_counted_as_a_confirmatory_look():
    stats = _stats()
    stats['manifest']['statistics'] = {'descriptive_only': True}
    a = L.confirmatory_access_audit(_reread_lock(2), stats=stats)
    assert a['descriptive_only'] is True
    assert a['status'] == 'not_applicable', a
    assert a['confirmatory_allowed'] is None
    assert a['downgrade_reasons'] == []
    assert a['confirmed_effective'] is None


def test_confirmed_stats_without_a_recorded_read_fail_closed():
    lock = _lock(test_reads=0, test_ids_fp=None,
                 history=[{'stage': 'preaccess'}])
    a = L.confirmatory_access_audit(lock, stats=_stats())
    assert a['status'] == 'fail', a
    assert a['confirmatory_allowed'] is False
    assert any('no held-out test read' in p for p in a['problems']), a


def test_stats_identity_mismatch_and_stored_block_tamper_fail():
    lock = _reread_lock(2)
    man = {'statistics': {'alpha': 0.05, 'test_access_policy': {
        'method': 'bonferroni', 'max_looks': 2}}}
    good = L.confirmatory_access_audit(lock, stats=_stats(), manifest=man)
    base = _stats()
    base['test_access'] = good
    base['confirmed_effective'] = good['confirmed_effective']
    replay = L.confirmatory_access_audit(lock, stats=base, manifest=man)
    assert replay['status'] == 'ok' and not replay['problems'], replay

    # A stats artifact from a different plan/slice cannot be attached to this
    # lock's access ledger.
    for key, value in (('manifest_sha256', 'OTHER'), ('test_slice_fp', 'OTHER')):
        bad = _stats()
        if key == 'manifest_sha256':
            bad['manifest'] = {'manifest_sha256': value}
        else:
            bad['admission'] = {'test_slice_fp': value}
        a = L.confirmatory_access_audit(lock, stats=bad, manifest=man)
        assert a['status'] == 'fail', (key, a)
        assert any('different' in p for p in a['problems']), (key, a)

    # A stored access block that claims fewer reads than the lock is tamper.
    tampered = dict(base)
    tampered['test_access'] = dict(good)
    tampered['test_access']['n_test_reads'] = 1
    a = L.confirmatory_access_audit(lock, stats=tampered, manifest=man)
    assert a['status'] == 'fail'
    assert any('n_test_reads' in p for p in a['problems']), a

    tampered = dict(base)
    tampered['confirmed_effective'] = ['c2']
    a = L.confirmatory_access_audit(lock, stats=tampered, manifest=man)
    assert a['status'] == 'fail'
    assert any('confirmed_effective' in p for p in a['problems']), a

    bad_p = _stats()
    bad_p['holm']['c1']['p_holm'] = 'not-a-p'
    a = L.confirmatory_access_audit(lock, stats=bad_p, manifest=man)
    assert a['status'] == 'fail'
    assert any('non-numeric' in p for p in a['problems']), a


def test_audit_gate_clean_downgraded_and_missing_artifacts():
    with tempfile.TemporaryDirectory() as td:
        empty = A.audit_test_access([td])
        assert empty['status'] == 'not_verifiable_locally'

        # Use the repository's current manifest hash so the lock is not
        # rejected merely for belonging to a different plan; the gate must
        # load the plan policy when the hash matches.
        cur = A._ta_current_manifest() or {}
        man_hash = cur.get('manifest_sha256') or 'MANIFEST'
        lock = _lock(manifest_sha256=man_hash)
        with open(os.path.join(td, '.freeze_lock.json'), 'w',
                  encoding='utf-8') as fh:
            json.dump(lock, fh)
        stats = _stats(manifest={'manifest_sha256': man_hash})
        with open(os.path.join(td, 'test_stats.json'), 'w',
                  encoding='utf-8') as fh:
            json.dump(stats, fh)
        out = A.audit_test_access([td])
        assert out['status'] == 'ok', out
        assert out['n_locks'] == 1 and out['n_confirmatory'] == 1
        assert out['n_stats_matched'] == 1
        assert not out['warnings'], out

        with open(os.path.join(td, '.freeze_lock.json'), 'w',
                  encoding='utf-8') as fh:
            json.dump(_reread_lock(2, manifest_sha256=man_hash), fh)
        out = A.audit_test_access([td])
        assert out['status'] == 'fail', out
        assert out['n_downgraded'] == 1
        assert out['locks'][os.path.join(td, '.freeze_lock.json')][
            'confirmatory_allowed'] is False

    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, 'test_stats.json'), 'w',
                  encoding='utf-8') as fh:
            json.dump(_stats(), fh)
        out = A.audit_test_access([td])
        assert out['status'] == 'fail'
        assert any('no freeze lock' in p for p in out['problems']), out


def test_audit_gate_uses_the_manifest_named_by_the_lock():
    with tempfile.TemporaryDirectory() as td:
        man_path = os.path.join(td, 'renamed_plan.json')
        man = {'statistics': {'alpha': 0.05, 'test_access_policy': {
            'method': 'bonferroni', 'max_looks': 2}}}
        with open(man_path, 'w', encoding='utf-8') as fh:
            json.dump(man, fh)
        man_hash = L.file_sha256(man_path)
        lock = _reread_lock(2, manifest_sha256=man_hash,
                            manifest_path=man_path)
        with open(os.path.join(td, '.freeze_lock.json'), 'w',
                  encoding='utf-8') as fh:
            json.dump(lock, fh)
        stats = _stats(manifest={'manifest_sha256': man_hash})
        with open(os.path.join(td, 'test_stats.json'), 'w',
                  encoding='utf-8') as fh:
            json.dump(stats, fh)
        out = A.audit_test_access([td])
        rec = out['locks'][os.path.join(td, '.freeze_lock.json')]
        assert out['status'] == 'ok', out
        assert rec['manifest_changed_since_lock'] is False
        assert abs(rec['alpha_effective'] - 0.025) < 1e-15
        assert rec['confirmed_effective'] == ['c1']


def test_reread_reason_is_recorded_by_the_freeze_lock():
    with tempfile.TemporaryDirectory() as td:
        man = {'manifest_sha256': 'AAA', 'frozen_at': '2026-10-30'}
        V12._check_or_write_freeze_lock(td, man, None, stage='preaccess')
        V12._check_or_write_freeze_lock(td, man, 'FP', stage='accessed')
        old = {k: os.environ.get(k) for k in
               ('V12_REFREEZE', 'V12_ALLOW_TEST_REREAD', 'V12_TEST_REREAD_REASON')}
        try:
            os.environ['V12_REFREEZE'] = '1'
            os.environ['V12_ALLOW_TEST_REREAD'] = '1'
            os.environ['V12_TEST_REREAD_REASON'] = 'fixed a broken eval loop'
            lock = V12._check_or_write_freeze_lock(
                td, man, None, stage='preaccess')
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        assert lock['test_reads'] == 2
        rereads = [h for h in lock['history'] if h.get('stage') == 'reread']
        assert rereads and rereads[-1]['reason'] == 'fixed a broken eval loop'
        a = L.confirmatory_access_audit(lock)
        assert a['n_rereads_without_reason'] == 0
        assert a['confirmatory_allowed'] is False


def test_confirmation_and_report_are_wired_to_the_access_audit():
    src = inspect.getsource(V12.finaltest_stats)
    assert 'confirmatory_access_audit' in src
    assert "stats['test_access']" in src
    assert "stats['confirmed_effective']" in src
    assert 'freeze_lock' in src
    rep = inspect.getsource(V12.build_report)
    assert 'confirmed_effective' in rep
    assert 'test_access' in rep
    assert 'test_access' in inspect.getsource(A.run_report)
    cli = inspect.getsource(A._run_cli)
    assert "'test_access': audit_test_access" in cli
    assert "'heldout': audit_test_access" in cli
    assert "'test_access', 'test_access_audit'" in cli


def test_manifest_loader_validates_the_access_policy_before_any_read():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'manifest.json')
        _write_manifest(
            path, list(range(12)),
            extra_stat={'test_access_policy': {
                'method': 'single_look', 'max_looks': 2}})
        try:
            V12._load_manifest(path)
        except SystemExit as exc:
            assert 'test_access_policy' in str(exc), str(exc)
        else:
            raise AssertionError('an invalid access policy was accepted')

        _write_manifest(
            path, list(range(12)),
            extra_stat={'test_access_policy': {
                'method': 'bonferroni', 'max_looks': 2}})
        loaded = V12._load_manifest(path)
        assert L.test_access_policy(loaded)['max_looks'] == 2
