"""CPU checks for executed-source registration (split packages).

`explib` and the drivers are assembled by executing `_parts/*.py` in the order
declared by a literal `_PARTS`.  A file-tree digest sees the bytes but not the
registration: a dead fragment or a reordered list changes the program while
every file hash (and `code_stamp`) stays identical.  These tests exercise the
derived registry identity, its use in source provenance, the reproduction
comparator and the freeze lock.  No training, GPU or test split is read.
"""
import copy
import json
import os
import subprocess
import sys
import tarfile
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403
from test_review_part_23 import _record

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


def _write(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(text)
    return path


def _split_pkg(root, parts, order=None, hist='legacy.py', extra_parts=()):
    """A minimal split package whose declaration is a plain literal."""
    pkg = os.path.join(root, 'pkg')
    order = list(order if order is not None else parts)
    _write(os.path.join(pkg, '__init__.py'),
           'from _split_loader import load_driver_package\n'
           f'_PARTS = {order!r}\n'
           "load_driver_package(globals(), __file__, _PARTS, "
           f"historical_path={hist!r})\n")
    for name in list(parts) + list(extra_parts):
        _write(os.path.join(pkg, '_parts', name), f'NAME = {name!r}\n')
    if hist is not None:
        _write(os.path.join(root, hist), '# historical entry point\n')
    return pkg


def test_every_real_split_package_is_fully_registered():
    reg = L.split_registry_manifest()
    assert reg['status'] == 'ok', reg['problems']
    assert reg['n_packages'] >= 13, reg['n_packages']
    assert reg['n_parts'] == sum(p['n_parts'] for p in reg['packages'].values())
    # Membership is exactly the fragments on disk: no dead source, no missing
    # registration.  Count from the tree so adding a fragment cannot silently
    # pass by changing the expected constant here.
    on_disk = 0
    for _pkg_dir, parts_dir in L._split_package_dirs(REPO):
        on_disk += sum(1 for n in os.listdir(parts_dir) if n.endswith('.py'))
    assert reg['n_parts'] == on_disk, (reg['n_parts'], on_disk)
    assert 'explib' in reg['packages'] and '_audit_gates' in reg['packages']
    assert reg['sha256'] == L.split_registry_manifest()['sha256']


def test_registry_tracks_order_and_membership_where_file_digest_cannot():
    with tempfile.TemporaryDirectory() as td:
        _split_pkg(td, ['a.py', 'b.py'], order=['b.py', 'a.py'])
        first = L.split_registry_manifest(root=td)
        tree_first = L.source_tree_manifest(root=td)
        assert first['status'] == 'ok', first['problems']
        # Reordering the declaration changes the source digest (the literal
        # lives in __init__.py) AND the executed-program identity.  The
        # registry is the part that says *why*: order and membership, not the
        # incidental bytes of the declaration file.
        _split_pkg(td, ['a.py', 'b.py'], order=['a.py', 'b.py'])
        second = L.split_registry_manifest(root=td)
        tree_second = L.source_tree_manifest(root=td)
        assert tree_first['sha256'] != tree_second['sha256']
        assert first['sha256'] != second['sha256']
        # A comment-only edit changes the file-tree digest but not the
        # executed registration: the registry keeps the two questions apart.
        _write(os.path.join(td, 'pkg', '__init__.py'),
               '# a comment that does not change registration\n'
               'from _split_loader import load_driver_package\n'
               "_PARTS = ['a.py', 'b.py']\n"
               "load_driver_package(globals(), __file__, _PARTS, "
               "historical_path='legacy.py')\n")
        third = L.split_registry_manifest(root=td)
        assert third['sha256'] == second['sha256']
        assert L.source_tree_manifest(root=td)['sha256'] != tree_second['sha256']
        # A fragment present on disk but omitted from _PARTS is dead source
        # and must be named, not silently ignored.
        _split_pkg(td, ['a.py', 'b.py', 'orphan.py'],
                   order=['a.py', 'b.py'])
        orphan = L.split_registry_manifest(root=td)
        assert orphan['status'] == 'invalid', orphan
        assert any('not registered: orphan.py' in p for p in orphan['problems']), orphan
        # A listed fragment that does not exist fails closed.
        _split_pkg(td, ['a.py'], order=['a.py', 'missing.py'])
        missing = L.split_registry_manifest(root=td)
        assert missing['status'] == 'invalid', missing
        assert any('missing' in p for p in missing['problems']), missing
        # Duplicate and path-like registrations are rejected, not deduplicated.
        _split_pkg(td, ['a.py', 'b.py'], order=['a.py', 'a.py'])
        duplicate = L.split_registry_manifest(root=td)
        assert duplicate['status'] == 'invalid', duplicate
        assert any('duplicate' in p for p in duplicate['problems']), duplicate
        _split_pkg(td, ['a.py'], order=['../escape.py'])
        escaped = L.split_registry_manifest(root=td)
        assert escaped['status'] == 'invalid', escaped
        assert any('plain *.py basename' in p for p in escaped['problems']), escaped
        # An out-of-tree symlink is executed by the loader but is invisible to
        # the in-tree source digest; the registry must refuse it.
        if hasattr(os, 'symlink'):
            with tempfile.TemporaryDirectory() as outside:
                target = _write(os.path.join(outside, 'outside.py'), 'X = 1\n')
                _split_pkg(td, ['b.py'], order=['a.py', 'b.py'])
                link = os.path.join(td, 'pkg', '_parts', 'a.py')
                if os.path.lexists(link):
                    os.unlink(link)
                try:
                    os.symlink(target, link)
                except (OSError, NotImplementedError):
                    pass
                else:
                    linked = L.split_registry_manifest(root=td)
                    assert linked['status'] == 'invalid', linked
                    assert any('outside the tree' in p for p in linked['problems']), linked
                    # The file-tree digest does not follow the link, so it
                    # alone would have accepted the external fragment.
                    assert 'pkg/_parts/a.py' not in L.source_tree_manifest(
                        root=td)['files']


def test_partial_source_scan_does_not_claim_a_registry():
    with tempfile.TemporaryDirectory() as td:
        _split_pkg(td, ['a.py'], order=['a.py'])
        full = L.source_tree_manifest(root=td)
        assert full['split_registry']['status'] == 'ok'
        partial = L.source_tree_manifest(root=td, include=['pkg/_parts/a.py'])
        assert partial['split_registry'] is None
        assert L.source_tree_summary(root=td)['split_registry']['status'] == 'ok'


def test_code_provenance_and_summary_carry_the_registry():
    cur = L.split_registry_manifest()
    summary = L.source_tree_summary()
    assert summary['split_registry']['sha256'] == cur['sha256']
    assert summary['split_registry']['status'] == 'ok'
    prov = L.code_provenance()
    assert (prov['code']['source_tree']['split_registry']['sha256']
            == cur['sha256'])
    # The plain file-tree digest is deliberately unchanged: old archives stay
    # verifiable, and the registry is an additional identity.
    assert summary['sha256'] == L.source_tree_manifest()['sha256']


def _with_registry(rec, registry):
    rec = copy.deepcopy(rec)
    rec['provenance']['code']['source_tree']['split_registry'] = registry
    return rec


def test_reproduction_distinguishes_registration_from_file_bytes():
    good = {'status': 'ok', 'sha256': 'a' * 64}
    other = {'status': 'ok', 'sha256': 'b' * 64}
    a = _with_registry(_record(), good)
    b = _with_registry(_record(), good)
    assert L.reproduction_diagnosis(a, b)['verdict'] == 'identical'
    changed = L.reproduction_diagnosis(a, _with_registry(_record(), other))
    assert changed['verdict'] == 'code_changed', changed
    assert changed['reproduced'] is False
    field = [f for f in changed['dimensions']['code']['fields']
             if f['field'] == 'split_registry'][0]
    assert field['status'] == 'mismatch', field
    # One-sided and present-but-invalid registries are unverified, never a
    # match; both-absent legacy records are named rather than promoted.
    one = L.reproduction_diagnosis(a, _record())
    assert one['verdict'] == 'provenance_incomplete', one
    invalid = L.reproduction_diagnosis(
        a, _with_registry(_record(), {'status': 'invalid', 'sha256': 'a' * 64}))
    assert invalid['verdict'] == 'provenance_incomplete', invalid
    legacy = L.reproduction_diagnosis(_record(), _record())
    assert legacy['verdict'] == 'identical', legacy
    assert any('predate split_registry' in n
               for d in legacy['dimensions'].values()
               for n in d.get('notes', [])), legacy


def test_freeze_lock_refuses_a_registration_change_without_byte_change():
    import explib.confirmation as V12
    with tempfile.TemporaryDirectory() as td:
        man = {'manifest_sha256': 'AAA', 'frozen_at': '2026-10-17'}
        src = L.source_tree_manifest()
        V12._check_or_write_freeze_lock(
            td, man, None, stage='preaccess', source_tree=src)
        changed = copy.deepcopy(src)
        changed['split_registry']['sha256'] = '0' * 64
        try:
            V12._check_or_write_freeze_lock(
                td, man, 'FP', stage='accessed', source_tree=changed)
            raise AssertionError('a changed registration must be refused')
        except SystemExit as e:
            message = str(e)
            assert 'measurement source changed' in message, message
            assert 'split-package registration' in message, message
        # A locally invalid registration is refused before the lock is used.
        invalid = copy.deepcopy(src)
        invalid['split_registry']['status'] = 'invalid'
        invalid['split_registry']['problems'] = ['dead fragment']
        try:
            V12._check_or_write_freeze_lock(
                td, man, 'FP', stage='analysis', source_tree=invalid)
            raise AssertionError('an invalid registration must be refused')
        except SystemExit as e:
            assert 'registration is invalid' in str(e), str(e)


def test_audit_gate_reports_local_and_recorded_identity():
    import audit_gates as A
    local = A.audit_split_registry()
    assert local['status'] == 'ok', local
    cur = L.split_registry_manifest()
    with tempfile.TemporaryDirectory() as td:
        match = os.path.join(td, 'match.json')
        with open(match, 'w', encoding='utf-8') as fh:
            json.dump({'provenance': {'code': {'source_tree': {
                'split_registry': {'status': 'ok', 'sha256': cur['sha256']}}}}}, fh)
        assert A.audit_split_registry([match])['status'] == 'ok'
        mismatch = os.path.join(td, 'mismatch.json')
        with open(mismatch, 'w', encoding='utf-8') as fh:
            json.dump({'provenance': {'code': {'source_tree': {
                'split_registry': {'status': 'ok', 'sha256': '0' * 64}}}}}, fh)
        bad = A.audit_split_registry([mismatch])
        assert bad['status'] == 'fail' and bad['n_mismatch'] == 1, bad
        absent = os.path.join(td, 'absent.json')
        with open(absent, 'w', encoding='utf-8') as fh:
            json.dump({'provenance': {'code': {'source_tree': {'sha256': 'x'}}}}, fh)
        assert A.audit_split_registry([absent])['status'] == 'not_verifiable'
    proc = subprocess.run([sys.executable, 'audit_gates.py', 'split_registry'],
                          cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)


def test_archive_verifier_recomputes_registry_from_packed_bytes():
    import archive_release as AR
    assert AR._SPLIT_REGISTRY_ALGORITHM == L.SPLIT_REGISTRY_ALGORITHM
    with tempfile.TemporaryDirectory() as td:
        _split_pkg(td, ['a.py', 'b.py'], order=['b.py', 'a.py'])
        st = L.source_tree_manifest(root=td)
        man = {'identity': {'source_tree': st}}
        tar_path = os.path.join(td, 'src.tar')
        with tarfile.open(tar_path, 'w') as tf:
            for rel in st['files']:
                tf.add(os.path.join(td, rel), arcname=rel)
        with tarfile.open(tar_path, 'r') as tf:
            assert AR._packed_split_registry_problems(man, tf) == []
            bad = copy.deepcopy(man)
            bad['identity']['source_tree']['split_registry']['sha256'] = '0' * 64
            problems = AR._packed_split_registry_problems(bad, tf)
            assert any('does not recompute' in p for p in problems), problems
            # Tampering with the recorded membership count is also refused.
            bad2 = copy.deepcopy(man)
            bad2['identity']['source_tree']['split_registry']['n_parts'] = 99
            assert any('n_parts' in p for p in
                       AR._packed_split_registry_problems(bad2, tf))
