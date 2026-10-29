"""CPU checks for byte-reproducible evidence archives (archive_release.py).

A release archive is only independently checkable if a third party can rebuild
the published object and compare hashes.  The default tar.gz records build
time in member mtimes and in the gzip header, so the same evidence packed
twice used to differ.  These checks pin the SOURCE_DATE_EPOCH contract, the
metadata-free payload digest, the verify-side replay and the fail-closed
behaviour of a malformed/absent timestamp.
"""
import io
import json
import os
import shutil
import tarfile
import tempfile
import sys as _sys

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

import pytest

if not _HAVE_TORCH:
    pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)

import archive_release as AR


@pytest.fixture
def repo_tmp():
    """A temporary directory under the repo (the archiver is repo-confined)."""
    td = tempfile.mkdtemp(dir=REPO, prefix='archive_test_')
    try:
        yield td
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _result_dir(td):
    results = os.path.join(td, 'results_x')
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump({'arm::seed0': {'ppl': 10.0}}, f)
    return os.path.basename(td), os.path.join(os.path.basename(td), 'results_x')


def _build(td, tag, epoch, outname='out'):
    rel_root, rel_results = _result_dir(td)
    previous = os.environ.get('SOURCE_DATE_EPOCH')
    os.environ['SOURCE_DATE_EPOCH'] = str(epoch)
    try:
        outdir = os.path.join(rel_root, outname)
        assert AR.cmd_make(tag, [rel_results], outdir=outdir) == 0
    finally:
        if previous is None:
            os.environ.pop('SOURCE_DATE_EPOCH', None)
        else:
            os.environ['SOURCE_DATE_EPOCH'] = previous
    return os.path.join(td, outname, f'release_{tag}.tar.gz')


def _manifest(path):
    with tarfile.open(path, 'r:gz') as tf:
        name = next(n for n in tf.getnames()
                    if n.endswith('.MANIFEST.sha256'))
        return json.load(tf.extractfile(name))


def _members(path):
    with tarfile.open(path, 'r:gz') as tf:
        return tf.getmembers()


def test_source_date_epoch_is_strict(monkeypatch):
    monkeypatch.delenv('SOURCE_DATE_EPOCH', raising=False)
    assert AR._source_date_epoch() is None
    assert AR._source_date_epoch('') is None
    monkeypatch.setenv('SOURCE_DATE_EPOCH', '1700000000')
    assert AR._source_date_epoch() == 1700000000
    # An explicit value is a deliberate override, not a second environment.
    assert AR._source_date_epoch('0') == 0
    for bad in ('-1', '1.5', '1e9', ' 12x', '2023-11-14', True):
        with pytest.raises(ValueError):
            AR._source_date_epoch(bad)
    with pytest.raises(ValueError):
        AR._source_date_epoch(AR._MAX_SOURCE_DATE_EPOCH + 1)
    assert AR._iso_utc(0) == '1970-01-01T00:00:00Z'


def test_same_epoch_rebuild_is_byte_identical(repo_tmp, monkeypatch):
    td = repo_tmp
    p1 = _build(td, 'twice', 1700000000, 'out')
    first = open(p1, 'rb').read()
    man1 = _manifest(p1)
    p2 = _build(td, 'twice', 1700000000, 'out')
    second = open(p2, 'rb').read()
    man2 = _manifest(p2)
    assert first == second
    assert man1['created_at'] == '2023-11-14T22:13:20Z'
    assert man1['payload_sha256'] == man2['payload_sha256']
    assert man1['reproducibility']['deterministic'] is True
    assert man1['reproducibility']['source_date_epoch'] == 1700000000
    assert man1['reproducibility']['member_mtime'] == 1700000000
    assert man1['reproducibility']['gzip_mtime'] == 1700000000
    # Every member, including the manifest, is pinned; owner fields are fixed.
    for member in _members(p1):
        assert member.mtime == 1700000000, member.name
        assert (member.uid, member.gid) == (0, 0), member.name
        assert not member.uname and not member.gname, member.name
    assert AR._gzip_mtime(p1) == 1700000000
    assert AR.cmd_verify(p1) == 0
    assert list(man1['files']) == sorted(man1['files'])


def test_payload_identity_is_independent_of_packaging_epoch(repo_tmp, monkeypatch):
    td = repo_tmp
    p1 = _build(td, 'epoch_a', 1700000000, 'a')
    p2 = _build(td, 'epoch_a', 1700000001, 'b')
    assert open(p1, 'rb').read() != open(p2, 'rb').read()
    m1, m2 = _manifest(p1), _manifest(p2)
    assert m1['payload_sha256'] == m2['payload_sha256']
    assert m1['payload_algorithm'] == AR._PAYLOAD_ALGORITHM
    assert AR.cmd_verify(p1) == 0
    assert AR.cmd_verify(p2) == 0


def test_payload_digest_detects_file_map_tampering(repo_tmp, monkeypatch):
    td = repo_tmp
    path = _build(td, 'payload', 1700000000)
    man = _manifest(path)
    assert AR._payload_problems(man) == []
    bad = json.loads(json.dumps(man))
    first = sorted(bad['files'])[0]
    bad['files'][first]['sha256'] = '0' * 64
    assert any('recompute' in p for p in AR._payload_problems(bad))
    bad = json.loads(json.dumps(man))
    bad['payload_algorithm'] = 'md5'
    assert any('algorithm' in p for p in AR._payload_problems(bad))
    # The byte count is part of the payload identity.  Deleting it must not
    # silently shrink the digest's coverage (the old reader skipped it).
    bad = json.loads(json.dumps(man))
    del bad['files'][first]['bytes']
    assert any('byte count' in p for p in AR._payload_problems(bad))
    # A legacy manifest without the content digest is not a tampering finding.
    legacy = {k: v for k, v in man.items()
              if k not in ('payload_sha256', 'payload_algorithm')}
    assert AR._payload_problems(legacy) == []


def test_reproducibility_contract_replays_metadata(repo_tmp, monkeypatch):
    td = repo_tmp
    path = _build(td, 'meta', 1700000000)
    man = _manifest(path)
    members = _members(path)
    assert AR._reproducibility_problems(man, path, members) == []

    bad = json.loads(json.dumps(man))
    bad['created_at'] = '2026-01-01T00:00:00Z'
    assert any('created_at' in p for p in
               AR._reproducibility_problems(bad, path, members))

    bad = json.loads(json.dumps(man))
    bad['reproducibility']['deterministic'] = 'yes'
    assert any('boolean' in p for p in
               AR._reproducibility_problems(bad, path, members))

    bad = json.loads(json.dumps(man))
    bad['reproducibility']['gzip_mtime'] = 1
    assert any('gzip_mtime' in p for p in
               AR._reproducibility_problems(bad, path, members))

    tampered_members = _members(path)
    tampered_members[0].mtime = 1
    assert any('member mtime' in p for p in
               AR._reproducibility_problems(man, path, tampered_members))

    # A non-deterministic manifest must not borrow a deterministic stamp.
    bad = json.loads(json.dumps(man))
    bad['reproducibility']['deterministic'] = False
    assert any('must not claim' in p for p in
               AR._reproducibility_problems(bad, path, members))


def _repack_with_manifest(src, dst, epoch, mutate):
    """Rewrite an archive with a modified manifest, keeping deterministic framing."""
    with tarfile.open(src, 'r:gz') as tf:
        entries = []
        for member in tf.getmembers():
            data = tf.extractfile(member).read() if member.isfile() else None
            entries.append((member, data))
    with AR._open_archive(dst, epoch) as out:
        for member, data in entries:
            if member.name.endswith('.MANIFEST.sha256'):
                man = json.loads(data)
                mutate(man)
                data = json.dumps(man, ensure_ascii=False, indent=2).encode('utf-8')
                member.size = len(data)
            AR._normalize_tarinfo(member, epoch)
            out.addfile(member, io.BytesIO(data))


def test_verify_fails_closed_on_tampered_metadata(repo_tmp, monkeypatch):
    td = repo_tmp
    path = _build(td, 'tamper', 1700000000)

    def _bad_payload(man):
        man['payload_sha256'] = '0' * 64

    bad = os.path.join(td, 'bad_payload.tar.gz')
    _repack_with_manifest(path, bad, 1700000000, _bad_payload)
    assert AR.cmd_verify(bad) == 1

    def _bad_time(man):
        man['created_at'] = '2026-01-01T00:00:00Z'

    bad = os.path.join(td, 'bad_time.tar.gz')
    _repack_with_manifest(path, bad, 1700000000, _bad_time)
    assert AR.cmd_verify(bad) == 1


def test_reproducible_flag_refuses_missing_epoch(repo_tmp, monkeypatch):
    td = repo_tmp
    _rel_root, rel_results = _result_dir(td)
    monkeypatch.delenv('SOURCE_DATE_EPOCH', raising=False)
    outdir = os.path.join(os.path.basename(td), 'out')
    assert AR.cmd_make('need_epoch', [rel_results], outdir=outdir,
                       reproducible=True) == 1
    assert not os.path.exists(
        os.path.join(td, 'out', 'release_need_epoch.tar.gz'))


def test_wall_clock_archive_still_verifies(repo_tmp, monkeypatch):
    td = repo_tmp
    _rel_root, rel_results = _result_dir(td)
    monkeypatch.delenv('SOURCE_DATE_EPOCH', raising=False)
    outdir = os.path.join(os.path.basename(td), 'wall')
    assert AR.cmd_make('wall', [rel_results], outdir=outdir) == 0
    path = os.path.join(td, 'wall', 'release_wall.tar.gz')
    man = _manifest(path)
    assert man['reproducibility']['deterministic'] is False
    assert man['payload_sha256']
    assert AR.cmd_verify(path) == 0
