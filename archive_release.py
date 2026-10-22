#!/usr/bin/env python3
"""archive_release.py — 不可变证据归档。

问题：README/设计文档称决策日志与完整证据入库，但仓库只保留源码——
复跑能力不等于可审计证据。源码仓库边界（design_notes §8：量值不入库）
仍然成立，因此本工具把「每个正式实验版本的可审计材料」打包为**带
SHA-256 清单的固定归档**，可发布到任何只读位置（release 附件、对象
存储、数据仓），而 Git 主仓仍只承载代码。

归档内容（对一个 tag/版本）：
  - 各 results_*/summary.json、bylen/distractor/niah 等落盘产物
  - analysis_*/ 全部统计 JSON 与 REPORT*.md
  - confirmation_frozen.json（+ 冻结锁 .freeze_lock.json 与 ckpt 哈希表）
  - run_time_state*.json（TimeGuard 账本）
  - 环境：python/pip/torch/CUDA 版本、requirements 快照
  - checkpoint 哈希清单（sha256 逐文件；可选把 .pt 本体一并入包）
  - git 状态：HEAD 哈希、dirty 文件列表

用法：
  python archive_release.py plan v13 --dirs results_lm_test analysis_v12
      # 只列出会入包的文件与总大小（dry-run）
  python archive_release.py make v13 --dirs results_lm_test analysis_v12 \
      [--include-ckpt] [--out archive/]
      # 产出 archive/release_v13.tar.gz + release_v13.MANIFEST.sha256
  python archive_release.py verify archive/release_v13.tar.gz

verify 独立于 make：任何拿到归档的人都能校验完整性。
"""
import argparse
import glob
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
import time

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except (ValueError, OSError):
            pass

REPO = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIRS = ['results_lm_test', 'analysis_v12']
_CKPT_SUFFIXES = ('.pt', '.ckpt', '.pth', '.safetensors')

ENV_FILES = ['confirmation_frozen.json', 'DECISION_LOG.md', 'RELEASES.md', 'requirements.txt',
            'requirements.lock', 'pyproject.toml',
            # data/cache provenance: a tokenizer or cache that changes the ids
            # must be visible in the archive, not only in a local directory.
            'wt103_cache/test_meta.json', 'wt103_cache/train_meta.json',
            'wt103_cache/meta.json', 'wt103_cache/val_meta.json']


_RELEASE_TAG_RE = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z')


def _release_tag(tag):
    """A tag is a filename component, not a path.

    ``release_{tag}.tar.gz`` is written into the output directory; a tag such
    as ``../x`` or ``a/b`` would escape that directory (or fail halfway
    through), so only a plain identifier is accepted.
    """
    if not isinstance(tag, str):
        raise ValueError(f'release tag must be a string, got {type(tag).__name__}')
    s = tag
    if not _RELEASE_TAG_RE.fullmatch(s):
        raise ValueError(
            f'release tag {tag!r} is not a safe filename component; use '
            'letters, digits, dot, underscore or hyphen (max 64 chars, '
            'starting with a letter/digit)')
    return s


_CACHE_PROVENANCE_PATTERNS = (
    'meta*.json',          # legacy meta.json / train_meta.json
    'test_meta*.json',     # tag-paired test metadata (test_meta_v2_*.json)
    'tok*.json',           # persisted train tokenizer (tok_v5_*.json)
    'test_tok*.json',      # persisted test tokenizer
    '*.idsfp.json',        # id-cache content fingerprints
)


def _safe_outdir(outdir):
    """Resolve a release output directory under the repository.

    The tag and every packed directory are already confined; `--out` was not,
    so `--out ../x` (or an absolute path) could write the archive outside the
    repository.  Keep the same containment rule as `_collect`.
    """
    if not isinstance(outdir, str) or not outdir.strip():
        raise ValueError(f'archive output directory must be a non-empty path, got {outdir!r}')
    _repo_real = os.path.realpath(REPO)
    _out = os.path.realpath(os.path.join(REPO, outdir))
    try:
        _inside = os.path.commonpath([_repo_real, _out]) == _repo_real
    except ValueError:
        _inside = False
    if not _inside:
        raise ValueError(
            f'archive output directory {outdir!r} resolves outside the repository '
            f'({_out!r}); write releases under {REPO!r}')
    return _out


def _cache_provenance_files(cache_dir=None):
    """Metadata/tokenizer files that make a corpus cache auditable.

    The loaders write TAG-PAIRED names (``meta_v5_*.json``,
    ``test_meta_v2_*.json``, ``tok_v5_*.json``).  The old fixed filename list
    in ENV_FILES matched none of them, so a release could claim to pack cache
    provenance while silently omitting every real metadata file.  Token-id
    arrays are deliberately NOT packed here (they are large); their
    fingerprints and the tokenizer identities are.
    """
    d = cache_dir or os.path.join(REPO, 'wt103_cache')
    if not os.path.isdir(d):
        return []
    out = set()
    for pat in _CACHE_PROVENANCE_PATTERNS:
        out.update(glob.glob(os.path.join(d, pat)))
    return sorted(p for p in out if os.path.isfile(p))


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _atomic_write_bytes(path, data):
    """Write a complete file and only then replace the previous artifact."""
    tmp = path + f'.tmp{os.getpid()}'
    try:
        with open(tmp, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _git_state():
    def _run(*args):
        try:
            return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            return None
    head = _run('rev-parse', 'HEAD')
    if head is None:
        return {'git': 'not a git repo / git unavailable'}
    dirty = _run('status', '--porcelain') or ''
    return {'git_head': head, 'git_branch': _run('rev-parse', '--abbrev-ref', 'HEAD'), 'dirty_files': dirty.splitlines()}


def _env_snapshot():
    env = {'python': sys.version.split()[0], 'platform': platform.platform(), 'captured_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
    try:
        import torch
        env['torch'] = torch.__version__
        env['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env['cuda'] = torch.version.cuda
            env['gpu'] = torch.cuda.get_device_name(0)
    except Exception as e:
        env['torch'] = f'unavailable ({type(e).__name__})'
    try:
        _req = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, timeout=120)
        if _req.returncode == 0:
            env['pip_freeze_head'] = _req.stdout.splitlines()[:400]
    except Exception:
        pass
    return env


def _collect(dirs, include_ckpt=False):
    """Files that would enter the archive, with sizes and hashes (lazily)."""
    files = []
    _repo_real = os.path.realpath(REPO)
    for d in dirs:
        base = os.path.realpath(os.path.join(REPO, d))
        try:
            _inside = os.path.commonpath([_repo_real, base]) == _repo_real
        except ValueError:
            _inside = False
        if not _inside:
            # `--dirs /tmp/x` or `../..` would otherwise pack host files under
            # `../../...` arcnames, producing an archive that escapes the
            # repository on extraction.
            raise ValueError(f'archive directory {d!r} resolves outside the repository '
                             f'({base!r}); only paths under {REPO!r} may be packed')
        if not os.path.isdir(base):
            print(f'[archive] WARNING: {d} does not exist locally — skipped (run on the experiment machine)')
            continue
        for root, _dirs, names in os.walk(base):
            for n in names:
                p = os.path.join(root, n)
                if not include_ckpt and n.endswith(_CKPT_SUFFIXES):
                    continue
                files.append(p)
    for f in ENV_FILES:
        p = os.path.join(REPO, f)
        if os.path.exists(p):
            files.append(p)
    files.extend(_cache_provenance_files())
    for f in ('results_lm_test/.freeze_lock.json',):
        p = os.path.join(REPO, f)
        if os.path.exists(p):
            files.append(p)
    # Training logs (gitignored, so they only exist on the experiment machine).
    # Pack every per-seed log in the evidence release; pack every
    # run_*.log at the repo root rather than trusting a hand-written list.
    try:
        for n in sorted(os.listdir(REPO)):
            if n.startswith('run_') and n.endswith('.log'):
                files.append(os.path.join(REPO, n))
    except OSError:
        pass
    # TimeGuard ledger / run-state files are part of the evidence the release
    # claims to preserve (module docstring), but they are gitignored and live
    # only on the experiment machine.
    files.extend(sorted(glob.glob(os.path.join(REPO, 'run_time_state*.json'))))
    # The freeze lock can already be reached by the directory walk above;
    # duplicate tar members with the same name are ambiguous on extraction,
    # so keep the first occurrence only.
    seen = set()
    unique = []
    for p in files:
        key = os.path.abspath(p)
        if key in seen:
            continue
        if os.path.islink(p):
            # tarfile stores the link itself, while the manifest hashes its
            # target; verify() then (correctly) refuses the non-regular member.
            # Fail at collect time instead of producing an unverifiable archive.
            raise ValueError(
                f'archive file {p!r} is a symbolic link; pack the target '
                'explicitly (a link member cannot be hash-verified as a '
                'regular file)')
        if not os.path.isfile(p):
            raise ValueError(
                f'archive file {p!r} is not a regular file; refusing to pack it')
        _p_real = os.path.realpath(p)
        try:
            _inside = os.path.commonpath([_repo_real, _p_real]) == _repo_real
        except ValueError:
            _inside = False
        if not _inside:
            raise ValueError(f'archive file {p!r} resolves outside the repository '
                             f'({_p_real!r}); refusing to pack it')
        seen.add(key)
        unique.append(p)
    return unique


def _ckpt_hash_table(dirs):
    """sha256 of every checkpoint file referenced by the release (even when the
    .pt bodies are not packed) — the weights stay auditable by hash."""
    rows = {}
    for d in dirs:
        base = os.path.join(REPO, d)
        if not os.path.isdir(base):
            continue
        for root, _dirs, names in os.walk(base):
            for n in names:
                if n.endswith(_CKPT_SUFFIXES):
                    p = os.path.join(root, n)
                    rows[os.path.relpath(p, REPO)] = _sha256(p)
    return rows


def cmd_plan(tag, dirs, include_ckpt=False):
    tag = _release_tag(tag)
    files = _collect(dirs, include_ckpt)
    total = 0
    print(f'[archive] plan for release {tag!r} ({len(files)} files):')
    for p in files:
        sz = os.path.getsize(p)
        total += sz
        print(f'  {sz:>12,d} B  {os.path.relpath(p, REPO)}')
    ck = _ckpt_hash_table(dirs)
    if ck:
        print(f'[archive] checkpoints found: {len(ck)} — hashed into the manifest' + (' and packed' if include_ckpt else ' (bodies NOT packed; use --include-ckpt to embed them)'))
    print(f'[archive] total: {total:,d} B (~{total / (1 << 20):.1f} MiB)')
    return files


def cmd_make(tag, dirs, include_ckpt=False, outdir='archive'):
    tag = _release_tag(tag)
    _required = ['DECISION_LOG.md', 'RELEASES.md', 'confirmation_frozen.json']
    _missing = [f for f in _required if not os.path.exists(os.path.join(REPO, f))]
    if _missing:
        print(f'[archive] REFUSE: formal release is missing required provenance file(s): {_missing}')
        return 1
    _present = [d for d in dirs
                if os.path.isdir(os.path.join(REPO, d))]
    if not _present:
        print('[archive] REFUSE: none of the named result directories exist '
              f'here ({list(dirs)}); a release of provenance files alone would '
              'contain no evidence — run this on the machine holding the '
              'artifacts')
        return 1
    files = _collect(dirs, include_ckpt)
    if not files:
        print('[archive] REFUSE: nothing to archive — the named result dirs do not exist here; run this on the machine holding the artifacts')
        return 1
    _outdir = _safe_outdir(outdir)
    os.makedirs(_outdir, exist_ok=True)
    base_name = f'release_{tag}'
    tar_path = os.path.join(_outdir, f'{base_name}.tar.gz')
    man = {'release': base_name, 'tag': tag, 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'created_by': 'archive_release.py', 'git': _git_state(), 'environment': _env_snapshot(), 'dirs': list(dirs), 'include_ckpt_bodies': bool(include_ckpt), 'files': {}, 'checkpoint_hashes': _ckpt_hash_table(dirs)}
    # Build the archive beside its final name; an interrupted or failed make
    # must not leave a truncated `release_<tag>.tar.gz` (or destroy a previous
    # good release with the same tag).
    tmp_tar = tar_path + f'.tmp{os.getpid()}'
    try:
        with tarfile.open(tmp_tar, 'w:gz') as tf:
            for p in files:
                rel = os.path.relpath(p, REPO)
                man['files'][rel] = {'sha256': _sha256(p), 'bytes': os.path.getsize(p)}
                tf.add(p, arcname=rel)
            man_path_rel = f'{base_name}.MANIFEST.sha256'
            man_bytes = json.dumps(man, ensure_ascii=False, indent=2).encode('utf-8')
            import io
            ti = tarfile.TarInfo(man_path_rel)
            ti.size = len(man_bytes)
            ti.mtime = int(time.time())
            tf.addfile(ti, io.BytesIO(man_bytes))
        tar_sha = _sha256(tmp_tar)
        os.replace(tmp_tar, tar_path)
    except BaseException:
        try:
            os.unlink(tmp_tar)
        except OSError:
            pass
        raise
    # external copy of the manifest + the archive's own hash
    man_ext = os.path.join(_outdir, man_path_rel)
    _atomic_write_bytes(man_ext, man_bytes)
    _atomic_write_bytes(os.path.join(_outdir, f'{base_name}.tar.gz.sha256'),
                        f'{tar_sha}  {os.path.basename(tar_path)}\n'.encode('utf-8'))
    print(f'[archive] wrote {os.path.relpath(tar_path, REPO)} ({os.path.getsize(tar_path):,d} B)')
    print(f'[archive] wrote {man_path_rel} ({len(man["files"])} files, {len(man["checkpoint_hashes"])} ckpt hashes)')
    print(f'[archive] archive sha256: {tar_sha}')
    print('[archive] publish the tar.gz + MANIFEST + .sha256 to a read-only location (release attachment / object store); the git repo itself stays source-only (design_notes §8)')
    _atomic_write_bytes(
        os.path.join(_outdir, f'{base_name}.release_record.json'),
        json.dumps({'release': base_name, 'tag': tag, 'created_at': man['created_at'],
                    'tar_sha256': tar_sha, 'n_files': len(man['files']),
                    'n_ckpt_hashes': len(man['checkpoint_hashes']),
                    'git_head': (man['git'] or {}).get('git_head')},
                   ensure_ascii=False, indent=2).encode('utf-8'))
    return 0


def _unsafe_member_name(name):
    """True when a tar member name could escape the extraction directory."""
    if not isinstance(name, str) or not name or '\\' in name:
        return True
    if name.startswith('/'):
        return True
    parts = name.split('/')
    return any(part in ('', '.', '..') for part in parts)


def cmd_verify(tar_path):
    if not os.path.exists(tar_path):
        print(f'[archive] REFUSE: {tar_path} not found')
        return 1
    side = tar_path + '.sha256'
    if os.path.exists(side):
        with open(side, encoding='utf-8') as _sf:
            _side_fields = _sf.read().split()
        if not _side_fields:
            print('[archive] NOTE: the .sha256 sidecar is empty; falling back to the manifest hashes inside the archive')
        else:
            want = _side_fields[0]
            got = _sha256(tar_path)
            print(f'[archive] archive sha256: {got}  {"OK" if got == want else "MISMATCH (expected " + want + ")"}')
            if got != want:
                return 1
    try:
        _tf = tarfile.open(tar_path, 'r:gz')
    except (tarfile.TarError, OSError, EOFError) as e:
        print(f'[archive] REFUSE: cannot read the archive ({type(e).__name__}: {e})')
        return 1
    with _tf as tf:
        names = tf.getnames()
        if len(names) != len(set(names)):
            print('[archive] REFUSE: the archive contains duplicate member names')
            return 1
        _bad_names = [n for n in names if _unsafe_member_name(n)]
        if _bad_names:
            print('[archive] REFUSE: unsafe member name(s) would escape the '
                  f'extraction directory: {_bad_names[:5]}')
            return 1
        man_name = next((n for n in names if n.endswith('.MANIFEST.sha256')), None)
        if man_name is None:
            print('[archive] REFUSE: no MANIFEST inside the archive')
            return 1
        _man_member = tf.getmember(man_name)
        if not _man_member.isfile():
            print('[archive] REFUSE: the MANIFEST member is not a regular file')
            return 1
        try:
            man = json.load(tf.extractfile(man_name))
        except (TypeError, ValueError, UnicodeDecodeError) as e:
            print(f'[archive] REFUSE: the MANIFEST is not valid JSON ({type(e).__name__}: {e})')
            return 1
        if not isinstance(man, dict) or not isinstance(man.get('files'), dict):
            print('[archive] REFUSE: the MANIFEST has no files mapping')
            return 1
        def _bad_file_entry(meta):
            if not isinstance(meta, dict):
                return True
            sha = meta.get('sha256')
            if not isinstance(sha, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', sha):
                return True
            size = meta.get('bytes')
            return (size is not None
                    and (isinstance(size, bool) or not isinstance(size, int) or size < 0))
        _bad_entries = [n for n, meta in man['files'].items()
                        if _bad_file_entry(meta)]
        if _bad_entries:
            print('[archive] REFUSE: the MANIFEST has malformed file entr(ies) '
                  f'(expected objects with a 64-hex sha256 and a non-negative '
                  f'integer byte count): {_bad_entries[:5]}')
            return 1
        _ck_raw = man.get('checkpoint_hashes')
        if _ck_raw is not None and not isinstance(_ck_raw, dict):
            print('[archive] REFUSE: the MANIFEST checkpoint_hashes is malformed '
                  f'(expected an object, got {type(_ck_raw).__name__})')
            return 1
        _bad_manifest_names = [n for n in man['files'] if _unsafe_member_name(n)]
        if _bad_manifest_names:
            print('[archive] REFUSE: unsafe manifest path(s): '
                  f'{_bad_manifest_names[:5]}')
            return 1
        bad = []
        for member in tf.getmembers():
            if member.name == man_name:
                continue
            if not member.isfile():
                # A symlink/hardlink/device member was previously skipped and
                # then counted as "present" because its name was in
                # tf.getnames(), so a manifest entry could be replaced by a
                # link that was never hashed.
                bad.append(f'{member.name}: member is not a regular file '
                           f'(type {member.type!r})')
                continue
            f = tf.extractfile(member)
            h = hashlib.sha256()
            while True:
                b = f.read(1 << 20)
                if not b:
                    break
                h.update(b)
            got = h.hexdigest()
            _meta = man['files'].get(member.name) or {}
            want = _meta.get('sha256')
            if want is None:
                bad.append(f'{member.name}: in tar but NOT in manifest')
            elif got != want:
                bad.append(f'{member.name}: hash mismatch (manifest {want[:12]}…, actual {got[:12]}…)')
            elif _meta.get('bytes') is not None and int(_meta['bytes']) != int(member.size):
                bad.append(f"{member.name}: byte count mismatch (manifest {_meta['bytes']}, actual {member.size})")
        missing = [n for n in man['files'] if n not in names]
        for n in missing:
            bad.append(f'{n}: in manifest but NOT in tar')
        ck = _ck_raw or {}
        _bad_ck = [n for n, h in ck.items()
                   if not isinstance(h, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', h)]
        if _bad_ck:
            print('[archive] REFUSE: the MANIFEST has malformed checkpoint hash '
                  f'entr(ies): {_bad_ck[:5]}')
            return 1
        print(f'[archive] verified {len(names)} members against the manifest; {len(ck)} checkpoint hashes recorded' + (' (bodies not packed — verify on the experiment machine by hash)' if ck else ''))
        if bad:
            print('[archive] FAIL:\n  - ' + '\n  - '.join(bad[:20]))
            return 1
        _git = man.get('git')
        _git_head = _git.get('git_head') if isinstance(_git, dict) else None
        print('[archive] VERIFY OK — release ' + str(man.get('tag')) + ' created ' + str(man.get('created_at')) + ' @ git ' + str(_git_head))
        return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description='immutable evidence archive for formal experiment releases')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p_plan = sub.add_parser('plan', help='dry-run: list what a release would contain')
    p_plan.add_argument('tag')
    p_plan.add_argument('--dirs', nargs='*', default=DEFAULT_DIRS)
    p_plan.add_argument('--include-ckpt', action='store_true')
    p_make = sub.add_parser('make', help='build the tar.gz + sha256 manifest')
    p_make.add_argument('tag')
    p_make.add_argument('--dirs', nargs='*', default=DEFAULT_DIRS)
    p_make.add_argument('--include-ckpt', action='store_true')
    p_make.add_argument('--out', default='archive')
    p_ver = sub.add_parser('verify', help='verify an archive against its manifest')
    p_ver.add_argument('tar')
    args = ap.parse_args(argv)
    if args.cmd == 'plan':
        cmd_plan(args.tag, args.dirs, args.include_ckpt)
        return 0
    if args.cmd == 'make':
        return cmd_make(args.tag, args.dirs, args.include_ckpt, args.out)
    if args.cmd == 'verify':
        return cmd_verify(args.tar)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
