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
import hashlib
import json
import os
import platform
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
ENV_FILES = ['confirmation_frozen.json', 'DECISION_LOG.md', 'RELEASES.md', 'requirements.txt',
            'requirements.lock', 'pyproject.toml',
            # data/cache provenance: a tokenizer or cache that changes the ids
            # must be visible in the archive, not only in a local directory.
            'wt103_cache/test_meta.json', 'wt103_cache/train_meta.json',
            'wt103_cache/meta.json', 'wt103_cache/val_meta.json']


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


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
    for d in dirs:
        base = os.path.join(REPO, d)
        if not os.path.isdir(base):
            print(f'[archive] WARNING: {d} does not exist locally — skipped (run on the experiment machine)')
            continue
        for root, _dirs, names in os.walk(base):
            for n in names:
                p = os.path.join(root, n)
                if os.path.islink(p) or not os.path.isfile(p):
                    # tarfile would store a symlink/device member, but the
                    # verifier only attests regular-file content.  Packing it
                    # would make `make` produce an archive its own `verify`
                    # rejects.  Leave it out instead of following it outside
                    # the repository.
                    print(f'[archive] WARNING: skipping non-regular file {os.path.relpath(p, REPO)} — only regular files are content-attested')
                    continue
                if not include_ckpt and (n.endswith('.pt') or n.endswith('.ckpt') or n.endswith('.pth') or n.endswith('.safetensors')):
                    continue
                files.append(p)
    for f in ENV_FILES:
        p = os.path.join(REPO, f)
        if os.path.exists(p) and not os.path.islink(p) and os.path.isfile(p):
            files.append(p)
    # The loaders write tag-suffixed metadata/tokenizer files (e.g.
    # meta_v5_sl512_....json), not the legacy fixed names above.  Pack those
    # small provenance files too; otherwise a release silently omits the cache
    # and tokenizer identity its own documentation promises.
    import glob as _glob
    for _pat in ('wt103_cache/meta_v5_*.json', 'wt103_cache/test_meta_v2_*.json',
                 'wt103_cache/tok_v5_*.json', 'wt103_cache/test_tok_v2_*.json',
                 'fineweb_cache/meta_v2_*.json'):
        for _p in sorted(_glob.glob(os.path.join(REPO, _pat))):
            if os.path.isfile(_p) and not os.path.islink(_p):
                files.append(_p)
    for f in ('results_lm_test/.freeze_lock.json',):
        p = os.path.join(REPO, f)
        if os.path.exists(p) and not os.path.islink(p) and os.path.isfile(p):
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
    # The freeze lock can already be reached by the directory walk above;
    # duplicate tar members with the same name are ambiguous on extraction,
    # so keep the first occurrence only.
    seen = set()
    unique = []
    for p in files:
        key = os.path.abspath(p)
        if key in seen:
            continue
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
                if n.endswith('.pt'):
                    p = os.path.join(root, n)
                    if os.path.islink(p) or not os.path.isfile(p):
                        continue
                    rows[os.path.relpath(p, REPO)] = _sha256(p)
    return rows


def cmd_plan(tag, dirs, include_ckpt=False):
    files = _collect(dirs, include_ckpt)
    total = 0
    print(f'[archive] plan for release {tag!r} ({len(files)} files):')
    for p in files:
        sz = os.path.getsize(p)
        total += sz
        print(f'  {sz:>12,d} B  {os.path.relpath(p, REPO)}')
    ck = _ckpt_hash_table(dirs)
    if ck:
        print(f'[archive] checkpoints found: {len(ck)} (.pt) — hashed into the manifest' + (' and packed' if include_ckpt else ' (bodies NOT packed; use --include-ckpt to embed them)'))
    print(f'[archive] total: {total:,d} B (~{total / (1 << 20):.1f} MiB)')
    return files


def _safe_tag(tag):
    """A release tag used in filenames must be a single path component."""
    if not isinstance(tag, str) or not tag or tag in ('.', '..'):
        return None
    if any(_c in tag for _c in ('/', '\\', '\x00', os.sep, os.altsep or os.sep)):
        return None
    return tag


def cmd_make(tag, dirs, include_ckpt=False, outdir='archive'):
    if _safe_tag(tag) is None:
        print(f'[archive] REFUSE: release tag {tag!r} is not a single safe filename component (path separators / empty / dot segments are not allowed)')
        return 1
    _required = ['DECISION_LOG.md', 'RELEASES.md', 'confirmation_frozen.json']
    _missing = [f for f in _required if not os.path.exists(os.path.join(REPO, f))]
    if _missing:
        print(f'[archive] REFUSE: formal release is missing required provenance file(s): {_missing}')
        return 1
    # Provenance files alone are not an evidence release.  _collect always
    # adds them, so the later `not files` guard could never fire when every
    # requested result directory is missing.  Require at least one actual
    # artifact under the requested dirs.
    _n_artifacts = 0
    for d in dirs:
        base = os.path.join(REPO, d)
        if not os.path.isdir(base):
            continue
        for root, _walk_dirs, names in os.walk(base):
            for n in names:
                if not include_ckpt and (n.endswith('.pt') or n.endswith('.ckpt') or n.endswith('.pth') or n.endswith('.safetensors')):
                    continue
                _p = os.path.join(root, n)
                if os.path.isfile(_p) and not os.path.islink(_p):
                    _n_artifacts += 1
    if not _n_artifacts:
        print('[archive] REFUSE: nothing to archive — the named result dirs do not exist here (or contain no files); run this on the machine holding the artifacts')
        return 1
    files = _collect(dirs, include_ckpt)
    if not files:
        print('[archive] REFUSE: nothing to archive — the named result dirs do not exist here; run this on the machine holding the artifacts')
        return 1
    os.makedirs(os.path.join(REPO, outdir), exist_ok=True)
    stamp = time.strftime('%Y%m%dT%H%M%S')
    base_name = f'release_{tag}'
    tar_path = os.path.join(REPO, outdir, f'{base_name}.tar.gz')
    _existing = [p for p in (tar_path, tar_path + '.sha256',
                             os.path.join(REPO, outdir, f'{base_name}.MANIFEST.sha256'),
                             os.path.join(REPO, outdir, f'{base_name}.release_record.json'))
                 if os.path.exists(p)]
    if _existing:
        print('[archive] REFUSE: release artifacts already exist for tag '
              f'{tag!r} ({[os.path.basename(p) for p in _existing]}) — an '
              'evidence archive is immutable; choose a new tag or move the old '
              'release aside explicitly')
        return 1
    man = {'release': base_name, 'tag': tag, 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'created_by': 'archive_release.py', 'git': _git_state(), 'environment': _env_snapshot(), 'dirs': list(dirs), 'include_ckpt_bodies': bool(include_ckpt), 'files': {}, 'checkpoint_hashes': _ckpt_hash_table(dirs)}
    with tarfile.open(tar_path, 'w:gz') as tf:
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
    # external copy of the manifest + the archive's own hash
    man_ext = os.path.join(REPO, outdir, man_path_rel)
    with open(man_ext, 'wb') as f:
        f.write(man_bytes)
    tar_sha = _sha256(tar_path)
    with open(os.path.join(REPO, outdir, f'{base_name}.tar.gz.sha256'), 'w', encoding='utf-8') as f:
        f.write(f'{tar_sha}  {os.path.basename(tar_path)}\n')
    print(f'[archive] wrote {os.path.relpath(tar_path, REPO)} ({os.path.getsize(tar_path):,d} B)')
    print(f'[archive] wrote {man_path_rel} ({len(man["files"])} files, {len(man["checkpoint_hashes"])} ckpt hashes)')
    print(f'[archive] archive sha256: {tar_sha}')
    print('[archive] publish the tar.gz + MANIFEST + .sha256 to a read-only location (release attachment / object store); the git repo itself stays source-only (design_notes §8)')
    with open(os.path.join(REPO, outdir, f'{base_name}.release_record.json'), 'w', encoding='utf-8') as f:
        json.dump({'release': base_name, 'tag': tag, 'created_at': man['created_at'], 'tar_sha256': tar_sha, 'n_files': len(man['files']), 'n_ckpt_hashes': len(man['checkpoint_hashes']), 'git_head': (man['git'] or {}).get('git_head')}, f, ensure_ascii=False, indent=2)
    return 0


def cmd_verify(tar_path):
    if not os.path.exists(tar_path):
        print(f'[archive] REFUSE: {tar_path} not found')
        return 1
    side = tar_path + '.sha256'
    if not os.path.exists(side):
        # The manifest travels INSIDE the tar, so without the external
        # checksum sidecar an attacker can rewrite the archive and its
        # manifest together and still pass this verifier.  A missing sidecar
        # is therefore a refusal, not an optional extra.
        print(f'[archive] REFUSE: external checksum sidecar {side} is missing — '
              'without it the archive and its in-archive manifest are not '
              'anchored to any independently published digest')
        return 1
    try:
        _parts = open(side, encoding='utf-8').read().split()
    except OSError as _e:
        print(f'[archive] REFUSE: cannot read {side} ({type(_e).__name__}: {_e})')
        return 1
    want = _parts[0] if _parts else ''
    if not want:
        print(f'[archive] REFUSE: {side} is empty or malformed — cannot verify the archive hash')
        return 1
    got = _sha256(tar_path)
    print(f'[archive] archive sha256: {got}  {"OK" if got == want else "MISMATCH (expected " + want + ")"}')
    if got != want:
        return 1
    try:
        tf = tarfile.open(tar_path, 'r:gz')
    except (tarfile.TarError, OSError, EOFError) as _e:
        print(f'[archive] REFUSE: {tar_path} is not a readable tar.gz archive '
              f'({type(_e).__name__}: {_e}) — the checksum anchor cannot rescue '
              'a corrupt file; rebuild or re-download the release')
        return 1
    with tf:
        names = tf.getnames()
        man_name = next((n for n in names if n.endswith('.MANIFEST.sha256')), None)
        if man_name is None:
            print('[archive] REFUSE: no MANIFEST inside the archive')
            return 1
        try:
            man = json.load(tf.extractfile(man_name))
        except Exception as _e:
            print(f'[archive] REFUSE: the archive manifest is not valid JSON ({type(_e).__name__}: {_e})')
            return 1
        if not isinstance(man, dict) or not isinstance(man.get('files'), dict):
            print('[archive] REFUSE: the archive manifest is not an object with a `files` mapping')
            return 1
        bad = []
        for member in tf.getmembers():
            if member.name == man_name:
                continue
            if not member.isfile():
                # A symlink/hardlink/device is not a content member.  The
                # manifest cannot attest its target, so silently skipping it
                # would let the member's name satisfy the presence check while
                # its bytes are never hashed.
                bad.append(f'{member.name}: non-regular tar member ({member.type!r}) is not allowed')
                continue
            f = tf.extractfile(member)
            if f is None:
                bad.append(f'{member.name}: in tar but its content cannot be read')
                continue
            h = hashlib.sha256()
            while True:
                b = f.read(1 << 20)
                if not b:
                    break
                h.update(b)
            got = h.hexdigest()
            _entry = man['files'].get(member.name)
            if _entry is None:
                bad.append(f'{member.name}: in tar but NOT in manifest')
                continue
            if not isinstance(_entry, dict):
                bad.append(f'{member.name}: manifest entry is {type(_entry).__name__}, not an object')
                continue
            want = _entry.get('sha256')
            if not isinstance(want, str) or not want:
                bad.append(f'{member.name}: manifest entry has no usable sha256')
            elif got != want:
                bad.append(f'{member.name}: hash mismatch (manifest {want[:12]}…, actual {got[:12]}…)')
        missing = [n for n in man['files'] if n not in names]
        for n in missing:
            bad.append(f'{n}: in manifest but NOT in tar')
        ck = man.get('checkpoint_hashes') or {}
        print(f'[archive] verified {len(names)} members against the manifest; {len(ck)} checkpoint hashes recorded' + (' (bodies not packed — verify on the experiment machine by hash)' if ck else ''))
        if bad:
            print('[archive] FAIL:\n  - ' + '\n  - '.join(bad[:20]))
            return 1
        print('[archive] VERIFY OK — release ' + str(man.get('tag')) + ' created ' + str(man.get('created_at')) + ' @ git ' + str((man.get('git') or {}).get('git_head')))
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
