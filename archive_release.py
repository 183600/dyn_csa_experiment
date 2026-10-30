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
  - **测量源码**：`exp_lib.source_tree_manifest()` 范围内的全部文件，
    以仓库相对路径入包；归档因此不依赖 Git 历史仍然自洽
  - **identity 块**：CODE_SEMANTICS、完整源码树摘要、依赖锁契约、
    执行环境指纹；`verify` 只凭归档即可重算源码树摘要
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
  SOURCE_DATE_EPOCH=1700000000 python archive_release.py make v13 \
      --reproducible --dirs results_lm_test analysis_v12
      # 字节可复现：同一证据重建得到同一 tar.gz/sha256
  python archive_release.py verify archive/release_v13.tar.gz

verify 独立于 make：任何拿到归档的人都能校验完整性。
"""
import argparse
import contextlib
import glob
import gzip
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
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


class _HashingReader:
    """File wrapper that hashes exactly the bytes tarfile consumes."""

    def __init__(self, stream):
        self._stream = stream
        self._hash = hashlib.sha256()
        self.nbytes = 0

    def read(self, size=-1):
        data = self._stream.read(size)
        if data:
            self._hash.update(data)
            self.nbytes += len(data)
        return data

    def hexdigest(self):
        return self._hash.hexdigest()


def _atomic_write_bytes(path, data):
    """Write a complete file and only then replace the previous artifact."""
    directory = os.path.dirname(os.path.abspath(path)) or '.'
    # PID alone is unique across processes but not across threads sharing one
    # process; a fixed sibling name could then be truncated by two writers.
    fd, tmp = tempfile.mkstemp(dir=directory,
                               prefix=os.path.basename(path) + '.',
                               suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as f:
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


# ---------------------------------------------------------------------------
# Byte-reproducible packaging
#
# Hashing every payload file makes a release *content*-verifiable, but the
# default tar.gz also records when it was built (member mtimes, the manifest
# member's mtime and the gzip header).  Two builds of identical evidence then
# produce different bytes and a different external .sha256, so a third party
# cannot rebuild the published archive and compare hashes.  Honour the
# reproducible-builds convention instead: when SOURCE_DATE_EPOCH is present,
# pin every timestamp/owner field and the gzip header to it.  The payload
# digest below is independent of that metadata, so content equality remains
# checkable even for legacy/wall-clock archives.
# ---------------------------------------------------------------------------
_PAYLOAD_ALGORITHM = 'sha256(sorted relpath NUL sha256 NUL bytes LF)'
_MAX_SOURCE_DATE_EPOCH = 253402300799  # 9999-12-31T23:59:59Z


def _source_date_epoch(value=None):
    """Parse SOURCE_DATE_EPOCH (or an explicit value) into a non-negative int.

    The value is a build timestamp, not a measurement.  Reject floats,
    negatives and non-numeric strings instead of silently truncating them:
    a malformed stamp would otherwise produce an archive that claims to be
    deterministic under a timestamp nobody can reproduce.
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


def _iso_utc(epoch):
    """Canonical UTC timestamp used by deterministic manifests."""
    try:
        return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(int(epoch)))
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError(f'invalid SOURCE_DATE_EPOCH {epoch!r}: {exc}') from None


def _normalize_tarinfo(ti, epoch):
    """Remove wall-clock and host ownership from one tar member."""
    ti.mtime = int(epoch)
    ti.uid = 0
    ti.gid = 0
    ti.uname = ''
    ti.gname = ''
    # gettarinfo may attach atime/ctime PAX records; they are host state, not
    # content, and would make two builds differ.
    ti.pax_headers = {}


@contextlib.contextmanager
def _open_archive(path, epoch):
    """Open a tar writer, with deterministic gzip framing when epoch is set."""
    if epoch is None:
        with tarfile.open(path, 'w:gz') as tf:
            yield tf
        return
    raw = open(path, 'wb')
    try:
        gz = gzip.GzipFile(filename='', mode='wb', fileobj=raw,
                           compresslevel=9, mtime=int(epoch))
        try:
            tf = tarfile.open(fileobj=gz, mode='w',
                              format=tarfile.PAX_FORMAT)
            try:
                yield tf
            finally:
                tf.close()
        finally:
            gz.close()
    finally:
        raw.close()


def _payload_digest(files):
    """Content digest of the packed file map, excluding tar/gzip metadata."""
    blob = ''.join(
        f'{rel}\0{files[rel]["sha256"]}\0{files[rel]["bytes"]}\n'
        for rel in sorted(files)
        if isinstance(files.get(rel), dict)
        and files[rel].get('sha256') is not None
        and files[rel].get('bytes') is not None)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def _payload_problems(man):
    """Why the manifest's payload digest does not describe its own file map."""
    has_digest = 'payload_sha256' in man
    has_algorithm = 'payload_algorithm' in man
    if not has_digest and not has_algorithm:
        # Legacy archives predate the content digest; absence is not a pass
        # claim, but it must not be reported as tampering either.
        return []
    problems = []
    if man.get('payload_algorithm') != _PAYLOAD_ALGORITHM:
        problems.append('payload_algorithm is not the documented algorithm')
    want = man.get('payload_sha256')
    if not isinstance(want, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', want):
        problems.append('payload_sha256 is missing or malformed')
        return problems
    files = man.get('files')
    if not isinstance(files, dict):
        problems.append('payload digest cannot be recomputed without a files map')
        return problems
    # The digest is defined over (relpath, sha256, byte count).  Skipping an
    # entry whose byte count was deleted would let that field be stripped
    # without changing the digest, so a modern manifest must carry an exact
    # size for every packed file.
    _missing_bytes = [rel for rel, meta in files.items()
                      if _bad_file_entry(meta, require_bytes=True)]
    if _missing_bytes:
        problems.append('payload file map has a missing/malformed byte count: '
                        + ', '.join(sorted(_missing_bytes)[:5]))
        return problems
    got = _payload_digest(files)
    if got.lower() != want.lower():
        problems.append('payload_sha256 does not recompute from the file map')
    return problems


def _gzip_mtime(path):
    """The 32-bit MTIME field of a gzip header (None if not gzip)."""
    try:
        with open(path, 'rb') as f:
            header = f.read(10)
    except OSError:
        return None
    if len(header) < 10 or header[:2] != b'\x1f\x8b':
        return None
    return int.from_bytes(header[4:8], 'little')


def _reproducibility_problems(man, tar_path, members):
    """Check a deterministic archive's timestamp/ownership contract.

    This runs on the archive alone (no exp_lib import), so a reader can decide
    whether a published tar.gz really is the byte-reproducible object its
    manifest claims.  Legacy archives without the block are not failures.
    """
    block = man.get('reproducibility')
    if block is None:
        return []
    if not isinstance(block, dict):
        return ['reproducibility block is not an object']
    deterministic = block.get('deterministic')
    if not isinstance(deterministic, bool):
        return ['reproducibility.deterministic must be a boolean']
    if not deterministic:
        if block.get('source_date_epoch') is not None:
            return ['a non-deterministic archive must not claim a SOURCE_DATE_EPOCH']
        return []
    problems = []
    epoch = block.get('source_date_epoch')
    if (isinstance(epoch, bool) or not isinstance(epoch, int)
            or epoch < 0 or epoch > _MAX_SOURCE_DATE_EPOCH):
        return [f'reproducibility.source_date_epoch is not a valid epoch: {epoch!r}']
    created_at = _iso_utc(epoch)
    if man.get('created_at') != created_at:
        problems.append('created_at does not equal the SOURCE_DATE_EPOCH UTC stamp')
    if block.get('created_at') not in (None, created_at):
        problems.append('reproducibility.created_at disagrees with SOURCE_DATE_EPOCH')
    if block.get('member_mtime') != epoch:
        problems.append('reproducibility.member_mtime disagrees with SOURCE_DATE_EPOCH')
    if block.get('gzip_mtime') != epoch:
        problems.append('reproducibility.gzip_mtime disagrees with SOURCE_DATE_EPOCH')
    if block.get('owner_normalized') is not True:
        problems.append('reproducibility.owner_normalized is not true')
    got_mtime = _gzip_mtime(tar_path)
    if got_mtime != epoch:
        problems.append(f'gzip header mtime is {got_mtime!r}, expected {epoch}')
    for member in members:
        if member.mtime != epoch:
            problems.append(
                f'{member.name}: member mtime is {member.mtime!r}, expected {epoch}')
        if member.uid != 0 or member.gid != 0:
            problems.append(
                f'{member.name}: member uid/gid is {member.uid}/{member.gid}, expected 0/0')
        if member.uname or member.gname:
            problems.append(
                f'{member.name}: member owner names are not normalised')
    return problems


def _reproducibility_block(epoch, created_at):
    """Manifest record of the packaging policy used for this archive."""
    if epoch is None:
        return {
            'deterministic': False,
            'source_date_epoch': None,
            'created_at': None,
            'member_mtime': None,
            'owner_normalized': False,
            'gzip_mtime': None,
            'note': ('wall-clock mtimes and the gzip header timestamp are '
                     'recorded; set SOURCE_DATE_EPOCH for a byte-identical '
                     'rebuild'),
        }
    return {
        'deterministic': True,
        'source_date_epoch': int(epoch),
        'created_at': created_at,
        'member_mtime': int(epoch),
        'owner_normalized': True,
        'gzip_mtime': int(epoch),
        'note': ('all member mtimes, uid/gid/uname/gname and the gzip header '
                 'mtime are pinned to SOURCE_DATE_EPOCH'),
    }


def _git_state(exclude=None):
    """HEAD/branch/dirty state, optionally hiding the archive output tree.

    The release writes its own tar.gz/manifest into ``--out``.  Without the
    exclusion, a second build sees those new untracked files and records a
    different dirty list, so the "reproducible" archive can never match the
    first build.  Build outputs are not measurement source, so they are
    excluded from the recorded work-tree state (the pathspec is only used for
    this diagnostic block, never for the source digest).
    """
    def _run(*args):
        try:
            return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            return None
    head = _run('rev-parse', 'HEAD')
    if head is None:
        return {'git': 'not a git repo / git unavailable'}
    if exclude:
        dirty = _run('status', '--porcelain', '--', '.',
                     f':(exclude){exclude}')
        if dirty is None:
            # An old Git without pathspec magic should still record *some*
            # state rather than silently claiming a clean tree.
            dirty = _run('status', '--porcelain') or ''
    else:
        dirty = _run('status', '--porcelain') or ''
    return {'git_head': head, 'git_branch': _run('rev-parse', '--abbrev-ref', 'HEAD'), 'dirty_files': dirty.splitlines()}


def _env_snapshot(captured_at=None):
    env = {'python': sys.version.split()[0], 'platform': platform.platform(),
           'captured_at': captured_at or time.strftime('%Y-%m-%dT%H:%M:%S')}
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


_SOURCE_TREE_ALGORITHM = 'sha256(sorted relpath NUL bytes NUL sha256 LF)'
_SOURCE_IDENTITY_SCHEMA = 1


def _measurement_source_manifest(root=None):
    """The measurement-relevant source tree, via the library's one definition.

    The archiver must not re-implement the scope: `exp_lib.source_tree_manifest`
    is what the freeze lock, run provenance and the source audit already bind.
    Importing the library here is deliberate — a release built from bytes that
    the measurement code cannot identify is not an evidence release.
    """
    _root = os.path.abspath(root or REPO)
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    try:
        import exp_lib as _L
    except Exception as e:
        raise RuntimeError(
            'cannot compute the measurement source identity '
            f'({type(e).__name__}: {e}); run the archiver from the repository '
            'envelope with exp_lib.py available') from e
    return _L.source_tree_manifest(root=_root)


def _release_identity(source_tree):
    """Compact, self-contained identity for one evidence release."""
    import exp_lib as _L
    env = _L.environment_snapshot()
    _env_keys = ('python', 'platform', 'machine', 'processor', 'torch',
                 'torch_git_version', 'numpy', 'cuda_available', 'cuda_version',
                 'cudnn_version', 'gpu_identity', 'determinism',
                 'cudnn_deterministic', 'cudnn_benchmark',
                 'deterministic_algorithms', 'tf32_matmul', 'tf32_cudnn',
                 'cublas_workspace_config', 'cpu_threads', 'pythonhashseed')
    return {
        'schema': _SOURCE_IDENTITY_SCHEMA,
        'code_semantics': _L.CODE_SEMANTICS,
        'code_stamp': _L.code_stamp(),
        'source_tree': source_tree,
        'source_tree_sha256': source_tree.get('sha256'),
        'dependency_contract': _L.dependency_contract(),
        'execution_fingerprint': _L.execution_fingerprint(env=env),
        # Archive both strict generations: v2 keeps old releases comparable,
        # v3 adds the NumPy/PyTorch numerical backend and every unpinned
        # runtime distribution.  `numerical_backend` is kept in readable form
        # so a verifier can say WHAT differed, not only that v3 did.
        'env_fp_v2': env.get('env_fp_v2'),
        'env_fp_v3': env.get('env_fp_v3'),
        'numerical_backend': env.get('numerical_backend'),
        'environment': {k: env.get(k) for k in _env_keys},
        'note': ('the measurement source files are packed at repository-relative '
                 'paths; verify recomputes source_tree.sha256 from the packed '
                 'bytes, so a release remains auditable without Git history'),
    }


def _bad_file_entry(meta, *, require_bytes=False):
    """Malformed per-file manifest entry?  Booleans are not byte counts."""
    if not isinstance(meta, dict):
        return True
    sha = meta.get('sha256')
    if not isinstance(sha, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', sha):
        return True
    size = meta.get('bytes')
    if size is None:
        return bool(require_bytes)
    return (isinstance(size, bool) or not isinstance(size, int) or size < 0)


def _release_identity_problems(man, *, identity=None):
    """Why a manifest's identity does not bind its packed bytes.

    Runs on the archived manifest alone: every source-tree entry must be in the
    per-file map with the same digest/size, the aggregate digest is recomputed
    with the documented algorithm, and the top-level identity fields must be
    present.  It does not import exp_lib, so `verify` stays independent of the
    code it is checking.
    """
    problems = []
    ident = man.get('identity') if identity is None else identity
    if not isinstance(ident, dict):
        return ['identity block is missing or not an object']
    if ident.get('schema') != _SOURCE_IDENTITY_SCHEMA:
        problems.append(f"identity schema is {ident.get('schema')!r}, "
                        f'expected {_SOURCE_IDENTITY_SCHEMA}')
    _cs = ident.get('code_semantics')
    if not isinstance(_cs, str) or not _cs.strip():
        problems.append('identity.code_semantics is empty or not a string')
    if not isinstance(ident.get('dependency_contract'), dict):
        problems.append('identity.dependency_contract is missing or not an object')
    st = ident.get('source_tree')
    if not isinstance(st, dict):
        return problems + ['identity.source_tree is missing or not an object']
    if st.get('algorithm') != _SOURCE_TREE_ALGORITHM:
        problems.append('identity.source_tree.algorithm is not the documented one')
    files = st.get('files')
    if not isinstance(files, dict) or not files:
        return problems + ['identity.source_tree.files is missing or empty']
    packed = man.get('files') if isinstance(man.get('files'), dict) else {}
    if not packed:
        problems.append('manifest files map is missing or empty')
    total = 0
    for rel, meta in files.items():
        if _unsafe_member_name(rel):
            problems.append(f'unsafe source-tree path: {rel!r}')
            continue
        if _bad_file_entry(meta, require_bytes=True):
            problems.append(f'malformed source-tree entry: {rel!r}')
            continue
        total += int(meta['bytes'])
        got = packed.get(rel)
        if not isinstance(got, dict):
            problems.append(f'source file not packed: {rel!r}')
            continue
        if str(got.get('sha256', '')).lower() != str(meta['sha256']).lower():
            problems.append(f'source-file digest disagrees with packed bytes: {rel!r}')
        if got.get('bytes') is not None and int(got['bytes']) != int(meta['bytes']):
            problems.append(f'source-file size disagrees with packed bytes: {rel!r}')
    blob = ''.join(f'{rel}\0{files[rel]["bytes"]}\0{files[rel]["sha256"]}\n'
                   for rel in sorted(files)
                   if isinstance(files.get(rel), dict)
                   and files[rel].get('sha256') is not None
                   and files[rel].get('bytes') is not None)
    digest = hashlib.sha256(blob.encode('utf-8')).hexdigest()
    if str(st.get('sha256', '')).lower() != digest:
        problems.append('identity.source_tree.sha256 does not recompute from its files')
    if str(ident.get('source_tree_sha256', '')).lower() != str(st.get('sha256', '')).lower():
        problems.append('identity.source_tree_sha256 disagrees with identity.source_tree.sha256')
    if st.get('n_files') != len(files):
        problems.append('identity.source_tree.n_files disagrees with its file map')
    if st.get('bytes') != total:
        problems.append('identity.source_tree.bytes disagrees with its file map')
    return problems


def _collect(dirs, include_ckpt=False, *, include_source=False, source_tree=None):
    """Files that would enter the archive, with sizes and hashes (lazily).

    `include_source` appends the exact measurement-source files (the same
    scope the freeze lock and run provenance bind).  The caller can pass a
    previously computed `source_tree` so the identity and the packed bytes are
    evaluated against one scan; a later scan would make a concurrent source
    edit look like a valid release.
    """
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
                # Do NOT drop checkpoint-suffixed files before the regular-file
                # / symlink check below: a link named `model.pt` would then be
                # skipped here but still hashed by `_ckpt_hash_table`, making
                # the manifest depend on bytes outside the archive/repository.
                files.append(os.path.join(root, n))
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
    if include_source:
        st = source_tree if source_tree is not None else _measurement_source_manifest()
        _src_files = st.get('files') if isinstance(st, dict) else None
        if not isinstance(_src_files, dict) or not _src_files:
            raise RuntimeError('the measurement source manifest is empty; refusing '
                               'to build a release that cannot bind its code')
        for rel in sorted(_src_files):
            files.append(os.path.join(REPO, rel))
    # The freeze lock can already be reached by the directory walk above;
    # duplicate tar members with the same name are ambiguous on extraction,
    # so keep the first occurrence only.
    seen = set()
    unique = []
    for p in files:
        key = os.path.abspath(p)
        if key in seen:
            continue
        # Validate every candidate BEFORE the checkpoint-body filter: a
        # symlinked or otherwise non-regular `model.pt` must be refused even
        # when the caller asked not to embed weight bodies.
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
        if not include_ckpt and p.endswith(_CKPT_SUFFIXES):
            # The checkpoint hash table still records the (already validated)
            # regular file, but its body is intentionally not packed.
            continue
        seen.add(key)
        unique.append(p)
    # os.walk order is filesystem-dependent; the archive manifest and tar
    # member order must be derived from the repository-relative names so two
    # builds of the same evidence produce the same bytes.
    unique.sort(key=lambda p: os.path.relpath(p, REPO).replace(os.sep, '/'))
    return unique


def _ckpt_hash_table(dirs):
    """sha256 of every checkpoint file referenced by the release (even when the
    .pt bodies are not packed) — the weights stay auditable by hash.

    The table is part of the release manifest, so it must obey the same
    containment/regular-file contract as the packed files: a symlink (or a
    directory named `*.pt`) would either hash bytes outside the repository or
    crash the archiver after the manifest was already built."""
    rows = {}
    _repo_real = os.path.realpath(REPO)
    for d in dirs:
        base = os.path.join(REPO, d)
        if not os.path.isdir(base):
            continue
        for root, _dirs, names in os.walk(base):
            for n in names:
                if not n.endswith(_CKPT_SUFFIXES):
                    continue
                p = os.path.join(root, n)
                if os.path.islink(p):
                    raise ValueError(
                        f'checkpoint {p!r} is a symbolic link; pack the target '
                        'explicitly (a link cannot be hashed as a regular file)')
                if not os.path.isfile(p):
                    raise ValueError(
                        f'checkpoint {p!r} is not a regular file; refusing to '
                        'hash it')
                _p_real = os.path.realpath(p)
                try:
                    _inside = os.path.commonpath([_repo_real, _p_real]) == _repo_real
                except ValueError:
                    _inside = False
                if not _inside:
                    raise ValueError(
                        f'checkpoint {p!r} resolves outside the repository '
                        f'({_p_real!r}); refusing to hash it')
                rows[os.path.relpath(p, REPO)] = _sha256(p)
    return dict(sorted(rows.items()))


def cmd_plan(tag, dirs, include_ckpt=False):
    tag = _release_tag(tag)
    source_tree = _measurement_source_manifest()
    files = _collect(dirs, include_ckpt, include_source=True,
                     source_tree=source_tree)
    total = 0
    print(f'[archive] plan for release {tag!r} ({len(files)} files):')
    for p in files:
        sz = os.path.getsize(p)
        total += sz
        print(f'  {sz:>12,d} B  {os.path.relpath(p, REPO)}')
    print(f'[archive] measurement source: {source_tree.get("n_files")} files, '
          f'{int(source_tree.get("bytes") or 0):,d} B, '
          f'sha256 {str(source_tree.get("sha256"))[:16]}…')
    ck = _ckpt_hash_table(dirs)
    if ck:
        print(f'[archive] checkpoints found: {len(ck)} — hashed into the manifest' + (' and packed' if include_ckpt else ' (bodies NOT packed; use --include-ckpt to embed them)'))
    print(f'[archive] total: {total:,d} B (~{total / (1 << 20):.1f} MiB)')
    return files


def cmd_make(tag, dirs, include_ckpt=False, outdir='archive', *,
             source_date_epoch=None, reproducible=False):
    tag = _release_tag(tag)
    try:
        epoch = _source_date_epoch(source_date_epoch)
        created_at = _iso_utc(epoch) if epoch is not None else time.strftime('%Y-%m-%dT%H:%M:%S')
    except ValueError as exc:
        print(f'[archive] REFUSE: {exc}')
        return 1
    if reproducible and epoch is None:
        print('[archive] REFUSE: --reproducible requires SOURCE_DATE_EPOCH '
              '(or --source-date-epoch); a wall-clock archive cannot be '
              'rebuilt byte-for-byte')
        return 1
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
    source_tree = _measurement_source_manifest()
    files = _collect(dirs, include_ckpt, include_source=True,
                     source_tree=source_tree)
    if not files:
        print('[archive] REFUSE: nothing to archive — the named result dirs do not exist here; run this on the machine holding the artifacts')
        return 1
    files.sort(key=lambda p: os.path.relpath(p, REPO).replace(os.sep, '/'))
    _outdir = _safe_outdir(outdir)
    os.makedirs(_outdir, exist_ok=True)
    _rel_outdir = os.path.relpath(_outdir, REPO).replace(os.sep, '/')
    identity = _release_identity(source_tree)
    _dep_status = (identity.get('dependency_contract') or {}).get('status')
    if _dep_status not in ('ok', 'partial'):
        print(f'[archive] WARNING: dependency contract status is {_dep_status!r}; '
              'the archive records the mismatch but this environment is not a '
              'verified lock environment')
    base_name = f'release_{tag}'
    tar_path = os.path.join(_outdir, f'{base_name}.tar.gz')
    man = {
        'schema': 2,
        'release': base_name,
        'tag': tag,
        'created_at': created_at,
        'created_by': 'archive_release.py',
        'git': _git_state(exclude=None if _rel_outdir in ('', '.') else _rel_outdir),
        'environment': _env_snapshot(captured_at=created_at),
        'identity': identity,
        'dirs': list(dirs),
        'include_ckpt_bodies': bool(include_ckpt),
        'payload_algorithm': _PAYLOAD_ALGORITHM,
        'files': {},
        'checkpoint_hashes': _ckpt_hash_table(dirs),
    }
    # Build the archive beside its final name; an interrupted or failed make
    # must not leave a truncated `release_<tag>.tar.gz` (or destroy a previous
    # good release with the same tag).
    tmp_tar = tar_path + f'.tmp{os.getpid()}'
    try:
        with _open_archive(tmp_tar, epoch) as tf:
            for p in files:
                rel = os.path.relpath(p, REPO)
                # Hash the bytes that are actually written into the archive,
                # not a separate earlier read of the source path.  A live log
                # or run-state file can change between those two reads, which
                # produced a release whose own manifest could never verify.
                st_before = os.stat(p)
                ti = tf.gettarinfo(p, arcname=rel)
                ti.size = int(st_before.st_size)
                if epoch is not None:
                    _normalize_tarinfo(ti, epoch)
                with open(p, 'rb') as src:
                    fst = os.fstat(src.fileno())
                    if ((fst.st_dev, fst.st_ino, fst.st_size, fst.st_mtime_ns)
                            != (st_before.st_dev, st_before.st_ino,
                                st_before.st_size, st_before.st_mtime_ns)):
                        raise RuntimeError(
                            f'archive input {p!r} changed between stat and '
                            'open; retry when the artifact is quiescent')
                    reader = _HashingReader(src)
                    tf.addfile(ti, reader)
                    fst_after = os.fstat(src.fileno())
                    st_after = os.stat(p)
                if ((fst_after.st_dev, fst_after.st_ino,
                     fst_after.st_size, fst_after.st_mtime_ns)
                        != (st_before.st_dev, st_before.st_ino,
                            st_before.st_size, st_before.st_mtime_ns)
                        or (st_after.st_dev, st_after.st_ino,
                            st_after.st_size, st_after.st_mtime_ns)
                        != (st_before.st_dev, st_before.st_ino,
                            st_before.st_size, st_before.st_mtime_ns)):
                    raise RuntimeError(
                        f'archive input {p!r} changed while it was being '
                        'packed; retry when the artifact is quiescent')
                if int(reader.nbytes) != int(ti.size):
                    raise RuntimeError(
                        f'archive input {p!r} produced {reader.nbytes} bytes '
                        f'but tar expected {ti.size}; nothing was published')
                man['files'][rel] = {'sha256': reader.hexdigest(),
                                     'bytes': int(reader.nbytes)}
            # Content identity is computed after every member is hashed; it
            # deliberately excludes tar/gzip metadata so two archives that
            # carry the same evidence compare equal even when one was built
            # without SOURCE_DATE_EPOCH.
            man['payload_sha256'] = _payload_digest(man['files'])
            man['reproducibility'] = _reproducibility_block(epoch, created_at)
            # The source manifest was scanned before packing.  A source edit
            # between that scan and the reads above would produce an archive
            # whose identity does not describe its own code; refuse to publish
            # it instead of leaving a release that cannot verify.
            _ident_problems = _release_identity_problems(man, identity=identity)
            if _ident_problems:
                raise RuntimeError(
                    'release identity does not match the packed measurement '
                    'source (was a source file edited while archiving?): '
                    + '; '.join(_ident_problems[:8]))
            man_path_rel = f'{base_name}.MANIFEST.sha256'
            man_bytes = json.dumps(man, ensure_ascii=False, indent=2).encode('utf-8')
            ti = tarfile.TarInfo(man_path_rel)
            ti.size = len(man_bytes)
            ti.mtime = epoch if epoch is not None else int(time.time())
            if epoch is not None:
                _normalize_tarinfo(ti, epoch)
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
    print(f'[archive] identity: code_semantics={identity.get("code_semantics")} '
          f'source_tree={str(identity.get("source_tree_sha256"))[:16]}… '
          f'({int(source_tree.get("n_files") or 0)} source files, '
          f'dependency={_dep_status})')
    print(f'[archive] archive sha256: {tar_sha}')
    print(f'[archive] payload sha256: {man["payload_sha256"]} '
          f'({"reproducible bytes" if epoch is not None else "wall-clock metadata; set SOURCE_DATE_EPOCH for byte-identical rebuilds"})')
    print('[archive] publish the tar.gz + MANIFEST + .sha256 to a read-only location (release attachment / object store); the git repo itself stays source-only (design_notes §8)')
    _atomic_write_bytes(
        os.path.join(_outdir, f'{base_name}.release_record.json'),
        json.dumps({'release': base_name, 'tag': tag, 'created_at': man['created_at'],
                    'tar_sha256': tar_sha, 'payload_sha256': man['payload_sha256'],
                    'reproducible': bool(epoch is not None),
                    'source_date_epoch': epoch,
                    'n_files': len(man['files']),
                    'n_ckpt_hashes': len(man['checkpoint_hashes']),
                    'code_semantics': identity.get('code_semantics'),
                    'source_tree_sha256': identity.get('source_tree_sha256'),
                    'execution_fingerprint': identity.get('execution_fingerprint'),
                    'dependency_status': _dep_status,
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
        # A manifest that carries the content digest (every current one) must
        # also carry each file's byte count: that field is part of the
        # payload identity, and dropping it must not silently shrink what the
        # digest covers.  Legacy archives without the digest keep the lenient
        # reader so they remain verifiable.
        _require_bytes = ('payload_sha256' in man
                          or 'payload_algorithm' in man)
        _bad_entries = [n for n, meta in man['files'].items()
                        if _bad_file_entry(meta, require_bytes=_require_bytes)]
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
        _ident = man.get('identity')
        if _ident is None:
            print('[archive] NOTE: no identity block (legacy release); the '
                  'measurement source-tree binding cannot be verified from '
                  'this archive alone')
        else:
            _ident_bad = _release_identity_problems(man, identity=_ident)
            if _ident_bad:
                bad.extend(f'identity: {p}' for p in _ident_bad[:20])
        for _problem in _payload_problems(man):
            bad.append(f'payload: {_problem}')
        for _problem in _reproducibility_problems(
                man, tar_path, tf.getmembers()):
            bad.append(f'reproducibility: {_problem}')
        _repro = man.get('reproducibility')
        if isinstance(_repro, dict) and _repro.get('deterministic') is True:
            print(f'[archive] reproducible build pinned to SOURCE_DATE_EPOCH='
                  f'{_repro.get("source_date_epoch")}')
        elif _repro is not None:
            print('[archive] NOTE: archive uses wall-clock metadata; '
                  'payload identity is still checkable but the tar.gz bytes '
                  'are not rebuildable')
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
    p_make.add_argument(
        '--source-date-epoch', type=int, default=None,
        help='pin tar/gzip metadata to this Unix timestamp (also honours the '
             'SOURCE_DATE_EPOCH environment variable)')
    p_make.add_argument(
        '--reproducible', action='store_true',
        help='require SOURCE_DATE_EPOCH and emit a byte-reproducible tar.gz')
    p_ver = sub.add_parser('verify', help='verify an archive against its manifest')
    p_ver.add_argument('tar')
    args = ap.parse_args(argv)
    if args.cmd == 'plan':
        cmd_plan(args.tag, args.dirs, args.include_ckpt)
        return 0
    if args.cmd == 'make':
        return cmd_make(args.tag, args.dirs, args.include_ckpt, args.out,
                        source_date_epoch=args.source_date_epoch,
                        reproducible=args.reproducible)
    if args.cmd == 'verify':
        return cmd_verify(args.tar)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
