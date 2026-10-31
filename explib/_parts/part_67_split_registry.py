# ---------------------------------------------------------------------------
# 128. The executed module is more than the files on disk: bind split-package
#      registration (order + membership) into the source identity
#
# `explib`, the study drivers and the audit package are assembled by executing
# `_parts/*.py` in the order declared by `_PARTS`.  `source_tree_manifest()`
# hashes every reviewed source file, but a file-tree hash does not say WHICH
# files were executed, in what order, or whether an entry resolved outside the
# tree:
#
#   * a fragment can be present but omitted from `_PARTS` (dead source that
#     still enters the source digest and makes provenance noisier, never
#     visibly "unused");
#   * the same fragments can be listed in a different order (later fragments
#     may overwrite names defined by earlier ones);
#   * a path-like entry (`/tmp/part.py`, `../outside.py`) would be executed by
#     the loader but is not in the repository tree, so the source digest
#     cannot bind its bytes at all.
#
# The first two are also caught by hashing `__init__.py` (the declaration is
# source), but the digest then conflates "same executed program" with
# comment-only edits.  The third is a real provenance hole.  The registry
# below is a derived, order-aware identity: ordered (package, part, sha256)
# tuples plus the declared historical entry point, validated to stay inside
# the checkout.  It is attached to `source_tree` (so the freeze lock,
# per-record provenance and release identity already carry it) without
# changing the historical file-tree digest, which keeps old archives
# verifiable.
# ---------------------------------------------------------------------------

import ast as _split_ast
import ntpath as _split_ntpath

SPLIT_REGISTRY_SCHEMA = 1
SPLIT_REGISTRY_ALGORITHM = (
    'sha256(sorted package NUL historical NUL order NUL part NUL sha256 LF)')

# Directories that are never part of the reviewed source envelope.  Reuse the
# one definition from the source-tree scope instead of maintaining a second
# skip list that could drift from it.
_SPLIT_REGISTRY_SKIP_DIRS = _SOURCE_TREE_SKIP_DIRS
_SPLIT_REGISTRY_SKIP_SUFFIXES = _SOURCE_TREE_SKIP_SUFFIXES
_SPLIT_REGISTRY_SKIP_PREFIXES = _SOURCE_TREE_SKIP_PREFIXES


def _split_rel(path, root):
    return os.path.relpath(path, root).replace(os.sep, '/')


def _split_is_plain_py_name(name):
    """A fragment entry is one basename, not a path on any platform.

    ``os.path`` on POSIX treats ``C:outside.py`` as an ordinary filename, but
    the same literal is drive-relative on Windows and would resolve outside
    the package directory.  The registry is a cross-platform identity, so the
    Windows drive/UNC rules are applied even when the audit runs on POSIX.
    """
    if not isinstance(name, str) or not name:
        return False
    if ('\x00' in name or name.startswith(('.', '/', '\\'))
            or '/' in name or '\\' in name
            or not name.endswith('.py')):
        return False
    try:
        drive, _tail = _split_ntpath.splitdrive(name)
    except (TypeError, ValueError):
        return False
    return not drive and not _split_ntpath.isabs(name)


def _split_hist_is_repo_relative(hist):
    """Whether ``historical_path`` is a plain repo-relative path.

    The loader joins this literal to the repository root to publish
    ``__file__``.  Absolute paths (including Windows drive/UNC spellings on a
    POSIX host, where ``os.path.isabs`` alone would miss them) and ``..``
    traversal would let fragments resolve resources outside the checkout.
    A path may name a file that no longer exists (the historical single-file
    script was split years ago); only its value is the identity.
    """
    if not isinstance(hist, str):
        return False
    text = hist.strip()
    if not text or '\x00' in text:
        return False
    posix = text.replace('\\', '/')
    if os.path.isabs(text) or _split_ntpath.isabs(text):
        return False
    try:
        drive, _tail = _split_ntpath.splitdrive(text)
    except (TypeError, ValueError):
        return False
    if drive:
        return False
    return not (posix.startswith('/') or posix.startswith('../')
                or '/../' in posix or posix in ('..', '.'))


def _split_registry_declaration(init_path):
    """Parse a split package's ``_PARTS`` and ``historical_path`` literals.

    The loader executes the same literals at import time.  They are read with
    ``ast.literal_eval`` rather than imported here: importing every driver
    would require CUDA/torch state, and a registry audit must work on a
    checkout before any experiment starts.  A non-literal declaration is an
    error, not something to guess.
    """
    out = {'parts': None, 'historical_path': None, 'problems': []}
    try:
        with open(init_path, 'rb') as fh:
            raw = fh.read()
    except OSError as e:
        out['problems'].append(
            f'cannot read package init {init_path}: {type(e).__name__}: {e}')
        return out
    try:
        tree = _split_ast.parse(raw, filename=init_path)
    except (SyntaxError, ValueError) as e:
        out['problems'].append(
            f'cannot parse package init {init_path}: {type(e).__name__}: {e}')
        return out

    declarations = []
    for node in tree.body:
        targets = []
        value = None
        if isinstance(node, _split_ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, _split_ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        for target in targets:
            if isinstance(target, _split_ast.Name) and target.id == '_PARTS':
                declarations.append((node.lineno, value))
    if not declarations:
        out['problems'].append('no _PARTS declaration found')
        return out
    if len(declarations) > 1:
        out['problems'].append(
            'multiple _PARTS declarations; the executed order is ambiguous')
        return out
    lineno, value = declarations[0]
    try:
        parsed = _split_ast.literal_eval(value)
    except (ValueError, TypeError, SyntaxError) as e:
        out['problems'].append(
            f'_PARTS (line {lineno}) is not a literal list/tuple: '
            f'{type(e).__name__}: {e}')
        return out
    if not isinstance(parsed, (list, tuple)):
        out['problems'].append(
            f'_PARTS (line {lineno}) is a {type(parsed).__name__}, '
            'not a list/tuple')
        return out
    out['parts'] = list(parsed)

    # The historical entry point is what the loader publishes as __file__.
    # Changing it can change resource lookups, so it is part of the identity.
    for node in tree.body:
        if not isinstance(node, _split_ast.Expr):
            continue
        call = node.value
        if not isinstance(call, _split_ast.Call):
            continue
        func = call.func
        name = func.id if isinstance(func, _split_ast.Name) else (
            func.attr if isinstance(func, _split_ast.Attribute) else '')
        if not name.startswith('load_'):
            continue
        for kw in call.keywords:
            if kw.arg != 'historical_path':
                continue
            try:
                hist = _split_ast.literal_eval(kw.value)
            except (ValueError, TypeError, SyntaxError):
                out['problems'].append(
                    f'historical_path (line {node.lineno}) is not a literal string')
                break
            if not isinstance(hist, str) or not hist.strip():
                out['problems'].append(
                    f'historical_path (line {node.lineno}) is not a non-empty string')
                break
            out['historical_path'] = hist.strip()
            break
        break
    return out


def _split_package_dirs(root):
    """Split-package directories (the parents of every ``_parts/`` directory).

    ``os.walk(..., followlinks=False)`` deliberately does not descend into a
    symlinked directory, but a symlinked package or ``_parts`` directory is
    still importable/executable through its link path.  Such a candidate is
    reported here (and then rejected by :func:`split_registry_manifest`)
    instead of disappearing from the registry entirely.  A symlinked
    directory is never traversed, so a link cycle cannot turn this scan into
    an unbounded walk.
    """
    out = []
    seen = set()

    def _add(pkg, parts):
        key = (os.path.abspath(pkg), os.path.abspath(parts))
        if key not in seen:
            seen.add(key)
            out.append((pkg, parts))

    for cur, dirs, _names in os.walk(root, followlinks=False):
        keep = []
        for d in sorted(dirs):
            if (d in _SPLIT_REGISTRY_SKIP_DIRS
                    or any(d.endswith(sfx) for sfx in _SPLIT_REGISTRY_SKIP_SUFFIXES)
                    or any(d.startswith(pref) for pref in _SPLIT_REGISTRY_SKIP_PREFIXES)):
                continue
            path = os.path.join(cur, d)
            if os.path.islink(path):
                # Do not follow the link.  Record a directly linked split
                # package (or a linked `_parts`), so the manifest can fail
                # closed rather than silently omitting it.
                if d == '_parts':
                    if os.path.isfile(os.path.join(cur, '__init__.py')):
                        _add(cur, path)
                elif (os.path.isfile(os.path.join(path, '__init__.py'))
                      and os.path.isdir(os.path.join(path, '_parts'))):
                    _add(path, os.path.join(path, '_parts'))
                continue
            keep.append(d)
        dirs[:] = keep
        if os.path.basename(cur) == '_parts':
            pkg = os.path.dirname(cur)
            if os.path.isfile(os.path.join(pkg, '__init__.py')):
                _add(pkg, cur)
            dirs[:] = []
    return sorted(out, key=lambda pair: _split_rel(pair[0], root))


def split_registry_manifest(root=None):
    """Identity of the *executed* split-package registration.

    For every package assembled from a ``_parts/`` directory this records the
    ordered fragment names, their content hashes and the declared
    ``historical_path``.  It fails closed (``status='invalid'``) on:

    * a listed fragment that does not exist / escapes the repository,
    * a fragment on disk that is not listed (silently dead source),
    * duplicate or path-like entries,
    * a package with no parseable ``_PARTS`` literal.

    The digest deliberately covers order and membership.  Reordering two
    fragments or dropping one from the list changes it even when no fragment
    byte changes; a comment-only edit to ``__init__.py`` does not.  Path-like
    entries and symlinks escaping the checkout are refused because the loader
    would execute bytes the source-tree digest never saw.  A full manifest
    includes this block; an explicit partial ``include`` scan leaves it
    ``None`` because it did not inspect the package declarations.
    """
    root = os.path.abspath(root or os.path.dirname(os.path.abspath(__file__)))
    root_real = os.path.realpath(root)
    problems = []
    packages = {}
    n_parts = 0

    def _inside(path):
        try:
            return os.path.commonpath(
                [root_real, os.path.realpath(path)]) == root_real
        except (OSError, ValueError):
            return False

    for pkg_dir, parts_dir in _split_package_dirs(root):
        pkg_rel = _split_rel(pkg_dir, root)
        init_path = os.path.join(pkg_dir, '__init__.py')
        pkg_problems = []
        pkg_is_link = os.path.islink(pkg_dir)
        parts_is_link = os.path.islink(parts_dir)
        parts_inside = _inside(parts_dir)
        init_inside = _inside(init_path)
        if pkg_is_link:
            pkg_problems.append(
                'package directory is a symlink; the source-tree digest does '
                'not traverse linked directories, so the executed layout is '
                'not the reviewed one')
        if parts_is_link:
            pkg_problems.append(
                '_parts directory is a symlink; the source-tree digest does '
                'not traverse linked directories, so fragment paths would '
                'not be the reviewed ones')
        if not parts_inside:
            pkg_problems.append('_parts directory resolves outside the tree')
        if not init_inside:
            pkg_problems.append('package init resolves outside the tree')
        if init_inside:
            decl = _split_registry_declaration(init_path)
        else:
            decl = {'parts': None, 'historical_path': None, 'problems': []}
        pkg_problems.extend(decl['problems'])
        # Entries and the on-disk inventory are only meaningful when the
        # declaration and the fragment directory are both regular, in-tree
        # paths.  For a linked/outside layout the registry is invalid anyway;
        # recording hashes through the link would imply an identity the source
        # tree did not review.
        can_read_parts = (parts_inside and init_inside
                          and not pkg_is_link and not parts_is_link)
        entries = []
        declared = decl['parts']
        if declared is None:
            declared = []
        seen = set()
        for order, name in enumerate(declared):
            if not isinstance(name, str) or not name:
                pkg_problems.append(f'_PARTS[{order}] is not a non-empty string')
                continue
            if not _split_is_plain_py_name(name):
                pkg_problems.append(
                    f'_PARTS[{order}]={name!r} is not a plain *.py basename')
                continue
            if name in seen:
                pkg_problems.append(f'duplicate fragment registration: {name}')
                continue
            seen.add(name)
            if not can_read_parts:
                continue
            path = os.path.join(parts_dir, name)
            if not os.path.isfile(path) or not _inside(path):
                pkg_problems.append(
                    f'registered fragment is missing or outside the tree: {name}')
                continue
            try:
                with open(path, 'rb') as fh:
                    data = fh.read()
            except OSError as e:
                pkg_problems.append(
                    f'cannot read fragment {name}: {type(e).__name__}: {e}')
                continue
            entries.append({'name': name, 'order': order,
                            'sha256': hashlib.sha256(data).hexdigest(),
                            'bytes': len(data)})
            n_parts += 1

        on_disk = []
        if can_read_parts:
            try:
                on_disk = sorted(
                    n for n in os.listdir(parts_dir)
                    if n.endswith('.py')
                    and os.path.isfile(os.path.join(parts_dir, n)))
            except OSError as e:
                pkg_problems.append(
                    f'cannot list {pkg_rel}/_parts: {type(e).__name__}: {e}')
        # If the declaration itself is unparseable, every fragment would look
        # unregistered; one root-cause problem is enough.
        if decl['parts'] is not None:
            for name in on_disk:
                if name not in seen:
                    pkg_problems.append(
                        f'fragment exists on disk but is not registered: {name}')

        hist = decl['historical_path']
        if hist is None:
            if not any('historical_path' in p for p in pkg_problems):
                pkg_problems.append('missing or non-literal historical_path')
        elif not _split_hist_is_repo_relative(hist):
            pkg_problems.append(
                f'historical_path is not a repo-relative path: {hist!r}')

        pkg_blob = ''.join(
            f'{pkg_rel}\0{hist or ""}\0{e["order"]}\0{e["name"]}\0{e["sha256"]}\n'
            for e in entries)
        pkg_sha = hashlib.sha256(pkg_blob.encode('utf-8')).hexdigest()
        packages[pkg_rel] = {
            'historical_path': hist,
            'sha256': pkg_sha,
            'n_parts': len(entries),
            'parts': entries,
            'problems': pkg_problems,
        }
        problems.extend(f'{pkg_rel}: {p}' for p in pkg_problems)

    blob = ''.join(f'{pkg}\0{packages[pkg]["sha256"]}\n'
                   for pkg in sorted(packages))
    digest = hashlib.sha256(blob.encode('utf-8')).hexdigest()
    return {
        'kind': 'split_registry',
        'schema': SPLIT_REGISTRY_SCHEMA,
        'algorithm': SPLIT_REGISTRY_ALGORITHM,
        'sha256': digest,
        'n_packages': len(packages),
        'n_parts': n_parts,
        'packages': packages,
        'problems': problems,
        'status': 'ok' if not problems else 'invalid',
        'scope': ('every package with a _parts/ directory and a literal _PARTS '
                  'declaration; order and membership are part of the digest, '
                  'the plain file-tree digest is unchanged for old archives'),
    }


def split_registry_summary(root=None):
    """Compact registry identity for ``source_tree``/provenance blocks."""
    reg = split_registry_manifest(root=root)
    out = {k: reg[k] for k in ('kind', 'schema', 'algorithm', 'sha256',
                               'n_packages', 'n_parts', 'status')}
    if reg.get('problems'):
        out['problems'] = list(reg['problems'])[:12]
    return out
