"""Execute ordered source fragments into a split module namespace.

Several importable modules in this repository keep their historical
single-file behaviour by loading review-sized fragments from a ``_parts/``
directory into the module globals.  Only the fragment order and package
location differ between them, so the loading mechanics live here: one
implementation for encoding, path handling and the compile filename used in
tracebacks.

A split package keeps its own thin ``__init__.py``: declare the fragments in
execution (dependency) order, then hand the package directory and that list to
:func:`load_split_package`.  Top-level packages use
:func:`load_root_package` and the study drivers under ``explib/`` use
:func:`load_driver_package`, so neither has to repeat the depth calculation or
the ``__file__`` handling.  Appending a fragment therefore stays a one-line
change in that package's ``_PARTS`` list, independent of how many split
packages exist.
"""

from __future__ import annotations

import ntpath
import os
from typing import Any, Iterable


def _safe_fragment_path(package_dir: str, part: Any) -> str:
    """Resolve one ``_PARTS`` entry and keep it inside the package.

    The loader's job is to execute reviewed fragments, not arbitrary paths
    that happen to be spelled in a list.  An absolute path, ``..`` entry,
    Windows drive/UNC name, or a symlink escaping the package would execute
    bytes the package's source tree never reviewed.  Registry audits reject
    such declarations; rejecting them at the execution point as well means a
    caller cannot bypass that check by invoking the loader directly.
    """
    if isinstance(part, os.PathLike):
        part = os.fspath(part)
    if not isinstance(part, str) or not part:
        raise ValueError(f'load_parts: fragment entry must be a non-empty '
                         f'string, got {part!r}')
    if ('\x00' in part or part.startswith(('.', '/', '\\'))
            or '/' in part or '\\' in part or not part.endswith('.py')):
        raise ValueError(
            f'load_parts: {part!r} is not a plain *.py basename inside the package')
    try:
        drive, _tail = ntpath.splitdrive(part)
    except (TypeError, ValueError):
        drive = ''
    if drive or ntpath.isabs(part):
        raise ValueError(
            f'load_parts: {part!r} is an absolute/drive path, not a package fragment')
    package_base = os.path.realpath(package_dir)
    fragment_dir = os.path.join(package_dir, '_parts')
    base = os.path.realpath(fragment_dir)
    try:
        fragment_inside = (os.path.commonpath([package_base, base])
                           == package_base)
    except (OSError, ValueError):
        fragment_inside = False
    if not fragment_inside:
        raise ValueError(
            f'load_parts: {fragment_dir!r} resolves outside the package; '
            'the loader only executes fragments inside the package')
    path = os.path.join(fragment_dir, part)
    try:
        inside = os.path.commonpath([base, os.path.realpath(path)]) == base
    except (OSError, ValueError):
        inside = False
    if not inside:
        raise ValueError(
            f'load_parts: {part!r} resolves outside {fragment_dir!r}; '
            'the loader only executes fragments inside the package')
    return path


def load_parts(namespace: dict[str, Any], package_dir: str,
               parts: Iterable[str]) -> None:
    """Run ``parts`` in order inside ``namespace``.

    ``package_dir`` is the package directory that contains ``_parts/``.
    Fragments are compiled with their real path as the filename, so a
    traceback or ``inspect.getsource`` call points at the reviewable fragment
    rather than at this loader.  Loading executes in the caller's globals,
    which preserves cross-fragment name lookup exactly as the historical
    single-file module did.
    """
    for part in parts:
        path = _safe_fragment_path(package_dir, part)
        with open(path, encoding='utf-8') as fh:
            source = fh.read()
        exec(compile(source, path, 'exec'), namespace)


def load_split_package(namespace: dict[str, Any], package_dir: str,
                       parts: Iterable[str], *, repo_depth: int = 1,
                       historical_path: str | None = None) -> None:
    """Bootstrap a split package into ``namespace``.

    ``package_dir`` is the package directory that holds ``_parts/``.
    ``repo_depth`` is the number of parent directories between it and the
    repository root (``1`` for a top-level package such as ``explib``, ``2``
    for a driver such as ``explib.rope``).

    ``historical_path`` is the repository-relative path of the module whose
    single-file behaviour the package preserves.  It becomes ``__file__``
    before any fragment executes, so resource lookups and
    ``os.path.dirname(__file__)`` inside the fragments resolve exactly as they
    did in the historical module.  ``_HERE`` and ``_REPO`` are also published
    in the namespace for fragments and callers that need the two directories.
    """
    repo_root = package_dir
    for _ in range(repo_depth):
        repo_root = os.path.dirname(repo_root)
    if historical_path is not None:
        namespace['__file__'] = os.path.join(repo_root, historical_path)
    namespace['_HERE'] = package_dir
    namespace['_REPO'] = repo_root
    load_parts(namespace, package_dir, parts)


def load_root_package(namespace: dict[str, Any], module_file: str,
                      parts: Iterable[str], *,
                      historical_path: str) -> None:
    """Bootstrap a top-level split package (``explib``, ``_audit_gates``, …).

    This is the common ``repo_depth=1`` case of :func:`load_split_package`.
    The package directory is derived from ``module_file`` (pass ``__file__``),
    so each package declares only its fragment order and historical script.
    """
    package_dir = os.path.dirname(os.path.abspath(module_file))
    load_split_package(namespace, package_dir, parts, repo_depth=1,
                       historical_path=historical_path)


def load_driver_package(namespace: dict[str, Any], module_file: str,
                        parts: Iterable[str], *,
                        historical_path: str) -> None:
    """Bootstrap one of the study drivers under ``explib/``.

    Driver packages (``explib.rope``, ``explib.confirmation``, …) live one
    directory below the repository root and each preserves a historical
    top-level script (``v7_supp.py`` …).  Passing the package's ``__file__``
    keeps the path handling in one place instead of rebuilding the directory
    and the ``repo_depth=2`` call in every ``__init__.py``.

    The resulting namespace is identical to calling
    :func:`load_split_package` with ``repo_depth=2``: ``_HERE`` is the package
    directory, ``_REPO`` is the repository root and ``__file__`` is the
    historical script path.
    """
    package_dir = os.path.dirname(os.path.abspath(module_file))
    load_split_package(namespace, package_dir, parts, repo_depth=2,
                       historical_path=historical_path)
