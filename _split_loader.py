"""Execute ordered source fragments into a split module namespace.

Several importable modules in this repository keep their historical
single-file behaviour by loading review-sized fragments from a ``_parts/``
directory into the module globals.  Only the fragment order and package
location differ between them, so the loading mechanics live here: one
implementation for encoding, path handling and the compile filename used in
tracebacks.

A split package keeps its own thin ``__init__.py``: declare the fragments in
execution (dependency) order, then hand the package directory and that list to
:func:`load_split_package`.  Appending a fragment therefore stays a one-line
change in that package's ``_PARTS`` list, independent of how many split
packages exist.
"""

from __future__ import annotations

import os
from typing import Any, Iterable


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
    fragment_dir = os.path.join(package_dir, '_parts')
    for part in parts:
        path = os.path.join(fragment_dir, part)
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
