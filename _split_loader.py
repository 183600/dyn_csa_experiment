"""Execute ordered source fragments into a split module namespace.

Several importable modules in this repository keep their historical
single-file behaviour by loading review-sized fragments from a ``_parts/``
directory into the module globals.  Only the fragment order and package
location differ between them, so the loading mechanics live here: one
implementation for encoding, path handling and the compile filename used in
tracebacks.

A split package keeps its own thin ``__init__.py``:

* set ``__file__`` to the historical module path, so relative paths in the
  fragments keep resolving against the repository root;
* declare the fragments in execution (dependency) order;
* call :func:`load_parts` with its module globals.

Appending a fragment therefore stays a one-line change in that package's
``_PARTS`` list, independent of how many split packages exist.
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
