"""Split implementation package for build_report.py.

Fragments are executed in ``_PARTS`` order into this module's globals; later
fragments rely on names defined by earlier ones, so keep the list ordered.
"""

import os as _os

from _split_loader import load_parts as _load_parts

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.dirname(_HERE)
__file__ = _os.path.join(_REPO, 'build_report.py')

_PARTS = [
'part_00_as_finite.py',
'part_01_sec_p1l.py',
]

_load_parts(globals(), _HERE, _PARTS)
del _load_parts

if __name__ == '__main__':
    _run_cli()
