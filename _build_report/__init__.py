"""Split implementation package for build_report.py.

Fragments are executed in ``_PARTS`` order into this module's globals; later
fragments rely on names defined by earlier ones, so keep the list ordered.
"""

import os as _os

from _split_loader import load_split_package

_HERE = _os.path.dirname(_os.path.abspath(__file__))

_PARTS = [
'part_00_as_finite.py',
'part_01_sec_p1l.py',
]

load_split_package(globals(), _HERE, _PARTS, repo_depth=1,
                   historical_path='build_report.py')
del load_split_package

if __name__ == '__main__':
    _run_cli()
