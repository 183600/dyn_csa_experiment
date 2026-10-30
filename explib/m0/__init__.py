"""Experiment driver moved from m0_supp.py into the measurement package."""

import os as _os

from _split_loader import load_split_package

_HERE = _os.path.dirname(_os.path.abspath(__file__))

_PARTS = [
'part_00_git_push.py',
]

load_split_package(globals(), _HERE, _PARTS, repo_depth=2,
                   historical_path='m0_supp.py')
del load_split_package

if __name__ == '__main__':
    _run_cli()
