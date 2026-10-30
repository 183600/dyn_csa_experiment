"""Experiment driver moved from v7_supp.py into the measurement package."""

import os as _os

from _split_loader import load_split_package

_HERE = _os.path.dirname(_os.path.abspath(__file__))

_PARTS = [
'part_00_soft_check_call.py',
'part_01_run_warmup_cell.py',
'part_02_run_lenphase_cell.py',
]

load_split_package(globals(), _HERE, _PARTS, repo_depth=2,
                   historical_path='v7_supp.py')
del load_split_package

if __name__ == '__main__':
    _run_cli()
