"""Experiment driver moved from v12_supp.py into the measurement package."""

import os as _os

from _split_loader import load_parts as _load_parts

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.dirname(_os.path.dirname(_HERE))
__file__ = _os.path.join(_REPO, 'v12_supp.py')

_PARTS = [
'part_00_make_guard.py',
'part_01_finaltest_bylen_cell.py',
'part_02_doc_level_analysis.py',
'part_03_finaltest_stats.py',
'part_04_plan_check.py',
]

_load_parts(globals(), _HERE, _PARTS)
del _load_parts

if __name__ == '__main__':
    _run_cli()
