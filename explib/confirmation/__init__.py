"""v12 confirmation driver (``python -m explib.confirmation``).

Separates the pre-registered final evaluation from the exploratory panels.
Modes: ``smoke``, ``plan``, ``dynamic-plan``, ``power``, ``phase <name>``
(P5C / P5H / P5PW / P5T / P5MP / P5S2 / P5S3), ``analysis``, ``report``,
``full`` and ``auto``.  Only P5T reads the held-out test split; the other
modes validate the plan, train the frozen arms, or rebuild statistics from
local artifacts.
"""

from _split_loader import load_driver_package

_PARTS = [
'part_00_make_guard.py',
'part_01_finaltest_bylen_cell.py',
'part_02_doc_level_analysis.py',
'part_03_finaltest_stats.py',
'part_04_plan_check.py',
]

load_driver_package(globals(), __file__, _PARTS,
                    historical_path='v12_supp.py')
del load_driver_package

if __name__ == '__main__':
    _run_cli()
