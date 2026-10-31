"""Experiment driver moved from v12_supp.py into the measurement package."""

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
