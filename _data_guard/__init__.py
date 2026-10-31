"""Split implementation package for data_guard.py.

Fragments are executed in ``_PARTS`` order into this module's globals; later
fragments rely on names defined by earlier ones, so keep the list ordered.
"""

from _split_loader import load_root_package

_PARTS = [
'part_00_UnhashableKey.py',
'part_01_LeakageSafePreprocessor.py',
'part_02_LeakageSafeTargetEncoder.py',
'part_03_clean_training_rows.py',
'part_04_data_quality_report.py',
]

load_root_package(globals(), __file__, _PARTS,
                  historical_path='data_guard.py')
del load_root_package
