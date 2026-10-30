"""Split implementation package for data_guard.py.

Fragments are executed in ``_PARTS`` order into this module's globals; later
fragments rely on names defined by earlier ones, so keep the list ordered.
"""

import os as _os

from _split_loader import load_split_package

_HERE = _os.path.dirname(_os.path.abspath(__file__))

_PARTS = [
'part_00_UnhashableKey.py',
'part_01_LeakageSafePreprocessor.py',
'part_02_LeakageSafeTargetEncoder.py',
'part_03_clean_training_rows.py',
'part_04_data_quality_report.py',
]

load_split_package(globals(), _HERE, _PARTS, repo_depth=1,
                   historical_path='data_guard.py')
del load_split_package
