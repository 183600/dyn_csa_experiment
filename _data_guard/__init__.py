"""Split implementation package for data_guard.py."""

import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.dirname(_HERE)
__file__ = _os.path.join(_REPO, 'data_guard.py')

_PARTS = [
'part_00_UnhashableKey.py',
'part_01_LeakageSafePreprocessor.py',
'part_02_LeakageSafeTargetEncoder.py',
'part_03_clean_training_rows.py',
'part_04_data_quality_report.py',
]

for _part in _PARTS:
    _path = _os.path.join(_HERE, '_parts', _part)
    with open(_path, encoding='utf-8') as _fh:
        _src = _fh.read()
    exec(compile(_src, _path, 'exec'), globals())
del _part, _path, _fh, _src
