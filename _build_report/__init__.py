"""Split implementation package for build_report.py.

Fragments are executed in ``_PARTS`` order into this module's globals; later
fragments rely on names defined by earlier ones, so keep the list ordered.
"""

import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.dirname(_HERE)
__file__ = _os.path.join(_REPO, 'build_report.py')

_PARTS = [
'part_00_as_finite.py',
'part_01_sec_p1l.py',
]

for _part in _PARTS:
    _path = _os.path.join(_HERE, '_parts', _part)
    with open(_path, encoding='utf-8') as _fh:
        _src = _fh.read()
    exec(compile(_src, _path, 'exec'), globals())
del _part, _path, _fh, _src

if __name__ == '__main__':
    _run_cli()
