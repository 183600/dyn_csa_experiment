"""Experiment driver moved from v9_supp.py into the measurement package."""

import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.dirname(_os.path.dirname(_HERE))
__file__ = _os.path.join(_REPO, 'v9_supp.py')

_PARTS = [
'part_00_v9_block_forward.py',
]

for _part in _PARTS:
    _path = _os.path.join(_HERE, '_parts', _part)
    with open(_path, encoding='utf-8') as _fh:
        _src = _fh.read()
    exec(compile(_src, _path, 'exec'), globals())
del _part, _path, _fh, _src

if __name__ == '__main__':
    _run_cli()
