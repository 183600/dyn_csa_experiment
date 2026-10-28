"""Split implementation package for audit_gates.py."""

import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.dirname(_HERE)
__file__ = _os.path.join(_REPO, 'audit_gates.py')

_PARTS = [
'part_00_load.py',
'part_01_audit_split_discipline.py',
'part_02_audit_sample_design.py',
'part_03_audit_methodology.py',
'part_04_audit_selection.py',
'part_05_audit_confirmation_verdicts.py',
]

for _part in _PARTS:
    _path = _os.path.join(_HERE, '_parts', _part)
    with open(_path, encoding='utf-8') as _fh:
        _src = _fh.read()
    exec(compile(_src, _path, 'exec'), globals())
del _part, _path, _fh, _src

# Keep module introspection on the implementation fragment (the historical
# script body lived in one file) while REPO still points at the repository root.
__file__ = _os.path.join(_HERE, '_parts', _PARTS[-1])
REPO = _REPO

if __name__ == '__main__':
    _run_cli()
