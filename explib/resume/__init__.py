"""Experiment driver moved from v9_supp.py into the measurement package."""

import os as _os

from _split_loader import load_parts as _load_parts

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.dirname(_os.path.dirname(_HERE))
__file__ = _os.path.join(_REPO, 'v9_supp.py')

_PARTS = [
'part_00_v9_block_forward.py',
]

_load_parts(globals(), _HERE, _PARTS)
del _load_parts

if __name__ == '__main__':
    _run_cli()
