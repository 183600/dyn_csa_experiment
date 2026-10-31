"""Split implementation package for build_report.py.

Fragments are executed in ``_PARTS`` order into this module's globals; later
fragments rely on names defined by earlier ones, so keep the list ordered.
"""

from _split_loader import load_root_package

_PARTS = [
'part_00_as_finite.py',
'part_01_sec_p1l.py',
]

load_root_package(globals(), __file__, _PARTS,
                  historical_path='build_report.py')
del load_root_package

if __name__ == '__main__':
    _run_cli()
