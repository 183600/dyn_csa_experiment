"""Open-block panel driver (``python -m explib.open_block``).

Varies the maximum block length against the rule for reading the current
block while keeping one top-k read budget.  Modes: ``smoke`` (synthetic
plumbing), ``pilot``, ``main``, ``aux``, ``extended`` and ``report``; only
the run modes train, and ``report`` rebuilds paired statistics from disk.
"""

from _split_loader import load_driver_package

_PARTS = [
'part_00_variant.py',
]

load_driver_package(globals(), __file__, _PARTS,
                    historical_path='open_block_supp.py')
del load_driver_package

if __name__ == '__main__':
    _run_cli()
