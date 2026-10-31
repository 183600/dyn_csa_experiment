"""Experiment driver moved from open_block_supp.py into the measurement package."""

from _split_loader import load_driver_package

_PARTS = [
'part_00_variant.py',
]

load_driver_package(globals(), __file__, _PARTS,
                    historical_path='open_block_supp.py')
del load_driver_package

if __name__ == '__main__':
    _run_cli()
