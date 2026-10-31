"""v11 statistical close-out driver (`python -m explib.length`).

Modes: `full` (default), `smoke`, `analysis`, `report` and
`phase <name>` for P4MT / P4MP / P4SS / P4F / P4SL.  It extends the
distractor probe, the crossover panels and the d=384 paired arms to the
pre-declared seed counts.
"""

from _split_loader import load_driver_package

_PARTS = [
'part_00_make_guard.py',
]

load_driver_package(globals(), __file__, _PARTS,
                    historical_path='v11_supp.py')
del load_driver_package

if __name__ == '__main__':
    _run_cli()
