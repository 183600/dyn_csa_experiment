"""v10 mechanism / scaling / statistics driver (`python -m explib.corruption`).

Modes: `full` (default), `smoke`, `analysis`, `report` and
`phase <name>` for P3MT / P3MTC / P3MP / P3T / P3SS / P3SL / P3SL2 / P3SL3.
It runs the distractor-injection mechanism probe, the crossover scaling panels,
the selector controls and the top-k seed completion.
"""

from _split_loader import load_driver_package

_PARTS = [
'part_00_make_guard.py',
'part_01_v10_analysis.py',
]

load_driver_package(globals(), __file__, _PARTS,
                    historical_path='v10_supp.py')
del load_driver_package

if __name__ == '__main__':
    _run_cli()
