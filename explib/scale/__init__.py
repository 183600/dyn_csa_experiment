"""v8 scale close-out driver (`python -m explib.scale`).

Modes: `full` (default), `smoke`, `analysis` and `phase <name>` for
P1S / P2R.  P1S completes the scale-panel seeds; P2R runs the RoPE + QK-norm
20k-step long run.
"""

from _split_loader import load_driver_package

_PARTS = [
'part_00_make_guard.py',
]

load_driver_package(globals(), __file__, _PARTS,
                    historical_path='v8_supp.py')
del load_driver_package

if __name__ == '__main__':
    _run_cli()
