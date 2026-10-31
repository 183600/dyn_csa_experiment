"""v7 reviewer-response driver (`python -m explib.rope`).

Modes: `full` (default), `smoke`, `analysis` and `phase <name>` for
P0R / P0W / P0E / P2S / P1L / P1T.  The suite covers RoPE + QK-norm, dense
warm-up, the 40k-step long run, scale re-seeding, length extrapolation and the
sequence-2048 top-k sweep.
"""

from _split_loader import load_driver_package

_PARTS = [
'part_00_soft_check_call.py',
'part_01_run_warmup_cell.py',
'part_02_run_lenphase_cell.py',
]

load_driver_package(globals(), __file__, _PARTS,
                    historical_path='v7_supp.py')
del load_driver_package

if __name__ == '__main__':
    _run_cli()
