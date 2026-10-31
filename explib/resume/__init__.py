"""v9 close-out driver (`python -m explib.resume`).

Modes: `full` (default), `smoke`, `analysis`, `report` and
`phase <name>` for P2T / P2S3.  P2T completes the sequence-2048 top-k sweep
at `batch_size=1`; P2S3 adds the third seed to the RoPE 20k run.
"""

from _split_loader import load_driver_package

_PARTS = [
'part_00_v9_block_forward.py',
]

load_driver_package(globals(), __file__, _PARTS,
                    historical_path='v9_supp.py')
del load_driver_package

if __name__ == '__main__':
    _run_cli()
