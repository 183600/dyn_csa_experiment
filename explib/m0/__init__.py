"""M0 / S1 driver for the B-tier data path and scale panels.

Modes: ``params`` (zero-GPU parameter audit), ``data`` (FineWeb-Edu cache),
``smoke``, ``kernel`` (G0 kernel/eager validation), ``analysis``, ``full``
(S1 by default, S2 via ``M0_PANEL``) and ``auto`` (gate, then panel chain).
The training modes are explicit; the remaining modes do not start a panel.
"""

from _split_loader import load_driver_package

_PARTS = [
'part_00_git_push.py',
]

load_driver_package(globals(), __file__, _PARTS,
                    historical_path='m0_supp.py')
del load_driver_package

if __name__ == '__main__':
    _run_cli()
