"""Split implementation package for csa_kernels.py.

Fragments are executed in ``_PARTS`` order into this module's globals; later
fragments rely on names defined by earlier ones, so keep the list ordered.
"""

from _split_loader import load_root_package

_PARTS = [
'part_00_env_mode.py',
'part_01_attn_bwd_ref.py',
'part_02_sparse_layer_forward.py',
]

load_root_package(globals(), __file__, _PARTS,
                  historical_path='csa_kernels.py')
del load_root_package
