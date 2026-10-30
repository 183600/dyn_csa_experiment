"""Split implementation package for csa_kernels.py.

Fragments are executed in ``_PARTS`` order into this module's globals; later
fragments rely on names defined by earlier ones, so keep the list ordered.
"""

import os as _os

from _split_loader import load_split_package

_HERE = _os.path.dirname(_os.path.abspath(__file__))

_PARTS = [
'part_00_env_mode.py',
'part_01_attn_bwd_ref.py',
'part_02_sparse_layer_forward.py',
]

load_split_package(globals(), _HERE, _PARTS, repo_depth=1,
                   historical_path='csa_kernels.py')
del load_split_package
