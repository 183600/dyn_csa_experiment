"""Compatibility shim: re-exports explib as the top-level module.

Historical script entry points import from ``exp_lib``; this shim ensures
the import path works regardless of whether the code is invoked as a package
or as a script from the repository root.
"""

CODE_SEMANTICS = 'v11.128'

import sys as _sys
import explib as _impl

_sys.modules[__name__] = _impl

if __name__ == '__main__':
    import warnings
    warnings.warn(
        'exp_lib is a shim; use ``import explib`` or ``python -m explib`` '
        'instead of running exp_lib as a script.',
        FutureWarning,
        stacklevel=2,
    )
