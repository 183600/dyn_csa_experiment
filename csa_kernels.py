"""(compatibility shim) implementation lives in _csa_kernels."""

import sys as _sys
import _csa_kernels as _impl

_sys.modules[__name__] = _impl
