"""(compatibility shim) implementation lives in explib."""

CODE_SEMANTICS = 'v11.128'

import sys as _sys
import explib as _impl

_sys.modules[__name__] = _impl
