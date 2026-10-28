"""(compatibility shim) implementation lives in _data_guard."""

import sys as _sys
import _data_guard as _impl

_sys.modules[__name__] = _impl
