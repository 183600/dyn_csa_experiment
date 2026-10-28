"""(compatibility shim) implementation lives in _audit_gates."""

import sys as _sys
import _audit_gates as _impl

_sys.modules[__name__] = _impl

if __name__ == '__main__':
    _impl._run_cli()
