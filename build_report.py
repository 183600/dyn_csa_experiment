"""(compatibility shim) implementation lives in _build_report."""

import sys as _sys
import _build_report as _impl

_sys.modules[__name__] = _impl

if __name__ == '__main__':
    _impl._run_cli()
