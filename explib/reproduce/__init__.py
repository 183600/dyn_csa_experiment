"""Cross-run reproduction comparator (``python -m explib.reproduce``).

The implementation lives in ``explib/_parts/part_37_reproduction.py`` so it is
available from the shared ``exp_lib`` namespace; this package is only the CLI
entry point.
"""

from explib import _reproduction_cli


def _run_cli():
    raise SystemExit(_reproduction_cli())


if __name__ == '__main__':
    _run_cli()
