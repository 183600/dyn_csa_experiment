#!/usr/bin/env python3
"""Compatibility entry point for the CPU review suite.

The checks are standard ``test_*.py`` modules discovered by pytest.  Running
this file directly invokes pytest so existing CI / ``make test`` commands keep
working without the historical script shell.
"""
import os

if __name__ == '__main__':
    import pytest

    raise SystemExit(pytest.main([os.path.dirname(os.path.abspath(__file__))]))
