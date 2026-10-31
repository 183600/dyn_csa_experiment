"""Regression checks: every 2x2 simple effect must itself be single-factor.

``factorial_design_audit`` already required the two simple-effect diffs to be
parallel and to avoid an attention-``kind`` swap.  It only checked that the
union of moved keys was a subset of all keys ever declared for the component,
so a bundle assembled from two different declared families (for example
``use_sink`` plus ``full_sink``) could still be read as one factor.  The gate
now applies the same ``cfg_keys_single_factor`` family test used by the
leave-one-out, sweep, probe and frozen-plan gates.
"""
import dataclasses
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


def test_shipped_factorial_axes_are_single_factor():
    rep = L.factorial_design_audit()
    assert rep['status'] == 'ok', rep['violations']
    for name, rec in rep['factorials'].items():
        assert rec['complete'] is True, (name, rec)
        assert set(rec['factor_single_factor']) == set(rec['factors']), (name, rec)
        assert all(rec['factor_single_factor'].values()), (name, rec)


def test_factorial_axis_spanning_two_families_is_refused():
    base = L.AttnCfg(kind='csa', dynamic=False, block_size=4,
                     overlap=L.OVERLAP, index_topk=32)

    def _dyn():
        return dataclasses.replace(base, dynamic=True,
                                   chunking='cosine_learnable',
                                   target_block_tokens=4)

    # Both sink levels move the same two keys; the diff is parallel, avoids a
    # kind swap, and every key is declared for `attention_sink`.  Only the
    # family test can see that it is two interventions, not one.
    def factory(_n, variant):
        off = dataclasses.replace(base, use_sink=False)
        off_dyn = dataclasses.replace(_dyn(), use_sink=False)
        on = dataclasses.replace(base, use_sink=True, full_sink=True)
        on_dyn = dataclasses.replace(_dyn(), use_sink=True, full_sink=True)
        return {'a0': [off], 'a1': [off_dyn],
                'b0': [on], 'b1': [on_dyn]}[variant]

    design = {'fixture': {
        'factors': ('dynamic_blocking', 'attention_sink'),
        'cells': {('control', 'control'): 'a0',
                  ('active', 'control'): 'a1',
                  ('control', 'active'): 'b0',
                  ('active', 'active'): 'b1'},
        'hypothesis': 'dynamic_blocking_x_attention_sink',
    }}
    rep = L.factorial_design_audit(
        design, factory=factory, present={'a0', 'a1', 'b0', 'b1'})
    assert rep['status'] == 'fail', rep
    assert rep['factorials']['fixture']['factor_single_factor'] == {
        'dynamic_blocking': True, 'attention_sink': False}, rep['factorials']
    assert any('attention_sink' in v and 'declared single-factor family' in v
               for v in rep['violations']), rep['violations']
