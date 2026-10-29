"""Regression checks for control-ladder audits and extreme-scale PPL spreads.

These are zero-GPU design checks.  They cover two failure modes that ordinary
design fixtures do not exercise:

* a design audit must not mutate the process RNG stream (a later training cell
  would otherwise inherit a different data/dropout order), and must fail closed
  on malformed nested declarations instead of raising ``TypeError``;
* a finite PPL panel can be extreme enough that naive ``np.std`` overflows even
  though the standard error is representable.  The analysis helpers therefore
  use the same guarded mean/SD as the paired-effect statistics.
"""
import copy

import numpy as np
import pytest
import torch

import exp_lib as L


def test_sink_audit_preserves_rng_state():
    torch.manual_seed(20261029)
    before = torch.get_rng_state().clone()
    report = L.sink_control_audit()
    assert report['status'] == 'ok', report['violations']
    assert torch.equal(before, torch.get_rng_state())


def test_control_audits_validate_declared_modes():
    sink = copy.deepcopy(L.SINK_CONTROL_DESIGN)
    sink['levels']['learned']['sink_mode'] = 'random'
    assert L.sink_control_audit(ladder=sink)['status'] == 'fail'
    sink = copy.deepcopy(L.SINK_CONTROL_DESIGN)
    sink['levels']['absent']['sink_mode'] = 'random'
    assert L.sink_control_audit(ladder=sink)['status'] == 'fail'
    content = copy.deepcopy(L.CONTENT_CONTROL_DESIGN)
    content['levels']['aligned']['content_mode'] = 'zero'
    assert L.content_control_audit(ladder=content)['status'] == 'fail'


def test_control_audits_fail_closed_on_nested_malformations():
    sink = copy.deepcopy(L.SINK_CONTROL_DESIGN)
    sink['absence_keys']['fixed'] = 1.5
    assert L.sink_control_audit(ladder=sink)['status'] == 'fail'
    sink = copy.deepcopy(L.SINK_CONTROL_DESIGN)
    sink['levels']['randomized']['arms']['fixed'] = {'not': 'an arm id'}
    assert L.sink_control_audit(ladder=sink)['status'] == 'fail'
    content = copy.deepcopy(L.CONTENT_CONTROL_DESIGN)
    content['levels']['misaligned']['arms']['fixed'] = ['not', 'an arm id']
    assert L.content_control_audit(ladder=content)['status'] == 'fail'


def test_extreme_finite_ppl_spread_stays_finite():
    ppls = [1.0e200, 1.1e200, 0.9e200]
    ci = L._seed_t_ci(ppls)
    assert ci is not None and all(np.isfinite(ci))
    assert np.isfinite(L._scaled_std_or_raw(ppls))
    # Ordinary magnitudes keep the historical numpy value exactly.
    ordinary = [1.0, 2.0, 3.0]
    assert L._scaled_std_or_raw(ordinary) == float(np.std(ordinary, ddof=1))
