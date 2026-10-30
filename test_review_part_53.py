"""CPU checks for the v10/v11 distractor-probe single-factor contract.

The P3MP/P4MP/P5MP probe quotes ``dense/randidx/allblocks - learned`` as a
mechanism reading, but its contrasts had no factor declaration and no
definition-site audit.  These checks pin the new registry: the dense reference
must stay architecture-level, the inference-time selector overrides and the
from-scratch selector controls must be verified single factors, and a record
whose declaration was not verified must not keep a ``single_factor=True``
label.  No result panel is read.
"""
import contextlib
import copy
import inspect
import io
import os
import sys as _sys

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

import pytest as _pytest


def _probe_rep(**kwargs):
    import explib.rope as V7
    return L.probe_contrast_factor_audit(
        factory=V7.make_layer_cfgs_v7, verbose=False, **kwargs)


def test_shipped_probe_contrasts_are_declared_and_verified():
    rep = _probe_rep()
    assert rep['kind'] == 'probe_contrast_factors'
    assert rep['status'] == 'ok', rep['violations']
    expected = {'dense', 'randidx', 'allblocks', 'csa_randtrain', 'csa_alltrain'}
    assert set(rep['pairs']) == expected, rep['pairs']
    assert all(e['verified'] is True for e in rep['pairs'].values()), rep['pairs']
    dense = rep['pairs']['dense']
    assert dense['actual_keys'] == ['kind'], dense
    assert dense['declared_single_factor'] is False, dense
    assert dense['actual_single_factor'] is False, dense
    assert dense['contrast_level'] == 'architecture_level', dense
    for tag in ('randidx', 'allblocks', 'csa_randtrain', 'csa_alltrain'):
        entry = rep['pairs'][tag]
        assert entry['actual_single_factor'] is True, (tag, entry)
        assert entry['declared_single_factor'] is True, (tag, entry)
        assert entry['actual_keys'] == sorted(
            L.PROBE_CONTRAST_FACTORS[tag]['factors']), (tag, entry)
        assert entry['contrast_level'] == 'single_factor', (tag, entry)
        assert not entry['extra_keys'] and not entry['missing_keys'], (tag, entry)
    # The registry is a closed table; a second copy would drift.
    assert L.PROBE_FACTOR_DECLARATIONS is L.PROBE_CONTRAST_FACTORS
    assert L.PROBE_CONTRAST_FACTOR_DECLARATIONS is L.PROBE_CONTRAST_FACTORS
    assert L.DISTRACTOR_PROBE_CONTRAST_FACTORS is L.PROBE_CONTRAST_FACTORS
    # NIAH and the probe share the inference override map, not a private copy.
    assert L.PROBE_CONTRAST_FACTORS['randidx']['override'] == 'randidx'
    assert L.PROBE_CONTRAST_FACTORS['allblocks']['override'] == 'allblocks'


def test_probe_declaration_drift_fails_closed():
    base = copy.deepcopy(L.PROBE_CONTRAST_FACTORS)
    # A second key in a single-factor declaration is refused.
    bad = copy.deepcopy(base)
    bad['randidx'] = dict(bad['randidx'], factors=['indexer_mode', 'index_topk'])
    rep = L.probe_contrast_factor_audit(
        bad, factory=L.make_layer_cfgs, verbose=False)
    assert rep['status'] == 'fail', rep
    assert rep['pairs']['randidx']['verified'] is False, rep['pairs']['randidx']
    assert any(('undeclared key' in v) or ('do not differ' in v)
               for v in rep['violations']), rep['violations']
    # A declared key that no longer differs is refused.
    bad = copy.deepcopy(base)
    bad['csa_alltrain'] = dict(bad['csa_alltrain'], factors=['use_sink'])
    rep = L.probe_contrast_factor_audit(
        bad, factory=L.make_layer_cfgs, verbose=False)
    assert rep['status'] == 'fail', rep
    assert rep['pairs']['csa_alltrain']['verified'] is False, rep['pairs']['csa_alltrain']
    # An unbuildable trained arm is a definition defect, not a crash.
    bad = copy.deepcopy(base)
    bad['csa_randtrain'] = dict(bad['csa_randtrain'], b='no_such_arm')
    rep = L.probe_contrast_factor_audit(
        bad, factory=L.make_layer_cfgs, verbose=False)
    assert rep['status'] == 'fail', rep
    assert any('construction failed' in v for v in rep['violations']), rep['violations']
    # A malformed declaration is refused before any factory call.
    for field, value in (('factors', []), ('factors', ['']),
                         ('single_factor', 'yes')):
        bad = copy.deepcopy(base)
        bad['dense'] = dict(bad['dense'], **{field: value})
        rep = L.probe_contrast_factor_audit(
            bad, factory=L.make_layer_cfgs, verbose=False)
        assert rep['status'] == 'fail', (field, value, rep)
    bad = copy.deepcopy(base)
    bad['randidx'] = dict(bad['randidx'], override='not_an_override')
    rep = L.probe_contrast_factor_audit(
        bad, factory=L.make_layer_cfgs, verbose=False)
    assert rep['status'] == 'fail', rep
    assert any('not in the shared' in v for v in rep['violations']), rep['violations']


def test_probe_observed_tags_close_the_record_side():
    tags = ['dense', 'randidx', 'allblocks', 'csa_randtrain', 'csa_alltrain']
    rep = _probe_rep(observed_tags=tags)
    assert rep['status'] == 'ok', rep
    assert set(rep['observed_tags']) == set(tags), rep
    assert not rep['undeclared_observed_tags'], rep
    # An analysis that emits a new contrast without a declaration must fail
    # closed instead of silently attaching no factor label.
    rep = _probe_rep(observed_tags=tags + ['retrain-gap'])
    assert rep['status'] == 'fail', rep
    assert rep['undeclared_observed_tags'] == ['retrain-gap'], rep
    assert any('without a factor declaration' in v for v in rep['violations']), rep['violations']
    # A panel that did not measure every declared tag is visible, not fatal.
    rep = _probe_rep(observed_tags=['randidx'])
    assert rep['status'] == 'ok', rep
    assert rep['warnings'], rep


def test_probe_metadata_clears_unverified_single_factor():
    rep = _probe_rep()
    for tag in ('randidx', 'allblocks'):
        rec = {'single_factor': True}
        L.attach_probe_factor_metadata(rec, tag, rep)
        assert rec['single_factor'] is True, (tag, rec)
        assert rec['factor_declaration_verified'] is True, (tag, rec)
        assert rec['factors'] == L.PROBE_CONTRAST_FACTORS[tag]['factors'], (tag, rec)
    # Dense is declared but explicitly not a single factor.
    dense = {'single_factor': False}
    L.attach_probe_factor_metadata(dense, 'dense', rep)
    assert dense['single_factor'] is False, dense
    assert dense['contrast_level'] == 'architecture_level', dense
    assert dense['factor_declaration_verified'] is True, dense
    # A failed definition-site audit must strip a True label, fail closed.
    bad = {'pairs': {'randidx': {'verified': False}}}
    rec = {'single_factor': True}
    L.attach_probe_factor_metadata(rec, 'randidx', bad)
    assert rec['single_factor'] is False, rec
    assert rec['declared_single_factor'] is True, rec
    assert rec['factor_declaration_verified'] is False, rec
    # An undeclared tag is not assumed to be a single factor.
    rec = {'single_factor': True}
    L.attach_probe_factor_metadata(rec, 'made_up', rep)
    assert rec['single_factor'] is False, rec
    assert rec['contrast_level'] == 'undeclared', rec
    assert rec['factor_declaration_verified'] is False, rec


def test_v10_v11_analysis_wire_the_probe_factor_report():
    import explib.corruption as V10
    import explib.length as V11
    for fn in (V10.v10_analysis, V11.v11_analysis):
        src = inspect.getsource(fn)
        assert 'probe_contrast_factor_audit' in src, fn.__name__
        assert 'attach_probe_factor_metadata' in src, fn.__name__
        assert "'factor_audit'" in src or '"factor_audit"' in src, fn.__name__


def test_audit_gate_and_report_wire_the_probe_block():
    import audit_gates as A
    with contextlib.redirect_stdout(io.StringIO()):
        direct = A.audit_probe_contrasts()
        sf = A.audit_single_factor()
    assert direct['status'] == 'ok', direct['violations']
    assert sf['probe_contrasts']['status'] == 'ok', sf['probe_contrasts']
    assert sf['status'] == 'ok', sf['status']
    src = inspect.getsource(A.run_report)
    assert "'probe_contrasts'" in src
    src_cli = inspect.getsource(A._run_cli)
    for alias in ('probe_contrasts', 'probe_controls', 'probe_factor',
                  'distractor_probe'):
        assert alias in src_cli, alias


def test_probe_registry_survives_an_audit_unchanged():
    before = copy.deepcopy(L.PROBE_CONTRAST_FACTORS)
    _probe_rep()
    assert L.PROBE_CONTRAST_FACTORS == before, 'the audit mutated the registry'
