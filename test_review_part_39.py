"""CPU checks for look-history attestation and the aggregate audit verdict.

Two reproducibility gaps are covered here:

1. ``look_history.json`` is the evidence behind the optional-stopping
   correction, but it is a mutable JSON file.  Deleting an early look leaves a
   shorter series whose p-values still replay, so recomputation alone cannot
   tell a one-look panel from a truncated multi-look one.  Every recorded look
   is now attested by a ``panel_look`` event in the hash-chained
   ``experiments.jsonl``; the audit cross-checks the two.

2. ``python audit_gates.py report`` wrote every child verdict but had no
   aggregate status, so the command exited 0 even when a gate failed.  The
   report now carries an explicit aggregate status and ``make audit`` exits
   non-zero on a real failure (while a source-only checkout with no local
   artifacts stays a warning, not a failure).

No test split, checkpoint or GPU is read.
"""
import contextlib
import io
import json
import os
import sys as _sys
import tempfile

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


_FP = ('steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_'
       'dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX')
_DELTAS = [-1.34, -1.18, -1.46, -1.21, -0.7, -1.1, 0.42, -0.48, 0.18, -0.04]
_DESIGN = {'dynamic_blocking': [('csa_dynamic', 'csa_fixed', 'x')]}
_KEY = 'component::dynamic_blocking::csa_dynamic|csa_fixed||'


def _summary(n):
    knobs = L.training_knobs()
    out = {}
    for i in range(n):
        out[f'csa_dynamic::seed{i}'] = dict(
            variant='csa_dynamic', seed=i, run_cfg=_FP, steps=1, tokens_seen=1,
            ppl=50.0 + _DELTAS[i], params=100, train_knobs=knobs)
        out[f'csa_fixed::seed{i}'] = dict(
            variant='csa_fixed', seed=i, run_cfg=_FP, steps=1, tokens_seen=1,
            ppl=50.0, params=100, train_knobs=knobs)
    return out


def _record_two_looks(td):
    recs6, _ = L._variant_seed_map(_summary(6))
    L.record_panel_look(td, recs6, design=_DESIGN, baselines={})
    recs10, _ = L._variant_seed_map(_summary(10))
    return L.record_panel_look(td, recs10, design=_DESIGN, baselines={})


def _read(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def _write(path, obj):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh)


def test_recorded_looks_are_attested_and_replay_clean():
    with tempfile.TemporaryDirectory() as td:
        hist = _record_two_looks(td)
        assert hist['n_looks'] == 2, hist
        assert hist['attestation']['kind'] == L.LOOK_ATTESTATION_KIND, hist
        for look in hist['looks']:
            assert look['look_sha256'] == L.look_entry_digest(look), look
        # Re-aggregating unchanged data must not append a second attestation
        # event for the same look.
        recs10, _ = L._variant_seed_map(_summary(10))
        L.record_panel_look(td, recs10, design=_DESIGN, baselines={})
        with open(os.path.join(td, 'experiments.jsonl'), encoding='utf-8') as fh:
            events = [json.loads(line) for line in fh if line.strip()]
        pl = [e for e in events if e.get('kind') == L.LOOK_LEDGER_EVENT_KIND]
        assert len(pl) == 2, [e.get('kind') for e in events]
        rep = L.verify_look_history_attestation(td, hist)
        assert rep['status'] == 'ok', rep
        assert rep['attested'] is True and rep['n_events'] == 2, rep


def test_truncated_look_history_fails_against_the_ledger():
    with tempfile.TemporaryDirectory() as td:
        hist = _record_two_looks(td)
        truncated = json.loads(json.dumps(hist))
        truncated['looks'] = truncated['looks'][:1]
        _write(os.path.join(td, 'look_history.json'), truncated)
        rep = L.verify_look_history_attestation(td, truncated)
        assert rep['status'] == 'fail', rep
        assert any('no such entry' in p for p in rep['problems']), rep
        with contextlib.redirect_stdout(io.StringIO()):
            aud = A.audit_look_history(paths=[td])
        assert aud['status'] == 'fail', aud
        assert any('panel_look' in p or 'no such entry' in p
                   for p in aud['problems']), aud


def test_removing_the_attestation_block_is_detected():
    with tempfile.TemporaryDirectory() as td:
        hist = _record_two_looks(td)
        stripped = json.loads(json.dumps(hist))
        stripped.pop('attestation', None)
        _write(os.path.join(td, 'look_history.json'), stripped)
        rep = L.verify_look_history_attestation(td, stripped)
        assert rep['status'] == 'fail', rep
        assert any('no attestation block' in p for p in rep['problems']), rep
        with contextlib.redirect_stdout(io.StringIO()):
            aud = A.audit_look_history(paths=[td])
        assert aud['status'] == 'fail', aud


def test_missing_ledger_is_a_warning_not_a_silent_pass():
    with tempfile.TemporaryDirectory() as td:
        hist = _record_two_looks(td)
        os.remove(os.path.join(td, 'experiments.jsonl'))
        os.remove(os.path.join(td, 'experiments.jsonl.anchor.json'))
        rep = L.verify_look_history_attestation(td, hist)
        assert rep['status'] == 'warnings', rep
        assert any('cannot be cross-checked' in w for w in rep['warnings']), rep
        # An unchanged re-aggregation repairs the missing events instead of
        # leaving the panel permanently unattestable.
        recs10, _ = L._variant_seed_map(_summary(10))
        repaired = L.record_panel_look(td, recs10, design=_DESIGN, baselines={})
        assert repaired['n_looks'] == 2, repaired
        rep = L.verify_look_history_attestation(td, repaired)
        assert rep['status'] == 'ok' and rep['n_events'] == 2, rep


def test_history_edited_after_attestation_warns_on_the_digest():
    with tempfile.TemporaryDirectory() as td:
        hist = _record_two_looks(td)
        edited = json.loads(json.dumps(hist))
        c = edited['looks'][0]['contrasts'][_KEY]
        c['deltas'][0] = -9.0
        # Keep the recorded p internally consistent: without the ledger
        # digest this edit is invisible to the p-replay layer.
        c['p_exact_signflip'] = L.exact_signflip_p(c['deltas'])['p']
        _write(os.path.join(td, 'look_history.json'), edited)
        rep = L.verify_look_history_attestation(td, edited)
        assert rep['status'] == 'warnings', rep
        assert any('no longer match the digest' in w for w in rep['warnings']), rep


def test_tampered_ledger_event_fails_the_attestation():
    with tempfile.TemporaryDirectory() as td:
        hist = _record_two_looks(td)
        path = os.path.join(td, 'experiments.jsonl')
        with open(path, encoding='utf-8') as fh:
            lines = [line for line in fh if line.strip()]
        obj = json.loads(lines[1])
        obj['n_contrasts'] = 999  # edit the chained body without re-hashing
        lines[1] = json.dumps(obj, sort_keys=True) + '\n'
        with open(path, 'w', encoding='utf-8') as fh:
            fh.writelines(lines)
        rep = L.verify_look_history_attestation(td, hist)
        assert rep['status'] == 'fail', rep
        assert any('does not replay' in p for p in rep['problems']), rep


def test_legacy_history_is_backfilled_on_first_upgrade():
    # An existing results directory may carry a pre-attestation look file.
    # Its first re-aggregation must attest the retained looks too, otherwise
    # every upgraded panel would be permanently "missing an event".
    with tempfile.TemporaryDirectory() as td:
        recs6, _ = L._variant_seed_map(_summary(6))
        snap = L.panel_look_snapshot(recs6, design=_DESIGN, baselines={})
        legacy = {'schema': 1, 'looks': [{
            'look': 1, 'recorded_at': '2026-01-01T00:00:00',
            'n_contrasts': len(snap), 'contrasts': snap}]}
        _write(os.path.join(td, 'look_history.json'), legacy)
        recs10, _ = L._variant_seed_map(_summary(10))
        hist = L.record_panel_look(td, recs10, design=_DESIGN, baselines={})
        assert hist['n_looks'] == 2, hist
        rep = L.verify_look_history_attestation(td, hist)
        assert rep['status'] == 'ok' and rep['n_events'] == 2, rep


def test_legacy_look_history_is_untouched_by_attestation():
    # A history written before attestation existed (and no ledger events) must
    # remain legacy, not be retroactively failed.
    legacy = {'schema': 1, 'looks': [
        {'look': 1, 'contrasts': {_KEY: {
            'seeds': [0, 1], 'deltas': [-1.0, -2.0],
            'p_exact_signflip': L.exact_signflip_p([-1.0, -2.0])['p']}}},
    ]}
    with tempfile.TemporaryDirectory() as td:
        _write(os.path.join(td, 'look_history.json'), legacy)
        rep = L.verify_look_history_attestation(td, legacy)
        assert rep['status'] == 'legacy', rep
        assert rep['attested'] is False and rep['n_events'] == 0, rep


def test_aggregate_report_status_is_explicit_and_actionable():
    ok = {'a': {'status': 'ok'}, 'b': {'status': 'not_verifiable_locally'}}
    assert A.audit_report_status(ok)['status'] == 'ok', A.audit_report_status(ok)
    warn = {'a': {'status': 'ok'}, 'b': {'status': 'ok', 'problems': ['x']}}
    got = A.audit_report_status(warn)
    assert got['status'] == 'warnings' and got['gates_warned'] == ['b'], got
    assert got['gates_with_problems'] == ['b'], got
    bad = {'a': {'status': 'ok'}, 'b': {'status': 'fail', 'problems': ['x']}}
    got = A.audit_report_status(bad)
    assert got['status'] == 'fail' and got['exit_code'] == 1, got
    assert got['gates_failed'] == ['b'], got
    empty = A.audit_report_status({})
    assert empty['status'] == 'ok' and empty['n_gates'] == 0, empty


def test_report_cli_exits_nonzero_on_aggregate_failure():
    old_argv = list(_sys.argv)
    old_run = A.run_report
    A.run_report = lambda *a, **k: {'status': 'fail'}
    try:
        _sys.argv = ['audit_gates.py', 'report']
        try:
            A._run_cli()
        except SystemExit as e:
            assert e.code == 1, e.code
        else:
            raise AssertionError('a failing aggregate report must exit non-zero')
    finally:
        A.run_report = old_run
        _sys.argv = old_argv
