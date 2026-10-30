"""CPU review checks (split group 14): tamper-evident experiment ledger.

`experiments.jsonl` is the execution record the run-order and evaluation-order
audits trust.  These checks exercise the per-event hash chain, the tail anchor,
the legacy/crash-window statuses and the zero-GPU audit wiring.
"""
import contextlib
import hashlib
import inspect
import io
import json
import os
import sys as _sys
import tempfile
import shutil

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)

_ZERO = '0' * 64


def _append(d, event):
    with contextlib.redirect_stdout(io.StringIO()):
        return L.log_experiment_event(d, event)


def _events(path):
    with open(path, encoding='utf-8') as fh:
        return [json.loads(x) for x in fh.read().splitlines() if x.strip()]


def _rewrite(path, records):
    with open(path, 'w', encoding='utf-8') as fh:
        for rec in records:
            fh.write(json.dumps(rec) + '\n')


def test_experiment_ledger_chains_and_anchor_verifies():
    with tempfile.TemporaryDirectory() as d:
        for i in range(3):
            _append(d, {'kind': 'cell', 'i': i, 'status': 'ok'})
        path = os.path.join(d, 'experiments.jsonl')
        events = _events(path)
        assert [e['kind'] for e in events] == ['cell'] * 3
        led = [e['_ledger'] for e in events]
        assert [x['seq'] for x in led] == [0, 1, 2], led
        assert led[0]['prev'] == _ZERO
        assert led[1]['prev'] == led[0]['sha256']
        assert led[2]['prev'] == led[1]['sha256']
        assert led[2]['sha256'] == L.ledger_link_sha256(
            2, led[1]['sha256'], events[2])
        anchor = json.load(open(path + '.anchor.json', encoding='utf-8'))
        assert anchor['count'] == 3, anchor
        assert anchor['last_sha256'] == led[2]['sha256'], anchor
        rep = L.verify_experiment_ledger(path)
        assert rep['status'] == 'ok', rep
        assert rep['n_events'] == 3 and rep['n_chained'] == 3, rep
        assert rep['n_legacy'] == 0 and not rep['problems'], rep


def test_experiment_ledger_detects_edit_delete_and_truncation():
    with tempfile.TemporaryDirectory() as d:
        for i in range(3):
            _append(d, {'kind': 'cell', 'i': i})
        path = os.path.join(d, 'experiments.jsonl')
        original = _events(path)

        edited = [dict(x) for x in original]
        edited[1]['i'] = 99
        _rewrite(path, edited)
        rep = L.verify_experiment_ledger(path)
        assert rep['status'] == 'fail', rep
        assert any('does not reproduce' in p for p in rep['problems']), rep

        _rewrite(path, original)
        assert L.verify_experiment_ledger(path)['status'] == 'ok'

        _rewrite(path, [original[0], original[2]])
        rep = L.verify_experiment_ledger(path)
        assert rep['status'] == 'fail', rep
        assert any('deleted, inserted or reordered' in p
                   or 'does not match the previous event link' in p
                   for p in rep['problems']), rep

        _rewrite(path, original[:2])
        rep = L.verify_experiment_ledger(path)
        assert rep['status'] == 'fail', rep
        assert any('truncated' in p for p in rep['problems']), rep

        # Appending over a truncated tail must not re-anchor the shorter
        # history: the evidence has to stay visible until it is inspected.
        before = open(path, encoding='utf-8').read()
        assert _append(d, {'kind': 'after'}) is None
        assert open(path, encoding='utf-8').read() == before
        assert L.verify_experiment_ledger(path)['status'] == 'fail'

        # Dropping _ledger from the tail (and re-pinning the anchor to the raw
        # bytes) must not turn an edited chained event into a tolerated legacy
        # one: a legacy line after chaining is a fail.
        _rewrite(path, original)
        tail_legacy = [dict(x) for x in original]
        tail_legacy[-1].pop('_ledger')
        _rewrite(path, tail_legacy)
        raw_tail = open(path, encoding='utf-8').read().splitlines()[-1]
        anchor = json.load(open(path + '.anchor.json', encoding='utf-8'))
        anchor['last_sha256'] = hashlib.sha256(
            raw_tail.encode('utf-8')).hexdigest()
        anchor['bytes'] = os.path.getsize(path)
        with open(path + '.anchor.json', 'w', encoding='utf-8') as fh:
            json.dump(anchor, fh)
        rep = L.verify_experiment_ledger(path)
        assert rep['status'] == 'fail', rep
        assert any('legacy event after chained events' in p
                   for p in rep['problems']), rep


def test_experiment_ledger_anchor_states_are_explicit():
    with tempfile.TemporaryDirectory() as d:
        for i in range(2):
            _append(d, {'kind': 'cell', 'i': i})
        path = os.path.join(d, 'experiments.jsonl')
        saved = json.load(open(path + '.anchor.json', encoding='utf-8'))
        _append(d, {'kind': 'cell', 'i': 2})
        # Simulate an append that crashed before replacing the anchor.
        with open(path + '.anchor.json', 'w', encoding='utf-8') as fh:
            json.dump(saved, fh)
        rep = L.verify_experiment_ledger(path)
        assert rep['status'] == 'warnings', rep
        assert any('anchor is behind' in w for w in rep['warnings']), rep

        # A behind anchor whose prefix still matches is the legitimate crash
        # window: the next append rescans and writes a current anchor.
        _append(d, {'kind': 'cell', 'i': 3})
        rep = L.verify_experiment_ledger(path)
        assert rep['status'] == 'ok' and rep['n_events'] == 4, rep

        # A missing sidecar is a warning, never a silent green pass.
        os.unlink(path + '.anchor.json')
        rep = L.verify_experiment_ledger(path)
        assert rep['status'] == 'warnings', rep
        assert any('no tail anchor' in w for w in rep['warnings']), rep

        # A malformed anchor fails closed.
        with open(path + '.anchor.json', 'w', encoding='utf-8') as fh:
            fh.write('[]')
        rep = L.verify_experiment_ledger(path)
        assert rep['status'] == 'fail', rep
        assert any('not a JSON object' in p for p in rep['problems']), rep


def test_experiment_ledger_legacy_prefix_and_empty_states():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'experiments.jsonl')
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(json.dumps({'kind': 'old', 'n': 1}) + '\n')
            fh.write(json.dumps({'kind': 'old', 'n': 2}))  # no trailing newline
        assert L.verify_experiment_ledger(path)['status'] == 'not_verifiable_locally'
        _append(d, {'kind': 'new'})
        # The appended event started a chain after the legacy prefix; the
        # missing trailing newline was repaired without changing the link.
        events = _events(path)
        assert len(events) == 3 and events[-1]['_ledger']['seq'] == 2, events
        rep = L.verify_experiment_ledger(path)
        assert rep['status'] == 'ok', rep
        assert rep['n_legacy'] == 2 and rep['n_chained'] == 1, rep

        empty = os.path.join(d, 'empty.jsonl')
        open(empty, 'w').close()
        assert L.verify_experiment_ledger(empty)['status'] == 'not_verifiable_locally'
        assert L.verify_experiment_ledger(os.path.join(d, 'nope.jsonl'))[
            'status'] == 'not_verifiable_locally'


def test_experiment_ledger_reserved_key_and_json_safety():
    with tempfile.TemporaryDirectory() as d:
        _append(d, {'kind': 'cell'})
        _append(d, {'kind': 'spoof', '_ledger': {'schema': 1, 'seq': 99,
                                                 'prev': _ZERO,
                                                 'sha256': _ZERO},
                    'v': np.int64(3), 'bad': float('nan'),
                    'tags': {'b', 'a'}})
        path = os.path.join(d, 'experiments.jsonl')
        line = open(path, encoding='utf-8').read().splitlines()[-1]

        def _reject(token):
            raise AssertionError(f'non-standard JSON token {token!r} was written')

        ev = json.loads(line, parse_constant=_reject)
        assert ev['_ledger']['seq'] == 1, ev
        assert ev['_ledger']['prev'] == _events(path)[0]['_ledger']['sha256']
        assert ev['v'] == 3 and ev['bad'] is None, ev
        assert sorted(ev['tags']) == ['a', 'b'], ev
        assert L.verify_experiment_ledger(path)['status'] == 'ok'


def test_experiment_ledger_audit_status_and_wiring():
    with tempfile.TemporaryDirectory() as good_dir, \
            tempfile.TemporaryDirectory() as bad_dir:
        for i in range(2):
            _append(good_dir, {'kind': 'cell', 'i': i})
            _append(bad_dir, {'kind': 'cell', 'i': i})
        good_path = os.path.join(good_dir, 'experiments.jsonl')
        bad_path = os.path.join(bad_dir, 'experiments.jsonl')
        bad = _events(bad_path)
        bad[0]['i'] = -1
        _rewrite(bad_path, bad)

        rep = A.audit_experiment_ledger([good_dir, bad_path])
        assert rep['status'] == 'fail', rep
        assert rep['n_ledgers'] == 2 and rep['n_ok'] == 1, rep
        assert rep['n_fail'] == 1 and rep['n_events'] == 4, rep

        rep = A.audit_experiment_ledger([good_dir])
        assert rep['status'] == 'ok', rep
        assert rep['n_ok'] == 1 and rep['n_legacy_events'] == 0, rep

        with tempfile.TemporaryDirectory() as empty_dir:
            rep = A.audit_experiment_ledger([empty_dir])
            assert rep['status'] == 'not_verifiable_locally', rep

        assert 'experiment_ledger' in (A.__doc__ or '')
        assert 'experiment_ledger' in inspect.getsource(A._run_cli)
        assert "'experiment_ledger': audit_experiment_ledger()" in \
            inspect.getsource(A.run_report)
        # Existing order audits keep reading the same events; the reserved
        # key must not hide the fields they consume.
        got = A._read_experiment_events(good_dir)
        assert len(got) == 2 and all(e['kind'] == 'cell' for e in got), got


def test_experiment_ledger_default_scan_skips_the_reports_own_trace(monkeypatch):
    # `analysis_audit/experiments.jsonl` is appended to by every report build;
    # including it in the report body would make two rebuilds of the same
    # evidence differ.  The default scan covers evidence ledgers only; an
    # explicit path still audits the report's own trace.
    with tempfile.TemporaryDirectory() as td:
        audit_dir = os.path.join(td, 'analysis_audit')
        result_dir = os.path.join(td, 'results_demo')
        os.makedirs(audit_dir)
        os.makedirs(result_dir)
        _append(audit_dir, {'kind': 'analysis', 'label': 'audit_report'})
        _append(result_dir, {'kind': 'cell', 'i': 0})
        monkeypatch.setattr(A, 'REPO', td)
        rep = A.audit_experiment_ledger()
        assert rep['n_ledgers'] == 1, rep
        assert list(rep['ledgers']) == ['results_demo/experiments.jsonl'], rep
        assert rep['status'] == 'ok', rep
        own = A.audit_experiment_ledger(
            [os.path.join(audit_dir, 'experiments.jsonl')])
        assert own['n_ledgers'] == 1 and own['n_ok'] == 1, own


def test_experiment_ledger_refuses_to_grow_corrupt_evidence():
    with tempfile.TemporaryDirectory() as d:
        for i in range(2):
            _append(d, {'kind': 'cell', 'i': i})
        path = os.path.join(d, 'experiments.jsonl')
        original = open(path, encoding='utf-8').read()

        # A malformed sidecar is not "no anchor": appending must not replace
        # it with a fresh anchor and erase the warning.
        with open(path + '.anchor.json', 'w', encoding='utf-8') as fh:
            fh.write('[]')
        assert _append(d, {'kind': 'after'}) is None
        assert open(path, encoding='utf-8').read() == original
        assert L.verify_experiment_ledger(path)['status'] == 'fail'

        # A corrupt/malformed line must not be re-anchored by the next append.
        with open(path + '.anchor.json', 'w', encoding='utf-8') as fh:
            json.dump({'count': 2}, fh)
        lines = original.split('\n')
        lines[1] = '{bad'
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(lines))
        before = open(path, encoding='utf-8').read()
        assert _append(d, {'kind': 'after'}) is None
        assert open(path, encoding='utf-8').read() == before
        assert L.verify_experiment_ledger(path)['status'] == 'fail'


def test_experiment_ledger_handles_unicode_line_separators():
    # JSON strings may contain U+2028/U+2029 unescaped.  They are not physical
    # ledger line breaks; treating them as such made a valid event unreadable.
    with tempfile.TemporaryDirectory() as d:
        _append(d, {'kind': 'cell', 'text': 'a\u2028b\u2029c\x85d'})
        _append(d, {'kind': 'cell', 'text': 'next'})
        path = os.path.join(d, 'experiments.jsonl')
        raw = open(path, encoding='utf-8').read()
        assert raw.count('\n') == 2, raw
        rep = L.verify_experiment_ledger(path)
        assert rep['status'] == 'ok' and rep['n_events'] == 2, rep
