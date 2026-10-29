"""CPU review checks (split group 13): analysis-artifact provenance.

A derived report number is reproducible only if the artifact that states it
can be tied to (a) the exact result body and (b) the bytes it consumed.  These
checks exercise the content digest and input manifest added to
``exp_lib.stamped_analysis`` and the corresponding local audit.
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


def _write(path, data):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, sort_keys=True)
    return path


def test_stamped_analysis_binds_body_and_inputs():
    td = tempfile.mkdtemp()
    try:
        inp = _write(os.path.join(td, 'bylen.json'), {'a::s0': {'ppl': 1.0}})
        rec = L.stamped_analysis(
            'unit_stats', {'comparison': {'p': 0.01, 'deltas': [1.0, 2.0]}},
            outdir=td, name='test_stats.json', inputs={'bylen': inp})
        assert len(rec['content_sha256']) == 64, rec
        assert rec['inputs']['bylen']['sha256'] == hashlib.sha256(
            open(inp, 'rb').read()).hexdigest()
        assert rec['inputs']['bylen']['bytes'] == os.path.getsize(inp)

        ok = L.verify_stamped_analysis(rec, base=td)
        assert ok['status'] == 'ok', ok

        # Provenance metadata is volatile: changing the timestamp/code section
        # must not change the result digest.  This is what makes an analysis
        # comparable across machines and reruns.
        rec2 = json.loads(json.dumps(rec))
        rec2['created_at'] = '2000-01-01T00:00:00'
        rec2['provenance'] = {'code': {'git_commit': 'other'}}
        assert L.analysis_content_sha256(rec2) == rec['content_sha256']
        assert L.verify_stamped_analysis(rec2, base=td)['status'] == 'ok'

        # A changed statistic/verdict must fail the body digest.
        bad = json.loads(json.dumps(rec))
        bad['comparison']['p'] = 0.02
        assert L.verify_stamped_analysis(bad, base=td)['status'] == 'fail'

        # A changed input must fail even when the result body is untouched.
        _write(inp, {'a::s0': {'ppl': 2.0}})
        changed = L.verify_stamped_analysis(rec, base=td)
        assert changed['status'] == 'fail', changed
        assert any('changed after the analysis' in p for p in changed['problems'])

        # A missing input is explicitly unverifiable, not a silent pass.
        os.unlink(inp)
        gone = L.verify_stamped_analysis(rec, base=td)
        assert gone['status'] == 'warnings', gone
        assert any('absent locally' in w for w in gone['warnings'])

        # Legacy artifacts without a digest are not accepted as verified.
        legacy = {'kind': 'old', 'provenance': {}, 'result': 1}
        leg = L.verify_stamped_analysis(legacy, base=td)
        assert leg['status'] == 'not_verifiable_locally', leg
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_stamped_analysis_rejects_malformed_digest_and_inputs():
    rec = L.stamped_analysis('unit', {'x': 1})
    bad = json.loads(json.dumps(rec))
    bad['content_sha256'] = 'not-a-digest'
    assert L.verify_stamped_analysis(bad)['status'] == 'fail'

    bad2 = json.loads(json.dumps(rec))
    bad2['inputs'] = ['not', 'an', 'object']
    assert L.verify_stamped_analysis(bad2)['status'] == 'fail'

    # An input entry without a recorded digest cannot be called verified.
    bad3 = json.loads(json.dumps(rec))
    bad3['inputs'] = {'x': {'path': 'x.json', 'exists': False}}
    assert L.verify_stamped_analysis(bad3)['status'] == 'fail'


def test_analysis_input_manifest_is_relative_and_explicit():
    td = tempfile.mkdtemp()
    try:
        with open(os.path.join(td, 'input.json'), 'w', encoding='utf-8') as fh:
            fh.write('{"x": 1}\n')
        m = L.analysis_input_manifest({'a': 'input.json'}, base=td)
        assert m['a']['path'] == 'input.json', m
        assert m['a']['exists'] is True and len(m['a']['sha256']) == 64, m
        missing = L.analysis_input_manifest(['nope.json'], base=td)
        assert missing['input_0']['exists'] is False, missing
        assert missing['input_0']['sha256'] is None, missing
        # Duplicate spellings of one path do not corrupt the recorded bytes.
        same = L.analysis_input_manifest(
            {'a': os.path.join(td, 'input.json'), 'b': 'input.json'}, base=td)
        assert same['a']['sha256'] == same['b']['sha256'], same
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_audit_analysis_integrity_scans_only_stamped_artifacts():
    td = tempfile.mkdtemp()
    try:
        inp = _write(os.path.join(td, 'input.json'), {'x': 1})
        rec = L.stamped_analysis('unit', {'answer': 42}, outdir=td,
                                 name='derived.json',
                                 inputs={'input': inp})
        # A plain data/input JSON is not in scope and must not be counted as a
        # failed or verified analysis.
        _write(os.path.join(td, 'plain.json'), {'raw': True})
        rep = A.audit_analysis_integrity([td])
        assert rep['status'] == 'ok', rep
        assert rep['n_checked'] == 1 and rep['n_skipped'] >= 1, rep
        assert rep['n_ok'] == 1, rep

        # Editing the artifact body after the fact must fail the audit.
        tampered = json.loads(json.dumps(rec))
        tampered['answer'] = 43
        L.atomic_write_json(os.path.join(td, 'derived.json'), tampered)
        rep2 = A.audit_analysis_integrity([td])
        assert rep2['status'] == 'fail', rep2
        assert rep2['n_fail'] == 1, rep2
        assert any('does not reproduce' in p for p in rep2['problems']), rep2

        # No stamped artifact at all is not a pass.
        empty = tempfile.mkdtemp()
        try:
            rep3 = A.audit_analysis_integrity([empty])
            assert rep3['status'] == 'not_verifiable_locally', rep3
        finally:
            shutil.rmtree(empty, ignore_errors=True)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_analysis_integrity_is_wired_into_report_and_cli():
    src = inspect.getsource(A.run_report)
    assert "'analysis_integrity': audit_analysis_integrity()" in src, src
    assert 'analysis_integrity' in (A.__doc__ or ''), (
        'analysis-integrity audit is not documented in the CLI help')
    assert 'analysis_integrity' in inspect.getsource(A._run_cli), (
        'analysis-integrity audit is not registered as a CLI command')
    # The confirmatory statistics entry point must bind its raw inputs, so a
    # regenerated test_stats.json can be tied to the bylen/manifest it saw.
    import explib.confirmation as V12
    fsrc = inspect.getsource(V12.finaltest_stats)
    assert "inputs=_analysis_inputs" in fsrc, fsrc
    assert "'bylen': bylen_path" in fsrc, fsrc
    psrc = inspect.getsource(V12.run_power)
    assert "inputs={'manifest'" in psrc, psrc


def test_stamped_analysis_event_records_content_digest():
    td = tempfile.mkdtemp()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rec = L.stamped_analysis('unit', {'x': 1}, outdir=td,
                                    name='a.json')
        events = [json.loads(line) for line in
                  open(os.path.join(td, 'experiments.jsonl'), encoding='utf-8')
                  .read().strip().splitlines()]
        ev = next(e for e in events if e.get('kind') == 'analysis')
        assert ev['content_sha256'] == rec['content_sha256'], (ev, rec)
    finally:
        shutil.rmtree(td, ignore_errors=True)
