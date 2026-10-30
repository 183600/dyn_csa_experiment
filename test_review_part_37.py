"""CPU checks for reproducible analysis reports and source-only hygiene.

The report layer had two reproducibility gaps: the wall-clock ``generated_at``
field was inside the content digest (so two identical audit runs looked like
different results), and the aggregate ``audit_report`` bound no evidence inputs
(so an edited ``results_*``/``analysis_*`` file did not invalidate it).  These
checks cover the SOURCE_DATE_EPOCH stamp, schema-aware hashing, evidence-input
binding, the report-to-report replay comparator/CLI, and the engineering rules
that keep the source-only tree free of vendored lint scratch and make the sdist
carry every regression test.  No result panel, test split or checkpoint is
read.
"""
import contextlib
import fnmatch
import io
import json
import os
import subprocess
import sys as _sys
import tempfile

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


def _write(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, sort_keys=True)
    return path


def _stamp(payload, inputs=None):
    return L.stamped_analysis('unit_report', payload, inputs=inputs)


def _run_cli(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = L._reproduction_cli(argv)
    return code, buf.getvalue()


def test_source_date_epoch_parser_matches_archive_contract():
    import archive_release as AR
    accepted = (None, '0', '1700000000', '253402300799')
    for value in accepted:
        if value is None:
            continue
        assert L.source_date_epoch(value) == AR._source_date_epoch(value)
    assert L.source_date_epoch('') is None
    assert L.source_date_epoch('  ') is None
    for bad in (True, False, -1, 1.5, '1.0', '2020-01-01', '1e9', ' 12x',
                '253402300800', '-0'):
        for fn in (L.source_date_epoch, AR._source_date_epoch):
            try:
                fn(bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f'{fn.__module__}.{fn.__name__} accepted {bad!r}')


def test_reproducible_timestamp_and_stamp_are_fixed_by_epoch():
    import archive_release as AR
    old = os.environ.get('SOURCE_DATE_EPOCH')
    try:
        os.environ['SOURCE_DATE_EPOCH'] = '1700000000'
        assert L.reproducible_timestamp() == '2023-11-14T22:13:20'
        assert L.reproducible_timestamp() == L.reproducible_timestamp()
        assert L.reproducible_timestamp(fmt='%Y-%m-%dT%H:%M:%SZ') == AR._iso_utc(1700000000)
        first = _stamp({'result': 1})
        second = _stamp({'result': 1})
        assert first['created_at'] == second['created_at'] == '2023-11-14T22:13:20'
        # No volatile field changes, so the whole JSON object is stable, not
        # merely its content digest.
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
        assert first['content_schema'] == 2
    finally:
        if old is None:
            os.environ.pop('SOURCE_DATE_EPOCH', None)
        else:
            os.environ['SOURCE_DATE_EPOCH'] = old


def test_generated_at_is_volatile_only_for_schema_v2():
    legacy = {'kind': 'old', 'answer': 1, 'generated_at': '2020-01-01T00:00:00'}
    h1 = L.analysis_content_sha256(legacy)
    legacy['generated_at'] = '2021-01-01T00:00:00'
    assert L.analysis_content_sha256(legacy) != h1, (
        'legacy artifacts must keep their historical digest semantics')

    current = _stamp({'answer': 1})
    current['generated_at'] = '2020-01-01T00:00:00'
    h2 = L.analysis_content_sha256(current)
    current['generated_at'] = '2021-01-01T00:00:00'
    assert L.analysis_content_sha256(current) == h2
    assert L.verify_stamped_analysis(current)['status'] == 'ok', (
        'changing a volatile timestamp must not invalidate the body digest')


def test_audit_evidence_inputs_is_deterministic_and_bounded():
    td = tempfile.mkdtemp()
    try:
        _write(os.path.join(td, 'results_a', 'summary.json'), {'a': 1})
        _write(os.path.join(td, 'results_a', 'nested', 'curve.csv'), {})
        with open(os.path.join(td, 'results_a', 'nested', 'curve.csv'), 'w',
                  encoding='utf-8') as fh:
            fh.write('step,ppl\n1,2\n')
        _write(os.path.join(td, 'analysis_b', 'stats.jsonl'), {})
        _write(os.path.join(td, 'analysis_audit', 'experiments.jsonl'), {})
        _write(os.path.join(td, 'confirmation_frozen.json'), {})
        _write(os.path.join(td, 'confirmation_frozen_dynamic.json'), {})
        with open(os.path.join(td, 'RELEASES.md'), 'w', encoding='utf-8') as fh:
            fh.write('# evidence\n')
        with open(os.path.join(td, 'results_a', 'weights.bin'), 'wb') as fh:
            fh.write(b'\x00' * 32)
        with open(os.path.join(td, 'results_a', 'huge.json'), 'w',
                  encoding='utf-8') as fh:
            fh.write('{"x": "' + 'a' * 2048 + '"}')
        # A FIFO (or other non-regular object) with a text suffix is not a
        # file the report can hash; it must be skipped, not opened as a
        # stream (which would block forever).
        fifo = os.path.join(td, 'results_a', 'stream.json')
        have_fifo = hasattr(os, 'mkfifo')
        if have_fifo:
            try:
                os.mkfifo(fifo)
            except OSError:
                have_fifo = False

        inputs1, scope1 = A.audit_evidence_inputs(td, max_file_bytes=1024)
        inputs2, scope2 = A.audit_evidence_inputs(td, max_file_bytes=1024)
        assert inputs1 == inputs2 and scope1 == scope2
        labels = list(inputs1)
        assert labels == sorted(labels)
        assert all(k.startswith('evidence:') for k in labels)
        joined = '\n'.join(labels)
        assert 'results_a/summary.json' in joined
        assert 'results_a/nested/curve.csv' in joined
        assert 'analysis_b/stats.jsonl' in joined
        assert 'analysis_audit' not in joined, (
            'the report must not bind itself to the ledger it appends to')
        assert 'weights.bin' not in joined, 'binary checkpoints are not report inputs'
        assert 'huge.json' not in joined
        assert 'results_a/huge.json' in scope1['skipped_large']
        if have_fifo:
            assert 'stream.json' not in joined
            assert any(e.get('path') == 'results_a/stream.json'
                       for e in scope1['skipped_unreadable']), scope1
        assert scope1['n_files'] == len(inputs1)
        assert scope1['n_bytes'] > 0
        assert scope1['max_file_bytes'] == 1024
        assert scope1['excluded_dirs'] == ['analysis_audit']
        # The report being replaced is not evidence about itself.  If the
        # caller writes it under analysis_*/, binding the previous copy would
        # make every rebuild depend on the previous run.
        report = _write(os.path.join(td, 'analysis_b', 'report.json'), {'v': 1})
        excluded_inputs, _ = A.audit_evidence_inputs(
            td, max_file_bytes=1024, exclude=report)
        assert 'evidence:analysis_b/report.json' not in excluded_inputs, excluded_inputs
        assert 'evidence:analysis_b/stats.jsonl' in excluded_inputs, excluded_inputs
        string_excluded, _ = A.audit_evidence_inputs(
            td, max_file_bytes=1024, exclude=[report])
        assert string_excluded == excluded_inputs
    finally:
        import shutil
        shutil.rmtree(td, ignore_errors=True)


def test_report_inputs_bind_the_evidence_bytes():
    td = tempfile.mkdtemp()
    try:
        evidence = _write(os.path.join(td, 'results_a', 'summary.json'), {'ppl': 10})
        inputs, scope = A.audit_evidence_inputs(td)
        rec = L.stamped_analysis('audit_report', {'verdict': 'ok'},
                                 inputs=inputs)
        assert rec['inputs_sha256'] and rec['inputs']
        assert L.verify_stamped_analysis(rec, base=td)['status'] == 'ok'
        _write(evidence, {'ppl': 11})
        got = L.verify_stamped_analysis(rec, base=td)
        assert got['status'] == 'fail', got
        assert any('changed after the analysis' in p for p in got['problems'])
        assert scope['n_files'] == 1
    finally:
        import shutil
        shutil.rmtree(td, ignore_errors=True)


def test_analysis_report_diff_separates_body_inputs_and_provenance():
    td = tempfile.mkdtemp()
    try:
        inp = _write(os.path.join(td, 'input.json'), {'x': 1})
        base = _stamp({'comparison': {'p': 0.01}, 'note': 'same'}, inputs={'x': inp})
        same = _stamp({'comparison': {'p': 0.01}, 'note': 'same'}, inputs={'x': inp})
        rep = L.analysis_report_diff(base, same)
        assert rep['verdict'] == 'reproduced', rep
        assert rep['dimensions']['body']['status'] == 'match'

        changed = _stamp({'comparison': {'p': 0.02}, 'note': 'same'}, inputs={'x': inp})
        rep = L.analysis_report_diff(base, changed)
        assert rep['verdict'] == 'body_changed', rep
        pointers = [d['pointer'] for d in rep['differences']]
        assert '/comparison/p' in pointers, pointers

        _write(inp, {'x': 2})
        new_input = _stamp({'comparison': {'p': 0.01}, 'note': 'same'},
                           inputs={'x': inp})
        rep = L.analysis_report_diff(base, new_input)
        assert rep['verdict'] == 'inputs_changed', rep
        assert rep['dimensions']['inputs']['status'] == 'mismatch'

        # A hand-edited body is an integrity failure, not a result change.
        tampered = json.loads(json.dumps(base))
        tampered['comparison']['p'] = 0.5
        rep = L.analysis_report_diff(base, tampered)
        assert rep['verdict'] == 'invalid', rep

        # Provenance is compared independently of the body.  Use the same
        # (already changed) input on both sides so only code differs.
        other = _stamp({'comparison': {'p': 0.01}, 'note': 'same'}, inputs={'x': inp})
        other['provenance']['code']['source_tree']['sha256'] = '0' * 64
        rep = L.analysis_report_diff(new_input, other)
        assert rep['verdict'] == 'provenance_changed', rep
        assert rep['dimensions']['code']['status'] == 'mismatch'

        # An append-only audit trace can be ignored without hiding a result change.
        left = _stamp({'comparison': {'p': 0.01},
                       'experiment_ledger': {'n_events': 5}}, inputs={'x': inp})
        right = _stamp({'comparison': {'p': 0.01},
                        'experiment_ledger': {'n_events': 6}}, inputs={'x': inp})
        assert L.analysis_report_diff(left, right)['verdict'] == 'body_changed'
        rep = L.analysis_report_diff(left, right, ignore='/experiment_ledger')
        assert rep['verdict'] == 'reproduced', rep
    finally:
        import shutil
        shutil.rmtree(td, ignore_errors=True)


def test_analysis_report_diff_cli_and_exit_codes():
    td = tempfile.mkdtemp()
    try:
        a = _stamp({'value': 1})
        b = _stamp({'value': 1})
        c = _stamp({'value': 2})
        pa = _write(os.path.join(td, 'a.json'), a)
        pb = _write(os.path.join(td, 'b.json'), b)
        pc = _write(os.path.join(td, 'c.json'), c)
        out = os.path.join(td, 'diff.json')
        code, text = _run_cli(['report', pa, pb, '--json', out])
        assert code == 0, text
        assert json.loads(text)['verdict'] == 'reproduced'
        assert json.load(open(out, encoding='utf-8'))['verdict'] == 'reproduced'
        code, text = _run_cli(['report', pa, pc])
        assert code == 1, text
        assert json.loads(text)['verdict'] == 'body_changed'
        code, text = _run_cli(['report', pa, pb, '--max-differences', '0'])
        assert code == 2, text
        code, text = _run_cli(['report', os.path.join(td, 'missing.json'), pb])
        assert code == 2, text
        assert json.loads(text)['verdict'] == 'error'
    finally:
        import shutil
        shutil.rmtree(td, ignore_errors=True)


def test_environment_snapshot_records_redacted_invocation():
    argv = ['train.py', '--seed', '1', '--token', 'abc',
            '--api-key=xyz', '--password', 'hunter2']
    ident = L._invocation_identity(argv)
    assert ident['argv'] == ['train.py', '--seed', '1', '--token', '***',
                             '--api-key=***', '--password', '***'], ident
    assert ident['entrypoint'] == 'train.py'
    assert ident['n_args'] == len(argv)
    assert ident['redacted'] is True
    assert L._invocation_identity(['python', '-m', 'explib.rope', 'plan'])[
        'python_module'] == 'explib.rope'
    env = L.environment_snapshot()
    assert isinstance(env.get('invocation'), dict)
    assert isinstance(env['invocation'].get('argv'), list)
    # The invocation must not change the execution-class fingerprint used to
    # pair same-environment runs.
    other = dict(env)
    other['invocation'] = {'argv': ['different.py'], 'entrypoint': 'different.py',
                           'python_module': None, 'n_args': 1, 'redacted': False}
    assert L.execution_fingerprint(env) == L.execution_fingerprint(other)


def test_source_tree_hygiene_and_sdist_coverage():
    tracked = subprocess.run(
        ['git', 'ls-files'], cwd=REPO, capture_output=True, text=True)
    if tracked.returncode != 0:
        import pytest as _pytest
        _pytest.skip('not a git checkout; tracked-file hygiene is not checkable')
    files = [p for p in tracked.stdout.splitlines() if p]
    for forbidden in ('.tmp_lint/', 'archive_test_', '__pycache__/',
                      '.pytest_cache/', 'results_', 'analysis_',
                      'audit_report.json'):
        hits = [p for p in files if p.startswith(forbidden)]
        assert not hits, f'tracked scratch artifact(s) {hits[:5]}'
    assert any(p.startswith('.github/workflows/') for p in files), (
        'CI configuration must remain in the repository')
    assert os.path.isfile(os.path.join(REPO, '.github', 'workflows', 'test.yml'))
    readmes = [p for p in files if fnmatch.fnmatch(os.path.basename(p), 'README*.md')]
    assert 'README.md' in readmes and 'README_zh.md' in readmes, readmes
    big = []
    for rel in files:
        path = os.path.join(REPO, rel)
        if os.path.isfile(path) and os.path.getsize(path) > 5 * 1024 * 1024:
            big.append((rel, os.path.getsize(path)))
    assert not big, f'tracked files exceed 5 MiB: {big}'

    with open(os.path.join(REPO, 'MANIFEST.in'), encoding='utf-8') as fh:
        lines = [ln.strip() for ln in fh if ln.strip().startswith('include ')]
    patterns = [ln.split(None, 1)[1] for ln in lines]
    tests = sorted(
        os.path.basename(p) for p in os.listdir(REPO)
        if p.startswith('test_') and p.endswith('.py'))
    missing = [t for t in tests
               if not any(fnmatch.fnmatch(t, pat) for pat in patterns)]
    assert not missing, f'sdist MANIFEST.in omits test module(s): {missing}'
    assert any(fnmatch.fnmatch('Makefile', pat) for pat in patterns)

    with open(os.path.join(REPO, '.gitattributes'), encoding='utf-8') as fh:
        attrs = fh.read()
    assert '* text=auto eol=lf' in attrs, (
        'a CRLF checkout would change source_tree_manifest() bytes')


def test_analysis_integrity_can_exclude_the_report_being_replaced():
    td = tempfile.mkdtemp()
    try:
        inp = _write(os.path.join(td, 'input.json'), {'x': 1})
        rec = L.stamped_analysis('unit', {'answer': 42}, outdir=td,
                                 name='derived.json', inputs={'inp': inp})
        assert A.audit_analysis_integrity([td])['n_checked'] == 1
        rep = A.audit_analysis_integrity(
            [td], exclude=[os.path.join(td, 'derived.json')])
        assert rep['n_checked'] == 0 and rep['status'] == 'not_verifiable_locally', rep
        # A single path and None are the natural spellings too; neither may be
        # iterated character by character or rejected.
        single = A.audit_analysis_integrity(
            [td], exclude=os.path.join(td, 'derived.json'))
        assert single['n_checked'] == 0, single
        assert A.audit_analysis_integrity([td], exclude=None)['n_checked'] == 1
        # An explicit path list keeps the artifact when it is not excluded.
        assert A.audit_analysis_integrity(
            [os.path.join(td, 'derived.json')])['n_ok'] == 1
        assert rec['content_schema'] == 2
    finally:
        import shutil
        shutil.rmtree(td, ignore_errors=True)


def test_json_pointer_supports_the_empty_key_and_rejects_bad_indices():
    # RFC 6901: "" is the whole document, "/" selects the member with the
    # empty key.  Treating "/" as the root made an ignored block under the
    # empty key impossible to remove.
    assert L._pointer_parts(None) == []
    assert L._pointer_parts('') == []
    assert L._pointer_parts('/') == ['']
    assert L._pointer_parts('//') == ['', '']
    doc = {'': {'value': 1}, 'list': [10, 20, 30]}
    assert L._drop_pointer(doc, '/') == {'list': [10, 20, 30]}
    assert L._drop_pointer(doc, '/list/1') == {
        '': {'value': 1}, 'list': [10, 30]}
    # A NaN-like negative/leading-zero/float index names no RFC array slot;
    # silently letting int() accept it deletes a different element.
    for pointer in ('/list/-1', '/list/01', '/list/1.0', '/list/-', '/list/9'):
        assert L._drop_pointer(doc, pointer) == doc, pointer


def test_analysis_report_diff_reports_a_schema_migration_without_a_body_change():
    current = _stamp({'comparison': {'p': 0.01}})
    legacy = json.loads(json.dumps(current))
    legacy.pop('content_schema', None)
    # Recompute the digest under the legacy schema this record now declares.
    legacy['content_sha256'] = L.analysis_content_sha256(legacy)
    assert L.analysis_content_sha256(legacy) != L.analysis_content_sha256(current)
    rep = L.analysis_report_diff(legacy, current)
    assert rep['verdict'] == 'schema_changed', rep
    assert rep['dimensions']['body']['status'] == 'schema_changed', rep
    assert rep['dimensions']['body']['n_differences'] == 0, rep
    assert rep['dimensions']['body']['original_schema'] is None, rep
    assert rep['dimensions']['body']['rerun_schema'] == 2, rep
    assert rep['stored_integrity'] == {
        'original': 'match', 'rerun': 'match'}, rep
    assert any('schema' in w for w in rep['warnings']), rep
    # Rebuilding with the same schema still compares as reproduced.
    assert L.analysis_report_diff(current, _stamp(
        {'comparison': {'p': 0.01}}))['verdict'] == 'reproduced'


def test_analysis_report_diff_truncation_flag_is_exact_at_the_limit():
    # Exactly `max_differences` differences are a complete report; only a
    # difference beyond the limit means the list was truncated.  The old
    # `len >= limit` test mislabeled the exact-boundary case.
    one = L.analysis_report_diff({'a': 1, 'b': 2}, {'a': 9, 'b': 2},
                                 max_differences=1)
    assert one['n_differences'] == 1 and one['truncated'] is False, one
    two = L.analysis_report_diff({'a': 1, 'b': 2}, {'a': 9, 'b': 8},
                                 max_differences=1)
    assert two['n_differences'] == 1 and two['truncated'] is True, two


def test_analysis_integrity_ignores_ledger_anchor_sidecars():
    # The report appends to analysis_audit/experiments.jsonl and rewrites its
    # anchor.  The anchor is not a stamped analysis; counting it as a skipped
    # artifact made the next rebuild's n_skipped (and thus the hashed body)
    # depend on the previous rebuild.
    td = tempfile.mkdtemp()
    try:
        _write(os.path.join(td, 'analysis_audit',
                            'experiments.jsonl.anchor.json'), {'count': 1})
        rep = A.audit_analysis_integrity([td])
        assert rep['n_checked'] == 0 and rep['n_skipped'] == 0, rep
        # An explicit anchor path is likewise not a stamped analysis.
        rep = A.audit_analysis_integrity(
            [os.path.join(td, 'analysis_audit',
                          'experiments.jsonl.anchor.json')])
        assert rep['n_checked'] == 0 and rep['n_skipped'] == 0, rep
    finally:
        import shutil
        shutil.rmtree(td, ignore_errors=True)
