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


def test_analysis_input_binding_cannot_be_stripped():
    td = tempfile.mkdtemp()
    try:
        inp = _write(os.path.join(td, 'bylen.json'), {'a::s0': {'ppl': 1.0}})
        rec = L.stamped_analysis('unit_stats', {'p': 0.01},
                                 inputs={'bylen': inp})
        assert len(rec['inputs_sha256']) == 64, rec
        assert L.verify_stamped_analysis(rec, base=td)['status'] == 'ok'
        # The raw inputs block carries abs_path/error diagnostics and is
        # excluded from content_sha256; inputs_sha256 is inside the hashed
        # body, so deletion or emptying the binding must fail.
        stripped = json.loads(json.dumps(rec))
        stripped.pop('inputs')
        got = L.verify_stamped_analysis(stripped, base=td)
        assert got['status'] == 'fail', got
        assert any('binding was stripped' in p for p in got['problems']), got
        emptied = json.loads(json.dumps(rec))
        emptied['inputs'] = {}
        got = L.verify_stamped_analysis(emptied, base=td)
        assert got['status'] == 'fail', got
        assert any('inputs_sha256 does not reproduce' in p for p in got['problems']), got
        # An edited recorded digest is detected the same way.
        edited = json.loads(json.dumps(rec))
        edited['inputs']['bylen']['sha256'] = '0' * 64
        got = L.verify_stamped_analysis(edited, base=td)
        assert got['status'] == 'fail', got
        # A moved checkout (different absolute root, same relative path) and
        # volatile error text do not change the canonical input identity.
        stable = json.loads(json.dumps(rec))
        stable['inputs']['bylen']['abs_path'] = '/elsewhere/bylen.json'
        stable['inputs']['bylen']['error'] = 'a different local note'
        assert L.analysis_inputs_sha256(stable['inputs']) == rec['inputs_sha256']
        # An artifact with no inputs still commits to the empty manifest.
        bare = L.stamped_analysis('unit', {'x': 1})
        assert bare['inputs'] == {} and len(bare['inputs_sha256']) == 64, bare
        assert L.verify_stamped_analysis(bare)['status'] == 'ok'
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
        # A bytes path is a valid OS path; decoding it must not make the
        # manifest (or its JSON digest) fail halfway through.
        raw = L.analysis_input_manifest(
            {'a': os.fsencode(os.path.join(td, 'input.json'))}, base=td)
        assert raw['a']['exists'] is True, raw
        assert raw['a']['sha256'] == same['a']['sha256'], raw
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
    # The aggregate report audits the other analysis artifacts, not the report
    # file it is about to replace: auditing itself made the body depend on the
    # previous run's freshness (fail on the first rebuild, ok on the next).
    assert "'analysis_integrity': audit_analysis_integrity(exclude=" in src, src
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


def test_design_and_record_audits_report_huge_counts_without_float_overflow():
    # A syntactically valid design/record may declare a token or parameter
    # count far beyond float64's range.  The audits must compare exact
    # integers and report an unrepresentable ratio as unverifiable, not crash
    # while converting the count to float.
    H = 10 ** 400
    with open(os.path.join(REPO, 'confirmation_frozen.json'),
              encoding='utf-8') as fh:
        manifest = json.load(fh)
    sig = manifest['binding']['training_recipe_signature']
    sig.update(steps=H, bs=1, sl=1, nt=H)
    rep = L.comparison_budget_ledger(manifest)
    tb = rep['token_budget']
    assert tb['consumed_tokens'] == H, tb
    assert tb['shared_by_all_arms'] is True, tb
    # H / H is exactly one epoch without ever forming H as a float.
    assert tb['definitions']['epochs_over_cache'] == 1.0, tb
    for rec in rep['arms'].values():
        assert rec['token_budget'] == H, rec
        # params / H underflows: report unverifiable, not a fabricated zero.
        assert rec['params_per_token'] is None, rec
    for comp in rep['comparisons'].values():
        assert (comp['compute_6nd_rel_diff'] is None
                or math.isfinite(comp['compute_6nd_rel_diff'])), comp
        assert isinstance(comp['compute_6nd_matched'], bool), comp

    # The read-budget proxy accepts a declared sequence length beyond float64
    # too: the block count is an exact integer ceiling and the fractional read
    # is reported through the same finite-ratio contract.
    cfgs = L.make_layer_cfgs(1, 'csa_dynamic')
    rb = L._ledger_read_budget(cfgs, H)
    assert all(0.0 <= layer['read_fraction'] <= 1.0 for layer in rb['layers']), rb
    assert rb['selected_read_fraction_mean'] is None or math.isfinite(
        rb['selected_read_fraction_mean']), rb

    # Record-level parameter/compute audits share the same untrusted-count
    # contract.  Equal huge counts are still an exact match; their relative
    # time/proxy rates are unverifiable rather than a float conversion crash.
    recs = {('a', 'p', 0): {'params': H, 'train_time_s': 10.0,
                            'tokens_seen': H},
            ('b', 'p', 0): {'params': H, 'train_time_s': 20.0,
                            'tokens_seen': H}}
    assert L._param_control(recs, 'a', 'b') == (0.0, True)
    assert L.param_pair_reason(recs[('a', 'p', 0)],
                               recs[('b', 'p', 0)]) is None
    assert L._param_control_detail(recs, 'a', 'b')['total_controlled'] is True
    # A near-tie beyond float64 has an unrepresentable ratio but the 1% gate
    # itself is still exactly decidable.
    near = {('a', 'p', 0): {'params': H}, ('b', 'p', 0): {'params': H + 1}}
    _rel, _ok = L._param_control(near, 'a', 'b')
    assert _rel is None and _ok is True, (_rel, _ok)
    assert L.param_pair_reason(near[('a', 'p', 0)],
                               near[('b', 'p', 0)]) is None
    comp = L._compute_control_detail(recs, 'a', 'b')
    # Equal huge products are an exact 6ND match even though each product is
    # not individually representable as a float.
    assert comp['compute_matched'] is True, comp
    assert comp['flops_proxy_6nd'] == {'a': H * H, 'b': H * H}, comp
    # A quotient whose denominator alone overflows float64 can still be
    # representable; divide exactly and round only the result.
    timed = {('a', 'p', 0): {'params': 1000, 'train_time_s': 1e300,
                             'tokens_seen': H},
             ('b', 'p', 0): {'params': 1000, 'train_time_s': 2e300,
                             'tokens_seen': H}}
    tc = L._compute_control_detail(timed, 'a', 'b')
    assert tc['time_ratio_a_over_b'] == 0.5, tc

    # The report builder's parameter-gap helper is the same exact-integer
    # gate: equal enormous counts match; an unrepresentable near-tie is
    # unverifiable instead of raising from int -> float conversion.
    assert BR.param_gap({'params': H}, {'params': H}) == 0.0
    assert BR.param_gap({'params': H}, {'params': H + 1}) is None
    assert BR.is_param_matched({'params': H}, {'params': H}) is True
    assert BR.is_param_matched({'params': H}, {'params': H + 1}) is True

    # The read-budget calibration formula multiplies a declared top-k by the
    # measured mean block length; that product must not be floated first when
    # the top-k exceeds float64's range.
    cal = {'precondition': {'read_budget_calibration': {
        'status': 'pending', 'fixed_block_size': 1,
        'topk_dynamic': H, 'mean_dynamic_block_tokens': 1.0,
        'matched_fixed_topk': H}}}
    cr = L.dynamic_read_budget_calibration(cal)
    assert cr['expected_topk'] == H, cr


def test_huge_parameter_breakdown_is_compared_exactly():
    H = 10 ** 400
    good = {'non_embedding': H, 'active': H}
    near = {'non_embedding': H, 'active': H + 1}
    bad = {'non_embedding': H, 'active': 2 * H}
    recs = {('a', 'p', 0): {'params': H, 'param_breakdown': good},
            ('b', 'p', 0): {'params': H, 'param_breakdown': near}}
    detail = L._param_control_detail(recs, 'a', 'b')
    assert detail['total_controlled'] is True, detail
    # H+1/H differs by 1/H, too small for a float, but the 1% gate is exact.
    assert detail['active_controlled'] is True, detail
    assert detail['active_rel_diff'] is None, detail
    assert L._param_capacity_controlled(detail) is True, detail
    assert L.param_pair_reason(recs[('a', 'p', 0)],
                               recs[('b', 'p', 0)]) is None

    recs[('b', 'p', 0)]['param_breakdown'] = bad
    detail = L._param_control_detail(recs, 'a', 'b')
    assert detail['active_controlled'] is False, detail
    assert L._param_capacity_controlled(detail) is False, detail
    why = L.param_pair_reason(recs[('a', 'p', 0)], recs[('b', 'p', 0)])
    assert why and 'breakdown' in why, why

    # A present-but-unusable breakdown fails closed instead of falling back to
    # the (matching) total count.
    unusable = {('a', 'p', 0): {'params': H, 'param_breakdown':
                                {'non_embedding': H, 'active': 'bad'}},
                ('b', 'p', 0): {'params': H, 'param_breakdown':
                                {'non_embedding': H, 'active': 2 * H}}}
    ud = L._param_control_detail(unusable, 'a', 'b')
    assert ud['breakdown_unverifiable'] is True, ud
    assert L._param_capacity_controlled(ud) is None, ud

    # The design-time close helper must not overflow on the same counts.
    assert L._design_close(H, H) is True
    assert L._design_close(H, H + 1) is False
