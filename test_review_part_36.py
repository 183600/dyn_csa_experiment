"""CPU checks for endpoint/checkpoint-selection provenance.

A quoted PPL is only comparable across arms when both arms quote the same
endpoint rule.  A final step, a best-validation step and an early-stopped step
are different estimands; selecting the best validation point per arm and then
comparing the selected values is a max-selection, and evaluating a selected
point on the same rows that chose it is optimistically biased.  These checks
cover the rule inference, the paired comparison gate, the manifest declaration,
the data-dependent-split rules, the audit gate/CLI wiring and the descriptive
integration into ``methodology_audit``.  No result panel, test split or trained
checkpoint is read.
"""
import contextlib
import io
import inspect
import json
import os
import sys as _sys
import tempfile

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


def _rec(arm, seed, ppl=None, **kw):
    rec = {'arm': arm, 'seed': seed}
    if ppl is not None:
        rec['ppl'] = ppl
    rec.update(kw)
    return rec


def _panel(n=3, **kw):
    out = []
    for s in range(n):
        out.append(_rec('a', s, 30.0 + s, **kw))
        out.append(_rec('b', s, 34.0 + s, **kw))
    return out


def _raises(fn, why):
    try:
        fn()
    except (TypeError, ValueError):
        return
    raise AssertionError(why)


def test_final_step_panel_is_clean():
    r = L.endpoint_selection_audit(
        _panel(steps=1000, steps_completed=1000), comparisons=[('a', 'b')])
    assert r['status'] == 'ok', r
    assert r['counts']['n_records'] == 6
    assert r['counts']['rule_counts'] == {'final_step': 6}
    assert r['counts']['n_data_dependent_records'] == 0
    c = r['comparisons']['a - b']
    assert c['status'] == 'ok', c
    assert c['n_pairs'] == 3 and c['seeds'] == [0, 1, 2]
    assert c['rule'] == 'final_step'
    assert c['warnings'] == [] and c['violations'] == []
    # A missing checkpoint digest is reported as a count, but it is not a
    # warning for a final-step panel: these records simply never saved one.
    assert r['counts']['n_checkpoint_identity_missing'] == 6
    assert 'no p-value' in r['role']
    assert not any(k.startswith('p_') for k in r['counts'])


def test_final_versus_best_is_not_a_controlled_comparison():
    recs = _panel(steps=1000, steps_completed=1000)
    for rec in recs:
        if rec['arm'] == 'b':
            rec.update({'reported_step': 800, 'best_step': 800,
                        'curve_health': {'best_step': 800}})
    r = L.endpoint_selection_audit(recs, comparisons=[('a', 'b')])
    assert r['status'] == 'fail', r
    c = r['comparisons']['a - b']
    assert c['rule_a'] == 'final_step' and c['rule_b'] == 'best_validation_step'
    assert c['rule'] is None
    assert any('different endpoint rules' in v for v in c['violations']), c


def test_mixed_rule_within_an_arm_fails_closed():
    recs = _panel(steps=1000, steps_completed=1000)
    # One b seed is quoted at its best step; the other b seeds are final.
    for rec in recs:
        if rec['arm'] == 'b' and rec['seed'] == 0:
            rec.update({'reported_step': 800, 'best_step': 800,
                        'curve_health': {'best_step': 800}})
    r = L.endpoint_selection_audit(recs, comparisons=[('a', 'b')])
    assert r['status'] == 'fail', r
    assert any('changes across seeds within an arm' in v
               for v in r['comparisons']['a - b']['violations'])


def test_early_stop_and_truncated_budgets_are_named():
    best = _panel(steps=1000, steps_completed=800,
                  early_stop={'stopped': True, 'best_step': 800},
                  selection_split='validation', eval_split='test')
    r = L.endpoint_selection_audit(best, comparisons=[('a', 'b')])
    assert r['status'] == 'warn', r
    assert r['comparisons']['a - b']['rule'] == 'early_stopped_best'
    assert r['counts']['n_data_dependent_records'] == 6
    stop = _panel(steps=1000, steps_completed=800,
                  early_stop={'stopped': True, 'best_step': 700},
                  selection_split='validation', eval_split='test')
    r = L.endpoint_selection_audit(stop, comparisons=[('a', 'b')])
    assert r['comparisons']['a - b']['rule'] == 'early_stopped_stop', r
    trunc = _panel(steps=1000, steps_completed=900)
    r = L.endpoint_selection_audit(trunc, comparisons=[('a', 'b')])
    assert r['status'] == 'warn', r
    assert r['comparisons']['a - b']['rule'] == 'truncated_step'
    assert any('truncated budget' in w
               for w in r['comparisons']['a - b']['warnings'])
    final = _panel(steps=1000, steps_completed=1000)
    r = L.endpoint_selection_audit(final, comparisons=[('a', 'b')])
    assert r['comparisons']['a - b']['rule'] == 'final_step'
    # Legacy/rope records carry only `steps`: it is the run length, not a
    # separate planned budget.  Keep that inference visible.
    legacy = _panel(steps=1000)
    r = L.endpoint_selection_audit(legacy, comparisons=[('a', 'b')])
    assert r['comparisons']['a - b']['rule'] == 'final_step'
    assert r['records']['a::s0']['steps_completed_inferred'] is True
    capped = _panel(steps=1000)
    for rec in capped:
        rec['cap_truncated'] = True
    r = L.endpoint_selection_audit(capped, comparisons=[('a', 'b')])
    assert r['comparisons']['a - b']['rule'] == 'truncated_step'


def test_data_dependent_selection_must_be_on_an_independent_split():
    good = _panel(steps=1000, steps_completed=800,
                  early_stop={'stopped': True, 'best_step': 800},
                  selection_split='validation', eval_split='test')
    r = L.endpoint_selection_audit(good, comparisons=[('a', 'b')])
    assert r['status'] == 'warn', r
    assert any('data-dependent selection' in w
               for w in r['comparisons']['a - b']['warnings'])
    same = _panel(steps=1000, steps_completed=800,
                  early_stop={'stopped': True, 'best_step': 800},
                  selection_split='validation', eval_split='validation')
    r = L.endpoint_selection_audit(same, comparisons=[('a', 'b')])
    assert r['status'] == 'fail', r
    assert any('same rows' in v or 'optimistically biased' in v
               for v in r['violations']), r['violations']
    on_test = _panel(steps=1000, steps_completed=800,
                     early_stop={'stopped': True, 'best_step': 800},
                     selection_split='test', eval_split='test')
    r = L.endpoint_selection_audit(on_test, comparisons=[('a', 'b')])
    assert r['status'] == 'fail', r
    assert any('selected on the test split' in v for v in r['violations'])
    missing = _panel(steps=1000, steps_completed=800,
                     early_stop={'stopped': True, 'best_step': 800})
    r = L.endpoint_selection_audit(missing, comparisons=[('a', 'b')])
    assert r['status'] == 'warn', r
    assert any('selection split' in w
               for w in r['comparisons']['a - b']['warnings'])


def test_manifest_endpoint_policy_and_preregistration():
    recs = _panel(steps=1000, steps_completed=1000)
    r = L.endpoint_selection_audit(recs, comparisons=[('a', 'b')],
                                   require_preregistered=True)
    assert r['status'] == 'fail', r
    assert any('no manifest endpoint policy' in v for v in r['violations'])
    r = L.endpoint_selection_audit(
        recs, comparisons=[('a', 'b')], require_preregistered=True,
        manifest={'endpoint_policy': {'kind': 'final_step'}})
    assert r['status'] == 'ok', r
    assert r['manifest_declarations'][0]['kind'] == 'final_step'
    r = L.endpoint_selection_audit(
        recs, comparisons=[('a', 'b')],
        manifest={'endpoint_policy': {'kind': 'best_validation_step',
                                       'selection_split': 'validation'}})
    assert r['status'] == 'fail', r
    assert any('manifest declares endpoint kind' in v for v in r['violations'])
    # A correctly declared data-dependent rule on independent rows is a
    # warning, not a failure: the reader still must not call it a final step.
    dep = _panel(steps=1000, steps_completed=800,
                 early_stop={'stopped': True, 'best_step': 800},
                 selection_split='validation', eval_split='test')
    r = L.endpoint_selection_audit(
        dep, comparisons=[('a', 'b')], require_preregistered=True,
        manifest={'endpoint_policy': {'comparison': 'a - b',
                                       'kind': 'early_stopped_best',
                                       'selection_split': 'validation'}})
    assert r['status'] == 'warn', r
    assert r['violations'] == []
    # Arms-based policy and malformed declarations.
    r = L.endpoint_selection_audit(
        recs, comparisons=[('a', 'b')],
        manifest={'endpoint_policy': {'arms': ['b', 'a'],
                                       'kind': 'final_step'}})
    assert r['status'] == 'ok', r
    r = L.endpoint_selection_audit(
        recs, comparisons=[('a', 'b')],
        manifest={'endpoint_policy': {'kind': 'not-a-rule'}})
    assert r['status'] == 'fail', r
    assert any('endpoint_policy' in v for v in r['violations'])


def test_checkpoint_step_and_identity_are_verified():
    recs = _panel(steps=1000, steps_completed=1000)
    for rec in recs:
        rec['checkpoint'] = {'mode': 'final', 'step': 1000, 'sha256': 'abc'}
    r = L.endpoint_selection_audit(recs, comparisons=[('a', 'b')])
    assert r['status'] == 'ok', r
    assert r['counts']['n_checkpoint_identity_missing'] == 0
    bad_step = [dict(x) for x in recs]
    bad_step[0]['checkpoint'] = {'mode': 'final', 'step': 900, 'sha256': 'abc'}
    r = L.endpoint_selection_audit(bad_step, comparisons=[('a', 'b')])
    assert r['status'] == 'fail', r
    assert any('checkpoint.step=900' in v for v in r['violations'])
    both = [dict(x) for x in recs]
    both[0]['checkpoint'] = {'mode': 'both', 'step': 1000, 'sha256': 'abc'}
    r = L.endpoint_selection_audit(both, comparisons=[('a', 'b')])
    assert r['status'] == 'fail', r
    assert any('mode "both"' in v for v in r['violations'])
    partial = [dict(x) for x in recs]
    partial[0].pop('checkpoint')
    r = L.endpoint_selection_audit(partial, comparisons=[('a', 'b')])
    assert r['status'] == 'warn', r
    assert any('partially attested' in w
               for w in r['comparisons']['a - b']['warnings'])
    r = L.endpoint_selection_audit(partial, comparisons=[('a', 'b')],
                                   require_checkpoint_identity=True)
    assert r['status'] == 'fail', r


def test_endpoint_inputs_fail_closed():
    _raises(lambda: L.endpoint_selection_audit('not-a-list'),
            'a string is not a record collection')
    _raises(lambda: L.endpoint_selection_audit([], min_n=1),
            'min_n below 2')
    _raises(lambda: L.endpoint_selection_audit([], min_n=True),
            'boolean min_n')
    _raises(lambda: L.endpoint_selection_audit([], require_preregistered=1),
            'non-boolean preregistration flag')
    bad_records = [
        [None],
        [_rec('a', 0), _rec('a', 0)],
        [{'arm': '', 'seed': 0}],
        [_rec('a', True)],
        [_rec('a', 0, ppl=0.0)],
        [_rec('a', 0, ppl=float('nan'))],
        [_rec('a', 0, steps=0, steps_completed=0)],
        [_rec('a', 0, steps=10, steps_completed=11)],
        [_rec('a', 0, steps=10, steps_completed=10, reported_step=11)],
        [_rec('a', 0, steps=10, steps_completed=10,
              selection_rule='not-a-rule')],
        [_rec('a', 0, steps=10, steps_completed=10,
              selection_split='not-a-split')],
        [_rec('a', 0, steps=10, steps_completed=10,
              checkpoint={'mode': 'not-a-mode'})],
        [_rec('a', 0, steps=10, steps_completed=10,
              early_stop={'stopped': 'yes'})],
        [_rec('a', 0, steps=10, steps_completed=10, best_step=11)],
    ]
    for recs in bad_records:
        r = L.endpoint_selection_audit(recs)
        assert r['status'] == 'fail', (recs, r)
        assert r['violations'], recs


def test_tags_are_not_pooled_and_summary_keys_are_accepted():
    recs = []
    for tag in ('warm', 'cold'):
        for s in range(3):
            recs.append(_rec('a', s, 30.0 + s, tag=tag,
                             steps=1000, steps_completed=1000))
            recs.append(_rec('b', s, 34.0 + s, tag=tag,
                             steps=1000, steps_completed=1000))
    r = L.endpoint_selection_audit(recs)
    assert r['status'] == 'ok', r
    assert set(r['comparisons']) == {'a - b [warm]', 'a - b [cold]'}
    assert all(c['n_pairs'] == 3 for c in r['comparisons'].values())
    r = L.endpoint_selection_audit(
        recs, comparisons=[('a', 'warm', 'b', 'cold')])
    assert list(r['comparisons']) == ['a[warm] - b[cold]']
    assert r['comparisons']['a[warm] - b[cold]']['n_pairs'] == 3
    summary = {}
    for s in range(3):
        summary[f'a::seed{s}'] = {'variant': 'a', 'seed': s, 'ppl': 30.0 + s,
                                  'steps': 1000, 'steps_completed': 1000}
        summary[f'b::seed{s}'] = {'variant': 'b', 'seed': s, 'ppl': 34.0 + s,
                                  'steps': 1000, 'steps_completed': 1000}
    r = L.endpoint_selection_audit(summary)
    assert r['status'] == 'ok', r
    assert r['counts']['n_records'] == 6
    tuple_recs = {
        (('a' if s % 2 == 0 else 'b'), s): {
            'ppl': 30.0 + s, 'steps': 1000, 'steps_completed': 1000}
        for s in range(3)}
    r = L.endpoint_selection_audit(tuple_recs)
    assert r['status'] == 'not_verifiable', r
    assert r['counts']['n_records'] == 3


def test_unknown_or_single_arm_is_not_verifiable():
    r = L.endpoint_selection_audit([_rec('a', 0, 30.0)])
    assert r['status'] == 'not_verifiable', r
    r = L.endpoint_selection_audit([_rec('a', 0), _rec('b', 0)])
    assert r['status'] == 'not_verifiable', r
    assert r['counts']['n_not_verifiable'] >= 0


def test_audit_gate_reads_summary_and_replays_the_rule():
    import audit_gates as A
    with tempfile.TemporaryDirectory() as td:
        summary = {}
        for s in range(3):
            summary[f'a::seed{s}'] = {'variant': 'a', 'seed': s,
                                      'ppl': 30.0 + s, 'steps': 1000,
                                      'steps_completed': 1000}
            summary[f'b::seed{s}'] = {'variant': 'b', 'seed': s,
                                      'ppl': 34.0 + s, 'steps': 1000,
                                      'steps_completed': 1000}
        path = os.path.join(td, 'summary.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(summary, f)
        with contextlib.redirect_stdout(io.StringIO()):
            clean = A.audit_endpoint_selection(td)
        assert clean['status'] == 'ok', clean
        assert 'endpoint_selection' in inspect.getsource(A.run_report)
        assert "'endpoint_selection': audit_endpoint_selection" in inspect.getsource(A._run_cli)
        assert "'checkpoint_selection': audit_endpoint_selection" in inspect.getsource(A._run_cli)
        # A tampered panel is a failure, not a missing-evidence warning.
        summary['b::seed0'].update({'reported_step': 800, 'best_step': 800,
                                    'curve_health': {'best_step': 800}})
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(summary, f)
        with contextlib.redirect_stdout(io.StringIO()):
            bad = A.audit_endpoint_selection(td)
        assert bad['status'] == 'fail', bad
        with contextlib.redirect_stdout(io.StringIO()):
            missing = A.audit_endpoint_selection(os.path.join(td, 'absent'))
        assert missing['status'] == 'not_verifiable_locally', missing


def test_methodology_audit_attaches_without_changing_decisions():
    base = {}
    endpoint = {}
    for s in range(3):
        for arm, ppl in (('a', 30.0 + s), ('b', 34.0 + s)):
            rec = {'variant': arm, 'seed': s, 'ppl': ppl, 'steps': 1000,
                   'steps_completed': 1000, 'run_cfg': 'cfg',
                   'tokens_seen': 1000}
            base[f'{arm}::seed{s}'] = dict(rec)
            rec2 = dict(rec)
            rec2['selection_split'] = None
            endpoint[f'{arm}::seed{s}'] = rec2
    design = {'comp': [('a', 'b', 'endpoint wiring')]}
    plain = L.methodology_audit(base, design=design, baselines={},
                                write=False, verbose=False, sesoi={})
    with_ep = L.methodology_audit(endpoint, design=design, baselines={},
                                  write=False, verbose=False, sesoi={})
    assert with_ep['endpoint_selection']['status'] == 'ok'
    assert with_ep['endpoint_selection']['counts']['rule_counts'] == {
        'final_step': 6}
    for key in ('mean', 'p_exact_signflip', 'p_holm_global',
                'significant_global', 'verified', 'deltas'):
        assert plain['components']['comp'][0].get(key) == \
            with_ep['components']['comp'][0].get(key), key
    assert plain['verified_components'] == with_ep['verified_components']
    assert plain['family'] == with_ep['family']
    # The block must survive the method's write step, not only live in memory.
    with tempfile.TemporaryDirectory() as td:
        L.methodology_audit(endpoint, design=design, baselines={},
                            outdir=td, write=True, verbose=False, sesoi={})
        with open(os.path.join(td, 'methodology_audit.json'),
                  encoding='utf-8') as f:
            written = json.load(f)
        assert written['endpoint_selection']['status'] == 'ok'
        assert written['endpoint_selection']['counts']['rule_counts'] == {
            'final_step': 6}
    # An early-stopped panel becomes a visible warning, not a verdict change.
    early = {}
    for s in range(3):
        for arm, ppl in (('a', 30.0 + s), ('b', 34.0 + s)):
            early[f'{arm}::seed{s}'] = {
                'variant': arm, 'seed': s, 'ppl': ppl, 'steps': 1000,
                'steps_completed': 800, 'run_cfg': 'cfg',
                'tokens_seen': 800,
                'early_stop': {'stopped': True, 'best_step': 800},
                'selection_split': 'validation', 'eval_split': 'test'}
    early_rep = L.methodology_audit(early, design=design, baselines={},
                                    write=False, verbose=False, sesoi={})
    assert early_rep['endpoint_selection']['status'] == 'warn'
    assert any('endpoint selection' in w for w in early_rep['warnings'])
    assert early_rep['verified_components'] == with_ep['verified_components']
