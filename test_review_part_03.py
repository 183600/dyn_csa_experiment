"""CPU review checks (split group 03; t_metric_policy_checks_record_values .. t_simple_baseline_nested_rows_and_fingerprint)."""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)

def test_metric_policy_checks_record_values():
    # The declared token-weighted PPL must match the record's own NLL
    # accounting; a per-sequence arithmetic mean is a different estimand.
    _tok = 100
    good = {'v::seed0': {'variant': 'v', 'ppl': 10.0, 'nll_sum': math.log(10.0) * _tok,
                         'n_tokens': _tok}}
    assert L.metric_policy_audit(good)['status'] == 'ok'
    # A PPL with no NLL/token accounting is unverifiable, not silently accepted.
    _legacy = L.metric_policy_audit({'v::seed0': {'variant': 'v', 'ppl': 10.0}})
    assert _legacy['status'] == 'warn' and any(
        'no NLL/token accounting' in w for w in _legacy['warnings']), _legacy
    bad = {'v::seed0': {'variant': 'v', 'ppl': 10.0, 'nll_sum': math.log(20.0) * _tok,
                        'n_tokens': _tok}}
    ma = L.metric_policy_audit(bad)
    assert ma['status'] == 'warn' and any('not exp(nll_sum/n_tokens)' in w for w in ma['warnings']), ma
    # Per-row accounting is accepted as the same token-weighted estimate.
    row = {'v::seed0': {'variant': 'v', 'ppl': 10.0,
                        'nll_per_row': [math.log(10.0) * 60, math.log(10.0) * 40],
                        'n_tokens_per_row': [60, 40]}}
    assert L.metric_policy_audit(row)['status'] == 'ok', L.metric_policy_audit(row)
    # A nested record reports its holdout PPL and must be checked against
    # the holdout NLL accounting, not the full-validation accounting.
    # `ppl` stays the full-validation value here (10.0) while the primary
    # estimate is the holdout PPL (5.0); the check must not compare the
    # holdout NLL against the full-validation PPL.
    nested = {'v::seed0': {'variant': 'v', 'ppl': 10.0,
                           'nll_sum': math.log(10.0) * 100, 'n_tokens': 100,
                           'ppl_holdout': 5.0,
                           'nll_sum_holdout': math.log(5.0) * 40,
                           'n_tokens_holdout': 40,
                           'ppl_estimation_rows': 'nested_validation_holdout'}}
    assert L.metric_policy_audit(nested)['status'] == 'ok', L.metric_policy_audit(nested)
    _pooled, _nl, _nt = L._pooled_ppl_from_records(list(nested.values()))
    assert abs(_pooled - 5.0) < 1e-12 and _nt == 40, (_pooled, _nl, _nt)
    # Intervals must use the same rows as the primary (holdout) PPL.  The
    # full-validation arrays here are deliberately incompatible with the
    # holdout estimate so a wrong-array fallback cannot look plausible.
    _nr = {'variant': 'v', 'seed': 0, 'ppl': math.exp(10.0),
           'nll_sum': 10.0 * 40, 'n_tokens': 40,
           'ppl_estimation_rows': 'nested_validation_holdout',
           'ppl_holdout': math.exp(2.0),
           'nll_sum_holdout': 2.0 * 10, 'n_tokens_holdout': 10,
           'nll_per_row': [100.0] * 4, 'n_tokens_per_row': [10] * 4,
           'nll_per_row_holdout': [10.0] * 2, 'n_tokens_per_row_holdout': [5] * 2}
    _nr2 = dict(_nr, seed=1)
    _pd, _pn, _pt = L._pooled_ppl_from_records([_nr, _nr2])
    assert abs(_pd - math.exp(2.0)) < 1e-12 and _pt == 20, (_pd, _pn, _pt)
    _dci = L._doc_bootstrap_ppl_ci([_nr, _nr2])
    assert _dci is not None and all(abs(x - math.exp(2.0)) < 1e-9 for x in _dci), _dci
    _cci = L._combined_seed_doc_ci([_nr, _nr2])
    assert _cci['status'] == 'ok' and _cci['n_docs'] == 2, _cci
    assert all(abs(x - math.exp(2.0)) < 1e-9 for x in _cci['ci95_ppl']), _cci
    # Missing holdout accounting must not silently pool the full-validation
    # totals.  A complete holdout per-row array is still a valid source, so
    # remove both forms to exercise the fail-closed path.
    _miss = dict(_nr)
    _miss.pop('nll_sum_holdout')
    _miss.pop('n_tokens_holdout')
    _miss.pop('nll_per_row_holdout')
    _miss.pop('n_tokens_per_row_holdout')
    assert L._pooled_ppl_from_records([_miss]) == (None, None, None)
    # Per-row holdout arrays alone are enough to recover the same totals.
    _rows_only = dict(_nr)
    _rows_only.pop('nll_sum_holdout')
    _rows_only.pop('n_tokens_holdout')
    assert abs(L._pooled_ppl_from_records([_rows_only])[0] - math.exp(2.0)) < 1e-12
    # The metric-policy audit must use those same holdout rows; falling back
    # to the full-validation arrays would compare the holdout PPL against a
    # different sample and flag a correct nested record.
    assert L.metric_policy_audit({'v::seed0': _rows_only})['status'] == 'ok', \
        L.metric_policy_audit({'v::seed0': _rows_only})
    # Missing holdout row arrays must not silently bootstrap full-validation rows.
    _miss_rows = dict(_nr)
    _miss_rows.pop('nll_per_row_holdout')
    _miss_rows.pop('n_tokens_per_row_holdout')
    assert L._doc_bootstrap_ppl_ci([_miss_rows]) is None
    assert L._combined_seed_doc_ci([_miss_rows, _miss_rows])['status'] == 'unverifiable'
    _bad_nested = json.loads(json.dumps(nested))
    _bad_nested['v::seed0']['ppl_holdout'] = 7.0
    _mn = L.metric_policy_audit(_bad_nested)
    assert _mn['status'] == 'warn' and any('not exp(nll_sum/n_tokens)' in w for w in _mn['warnings']), _mn
    # Boundary F1 must be the harmonic mean of the reported precision/recall.
    b = {'b::seed0': {'variant': 'b', 'ppl': 10.0,
                      'stats': {'L0': {'bnd_f1': 0.9, 'bnd_prec': 0.2,
                                       'bnd_rec': 0.2, 'bnd_rand': 0.1}}}}
    mb = L.metric_policy_audit(b)
    assert mb['status'] == 'warn' and any('non-harmonic' in w for w in mb['warnings']), mb
    # Existence is not measurement: a boundary record with all keys present
    # but None must fail, not pass through the range/filter checks.
    bn = {'b::seed0': {'variant': 'b', 'ppl': 10.0,
                       'stats': {'L0': {'bnd_f1': None, 'bnd_prec': None,
                                        'bnd_rec': None, 'bnd_rand': None}}}}
    mbn = L.metric_policy_audit(bn)
    assert mbn['status'] == 'warn', mbn
    _bc = next(c for c in mbn['checks'] if c['metric'] == 'boundary_alignment')
    assert _bc['ok'] is False and _bc['missing'], _bc
    # Retrieval accuracy outside [0,1] is not a usable endpoint.
    n = {'n::seed0': {'variant': 'n', 'task': 'niah_natural', 'acc': 1.5,
                      'nll': 1.0, 'n_queries': 10}}
    mn = L.metric_policy_audit(n)
    assert mn['status'] == 'warn' and any('invalid accuracy' in w for w in mn['warnings']), mn
    # An empty retrieval record is incomplete, not a pass.
    me = L.metric_policy_audit({'x::seed0': {'variant': 'x', 'task': 'niah_natural'}})
    assert me['status'] == 'warn', me

    # The audit's record checks must honour a caller-supplied policy, not the
    # module default.  Here the custom policy declares ROC-AUC as primary and
    # a 0.5 extreme-imbalance threshold (so a 10% positive rate is NOT rare).
    _custom = json.loads(json.dumps(L.METRIC_POLICY))
    _custom['binary_classification']['primary'] = 'roc_auc'
    _custom['binary_classification']['extreme_imbalance_threshold'] = 0.5
    _cust_rec = {'r': {'variant': 'r', 'task': 'binary_classification',
                       'prevalence': 0.1, 'primary_metric': 'roc_auc',
                       'roc_auc': 0.9, 'pr_auc': 0.8,
                       'recommended_primary_metric': 'pr_auc',
                       'recommended_primary_available': True}}
    _ca = L.metric_policy_audit(_cust_rec, policy=_custom)
    _cw = ' '.join(_ca['warnings'])
    assert 'declares primary_metric' not in _cw, _ca
    assert 'does not match the policy endpoint' not in _cw, _ca
    assert 'is not defined for task' not in _cw, _ca
    # A malformed caller policy entry is absent, not an exception.
    _broken = {'binary_classification': ['not', 'a', 'mapping']}
    _ba = L.metric_policy_audit(_cust_rec, policy=_broken)
    assert _ba['status'] == 'warn', _ba


def test_retrieval_metrics_and_niah_trivial_floor():
    # A many-way retrieval endpoint must expose its trivial floors: micro
    # accuracy alone can be high merely because one answer class dominates.
    rm = L.retrieval_metrics([0, 0, 0, 1], [0, 0, 0, 0])
    assert rm['n_correct'] == 3 and rm['n'] == 4, rm
    assert abs(rm['acc'] - 0.75) < 1e-12 and abs(rm['balanced_accuracy'] - 0.5) < 1e-12, rm
    assert abs(rm['chance_accuracy'] - 0.5) < 1e-12, rm
    assert abs(rm['majority_accuracy'] - 0.75) < 1e-12, rm
    assert abs(rm['acc_lift_over_chance'] - 0.25) < 1e-12, rm
    assert abs(rm['acc_lift_over_majority']) < 1e-12, rm
    # A multiclass endpoint must name macro/micro/weighted F1.  Micro equals
    # top-1 accuracy for single-label predictions; macro treats the two
    # observed answers equally; weighted follows support (3:1 here).
    assert abs(rm['f1_micro'] - rm['acc']) < 1e-12, rm
    assert abs(rm['f1_macro'] - (6.0 / 7.0) / 2.0) < 1e-12, rm['f1_macro']
    assert abs(rm['f1_weighted'] - (3.0 * (6.0 / 7.0) / 4.0)) < 1e-12, rm['f1_weighted']
    assert rm['per_class']['0']['predicted'] == 4, rm['per_class']
    assert abs(rm['per_class']['0']['precision'] - 0.75) < 1e-12, rm['per_class']
    assert rm['n_undeclared_predictions'] == 0, rm
    # A prediction of a class absent from the answer support is legitimate
    # (top-1 can emit any token): it counts as an undeclared prediction and
    # must not make the per-class predicted counts fail to sum to n_queries.
    rm_u = L.retrieval_metrics([0, 0, 1], [0, 2, 1])
    assert rm_u['n_undeclared_predictions'] == 1, rm_u
    assert abs(rm_u['f1_micro'] - rm_u['acc']) < 1e-12, rm_u
    _rec_u = dict(rm_u, task='niah_natural', nll=1.0, n_queries=rm_u['n'])
    assert L.metric_policy_audit({'u::seed0': _rec_u})['status'] == 'ok', \
        L.metric_policy_audit({'u::seed0': _rec_u})
    # The NIAH copy rule is a deterministic lexical floor: it must recover a
    # planted key->value pair by copying after the nearest earlier key, and
    # count a query with no earlier key as missing context (not as a hit).
    _ids = np.array([[1, 2, 3, 7, 9, 4, 5, 6, 1, 7, 9]])
    _rule = L.niah_copy_rule_baseline(_ids, np.array([[3]]), np.array([[9]]),
                                      target=4)
    assert _rule['n_correct'] == 1 and _rule['acc'] == 1.0, _rule
    assert _rule['n_missing_context'] == 0 and _rule['rule'] == 'nearest_previous_key_copy', _rule
    _no_ctx = L.niah_copy_rule_baseline(
        np.array([[1, 2, 3, 4, 7, 9]]), np.array([[1]]), np.array([[9]]),
        target=4)
    assert _no_ctx['acc'] == 0.0 and _no_ctx['n_missing_context'] == 1, _no_ctx
    # A target zone wider than the scored inputs cannot define a query offset;
    # it must fail closed instead of silently marking every query as missing.
    try:
        L.niah_copy_rule_baseline(_ids, np.array([[3]]), np.array([[9]]),
                                  target=_ids.shape[1])
    except ValueError as e:
        assert 'exceeds' in str(e), e
    else:
        raise AssertionError('a copy-rule target beyond the sequence must be refused')
    # Mixed label identities must not be merged by NumPy dtype promotion.
    rm2 = L.retrieval_metrics([1, '1'], [1, '1'])
    assert rm2['n_classes'] == 2 and rm2['acc'] == 1.0, rm2
    # A declared label set must cover every true answer: otherwise accuracy
    # counts a class that the support/chance/macro-recall computation drops.
    for _bad in (dict(labels=[0, 1]), dict(labels=[0, 1, 1])):
        try:
            L.retrieval_metrics([0, 1, 2], [0, 1, 2], **_bad)
        except ValueError as e:
            assert ('not in the declared labels' in str(e)
                    or 'duplicate class' in str(e)), (e, _bad)
        else:
            raise AssertionError(f'malformed declared labels were accepted: {_bad}')
    # Declared-but-absent labels stay listed with a null recall and do not
    # change the observed-class chance floor.
    rake = L.retrieval_metrics([0, 0, 1], [0, 0, 0], labels=[0, 1, 2])
    assert rake['n_classes'] == 2 and abs(rake['chance_accuracy'] - 0.5) < 1e-12, rake
    assert rake['per_class_recall']['2'] is None, rake
    # A complete NIAH record is accepted and its floors are recomputed.
    rec = dict(rm, task='niah_natural', nll=1.25, n_queries=rm['n'])
    good = {'n::seed0': rec}
    assert L.metric_policy_audit(good)['status'] == 'ok', L.metric_policy_audit(good)
    # F1 averages must be complete, reproducible from per-class counts, and
    # micro F1 must equal accuracy on a single-label endpoint.
    for _mut, _why in (({'f1_weighted': None}, 'f1-aggregate-invalid'),
                       ({'f1_micro': 0.1}, 'f1-micro-not-accuracy'),
                       ({'f1_macro': 0.99}, 'f1-aggregate-inconsistent')):
        _bad_f1 = json.loads(json.dumps(good))
        _bad_f1['n::seed0'].update(_mut)
        _fa = L.metric_policy_audit(_bad_f1)
        assert _fa['status'] == 'warn', (_mut, _fa)
        assert any(_why in str(c.get('invalid')) for c in _fa['checks']), (_mut, _fa)
    _miss_f1 = json.loads(json.dumps(good))
    del _miss_f1['n::seed0']['f1_weighted']
    _mfa = L.metric_policy_audit(_miss_f1)
    assert _mfa['status'] == 'warn' and any(
        'f1-aggregate-missing' in str(c.get('invalid')) for c in _mfa['checks']), _mfa
    # A persisted lexical copy-rule floor must be reproducible from its exact
    # counts, and a model that does not beat it is called out (the rule is the
    # simplest baseline that can solve this copy task).
    with_rule = json.loads(json.dumps(good))
    with_rule['n::seed0']['copy_rule_baseline'] = {
        'n': 4, 'n_correct': 3, 'acc': 0.75, 'n_missing_context': 0,
        'rule': 'nearest_previous_key_copy'}
    ar = L.metric_policy_audit(with_rule)
    assert ar['status'] == 'warn' and any('copy rule' in w for w in ar['warnings']), ar
    bad_rule = json.loads(json.dumps(with_rule))
    bad_rule['n::seed0']['copy_rule_baseline']['acc'] = 0.9
    abr = L.metric_policy_audit(bad_rule)
    assert abr['status'] == 'warn' and any(
        'copy-rule-acc-count-inconsistent' in str(c.get('invalid'))
        for c in abr['checks']), abr
    assert rm['recommended_primary_metric'] == 'top1_accuracy', rm
    # Extreme answer-class imbalance switches the conditional primary to
    # macro recall; a top-1 headline would otherwise be dominated by the
    # frequent answers.
    _y_imb = [0] * 1000 + [1]
    _p_imb = [0] * 1001
    ri = L.retrieval_metrics(_y_imb, _p_imb)
    assert ri['minority_prevalence'] < 1e-3, ri
    assert ri['recommended_primary_metric'] == 'macro_recall', ri
    assert ri['recommended_primary_available'] is True, ri
    _ri_rec = dict(ri, task='niah_natural', nll=1.0, n_queries=ri['n'])
    _ri_good = {'n::seed0': _ri_rec}
    _ra = L.metric_policy_audit(_ri_good)
    assert _ra['status'] == 'ok', _ra
    _bad_pri = json.loads(json.dumps(_ri_good))
    _bad_pri['n::seed0']['recommended_primary_metric'] = 'top1_accuracy'
    _rbp = L.metric_policy_audit(_bad_pri)
    assert _rbp['status'] == 'warn', _rbp
    assert any('conditional-primary-mismatch' in str(c.get('invalid'))
               for c in _rbp['checks']), _rbp
    _miss_pri = json.loads(json.dumps(_ri_good))
    _miss_pri['n::seed0'].pop('recommended_primary_metric')
    _rmp = L.metric_policy_audit(_miss_pri)
    assert _rmp['status'] == 'warn', _rmp
    assert any('conditional-primary-missing' in str(c.get('invalid'))
               for c in _rmp['checks']), _rmp
    # A support table that does not sum to n_queries describes another sample.
    bad_sup = json.loads(json.dumps(good))
    _first_sup = next(iter(bad_sup['n::seed0']['answer_support']))
    bad_sup['n::seed0']['answer_support'][_first_sup] += 1
    ms = L.metric_policy_audit(bad_sup)
    assert ms['status'] == 'warn', ms
    assert any('answer-support-sum-mismatch' in str(c.get('invalid'))
               for c in ms['checks']), ms
    # Per-class recall must reproduce the recorded correct count/support.
    bad_pcr = json.loads(json.dumps(good))
    _rec_name = next(iter(bad_pcr['n::seed0']['per_class_recall']))
    bad_pcr['n::seed0']['per_class_recall'][_rec_name] = 0.5
    mp = L.metric_policy_audit(bad_pcr)
    assert mp['status'] == 'warn', mp
    # A null recall for a class with positive support drops that class from
    # macro recall and from the correct-count reconciliation: that is missing
    # evidence, not the documented "absent class" null.
    bad_null = json.loads(json.dumps(good))
    _pos_name = next(k for k, v in bad_null['n::seed0']['answer_support'].items()
                     if v > 0)
    bad_null['n::seed0']['per_class_recall'][_pos_name] = None
    mn = L.metric_policy_audit(bad_null)
    assert mn['status'] == 'warn', mn
    assert any('per-class-recall-missing' in str(c.get('invalid'))
               for c in mn['checks']), mn
    # Present-but-malformed floor evidence must fail closed instead of being
    # treated like a legacy record that omits the table entirely.
    for _key, _val, _why in (
            ('answer_support', 'not-a-table', 'answer-support-invalid'),
            ('per_class_recall', ['not', 'a', 'table'], 'per-class-recall-invalid')):
        bad_shape = json.loads(json.dumps(good))
        bad_shape['n::seed0'][_key] = _val
        msh = L.metric_policy_audit(bad_shape)
        assert msh['status'] == 'warn', (msh, _key)
        assert any(_why in str(c.get('invalid')) for c in msh['checks']), (msh, _key)
    # A missing or non-numeric accuracy is an invalid record, not a reason to
    # abort the whole audit with float(None)/max([]).
    for _mut in ('missing', 'nonnumeric'):
        _bad_acc = json.loads(json.dumps(good))
        if _mut == 'missing':
            del _bad_acc['n::seed0']['acc']
        else:
            _bad_acc['n::seed0']['acc'] = 'not-a-number'
        _macc = L.metric_policy_audit(_bad_acc)
        assert _macc['status'] == 'warn', (_mut, _macc)
        assert any('retrieval endpoint is not usable' in w for w in _macc['warnings']), (_mut, _macc)
    # An empty support table has no majority class; it must fail closed rather
    # than raise from max() on an empty sequence.
    _empty_sup = json.loads(json.dumps(good))
    _empty_sup['n::seed0']['answer_support'] = {}
    _mes = L.metric_policy_audit(_empty_sup)
    assert _mes['status'] == 'warn', _mes
    assert any('answer-support-sum-mismatch' in str(c.get('invalid'))
               for c in _mes['checks']), _mes
    # A malformed POLICY (not just a malformed record) must fail closed
    # instead of crashing on .get/.items of a scalar entry.
    for _bad_policy in ([], {'x': None}, {'x': 1}):
        _mpol = L.metric_policy_audit(good, policy=_bad_policy)
        assert _mpol['status'] == 'warn', (_bad_policy, _mpol)
    # A malformed SUMMARY is not a mapping of records either: warn, do not
    # abort on `.values()`.
    for _bad_summary in (1, True, 'x', [None], [1]):
        _msum = L.metric_policy_audit(_bad_summary)
        assert _msum['status'] == 'warn' and _msum['warnings'], (_bad_summary, _msum)
    # A malformed threshold in an otherwise valid policy entry is reported but
    # does not abort the record audit.
    _bad_thr_policy = json.loads(json.dumps(L.METRIC_POLICY))
    _bad_thr_policy['binary_classification']['extreme_imbalance_threshold'] = 'bad'
    _mth = L.metric_policy_audit({'r': {'variant': 'r',
                                        'task': 'binary_classification',
                                        'prevalence': 1e-4,
                                        'recommended_primary_metric': 'f1'}},
                                 policy=_bad_thr_policy)
    assert _mth['status'] == 'warn', _mth
    # A support count above the float range must fail the record closed, not
    # abort the audit with an OverflowError when recall*support is formed.
    _huge = 10 ** 400
    _huge_rec = {'variant': 'h', 'task': 'niah_natural', 'acc': 0.5,
                 'nll': 1.0, 'n_queries': _huge, 'n_correct': _huge // 2,
                 'chance_accuracy': 0.5, 'majority_accuracy': 0.5,
                 'balanced_accuracy': 0.5,
                 'answer_support': {'a': _huge // 2, 'b': _huge - _huge // 2},
                 'per_class_recall': {'a': 0.5, 'b': 0.5}}
    _mh = L.metric_policy_audit({'h::seed0': _huge_rec})
    assert _mh['status'] == 'warn', _mh
    # Forged accuracy / majority floor / macro recall are rejected.
    for key, val in (('acc', 0.9), ('majority_accuracy', 0.1),
                     ('balanced_accuracy', 0.99), ('chance_accuracy', 0.01)):
        bad = json.loads(json.dumps(good))
        bad['n::seed0'][key] = val
        ma = L.metric_policy_audit(bad)
        assert ma['status'] == 'warn', (key, ma)
        assert any('retrieval endpoint is not usable' in w for w in ma['warnings']), (key, ma)
    # A legacy rate without n_correct or a floor is unverifiable, not a pass.
    legacy = {'n::seed0': {'variant': 'n', 'task': 'niah_natural', 'acc': 0.5,
                           'nll': 1.0, 'n_queries': 10}}
    ml = L.metric_policy_audit(legacy)
    assert ml['status'] == 'warn' and any('trivial floor' in w for w in ml['warnings']), ml


def test_baseline_tuning_audit():
    trials = L.BASELINE_TUNING_PROTOCOL['trials'][:2]
    def _arm(ppls):
        rows = [{'config': dict(c), 'ppl': p} for c, p in zip(trials, ppls)]
        best = min(rows, key=lambda r: r['ppl'])
        return {'trials': rows, 'selected': dict(best['config']),
                'selected_metric_value': best['ppl']}
    protocol = dict(L.BASELINE_TUNING_PROTOCOL, arms=['full_matched', 'csa_dynamic'])
    art = {'selection_metric': 'ppl', 'selection_split': 'validation',
           'direction': 'lower_is_better',
           'arms': {'full_matched': _arm([10.0, 11.0]), 'csa_dynamic': _arm([9.0, 9.5])}}
    r = L.baseline_tuning_audit(art, protocol=protocol)
    assert r['tuning_verified'] is True, r
    # An artifact carries the protocol it was produced under (the smoke panel
    # deliberately schedules a subset); the audit must use it when the caller
    # does not pass one explicitly.
    r_auto = L.baseline_tuning_audit(dict(art, protocol=protocol))
    assert r_auto['tuning_verified'] is True, r_auto
    bad = json.loads(json.dumps(art))
    bad['arms']['csa_dynamic']['trials'] = bad['arms']['csa_dynamic']['trials'][:1]
    r2 = L.baseline_tuning_audit(bad, protocol=protocol)
    assert r2['tuning_verified'] is False and any('same number' in p or 'same trial grid' in p for p in r2['problems']), r2
    leak = json.loads(json.dumps(art))
    leak['selection_split'] = 'test'
    r3 = L.baseline_tuning_audit(leak, protocol=protocol)
    assert r3['tuning_verified'] is False and any('validation' in p for p in r3['problems']), r3


def test_baseline_family_declared_contrasts_and_undeclared_arms():
    # The tuned multiplicity family is the set of DECLARED method-vs-baseline
    # contrasts, not `len(arms)-1`.  With two parameter-matching groups the
    # two differ, and using the arm count inflates the planning floor.
    trials = L.BASELINE_TUNING_PROTOCOL['trials'][:2]

    def _arm(ppls):
        rows = [{'config': dict(c), 'ppl': p} for c, p in zip(trials, ppls)]
        best = min(rows, key=lambda r: r['ppl'])
        return {'trials': rows, 'selected': dict(best['config']),
                'selected_metric_value': best['ppl']}

    arms = ['a', 'b', 'c', 'd', 'e']
    protocol = dict(L.BASELINE_TUNING_PROTOCOL, arms=arms, trials=trials,
                    search_seeds=[], final_seeds=[],
                    arm_groups=[
                        {'arms': ['a', 'b', 'c'], 'method_arm': 'a'},
                        {'arms': ['d', 'e'], 'method_arm': 'd'},
                    ])
    plan, basis = L._tuning_declared_contrasts(protocol, arms)
    assert plan == [('a', 'b'), ('a', 'c'), ('d', 'e')], (plan, basis)
    assert basis == 'declared_arm_groups' and len(plan) != len(arms) - 1
    art = {'selection_metric': 'ppl', 'selection_split': 'validation',
           'direction': 'lower_is_better',
           'arms': {a: _arm([10.0, 11.0]) for a in arms}}
    out = L.baseline_tuning_audit(art, protocol=protocol)
    assert out['final_family_size'] == 3, out
    assert out['final_family_basis'] == 'declared_arm_groups', out
    assert out['final_family_contrasts'] == [['a', 'b'], ['a', 'c'], ['d', 'e']], out
    # A verified equal-search artifact is not automatically a tuned claim:
    # this two-level grid is coarse, so the adequacy gate must refuse it.
    assert out['tuning_verified'] is True and out['tuned_claim_allowed'] is False, out
    assert out['tuned_claim_reasons'], out
    # An undeclared arm must not enlarge the family or buy coverage.
    extra = json.loads(json.dumps(art))
    extra['arms']['f'] = _arm([10.0, 11.0])
    rex = L.baseline_tuning_audit(extra, protocol=protocol)
    assert rex['tuning_verified'] is False, rex
    assert any('undeclared arm' in p for p in rex['problems']), rex

    # Search-to-final DATA transfer: a config tuned on one corpus/validation
    # slice is not tuned for a final estimate on another.  Steps and tokens
    # matching does not make the data identity match.
    def _row(cfg, ppl, dfp='DFP', vfp='VFP'):
        return {'config': dict(cfg), 'ppl': float(ppl), 'steps': 10,
                'tokens_seen': 100, 'seeds': [100],
                'seed_ppls': {'100': float(ppl)},
                'data_fp': dfp, 'val_fp': vfp, 'data_version': 'DV'}
    dp = dict(L.BASELINE_TUNING_PROTOCOL, arms=['a', 'b'], trials=trials,
              search_seeds=[100], final_seeds=[0], inherits_tuning={})
    dp.pop('arm_groups', None)
    rows = {'a': [_row(trials[0], 10.0), _row(trials[1], 11.0)],
            'b': [_row(trials[0], 9.0), _row(trials[1], 9.5)]}
    d_art = {'selection_metric': 'ppl', 'selection_split': 'validation',
             'direction': 'lower_is_better', 'search_seeds': [100],
             'final_seeds': [0],
             'arms': {a: {'trials': rr,
                          'selected': dict(min(rr, key=lambda x: x['ppl'])['config']),
                          'selected_metric_value': min(x['ppl'] for x in rr)}
                      for a, rr in rows.items()},
             'final_records': {
                 f'{a}::seed0': {'seed': 0, 'variant': a, 'ppl': 9.0,
                                 'steps': 10, 'tokens_seen': 100,
                                 'data_fp': 'DFP', 'val_fp': 'VFP',
                                 'data_version': 'DV',
                                 'hyperparams': dict(min(rows[a], key=lambda x: x['ppl'])['config'])}
                 for a in ('a', 'b')}}
    okd = L.baseline_tuning_audit(d_art, protocol=dp)
    assert okd['tuning_verified'] is True, okd
    assert okd['data_transfer']['trial_fingerprints']['data_fp']['a']['values'] == ['DFP']
    bad_trial = json.loads(json.dumps(d_art))
    bad_trial['arms']['b']['trials'][0]['data_fp'] = 'OTHER'
    rbt = L.baseline_tuning_audit(bad_trial, protocol=dp)
    assert rbt['tuning_verified'] is False, rbt
    assert any('different data_fp' in p for p in rbt['problems']), rbt
    bad_final = json.loads(json.dumps(d_art))
    bad_final['final_records']['b::seed0']['data_fp'] = 'OTHER'
    rbf = L.baseline_tuning_audit(bad_final, protocol=dp)
    assert rbf['tuning_verified'] is False, rbf
    assert any('tuning trials used' in p for p in rbf['problems']), rbf


def test_baseline_horizon_transfer_audit():
    # A config selected at a shorter search horizon is not the tuned optimum at
    # the final horizon; equal search across arms does not catch this.
    trials = L.BASELINE_TUNING_PROTOCOL['trials'][:2]
    protocol = dict(L.BASELINE_TUNING_PROTOCOL, arms=['a'], trials=trials,
                    search_seeds=[100], final_seeds=[0])

    def _art(search_steps, final_steps, trial_steps=None):
        rows = []
        for _i, _cfg in enumerate(trials):
            _st = trial_steps if trial_steps is not None else search_steps
            rows.append({'config': dict(_cfg), 'ppl': 10.0 + _i,
                         'steps': _st, 'tokens_seen': _st * 8})
        best = min(rows, key=lambda r: r['ppl'])
        return {'selection_metric': 'ppl', 'selection_split': 'validation',
                'direction': 'lower_is_better', 'search_steps': search_steps,
                'search_seeds': [100], 'final_seeds': [0],
                'arms': {'a': {'trials': rows, 'selected': dict(best['config']),
                               'selected_metric_value': best['ppl']}},
                'final_records': {
                    'a::seed0': {'variant': 'a', 'seed': 0, 'ppl': 6.0,
                                 'steps': final_steps,
                                 'tokens_seen': final_steps * 8,
                                 'hyperparams': dict(best['config'],
                                                     steps=final_steps,
                                                     tokens_seen=final_steps * 8)}}}

    good = L.baseline_tuning_audit(_art(10, 10), protocol=protocol)
    assert good['tuning_verified'] is True, good
    assert good['budget_transfer']['equal'] is True, good['budget_transfer']
    assert good['budget_transfer']['tokens_equal'] is True, good['budget_transfer']
    assert not any('different training horizon' in w for w in good['warnings']), good

    mismatch = L.baseline_tuning_audit(_art(10, 20), protocol=protocol)
    assert mismatch['tuning_verified'] is True, mismatch
    assert mismatch['budget_transfer']['equal'] is False, mismatch['budget_transfer']
    assert mismatch['budget_transfer']['mismatch_arms'] == ['a'], mismatch['budget_transfer']
    assert mismatch['budget_transfer']['tokens_equal'] is False, mismatch['budget_transfer']
    assert mismatch['budget_transfer']['token_mismatch_arms'] == ['a'], mismatch['budget_transfer']
    assert any('NOT a fully tuned baseline at the final budget' in w
               for w in mismatch['warnings']), mismatch['warnings']

    malformed = L.baseline_tuning_audit(_art(0, 10), protocol=protocol)
    assert malformed['tuning_verified'] is False, malformed
    assert any('search_steps' in p for p in malformed['problems']), malformed

    trial_mismatch = L.baseline_tuning_audit(_art(10, 10, trial_steps=5), protocol=protocol)
    assert trial_mismatch['tuning_verified'] is False, trial_mismatch
    assert any('does not describe the search budget' in p
               for p in trial_mismatch['problems']), trial_mismatch


def test_tuning_grid_boundary_adequacy():
    # Equal search is not enough if the winner sits on the edge of the grid:
    # the true optimum may be outside the searched range.
    trials = [
        {'lr': 0.0001, 'weight_decay': 0.1, 'warmup': 50},
        {'lr': 0.0003, 'weight_decay': 0.1, 'warmup': 50},
        {'lr': 0.001, 'weight_decay': 0.1, 'warmup': 50},
    ]
    def _art(selected_i, ppls):
        rows = [{'config': dict(c), 'ppl': p} for c, p in zip(trials, ppls)]
        best = rows[selected_i]
        return {'selection_metric': 'ppl', 'selection_split': 'validation',
                'direction': 'lower_is_better',
                'arms': {'a': {'trials': rows, 'selected': dict(best['config']),
                               'selected_metric_value': best['ppl']}}}
    protocol = dict(L.BASELINE_TUNING_PROTOCOL, arms=['a'], trials=trials,
                    search_seeds=[], final_seeds=[])
    edge = L.baseline_tuning_audit(_art(2, [10.0, 9.0, 8.0]), protocol=protocol)
    assert edge['tuning_verified'] is True, edge
    assert edge['grid_adequacy'] == 'warn', edge
    assert edge['grid_boundary_hits'] == ['a:lr'], edge
    assert edge['grid_coverage']['a']['lr']['position'] == 2, edge['grid_coverage']
    assert any('boundary of the declared search grid' in w for w in edge['warnings']), edge
    inside = L.baseline_tuning_audit(_art(1, [10.0, 8.0, 9.0]), protocol=protocol)
    assert inside['grid_adequacy'] == 'ok' and not inside['grid_boundary_hits'], inside
    # A capped search that stopped at the first two declared levels has a
    # boundary at its own maximum even though more levels were declared.
    capped = _art(1, [10.0, 8.0])
    capped['arms']['a']['trials'] = capped['arms']['a']['trials'][:2]
    capped['arms']['a']['selected'] = dict(trials[1])
    rc = L.baseline_tuning_audit(capped, protocol=protocol)
    assert rc['grid_adequacy'] == 'warn', rc
    assert rc['grid_coverage']['a']['lr']['n_searched_levels'] == 2, rc['grid_coverage']
    assert any('searched level' in w for w in rc['warnings']), rc
    # An artifact cannot label a strict-subset search as complete: that would
    # let the tuned-baseline gate accept a truncated grid.
    forged_complete = json.loads(json.dumps(capped))
    forged_complete['search_complete'] = True
    rfc = L.baseline_tuning_audit(forged_complete, protocol=protocol)
    assert rfc['tuning_verified'] is False and rfc['search_complete'] is False, rfc
    assert any('claims search_complete=True' in p for p in rfc['problems']), rfc
    non_bool = json.loads(json.dumps(capped))
    non_bool['search_complete'] = 'yes'
    rnb = L.baseline_tuning_audit(non_bool, protocol=protocol)
    assert rnb['tuning_verified'] is False and any(
        'search_complete must be a boolean' in p for p in rnb['problems']), rnb
    # Trials selected on different evaluation token counts are not comparable.
    _diff_tok = _art(1, [10.0, 8.0, 9.0])
    for _i, _t in enumerate(_diff_tok['arms']['a']['trials']):
        _t['n_tokens'] = 100 if _i < 2 else 50
    rt = L.baseline_tuning_audit(_diff_tok, protocol=protocol)
    assert rt['tuning_verified'] is False and any('different evaluation token counts' in p for p in rt['problems']), rt
    # A selected value outside the declared levels is not a tuned winner.
    outside = _art(1, [10.0, 8.0, 9.0])
    outside['arms']['a']['selected'] = {'lr': 0.002, 'weight_decay': 0.1, 'warmup': 50}
    outside['arms']['a']['trials'].append({'config': dict(outside['arms']['a']['selected']), 'ppl': 7.0})
    outside['arms']['a']['selected_metric_value'] = 7.0
    ro = L.baseline_tuning_audit(outside, protocol=protocol)
    assert ro['tuning_verified'] is False and any('outside the declared trial levels' in p for p in ro['problems']), ro
    # Equal marginal coverage is not full factorial coverage: an OFAT winner
    # cannot be described as a jointly tuned optimum.
    ofat_trials = [
        {'lr': 0.0001, 'weight_decay': 0.1, 'warmup': 50},
        {'lr': 0.0003, 'weight_decay': 0.1, 'warmup': 50},
        {'lr': 0.0003, 'weight_decay': 0.01, 'warmup': 50},
    ]
    ofat_proto = dict(protocol, trials=ofat_trials)
    def _ofat_art(ppls):
        rows = [{'config': dict(c), 'ppl': p} for c, p in zip(ofat_trials, ppls)]
        best = min(rows, key=lambda r: r['ppl'])
        return {'selection_metric': 'ppl', 'selection_split': 'validation',
                'direction': 'lower_is_better',
                'arms': {'a': {'trials': rows, 'selected': dict(best['config']),
                               'selected_metric_value': best['ppl']}}}
    rof = L.baseline_tuning_audit(_ofat_art([10.0, 9.0, 8.0]), protocol=ofat_proto)
    assert rof['tuning_verified'] is True, rof
    assert rof['grid_interaction_untested'] == ['a:lr|weight_decay'], rof
    assert rof['grid_factorial']['a']['full_factorial'] is False, rof['grid_coverage']
    assert rof['grid_factorial']['a']['pairwise']['lr|weight_decay']['n_missing'] == 1, rof
    assert any('pairwise interaction' in w for w in rof['warnings']), rof['warnings']
    full_trials = ofat_trials + [{'lr': 0.0001, 'weight_decay': 0.01, 'warmup': 50}]
    full_proto = dict(protocol, trials=full_trials)
    frows = [{'config': dict(c), 'ppl': 10.0 - i} for i, c in enumerate(full_trials)]
    fbest = min(frows, key=lambda r: r['ppl'])
    rfull = L.baseline_tuning_audit(
        {'selection_metric': 'ppl', 'selection_split': 'validation',
         'direction': 'lower_is_better',
         'arms': {'a': {'trials': frows, 'selected': dict(fbest['config']),
                        'selected_metric_value': fbest['ppl']}}},
        protocol=full_proto)
    assert rfull['tuning_verified'] is True, rfull
    assert rfull['grid_interaction_untested'] == [], rfull
    assert rfull['grid_factorial']['a']['full_factorial'] is True, rfull['grid_coverage']
    # The search design is a declared design choice, not an implicit pile of
    # trials: a declared full factorial must actually be complete, while a
    # declared marginal/OFAT search is allowed but must not claim a joint optimum.
    declared_full = dict(ofat_proto, search_design='full_factorial')
    rbad = L.baseline_tuning_audit(_ofat_art([10.0, 9.0, 8.0]), protocol=declared_full)
    assert rbad['tuning_verified'] is False and any(
        'declares a full_factorial' in p for p in rbad['problems']), rbad
    declared_ofat = dict(ofat_proto, search_design='one_factor_at_a_time')
    rofat = L.baseline_tuning_audit(_ofat_art([10.0, 9.0, 8.0]), protocol=declared_ofat)
    assert rofat['tuning_verified'] is True and rofat['search_design'] == 'one_factor_at_a_time', rofat
    assert any('marginal/OFAT' in w for w in rofat['warnings']), rofat
    bad_design = dict(ofat_proto, search_design='whatever')
    rbd = L.baseline_tuning_audit(_ofat_art([10.0, 9.0, 8.0]), protocol=bad_design)
    assert rbd['tuning_verified'] is False and any('search_design' in p for p in rbd['problems']), rbd
    empty_note = dict(ofat_proto, search_design='hand_designed_grid', search_design_note='')
    ren = L.baseline_tuning_audit(_ofat_art([10.0, 9.0, 8.0]), protocol=empty_note)
    assert ren['tuning_verified'] is False and any('search_design_note' in p for p in ren['problems']), ren


def test_baseline_multi_seed_selection_audit():
    trials = L.BASELINE_TUNING_PROTOCOL['trials'][:2]

    def _row(cfg, ppls, seeds=(0, 1, 2)):
        vals = [ppls[s] for s in seeds]
        return {'config': dict(cfg), 'ppl': float(np.mean(vals)),
                'ppl_std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                'n_seeds': len(vals), 'seeds': list(seeds),
                'seed_ppls': {str(s): float(ppls[s]) for s in seeds},
                'steps': 10, 'tokens_seen': 100}
    def _arm(rows):
        best = min(rows, key=lambda r: r['ppl'])
        return {'trials': rows, 'selected': dict(best['config']),
                'selected_metric_value': best['ppl']}
    protocol = dict(L.BASELINE_TUNING_PROTOCOL, arms=['full_matched', 'csa_dynamic'])
    art = {'selection_metric': 'ppl', 'selection_split': 'validation',
           'direction': 'lower_is_better', 'search_seeds': [0, 1, 2],
           'arms': {
               'full_matched': _arm([_row(trials[0], {0: 10, 1: 10.2, 2: 9.8}),
                                     _row(trials[1], {0: 11, 1: 11.1, 2: 10.9})]),
               'csa_dynamic': _arm([_row(trials[0], {0: 9, 1: 9.2, 2: 8.8}),
                                    _row(trials[1], {0: 9.5, 1: 9.6, 2: 9.4})]),
           }}
    r = L.baseline_tuning_audit(art, protocol=protocol)
    assert r['tuning_verified'] is True, r
    assert r['search_seed_balanced'] is True and r['search_seeds'] == [0, 1, 2], r
    # A trial's aggregate must be the mean of its own per-seed values.
    bad = json.loads(json.dumps(art))
    bad['arms']['csa_dynamic']['trials'][0]['ppl'] += 0.5
    rb = L.baseline_tuning_audit(bad, protocol=protocol)
    assert rb['tuning_verified'] is False and any('not the mean' in p for p in rb['problems']), rb
    # Selecting some arms on more seeds than others is unequal tuning.
    bad2 = json.loads(json.dumps(art))
    for t in bad2['arms']['full_matched']['trials']:
        t['seeds'] = [0, 1]
        t['seed_ppls'] = {k: v for k, v in t['seed_ppls'].items() if k in ('0', '1')}
        t['n_seeds'] = 2
        t['ppl'] = float(np.mean(list(t['seed_ppls'].values())))
        t['ppl_std'] = float(np.std(list(t['seed_ppls'].values()), ddof=1))
    rb2 = L.baseline_tuning_audit(bad2, protocol=protocol)
    assert rb2['tuning_verified'] is False and any('same search-seed set' in p for p in rb2['problems']), rb2
    # Every trial WITHIN an arm must use one search-seed set too.  Merely
    # having the same collection of per-trial sets in every arm is not an
    # equal search: config A and config B would be selected on different
    # draws.
    bad3 = json.loads(json.dumps(art))
    for _arm_name, _entry in bad3['arms'].items():
        _rows = _entry['trials']
        for _row, _keys in zip(_rows, (('0', '1'), ('1', '2'))):
            _row['seeds'] = [int(k) for k in _keys]
            _row['seed_ppls'] = {k: _row['seed_ppls'][k] for k in _keys}
            _row['n_seeds'] = len(_keys)
            _vals = list(_row['seed_ppls'].values())
            _row['ppl'] = float(np.mean(_vals))
            _row['ppl_std'] = float(np.std(_vals, ddof=1))
        _best = min(_rows, key=lambda r: r['ppl'])
        _entry['selected'] = dict(_best['config'])
        _entry['selected_metric_value'] = _best['ppl']
    rb3 = L.baseline_tuning_audit(bad3, protocol=protocol)
    assert rb3['tuning_verified'] is False, rb3
    assert rb3['search_seed_balanced'] is False, rb3
    assert any('more than one search-seed set' in p for p in rb3['problems']), rb3


def test_partial_skipped_seed_keeps_controlled_contrast():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()
    s = {}
    for i in range(8):
        ka = knobs if i else dict(knobs, grad_clip_norm=2.0)
        s[f'a::seed{i}'] = dict(variant='a', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                                ppl=49.0, params=100, train_knobs=ka)
        s[f'b::seed{i}'] = dict(variant='b', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                                ppl=50.0, params=100, train_knobs=knobs)
    r = L.methodology_audit(s, design={'c': [('a', 'b', 'x')]}, baselines={}, verbose=False)
    e = r['components']['c'][0]
    assert e['n_paired'] == 7 and e['n_skipped'] == 1, e
    assert e['controlled'] is True and e['fully_paired'] is False, e
    # The six+ controlled pairs still support the claim; the rejected seed is
    # named instead of silently shrinking the evidence or erasing the contrast.
    assert 'c' in r['verified_components'], r['verified_components']
    assert any('controlled pair(s)' in w for w in r['warnings']), r['warnings']


def test_audit_incomplete_components_alias():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()
    s = {}
    for i in range(3):
        for v, ppl in (('full', 50.0), ('full-A', 49.0)):
            s[f'{v}::seed{i}'] = dict(variant=v, seed=i, run_cfg=fp, steps=1,
                                      tokens_seen=1, ppl=ppl, params=100, train_knobs=knobs)
    design = {'A': [('full', 'full-A', ''), ('full', 'full-A2', '')]}
    r = L.methodology_audit(s, design=design, baselines={}, verbose=False)
    assert r['ablation_completeness']['incomplete_components'] == ['A'], r['ablation_completeness']
    assert r['incomplete_components'] == ['A'], r['incomplete_components']
    assert r['missing_ablation_arms']['A'] == ['full-A2'], r['missing_ablation_arms']


def test_baseline_seed_partition_and_coverage_audit():
    trials = L.BASELINE_TUNING_PROTOCOL['trials'][:2]

    def _row(cfg, ppl):
        return {'config': dict(cfg), 'ppl': float(ppl), 'steps': 10,
                'tokens_seen': 100, 'seeds': [0, 1, 2],
                'seed_ppls': {'0': float(ppl), '1': float(ppl) + 0.1,
                              '2': float(ppl) - 0.1}}

    def _arm(ppls):
        rows = [_row(c, p) for c, p in zip(trials, ppls)]
        best = min(rows, key=lambda r: r['ppl'])
        return {'trials': rows, 'selected': dict(best['config']),
                'selected_metric_value': best['ppl']}

    protocol = dict(L.BASELINE_TUNING_PROTOCOL, arms=['full_matched', 'csa_dynamic'],
                    search_seeds=[0, 1, 2], final_seeds=[0, 1, 2])
    chosen = {'full_matched': dict(trials[0]), 'csa_dynamic': dict(trials[0])}
    art = {
        'selection_metric': 'ppl', 'selection_split': 'validation',
        'direction': 'lower_is_better', 'search_seeds': [0, 1, 2],
        'final_seeds': [0, 1, 2],
        'arms': {'full_matched': _arm([10.0, 11.0]),
                 'csa_dynamic': _arm([9.0, 9.5])},
        'final_records': {
            f'{a}::seed{s}': {'seed': s, 'variant': a, 'ppl': 9.0 + s / 10.0,
                              'params': 100, 'train_knobs': L.training_knobs(),
                              'hyperparams': dict(chosen[a])}
            for a in chosen for s in (0, 1, 2)},
    }
    # The final records were measured on the selection seeds: refuse.
    bad = L.baseline_tuning_audit(dict(art, protocol=protocol), protocol=protocol)
    assert bad['tuning_verified'] is False, bad
    assert bad['search_final_seed_overlap'] == [0, 1, 2], bad
    assert any('overlap' in p for p in bad['problems']), bad

    # A disjoint partition is accepted (the 3-seed final floor is a warning,
    # because the tuned family itself is resolution-limited).
    good_protocol = dict(protocol, search_seeds=[100, 101, 102])
    good = L.baseline_tuning_audit(dict(art, protocol=good_protocol,
                                        search_seeds=[100, 101, 102]),
                                   protocol=good_protocol)
    assert good['tuning_verified'] is True, good
    assert good['search_final_seed_disjoint'] is True, good
    assert good['search_final_seed_overlap'] == [], good

    # Intervention controls inherit their parent's equal-search evidence, but
    # an unknown parent is reported instead of silently covering the arm.
    cov_protocol = dict(protocol, arms=['csa_fixed_rope'],
                        inherits_tuning={'csa_fixed_rope_randidx': 'csa_fixed_rope'})
    cov = L.baseline_tuning_audit(
        dict(art, protocol=cov_protocol,
             arms={'csa_fixed_rope': _arm([10.0, 11.0])},
             final_records={}),
        protocol=cov_protocol)
    assert cov['coverage'].get('csa_fixed_rope_randidx') == 'csa_fixed_rope', cov
    miss_protocol = dict(protocol, arms=['csa_fixed_rope'],
                         inherits_tuning={'csa_fixed_rope_randidx': 'absent_parent'})
    miss = L.baseline_tuning_audit(
        dict(art, protocol=miss_protocol,
             arms={'csa_fixed_rope': _arm([10.0, 11.0])},
             final_records={}),
        protocol=miss_protocol)
    assert any('absent_parent' in w for w in miss['warnings']), miss


def test_tuning_selection_stability_audit():
    def row(cfg, ppls):
        return {'config': dict(cfg), 'ppl': float(np.mean(list(ppls.values()))),
                'ppl_std': float(np.std(list(ppls.values()), ddof=1)),
                'seeds': sorted(ppls), 'seed_ppls': {str(k): float(v) for k, v in ppls.items()},
                'steps': 10, 'tokens_seen': 100}
    declared = L.BASELINE_TUNING_PROTOCOL['trials'][:2]
    stable = [row(declared[0], {0: 10, 1: 10, 2: 10}),
              row(declared[1], {0: 11, 1: 11, 2: 11})]
    s = L.tuning_selection_stability(stable)
    assert s['status'] == 'ok' and s['stable'] is True, s
    unstable = [row(declared[0], {0: 9, 1: 9, 2: 12}),
                row(declared[1], {0: 11, 1: 11, 2: 9})]
    u = L.tuning_selection_stability(unstable)
    assert u['status'] == 'ok' and u['stable'] is False, u
    assert u['n_distinct_configs'] == 2, u
    # The audit verifies a recorded stability claim instead of trusting it.
    trials = unstable
    best = min(trials, key=lambda t: t['ppl'])
    art = {'selection_metric': 'ppl', 'selection_split': 'validation',
           'direction': 'lower_is_better', 'search_seeds': [0, 1, 2],
           'arms': {'a': {'trials': trials, 'selected': dict(best['config']),
                          'selected_metric_value': best['ppl'],
                          'selection_stability': {'stable': True}}}}
    protocol = dict(L.BASELINE_TUNING_PROTOCOL, arms=['a'], trials=declared,
                    inherits_tuning={}, search_seeds=[0, 1, 2],
                    final_seeds=[3, 4, 5])
    out = L.baseline_tuning_audit(art, protocol=protocol)
    assert out['tuning_verified'] is False, out
    assert any('selection_stability' in p for p in out['problems']), out
    # Without the false record the same artifact is merely warning-level: the
    # unstable pick is disclosed, not silently accepted.
    del art['arms']['a']['selection_stability']
    out2 = L.baseline_tuning_audit(art, protocol=protocol)
    assert out2['tuning_verified'] is True, out2
    assert out2['selection_stable_all'] is False, out2
    assert any('not leave-one-search-seed-out stable' in w for w in out2['warnings']), out2
    # Legacy rows without per-seed values can never report stability as a pass.
    no_seed = L.tuning_selection_stability([{'config': {'lr': 1e-4}, 'ppl': 1.0}])
    assert no_seed['status'] == 'unavailable', no_seed
    # One seed under two JSON spellings is a repeated draw, not two draws.
    dup = L.tuning_selection_stability([
        {'config': {'lr': 1e-4}, 'seed_ppls': {'0': 10.0, 0: 10.1,
                                                1: 10.2, 2: 10.3}}])
    assert dup['status'] == 'unavailable' and 'repeats seed' in dup['reason'], dup


def test_paired_row_bootstrap_delta():
    # Evaluation-row bootstrap for a paired token-weighted PPL difference:
    # the SAME rows are resampled for both arms and every shared seed.
    def _pair(sign=1.0, n_seeds=8, n_rows=16, ntok=50):
        out = []
        for s in range(n_seeds):
            a_nll = [2.0 + 0.01 * j for j in range(n_rows)]
            b_nll = [2.0 + 0.01 * j - sign * 0.2 for j in range(n_rows)]
            out.append(({'variant': 'a', 'seed': s,
                         'ppl': math.exp(sum(a_nll) / (n_rows * ntok)),
                         'nll_per_row': a_nll,
                         'n_tokens_per_row': [ntok] * n_rows},
                        {'variant': 'b', 'seed': s,
                         'ppl': math.exp(sum(b_nll) / (n_rows * ntok)),
                         'nll_per_row': b_nll,
                         'n_tokens_per_row': [ntok] * n_rows}))
        return out
    r = L.paired_row_bootstrap_delta(_pair(+1.0))
    assert r['status'] == 'ok' and r['n_pairs'] == 8 and r['n_rows'] == 16, r
    assert r['mean_delta'] > 0 and r['ci95_delta_ppl'][0] > 0, r
    assert r['n_seeds_used'] == 8, r
    # Deterministic draws for the same seed.
    assert L.paired_row_bootstrap_delta(_pair(+1.0))['ci95_delta_ppl'] == r['ci95_delta_ppl']
    # `conf` is coverage, not a tail probability: a 95% interval must be at
    # least as wide as a 50% interval on the same draws.  (Using conf as
    # alpha silently produced a 5% interval here.)
    _lo50 = L.paired_row_bootstrap_delta(_pair(+1.0), conf=0.50)
    _w95 = r['ci95_delta_ppl'][1] - r['ci95_delta_ppl'][0]
    _w50 = _lo50['ci95_delta_ppl'][1] - _lo50['ci95_delta_ppl'][0]
    assert _w95 > _w50, (r['ci95_delta_ppl'], _lo50['ci95_delta_ppl'])
    rn = L.paired_row_bootstrap_delta(_pair(-1.0))
    assert rn['status'] == 'ok' and rn['mean_delta'] < 0 and rn['ci95_delta_ppl'][1] < 0, rn
    # Invalid coverage/draw-count/seed are part of the measurement identity:
    # never silently build a 95% interval while echoing conf=1.0 or 0.5.
    for _bad in (True, 0, 1, 1.5, -0.1, float('nan'), 'x'):
        _r = L.paired_row_bootstrap_delta(_pair(+1.0), conf=_bad)
        assert _r['status'] == 'unverifiable' and _r.get('reason'), (_bad, _r)
        assert _r['conf'] is None and _r['ci95_delta_ppl'] is None, (_bad, _r)
    for _bad in (True, 0, 19, 1.5, 'x', None, float('nan')):
        _r = L.paired_row_bootstrap_delta(_pair(+1.0), n_boot=_bad)
        assert _r['status'] == 'unverifiable' and _r.get('reason'), (_bad, _r)
    for _bad in (True, 1.5, 'x', None, float('nan')):
        _r = L.paired_row_bootstrap_delta(_pair(+1.0), seed=_bad)
        assert _r['status'] == 'unverifiable' and 'seed' in _r.get('reason', ''), (_bad, _r)
    # A paired row bootstrap is not invented when either arm lacks per-row
    # accounting, when row counts differ, or when nested holdout rows were
    # reported without their per-row arrays.
    bad = _pair(+1.0)
    bad[0][0].pop('nll_per_row')
    assert L.paired_row_bootstrap_delta(bad)['status'] == 'unverifiable'
    bad2 = _pair(+1.0)
    bad2[0][1]['nll_per_row'] = bad2[0][1]['nll_per_row'][:-1]
    bad2[0][1]['n_tokens_per_row'] = bad2[0][1]['n_tokens_per_row'][:-1]
    assert L.paired_row_bootstrap_delta(bad2)['status'] == 'unverifiable'
    bad3 = _pair(+1.0)
    for _a, _b in bad3:
        _a['ppl_estimation_rows'] = 'nested_validation_holdout'
        _b['ppl_estimation_rows'] = 'nested_validation_holdout'
    assert L.paired_row_bootstrap_delta(bad3)['status'] == 'unverifiable'
    for _a, _b in bad3:
        _a['nll_per_row_holdout'] = list(_a['nll_per_row'])[:5]
        _a['n_tokens_per_row_holdout'] = [50] * 5
        _b['nll_per_row_holdout'] = list(_b['nll_per_row'])[:5]
        _b['n_tokens_per_row_holdout'] = [50] * 5
    rh = L.paired_row_bootstrap_delta(bad3)
    assert rh['status'] == 'ok' and rh['n_rows'] == 5, rh
    # A pair whose two arms use different estimands is not a paired row
    # comparison, even when both arrays happen to have the same length.
    mixed_pair = _pair(+1.0)
    mixed_pair[0][0]['ppl_estimation_rows'] = 'nested_validation_holdout'
    mixed_pair[0][0]['nll_per_row_holdout'] = list(mixed_pair[0][0]['nll_per_row'])
    mixed_pair[0][0]['n_tokens_per_row_holdout'] = list(mixed_pair[0][0]['n_tokens_per_row'])
    rmix = L.paired_row_bootstrap_delta(mixed_pair)
    assert rmix['status'] == 'unverifiable' and 'different estimands' in rmix['reason'], rmix
    # Nor may different shared pairs contribute intervals from different
    # estimands to one mean/CI.
    mixed_group = _pair(+1.0)
    for _a, _b in mixed_group[:1]:
        for _r in (_a, _b):
            _r['ppl_estimation_rows'] = 'nested_validation_holdout'
            _r['nll_per_row_holdout'] = list(_r['nll_per_row'])
            _r['n_tokens_per_row_holdout'] = list(_r['n_tokens_per_row'])
    rmg = L.paired_row_bootstrap_delta(mixed_group)
    assert rmg['status'] == 'unverifiable' and 'mix' in rmg['reason'], rmg
    # The methodology audit attaches the interval to every controlled contrast
    # (and reports unverifiable rather than inventing rows).
    fp = ('steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_'
          'dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX')
    summary = {}
    for s in range(6):
        for arm, off in (('a', 0.0), ('b', -0.2)):
            nll = [2.0 + off + 0.01 * j for j in range(8)]
            ntok = [50] * 8
            ppl = math.exp(sum(nll) / sum(ntok))
            summary[f'{arm}::seed{s}'] = {
                'variant': arm, 'seed': s, 'run_cfg': fp, 'steps': 1,
                'tokens_seen': 400, 'params': 100, 'ppl': ppl,
                'nll_sum': math.log(ppl) * sum(ntok), 'n_tokens': sum(ntok),
                'nll_per_row': nll, 'n_tokens_per_row': ntok,
                'train_knobs': L.training_knobs(),
            }
    ra = L.methodology_audit(summary, design={'c': [('a', 'b', 'x')]},
                             baselines={}, verbose=False)
    _ent = ra['components']['c'][0]
    _boot = _ent.get('paired_row_bootstrap_delta') or {}
    assert _boot.get('status') == 'ok' and _boot.get('n_rows') == 8, _boot
    # b has the lower NLL, so delta = PPL_a - PPL_b is positive.
    assert _boot.get('mean_delta') > 0 and _boot.get('ci95_delta_ppl')[0] > 0, _boot


def test_methodology_audit_requires_panel_to_use_tuned_config():
    with tempfile.TemporaryDirectory() as d:
        trials = [{'lr': 3e-4, 'weight_decay': 0.1, 'warmup': 1},
                  {'lr': 1e-4, 'weight_decay': 0.1, 'warmup': 1},
                  {'lr': 1e-3, 'weight_decay': 0.1, 'warmup': 1}]
        protocol = dict(L.BASELINE_TUNING_PROTOCOL, arms=['a', 'b'], trials=trials,
                        inherits_tuning={}, search_seeds=[100, 101, 102],
                        final_seeds=[0, 1, 2])
        art = {'protocol': protocol, 'selection_metric': 'ppl',
               'selection_split': 'validation', 'direction': 'lower_is_better',
               'search_seeds': [100, 101, 102], 'final_seeds': [0, 1, 2],
               'arms': {
                   'a': {'selected': dict(trials[0]), 'selected_metric_value': 2.0,
                         'trials': [{'config': c, 'ppl': p}
                                    for c, p in zip(trials, (2.0, 3.0, 4.0))]},
                   'b': {'selected': dict(trials[0]), 'selected_metric_value': 2.1,
                         'trials': [{'config': c, 'ppl': p}
                                    for c, p in zip(trials, (2.1, 3.1, 4.1))]},
               }}
        L.atomic_write_json(os.path.join(d, 'baseline_tuning.json'), art)
        knobs = L.training_knobs()

        def _summary(b_lr=3e-4):
            out = {}
            for i in range(8):
                out[f'a::seed{i}'] = dict(variant='a', seed=i, run_cfg='same_cfg',
                                          steps=1, tokens_seen=10, ppl=50.0, params=100,
                                          train_knobs=knobs,
                                          hyperparams={'lr': 3e-4, 'weight_decay': 0.1,
                                                       'warmup': 1, 'mlp_ratio': 4.0})
                out[f'b::seed{i}'] = dict(variant='b', seed=i, run_cfg='same_cfg',
                                          steps=1, tokens_seen=10, ppl=51.0 + i * 0.0,
                                          params=100, train_knobs=knobs,
                                          hyperparams={'lr': b_lr, 'weight_decay': 0.1,
                                                       'warmup': 1, 'mlp_ratio': 4.0})
            return out

        design = {'comp': [('a', 'b', 'x')]}
        r = L.methodology_audit(_summary(), baselines=design, outdir=d, verbose=False)
        assert r['baseline_tuning']['tuning_verified'] is True, r['baseline_tuning']
        assert r['baseline_tuning']['panel_arm_hyperparams_match'] == {'a': True, 'b': True}, r
        assert 'comp' in r['tuned_verified_baselines'], r.get('tuned_verified_baselines')
        # The panel ran b at a config that was NOT selected: the artifact may
        # exist, but the contrast is not tuned-baseline evidence.
        r2 = L.methodology_audit(_summary(b_lr=1e-4), baselines=design, outdir=d, verbose=False)
        assert r2['baseline_tuning']['panel_arm_hyperparams_match']['b'] is False, r2['baseline_tuning']
        assert 'comp' in r2['untuned_verified_baselines'], r2.get('untuned_verified_baselines')
        assert any('not trained at the tuned config' in w for w in r2['warnings']), r2['warnings']
        # Equal search over a coarse two-level lr grid is not a "well-tuned"
        # baseline: the panel report must keep the contrast in the untuned
        # bucket and expose the tuning caveat, not silently promote it.
        _trials2 = trials[:2]
        _proto2 = dict(protocol, trials=_trials2)
        _art2 = {'protocol': _proto2, 'selection_metric': 'ppl',
                 'selection_split': 'validation', 'direction': 'lower_is_better',
                 'search_seeds': [100, 101, 102], 'final_seeds': [0, 1, 2],
                 'arms': {
                     'a': {'selected': dict(_trials2[0]), 'selected_metric_value': 2.0,
                           'trials': [{'config': c, 'ppl': p}
                                      for c, p in zip(_trials2, (2.0, 3.0))]},
                     'b': {'selected': dict(_trials2[0]), 'selected_metric_value': 2.1,
                           'trials': [{'config': c, 'ppl': p}
                                      for c, p in zip(_trials2, (2.1, 3.1))]},
                 }}
        L.atomic_write_json(os.path.join(d, 'baseline_tuning.json'), _art2)
        r3 = L.methodology_audit(_summary(), baselines=design, outdir=d, verbose=False)
        assert r3['baseline_tuning']['tuning_verified'] is True, r3['baseline_tuning']
        assert r3['baseline_tuning']['tuned_claim_allowed'] is False, r3['baseline_tuning']
        assert 'comp' not in r3['tuned_verified_baselines'], r3.get('tuned_verified_baselines')
        assert 'comp' in r3['untuned_verified_baselines'], r3.get('untuned_verified_baselines')
        assert any('does not support a "well-tuned baseline" claim' in w
                   for w in r3['warnings']), r3['warnings']
        assert any('baseline tuning:' in w and 'coarse' in w
                   for w in r3['warnings']), r3['warnings']
        # A search horizon shorter than the final horizon is not a fully
        # tuned baseline: the selected lr may be optimal for the cheap search
        # budget, not for the budget that is reported.
        _art4 = json.loads(json.dumps(_art2))
        _art4['search_steps'] = 1
        _art4['final_records'] = {}
        for _arm in ('a', 'b'):
            for _seed in (0, 1, 2):
                _art4['final_records'][f'{_arm}::seed{_seed}'] = {
                    'variant': _arm, 'seed': _seed, 'ppl': 50.0, 'steps': 2,
                    'hyperparams': dict(_trials2[0], steps=2)}
        L.atomic_write_json(os.path.join(d, 'baseline_tuning.json'), _art4)
        r4 = L.methodology_audit(_summary(), baselines=design, outdir=d, verbose=False)
        assert r4['baseline_tuning']['budget_transfer']['equal'] is False, r4['baseline_tuning']
        assert r4['baseline_tuning']['tuned_claim_allowed'] is False, r4['baseline_tuning']
        assert 'comp' not in r4['tuned_verified_baselines'], r4.get('tuned_verified_baselines')
        assert 'comp' in r4['untuned_verified_baselines'], r4.get('untuned_verified_baselines')
        assert any('different training horizon' in w for w in r4['warnings']), r4['warnings']


def test_baseline_contrast_plan_is_group_aware():
    arms = list(L.BASELINE_TUNING_PROTOCOL['arms'])
    groups = BT._protocol_arm_groups(L.BASELINE_TUNING_PROTOCOL, L.RUN, arms)
    plan = BT._contrast_plan(L.BASELINE_TUNING_PROTOCOL, groups, arms)
    assert ('csa_dynamic', 'full_matched') in plan and ('csa_dynamic', 'csa_fixed') in plan, plan
    assert ('csa_fixed_rope', 'full_rope') in plan, plan
    # Cross-group contrasts would also change the parameter-matching reference.
    assert not any(a == 'csa_fixed_rope' and b in ('full_matched', 'csa_dynamic')
                   for a, b in plan), plan
    assert not any(a == 'csa_dynamic' and b in ('full_rope', 'csa_fixed_rope')
                   for a, b in plan), plan
    legacy = BT._contrast_plan({'method_arm': 'csa_dynamic'}, groups, arms)
    assert ('csa_dynamic', 'full_matched') in legacy and ('csa_dynamic', 'full_rope') in legacy, legacy
    # The default protocol must be able to confirm its own declared family and
    # must not evaluate the final records on the selection draws.
    proto = L.BASELINE_TUNING_PROTOCOL
    need = L.signflip_min_n_for_holm(len(plan), 0.05)
    assert need is not None and len(proto['final_seeds']) >= need, (need, proto['final_seeds'])
    assert not (set(proto['search_seeds']) & set(proto['final_seeds'])), proto


def test_baseline_driver_refuses_overlapping_final_seeds():
    protocol = dict(L.BASELINE_TUNING_PROTOCOL, arms=['csa_fixed', 'csa_dynamic'],
                    search_seeds=[0], final_seeds=[0])
    try:
        BT.run_tuning(dict(L.RUN, outdir=tempfile.mkdtemp(prefix='tune_overlap_')),
                      outdir=tempfile.mkdtemp(prefix='tune_overlap_out_'),
                      protocol=protocol, final=True)
    except ValueError as e:
        assert 'overlap' in str(e), e
    else:
        raise AssertionError('overlapping search/final seeds were not refused')
    # An empty final partition must not silently fall back to seed 0 (which
    # could be a search draw).
    empty = dict(L.BASELINE_TUNING_PROTOCOL, arms=['csa_fixed', 'csa_dynamic'],
                 search_seeds=[0], final_seeds=[])
    try:
        BT.run_tuning(dict(L.RUN, outdir=tempfile.mkdtemp(prefix='tune_empty_')),
                      outdir=tempfile.mkdtemp(prefix='tune_empty_out_'),
                      protocol=empty, final=True)
    except ValueError as e:
        assert 'final_seeds' in str(e), e
    else:
        raise AssertionError('an empty final seed partition was not refused')
    # A scalar zero is a valid single final seed; `or []` used to erase it.
    _search, _final, _shared = BT._search_final_seeds(
        {'search_seeds': [1], 'final_seeds': 0})
    assert (_search, _final, _shared) == ([1], [0], []), (_search, _final, _shared)


def test_nested_validation_row_split_and_subset_ppl():
    sel, ev, meta = L.validation_row_split(17, seed=123, n_folds=2,
                                           selection_fold=0, evaluation_fold=1)
    assert sorted(sel + ev) == list(range(17)), (sel, ev)
    assert not (set(sel) & set(ev)), (sel, ev)
    assert meta['disjoint'] is True and meta['covers_all_rows'] is True, meta
    assert meta['n_selection_rows'] + meta['n_evaluation_rows'] == 17, meta
    # Validation rows are consecutive stream chunks, so the folds must be
    # contiguous in time order: selection is an earlier block and evaluation a
    # later one.  A seeded row permutation would interleave future chunks into
    # selection and split the same document across the two stages repeatedly.
    assert sel == list(range(len(sel))) and ev == list(range(len(sel), 17)), (sel, ev)
    assert max(sel) < min(ev)
    assert meta['policy'] == 'chronological_contiguous_row_folds', meta
    assert meta['chronological'] is True and meta['selection_precedes_evaluation'] is True, meta
    assert meta['selection_row_range'] == [0, len(sel) - 1], meta
    assert meta['evaluation_row_range'] == [len(sel), 16], meta
    try:
        L.validation_row_split(17, selection_fold=1, evaluation_fold=0)
    except ValueError as e:
        assert 'precede' in str(e), e
    else:
        raise AssertionError('a reverse-time nested split must be refused')
    # Deterministic across calls and independent of the row order in which
    # the same physical rows are presented (indices are the identity).
    sel2, ev2, meta2 = L.validation_row_split(17, seed=123)
    assert (sel, ev, meta) == (sel2, ev2, meta2), 'row split must be deterministic'
    # A different seed must actually move rows (not silently ignore the seed).
    sel3, ev3, _ = L.validation_row_split(17, seed=124)
    assert (sel3, ev3) != (sel, ev), 'row-split seed did not change the folds'
    rec = {'nll_per_row': [1.0, 2.0, 3.0, 4.0],
           'n_tokens_per_row': [10, 10, 10, 10]}
    st = L.record_subset_ppl(rec, [1, 3])
    assert st['n_tokens'] == 20 and abs(st['nll'] - 0.3) < 1e-12, st
    assert abs(st['ppl'] - math.exp(0.3)) < 1e-12, st
    # Missing accounting or out-of-range rows are unverifiable, not silently
    # substituted with the full-validation number.
    assert L.record_subset_ppl({'ppl': 9.0}, [0]) is None
    assert L.record_subset_ppl(rec, [99]) is None
    assert L.record_subset_ppl(rec, []) is None
    # A fractional index must not be truncated to a different row, booleans
    # are flags not row ids, and repeating a row must not double-count its
    # tokens in the held-out pooled PPL.
    assert L.record_subset_ppl(rec, [1.5]) is None
    assert L.record_subset_ppl(rec, [True]) is None
    assert L.record_subset_ppl(rec, [1, 1]) is None
    from decimal import Decimal as _Decimal
    assert L.record_subset_ppl(rec, [_Decimal('1')])['n_tokens'] == 10
    # Token counts are exact integer denominators.  A fractional count must
    # not be used as a pooled denominator (and a boolean is not a count).
    for _bad_ntok in ([10, 10, 10, 10.5], [10, 10, 10, True],
                      [10, 10, 10, float('nan')]):
        assert L.record_subset_ppl({'nll_per_row': [1.0, 2.0, 3.0, 4.0],
                                    'n_tokens_per_row': _bad_ntok}, [1, 3]) is None, _bad_ntok
    assert L._pooled_ppl_from_records([{'nll_sum': 1.0, 'n_tokens': 10.5}]) == (None, None, None)
    assert L._group_params([{'params': 12.5}, {'params': True}]) is None
    assert L._group_params([{'params': 12.0}]) == 12
    for _bad_alpha in (True, 0, 1.0, float('nan')):
        try:
            L._seed_t_ci([1.0, 2.0, 3.0], alpha=_bad_alpha)
        except ValueError:
            pass
        else:
            raise AssertionError(f'_seed_t_ci accepted alpha={_bad_alpha!r}')
    try:
        L.validation_row_split(1)
    except ValueError:
        pass
    else:
        raise AssertionError('a one-row validation slice cannot be split')
    # Fold indices must select their actual contiguous blocks, not silently
    # fall back to "first block vs all remaining rows".
    sel4, ev4, m4 = L.validation_row_split(17, seed=123, n_folds=4,
                                           selection_fold=1, evaluation_fold=3)
    folds = m4['fold_ranges']
    assert sel4 == list(range(folds[1][0], folds[1][1] + 1)), (sel4, m4)
    assert ev4 == list(range(folds[3][0], folds[3][1] + 1)), (ev4, m4)
    assert m4['fold_sizes'] == [5, 4, 4, 4], m4
    assert m4['covers_all_rows'] is False and m4['disjoint'] is True, m4
    try:
        L.validation_row_split(17, n_folds=18)
    except ValueError:
        pass
    else:
        raise AssertionError('more folds than rows must be refused')


def test_nested_tuning_artifact_audit():
    trials = [{'lr': 1e-4, 'weight_decay': 0.1, 'warmup': 5},
              {'lr': 3e-4, 'weight_decay': 0.1, 'warmup': 5}]
    protocol = dict(L.BASELINE_TUNING_PROTOCOL, arms=['a'], trials=trials,
                    search_seeds=[100, 101, 102],
                    final_seeds=[0, 1, 2, 3, 4, 5, 6, 7],
                    inherits_tuning={})
    protocol.pop('arm_groups', None)

    def _row(cfg, nlls):
        seed_ppls = {str(s): math.exp(v / 100.0)
                     for s, v in zip(protocol['search_seeds'], nlls)}
        nll_sum = float(sum(nlls))
        n_tok = 100 * len(nlls)
        ppl = math.exp(nll_sum / n_tok)
        return {'config': dict(cfg), 'ppl': ppl,
                'ppl_std': float(np.std(list(seed_ppls.values()), ddof=1)),
                'ppl_aggregation': 'pooled_token_nll',
                'nll_sum': nll_sum, 'n_tokens': n_tok,
                'seed_ppls': seed_ppls,
                'seed_nll_sum': {str(s): float(v) for s, v in
                                 zip(protocol['search_seeds'], nlls)},
                'seed_n_tokens': {str(s): 100 for s in protocol['search_seeds']},
                'seeds': list(protocol['search_seeds']),
                'steps': 10, 'tokens_seen': 1000,
                'selection_row_split_applied': True,
                'selection_row_fp': 'SELFP', 'evaluation_row_fp': 'EVALFP'}

    rows = [_row(trials[0], [100, 101, 99]), _row(trials[1], [110, 111, 109])]
    best = min(rows, key=lambda r: r['ppl'])
    art = {'selection_metric': 'ppl', 'selection_split': 'validation',
           'direction': 'lower_is_better', 'search_seeds': [100, 101, 102],
           'final_seeds': protocol['final_seeds'],
           'selection_row_split': {'applied': True, 'disjoint': True,
                                   'covers_all_rows': True,
                                   'policy': 'chronological_contiguous_row_folds',
                                   'chronological': True,
                                   'selection_precedes_evaluation': True,
                                   'selection_row_range': [0, 7],
                                   'evaluation_row_range': [8, 15],
                                   'selection_row_fp': 'SELFP',
                                   'evaluation_row_fp': 'EVALFP',
                                   'n_selection_rows': 8, 'n_evaluation_rows': 8},
           'arms': {'a': {'trials': rows, 'selected': dict(best['config']),
                          'selected_metric_value': best['ppl']}},
           'final_records': {
               f'a::seed{s}': {'seed': s, 'variant': 'a', 'ppl': 5.0,
                               'ppl_holdout': 5.1, 'holdout_row_fp': 'EVALFP',
                               'params': 100, 'train_knobs': L.training_knobs(),
                               'hyperparams': dict(best['config'])}
               for s in protocol['final_seeds']}}
    ok = L.baseline_tuning_audit(art, protocol=protocol)
    assert ok['tuning_verified'] is True, ok
    assert ok['selection_row_split_applied'] is True and ok['nested_final_evaluation'] is True, ok
    # A winner inside one standard error of the runner-up is a tie, not a
    # uniquely tuned optimum.  The audit recomputes the margin, warns, and
    # refuses a self-reported margin that does not match the trials.
    tie = json.loads(json.dumps(art))
    tie['arms']['a']['trials'][1] = _row(trials[1], [100, 101, 99])
    rt = L.baseline_tuning_audit(tie, protocol=protocol)
    assert rt['tuning_verified'] is True, rt
    assert any('inside one standard error' in w for w in rt['warnings']), rt['warnings']
    fake = json.loads(json.dumps(art))
    fake['arms']['a']['selection_margin'] = 123.0
    rf = L.baseline_tuning_audit(fake, protocol=protocol)
    assert rf['tuning_verified'] is False, rf
    assert any('selection_margin' in p for p in rf['problems']), rf
    # A claimed nested run without held-out final rows is refused: the tuned
    # contrast would otherwise fall back to the selection documents.
    bad = json.loads(json.dumps(art))
    del bad['final_records']['a::seed0']['ppl_holdout']
    rb = L.baseline_tuning_audit(bad, protocol=protocol)
    assert rb['tuning_verified'] is False, rb
    assert any('held-out-row PPL' in p for p in rb['problems']), rb
    # A final record scored on the wrong row fold is refused too.
    bad2 = json.loads(json.dumps(art))
    bad2['final_records']['a::seed0']['holdout_row_fp'] = 'OTHER'
    rb2 = L.baseline_tuning_audit(bad2, protocol=protocol)
    assert rb2['tuning_verified'] is False and any('different held-out' in p for p in rb2['problems']), rb2
    # Final records must attest the selected knobs.  A missing descriptor or a
    # missing knob is a failed verification, not "nothing to compare".
    bad_hp = json.loads(json.dumps(art))
    bad_hp['final_records']['a::seed0']['hyperparams'] = {}
    rbh = L.baseline_tuning_audit(bad_hp, protocol=protocol)
    assert rbh['tuning_verified'] is False and any('unattested' in p for p in rbh['problems']), rbh
    bad_hp2 = json.loads(json.dumps(art))
    del bad_hp2['final_records']['a::seed1']['hyperparams']
    rbh2 = L.baseline_tuning_audit(bad_hp2, protocol=protocol)
    assert rbh2['tuning_verified'] is False and any('no structured hyperparams' in p for p in rbh2['problems']), rbh2
    # A legacy/seeded random row permutation is disjoint but interleaves future
    # chunks into selection, so it cannot pass the nested-validation gate.
    bad_policy = json.loads(json.dumps(art))
    bad_policy['selection_row_split']['policy'] = 'seeded_splitmix64_row_folds'
    rp = L.baseline_tuning_audit(bad_policy, protocol=protocol)
    assert rp['tuning_verified'] is False, rp
    assert any('chronological_contiguous_row_folds' in p for p in rp['problems']), rp
    # Reverse-time ranges are refused as well.
    bad_range = json.loads(json.dumps(art))
    bad_range['selection_row_split']['selection_row_range'] = [8, 15]
    bad_range['selection_row_split']['evaluation_row_range'] = [0, 7]
    rr = L.baseline_tuning_audit(bad_range, protocol=protocol)
    assert rr['tuning_verified'] is False, rr
    assert any('selection-before-evaluation' in p for p in rr['problems']), rr
    # A search trial selected on different rows than declared is refused.
    bad3 = json.loads(json.dumps(art))
    bad3['arms']['a']['trials'][0]['selection_row_fp'] = 'OTHER'
    rb3 = L.baseline_tuning_audit(bad3, protocol=protocol)
    assert rb3['tuning_verified'] is False and any('selected on different rows' in p for p in rb3['problems']), rb3
    # The trial must also record the SAME held-out fold the artifact will use
    # for the final comparison.
    bad_eval = json.loads(json.dumps(art))
    bad_eval['arms']['a']['trials'][0]['evaluation_row_fp'] = 'OTHER'
    rbe = L.baseline_tuning_audit(bad_eval, protocol=protocol)
    assert rbe['tuning_verified'] is False and any('different evaluation fold' in p for p in rbe['problems']), rbe
    # A truthy non-boolean split flag is not an applied nested split.
    bad_flag = json.loads(json.dumps(art))
    bad_flag['arms']['a']['trials'][0]['selection_row_split_applied'] = 'yes'
    rbf = L.baseline_tuning_audit(bad_flag, protocol=protocol)
    assert rbf['tuning_verified'] is False and any('did not use the declared nested' in p for p in rbf['problems']), rbf
    # Extra final records outside the declared final-seed plan must not enlarge
    # the effective n.
    extra = json.loads(json.dumps(art))
    extra['final_records']['a::seed99'] = dict(extra['final_records']['a::seed0'], seed=99)
    rxe = L.baseline_tuning_audit(extra, protocol=protocol)
    assert rxe['tuning_verified'] is False and any('undeclared seed' in p for p in rxe['problems']), rxe

    # More than two folds intentionally leaves unused blocks, so
    # `covers_all_rows=False` is correct and must not fail the nested gate.
    # The recorded fold description is also re-derived from
    # (n_rows, seed, n_folds, selection_fold, evaluation_fold).
    mf_protocol = dict(protocol, selection_row_split={
        'enabled': True, 'seed': 123, 'n_folds': 4,
        'selection_fold': 1, 'evaluation_fold': 3})
    _sel, _ev, _mf_meta = L.validation_row_split(
        17, seed=123, n_folds=4, selection_fold=1, evaluation_fold=3)
    mf = json.loads(json.dumps(art))
    mf['selection_row_split'] = dict(_mf_meta, applied=True)
    for _t in mf['arms']['a']['trials']:
        _t['selection_row_fp'] = _mf_meta['selection_row_fp']
        _t['evaluation_row_fp'] = _mf_meta['evaluation_row_fp']
    for _r in mf['final_records'].values():
        _r['holdout_row_fp'] = _mf_meta['evaluation_row_fp']
    rmf = L.baseline_tuning_audit(mf, protocol=mf_protocol)
    assert rmf['tuning_verified'] is True, rmf
    assert rmf['selection_row_split_applied'] is True, rmf
    # A self-consistent but edited range/fingerprint must be refused because
    # it cannot be reproduced from the declared fold parameters.
    mf_bad = json.loads(json.dumps(mf))
    mf_bad['selection_row_split']['evaluation_row_range'] = [0, 3]
    rmb = L.baseline_tuning_audit(mf_bad, protocol=mf_protocol)
    assert rmb['tuning_verified'] is False, rmb
    assert any('not reproducible' in p for p in rmb['problems']), rmb
    # A document-group-aware selection split is valid even though a row-only
    # replay would place the boundary at a different row.  The audit must not
    # reject it merely because (n_rows, seed, folds) cannot reproduce it.
    _grp_protocol = dict(protocol, selection_row_split={
        'enabled': True, 'seed': 20261020, 'n_folds': 2,
        'selection_fold': 0, 'evaluation_fold': 1})
    _g_sel, _g_ev, _g_meta = L.validation_row_split(
        4, seed=20261020, n_folds=2, selection_fold=0, evaluation_fold=1,
        groups=[0, 1, 1, 1])
    grp = json.loads(json.dumps(art))
    grp['selection_row_split'] = dict(_g_meta, applied=True)
    for _t in grp['arms']['a']['trials']:
        _t['selection_row_fp'] = _g_meta['selection_row_fp']
        _t['evaluation_row_fp'] = _g_meta['evaluation_row_fp']
    for _r in grp['final_records'].values():
        _r['holdout_row_fp'] = _g_meta['evaluation_row_fp']
    rg = L.baseline_tuning_audit(grp, protocol=_grp_protocol)
    assert rg['tuning_verified'] is True, rg


def test_nested_final_uses_holdout_rows():
    base = dict(L.RUN, d=8, n_layers=1, n_heads=1, d_head=8, seq_len=8,
                batch_size=1, n_train_tokens=32, steps=3, warmup=1)
    trials = [{'lr': 1e-4, 'weight_decay': 0.1, 'warmup': 1}]
    protocol = dict(L.BASELINE_TUNING_PROTOCOL, arms=['a', 'b'], trials=trials,
                    search_seeds=[100, 101, 102], final_seeds=[0, 1, 2, 3],
                    inherits_tuning={}, method_arm='a')
    protocol.pop('arm_groups', None)
    out = tempfile.mkdtemp(prefix='nested_final_')
    selected = {a: {'selected': dict(trials[0]), 'trials': [{}],
                    'selected_metric_value': 1.0} for a in ('a', 'b')}
    L.atomic_write_json(os.path.join(out, 'baseline_tuning.json'),
                        {'arms': selected, 'protocol': protocol})
    orig_run, orig_wm, orig_ev = L.run, L.write_run_manifest, L.log_experiment_event
    _, ev, _ = L.validation_row_split(8)

    def fake_run(cfg, seeds=None, guard=None, label=None, **kw):
        os.makedirs(cfg['outdir'], exist_ok=True)
        summary = {}
        for v in cfg['variants']:
            for sd in seeds:
                base_nll = 1.0 if v == 'a' else 2.0
                rows = [base_nll + (0.5 if i in ev else 0.0) for i in range(8)]
                summary[f'{v}::seed{sd}'] = {
                    'variant': v, 'seed': int(sd),
                    'ppl': float(np.exp(np.mean(rows))),
                    'nll_sum': float(sum(rows)), 'n_tokens': 8,
                    'nll_per_row': rows, 'n_tokens_per_row': [1] * 8,
                    'steps': cfg['steps'],
                    'tokens_seen': cfg['steps'] * cfg['batch_size'] * cfg['seq_len'],
                    'params': 100, 'run_cfg': 'same', 'data_fp': 'd', 'val_fp': 'v',
                    'train_knobs': L.training_knobs(),
                    'hyperparams': {'lr': cfg['lr'], 'weight_decay': cfg['weight_decay'],
                                    'warmup': cfg['warmup']}}
        with open(os.path.join(cfg['outdir'], 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f)

    L.run, L.write_run_manifest, L.log_experiment_event = fake_run, (lambda *a, **k: None), (lambda *a, **k: None)
    try:
        BT._run_final(base, out, protocol, selected)
    finally:
        L.run, L.write_run_manifest, L.log_experiment_event = orig_run, orig_wm, orig_ev
        comp_path = os.path.join(out, 'final', 'tuned_comparison.json')
        comp = json.load(open(comp_path, encoding='utf-8'))
        shutil.rmtree(out, ignore_errors=True)
    assert comp['estimation_rows'] == 'nested_validation_holdout', comp
    # Holdout rows all carry the +0.5 shift, so the expected held-out delta is
    # exp(1.5)-exp(2.5), NOT the full-validation mean difference.
    expect = math.exp(1.5) - math.exp(2.5)
    assert abs(comp['contrasts']['a - b']['mean'] - expect) < 1e-12, comp['contrasts']
    assert comp['final_evaluation_rows']['applied'] is True, comp['final_evaluation_rows']


def test_nested_final_partial_holdout_falls_back_wholly():
    base = dict(L.RUN, d=8, n_layers=1, n_heads=1, d_head=8, seq_len=8,
                batch_size=1, n_train_tokens=32, steps=3, warmup=1)
    trials = [{'lr': 1e-4, 'weight_decay': 0.1, 'warmup': 1}]
    protocol = dict(L.BASELINE_TUNING_PROTOCOL, arms=['a', 'b'], trials=trials,
                    search_seeds=[100, 101, 102], final_seeds=[0, 1],
                    inherits_tuning={}, method_arm='a')
    protocol.pop('arm_groups', None)
    out = tempfile.mkdtemp(prefix='nested_final_partial_')
    selected = {a: {'selected': dict(trials[0]), 'trials': [{}],
                    'selected_metric_value': 1.0} for a in ('a', 'b')}
    L.atomic_write_json(os.path.join(out, 'baseline_tuning.json'),
                        {'arms': selected, 'protocol': protocol})
    orig_run, orig_wm, orig_ev = L.run, L.write_run_manifest, L.log_experiment_event
    _, ev, _ = L.validation_row_split(8)

    def fake_run(cfg, seeds=None, guard=None, label=None, **kw):
        os.makedirs(cfg['outdir'], exist_ok=True)
        summary = {}
        for v in cfg['variants']:
            for sd in seeds:
                base_nll = 1.0 if v == 'a' else 2.0
                rows = [base_nll + (0.5 if i in ev else 0.0) for i in range(8)]
                toks = [1] * 8
                # One record has a malformed per-row token count.  Its row
                # length still matches the others, so the holdout layout is
                # accepted and only record_subset_ppl() fails for this cell.
                if v == 'b' and int(sd) == 0:
                    toks[int(ev[0])] = None
                summary[f'{v}::seed{sd}'] = {
                    'variant': v, 'seed': int(sd),
                    'ppl': float(np.exp(np.mean(rows))),
                    'nll_sum': float(sum(rows)), 'n_tokens': 8,
                    'nll_per_row': rows, 'n_tokens_per_row': toks,
                    'steps': cfg['steps'],
                    'tokens_seen': cfg['steps'] * cfg['batch_size'] * cfg['seq_len'],
                    'params': 100, 'run_cfg': 'same', 'data_fp': 'd', 'val_fp': 'v',
                    'train_knobs': L.training_knobs(),
                    'hyperparams': {'lr': cfg['lr'], 'weight_decay': cfg['weight_decay'],
                                    'warmup': cfg['warmup']}}
        with open(os.path.join(cfg['outdir'], 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f)

    L.run, L.write_run_manifest, L.log_experiment_event = fake_run, (lambda *a, **k: None), (lambda *a, **k: None)
    try:
        BT._run_final(base, out, protocol, selected)
    finally:
        L.run, L.write_run_manifest, L.log_experiment_event = orig_run, orig_wm, orig_ev
        try:
            comp = json.load(open(os.path.join(out, 'final', 'tuned_comparison.json'), encoding='utf-8'))
            final = json.load(open(os.path.join(out, 'final', 'summary.json'), encoding='utf-8'))
        finally:
            shutil.rmtree(out, ignore_errors=True)
    # No record may be switched to the holdout when ANY cell lacks the row
    # accounting: a mixed fold/full-validation contrast is a different sample.
    assert all('ppl_holdout' not in r for r in final.values()), final
    assert comp['estimation_rows'] == 'full_validation', comp
    assert comp['final_evaluation_rows']['applied'] is False, comp['final_evaluation_rows']
    # The whole contrast is on the full-validation PPLs, not a mixture of
    # one record's holdout PPL and another's full-validation PPL.
    expect = final['a::seed0']['ppl'] - final['b::seed0']['ppl']
    assert abs(comp['contrasts']['a - b']['mean'] - expect) < 1e-12, comp['contrasts']


def test_baseline_tuning_driver_multiseed_selection():
    out = tempfile.mkdtemp(prefix='tune_multiseed_')
    trials = [{'lr': 1e-4, 'weight_decay': 0.1, 'warmup': 2},
              {'lr': 3e-4, 'weight_decay': 0.1, 'warmup': 2}]
    protocol = dict(L.BASELINE_TUNING_PROTOCOL, arms=['csa_fixed', 'csa_dynamic'], trials=trials,
                    search_seeds=[0, 1, 2], final_seeds=[0, 1, 2])
    base = dict(L.RUN, d=8, n_layers=1, n_heads=1, d_head=8, seq_len=8,
                batch_size=1, n_train_tokens=32, steps=3, warmup=1)
    calls = []
    orig_run = L.run
    orig_guard = L.overfit_guard

    def fake_run(cfg, seeds=None, guard=None, label=None, **kw):
        calls.append({'variants': list(cfg['variants']), 'seeds': list(seeds)})
        os.makedirs(cfg['outdir'], exist_ok=True)
        summary = {}
        for v in cfg['variants']:
            for s in seeds:
                # lr=3e-4 is better for both arms; a small seed-dependent
                # offset makes the per-seed mean/std real.
                ppl = 20.0 - cfg['lr'] * 10000.0 + 0.1 * int(s) + (1.0 if v == 'csa_dynamic' else 0.0)
                summary[f'{v}::seed{s}'] = {
                    'variant': v, 'seed': int(s), 'ppl': float(ppl),
                    'steps': cfg['steps'], 'tokens_seen': cfg['steps'] * cfg['batch_size'] * cfg['seq_len'],
                    'params': 100, 'run_cfg': f'fake::{cfg["lr"]}::{v}',
                }
        with open(os.path.join(cfg['outdir'], 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f)

    L.run = fake_run
    L.overfit_guard = lambda *a, **k: None
    try:
        art = BT.run_tuning(base, outdir=out, protocol=protocol, final=False)
    finally:
        L.run = orig_run
        L.overfit_guard = orig_guard
        shutil.rmtree(out, ignore_errors=True)
    assert calls and all(c['seeds'] == [0, 1, 2] for c in calls), calls
    assert art['search_seeds'] == [0, 1, 2], art.get('search_seeds')
    for arm, entry in art['arms'].items():
        assert len(entry['trials']) == len(trials), entry
        for row in entry['trials']:
            assert row['n_seeds'] == 3 and len(row['seed_ppls']) == 3, row
            assert abs(row['ppl'] - float(np.mean(list(row['seed_ppls'].values())))) < 1e-12, row
        assert entry['selected']['lr'] == 3e-4, (arm, entry['selected'])
        # Selection uncertainty is recorded, not hidden: the winning config is
        # compared with the runner-up in standard-error units.
        assert entry['selection_margin'] is not None and entry['selection_margin'] > 0, entry
        # The trials were paired on the same search seeds; here every seed
        # moves by exactly the same margin, so the paired SE is 0 and the
        # z-score is undefined (decisive, not "not computable").
        assert entry['selection_margin_method'] == 'paired_search_seeds', entry
        assert entry['selection_margin_se'] == 0.0 and entry['selection_margin_in_se'] is None, entry
        assert entry['selection_decisive'] is True, entry
        assert entry['runner_up'] is not None and entry['runner_up']['config'] != entry['selected'], entry
    # Pairing matters: a consistent per-seed gain can be decisive even when
    # the marginal SDs make the independent-SE approximation look like a tie.
    _b = {'ppl': 10.0, 'ppl_std': 1.0, 'n_seeds': 3,
          'seed_ppls': {'0': 9.0, '1': 11.0, '2': 10.0}}
    _r = {'ppl': 10.1, 'ppl_std': 1.0, 'n_seeds': 3,
          'seed_ppls': {'0': 9.1, '1': 11.1, '2': 10.1}}
    _m = L.tuning_selection_margin(_b, _r)
    assert _m['method'] == 'paired_search_seeds' and _m['se'] == 0.0, _m
    assert _m['decisive'] is True and _m['decisive_independent'] is False, _m
    assert abs(_m['margin'] - 0.1) < 1e-12 and _m['n_paired'] == 3, _m
    # Without per-seed values the legacy independent-SE fallback is used and
    # labelled as such.
    _mf = L.tuning_selection_margin({'ppl': 10.0, 'ppl_std': 1.0, 'n_seeds': 3},
                                    {'ppl': 10.1, 'ppl_std': 1.0, 'n_seeds': 3})
    assert _mf['method'] == 'independent_marginal_se' and _mf['decisive'] is False, _mf
    # A seed repeated under "0" and 0 must not be collapsed into one draw and
    # used as if it were a valid paired set.
    _dup = L.tuning_selection_margin(
        {'ppl': 10.0, 'ppl_std': 1.0, 'n_seeds': 3,
         'seed_ppls': {'0': 9.0, 0: 9.1, '1': 11.0, '2': 10.0}},
        {'ppl': 10.1, 'ppl_std': 1.0, 'n_seeds': 3,
         'seed_ppls': {'0': 9.1, '1': 11.1, '2': 10.1}})
    assert _dup['method'] == 'invalid_seed_ppls' and _dup.get('reason'), _dup
    assert _dup['se'] is None and _dup['decisive'] is None, _dup


def test_simple_baseline_ngram_and_audit():
    rng = np.random.default_rng(7)
    vocab = 32
    train = rng.integers(0, vocab, size=600)
    val = rng.integers(0, vocab, size=(4, 25))
    proto = dict(L.SIMPLE_BASELINE_POLICY,
                 arms={'unigram': {'order': 1}, 'bigram': {'order': 2}},
                 grids={'unigram': {'alpha': [0.1, 1.0]},
                        'bigram': {'alpha': [0.1, 1.0], 'interpolation': [0.5, 1.0]}})
    u = L.ngram_lm_score(train, val, vocab, order=1, alpha=0.1)
    b = L.ngram_lm_score(train, val, vocab, order=2, alpha=0.1, interpolation=0.5)
    assert u['n_tokens'] == val.size and b['n_tokens'] == val.size, (u, b)
    assert math.isfinite(u['ppl']) and math.isfinite(b['ppl']) and u['ppl'] > 0 and b['ppl'] > 0
    # The default floor grid must be wide enough to be called tuned, not just
    # the least-bad of two arbitrary smoothing values.
    assert len(L.SIMPLE_BASELINE_POLICY['grids']['unigram']['alpha']) >= 5
    assert len(L.SIMPLE_BASELINE_POLICY['grids']['bigram']['alpha']) >= 5
    assert len(L.SIMPLE_BASELINE_POLICY['grids']['bigram']['interpolation']) >= 5
    art = L.simple_baseline_fit(train, val, vocab, protocol=proto)
    assert art['best_baseline'] in ('unigram', 'bigram')
    best = min(e['selected_metric_value'] for e in art['baselines'].values())
    assert abs(float(art['best_ppl']) - float(best)) < 1e-12, art
    assert art['data']['val_fp'] and art['data']['n_eval_tokens'] == val.size
    # The count floor must be better than uniform tokens; both values are
    # recorded so the audit can reject a broken floor.
    assert art['uniform_ppl'] == vocab and art['data']['vocab_size'] == vocab, art
    assert art['best_ppl'] <= art['uniform_ppl'] * 1.05, art
    audit = L.simple_baseline_audit(art, protocol=proto)
    assert audit['baseline_verified'] is True, audit
    assert audit['uniform_ppl'] == vocab and audit['best_ppl_vs_uniform'] <= 1.05, audit
    assert audit['search_design'] == 'full_factorial', audit
    assert all(v['complete'] for v in audit['grid_coverage'].values()), audit['grid_coverage']
    # A declared full-factorial count search must actually contain every cell.
    bad_cell = json.loads(json.dumps(art))
    _bgram = [t for t in bad_cell['baselines']['bigram']['trials']
              if (t.get('config') or {}).get('interpolation') == 0.5]
    assert _bgram, bad_cell['baselines']['bigram']['trials']
    bad_cell['baselines']['bigram']['trials'].remove(_bgram[0])
    rbc = L.simple_baseline_audit(bad_cell, protocol=proto)
    assert rbc['baseline_verified'] is False and any(
        'full_factorial' in p and 'not run' in p for p in rbc['problems']), rbc
    # A floor worse than uniform tokens is a data/smoothing bug, not a weak
    # baseline; the audit refuses it.
    bad_uniform = json.loads(json.dumps(art))
    bad_uniform['uniform_ppl'] = 1.0
    ru = L.simple_baseline_audit(bad_uniform, protocol=proto)
    assert ru['baseline_verified'] is False and any(
        'uniform-token floor' in x for x in (ru['problems'] + ru['warnings'])), ru
    # A present-but-malformed floor field is unverifiable, not an absent
    # legacy field: do not silently ignore a bool/string/non-finite value.
    for _key, _val in (('uniform_ppl', True), ('uniform_ppl', '32'),
                       ('uniform_ppl', float('nan')),
                       ('best_ppl_vs_uniform', True),
                       ('best_ppl_vs_uniform', '0.9'),
                       ('best_ppl_vs_uniform', float('inf'))):
        _bad_floor = json.loads(json.dumps(art))
        _bad_floor[_key] = _val
        _rbf = L.simple_baseline_audit(_bad_floor, protocol=proto)
        assert _rbf['baseline_verified'] is False and any(
            _key in x for x in _rbf['problems']), (_key, _val, _rbf)
    # A cell outside the declared search space is not equal-search under ANY
    # design (hand-designed grids included), and the declared per-arm order
    # must be the model that actually ran.
    outside = json.loads(json.dumps(art))
    outside['baselines']['unigram']['trials'][0]['config']['alpha'] = 99.0
    rout = L.simple_baseline_audit(outside, protocol=dict(
        proto, search_design='hand_designed_grid'))
    assert rout['baseline_verified'] is False and any(
        'outside the declared search grid' in p for p in rout['problems']), rout
    wrong_order = json.loads(json.dumps(art))
    wrong_order['baselines']['bigram']['order'] = 1
    rord = L.simple_baseline_audit(wrong_order, protocol=proto)
    assert rord['baseline_verified'] is False and any(
        'differs from the declared order' in p for p in rord['problems']), rord
    bad_design = dict(proto, search_design='whatever')
    rbd = L.simple_baseline_audit(art, protocol=bad_design)
    assert rbd['baseline_verified'] is False and any(
        'search_design' in p for p in rbd['problems']), rbd
    # A selected value that is not the selected config's own trial value is a
    # fabricated result, even if it happens to equal the grid minimum.
    bad = json.loads(json.dumps(art))
    arm = bad['best_baseline']
    bad['baselines'][arm]['selected_metric_value'] = float(bad['baselines'][arm]['selected_metric_value']) + 1.0
    rbad = L.simple_baseline_audit(bad, protocol=proto)
    assert rbad['baseline_verified'] is False and any('does not match' in p for p in rbad['problems']), rbad
    # Per-row accounting must describe the reported floor, not a different
    # sample; otherwise the row bootstrap would resample unscored rows.
    bad_rows = json.loads(json.dumps(art))
    _be = bad_rows['baselines'][bad_rows['best_baseline']]
    if _be.get('reported_nll_per_row'):
        _be['reported_nll_per_row'][0] += 1.0
        rbr = L.simple_baseline_audit(bad_rows, protocol=proto)
        assert rbr['baseline_verified'] is False and any('per-row NLL sum' in p for p in rbr['problems']), rbr
        bad_rows2 = json.loads(json.dumps(art))
        bad_rows2['baselines'][bad_rows2['best_baseline']]['reported_n_tokens_per_row'][0] += 1
        rbr2 = L.simple_baseline_audit(bad_rows2, protocol=proto)
        assert rbr2['baseline_verified'] is False and any('per-row token counts' in p for p in rbr2['problems']), rbr2
    # Missing an arm is not evidence that the floor was tuned.
    bad2 = json.loads(json.dumps(art))
    bad2['baselines'].pop('unigram')
    r2 = L.simple_baseline_audit(bad2, protocol=proto)
    assert r2['baseline_verified'] is False and any('missing' in p for p in r2['problems']), r2
    # A flat 1-D evaluation stream is ONE row: only its first token is
    # unconditioned, and the two call forms must score the same tokens.
    flat = val[0]
    for order, lam in ((2, 0.5), (2, 1.0)):
        two_d = L.ngram_lm_score(train, val[:1], vocab, order=order, alpha=0.1, interpolation=lam)
        one_d = L.ngram_lm_score(train, flat, vocab, order=order, alpha=0.1, interpolation=lam)
        assert abs(two_d['nll'] - one_d['nll']) < 1e-12, (order, lam, two_d, one_d)
        assert abs(two_d['ppl'] - one_d['ppl']) < 1e-9, (order, lam, two_d, one_d)
    # A training stream with fewer than two tokens has no bigram table at
    # all.  Scoring order=2 through the empty table must fall back to the
    # add-alpha unseen conditional instead of raising IndexError.
    for _short in (np.empty(0, dtype=np.int64), np.array([0], dtype=np.int64)):
        _short_score = L.ngram_lm_score(_short, np.array([0, 1, 2]), vocab,
                                        order=2, alpha=0.1, interpolation=1.0)
        assert math.isfinite(_short_score['ppl']) and _short_score['ppl'] > 0, _short_score

    # interpolation=0 must reduce to the unigram model.
    uni = L.ngram_lm_score(train, val, vocab, order=1, alpha=0.1)
    zero = L.ngram_lm_score(train, val, vocab, order=2, alpha=0.1, interpolation=0.0)
    assert abs(uni['nll'] - zero['nll']) < 1e-12 and abs(uni['ppl'] - zero['ppl']) < 1e-9, (uni, zero)
    # An unseen context still gets the add-alpha conditional 1/V, then the
    # declared interpolation.  Falling back to the unigram probability would
    # silently drop the interpolation term for exactly those transitions.
    _ut = np.array([0, 0, 0])
    _ue = np.array([[1, 0]])
    _a, _lam, _v = 0.1, 0.5, 2
    _up = (np.bincount(_ut, minlength=_v) + _a) / (_ut.size + _a * _v)
    _want = (-math.log(_up[1]) - math.log(_lam * (1.0 / _v) + (1.0 - _lam) * _up[0])) / 2
    _got = L.ngram_lm_score(_ut, _ue, _v, order=2, alpha=_a, interpolation=_lam)
    assert abs(_got['nll'] - _want) < 1e-12, (_got, _want)
    # The count-model floor must not be fitted/scored on truncated ids or a
    # fractional vocabulary: those silently change the tokens being counted.
    for _bad in ([1.0, 2.0, 3.0], [True, False, True], np.array([1.5, 2.5])):
        try:
            L.ngram_lm_score(_bad, val, vocab, order=2)
        except TypeError:
            pass
        else:
            raise AssertionError(f'float/bool train ids were accepted: {_bad!r}')
        try:
            L.ngram_lm_score(train, _bad, vocab, order=2)
        except TypeError:
            pass
        else:
            raise AssertionError(f'float/bool eval ids were accepted: {_bad!r}')
    for _bad_vocab in (2.5, True, 0, -3):
        try:
            L.ngram_lm_score(train, val, _bad_vocab, order=1)
        except ValueError:
            pass
        else:
            raise AssertionError(f'invalid vocab_size {_bad_vocab!r} was accepted')
    # Discrete model order and smoothing constants are exact scalars too:
    # True == 1 would silently run the unigram arm as a "bigram" trial.
    for _bad_order in (True, 1.5, '1.5', 3):
        try:
            L.ngram_lm_score(train, val, vocab, order=_bad_order)
        except ValueError:
            pass
        else:
            raise AssertionError(f'invalid order {_bad_order!r} was accepted')
    for _bad_alpha in (True, float('nan'), float('inf'), 0, -1):
        try:
            L.ngram_lm_score(train, val, vocab, order=1, alpha=_bad_alpha)
        except ValueError:
            pass
        else:
            raise AssertionError(f'invalid alpha {_bad_alpha!r} was accepted')
    try:
        L.simple_baseline_fit(train, val, vocab, protocol=dict(
            proto, arms={'u': {'order': 1.5}}, grids={'u': {'alpha': [0.1]}}))
    except ValueError:
        pass
    else:
        raise AssertionError('a fractional declared order must not be truncated')
    # The grid path must not pre-coerce booleans/non-finite values before
    # ngram_lm_score validates them (float(True) == 1.0 used to slip through).
    for _bad_grid in (
            {'arms': {'u': {'order': 1}}, 'grids': {'u': {'alpha': [True]}}},
            {'arms': {'u': {'order': 1}}, 'grids': {'u': {'alpha': [float('nan')]}}},
            {'arms': {'b': {'order': 2}},
             'grids': {'b': {'alpha': [0.1], 'interpolation': [True]}}},
            {'arms': {'b': {'order': 2}},
             'grids': {'b': {'alpha': [0.1], 'interpolation': [float('inf')]}}}):
        try:
            L.simple_baseline_fit(train, val, vocab, protocol=_bad_grid)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f'invalid grid hyperparameter accepted: {_bad_grid!r}')


def test_simple_baseline_grid_adequacy():
    rng = np.random.default_rng(23)
    vocab = 24
    train = rng.integers(0, vocab, size=400)
    val = rng.integers(0, vocab, size=(4, 20))
    proto = dict(L.SIMPLE_BASELINE_POLICY,
                 arms={'uni': {'order': 1}},
                 grids={'uni': {'alpha': [0.001, 0.01, 0.1]}})
    art = L.simple_baseline_fit(train, val, vocab, protocol=proto)
    entry = art['baselines']['uni']
    # Force the artifact to select the largest tested smoothing level, with a
    # clear (non-tie) margin over the runner-up.
    by_alpha = {float(t['config']['alpha']): t for t in entry['trials']}
    assert 0.1 in by_alpha, by_alpha
    by_alpha[0.001]['ppl'] = 2.0
    by_alpha[0.01]['ppl'] = 1.5
    by_alpha[0.1]['ppl'] = 1.0
    entry['selected'] = dict(by_alpha[0.1]['config'])
    entry['selected_metric_value'] = 1.0
    art['best_ppl'] = 1.0
    art['best_ppl_vs_uniform'] = 1.0 / float(vocab)
    for k in ('reported_nll_per_row', 'reported_n_tokens_per_row',
              'reported_estimation_rows', 'holdout_ppl', 'holdout_nll',
              'holdout_n_tokens', 'holdout_metric_value'):
        entry.pop(k, None)
    audit = L.simple_baseline_audit(art, protocol=proto)
    assert audit['baseline_verified'] is True, audit
    assert audit['grid_adequacy'] == 'warn', audit
    assert any('max of the declared range' in w and '0.1' in w for w in audit['warnings']), audit
    assert audit['grid_boundary_hits'] and 'uni:alpha' in audit['grid_boundary_hits'][0], audit
    # Equal best/runner-up values are a tie, not a decisive tuned pick.
    tie = json.loads(json.dumps(art))
    for t in tie['baselines']['uni']['trials']:
        t['ppl'] = 1.0
    tie['best_ppl'] = 1.0
    tie_audit = L.simple_baseline_audit(tie, protocol=proto)
    assert tie_audit['selection_tie_arms'] == ['uni'], tie_audit
    assert any('tie' in w for w in tie_audit['warnings']), tie_audit
    # Interpolation is a second continuous knob: selecting the pure-bigram
    # corner (1.0) means the optimum may lie outside the tested range.
    proto_i = dict(L.SIMPLE_BASELINE_POLICY,
                   arms={'bi': {'order': 2}},
                   grids={'bi': {'alpha': [0.1, 1.0],
                                 'interpolation': [0.25, 0.5, 0.75]}})
    art_i = L.simple_baseline_fit(train, val, vocab, protocol=proto_i)
    entry_i = art_i['baselines']['bi']
    for _t in entry_i['trials']:
        _lam = _t['config']['interpolation']
        _t['ppl'] = {0.25: 3.0, 0.5: 2.0, 0.75: 1.0}[_lam]
    _best_i = [t for t in entry_i['trials']
               if t['config']['interpolation'] == 0.75][0]
    entry_i['selected'] = dict(_best_i['config'])
    entry_i['selected_metric_value'] = 1.0
    art_i['best_ppl'] = 1.0
    art_i['best_ppl_vs_uniform'] = 1.0 / float(vocab)
    for _k in ('reported_nll_per_row', 'reported_n_tokens_per_row',
               'reported_estimation_rows', 'holdout_ppl', 'holdout_nll',
               'holdout_n_tokens', 'holdout_metric_value'):
        entry_i.pop(_k, None)
    ai = L.simple_baseline_audit(art_i, protocol=proto_i)
    assert ai['baseline_verified'] is True, ai
    assert 'bi:interpolation=0.75(max)' in ai['grid_boundary_hits'], ai
    assert any('bigram/unigram mix optimum' in w for w in ai['warnings']), ai


def test_simple_baseline_nested_rows_and_fingerprint():
    rng = np.random.default_rng(19)
    vocab = 24
    train = rng.integers(0, vocab, size=500)
    val = rng.integers(0, vocab, size=(8, 15))
    proto = dict(L.SIMPLE_BASELINE_POLICY,
                 arms={'unigram': {'order': 1}, 'bigram': {'order': 2}},
                 grids={'unigram': {'alpha': [0.1, 1.0]},
                        'bigram': {'alpha': [0.1, 1.0], 'interpolation': [0.5, 1.0]}})
    sel, ev, meta = L.validation_row_split(8, seed=20261020, n_folds=2,
                                           selection_fold=0, evaluation_fold=1)
    art = L.simple_baseline_fit(train, val, vocab, protocol=proto,
                                selection_rows=sel, evaluation_rows=ev,
                                row_split=meta)
    data = art['data']
    assert data['nested_selection'] is True, data
    assert data['n_eval_tokens'] == int(val[ev].size), data
    assert data['selection_fp'] != data['eval_fp'], data
    assert data['row_split']['evaluation_row_fp'] == meta['evaluation_row_fp'], data
    assert art['best_ppl'] == art['baselines'][art['best_baseline']]['holdout_metric_value'], art
    audit = L.simple_baseline_audit(art, protocol=proto)
    assert audit['baseline_verified'] is True, audit
    assert audit['nested_selection'] is True and audit['best_ppl_estimation_rows'] == 'nested_validation_holdout', audit
    # A floor reported on the selection fold is selection-biased, even if it
    # is the minimum selection value.
    bad = json.loads(json.dumps(art))
    bad['best_ppl'] = bad['baselines'][bad['best_baseline']]['selected_metric_value']
    rbad = L.simple_baseline_audit(bad, protocol=proto)
    assert rbad['baseline_verified'] is False and any('held-out' in p for p in rbad['problems']), rbad
    # The held-out evaluation must be on disjoint rows.
    try:
        L.simple_baseline_fit(train, val, vocab, protocol=proto,
                              selection_rows=sel, evaluation_rows=sel,
                              row_split=meta)
    except ValueError:
        pass
    else:
        raise AssertionError('overlapping selection/evaluation rows were accepted')
    # A row-split fingerprint that does not describe the supplied rows is a
    # fabricated nested claim.
    tampered = dict(meta, evaluation_row_fp='not-the-eval-rows')
    try:
        L.simple_baseline_fit(train, val, vocab, protocol=proto,
                              selection_rows=sel, evaluation_rows=ev,
                              row_split=tampered)
    except ValueError:
        pass
    else:
        raise AssertionError('mismatched evaluation_row_fp was accepted')
    # The floor must use the same declared fold layout as the panel: a
    # reproducible but different split is still an unfair comparison.
    sel2, ev2, meta2 = L.validation_row_split(8, seed=123, n_folds=2,
                                              selection_fold=0, evaluation_fold=1)
    art2 = L.simple_baseline_fit(train, val, vocab, protocol=proto,
                                 selection_rows=sel2, evaluation_rows=ev2,
                                 row_split=meta2)
    rsplit = L.simple_baseline_audit(art2, protocol=proto)
    assert rsplit['baseline_verified'] is False and any(
        'different validation folds' in p for p in rsplit['problems']), rsplit
    # The reported token denominator must be the held-out denominator.
    bad_tok = json.loads(json.dumps(art))
    bad_tok['data']['n_eval_tokens'] = int(val.size)
    rtok = L.simple_baseline_audit(bad_tok, protocol=proto)
    assert rtok['baseline_verified'] is False and any('held-out token' in p for p in rtok['problems']), rtok

