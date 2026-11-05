"""Family-level correction for the within-checkpoint lesion ladder.

A lesion artifact can contain many ``(component, arm)`` panels, and a scan can
contain many artifacts.  ``lesion_effect`` tests each panel at the nominal
alpha; reading the best panel, or collecting every ``necessary``/``redundant``
verdict across artifacts, is a multiple-comparison claim.  These tests pin the
family layer: Holm over the exact superiority p and Holm over the paired-TOST
p, the separation of single-panel from family-level verdicts, and the
fail-closed treatment of family declarations.
"""
import contextlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


def _records(component, arm, *, n=6, lesion=11.0, intact=10.0):
    return [{'component': component, 'arm': arm, 'seed': s,
             'checkpoint_sha256': f'ck-{component}-{arm}-{s}',
             'eval_fp': 'same-eval-rows',
             'intact_ppl': intact, 'lesion_ppl': lesion,
             'sensitivity': {'changed': True, 'weights_unchanged': True}}
            for s in range(n)]


def _two_necessary_panels(n=6):
    return (_records('attention_sink', 'csa_fixed', n=n)
            + _records('attention_sink', 'csa_dynamic', n=n))


def test_two_weak_panels_do_not_survive_family_holm():
    rep = L.lesion_panel_audit({'records': _two_necessary_panels(6)})
    # Each panel alone clears alpha (2/64 = 0.03125), but neither survives a
    # two-panel Holm correction (2 * 0.03125 = 0.0625).
    assert rep['status'] == 'warn', rep
    assert rep['claim_allowed_panels'] == [
        'attention_sink/csa_dynamic', 'attention_sink/csa_fixed']
    assert rep['family_claim_allowed_panels'] == [], rep['family']
    assert rep['single_panel_only_panels'] == [
        'attention_sink/csa_dynamic', 'attention_sink/csa_fixed']
    fam = rep['family']
    assert fam['status'] == 'warn', fam
    assert fam['family']['m'] == 2
    assert fam['family']['min_n_for_alpha'] == 7
    assert fam['superiority']['reject_panels'] == []
    for name, row in fam['superiority']['panels'].items():
        assert row['p_raw'] == 0.03125, (name, row)
        assert row['p_holm'] == 0.0625, (name, row)
        assert row['reject'] is False, (name, row)
        assert row['resolution_limited'] is True, (name, row)
    for name, row in fam['panels'].items():
        assert row['single_panel_verdict'] == 'necessary', (name, row)
        assert row['single_panel_claim_allowed'] is True, (name, row)
        assert row['family_verdict'] == 'single_panel_only', (name, row)
        assert row['family_claim_allowed'] is False, (name, row)
        assert 'does not survive Holm' in row['family_claim'], (name, row)
    # The single-panel fields themselves are not rewritten.
    raw = rep['panels']['attention_sink/csa_fixed']
    assert raw['verdict'] == 'necessary' and raw['claim_allowed'] is True


def test_family_claim_survives_with_enough_seeds():
    rep = L.lesion_panel_audit({'records': _two_necessary_panels(8)})
    # n=8: 2/256 = 0.0078125 raw; 2 * it = 0.015625 <= 0.05.
    assert rep['status'] == 'ok', rep
    assert rep['family_claim_allowed_panels'] == [
        'attention_sink/csa_dynamic', 'attention_sink/csa_fixed']
    assert rep['single_panel_only_panels'] == []
    fam = rep['family']
    assert fam['superiority']['reject_panels'] == [
        'attention_sink/csa_dynamic', 'attention_sink/csa_fixed']
    for row in fam['panels'].values():
        assert row['family_verdict'] == 'necessary', row
        assert row['family_claim_allowed'] is True, row


def test_redundant_claim_uses_the_tost_family():
    records = (_records('content_pooling', 'csa_fixed', lesion=10.05)
               + _records('content_pooling', 'csa_dynamic', lesion=10.05))
    rep = L.lesion_panel_audit({'records': records})
    assert rep['status'] == 'ok', rep
    fam = rep['family']
    assert fam['equivalence']['reject_panels'] == [
        'content_pooling/csa_dynamic', 'content_pooling/csa_fixed']
    assert rep['family_claim_allowed_panels'] == [
        'content_pooling/csa_dynamic', 'content_pooling/csa_fixed']
    for row in fam['panels'].values():
        assert row['single_panel_verdict'] == 'redundant', row
        assert row['claim_kind'] == 'equivalence', row
        assert row['family_verdict'] == 'redundant', row
        assert row['family_claim_allowed'] is True, row


def test_superiority_and_equivalence_are_separate_families():
    records = (_records('attention_sink', 'csa_fixed', n=6, lesion=11.0)
               + _records('content_pooling', 'csa_dynamic', n=6,
                          lesion=10.05))
    rep = L.lesion_panel_audit({'records': records})
    fam = rep['family']
    assert fam['family']['m'] == 2
    # The necessary panel is now single-panel only; the redundant panel's
    # equivalence claim is corrected in its own family and survives.
    assert rep['single_panel_only_panels'] == ['attention_sink/csa_fixed']
    assert rep['family_claim_allowed_panels'] == ['content_pooling/csa_dynamic']
    assert fam['panels']['attention_sink/csa_fixed']['claim_kind'] == 'superiority'
    assert fam['panels']['content_pooling/csa_dynamic']['claim_kind'] == 'equivalence'
    assert rep['status'] == 'warn', rep


def test_declared_family_excludes_uncorrected_panels():
    records = _two_necessary_panels(6)
    rep = L.lesion_panel_audit({
        'records': records,
        'family': ['attention_sink/csa_fixed'],
    })
    fam = rep['family']
    assert fam['family']['declared'] is True
    assert fam['family']['names'] == ['attention_sink/csa_fixed']
    assert fam['family']['m'] == 1
    assert rep['family_claim_allowed_panels'] == [
        'attention_sink/csa_fixed']
    excluded = fam['panels']['attention_sink/csa_dynamic']
    assert excluded['in_family'] is False
    assert excluded['family_verdict'] == 'not_in_declared_family'
    assert excluded['family_claim_allowed'] is False
    assert fam['excluded_panels'] == ['attention_sink/csa_dynamic']
    assert any('outside the declared family' in w for w in rep['warnings']), \
        rep['warnings']
    assert rep['status'] == 'warn', rep


def test_invalid_family_declaration_fails_closed():
    records = _two_necessary_panels(6)
    for family in (['does-not-exist'],
                   ['attention_sink/csa_fixed', 'attention_sink/csa_fixed'],
                   [], '', {'attention_sink/csa_fixed'}):
        rep = L.lesion_panel_audit({'records': records, 'family': family})
        assert rep['status'] == 'fail', (family, rep)
        assert rep['problems'], (family, rep)


def test_unusable_p_still_occupies_a_family_slot():
    records = (_records('attention_sink', 'csa_fixed', n=6, lesion=11.0)
               + _records('attention_sink', 'csa_dynamic', n=3, lesion=11.0))
    rep = L.lesion_panel_audit({'records': records})
    fam = rep['family']
    assert fam['family']['m'] == 2
    # The n=3 panel is underpowered and cannot be a claim, but it still makes
    # the n=6 panel's Holm multiplier 2 instead of 1.
    assert fam['panels']['attention_sink/csa_dynamic']['single_panel_verdict'] \
        == 'underpowered'
    assert fam['panels']['attention_sink/csa_fixed'][
        'p_holm_superiority'] == 0.0625
    assert rep['single_panel_only_panels'] == ['attention_sink/csa_fixed']


def test_single_panel_family_does_not_penalise_itself():
    rep = L.lesion_panel_audit({'records': _records(
        'attention_sink', 'csa_dynamic', n=6, lesion=11.0)})
    assert rep['status'] == 'ok', rep
    assert rep['family']['family']['m'] == 1
    assert rep['family_claim_allowed_panels'] == [
        'attention_sink/csa_dynamic']
    assert rep['single_panel_only_panels'] == []
    assert rep['family']['panels']['attention_sink/csa_dynamic'][
        'family_verdict'] == 'necessary'


def test_family_audit_rejects_non_mapping_and_marks_empty_family():
    bad = L.lesion_family_audit([])
    assert bad['status'] == 'fail' and bad['problems'], bad
    empty = L.lesion_family_audit({})
    assert empty['status'] == 'not_verifiable', empty
    # The family audit is a reader: it does not mutate the panel payloads.
    panels = L.lesion_panel_audit({'records': _two_necessary_panels(6)})['panels']
    before = json.dumps({k: v['verdict'] for k, v in panels.items()},
                        sort_keys=True)
    L.lesion_family_audit(panels)
    after = json.dumps({k: v['verdict'] for k, v in panels.items()},
                       sort_keys=True)
    assert before == after


def test_scan_pools_lesion_panels_across_artifacts(tmp_path, monkeypatch):
    import audit_gates as A
    for name, component in (('results_a', 'attention_sink'),
                            ('results_b', 'content_pooling')):
        d = tmp_path / name
        d.mkdir()
        records = _records(component, 'csa_dynamic', n=6, lesion=11.0)
        with open(d / 'lesion.json', 'w', encoding='utf-8') as f:
            json.dump({'records': records}, f)
    monkeypatch.setattr(A, 'REPO', str(tmp_path))
    with contextlib.redirect_stdout(io.StringIO()):
        rep = A.audit_lesion()
    cross = rep['family_across_artifacts']
    assert cross is not None, rep
    assert cross['family']['m'] == 2, cross
    assert rep['status'] == 'warn', rep
    assert cross['family_claim_allowed_panels'] == []
    assert len(cross['single_panel_only_panels']) == 2
    assert any('cross-artifact lesion family' in w for w in rep['warnings']), \
        rep['warnings']
