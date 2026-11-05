"""2x2 difference-in-differences: interaction is not a difference in significance.

The frozen v12 sink-matched / no-sink pair is a 2x2 sensitivity design, not two
independent tests.  These checks exercise the declared cell algebra, the exact
sign-flip interaction, the family max-T / named Holm fallback, the raw-vs-log
scale sensitivity, the artifact replay and the audit/CLI/Makefile wiring.  No
training, no split read, and no frozen p/Holm/confirmed change.
"""
import inspect
import json
import math
import os
import tempfile

import pytest

from test_support import *  # noqa: F401,F403

REPO = os.path.dirname(os.path.abspath(__file__))
DESIGN = L.CONFIRMATION_SINK_2X2
CELL_NAMES = sorted(DESIGN['cells'])
BASE = {
    'csa_sink': 10.0,
    'dense_sink': 10.0,
    'csa_nosink': 10.0,
    'dense_nosink': 10.0,
}
CONFIRMATION_ARMS = ('csa_fixed_rope', 'full_rope_sink',
                     'csa_fixed_rope_nosink', 'full_rope')


def _noise(cell, seed, amplitude=0.01):
    """Deterministic, cell-and-seed-specific multiplicative noise."""
    idx = CELL_NAMES.index(cell)
    return math.exp(amplitude * math.sin((seed + 1) * (idx + 2) * 1.7))


def _arm_noise(arm, seed, amplitude=0.01):
    idx = sorted(CONFIRMATION_ARMS).index(arm)
    return math.exp(amplitude * math.sin((seed + 1) * (idx + 2) * 1.7))


def _panel(effect_sink=-0.6, effect_nosink=-0.4, *, n=12, base=None,
           noise=0.01, nosink_noise=None):
    """Build four cells whose log-scale simple effects are exact in expectation.

    ``delta_sink = log(csa_sink) - log(dense_sink)`` and similarly for
    ``no_sink``; the interaction is ``delta_sink - delta_nosink``.
    """
    base = dict(BASE if base is None else base)
    cells = {cell: {} for cell in CELL_NAMES}
    for seed in range(n):
        for cell in CELL_NAMES:
            level = base[cell]
            amp = noise
            if nosink_noise is not None and cell in ('csa_nosink', 'dense_nosink'):
                amp = nosink_noise
            value = level * _noise(cell, seed, amp)
            if cell == 'dense_sink':
                value = base['csa_sink'] * math.exp(-effect_sink) * _noise(cell, seed, noise)
            elif cell == 'dense_nosink':
                value = base['csa_nosink'] * math.exp(-effect_nosink) * _noise(cell, seed, amp)
            cells[cell][seed] = value
    return cells


def _assert_close(got, want, tol=0.03, msg=''):
    assert got is not None, (msg, got, want)
    assert abs(float(got) - float(want)) <= tol, (msg, got, want)


def test_design_algebra_and_full_two_by_two_required():
    norm = L._ci_normalise_design(DESIGN)
    assert norm['contrast_factor'] == 'architecture'
    assert norm['moderator_factor'] == 'sink'
    assert norm['contrast_levels'] == ['csa', 'dense']
    assert norm['moderator_levels'] == ['sink', 'no_sink']
    assert norm['cell_roles']['csa']['sink'] == 'csa_sink'
    assert norm['cell_roles']['dense']['no_sink'] == 'dense_nosink'
    bad = [
        {'factors': ['a', 'a'], 'levels': {'a': ['x', 'y']}, 'cells': {}},
        {'factors': ['a', 'b'], 'levels': {'a': ['x', 'x'], 'b': ['u', 'v']},
         'cells': {}},
        {'factors': ['a', 'b'], 'levels': {'a': ['x', 'y'], 'b': ['u', 'v']},
         'cells': {'c1': {'a': 'x', 'b': 'u'}}},
        {'factors': ['a', 'b'], 'levels': {'a': ['x', 'y'], 'b': ['u', 'v']},
         'cells': {'c1': {'a': 'x', 'b': 'u'}, 'c2': {'a': 'x', 'b': 'u'},
                   'c3': {'a': 'x', 'b': 'v'}, 'c4': {'a': 'y', 'b': 'v'}}},
    ]
    for design in bad:
        with pytest.raises(ValueError):
            L._ci_normalise_design(design)
    wrong = dict(DESIGN)
    wrong['contrast_factor'] = 'kernel'
    with pytest.raises(ValueError):
        L._ci_normalise_design(wrong)


def test_exact_interaction_and_family_max_t():
    cells = _panel(effect_sink=-0.6, effect_nosink=-0.4, n=12, noise=0.02)
    out = L.difference_in_differences(DESIGN, cells, alpha=0.05, min_n=3)
    assert out['status'] == 'ok', out
    assert out['n'] == 12
    prim = out['scales']['log_ppl']
    _assert_close(prim['interaction']['mean'], -0.2, tol=0.03)
    assert prim['interaction']['p_exact_signflip'] <= 0.05, prim['interaction']
    assert prim['interaction']['p_family'] is not None
    _name = prim['interaction']['name']
    _raw_t = prim['family']['p_raw_studentized'][_name]
    assert _raw_t is not None
    assert prim['interaction']['p_family'] >= _raw_t - 1e-15
    assert prim['interaction']['p_family_statistic'] == 'studentized t max-T'
    assert out['interpretation']['classification'] == 'effect_modification'
    assert out['interaction_claim_allowed'] is True
    assert len(prim['marginals']) == 2
    # Raw-scale companion is present and independently computable.
    raw = out['scales']['ppl']['interaction']
    assert raw['n'] == 12 and raw['mean'] is not None
    assert out['scale_agreement']['sign_agree'] is True


def test_zero_interaction_is_not_a_claim_even_with_two_significant_marginals():
    cells = _panel(effect_sink=-0.6, effect_nosink=-0.6, n=12, noise=0.02)
    out = L.difference_in_differences(DESIGN, cells)
    prim = out['scales']['log_ppl']
    _assert_close(prim['interaction']['mean'], 0.0, tol=0.02)
    assert prim['interaction']['p_exact_signflip'] == 0.0 or prim['interaction']['p_exact_signflip'] >= 0.5
    assert out['interaction_claim_allowed'] is False
    assert out['interpretation']['classification'] in (
        'parallel_effects', 'no_evidence_of_interaction')


def test_difference_in_significance_fallacy_is_flagged():
    # Both marginals are negative on average, but the no-sink marginal has very
    # large seed-to-seed noise: it is not significant, while the sink marginal
    # is.  The interaction (which is driven by that noise) is not significant.
    cells = _panel(effect_sink=-0.8, effect_nosink=-0.8, n=12, noise=0.005,
                   nosink_noise=2.2)
    out = L.difference_in_differences(DESIGN, cells)
    interp = out['interpretation']
    prim = out['scales']['log_ppl']
    assert interp['difference_in_significance_fallacy'] is True, (interp, prim)
    assert interp['interaction_detected_raw'] is False
    assert interp['classification'] in ('difference_in_significance_only',
                                         'difference_in_significance_unresolved')
    assert any('difference in significance' in w.lower()
               or 'difference-in-significance' in w.lower()
               for w in out['warnings'])
    assert out['interaction_claim_allowed'] is False


def test_log_and_raw_scales_can_disagree():
    # Equal log effects but very different baseline PPL levels: the log
    # interaction is zero while the raw-PPL interaction is not.
    base = {'csa_sink': 10.0, 'dense_sink': 10.0,
            'csa_nosink': 20.0, 'dense_nosink': 20.0}
    cells = _panel(effect_sink=-0.5, effect_nosink=-0.5, n=12, base=base,
                   noise=0.01)
    out = L.difference_in_differences(DESIGN, cells)
    assert out['scale_agreement']['sign_agree'] is False
    assert out['scale_agreement']['scale_sensitive'] is True
    assert out['status'] == 'warnings'
    assert any('scale' in w.lower() for w in out['warnings'])


def test_missing_cell_or_common_seeds_are_not_imputed():
    cells = _panel(n=8)
    del cells['dense_sink']
    out = L.difference_in_differences(DESIGN, cells)
    assert out['status'] == 'not_estimable'
    assert 'missing cell' in out['reason']

    cells = _panel(n=8)
    cells['extra'] = {0: 1.0}
    out = L.difference_in_differences(DESIGN, cells)
    assert out['status'] == 'not_estimable'
    assert 'undeclared' in out['reason']

    cells = _panel(n=8)
    for cell in cells:
        cells[cell] = {seed: value for seed, value in cells[cell].items()
                       if seed < 2}
    out = L.difference_in_differences(DESIGN, cells, min_n=3)
    assert out['status'] == 'not_estimable'
    assert 'common seed' in out['reason']

    cells = _panel(n=8)
    cells['csa_sink'][100] = 11.0
    out = L.difference_in_differences(DESIGN, cells, min_n=3)
    assert out['status'] == 'ok'
    assert out['n'] == 8
    assert 100 in out['missing_seeds']['dense_sink']


def test_invalid_measurements_and_arguments_fail_closed():
    for bad in (0.0, -1.0, float('nan'), float('inf'), True):
        cells = _panel(n=8)
        cells['csa_sink'][0] = bad
        out = L.difference_in_differences(DESIGN, cells)
        assert out['status'] == 'not_estimable', (bad, out)
    cells = _panel(n=8)
    cells['csa_sink']['not-a-seed'] = 10.0
    assert L.difference_in_differences(DESIGN, cells)['status'] == 'not_estimable'
    with pytest.raises(ValueError):
        L.difference_in_differences(DESIGN, cells, alpha=1.5)
    with pytest.raises(ValueError):
        L.difference_in_differences(DESIGN, cells, min_n=1)
    with pytest.raises(ValueError):
        L.difference_in_differences(DESIGN, cells, margin=0.0)
    with pytest.raises(ValueError):
        L.difference_in_differences(DESIGN, cells, margin=0.1,
                                    margin_relative_pct=5.0)
    # Consistent declarations are accepted.
    out = L.difference_in_differences(
        DESIGN, _panel(n=8), margin=math.log1p(0.05),
        margin_relative_pct=5.0)
    assert out['margin'] is not None
    assert out['scales']['log_ppl']['tost']['margin'] == out['margin']


def test_tost_is_reported_only_with_a_declared_margin():
    cells = _panel(effect_sink=-0.2, effect_nosink=-0.2, n=12, noise=0.01)
    out = L.difference_in_differences(DESIGN, cells)
    assert out['scales']['log_ppl']['tost'] is None
    out = L.difference_in_differences(DESIGN, cells, margin=0.5)
    tost = out['scales']['log_ppl']['tost']
    assert isinstance(tost, dict)
    assert tost['margin'] == 0.5
    assert tost['equivalent'] is True
    assert out['margin'] == 0.5


def test_artifact_hash_replay_and_created_at_identity():
    cells = _panel(n=8)
    art = L.build_contrast_interaction(DESIGN, cells, alpha=0.05, min_n=3)
    assert art['kind'] == 'contrast_interaction'
    assert len(art['content_sha256']) == 64
    assert L.validate_contrast_interaction(art)['status'] == 'ok'
    moved = json.loads(json.dumps(art))
    moved['created_at'] = '1999-01-01T00:00:00'
    assert L.validate_contrast_interaction(moved)['status'] == 'ok'
    # Tamper the result, then recompute the hash to bypass the easy check:
    # the replay must still catch it.
    tampered = json.loads(json.dumps(art))
    tampered['result']['scales']['log_ppl']['interaction']['mean'] += 0.1
    tampered['content_sha256'] = L._ci_content_sha(tampered)
    verdict = L.validate_contrast_interaction(tampered)
    assert verdict['status'] == 'fail'
    assert any('replay' in p for p in verdict['problems'])
    # A frozen-decision key is refused even when the hash is recomputed.
    forbidden = json.loads(json.dumps(art))
    forbidden['result']['confirmed'] = ['x']
    forbidden['content_sha256'] = L._ci_content_sha(forbidden)
    verdict = L.validate_contrast_interaction(forbidden)
    assert verdict['status'] == 'fail'
    assert any('frozen-decision' in p for p in verdict['problems'])


def test_artifact_not_estimable_roundtrips_without_becoming_a_claim():
    cells = _panel(n=8)
    del cells['dense_sink']
    art = L.build_contrast_interaction(DESIGN, cells)
    verdict = L.validate_contrast_interaction(art)
    assert verdict['status'] == 'not_estimable'
    assert verdict['result']['interaction_claim_allowed'] is False
    assert 'missing cell' in verdict['result']['reason']


def test_writer_noop_conflict_overwrite():
    cells = _panel(n=8)
    art = L.build_contrast_interaction(DESIGN, cells)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'ci.json')
        assert L.write_contrast_interaction(path, art)['status'] == 'written'
        assert L.write_contrast_interaction(path, art)['status'] == 'noop'
        other = L.build_contrast_interaction(DESIGN, _panel(n=8, effect_sink=-0.7))
        assert L.write_contrast_interaction(path, other)['status'] == 'conflict'
        assert L.write_contrast_interaction(path, other, overwrite=True)['status'] == 'written'
        with pytest.raises(ValueError):
            bad = json.loads(json.dumps(art))
            bad['content_sha256'] = '0' * 64
            L.write_contrast_interaction(os.path.join(td, 'bad.json'), bad)


def test_compare_reports_exact_paths():
    cells = _panel(n=8)
    out = L.difference_in_differences(DESIGN, cells)
    other = json.loads(json.dumps(out))
    other['scales']['log_ppl']['interaction']['mean'] += 0.01
    cmp = L.compare_contrast_interaction_results(out, other)
    assert cmp['match'] is False
    assert any('interaction.mean' in d for d in cmp['differences'])


def test_audit_gate_ok_warnings_fail_and_missing():
    cells = _panel(effect_sink=-0.6, effect_nosink=-0.4, n=12)
    art = L.build_contrast_interaction(DESIGN, cells)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'ci.json')
        L.write_contrast_interaction(path, art)
        assert A.audit_contrast_interaction(path, verbose=False)['status'] == 'ok'
        tampered = json.loads(json.dumps(art))
        tampered['result']['interpretation']['classification'] = 'parallel_effects'
        tampered['content_sha256'] = L._ci_content_sha(tampered)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(tampered, f)
        gate = A.audit_contrast_interaction(path, verbose=False)
        assert gate['status'] == 'fail'
        assert gate['problems']
    # Warnings result maps to warnings, not ok.
    base = {'csa_sink': 10.0, 'dense_sink': 10.0,
            'csa_nosink': 20.0, 'dense_nosink': 20.0}
    warn_cells = _panel(effect_sink=-0.5, effect_nosink=-0.5, n=12,
                        base=base)
    warn_art = L.build_contrast_interaction(DESIGN, warn_cells)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'ci.json')
        L.write_contrast_interaction(path, warn_art)
        assert A.audit_contrast_interaction(path, verbose=False)['status'] == 'warnings'
    missing = A.audit_contrast_interaction(
        os.path.join(tempfile.gettempdir(), 'definitely-missing-ci.json'),
        verbose=False)
    assert missing['status'] == 'not_verifiable_locally'


def test_test_stats_embedded_block_replays(monkeypatch):
    cells = _panel(effect_sink=-0.6, effect_nosink=-0.4, n=12, noise=0.02)
    by = {}
    for arm, cell in zip(
            CONFIRMATION_ARMS,
            ('csa_sink', 'dense_sink', 'csa_nosink', 'dense_nosink')):
        by[arm] = {'4096': {'ppl_by_seed': {
            str(seed): cells[cell][seed] for seed in range(12)}}}
    stored = L.confirmation_sink_interaction(by)
    assert stored['status'] in ('ok', 'warnings'), stored
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, 'analysis_v12'))
        path = os.path.join(td, 'analysis_v12', 'test_stats.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'by_length': by, 'sink_factorial': stored}, f)
        monkeypatch.setattr(A, 'REPO', td)
        gate = A.audit_contrast_interaction(verbose=False)
        assert gate['status'] == 'ok', gate
        assert gate['source'] == 'test_stats.sink_factorial'
        assert gate['classification'] == stored['interpretation']['classification']
        # Tamper the stored block: the same by-length cells no longer replay it.
        bad = json.loads(json.dumps(stored))
        bad['interpretation']['classification'] = 'parallel_effects'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'by_length': by, 'sink_factorial': bad}, f)
        gate = A.audit_contrast_interaction(verbose=False)
        assert gate['status'] == 'fail'
        # A missing four-arm panel is honestly not verifiable.
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'by_length': {'only_one_arm': {}}}, f)
        gate = A.audit_contrast_interaction(verbose=False)
        assert gate['status'] == 'not_verifiable_locally'


def test_v12_wiring_and_markdown():
    src = inspect.getsource(V12.finaltest_stats)
    assert 'confirmation_sink_interaction' in src
    assert 'sink_factorial' in src
    report_src = inspect.getsource(V12.build_report)
    assert 'sink_factorial' in report_src
    gate_src = inspect.getsource(A.audit_contrast_interaction)
    assert 'sink_factorial' in gate_src
    makefile = open(os.path.join(REPO, 'Makefile'), encoding='utf-8').read()
    assert 'sink-interaction' in makefile
    assert 'contrast_interaction' in makefile
    cli_src = inspect.getsource(A._run_cli)
    assert "'contrast_interaction'" in cli_src
    report_gate_src = inspect.getsource(A.run_report)
    assert "res['contrast_interaction']" in report_gate_src
    text = L.contrast_interaction_markdown(
        L.difference_in_differences(DESIGN, _panel(n=8)))
    assert '2x2 interaction' in text
    assert 'exploratory' in text
    # The protected files are still present and in place.
    for rel in ('README.md', 'README_zh.md', '.github/workflows/test.yml'):
        assert os.path.exists(os.path.join(REPO, rel)), rel


def test_confirmation_helper_rejects_incomplete_or_bad_endpoint():
    out = L.confirmation_sink_interaction({})
    assert out['status'] == 'not_estimable'
    by = {'csa_fixed_rope': {'4096': {'ppl_by_seed': {0: 10.0}}}}
    out = L.confirmation_sink_interaction(by, endpoint='not-an-endpoint')
    assert out['status'] == 'not_estimable'
