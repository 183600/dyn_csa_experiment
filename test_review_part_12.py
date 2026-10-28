"""CPU review checks (split group 12; t_signflip_power_table_scales_exactly .. t_compare_token_errors_reports_bucket_inference_and_markdown)."""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)

def test_signflip_power_table_scales_exactly():
    def _reference(deltas, m_family, alpha, qs, expected_sign):
        abs_d = np.asarray([abs(float(x)) for x in deltas], dtype=float)
        n = int(abs_d.size)
        total = 1 << n
        masks = np.arange(total, dtype=np.int64)[:, None]
        bits = (1 << np.arange(n, dtype=np.int64))[None, :]
        signs = np.where((masks & bits) != 0, 1.0, -1.0)
        sums = signs @ abs_d
        abs_sums = np.abs(sums)
        p = (abs_sums[None, :] >= abs_sums[:, None] - 1e-12).mean(axis=1)
        p_holm = np.minimum(1.0, p * float(m_family))
        k_plus = np.asarray([bin(i).count('1') for i in range(total)], dtype=int)
        if expected_sign == 'negative':
            sign_ok = sums < 0
            k_power = n - k_plus
            k_same = k_power
        elif expected_sign == 'positive':
            sign_ok = sums > 0
            k_power = k_plus
            k_same = k_power
        else:
            sign_ok = np.ones(total, dtype=bool)
            k_power = k_plus
            k_same = np.maximum(k_plus, n - k_plus)
        power = {}
        for q in qs:
            prob = (q ** k_power) * ((1.0 - q) ** (n - k_power))
            power[f'q={q:.2f}'] = float(
                (prob * ((p_holm < alpha) & sign_ok)).sum())
        min_same = None
        for s in range(n + 1):
            sel = (k_same == s) & sign_ok
            if sel.any() and float(p_holm[sel].min()) < alpha:
                min_same = int(s)
                break
        return {'power': power, 'min_same_sign_for_alpha': min_same,
                'min_attainable_p_holm': float(p_holm.min())}

    qs = [0.5, 0.8, 0.95, 1.0]
    for n in range(2, 10):
        deltas = [0.2 + 0.11 * i for i in range(n)]
        for expected_sign in ('positive', 'negative', None):
            got = V12._signflip_power_table(deltas, 6, 0.05, qs, expected_sign)
            ref = _reference(deltas, 6, 0.05, qs, expected_sign)
            assert got['enumerated'] == 1 << n, got
            for q in qs:
                key = f'q={q:.2f}'
                assert abs(got['power'][key] - ref['power'][key]) < 1e-12, (
                    n, expected_sign, key, got['power'][key], ref['power'][key])
            assert got['min_same_sign_for_alpha'] == ref['min_same_sign_for_alpha'], got
            assert abs(got['min_attainable_p_holm']
                       - ref['min_attainable_p_holm']) < 1e-15, got

    # The old non-directional branch credited both all-positive and
    # all-negative patterns with k=n at q=1, yielding power=2.  The actual
    # positive-sign mass at q=1 is one pattern (power=1).
    ones = V12._signflip_power_table([1.0] * 8, 6, 0.05, [1.0], None)
    assert abs(ones['power']['q=1.00'] - 1.0) < 1e-12, ones
    assert ones['min_same_sign_for_alpha'] == 8, ones
    half = V12._signflip_power_table([1.0] * 8, 6, 0.05, [0.5], None)
    assert abs(half['power']['q=0.50'] - 2.0 / 256.0) < 1e-12, half

    big = V12._signflip_power_table(
        [0.2 + 0.11 * i for i in range(16)], 8, 0.05, [0.95], 'negative')
    assert big['n'] == 16 and big['enumerated'] == 1 << 16, big
    assert 0.0 <= big['power']['q=0.95'] <= 1.0, big
    assert big['min_attainable_p_holm'] > 0.0, big
    try:
        V12._signflip_power_table([1.0] * 21, 8, 0.05, [0.95], 'negative')
    except ValueError as exc:
        assert 'capped at n<=20' in str(exc), exc
    else:
        raise AssertionError('expected the n>20 power table to fail closed')


def test_sensitivity_audit_validates_seed_and_comparison_identity():
    with tempfile.TemporaryDirectory() as td:
        mp = os.path.join(td, 'manifest.json')
        manifest = {
            'manifest_version': 3,
            'seeds': list(range(12)),
            'statistics': {'alpha': 0.05, 'min_power_for_test': 0.80},
            'primary_comparisons': [
                {'name': 'a', 'expected_sign': 'negative', 'sesoi': 0.01},
            ],
        }

        def _audit():
            with open(mp, 'w', encoding='utf-8') as f:
                json.dump(manifest, f)
            with contextlib.redirect_stdout(io.StringIO()):
                return A.audit_design_sensitivity(
                    manifest_path=mp, pilot_path=os.path.join(td, 'missing.json'))

        manifest['seeds'] = list(range(11)) + [10]
        rep = _audit()
        assert rep['status'] == 'fail', rep
        assert any('manifest.seeds contains duplicates' in v
                   for v in rep['violations']), rep['violations']

        manifest['seeds'] = list(range(12))
        manifest['statistics']['n_seeds'] = 13
        rep = _audit()
        assert rep['status'] == 'fail', rep
        assert any('statistics.n_seeds=13 does not equal len(seeds)=12' in v
                   for v in rep['violations']), rep['violations']
        manifest['statistics'].pop('n_seeds')

        manifest['primary_comparisons'] = [
            {'name': 'a', 'expected_sign': 'negative', 'sesoi': 0.01},
            {'name': 'a', 'expected_sign': 'negative', 'sesoi': 0.01},
        ]
        rep = _audit()
        assert rep['status'] == 'fail', rep
        assert any('duplicate primary-comparison name' in v
                   for v in rep['violations']), rep['violations']
        assert len(rep['comparisons']) == 1, rep

        manifest['primary_comparisons'] = [
            {'name': 'a', 'expected_sign': 'negative', 'sesoi': 0.01,
             'require_sesoi_power': 'false'},
        ]
        rep = _audit()
        assert rep['status'] == 'fail', rep
        assert any('must be a boolean' in v for v in rep['violations']), rep['violations']


def test_audit_design_sensitivity_manifest_and_pilot():
    with tempfile.TemporaryDirectory() as td:
        mp = os.path.join(td, 'manifest.json')
        pp = os.path.join(td, 'power_validation.json')
        manifest = {
            'manifest_version': 3,
            'seeds': list(range(12)),
            'statistics': {'alpha': 0.05, 'min_power_for_test': 0.80,
                           'sesoi': 0.01, 'require_sesoi_power': True},
            'primary_comparisons': [
                {'name': 'a', 'expected_sign': 'negative',
                 'sesoi': 0.01, 'require_sesoi_power': True},
            ],
        }
        with open(mp, 'w', encoding='utf-8') as f:
            json.dump(manifest, f)
        with open(pp, 'w', encoding='utf-8') as f:
            json.dump({'comparisons': {
                'a': {'deltas': [-1.0] * 10 + [-0.5, -2.0]}}}, f)
        with contextlib.redirect_stdout(io.StringIO()):
            rep = A.audit_design_sensitivity(manifest_path=mp, pilot_path=pp)
        assert rep['status'] == 'fail', rep
        assert rep['pilot_status'] == 'available', rep
        cell = rep['comparisons']['a']
        assert cell['pilot']['status'] == 'available', cell['pilot']
        assert cell['pilot']['n'] == 12, cell['pilot']
        assert cell['sesoi']['status'] == 'underpowered', cell['sesoi']
        assert any('a: require_sesoi_power=true and the planned n is '
                   'underpowered' in v for v in rep['violations']), rep['violations']

        manifest['statistics'].pop('sesoi')
        manifest['statistics'].pop('require_sesoi_power')
        manifest['primary_comparisons'][0].pop('sesoi')
        manifest['primary_comparisons'][0].pop('require_sesoi_power')
        with open(mp, 'w', encoding='utf-8') as f:
            json.dump(manifest, f)
        with contextlib.redirect_stdout(io.StringIO()):
            rep2 = A.audit_design_sensitivity(manifest_path=mp, pilot_path=pp)
        assert rep2['status'] == 'ok', rep2
        assert rep2['comparisons']['a']['sesoi']['status'] == 'not_declared', rep2
        assert any('declares no SESOI' in w for w in rep2['warnings']), rep2['warnings']

        with contextlib.redirect_stdout(io.StringIO()):
            rep3 = A.audit_design_sensitivity(
                manifest_path=mp, pilot_path=os.path.join(td, 'missing.json'))
        assert rep3['status'] == 'not_verifiable', rep3
        assert rep3['pilot_status'] == 'not_available', rep3

        with open(pp, 'w', encoding='utf-8') as f:
            json.dump({'comparisons': {'b': {'deltas': [-1.0, -2.0]}}}, f)
        with contextlib.redirect_stdout(io.StringIO()):
            rep4 = A.audit_design_sensitivity(manifest_path=mp, pilot_path=pp)
        assert rep4['status'] == 'not_verifiable', rep4
        assert rep4['pilot_status'] == 'no_matching_comparisons', rep4
        assert any('none carry a deltas list' in w for w in rep4['warnings']), rep4['warnings']


def test_attrition_power_audit_and_design_gate():
    base = L.attrition_power_audit(
        12, 6, q=0.95, pair_retention=1.0, target_power=0.80,
        require_complete_coverage=True, min_complete_pairs=8)
    assert base['status'] == 'ok', base['violations']
    assert base['runnable_at'] == 12 and base['family_floor'] == 8, base
    assert abs(base['conditional_power']
               - L.signflip_power(12, 6, 0.95, 0.05)) < 1e-15, base
    assert abs(base['unconditional_power'] - base['conditional_power']) < 1e-15
    assert base['probability_runnable'] == 1.0, base
    assert base['target_met'] is True and base['coverage_bottleneck'] is False
    assert base['min_pair_retention_for_target'] is not None, base
    assert 0.99 < base['min_pair_retention_for_target'] < 1.0, base
    assert base['required_nominal_seeds'] == 12, base
    _sc = {s['pair_retention']: s for s in base['scenarios']}
    assert abs(_sc[0.95]['unconditional_power'] - 0.95 ** 12
               * base['conditional_power']) < 1e-12, _sc[0.95]

    # With complete coverage, 5% pair loss multiplies power by P(K=12), so
    # the conditional 0.882 becomes an unconditional 0.476.  No nominal n can
    # recover 80% here because the all-12 requirement caps availability.
    loss = L.attrition_power_audit(
        12, 6, q=0.95, pair_retention=0.95, target_power=0.80,
        require_complete_coverage=True)
    assert loss['status'] == 'impossible', loss
    assert loss['failure_reason'] == 'power', loss
    assert loss['coverage_bottleneck'] is True, loss
    assert abs(loss['probability_runnable'] - 0.95 ** 12) < 1e-12, loss
    assert abs(loss['unconditional_power'] - 0.95 ** 12
               * loss['conditional_power']) < 1e-12, loss
    assert loss['power_loss'] > 0.3 and loss['required_nominal_seeds'] is None

    # The same 5% loss with an 8-pair floor is a different design: the panel
    # is almost certainly runnable, but the smaller effective n drops power to
    # 0.742, which is still below target.  The audit must separate those two
    # failure modes instead of reporting one "attrition" number.
    floor = L.attrition_power_audit(
        12, 6, q=0.95, pair_retention=0.95, target_power=0.80,
        require_complete_coverage=False, min_complete_pairs=8,
        availability_target=0.95)
    assert floor['runnable_at'] == 8, floor
    assert floor['probability_runnable'] > 0.999, floor
    assert floor['unconditional_power'] > loss['unconditional_power'], floor
    assert floor['status'] == 'underpowered', floor
    assert floor['required_nominal_seeds'] == 13, floor
    assert floor['failure_reason'] is None, floor

    # min_complete_pairs below the Holm resolution floor is not a lower bar.
    below = L.attrition_power_audit(
        12, 6, q=0.95, pair_retention=0.95,
        require_complete_coverage=False, min_complete_pairs=3)
    assert below['runnable_at'] == 8, below
    assert any('below the Holm resolution floor' in w
               for w in below['warnings']), below

    # An undeclared retention is visible as not_declared, never as a pass;
    # requiring it turns the missing declaration into a failure.
    nd = L.attrition_power_audit(12, 6, q=0.95, target_power=0.80)
    assert nd['status'] == 'not_declared', nd
    assert nd['min_pair_retention_for_target'] is not None, nd
    assert any('no pair_retention' in w for w in nd['warnings']), nd
    need = L.attrition_power_audit(
        12, 6, q=0.95, target_power=0.80, require_attrition_power=True)
    assert need['status'] == 'fail', need
    assert any('no pair_retention' in v for v in need['violations']), need
    required = L.attrition_power_audit(
        12, 6, q=0.95, pair_retention=0.95, target_power=0.80,
        require_attrition_power=True)
    assert required['status'] == 'fail', required
    assert any('attrition-adjusted power' in v
               for v in required['violations']), required

    assert L.attrition_power_audit(
        12, 6, q=1.0, pair_retention=1.0,
        target_power=0.99)['unconditional_power'] == 1.0
    for kwargs in (dict(n_seeds=1), dict(m_comparisons=0), dict(q=-0.01),
                   dict(pair_retention=1.5), dict(alpha=0.0),
                   dict(target_power=1.0), dict(require_complete_coverage='yes'),
                   dict(min_complete_pairs=-1), dict(availability_target=0.0),
                   dict(retention_scenarios='x'), dict(n_max=1)):
        _kw = dict(n_seeds=12, m_comparisons=6, q=0.95)
        _kw.update(kwargs)
        bad = L.attrition_power_audit(**_kw)
        assert bad['status'] == 'fail', (kwargs, bad)

    with tempfile.TemporaryDirectory() as td:
        mp = os.path.join(td, 'm.json')
        pp = os.path.join(td, 'p.json')
        manifest = {
            'manifest_version': 3, 'seeds': list(range(12)),
            'statistics': {
                'alpha': 0.05, 'min_power_for_test': 0.80,
                'power_q_for_test': 0.95,
                'require_complete_coverage': True,
                'min_seeds_for_confirmation': 8,
                'attrition': {'pair_retention': 0.95,
                              'require_attrition_power': True}},
            'primary_comparisons': [
                {'name': 'a', 'expected_sign': 'negative'}],
        }
        with open(mp, 'w', encoding='utf-8') as f:
            json.dump(manifest, f)
        with open(pp, 'w', encoding='utf-8') as f:
            json.dump({'comparisons': {'a': {'deltas': [-1.0] * 12}}}, f)
        with contextlib.redirect_stdout(io.StringIO()):
            rep = A.audit_design_sensitivity(manifest_path=mp, pilot_path=pp)
        assert rep['status'] == 'fail', rep
        assert rep['attrition']['declared'] is True, rep['attrition']
        assert rep['comparisons']['a']['attrition']['status'] == 'fail', rep
        assert any('a: require_attrition_power=true' in v
                   for v in rep['violations']), rep['violations']

        manifest['statistics']['attrition']['pair_retention'] = 1.0
        with open(mp, 'w', encoding='utf-8') as f:
            json.dump(manifest, f)
        with contextlib.redirect_stdout(io.StringIO()):
            rep = A.audit_design_sensitivity(manifest_path=mp, pilot_path=pp)
        assert rep['status'] == 'ok', rep
        assert rep['comparisons']['a']['attrition']['status'] == 'ok', rep

        manifest['statistics'].pop('attrition')
        with open(mp, 'w', encoding='utf-8') as f:
            json.dump(manifest, f)
        with contextlib.redirect_stdout(io.StringIO()):
            rep = A.audit_design_sensitivity(manifest_path=mp, pilot_path=pp)
        assert rep['attrition']['declared'] is False, rep['attrition']
        assert any('no statistics.attrition.pair_retention' in w
                   for w in rep['warnings']), rep['warnings']

        manifest['statistics']['attrition'] = {'pair_retention': 1.0}
        manifest['primary_comparisons'][0]['expected_sign'] = -1
        with open(mp, 'w', encoding='utf-8') as f:
            json.dump(manifest, f)
        with contextlib.redirect_stdout(io.StringIO()):
            rep = A.audit_design_sensitivity(manifest_path=mp, pilot_path=pp)
        assert rep['status'] == 'ok', rep
        assert rep['comparisons']['a']['expected_sign'] == -1, rep
        assert rep['comparisons']['a']['attrition']['status'] == 'ok', rep

        manifest['statistics']['attrition'] = {
            'pair_retention': 1.0, 'require_attrition_power': 'false'}
        manifest['primary_comparisons'][0]['expected_sign'] = 'negative'
        with open(mp, 'w', encoding='utf-8') as f:
            json.dump(manifest, f)
        with contextlib.redirect_stdout(io.StringIO()):
            rep = A.audit_design_sensitivity(manifest_path=mp, pilot_path=pp)
        assert rep['status'] == 'fail', rep
        assert rep['attrition']['require_attrition_power'] is False, rep['attrition']
        assert rep['comparisons']['a']['attrition'][
            'require_attrition_power'] is False, rep
        assert any('require_attrition_power' in v and 'must be a boolean' in v
                   for v in rep['violations']), rep['violations']

        # A non-directional comparison never enters the per-comparison
        # attrition audit.  The top-level gate must still reject a malformed
        # pre-registered boolean instead of coercing "yes" to True.
        manifest['statistics']['attrition'] = {'pair_retention': 1.0}
        manifest['statistics']['require_complete_coverage'] = 'yes'
        manifest['primary_comparisons'][0]['expected_sign'] = None
        with open(mp, 'w', encoding='utf-8') as f:
            json.dump(manifest, f)
        with contextlib.redirect_stdout(io.StringIO()):
            rep = A.audit_design_sensitivity(manifest_path=mp, pilot_path=pp)
        assert rep['status'] == 'fail', rep
        assert rep['attrition']['require_complete_coverage'] is False, rep['attrition']
        assert any('require_complete_coverage' in v and 'must be a boolean' in v
                   for v in rep['violations']), rep['violations']

    huge = L.attrition_power_audit(
        12, 10 ** 400, q=0.95, pair_retention=1.0)
    assert huge['status'] == 'fail', huge
    assert huge['family_floor'] is None, huge

    big = L.attrition_power_audit(
        1000, 6, q=0.95, pair_retention=1.0, n_max=8)
    assert big['status'] == 'ok', big
    assert big['n_max'] == 8, big
    assert big['required_nominal_seeds'] is None, big
    assert big['conditional_power'] == 1.0 and big['unconditional_power'] == 1.0, big

    import inspect
    assert 'attrition_power_audit' in inspect.getsource(A.audit_design_sensitivity)
    assert 'attrition_power_audit' in inspect.getsource(V12.run_power)


def test_unit_precision_design_planning():
    deltas = [-0.20] * 8 + [0.05] * 40
    base = L.unit_precision_design(
        48, pilot_unit_deltas=deltas, unit_label='document',
        sesoi=0.05, target_power=0.80)
    assert base['status'] == 'ok', base['violations']
    assert base['n_units'] == 48 and base['unit_label'] == 'document'
    _sd = float(np.std(deltas, ddof=1))
    _tc = L._t_critical(0.95, 47.0)
    assert abs(base['precision']['ci_halfwidth']
               - _tc * _sd / math.sqrt(48.0)) < 1e-12, base['precision']
    assert base['pilot']['sd'] == _sd, base['pilot']
    assert base['mde']['absolute'] > 0.0, base['mde']
    assert base['population']['claim_allowed'] is False, base['population']
    assert any('min_units_for_population' in w for w in base['warnings']), base['warnings']
    assert base['sesoi']['status'] in ('ok', 'underpowered'), base['sesoi']

    large = L.unit_precision_design(
        64, pilot_unit_deltas=deltas, unit_label='document')
    assert large['status'] == 'ok', large['violations']
    assert large['population']['claim_allowed'] is True, large['population']
    row = L.unit_precision_design(
        640, pilot_unit_deltas=deltas, unit_label='row')
    assert row['status'] == 'ok', row['violations']
    assert row['population']['claim_allowed'] is False, row['population']
    assert any('evaluation window' in w for w in row['warnings']), row['warnings']

    tight = L.unit_precision_design(
        48, pilot_unit_sd=0.1, target_halfwidth=0.01)
    assert tight['status'] == 'ok', tight['violations']
    assert tight['precision']['meets_target'] is False, tight['precision']
    assert tight['precision']['n_required_for_target_halfwidth'] > 48, tight['precision']
    loose = L.unit_precision_design(
        48, pilot_unit_sd=0.1, target_halfwidth=1.0)
    assert loose['precision']['meets_target'] is True, loose['precision']
    assert loose['precision']['n_required_for_target_halfwidth'] <= 48, loose['precision']

    under = L.unit_precision_design(
        48, pilot_unit_sd=0.1, sesoi=0.001, target_power=0.80,
        require_sesoi_power=True)
    assert under['status'] == 'fail', under
    assert under['sesoi']['status'] == 'underpowered', under['sesoi']
    assert under['sesoi']['required_n_for_target_power'] > 48, under['sesoi']
    assert any('underpowered at the pre-registered SESOI' in v
               for v in under['violations']), under['violations']
    advisory = L.unit_precision_design(
        48, pilot_unit_sd=0.1, sesoi=0.001, target_power=0.80)
    assert advisory['status'] == 'ok', advisory
    assert advisory['sesoi']['status'] == 'underpowered', advisory['sesoi']
    assert any('SESOI' in w for w in advisory['warnings']), advisory['warnings']

    bare = L.unit_precision_design(48)
    assert bare['status'] == 'not_verifiable', bare
    assert bare['pilot']['status'] == 'not_provided', bare['pilot']

    # A pilot with too few/zero-variance units has no SD, so MDE/SESOI power
    # cannot be computed; that is not a green "ok" design.
    one = L.unit_precision_design(48, pilot_unit_deltas=[0.1])
    assert one['status'] == 'not_verifiable', one
    zero_sd = L.unit_precision_design(48, pilot_unit_deltas=[0.1, 0.1])
    assert zero_sd['status'] == 'not_verifiable', zero_sd

    # Regression: the normal-approximation n0 can exceed n_max even though a
    # smaller feasible n exists.  The old 50-step upward scan returned None
    # here (n0=112936 > n_max=100000, true minimum=74871).
    _sd, _alpha = 54.45357975358687, 0.35524694384435124
    _margin, _target_power = 0.12912223662142594, 0.4492389684641835
    missed = L.unit_precision_design(
        48, pilot_unit_sd=_sd, alpha=_alpha, sesoi=_margin,
        target_power=_target_power)
    _needed = missed['sesoi']['required_n_for_target_power']
    assert _needed is not None and 2 < _needed <= 100000, missed['sesoi']
    assert (L._unit_power_at(_needed, _sd, _margin, _alpha) + 1e-15
            >= _target_power), missed['sesoi']
    assert (L._unit_power_at(_needed - 1, _sd, _margin, _alpha) + 1e-15
            < _target_power), missed['sesoi']

    for kwargs in (dict(alpha=float('nan')), dict(target_power=1.0),
                   dict(pilot_unit_sd=-1.0), dict(pilot_unit_deltas='abc'),
                   dict(unit_label='line'), dict(target_halfwidth=0.0),
                   dict(require_sesoi_power='yes'), dict(n_max=1),
                   dict(min_units_for_population=1)):
        bad = L.unit_precision_design(48, **kwargs)
        assert bad['status'] == 'fail', (kwargs, bad)
    assert L.unit_precision_design(True)['status'] == 'fail'


def test_paired_unit_deltas_prefers_independent_units():
    a = [{'ppl_per_article': [10.0, 12.0, 14.0],
          'article_ids': ['a', 'b', 'c'],
          'ppl_per_doc': [10.0, 12.0], 'doc_ids': ['x', 'y'],
          'ppl_per_seq': [10.0, 12.0]},
         {'ppl_per_article': [10.5, 12.5, 14.5],
          'article_ids': ['a', 'b', 'c'],
          'ppl_per_doc': [10.5, 12.5], 'doc_ids': ['x', 'y'],
          'ppl_per_seq': [10.5, 12.5]}]
    b = [{'ppl_per_article': [9.0, 10.0, 11.0],
          'article_ids': ['a', 'b', 'c'],
          'ppl_per_doc': [9.5, 10.5], 'doc_ids': ['x', 'y'],
          'ppl_per_seq': [9.0, 11.0]},
         {'ppl_per_article': [9.5, 10.5, 11.5],
          'article_ids': ['a', 'b', 'c'],
          'ppl_per_doc': [10.0, 11.0], 'doc_ids': ['x', 'y'],
          'ppl_per_seq': [9.5, 11.5]}]
    got = L.paired_unit_deltas(a, b)
    assert got['status'] == 'ok', got
    assert got['unit'] == 'article' and got['n_units'] == 3, got
    assert got['n_seeds'] == 2 and got['ids'] == ['a', 'b', 'c'], got
    assert all(abs(x - y) < 1e-12 for x, y in zip(got['deltas'], [1.0, 2.0, 3.0])), got['deltas']

    no_article = [dict(x) for x in a]
    no_article_b = [dict(x) for x in b]
    for rows in (no_article, no_article_b):
        for cell in rows:
            cell.pop('ppl_per_article')
            cell.pop('article_ids')
    doc = L.paired_unit_deltas(no_article, no_article_b)
    assert doc['status'] == 'ok' and doc['unit'] == 'document', doc
    assert abs(doc['deltas'][0] - 0.5) < 1e-12, doc['deltas']

    for rows in (no_article, no_article_b):
        for cell in rows:
            cell.pop('ppl_per_doc')
            cell.pop('doc_ids')
    row = L.paired_unit_deltas(no_article, no_article_b)
    assert row['status'] == 'ok' and row['unit'] == 'row', row

    bad_ids = [dict(x) for x in a]
    bad_ids[1]['article_ids'] = ['c', 'b', 'a']
    bad = L.paired_unit_deltas(bad_ids, b, unit='article')
    assert bad['status'] == 'not_available', bad
    assert any('identities differ' in v for v in bad['candidates'].values()), bad

    short = [dict(x) for x in a]
    short[1]['ppl_per_article'] = [10.0, 12.0]
    short[1]['article_ids'] = ['a', 'b']
    bad = L.paired_unit_deltas(short, b, unit='article')
    assert bad['status'] == 'not_available', bad

    low = [dict(x) for x in a]
    low[0] = dict(low[0])
    low[0]['ppl_per_seq'] = [0.5, 12.0]
    bad = L.paired_unit_deltas(low, b, unit='row')
    assert bad['status'] == 'not_available', bad
    try:
        L.paired_unit_deltas(a, b, unit='line')
    except ValueError:
        pass
    else:
        raise AssertionError('invalid unit label was accepted')


def test_audit_evaluation_design_and_wiring():
    with tempfile.TemporaryDirectory() as td:
        mp = os.path.join(td, 'manifest.json')
        pp = os.path.join(td, 'power_validation.json')
        manifest = {
            'statistics': {'alpha': 0.05, 'min_power_for_test': 0.80},
            'eval': {'n_seqs_clean': 48},
            'binding': {'test_slice': {'n_seqs': 48}},
            'primary_comparisons': [{'name': 'a'}],
            'evaluation_design': {'n_units': 48, 'unit': 'document'},
        }
        with open(mp, 'w', encoding='utf-8') as f:
            json.dump(manifest, f)
        with open(pp, 'w', encoding='utf-8') as f:
            json.dump({'comparisons': {
                'a': {'unit_deltas': [-0.2] * 8 + [0.05] * 40}}}, f)
        with contextlib.redirect_stdout(io.StringIO()):
            rep = A.audit_evaluation_design(manifest_path=mp, pilot_path=pp)
        assert rep['status'] == 'ok', rep
        assert rep['n_units'] == 48 and rep['pilot_status'] == 'available', rep
        assert rep['population']['claim_allowed'] is False, rep['population']
        cell = rep['comparisons']['a']
        assert cell['status'] == 'ok' and cell['precision']['ci_halfwidth'] > 0, cell

        for _bad_pilot in (
                {'comparisons': {'a': {'unit_deltas': []}}},
                {'comparisons': {'a': {'unit_deltas': [0.1]}}},
                {'comparisons': {'a': {'unit_deltas': [0.1, 0.1]}}}):
            _bad = L.evaluation_design_audit(manifest, _bad_pilot)
            assert _bad['status'] == 'not_verifiable', (_bad_pilot, _bad)

        du = L.evaluation_design_units(manifest)
        assert du['raw_n_units'] == 48 and du['n_units'] == 48, du
        assert du['unit_label'] == 'document' and du['label_declared'] is True, du
        du = L.evaluation_design_units({'eval': {'n_seqs_clean': 32}})
        assert du['raw_n_units'] == 32 and du['n_units'] == 32, du
        assert du['unit_label'] == 'document' and du['label_declared'] is False, du
        du = L.evaluation_design_units({
            'evaluation_design': {'n_units': None, 'unit_count': 40},
            'eval': {'n_seqs_clean': None},
            'binding': {'test_slice': {'n_seqs': 48}}})
        assert du['raw_n_units'] == 40 and du['n_units'] == 40, du
        du = L.evaluation_design_units({
            'eval': {'n_seqs_clean': None},
            'binding': {'test_slice': {'n_seqs': 48}}})
        assert du['raw_n_units'] == 48 and du['n_units'] == 48, du
        du = L.evaluation_design_units(
            {'evaluation_design': {'n_units': 64, 'unit': 'Article'}})
        assert du['raw_n_units'] == 64 and du['unit_label'] == 'article', du
        assert du['label_declared'] is True, du

        with contextlib.redirect_stdout(io.StringIO()):
            rep2 = A.audit_evaluation_design(
                manifest_path=mp, pilot_path=os.path.join(td, 'missing.json'))
        assert rep2['status'] == 'not_verifiable', rep2
        assert rep2['pilot_status'] == 'not_available', rep2
        assert any('no per-unit pilot deltas' in w for w in rep2['warnings']), rep2

        manifest['evaluation_design']['supports_population_claim'] = True
        with open(mp, 'w', encoding='utf-8') as f:
            json.dump(manifest, f)
        with contextlib.redirect_stdout(io.StringIO()):
            rep3 = A.audit_evaluation_design(manifest_path=mp, pilot_path=pp)
        assert rep3['status'] == 'fail', rep3
        assert any('supports_population_claim=true' in v for v in rep3['violations']), rep3

        manifest['evaluation_design']['n_units'] = 64
        manifest['eval']['n_seqs_clean'] = 64
        with open(mp, 'w', encoding='utf-8') as f:
            json.dump(manifest, f)
        with contextlib.redirect_stdout(io.StringIO()):
            rep3b = A.audit_evaluation_design(manifest_path=mp, pilot_path=pp)
        assert rep3b['status'] == 'fail', rep3b
        assert any('no machine-readable probability sampling design' in v
                   for v in rep3b['violations']), rep3b

        manifest['evaluation_design']['test_slice_sampling'] = {'method': 'prefix'}
        with open(mp, 'w', encoding='utf-8') as f:
            json.dump(manifest, f)
        with contextlib.redirect_stdout(io.StringIO()):
            rep4 = A.audit_evaluation_design(manifest_path=mp, pilot_path=pp)
        assert rep4['status'] == 'fail', rep4
        assert any('deterministic prefix' in v for v in rep4['violations']), rep4

        manifest['evaluation_design']['supports_population_claim'] = 'yes'
        with open(mp, 'w', encoding='utf-8') as f:
            json.dump(manifest, f)
        with contextlib.redirect_stdout(io.StringIO()):
            rep5 = A.audit_evaluation_design(manifest_path=mp, pilot_path=pp)
        assert rep5['status'] == 'fail', rep5
        assert any('supports_population_claim must be a boolean' in v
                   for v in rep5['violations']), rep5

    with contextlib.redirect_stdout(io.StringIO()):
        frozen = A.audit_evaluation_design(
            manifest_path=os.path.join(REPO, 'confirmation_frozen.json'),
            pilot_path=os.path.join(REPO, 'analysis_v12', 'does_not_exist.json'))
    assert frozen['n_units'] == 48, frozen
    assert frozen['population']['claim_allowed'] is False, frozen
    assert frozen['status'] == 'not_verifiable', frozen

    import inspect
    assert "'evaluation': audit_evaluation_design()" in inspect.getsource(A.run_report)
    assert 'evaluation' in (A.__doc__ or ''), 'evaluation is not documented in the audit CLI'


def test_power_preflight_uses_frozen_test_units():
    manifest = {
        'frozen_at': 'test', 'code_semantics_expected': L.CODE_SEMANTICS,
        'arms': ['a', 'b'], 'seeds': [0, 1],
        'primary_comparisons': [{
            'name': 'a-b', 'a': 'a', 'b': 'b',
            'endpoint': 'by_len.ppl@512', 'expected_sign': 'positive'}],
        'statistics': {
            'test': 'exact two-sided sign-flip',
            'correction': 'holm step-down', 'alpha': 0.05, 'n_seeds': 2,
            'descriptive_only': True, 'min_power_for_test': 0.8,
            'require_power_gate': False},
        'eval': {'n_seqs_clean': 64},
        'evaluation_design': {'n_units': 64, 'unit': 'document'},
    }
    docs = ['d1', 'd2', 'd3', 'd4']

    def rec(ppl, vals):
        return {'by_len': {'512': {
            'ppl': ppl, 'truncated': False,
            'ppl_per_doc': vals, 'doc_ids': list(docs)}}}

    summary = {
        'a::seed0': rec(2.0, [3.0, 2.0, 2.0, 1.0]),
        'b::seed0': rec(1.0, [1.0, 1.0, 1.0, 1.0]),
        'a::seed1': rec(2.0, [3.0, 2.0, 2.0, 1.0]),
        'b::seed1': rec(1.0, [1.0, 1.0, 1.0, 1.0]),
    }
    with tempfile.TemporaryDirectory() as td:
        mp = os.path.join(td, 'manifest.json')
        sp = os.path.join(td, 'summary.json')
        op = os.path.join(td, 'power.json')
        with open(mp, 'w', encoding='utf-8') as f:
            json.dump(manifest, f)
        with open(sp, 'w', encoding='utf-8') as f:
            json.dump(summary, f)
        _old_current = L.result_is_current
        L.result_is_current = lambda *a, **k: True
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                V12.run_power(manifest_path=mp, summary=sp, out=op)
        finally:
            L.result_is_current = _old_current
        with open(op, encoding='utf-8') as f:
            rep = json.load(f)
    cell = rep['comparisons']['a-b']
    up = cell['unit_precision']
    assert up['status'] == 'ok', up
    assert up['n_units'] == 64 and up['pilot_n_units'] == 4, up
    assert up['unit_label'] == 'document' and up['pilot_unit'] == 'document', up
    assert cell['evaluation_unit']['n_units'] == 4, cell['evaluation_unit']

    import inspect
    assert 'evaluation_design_units' in inspect.getsource(V12.run_power)


def test_directional_claim_verdicts_three_way():
    base = {
        'expected_sign': 'negative', 'p_holm': 0.01, 'direction_ok': True,
        'n_effective': 12, 'min_seeds_for_confirmation': 8,
        'coverage_ok': True, 'param_matched': True,
        'param_match_complete': True, 'same_init_core_fp': True,
        'mechanism_claim_allowed': False,
    }

    def _case(**over):
        c = dict(base)
        c.update(over)
        return c

    comps = {
        'sup': _case(),
        'ref': _case(direction_ok=False),
        'inc': _case(p_holm=0.5),
        'nd': _case(expected_sign=None, direction_ok=None),
        'omit': {'omitted': 'no complete seed pairs'},
        'opp_inc': _case(p_holm=0.4, direction_ok=False),
    }
    frozen = json.loads(json.dumps(comps))
    rep = L.directional_claim_verdicts(
        comps, names=['sup', 'ref', 'inc', 'nd', 'omit', 'opp_inc'],
        alpha=0.05, require_complete_coverage=True,
        require_param_match=True, default_min_seeds=8)
    assert comps == frozen, 'the helper must not mutate its input records'
    assert rep['supported'] == ['sup'], rep['supported']
    assert rep['refuted'] == ['ref'], rep['refuted']
    assert rep['inconclusive'] == ['inc', 'opp_inc'], rep['inconclusive']
    assert rep['not_testable'] == ['omit'], rep['not_testable']
    assert rep['significant_nondirectional'] == ['nd'], rep
    assert rep['opposite_nonsignificant'] == ['opp_inc'], rep
    assert rep['verdicts']['ref']['reason'] == 'holm_significant_and_direction_opposite', rep
    assert rep['verdicts']['ref']['claim_scope'] == 'system_level', rep['verdicts']['ref']
    assert rep['verdicts']['nd']['claim_scope'] == 'nondirectional', rep['verdicts']['nd']
    assert rep['counts']['refuted'] == 1, rep['counts']
    assert 'not equivalence' in rep['rule'], rep['rule']

    # A reverse-significant result that fails a validity gate is NOT a
    # refutation; it stays inconclusive and is surfaced by name so the report
    # can explain why the evidence was not admitted.
    weak = L.directional_claim_verdicts(
        {'ref': _case(direction_ok=False, coverage_ok=False)},
        names=['ref'], alpha=0.05, require_complete_coverage=True,
        require_param_match=True, default_min_seeds=8)
    assert weak['refuted'] == [], weak
    assert weak['reverse_significant_unverified'] == ['ref'], weak
    assert weak['verdicts']['ref']['verdict'] == 'inconclusive', weak
    assert 'coverage_incomplete' in weak['verdicts']['ref']['gate_failures'], weak

    # A definite parameter mismatch always invalidates the claim; a disclosed
    # matching contract additionally requires a complete record (None is not a
    # pass), while legacy manifests without the contract keep the old path.
    _v = L.directional_claim_verdicts(
        {'m': _case(param_matched=False)},
        names=['m'], require_param_match=False, default_min_seeds=8)
    assert _v['supported'] == [] and _v['verdicts']['m']['validity_ok'] is False, _v
    _v = L.directional_claim_verdicts(
        {'m': _case(param_match_complete=None)},
        names=['m'], require_param_match=True, default_min_seeds=8)
    assert _v['supported'] == [], _v
    _v = L.directional_claim_verdicts(
        {'m': _case(param_match_complete=None)},
        names=['m'], require_param_match=False, default_min_seeds=8)
    assert _v['supported'] == ['m'], _v
    _v = L.directional_claim_verdicts(
        {'m': _case(n_effective=3)},
        names=['m'], default_min_seeds=8)
    assert _v['supported'] == [] and any(
        'min_seeds_for_confirmation' in x
        for x in _v['verdicts']['m']['gate_failures']), _v

    # A malformed or negative floor is corrupt evidence, not a licence to
    # fall back to the default (or to compare against a negative bar).  The
    # same fail-closed rule applies to the observed effective n.
    for _bad_min in (-1, 8.5, True, 'banana'):
        _v = L.directional_claim_verdicts(
            {'m': _case(min_seeds_for_confirmation=_bad_min)},
            names=['m'], default_min_seeds=8)
        assert _v['supported'] == [], (_bad_min, _v)
        assert any('n floor is unverifiable' in x
                   for x in _v['verdicts']['m']['gate_failures']), (_bad_min, _v)
    for _bad_n in (-5, 1.5, True, 'banana'):
        _v = L.directional_claim_verdicts(
            {'m': _case(n_effective=_bad_n)},
            names=['m'], default_min_seeds=8)
        assert _v['supported'] == [], (_bad_n, _v)
        assert 'n_effective_unverifiable' in _v['verdicts']['m']['gate_failures'], (_bad_n, _v)

    # p == alpha is not below alpha; two-sided p cannot be a boolean/NaN/out
    # of range.  The helper adds no test, so the boundary must be exact.
    _v = L.directional_claim_verdicts({'m': _case(p_holm=0.05)}, names=['m'])
    assert _v['verdicts']['m']['verdict'] == 'inconclusive', _v

    def _raises(fn):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError('expected ValueError')

    _raises(lambda: L.directional_claim_verdicts({'m': _case(expected_sign='up')}, names=['m']))
    _raises(lambda: L.directional_claim_verdicts({'m': _case(direction_ok=1)}, names=['m']))
    _raises(lambda: L.directional_claim_verdicts({'m': _case(p_holm=True)}, names=['m']))
    _raises(lambda: L.directional_claim_verdicts({'m': _case(p_holm=float('nan'))}, names=['m']))
    _raises(lambda: L.directional_claim_verdicts({'m': _case(p_holm=-0.01)}, names=['m']))
    _raises(lambda: L.directional_claim_verdicts({'m': _case()}, names='m'))
    _raises(lambda: L.directional_claim_verdicts({'m': _case()}, names=['m', 'm']))
    _raises(lambda: L.directional_claim_verdicts({'m': _case()}, names=['m'], alpha=1.0))
    _raises(lambda: L.directional_claim_verdicts({'m': _case()}, names=['m'],
                                                  default_min_seeds=-1))

    import inspect
    src = inspect.getsource(V12.finaltest_stats)
    assert 'directional_claim_verdicts' in src and 'verdict_consistency' in src, src
    assert inspect.getsource(V12._refutation_markdown)


def test_finaltest_verdicts_and_refutation_markdown():
    manifest = {
        'frozen_at': '2026-10-27', 'code_semantics_expected': L.CODE_SEMANTICS,
        'arms': ['a_arm', 'b_arm'], 'seeds': [0, 1, 2, 3, 4, 5],
        'primary_comparisons': [{
            'name': 'c1', 'a': 'a_arm', 'b': 'b_arm',
            'endpoint': 'by_len.ppl@4096', 'expected_sign': 'negative',
            'mechanism_claim_allowed': False}],
        'statistics': {
            'test': 'exact two-sided sign-flip', 'correction': 'holm step-down',
            'descriptive_only': True, 'require_complete_coverage': True,
            'min_seeds_for_confirmation': 2, 'n_seeds': 6},
    }

    def _rec(arm, seed, ppl, msha):
        return {'variant': arm, 'seed': seed, 'split': 'test',
                'data_fp': 'FP', 'by_len': {'4096': {
                    'ppl': ppl, 'n_tok': 1000, 'truncated': False}},
                '_code': L.code_stamp(), 'manifest_sha256': msha,
                'init_core_fp': f'I{seed}',
                'param_breakdown': {'active': 1000}, 'params': 1000}

    with tempfile.TemporaryDirectory() as td:
        mp = os.path.join(td, 'm.json')
        with open(mp, 'w', encoding='utf-8') as f:
            json.dump(manifest, f)
        m = V12._load_manifest(mp)
        outdir = os.path.join(td, 'res')
        os.makedirs(outdir)
        V12._check_or_write_freeze_lock(outdir, m, 'FP', stage='accessed')
        bp = os.path.join(outdir, 'bylen.json')

        def _run(avals, tag):
            raw = {}
            for s, a_ppl in enumerate(avals):
                raw[f'a_arm::s{s}'] = _rec('a_arm', s, a_ppl, m['manifest_sha256'])
                raw[f'b_arm::s{s}'] = _rec('b_arm', s, 12.0, m['manifest_sha256'])
            with open(bp, 'w', encoding='utf-8') as f:
                json.dump(raw, f)
            with contextlib.redirect_stdout(io.StringIO()):
                return V12.finaltest_stats(
                    manifest=m, bylen_path=bp,
                    probe_path=os.path.join(td, 'no_probe.json'),
                    out=os.path.join(td, f'stats_{tag}.json'))

        # a - b is negative: the pre-registered direction is supported.
        st = _run([10.0] * 6, 'sup')
        assert st['confirmed'] == ['c1'], st['confirmed']
        assert st['verdicts']['c1']['verdict'] == 'supported', st['verdicts']
        assert st['verdict_summary']['supported'] == ['c1'], st['verdict_summary']
        assert st['verdict_consistency']['match'] is True, st
        assert V12._refutation_markdown(st, 0.05) == []

        # a - b is positive and Holm-significant: a directional conflict, not
        # "not confirmed" and not an exploratory result.
        st = _run([14.0] * 6, 'ref')
        assert st['confirmed'] == [], st['confirmed']
        assert st['refuted'] == ['c1'], st['refuted']
        assert st['verdicts']['c1']['verdict'] == 'refuted', st['verdicts']
        assert st['verdicts']['c1']['reason'] == 'holm_significant_and_direction_opposite', st
        md = '\n'.join(V12._refutation_markdown(st, 0.05))
        assert '反证' in md and '不是「未确认」' in md, md
        assert 'system_level' in md and 'SESOI' in md, md
        assert st['not_confirmed_reasons']['c1']['verdict'] == 'refuted', st['not_confirmed_reasons']

        # A non-significant reversal is inconclusive and may be called out as
        # an opposite non-significant readout, never as a refutation.
        st = _run([13.0, 11.9, 12.1, 12.1, 12.1, 12.1], 'inc')
        assert st['confirmed'] == [] and st['refuted'] == [], st
        assert st['inconclusive'] == ['c1'], st['inconclusive']
        assert st['verdicts']['c1']['verdict'] == 'inconclusive', st['verdicts']
        assert st['opposite_nonsignificant'] == ['c1'], st
        md = '\n'.join(V12._refutation_markdown(st, 0.05))
        assert '方向相反但未达显著' in md and '不是支持也不是反证' in md, md

    import inspect
    src = inspect.getsource(V12.build_report)
    assert '_refutation_markdown' in src, src


def test_audit_confirmation_verdicts_and_wiring():
    def _rec(sign, dok, p, **over):
        c = {'expected_sign': sign, 'direction_ok': dok,
             'n_effective': 12, 'min_seeds_for_confirmation': 8,
             'coverage_ok': True, 'param_matched': True,
             'param_match_complete': True, 'same_init_core_fp': True,
             'mechanism_claim_allowed': False}
        c.update(over)
        return c

    comps = {'c1': _rec('negative', True, 0.01),
             'c2': _rec('negative', False, 0.02),
             'c3': _rec('negative', True, 0.5),
             'c4': {'omitted': 'no complete pairs'},
             # Legacy ``significant_nondirectional`` keeps a Holm-significant
             # record whose direction is unverifiable even when a validity
             # gate failed; the helper classifies it as inconclusive.
             'c5': _rec('negative', None, 0.03, coverage_ok=False)}
    holm = {k: {'p_holm': {'c1': 0.01, 'c2': 0.02, 'c3': 0.5,
                           'c4': 1.0, 'c5': 0.03}[k]}
            for k in comps}
    rep = L.directional_claim_verdicts(
        {k: dict(v, p_holm=holm[k]['p_holm']) for k, v in comps.items()},
        names=list(comps), require_complete_coverage=True,
        require_param_match=True, default_min_seeds=12)
    ft = {'alpha': 0.05, 'comparisons': comps,
          'holm': holm,
          'manifest': {'seeds': list(range(12))},
          'coverage_requirement': {'require_complete_coverage': True},
          'parameter_binding': {'required_by_manifest': True},
          'verdicts': rep['verdicts'],
          'confirmed': rep['supported'], 'refuted': rep['refuted'],
          'inconclusive': rep['inconclusive'],
          'not_testable': rep['not_testable'],
          'significant_nondirectional': ['c5'],
          'opposite_nonsignificant': rep['opposite_nonsignificant'],
          'reverse_significant_unverified': rep['reverse_significant_unverified']}
    # The omitted comparison supplies no p_holm; the explicit holm map keeps
    # the family slot at p=1.0, exactly as finaltest_stats does.
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'test_stats.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'finaltest': ft}, f)
        with contextlib.redirect_stdout(io.StringIO()):
            got = A.audit_confirmation_verdicts(p)
        assert got['status'] == 'warnings', got
        assert got['n_supported'] == 1 and got['n_refuted'] == 1, got
        assert got['n_inconclusive'] == 2 and got['n_not_testable'] == 1, got
        assert any('refuted' in w and 'not "not confirmed"' in w
                   for w in got['warnings']), got['warnings']

        bad = json.loads(json.dumps({'finaltest': ft}))
        bad['finaltest']['refuted'] = []
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(bad, f)
        with contextlib.redirect_stdout(io.StringIO()):
            got = A.audit_confirmation_verdicts(p)
        assert got['status'] == 'fail', got
        assert any("top-level 'refuted'" in x for x in got['problems']), got

        tampered = json.loads(json.dumps({'finaltest': ft}))
        tampered['finaltest']['verdicts']['c2']['verdict'] = 'supported'
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(tampered, f)
        with contextlib.redirect_stdout(io.StringIO()):
            got = A.audit_confirmation_verdicts(p)
        assert got['status'] == 'fail', got
        assert any('c2' in x and 'supported' in x for x in got['problems']), got

        legacy = json.loads(json.dumps({'finaltest': ft}))
        legacy['finaltest'].pop('verdicts')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(legacy, f)
        with contextlib.redirect_stdout(io.StringIO()):
            got = A.audit_confirmation_verdicts(p)
        assert got['status'] == 'not_verifiable_locally', got
        assert 'predates' in got['reason'], got

    with contextlib.redirect_stdout(io.StringIO()):
        got = A.audit_confirmation_verdicts(os.path.join(tempfile.gettempdir(),
                                                        'no_such_stats.json'))
    assert got['status'] == 'not_verifiable_locally', got

    import inspect
    assert 'audit_confirmation_verdicts()' in inspect.getsource(A.run_report)
    assert "'verdicts': audit_confirmation_verdicts" in inspect.getsource(A)


def test_error_bucket_inference_split_half_and_audit():
    # Synthetic clusters: bucket y is worse in every row, bucket x only at a
    # few positions.  The discovery/replication split must choose y without
    # using the replication half for selection, and the replay gate must detect
    # a tampered verdict.
    rows, buckets, d = [], [], []
    for r in range(16):
        for j in range(8):
            rows.append(r)
            buckets.append('y' if j else 'x')
            d.append(1.0 if j else 0.5)
    rep = L.paired_error_bucket_inference(d, buckets, rows, min_clusters=3,
                                          min_tokens=5, split_seed=7)
    assert rep['status'] == 'ok', rep
    assert rep['selected']['bucket'] == 'y', rep['selected']
    assert rep['selected']['same_sign'] is True, rep['selected']
    assert rep['selected']['replicated'] is True, rep['selected']
    assert rep['selected']['full_sample_p_holm'] < 0.01, rep['selected']
    split = rep['split']
    assert split['disjoint'] is True, split
    assert not (set(split['discovery_clusters']) & set(split['replication_clusters']))
    assert len(split['discovery_clusters']) + len(split['replication_clusters']) == 16
    # The full-sample Holm path is present for every bucket, and the selected
    # bucket is marked post-hoc rather than promoted to confirmation.
    for b, rec in rep['buckets'].items():
        assert rec['p_holm'] is not None, (b, rec)
        assert rec['role'] == 'post_hoc_full_sample_descriptive', rec
    again = L.paired_error_bucket_inference(d, buckets, rows, min_clusters=3,
                                            min_tokens=5, split_seed=7)
    assert again['split'] == split, 'split must be deterministic'
    # The declared seed must drive the SHA-256 cluster order, not just be
    # echoed in the artifact.  Otherwise a tampered seed would be unauditable.
    other_orders = {
        tuple(L.paired_error_bucket_inference(
            d, buckets, rows, min_clusters=3, min_tokens=5,
            split_seed=_seed)['split']['cluster_order'])
        for _seed in (8, 9, 10)
    }
    assert tuple(split['cluster_order']) not in other_orders, (
        'different split seeds must reshuffle the cluster order')
    # No positive discovery candidate is a different state from "no data".
    neg = L.paired_error_bucket_inference([-x for x in d], buckets, rows,
                                          min_clusters=3, min_tokens=5,
                                          split_seed=7)
    assert neg['status'] == 'no_candidate', neg['status']
    empty = L.paired_error_bucket_inference([], [], [])
    assert empty['status'] == 'no_data', empty
    # Fail-closed validation.
    for args, kw in ((([1.0, 2.0], ['a'], [0, 1]), {}),
                     (([True], ['a'], [0]), {}),
                     (([1.0], ['a'], [0]), {'min_clusters': 1}),
                     (([1.0], ['a'], [0]), {'min_tokens': 0})):
        try:
            L.paired_error_bucket_inference(*args, **kw)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f'inference accepted invalid input {args!r} {kw!r}')

    # Replay gate accepts the authentic artifact and rejects a flipped verdict.
    td = tempfile.mkdtemp(prefix='bucket_inf_', dir=REPO)
    art = os.path.join(td, 'paired_errors.json')
    try:
        rep['family'] = 'synthetic'
        with open(art, 'w', encoding='utf-8') as f:
            json.dump({'kind': 'paired_token_error_comparison',
                       'bucket_inference': [rep]}, f)
        aud = A.audit_error_case_inference(art)
        assert aud['status'] == 'ok', aud
        bad_seed = json.loads(json.dumps(rep))
        bad_seed['split_seed'] += 1
        with open(art, 'w', encoding='utf-8') as f:
            json.dump({'bucket_inference': [bad_seed]}, f)
        aud_seed = A.audit_error_case_inference(art)
        assert aud_seed['status'] == 'fail', aud_seed
        assert any('seed' in x for x in aud_seed['problems']), aud_seed
        # A no_data family is auditable only when it really is empty; a
        # fabricated token count must not turn "no data" into a passing claim.
        empty_fam = json.loads(json.dumps(empty))
        empty_fam['family'] = 'empty'
        with open(art, 'w', encoding='utf-8') as f:
            json.dump({'bucket_inference': [empty_fam]}, f)
        aud_empty = A.audit_error_case_inference(art)
        assert aud_empty['status'] == 'ok', aud_empty
        bad_empty = json.loads(json.dumps(empty_fam))
        bad_empty['n_tokens'] = 1
        with open(art, 'w', encoding='utf-8') as f:
            json.dump({'bucket_inference': [bad_empty]}, f)
        aud_bad_empty = A.audit_error_case_inference(art)
        assert aud_bad_empty['status'] == 'fail', aud_bad_empty
        assert any('zero finite' in x for x in aud_bad_empty['problems']), aud_bad_empty
        sparse_rows, sparse_buckets, sparse_d = [], [], []
        for _c in ('c2', 'c5', 'c3', 'c6'):
            for _ in range(2):
                sparse_rows.append(_c)
                sparse_buckets.append('y')
                sparse_d.append(1.0)
        for _c in ('c4', 'c7', 'c1', 'c0'):
            for _ in range(2):
                sparse_rows.append(_c)
                sparse_buckets.append('x')
                sparse_d.append(-0.5)
        sparse = L.paired_error_bucket_inference(
            sparse_d, sparse_buckets, sparse_rows, min_clusters=2,
            min_tokens=1, split_seed=7)
        assert sparse['status'] == 'ok', sparse
        assert sparse['selected']['bucket'] == 'y', sparse['selected']
        assert sparse['buckets']['y']['replication']['mean'] is None, sparse
        assert sparse['buckets']['x']['discovery']['mean'] is None, sparse
        with open(art, 'w', encoding='utf-8') as f:
            json.dump({'bucket_inference': [sparse]}, f)
        aud_sparse = A.audit_error_case_inference(art)
        assert aud_sparse['status'] == 'ok', aud_sparse
        bad = json.loads(json.dumps(rep))
        bad['selected']['replicated'] = not bad['selected']['replicated']
        with open(art, 'w', encoding='utf-8') as f:
            json.dump({'bucket_inference': [bad]}, f)
        aud2 = A.audit_error_case_inference(art)
        assert aud2['status'] == 'fail', aud2
        assert any('replicated' in x for x in aud2['problems']), aud2
        missing = A.audit_error_case_inference(os.path.join(td, 'absent.json'))
        assert missing['status'] == 'not_verifiable', missing
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_compare_token_errors_reports_bucket_inference_and_markdown():
    import torch.nn.functional as _F

    class _M:
        def __init__(self, correct_logit):
            self.correct_logit = float(correct_logit)
            self.training = True

        def eval(self):
            self.training = False
            return self

        def train(self, mode=True):
            self.training = bool(mode)
            return self

        def trunk(self, ids):
            return ids.unsqueeze(-1).float()

        def head(self, h):
            ids = h[..., 0].long()
            lg = torch.zeros((ids.shape[0], ids.shape[1], 32), dtype=torch.float32)
            tgt = ((ids + 1) % 32).unsqueeze(-1)
            return lg.scatter(2, tgt, self.correct_logit)

    vb = (np.arange(20 * 8, dtype=np.int64).reshape(20, 8) % 32)
    a, b = _M(7.0), _M(3.0)
    train_ids = np.arange(320, dtype=np.int64) % 32
    bnd = np.zeros_like(vb, dtype=bool)
    bnd[:, 1::2] = True
    r = L.compare_token_errors(a, b, vb, 32, torch.device('cpu'), top_n=0,
                               train_ids=train_ids, val_bnd=bnd)
    fams = {f['family']: f for f in (r.get('bucket_inference') or [])}
    assert {'position', 'target_rarity', 'boundary'} <= set(fams), fams
    for fam in fams.values():
        assert fam['status'] in ('ok', 'no_candidate', 'insufficient_clusters'), fam
    pos = fams['position']
    assert pos['status'] == 'ok', pos
    assert pos['selected'] is not None and pos['selected']['replicated'] is True, pos['selected']
    assert pos['selected']['bucket'].startswith('position:'), pos['selected']
    # The split is row-level and disjoint: no token row leaks across halves.
    assert pos['split']['disjoint'] is True, pos['split']
    for bname, rec in pos['buckets'].items():
        for u in rec.get('cluster_units') or []:
            assert (u['cluster'] in pos['split']['discovery_clusters']) != (
                u['cluster'] in pos['split']['replication_clusters']), (bname, u)
    # Auxiliary families fail soft (recorded reason) rather than aborting the
    # paired comparison when their inputs are malformed.
    r2 = L.compare_token_errors(a, b, vb, 32, torch.device('cpu'), top_n=0,
                                train_ids=[], val_bnd=np.zeros((3, 3), dtype=bool))
    assert 'target_rarity' in r2['bucket_family_reasons'], r2['bucket_family_reasons']
    assert 'boundary' in r2['bucket_family_reasons'], r2['bucket_family_reasons']
    assert r2['bucket_inference'], r2
    r3 = L.compare_token_errors(a, b, vb, 32, torch.device('cpu'), top_n=0,
                                bucket_audit=False)
    assert r3['bucket_inference'] is None, r3
    # Artifact renderers expose the stability claim and its boundary.
    paired = {'bucket_inference': [pos],
              'bucket_family_reasons': {'target_rarity': 'unavailable: no train_ids'}}
    ea_txt = '\n'.join(EA._render_paired_bucket_inference(paired))
    assert 'replicated=True' in ea_txt and 'not external replication' in ea_txt, ea_txt
    br_txt = '\n'.join(BR._render_paired_error_inference(paired))
    assert 'replicated=True' in br_txt and 'post-hoc' in br_txt, br_txt

