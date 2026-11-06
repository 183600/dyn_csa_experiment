"""Zero seed columns are sign-neutral and must not trigger a scale branch.

A future panel can be stored with one row per planned seed, including seeds
whose paired delta is exactly zero (a tie, or a padded seed slot).  For the
exact sign-flip family tests, such a column cannot change any signed sum: it
only duplicates every sign pattern.  Counting it as an informative seed made
the exact-TOST/MITM, max-T, curve-level and sequential tests choose an
approximation (or refuse the panel) even though only a few columns carried
information.  These CPU-only checks pin the effective-width branch.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

import pytest

ACTIVE = [-0.20, 0.15, -0.10, 0.25, -0.05, 0.12,
          -0.18, 0.22, -0.08, 0.30, -0.14, 0.18]
PAD = 20

# n=32 > the 20/40 exact caps, but only 12 columns are non-zero.
DELTAS = ACTIVE + [0.0] * PAD


def test_exact_tost_uses_the_nonzero_seed_width():
    base = L._etost_upper_tail(ACTIVE)
    padded = L._etost_upper_tail(DELTAS)
    assert base['p'] == pytest.approx(padded['p'])
    assert base['method'] == padded['method'] == 'exact'
    # n stays the declared count; the floor is still controlled by the
    # non-zero support, which the two vectors share.
    assert padded['n'] == len(DELTAS)
    assert padded['n_nonzero'] == len(ACTIVE)
    assert padded['p_floor'] == base['p_floor']


def test_max_t_zero_columns_do_not_count_toward_the_cap():
    rows = {
        'a': ACTIVE + [0.0] * PAD,
        'b': [0.10 + 0.01 * i for i in range(len(ACTIVE))] + [0.0] * PAD,
    }
    padded = L.maxt_signflip_adjust(rows)
    base = L.maxt_signflip_adjust({k: v[:len(ACTIVE)] for k, v in rows.items()})
    assert padded['n_seeds'] == len(DELTAS)
    assert padded['n_seeds_effective'] == len(ACTIVE)
    assert padded['n_patterns'] == 1 << len(ACTIVE)
    assert padded['p_floor_single'] == base['p_floor_single']
    for name in rows:
        assert padded['contrasts'][name]['p_maxt'] == pytest.approx(
            base['contrasts'][name]['p_maxt'])
        assert padded['contrasts'][name]['p_raw_signflip'] == pytest.approx(
            base['contrasts'][name]['p_raw_signflip'])
        assert padded['contrasts'][name]['mean'] == pytest.approx(
            base['contrasts'][name]['mean'] * len(ACTIVE) / len(DELTAS))


def test_curve_exact_path_ignores_zero_padded_seeds_without_a_margin():
    rows = {
        'a': ACTIVE + [0.0] * PAD,
        'b': [0.10 + 0.01 * i for i in range(len(ACTIVE))] + [0.0] * PAD,
    }
    seeds = list(range(len(DELTAS)))
    padded = L.curve_simultaneous_inference(
        list(rows), rows, seeds=seeds, min_n=3)
    base = L.curve_simultaneous_inference(
        list(rows), {k: v[:len(ACTIVE)] for k, v in rows.items()},
        seeds=list(range(len(ACTIVE))), min_n=3)
    assert padded['status'] in ('ok', 'warnings')
    assert padded['method'] == 'exact_signflip_maxt_iut'
    assert padded['family']['n_patterns'] == 1 << len(ACTIVE)
    for name in rows:
        for key in ('p_pointwise_lower', 'p_pointwise_upper',
                    'p_pointwise_two_sided'):
            assert padded['per_condition'][name][key] == pytest.approx(
                base['per_condition'][name][key])
    assert padded['flatness']['p_range'] == pytest.approx(
        base['flatness']['p_range'])


def test_sequential_exact_enumeration_uses_nonzero_union_seeds():
    d1 = ACTIVE[:10]
    d2 = ACTIVE + [0.0] * PAD
    padded = L.sequential_signflip_p([
        (list(range(10)), d1),
        (list(range(len(DELTAS))), d2),
    ])
    base = L.sequential_signflip_p([
        (list(range(10)), d1),
        (list(range(len(ACTIVE))), ACTIVE),
    ])
    assert padded['method'] == base['method'] == (
        'exact-max-over-looks-final-direction')
    assert padded['n_union_seeds'] == len(DELTAS)
    assert padded['n_effective_seeds'] == len(ACTIVE)
    assert padded['p_sequential'] == pytest.approx(base['p_sequential'])


def test_confirmation_family_zero_columns_do_not_count_toward_the_cap():
    seeds = list(range(len(DELTAS)))
    rows = {
        'a': ACTIVE + [0.0] * PAD,
        'b': [0.10 + 0.01 * i for i in range(len(ACTIVE))] + [0.0] * PAD,
    }
    comps = {
        name: {'per_seed': vals, 'seeds': seeds,
               'p_exact_signflip': L.exact_signflip_p(vals)['p']}
        for name, vals in rows.items()
    }
    padded = L.confirmation_family_correction(comps, family_names=list(rows))
    base_comps = {
        name: {'per_seed': vals[:len(ACTIVE)],
               'seeds': list(range(len(ACTIVE))),
               'p_exact_signflip': L.exact_signflip_p(vals[:len(ACTIVE)])['p']}
        for name, vals in rows.items()
    }
    base = L.confirmation_family_correction(
        base_comps, family_names=list(rows))
    assert padded['status'] == base['status'] == 'ok'
    assert padded['n_seeds'] == len(DELTAS)
    assert padded['n_seeds_effective'] == len(ACTIVE)
    assert padded['n_patterns'] == 1 << len(ACTIVE)
    for name in rows:
        assert padded['comparisons'][name]['p_maxt_score'] == pytest.approx(
            base['comparisons'][name]['p_maxt_score'])
