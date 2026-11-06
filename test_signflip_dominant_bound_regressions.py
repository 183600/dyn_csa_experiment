"""Exact-dominant sign-flip panels beyond the meet-in-the-middle size cap.

A scale-up seed panel can pass the 46-seed exact-MITM limit while still being
analytically simple: one delta outweighs the sum of all the others.  The
saddlepoint tail is not reliable there (it understates p), so the exact
closed-form "every sign assignment is at least as extreme" condition must be
checked before falling back to it.
"""

import itertools

import exp_lib as L
import build_report as BR


def test_dominant_panel_above_mitm_cutoff_is_exact_one():
    # At n=47 the exact meet-in-the-middle table no longer fits the declared
    # cap.  The smallest attainable |sum| equals the observed |sum| here, so
    # every one of the 2**47 sign assignments is at least as extreme and the
    # exact two-sided p is 1.  The saddlepoint branch used to report ~0.69.
    for n in (47, 48, 60):
        out = L.exact_signflip_p([1.0] + [-0.01] * (n - 1))
        assert out['p'] == 1.0, (n, out)
        assert out['method'] == 'exact-mitm-dominant', (n, out)


def test_report_builder_uses_the_same_dominant_branches():
    # build_report keeps a torch-free mirror of exact_signflip_p.  When only
    # the audit copy learned about concentrated panels, the zero-GPU report
    # printed the anti-conservative saddlepoint p (~0.66-0.69) for exactly the
    # panels where the exact test is 1.  Mirror both the 41..46 exact-MITM
    # route and the >46 closed-form route.
    for n in (41, 45, 46, 47, 60):
        deltas = [1.0] + [-0.01] * (n - 1)
        audit = L.exact_signflip_p(deltas)
        report = BR.exact_signflip(deltas)
        assert report['p_exact_signflip'] == audit['p'], (n, report, audit)
        assert report['method'] == audit['method'], (n, report, audit)
        assert report['p_exact_signflip'] == 1.0, (n, report)
    # A balanced panel above the cap must still reach the saddlepoint branch
    # in both implementations, not be forced to 1 by the lower bound.
    balanced = [1.0] + [-0.03] * 46
    audit = L.exact_signflip_p(balanced)
    report = BR.exact_signflip(balanced)
    assert audit['method'] == report['method'] == 'saddlepoint'
    assert report['p_exact_signflip'] == audit['p'], (report, audit)


def test_balanced_panel_is_not_forced_to_one_by_the_bound():
    # The closed-form condition is a lower bound on every attainable |sum|;
    # a genuinely balanced panel does not satisfy it and must still go through
    # the approximation branch rather than being labelled exact p=1.
    out = L.exact_signflip_p([1.0] + [-0.03] * 46)
    assert out['method'] == 'saddlepoint', out
    assert 0.0 < out['p'] < 1.0, out


def test_zero_deltas_do_not_force_an_exact_panel_onto_saddlepoint():
    # Zero deltas are sign-neutral: they leave p unchanged and only lower the
    # resolution floor.  Branch selection used the raw count n, so 44 exact
    # zeros pushed a 3-effect panel to the n>46 saddlepoint tail.  That
    # returned 0.5189 while the exact two-sided p over the three non-zero
    # effects is 0.75.
    deltas = [1.0, -0.2, 0.1] + [0.0] * 44
    out = L.exact_signflip_p(deltas)
    assert out['n'] == 47 and out['n_nonzero'] == 3, out
    assert out['method'] == 'exact', out
    observed = abs(sum(deltas))
    count = 0
    for signs in itertools.product((-1.0, 1.0), repeat=3):
        total = sum(sign * value for sign, value in zip(signs, deltas[:3]))
        count += int(abs(total) >= observed - 1e-12)
    assert out['p'] == count / 8.0 == 0.75, out
    assert out['p_floor'] == 0.25, out
    # The torch-free report mirror must select the same branch and p.
    report = BR.exact_signflip(deltas)
    assert report['p_exact_signflip'] == out['p'], (report, out)
    assert report['method'] == out['method'], (report, out)
    # An all-zero panel keeps its historical branch label (the visible suite
    # pins exact-mitm at n=24) while still returning p=1.
    zero = L.exact_signflip_p([0.0] * 24)
    assert zero['p'] == 1.0 and zero['method'] == 'exact-mitm', zero
    assert BR.exact_signflip([0.0] * 24)['method'] == 'exact-mitm'
