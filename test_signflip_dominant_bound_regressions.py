"""Exact-dominant sign-flip panels beyond the meet-in-the-middle size cap.

A scale-up seed panel can pass the 46-seed exact-MITM limit while still being
analytically simple: one delta outweighs the sum of all the others.  The
saddlepoint tail is not reliable there (it understates p), so the exact
closed-form "every sign assignment is at least as extreme" condition must be
checked before falling back to it.
"""

import exp_lib as L


def test_dominant_panel_above_mitm_cutoff_is_exact_one():
    # At n=47 the exact meet-in-the-middle table no longer fits the declared
    # cap.  The smallest attainable |sum| equals the observed |sum| here, so
    # every one of the 2**47 sign assignments is at least as extreme and the
    # exact two-sided p is 1.  The saddlepoint branch used to report ~0.69.
    for n in (47, 48, 60):
        out = L.exact_signflip_p([1.0] + [-0.01] * (n - 1))
        assert out['p'] == 1.0, (n, out)
        assert out['method'] == 'exact-mitm-dominant', (n, out)


def test_balanced_panel_is_not_forced_to_one_by_the_bound():
    # The closed-form condition is a lower bound on every attainable |sum|;
    # a genuinely balanced panel does not satisfy it and must still go through
    # the approximation branch rather than being labelled exact p=1.
    out = L.exact_signflip_p([1.0] + [-0.03] * 46)
    assert out['method'] == 'saddlepoint', out
    assert 0.0 < out['p'] < 1.0, out
