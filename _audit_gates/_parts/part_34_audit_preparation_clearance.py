# The audit CLI help text lives in part_00_load.py and must remain the
# package __doc__; this fragment deliberately has no module docstring.
#
# Launch gate for the prospective preparation clearance.
#
# A design protocol says what the study intends to do.  A clearance says the
# machinery needed to do it was ready before the first cell: a complete
# randomized allocation, the registered disposition policy, a passed pre-test
# power artifact, the same analysis closure, a passed pre-run rehearsal, an
# unread held-out split and a pinned dependency stack.  This gate replays the
# stored clearance, checks its protocol binding, and orders it against the
# first event.  Missing evidence is `not_verifiable_locally`, never a pass.
#
# It creates no test, p-value, Holm slot or confirmation verdict and reads no
# measurement data.


import json


def audit_preparation_clearance(path=None, protocol_path=None, verbose=True,
                                events=None, panel_dirs=None):
    """Replay the prospective preparation clearance for one study."""
    L = _import_exp_lib()
    rep = L.preparation_clearance_audit(
        path=path, protocol_path=protocol_path, events=events,
        panel_dirs=panel_dirs, verbose=False)
    if verbose:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    return rep
