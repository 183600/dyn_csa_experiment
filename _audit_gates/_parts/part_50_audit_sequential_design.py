# The audit CLI help text lives in part_00_load.py and must remain the
# package __doc__; this fragment deliberately has no module docstring.
#
# Design/preparation gate for a prospective group-sequential plan.
#
# A study that may add seeds after an interim look must say so *before* the
# first look.  This gate replays the hash-bound sequential_design.json: the
# declared look schedule, the alpha-spending rule, the exact integer
# boundaries, the exact FWER and the sign-test power.  It also checks that the
# plan precedes the first recorded look/event, that its n_max agrees with the
# frozen manifest, that the protocol's stopping rule is not contradicted, and
# that every observed look is on the registered schedule.
#
# It adds no test, p-value, Holm slot or confirmation verdict and reads no
# measurement data.


import json


def audit_sequential_design(path=None, manifest_path=None, protocol_path=None,
                            look_history_path=None, events_path=None,
                            verbose=True):
    """Replay a prospective group-sequential plan and its timing."""
    L = _import_exp_lib()  # noqa: F821
    rep = L.sequential_design_audit(
        path=path, manifest_path=manifest_path, protocol_path=protocol_path,
        look_history_path=look_history_path, events_path=events_path,
        verbose=False)
    if verbose:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    return rep


def audit_sequential_design_artifact(path=None, *, verbose=True):
    """Single-artifact replay alias for :func:`audit_sequential_design`."""
    return audit_sequential_design(path=path, verbose=verbose)
