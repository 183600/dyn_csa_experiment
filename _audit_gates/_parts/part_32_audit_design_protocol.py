# The audit CLI help text lives in part_00_load.py and must remain the
# package __doc__; this fragment deliberately has no module docstring.
#
# Gate for the prospective design protocol (one pre-run registration).
#
# The result-level audits assume the design was fixed before the run.  This
# gate checks the object that records that fact: a hash-bound
# ``design_protocol.json`` compiled from the frozen manifest, the run-order
# plan, the failed-cell disposition policy and the held-out access policy.  It
# fails on a tampered body, a manifest mismatch, a missing confirmatory design
# decision, or a protocol written after the first recorded event.  No protocol
# is ``not_verifiable_locally`` -- an absence is not a pass.
#
# The gate adds no test, p-value, Holm slot or run; it only reads JSON.


import json


def audit_design_protocol(path=None, manifest_path=None, verbose=True):
    """Replay the prospective design registration for one study.

    ``path`` may name a protocol; otherwise the default locations are tried
    (repository root, ``analysis_v12/`` and ``results_*/``).  ``manifest_path``
    selects the freeze to bind against; it defaults to the fixed-chunk
    ``confirmation_frozen.json``.  Missing artifacts are reported honestly;
    the caller decides whether an unregistered design blocks a run.
    """
    L = _import_exp_lib()
    rep = L.design_protocol_audit(path=path, manifest_path=manifest_path,
                                  verbose=False)
    if verbose:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    return rep
