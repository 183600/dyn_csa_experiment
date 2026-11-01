# Audit gate: validation -> test transfer / shrinkage of the pre-test effect.
#
# The transfer block itself lives in ``explib`` (part_71) so the confirmation
# analysis can build it next to the frozen final-test statistics.  This gate
# replays every derived number from the stored sufficient statistics (seeds and
# the two per-seed log-ratio vectors) and refuses a block whose means,
# intervals, exact p-values, family Holm values or classification labels do not
# reproduce.  Missing evidence is ``not_verifiable``, never green.

import json
import os


# Shared replay loader (part_00): unreadable evidence is "no evidence".
_vt_gate_load = _gate_load


def audit_validation_transfer(path=None):
    """Replay the validation-to-test transfer block of the v12 analysis.

    This is a descriptive transportability/shrinkage readout of the frozen
    comparisons: for each seed it compares the validation log PPL ratio with
    the test log PPL ratio on the same checkpoint.  The gate does not add a
    confirmatory test, does not read a split, and never changes a frozen
    p/Holm/verified/confirmed value.  A block that is present but whose stored
    statistics do not replay is a ``fail``; a missing block or a plan without
    the pre-test validation artifact is ``not_verifiable_locally``.
    """
    L = _import_exp_lib()  # noqa: F821
    p = path or os.path.join(REPO, 'analysis_v12', 'stats.json')  # noqa: F821
    out = {'kind': 'validation_transfer_audit', 'status': 'not_verifiable_locally',
           'path': p, 'n_comparisons': 0, 'n_ok': 0, 'n_not_estimable': 0,
           'class_counts': {}, 'n_opposite_sign': 0, 'n_attenuated': 0,
           'n_amplified': 0, 'n_not_distinguishable': 0,
           'family_holm_rejections': [], 'problems': [], 'warnings': [],
           'frozen_decision_changed': False}
    if not os.path.exists(p):
        out['reason'] = 'no analysis_v12/stats.json on disk'
        return out
    data = _vt_gate_load(p)
    if not isinstance(data, dict):
        out['status'] = 'fail'
        out['problems'].append(f'{p} exists but is not a readable JSON object')
        return out
    block = data.get('validation_transfer')
    if not isinstance(block, dict):
        # Older analyses predate the block.  Absence is not a failed transfer;
        # it means the local artifact cannot answer the question.
        out['reason'] = 'the stats artifact carries no validation_transfer block'
        return out
    out['block_status'] = block.get('status')
    if block.get('status') == 'not_estimable' and not block.get('family'):
        # The analysis writes a minimal placeholder when the pre-test power
        # artifact is missing.  It carries no family to replay; it is missing
        # evidence, not a failed transfer.
        out['status'] = 'not_verifiable_locally'
        out['reason'] = (block.get('reason')
                         or 'the transfer block has no comparison family')
        return out
    _family = block.get('family')
    out['n_comparisons'] = (len(_family)
                            if isinstance(_family, (list, tuple)) else 0)
    out['n_ok'] = block.get('n_ok')
    out['n_not_estimable'] = block.get('n_not_estimable')
    summary = block.get('summary') if isinstance(block.get('summary'), dict) else {}
    _classes = summary.get('class_counts')
    out['class_counts'] = (dict(_classes)
                           if isinstance(_classes, dict) else {})
    out['n_opposite_sign'] = summary.get('n_opposite_sign', 0)
    out['n_attenuated'] = summary.get('n_attenuated', 0)
    out['n_amplified'] = summary.get('n_amplified', 0)
    out['n_not_distinguishable'] = summary.get('n_not_distinguishable', 0)
    out['n_incomplete_coverage'] = summary.get('n_incomplete_coverage', 0)
    if out['n_incomplete_coverage']:
        out['warnings'].append(
            f"{out['n_incomplete_coverage']} comparison(s) use only the seeds "
            'present on both validation and test; the missing seeds are named '
            'inside the block')
    _rejections = summary.get('family_holm_rejections')
    out['family_holm_rejections'] = (list(_rejections)
                                      if isinstance(_rejections, (list, tuple))
                                      else [])
    manifest = None
    manifest_path = os.path.join(REPO, 'confirmation_frozen.json')  # noqa: F821
    if os.path.exists(manifest_path):
        manifest = _vt_gate_load(manifest_path)
    rep = L.replay_validation_transfer(block, manifest=manifest)
    out['problems'].extend(rep.get('problems') or [])
    out['warnings'].extend(rep.get('warnings') or [])
    if rep.get('status') == 'fail':
        out['status'] = 'fail'
        return out
    status = block.get('status')
    if status == 'not_estimable':
        out['status'] = 'not_verifiable_locally'
        out['reason'] = block.get('reason') or 'no comparison has a complete validation/test cell'
    elif status == 'partial':
        out['status'] = 'warn'
        out['reason'] = (block.get('reason')
                         or 'some frozen comparisons have no complete validation/test cell')
    elif status == 'ok':
        out['status'] = 'warn' if out['n_incomplete_coverage'] else 'ok'
    else:
        out['status'] = 'fail'
        out['problems'].append(f'unknown transfer status {status!r}')
    out['note'] = ('descriptive validation-to-test transport only; not in the '
                   'frozen Holm family and never a new confirmation test')
    return out
