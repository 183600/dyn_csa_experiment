"""Split implementation package for audit_gates.py.

Fragments are executed in ``_PARTS`` order, which is dependency order rather
than numeric filename order.  Keep a new fragment in its required execution
position instead of re-sorting the list.
"""

import os as _os

from _split_loader import load_parts as _load_parts

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.dirname(_HERE)
__file__ = _os.path.join(_REPO, 'audit_gates.py')

_PARTS = [
'part_00_load.py',
'part_01_audit_split_discipline.py',
'part_02_audit_sample_design.py',
'part_03_audit_methodology.py',
'part_04_audit_selection.py',
'part_06_audit_data_processing.py',
'part_07_reproduction.py',
'part_05_audit_confirmation_verdicts.py',
'part_08_audit_missingness.py',
'part_09_audit_trajectory.py',
'part_10_audit_disposition.py',
'part_11_audit_test_access.py',
'part_12_audit_arm_ranking.py',
'part_13_audit_arm_rank_units.py',
'part_14_audit_winner_curse.py',
'part_15_audit_effect_sensitivity.py',
'part_16_audit_length_position.py',
]

_load_parts(globals(), _HERE, _PARTS)
del _load_parts

# Keep module introspection on the implementation fragment (the historical
# script body lived in one file) while REPO still points at the repository root.
__file__ = _os.path.join(
    _HERE, '_parts', 'part_05_audit_confirmation_verdicts.py')
REPO = _REPO

if __name__ == '__main__':
    _run_cli()
