"""Split implementation package for audit_gates.py.

Fragments are executed in ``_PARTS`` order, which is dependency order rather
than numeric filename order.  Keep a new fragment in its required execution
position instead of re-sorting the list.
"""

import os as _os

from _split_loader import load_root_package

_PARTS = [
'part_00_load.py',
'part_01_audit_split_discipline.py',
'part_02_audit_sample_design.py',
'part_03_audit_methodology.py',
'part_04_audit_selection.py',
'part_06_audit_data_processing.py',
'part_07_reproduction.py',
'part_29_audit_panel_synthesis.py',
'part_30_audit_panel_scale_sensitivity.py',
'part_31_audit_contrast_network.py',
'part_32_audit_design_protocol.py',
'part_33_split_registry.py',
'part_34_audit_preparation_clearance.py',
'part_35_audit_cleaning_sensitivity.py',
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
'part_17_audit_practical.py',
'part_18_audit_family_correction.py',
'part_19_audit_environment_identity.py',
'part_20_audit_length_strata.py',
'part_21_audit_unit_quantiles.py',
'part_22_audit_endpoint_sensitivity.py',
'part_23_audit_order_dependence.py',
'part_24_audit_pooled_order.py',
'part_25_audit_data_exposure.py',
'part_26_audit_unit_order.py',
'part_27_audit_cleaning_replay.py',
'part_28_audit_plan_conformance.py',
]

load_root_package(globals(), __file__, _PARTS,
                  historical_path='audit_gates.py')
del load_root_package

# Keep module introspection on the implementation fragment (the historical
# script body lived in one file) while REPO still points at the repository root.
__file__ = _os.path.join(
    _HERE, '_parts', 'part_05_audit_confirmation_verdicts.py')
REPO = _REPO

if __name__ == '__main__':
    _run_cli()
