"""Measurement library assembled from ordered source fragments.

``_parts/*.py`` are executed in ``_PARTS`` order into this module's globals,
which reproduces the historical single-file body while keeping each fragment
reviewable.  Later fragments may use names defined by earlier ones, so append a
new part (or insert a self-contained one) and keep the list in execution order.
``exp_lib.py`` re-exports this module for the historical import path.
"""

import os as _os

from _split_loader import load_parts as _load_parts

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.dirname(_HERE)
__file__ = _os.path.join(_REPO, 'exp_lib.py')

_PARTS = [
'part_00_kernel_stamp.py',
'part_01_paired_nll_unit_influence.py',
'part_02_hoist_blocks_from_cuts.py',
'part_03_pool_geom_flat_single.py',
'part_04_attn_full_width_ok.py',
'part_05_RMSNorm.py',
'part_06_tokenise_row_chunks.py',
'part_07_wikitext_article_text.py',
'part_08_load_fineweb_edu.py',
'part_09_param_breakdown.py',
'part_10_paired_bootstrap_ci.py',
'part_11_log_binom_cdf_half.py',
'part_12_scale_points.py',
'part_13_scale_ladder_audit.py',
'part_14_sweep_contrast_factor_audit.py',
'part_15_paired_unit_deltas.py',
'part_16_comparison_budget_ledger.py',
'part_17_validation_row_split.py',
'part_18_simple_baseline_audit.py',
'part_19_retrieval_metrics.py',
'part_20_metric_policy_audit.py',
'part_21_plan_row_split.py',
'part_22_baseline_tuning_audit.py',
'part_23_panel_look_snapshot.py',
'part_24_ablation_coverage_across_panels.py',
'part_25_factorial_interaction_audit.py',
'part_26_print_methodology_audit.py',
'part_27_dependency_contract.py',
'part_28_PrefixStats.py',
'part_29_ledger_curve.py',
'part_30_context_target_counts.py',
'part_31_overfit_sanity_check_impl.py',
'part_32_pooled_ppl_from_records.py',
'part_33_plot.py',
'part_34_dump_aggregate.py',
'part_35_run_allocation.py',
'part_36_ablation_controls.py',
'part_37_reproduction.py',
'part_38_variance_components.py',
'part_39_factorial_power.py',
'part_40_ablation_joint_correction.py',
'part_41_missingness_sensitivity.py',
'part_42_trajectory_stability.py',
'part_43_run_disposition.py',
'part_44_component_lesion.py',
'part_45_endpoint_selection.py',
'part_46_reproducibility.py',
'part_47_test_access.py',
'part_48_arm_rank_inference.py',
'part_49_arm_rank_units.py',
'part_50_winner_curse.py',
'part_51_effect_sensitivity.py',
'part_52_length_position.py',
'part_53_confirmation_practical.py',
'part_54_family_correction.py',
]

_load_parts(globals(), _HERE, _PARTS)
del _load_parts
