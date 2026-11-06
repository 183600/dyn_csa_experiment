"""Measurement library assembled from ordered source fragments.

``_parts/*.py`` are executed in ``_PARTS`` order into this module's globals,
which reproduces the historical single-file body while keeping each fragment
reviewable.  Later fragments may use names defined by earlier ones, so append a
new part (or insert a self-contained one) and keep the list in execution order.
``exp_lib.py`` re-exports this module for the historical import path.

Provenance: the hash of this part list binds the exact fragment set used for
every record.  Two runs with different part lists are different numerical
stacks (per env_fp_v3 policy).
"""

from _split_loader import load_root_package

import hashlib
import json
import os
import pathlib

_PARTS_HASH = None
_PARTS_VALIDATED = False


def _validate_parts_parts():
    """Validate that every part in _PARTS exists inside the package's _parts/.

    Returns the canonical part-list hash, or raises ImportError on failure.
    """
    global _PARTS_HASH, _PARTS_VALIDATED
    if _PARTS_VALIDATED:
        return _PARTS_HASH

    # Use __path__[0] because __file__ may point to the shim (exp_lib.py),
    # not the explib/ package directory.
    fragment_dir = pathlib.Path(__path__[0]) / '_parts'
    if not fragment_dir.is_dir():
        raise ImportError(f'_parts/ directory not found at {fragment_dir}')

    missing = []
    for part in _PARTS:
        p = fragment_dir / part
        if not p.is_file():
            missing.append(part)

    if missing:
        raise ImportError(
            f'Missing _parts/ entries: {missing}.  '
            'Add the new part to _PARTS in the correct order, or create the file.')

    canonical = json.dumps(sorted(_PARTS), sort_keys=True, separators=(',', ':'))
    _PARTS_HASH = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    _PARTS_VALIDATED = True
    return _PARTS_HASH


def part_list_hash():
    """Return the SHA-256 hash of the canonical _PARTS list.

    This binds the exact fragment set used in every run.  Records that carry
    a different part-list hash have been generated with a different measurement
    library configuration and are not directly comparable.
    """
    _validate_parts_parts()
    return _PARTS_HASH


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
'part_55_ablation_manipulation.py',
'part_56_length_strata.py',
'part_57_unit_quantiles.py',
'part_58_endpoint_sensitivity.py',
'part_59_ordered_unit_inference.py',
'part_60_pooled_order_inference.py',
'part_61_training_data_exposure.py',
'part_62_cleaning_replay.py',
'part_63_cross_panel_synthesis.py',
'part_64_cross_panel_scale_sensitivity.py',
'part_65_contrast_network.py',
'part_66_design_protocol.py',
'part_67_split_registry.py',
'part_68_preparation_clearance.py',
'part_69_cleaning_sensitivity.py',
'part_70_analysis_rehearsal.py',
'part_71_validation_transfer.py',
'part_72_budget_truncation.py',
'part_73_ablation_lattice.py',
'part_74_design_estimability.py',
'part_75_article_sidecar.py',
'part_76_type_sm_design.py',
'part_77_exact_tost.py',
'part_78_global_fdr.py',
'part_79_ablation_ranking.py',
'part_80_analysis_multiverse.py',
'part_81_ablation_screening.py',
'part_82_numerical_canary.py',
'part_83_contrast_interaction.py',
'part_84_sequential_design.py',
'part_85_budget_ablation.py',
]

load_root_package(globals(), __file__, _PARTS,
                  historical_path='exp_lib.py')
del load_root_package
