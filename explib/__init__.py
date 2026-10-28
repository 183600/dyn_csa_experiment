"""Measurement library (split by responsibility; source-compatible with exp_lib)."""

import os as _os

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
]

for _part in _PARTS:
    _path = _os.path.join(_HERE, '_parts', _part)
    with open(_path, encoding='utf-8') as _fh:
        _src = _fh.read()
    exec(compile(_src, _path, 'exec'), globals())
del _part, _path, _fh, _src
