#!/usr/bin/env python3
"""Fail-closed validator for the independent Dynamic-CSA/HCA follow-up plan.

This validates the *design record*, not model quality or experimental results.
It deliberately does not read, edit, or supersede confirmation_frozen.json.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

REQUIRED_FACTORIAL = {
    ("CSA", "fixed"),
    ("CSA", "dynamic"),
    ("HCA", "fixed"),
    ("HCA", "dynamic"),
}
REQUIRED_HYPOTHESES = {
    "dynamic_minus_fixed_within_CSA",
    "dynamic_minus_fixed_within_HCA",
    "HCA_minus_CSA_under_fixed_segmentation",
    "attention_family_by_segmentation_interaction",
}
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def validate(plan: dict, require_ready: bool = False) -> list[str]:
    errors: list[str] = []

    def need(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    need(isinstance(plan, dict), "root must be a JSON object")
    if not isinstance(plan, dict):
        return errors

    need(plan.get("schema_version") == 1, "schema_version must equal 1")
    need(bool(plan.get("study_id")), "study_id is required")

    freeze = plan.get("freeze") or {}
    design = plan.get("design") or {}
    compute = plan.get("compute_matching") or {}
    evaluation = plan.get("evaluation") or {}
    stats = plan.get("statistics") or {}
    gates = plan.get("algorithmic_gates") or {}

    arms = design.get("factorial_arms") or []
    seen_ids = set()
    seen_cells = set()
    for arm in arms:
        if not isinstance(arm, dict):
            errors.append("each factorial arm must be an object")
            continue
        aid = arm.get("id")
        need(bool(aid) and aid not in seen_ids, f"factorial arm id missing or duplicated: {aid!r}")
        seen_ids.add(aid)
        cell = (arm.get("family"), arm.get("segmentation"))
        need(cell not in seen_cells, f"duplicate factorial cell: {cell}")
        seen_cells.add(cell)
        need(bool(arm.get("variant")), f"{aid}: variant is required")
    need(seen_cells == REQUIRED_FACTORIAL,
         f"factorial cells must be exactly {sorted(REQUIRED_FACTORIAL)}")

    dense = design.get("dense_controls") or []
    dense_kinds = {x.get("matching") for x in dense if isinstance(x, dict)}
    need({"parameter_count", "measured_training_flops"} <= dense_kinds,
         "separate parameter-matched and measured-FLOPs-matched Dense controls are required")

    seeds = design.get("seeds") or []
    need(isinstance(seeds, list) and len(seeds) >= 12,
         "at least 12 predeclared seeds are required for the default four-test family")
    need(len(seeds) == len(set(seeds)), "seed list contains duplicates")
    need(all(isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in seeds),
         "seeds must be non-negative integers")
    need(design.get("same_seed_data_order") is True, "paired arms must share per-seed data order")
    need(design.get("same_training_token_budget") is True, "paired arms must share the registered token budget")
    need(design.get("same_optimizer_schedule") is True, "paired arms must share optimizer/schedule")

    tol = compute.get("training_flops_tolerance_fraction")
    need(compute.get("profiler_measured") is True, "FLOPs matching must use profiler-measured FLOPs")
    need(isinstance(tol, (int, float)) and not isinstance(tol, bool) and 0 < tol <= 0.01,
         "training FLOPs tolerance must be in (0, 0.01]")
    need(compute.get("inference_flops_reported") is True, "inference FLOPs must be reported")
    need(compute.get("wall_clock_is_not_flops_proxy") is True,
         "wall-clock time cannot be used as a FLOPs proxy")
    need(compute.get("unsupported_operator_policy") == "refuse_flops_matched_claim",
         "unsupported profiler operators must fail closed")

    min_docs = evaluation.get("minimum_distinct_documents")
    need(isinstance(min_docs, int) and not isinstance(min_docs, bool) and min_docs >= 64,
         "evaluation must include at least 64 distinct documents (or the full test set)")
    need(evaluation.get("document_ids_frozen_before_test_access") is True,
         "document IDs must be frozen before test access")
    reps = evaluation.get("document_level_bootstrap_replicates")
    need(isinstance(reps, int) and reps >= 10000,
         "document-level bootstrap must use at least 10,000 replicates")
    need(evaluation.get("test_data_used_for_tuning") is False,
         "test data must not be used for tuning or model selection")
    need(evaluation.get("document_level_bootstrap_replicates") and
         evaluation.get("document_bootstrap_unit", "document") == "document",
         "bootstrap unit must be document, not window")

    hypotheses = set(stats.get("primary_hypotheses") or [])
    need(hypotheses == REQUIRED_HYPOTHESES,
         f"primary hypotheses must be exactly {sorted(REQUIRED_HYPOTHESES)}")
    need(stats.get("paired_by") == "seed", "primary comparisons must be paired by seed")
    need(stats.get("test") == "exact_two_sided_sign_flip", "exact two-sided sign-flip test required")
    need(stats.get("correction") == "Holm", "Holm correction required")
    alpha = stats.get("familywise_alpha")
    need(isinstance(alpha, (int, float)) and 0 < alpha <= 0.05,
         "familywise alpha must be in (0, 0.05]")
    n = len(seeds) if isinstance(seeds, list) else 0
    m = len(REQUIRED_HYPOTHESES)
    p_floor = 2 / (2 ** n) if n else math.inf
    need(m * p_floor < alpha if n else False,
         f"exact sign-flip resolution cannot clear first Holm threshold: m*2/2^n={m*p_floor:g}")
    need(stats.get("power_simulation_completed") is True,
         "simulation-based power analysis must be completed before freeze")
    need(stats.get("report_per_seed_differences") is True, "per-seed differences must be reported")
    need(stats.get("report_effect_sizes_and_confidence_intervals") is True,
         "effect sizes and confidence intervals must be reported")

    required_gates = (
        "prefix_invariance",
        "future_perturbation_and_gradient",
        "all_block_reference_equivalence",
        "fp32_and_bf16_forward_backward",
        "duplicate_selection_and_overlap_tests",
        "pure_hca_layer_registry_verified",
        "eager_and_kernel_parity",
    )
    for key in required_gates:
        need(gates.get(key) is True, f"algorithmic gate is not passed: {key}")

    if require_ready:
        need(plan.get("status") == "preregistered", "status must be preregistered")
        need(bool(freeze.get("registered_at_utc")), "registered_at_utc is required")
        need(bool(SHA_RE.fullmatch(str(freeze.get("source_commit") or ""))),
             "freeze.source_commit must be a 40-character Git SHA")
        need(freeze.get("test_split_accessed") is False,
             "test split must not have been accessed before registration")
        need(freeze.get("test_document_ids_frozen") is True,
             "test document IDs must be frozen")
        need(bool(freeze.get("output_dir")) and
             freeze.get("output_dir") != "results_lm_test",
             "follow-up output_dir must be separate from v12 results")
        for arm in arms:
            variant = str(arm.get("variant", ""))
            need(bool(variant) and not variant.startswith("TODO"),
                 f"{arm.get('id')}: replace TODO with an implemented pure variant")
        for arm in dense:
            variant = str(arm.get("variant", ""))
            need(bool(variant) and not variant.startswith("TODO"),
                 f"{arm.get('id')}: replace TODO with an implemented Dense variant")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--require-ready", action="store_true",
                        help="require a frozen, test-unaccessed preregistration")
    args = parser.parse_args()
    try:
        plan = json.loads(args.manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"INVALID: cannot read JSON: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    errors = validate(plan, require_ready=args.require_ready)
    if errors:
        print(f"NOT READY ({len(errors)} issue(s))")
        for error in errors:
            print(f"- {error}")
        return 1
    print("STRUCTURE OK")
    if not args.require_ready:
        print("NOTE: structural validity is not preregistration, execution, or scientific validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
