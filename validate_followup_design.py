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

# The power-simulation artifact must certify the plan it is quoted by: the
# binding below cross-checks study id, hypothesis family, seed count, target
# power, familywise alpha and the minimum simulated power.  A boolean alone
# ("power_simulation_completed": true) stops being acceptable evidence.
POWER_ARTIFACT_TOOL = "followup_power_sim.py"


def _power_artifact_errors(stats: dict, study_id, seeds, alpha, target_power, *,
                            repo: Path | None, manifest_dir: Path | None) -> list[str]:
    """Bind statistics.power_simulation_artifact to the design it certifies.

    Fail-closed: an unreadable artifact, a tool mismatch, or any field that
    does not match the declared plan is an error — the boolean
    ``power_simulation_completed`` alone is not evidence.
    """
    errors: list[str] = []

    def _safe_float(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        try:
            f = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return f if math.isfinite(f) else None

    plan_alpha = _safe_float(alpha)
    plan_target = _safe_float(target_power)
    seed_list = list(seeds) if isinstance(seeds, list) else []
    art = stats.get("power_simulation_artifact")
    if not isinstance(art, str) or not art.strip() or art.startswith("TODO"):
        errors.append("statistics.power_simulation_artifact must name the "
                      "followup_power_sim.py output JSON when "
                      "power_simulation_completed is true")
        return errors
    candidates = []
    for base in (manifest_dir, repo):
        if base is not None:
            p = (Path(base) / art).resolve()
            if p not in candidates:
                candidates.append(p)
    payload = None
    for p in candidates:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            break
        except Exception:
            continue
    if not isinstance(payload, dict):
        errors.append(f"power_simulation_artifact {art!r} is not a readable "
                      f"JSON object (searched: {[str(c) for c in candidates]})")
        return errors
    if payload.get("tool") != POWER_ARTIFACT_TOOL:
        errors.append(f"power_simulation_artifact {art!r} was not produced by "
                      f"{POWER_ARTIFACT_TOOL} (tool={payload.get('tool')!r})")
    if payload.get("status") != "ok":
        errors.append(f"power_simulation_artifact status is {payload.get('status')!r}, "
                      f"not 'ok'")
    if payload.get("study_id") != study_id:
        errors.append(f"power_simulation_artifact study_id {payload.get('study_id')!r} "
                      f"does not match the plan's {study_id!r}")
    if set(payload.get("hypotheses") or []) != REQUIRED_HYPOTHESES:
        errors.append(f"power_simulation_artifact hypotheses "
                      f"{sorted(payload.get('hypotheses') or [])} do not match the "
                      f"pre-registered family {sorted(REQUIRED_HYPOTHESES)}")
    if payload.get("test") != "exact_two_sided_sign_flip" or payload.get("correction") != "holm":
        errors.append("power_simulation_artifact must record the exact "
                      "two-sided sign-flip test with Holm correction")
    art_seeds = payload.get("n_seeds")
    if not isinstance(art_seeds, int) or isinstance(art_seeds, bool) or art_seeds != len(seed_list):
        errors.append(f"power_simulation_artifact n_seeds {art_seeds!r} does not match "
                      f"the declared seed panel ({len(seed_list)} seeds)")
    art_alpha = _safe_float(payload.get("alpha"))
    if plan_alpha is None or art_alpha is None or abs(art_alpha - plan_alpha) > 1e-12:
        errors.append(f"power_simulation_artifact alpha {payload.get('alpha')!r} differs from "
                      f"the declared familywise alpha {alpha!r}")
    art_target = _safe_float(payload.get("target_power"))
    if plan_target is None or art_target is None or abs(art_target - plan_target) > 1e-12:
        errors.append(f"power_simulation_artifact target_power {payload.get('target_power')!r} "
                      f"differs from the declared target {target_power!r}")
    reps = payload.get("mc_replicates")
    if not isinstance(reps, int) or isinstance(reps, bool) or reps < 200:
        errors.append(f"power_simulation_artifact mc_replicates {reps!r} is below the "
                      f"200-replicate floor (fewer certifies nothing)")
    min_power = payload.get("min_power")
    mp = _safe_float(min_power)
    if mp is None or (plan_target is not None and mp < plan_target):
        errors.append(f"power_simulation_artifact min_power {min_power!r} does not clear "
                      f"the declared target power {target_power!r} — increase the seed "
                      f"count or revise the design, do not shrink the target")
    return errors


def _registry_errors(plan: dict, repo: Path) -> list[str]:
    """Check that the design's variant IDs exist AND have the declared shape.

    The structural validator alone cannot tell a pure-HCA arm from the hybrid
    alternation: both are just strings.  Importing the repository's variant
    registry (``exp_lib.make_layer_cfgs`` after ``v7_supp`` installs the RoPE
    variants) turns the design record into a machine-checkable claim about the
    code that would actually run.

    This is a *design/code consistency* check, not evidence that the arms
    train well or that the hypotheses are true.
    """
    import sys

    errors: list[str] = []
    repo = Path(repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import exp_lib as L  # noqa: PLC0415 — imported lazily, tests may skip
        import v7_supp  # noqa: F401,PLC0415 — installs the *_rope registry
    except Exception as exc:  # pragma: no cover - environment dependent
        return [f"cannot import the variant registry from {repo}: "
                f"{type(exc).__name__}: {exc}"]

    design = plan.get("design") if isinstance(plan.get("design"), dict) else {}
    for arm in design.get("factorial_arms") or []:
        if not isinstance(arm, dict):
            continue
        variant = arm.get("variant")
        aid = arm.get("id")
        if not isinstance(variant, str) or not variant or variant.startswith("TODO"):
            errors.append(f"{aid}: variant is not an implemented registry entry: {variant!r}")
            continue
        try:
            cfgs = L.make_layer_cfgs(3, variant)
        except Exception as exc:
            errors.append(f"{aid}: make_layer_cfgs({variant!r}) failed: "
                          f"{type(exc).__name__}: {exc}")
            continue
        if not cfgs:
            errors.append(f"{aid}: make_layer_cfgs({variant!r}) returned no layers")
            continue
        family = str(arm.get("family", "")).lower()
        if family in ("csa", "hca"):
            kinds = [str(getattr(c, "kind", "")) for c in cfgs]
            if any(k != family for k in kinds):
                errors.append(
                    f"{aid}: variant {variant!r} is not a pure {family.upper()} arm "
                    f"(layer kinds {kinds}); a hybrid arm cannot answer the "
                    f"{family.upper()} hypothesis")
        seg = str(arm.get("segmentation", ""))
        dynamic = [bool(getattr(c, "dynamic", False)) for c in cfgs]
        if seg == "dynamic" and not all(dynamic):
            errors.append(f"{aid}: declared dynamic but not every layer is dynamic: {dynamic}")
        if seg == "fixed" and any(dynamic):
            errors.append(f"{aid}: declared fixed but at least one layer is dynamic: {dynamic}")
        # pattern_seed (review-15): a control arm may declare the
        # connection-pattern seed explicitly; the registry must resolve the
        # SAME seed on every layer, and only the hash-based re-randomized
        # controls (random_each / random_dist) may carry a nonzero one.
        declared_ps = arm.get("pattern_seed")
        if declared_ps is not None:
            if (not isinstance(declared_ps, int) or isinstance(declared_ps, bool)
                    or declared_ps < 0):
                errors.append(f"{aid}: pattern_seed must be a non-negative integer, "
                              f"got {declared_ps!r}")
            else:
                resolved = {int(getattr(c, "pattern_seed", 0) or 0) for c in cfgs}
                modes = {str(getattr(c, "indexer_mode", "learned")) for c in cfgs}
                if len(resolved) != 1 or resolved != {int(declared_ps)}:
                    errors.append(f"{aid}: declared pattern_seed={declared_ps} but the "
                                  f"variant resolves to {sorted(resolved)}")
                if declared_ps != 0 and not modes <= {"random_each", "random_dist"}:
                    errors.append(f"{aid}: a nonzero pattern_seed requires the hash-based "
                                  f"controls (random_each/random_dist), got modes {sorted(modes)}")

    for entry in design.get("selection_controls") or []:
        # The protocol's mechanism-isolation selection controls (random /
        # re-randomized / distance-matched / nearest / stride selectors, each
        # optionally under connection-pattern seeds independent of the
        # training seed).  Same registry checks as a factorial arm: the
        # declared variant must exist, and a declared pattern_seed must be
        # what the registry actually resolves.
        if not isinstance(entry, dict):
            continue
        variant = entry.get("variant")
        sid = entry.get("id")
        if not isinstance(variant, str) or not variant or variant.startswith("TODO"):
            errors.append(f"{sid}: selection-control variant is not an implemented "
                          f"registry entry: {variant!r}")
            continue
        try:
            cfgs = L.make_layer_cfgs(3, variant)
        except Exception as exc:
            errors.append(f"{sid}: make_layer_cfgs({variant!r}) failed: "
                          f"{type(exc).__name__}: {exc}")
            continue
        declared_ps = entry.get("pattern_seed")
        if declared_ps is not None:
            if (not isinstance(declared_ps, int) or isinstance(declared_ps, bool)
                    or declared_ps < 0):
                errors.append(f"{sid}: pattern_seed must be a non-negative integer, "
                              f"got {declared_ps!r}")
            else:
                resolved = {int(getattr(c, "pattern_seed", 0) or 0) for c in cfgs}
                if len(resolved) != 1 or resolved != {int(declared_ps)}:
                    errors.append(f"{sid}: declared pattern_seed={declared_ps} but the "
                                  f"variant resolves to {sorted(resolved)}")

    for entry in design.get("dense_controls") or []:
        if not isinstance(entry, dict):
            continue
        variant = entry.get("variant")
        eid = entry.get("id")
        if not isinstance(variant, str) or not variant or variant.startswith("TODO"):
            errors.append(f"{eid}: dense variant is not an implemented registry entry: {variant!r}")
            continue
        try:
            cfgs = L.make_layer_cfgs(3, variant)
        except Exception as exc:
            errors.append(f"{eid}: make_layer_cfgs({variant!r}) failed: "
                          f"{type(exc).__name__}: {exc}")
            continue
        if any(str(getattr(c, "kind", "")) != "full" for c in cfgs):
            errors.append(f"{eid}: variant {variant!r} is not a dense/full-attention arm")
        # A FLOPs-matched declaration must point at a run cfg that actually
        # routes the arm through the analytical solver (flops_matched set +
        # reference); the registry entry alone only fixes the layer shape.
        if str(entry.get("matching", "")) == "measured_training_flops":
            _note = str(entry.get("note", ""))
            if "analytical" not in _note and "flops_matched_mlp_ratio" not in _note:
                errors.append(
                    f"{eid}: a measured_training_flops control must document its capacity "
                    f"planning source (run-cfg flops_matched set -> flops_matched_mlp_ratio) "
                    f"and that the 1% profiler gate, not the analytical equality, accepts "
                    f"the arm as compute-matched")
    return errors


def validate(plan: dict, require_ready: bool = False, *,
             verify_registry: bool = False, repo: Path | None = None,
             manifest_dir: Path | None = None) -> list[str]:
    errors: list[str] = []

    def need(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    def section(name: str) -> dict:
        value = plan.get(name)
        if value is None:
            return {}
        if not isinstance(value, dict):
            need(False, f"{name} must be a JSON object")
            return {}
        return value

    def object_list(name: str, owner: dict) -> list:
        value = owner.get(name)
        if value is None:
            return []
        if not isinstance(value, list):
            need(False, f"{name} must be a JSON array")
            return []
        return value

    def add_unique(values: set, value: object, message: str,
                   duplicate_message: str) -> None:
        try:
            duplicate = value in values
        except TypeError:
            need(False, message)
            return
        need(not duplicate, duplicate_message)
        values.add(value)

    need(isinstance(plan, dict), "root must be a JSON object")
    if not isinstance(plan, dict):
        return errors

    need(plan.get("schema_version") == 1, "schema_version must equal 1")
    need(bool(plan.get("study_id")), "study_id is required")

    freeze = section("freeze")
    design = section("design")
    compute = section("compute_matching")
    evaluation = section("evaluation")
    stats = section("statistics")
    gates = section("algorithmic_gates")

    arms = object_list("factorial_arms", design)
    seen_ids = set()
    seen_cells = set()
    for arm in arms:
        if not isinstance(arm, dict):
            errors.append("each factorial arm must be an object")
            continue
        aid = arm.get("id")
        need(bool(aid), f"factorial arm id missing: {aid!r}")
        if aid is not None:
            add_unique(seen_ids, aid,
                       f"factorial arm id must be hashable: {aid!r}",
                       f"factorial arm id duplicated: {aid!r}")
        cell = (arm.get("family"), arm.get("segmentation"))
        add_unique(seen_cells, cell,
                   f"factorial cell must be hashable: {cell!r}",
                   f"duplicate factorial cell: {cell}")
        need(bool(arm.get("variant")), f"{aid}: variant is required")
    need(seen_cells == REQUIRED_FACTORIAL,
         f"factorial cells must be exactly {sorted(REQUIRED_FACTORIAL)}")

    dense = object_list("dense_controls", design)
    dense_kinds = set()
    for entry in dense:
        if not isinstance(entry, dict):
            need(False, "each dense control must be an object")
            continue
        kind = entry.get("matching")
        try:
            dense_kinds.add(kind)
        except TypeError:
            need(False, "dense control matching must be hashable")
    need({"parameter_count", "measured_training_flops"} <= dense_kinds,
         "separate parameter-matched and measured-FLOPs-matched Dense controls are required")

    seeds = design.get("seeds")
    need(isinstance(seeds, list) and len(seeds) >= 12,
         "at least 12 predeclared seeds are required for the default four-test family")
    if isinstance(seeds, list):
        valid_seeds = all(isinstance(s, int) and not isinstance(s, bool) and s >= 0
                          for s in seeds)
        need(valid_seeds, "seeds must be non-negative integers")
        if valid_seeds:
            need(len(seeds) == len(set(seeds)), "seed list contains duplicates")
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
    need(isinstance(reps, int) and not isinstance(reps, bool) and reps >= 10000,
         "document-level bootstrap must use at least 10,000 replicates")
    need(evaluation.get("test_data_used_for_tuning") is False,
         "test data must not be used for tuning or model selection")
    need(evaluation.get("document_level_bootstrap_replicates") and
         evaluation.get("document_bootstrap_unit", "document") == "document",
         "bootstrap unit must be document, not window")

    raw_hypotheses = stats.get("primary_hypotheses")
    if raw_hypotheses is None:
        raw_hypotheses = []
    hypotheses = set()
    if not isinstance(raw_hypotheses, list):
        need(False, "primary_hypotheses must be a JSON array")
    else:
        for item in raw_hypotheses:
            try:
                hypotheses.add(item)
            except TypeError:
                need(False, "primary hypotheses must be hashable strings")
    need(hypotheses == REQUIRED_HYPOTHESES,
         f"primary hypotheses must be exactly {sorted(REQUIRED_HYPOTHESES)}")
    need(stats.get("paired_by") == "seed", "primary comparisons must be paired by seed")
    need(stats.get("test") == "exact_two_sided_sign_flip", "exact two-sided sign-flip test required")
    need(stats.get("correction") == "Holm", "Holm correction required")
    alpha = stats.get("familywise_alpha")
    try:
        alpha_ok = (isinstance(alpha, (int, float)) and not isinstance(alpha, bool)
                    and math.isfinite(float(alpha)) and 0 < alpha <= 0.05)
    except (TypeError, ValueError, OverflowError):
        alpha_ok = False
    need(alpha_ok, "familywise alpha must be in (0, 0.05]")
    n = len(seeds) if isinstance(seeds, list) else 0
    m = len(REQUIRED_HYPOTHESES)
    p_floor = 2 / (2 ** n) if n else math.inf
    need(alpha_ok and n > 0 and m * p_floor < alpha,
         f"exact sign-flip resolution cannot clear first Holm threshold: m*2/2^n={m*p_floor:g}")
    sim_done = stats.get("power_simulation_completed") is True
    need(sim_done, "simulation-based power analysis must be completed before freeze")
    if sim_done:
        # Fail closed: bind the boolean to the artifact followup_power_sim.py
        # actually produced — study id, family, seed panel, alpha, target and
        # minimum simulated power must all match the declared plan.
        errors.extend(_power_artifact_errors(
            stats, plan.get("study_id"), seeds, alpha, stats.get("target_power"),
            repo=repo, manifest_dir=manifest_dir))
    need(stats.get("report_per_seed_differences") is True, "per-seed differences must be reported")
    need(stats.get("report_effect_sizes_and_confidence_intervals") is True,
         "effect sizes and confidence intervals must be reported")
    # A pre-registered test needs a stopping rule: without one, the number of
    # looks (and therefore the error rate) is chosen after seeing the data.
    # This study is a single fixed-n look; interim looks/sequential stopping
    # would require their own alpha-spending registration.
    need(stats.get("stopping_rule") == "single_look_fixed_n",
         "statistics.stopping_rule must be 'single_look_fixed_n' for this design")
    need(stats.get("n_looks") == 1,
         "statistics.n_looks must be 1 (no interim analysis registered)")
    need(stats.get("interim_analysis_planned") is False,
         "interim_analysis_planned must be false for a single-look design")

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
            if not isinstance(arm, dict):
                continue
            variant = str(arm.get("variant", ""))
            need(bool(variant) and not variant.startswith("TODO"),
                 f"{arm.get('id')}: replace TODO with an implemented pure variant")
        for arm in dense:
            if not isinstance(arm, dict):
                continue
            variant = str(arm.get("variant", ""))
            need(bool(variant) and not variant.startswith("TODO"),
                 f"{arm.get('id')}: replace TODO with an implemented Dense variant")

    if verify_registry or require_ready:
        if repo is None:
            errors.append("registry verification requested without a repository path")
        else:
            errors.extend(_registry_errors(plan, Path(repo)))

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--require-ready", action="store_true",
                        help="require a frozen, test-unaccessed preregistration")
    parser.add_argument("--verify-registry", action="store_true",
                        help="import the repository variant registry and check that "
                             "every declared arm exists and is pure (implied by --require-ready)")
    parser.add_argument("--repo", type=Path, default=None,
                        help="repository root for --verify-registry (default: the manifest's directory)")
    args = parser.parse_args()
    try:
        plan = json.loads(args.manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"INVALID: cannot read JSON: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    errors = validate(plan, require_ready=args.require_ready,
                      verify_registry=args.verify_registry,
                      repo=args.repo or args.manifest.resolve().parent,
                      manifest_dir=args.manifest.resolve().parent)
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
