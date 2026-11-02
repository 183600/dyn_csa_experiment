# Suspected Bugs Found in Codebase

This report identifies potential bugs across the Python source files in the
dyn_csa_experiment codebase. Focus areas: logic errors at scale, off-by-one
errors, division-by-zero risks, integer overflow risks, incorrect assumptions
about data sizes, and missing edge case handling.

## File: baseline_tune.py

### Bug 1: Division by zero risk (Line 424)
**Location**: `baseline_tune.py`, line 424
**Description**: `_nll_tot / _tok_tot` where `_tok_tot = int(sum(selection_stats[s]['n_tokens'] for s in selection_stats))`. If all `n_tokens` values are 0 or `selection_stats` is empty, `_tok_tot` becomes 0, raising `ZeroDivisionError` before `_safe_exp_ppl()` is even called.
**Risk**: Critical - crashes the entire tuning run.
**Fix**: Add a guard `if _tok_tot == 0: mean = float('nan')` or similar before the division.

### Bug 2: NaN propagation from empty full_seed_ppls (Line 450)
**Location**: `baseline_tune.py`, line 450
**Description**: `_mean_std(list(full_seed_ppls.values()))[0]` - if `full_seed_ppls` is empty (no usable PPL records), `_mean_std` returns `(nan, nan)`, and accessing `[0]` gives `nan`. This NaN propagates through the artifact.
**Risk**: High - corrupts artifact data silently.
**Fix**: Check `if not full_seed_ppls: mean = float('nan')` before calling `_mean_std`.

### Bug 3: Empty set iteration edge case (Lines 458-459)
**Location**: `baseline_tune.py`, lines 458-459
**Description**: `next(iter(_steps)) if len(_steps) == 1 else None` and `next(iter(_tokens)) if len(_tokens) == 1 else None`. If `_steps` or `_tokens` is an empty set (length 0), these correctly return None since `len(...) == 1` is False. However, if the sets contain unexpected values, the `next(iter(...))` pattern could behave unexpectedly.
**Risk**: Medium - subtle behavior change with different data.
**Fix**: Use `next(iter(_steps), None)` with a default value for clarity.

### Bug 4: Missing keys in budget_by_seed (Line 460-461)
**Location**: `baseline_tune.py`, lines 460-461
**Description**: `'budget_by_seed': {str(s): [records[s].get('steps'), records[s].get('tokens_seen')] for s in sorted(records)}`. If `records[s]` lacks `'steps'` or `'tokens_seen'` keys, both values default to `None`, which may mask data completeness issues.
**Risk**: Medium - incomplete data not detected.
**Fix**: Add explicit key existence checks or document the expected record format.

## File: fuse_analysis.py

### Bug 5: Empty array passed to exact_signflip_p (Line 381)
**Location**: `fuse_analysis.py`, line 381
**Description**: `_rf1 = L.exact_signflip_p([float(x) for x in np.asarray(d_f1, dtype=float).reshape(-1)])`. If `d_f1` is empty (no finite per-seed values pass the gate), an empty array is passed, which may cause undefined behavior in the signflip test.
**Risk**: Medium - p-value computation undefined for empty sample.
**Fix**: Add guard `if not d_f1.size: p_f1 = None` before calling `exact_signflip_p`.

### Bug 6: signflip p-value floor with n=0 (Line 394)
**Location**: `fuse_analysis.py`, line 394
**Description**: `_ppl_floor0 = L.signflip_p_floor(_n_ppl)`. If `_n_ppl` is 0 (no paired samples), the p-value floor may be undefined or return an unexpected value, causing incorrect floor comparison logic.
**Risk**: Medium - incorrect significance assessment.
**Fix**: Add guard `if _n_ppl <= 0: _ppl_floor0 = float('nan')` before calling `signflip_p_floor`.

### Bug 7: Gate logic nested conditions (Lines 332-336, 367-404)
**Location**: `fuse_analysis.py`, lines 332-336 and 367-404
**Description**: The pair_reason gate and subsequent analysis have deeply nested conditions (`_reasons`, `_why`, `_ok_seeds`, `_gate_ok`, `_keep`). A bug in any one condition cascade could silently skip all analysis output or produce incorrect results without clear error signals.
**Risk**: High - hard to reason about correctness for new users/maintainers.
**Fix**: Add more explicit intermediate variable names and assertions; consider refactoring into smaller functions.

## File: run_gapfill.py

### Bug 8: Empty seeds list handling (Line 543)
**Location**: `run_gapfill.py`, line 543
**Description**: `L.set_seed(int((cfg.get('seeds') or [0])[0]))`. If `cfg.get('seeds')` returns `None`, the expression evaluates to `int(([0])[0])` = 0. If `cfg.get('seeds')` returns `[]` (empty list, which is falsy), the same thing happens - seed 0 is used even though the user explicitly provided no seeds. This could silently default to seed 0 when no seeds were intended.
**Risk**: Medium - unexpected seed 0 usage masks user intent.
**Fix**: Change to `seeds = cfg.get('seeds') or []; int(seeds[0]) if seeds else int(0)` with explicit empty-check.

### Bug 9: _num_or_none in par.py vs run_gapfill.py inconsistency
**Location**: `par.py`, line 40-45 vs `run_gapfill.py`, line 36-47
**Description**: The `_num_or_none` in `par.py` returns `None` for booleans via `isinstance(v, (bool, np.bool_))` check at the top. The one in `run_gapfill.py` has the same check. However, `par.py`'s version does `int(str(value).strip())` which would raise `ValueError` for string `'3.5'` (converted to float then int would fail), while `run_gapfill.py`'s version tries `float(v)` first. This inconsistency could cause different behavior for the same input across the codebase.
**Risk**: Low - different code paths handle edge cases differently.
**Fix**: Align both implementations to use the more robust `run_gapfill.py` approach.

## File: par.py

### Bug 10: ctx_active with None ctx (Line 178-183)
**Location**: `par.py`, lines 178-183
**Description**: `ctx_active(ctx, n_cells=1)` calls `bool(ctx) and int(ctx.get("workers", 0)) > 1 and n_cells > 1`. If `ctx` is `None`, `bool(None` is `False`, so the function returns `False` without error. However, if `ctx` is some truthy but incomplete object missing `"workers"`, `ctx.get("workers", 0)` returns 0, and `int(0) > 1` is `False`, so the function returns `False`. This masks the fact that `ctx` is malformed.
**Risk**: Low - silently falls back to serial execution instead of raising an error.
**Fix**: Add `if ctx is None: return False` at the top for clarity.

### Bug 11: GPU slot validation edge case (Lines 85-94)
**Location**: `par.py`, lines 85-94 in `gpu_slots`
**Description**: The check `len(tokens) != count or any(not item or item == "-1" for item in tokens) or len(set(tokens)) != len(tokens)` validates CUDA_VISIBLE_DEVICES mapping. However, if `tokens` contains non-string items (e.g., integers from a different code path), `item.strip()` would raise `AttributeError`. The error check assumes tokens are always strings.
**Risk**: Low - depends on caller ensuring string tokens.
**Fix**: Add `str(item).strip()` or ensure type consistency earlier.

## File: error_analysis.py

### Bug 12: Potential truncation without warning (Lines 355-356)
**Location**: `error_analysis.py`, lines 355-356
**Description**: `val_batch = np.ascontiguousarray(val_batch[:n_rows, :seq_len])`. If `n_rows > val_batch.shape[0]` or `seq_len > val_batch.shape[1]`, the slice silently truncates data. While there's a check `n_rows = min(n_rows, int(val_batch.shape[0]))` at line 355, there's no explicit guard for `seq_len`.
**Risk**: Medium - model may be evaluated on fewer tokens than expected without warning.
**Fix**: Add `seq_len = min(seq_len, val_batch.shape[1])` with a warning.

### Bug 13: context_scan_positions = 0 behavior (Line 346-347)
**Location**: `error_analysis.py`, lines 346-347
**Description**: `if args.context_scan_positions < 0: raise SystemExit(...)` rejects negative values, but 0 means "scan the whole evaluated split" (per the argparse help). There's no explicit check for this in the later code that uses `context_scan_max_positions=(args.context_scan_positions or None)` - when 0 is passed, it becomes `None` due to `or None`, which then means "unbounded" in the called function. This is consistent behavior but could be confusing if the user expects 0 to mean "scan 0 positions".
**Risk**: Low - documented behavior, but worth noting.
**Fix**: Document that 0 means "unbounded scan" in the help string.

## File: _csa_kernels/_parts/part_02_sparse_layer_forward.py

### Bug 14: Potential division by zero in chunk size calculation (Line 994)
**Location**: `_csa_kernels/_parts/part_02_sparse_layer_forward.py`, line 994
**Description**: `t_chunk = max(1, min(T, (192 << 20) // max(int(R) * int(Bm) * 4 * (int(nIH) + 8), 1)))`. If `R * Bm * 4 * (nIH + 8)` evaluates to 0 (unlikely but possible with zero-sized tensors), the `max(..., 1)` prevents division by zero, but the calculation could still produce unexpected results.
**Risk**: Low - defensive `max(1, ...)` prevents actual zero-division.
**Fix**: Add a debug log when the denominator is very small to aid tuning.

### Bug 15: Index out of range in kernel stamp (Line 1195)
**Location**: `_csa_kernels/_parts/part_00_kernel_stamp.py`, line 1195
**Description**: `z = q.new_zeros((int(R), 0, int(nh), int(hd)))`. If `R` is 0, this creates a tensor with 0 first dimension, which may cause issues in subsequent operations that assume at least one row.
**Risk**: Low - depends on the calling context ensuring R > 0.
**Fix**: Add `if R <= 0: return q.new_zeros((0, 0, int(nh), int(hd)))` early return.