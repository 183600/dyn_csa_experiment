#!/usr/bin/env python3
"""Algorithmic acceptance-gate runner for the follow-up confirmatory study.

FOLLOWUP_CONFIRMATORY_PROTOCOL.md ("Algorithmic acceptance gates before
training") requires seven correctness gates to run BEFORE any confirmatory
training of the factorial arms:

    prefix invariance / future-token perturbation (+ gradients) /
    all-block reference equivalence / FP32-and-BF16 forward+backward /
    duplicate-selection and overlap semantics / pure-HCA layer registry /
    eager-vs-kernel parity.

Until now those gates existed only as scattered tests plus seven BOOLEANS in
``followup_design.template.json`` — booleans a hand can set without evidence.
``validate_followup_design.py`` already refuses to trust a bare
``power_simulation_completed`` flag (review-16, N117: the boolean must bind to
a ``followup_power_sim.py`` artifact), but the ``algorithmic_gates`` block had
no producer and no binding, so a "gate passed" flag was still a rubber stamp.

This tool is the missing producer.  It runs every CPU-runnable gate over the
follow-up rope arms (the four factorial cells, the registered controls the
gates need, and the same-backbone dense isolation arm) at tiny geometry, and
writes one JSON artifact that ``validate_followup_design.py`` binds
field-by-field: setting ``algorithmic_gates.<name> = true`` then REQUIRES an
``algorithmic_gates_artifact`` whose tool, study id, arm coverage and
per-gate pass status all match.

CPU scope, disclosed instead of hidden: the runner executes the eager rope
paths and the csa_kernels torch-reference backend (impl='ref') on CPU.  The
Triton kernels and the CUDA-only G0 benchmark layer remain GPU work — the
artifact records ``backends_tested`` and ``device`` so a CPU artifact can
never masquerade as the full CUDA/Triton parity run.  These are correctness
gates on the algorithm, not training, not test-split access, and not evidence
that any method works.

Gates and what each actually checks (tiny geometry, deterministic seeds):

  1  prefix_invariance           — label shift + f(x[:T']) == f(x)[:T'] +
                                   strict readability + incremental-decode
                                   consistency, over the seven rope arms
                                   (reuses ``audit_gates.audit_causality``,
                                   which already covers the four factorial
                                   cells, both selection controls and the
                                   dense reference);
  2  future_perturbation_and_gradient
                                 — perturbing tokens at/after a cut leaves
                                   prefix outputs unchanged (forward) AND the
                                   gradient of a prefix output w.r.t. the
                                   input embeddings of future positions is
                                   exactly zero (backward leak), for the four
                                   factorial arms + the same-backbone dense
                                   control ``full_rope_cos_sink``;
  3  all_block_reference_equivalence
                                 — (a) batched forward == per-row unbatched
                                   forward (the row-chunking equivalence
                                   contract, design_notes §6.1) for all four
                                   factorial arms, and (b) at a geometry where
                                   every readable block is selected, the
                                   topk-clamped arm ``csa_fixed_rope`` and the
                                   explicit all-blocks arm
                                   ``csa_fixed_rope_allblocks`` produce the
                                   same outputs from the same weights;
  4  fp32_and_bf16_forward_backward
                                 — forward + backward for all four factorial
                                   arms in fp32 and under bf16 autocast:
                                   every gradient finite in both dtypes, fp32
                                   forward deterministic across a re-run, and
                                   the bf16-vs-fp32 forward deviation within
                                   the declared dtype tolerance table
                                   (fp32 1e-5 / bf16 1e-1 relative — the same
                                   backends tiering the m0 G0 suite uses);
  5  duplicate_selection_and_overlap_tests
                                 — the indexer's selected-block table has no
                                   duplicate index within a row under
                                   adversarial all-tie scores (stable top-k
                                   semantics), the readable mask never includes
                                   the block containing the query position
                                   itself (the open/current block is served by
                                   the per-token window branch, never twice),
                                   and ``block_readable`` keeps the strict
                                   inequality pos > last_tok;
  6  pure_hca_layer_registry_verified
                                 — ``hca_fixed_rope`` / ``hca_dynamic_rope``
                                   resolve to kind='hca' in EVERY layer with
                                   the declared segmentation, while
                                   ``hybrid_fixed_rope`` is verified NOT to
                                   be pure HCA (negative control: the hybrid
                                   alternation cannot answer an HCA
                                   hypothesis);
  7  eager_and_kernel_parity     — the three core operators (pool / indexer /
                                   attention) agree between the eager path
                                   and the csa_kernels torch-reference
                                   twin, and the model-level eager-vs-ref
                                   comparison passes for csa_fixed /
                                   csa_dynamic / hybrid_dynamic (the non-rope
                                   families the kernel path supports), on the
                                   ref backend at fp32.

Exit status: 0 only if every gate passes; the artifact records per-gate
status and evidence either way.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
from pathlib import Path

TOOL_NAME = 'followup_gate_runner.py'
SCHEMA_VERSION = 1
# Mirrors REQUIRED gate ids in validate_followup_design.py (the validator owns
# the canonical list; a mismatch here is caught by the binding tests).
GATE_KEYS = (
    'prefix_invariance',
    'future_perturbation_and_gradient',
    'all_block_reference_equivalence',
    'fp32_and_bf16_forward_backward',
    'duplicate_selection_and_overlap_tests',
    'pure_hca_layer_registry_verified',
    'eager_and_kernel_parity',
)
# The four factorial cells of FOLLOWUP_CONFIRMATORY_PROTOCOL.md plus the
# same-backbone dense isolation arm (review-17) used by the gradient gate.
FACTORIAL_ARMS = ('csa_fixed_rope', 'csa_dynamic_rope',
                  'hca_fixed_rope', 'hca_dynamic_rope')
DENSE_ISOLATION_ARM = 'full_rope_cos_sink'
# Declared dtype tolerance table for gate 4 (same backends tiering as the m0
# G0 suite: fp32 1e-5 observed, bf16 relative 1e-1).
TOL_FP32 = 1e-5
TOL_BF16_REL = 1e-1


def _repo_import(repo: Path):
    """Import exp_lib + v7_supp (rope registry) from the given repo root."""
    repo = Path(repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import exp_lib as L  # noqa: PLC0415
    import v7_supp  # noqa: F401,PLC0415 — installs the *_rope variant registry
    return L


def _tiny_model(L, variant, *, seed=0, vocab=128, d=64, n_layers=4,
                n_heads=4, d_head=16, seq=96):
    """A tiny rope-arm model at fixed seed (deterministic construction)."""
    import copy as _copy
    import torch
    L.set_seed(seed)
    cfgs = L.make_layer_cfgs(n_layers, variant)
    model = L.SmallGPT(vocab, d, n_layers, n_heads, d_head, seq,
                       _copy.deepcopy(cfgs))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Gate 1: prefix invariance (reuse the existing zero-experiment audit, which
# already covers the seven rope arms incl. the four factorial cells).
# ---------------------------------------------------------------------------

def gate_prefix_invariance(L, repo: Path):
    import audit_gates as AG
    out = AG.audit_causality()
    status = 'pass' if out.get('status') == 'pass' else 'fail'
    return {
        'status': status,
        'source': 'audit_gates.audit_causality',
        'checks': ['label shift', 'prefix invariance f(x[:T\']) == f(x)[:T\']',
                   'strict readability gate pos > last_tok',
                   'incremental decode vs full forward'],
        'variants': out.get('variants'),
        'failures': out.get('failures'),
    }


# ---------------------------------------------------------------------------
# Gate 2: future-token perturbation, forward AND backward.
# ---------------------------------------------------------------------------

def gate_future_perturbation_and_gradient(L, repo: Path):
    import torch
    arms = list(FACTORIAL_ARMS) + [DENSE_ISOLATION_ARM]
    fails = []
    evidence = []
    V, T = 3, 96
    for arm in arms:
        L.set_seed(0)
        ids = torch.randint(0, 128, (V, T + 1), dtype=torch.long,
                            generator=torch.Generator().manual_seed(11))
        _sentinel = torch.randint(0, 128, (V, T), dtype=torch.long,
                                  generator=torch.Generator().manual_seed(12))
        model = _tiny_model(L, arm)
        with torch.inference_mode():
            full = model(ids[:, :T])
        row = {'arm': arm}
        worst_fwd = 0.0
        worst_grad = 0.0
        for cut in (24, 48, 72):
            pert = ids[:, :T].clone()
            pert[:, cut:] = _sentinel[:, cut:]
            with torch.inference_mode():
                out_pert = model(pert)
            diff = float((out_pert[:, :cut] - full[:, :cut]).abs().max())
            worst_fwd = max(worst_fwd, diff)
            if diff > 1e-4:
                fails.append(f'{arm}: future perturbation changed prefix '
                             f'outputs before token {cut} (max |d|={diff:.3e})')
            # Backward leak: gradient of a PREFIX output w.r.t. the input
            # embedding rows at/after the cut must be exactly zero.
            model.zero_grad(set_to_none=True)
            store = {}

            def _hook(_mod, _inp, out):
                out.retain_grad()
                store['emb'] = out
                return None

            handle = model.tok.register_forward_hook(_hook)
            try:
                h = model.trunk(ids[:, :T])
                gfuture = torch.autograd.grad(h[:, :cut].sum(), store['emb'])[0]
            finally:
                handle.remove()
            gmax = float(gfuture[:, cut:].abs().max()) if gfuture.shape[0] else 0.0
            worst_grad = max(worst_grad, gmax)
            if gmax != 0.0:
                fails.append(f'{arm}: future-gradient leak at cut={cut} '
                             f'(max |grad|={gmax:.3e} must be exactly 0)')
        row.update({'max_forward_dev': worst_fwd, 'max_future_grad': worst_grad})
        evidence.append(row)
        del model
    return {
        'status': 'fail' if fails else 'pass',
        'checks': ['forward: perturbing tokens at/after cut leaves prefix outputs unchanged',
                   'backward: grad(prefix output, future input embeddings) == 0 exactly'],
        'arms': arms,
        'per_arm': evidence,
        'failures': fails,
    }


# ---------------------------------------------------------------------------
# Gate 3: all-block reference equivalence.
# ---------------------------------------------------------------------------

def gate_all_block_reference_equivalence(L, repo: Path):
    import torch
    fails = []
    evidence = []
    V, T = 3, 96
    # (a) batched forward == per-row unbatched forward (row-chunking contract)
    for arm in FACTORIAL_ARMS:
        L.set_seed(0)
        ids = torch.randint(0, 128, (V, T), dtype=torch.long,
                           generator=torch.Generator().manual_seed(13))
        model = _tiny_model(L, arm)
        with torch.inference_mode():
            batched = model(ids)
            rows = torch.cat([model(ids[i:i + 1]) for i in range(V)], dim=0)
        diff = float((batched - rows).abs().max())
        evidence.append({'arm': arm, 'check': 'batched_vs_per_row',
                         'max_abs_dev': diff})
        if diff > 1e-5:
            fails.append(f'{arm}: batched forward differs from per-row '
                         f'unbatched forward (max |d|={diff:.3e})')
        del model
    # (b) every-block-selected equivalence: at this geometry the whole block
    # roster is readable and smaller than topk=32, so csa_fixed_rope's
    # selection is clamped to all blocks (lightning_indexer k=min(topk,B)).
    # The explicitly-all-blocks arm must then produce the same outputs from
    # the same weights: the clamp path and the declared-all path agree.
    L.set_seed(0)
    ids = torch.randint(0, 128, (2, 64), dtype=torch.long,
                        generator=torch.Generator().manual_seed(14))
    a = _tiny_model(L, 'csa_fixed_rope', seq=64)
    b = _tiny_model(L, 'csa_fixed_rope_allblocks', seq=64)
    with torch.inference_mode():
        oa, ob = a(ids), b(ids)
    diff = float((oa - ob).abs().max())
    evidence.append({'check': 'topk_clamped_equals_allblocks_arm',
                     'arm_a': 'csa_fixed_rope', 'arm_b': 'csa_fixed_rope_allblocks',
                     'max_abs_dev': diff})
    if diff > 1e-5:
        fails.append(f'all-blocks equivalence: csa_fixed_rope (topk clamped '
                     f'to all {16} blocks) differs from csa_fixed_rope_allblocks '
                     f'(max |d|={diff:.3e})')
    return {
        'status': 'fail' if fails else 'pass',
        'checks': ['batched forward == per-row unbatched forward (row-chunking contract, design_notes 6.1)',
                   'every-block-selected: topk-clamped arm == explicit all-blocks arm from the same weights'],
        'arms': list(FACTORIAL_ARMS) + ['csa_fixed_rope_allblocks'],
        'per_arm': evidence,
        'failures': fails,
    }


# ---------------------------------------------------------------------------
# Gate 4: fp32 and bf16 forward/backward.
# ---------------------------------------------------------------------------

def gate_fp32_and_bf16_forward_backward(L, repo: Path):
    import torch
    fails = []
    evidence = []
    V, T = 2, 96
    tgts = torch.randint(0, 128, (V, T), dtype=torch.long,
                         generator=torch.Generator().manual_seed(15))
    for arm in FACTORIAL_ARMS:
        row = {'arm': arm}
        ids = torch.randint(0, 128, (V, T), dtype=torch.long,
                           generator=torch.Generator().manual_seed(16))
        # fp32 forward + backward
        L.set_seed(0)
        m1 = _tiny_model(L, arm)
        ce1 = m1.ce_loss(ids, tgts, 128)
        ce1.backward()
        gmax1 = max((float(p.grad.abs().max()) if p.grad is not None else 0.0)
                    for p in m1.parameters())
        if not math.isfinite(gmax1) or not math.isfinite(float(ce1)):
            fails.append(f'{arm}: fp32 loss/gradient not finite (ce={float(ce1)}, '
                         f'max|g|={gmax1})')
        row['fp32'] = {'ce': float(ce1), 'max_abs_grad': gmax1}
        # fp32 determinism across a re-run of the same weights
        with torch.inference_mode():
            f1 = m1.trunk(ids)
            f1b = m1.trunk(ids)
        det = float((f1 - f1b).abs().max())
        row['fp32']['forward_rerun_max_dev'] = det
        if det != 0.0:
            fails.append(f'{arm}: fp32 forward is not deterministic across a '
                         f're-run (max |d|={det:.3e})')
        # bf16 autocast forward + backward (CPU autocast; the same numeric
        # family the CUDA training path uses)
        L.set_seed(0)
        m2 = _tiny_model(L, arm)
        try:
            with torch.autocast('cpu', dtype=torch.bfloat16):
                ce2 = m2.ce_loss(ids, tgts, 128)
            ce2.backward()
            gmax2 = max((float(p.grad.abs().max()) if p.grad is not None else 0.0)
                        for p in m2.parameters())
            with torch.autocast('cpu', dtype=torch.bfloat16):
                f2 = m2.trunk(ids)
        except Exception as exc:  # honest failure, not a silent skip
            fails.append(f'{arm}: bf16 autocast forward/backward raised '
                         f'{type(exc).__name__}: {exc}')
            row['bf16'] = {'error': f'{type(exc).__name__}: {exc}'}
            evidence.append(row)
            continue
        if not math.isfinite(gmax2) or not math.isfinite(float(ce2)):
            fails.append(f'{arm}: bf16 loss/gradient not finite (ce={float(ce2)}, '
                         f'max|g|={gmax2})')
        rel = float((f2.float() - f1).abs().max() / (f1.abs().max() + 1e-12))
        row['bf16'] = {'ce': float(ce2), 'max_abs_grad': gmax2,
                       'forward_rel_dev_vs_fp32': rel,
                       'tolerance_table': {'fp32_abs': TOL_FP32,
                                           'bf16_rel': TOL_BF16_REL}}
        if rel > TOL_BF16_REL:
            fails.append(f'{arm}: bf16 forward deviates from fp32 beyond the '
                         f'declared tolerance (rel={rel:.3e} > {TOL_BF16_REL})')
        evidence.append(row)
        del m1, m2
    return {
        'status': 'fail' if fails else 'pass',
        'checks': ['fp32 forward+backward, finite loss and gradients',
                   'fp32 forward deterministic across a same-weights re-run',
                   'bf16 autocast forward+backward, finite loss and gradients',
                   'bf16-vs-fp32 forward deviation within the declared tolerance table'],
        'arms': list(FACTORIAL_ARMS),
        'per_arm': evidence,
        'failures': fails,
    }


# ---------------------------------------------------------------------------
# Gate 5: duplicate selection and overlap semantics.
# ---------------------------------------------------------------------------

def gate_duplicate_selection_and_overlap(L, repo: Path):
    import torch
    fails = []
    evidence = []
    # (a) duplicate-freeness of the selected-block table under adversarial
    # all-tie scores (stable top-k: ties resolve to the earliest indices,
    # never the same index twice).
    n, B, nIH, hd, kv, d_model = 24, 12, 4, 8, 32, 64
    L.set_seed(0)
    H = torch.zeros(n, d_model)
    comp_kv = torch.randn(B, kv)
    W_DQ = torch.zeros(d_model, nIH * hd)
    W_DK = torch.zeros(kv, nIH * hd)
    W_w = torch.zeros(nIH, d_model)
    last_tok = torch.arange(B, dtype=torch.long) * 2  # block j readable after 2j
    box = []
    m, idx, soft = L.lightning_indexer(H, comp_kv, last_tok, W_DQ, W_DK, W_w,
                                       nIH, topk=6, return_mask=True,
                                       out_valid=box)
    valid = box[0]
    k = idx.shape[1]
    dup = 0
    bad_causal = 0
    for r in range(idx.shape[0]):
        lane = idx[r][valid[r]]
        if len(set(lane.tolist())) != int(lane.numel()):
            dup += 1
            fails.append(f'all-tie scores: row {r} selected duplicate blocks '
                         f'{sorted(lane.tolist())} in its VALID lanes')
        if bool((last_tok[lane.long()] >= r).any()):
            bad_causal += 1
            fails.append(f'all-tie scores: row {r} selected a block whose '
                         f'last_tok >= pos (non-causal selection)')
    row_true = m.sum(dim=1) if m is not None else None
    evidence.append({'check': 'all_tie_topk_uniqueness', 'k': int(k),
                     'rows_with_duplicate_valid_lanes': dup,
                     'rows_with_noncausal_selection': bad_causal,
                     'note': 'invalid lanes carry a masked 0 sentinel and are '
                             'excluded; only valid lanes count as selections'})
    # (b) the readable mask never includes the block that CONTAINS the query
    # position (the open/current block is served by the per-token window
    # branch; a pooled read of it would double-count those tokens).
    pos = torch.arange(24)
    last = torch.arange(12) * 2
    readable = L.block_readable(pos, last)
    # position i lives in open block ceil((i+1)/2)-ish; the strict
    # inequality pos > last_tok already forbids pos == last_tok; additionally
    # verify pos > last_tok excludes the block whose last_tok >= pos.
    violating = int(((last[None, :] >= pos[:, None]) & readable).sum())
    if violating != 0:
        fails.append('readable mask includes a block whose last_tok >= pos '
                     '(open/current block would be read twice: pooled + window)')
    evidence.append({'check': 'open_block_never_pooled',
                     'violating_cells': violating})
    # (c) strict inequality, direct.
    want = pos[:, None] > last[None, :]
    if not torch.equal(readable, want):
        fails.append('block_readable != strict (pos > last_tok) table')
    evidence.append({'check': 'strict_inequality_table',
                    'equals_strict': bool(torch.equal(readable, want))})
    return {
        'status': 'fail' if fails else 'pass',
        'checks': ['top-k selected-block table has no duplicate index per row '
                   '(adversarial all-tie scores, stable ordering)',
                   'the open/current block is never read as a pooled block '
                   '(strict pos > last_tok), so window + block reads cannot '
                   'double-count a token',
                   'block_readable equals the strict inequality table'],
        'evidence': evidence,
        'failures': fails,
    }


# ---------------------------------------------------------------------------
# Gate 6: pure-HCA layer registry.
# ---------------------------------------------------------------------------

def gate_pure_hca_layer_registry(L, repo: Path):
    import v7_supp as V7
    fails = []
    evidence = []
    for variant, want_dynamic in (('hca_fixed_rope', False),
                                 ('hca_dynamic_rope', True)):
        cfgs = L.make_layer_cfgs(4, variant)
        kinds = [str(getattr(c, 'kind', '')) for c in cfgs]
        dyn = [bool(getattr(c, 'dynamic', False)) for c in cfgs]
        ok = all(k == 'hca' for k in kinds) and all(d == want_dynamic for d in dyn)
        evidence.append({'variant': variant, 'layer_kinds': kinds,
                         'dynamic_flags': dyn, 'pure_hca': ok})
        if not ok:
            fails.append(f'{variant}: not a pure HCA arm (kinds={kinds}, '
                         f'dynamic={dyn})')
    # Negative control: the hybrid alternation must NOT be pure HCA.
    cfgs = L.make_layer_cfgs(4, 'hybrid_fixed_rope')
    kinds = [str(getattr(c, 'kind', '')) for c in cfgs]
    hybrid_is_pure = all(k == 'hca' for k in kinds)
    evidence.append({'variant': 'hybrid_fixed_rope', 'layer_kinds': kinds,
                     'pure_hca': bool(hybrid_is_pure),
                     'note': 'negative control: the hybrid alternation is not '
                             'pure HCA and cannot answer the HCA hypothesis'})
    if hybrid_is_pure:
        fails.append('hybrid_fixed_rope unexpectedly resolved to pure HCA '
                     '(the negative control is broken)')
    if not (hasattr(V7, 'PURE_HCA_VARIANTS')
            and set(V7.PURE_HCA_VARIANTS) == {'hca_fixed_rope', 'hca_dynamic_rope'}
            and hasattr(V7, 'FACTORIAL_VARIANTS')
            and set(V7.FACTORIAL_VARIANTS) == set(FACTORIAL_ARMS)):
        fails.append('v7_supp pure-HCA/factorial registry lists are missing or '
                     'changed; the gate runner and the registry disagree')
    return {
        'status': 'fail' if fails else 'pass',
        'checks': ['hca_fixed_rope / hca_dynamic_rope: kind=hca in EVERY layer '
                   'with the declared segmentation',
                   'hybrid_fixed_rope verified NOT pure HCA (negative control)',
                   'v7_supp PURE_HCA_VARIANTS / FACTORIAL_VARIANTS registry lists intact'],
        'evidence': evidence,
        'failures': fails,
    }


# ---------------------------------------------------------------------------
# Gate 7: eager vs kernel-path parity (ref backend on CPU).
# ---------------------------------------------------------------------------

def gate_eager_and_kernel_parity(L, repo: Path):
    import torch
    import csa_kernels as CK
    fails = []
    evidence = []
    dev = torch.device('cpu')
    backends = ['ref']
    if dev.type == 'cuda' and CK.triton_importable():
        backends.append('triton')
    worst_op = 0.0
    for impl in backends:
        tol = 1e-2 if impl == 'triton' else 1e-3
        for seed in (0, 1):
            res = CK.selfcheck_operators(device=dev, dtype=torch.float32,
                                         impl=impl, seed=seed, verbose=False)
            worst_k = max(res, key=res.get)
            worst = res[worst_k]
            worst_op = max(worst_op, worst)
            evidence.append({'impl': impl, 'seed': seed,
                             'worst_operator': worst_k,
                             'max_rel_err': float(worst), 'tolerance': tol})
            if worst >= tol:
                fails.append(f'operator parity impl={impl} seed={seed}: '
                             f'{worst_k}={worst:.3e} >= tol {tol}')
    for impl in backends:
        for variant in ('csa_fixed', 'csa_dynamic', 'hybrid_dynamic'):
            gmax_tol = 5e-2 if impl == 'triton' else 1e-3
            res = CK.selfcheck_model(variant=variant, impl=impl, seed=0,
                                     T=64, B=2, device=dev, verbose=False)
            ok = (res['ce_abs_diff'] < 1e-5 and res['grad_norm_rel'] < 1e-4
                  and res['grad_max_rel'] < gmax_tol)
            evidence.append({'impl': impl, 'model_variant': variant,
                             'ce_abs_diff': res['ce_abs_diff'],
                             'grad_norm_rel': res['grad_norm_rel'],
                             'grad_max_rel': res['grad_max_rel'], 'ok': bool(ok)})
            if not ok:
                fails.append(f'model parity impl={impl} {variant}: {res}')
    return {
        'status': 'fail' if fails else 'pass',
        'checks': ['pool / indexer / attention operators: eager vs reference '
                   'twin (values AND gradients), ref backend, fp32, seeds 0/1',
                   'model-level eager-vs-ref forward loss + full parameter '
                   'gradients for csa_fixed / csa_dynamic / hybrid_dynamic'],
        'backends_tested': backends,
        'device': str(dev),
        'triton_disclosure': 'Triton kernels and the CUDA G0 benchmark are '
                             'GPU-only; a CPU artifact records the ref backend '
                             'only and cannot certify the Triton path',
        'evidence': evidence,
        'failures': fails,
    }


GATE_RUNNERS = {
    'prefix_invariance': gate_prefix_invariance,
    'future_perturbation_and_gradient': gate_future_perturbation_and_gradient,
    'all_block_reference_equivalence': gate_all_block_reference_equivalence,
    'fp32_and_bf16_forward_backward': gate_fp32_and_bf16_forward_backward,
    'duplicate_selection_and_overlap_tests': gate_duplicate_selection_and_overlap,
    'pure_hca_layer_registry_verified': gate_pure_hca_layer_registry,
    'eager_and_kernel_parity': gate_eager_and_kernel_parity,
}


def run_gates(study_id: str, repo: Path, *, device_note: str = 'cpu') -> dict:
    if set(GATE_RUNNERS) != set(GATE_KEYS):
        raise RuntimeError('internal gate list mismatch: the runner table and '
                           'GATE_KEYS disagree; update both together')
    L = _repo_import(repo)
    import torch
    artifact = {
        'schema_version': SCHEMA_VERSION,
        'tool': TOOL_NAME,
        'study_id': str(study_id),
        'status': None,
        'generated_at_utc': _dt.datetime.now(_dt.timezone.utc).isoformat(),
        'device': device_note,
        'torch': torch.__version__,
        'cuda_available': bool(torch.cuda.is_available()),
        'code_stamp': L.code_stamp(),
        'arms': {'factorial': list(FACTORIAL_ARMS),
                 'dense_isolation': DENSE_ISOLATION_ARM},
        'gates': {},
        'notes': [
            'Correctness gates on the algorithm only: no training, no '
            'test-split access, no evidence that any method works.',
            'A CPU artifact certifies the eager paths and the csa_kernels '
            'torch-reference backend; the Triton/CUDA parity layer requires '
            'a GPU run of this same tool.',
        ],
    }
    failures = []
    for key in GATE_KEYS:
        try:
            res = GATE_RUNNERS[key](L, repo)
        except Exception as exc:  # a crashed gate is a failed gate, never a skip
            res = {'status': 'fail',
                   'failures': [f'{type(exc).__name__}: {exc}']}
        artifact['gates'][key] = res
        if res.get('status') != 'pass':
            failures.append(key)
    artifact['status'] = 'ok' if not failures else 'fail'
    artifact['failed_gates'] = failures
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--study-id', default=None,
                        help='study id the artifact certifies (binds the '
                             'artifact to the design record; default reads it '
                             'from --design)')
    parser.add_argument('--design', type=Path, default=None,
                        help='optional design JSON to read the study id from')
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parent,
                        help='repository root holding exp_lib / v7_supp')
    parser.add_argument('--out', type=Path, default=Path('analysis_followup/algorithmic_gates.json'),
                        help='output artifact path (parent dirs are created)')
    args = parser.parse_args()
    study_id = args.study_id
    if study_id is None and args.design is not None:
        try:
            study_id = json.loads(args.design.read_text(encoding='utf-8')).get('study_id')
        except Exception:
            study_id = None
    if not study_id or not str(study_id).strip():
        print('INVALID: --study-id (or a readable --design with a study_id) '
              'is required — an artifact that cannot be bound to a plan '
              'certifies nothing', file=sys.stderr)
        return 2
    artifact = run_gates(str(study_id), args.repo)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=1, sort_keys=False) + '\n',
                   encoding='utf-8')
    print(f'[gates] study={artifact["study_id"]} device={artifact["device"]} '
          f'torch={artifact["torch"]}')
    for key in GATE_KEYS:
        res = artifact['gates'][key]
        line = f'[gates]   {key:<42} {res.get("status")}'
        fl = res.get('failures') or []
        print(line + (f'  ({len(fl)} failure(s))' if fl else ''))
        for f in fl[:3]:
            print(f'[gates]       - {f}')
    print(f'[gates] status: {artifact["status"]}  '
          f'failed: {artifact["failed_gates"] or "none"}')
    print(f'[gates] artifact: {out}')
    return 0 if artifact['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
