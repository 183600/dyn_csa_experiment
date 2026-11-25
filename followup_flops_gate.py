#!/usr/bin/env python3
"""Measured-FLOPs acceptance gate + measurement harness for the follow-up design.

FOLLOWUP_CONFIRMATORY_PROTOCOL.md (compute_matching) requires, for the
FLOPs-matched Dense arm:

    "for FLOPs matching, tune only predeclared capacity knobs and require the
     measured total training-FLOPs ratio to be within 1% of the target" and
     "If the profiler cannot reliably count an operator (e.g. custom Triton
     kernels), disclose the unsupported operators and refuse to call the arm
     FLOPs-matched".

review-15 closed the ANALYTICAL planning half (`exp_lib.flops_matched_mlp_ratio`).
This tool is the measurement/acceptance half:

``measure``  profile K measured training steps (after warm-up) of a registry
             variant with `torch.utils.flop_counter.FlopCounterMode` — the
             op-dispatch-level counter that DOES attribute
             `scaled_dot_product_attention` (the CPU/CUDA profilers' flops
             registry covers only mm/bmm/addmm and silently drops the
             attention kernels).  A torch.profiler pass runs alongside purely
             to produce the disclosure table of ops that carry time but no
             profiler-level FLOPs attribution, so the artifact shows exactly
             what the dispatch counter had to cover.
``gate``     compare two measurement artifacts: PASS only when both
             measurements certify on the declared device, the repeat block
             reproduces the count exactly, and the total-training-FLOPs ratio
             is within tolerance.

Fail-closed properties (each refuses the FLOPS_MATCHED claim):

- a custom CSA kernel path active in the process is INVISIBLE to op-level
  counting (it never dispatches aten ops) — the measurement refuses instead
  of undercounting; run it with the eager path or reconcile independently;
- a non-reproducing repeat block (op-level counting is shape-deterministic,
  so any difference means the computation itself changed) refuses;
- a measurement on a device other than the declared certifying device never
  certifies (CPU runs are smoke evidence, recorded as such);
- totals are forward+backward training-step FLOPs per the SAME counter for
  both arms — never wall-clock time (the protocol forbids that proxy).

SDPA counting note: FlopCounterMode applies the FULL attention formula
4*B*H*T_q*T_k*D regardless of a causal mask.  The repo's analytical model
(`exp_lib.analytic_attn_flops_token`) uses the same non-halved convention,
so the measured and analytical cross-check columns share one accounting
convention.  Comparisons are still arm-vs-arm under one counter, which is
what the 1% gate is about.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

import exp_lib as L
import v7_supp  # noqa: F401 — installs the *_rope variant registry

SCHEMA_VERSION = 1
TOOL_NAME = 'followup_flops_gate.py'
# Ops that carry real compute but get no flops attribution from
# torch.profiler(with_flops=True); their self-time is disclosed per artifact.
_PROFILER_UNSUPPORTED_PATTERNS = (
    'scaled_dot_product', 'flash_attention', 'efficient_attention',
    'triton', 'csa_kernels', 'block_level_gather',
)


def _pos_int(value, name):
    if isinstance(value, bool):
        return None
    try:
        if int(value) != value:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    n = int(value)
    return n if n > 0 else None


def _safe_float(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return f if math.isfinite(f) else None


def _kernel_path_state():
    """(module_available, kernel_enabled) for the disclosure/refusal logic."""
    try:
        available = L._CK is not None
    except AttributeError:
        available = False
    enabled = False
    if available:
        try:
            enabled = bool(L._CK.kernel_enabled())
        except Exception:
            enabled = False
    return available, enabled


def _train_step(model, ids, tgt, vocab, opt):
    loss = model(ids, targets=tgt, vocab=vocab)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return float(loss.detach())


def _measure_variant(args):
    problems = []
    geometry_keys = ('vocab', 'd', 'n_layers', 'n_heads', 'd_head', 'seq', 'batch')
    geometry = {}
    for key in geometry_keys:
        val = _pos_int(getattr(args, key), key)
        if val is None:
            problems.append(f'--{key} must be a positive exact integer')
        geometry[key] = int(getattr(args, key) if val is None else val)
    mr = _safe_float(getattr(args, 'mlp_ratio', None))
    if mr is None or mr <= 0.0:
        problems.append('--mlp-ratio must be a positive finite number')
    steps = _pos_int(getattr(args, 'steps', None), 'steps')
    warmup = _pos_int(getattr(args, 'warmup', None), 'warmup')
    if steps is None:
        problems.append('--steps must be a positive exact integer')
    if warmup is None:
        problems.append('--warmup must be a positive exact integer')
    device = str(getattr(args, 'device', 'cpu'))
    try:
        torch.device(device)
    except Exception:
        problems.append(f'--device {device!r} is not a valid torch device')
    if problems:
        return {'schema_version': SCHEMA_VERSION, 'tool': TOOL_NAME,
                'mode': 'measure', 'status': 'refused', 'problems': problems,
                'note': 'fail-closed: a malformed measurement config certifies nothing'}

    mod_avail, kern_on = _kernel_path_state()
    if kern_on:
        return {'schema_version': SCHEMA_VERSION, 'tool': TOOL_NAME,
                'mode': 'measure', 'status': 'refused',
                'problems': ['the custom CSA kernel path is active in this process; '
                             'custom kernels never dispatch aten ops, so op-level '
                             'FLOP counting would undercount the arm — run the '
                             'measurement on the eager path (disable the kernel '
                             'module) or reconcile the kernel FLOPs independently'],
                'kernel_module_available': mod_avail, 'kernel_path_active': True,
                'note': 'refuse_flops_matched_claim'}

    try:
        cfgs = [__import__('copy').deepcopy(c)
                for c in L.make_layer_cfgs(int(args.n_layers), str(args.variant))]
    except Exception as exc:
        return {'schema_version': SCHEMA_VERSION, 'tool': TOOL_NAME,
                'mode': 'measure', 'status': 'refused',
                'problems': [f'make_layer_cfgs({args.variant!r}) failed: '
                             f'{type(exc).__name__}: {exc}'],
                'note': 'fail-closed: unknown variant'}

    torch.manual_seed(20261126)
    model = L.SmallGPT(int(args.vocab), int(args.d), int(args.n_layers),
                       int(args.n_heads), int(args.d_head), int(args.seq),
                       cfgs, mlp_ratio=float(args.mlp_ratio)).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-8)
    ids = torch.randint(0, int(args.vocab), (int(args.batch), int(args.seq)), device=device)
    tgt = torch.randint(0, int(args.vocab), (int(args.batch), int(args.seq)), device=device)

    for _ in range(int(args.warmup)):
        _train_step(model, ids, tgt, int(args.vocab), opt)

    from torch.utils.flop_counter import FlopCounterMode
    mode = FlopCounterMode(display=False)
    with mode:
        for _ in range(int(args.steps)):
            _train_step(model, ids, tgt, int(args.vocab), opt)
    flops_total = int(mode.get_total_flops())

    # Reproducibility: op-level counting is shape-deterministic, so a second
    # identical block must reproduce the count EXACTLY.  Any difference means
    # the computation itself changed between blocks (cache state, dynamic
    # selection instability) and the number is not a measurement of a fixed
    # recipe — refuse instead of averaging it away.
    mode2 = FlopCounterMode(display=False)
    with mode2:
        for _ in range(int(args.steps)):
            _train_step(model, ids, tgt, int(args.vocab), opt)
    flops_repeat = int(mode2.get_total_flops())
    reproducible = (flops_total == flops_repeat)

    # Disclosure table: ops with self-time but no torch.profiler flops
    # attribution (the attention kernels).  Purely informative — the
    # dispatch-level counter above is the measurement — but the artifact must
    # show what the profiler alone would have missed.
    profiler_unsupported = {}
    try:
        from torch.profiler import ProfilerActivity, profile
        activities = [ProfilerActivity.CPU]
        if device.startswith('cuda'):
            activities.append(ProfilerActivity.CUDA)
        with profile(activities=activities, with_flops=True) as prof:
            _train_step(model, ids, tgt, int(args.vocab), opt)
        for evt in prof.key_averages():
            key = str(evt.key)
            if evt.flops > 0 or 'Profiler' in key:
                continue
            if any(p in key for p in _PROFILER_UNSUPPORTED_PATTERNS):
                t = float(getattr(evt, 'self_cpu_time_total', 0.0) or 0.0)
                if device.startswith('cuda'):
                    t = float(getattr(evt, 'self_device_time_total', 0.0) or 0.0)
                profiler_unsupported[key] = round(t / 1000.0, 3)  # usec -> ms
    except Exception as exc:  # pragma: no cover - environment dependent
        profiler_unsupported = {'<profiler failed>': f'{type(exc).__name__}: {exc}'}

    per_step = flops_total / float(int(args.steps))
    tokens_per_step = int(args.batch) * int(args.seq)
    # Analytical cross-check column: attention core + MLP per token per layer
    # (projections / head / pooling excluded for every arm, forward only) —
    # the same accounting as the frozen compute_budget_disclosure.  Recorded
    # for cross-checking, never the acceptance number.
    try:
        an_tok_layer = float(L.analytic_train_flops_token(
            cfgs[0], float(args.mlp_ratio), d=int(args.d), seq_len=int(args.seq),
            n_heads=int(args.n_heads), d_head=int(args.d_head)))
    except Exception:
        an_tok_layer = None
    certifying_device = str(getattr(args, 'certifying_device', 'cuda'))
    certifies = device.startswith(certifying_device)
    status = 'ok' if (reproducible and flops_total > 0) else 'refused'
    notes = ['measured with torch.utils.flop_counter.FlopCounterMode at the aten '
             'dispatch level (mm/bmm/addmm/SDPA); forward+backward+optimizer step '
             'per training step; deterministic shape-based counting']
    if not reproducible:
        notes.append('the repeat block did NOT reproduce the count — the measurement '
                     'is not of a fixed computation and certifies nothing')
    if not certifies:
        notes.append(f'device {device!r} is not the certifying device '
                     f'{certifying_device!r}: smoke evidence only')
    if an_tok_layer is not None:
        notes.append('analytic_attn_mlp_forward_total is the planning cross-check '
                     '(projections/head excluded), never the acceptance number; '
                     'measured totals legitimately exceed it')
    declared_steps = getattr(args, 'declared_total_steps', None)
    art = {
        'schema_version': SCHEMA_VERSION,
        'tool': TOOL_NAME,
        'mode': 'measure',
        'status': status,
        'problems': [] if status == 'ok' else [
            'the FLOPs count is not reproducible across identical measured blocks'
            if not reproducible else 'the measured FLOPs count is not positive'],
        'variant': str(args.variant),
        'mlp_ratio': float(args.mlp_ratio),
        'geometry': geometry,
        'device': device,
        'torch_version': str(torch.__version__),
        'warmup_steps': int(args.warmup),
        'measured_steps': int(args.steps),
        'flops_total_measured': flops_total,
        'flops_repeat_block': flops_repeat,
        'reproducible': bool(reproducible),
        'flops_per_step': per_step,
        'tokens_per_step': tokens_per_step,
        'analytic_attn_mlp_forward_per_token_layer': an_tok_layer,
        'analytic_attn_mlp_forward_total': (
            an_tok_layer * tokens_per_step * int(args.n_layers)
            if an_tok_layer is not None else None),
        'declared_total_steps': (int(declared_steps)
                                 if _pos_int(declared_steps, 'dts') is not None else None),
        'profiler_unsupported_ops_ms': profiler_unsupported,
        'kernel_module_available': mod_avail,
        'kernel_path_active': bool(kern_on),
        'certifying_device': certifying_device,
        'certifies': bool(certifies),
        'unsupported_operator_policy': 'refuse_flops_matched_claim',
        'note': ' | '.join(notes),
    }
    return art


def _gate(ref, cand, tolerance, certifying_device, allow_noncertifying):
    problems = []
    for label, art in (('reference', ref), ('candidate', cand)):
        if not isinstance(art, dict):
            problems.append(f'{label}: artifact is not a JSON object')
            continue
        if art.get('tool') != TOOL_NAME:
            problems.append(f'{label}: artifact was not produced by {TOOL_NAME}')
        if art.get('status') != 'ok':
            problems.append(f'{label}: measurement status is {art.get("status")!r}')
        if art.get('kernel_path_active'):
            problems.append(f'{label}: measured with the custom kernel path active '
                            f'(invisible to op-level counting) — refuse_flops_matched_claim')
        if art.get('reproducible') is not True:
            problems.append(f'{label}: measurement is not reproducible')
        if not allow_noncertifying and art.get('certifies') is not True:
            problems.append(f'{label}: measurement does not certify on '
                            f'{certifying_device!r} (device {art.get("device")!r})')
        # The ratio is computed from flops_per_step (x declared_total_steps),
        # so THAT is the number that must be a positive finite measurement —
        # flops_total_measured is checked too, and an inconsistency between
        # the two is itself a refusal (the artifact must describe one run).
        t = _safe_float(art.get('flops_total_measured'))
        f = _safe_float(art.get('flops_per_step'))
        if f is None or f <= 0:
            problems.append(f'{label}: flops_per_step is not a positive finite '
                            f'number: {art.get("flops_per_step")!r}')
        if t is None or t <= 0:
            problems.append(f'{label}: flops_total_measured is not a positive finite '
                            f'number: {art.get("flops_total_measured")!r}')
        if t is not None and f is not None and t > 0 and f > 0:
            steps = _pos_int(art.get('measured_steps'), 'ms')
            if steps is not None:
                per_from_total = t / float(steps)
                if abs(per_from_total - f) > 1e-6 * max(1.0, f):
                    problems.append(f'{label}: flops_total_measured/measured_steps '
                                    f'({per_from_total:.6g}) is inconsistent with '
                                    f'flops_per_step ({f:.6g})')

    def _total(art):
        steps = _pos_int(art.get('declared_total_steps'), 'dts') if isinstance(art, dict) else None
        per_step = _safe_float(art.get('flops_per_step')) if isinstance(art, dict) else None
        if per_step is None or per_step <= 0:
            return None
        return per_step * (steps if steps is not None else 1)

    ref_total, cand_total = _total(ref), _total(cand)
    ratio = None
    if ref_total is not None and cand_total is not None and ref_total > 0:
        ratio = cand_total / ref_total
        if abs(ratio - 1.0) > float(tolerance):
            problems.append(f'total-training-FLOPs ratio {ratio:.6f} exceeds the '
                            f'{float(tolerance):.2%} tolerance')
    else:
        problems.append('cannot compute the FLOPs ratio from the artifacts '
                        '(missing/invalid per-step totals)')
    verdict = 'REFUSED' if problems else 'FLOPS_MATCHED'
    out = {
        'schema_version': SCHEMA_VERSION,
        'tool': TOOL_NAME,
        'mode': 'gate',
        'verdict': verdict,
        'problems': problems,
        'tolerance': float(tolerance),
        'certifying_device': certifying_device,
        'allow_noncertifying': bool(allow_noncertifying),
        'reference_total_training_flops': ref_total,
        'candidate_total_training_flops': cand_total,
        'ratio': ratio,
        'note': ('PASS requires: both measurements ok, reproducible, certifying on '
                 f'{certifying_device!r}, kernel path inactive, and |ratio-1| <= '
                 f'{float(tolerance):.2%}; wall-clock time is never a FLOPs proxy'),
    }
    if allow_noncertifying and verdict == 'FLOPS_MATCHED':
        out['verdict'] = 'FLOPS_MATCHED_SMOKE_ONLY'
        out['note'] += ('; SMOKE_ONLY: at least one measurement ran on a '
                        'non-certifying device and cannot support the '
                        'equal-compute acceptance')
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='cmd', required=True)

    m = sub.add_parser('measure', help='profile a variant and write a measurement JSON')
    m.add_argument('--variant', required=True, help='registry variant id')
    m.add_argument('--mlp-ratio', required=True, type=float,
                   help='MLP hidden ratio for this arm (use exp_lib.flops_matched_mlp_ratio '
                        'to solve the FLOPs-matched dense knob)')
    m.add_argument('--vocab', type=int, default=8192)
    m.add_argument('--d', type=int, default=256)
    m.add_argument('--n-layers', type=int, default=6)
    m.add_argument('--n-heads', type=int, default=8)
    m.add_argument('--d-head', type=int, default=32)
    m.add_argument('--seq', type=int, default=512)
    m.add_argument('--batch', type=int, default=12)
    m.add_argument('--steps', type=int, default=5)
    m.add_argument('--warmup', type=int, default=3)
    m.add_argument('--device', default='cpu')
    m.add_argument('--certifying-device', default='cuda',
                   help='the device family the protocol acceptance requires (default cuda)')
    m.add_argument('--declared-total-steps', default=None, type=int,
                   help='registered total training steps for this arm; the gate scales '
                        'per-step FLOPs by it to compare total training FLOPs')
    m.add_argument('--out', type=Path, required=True)

    g = sub.add_parser('gate', help='compare two measurement artifacts (1%% gate)')
    g.add_argument('reference', type=Path)
    g.add_argument('candidate', type=Path)
    g.add_argument('--tolerance', type=float, default=0.01)
    g.add_argument('--certifying-device', default='cuda')
    g.add_argument('--allow-noncertifying', action='store_true',
                   help='evaluate the ratio anyway, but the verdict is marked SMOKE_ONLY '
                        'and cannot support the equal-compute acceptance')
    g.add_argument('--out', type=Path, default=None)

    args = parser.parse_args(argv)
    if args.cmd == 'measure':
        art = _measure_variant(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(art, indent=2) + '\n', encoding='utf-8')
        if art['status'] != 'ok':
            print(f'REFUSED ({len(art["problems"])} problem(s)); artifact written to {args.out}')
            for p in art['problems']:
                print(f'- {p}')
            return 1
        print(f'[flops-measure] variant={art["variant"]} device={art["device"]} '
              f'per_step={art["flops_per_step"]:.6g} reproducible={art["reproducible"]} '
              f'certifies={art["certifies"]}')
        print(f'[flops-measure] profiler-unattributed compute ops (disclosed): '
              f'{list(art["profiler_unsupported_ops_ms"])[:4]}')
        print(f'[flops-measure] artifact: {args.out}')
        return 0

    try:
        ref = json.loads(args.reference.read_text(encoding='utf-8'))
        cand = json.loads(args.candidate.read_text(encoding='utf-8'))
    except Exception as exc:
        print(f'INVALID: cannot read artifacts: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
    verdict = _gate(ref, cand, args.tolerance, args.certifying_device,
                    args.allow_noncertifying)
    text = json.dumps(verdict, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + '\n', encoding='utf-8')
    print(text)
    return 0 if verdict['verdict'] == 'FLOPS_MATCHED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
