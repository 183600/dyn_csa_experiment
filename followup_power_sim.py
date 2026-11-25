#!/usr/bin/env python3
"""Simulation-based power analysis for the follow-up factorial design (pre-freeze).

FOLLOWUP_CONFIRMATORY_PROTOCOL.md ("Primary hypotheses and statistics")
requires, BEFORE any test-split access:

    "Before test access, run a simulation-based power analysis using plausible
     paired effect sizes and the observed validation-set variance.  Increase
     the seed count if the target power is not reached."

`v12_supp._power_enumeration` is an EXACT enumeration over the OBSERVED
validation magnitudes — a post-run design aid that cannot answer "will THIS
seed count detect the effect we pre-registered".  This tool is the missing
complement: it simulates the pre-registered four-hypothesis Holm family under
declared plausible paired effect sizes and declared seed-level variance
(or a bootstrap over the observed validation deltas), using the EXACT
statistics the confirmatory analysis will use — `exp_lib.exact_signflip_p`
plus `exp_lib.holm_adjust` — and reports per-hypothesis power, the
minimum-power floor, and (with `--seed-sweep`) the minimum seed count that
clears the declared target power.

The artifact JSON binds to `validate_followup_design.py`: setting
`statistics.power_simulation_completed = true` now REQUIRES
`statistics.power_simulation_artifact` to name an artifact produced by this
tool whose study id, hypothesis family, seed count, target power, alpha and
minimum simulated power all match the plan — the boolean alone stops being
a rubber stamp.

This is a design aid, not a guarantee: the projection is only as honest as
the declared effect sizes and the variance estimate.  It must be computed on
validation data / planning assumptions BEFORE any test-split access, and the
spec inputs (means, sds / observed deltas) must be recorded next to the
artifact so the projection is auditable.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import exp_lib as L
from validate_followup_design import REQUIRED_HYPOTHESES

SCHEMA_VERSION = 1
TOOL_NAME = 'followup_power_sim.py'
# Monte-Carlo floors: fewer than 200 replicates certifies nothing (the Wilson
# CI at 200 reps is wider than any practical power target band); more than
# 200k is a waste on a design aid that runs before training.
MIN_MC_REPLICATES = 200
MAX_MC_REPLICATES = 200000
# The protocol's own floor: a ready plan needs >= 12 predeclared seeds, so the
# recommendation logic never suggests less; single-n simulation below 12 is
# still allowed for curve drawing.
MIN_PLAN_SEEDS = 12
_BAD_NUMS = (bool, np.bool_, np.datetime64, np.timedelta64, complex, np.complexfloating)


def _as_exact_int(value):
    """Exact non-negative integer or None (booleans are flags, not counts)."""
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        if int(value) != value:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    n = int(value)
    return n if n >= 0 else None


def _finite_level(value, lo, hi, hi_inclusive=False):
    """A real scalar in (lo, hi) — or (lo, hi] when inclusive — or None.

    Refuses booleans/complex/strings: a familywise alpha of 0.05 must be a
    real number, not a flag or a stringified JSON artifact.
    """
    if isinstance(value, _BAD_NUMS):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    ok = (lo < f <= hi) if hi_inclusive else (lo < f < hi)
    if not math.isfinite(f) or not ok:
        return None
    return f


def _spec_problems(spec):
    """Fail-closed spec validation. Returns a list of human-readable problems."""
    problems = []
    if not isinstance(spec, dict):
        return ['spec root is not a JSON object']
    if spec.get('schema_version') != SCHEMA_VERSION:
        problems.append(f'schema_version must equal {SCHEMA_VERSION}')
    sid = spec.get('study_id')
    if not isinstance(sid, str) or not sid.strip():
        problems.append('study_id must be a non-empty string')
    hyps = spec.get('hypotheses')
    if not isinstance(hyps, list) or {str(h) for h in (hyps or [])} != set(REQUIRED_HYPOTHESES):
        problems.append(f'hypotheses must be exactly the four pre-registered ids: '
                        f'{sorted(REQUIRED_HYPOTHESES)}')
    alpha = _finite_level(spec.get('alpha'), 0.0, 0.05, hi_inclusive=True)
    if alpha is None:
        problems.append('alpha must be a finite number in (0, 0.05] (the familywise level '
                        'the confirmatory analysis will use)')
    target = _finite_level(spec.get('target_power'), 0.0, 1.0)
    if target is None:
        problems.append('target_power must be a finite number in (0, 1)')
    n = _as_exact_int(spec.get('n_seeds'))
    if n is None or n < 4:
        problems.append('n_seeds must be an exact integer >= 4 (smaller panels cannot even '
                        'be simulated meaningfully)')
    reps = _as_exact_int(spec.get('mc_replicates'))
    if reps is None or not (MIN_MC_REPLICATES <= reps <= MAX_MC_REPLICATES):
        problems.append(f'mc_replicates must be an exact integer in '
                        f'[{MIN_MC_REPLICATES}, {MAX_MC_REPLICATES}]')
    if _as_exact_int(spec.get('rng_seed')) is None:
        problems.append('rng_seed must be an exact non-negative integer (the simulation is '
                        'deterministic only when the seed is recorded)')
    em = spec.get('effect_model')
    if not isinstance(em, dict):
        problems.append('effect_model must be a JSON object')
        return problems
    dist = em.get('dist')
    if dist not in ('normal', 'bootstrap'):
        problems.append("effect_model.dist must be 'normal' (declared mean/sd) or "
                        "'bootstrap' (resample the observed validation deltas)")
    hyp_set = set(REQUIRED_HYPOTHESES)
    if dist == 'normal':
        means, sds = em.get('mean'), em.get('sd')
        for label, table, strict_pos in (('mean', means, False), ('sd', sds, True)):
            if not isinstance(table, dict) or set(map(str, table or {})) != hyp_set:
                problems.append(f'effect_model.{label} must map each of the four '
                                f'hypothesis ids to a number')
                continue
            for k, v in table.items():
                f = _finite_level(v, -math.inf, math.inf) if not isinstance(v, _BAD_NUMS) else None
                if f is None or math.isnan(f):
                    problems.append(f'effect_model.{label}[{k}] is not a finite real number')
                elif strict_pos and f <= 0.0:
                    problems.append(f'effect_model.sd[{k}] must be > 0 (a zero variance panel '
                                    f'would certify power from a point mass)')
    elif dist == 'bootstrap':
        obs = em.get('observed_deltas')
        if not isinstance(obs, dict) or set(map(str, obs or {})) != hyp_set:
            problems.append('effect_model.observed_deltas must map each of the four '
                            'hypothesis ids to a list of validation-panel deltas')
        else:
            for k, vals in obs.items():
                if not isinstance(vals, list) or len(vals) < 4:
                    problems.append(f'effect_model.observed_deltas[{k}] needs at least 4 '
                                    f'finite deltas to resample')
                    continue
                for v in vals:
                    if isinstance(v, _BAD_NUMS) or not isinstance(v, (int, float)) \
                            or not math.isfinite(float(v)):
                        problems.append(f'effect_model.observed_deltas[{k}] contains a '
                                        f'non-measurement value: {v!r}')
                        break
    return problems


# Batched enumeration limit matches exp_lib's own exact-enumeration ceiling
# (2^20 sign patterns per replicate row, chunked so the (patterns x
# replicates) statistic block stays a few million entries); larger panels
# fall back to the per-row exp_lib call (MITM to n=40, saddlepoint beyond)
# — same contract, slower path.
_BATCH_ENUM_MAX_N = 20


def _batch_exact_signflip_p(D):
    """Exact two-sided sign-flip p-values for a matrix of replicate panels.

    Mirrors the ``exp_lib.exact_signflip_p`` contract row-for-row: the
    power-of-two rescale, the ``1e-12 * mean|d|`` tie slack and the exact
    2^n enumeration are the same — only the loop is vectorised across
    replicates so a thousands-of-replicates simulation is one pass instead
    of thousands of Python calls.  Floating summation order differs from the
    scalar path by ~1e-16 relative; only exact ties could flip, and the tie
    slack exists precisely to absorb that.
    """
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[1] == 0 or D.shape[0] == 0:
        return np.full(max(D.shape[0] if D.ndim == 2 else 0, 0), float('nan'))
    R, n = D.shape
    if n > _BATCH_ENUM_MAX_N:
        return np.asarray([L.exact_signflip_p(list(row))['p'] for row in D])
    amax = np.abs(D).max(axis=1)
    _, e = np.frexp(amax)
    Dn = D * np.ldexp(np.ones_like(amax), -e)[:, None]
    obs = np.abs(Dn.mean(axis=1))
    tol = 1e-12 * (np.abs(Dn).sum(axis=1) / n)
    P = 1 << n
    positions = np.arange(n, dtype=np.int64)
    cnt = np.zeros(R, dtype=np.int64)
    # Chunk the sign-pattern axis so the (patterns, replicates) statistic
    # block stays a few million entries regardless of R.
    chunk = max(1, min(P, (1 << 22) // max(R, 1)))
    for off in range(0, P, chunk):
        bits = np.arange(off, min(off + chunk, P), dtype=np.int64)
        S = np.where((bits[:, None] & (1 << positions)) != 0, 1.0, -1.0)
        # (signs . deltas) is the SUM; the exp_lib statistic is the MEAN of
        # the signed deltas, so divide the pattern block by n here.
        stats = np.abs(S @ Dn.T) / float(n)
        cnt += (stats >= (obs - tol)[None, :]).sum(axis=0)
    return cnt / float(P)


def simulate_power(spec, n_seeds=None):
    """Run the Monte-Carlo power simulation. Returns the artifact dict.

    Pure function of (spec, n_seeds): the RNG is re-seeded from
    `spec['rng_seed']` every call, so the same spec always yields the same
    artifact and `--seed-sweep` points are reproducible independently.
    """
    problems = _spec_problems(spec)
    if problems:
        return {'schema_version': SCHEMA_VERSION, 'tool': TOOL_NAME,
                'status': 'invalid_spec', 'problems': problems, 'note':
                'fail-closed: a malformed planning spec certifies nothing'}
    n = int(n_seeds if n_seeds is not None else spec['n_seeds'])
    if n < 4:
        return {'schema_version': SCHEMA_VERSION, 'tool': TOOL_NAME,
                'status': 'invalid_spec',
                'problems': [f'n_seeds={n} < 4: cannot simulate a meaningful panel'],
                'note': 'fail-closed'}
    alpha = float(spec['alpha'])
    target = float(spec['target_power'])
    reps = int(spec['mc_replicates'])
    model = {'dist': spec['effect_model']['dist']}
    if model['dist'] == 'normal':
        model['mean'] = {str(k): float(v) for k, v in spec['effect_model']['mean'].items()}
        model['sd'] = {str(k): float(v) for k, v in spec['effect_model']['sd'].items()}
    else:
        model['observed_deltas'] = {str(k): [float(x) for x in v]
                                    for k, v in spec['effect_model']['observed_deltas'].items()}
    hyps = sorted(REQUIRED_HYPOTHESES)
    rng = np.random.default_rng(int(spec['rng_seed']))
    reps = int(spec['mc_replicates'])
    # Draw the whole (replicates x n_seeds) delta matrix per hypothesis up
    # front, then compute every replicate's exact sign-flip p in ONE batched
    # pass (_batch_exact_signflip_p) and replay the Holm step-down per
    # replicate with exp_lib.holm_adjust — the same statistics the
    # confirmatory analysis will use, without thousands of scalar calls.
    p_mats = {}
    for h in hyps:
        if model['dist'] == 'normal':
            D = rng.normal(model['mean'][h], model['sd'][h], size=(reps, n))
        else:
            D = rng.choice(np.asarray(model['observed_deltas'][h], dtype=float),
                           size=(reps, n), replace=True)
        p_mats[h] = _batch_exact_signflip_p(D)
    rejects = {h: 0 for h in hyps}
    n_all = 0
    n_any = 0
    # A replicate with any non-finite p-value counts as "family untestable"
    # for the joint readouts (the confirmatory analysis would not confirm
    # from a corrupted panel); continuous draws make this measure-zero, the
    # bootstrap panel can contain exact zeros (handled by the exact test).
    for r in range(reps):
        holm = L.holm_adjust([(h, float(p_mats[h][r])) for h in hyps], alpha=alpha)
        rej = {h: bool(holm[h]['reject']) for h in hyps}
        for h in hyps:
            rejects[h] += int(rej[h])
        n_all += int(all(rej.values()))
        n_any += int(any(rej.values()))
    m = len(hyps)

    def _wilson(k, r):
        p = k / r
        z = 1.959963984540054
        denom = 1.0 + z * z / r
        centre = (p + z * z / (2.0 * r)) / denom
        half = z * math.sqrt(p * (1.0 - p) / r + z * z / (4.0 * r * r)) / denom
        return [max(0.0, centre - half), min(1.0, centre + half)]

    powers = {h: {'power': rejects[h] / reps,
                  'ci95_mc': _wilson(rejects[h], reps),
                  'rejects': int(rejects[h])} for h in hyps}
    min_power = min(powers[h]['power'] for h in hyps)
    analytic = {f'q={q:.2f}': L.signflip_power(n, m, q, alpha)
                for q in (0.80, 0.85, 0.90, 0.95)}
    em_record = {'dist': model['dist']}
    if model['dist'] == 'normal':
        em_record['mean'] = model['mean']
        em_record['sd'] = model['sd']
    else:
        em_record['observed_deltas'] = model['observed_deltas']
    return {
        'schema_version': SCHEMA_VERSION,
        'tool': TOOL_NAME,
        'status': 'ok',
        'study_id': str(spec['study_id']),
        'hypotheses': hyps,
        'test': 'exact_two_sided_sign_flip',
        'correction': 'holm',
        'alpha': alpha,
        'n_seeds': n,
        'mc_replicates': reps,
        'rng_seed': int(spec['rng_seed']),
        'effect_model': em_record,
        'powers': powers,
        'power_all_four': n_all / reps,
        'power_any': n_any / reps,
        'min_power': float(min_power),
        'target_power': target,
        'meets_target': bool(min_power >= target),
        'analytic_projection_conservative': analytic,
        'seed_sweep': None,
        'recommended_n_seeds': None,
        'note': ('design aid computed on declared effect sizes / validation variance '
                 'BEFORE test access; not a guarantee — the projection is only as honest '
                 'as the declared inputs, which are recorded in effect_model'),
    }


def run_seed_sweep(spec, ns):
    """Simulate each n; attach the sweep and the minimum seed recommendation."""
    base = simulate_power(spec)
    if base['status'] != 'ok':
        return base
    sweep = []
    for n in ns:
        art = simulate_power(spec, n_seeds=int(n))
        if art['status'] != 'ok':
            return art
        sweep.append({'n_seeds': int(n),
                      'powers': {h: round(art['powers'][h]['power'], 6) for h in art['hypotheses']},
                      'min_power': art['min_power'],
                      'meets_target': art['meets_target']})
    rec = next((s['n_seeds'] for s in sweep
                if s['meets_target'] and s['n_seeds'] >= MIN_PLAN_SEEDS), None)
    base['seed_sweep'] = sweep
    base['recommended_n_seeds'] = rec
    if rec is None:
        base['note'] += ('; no swept n clears the target power at the declared effect '
                         'sizes — increase the seed count beyond the sweep or revise '
                         'the design, do not shrink the target')
    return base


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('spec', type=Path, help='planning spec JSON (see module docstring)')
    parser.add_argument('--out', type=Path, default=None,
                        help='artifact JSON path (default: <spec dir>/power_sim.json)')
    parser.add_argument('--seed-sweep', default=None,
                        help="comma-separated extra seed counts to project, e.g. "
                             "'12,16,20,24,32'; the artifact records the minimum n "
                             "that clears the target power")
    args = parser.parse_args(argv)
    try:
        spec = json.loads(args.spec.read_text(encoding='utf-8'))
    except Exception as exc:
        print(f'INVALID: cannot read spec {args.spec}: {type(exc).__name__}: {exc}',
              file=sys.stderr)
        return 2
    if args.seed_sweep is not None:
        try:
            ns = [int(x) for x in str(args.seed_sweep).split(',') if str(x).strip()]
        except ValueError:
            print(f'INVALID: --seed-sweep must be comma-separated integers: '
                  f'{args.seed_sweep!r}', file=sys.stderr)
            return 2
        if not ns or any(n < 4 for n in ns):
            print('INVALID: every swept seed count must be >= 4', file=sys.stderr)
            return 2
        art = run_seed_sweep(spec, ns)
    else:
        art = simulate_power(spec)
    out = args.out or args.spec.parent / 'power_sim.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(art, indent=2, sort_keys=False) + '\n', encoding='utf-8')
    if art['status'] != 'ok':
        print(f'INVALID SPEC ({len(art["problems"])} problem(s)); artifact written to {out}')
        for p in art['problems']:
            print(f'- {p}')
        return 1
    print(f'[power-sim] study={art["study_id"]} n_seeds={art["n_seeds"]} '
          f'reps={art["mc_replicates"]} alpha={art["alpha"]}')
    for h in art['hypotheses']:
        pw = art['powers'][h]
        print(f'[power-sim]   {h:52s} power={pw["power"]:.3f} '
              f'ci95=({pw["ci95_mc"][0]:.3f}, {pw["ci95_mc"][1]:.3f})')
    print(f'[power-sim] min_power={art["min_power"]:.3f} target={art["target_power"]:.2f} '
          f'meets_target={art["meets_target"]}')
    if art.get('recommended_n_seeds') is not None:
        print(f'[power-sim] recommended_n_seeds={art["recommended_n_seeds"]}')
    print(f'[power-sim] artifact: {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
