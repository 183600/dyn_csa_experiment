#!/usr/bin/env python3
"""Open-block panel driver: maximum block length × current-open-block reading.

The main experiment is a single controlled matrix on the dynamic CSA model:

    max_block_tokens  ∈ {32, 64, 128, None}
    open_block_mode   ∈ {closed, prefix_summary, prefix_singletons}

with the eight cells of the main matrix (32/64 only closed, 128 and None
with all three reading rules).  `None` means "dynamic boundaries only": no
forced cut from a maximum length.  All arms share the same position-table
capacity (`pool_position_capacity=129`, truncation) and the same training
recipe; only the two declared factors change.

Commands
--------
python open_block_supp.py smoke
    Zero-dataset end-to-end check: all eight cells, tiny model, a few steps,
    finite loss, open-block metadata present.

python open_block_supp.py pilot  [--seeds 0] [--steps 100]
    One short seed per cell; excluded from the formal statistics.

python open_block_supp.py main   [--seeds 0,1,2,3] [--steps 1500] [--max-hours H]
    The 8-cell × 4-seed panel.  Resumable: a cell whose record is current
    (same code/recipe and a finite PPL) is skipped.

python open_block_supp.py aux    [--seeds ...]
    The outside-window auxiliary panel: the same three reading rules at
    128/None, but the prefix is constructed from U_t = P_t \\ W_t only.

python open_block_supp.py extended [--seeds ...]
    Contingency panel: all 4 maximum-block values x all 3 reading rules
    (12 cells).  Only run this if the main matrix shows an interaction trend;
    it is not part of the 8-cell primary panel.

python open_block_supp.py report [--outdir ...]
    Paired per-seed deltas, interaction effects, exact sign-flip p-values,
    Holm adjustment, and a Markdown report.  No GPU needed.

Artifacts
---------
<outdir>/summary.json          per-cell records (resumable)
<outdir>/run_manifest.json     recipe / code / data provenance
<analysis_open_block>/stats.json + OPEN_BLOCK_REPORT.md
"""
import argparse
import json
import math
import os
import sys
import time

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except (ValueError, OSError):
            pass

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import numpy as np
import torch

import exp_lib as L


MAIN_MATRIX = (
    (32, 'closed', 'full_prefix'),
    (64, 'closed', 'full_prefix'),
    (128, 'closed', 'full_prefix'),
    (128, 'summary', 'full_prefix'),
    (128, 'singletons', 'full_prefix'),
    (None, 'closed', 'full_prefix'),
    (None, 'summary', 'full_prefix'),
    (None, 'singletons', 'full_prefix'),
)
EXTENDED_MATRIX = tuple(
    (mb, mode, 'full_prefix')
    for mb in L.OPEN_BLOCK_MAX_VALUES
    for mode in L.OPEN_BLOCK_MODES_SHORT
)
AUX_MATRIX = (
    (128, 'closed', 'outside_window'),
    (128, 'summary', 'outside_window'),
    (128, 'singletons', 'outside_window'),
    (None, 'closed', 'outside_window'),
    (None, 'summary', 'outside_window'),
    (None, 'singletons', 'outside_window'),
)

BASE_CFG = dict(
    dataset='wikitext', d=256, n_layers=6, n_heads=8, d_head=32,
    seq_len=512, batch_size=12, n_train_tokens=8_000_000, steps=1500,
    lr=0.0003, weight_decay=0.1, warmup=50, comp_lambda=0.05,
    delta_lr_mult=10.0, eval_every=250, eval_subset=128, val_seqs=512,
    seeds=[0, 1, 2, 3], outdir='results_open_block',
)


def _variant(max_block, mode, scope):
    return L.open_block_variant_name(max_block, mode, scope)


def _recipe(cfg, *, matrix, outdir):
    d = dict(cfg)
    d['matrix'] = [list(x) for x in matrix]
    d['outdir'] = outdir
    d['code'] = L.code_stamp()
    d['semantics'] = L.CODE_SEMANTICS
    return d


def _run_fingerprint(cfg, *, matrix, outdir):
    r = _recipe(cfg, matrix=matrix, outdir=outdir)
    # The fingerprint identifies the training recipe, not which cells of the
    # matrix happen to be scheduled in this invocation: adding a seed or an
    # auxiliary cell must not invalidate already-finished cells.
    r.pop('seeds', None)
    r.pop('outdir', None)
    r.pop('matrix', None)
    return json.dumps(r, sort_keys=True, separators=(',', ':'))


def _atomic_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + f'.tmp{os.getpid()}'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def _load_summary(outdir):
    p = os.path.join(outdir, 'summary.json')
    if os.path.exists(p):
        try:
            with open(p, encoding='utf-8') as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {}
        except Exception as e:
            print(f'[open-block] WARNING: unreadable summary {p} ({type(e).__name__}: {e}); starting a fresh one')
    return {}


def _save_summary(outdir, summary):
    _atomic_json(os.path.join(outdir, 'summary.json'), summary)


def _is_current(rec, fp, *, data_fp=None, eval_fp=None):
    if not isinstance(rec, dict):
        return False
    if rec.get('run_fingerprint') != fp:
        return False
    if rec.get('code') != L.code_stamp():
        return False
    # The recipe fingerprint identifies the settings, not the bytes those
    # settings were run on.  A rebuilt cache with the same dataset/config would
    # otherwise be silently skipped and its old PPL reused.
    if data_fp is not None and rec.get('data_fp') != data_fp:
        return False
    if eval_fp is not None and rec.get('eval_fp') != eval_fp:
        return False
    ppl = rec.get('ppl')
    if not (isinstance(ppl, (int, float)) and not isinstance(ppl, bool)
            and math.isfinite(float(ppl)) and float(ppl) > 0):
        return False
    # The analysis consumes token-pooled NLL, not only PPL.  Skipping a record
    # that has no exact n_tokens/nll_sum would make it look "current" forever
    # while `analyse` excludes it as unusable (a resume that can never repair
    # the cell).  A fractional token count is not a denominator either.
    return _nll_per_token(rec) is not None


def _synthetic_data(cfg, vocab=97, n_train=40000, val_rows=8):
    rng = np.random.RandomState(cfg.get('seed', 0) if 'seed' in cfg else 0)
    train = rng.randint(0, vocab, size=int(n_train)).astype(np.int64)
    val = rng.randint(0, vocab, size=(int(val_rows), int(cfg['seq_len']))).astype(np.int64)
    return train, val, int(vocab)


def _wikitext_data(cfg):
    train, val, vocab, _tok, val_bnd = L.load_wikitext(
        int(cfg['seq_len']), int(cfg['n_train_tokens']),
        vocab_size=8192, val_seqs=int(cfg.get('val_seqs', 512)))
    return train, val, int(vocab), val_bnd


def _run_one(cfg, *, variant, seed, train_ids, val_batch, vocab, val_bnd,
             outdir, fp, device, max_hours=None, prov=None):
    key = f'{variant}::seed{int(seed)}'
    summary = _load_summary(outdir)
    _data_fp = L._ids_fp(train_ids) if train_ids is not None else None
    _eval_fp = L._ids_fp(np.ascontiguousarray(val_batch).reshape(-1)) if val_batch is not None else None
    if _is_current(summary.get(key), fp, data_fp=_data_fp, eval_fp=_eval_fp):
        print(f'[open-block] skip {key} (current record)')
        return summary[key]
    if max_hours is not None and max_hours <= 0:
        print(f'[open-block] stopping before {key}: --max-hours exhausted')
        return None
    print(f'\n===== open-block {key} =====')
    t0 = time.time()
    rec = L.train_variant(
        variant, train_ids, val_batch, vocab, seed=int(seed),
        d=int(cfg['d']), n_layers=int(cfg['n_layers']),
        n_heads=int(cfg['n_heads']), d_head=int(cfg['d_head']),
        seq_len=int(cfg['seq_len']), batch_size=int(cfg['batch_size']),
        steps=int(cfg['steps']), lr=float(cfg['lr']),
        weight_decay=float(cfg['weight_decay']), warmup=int(cfg['warmup']),
        comp_lambda=float(cfg['comp_lambda']),
        delta_lr_mult=float(cfg.get('delta_lr_mult', 10.0)),
        eval_every=int(cfg.get('eval_every', 0)),
        eval_subset=int(cfg.get('eval_subset', 128)),
        val_bnd=val_bnd, device=device, mlp_ratio=4.0,
        measure_eff=True, eager_batch=False,
        amp_dtype=cfg.get('amp_dtype', 'auto'))
    rec = dict(rec)
    rec.pop('_model', None)
    rec['key'] = key
    rec['variant'] = variant
    rec['seed'] = int(seed)
    rec['run_fingerprint'] = fp
    rec['code'] = L.code_stamp()
    rec['semantics'] = L.CODE_SEMANTICS
    rec['elapsed_s'] = time.time() - t0
    if prov is None:
        prov = L.experiment_provenance(cfg, train_ids=train_ids, val_batch=val_batch, vocab=vocab)
    rec['provenance'] = prov
    rec['data_version'] = (prov.get('data') or {}).get('data_version')
    rec['corpus_fp'] = (prov.get('data') or {}).get('corpus_fp')
    rec['hyperparams'] = L.run_hyperparameters(
        cfg, seed=int(seed), d=int(cfg['d']), n_layers=int(cfg['n_layers']),
        n_heads=int(cfg['n_heads']), d_head=int(cfg['d_head']),
        vocab=int(vocab), mlp_ratio=4.0,
        amp_dtype=rec.get('amp_dtype', 'auto'),
        eager_batch=rec.get('eager_batch', False),
        data_version=rec.get('data_version'))
    rec['hyperparams']['open_block_config'] = rec.get('open_block_config')
    summary[key] = rec
    _save_summary(outdir, summary)
    _nll_tok = _nll_per_token(rec)
    _losses = rec.get('losses') if isinstance(rec.get('losses'), list) else []
    L.log_experiment_event(outdir, {
        'kind': 'cell', 'tracking_schema': 2, 'status': 'ok', 'key': key,
        'variant': variant, 'seed': int(seed), 'run_cfg': fp,
        'code': rec.get('code'), 'provenance': prov,
        'hyperparams': rec.get('hyperparams'), 'data_fp': rec.get('data_fp'),
        'val_fp': rec.get('eval_fp'), 'data_version': rec.get('data_version'),
        'corpus_fp': rec.get('corpus_fp'), 'rng_fp': rec.get('rng_fp'),
        'started_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'elapsed_s': rec.get('elapsed_s'),
        'metrics': {'ppl': rec.get('ppl'), 'nll_per_token': _nll_tok,
                    'params': rec.get('params'), 'steps': rec.get('steps'),
                    'tokens_seen': rec.get('tokens_seen'),
                    'train_time_s': rec.get('train_time_s'),
                    'final_loss_smoothed': rec.get('final_loss_smoothed')},
        'curve_summary': {'n_points': len(_losses),
                          'first': _losses[0] if _losses else None,
                          'last': _losses[-1] if _losses else None,
                          'best': min(_losses) if _losses else None},
        'losses': _losses, 'ppl_history': rec.get('ppl_history'),
    })
    print(f'[open-block] {key}: NLL/token='
          f'{(f"{_nll_tok:.6f}" if _nll_tok is not None else "n/a")} PPL={rec["ppl"]:.4f}')
    return rec


def run_panel(cfg, matrix, *, device, synthetic=False, max_hours=None, label='main'):
    outdir = cfg['outdir']
    os.makedirs(outdir, exist_ok=True)
    fp = _run_fingerprint(cfg, matrix=matrix, outdir=outdir)
    cfg = dict(cfg)
    cfg['seeds'] = L.normalize_seed_list(cfg.get('seeds'))
    if synthetic:
        train, val, vocab = _synthetic_data(cfg)
        val_bnd = None
    else:
        train, val, vocab, val_bnd = _wikitext_data(cfg)
    variants = list(dict.fromkeys(_variant(mb, m, sc) for mb, m, sc in matrix))
    planned = [f'{v}::seed{s}' for s in cfg['seeds'] for v in variants]
    first_seed = int(cfg['seeds'][0])
    _prov_cfg = dict(cfg)
    if synthetic:
        _prov_cfg['dataset'] = 'synthetic'
    _prov_cfg['data_extra'] = {'open_block_matrix': [list(x) for x in matrix],
                               'synthetic': bool(synthetic)}
    prov = L.experiment_provenance(_prov_cfg, train_ids=train, val_batch=val, vocab=vocab)
    _data = prov.get('data') or {}
    manifest = {
        'kind': 'open_block_panel', 'label': label, 'tracking_schema': 2,
        'status': 'preflight',
        'code': L.code_stamp(), 'semantics': L.CODE_SEMANTICS,
        'recipe': _recipe(cfg, matrix=matrix, outdir=outdir),
        'run_fingerprint': fp,
        'provenance': prov,
        'data': {'train_len': int(len(train)), 'val_shape': list(np.asarray(val).shape),
                 'vocab': int(vocab), 'synthetic': bool(synthetic),
                 'train_fp': _data.get('train_fp'), 'val_fp': _data.get('val_fp'),
                 'corpus_fp': _data.get('corpus_fp'),
                 'data_version': _data.get('data_version')},
        'started_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'cells': planned, 'n_planned': len(planned),
    }

    def _save_manifest(**updates):
        manifest.update(updates)
        _atomic_json(os.path.join(outdir, 'run_manifest.json'), manifest)

    _save_manifest()
    # The same pre-flight contract as exp_lib.run(): every RNG source pinned,
    # the actual loader's label/vocab/repeat contract checked, and the real
    # architecture overfitted on one fixed batch before any panel cell is
    # trained.  Failed evidence is persisted and refuses the panel.
    try:
        L.set_seed(first_seed)
        seed_contract = L.seed_contract_check(seed=first_seed)
        seed_contract.update({'run_fingerprint': fp, 'provenance': prov})
        _atomic_json(os.path.join(outdir, 'seed_contract.json'), seed_contract)
        if not seed_contract.get('passed'):
            _save_manifest(status='failed', stage='rng_seed_contract',
                           error=str(seed_contract.get('diagnosis')))
            raise RuntimeError('open-block seed contract check FAILED: '
                               + str(seed_contract.get('diagnosis')))
        data_contract = L.batch_contract_check(
            train, cfg['seq_len'], cfg['batch_size'], device,
            seed=first_seed, vocab=vocab)
        data_contract.update({'run_fingerprint': fp, 'provenance': prov})
        _atomic_json(os.path.join(outdir, 'data_contract.json'), data_contract)
        if not data_contract.get('passed'):
            _save_manifest(status='failed', stage='batch_iterator_contract',
                           error=str(data_contract.get('diagnosis')))
            raise RuntimeError('open-block batch iterator contract check FAILED: '
                               + str(data_contract.get('diagnosis')))
        _probe_seq = L.probe_seq_len(cfg['seq_len'])
        sanity = list(L.overfit_guard(
            variants, seed=first_seed, steps=200, seq_len=_probe_seq,
            batch_size=2, vocab=vocab, train_ids=train,
            arch={'d': int(cfg['d']), 'n_layers': int(cfg['n_layers']),
                  'n_heads': int(cfg['n_heads']), 'd_head': int(cfg['d_head'])},
            mlp_ratio=4.0))
        if not sanity:
            sanity = list(L.overfit_reports_for(
                variants, source='data', data_fp=_data.get('train_fp')))
    except RuntimeError as _fe:
        sanity_doc = {
            'kind': 'small_batch_overfit_sanity_panel',
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'run_fingerprint': fp, 'provenance': prov, 'passed': False,
            'error': f'{_fe}', 'reports': list(L.overfit_reports().values()),
        }
        _atomic_json(os.path.join(outdir, 'overfit_sanity.json'), sanity_doc)
        _save_manifest(status='failed', stage='preflight', error=f'{_fe}')
        raise
    sanity_doc = {
        'kind': 'small_batch_overfit_sanity_panel',
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'run_fingerprint': fp, 'provenance': prov,
        'passed': bool(sanity) and all(bool(r.get('passed')) for r in sanity),
        'reports': sanity,
    }
    _atomic_json(os.path.join(outdir, 'overfit_sanity.json'), sanity_doc)
    _save_manifest(status='running', seed_contract=seed_contract,
                   data_contract=data_contract, sanity=sanity_doc)
    L.log_experiment_event(outdir, {
        'kind': 'panel_start', 'tracking_schema': 2, 'label': label,
        'run_cfg': fp, 'code': L.code_stamp(), 'seeds': list(cfg['seeds']),
        'variants': variants, 'n_cells_planned': len(planned),
        'provenance': prov, 'data_version': _data.get('data_version'),
        'seed_contract': {'passed': bool(seed_contract.get('passed'))},
        'data_contract': {'passed': bool(data_contract.get('passed'))},
        'sanity': {'passed': sanity_doc['passed'], 'n_reports': len(sanity)},
    })
    deadline = None if max_hours is None else time.time() + float(max_hours) * 3600.0
    done = 0
    for seed in cfg['seeds']:
        for mb, mode, scope in matrix:
            variant = _variant(mb, mode, scope)
            remain = None if deadline is None else (deadline - time.time()) / 3600.0
            rec = _run_one(cfg, variant=variant, seed=seed, train_ids=train,
                           val_batch=val, vocab=vocab, val_bnd=val_bnd,
                           outdir=outdir, fp=fp, device=device, max_hours=remain,
                           prov=prov)
            if rec is not None:
                done += 1
    summary = _load_summary(outdir)
    present = {k for k in planned if isinstance(summary.get(k), dict)
               and L.ppl_is_usable((summary.get(k) or {}).get('ppl'))}
    missing = [k for k in planned if k not in present]
    status = 'complete' if not missing else 'incomplete'
    _save_manifest(status=status, finished_at=time.strftime('%Y-%m-%dT%H:%M:%S'),
                   completed_cells=done, n_measured=len(present),
                   missing_cells=missing)
    L.log_experiment_event(outdir, {
        'kind': 'panel_finish', 'tracking_schema': 2, 'label': label,
        'status': status, 'run_cfg': fp, 'code': L.code_stamp(),
        'n_cells_planned': len(planned), 'n_cells_measured': len(present),
        'n_cells_missing': len(missing), 'missing_cells': missing[:20],
        'data_version': _data.get('data_version'),
    })
    print(f'\n[open-block] panel {label}: {done} cell(s) processed, '
          f'{len(present)}/{len(planned)} measured, status={status}; '
          f'summary={os.path.join(outdir, "summary.json")}')
    return outdir


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _nll_per_token(rec):
    # Token counts are exact integers.  float(1.5) would otherwise pass and
    # int() in the arm summary would truncate it to a different denominator,
    # making two different token budgets look like one pooled measurement.
    if not isinstance(rec, dict):
        return None
    n = L._exact_count(rec.get('n_tokens'), minimum=1)
    v = L._finite_scalar(rec.get('nll_sum'))
    if n is None or v is None:
        return None
    return v / n


def _rec_by_cell(summary, run_fingerprint=None, problems=None):
    """Index usable cells, refusing a key that disagrees with the record.

    The summary key is ``<variant>::seed<n>``, but the record itself is the
    authority.  A key/record seed mismatch (or a variant mismatch) would pair
    one seed's arm with another seed's result while the statistics still look
    like a clean paired design, so those cells are excluded and named.
    """
    out = {}
    for key, rec in summary.items():
        if not isinstance(rec, dict):
            continue
        if run_fingerprint is not None and rec.get('run_fingerprint') != run_fingerprint:
            continue
        if not L.ppl_is_usable(rec.get('ppl')):
            continue
        v, sep, s = key.partition('::seed')
        try:
            seed = int(s)
        except Exception:
            if problems is not None:
                problems.append({'key': str(key), 'reason': 'key carries no integer seed'})
            continue
        if not sep:
            if problems is not None:
                problems.append({'key': str(key), 'reason': 'key carries no ::seed suffix'})
            continue
        rec_seed = L._as_seed_int(rec.get('seed'))
        if rec_seed is None:
            if problems is not None:
                problems.append({'key': str(key), 'reason': 'record carries no usable seed'})
            continue
        if rec_seed != seed:
            if problems is not None:
                problems.append({'key': str(key), 'reason': f'record seed {rec_seed} != key seed {seed}'})
            continue
        rec_var = rec.get('variant')
        if rec_var is not None and str(rec_var) != str(v):
            if problems is not None:
                problems.append({'key': str(key), 'reason': f'record variant {rec_var!r} != key variant {v!r}'})
            continue
        nll = _nll_per_token(rec)
        if nll is None:
            if problems is not None:
                problems.append({'key': str(key), 'reason': 'nll/token is missing or non-finite'})
            continue
        out[(v, seed)] = {'record': rec, 'nll': nll, 'ppl': float(rec['ppl'])}
    return out


def _arm_summary(cells):
    """Per-arm n, mean ± SD and token-pooled PPL over the measured seeds.

    The paired contrast table answers "is a − b different?"; this table keeps
    the raw arm level visible (n, seed spread, pooled estimate) so a small
    paired delta is not quoted without the seed-level variability that bounds
    it.
    """
    groups = {}
    for (v, s), cell in cells.items():
        groups.setdefault(v, []).append((int(s), cell))
    out = {}
    for v, rows in sorted(groups.items()):
        rows.sort(key=lambda x: x[0])
        ppls = [float(c['ppl']) for _s, c in rows]
        nlls = [float(c['nll']) for _s, c in rows]
        n_tok = []
        for _s, c in rows:
            r = c.get('record') or {}
            n = L._exact_count(r.get('n_tokens'), minimum=1)
            n_tok.append(n)
        pooled = None
        if all(n is not None for n in n_tok):
            # `nlls` are already per-token values.  Pooling must recover each
            # seed's NLL SUM (nll_per_token * n_tokens) before dividing by the
            # combined token count; summing the per-token means and dividing by
            # the total token count would make the pooled PPL shrink as more
            # seeds (or more tokens) are added.
            _nll = math.fsum(nll * int(n) for nll, n in zip(nlls, n_tok))
            _nt = int(sum(n_tok))
            if _nt > 0:
                pooled = L._safe_exp_ppl(_nll / _nt)
        out[v] = {
            'n_seeds': len(rows),
            'seeds': [s for s, _c in rows],
            'ppl_mean': float(np.mean(ppls)),
            'ppl_std': float(np.std(ppls, ddof=1)) if len(ppls) > 1 else 0.0,
            'nll_per_token_mean': float(np.mean(nlls)),
            'nll_per_token_std': float(np.std(nlls, ddof=1)) if len(nlls) > 1 else 0.0,
            'ppl_pooled': pooled,
            'n_tokens_total': (int(sum(n_tok)) if all(n is not None for n in n_tok) else None),
        }
    return out


def _paired(cells, va, vb, seeds):
    """Paired per-seed deltas that ALSO clear the controlled-comparator gate.

    A seed whose records differ in an undeclared setting is excluded from the
    delta list (and named in `rejected`); including it would let an optimizer /
    schedule / evaluation-slice change masquerade as the block-reading effect.
    """
    ds, ppl_ds, used, rejected = [], [], [], []
    _seen = set()
    for s in seeds:
        # A repeated seed is one draw, not two independent observations.
        if s in _seen:
            continue
        _seen.add(s)
        a, b = cells.get((va, s)), cells.get((vb, s))
        if a is None or b is None:
            continue
        reason = _controlled_reason(a['record'], b['record'])
        if reason is not None:
            rejected.append({'seed': int(s), 'reason': reason})
            continue
        ds.append(float(a['nll'] - b['nll']))
        ppl_ds.append(float(a['ppl'] - b['ppl']))
        used.append(int(s))
    return ds, ppl_ds, used, rejected


def _cell_diagnostics(rec):
    """Aggregate per-layer block/prefix diagnostics for one cell."""
    stats = rec.get('stats') if isinstance(rec, dict) else None
    if not isinstance(stats, dict):
        return None
    keys_mean = ('len_mean', 'len_std', 'avg_len', 'frac_at_min', 'frac_at_max',
                 'prefix_age_mean', 'prefix_age_p95', 'outside_window_len_mean',
                 'outside_window_nonzero_frac', 'prefix_len_mean',
                 'prefix_selected_rate', 'prefix_selected_any_rate',
                 'closed_selected_mean', 'selected_mean', 'candidates_mean',
                 'unused_slots_mean', 'history_displaced_mean',
                 'prefix_attention_mass_mean', 'position_truncated_frac',
                 'position_truncated_frac_of_block_tokens',
                 'position_truncated_tokens', 'position_truncated_blocks',
                 'forced_only', 'natural_only', 'forced_and_natural',
                 'forced_total', 'natural_total', 'len_p50', 'len_p95')
    out = {k: [] for k in keys_mean}
    out['len_max'] = []
    for _layer, d in stats.items():
        if not isinstance(d, dict):
            continue
        for k in keys_mean:
            v = d.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
                out[k].append(float(v))
            ob = d.get('open_block')
            if isinstance(ob, dict):
                v2 = ob.get(k)
                if isinstance(v2, (int, float)) and not isinstance(v2, bool) and math.isfinite(float(v2)):
                    out[k].append(float(v2))
        lm = d.get('len_max')
        if isinstance(lm, (int, float)) and math.isfinite(float(lm)):
            out['len_max'].append(float(lm))
    agg = {}
    for k, vals in out.items():
        if not vals:
            agg[k] = None
        elif k == 'len_max':
            agg[k] = max(vals)
        elif (k.startswith('position_truncated_tokens') or k.startswith('position_truncated_blocks')
              or k in ('forced_only', 'natural_only', 'forced_and_natural',
                       'forced_total', 'natural_total')):
            agg[k] = sum(vals)
        else:
            agg[k] = sum(vals) / len(vals)
    return agg


def _cell_efficiency(rec):
    if not isinstance(rec, dict):
        return None
    eff = rec.get('efficiency') if isinstance(rec.get('efficiency'), dict) else {}
    out = {k: eff.get(k) for k in
           ('prefill_ms_p50', 'prefill_ms_p95', 'tokens_per_s_prefill',
            'prefix_prefill_ms_p50', 'prefix_prefill_ms_p95',
            'peak_mem_allocated_mb', 'peak_mem_reserved_mb', 'status')}
    out['train_tokens_per_s'] = rec.get('train_tokens_per_s')
    out['train_time_s'] = rec.get('train_time_s')
    out['peak_memory'] = rec.get('peak_memory')
    return out if any(v is not None for v in out.values()) else None


def _controlled_reason(a, b):
    """Only the declared experimental factors may differ between two arms."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return 'missing record'
    _ha_raw, _hb_raw = a.get('hyperparams'), b.get('hyperparams')
    if _ha_raw is not None and not isinstance(_ha_raw, dict):
        return 'hyperparams is malformed (not an object)'
    if _hb_raw is not None and not isinstance(_hb_raw, dict):
        return 'hyperparams is malformed (not an object)'
    ha, hb = _ha_raw or {}, _hb_raw or {}
    keys = ('steps', 'tokens_seen', 'n_train_tokens', 'seq_len', 'batch_size',
            'lr', 'weight_decay', 'warmup', 'comp_lambda', 'delta_lr_mult',
            'd', 'n_layers', 'n_heads', 'd_head', 'mlp_ratio', 'vocab_size',
            'dataset', 'data_version', 'val_seqs', 'eval_every', 'eval_subset',
            'code', 'amp_dtype', 'eager_batch')
    for k in keys:
        va, vb = ha.get(k), hb.get(k)
        if va is None and vb is None:
            # Legacy/legacy is the only permissive path: both records omit the
            # setting, so it cannot be shown to differ (and cannot be shown to
            # match either).  A ONE-SIDED value is unverifiable, not a pass.
            continue
        if (va is None) != (vb is None):
            return f'{k} is unverifiable (only one record carries it)'
        if va != vb:
            return f'{k} differs ({va!r} != {vb!r})'
    # The compact `hyperparams` copy pins lr/wd/schedule lengths but NOT the
    # optimizer identity, betas/eps, clipping, LR-schedule shape, augmentation
    # or precision mode.  Those live in the nested/top-level `train_knobs`
    # descriptor.  Comparing it here is what prevents "same lr" from being
    # mistaken for "same optimizer recipe" after a future backend/default
    # change.  Both-absent is the legacy path; a one-sided descriptor is
    # unverifiable, exactly like the hyperparameter keys above.
    _ka = ha.get('train_knobs') if 'train_knobs' in ha else a.get('train_knobs')
    _kb = hb.get('train_knobs') if 'train_knobs' in hb else b.get('train_knobs')
    if (_ka is None) != (_kb is None):
        return 'train_knobs is unverifiable (only one record carries it)'
    if _ka is not None and _ka != _kb:
        return 'train_knobs differs (optimizer / betas / schedule / clipping / augmentation / precision)'
    # The actual loss composition is derived from the built model, not from
    # the variant name.  Open-block cells must not differ here: only max_block
    # and the reading rule are declared factors.
    _oa, _ob = a.get('train_objective'), b.get('train_objective')
    if (_oa is None) != (_ob is None):
        return 'train_objective is unverifiable (only one record carries it)'
    if _oa is not None and _oa != _ob:
        return 'train_objective differs (the objective/regularizer/policy is not the same)'
    # Same seed number is not the same starting point: architectures consume
    # RNG differently.  The shared-trunk init digest is the matched-init gate
    # used by the main panels; a one-sided digest is unverifiable.
    _ia, _ib = a.get('init_core_fp'), b.get('init_core_fp')
    _ha = _ia is not None and str(_ia) != ''
    _hb = _ib is not None and str(_ib) != ''
    if _ha != _hb:
        return 'init_core_fp is unverifiable (only one record carries it)'
    if _ha and str(_ia) != str(_ib):
        return 'init_core_fp differs (the arms did not start from the same shared-trunk initialization)'
    if a.get('code') != b.get('code'):
        return f"code stamp differs ({a.get('code')!r} != {b.get('code')!r})"
    pa, pb = a.get('params'), b.get('params')
    if (pa is None) != (pb is None):
        return 'parameter count is unverifiable (only one record carries it)'
    if pa is not None and pa != pb:
        return f"parameter count differs ({pa!r} != {pb!r})"
    if a.get('run_fingerprint') != b.get('run_fingerprint'):
        return 'run_fingerprint differs (recipe/backend/matrix changed)'
    for k in ('data_fp', 'val_fp', 'eval_fp', 'eval_subset_fp', 'rng_fp',
              'holdout_row_fp'):
        va, vb = a.get(k), b.get(k)
        _ha = va is not None and str(va) != ''
        _hb = vb is not None and str(vb) != ''
        if _ha != _hb:
            return f'{k} is unverifiable (only one record carries it)'
        if _ha and str(va) != str(vb):
            return f'{k} differs ({va!r} != {vb!r})'
    _oa_raw, _ob_raw = a.get('open_block_config'), b.get('open_block_config')
    if _oa_raw is not None and not isinstance(_oa_raw, dict):
        return 'open_block_config is malformed (not an object)'
    if _ob_raw is not None and not isinstance(_ob_raw, dict):
        return 'open_block_config is malformed (not an object)'
    oa, ob = _oa_raw or {}, _ob_raw or {}
    if oa.get('fingerprint') is not None and oa.get('fingerprint') == ob.get('fingerprint'):
        return None
    # The only allowed layer-level differences are max_block/open mode/scope.
    la, lb = oa.get('layers') or [], ob.get('layers') or []
    if not isinstance(la, list) or not isinstance(lb, list):
        return 'open_block_config layers are malformed (not a list)'
    if len(la) != len(lb):
        return 'different layer counts'
    if any(not isinstance(x, dict) for x in la + lb):
        return 'open_block_config layer rows are malformed (not objects)'
    # `open_block_config` carries both names for the same setting; the
    # descriptor must not reject a legitimate max-block contrast just because
    # the alias appears in the layer row.
    allowed = {'max_block', 'max_block_tokens',
               'open_block_mode', 'open_block_scope'}
    for i, (x, y) in enumerate(zip(la, lb)):
        for k in set(x) | set(y):
            if k in allowed:
                continue
            if x.get(k) != y.get(k):
                return f'layer {i} differs in undeclared field {k}'
    return None


def _controlled_interaction(a, b, c, d):
    """Control gate for the 2x2 interaction (a-b at low, c-d at high).

    Each within-cap pair and the two cap groups must share one controlled
    context (data/validation/eval fingerprints, RNG context, budget, ...).
    """
    return (_controlled_reason(a, b)
            or _controlled_reason(c, d)
            or _controlled_reason(a, c)
            or _controlled_reason(b, d))


def _family_holm_floor(ns, m_comparisons):
    """Holm resolution floor for a mixed-n family.

    The first Holm gate is reachable through the BEST-POWERED member, so the
    family floor uses max(n), not min(n).  A missing/zero-n comparison cannot
    carry the family and is ignored (it still occupies its declared m slot).

    Items may be plain seed counts (legacy callers) or result dicts carrying
    ``n`` and ``n_nonzero``.  Exact zero paired differences do not change the
    sign-flip p value but DO lower its attainable resolution, so the
    dict form uses the non-zero pair count (mapping all-zero to 1, floor 1)
    exactly like ``exp_lib.paired_effect_stats``.  Reading only ``n`` here
    would overstate the family's power whenever a contrast has ties.
    """
    used = []
    for item in (ns or []):
        if isinstance(item, dict):
            _n = _as_positive_int(item.get('n'))
            if _n is None:
                continue
            _nn_raw = item.get('n_nonzero')
            # n_nonzero=0 is a REAL measurement (all ties), not a missing
            # value: use the non-negative parser so it maps to floor 1.
            _nn = (L._as_seed_int(_nn_raw)
                   if _nn_raw is not None else None)
            if _nn is None or _nn < 0 or _nn > _n:
                used.append(_n)
            else:
                used.append(_nn if _nn > 0 else 1)
            continue
        _n = _as_positive_int(item)
        if _n is not None:
            used.append(_n)
    m = _as_positive_int(m_comparisons)
    if not used or m is None:
        return None
    return L.holm_min_attainable_p(max(used), m, 0.05)


def _primary_effective_n(results, names):
    """Smallest effective pair count among the measured primary contrasts.

    Exact zero differences keep their n but add no sign resolution, so the
    floor is governed by ``n_nonzero`` (0 maps to 1, p floor 1.0).  A missing
    or out-of-range count falls back to the raw n rather than pretending the
    contrast has more resolution.
    """
    used = []
    for name in (names or []):
        r = (results or {}).get(name)
        if not isinstance(r, dict):
            continue
        n = _as_positive_int(r.get('n'))
        if n is None:
            continue
        nn = L._as_seed_int(r.get('n_nonzero'))
        if nn is not None and 0 <= nn <= n:
            used.append(nn if nn > 0 else 1)
        else:
            used.append(n)
    return min(used) if used else 0


def _as_positive_int(v):
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
        return None
    v = int(v)
    return v if v > 0 else None


def analyse(outdir, *, seeds=None, analysis_dir='analysis_open_block'):
    summary = _load_summary(outdir)
    _fps = {rec.get('run_fingerprint') for rec in summary.values()
            if isinstance(rec, dict) and L.ppl_is_usable(rec.get('ppl'))}
    _fps.discard(None)
    if len(_fps) > 1:
        raise SystemExit(
            f'{outdir}/summary.json mixes {len(_fps)} run fingerprints (e.g. a '
            'pilot and a main panel); refusing to pool them. Use a separate '
            'outdir for the pilot or remove the stale records explicitly.')
    _fp = next(iter(_fps)) if len(_fps) == 1 else None
    _integrity = []
    cells = _rec_by_cell(summary, run_fingerprint=_fp, problems=_integrity)
    if not cells:
        raise SystemExit(f'no usable records in {outdir}/summary.json'
                         + (f' (excluded: {_integrity})' if _integrity else ''))
    avail_seeds = sorted({s for (_, s) in cells})
    _raw_seeds = avail_seeds if seeds is None else seeds
    _parsed = [L._as_seed_int(s) for s in _raw_seeds]
    if any(s is None for s in _parsed):
        raise SystemExit(f'open-block seeds must be integers, got {_raw_seeds!r}')
    # Repeating a seed must not fabricate extra paired observations.
    seeds = list(dict.fromkeys(_parsed))
    results = {}

    def variant(mb, mode, scope='full_prefix'):
        return _variant(mb, mode, scope)

    def add(name, va, vb, *, family):
        if name in results:
            return results[name]
        ds, ppl_ds, used, rejected = _paired(cells, va, vb, seeds)
        reasons = [f"seed{r['seed']}: {r['reason']}" for r in rejected]
        st = L.paired_effect_stats(ds, alpha=0.05, min_n=3) if ds else {'n': 0}
        st = dict(st)
        st.update({
            'name': name, 'family': family, 'a': va, 'b': vb,
            'seeds': used, 'per_seed_delta_nll': ds, 'per_seed_delta_ppl': ppl_ds,
            'delta_convention': 'a - b; negative NLL delta means a is better',
            'controlled_violations': reasons,
            'n_control_rejected': len(rejected),
            'control_rejected_seeds': [r['seed'] for r in rejected],
            'p_holm': None,
        })
        results[name] = st
        return st

    # Primary family (pre-registered three comparisons).
    primary = []
    primary.append(add('summary - closed @128',
                       variant(128, 'summary'), variant(128, 'closed'), family='primary'))
    primary.append(add('singletons - summary @128',
                       variant(128, 'singletons'), variant(128, 'summary'), family='primary'))
    # Interaction: (summary - closed at None) - (summary - closed at 128),
    # computed seed-wise as Δ_none - Δ_128.
    d_int, p_int, used_int, int_rejected = [], [], [], []
    for s in seeds:
        a = cells.get((variant(None, 'summary'), s))
        b = cells.get((variant(None, 'closed'), s))
        c = cells.get((variant(128, 'summary'), s))
        d = cells.get((variant(128, 'closed'), s))
        if None in (a, b, c, d):
            continue
        # The interaction contrasts two controlled pairs.  A seed that fails
        # either pair's gate cannot contribute a controlled interaction.
        _why = _controlled_interaction(a['record'], b['record'],
                                       c['record'], d['record'])
        if _why is not None:
            int_rejected.append({'seed': int(s), 'reason': _why})
            continue
        d_int.append(float((a['nll'] - b['nll']) - (c['nll'] - d['nll'])))
        p_int.append(float((a['ppl'] - b['ppl']) - (c['ppl'] - d['ppl'])))
        used_int.append(int(s))
    st = dict(L.paired_effect_stats(d_int, alpha=0.05, min_n=3) if d_int else {'n': 0})
    st.update({'name': 'interaction: summary effect (none - 128)',
               'family': 'primary', 'a': 'summary-none - closed-none',
               'b': 'summary-128 - closed-128', 'seeds': used_int,
               'per_seed_delta_nll': d_int, 'per_seed_delta_ppl': p_int,
               'delta_convention': 'negative = prefix summary gains more when no cap',
               'controlled_violations': [f"seed{r['seed']}: {r['reason']}" for r in int_rejected],
               'n_control_rejected': len(int_rejected),
               'control_rejected_seeds': [r['seed'] for r in int_rejected],
               'p_holm': None})
    results[st['name']] = st

    # Secondary descriptive cells.
    for mb in (32, 64, 128, None):
        for mode in ('summary', 'singletons'):
            add(f'{mode} - closed @{mb}',
                variant(mb, mode), variant(mb, 'closed'), family='descriptive')
    add('singletons - summary @none',
        variant(None, 'singletons'), variant(None, 'summary'), family='descriptive')
    # Auxiliary outside-window panel, if present.
    for mb in (128, None):
        for mode in ('summary', 'singletons'):
            va, vb = variant(mb, mode, 'outside_window'), variant(mb, 'closed', 'outside_window')
            _sa = {s for (v, s) in cells if v == va}
            _sb = {s for (v, s) in cells if v == vb}
            if _sa & _sb:
                add(f'{mode} - closed @{mb} (outside-window)',
                    va, vb, family='auxiliary_outside_window')

    # Holm within the declared primary family.
    prim_names = [n for n, r in results.items() if r.get('family') == 'primary']
    # Pre-registration: every primary contrast must state its mechanism,
    # predicted sign, metric and falsifier BEFORE the numbers are read.  The
    # audit also checks that the declared metric is defined in METRIC_POLICY.
    _hyp = L.hypothesis_audit(L.OPEN_BLOCK_HYPOTHESIS_DESIGN, 'open_block',
                              names=prim_names)
    for _n in prim_names:
        results[_n]['hypothesis'] = (_hyp.get('hypotheses') or {}).get(_n)
    # The Holm family is the DECLARED primary family, not only the comparisons
    # that happened to produce a finite p.  A declared-but-unmeasurable
    # contrast keeps a p=1 slot; dropping it would shrink m and make the
    # remaining tests look better than the study design allows.
    fam = []
    for n in prim_names:
        p = results[n].get('p_exact_signflip')
        try:
            p = float(p)
        except (TypeError, ValueError, OverflowError):
            p = float('nan')
        fam.append((n, p if math.isfinite(p) and 0.0 <= p <= 1.0 else 1.0))
    if fam:
        adj = L.holm_adjust(fam)
        for n, d in adj.items():
            results[n]['p_holm'] = d.get('p_holm')

    diagnostics = {}
    efficiency = {}
    for (v, s), cell in cells.items():
        d = _cell_diagnostics(cell['record'])
        if d is not None:
            diagnostics[f'{v}::seed{s}'] = d
        e = _cell_efficiency(cell['record'])
        if e is not None:
            efficiency[f'{v}::seed{s}'] = e
    _prim_n = [int(results[n].get('n') or 0) for n in prim_names if results[n].get('n')]
    _prim_n_min = min(_prim_n) if _prim_n else 0
    _prim_n_max = max(_prim_n) if _prim_n else 0
    # Carry n_nonzero through the family floor: reading only the pair count
    # would claim a resolution the tied contrasts cannot reach.
    _prim_entries = [results[n] for n in prim_names if results[n].get('n')]
    _holm_floor = (_family_holm_floor(_prim_entries, len(prim_names))
                   if _prim_entries and prim_names else None)
    _holm_min_n = (L.signflip_min_n_for_holm(len(prim_names), 0.05)
                   if prim_names else None)
    # Exact ties do not contribute resolution.  Use each contrast's non-zero
    # pair count (the same number that sets its p floor), not its raw n: an
    # all-tied primary contrast at n=8 still has floor 1.0 and cannot reach a
    # family-corrected claim.
    _prim_eff_n_min = _primary_effective_n(results, prim_names)
    # Every primary contrast must clear the family-corrected seed floor for ANY
    # confirmatory claim.  The weakest measured contrast governs that gate, not
    # the best-powered one.  n=6 only suffices for a single RAW two-sided test
    # (floor 0.03125); with the three-comparison Holm family the minimum is
    # n=7 (3 * 2/2**7 = 0.0469).  Surfacing this prevents the old
    # "6-seed extension" wording from promising a significance claim it cannot
    # reach.
    _seeds_sufficient = (bool(_holm_min_n is not None and _prim_eff_n_min > 0
                              and _prim_eff_n_min >= _holm_min_n))
    os.makedirs(analysis_dir, exist_ok=True)
    stats = {
        'outdir': outdir, 'code': L.code_stamp(), 'semantics': L.CODE_SEMANTICS,
        'seeds_requested': seeds,
        'cells_available': sorted(f'{v}::seed{s}' for (v, s) in cells),
        'record_integrity': {'n_excluded': len(_integrity), 'excluded': _integrity},
        'arm_summary': _arm_summary(cells),
        'primary_family': prim_names,
        'primary_family_size': len(prim_names),
        'primary_holm_floor': _holm_floor,
        'primary_n_min': _prim_n_min,
        'primary_n_max': _prim_n_max,
        'primary_effective_n_min': _prim_eff_n_min,
        'primary_holm_min_seeds': _holm_min_n,
        'primary_seeds_sufficient': _seeds_sufficient,
        'comparisons': results,
        'hypothesis_audit': _hyp,
        'hypotheses': {n: (_hyp.get('hypotheses') or {}).get(n) for n in prim_names},
        'diagnostics': diagnostics,
        'efficiency': efficiency,
        'notes': [
            'delta = a - b; a negative NLL delta means arm a is better.',
            'p_exact_signflip is two-sided; with n=4 the minimum attainable raw p is 0.125, with n=6 it is 0.03125. A family-corrected claim needs the Holm floor, not the raw floor.',
            'A failure to reach p<0.05 at n=4 is a resolution limit, not evidence of no effect.',
            'The 4-seed panel is descriptive for the primary family; Holm-adjusted p-values are reported with their floors.',
            'Interaction I = (summary - closed | no cap) - (summary - closed | 128); negative means the summary benefit grows when the cap is removed.',
        ] + ([f'primary seed coverage is INSUFFICIENT for a family-corrected claim: '
              f'minimum n={_holm_min_n} for {len(prim_names)} Holm comparisons, '
              f'weakest contrast has effective n={_prim_eff_n_min} (raw n={_prim_n_min})']
             if _holm_min_n is not None and not _seeds_sufficient else [])
        + [f'hypothesis audit: {w}' for w in (_hyp.get('warnings') or [])]
        + [f'record integrity: {r["key"]} excluded ({r["reason"]})' for r in _integrity],
    }
    _atomic_json(os.path.join(analysis_dir, 'stats.json'), stats)
    md = _render_markdown(stats)
    with open(os.path.join(analysis_dir, 'OPEN_BLOCK_REPORT.md'), 'w', encoding='utf-8') as f:
        f.write(md)
    print(md)
    return stats


def _fmt(x, nd=4):
    if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x)):
        return f'{float(x):.{nd}f}'
    return '—'


def _render_markdown(stats):
    lines = ['# Open-block panel: maximum block length × current-block reading', '']
    lines.append(f'Code: `{stats["code"]}`  |  cells: {len(stats["cells_available"])}  |  seeds: {stats["seeds_requested"]}')
    if stats.get('primary_holm_floor') is not None:
        _n_txt = (str(stats.get('primary_n_min')) if stats.get('primary_n_min') == stats.get('primary_n_max')
                  else f'{stats.get("primary_n_min")}..{stats.get("primary_n_max")}')
        lines.append(f'Primary family Holm floor: `{_fmt(stats.get("primary_holm_floor"))}` at family n={_n_txt} (max n governs the floor); minimum seeds for a confirmatory claim: `{stats.get("primary_holm_min_seeds")}`.')
    if stats.get('primary_seeds_sufficient') is False:
        lines += ['', f'> **Resolution limit:** the weakest primary contrast has effective n={stats.get("primary_effective_n_min")} (raw n={stats.get("primary_n_min")}), '
                  f'but the {stats.get("primary_family_size")}-comparison Holm family needs n>='
                  f'`{stats.get("primary_holm_min_seeds")}` before any corrected claim is reachable. '
                  'This panel is descriptive; adding seeds (not re-reading the same p-values) is the only fix.']
    if stats.get('record_integrity', {}).get('n_excluded'):
        lines += ['', '> **Excluded records:** ' + '; '.join(
            f'`{r.get("key")}` ({r.get("reason")})'
            for r in stats['record_integrity']['excluded']) + '.']
    lines += ['', 'Delta convention: `a - b` on token NLL; negative means `a` is better.', '']
    _arms = stats.get('arm_summary') or {}
    if _arms:
        lines += ['## Arm summary (Mean ± SD over seeds)', '',
                  '| arm | n seeds | PPL mean ± SD | NLL/token mean ± SD | pooled PPL | seeds |',
                  '|---|---|---|---|---|---|']
        for _v, _a in sorted(_arms.items()):
            lines.append('| {} | {} | {} ± {} | {} ± {} | {} | {} |'.format(
                _v, _a.get('n_seeds'), _fmt(_a.get('ppl_mean'), 3),
                _fmt(_a.get('ppl_std'), 3), _fmt(_a.get('nll_per_token_mean'), 6),
                _fmt(_a.get('nll_per_token_std'), 6), _fmt(_a.get('ppl_pooled'), 3),
                ','.join(str(x) for x in (_a.get('seeds') or []))))
        lines += ['', '> `±` is the seed-level standard deviation (training randomness), '
                  'not a document-sampling interval. `pooled PPL` re-pools token NLL within the arm.', '']
    lines.append('| comparison | family | n | mean ΔNLL | SD | dz | exact p | Holm p | p floor | CI95 | controlled |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for name, r in stats['comparisons'].items():
        ctrl = 'ok' if not r.get('controlled_violations') else '; '.join(r['controlled_violations'])
        ci_s = f'[{_fmt(r.get("ci95_low"))}, {_fmt(r.get("ci95_high"))}]'
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            name, r.get('family', ''), r.get('n', 0), _fmt(r.get('mean')),
            _fmt(r.get('std')), _fmt(r.get('cohens_dz')),
            _fmt(r.get('p_exact_signflip')), _fmt(r.get('p_holm')),
            _fmt(r.get('min_reachable_p')), ci_s, ctrl))
    if stats.get('hypotheses'):
        lines += ['', '## Pre-registered primary hypotheses', '',
                  '| comparison | predicted sign | mechanism | falsifier |',
                  '|---|---|---|---|']
        for _n, _h in stats['hypotheses'].items():
            if not isinstance(_h, dict):
                lines.append(f'| {_n} | MISSING | — | — |')
                continue
            lines.append('| {} | {} | {} | {} |'.format(
                _n, _h.get('predicted_sign'), _h.get('mechanism') or '—',
                _h.get('falsifier') or '—'))
        if (stats.get('hypothesis_audit') or {}).get('status') != 'ok':
            lines.append('')
            lines.append('Hypothesis audit is not clean: '
                         + '; '.join((stats.get('hypothesis_audit') or {}).get('warnings') or []))
    if stats.get('diagnostics'):
        lines += ['', '## Diagnostics (per-cell means over layers; seed-level in stats.json)', '',
                  '| cell | len_mean | len_p50 | len_p95 | len_max | frac_at_max | forced_only | natural_only | both | prefix_age_mean | outside_nonzero | prefix_sel_rate | history_displaced | pos_trunc_frac |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for cell, d in sorted(stats['diagnostics'].items()):
            lines.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
                cell, _fmt(d.get('len_mean')), _fmt(d.get('len_p50'), 2),
                _fmt(d.get('len_p95'), 2), _fmt(d.get('len_max'), 0),
                _fmt(d.get('frac_at_max')), _fmt(d.get('forced_only'), 0),
                _fmt(d.get('natural_only'), 0), _fmt(d.get('forced_and_natural'), 0),
                _fmt(d.get('prefix_age_mean')),
                _fmt(d.get('outside_window_nonzero_frac')),
                _fmt(d.get('prefix_selected_rate')),
                _fmt(d.get('history_displaced_mean')),
                _fmt(d.get('position_truncated_frac')
                     if d.get('position_truncated_frac') is not None
                     else d.get('position_truncated_frac_of_block_tokens'))))
    if stats.get('efficiency'):
        lines += ['', '## Efficiency (from the measured harness; no KV cache)', '',
                  '| cell | train tokens/s | tokens/s prefill | prefill p50 ms | prefix-prefill p50 ms | peak alloc MB |',
                  '|---|---|---|---|---|---|']
        for cell, e in sorted(stats['efficiency'].items()):
            lines.append('| {} | {} | {} | {} | {} | {} |'.format(
                cell, _fmt(e.get('train_tokens_per_s'), 1),
                _fmt(e.get('tokens_per_s_prefill'), 1),
                _fmt(e.get('prefill_ms_p50'), 1), _fmt(e.get('prefix_prefill_ms_p50'), 1),
                _fmt(e.get('peak_mem_allocated_mb'), 1)))
    lines += ['', '## How to read the primary comparisons', '',
              '- `summary - closed @128`: does a causal summary of the open block help under the default cap?',
              '- `singletons - summary @128`: representation granularity at a fixed top-k budget.',
              '- `interaction`: does removing the cap make the summary benefit larger? A negative value is consistent with a waiting bottleneck, but only alongside block-length, prefix-selection and window-coverage statistics.',
              '- Prefix candidates compete for the same K slots; an empty/rare prefix selection is a selection-path fact, not evidence that prefix information is useless.',
              '- The primary family measured n={}–{} pair(s); the exact-test floor at its weakest effective n is {}. Report the effect and its uncertainty, and do not read a non-significant result as a null result when the floor is above alpha. '.format(
                  stats.get('primary_n_min'), stats.get('primary_n_max'),
                  _fmt(L.signflip_p_floor(stats.get('primary_effective_n_min') or 0)))
              + (f'With the {stats.get("primary_family_size")}-comparison Holm family the minimum seed count for any confirmatory claim is {stats.get("primary_holm_min_seeds")} (Holm floor {_fmt(stats.get("primary_holm_floor"))}). '
                 + ('The current panel does NOT meet that floor; extend it before making a significance claim.'
                    if stats.get('primary_seeds_sufficient') is False else
                    'The current panel meets that floor.')
                 if stats.get('primary_holm_min_seeds') else 'Extend the key cells only if a significance claim is needed.')]
    return '\n'.join(lines) + '\n'


def cmd_smoke(args):
    torch.manual_seed(0)
    cfg = dict(BASE_CFG, d=64, n_layers=2, n_heads=4, d_head=16, seq_len=64,
               batch_size=4, n_train_tokens=8000, steps=3, warmup=1,
               eval_every=0, eval_subset=4, seeds=[0],
               outdir=args.outdir or 'results_open_block_smoke')
    outdir = run_panel(cfg, MAIN_MATRIX, device=L.DEVICE, synthetic=True,
                       label='smoke')
    summary = _load_summary(outdir)
    got = {k.split('::')[0] for k in summary if '::' in k}
    want = {_variant(mb, m, sc) for mb, m, sc in MAIN_MATRIX}
    missing = want - got
    assert not missing, f'smoke cells missing: {missing}'
    for k, r in summary.items():
        assert isinstance(r.get('open_block_config'), dict), k
        assert r['open_block_config'].get('layers'), k
        assert math.isfinite(float(r['ppl'])), k
    print(f'\n[open-block] smoke PASSED ({len(got)}/{len(want)} cells)')
    return 0


def _parse_seeds(s):
    return [int(x) for x in str(s).split(',') if str(x).strip()]


def _announce_seed_plan(cfg, label='main'):
    """State the family-corrected seed requirement before a panel is trained.

    The primary family has three declared Holm comparisons, so n=4 (and even
    n=6) cannot reach a corrected claim.  Printing the floor here avoids
    spending GPU time on a panel whose planned seed count already rules out the
    claim it is meant to test.
    """
    _seeds = []
    for s in (cfg.get('seeds') or []):
        _s = L._as_seed_int(s)
        if _s is not None and _s not in _seeds:
            _seeds.append(_s)
    need = L.signflip_min_n_for_holm(len(L.OPEN_BLOCK_HYPOTHESIS_DESIGN), 0.05)
    sufficient = bool(need is not None and len(_seeds) >= need)
    out = {'label': label, 'n_seeds': len(_seeds),
           'primary_family_size': len(L.OPEN_BLOCK_HYPOTHESIS_DESIGN),
           'primary_holm_min_seeds': need, 'seeds_sufficient': sufficient}
    print(f'[open-block seed-plan] {label}: n={len(_seeds)}, primary family='
          f'{out["primary_family_size"]}, family-corrected minimum n={need}, '
          f'sufficient={sufficient}')
    if not sufficient:
        print(f'[open-block seed-plan] WARNING: {label} at n={len(_seeds)} is '
              f'descriptive only; a family-corrected claim needs n>={need}. '
              'Extend the key cells with more seeds before reading a p-value as significant.')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['smoke', 'pilot', 'main', 'aux', 'extended', 'report'])
    ap.add_argument('--seeds', default=None)
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--seq-len', type=int, default=None)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--max-hours', type=float, default=None)
    ap.add_argument('--analysis-dir', default='analysis_open_block')
    args = ap.parse_args()

    if args.command == 'smoke':
        return cmd_smoke(args)

    if args.command == 'report':
        outdir = args.outdir or BASE_CFG['outdir']
        analyse(outdir, seeds=_parse_seeds(args.seeds) if args.seeds else None,
                analysis_dir=args.analysis_dir)
        return 0

    if args.command in ('pilot', 'main'):
        matrix = MAIN_MATRIX
    elif args.command == 'extended':
        matrix = EXTENDED_MATRIX
    else:
        matrix = AUX_MATRIX
    cfg = dict(BASE_CFG)
    cfg['seeds'] = _parse_seeds(args.seeds) if args.seeds else ([0] if args.command == 'pilot' else BASE_CFG['seeds'])
    if args.steps is not None:
        cfg['steps'] = int(args.steps)
    if args.command == 'pilot':
        cfg['steps'] = int(args.steps or 100)
    if args.seq_len is not None:
        cfg['seq_len'] = int(args.seq_len)
    if args.command == 'aux':
        cfg['outdir'] = 'results_open_block_aux'
    if args.command == 'extended':
        cfg['outdir'] = 'results_open_block_extended'
    if args.outdir is not None:
        cfg['outdir'] = args.outdir
    if args.command in ('main', 'aux', 'extended'):
        _announce_seed_plan(cfg, label=args.command)
    run_panel(cfg, matrix, device=L.DEVICE, synthetic=False,
              max_hours=args.max_hours,
              label='pilot' if args.command == 'pilot' else args.command)
    if args.command in ('main', 'aux', 'extended'):
        try:
            analyse(cfg['outdir'], seeds=cfg['seeds'], analysis_dir=args.analysis_dir)
        except SystemExit as e:
            print(f'[open-block] report skipped: {e}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
