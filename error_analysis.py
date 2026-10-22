#!/usr/bin/env python3
"""Bad-case / error analysis for a saved CSA checkpoint.

Zero training: load one checkpoint, score a fixed validation split, and write
  analysis_errors/bad_cases.jsonl   worst tokens + worst sequences (with cause triage)
  analysis_errors/bad_cases.md      human-readable samples + a manual-review checklist
  analysis_errors/summary.json      bucket statistics, label checks and heuristic cause counts

The analysis is deliberately restricted to the validation split.  The frozen
test split must only be touched through v12's one-shot freeze protocol; feeding
it here repeatedly would leak test information into later design choices.

Usage:
  python error_analysis.py --ckpt results_v10_mech/ckpt/csa_fixed_rope_seed0.pt
  python error_analysis.py --ckpt ... --max-rows 64 --top-tokens 80
"""
import argparse
import hashlib
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

import numpy as np
import torch

import exp_lib as L

try:  # installs the RoPE / v7 layer variants used by the frozen checkpoints
    import v7_supp  # noqa: F401
except Exception as _v7e:
    print(f'[errors] NOTE: v7_supp not importable ({type(_v7e).__name__}: {_v7e}); only built-in layer variants can be loaded')


def _sha256(path, cap=None):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _load_model(path):
    try:
        d = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    except TypeError:
        d = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(d, dict) or 'sd' not in d:
        raise SystemExit(f'[errors] REFUSE: {path} is not a CSA checkpoint (no state_dict)')
    d = dict(d)
    sd = d['sd']
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    vocab = int(d.get('vocab', 8192))
    dim = int(d.get('d', 256))
    n_layers = int(d.get('n_layers', 6))
    n_heads = int(d.get('n_heads', 8))
    d_head = int(d.get('d_head', 32))
    max_seq = int(d.get('train_len') or d.get('max_seq') or 512)
    cfgs = d.get('cfg')
    if not cfgs:
        raise SystemExit(f'[errors] REFUSE: {path} carries no layer configs')
    model = L.SmallGPT(vocab, dim, n_layers, n_heads, d_head, max_seq,
                       cfgs, mlp_ratio=float(d.get('mlp_ratio', 4.0)))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f'[errors] REFUSE: checkpoint does not match the model (missing={len(missing)}, unexpected={len(unexpected)})')
    model.eval()
    return model, d


def _paired_vocab_guard(meta_a, meta_b):
    """Refuse a paired per-token comparison across different vocabularies.

    The same integer id means a different token in two tokenizers, so a
    position-wise delta would silently compare different labels.  This is a
    validity check, not a preference: the corpus/vocabulary must be identical
    before any paired error claim is made.
    """
    def _v(meta, label):
        raw = (meta or {}).get('vocab', 8192)
        if isinstance(raw, bool):
            raise SystemExit(f'[errors] REFUSE: {label} checkpoint has a boolean vocab ({raw!r})')
        try:
            f = float(raw)
        except (TypeError, ValueError, OverflowError):
            raise SystemExit(f'[errors] REFUSE: {label} checkpoint vocab is not an integer ({raw!r})') from None
        if not math.isfinite(f) or f != math.floor(f) or f < 1:
            raise SystemExit(f'[errors] REFUSE: {label} checkpoint vocab is not a positive integer ({raw!r})')
        return int(f)
    va = _v(meta_a, 'first')
    vb = _v(meta_b, 'comparison')
    if va != vb:
        raise SystemExit(f'[errors] REFUSE: paired checkpoints use different vocabularies ({va} vs {vb}); per-token labels are not comparable')


def _roundtrip(d):
    return json.loads(json.dumps(d, default=str))


def _one_line(value, limit=160):
    text = '' if value is None else str(value)
    return text.replace('\r', '\\r').replace('\n', '\\n')[:limit]


def _print_bad_cases(res, *, n_tokens=10, n_sequences=3):
    """Print the worst samples so the reviewer does not have to open JSONL first.

    The console view deliberately repeats the manual-review caveat: the cause
    label is a triage prompt, not an annotation verdict.
    """
    if n_tokens <= 0 and n_sequences <= 0:
        return
    print('\n[errors] worst-token samples (context -> target | predicted; cause is a heuristic):')
    tokens = res.get('worst_tokens') if isinstance(res.get('worst_tokens'), list) else []
    for i, t in enumerate(tokens[:max(0, int(n_tokens))]):
        if not isinstance(t, dict):
            continue
        topk = t.get('topk_text')
        if not (isinstance(topk, list) and any(x is not None for x in topk)):
            topk = t.get('topk_ids') or []
        if isinstance(topk, list):
            topk_txt = ', '.join(_one_line(x, 24) for x in topk[:5])
        else:
            topk_txt = _one_line(topk, 80)
        flags = []
        if t.get('same_context_ambiguous'):
            flags.append('prefix has multiple targets')
        if t.get('rare_target'):
            flags.append('rare target')
        print(f"  #{i + 1} row={t.get('row')} pos={t.get('position')} CE={t.get('ce')}  "
              f"target={t.get('target_id')}:{_one_line(t.get('target_text'), 48)!r}  "
              f"pred={t.get('pred_id')}:{_one_line(t.get('pred_text'), 48)!r}")
        print(f"      context={_one_line(t.get('context_text'), 180)!r}")
        print(f"      top-5=[{topk_txt}]  cause={t.get('cause_primary')} "
              f"causes={t.get('causes')}" + (f"  flags={flags}" if flags else ''))
    seqs = res.get('worst_sequences') if isinstance(res.get('worst_sequences'), list) else []
    if n_sequences > 0 and seqs:
        print('[errors] worst sequences:')
        for i, sq in enumerate(seqs[:int(n_sequences)]):
            if not isinstance(sq, dict):
                continue
            print(f"  seq #{i + 1} row={sq.get('row')} mean_CE={sq.get('mean_ce')} "
                  f"n_tokens={sq.get('n_tokens')} context={_one_line(sq.get('context'), 240)!r}")
    counts = res.get('cause_counts') if isinstance(res.get('cause_counts'), dict) else {}
    if counts:
        tally = ', '.join(f'{k}={v}' for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
        print(f'[errors] worst-token cause tally: {tally}')
    ca = res.get('context_ambiguity')
    if isinstance(ca, dict):
        _scan = ca.get('scan') if isinstance(ca.get('scan'), dict) else {}
        _scan_txt = ('whole split' if ca.get('scan_complete')
                     else f"sampled {_scan.get('n_rows_scanned')}/{_scan.get('n_rows_total')} rows")
        print(f"[errors] prefix ambiguity: {ca.get('n_ambiguous_worst')}/{ca.get('n_worst_reviewed')} "
              f'reviewed worst tokens share a prefix with another target; '
              f"{ca.get('n_contexts_with_multiple_targets')} ambiguous context(s) in the evaluated rows "
              f'({_scan_txt})')
    print('[errors] Read the context and label before acting: annotation error, genuine ambiguity and a model blind spot need different fixes.\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--compare-ckpt', default=None, help='optional second checkpoint for paired per-token delta analysis')
    ap.add_argument('--outdir', default='analysis_errors')
    ap.add_argument('--dataset', default='wikitext', choices=['wikitext', 'fineweb'])
    ap.add_argument('--seq-len', type=int, default=None, help='evaluation sequence length (default: checkpoint max_seq)')
    ap.add_argument('--n-train-tokens', type=int, default=8000000)
    ap.add_argument('--val-seqs', type=int, default=128)
    ap.add_argument('--max-rows', type=int, default=64)
    ap.add_argument('--top-tokens', type=int, default=80)
    ap.add_argument('--top-sequences', type=int, default=30)
    ap.add_argument('--print-samples', type=int, default=10, help='worst-token samples to print (0 disables)')
    ap.add_argument('--print-sequences', type=int, default=3, help='worst sequences to print (0 disables)')
    ap.add_argument('--chunk', type=int, default=8)
    ap.add_argument('--context-scan-positions', type=int, default=2000000,
                    help='bound the prefix-ambiguity scan (0 = scan the whole evaluated split)')
    ap.add_argument('--device', default=None)
    args = ap.parse_args(argv)
    device = torch.device(args.device) if args.device else L.DEVICE
    os.makedirs(args.outdir, exist_ok=True)
    model, meta = _load_model(args.ckpt)
    model.to(device)
    seq_len = int(args.seq_len or meta.get('train_len') or meta.get('max_seq') or 512)
    cfg = {'dataset': args.dataset, 'seq_len': seq_len, 'n_train_tokens': args.n_train_tokens,
           'val_seqs': args.val_seqs, 'vocab_size': int(meta.get('vocab', 8192))}
    train_ids, val_batch, vocab, tok, val_bnd = L.load_run_corpus(cfg)
    if args.dataset != 'wikitext':
        tok = None
    n_rows = int(args.max_rows)
    if n_rows < 1:
        raise SystemExit(f'--max-rows must be a positive integer, got {n_rows}')
    if args.print_samples < 0 or args.print_sequences < 0:
        raise SystemExit('--print-samples/--print-sequences must be non-negative integers')
    if args.context_scan_positions < 0:
        raise SystemExit('--context-scan-positions must be a non-negative integer (0 means unbounded)')
    n_rows = min(n_rows, int(val_batch.shape[0]))
    val_batch = np.ascontiguousarray(val_batch[:n_rows, :seq_len])
    if val_bnd is not None:
        _vb = np.asarray(val_bnd)
        if _vb.ndim != 2 or _vb.shape[0] < n_rows or _vb.shape[1] < seq_len:
            print(f'[errors] WARNING: boundary mask shape {_vb.shape} cannot cover {n_rows} rows x {seq_len} tokens — omitting boundary-token statistics instead of silently truncating them')
            val_bnd = None
        else:
            val_bnd = np.ascontiguousarray(_vb[:n_rows, :seq_len])
    print(f'[errors] scoring {n_rows} validation rows x {seq_len} tokens with {os.path.basename(args.ckpt)}')
    res = L.analyze_errors(model, val_batch, vocab, device, tokenizer=tok,
                           top_n=args.top_sequences, top_tokens=args.top_tokens,
                           chunk=args.chunk, val_bnd=val_bnd, train_ids=train_ids,
                           context_scan_max_positions=(args.context_scan_positions or None))
    res['checkpoint'] = os.path.abspath(args.ckpt)
    res['checkpoint_sha256'] = _sha256(args.ckpt)
    res['checkpoint_meta'] = {k: _roundtrip(meta.get(k)) for k in
                              ('variant', 'seed', 'step', 'train_fp', 'eval_fp', 'code', 'curve_health', 'best_step', 'best_val_ppl')}
    res['split'] = 'validation'
    res['created_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    res['warning'] = 'validation-only error analysis; do not run this on the frozen test split outside the v12 one-shot protocol'
    paired = None
    if args.compare_ckpt:
        model_b, meta_b = _load_model(args.compare_ckpt)
        _paired_vocab_guard(meta, meta_b)
        model_b.to(device)
        paired = L.compare_token_errors(model, model_b, val_batch, vocab, device,
                                        tokenizer=tok, top_n=args.top_tokens, chunk=args.chunk)
        paired['checkpoint_a'] = os.path.abspath(args.ckpt)
        paired['checkpoint_b'] = os.path.abspath(args.compare_ckpt)
        paired['checkpoint_a_sha256'] = res['checkpoint_sha256']
        paired['checkpoint_b_sha256'] = _sha256(args.compare_ckpt)
        paired['split'] = 'validation'
        L.atomic_write_json(os.path.join(args.outdir, 'paired_errors.json'), paired, indent=2)
        with open(os.path.join(args.outdir, 'paired_bad_cases.jsonl'), 'w', encoding='utf-8') as f:
            for i, rec in enumerate(paired.get('worst_tokens_for_b') or []):
                f.write(json.dumps(L._json_clean({'kind': 'paired_token', 'rank': i + 1, **rec}), ensure_ascii=False, allow_nan=False) + '\n')
        del model_b
    # JSONL: one line per bad case, easiest to filter/grep downstream.
    jsonl = os.path.join(args.outdir, 'bad_cases.jsonl')
    with open(jsonl, 'w', encoding='utf-8') as f:
        for i, rec in enumerate(res.get('worst_tokens') or []):
            f.write(json.dumps(L._json_clean({'kind': 'token', 'rank': i + 1, **rec}), ensure_ascii=False, allow_nan=False) + '\n')
        for i, rec in enumerate(res.get('worst_sequences') or []):
            f.write(json.dumps(L._json_clean({'kind': 'sequence', 'rank': i + 1, **rec}), ensure_ascii=False, allow_nan=False) + '\n')
    L.atomic_write_json(os.path.join(args.outdir, 'summary.json'), res, indent=2)
    lines = ['# Bad-case analysis', '',
             f'- checkpoint: `{os.path.abspath(args.ckpt)}`',
             f'- sha256: `{res["checkpoint_sha256"]}`',
             f'- split: **validation only** ({n_rows} rows x {seq_len} tokens)',
             f'- pooled CE: {res.get("mean_ce"):.4f}; pooled PPL: {res.get("pooled_ppl"):.3f}' if res.get('mean_ce') is not None else '- pooled CE: n/a',
             f'- checkpoint health: `{(meta.get("curve_health") or {}).get("status", "unknown")}`', '']
    if res.get('rare_target_basis'):
        _rb = str(res.get('rare_target_basis'))
        _rp = res.get('rare_target_problem')
        lines += [f'- rare-target basis: `{_rb}`' + (f' — {_rp}' if _rp else ''), '']
    lines += ['## Position / target buckets', '',
              '| bucket | n | mean CE | PPL |', '|---|---|---|---|']
    for k, v in sorted((res.get('buckets') or {}).items()):
        _mce = v.get('mean_ce')
        _ppl = v.get('perplexity')
        lines.append(f'| {k} | {v.get("n")} | {("n/a" if _mce is None else f"{_mce:.3f}")} | {("n/a" if _ppl is None else f"{_ppl:.2f}")} |')
    _counts = res.get('cause_counts') if isinstance(res.get('cause_counts'), dict) else {}
    _guide = res.get('cause_guide') if isinstance(res.get('cause_guide'), dict) else {}
    if _counts:
        lines += ['', '## Error-cause triage (heuristic, not a verdict)', '',
                  '> These labels only direct the manual review. An "annotation_or_rare_target" case may be a correct but rare label; an "ambiguous_context" case may have no defensible single answer. Read the context and the label before changing the model.', '',
                  '| cause | worst-token count | what to check |', '|---|---|---|']
        for c, n in sorted(_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f'| {c} | {n} | {(_guide.get(c) or "").replace("|", "\\|")} |')
    _ca = res.get('context_ambiguity') if isinstance(res.get('context_ambiguity'), dict) else {}
    if _ca:
        _scan = _ca.get('scan') if isinstance(_ca.get('scan'), dict) else {}
        _scan_txt = ('whole evaluated split' if _ca.get('scan_complete')
                     else f'{_scan.get("n_rows_scanned")}/{_scan.get("n_rows_total")} row(s), '
                          f'{_scan.get("n_positions_scanned")} position(s)')
        lines += ['', '**Prefix ambiguity (label inconsistency vs genuine ambiguity):**', '',
                  f'- contexts with more than one observed next token: {_ca.get("n_contexts_with_multiple_targets")} '
                  f'covering {_ca.get("n_positions_with_ambiguous_context")} scored position(s) '
                  f'(scan: {_scan_txt})',
                  f'- reviewed worst tokens: {_ca.get("n_ambiguous_worst")} / {_ca.get("n_worst_reviewed")} '
                  f'share their prefix with a different target',
                  f'- {_ca.get("note")}']
        if _ca.get('scan_note'):
            lines.append(f'- {_ca.get("scan_note")}')
    _all = res.get('cause_counts_all') if isinstance(res.get('cause_counts_all'), dict) else {}
    if _all:
        lines += ['', '**Whole-split cause tally (finite targets, heuristic):**', '',
                  '| cause | n | share | mean CE |', '|---|---|---|---|']
        _rows = [kv for kv in _all.items() if isinstance(kv[1], dict) and kv[1].get('n')]
        for c, v in sorted(_rows, key=lambda kv: (-(kv[1].get('n') or 0), kv[0])):
            lines.append(f'| {c} | {v.get("n")} | {(v.get("share") or 0) * 100:.1f}% | {v.get("mean_ce"):.3f} |')
    lines += ['', '## Worst tokens', '',
              '> Read the context before changing the model: a rare-but-correct target, a genuinely ambiguous continuation and a confident blind spot need different fixes.', '',
              '| # | row | pos | CE | context (truncated) | target | predicted | top-5 | train count | entropy | rare | ctx variants | cause | boundary |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    _tokens = res.get('worst_tokens') if isinstance(res.get('worst_tokens'), list) else []
    for i, t in enumerate(_tokens):
        tgt = (t.get('target_text') or '').replace('|', '\\|').replace('\n', '\\n')[:80]
        prd = (t.get('pred_text') or '').replace('|', '\\|').replace('\n', '\\n')[:80]
        ctx = (t.get('context_text') or '').replace('|', '\\|').replace('\n', '\\n')[:180]
        _top = t.get('topk_text')
        if not (isinstance(_top, list) and any(x is not None for x in _top)):
            _top = t.get('topk_ids') or []
        _top_txt = ', '.join(str(x).replace('|', '\\|').replace('\n', '\\n')[:24]
                             for x in (_top[:5] if isinstance(_top, list) else []))
        lines.append(f'| {i + 1} | {t["row"]} | {t["position"]} | {t["ce"]:.3f} | {ctx} | `{t.get("target_id")}` {tgt} | `{t.get("pred_id")}` {prd} | {_top_txt} | {t.get("target_train_count")} | {t["entropy"]:.3f} | {t.get("rare_target")} | {t.get("same_context_target_variants")} | {t.get("cause_primary")} | {t.get("is_boundary_token")} |')
    lines += ['', '## Worst sequences', '',
              '| # | row | mean CE | NLL | context (truncated) |', '|---|---|---|---|---|']
    for i, s in enumerate(res.get('worst_sequences') or []):
        ctx = (s.get('context') or '').replace('|', '\\|').replace('\n', ' ')[:240]
        lines.append(f'| {i + 1} | {s["row"]} | {s["mean_ce"]:.3f} | {s["nll"]:.2f} | {ctx} |')
    if paired:
        _md = paired.get('mean_delta_b_minus_a')
        _npf = paired.get('n_paired_finite')
        _ntot = paired.get('n_tokens')
        _cov = (f'{_npf}/{_ntot} positions ({100.0 * _npf / _ntot:.1f}%)'
                if (isinstance(_npf, int) and isinstance(_ntot, int) and _ntot > 0) else 'n/a')
        _md_txt = (f'**{_md:+.4f}**' if isinstance(_md, (int, float)) and math.isfinite(_md)
                   else '**n/a — no position is finite in both arms**')
        lines += ['', '## Paired arm comparison', '',
                  f'- paired finite coverage: {_cov}',
                  f'- B `{paired.get("checkpoint_b")}` minus A `{paired.get("checkpoint_a")}` mean CE: {_md_txt}',
                  f'- B worse at {paired.get("b_worse_at_n")} token positions; A worse at {paired.get("a_worse_at_n")}.', '',
                  '> Only positions finite in BOTH checkpoints enter the paired delta. Per-arm non-finite counts are in `paired_errors.json`; a low coverage means the delta describes a small common support, not the whole split.', '',
                  '| # | row | pos | target | CE A | CE B | delta (B−A) |', '|---|---|---|---|---|---|---|']
        for i, t in enumerate(paired.get('worst_tokens_for_b') or []):
            tg = (t.get('target_text') or '').replace('|', '\\|').replace('\n', '\\n')[:60]
            lines.append(f'| {i + 1} | {t["row"]} | {t["position"]} | `{t["target_id"]}` {tg} | {t["ce_a"]:.3f} | {t["ce_b"]:.3f} | {t["delta_b_minus_a"]:+.3f} |')
    lines += ['', '> This file is a diagnostic starting point, not a result table. Inspect the labelled target and context before declaring a model failure: annotation errors, near-duplicates, genuinely ambiguous text and rare tokens all look like "high CE". The automatic cause label is a triage heuristic; only a manual read of the context and label can separate an annotation error from an ambiguous example from a model blind spot.', '']
    L.atomic_write_text(os.path.join(args.outdir, 'bad_cases.md'), '\n'.join(lines) + '\n')
    _print_bad_cases(res, n_tokens=args.print_samples, n_sequences=args.print_sequences)
    print(f'[errors] wrote {jsonl}, {os.path.join(args.outdir, "bad_cases.md")}, {os.path.join(args.outdir, "summary.json")}' + (' + paired comparison' if paired else ''))
    return res


if __name__ == '__main__':
    main()
