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


def _roundtrip(d):
    return json.loads(json.dumps(d, default=str))


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
    ap.add_argument('--chunk', type=int, default=8)
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
                           chunk=args.chunk, val_bnd=val_bnd, train_ids=train_ids)
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
    lines += ['## Position / target buckets', '',
              '| bucket | n | mean CE | PPL |', '|---|---|---|---|']
    for k, v in sorted((res.get('buckets') or {}).items()):
        _mce = v.get('mean_ce')
        _ppl = v.get('perplexity')
        lines.append(f'| {k} | {v.get("n")} | {("n/a" if _mce is None else f"{_mce:.3f}")} | {("n/a" if _ppl is None else f"{_ppl:.2f}")} |')
    _counts = res.get('cause_counts') or {}
    _guide = res.get('cause_guide') or {}
    if _counts:
        lines += ['', '## Error-cause triage (heuristic, not a verdict)', '',
                  '> These labels only direct the manual review. An "annotation_or_rare_target" case may be a correct but rare label; an "ambiguous_context" case may have no defensible single answer. Read the context and the label before changing the model.', '',
                  '| cause | worst-token count | what to check |', '|---|---|---|']
        for c, n in sorted(_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f'| {c} | {n} | {(_guide.get(c) or "").replace("|", "\\|")} |')
    _all = res.get('cause_counts_all') or {}
    if _all:
        lines += ['', '**Whole-split cause tally (finite targets, heuristic):**', '',
                  '| cause | n | share | mean CE |', '|---|---|---|---|']
        for c, v in sorted(_all.items(), key=lambda kv: (-(kv[1].get('n') or 0), kv[0])):
            if not isinstance(v, dict) or not v.get('n'):
                continue
            lines.append(f'| {c} | {v.get("n")} | {(v.get("share") or 0) * 100:.1f}% | {v.get("mean_ce"):.3f} |')
    lines += ['', '## Worst tokens', '',
              '| # | row | pos | CE | target | predicted | entropy | rare | cause | boundary |', '|---|---|---|---|---|---|---|---|---|---|']
    for i, t in enumerate(res.get('worst_tokens') or []):
        tgt = (t.get('target_text') or '').replace('|', '\\|').replace('\n', '\\n')[:80]
        prd = (t.get('pred_text') or '').replace('|', '\\|').replace('\n', '\\n')[:80]
        lines.append(f'| {i + 1} | {t["row"]} | {t["position"]} | {t["ce"]:.3f} | `{t.get("target_id")}` {tgt} | `{t.get("pred_id")}` {prd} | {t["entropy"]:.3f} | {t.get("rare_target")} | {t.get("cause_primary")} | {t.get("is_boundary_token")} |')
    lines += ['', '## Worst sequences', '',
              '| # | row | mean CE | NLL | context (truncated) |', '|---|---|---|---|---|']
    for i, s in enumerate(res.get('worst_sequences') or []):
        ctx = (s.get('context') or '').replace('|', '\\|').replace('\n', ' ')[:240]
        lines.append(f'| {i + 1} | {s["row"]} | {s["mean_ce"]:.3f} | {s["nll"]:.2f} | {ctx} |')
    if paired:
        lines += ['', '## Paired arm comparison', '',
                  f'- B `{paired.get("checkpoint_b")}` minus A `{paired.get("checkpoint_a")}` mean CE: **{paired.get("mean_delta_b_minus_a"):+.4f}**',
                  f'- B worse at {paired.get("b_worse_at_n")} token positions; A worse at {paired.get("a_worse_at_n")}.', '',
                  '| # | row | pos | target | CE A | CE B | delta (B−A) |', '|---|---|---|---|---|---|---|']
        for i, t in enumerate(paired.get('worst_tokens_for_b') or []):
            tg = (t.get('target_text') or '').replace('|', '\\|').replace('\n', '\\n')[:60]
            lines.append(f'| {i + 1} | {t["row"]} | {t["position"]} | `{t["target_id"]}` {tg} | {t["ce_a"]:.3f} | {t["ce_b"]:.3f} | {t["delta_b_minus_a"]:+.3f} |')
    lines += ['', '> This file is a diagnostic starting point, not a result table. Inspect the labelled target and context before declaring a model failure: annotation errors, near-duplicates, genuinely ambiguous text and rare tokens all look like "high CE". The automatic cause label is a triage heuristic; only a manual read of the context and label can separate an annotation error from an ambiguous example from a model blind spot.', '']
    L.atomic_write_text(os.path.join(args.outdir, 'bad_cases.md'), '\n'.join(lines) + '\n')
    print(f'[errors] wrote {jsonl}, {os.path.join(args.outdir, "bad_cases.md")}, {os.path.join(args.outdir, "summary.json")}' + (' + paired comparison' if paired else ''))
    return res


if __name__ == '__main__':
    main()
