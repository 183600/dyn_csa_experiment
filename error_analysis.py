#!/usr/bin/env python3
"""Bad-case / error analysis for a saved CSA checkpoint.

Zero training: load one checkpoint, score a fixed validation split, and write
  analysis_errors/bad_cases.jsonl   worst tokens + worst sequences
  analysis_errors/bad_cases.md      human-readable samples
  analysis_errors/summary.json      bucket statistics and label checks

The analysis is deliberately restricted to the validation split.  The frozen
test split must only be touched through v12's one-shot freeze protocol; feeding
it here repeatedly would leak test information into later design choices.

Usage:
  python error_analysis.py --ckpt results_v10_mech/ckpt/csa_fixed_rope_seed0.pt
  python error_analysis.py --ckpt ... --max-rows 64 --top-tokens 80
"""
import argparse
import re
import hashlib
import json
import math
import os
import sys
import tempfile
import time

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except (ValueError, OSError):
            pass

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
sys.path.insert(0, REPO)

import numpy as np
import torch

import exp_lib as L

try:  # installs the RoPE / v7 layer variants used by the frozen checkpoints
    import v7_supp  # noqa: F401
except Exception as _v7e:
    print(f'[errors] NOTE: v7_supp not importable ({type(_v7e).__name__}: {_v7e}); only built-in layer variants can be loaded')


def _atomic_write_lines(path, lines):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=os.path.basename(path) + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            for _line in lines:
                f.write(_line)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _fmt_num(v, fmt, fallback='n/a'):
    return fmt.format(v) if isinstance(v, (int, float)) and math.isfinite(v) else fallback


def _pooled_line(res):
    _mc = res.get('mean_ce')
    if _mc is None:
        return '- pooled CE: n/a'
    return f'- pooled CE: {_fmt_num(_mc, "{:.4f}")}; pooled PPL: {_fmt_num(res.get("pooled_ppl"), "{:.3f}")}'


def _nonnegative_int(text):
    """argparse type for row/rank counts: a negative value is not a subset."""
    if not isinstance(text, str):
        text = str(text)
    if re.fullmatch(r'(?:0|[1-9][0-9]*)', text) is None:
        raise argparse.ArgumentTypeError(
            f'expected a non-negative integer, got {text!r}') from None
    try:
        value = int(text)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f'expected a non-negative integer, got {text!r}') from None
    if value < 0:
        raise argparse.ArgumentTypeError(
            f'expected a non-negative integer, got {text!r}')
    return value


def _positive_int(text):
    """argparse type for geometry sizes: zero would size no tensor."""
    value = _nonnegative_int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(
            f'expected a positive integer, got {text!r}')
    return value


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
    except Exception:
        # Fall back to a plain load when the mmap path cannot serve this
        # file; if that also fails, let the caller refuse.
        try:
            d = torch.load(path, map_location='cpu', weights_only=False)
        except Exception as _e:
            raise SystemExit(f'[errors] REFUSE: {path} cannot be loaded as a checkpoint ({type(_e).__name__}: {_e})') from _e
    if not isinstance(d, dict) or 'sd' not in d:
        raise SystemExit(f'[errors] REFUSE: {path} is not a CSA checkpoint (no state_dict)')
    d = dict(d)
    sd = d['sd']
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}

    def _exact_positive_int(_k, _default):
        # Architecture dimensions are exact positive integer counts (same
        # contract as _compare_reason): a bool/fractional/out-of-range value
        # must not be rounded to a different architecture.
        _raw = d.get(_k, _default)
        _v = L._as_seed_int(_raw)
        if _v is None or _v <= 0:
            raise SystemExit(f'[errors] REFUSE: checkpoint {_k}={_raw!r} is not a positive exact integer')
        return _v

    vocab = _exact_positive_int('vocab', 8192)
    dim = _exact_positive_int('d', 256)
    n_layers = _exact_positive_int('n_layers', 6)
    n_heads = _exact_positive_int('n_heads', 8)
    d_head = _exact_positive_int('d_head', 32)
    _ms_raw = d.get('train_len') or d.get('max_seq') or 512
    max_seq = L._as_seed_int(_ms_raw)
    if max_seq is None or max_seq <= 0:
        raise SystemExit(f'[errors] REFUSE: checkpoint train_len/max_seq={_ms_raw!r} is not a positive exact integer')
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


def _compare_reason(meta_a, meta_b, seq_len, max_seq_a=None, max_seq_b=None):
    """Why two checkpoints cannot be compared token-for-token (None = OK).

    `compare_token_errors` pairs token positions and decoded strings, which is
    only meaningful when both checkpoints share the token space and the model
    geometry.  It also runs the second model on the first model's validation
    batch, so a shorter position table would fail after most of the work was
    done."""
    _arch = {'vocab': 8192, 'd': 256, 'n_layers': 6, 'n_heads': 8,
             'd_head': 32}
    for _k, _default in _arch.items():
        # Architecture dimensions are exact positive integer counts.  A bool,
        # a fractional value or a value above the exact-float range must not be
        # rounded to a different architecture and then accepted as aligned.
        _va = L._as_seed_int(meta_a.get(_k, _default))
        _vb = L._as_seed_int(meta_b.get(_k, _default))
        if _va is None or _vb is None or _va <= 0 or _vb <= 0:
            return f'{_k} is not a positive exact integer in both checkpoints'
        if _va != _vb:
            return f'architecture differs on {_k} ({_va!r} vs {_vb!r})'
    _mr_a = meta_a.get('mlp_ratio', 4.0)
    _mr_b = meta_b.get('mlp_ratio', 4.0)
    if (isinstance(_mr_a, (bool, np.bool_, np.datetime64, np.timedelta64,
                           complex, np.complexfloating))
            or isinstance(_mr_b, (bool, np.bool_, np.datetime64, np.timedelta64,
                                  complex, np.complexfloating))):
        return 'mlp_ratio is a boolean/temporal/complex value in at least one checkpoint'
    try:
        _mr_a = float(_mr_a)
        _mr_b = float(_mr_b)
    except (TypeError, ValueError, OverflowError):
        return 'mlp_ratio is not numeric in both checkpoints'
    if not (math.isfinite(_mr_a) and math.isfinite(_mr_b)
            and _mr_a > 0.0 and _mr_b > 0.0):
        return 'mlp_ratio is not a finite positive number in both checkpoints'
    if _mr_a != _mr_b:
        return f'architecture differs on mlp_ratio ({_mr_a!r} vs {_mr_b!r})'
    if meta_a.get('cfg') != meta_b.get('cfg'):
        return 'layer configuration differs between the checkpoints'
    for _k in ('train_fp',):
        # Mirror of exp_lib.pair_reason: a boolean/temporal/complex flag is
        # not a corpus fingerprint, so both-boolean must be refused rather
        # than pair two garbage-provenance checkpoints as token-aligned.
        _va, _vb = meta_a.get(_k), meta_b.get(_k)
        if (isinstance(_va, (bool, np.bool_)) or L._is_nonmeasurement_scalar(_va)
                or isinstance(_vb, (bool, np.bool_)) or L._is_nonmeasurement_scalar(_vb)):
            return f'unverifiable {_k} (a non-measurement value is not a corpus fingerprint)'
        if (_va is None) != (_vb is None):
            return f'unverifiable {_k} (only one checkpoint records it)'
        if _va is not None and str(_va) != str(_vb):
            return f'the checkpoints were measured on different {_k} slices'

    def _eval_slice_fp(meta):
        # A checkpoint may persist its evaluation-slice fingerprint under
        # either `eval_fp` or `val_fp`.  Treat the two spellings as one
        # identity so a one-sided/mismatched slice cannot slip through.
        for _name in ('eval_fp', 'val_fp'):
            if meta.get(_name) is not None:
                return _name, meta.get(_name)
        return None, None

    def _eval_slice_conflict(meta):
        # A checkpoint may persist its evaluation-slice fingerprint under
        # either spelling, but a single checkpoint carrying BOTH with
        # different values is internally inconsistent: taking the first one
        # would let a mismatched slice pass as token-aligned.
        _ea, _eb = meta.get('eval_fp'), meta.get('val_fp')
        return (_ea is not None and _eb is not None
                and str(_ea) != str(_eb))

    if _eval_slice_conflict(meta_a) or _eval_slice_conflict(meta_b):
        return ('unverifiable evaluation-slice fingerprint (a checkpoint '
                'records conflicting eval_fp/val_fp values)')

    _ka, _va = _eval_slice_fp(meta_a)
    _kb, _vb = _eval_slice_fp(meta_b)
    if (isinstance(_va, (bool, np.bool_)) or L._is_nonmeasurement_scalar(_va)
            or isinstance(_vb, (bool, np.bool_)) or L._is_nonmeasurement_scalar(_vb)):
        return 'unverifiable evaluation-slice fingerprint (a non-measurement value is not an evaluation-slice fingerprint)'
    if (_va is None) != (_vb is None):
        return 'unverifiable evaluation-slice fingerprint (only one checkpoint records it)'
    if _va is not None and str(_va) != str(_vb):
        return (f'the checkpoints were measured on different evaluation slices '
                f'({_ka}={_va!r} vs {_kb}={_vb!r})')
    _ca, _cb = meta_a.get('code'), meta_b.get('code')
    if (isinstance(_ca, (bool, np.bool_)) or L._is_nonmeasurement_scalar(_ca)
            or isinstance(_cb, (bool, np.bool_)) or L._is_nonmeasurement_scalar(_cb)):
        return 'unverifiable code semantics (a non-measurement value is not a code-semantics fingerprint)'
    if (_ca is None) != (_cb is None):
        return 'unverifiable code semantics (only one checkpoint records it)'
    if _ca is not None and str(_ca) != str(_cb):
        return f'checkpoint code semantics differ ({_ca!r} vs {_cb!r})'
    _sl = L._as_seed_int(seq_len)
    if _sl is None or _sl <= 0:
        return 'sequence length is not an exact positive integer'
    _msb = max_seq_b if max_seq_b is not None else meta_b.get('train_len') or meta_b.get('max_seq')
    if _msb is not None:
        _msb = L._as_seed_int(_msb)
        if _msb is None or _msb <= 0:
            return 'checkpoint B position-table capacity is not an exact positive integer'
        if _sl > _msb:
            return (f'requested seq_len={_sl} exceeds checkpoint B position-table '
                    f'capacity max_seq={_msb}')
    # A's capacity is checked symmetrically: B is run on A's validation
    # batch, but A itself is also evaluated at seq_len, so an over-capacity
    # table must be refused before either model runs.
    _msa = max_seq_a if max_seq_a is not None else meta_a.get('train_len') or meta_a.get('max_seq')
    if _msa is not None:
        _msa = L._as_seed_int(_msa)
        if _msa is None or _msa <= 0:
            return 'checkpoint A position-table capacity is not an exact positive integer'
        if _sl > _msa:
            return (f'requested seq_len={_sl} exceeds checkpoint A position-table '
                    f'capacity max_seq={_msa}')
    return None


def _roundtrip(d):
    return json.loads(json.dumps(d, default=str))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--compare-ckpt', default=None, help='optional second checkpoint for paired per-token delta analysis')
    ap.add_argument('--outdir', default='analysis_errors')
    ap.add_argument('--dataset', default='wikitext', choices=['wikitext', 'fineweb'])
    ap.add_argument('--seq-len', type=_positive_int, default=None, help='evaluation sequence length (default: checkpoint max_seq)')
    ap.add_argument('--n-train-tokens', type=_positive_int, default=8000000)
    ap.add_argument('--val-seqs', type=_positive_int, default=128)
    ap.add_argument('--max-rows', type=_nonnegative_int, default=64)
    ap.add_argument('--top-tokens', type=_nonnegative_int, default=80)
    ap.add_argument('--top-sequences', type=_nonnegative_int, default=30)
    ap.add_argument('--chunk', type=_positive_int, default=8)
    ap.add_argument('--device', default=None)
    args = ap.parse_args(argv)
    device = torch.device(args.device) if args.device else L.DEVICE
    os.makedirs(args.outdir, exist_ok=True)
    model, meta = _load_model(args.ckpt)
    model.to(device)
    if args.seq_len is not None:
        seq_len = args.seq_len
    else:
        _ms = meta.get('train_len') or meta.get('max_seq')
        if _ms is None:
            seq_len = 512
        else:
            seq_len = L._as_seed_int(_ms)
            if seq_len is None or seq_len <= 0:
                raise SystemExit(f'[errors] REFUSE: checkpoint train_len/max_seq={_ms!r} is not a positive exact integer')
    # Refuse before any corpus work when the requested evaluation length
    # exceeds the checkpoint's position-table capacity (the paired path's
    # capacity check only covers --compare-ckpt).
    _cap = L._as_seed_int(meta.get('train_len') or meta.get('max_seq'))
    if _cap is not None and seq_len > _cap:
        raise SystemExit(f'[errors] REFUSE: requested seq_len={seq_len} exceeds checkpoint position-table capacity (max_seq={_cap})')
    _vocab_cfg = L._as_seed_int(meta.get('vocab', 8192))
    if _vocab_cfg is None or _vocab_cfg <= 0:
        raise SystemExit(f"[errors] REFUSE: checkpoint vocab={meta.get('vocab', 8192)!r} is not a positive exact integer")
    cfg = {'dataset': args.dataset, 'seq_len': seq_len, 'n_train_tokens': args.n_train_tokens,
           'val_seqs': args.val_seqs, 'vocab_size': _vocab_cfg}
    train_ids, val_batch, vocab, tok, val_bnd = L.load_run_corpus(cfg)
    if args.dataset != 'wikitext':
        tok = None
    if val_batch.shape[1] < seq_len:
        raise SystemExit(f'[errors] REFUSE: corpus rows are {int(val_batch.shape[1])} tokens wide, fewer than seq_len={seq_len} — refusing instead of scoring (and labelling) truncated rows')
    n_rows = min(int(args.max_rows), int(val_batch.shape[0]))
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
        _why = _compare_reason(meta, meta_b, seq_len,
                               max_seq_a=getattr(model, 'max_seq', None),
                               max_seq_b=getattr(model_b, 'max_seq', None))
        if _why is not None:
            raise SystemExit(f'[errors] REFUSE: --compare-ckpt is not a token-aligned '
                             f'comparison: {_why}. Compare checkpoints from the same '
                             'tokenizer/evaluation slice and geometry (e.g. two arms '
                             'from the same run panel).')
        model_b.to(device)
        paired = L.compare_token_errors(model, model_b, val_batch, vocab, device,
                                        tokenizer=tok, top_n=args.top_tokens, chunk=args.chunk)
        paired['checkpoint_a'] = os.path.abspath(args.ckpt)
        paired['checkpoint_b'] = os.path.abspath(args.compare_ckpt)
        paired['checkpoint_a_sha256'] = res['checkpoint_sha256']
        paired['checkpoint_b_sha256'] = _sha256(args.compare_ckpt)
        paired['split'] = 'validation'
        L.atomic_write_json(os.path.join(args.outdir, 'paired_errors.json'), paired, indent=2)
        _lines = [json.dumps(L._json_clean({'kind': 'paired_token', 'rank': i + 1, **rec}), ensure_ascii=False, allow_nan=False) + '\n'
                  for i, rec in enumerate(paired.get('worst_tokens_for_b') or [])]
        _atomic_write_lines(os.path.join(args.outdir, 'paired_bad_cases.jsonl'), _lines)
        del model_b
    # JSONL: one line per bad case, easiest to filter/grep downstream.
    jsonl = os.path.join(args.outdir, 'bad_cases.jsonl')
    _lines = [json.dumps(L._json_clean({'kind': 'token', 'rank': i + 1, **rec}), ensure_ascii=False, allow_nan=False) + '\n'
              for i, rec in enumerate(res.get('worst_tokens') or [])]
    _lines += [json.dumps(L._json_clean({'kind': 'sequence', 'rank': i + 1, **rec}), ensure_ascii=False, allow_nan=False) + '\n'
               for i, rec in enumerate(res.get('worst_sequences') or [])]
    _atomic_write_lines(jsonl, _lines)
    L.atomic_write_json(os.path.join(args.outdir, 'summary.json'), res, indent=2)
    lines = ['# Bad-case analysis', '',
             f'- checkpoint: `{os.path.abspath(args.ckpt)}`',
             f'- sha256: `{res["checkpoint_sha256"]}`',
             f'- split: **validation only** ({n_rows} rows x {seq_len} tokens)',
             _pooled_line(res),
             f'- checkpoint health: `{(meta.get("curve_health") or {}).get("status", "unknown")}`', '']
    lines += ['## Position / target buckets', '',
              '| bucket | n | mean CE | PPL |', '|---|---|---|---|']
    for k, v in sorted((res.get('buckets') or {}).items()):
        _mce = v.get('mean_ce')
        _ppl = v.get('perplexity')
        lines.append(f'| {k} | {v.get("n")} | {("n/a" if _mce is None else f"{_mce:.3f}")} | {("n/a" if _ppl is None else f"{_ppl:.2f}")} |')
    lines += ['', '## Worst tokens', '',
              '| # | row | pos | CE | target | predicted | entropy | rare | boundary |', '|---|---|---|---|---|---|---|---|---|']
    for i, t in enumerate(res.get('worst_tokens') or []):
        tgt = (t.get('target_text') or '').replace('|', '\\|').replace('\n', '\\n')[:80]
        prd = (t.get('pred_text') or '').replace('|', '\\|').replace('\n', '\\n')[:80]
        lines.append(f'| {i + 1} | {t["row"]} | {t["position"]} | {t["ce"]:.3f} | `{t.get("target_id")}` {tgt} | `{t.get("pred_id")}` {prd} | {t["entropy"]:.3f} | {t.get("rare_target")} | {t.get("is_boundary_token")} |')
    lines += ['', '## Worst sequences', '',
              '| # | row | mean CE | NLL | context (truncated) |', '|---|---|---|---|---|']
    for i, s in enumerate(res.get('worst_sequences') or []):
        ctx = (s.get('context') or '').replace('|', '\\|').replace('\n', ' ')[:240]
        lines.append(f'| {i + 1} | {s["row"]} | {s["mean_ce"]:.3f} | {s["nll"]:.2f} | {ctx} |')
    if paired:
        lines += ['', '## Paired arm comparison', '',
                  f'- B `{paired.get("checkpoint_b")}` minus A `{paired.get("checkpoint_a")}` mean CE: **{_fmt_num(paired.get("mean_delta_b_minus_a"), "{:+.4f}")}**',
                  f'- B worse at {paired.get("b_worse_at_n")} token positions; A worse at {paired.get("a_worse_at_n")}.', '',
                  '| # | row | pos | target | CE A | CE B | delta (B−A) |', '|---|---|---|---|---|---|---|']
        for i, t in enumerate(paired.get('worst_tokens_for_b') or []):
            tg = (t.get('target_text') or '').replace('|', '\\|').replace('\n', '\\n')[:60]
            lines.append(f'| {i + 1} | {t["row"]} | {t["position"]} | `{t["target_id"]}` {tg} | {t["ce_a"]:.3f} | {t["ce_b"]:.3f} | {t["delta_b_minus_a"]:+.3f} |')
    lines += ['', '> This file is a diagnostic starting point, not a result table. Inspect the labelled target and context before declaring a model failure: annotation errors, near-duplicates, genuinely ambiguous text and rare tokens all look like "high CE".', '']
    L.atomic_write_text(os.path.join(args.outdir, 'bad_cases.md'), '\n'.join(lines) + '\n')
    print(f'[errors] wrote {jsonl}, {os.path.join(args.outdir, "bad_cases.md")}, {os.path.join(args.outdir, "summary.json")}' + (' + paired comparison' if paired else ''))
    return res


if __name__ == '__main__':
    main()
