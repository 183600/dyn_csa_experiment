#!/usr/bin/env python3
"""mech_selection.py — selection-attribution mechanism probe (review response).

Produces the `analysis_mechanism/selection_attribution.json` artifact whose
contract `audit_gates.audit_selection` defines and validates.  The artifact
closes the repository's declared mechanism-evidence gap ("no selection-level
mechanism evidence", design notes §25.5 / audit note) by measuring, per
(arm, eval_len, seed):

- ``recall_at_k``  — how much of the ORACLE's top-mass blocks the learned
  top-k selection captures (fraction of oracle blocks found);
- ``mass_covered`` — the fraction of the oracle attention mass that falls
  inside the learned selection;
- ``random_baseline`` — k / n_blocks;
- outcome readouts (extra, descriptive): PPL of the learned selection, of an
  ORACLE-FORCED selection, of the read-everything (allblocks) pass, and of the
  frozen fixed-hash random control at the same k.

Oracle definition (self-consistent, same model): the block-level softmax mass
the layer itself would allocate if it could read EVERY compressed block —
i.e. the layer's own q/k_blk operands under an all-block read, softmax over
the readable blocks, averaged over heads.  It is a *dependency oracle for the
selection question*, not a claim about token-level attention.

Hard guarantees:
- NEVER reads the WikiText-103 test split: the probe refuses any path that
  mentions the frozen P5T results dir, and its data modes are `synthetic`
  ids or the VALIDATION stream only.
- zero training; read-only on the given checkpoints;
- the frozen plan's measurement semantics are untouched (cfg mutations are
  restored; the model is rebuilt per seed).

Usage:
  python mech_selection.py run --results <dir> --arm csa_fixed_rope \
      --seeds 0,1,2 [--eval-len 2048] [--n-seqs 8] [--data valid|synthetic] \
      [--out analysis_mechanism/selection_attribution.json]
  python mech_selection.py run --ckpt <path.pt> --arm csa_fixed_rope --seed 0 ...
  python mech_selection.py smoke          # tiny synthetic model, end-to-end self-check
  python mech_selection.py audit [path]    # re-run audit_gates.audit_selection
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402


def _fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"[mech] REFUSE: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _device_arg() -> "torch.device":
    import torch

    want = os.environ.get("MECH_DEVICE", "cpu")
    if want.startswith("cuda") and torch.cuda.is_available():
        return torch.device(want)
    return torch.device("cpu")


_TEST_SPLIT_HINTS = ("results_lm_test",)


def _guard_no_test_split(path: str) -> None:
    norm = os.path.normpath(os.path.abspath(path))
    for hint in _TEST_SPLIT_HINTS:
        if hint in norm.split(os.sep):
            _fail(f"the mechanism probe must never read the frozen test-split results dir; refused {path}")


def _load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _atomic_write_json(path, payload):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, ".mech_selection.tmp.json")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------- probe core
class _Capture:
    """Records every closed-path attention call's selection operands."""

    def __init__(self):
        self.calls = []


def _install_capture(cap: _Capture):
    import exp_lib as L

    _orig = L._gathered_attention_trainable

    def _gat_wrap(*a, **kw):
        sel_valid = kw.get("sel_valid")
        cap.calls.append({
            "q": a[0].detach(),
            "k_blk": a[1].detach(),
            "topk_idx": a[3].detach(),
            "last_tok": a[4].detach(),
            "sel_valid": (sel_valid.detach() if sel_valid is not None else None),
            "scale": float(a[8]),
        })
        return _orig(*a, **kw)

    L._gathered_attention_trainable = _gat_wrap
    return _orig


def _restore_capture(orig):
    import exp_lib as L

    L._gathered_attention_trainable = orig


def _install_replay(replay: list):
    """Force the selection to the recorded oracle blocks, call-order aligned."""
    import exp_lib as L

    _orig = L._indexer_trainable
    _state = {"i": 0}

    def _it_wrap(H, comp_kv, last_tok, W_DQ, W_DK, W_w, nIH, topk, *,
                 random_select=False, pre_qI=None, pre_w=None,
                 out_valid=None, causal_ends_host=None, need_soft=True):
        if _state["i"] >= len(replay):
            raise RuntimeError(
                "mech probe replay: more indexer calls than captured — the "
                "forward/call order changed between passes")
        idx, valid = replay[_state["i"]]
        _state["i"] += 1
        if out_valid is not None:
            out_valid[:] = [valid]
        return (None, idx, None)

    L._indexer_trainable = _it_wrap
    return _orig, _state


def _restore_replay(orig):
    import exp_lib as L

    L._indexer_trainable = orig


def _oracle_from_call(call, k: int):
    """Top-k blocks by the layer's own block-level attention mass (allblocks oracle).

    Returns (idx [T, k] int32, valid [T, k] bool, mass [T, B] float,
    readable [T, B] bool).  k_eff = min(k, B); rows whose readable count is
    below k keep the same padding convention as `_indexer_selection` — the
    padded slots are flagged invalid in `valid`.
    """
    import torch
    import exp_lib as L

    q, k_blk, last_tok, scale = call["q"], call["k_blk"], call["last_tok"], call["scale"]
    T, B = int(q.shape[0]), int(k_blk.shape[0])
    dev = q.device
    if B == 0 or T == 0:
        return None
    readable = L.block_readable(L._arange_cache(T, dev), last_tok)
    if q.dim() != 3 or k_blk.dim() != 3:
        raise RuntimeError(f"mech probe: unexpected operand ranks q={tuple(q.shape)} k_blk={tuple(k_blk.shape)}")
    scores = torch.einsum("thd,bhd->thb", q.to(torch.float32), k_blk.to(torch.float32)) * scale
    scores = scores.masked_fill(~readable[:, None, :], float("-inf"))
    mass = torch.softmax(scores, dim=-1).mean(dim=1)  # [T, B]
    mass_row = mass.clone()
    kk = int(min(k, B))
    mass_m = mass_row.masked_fill(~readable, float("-inf"))
    vals, idx = torch.topk(mass_m, kk, dim=-1)
    valid = torch.isfinite(vals)
    idx = idx.to(torch.int32)
    return idx, valid, mass_row, readable


def _row_sets(idx, valid):
    out = []
    idx_np = idx.cpu().numpy()
    valid_np = (valid.cpu().numpy() if valid is not None
                else np.ones(idx_np.shape, dtype=bool))
    for r in range(idx_np.shape[0]):
        out.append(set(int(b) for b, ok in zip(idx_np[r], valid_np[r]) if ok))
    return out


def _probe_arm(model, val_batch, device, k: int):
    """Run the four passes and return per-seed mechanism measurements.

    Passes (all under no_grad, cfg mutations restored):
      1. learned   — normal forward; capture the learned selections;
      2. allblocks — index_topk -> every block; capture q/k_blk and compute the
                     oracle mass + oracle top-k; ppl_allblocks readout;
      3. oracle-forced — replay the oracle selections; ppl_oracle readout;
      4. random-hash — the frozen fixed-hash control at the same k (readout).
    """
    import torch
    import exp_lib as L

    model.eval()
    blocks = list(model.blocks)
    saved_topk = [int(getattr(blk.attn.cfg, "index_topk", 32)) for blk in blocks]
    saved_mode = [getattr(blk.attn.cfg, "indexer_mode", "learned") for blk in blocks]
    cap = _Capture()
    orig_gat = _install_capture(cap)
    orig_it = L._indexer_trainable
    try:
        with torch.no_grad():
            # -- pass 1: learned ------------------------------------------------
            cap.calls.clear()
            ppl_learned = float(L.eval_ppl(model, val_batch, device, chunk=2))
            learned = [(c["topk_idx"], c["sel_valid"]) for c in cap.calls]

            # -- pass 2: allblocks oracle --------------------------------------
            for blk in blocks:
                blk.attn.cfg.index_topk = 10 ** 6
            cap.calls.clear()
            ppl_allblocks = float(L.eval_ppl(model, val_batch, device, chunk=2))
            if len(cap.calls) != len(learned):
                raise RuntimeError(
                    f"mech probe: pass-order mismatch — learned pass had {len(learned)} "
                    f"selection calls, allblocks pass had {len(cap.calls)}; the two eval_ppl "
                    f"invocations are not comparable")
            oracle, metrics = [], []
            for call, (l_idx, l_valid) in zip(cap.calls, learned):
                got = _oracle_from_call(call, k)
                if got is None:
                    oracle.append(None)
                    metrics.append(None)
                    continue
                o_idx, o_valid, mass, readable = got
                oracle.append((o_idx, o_valid))
                l_sets = _row_sets(l_idx, l_valid)
                o_sets = _row_sets(o_idx, o_valid)
                mass_np = mass.cpu().numpy()
                read_np = readable.cpu().numpy()
                rec_vals, cov_vals = [], []
                for r in range(len(l_sets)):
                    o_set = o_sets[r]
                    denom = float(read_np[r].sum())
                    if not o_set or denom <= 0.0:
                        continue
                    l_set = l_sets[r]
                    if l_set:
                        inter = len(l_set & o_set)
                        rec_vals.append(inter / len(o_set))
                        cov_vals.append(float(sum(mass_np[r, b] for b in l_set)) / denom)
                    else:
                        rec_vals.append(0.0)
                        cov_vals.append(0.0)
                metrics.append((rec_vals, cov_vals))
            for blk, old in zip(blocks, saved_topk):
                blk.attn.cfg.index_topk = old

            # -- pass 3: oracle-forced readout ----------------------------------
            # keep a 1:1 call alignment: a call with no measurable oracle falls
            # back to replaying that call's own learned selection (no-op).
            replay = [(o if o is not None else learned[i]) for i, o in enumerate(oracle)]
            _install_replay(replay)
            try:
                ppl_oracle = float(L.eval_ppl(model, val_batch, device, chunk=2))
            finally:
                _restore_replay(orig_it)

            # -- pass 4: frozen fixed-hash random control ----------------------
            for blk in blocks:
                blk.attn.cfg.indexer_mode = "random"
            try:
                ppl_random = float(L.eval_ppl(model, val_batch, device, chunk=2))
            finally:
                for blk, old in zip(blocks, saved_mode):
                    blk.attn.cfg.indexer_mode = old
    finally:
        _restore_capture(orig_gat)
        L._indexer_trainable = orig_it
        for blk, old in zip(blocks, saved_topk):
            blk.attn.cfg.index_topk = old
        for blk, old in zip(blocks, saved_mode):
            blk.attn.cfg.indexer_mode = old

    recalls = [v for m in metrics if m for v in m[0]]
    covers = [v for m in metrics if m for v in m[1]]
    n_queries = int(sum(len(m[0]) for m in metrics if m))
    B_seen = [int(c["k_blk"].shape[0]) for c in cap.calls if int(c["k_blk"].shape[0])]
    return {
        "n_calls": len(cap.calls),
        "n_queries": n_queries,
        "n_blocks": (int(max(B_seen)) if B_seen else 0),
        "recall_at_k": (float(np.mean(recalls)) if recalls else None),
        "mass_covered": (float(np.mean(covers)) if covers else None),
        "random_baseline": (float(k) / float(max(B_seen)) if B_seen else None),
        "ppl_learned": ppl_learned,
        "ppl_oracle_forced": ppl_oracle,
        "ppl_allblocks": ppl_allblocks,
        "ppl_random_hash": ppl_random,
    }


def _cell_payload(arm, eval_len, k, stats_per_seed):
    per_seed = {}
    for seed, st in sorted(stats_per_seed.items()):
        per_seed[str(int(seed))] = {
            "recall_at_k": st["recall_at_k"],
            "mass_covered": st["mass_covered"],
            "random_baseline": st["random_baseline"],
            "n_queries": int(st["n_queries"]),
            "ppl_learned": st["ppl_learned"],
            "ppl_oracle_forced": st["ppl_oracle_forced"],
            "ppl_allblocks": st["ppl_allblocks"],
            "ppl_random_hash": st["ppl_random_hash"],
        }
    agg = [st for st in stats_per_seed.values() if st["recall_at_k"] is not None]
    return {
        "arm": arm,
        "eval_len": int(eval_len),
        "k": int(k),
        "n_blocks": int(max((st["n_blocks"] for st in stats_per_seed.values()), default=0)),
        "n_queries": int(sum(int(st["n_queries"]) for st in stats_per_seed.values())),
        "recall_at_k": (float(np.mean([st["recall_at_k"] for st in agg])) if agg else None),
        "mass_covered": (float(np.mean([st["mass_covered"] for st in agg])) if agg else None),
        "random_baseline": stats_per_seed[list(stats_per_seed)[0]]["random_baseline"],
        "per_seed": per_seed,
    }


ARTIFACT_DEFINITION = {
    "probe": "mech_selection.py",
    "oracle": ("block-level softmax mass of the layer's own q/k_blk operands under an "
               "all-block read (softmax over readable blocks, averaged over heads) — "
               "a dependency oracle for the selection question, not token-level attention"),
    "recall_at_k": "fraction of the oracle top-k blocks found by the learned selection",
    "mass_covered": "fraction of the oracle mass inside the learned selection",
    "random_baseline": "k / n_blocks",
    "ppl_readouts": ("descriptive outcome readouts: learned / oracle-forced / allblocks "
                     "(read everything) / frozen fixed-hash random at the same k"),
    "data": "validation stream or synthetic ids; the test split is never read",
}


# ---------------------------------------------------------------- model/data
def _find_ckpt(results, arm, seed, ckpt_dir=None, explicit=None):
    import exp_lib as L

    if explicit:
        return explicit
    cand_dirs = [ckpt_dir] if ckpt_dir else []
    cand_dirs += [os.path.join(results, "ckpt"), os.path.join(results, "checkpoints"), results]
    for d in cand_dirs:
        if not d:
            continue
        p = os.path.join(d, L.direct_ckpt_name(arm, seed))
        if os.path.exists(p):
            return p
    _fail(f"no checkpoint for arm {arm!r} seed {seed} under {results!r} "
          f"(looked for {L.direct_ckpt_name(arm, seed)} in {cand_dirs}); pass --ckpt explicitly")


def _load_model(ckpt_path, device):
    import torch
    import exp_lib as L
    import v7_supp  # noqa: F401 — installs the *_rope registry + SmallGPT factory

    _guard_no_test_split(ckpt_path)
    d = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = (int(d.get("d", 256)), int(d.get("n_layers", 6)),
            int(d.get("n_heads", 8)), int(d.get("d_head", 32)))
    vocab = int(d.get("vocab", 8192))
    cfgs = [copy.deepcopy(c) for c in d["cfg"]]
    model = L.SmallGPT(vocab, *arch, d.get("train_len", 512), cfgs,
                       mlp_ratio=d.get("mlp_ratio", 4.0)).to(device)
    model.load_state_dict(d.get("sd"))
    model.eval()
    k = None
    for c in cfgs:
        if getattr(c, "kind", "") in ("csa",):
            k = int(getattr(c, "index_topk", 32))
            break
    if k is None:
        _fail(f"{ckpt_path}: no CSA layer found — the probe measures CSA selection only")
    return model, k, arch


def _val_batch(eval_len, n_seqs, data_mode, device, seed=0):
    import torch

    if data_mode == "synthetic":
        g = torch.Generator().manual_seed(int(seed))
        ids = torch.randint(0, 8192, (int(n_seqs), int(eval_len)), generator=g)
        return ids.numpy().astype(np.int64)
    if data_mode == "valid":
        import exp_lib as L

        n_train = int(os.environ.get("MECH_TRAIN_TOKENS", "8000000"))
        got = L.load_wikitext(int(eval_len), n_train)
        val_ids = np.asarray(got[1]).reshape(-1)
        need = int(n_seqs) * int(eval_len)
        if val_ids.size < need:
            _fail(f"validation stream has {val_ids.size} ids, need {need}; lower --n-seqs")
        return val_ids[:need].reshape(int(n_seqs), int(eval_len)).astype(np.int64)
    _fail(f"unknown data mode {data_mode!r} (synthetic|valid; the test split is not an option)")


# ---------------------------------------------------------------- commands
def cmd_run(args):
    device = _device_arg()
    seeds = [int(s) for s in str(args.seeds).split(",") if str(s).strip() != ""]
    if not seeds:
        _fail("--seeds is empty")
    if args.ckpt and len(seeds) > 1:
        _fail("--ckpt loads ONE checkpoint; use --results with --seeds")
    stats, k_used, eval_len = {}, None, int(args.eval_len)
    for seed in seeds:
        ckpt = _find_ckpt(args.results, args.arm, seed, ckpt_dir=args.ckpt_dir,
                          explicit=args.ckpt)
        model, k, _arch = _load_model(ckpt, device)
        k_used = k
        batch = _val_batch(eval_len, args.n_seqs, args.data, device, seed=seed)
        st = _probe_arm(model, batch, device, k)
        stats[seed] = st
        print(f"[mech] {args.arm} seed {seed}: recall@k={st['recall_at_k']:.4f} "
              f"mass={st['mass_covered']:.4f} base={st['random_baseline']:.4f} "
              f"ppl learned={st['ppl_learned']:.3f} oracle={st['ppl_oracle_forced']:.3f} "
              f"allblocks={st['ppl_allblocks']:.3f} random={st['ppl_random_hash']:.3f}",
              flush=True)
        del model
    cell = _cell_payload(args.arm, eval_len, k_used, stats)
    cell.update({"ckpt_provenance": {"results": args.results, "seeds": seeds,
                                     "data_mode": args.data}})
    artifact = {"schema_version": 1, "definition": ARTIFACT_DEFINITION, "cells": [cell]}
    _atomic_write_json(args.out, artifact)
    print(f"[mech] wrote {args.out} ({len(seeds)} seed(s)); re-run: "
          f"python audit_gates.py selection (reads analysis_mechanism/selection_attribution.json)")
    return 0


def cmd_smoke(args):
    import torch
    import exp_lib as L
    import v7_supp  # noqa: F401

    device = _device_arg()
    torch.manual_seed(7)
    cfgs = [L.AttnCfg(kind="csa", dynamic=False, block_size=4, overlap=L.OVERLAP,
                      index_topk=4) for _ in range(2)]
    model = L.SmallGPT(64, 32, 2, 2, 16, 128, cfgs, mlp_ratio=2).to(device)
    model.eval()
    g = torch.Generator().manual_seed(11)
    batch = torch.randint(0, 64, (2, 128), generator=g).numpy().astype(np.int64)
    st = _probe_arm(model, batch, device, k=4)
    if st["recall_at_k"] is None or st["mass_covered"] is None:
        _fail("smoke: no measurable rows — probe produced nothing")
    cell = _cell_payload("csa_fixed_smoke", 128, 4, {0: st})
    artifact = {"schema_version": 1, "definition": ARTIFACT_DEFINITION, "cells": [cell]}

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "selection_attribution.json")
        _atomic_write_json(out, artifact)
        import audit_gates as AG

        res = AG.audit_selection(out)
        status = res.get("status")
        problems = res.get("problems") or []
        print(f"[mech] smoke audit status={status} problems={problems}")
        if status != "ok" or problems:
            for p in problems:
                print(f"[mech]   - {p}")
            _fail("smoke: audit_selection did not accept the artifact")
    print(f"[mech] smoke OK: recall@k={st['recall_at_k']:.4f} mass={st['mass_covered']:.4f} "
          f"base={st['random_baseline']:.4f} "
          f"ppl learned={st['ppl_learned']:.3f} oracle={st['ppl_oracle_forced']:.3f} "
          f"allblocks={st['ppl_allblocks']:.3f} random={st['ppl_random_hash']:.3f}")
    return 0


def cmd_audit(args):
    import audit_gates as AG

    res = AG.audit_selection(args.path)
    problems = res.get("problems") or []
    print(f"status={res.get('status')} cells={len(res.get('cells') or [])} problems={len(problems)}")
    for p in problems:
        print(f"  - {p}")
    return 0 if (res.get("status") == "ok" and not problems) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="selection-attribution mechanism probe")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="probe trained checkpoints and write the artifact")
    p_run.add_argument("--results", default=None, help="results dir holding the checkpoints")
    p_run.add_argument("--ckpt", default=None, help="explicit single checkpoint path")
    p_run.add_argument("--arm", required=True)
    p_run.add_argument("--seeds", default="0", help="comma-separated seed list")
    p_run.add_argument("--eval-len", type=int, default=2048)
    p_run.add_argument("--n-seqs", type=int, default=8)
    p_run.add_argument("--data", choices=("valid", "synthetic"), default="valid")
    p_run.add_argument("--ckpt-dir", default=None)
    p_run.add_argument("--out", default=os.path.join("analysis_mechanism", "selection_attribution.json"))
    p_run.set_defaults(func=cmd_run)

    p_smoke = sub.add_parser("smoke", help="tiny synthetic end-to-end self-check")
    p_smoke.set_defaults(func=cmd_smoke)

    p_audit = sub.add_parser("audit", help="re-run audit_gates.audit_selection")
    p_audit.add_argument("path", nargs="?", default=os.path.join("analysis_mechanism", "selection_attribution.json"))
    p_audit.set_defaults(func=cmd_audit)

    args = ap.parse_args(argv)
    if args.cmd == "run":
        if not (args.ckpt or args.results):
            _fail("run needs --results or --ckpt")
        if args.results:
            _guard_no_test_split(args.results)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
