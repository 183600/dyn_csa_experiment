#!/usr/bin/env python3
"""CPU-only checks for the open-block reading rule.

These are the substantive gates the panel must pass before any GPU run:

  K1  config / segmenter: `max_block=None` is an explicit no-cap bound,
      the position capacity is independent of it, and the default is closed.
  K2  optimized causal prefix pooling == per-query reference (forward and
      gradients), for full-prefix and outside-window scopes.  At a block end
      the full-prefix summary equals the closed block pooling exactly.
  K3  prefix invariance: a truncated input gives the same outputs on the
      common prefix.
  K4  future perturbation: replacing x_{t+1:} leaves the output at t
      unchanged; its gradient w.r.t. the future input embeddings is zero.
  K5  candidate masks: future/out-of-prefix candidates are never selected;
      selected ids are unique and padding is invalid.
  K6  edge cases: length 1, first token, natural/forced cuts, window edge,
      no-cap tail.
  K7  closed regression: the default closed path is unchanged (and the
      open modes leave the dynamic cut table untouched).
  K8  singleton semantics: a length-1 pooled block, not a raw-token K/V.
  K9  HCA budget declaration: open HCA must declare the extra-read budget.

Run directly (`python test_open_block.py`) or through
`python test_review_fixes.py` (the CI entry point imports CHECKS).
"""
import math
import os
import sys

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


def _bid_from_cuts(n, cuts, min_block=2, max_block=12):
    return L.blocks_from_cuts(n, np.asarray(cuts, dtype=bool), min_block, max_block,
                              torch.device('cpu'))


def _tiny(mode='closed', scope='full_prefix', max_block=8, seq=32, window=6,
          vocab=97, d=32, n_layers=2, n_heads=2, d_head=16, capacity=31,
          seed=0, kind='csa'):
    torch.manual_seed(seed)
    cfg = L.AttnCfg(kind=kind, dynamic=True, chunking='cosine_learnable',
                    target_block_tokens=4, overlap=4, index_topk=6,
                    temperature=0.1, min_block=2, max_block=max_block,
                    pool_position_capacity=capacity, sliding_window=window,
                    open_block_mode=mode, open_block_scope=scope)
    if kind == 'hca' and mode != 'closed':
        cfg = L.AttnCfg(kind=kind, dynamic=True, chunking='cosine_learnable',
                        target_block_tokens=4, index_topk=6, temperature=0.1,
                        min_block=2, max_block=max_block,
                        pool_position_capacity=capacity, sliding_window=window,
                        open_block_mode=mode, open_block_scope=scope,
                        selection_budget='all_closed_plus_prefix')
    return L.SmallGPT(vocab, d, n_layers, n_heads, d_head, seq, [cfg, cfg],
                      mlp_ratio=2)


# ---------------------------------------------------------------------------
# K1 config / segmenter
# ---------------------------------------------------------------------------

def t_config_and_segmenter():
    c = L.AttnCfg()
    assert c.max_block == 128 and c.position_capacity() == 129
    assert c.open_block_mode == 'closed', 'the default must preserve the historical closed rule'
    assert c.position_policy() == 'truncate'
    n = 23
    cuts = np.zeros(n - 1, dtype=bool)
    cuts[5] = cuts[11] = True
    for mb in (32, 64, 128, None):
        b = L.blocks_from_cuts(n, cuts, 2, mb, torch.device('cpu'))
        assert int(b.numel()) == n
        if mb is not None:
            assert int(torch.bincount(b).max()) <= int(mb), (mb, b.tolist())
    # no forced cap: no cut -> one block; max_block=8 -> several blocks
    b_none = _bid_from_cuts(n, np.zeros(n - 1, dtype=bool), max_block=None)
    assert int(b_none.max()) == 0, b_none.tolist()
    b_8 = _bid_from_cuts(n, np.zeros(n - 1, dtype=bool), max_block=8)
    assert int(torch.bincount(b_8).max()) <= 8
    assert int(b_8.max()) > 0
    # explicit None must be bit-identical to an explicit bound >= n
    b_big = _bid_from_cuts(n, cuts, max_block=n + 5)
    b_none2 = _bid_from_cuts(n, cuts, max_block=None)
    assert torch.equal(b_none2, b_big)
    # position tables are independent of max_block
    m32 = L.SmallGPT(32, 16, 2, 2, 8, 8,
                     [L.AttnCfg(kind='csa', max_block=32, pool_position_capacity=129)] * 2)
    mnone = L.SmallGPT(32, 16, 2, 2, 8, 8,
                       [L.AttnCfg(kind='csa', max_block=None, pool_position_capacity=129)] * 2)
    assert int(m32.blocks[0].attn.B_pos_a.shape[0]) == 129
    assert int(mnone.blocks[0].attn.B_pos_a.shape[0]) == 129
    # invalid declarations fail closed
    for kwargs in ({'open_block_mode': 'prefix_summary', 'kind': 'full'},
                   {'open_block_mode': 'nope'},
                   {'pool_position_capacity': 0},
                   {'max_block': 1, 'min_block': 2}):
        try:
            L.AttnCfg(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f'AttnCfg accepted invalid settings {kwargs}')


# ---------------------------------------------------------------------------
# K2 prefix pooling: reference, gradients, and block-end equality
# ---------------------------------------------------------------------------

def _pool_fixture(seed=0, T=41, Fd=7, ov=3, cap=64):
    rng = np.random.RandomState(seed)
    cuts = np.zeros(T - 1, dtype=bool)
    for i in (1, 4, 7, 9, 14, 17, 20, 25, 28, 31, 36, 38):
        cuts[i] = True
    bid = L.blocks_from_cuts(T, cuts, 2, 12, torch.device('cpu'))
    g = torch.Generator().manual_seed(seed + 1)
    Xa = torch.randn(T, Fd, generator=g, requires_grad=True)
    Xb = torch.randn(T, Fd, generator=g, requires_grad=True)
    Za = torch.randn(T, Fd, generator=g, requires_grad=True) * 2
    Zb = torch.randn(T, Fd, generator=g, requires_grad=True) * 2
    Ba = torch.randn(cap, Fd, generator=g, requires_grad=True)
    Bb = torch.randn(cap, Fd, generator=g, requires_grad=True)
    return bid, Xa, Xb, Za, Zb, Ba, Bb


def t_prefix_pool_reference():
    bid, Xa, Xb, Za, Zb, Ba, Bb = _pool_fixture()
    T = int(bid.numel())
    nb = int(bid.max()) + 1
    for scope, window in (('full_prefix', 6), ('outside_window', 3),
                          ('outside_window', 0)):
        out = L.causal_prefix_pool(Xa, Xb, Za, Zb, bid, Ba, Bb, 3,
                                   n_blocks=nb, scope=scope, window=window)
        ref = L.causal_prefix_pool_reference(Xa, Xb, Za, Zb, bid, Ba, Bb, 3,
                                             n_blocks=nb, scope=scope, window=window)
        assert out.shape == (T, Xa.shape[1])
        assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3), (
            scope, window, float((out - ref).detach().abs().max()))
        g = torch.randn(T, Xa.shape[1], generator=torch.Generator().manual_seed(9))
        go = torch.autograd.grad((out * g).sum(), [Xa, Za, Ba], retain_graph=True)
        gr = torch.autograd.grad((ref * g).sum(), [Xa, Za, Ba], retain_graph=True)
        for a, b in zip(go, gr):
            assert torch.allclose(a, b, atol=2e-3, rtol=2e-3), (
                scope, window, float((a - b).detach().abs().max()))
    # A future token in the SAME block must not change an earlier prefix,
    # even when its logit is far outside the past range.  A block-wide shift
    # (rather than a prefix-wide one) would underflow the earlier rows to
    # zero and make them numerically depend on the future.
    for scope, window in (('full_prefix', 6), ('outside_window', 3)):
        _Za = torch.zeros_like(Za)
        _Za[-1] = 1e6
        _out = L.causal_prefix_pool(Xa, Xb, _Za, Zb, bid, Ba, Bb, 3,
                                    n_blocks=nb, scope=scope, window=window)
        _ref = L.causal_prefix_pool_reference(Xa, Xb, _Za, Zb, bid, Ba, Bb, 3,
                                              n_blocks=nb, scope=scope, window=window)
        assert torch.allclose(_out, _ref, atol=1e-4, rtol=1e-4), (
            scope, window, 'large-future-logit', float((_out - _ref).abs().max()))
    # At each block's end the full-prefix summary is the closed block pooling.
    comp, last_tok, B = L.pool_variable_blocks(Xa, Xb, Za, Zb, bid, Ba, Bb, 3,
                                               n_blocks=nb)
    pref = L.causal_prefix_pool(Xa, Xb, Za, Zb, bid, Ba, Bb, 3, n_blocks=nb,
                                scope='full_prefix')
    ends = torch.cumsum(torch.bincount(bid, minlength=B), 0)
    for b in range(B):
        t = int(ends[b]) - 1
        assert torch.allclose(comp[b], pref[t], atol=5e-4, rtol=5e-4), (
            b, t, float((comp[b] - pref[t]).detach().abs().max()))
    # Padding / empty outside-window prefixes produce a finite zero row and
    # never enter a summary through a NaN.
    pg = L._open_prefix_geometry(bid, n_blocks=nb, scope='outside_window', window=1)
    assert pg['plen'].min() >= 0
    assert torch.isfinite(pref).all()


# ---------------------------------------------------------------------------
# K3 / K4 causality
# ---------------------------------------------------------------------------

def _trunk_fp(m, ids):
    with torch.no_grad():
        return m.trunk(ids)


def t_prefix_invariance():
    seq, cut = 56, 33
    g = torch.Generator().manual_seed(4)
    ids = torch.randint(0, 97, (1, seq), generator=g)
    for mode, scope in (('closed', 'full_prefix'),
                        ('prefix_summary', 'full_prefix'),
                        ('prefix_summary', 'outside_window'),
                        ('prefix_singletons', 'full_prefix'),
                        ('prefix_singletons', 'outside_window')):
        m = _tiny(mode=mode, scope=scope, max_block=None, seq=seq, window=10)
        m.eval()
        full = _trunk_fp(m, ids)
        trunc = _trunk_fp(m, ids[:, :cut])
        d = float((full[:, :cut] - trunc).abs().max())
        assert d <= 3e-5, (mode, scope, d)


def t_future_causality():
    seq, tcut = 44, 19
    g = torch.Generator().manual_seed(5)
    ids = torch.randint(0, 97, (1, seq), generator=g)
    for mode, scope in (('closed', 'full_prefix'),
                        ('prefix_summary', 'full_prefix'),
                        ('prefix_summary', 'outside_window'),
                        ('prefix_singletons', 'full_prefix'),
                        ('prefix_singletons', 'outside_window')):
        m = _tiny(mode=mode, scope=scope, max_block=None, seq=seq, window=8)
        m.eval()
        h1 = _trunk_fp(m, ids)
        ids2 = ids.clone()
        ids2[:, tcut + 1:] = torch.randint(0, 97, (1, seq - tcut - 1),
                                           generator=torch.Generator().manual_seed(6))
        h2 = _trunk_fp(m, ids2)
        d = float((h1[:, tcut] - h2[:, tcut]).abs().max())
        assert d <= 1e-5, (mode, scope, 'future perturbation', d)
        # Future-gradient check: output at t must have zero gradient w.r.t.
        # input embeddings after t.
        m.train()
        store = {}

        def hook(_mod, _inp, out):
            out.retain_grad()
            store['emb'] = out

        handle = m.tok.register_forward_hook(hook)
        try:
            h = m.trunk(ids)
            gfuture = torch.autograd.grad(h[:, tcut].sum(), store['emb'])[0]
        finally:
            handle.remove()
        assert float(gfuture[0, tcut + 1:].abs().max()) == 0.0, (
            mode, scope, 'future gradient', float(gfuture[0, tcut + 1:].abs().max()))


# ---------------------------------------------------------------------------
# K5 candidate masks
# ---------------------------------------------------------------------------

def t_candidate_masks():
    T, C, k = 7, 11, 5
    scores = torch.randn(T, C)
    valid = torch.zeros(T, C, dtype=torch.bool)
    for t in range(T):
        valid[t, :min(C, t + 2)] = True
    # make every invalid column hugely attractive: a mask bug would select it
    scores = scores.masked_fill(~valid, 1e6)
    idx = L._indexer_selection(scores, valid, k)
    box = []
    idx2 = L._indexer_selection(scores, valid, k, out_valid=box)
    assert torch.equal(idx, idx2)
    sv = box[0]
    for t in range(T):
        sel = [int(idx[t, j]) for j in range(idx.shape[1]) if bool(sv[t, j])]
        assert len(sel) == len(set(sel)), ('duplicate candidates', t, sel)
        for j, c in enumerate(idx[t].tolist()):
            if sv[t, j]:
                assert valid[t, c], (t, j, c)
    # open indexer: future closed blocks and j>=plen are never selected
    torch.manual_seed(7)
    n, B, d, kv, nIH, hd = 9, 3, 12, 10, 2, 4
    bid = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    geo = L._open_prefix_geometry(bid, n_blocks=3, scope='full_prefix', window=99)
    ends = torch.cumsum(torch.bincount(bid, minlength=3), 0)
    last_tok = (ends - 1).clamp_min(0)
    H = torch.randn(n, d)
    comp_kv = torch.randn(B, kv)
    tok_kv = torch.randn(n, kv)
    W_DQ = torch.nn.Parameter(torch.randn(d, nIH * hd))
    W_DK = torch.nn.Parameter(torch.randn(kv, nIH * hd))
    W_w = torch.nn.Linear(d, nIH, bias=False)
    idx, sv, soft = L.lightning_indexer_open(
        H, comp_kv, last_tok, W_DQ, W_DK, W_w.weight, nIH, 4,
        prefix_geo=geo, prefix_token_kv=tok_kv, prefix_kind='singletons',
        need_soft=True)
    for t in range(n):
        for j in range(idx.shape[1]):
            if not bool(sv[t, j]):
                continue
            c = int(idx[t, j])
            if c < B:
                assert int(last_tok[c]) < t, ('future block selected', t, c, int(last_tok[c]))
            else:
                p = c - B
                assert p < int(geo['plen'][t]), ('out-of-prefix singleton selected', t, p, int(geo['plen'][t]))
                assert int(geo['token_idx'][t, p]) <= t
    assert torch.isfinite(soft).all() and float(soft.detach().min()) >= 0.0


# ---------------------------------------------------------------------------
# K6 edge cases
# ---------------------------------------------------------------------------

def t_edge_cases():
    cases = [(1, 'closed'), (1, 'prefix_summary'), (1, 'prefix_singletons'),
             (2, 'prefix_summary'), (3, 'prefix_singletons')]
    for seq, mode in cases:
        m = _tiny(mode=mode, scope='full_prefix', max_block=8, seq=seq,
                  window=3, capacity=5)
        ids = torch.randint(0, 97, (1, seq), generator=torch.Generator().manual_seed(seq))
        m.eval()
        with torch.no_grad():
            h = m.trunk(ids)
        assert torch.isfinite(h).all(), (seq, mode)
    # forced cut, natural cut, no-cap tail, window edge
    for mb, mode, scope in ((4, 'prefix_summary', 'full_prefix'),
                            (None, 'prefix_singletons', 'full_prefix'),
                            (4, 'prefix_summary', 'outside_window'),
                            (None, 'prefix_singletons', 'outside_window')):
        m = _tiny(mode=mode, scope=scope, max_block=mb, seq=33, window=5,
                  capacity=4)
        ids = torch.randint(0, 97, (1, 33), generator=torch.Generator().manual_seed(11))
        m.train()
        h = m.trunk(ids)
        loss = h.float().pow(2).mean()
        loss.backward()
        assert torch.isfinite(loss).all(), (mb, mode, scope)
        for p in m.parameters():
            assert p.grad is None or torch.isfinite(p.grad).all(), (mb, mode, scope)


# ---------------------------------------------------------------------------
# K7 closed regression / untouched cut table
# ---------------------------------------------------------------------------

def t_closed_regression():
    torch.manual_seed(12)
    ids = torch.randint(0, 97, (2, 40), generator=torch.Generator().manual_seed(12))
    m_default = L.SmallGPT(97, 32, 2, 2, 16, 40,
                           L.make_layer_cfgs(2, 'csa_dynamic'), mlp_ratio=2)
    cfg_closed = L.make_layer_cfgs(2, 'csa_dynamic_mb128_closed')
    m_closed = L.SmallGPT(97, 32, 2, 2, 16, 40, cfg_closed, mlp_ratio=2)
    # identical initialisation: the new fields consume no RNG
    sd0, sd1 = m_default.state_dict(), m_closed.state_dict()
    assert set(sd0) == set(sd1)
    for k in sd0:
        if sd0[k].is_floating_point():
            assert torch.equal(sd0[k], sd1[k]), k
    # Different `max_block` arms share the same parameter shapes (position
    # capacity is decoupled), so their initialisation must be identical.
    m32 = L.SmallGPT(97, 32, 2, 2, 16, 40,
                     L.make_layer_cfgs(2, 'csa_dynamic_mb32_closed'), mlp_ratio=2)
    sd32 = m32.state_dict()
    assert set(sd32) == set(sd0)
    for k in sd0:
        if sd0[k].is_floating_point():
            assert torch.equal(sd0[k], sd32[k]), ('init differs by max_block', k)
    m_default.eval(); m_closed.eval()
    with torch.no_grad():
        h0 = m_default.trunk(ids)
        h1 = m_closed.trunk(ids)
    assert torch.equal(h0, h1), float((h0 - h1).abs().max())
    # open modes must not mutate the dynamic cut table relative to closed:
    # same seed -> same model parameters -> same gate decisions -> same cuts.
    m_open = _tiny(mode='prefix_summary', scope='full_prefix', max_block=None,
                   seq=40, window=8, seed=12)
    m_closed2 = _tiny(mode='closed', scope='full_prefix', max_block=None,
                      seq=40, window=8, seed=12)
    for mm, on in ((m_open, True), (m_closed2, True)):
        mm.eval()
        L.enable_block_stats(mm, on)
        with torch.no_grad():
            mm.trunk(ids[:1])
    for c_open, c_closed in zip(m_open.blocks[0].attn._cuts,
                                m_closed2.blocks[0].attn._cuts):
        assert np.array_equal(np.asarray(c_open), np.asarray(c_closed)), (c_open, c_closed)
    L.enable_block_stats(m_open, False)
    L.enable_block_stats(m_closed2, False)


def t_forced_vs_natural_cuts():
    st = L._BlockStats(min_block=2, max_block=5)
    bid = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    gate = torch.tensor([False, False, False, False, True, False, False, True,
                         False, False, False])
    st.add_segmentation(bid, gate, 5)
    assert st.forced_and_natural == 1, (st.forced_and_natural, 'both')
    assert st.natural_only == 1, (st.natural_only, 'natural only')
    assert st.forced_only == 0, (st.forced_only, 'forced only')
    st2 = L._BlockStats(min_block=2, max_block=5)
    bid2 = torch.tensor([0] * 5 + [1] * 5 + [2] * 5)
    st2.add_segmentation(bid2, torch.zeros(14, dtype=torch.bool), 5)
    assert st2.forced_only == 2, (st2.forced_only, 'two forced non-tail ends')
    assert st2.natural_only == 0 and st2.forced_and_natural == 0
    st3 = L._BlockStats(min_block=2, max_block=None)
    st3.add_segmentation(bid, torch.zeros(11, dtype=torch.bool), None)
    assert st3.forced_total == 0
    # Capacity is a table size: length == capacity is fully represented
    # (offsets 0..cap-1); only length > capacity truncates.
    stc = L._BlockStats(position_capacity=4)
    stc._add(np.array([4.0, 5.0, 7.0]))
    assert stc.pos_truncated_tokens == (0 + 1 + 3), stc.pos_truncated_tokens
    assert stc.pos_truncated_blocks == 2, stc.pos_truncated_blocks


# ---------------------------------------------------------------------------
# K8 singleton semantics
# ---------------------------------------------------------------------------

def t_singleton_is_length_one_pooled():
    torch.manual_seed(13)
    T, Fd, cap = 6, 5, 12
    Xa = torch.randn(T, Fd)
    Xb = torch.randn(T, Fd)
    Za = torch.randn(T, Fd)
    Zb = torch.randn(T, Fd)
    Ba = torch.randn(cap, Fd)
    Bb = torch.randn(cap, Fd)
    # Six length-1 blocks, overlap=0: each pooled row is exactly its A token.
    bid = torch.arange(T)
    comp, _lt, B = L.pool_variable_blocks(Xa, Xb, Za, Zb, bid, Ba, Bb, 0,
                                          n_blocks=T)
    assert torch.allclose(comp, Xa, atol=1e-6, rtol=1e-6), (
        float((comp - Xa).abs().max()))
    # With overlap>0 a length-1 block would mix the previous block, which is
    # why `prefix_singletons` explicitly uses overlap=0 for its token-pure
    # representation; the raw-token variant is a different (future) mode.
    comp_ov, _lt2, _B2 = L.pool_variable_blocks(
        Xa, Xb, Za, Zb, torch.arange(T // 2).repeat_interleave(2), Ba, Bb, 3,
        n_blocks=T // 2)
    assert comp_ov.shape[0] == T // 2


# ---------------------------------------------------------------------------
# K9 HCA budget declaration
# ---------------------------------------------------------------------------

def t_hca_budget_declaration():
    try:
        L.AttnCfg(kind='hca', open_block_mode='prefix_summary')
    except ValueError:
        pass
    else:
        raise AssertionError('open HCA accepted the shared_topk budget label')
    c = L.AttnCfg(kind='hca', open_block_mode='prefix_summary',
                  selection_budget='all_closed_plus_prefix')
    assert c.selection_budget == 'all_closed_plus_prefix'


def t_open_candidate_edge_cases():
    # Empty query sets and zero available candidates must return the same
    # shapes the non-empty path promises, not crash inside a gather.
    d, nIH = 8, 2
    W_DQ, W_DK, W_w = torch.randn(d, d), torch.randn(d, d), torch.randn(nIH, d)
    empty_bid = torch.empty(0, dtype=torch.long)
    geo0 = L._open_prefix_geometry(empty_bid, n_blocks=0, scope='full_prefix',
                                   candidate_table=False)
    idx, valid, soft = L.lightning_indexer_open(
        torch.empty(0, d), torch.empty(0, d), torch.empty(0, dtype=torch.long),
        W_DQ, W_DK, W_w, nIH, 4, prefix_geo=geo0, prefix_kv=torch.empty(0, d))
    assert idx.shape == valid.shape == soft.shape == (0, 4), (idx.shape, valid.shape, soft.shape)

    n = 5
    H = torch.randn(n, d)
    bid = torch.tensor([0, 0, 0, 1, 1])
    geo = L._open_prefix_geometry(bid, n_blocks=2, scope='outside_window',
                                  window=100, candidate_table=True)
    idx, valid, soft = L.lightning_indexer_open(
        H, torch.empty(0, d), torch.empty(0, dtype=torch.long),
        W_DQ, W_DK, W_w, nIH, 4, prefix_geo=geo,
        prefix_token_kv=torch.randn(n, d), prefix_kind='singletons')
    assert idx.shape == valid.shape == soft.shape == (n, 4)
    assert not valid.any() and not soft.any(), 'no candidate may be read or receive soft mass'

    h, hd, w, K = 2, 3, 2, 4
    q = torch.randn(n, h, hd)
    k_sw, v_sw = torch.randn(n, h, hd), torch.randn(n, h, hd)
    topk = torch.zeros(n, K, dtype=torch.int32)
    sel = torch.zeros(n, K, dtype=torch.bool)
    out, mass = L.gathered_attention_open(
        q, torch.empty(0, h, hd), torch.empty(0, h, hd), topk, sel,
        torch.empty(0, dtype=torch.long), k_sw, v_sw, w, 0.5,
        prefix_geo=geo, prefix_kind='singletons',
        prefix_k=torch.empty(n, h, hd), prefix_v=torch.empty(n, h, hd),
        return_prefix_mass=True)
    assert out.shape == q.shape and torch.isfinite(out).all(), out.shape
    assert mass.shape == (n,) and torch.isfinite(mass).all(), mass


def t_open_analysis_control_gate():
    import open_block_supp as OB
    hp = {'steps': 10, 'tokens_seen': 100, 'n_train_tokens': 100,
          'seq_len': 8, 'batch_size': 1, 'lr': 3e-4, 'weight_decay': 0.1,
          'warmup': 1, 'comp_lambda': 0.0, 'delta_lr_mult': 10.0,
          'd': 8, 'n_layers': 1, 'n_heads': 1, 'd_head': 8, 'mlp_ratio': 4,
          'vocab_size': 32, 'dataset': 'unit', 'data_version': 'v',
          'val_seqs': 2, 'eval_every': 0, 'eval_subset': 2,
          'code': 'c', 'amp_dtype': None, 'eager_batch': False}

    def _rec(mode, **kw):
        r = {'hyperparams': dict(hp), 'code': 'c', 'params': 100,
             'run_fingerprint': 'r', 'data_fp': 'd', 'val_fp': 'v',
             'eval_fp': 'e', 'eval_subset_fp': 's', 'rng_fp': 'g',
             'open_block_config': {'layers': [
                 {'max_block': 128, 'open_block_mode': mode,
                  'open_block_scope': 'full_prefix'}]}}
        r.update(kw)
        return r
    a = _rec('summary')
    b = _rec('closed')
    assert OB._controlled_reason(a, b) is None
    # Only the declared experimental factor may differ.
    bad_hp = dict(hp, lr=1e-3)
    assert 'lr differs' in OB._controlled_reason(
        a, _rec('closed', hyperparams=bad_hp))
    # A one-sided descriptor is unverifiable, not a pass.
    one = _rec('closed')
    one.pop('data_fp')
    assert 'unverifiable' in OB._controlled_reason(a, one)
    one = _rec('closed')
    one['hyperparams'] = {k: v for k, v in hp.items() if k != 'lr'}
    assert 'unverifiable' in OB._controlled_reason(a, one)
    # The paired contrast drops the uncontrolled seed instead of testing it.
    cells = {('a', 0): {'record': a, 'nll': 1.0, 'ppl': 2.0},
             ('b', 0): {'record': b, 'nll': 1.1, 'ppl': 2.1},
             ('a', 1): {'record': a, 'nll': 1.0, 'ppl': 2.0},
             ('b', 1): {'record': _rec('closed', hyperparams=bad_hp),
                        'nll': 1.1, 'ppl': 2.1}}
    ds, pds, used, rejected = OB._paired(cells, 'a', 'b', [0, 1])
    assert used == [0] and len(ds) == 1 and len(pds) == 1, (used, ds, rejected)
    assert len(rejected) == 1 and rejected[0]['seed'] == 1, rejected
    assert 'lr differs' in rejected[0]['reason'], rejected


CHECKS = [
    ('K1 open-block config/segmenter', t_config_and_segmenter),
    ('K2 prefix pooling reference/grad', t_prefix_pool_reference),
    ('K3 prefix invariance', t_prefix_invariance),
    ('K4 future perturbation/gradient', t_future_causality),
    ('K5 candidate masks', t_candidate_masks),
    ('K6 edge cases', t_edge_cases),
    ('K7 closed regression/cut table', t_closed_regression),
    ('K8 singleton semantics', t_singleton_is_length_one_pooled),
    ('K9 HCA budget declaration', t_hca_budget_declaration),
    ('K10 forced-vs-natural cuts', t_forced_vs_natural_cuts),
    ('K11 open candidate edge cases', t_open_candidate_edge_cases),
    ('K12 analysis control gate drops uncontrolled seeds', t_open_analysis_control_gate),
]


def main():
    fails = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f'[open-block] PASS  {name}')
        except Exception as e:
            fails += 1
            print(f'[open-block] FAIL  {name}: {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
    print(f'[open-block] {len(CHECKS) - fails}/{len(CHECKS)} passed')
    return 1 if fails else 0


if __name__ == '__main__':
    raise SystemExit(main())
