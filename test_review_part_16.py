"""CPU review checks (split group 16): ablation control ladders.

A leave-one-out contrast shows that a component is associated with a PPL
change.  It does not show that the *declared mechanism* produced it: the
control may also change magnitude, norm, parameter count or read budget.
These checks cover the matched content placebo (`shift`: the same pooled
content moved back one block) and the design gate that keeps it distinct from
the older zero-content presence control.
"""
import json
import os
import sys as _sys

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


def _tiny(variant, *, vocab=32, d=16, n_layers=2, n_heads=2, d_head=8, seq_len=16):
    import torch
    torch.manual_seed(1234)
    return L.SmallGPT(vocab, d, n_layers, n_heads, d_head, seq_len,
                      L.make_layer_cfgs(n_layers, variant))


def test_shift_content_helper_is_causal_norm_matched_and_gradient_correct():
    import torch
    table = torch.arange(12, dtype=torch.float32).reshape(4, 3).requires_grad_(True)
    shifted = L._shift_block_content(table)
    assert shifted.shape == table.shape
    assert torch.equal(shifted[0], torch.zeros(3))
    assert torch.equal(shifted[1:], table[:-1])
    # Changing a later row must never change an earlier output row.
    changed = table.detach().clone()
    changed[3] += 100.0
    shifted_changed = L._shift_block_content(changed)
    assert torch.equal(shifted_changed[:3], shifted.detach()[:3])
    # After the caller's normalisation a shifted row has exactly the norm of
    # its source row; the first row is the zero-content value.
    normed = torch.nn.functional.normalize(shifted, dim=-1)
    ref = torch.nn.functional.normalize(table.detach(), dim=-1)
    assert torch.allclose(normed[1:].detach(), ref[:-1], atol=1e-6)
    assert torch.allclose(normed[0], torch.zeros(3))
    # Gradient flows from row i+1 to source row i, never to the current row.
    shifted.sum().backward()
    grad = table.grad
    assert grad is not None
    assert torch.equal(grad[0], torch.ones(3))
    assert torch.equal(grad[3], torch.zeros(3))
    # A single-row table has no predecessor and is the zero control.
    one = torch.ones(1, 3, requires_grad=True)
    assert torch.equal(L._shift_block_content(one), torch.zeros(1, 3))
    for bad in (torch.ones(3), [[1.0, 2.0]], None):
        try:
            L._shift_block_content(bad)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f'non-2D table accepted: {bad!r}')


def test_shift_arm_preserves_parameters_but_changes_attention():
    import torch
    vocab, seq = 32, 16
    ids = torch.arange(seq).reshape(1, seq) % vocab
    models = {v: _tiny(v, vocab=vocab, seq_len=seq) for v in
              ('csa_fixed', 'csa_fix_shiftcont', 'csa_fix_zerocont')}
    params = {v: sum(p.numel() for p in m.parameters()) for v, m in models.items()}
    assert len(set(params.values())) == 1, params
    outs = {}
    with torch.no_grad():
        for v, m in models.items():
            outs[v] = m(ids)
    assert not torch.allclose(outs['csa_fixed'], outs['csa_fix_shiftcont']), \
        'the shifted-content arm must not be a no-op'
    assert not torch.allclose(outs['csa_fix_shiftcont'], outs['csa_fix_zerocont']), \
        'presence and alignment controls must be different manipulations'
    # Same seed/input is deterministic.
    again = _tiny('csa_fix_shiftcont', vocab=vocab, seq_len=seq)
    with torch.no_grad():
        assert torch.equal(again(ids), outs['csa_fix_shiftcont'])
    # Unknown content modes fail closed at config construction.
    for bad in ('random', '', None, 'Pooled'):
        try:
            L.AttnCfg(kind='csa', content_mode=bad)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f'bad content_mode accepted: {bad!r}')


def test_shift_mode_is_causal_on_a_prefix():
    import torch
    torch.manual_seed(7)
    m = _tiny('csa_fix_shiftcont', seq_len=16)
    ids = torch.arange(16).reshape(1, 16) % 32
    with torch.no_grad():
        full = m(ids)
        prefix = m(ids[:, :8])
    # The first 8 positions cannot depend on tokens after position 7.  The
    # fixed-block shifted table is causal, so the prefix logits match.
    assert torch.allclose(full[:, :8], prefix, atol=1e-5), \
        (full[:, :8] - prefix).abs().max()
    # Open-block prefix representations have a different candidate geometry;
    # shifting their content is refused instead of silently doing something
    # that is not the declared manipulation.
    cfg = L.make_layer_cfgs(1, 'csa_fix_shiftcont')[0]
    import dataclasses
    bad_cfg = dataclasses.replace(cfg, open_block_mode='prefix_summary')
    try:
        L.SmallGPT(32, 16, 1, 2, 8, 16, [bad_cfg])
        m2 = L.SmallGPT(32, 16, 1, 2, 8, 16, [bad_cfg])
        with torch.no_grad():
            m2(ids[:, :8])
    except RuntimeError as e:
        assert 'closed blocks' in str(e) or 'shift' in str(e)
    else:
        raise AssertionError('shift + open-block prefix was accepted')


def test_batched_shift_helper_matches_row_semantics():
    import torch
    import csa_kernels as CK
    B, Bm, d = 2, 3, 2
    comp = torch.arange(B * Bm * d, dtype=torch.float32).reshape(B * Bm, d)
    counts = [torch.ones(3), torch.ones(2)]
    out = CK._shift_content_batched(comp, counts, B, Bm)
    # Row 0 of each batch has no predecessor -> zero.
    assert torch.equal(out[0], torch.zeros(d))
    assert torch.equal(out[3], torch.zeros(d))
    # Row i -> source row i-1 inside the same batch.
    assert torch.equal(out[1], comp[0]) and torch.equal(out[2], comp[1])
    assert torch.equal(out[4], comp[3])
    # A padded block (batch 1 has only two real blocks) is zero.
    assert torch.equal(out[5], torch.zeros(d))
    one = CK._shift_content_batched(comp[:3], [torch.ones(3)], 1, 3)
    assert torch.equal(one, L._shift_block_content(comp[:3]))
    # Batched (3-D) form used by the CUDA kernel/ref dispatch.
    comp3 = comp.reshape(B, Bm, d)
    out3 = CK._shift_content_batched(comp3, counts, B, Bm)
    assert torch.equal(out3.reshape(B * Bm, d), out)
    for bad_counts in ([torch.ones(3)], [], [torch.ones(3)] * 3):
        try:
            CK._shift_content_batched(comp, bad_counts, B, Bm)
        except ValueError:
            pass
        else:
            raise AssertionError(f'wrong counts accepted: {bad_counts!r}')


def test_content_control_ladder_audit_accepts_shipped_design():
    rep = L.content_control_audit()
    assert rep['status'] == 'ok', rep['violations']
    assert rep['treatment_component'] == 'content_alignment'
    assert rep['presence_component'] == 'content_pooling'
    roles = {k: v['role'] for k, v in rep['levels'].items()}
    assert roles == {'aligned': 'treatment',
                     'misaligned': 'alignment_placebo',
                     'absent': 'presence_control'}, roles
    assert all(v['status'] == 'ok' for base in rep['base_checks'].values()
               for k, v in base.items() if isinstance(v, dict) and 'status' in v)
    assert 'csa_fix_shiftcont' in L.RUN_ABL['variants']
    assert 'csa_dyn_shiftcont' in L.RUN_ABL['variants']
    # Both content controls are declared leave-one-out families.
    assert ('csa_fixed', 'csa_fix_shiftcont') in {
        (a, b) for a, b, *_ in L.ABLATION_DESIGN['content_alignment']}
    assert ('csa_fixed', 'csa_fix_zerocont') in {
        (a, b) for a, b, *_ in L.ABLATION_DESIGN['content_pooling']}
    # Factory-level single-factor and completeness gates see the new arms.
    assert L.single_factor_design_audit(verbose=False)['status'] == 'ok'
    assert L.ablation_completeness_audit(
        L.ABLATION_DESIGN, L.RUN_ABL['variants'])['complete'] is True
    assert L.hypothesis_coverage_audit()['status'] == 'ok'
    # The matched placebo has its own 2x2 against dynamic blocking, so the
    # alignment question is not inferred from the fixed-block cell alone.
    fac = L.FACTORIAL_DESIGN['dynamic_blocking_x_content_alignment']
    assert fac['factors'] == ('dynamic_blocking', 'content_alignment')
    assert fac['cells'][('control', 'control')] == 'csa_fix_shiftcont'
    assert fac['cells'][('active', 'control')] == 'csa_dyn_shiftcont'
    assert fac['cells'][('control', 'active')] == 'csa_fixed'
    assert fac['cells'][('active', 'active')] == 'csa_dynamic'
    rep = L.factorial_design_audit()
    assert rep['status'] == 'ok', rep['violations']
    assert rep['factorials']['dynamic_blocking_x_content_alignment']['complete'] is True
    assert L.interaction_sesoi_audit()['status'] == 'ok'


def test_content_control_ladder_audit_fails_closed_on_drift():
    # Missing presence level: alignment alone cannot separate content from
    # magnitude.
    ladder = json.loads(json.dumps(L.CONTENT_CONTROL_DESIGN))
    ladder['levels'].pop('absent')
    assert L.content_control_audit(ladder=ladder)['status'] == 'fail'
    # An unscheduled arm.
    ladder = json.loads(json.dumps(L.CONTENT_CONTROL_DESIGN))
    ladder['levels']['misaligned']['arms']['fixed'] = 'csa_fix_missing'
    rep = L.content_control_audit(ladder=ladder)
    assert rep['status'] == 'fail' and any(
        'not scheduled' in v for v in rep['violations']), rep
    # A relabelled copy must not pass as an independent control level.
    ladder = json.loads(json.dumps(L.CONTENT_CONTROL_DESIGN))
    ladder['levels']['misaligned']['arms']['fixed'] = 'csa_fixed'
    rep = L.content_control_audit(ladder=ladder)
    assert rep['status'] == 'fail' and any(
        'reused' in v or 'treatment arm' in v for v in rep['violations']), rep
    # A control that also moves a second knob is confounded.
    def _factory_extra(n_layers, variant):
        cfgs = L.make_layer_cfgs(n_layers, variant)
        if variant == 'csa_fix_shiftcont':
            return [__import__('dataclasses').replace(c, index_topk=8) for c in cfgs]
        return cfgs
    rep = L.content_control_audit(factory=_factory_extra)
    assert rep['status'] == 'fail' and any(
        'index_topk' in v for v in rep['violations']), rep
    # A level naming an undeclared content mode is not a placebo.
    ladder = json.loads(json.dumps(L.CONTENT_CONTROL_DESIGN))
    ladder['levels']['misaligned']['content_mode'] = 'random'
    rep = L.content_control_audit(ladder=ladder)
    assert rep['status'] == 'fail' and any(
        'content_mode' in v and 'random' in v for v in rep['violations']), rep
    # Malformed declarations do not crash (including a non-mapping arm table).
    for bad in ([], 'x', {'base_arms': {}, 'levels': {}}):
        rep = L.content_control_audit(ladder=bad)
        assert rep['status'] == 'fail', rep
    ladder = json.loads(json.dumps(L.CONTENT_CONTROL_DESIGN))
    ladder['levels']['misaligned']['arms'] = ['csa_fix_shiftcont']
    rep = L.content_control_audit(ladder=ladder)
    assert rep['status'] == 'fail', rep
    # A second level claiming an already-used role must not be silently
    # dropped by the role->level lookup (the set comparison alone would pass).
    ladder = json.loads(json.dumps(L.CONTENT_CONTROL_DESIGN))
    ladder['levels']['absent_copy'] = json.loads(
        json.dumps(ladder['levels']['absent']))
    rep = L.content_control_audit(ladder=ladder)
    assert rep['status'] == 'fail' and any(
        'exactly once' in v for v in rep['violations']), rep


def test_audit_gate_entry_reports_content_controls():
    A = __import__('audit_gates')
    rep = A.audit_content_controls()
    assert rep['status'] == 'ok', rep
    sf = A.audit_single_factor()
    assert sf['content_controls']['status'] == 'ok', sf['content_controls']
    assert sf['status'] == 'ok', sf

def test_shift_arm_matches_kernel_reference_path():
    import torch
    import csa_kernels as CK
    torch.manual_seed(1234)
    model = _tiny('csa_fix_shiftcont', seq_len=16)
    ids = torch.arange(16).reshape(1, 16) % 32
    with torch.no_grad():
        eager = model(ids)
    with CK.kernel_env('ref'):
        with torch.no_grad():
            reference = model(ids)
    assert torch.allclose(eager, reference, atol=1e-5), \
        (eager - reference).abs().max()
    # The kernel path must not silently fall back to pooled content: its
    # output is also different from the pooled arm.
    pooled = _tiny('csa_fixed', seq_len=16)
    with torch.no_grad():
        pooled_out = pooled(ids)
    assert not torch.allclose(reference, pooled_out), \
        'kernel shifted-content path collapsed to pooled content'


def test_rope_shift_arm_is_not_silently_pooled():
    """Every attention read path must honour content_mode='shift'.

    The RoPE layer had its own pooled/zero K/V construction and only checked
    `zero`, so a shift arm ran exactly as the pooled treatment.  That turns the
    alignment placebo into a duplicate of the reference and makes the contrast
    look null by construction.
    """
    import dataclasses
    import torch
    import explib.rope as V7

    d, n_heads, d_head, T = 16, 2, 8, 16
    base = V7.AttnCfgRope(kind='csa', dynamic=False, block_size=4, overlap=2,
                          index_topk=32, rope=True, rope_dim=8, qk_norm=True)
    x = torch.randn(1, T, d)

    def _build(mode):
        cfg = dataclasses.replace(base, content_mode=mode)
        torch.manual_seed(99)
        return V7.HybridAttentionRoPE(d, n_heads, d_head, cfg)

    with torch.no_grad():
        pooled = _build('pooled')(x)
        shift = _build('shift')(x)
        zero = _build('zero')(x)
    assert not torch.allclose(shift, pooled), \
        'RoPE shifted-content path collapsed to pooled content'
    assert not torch.allclose(shift, zero), \
        'presence and alignment controls must stay distinct in the RoPE path'
