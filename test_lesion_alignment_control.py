"""Within-checkpoint lesion of content *alignment*, not merely presence.

The between-model `content_alignment` contrast trains on shifted pooled
content.  It cannot tell whether the trained checkpoint actually reads the
content of its own block.  This module adds the complementary inference-time
lesion: keep the weights and the pooled table byte-identical, read the same
table one block back for the forward pass, and require the operation to be a
real, weight-preserving manipulation.  It also checks that the lesion fails
closed on an arm that has no aligned content to remove.
"""
import contextlib
import os
import sys as _sys

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)

import torch


def _tiny(variant, *, vocab=48, d=24, n_layers=2, n_heads=2, d_head=12,
          seq_len=24):
    torch.manual_seed(1234)
    return L.SmallGPT(vocab, d, n_layers, n_heads, d_head, seq_len,
                      L.make_layer_cfgs(n_layers, variant))


def _ids(seq_len=24, vocab=48):
    return torch.arange(seq_len).reshape(1, seq_len) % vocab


def test_alignment_lesion_changes_forward_without_touching_weights():
    for variant in ('csa_fixed', 'csa_dynamic'):
        m = _tiny(variant)
        ids = _ids()
        state = L.component_lesion_state(m, 'content_alignment')
        # The content table is computed, not a parameter tensor; the forward
        # manipulation check is the applicable test.
        assert state['status'] == 'not_parameterized', (variant, state)
        before = L.state_dict_sha256(m.state_dict())
        attn = m.blocks[0].attn
        original_cfg = attn.cfg
        sens = L.lesion_sensitivity(m, ids, 'content_alignment')
        assert sens['status'] == 'changed', (variant, sens)
        assert sens['changed'] is True
        assert sens['weights_unchanged'] is True
        assert L.state_dict_sha256(m.state_dict()) == before
        # Every cfg object is restored (not merely an equal copy).
        for blk in m.blocks:
            assert blk.attn.cfg is not None
            assert blk.attn.cfg.content_mode == 'pooled', variant
        assert attn.cfg is not None


def test_alignment_lesion_context_moves_only_content_mode_and_restores():
    m = _tiny('csa_fixed')
    ids = _ids()
    attn = m.blocks[0].attn
    original_cfg = attn.cfg
    with torch.no_grad():
        intact = m(ids)
        with L.component_lesion(m, 'content_alignment') as rec:
            assert rec['component'] == 'content_alignment'
            assert rec['cfg_overrides'] == {'content_mode': 'shift'}
            for blk in m.blocks:
                assert blk.attn.cfg.content_mode == 'shift'
            lesion = m(ids)
        assert m.blocks[0].attn.cfg is original_cfg
        assert m.blocks[0].attn.cfg.content_mode == 'pooled'
        assert torch.allclose(m(ids), intact)
    assert not torch.allclose(intact, lesion), (
        'the alignment lesion must be a real forward manipulation')


def test_alignment_lesion_fails_closed_without_aligned_content():
    # An arm already trained/evaluated with shifted or zero content has no
    # aligned table to lesion; a no-op context must not masquerade as a pass.
    for variant in ('csa_fix_shiftcont', 'csa_fix_zerocont'):
        m = _tiny(variant)
        attn = m.blocks[0].attn
        before = attn.cfg
        try:
            with L.component_lesion(m, 'content_alignment'):
                pass
        except ValueError as exc:
            assert 'no applicable layer' in str(exc), (variant, exc)
        else:
            raise AssertionError(
                f'{variant}: alignment lesion on non-pooled content must fail')
        assert m.blocks[0].attn.cfg is before


def test_alignment_design_gate_and_coverage_are_explicit():
    rep = L.lesion_design_audit(build_models=False)
    assert rep['status'] == 'ok', rep['violations']
    rec = rep['components']['content_alignment']
    assert rec['status'] == 'ok'
    assert set(rec['arms']) == {'csa_fixed', 'csa_dynamic'}
    for arm, arec in rec['arms'].items():
        assert arec['single_factor_family'] is True, (arm, arec)
        assert set(arec['effective_cfg_keys']) == {'content_mode'}, (arm, arec)
    cov = rep['coverage']
    assert cov['kind'] == 'lesion_coverage'
    assert cov['components']['content_alignment']['inference_lesion'] is True
    assert cov['components']['content_alignment']['lesion_components'] == [
        'content_alignment']
    # Coverage is not a completeness claim: known non-togglable components are
    # named with a reason, and an inference-time probe elsewhere is disclosed.
    assert cov['components']['compression_regularizer']['inference_lesion'] is False
    assert 'training-time loss' in cov['components']['compression_regularizer']['note']
    assert cov['components']['learned_selection']['inference_lesion'] is False
    assert 'v10/v11' in cov['components']['learned_selection']['note']
    assert cov['n_with_inference_lesion'] == cov['n_ablation_components'] - len(
        cov['without_inference_lesion'])
    assert 'content_alignment' in cov['with_inference_lesion']
    # A missing lesion limits the claim to a training effect; it is not a
    # verdict that the component is unused at inference.
    aligned = cov['components']['content_alignment']
    assert aligned['claim_channels'] == {
        'training_effect': True, 'inference_use': True}
    assert aligned['claim_limit'] is None
    loss = cov['components']['compression_regularizer']
    assert loss['claim_channels'] == {
        'training_effect': True, 'inference_use': False}
    assert 'training-effect only' in loss['claim_limit']
    assert cov['inference_use_claim_allowed'] == cov['with_inference_lesion']


def test_alignment_panel_audit_accepts_the_new_component():
    records = []
    for seed in range(6):
        records.append({
            'component': 'content_alignment',
            'arm': 'csa_fixed',
            'seed': seed,
            'checkpoint_sha256': f'{seed:064x}',
            'intact_ppl': 10.0,
            'lesion_ppl': 10.5,
            'eval_fp': 'same-eval-rows',
            'sensitivity': {'changed': True, 'weights_unchanged': True},
        })
    rep = L.lesion_panel_audit(records)
    assert rep['status'] == 'ok', rep
    panel = rep['panels']['content_alignment/csa_fixed']
    assert panel['verdict'] == 'necessary', panel
    assert panel['n'] == 6
    assert rep['claim_allowed_panels'] == ['content_alignment/csa_fixed']
