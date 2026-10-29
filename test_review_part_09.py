"""CPU review checks (split group 09; t_provenance_extra_ids_alias_narrow_and_sample_bound .. t_archive_manifest_covers_bytes_actually_written)."""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)

def test_provenance_extra_ids_alias_narrow_and_sample_bound():
    ids = np.arange(100, dtype=np.int64)
    a = L.experiment_provenance(
        {'dataset': 'wikitext', 'n_train_tokens': 100, 'seq_len': 10,
         'data_extra': {'task': 'niah', 'seed': 1}}, train_ids=ids, vocab=100)
    b = L.experiment_provenance(
        {'dataset': 'wikitext', 'n_train_tokens': 100, 'seq_len': 10,
         'data_extra': {'task': 'corrupt', 'seed': 2}}, train_ids=ids, vocab=100)
    assert a['data']['corpus_fp'] != b['data']['corpus_fp'], (a['data'], b['data'])
    # A read-only view over writable storage is still mutable.
    base = np.arange(10, dtype=np.int64)
    view = base[:]
    view.flags.writeable = False
    f1 = L._ids_fp(view)
    base[0] = 99
    assert L._ids_fp(view) != f1, 'memo returned the pre-mutation digest for a mutable base'
    # Narrowing is transport-only and must not wrap a negative id.
    try:
        L._ids_narrow(np.array([-2 ** 31 - 1, 0, 5], dtype=np.int64))
        raise AssertionError('negative out-of-range ids were silently narrowed')
    except ValueError:
        pass
    for _m in (1, 4, 15, 16):
        _s, _sampled = L._audit_sample_flat(np.arange(100), _m)
        assert _sampled and _s.size <= _m, (_m, _s.size)


def test_v7_eval_length_reports_short_panel():
    cfg = V7.make_layer_cfgs_v7(1, 'csa_fixed_rope')[0]
    model = V7.SmallGPTRoPE(64, 32, 1, 2, 16, 16, [cfg])
    ids = (np.arange(40).reshape(2, 20) % 64).astype(np.int64)
    out = V7.eval_length_gen(model, ids, [32], 'cpu', n_seq=2, max_pos=None)
    assert 32 in out and out[32].get('error'), out
    assert out[32].get('truncated') is False, out
    try:
        V7.eval_length_gen(model, ids.reshape(-1), [32], 'cpu', n_seq=2, max_pos=None)
        raise AssertionError('a 1-D eval panel must be refused')
    except ValueError:
        pass


def test_ladder_params_use_the_tier_geometry():
    import explib.m0 as M0
    name = 'S9'
    L.SCALE_LADDER[name] = {'d': 16, 'n_layers': 1, 'n_heads': 1, 'd_head': 16}
    L.RUN_S9 = {'seq_len': 64, 'outdir': 'unused', 'seeds': [0], 'steps': 1,
                'batch_size': 1, 'n_train_tokens': 64, 'lr': 0.1, 'warmup': 0,
                'vocab_size': L.B_VOCAB}
    try:
        got, _per_layer = M0._ladder_params(name)
        cfgs = L.make_layer_cfgs(1, 'csa_dynamic')
        want = L.count_params(L.SmallGPT(L.B_VOCAB, 16, 1, 1, 16, 64, cfgs))
        assert got == want, (got, want)
    finally:
        L.SCALE_LADDER.pop(name, None)
        if hasattr(L, 'RUN_S9'):
            delattr(L, 'RUN_S9')


def test_audit_artifacts_resolve_from_repo_not_cwd():
    panel = tempfile.mkdtemp(prefix='results_zz_audit_', dir=REPO)
    old = os.getcwd()
    try:
        with open(os.path.join(panel, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'csa::seed0': {'variant': 'csa', 'seed': 0, 'ppl': 1.0,
                                      'params': 10, 'stats': {}}}, f)
        os.chdir(tempfile.gettempdir())
        out = A.audit_attribution(outdirs=[os.path.basename(panel)])
        assert os.path.basename(panel) in out['panels'], out
    finally:
        os.chdir(old)
        shutil.rmtree(panel, ignore_errors=True)


def test_row_identity_is_numeric_dtype_and_width_safe():
    import data_guard as DG
    i = np.array([[1, 2]], dtype=np.int64)
    f = np.array([[1.0, 2.0]])
    assert DG._row_keys(i) == DG._row_keys(f), 'equal int/float rows must share a key'
    assert len(DG._row_keys(i)[0]) == 16, 'row digests must not be only 64-bit'
    # A large int that float64 cannot represent must not collide with its
    # rounded float image.
    assert DG._row_keys(np.array([[2 ** 53 + 1]], dtype=np.int64)) != \
        DG._row_keys(np.array([[float(2 ** 53)]]))
    assert A._row_overlap_count(np.array([[1., 2.], [3., 4.]]),
                                np.array([[1, 2]])) == 1
    assert DG._matrix_fingerprint(np.array([1., 2.])) == \
        DG._matrix_fingerprint(np.array([[1.], [2.]]))
    # A complex table cannot use the canonical fingerprint, but a null
    # content_fp must not become an accidental identity match.
    fp = DG.split_content_fingerprints(
        np.array([[1 + 2j], [3 + 4j]]), None,
        {'train': np.array([0, 1]), 'val': np.array([], dtype=np.int64),
         'test': np.array([], dtype=np.int64)})
    assert all(v.get('content_fp') for v in fp.values()), fp
    assert fp['train']['content_fp'] != fp['val']['content_fp'], fp


def test_evidence_and_dynamic_audits_reject_malformed_thresholds():
    import copy
    with open(os.path.join(REPO, 'confirmation_frozen.json'), encoding='utf-8') as f:
        main = json.load(f)
    bad = copy.deepcopy(main)
    bad['scope']['confirmatory_claims'] = ['not', 'text']
    ev = L.evidence_level_audit(bad, manifest_path='confirmation_frozen.json')
    assert ev['status'] == 'fail', ev
    with open(os.path.join(REPO, 'confirmation_frozen_dynamic.json'), encoding='utf-8') as f:
        dyn = json.load(f)
    for key, val in (('alpha', float('nan')),
                     ('min_power_for_test', float('inf')),
                     ('min_power_for_test', float('nan'))):
        m = copy.deepcopy(dyn)
        m['statistics'][key] = val
        r = L.dynamic_design_audit(m, manifest_path='confirmation_frozen_dynamic.json')
        assert r['status'] == 'fail', (key, val, r)


def test_dynamic_design_requires_explicit_mapping_and_orientation():
    import copy
    with open(os.path.join(REPO, 'confirmation_frozen_dynamic.json'), encoding='utf-8') as f:
        dyn = json.load(f)
    # A new comparison must state its evidence level; inheriting the manifest
    # level silently is exactly how an exploratory contrast becomes "frozen".
    bad = copy.deepcopy(dyn)
    bad['primary_comparisons'][0].pop('evidence_level')
    r = L.dynamic_design_audit(bad)
    assert r['status'] == 'fail', r
    assert any('no explicit evidence_level' in v for v in r['violations']), r['violations']
    # Endpoint drift is a post-hoc outcome change, not a confirmation.
    bad = copy.deepcopy(dyn)
    bad['primary_comparisons'][0]['endpoint'] = 'by_len.ppl@2048'
    r = L.dynamic_design_audit(bad)
    assert r['status'] == 'fail' and any(
        'does not equal the hypothesis endpoint' in v for v in r['violations']), r['violations']
    # A reversed contrast must carry the reversed expected sign.
    bad = copy.deepcopy(dyn)
    bad['primary_comparisons'][0]['expected_sign'] = 1
    r = L.dynamic_design_audit(bad)
    assert r['status'] == 'fail' and any(
        'expected_sign=1 is inconsistent' in v for v in r['violations']), r['violations']
    # A declared hypothesis without any scheduled comparison is not a design.
    bad = copy.deepcopy(dyn)
    bad['primary_comparisons'] = [
        c for c in bad['primary_comparisons']
        if c.get('hypothesis_id') != 'H2_retrieval_precision']
    r = L.dynamic_design_audit(bad)
    assert r['status'] == 'fail' and any(
        'H2_retrieval_precision has no primary comparison' in v
        for v in r['violations']), r['violations']
    # An extra hypothesis object is validated, not silently ignored.
    bad = copy.deepcopy(dyn)
    bad['hypotheses']['HX_extra'] = 'not-an-object'
    r = L.dynamic_design_audit(bad)
    assert r['status'] == 'fail' and any(
        'malformed extra claim' in v for v in r['violations']), r['violations']
    # System-level H3/H5 cannot carry the mechanism flag even if all budgets
    # happened to match.
    bad = copy.deepcopy(dyn)
    h3 = next(c for c in bad['primary_comparisons']
              if c.get('hypothesis_id') == 'H3_long_range_coverage')
    h3['mechanism_claim_allowed'] = True
    r = L.dynamic_design_audit(bad)
    assert r['status'] == 'fail' and any(
        'not declared mechanism_claim_eligible' in v for v in r['violations']), r['violations']
    # A mechanism-eligible hypothesis must actually be tested by a
    # single-factor pair.  Replacing H1's random-vs-random segmentation pair
    # with a kind swap (HCA vs CSA) makes it a multi-factor contrast even
    # though the hypothesis is still labeled mechanism-eligible.
    bad = copy.deepcopy(dyn)
    h1 = next(c for c in bad['primary_comparisons']
              if c.get('hypothesis_id') == 'H1_segmentation_quality'
              and c.get('a') == 'csa_dynamic_rope_randidx_seeded')
    h1['a'], h1['b'] = 'hybrid_dynamic_rope', 'csa_dynamic_rope'
    r = L.dynamic_design_audit(bad)
    assert r['status'] == 'fail', r
    assert any('single-factor family' in v for v in r['violations']), r['violations']


def test_dynamic_power_and_read_calibration_are_machine_checked():
    import copy
    with open(os.path.join(REPO, 'confirmation_frozen_dynamic.json'), encoding='utf-8') as f:
        dyn = json.load(f)
    # The exact p floor and Holm floor are arithmetic, not prose.
    for key, val in (('p_floor', 1e-3),
                     ('min_attainable_p_holm', 0.01),
                     ('n_seeds', 15),
                     ('min_power_for_test', 0.99),
                     ('power_q_for_test', 0.90)):
        bad = copy.deepcopy(dyn)
        bad['statistics'][key] = val
        r = L.dynamic_design_audit(bad)
        assert r['status'] == 'fail', (key, val, r['violations'])
    # A power-gate "passed" label without a numeric power is not a gate.
    bad = copy.deepcopy(dyn)
    bad['precondition']['power_gate'] = {'status': 'passed'}
    r = L.dynamic_design_audit(bad)
    assert r['status'] == 'fail' and any(
        'power is missing' in v for v in r['violations']), r['violations']
    # Measured calibration must contain the realized mean and the concrete arm
    # derived from it; a target_block_tokens guess is not a measurement.
    bad = copy.deepcopy(dyn)
    cal = bad['precondition']['read_budget_calibration']
    cal['status'] = 'measured'
    r = L.dynamic_design_audit(bad)
    assert r['status'] == 'fail' and any(
        'positive mean_dynamic_block_tokens' in v for v in r['violations']), r['violations']
    # Formula inconsistency is refused.
    bad = copy.deepcopy(dyn)
    cal = bad['precondition']['read_budget_calibration']
    cal.update(status='measured', mean_dynamic_block_tokens=4.5,
               matched_fixed_topk=32, arm='csa_fixed_rope_topk32')
    r = L.dynamic_design_audit(bad)
    assert r['status'] == 'fail' and any(
        'does not equal the calibration formula result 36' in v
        for v in r['violations']), r['violations']
    # A formula-consistent k with no trained/declared arm is still a guess.
    bad = copy.deepcopy(dyn)
    cal = bad['precondition']['read_budget_calibration']
    cal.update(status='measured', mean_dynamic_block_tokens=4.5,
               matched_fixed_topk=36, arm='csa_fixed_rope_topk36')
    r = L.dynamic_design_audit(bad)
    assert r['status'] == 'fail' and any(
        'is not in manifest.arms' in v for v in r['violations']), r['violations']
    # With the measured arm present the draft is calibration-consistent again
    # (it is still a non-runnable draft, not a confirmation).
    good = copy.deepcopy(bad)
    good['arms'].append('csa_fixed_rope_topk36')
    good['binding']['per_arm_recipe_signature']['csa_fixed_rope_topk36'] = {'mr': 4.0}
    r = L.dynamic_design_audit(good)
    assert r['status'] == 'ok', r['violations']
    assert r['calibration']['status'] == 'ok' and r['calibration']['expected_topk'] == 36, r['calibration']
    assert r['promotion_ready'] is False, r


def test_budget_ledger_mechanism_claim_needs_confirmation_and_calibration():
    import copy
    with open(os.path.join(REPO, 'confirmation_frozen_dynamic.json'), encoding='utf-8') as f:
        dyn = json.load(f)
    # A draft cannot turn a matched budget into a mechanism claim.
    bad = copy.deepcopy(dyn)
    bad['primary_comparisons'][0]['mechanism_claim_allowed'] = True
    r = L.comparison_budget_ledger(bad)
    assert r['status'] == 'fail' and any(
        'evidence_level=' in v and 'mechanism claim' in v for v in r['violations']), r['violations']
    # Promote the same design: the claim is now allowed only because the
    # manifest is confirmatory AND the dynamic-vs-fixed calibration is measured.
    good = copy.deepcopy(dyn)
    good['status'] = 'frozen'
    good['frozen_at'] = '2026-10-23'
    good['evidence_level'] = 'confirmatory'
    good['precondition']['runnable'] = True
    good['precondition']['read_budget_calibration'].update(
        status='measured', mean_dynamic_block_tokens=4.0,
        matched_fixed_topk=32, arm='csa_fixed_rope_topk32')
    good['precondition']['power_gate'] = {
        'status': 'passed', 'power': 0.957, 'q': 0.95, 'min_power': 0.8}
    good['primary_comparisons'][0]['evidence_level'] = 'confirmatory'
    good['primary_comparisons'][0]['mechanism_claim_allowed'] = True
    r = L.comparison_budget_ledger(good)
    assert r['status'] == 'ok', r['violations']
    entry = r['comparisons'][good['primary_comparisons'][0]['name']]
    assert entry['mechanism_claim_allowed'] is True, entry
    assert entry['read_budget_calibration_ok'] is True, entry
    assert entry['compute_proxy_matched'] is True, entry
    # A pending calibration breaks the mechanism claim even with the same
    # confirmatory label: a status must be measured, not merely present.
    pend = copy.deepcopy(good)
    pend['precondition']['read_budget_calibration']['status'] = 'pending'
    rp = L.comparison_budget_ledger(pend)
    assert rp['status'] == 'fail' and any(
        'MEASURED read-budget calibration' in v for v in rp['violations']), rp['violations']
    # Dropping the confirmatory level drops the claim.
    exp = copy.deepcopy(good)
    exp['evidence_level'] = 'exploratory'
    re = L.comparison_budget_ledger(exp)
    assert re['status'] == 'fail' and any(
        'evidence_level=' in v and 'mechanism claim' in v for v in re['violations']), re['violations']
    # A different served cache is a data treatment, not a nuisance.
    with open(os.path.join(REPO, 'confirmation_frozen.json'), encoding='utf-8') as f:
        main = json.load(f)
    tok = copy.deepcopy(main)
    tok['binding']['per_arm_recipe_signature']['full_rope']['nt'] = 4000000
    rt = L.comparison_budget_ledger(tok)
    assert rt['status'] == 'fail' and any(
        'served train-cache budget' in p for p in rt['problems']), rt['problems']
    # Boolean counts must not silently become 1.
    bool_step = copy.deepcopy(main)
    bool_step['binding']['training_recipe_signature']['steps'] = True
    rb = L.comparison_budget_ledger(bool_step)
    assert rb['status'] == 'fail' and any(
        'positive steps/bs/sl' in p for p in rb['problems']), rb['problems']


def test_evidence_audit_checks_power_floor_and_family_identity():
    import copy
    with open(os.path.join(REPO, 'confirmation_frozen.json'), encoding='utf-8') as f:
        main = json.load(f)
    for key, val in (('p_floor', 1e-4),
                     ('min_attainable_p_holm', 0.01),
                     ('n_seeds', 11),
                     ('alpha', float('nan'))):
        bad = copy.deepcopy(main)
        bad['statistics'][key] = val
        r = L.evidence_level_audit(bad, manifest_path=os.path.join(REPO, 'confirmation_frozen.json'))
        assert r['status'] == 'fail', (key, val, r['violations'])
    # A stale per-q power table is a design record defect, not a rounding nit.
    stale = copy.deepcopy(main)
    stale['statistics']['power_first_holm_gate']['q=0.95'] = 0.50
    r = L.evidence_level_audit(stale, manifest_path=os.path.join(REPO, 'confirmation_frozen.json'))
    assert r['status'] == 'fail' and any(
        'power_first_holm_gate' in v for v in r['violations']), r['violations']
    # Duplicate comparison names would silently merge two family slots.
    dup = copy.deepcopy(main)
    dup['primary_comparisons'].append(copy.deepcopy(dup['primary_comparisons'][0]))
    r = L.evidence_level_audit(dup, manifest_path=os.path.join(REPO, 'confirmation_frozen.json'))
    assert r['status'] == 'fail' and any(
        'duplicate primary comparison name' in v for v in r['violations']), r['violations']
    # Duplicate frozen seeds are not extra replication.
    dseed = copy.deepcopy(main)
    dseed['seeds'][1] = dseed['seeds'][0]
    r = L.evidence_level_audit(dseed, manifest_path=os.path.join(REPO, 'confirmation_frozen.json'))
    assert r['status'] == 'fail' and any(
        'seeds contain duplicates' in v for v in r['violations']), r['violations']
    # A malformed direction word is not a direction.
    badsig = copy.deepcopy(main)
    badsig['primary_comparisons'][0]['expected_sign'] = 'maybe'
    r = L.evidence_level_audit(badsig, manifest_path=os.path.join(REPO, 'confirmation_frozen.json'))
    assert r['status'] == 'fail' and any(
        'expected_sign' in v for v in r['violations']), r['violations']


def test_dynamic_audit_names_unhashable_arm_values():
    import copy
    with open(os.path.join(REPO, 'confirmation_frozen_dynamic.json'), encoding='utf-8') as f:
        dyn = json.load(f)
    bad = copy.deepcopy(dyn)
    bad['primary_comparisons'][0]['a'] = ['csa_dynamic_rope']
    bad['primary_comparisons'][0]['b'] = {'name': 'csa_fixed_rope'}
    # A malformed arm identity must be named and the rest of the family must
    # still be audited; it used to raise `TypeError: unhashable type` from the
    # seen-pairs set.
    r = L.dynamic_design_audit(bad)
    assert r['status'] == 'fail', r
    assert any('is not a non-empty arm name' in v for v in r['violations']), r['violations']
    # The well-formed sibling comparison is still represented.
    assert len(r['comparisons']) == len(bad['primary_comparisons']), r['comparisons']


def test_integer_fingerprint_cast_stays_in_range():
    import warnings
    import data_guard as DG
    # The float64 image of int64_max is 2**63 (outside int64), and the same is
    # true for the largest datetime64[ns].  An unguarded cast warns/overflows
    # even though the exact tail makes the digest correct.
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        for arr in (np.array([[2 ** 63 - 1]], dtype=np.int64),
                    np.array([[np.datetime64('2262-04-11T23:47:16.854775807')]])):
            fp = DG._matrix_fingerprint(arr)
            assert isinstance(fp, str) and len(fp) == 64, fp
        # A native int64 and its exact object twin share one digest.
        assert (DG._matrix_fingerprint(np.array([[2 ** 63 - 1]], dtype=np.int64))
                == DG._matrix_fingerprint(np.array([[2 ** 63 - 1]], dtype=object)))


def test_index_float_boundary_casts_are_guarded():
    import warnings
    import data_guard as DG
    # The signed int64 boundaries, not just the positive one: a far-negative
    # float used to reach the cast (warning/implementation-defined overflow)
    # before the negativity check could reject it.
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        for value in (-1e20, 1e20, 2.0 ** 63, -2.0 ** 63):
            try:
                DG._validated_index(np.array([value]))
                raise AssertionError(f'out-of-range/negative float index {value!r} was accepted')
            except ValueError:
                pass


def test_segment_sum_accumulates_precisely():
    # Same failure mode the fused scatter helper fixed: a large global prefix
    # rounds a later small segment to zero in fp32.  The output dtype must stay
    # fp32, but the accumulation must not be an fp32 global cumsum.
    idx = torch.cat([torch.zeros(64, dtype=torch.long),
                     torch.ones(1000, dtype=torch.long)])
    vals = torch.cat([torch.full((64,), 1e12), torch.ones(1000)])
    out = L._det_segment_sum(2, idx, vals, dtype=torch.float32)
    assert out.dtype == torch.float32, out.dtype
    assert abs(float(out[1]) - 1000.0) < 1.0, out
    # Integer destinations keep their existing per-element cast semantics.
    iout = L._det_segment_sum(2, torch.tensor([1, 0, 1, 0]),
                              torch.tensor([2.5, 3.5, 4.5, 5.5]),
                              dtype=torch.long)
    assert iout.tolist() == [8, 6], iout.tolist()


def test_confirmation_admission_rejects_nonobject_payload():
    # A JSON list/string at the bylen path is not a record mapping; it used to
    # raise AttributeError (`.items()`), bypassing the documented fail-closed
    # refusal path.
    for bad in ([], 'records', 3):
        try:
            V12._admit_confirmation_records(bad, {})
            raise AssertionError(f'non-object bylen payload {bad!r} was admitted')
        except SystemExit:
            pass


def test_pool_geometry_widens_instead_of_wrapping():
    import csa_kernels as CK
    n = 2 ** 31 + 9
    # Host geometry: int32 on the historical path, int64 once a value/its
    # pairwise sum leaves the int32 range (the same exact integers).
    p32 = L._pool_parts_flat([5], 1, 3, 1000)
    assert p32 is not None and p32[0].dtype == np.int32, p32
    p = L._pool_parts_flat([5], 1, 3, n)
    assert p is not None and p[0].dtype == np.int64, p
    assert p[0][:8].tolist() == [5, 5, 0, 4, -1, 0, 0, 0], p[0].tolist()
    v = L._pool_geom_flat_var([5], 1, 3, n, 5, 2, 8, 8)
    assert v is not None and v[0].dtype == np.int64, v
    assert v[0].tolist() == [0, 1, 2, 3, 4, 0, 1, 0, 1, 0, 1, 2, 3, 4], v[0].tolist()
    s = L._pool_geom_flat_single([5], 1, n, 5, 8)
    assert s is not None and s[0].dtype == np.int64, s
    assert s[0].tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4], s[0].tolist()
    # PoolGeometry also stores the widened tables (not wrapped negatives).
    g = CK.PoolGeometry([[2, 3]], 2, n, torch.device('cpu'), None)
    assert g.starts.dtype == torch.int64, g.starts.dtype
    assert g.last_tok.tolist() == [[1, 4]], g.last_tok.tolist()
    ga, av = g.gather_geom()
    assert ga.tolist() == [[[0, 1, 2], [2, 3, 4]]], ga.tolist()
    assert av[..., 0].tolist() == [[[True, True, False], [True, True, True]]], av.tolist()
    g2 = CK.PoolGeometry([[2, 3]], 2, 10, torch.device('cpu'), None)
    assert g2.starts.dtype == torch.int32, g2.starts.dtype


def test_device_segment_geometry_matches_host_at_scale():
    # `_device_segment_geometry` preallocates the successor orbit from a
    # block-count bound.  The bound must use the SMALLEST possible successor
    # stride: f(p) = min(p+max_block, max(p+min_block, cut+1)), so when
    # max_block < min_block the first branch wins and the effective stride is
    # max_block.  Bounding by min_block alone under-allocates `starts` on a
    # long sequence and silently truncates the block table (the remaining
    # tokens collapse into one oversized final block), even though the host
    # `_segment` it must replicate has no such assumption.
    n = 10003
    cuts = np.zeros((2, n - 1), dtype=bool)
    cuts[0, ::7] = True
    cuts[1, ::113] = True
    for mb, xb in ((64, 16), (128, 1), (2, 1), (16, 16), (2, 128), (2, None)):
        bid, counts, hon = L._device_segment_geometry(
            n, torch.from_numpy(cuts), mb, xb, 'cpu')
        for b in range(cuts.shape[0]):
            _hb, _hh, hc = L._segment(n, cuts[b].tolist(), mb, xb)
            exp_bid = np.repeat(np.arange(len(hc), dtype=np.int64), hc)
            got_c = counts[b].tolist()
            while got_c and got_c[-1] == 0:  # drop the power-of-two padding
                got_c.pop()
            assert got_c == [int(c) for c in hc], (mb, xb, b, got_c[:8], list(hc)[:8])
            assert np.array_equal(bid[b].numpy(), exp_bid), (mb, xb, b)
            # A truncated table cannot reach the last host block id.
            assert int(bid[b].max()) == len(hc) - 1, (mb, xb, b, int(bid[b].max()), len(hc))
            exp_h = np.zeros(max(n - 1, 0), dtype=np.int64)
            for _slot in _hh:
                exp_h[int(_slot)] = 1
            got_h = np.zeros(max(n - 1, 0), dtype=np.int64)
            _hs = hon[b].numpy()[:n - 1]
            _ok = _hs >= 0
            got_h[_hs[_ok]] = 1
            assert np.array_equal(got_h, exp_h), (mb, xb, b)


def test_aggregate_wrapper_preserves_pairs_and_splits_code():
    # `aggregate_with_exclusions` is what every panel dump uses.  Its old
    # per-variant grouping removed the baseline before `aggregate` could pair,
    # so every shipped aggregate silently lost dPPL_vs_full / n_paired.
    base = {'run_cfg': 'cfg1', 'steps': 10, 'tokens_seen': 1000,
            'params': 100, '_code': 'v1'}
    summary = {}
    for variant, ppl in (('csa_dynamic', 10.0), ('full', 13.0)):
        for seed in (0, 1):
            summary[f'{variant}::s{seed}'] = dict(
                base, variant=variant, seed=seed, ppl=ppl + seed,
                nll_sum=(ppl + seed) * 1000, n_tokens=1000)
    agg, rep, admitted = L.aggregate_with_exclusions(summary)
    assert len(admitted) == 4, admitted
    arm = agg['csa_dynamic']
    assert arm.get('dPPL_vs_full_mean') == -3.0, arm
    assert arm.get('n_paired') == 2, arm
    # A non-nested record must be labelled full_validation; a set {False} is
    # truthy and used to mislabel every ordinary panel as nested-holdout.
    assert arm.get('ppl_primary_estimand') == 'full_validation', arm
    assert agg['full'].get('ppl_primary_estimand') == 'full_validation'
    # Two code stamps are two measurements, not two seeds: split, never pool.
    mixed = {k: dict(v, _code=('v1' if v['seed'] == 0 else 'v2'))
             for k, v in summary.items()}
    agg2, rep2, _ = L.aggregate_with_exclusions(mixed)
    assert len(agg2) == 4 and any('@fp' in k for k in agg2), list(agg2)
    assert all(v['n_seeds'] == 1 for v in agg2.values()), agg2
    assert all(v.get('dPPL_vs_full_mean') is not None for k, v in agg2.items()
               if k.startswith('csa_dynamic@')), agg2
    assert len(rep2['fingerprint_split_groups']) == 2, rep2
    # A nested-holdout record keeps its declared estimand label.
    nested = dict(summary['csa_dynamic::s0'], seed=0,
                  ppl_estimation_rows='nested_validation_holdout',
                  ppl_holdout=9.0)
    a1, _r1, _ad1 = L.aggregate_with_exclusions(
        {f'csa_dynamic::s0': nested,
         'full::s0': dict(summary['full::s0'])})
    assert a1['csa_dynamic']['ppl_primary_estimand'] == 'nested_validation_holdout', a1
    assert a1['csa_dynamic']['ppl_primary'] == 9.0, a1


def test_code_semantics_enters_arm_and_control_identity():
    def rec(v, seed, code, ppl):
        return {'variant': v, 'seed': seed, '_code': code, 'run_cfg': 'cfg',
                'steps': 10, 'tokens_seen': 1000, 'params': 100, 'ppl': ppl,
                'nll_sum': ppl * 1000, 'n_tokens': 1000}
    recs = {('a', '', 0): rec('a', 0, 'v1', 10.0),
            ('a', '', 1): rec('a', 1, 'v1', 11.0),
            ('b', '', 0): rec('b', 0, 'v2', 12.0),
            ('b', '', 1): rec('b', 1, 'v2', 13.0)}
    cv = L.controlled_variables_report(recs, ['a', 'b'])
    assert cv['code_equal'] is False and cv['controlled'] is False, cv
    assert '_code' in cv['differences'], cv
    recs['a', '', 1] = dict(recs['a', '', 1], _code='v2')
    cv2 = L.controlled_variables_report(recs, ['a', 'b'])
    assert 'a' in cv2['within_arm_mixed'], cv2
    assert cv2['controlled'] is False, cv2
    # Arm summaries split by code stamp and use the correct estimand label.
    summary = {f'a::s{s}': rec('a', s, ('v1' if s == 0 else 'v2'), 10.0 + s)
               for s in (0, 1)}
    rep = L.methodology_audit(summary, design={}, baselines={}, write=False,
                              verbose=False)
    keys = [k for k in rep['arm_summary'] if k.startswith('a')]
    assert len(keys) == 2, rep['arm_summary']
    assert all(rep['arm_summary'][k]['n_seeds'] == 1 for k in keys)
    assert all(rep['arm_summary'][k]['ppl_estimand'] == 'full_validation'
               for k in keys), rep['arm_summary']


def test_confirmation_parameter_binding():
    m = _manifest_for_admission(seeds=(0, 1))
    m['compute_budget_disclosure'] = {'parameter_matching': 'matched by mr'}
    m['statistics']['descriptive_only'] = False
    good = {}
    for v in ('a_arm', 'b_arm'):
        for s in (0, 1):
            good[f'{v}::s{s}'] = dict(_rec(v, s, 'FP'), params=100)
    adm, _fp = V12._admit_confirmation_records(good, m)
    assert len(adm) == 4, adm
    bad = {k: dict(r) for k, r in good.items()}
    del bad['a_arm::s1']['params']
    adm2, _fp2 = V12._admit_confirmation_records(bad, m)
    assert ('a_arm', 1) not in adm2 and len(adm2) == 3, adm2
    # Descriptive-only smoke plans keep the legacy permissive path.
    m_desc = dict(m, statistics=dict(m['statistics'], descriptive_only=True))
    adm3, _fp3 = V12._admit_confirmation_records(bad, m_desc)
    assert len(adm3) == 4, adm3
    # Per-seed parameter identity: partial/inconsistent evidence is not a pass.
    ra = {0: {'params': 100}, 1: {'params': 100}}
    rb = {0: {'params': 100}, 1: {'params': 100}}
    d = V12._param_match_detail(ra, rb, [0, 1], 'a', 'b')
    assert d['param_match_complete'] is True and d['param_matched'] is True, d
    # A small total difference is still within the 1% tolerance.
    rb_ok = {0: {'params': 100}, 1: {'params': 100}}
    d_ok = V12._param_match_detail(ra, rb_ok, [0, 1], 'a', 'b')
    assert d_ok['param_matched'] is True, d_ok
    d2 = V12._param_match_detail(ra, {0: {'params': 100}}, [0, 1], 'a', 'b')
    assert d2['param_match_complete'] is False and d2['param_matched'] is None, d2
    d3 = V12._param_match_detail({0: {'params': 100}, 1: {'params': 200}},
                                 rb, [0, 1], 'a', 'b')
    assert d3['param_match_complete'] is False and d3['param_matched'] is None, d3
    assert d3['param_match_reason'], d3
    # The manifest disclosure is what makes params mandatory; descriptive-only
    # smoke plans keep the legacy path.
    assert V12._manifest_requires_params(m) is True
    assert V12._manifest_requires_params(m_desc) is False
    assert V12._manifest_requires_params({'statistics': {}}) is False
    # Two internally consistent arms beyond 1% are a real mismatch, not None.
    d4 = V12._param_match_detail(ra, {0: {'params': 200}, 1: {'params': 200}},
                                 [0, 1], 'a', 'b')
    assert d4['param_match_complete'] is True and d4['param_matched'] is False, d4


def test_paired_fragility_and_seed_drop_jackknife():
    # 12 equal same-sign deltas: the exact test is at its floor; the verdict
    # survives dropping six least-influential seeds and needs three flips
    # before p >= alpha.
    d12 = [0.5] * 12
    r = L.paired_fragility(d12, seeds=list(range(12)))
    assert r['n'] == 12 and r['n_nonzero'] == 12, r
    assert abs(r['p'] - L.signflip_p_floor(12)) < 1e-15, r
    assert r['same_sign_count'] == 12 and r['same_sign_fraction'] == 1.0, r
    assert r['drop_fragility'] == 6, r['drop_fragility']
    assert r['flip_fragility'] == 3 and r['flip_stop_reason'] == 'p>=alpha', r
    assert r['ci_excludes_zero'] is True, r
    assert r['most_influential_tied_count'] == 12, r
    assert len(r['loo']) == 12 and r['loo_p_min'] == r['loo_p_max'], r
    assert all(not v['removes_significance'] for v in r['loo']), r['loo']
    # A borderline n=6 panel: one seed removal already overturns the raw test.
    r6 = L.paired_fragility([0.5] * 6, seeds=list(range(6)))
    assert r6['drop_fragility'] == 0, r6['drop_fragility']
    assert all(v['removes_significance'] for v in r6['loo']), r6['loo']
    assert r6['flip_fragility'] == 1, r6
    # A contrary outlier is the most influential seed and kills the panel.
    r_out = L.paired_fragility([0.5] * 11 + [-10.0], seeds=list(range(12)))
    assert r_out['p'] >= 0.05 and r_out['flip_fragility'] == 0, r_out
    # Influence is the ABSOLUTE p change: removing the contrary seed makes the
    # panel significant, which the signed p-increase alone would hide.
    assert r_out['most_influential_seed'] == 11, r_out['most_influential_seed']
    assert r_out['most_influential_p_shift'] > 0.9, r_out['most_influential_p_shift']
    assert r_out['most_influential_tied_count'] == 1, r_out
    # Dropped non-finite values keep their seed label out of the LOO pass.
    r_nan = L.paired_fragility([0.5, float('nan'), 0.5, 0.5, 0.5],
                               seeds=[0, 1, 2, 3, 4])
    assert r_nan['n'] == 4 and r_nan['n_dropped'] == 1, r_nan
    assert [v['seed'] for v in r_nan['loo']] == [0, 2, 3, 4], r_nan['loo']
    # Degenerate inputs are reported, not crashed.
    r1 = L.paired_fragility([0.5], seeds=[7])
    assert r1['n'] == 1 and r1['flip_fragility'] == 0, r1
    r0 = L.paired_fragility([], seeds=[])
    assert r0['n'] == 0 and r0['drop_fragility'] is None, r0
    # Seed identity contract: length/duplicate/type errors are explicit.
    for _bad in (lambda: L.paired_fragility([0.1, 0.2], seeds=[0]),
                 lambda: L.paired_fragility([0.1, 0.2], seeds=[0, 0]),
                 lambda: L.paired_fragility([0.1, 0.2], seeds=[0, 1.5]),
                 lambda: L.paired_fragility([0.1, 0.2], seeds=[0, True]),
                 lambda: L.paired_fragility('ab'),
                 lambda: L.paired_fragility([0.1, 0.2], seeds='01'),
                 lambda: L.paired_fragility(3)):
        try:
            _bad()
        except ValueError:
            pass
        else:
            raise AssertionError('malformed fragility input must raise ValueError')
    for _bad_alpha in (True, 0, 1.0, 1.5, 'x'):
        try:
            L.paired_fragility([0.1, 0.2], alpha=_bad_alpha)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid fragility alpha must raise ValueError')
    # Family jackknife: dropping a seed from the n=6 comparison loses it,
    # while the n=12 comparison keeps its Holm rejection in every fold.
    fam = L.holm_decision_jackknife({
        'fragile': {'deltas': [0.5] * 6, 'seeds': list(range(6))},
        'stable': {'deltas': [0.5] * 12, 'seeds': list(range(12))},
    }, alpha=0.05)
    assert fam['rejected_full'] == ['fragile', 'stable'], fam['rejected_full']
    assert fam['unstable_folds'] == list(range(6)), fam['unstable_folds']
    assert fam['always_rejected'] == ['stable'], fam['always_rejected']
    assert fam['sometimes_rejected'] == ['fragile'], fam['sometimes_rejected']
    assert fam['all_rejections_stable'] is False, fam
    _lost = next(f for f in fam['folds'] if f['seed'] == 0)
    assert _lost['lost'] == ['fragile'], _lost
    assert _lost['n_pairs']['fragile'] == 5, _lost['n_pairs']
    # A seed absent from a comparison leaves that comparison's full p intact.
    _absent = next(f for f in fam['folds'] if f['seed'] == 6)
    assert _absent['n_pairs']['fragile'] == 6, _absent['n_pairs']
    assert _absent['raw_p']['fragile'] == L.exact_signflip_p([0.5] * 6)['p'], _absent
    # A declared-but-omitted comparison keeps its p=1 family slot: with m=3 the
    # n=6 comparison (raw p=0.03125) is no longer Holm-rejected, which is the
    # correct correction and the opposite of silently shrinking m to 2.
    fam_decl = L.holm_decision_jackknife({
        'fragile': {'deltas': [0.5] * 6, 'seeds': list(range(6))},
        'stable': {'deltas': [0.5] * 12, 'seeds': list(range(12))},
    }, alpha=0.05, family_names=['omitted'])
    assert fam_decl['n_comparisons'] == 3, fam_decl
    assert fam_decl['rejected_full'] == ['stable'], fam_decl['rejected_full']
    assert 'omitted' in fam_decl['never_rejected'], fam_decl['never_rejected']
    assert fam_decl['per_comparison']['omitted']['p_holm_min'] == 1.0, fam_decl
    for _bad_family in ('ab', b'ab', 3):
        try:
            L.holm_decision_jackknife({'x': [0.5, 0.5]}, family_names=_bad_family)
        except ValueError:
            pass
        else:
            raise AssertionError('a non-iterable/string family_names must raise ValueError')
    # The composed helper keeps the declared family size and never edits p.
    rob = L.confirmation_robustness(
        {'fragile': {'deltas': [0.5] * 6, 'seeds': list(range(6))}},
        alpha=0.05, family_size=6, confirmed=['fragile'])
    assert rob['family_size'] == 6, rob
    # n=6 cannot clear a Holm/6 gate even with 6/6 same-sign pairs, so the
    # planning margin is explicitly None (not a fabricated 6).
    assert rob['per_comparison']['fragile']['min_same_sign_for_family_gate'] is None, rob
    assert rob['per_comparison']['fragile']['same_sign_margin'] is None, rob
    assert rob['confirmed_full'] == ['fragile'], rob
    assert rob['summary']['all_holm_rejections_stable_under_one_seed_drop'] is False, rob
    assert 'does not change the reported p' in rob['per_comparison']['fragile']['note']
    # A declared family smaller than the supplied comparisons must fail closed
    # instead of silently relaxing the Holm correction.
    try:
        L.confirmation_robustness(
            {'a': {'deltas': [0.5] * 6, 'seeds': list(range(6))},
             'b': {'deltas': [0.5] * 6, 'seeds': list(range(6))}},
            alpha=0.05, family_size=1)
    except ValueError:
        pass
    else:
        raise AssertionError('family_size smaller than the comparison set must be refused')


def test_confirmation_robustness_never_edits_the_verdict():
    pairs = {'cmp': {'deltas': [0.5] * 6, 'seeds': list(range(6))}}
    stats = {'comparisons': {'cmp': {
                 'direction_ok': True, 'param_matched': True,
                 'param_match_complete': True, 'n_effective': 6,
                 'min_seeds_for_confirmation': 3, 'seeds': list(range(6))}},
             'holm': {'cmp': {'p_holm': 0.01}},
             'confirmed': ['cmp']}
    before = dict(stats)
    rob = V12._confirmation_robustness_block(
        stats, pairs, alpha=0.05, family_size=6, param_required=True)
    assert rob is not None and stats['robustness'] is rob, stats
    # The frozen verdict itself is untouched: the helper only adds a block.
    assert stats['confirmed'] == ['cmp'] and stats['holm'] == before['holm'], stats
    assert rob['confirmed_without_coverage_full'] == ['cmp'], rob
    assert rob['confirmed_without_coverage_by_fold'] == {
        '0': [], '1': [], '2': [], '3': [], '4': [], '5': []}, rob
    assert rob['confirmed_without_coverage_stable'] is False, rob
    assert 'complete-coverage rule is not replayed' in rob['coverage_gate_note']
    # The rendered section carries the boundary verbatim and names the row.
    md = V12._robustness_markdown(rob)
    assert md and '结论稳健性' in md[0], md
    assert any('post-hoc' in line for line in md), md
    assert any('| cmp |' in line for line in md), md
    assert any('不能被描述为' in line for line in md), md
    assert any('并列 6 个' in line for line in md), md
    _cmp_row = next(line for line in md if line.startswith('| cmp |'))
    assert '[' in _cmp_row and ']' in _cmp_row, _cmp_row
    assert V12._robustness_markdown({}) == []
    assert V12._robustness_markdown(None) == []
    # A declared-but-omitted comparison stays in the family jackknife at p=1.
    stats_omit = {'comparisons': {'cmp': dict(stats['comparisons']['cmp'])},
                  'holm': dict(stats['holm']), 'confirmed': ['cmp']}
    rob_omit = V12._confirmation_robustness_block(
        stats_omit, {'cmp': pairs['cmp']}, alpha=0.05, family_size=6,
        param_required=True, family_names=['cmp', 'omitted'])
    assert rob_omit['family_jackknife']['n_comparisons'] == 2, rob_omit['family_jackknife']
    assert 'omitted' in rob_omit['family_jackknife']['never_rejected'], rob_omit
    # A stable n=12 comparison keeps the direction/parameter verdict in every
    # fold, so the same block reports stability rather than a false alarm.
    pairs12 = {'cmp': {'deltas': [0.5] * 12, 'seeds': list(range(12))}}
    stats12 = {'comparisons': {'cmp': {
                   'direction_ok': True, 'param_matched': True,
                   'param_match_complete': True, 'n_effective': 12,
                   'min_seeds_for_confirmation': 3, 'seeds': list(range(12))}},
               'holm': {'cmp': {'p_holm': 0.01}}, 'confirmed': ['cmp']}
    rob12 = V12._confirmation_robustness_block(
        stats12, pairs12, alpha=0.05, family_size=6, param_required=True)
    assert rob12['confirmed_without_coverage_stable'] is True, rob12
    # A panel sitting exactly on the declared seed floor cannot survive a drop:
    # the n-gate is replayed alongside direction/parameters (only complete
    # coverage is not).
    stats_floor = {'comparisons': {'cmp': dict(
                       stats12['comparisons']['cmp'], min_seeds_for_confirmation=12)},
                   'holm': {'cmp': {'p_holm': 0.01}}, 'confirmed': ['cmp']}
    rob_floor = V12._confirmation_robustness_block(
        stats_floor, pairs12, alpha=0.05, family_size=6, param_required=True)
    assert rob_floor['confirmed_without_coverage_full'] == ['cmp'], rob_floor
    assert rob_floor['confirmed_without_coverage_by_fold'] == {
        str(s): [] for s in range(12)}, rob_floor
    assert rob_floor['confirmed_without_coverage_stable'] is False, rob_floor
    # No pairs -> no block (an omitted comparison cannot fabricate robustness).
    stats3 = {'comparisons': {}, 'holm': {}, 'confirmed': []}
    assert V12._confirmation_robustness_block(
        stats3, {}, alpha=0.05, family_size=6, param_required=False) is None
    assert 'robustness' not in stats3, stats3


def test_doc_level_breadth_diagnostics():
    na = [1.0, 2.0, 3.0, 100.0]
    nb = [2.0, 2.0, 2.0, 2.0]
    recs_a = {s: {'by_len': {512: {'ppl_per_seq': list(na)}}} for s in (0, 1)}
    recs_b = {s: {'by_len': {512: {'ppl_per_seq': list(nb)}}} for s in (0, 1)}
    got = V12._doc_level_analysis(recs_a, recs_b, [0, 1], 512)
    assert got is not None, got
    # Δ = a-b per document: [-1, 0, +1, +98].
    assert got['doc_delta_negative'] == 1, got
    assert got['doc_delta_positive'] == 2, got
    assert got['doc_delta_zero'] == 1, got
    assert abs(got['doc_a_better_fraction'] - 0.25) < 1e-12, got
    assert abs(got['per_doc_delta_median'] - 0.5) < 1e-12, got
    assert abs(got['top_doc_abs_share'] - 98.0 / 100.0) < 1e-12, got
    assert 'outlier-sensitive' in got['breadth_note'], got
    # Mismatched per-document row counts are refused, not silently truncated.
    bad_a = {s: {'by_len': {512: {'ppl_per_seq': [1.0, 2.0]}}} for s in (0, 1)}
    refused = V12._doc_level_analysis(bad_a, recs_b, [0, 1], 512)
    assert refused is not None and refused['comparable'] is False, refused


def test_window_geometry_is_built_per_chunk():
    # The full [n, w] window table is O(n*w) and is only consumed by the
    # full-width mask hoist.  The row-chunked attention path must build the
    # rows it reads and produce exactly the values of the full table.
    torch = __import__('torch')
    n, w = 64, 6
    full_i, full_v = L._window_geometry(n, w, 'cpu')
    for (s, e) in ((0, n), (0, 1), (17, 33), (n - 3, n), (n, n)):
        for off in (0, 5, 123):
            gi, gv = L._window_geometry_rows(n, w, 'cpu', s, e, offset=off)
            ei = ((full_i[s:e] + off).to(torch.int32) if off else full_i[s:e])
            assert gi.shape == (max(0, e - s), w) and gv.shape == gi.shape
            if s < e:
                assert torch.equal(gi, ei)
                assert torch.equal(gv, full_v[s:e])

    # End-to-end: force the chunked branch and make any full-table build
    # fail, so a regression that reintroduces the O(n*w) allocation is
    # caught rather than merely slow.
    nh, hd, nb, B = 2, 8, 5, 11
    q = torch.randn(64, nh, hd, dtype=torch.float64)
    kb = torch.randn(B, nh, hd, dtype=torch.float64)
    vb = torch.randn(B, nh, hd, dtype=torch.float64)
    ks = torch.randn(64, nh, hd, dtype=torch.float64)
    vs = torch.randn(64, nh, hd, dtype=torch.float64)
    idx = torch.randint(0, B, (64, nb))
    last = torch.randint(0, 64, (B,))
    valid = torch.rand(64, nb) < 0.8
    ref = L.gathered_attention(q, kb, vb, idx, last, ks, vs, w, 1.0,
                               sel_valid=valid)
    old_ok, old_geo = L._attn_full_width_ok, L._window_geometry

    def _no_full(*_a, **_k):
        raise AssertionError('the chunked path built the full window table')

    try:
        L._attn_full_width_ok = lambda *_a, **_k: False
        L._window_geometry = _no_full
        got = L.gathered_attention(q, kb, vb, idx, last, ks, vs, w, 1.0,
                                   sel_valid=valid)
    finally:
        L._attn_full_width_ok, L._window_geometry = old_ok, old_geo
    assert torch.equal(ref, got), (ref - got).abs().max().item()


def test_causal_quantile_positions_are_exact():
    # The causal quantile rank is q * c with q = 1/target (0.001 above 1000
    # target tokens).  A float32 product can round an exact integer rank DOWN
    # (1/41 * 41 -> 0.99999994) and loses the column itself above 2**24.  The
    # rank/remainder must be computed with the integer division instead.
    torch = __import__('torch')
    n = 100
    x = (torch.arange(n, dtype=torch.float32) * 100.0).reshape(1, n)
    cols = torch.arange(n, dtype=torch.long)
    for full in (False, True):
        out = L._causal_quantile_exact(x, cols, 41, _full_range=full)
        assert out.dtype == x.dtype, (full, out.dtype)
        # c=41 has exact position 1.0, so the second-smallest value (100) is
        # selected exactly; the old float32 product interpolated from rank 0
        # with frac=0.99999994 and returned 99.99999.
        assert out[0, 41].item() == 100.0, (full, out[0, 41].item())
        assert out[0, 82].item() == 200.0, (full, out[0, 82].item())
        assert out[0, 0].item() == float('-inf'), (full, out[0, 0].item())
    # Same exact ranks through the public segmenter entry point (the caller
    # passes the 1-D similarity row, not the private [1, n] batch).
    tau = L.causal_adaptive_threshold(x.reshape(-1), 41)
    assert tau[41].item() == 100.0, tau[41].item()
    assert tau[82].item() == 200.0, tau[82].item()
    assert tau[0].item() == float('-inf'), tau[0].item()

    # Positions stay exact beyond float32's 2**24 integer range and beyond
    # 2**53 (the float64 integer range), because no float is used for lo/rem.
    c = torch.tensor([2 ** 24 + 1, 2 ** 24 + 2, 2 ** 53 + 1, 1234567890123],
                     dtype=torch.long)
    for target, den in ((3, 3), (41, 41), (1001, 1000)):
        _q, lo, frac, _lc, _hc, use_hi, _fr = L._causal_quantile_positions(
            c, target, 2 ** 63, torch.float64)
        assert torch.equal(lo, torch.div(c, den, rounding_mode='floor')), (target, lo)
        rem = c - lo * den
        assert torch.equal(use_hi, (rem > 0) & (lo + 1 <= c)), (target, use_hi)
        assert torch.equal(frac, rem.to(torch.float64) / float(den)), (target, frac)
    # A column whose position is an exact integer (rem == 0) must use the
    # lower order statistic directly, not a phantom interpolation.
    c2 = torch.tensor([41, 82, 123], dtype=torch.long)
    _q, lo2, frac2, _lc, _hc, use_hi2, _fr = L._causal_quantile_positions(
        c2, 41, 1000, torch.float32)
    assert lo2.tolist() == [1, 2, 3], lo2
    assert frac2.tolist() == [0.0, 0.0, 0.0], frac2
    assert not bool(use_hi2.any()), use_hi2


def test_per_token_document_sidecar():
    rows = ['alpha document', 'beta document', 'gamma document']
    lengths = [5, 3, 7]
    cap = 10
    tok = L._token_doc_ids_for_capped_rows(rows, lengths, cap)
    doc = L._doc_ids_for_capped_rows(rows, lengths, cap)
    assert tok.dtype == np.uint64 and tok.size == cap, (tok.dtype, tok.size)
    assert doc.tolist() == [int(L._doc_id_u64(L._doc_exact_fp(r))) for r in rows]
    ids = [int(L._doc_id_u64(L._doc_exact_fp(r))) for r in rows]
    assert tok.tolist() == [ids[0]] * 5 + [ids[1]] * 3 + [ids[2]] * 2
    # The two sidecars must name the same document set even when the cap
    # truncates the final document.
    assert np.array_equal(np.unique(tok), np.unique(doc))
    # A cap that ends exactly at a document boundary keeps the EOS with the
    # document that emitted it (length includes the EOS).
    assert L._token_doc_ids_for_capped_rows(rows, lengths, 5).tolist() == [ids[0]] * 5
    # Legacy-cache reconstruction from EOS boundaries: one EOS per complete
    # document, and the final (cap-truncated) document has no EOS.
    eos = 99
    toks = np.asarray([1, 2, eos, 3, eos, 4, 5], dtype=np.int64)
    docs = np.asarray([10, 20, 30], dtype=np.uint64)
    rec, why = L._token_doc_ids_from_eos(toks, docs, eos)
    assert why is None and rec.tolist() == [10] * 3 + [20] * 2 + [30] * 2, (why, rec)
    complete = np.asarray([1, eos, 2, eos, 3, eos], dtype=np.int64)
    rec2, why2 = L._token_doc_ids_from_eos(complete, docs, eos)
    assert why2 is None and rec2.tolist() == [10] * 2 + [20] * 2 + [30] * 2, (why2, rec2)
    # The invariant must fail closed when the EOS count/order cannot describe
    # the sidecar (rather than shifting every document boundary silently).
    bad1, _ = L._token_doc_ids_from_eos(toks[:1], docs, eos)
    assert bad1 is None
    bad2, _ = L._token_doc_ids_from_eos(
        np.asarray([1, eos, 2, eos, 3], dtype=np.int64), docs[:2], eos)
    assert bad2 is None
    bad3, _ = L._token_doc_ids_from_eos(toks, docs, None)
    assert bad3 is None


def test_document_level_eval_uses_real_boundaries():
    import copy as _copy
    torch.manual_seed(0)
    dev = torch.device('cpu')
    cfgs = L.make_layer_cfgs(2, 'full_rope')
    model = L.SmallGPT(256, 64, 2, 4, 16, 64, _copy.deepcopy(cfgs)).to(dev)
    model.eval()
    ids = np.asarray(np.random.default_rng(11).integers(0, 256, size=(3, 65)),
                     dtype=np.int64)
    # Deliberately make documents span row boundaries and rows mix documents.
    doc = np.empty_like(ids, dtype=np.uint64)
    doc[0, :33] = 11
    doc[0, 33:] = 22
    doc[1, :21] = 22
    doc[1, 21:] = 33
    doc[2, :11] = 33
    doc[2, 11:] = 44
    for Ln in (32, 64):
        cell = V7.eval_length_gen(model, ids, [Ln], dev, max_pos=None,
                                  doc_ids=doc)[Ln]
        assert cell['doc_granularity'] == 'document', cell
        assert cell['n_docs'] == int(np.unique(doc[:3, 1:Ln]).size), cell
        assert sum(cell['n_tokens_per_doc']) == cell['n_tok'], cell
        assert len(cell['ppl_per_doc']) == cell['n_docs'], cell
        # The per-document NLLs partition the same totals as the row-level
        # accounting; only the grouping changes.  The tiny difference is
        # float32 per-token values summed in a different order, so compare
        # relatively rather than to the last bit.
        assert abs(sum(cell['nll_per_doc']) - cell['nll_sum']) <= 1e-5 * max(1.0, abs(cell['nll_sum'])), cell
        assert sum(cell['n_tokens_per_doc']) == cell['n_tok'], cell
        # Token-weighted pooled PPL identity: exp(sum n_d*log ppl_d / sum n_d).
        _logs = [n * math.log(p) for n, p in
                 zip(cell['n_tokens_per_doc'], cell['ppl_per_doc'])]
        _pooled = math.exp(sum(_logs) / sum(cell['n_tokens_per_doc']))
        assert abs(_pooled - cell['ppl']) / cell['ppl'] < 1e-6, (Ln, _pooled, cell['ppl'])
    # One document per row: the document and row readouts must coincide.
    doc2 = np.repeat(np.asarray([[101], [202], [303]], dtype=np.uint64), 65, axis=1)
    one = V7.eval_length_gen(model, ids, [64], dev, max_pos=None, doc_ids=doc2)[64]
    assert one['doc_granularity'] == 'document' and one['n_docs'] == 3, one
    for a, b in zip(one['ppl_per_doc'], one['ppl_per_seq']):
        assert abs(a - b) < 1e-9 * max(1.0, abs(b)), (a, b)
    # The truncated branch must aggregate the SAME (start, end) token slice.
    trunc = V7.eval_length_gen(model, ids, [48, 64], dev, max_pos=32,
                               doc_ids=doc)
    for Ln in (48, 64):
        cell = trunc[Ln]
        assert cell['truncated'] is True and cell['doc_granularity'] == 'document', cell
        assert cell['n_tok'] == 3 * 31 and sum(cell['n_tokens_per_doc']) == 3 * 31, cell
        assert abs(sum(cell['nll_per_doc']) - cell['nll_sum']) <= 1e-5 * max(1.0, abs(cell['nll_sum'])), cell
    # A misaligned, float or negative mapping is refused before scoring.
    for bad in (doc[:, :-1], doc.astype(np.float64), -(doc.astype(np.int64))):
        try:
            V7.eval_length_gen(model, ids, [32], dev, max_pos=None, doc_ids=bad)
        except ValueError:
            pass
        else:
            raise AssertionError('eval_length_gen accepted a bad document mapping')


def test_document_aggregation_omits_documents_absent_from_the_slice():
    # A global document map is built on the LONGEST evaluated prefix.  For a
    # shorter by-length cell, documents that have not started yet must not be
    # emitted with 0 tokens and NaN PPL: n_docs would be inflated and the
    # per-document vectors would contain non-units.
    full = np.asarray([[1, 1, 2, 2, 3],
                       [1, 2, 2, 2, 3]], dtype=np.uint64)
    full_nll = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    keys, inv = L.document_index(full)
    short = L.aggregate_nll_by_document(
        full_nll[:, :4], full[:, :4],
        document_index_map=(keys, inv[:, :4]))
    assert short['n_docs'] == 2, short
    assert short['doc_ids'] == ['0000000000000001', '0000000000000002'], short
    assert short['n_tokens_per_doc'] == [3, 5], short
    assert all(math.isfinite(p) and p > 0 for p in short['ppl_per_doc']), short
    assert short['n_tokens_total'] == 8 and short['n_docs'] == len(short['nll_per_doc']), short
    # The explicit map must be a valid (keys, inverse) for THIS slice; a
    # stale/out-of-range/float map is refused instead of silently dropping
    # tokens in the scatter.
    for bad in ((keys, inv[:, :4] + 100),
                (keys.astype(np.float64), inv[:, :4]),
                (keys, inv[:, :4].astype(np.float64)),
                (np.asarray([10, 20, 30]), inv[:, :4])):
        try:
            L.aggregate_nll_by_document(
                full_nll[:, :4], full[:, :4], document_index_map=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'aggregate_nll_by_document accepted {bad!r}')
    # End-to-end: when two lengths are evaluated together, the shorter cell
    # must contain only the documents present in that prefix.
    import copy as _copy
    torch.manual_seed(0)
    dev = torch.device('cpu')
    cfgs = L.make_layer_cfgs(2, 'full_rope')
    model = L.SmallGPT(256, 64, 2, 4, 16, 64, _copy.deepcopy(cfgs)).to(dev)
    model.eval()
    ids = np.asarray(np.random.default_rng(12).integers(0, 256, size=(2, 33)),
                     dtype=np.int64)
    doc = np.empty_like(ids, dtype=np.uint64)
    doc[:, :20] = 7
    doc[:, 20:] = 9
    cells = V7.eval_length_gen(model, ids, [8, 32], dev, max_pos=None,
                               doc_ids=doc)
    assert cells[8]['n_docs'] == 1, cells[8]
    assert cells[8]['n_tokens_per_doc'] == [14], cells[8]
    assert cells[32]['n_docs'] == 2, cells[32]
    assert cells[32]['n_tokens_per_doc'] == [38, 24], cells[32]
    assert all(math.isfinite(p) for p in cells[8]['ppl_per_doc']), cells[8]


def test_doc_level_analysis_prefers_document_cells():
    na = [1.0, 2.0, 3.0, 100.0]
    nb = [2.0, 2.0, 2.0, 2.0]
    docs = ['0000000000000001', '0000000000000002',
            '0000000000000003', '0000000000000004']
    toks = [10, 10, 10, 10]
    recs_a = {s: {'by_len': {512: {'ppl_per_seq': list(na),
                                    'ppl_per_doc': list(na),
                                    'n_tokens_per_doc': list(toks),
                                    'doc_ids': list(docs)}}} for s in (0, 1)}
    recs_b = {s: {'by_len': {512: {'ppl_per_seq': list(nb),
                                    'ppl_per_doc': list(nb),
                                    'n_tokens_per_doc': list(toks),
                                    'doc_ids': list(docs)}}} for s in (0, 1)}
    got = V12._doc_level_analysis(recs_a, recs_b, [0, 1], 512)
    assert got is not None and got['granularity'] == 'document', got
    assert got['n_docs'] == 4, got
    assert got['doc_delta_negative'] == 1 and got['doc_delta_positive'] == 2, got
    assert got['doc_delta_zero'] == 1, got
    # Mixing document-level and row-level cells would silently change the
    # estimand; the analysis must refuse rather than fall back per record.
    mixed = {s: {'by_len': {512: {'ppl_per_seq': list(na)}}} for s in (0,)}
    mixed.update({s: recs_a[s] for s in (1,)})
    bad = V12._doc_level_analysis(mixed, recs_b, [0, 1], 512)
    assert bad is not None and bad['comparable'] is False, bad
    # Different documents (token counts) on the two sides are not paired.
    recs_b_bad = {s: {'by_len': {512: {'ppl_per_doc': list(nb),
                                        'n_tokens_per_doc': [99, 10, 10, 10],
                                        'doc_ids': list(docs)}}} for s in (0, 1)}
    bad2 = V12._doc_level_analysis(recs_a, recs_b_bad, [0, 1], 512)
    assert bad2 is not None and bad2['comparable'] is False, bad2
    # A document-level by_len cell is admissible only when its per-document
    # token counts partition the scored targets exactly; a partial cell is
    # rejected by the same gate that protects the row-level endpoint.
    good_cell = {'n_tok': 4 * 7, 'truncated': False, 'ppl_per_seq': [1.0] * 4,
                 'n_seq': 4, 'doc_granularity': 'document',
                 'nll_per_doc': [1.0, 2.0], 'n_tokens_per_doc': [20, 8],
                 'ppl_per_doc': [1.0, 2.0],
                 'doc_ids': ['0000000000000001', '0000000000000002'],
                 'n_docs': 2}
    assert V12._by_len_shape_reason({'by_len': {8: dict(good_cell)}}, 4) is None
    broken = dict(good_cell)
    broken['n_tokens_per_doc'] = [20, 7]
    assert 'n_tokens_per_doc sums' in str(
        V12._by_len_shape_reason({'by_len': {8: broken}}, 4))
    missing = dict(good_cell)
    del missing['nll_per_doc']
    assert 'missing' in str(
        V12._by_len_shape_reason({'by_len': {8: missing}}, 4))
    # A per-document NLL vector is a measurement, not a decorative field: it
    # must be finite and must reconstruct the cell's recorded nll_sum.
    summed = dict(good_cell, nll_sum=3.0)
    assert V12._by_len_shape_reason({'by_len': {8: summed}}, 4) is None
    bad_nll_sum = dict(good_cell, nll_sum=4.0)
    assert 'nll_per_doc sums' in str(
        V12._by_len_shape_reason({'by_len': {8: bad_nll_sum}}, 4))
    bad_nll = dict(good_cell, nll_per_doc=[1.0, float('nan')])
    assert 'nll_per_doc contains' in str(
        V12._by_len_shape_reason({'by_len': {8: bad_nll}}, 4))
    # A zero-target document is not an evaluation unit: the aggregator drops
    # it, so a cell that still declares it is internally inconsistent.
    zero_doc = dict(good_cell, n_tokens_per_doc=[28, 0])
    assert 'at least one scored' in str(
        V12._by_len_shape_reason({'by_len': {8: zero_doc}}, 4))


def test_paired_token_pooled_document_ci():
    # One seed, two documents with unequal lengths: the point must be the
    # token-pooled PPL difference, not the equal-weighted mean document Δ.
    tokens = [10, 30]
    a = [[2.0, 6.0]]
    b = [[1.0, 7.0]]
    got = L._paired_nll_doc_ci(a, b, tokens)
    assert got is not None, got
    want = math.exp(8.0 / 40.0) - math.exp(8.0 / 40.0)
    assert abs(got['delta_ppl'] - want) < 1e-12, got
    assert got['n_docs'] == 2 and got['n_seeds'] == 1 and got['df'] == 1, got
    lo, hi = got['delta_ppl_ci95']
    assert lo <= got['delta_ppl'] <= hi, got
    # Two seeds with different pooled effects: the point is the mean of the
    # per-seed pooled PPL differences (the frozen comparison's estimand).
    a2 = [[2.0, 6.0, 30.0], [3.0, 6.0, 30.0]]
    b2 = [[3.0, 5.0, 31.0], [3.0, 5.0, 31.0]]
    toks2 = [10, 20, 70]
    got2 = L._paired_nll_doc_ci(a2, b2, toks2)
    per_seed = [math.exp(sum(x) / 100.0) - math.exp(sum(y) / 100.0)
                 for x, y in zip(a2, b2)]
    assert abs(got2['delta_ppl'] - sum(per_seed) / 2.0) < 1e-12, got2
    assert got2['n_docs'] == 3 and got2['n_seeds'] == 2 and got2['df'] == 2, got2
    lo2, hi2 = got2['delta_ppl_ci95']
    assert lo2 <= got2['delta_ppl'] <= hi2, got2
    assert got2['delta_log_ratio_ci95'] is not None, got2
    assert got2['ratio_ci95'][0] <= math.exp(got2['delta_log_ratio']) <= got2['ratio_ci95'][1], got2
    # A single document is a point estimate, not a fabricated interval.
    one = L._paired_nll_doc_ci([[3.0]], [[2.0]], [10])
    assert one['delta_ppl_ci95'] is None and one['df'] is None, one
    # Divergent CE is unavailable rather than an inf-valued interval.
    assert L._paired_nll_doc_ci([[2000.0, 2000.0]], [[0.0, 0.0]], [1, 1]) is None
    # Shape/token/probability guards.
    for _a, _b, _t in (([[1.0, 2.0]], [[1.0]], [1, 1]),
                       ([[1.0, 2.0]], [[1.0, 2.0]], [1]),
                       ([[1.0, 2.0]], [[1.0, 2.0]], [1, 0]),
                       ([[1.0, 2.0]], [[1.0, 2.0]], [1, float('nan')])):
        try:
            L._paired_nll_doc_ci(_a, _b, _t)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted malformed paired NLL input {(_a, _b, _t)!r}')
    for _conf in (True, 0.0, 1.0, float('nan')):
        try:
            L._paired_nll_doc_ci([[1.0, 2.0]], [[1.0, 2.0]], [1, 1], conf=_conf)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted invalid conf={_conf!r}')


def test_doc_level_analysis_uses_the_pooled_estimand_ci():
    docs = ['0000000000000001', '0000000000000002', '0000000000000003']
    toks = [10, 20, 70]
    nll_a = [2.0, 6.0, 30.0]
    nll_b = [3.0, 5.0, 31.0]
    recs_a = {s: {'by_len': {512: _document_cell(nll_a, toks, docs)}}
              for s in (0, 1)}
    recs_b = {s: {'by_len': {512: _document_cell(nll_b, toks, docs)}}
              for s in (0, 1)}
    got = V12._doc_level_analysis(recs_a, recs_b, [0, 1], 512)
    assert got is not None and got['granularity'] == 'document', got
    want = math.exp(38.0 / 100.0) - math.exp(39.0 / 100.0)
    assert 'pooled_delta_ppl' in got and abs(got['pooled_delta_ppl'] - want) < 1e-12, got
    assert 'pooled_delta_ppl_ci95' in got, got
    lo, hi = got['pooled_delta_ppl_ci95']
    assert lo <= got['pooled_delta_ppl'] <= hi, got
    assert got['pooled_ci_n_docs'] == 3 and got['pooled_ci_df'] == 2, got
    assert got['pooled_ci_method'].startswith('cluster-robust'), got
    # Equal-weight document mean is a different estimand; the old bootstrap is
    # retained only as a descriptive field, not as the primary CI.
    assert abs(got['per_doc_delta_mean'] - got['pooled_delta_ppl']) > 1e-6, got
    assert got['per_doc_delta_ppl_ci95'] is not None, got
    assert 'cluster' in V12._doc_ci_markdown_cell(got), got
    # A vector that does not reconstruct the recorded nll_sum is refused, and
    # the refusal is explicit rather than a silently wrong interval.
    bad_cell = _document_cell(nll_a, toks, docs)
    bad_cell['nll_per_doc'] = [nll_a[0] + 1.0, nll_a[1], nll_a[2]]
    bad = {s: {'by_len': {512: dict(bad_cell)}} for s in (0, 1)}
    refused = V12._doc_level_analysis(bad, recs_b, [0, 1], 512)
    assert 'pooled_delta_ppl_ci95' not in refused, refused
    assert 'partition' in refused.get('pooled_ci_unavailable_reason', ''), refused
    # A ppl_per_doc that disagrees with nll/tokens is equally unusable.
    bad_ppl = {s: {'by_len': {512: dict(
        _document_cell(nll_a, toks, docs),
        ppl_per_doc=[1.0, 1.0, 1.0])}} for s in (0, 1)}
    refused2 = V12._doc_level_analysis(bad_ppl, recs_b, [0, 1], 512)
    assert 'pooled_delta_ppl_ci95' not in refused2, refused2
    assert 'inconsistent' in refused2.get('pooled_ci_unavailable_reason', ''), refused2
    # The cluster interval must also be refused when the cell has no finite
    # per-seed pooled PPL to tie its point estimate to: a missing primary
    # value is not evidence that the document vectors reproduce it.
    _no_ppl_a = _document_cell(nll_a, toks, docs)
    _no_ppl_b = _document_cell(nll_b, toks, docs)
    _no_ppl_a.pop('ppl')
    _no_ppl_b.pop('ppl')
    no_ppl = ({s: {'by_len': {512: dict(_no_ppl_a)}} for s in (0, 1)},
              {s: {'by_len': {512: dict(_no_ppl_b)}} for s in (0, 1)})
    refused3 = V12._doc_level_analysis(no_ppl[0], no_ppl[1], [0, 1], 512)
    assert 'pooled_delta_ppl_ci95' not in refused3, refused3
    assert 'pooled PPL' in refused3.get('pooled_ci_unavailable_reason', ''), refused3
    # Row-level cells keep the labelled row bootstrap and never claim a
    # token-pooled document interval.
    rows_a = {s: {'by_len': {512: {'ppl_per_seq': [1.0, 2.0, 4.0]}}}
              for s in (0, 1)}
    rows_b = {s: {'by_len': {512: {'ppl_per_seq': [2.0, 2.0, 2.0]}}}
              for s in (0, 1)}
    row = V12._doc_level_analysis(rows_a, rows_b, [0, 1], 512)
    assert row is not None and row['granularity'] == 'row', row
    assert 'pooled_delta_ppl_ci95' not in row, row
    assert 'bootstrap' in V12._doc_ci_markdown_cell(row), row


def test_doc_ci_markdown_cell_prefers_pooled_cluster():
    assert V12._doc_ci_markdown_cell({}) == '—'
    assert V12._doc_ci_markdown_cell(None) == '—'
    dsm = {'granularity': 'document',
           'pooled_delta_ppl_ci95': [-1.25, 0.5],
           'per_doc_delta_ppl_ci95': [-9.0, 9.0]}
    cell = V12._doc_ci_markdown_cell(dsm)
    assert 'cluster' in cell and 'token 池化' in cell and '-1.25' in cell, cell
    fallback = {'granularity': 'document',
                'per_doc_delta_ppl_ci95': [-3.0, 2.0]}
    cell2 = V12._doc_ci_markdown_cell(fallback)
    assert 'bootstrap' in cell2 and '等权' in cell2, cell2
    row = {'granularity': 'row', 'per_doc_delta_ppl_ci95': [-1.0, 1.0]}
    cell3 = V12._doc_ci_markdown_cell(row)
    assert '行 bootstrap' in cell3, cell3
    # A non-finite/invalid pooled interval must fall back, not render garbage.
    bad = {'granularity': 'document', 'pooled_delta_ppl_ci95': [float('nan'), 1.0],
           'per_doc_delta_ppl_ci95': [-2.0, 2.0]}
    cell4 = V12._doc_ci_markdown_cell(bad)
    assert 'bootstrap' in cell4, cell4


def test_paired_init_core_audit():
    # The digest is seed-dependent, so a healthy panel has DIFFERENT values
    # across seeds and EQUAL values across the two arms within a seed.
    a = {0: {'init_core_fp': 'aa'}, 1: {'init_core_fp': 'bb'},
         2: {'init_core_fp': 'cc'}}
    b = {0: {'init_core_fp': 'aa'}, 1: {'init_core_fp': 'XX'}, 2: {}}
    r = L.paired_init_core_audit(a, b, [0, 1, 2])
    assert r['n_pairs'] == 3 and r['n_verified'] == 1, r
    assert r['n_mismatch'] == 1 and r['n_unverified'] == 1, r
    assert r['mismatch_seeds'] == [1] and r['unverified_seeds'] == [2], r
    assert r['keep_seeds'] == [0, 2], r
    assert r['pairing_ok'] is False, r
    # All pairs verified: the across-seed values still differ, which must NOT
    # be misread as a mismatch.
    ok = L.paired_init_core_audit(a, {0: {'init_core_fp': 'aa'},
                                      1: {'init_core_fp': 'bb'},
                                      2: {'init_core_fp': 'cc'}}, [0, 1, 2])
    assert ok['pairing_ok'] is True and ok['n_mismatch'] == 0, ok
    # No digest anywhere is UNVERIFIED (None), never the vacuous True that made
    # the old global `len(values) <= 1` test look like a pass.
    empty = L.paired_init_core_audit({0: {}, 1: {}}, {0: {}, 1: {}}, [0, 1])
    assert empty['pairing_ok'] is None and empty['n_unverified'] == 2, empty
    # A one-sided digest is unverifiable, not equality.
    one = L.paired_init_core_audit({0: {'init_core_fp': 'a'}}, {0: {}}, [0])
    assert one['n_unverified'] == 1 and one['pairing_ok'] is None, one
    # A malformed non-string value must not stringify into a match.
    mal = L.paired_init_core_audit({0: {'init_core_fp': 123}},
                                   {0: {'init_core_fp': '123'}}, [0])
    assert mal['n_unverified'] == 1 and mal['pairing_ok'] is None, mal
    # A missing record is not a pair either.
    miss = L.paired_init_core_audit({0: {'init_core_fp': 'a'}}, {}, [0])
    assert miss['n_missing'] == 1 and miss['pairing_ok'] is None, miss
    assert L.paired_init_core_audit({}, {}, [])['pairing_ok'] is None
    # Seed identity is validated exactly like the other fragility helpers.
    for _bad in ('0', 0.0, True, 3):
        try:
            L.paired_init_core_audit({}, {}, _bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'seed collection {_bad!r} was accepted')
    for _call in (lambda: L.paired_init_core_audit([], {}, [0]),
                  lambda: L.paired_init_core_audit({}, {}, [0, 0]),
                  lambda: L.paired_init_core_audit({}, {}, [0, [0]]),
                  lambda: L.paired_init_core_audit({}, {}, [0, True])):
        try:
            _call()
        except ValueError:
            pass
        else:
            raise AssertionError('malformed audit input must raise ValueError')


def test_v12_init_core_fp_pairing():
    class _EmptyModel:
        def state_dict(self):
            return {}

    # The metadata must actually travel with the weights, not only with the
    # training summary; the P5T writer reads it from this payload.
    _pl = L._checkpoint_payload(_EmptyModel(), variant='v', seed=0, step=1,
                                val_ppl=1.0, train_ce=1.0, health={},
                                extra={'init_core_fp': 'DEADBEEF'})
    assert _pl.get('init_core_fp') == 'DEADBEEF', _pl

    def _mk_manifest(td, seeds):
        m = {'frozen_at': '2026-10-22', 'code_semantics_expected': L.CODE_SEMANTICS,
             'arms': ['a_arm', 'b_arm'], 'seeds': list(seeds),
             'primary_comparisons': [{'name': 'c1', 'a': 'a_arm', 'b': 'b_arm',
                                      'endpoint': 'by_len.ppl@4096',
                                      'expected_sign': 'negative'}],
             'statistics': {'test': 'exact two-sided sign-flip',
                            'correction': 'holm step-down',
                            'descriptive_only': True,
                            'require_complete_coverage': True,
                            'min_seeds_for_confirmation': 2}}
        p = os.path.join(td, 'm.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(m, f)
        return V12._load_manifest(p)

    def _rec(v, s, init, ppl, msha):
        r = {'variant': v, 'seed': s, 'split': 'test', 'data_fp': 'FP',
             'by_len': {'4096': {'ppl': ppl, 'n_tok': 1000,
                                 'truncated': False}},
             '_code': L.code_stamp(), 'manifest_sha256': msha}
        if init is not None:
            r['init_core_fp'] = init
        return r

    with tempfile.TemporaryDirectory() as td:
        m = _mk_manifest(td, [0, 1])
        outdir = os.path.join(td, 'res')
        os.makedirs(outdir)
        V12._check_or_write_freeze_lock(outdir, m, 'FP', stage='accessed')
        bp = os.path.join(outdir, 'bylen.json')

        def _run(raw, tag):
            with open(bp, 'w', encoding='utf-8') as f:
                json.dump(raw, f)
            return V12.finaltest_stats(
                manifest=m, bylen_path=bp,
                probe_path=os.path.join(td, 'no_probe.json'),
                out=os.path.join(td, f'stats_{tag}.json'))

        # Healthy: each seed has its own init, both arms agree per seed.
        raw = {'a_arm::s0': _rec('a_arm', 0, 'I0', 10.0, m['manifest_sha256']),
               'b_arm::s0': _rec('b_arm', 0, 'I0', 12.0, m['manifest_sha256']),
               'a_arm::s1': _rec('a_arm', 1, 'I1', 10.0, m['manifest_sha256']),
               'b_arm::s1': _rec('b_arm', 1, 'I1', 13.0, m['manifest_sha256'])}
        st = _run(raw, 'ok')
        c = st['comparisons']['c1']
        assert c['n_effective'] == 2 and c['coverage_ok'] is True, c
        assert c['same_init_core_fp'] is True, c
        assert c['init_core_fp']['pairing_ok'] is True, c['init_core_fp']
        assert c['init_core_fp_recorded'] is True, c
        assert st['validity']['same_init_core_fp'] is True, st['validity']
        assert st['validity']['init_core_fp_complete'] is True, st['validity']

        # One arm's seed-1 digest differs: that seed is not a matched pair and
        # must be dropped by name, so a 2-seed complete-coverage plan cannot
        # confirm from the remnants.
        bad = {k: dict(v) for k, v in raw.items()}
        bad['b_arm::s1']['init_core_fp'] = 'OTHER'
        st2 = _run(bad, 'mismatch')
        c2 = st2['comparisons']['c1']
        assert 'omitted' in c2, c2
        assert c2['init_core_fp']['mismatch_seeds'] == [1], c2['init_core_fp']
        assert c2['dropped_pairs'][0]['seed'] == 1, c2['dropped_pairs']
        assert 'init_core_fp' in c2['dropped_pairs'][0]['reason'], c2
        assert st2['validity']['same_init_core_fp'] is False, st2['validity']
        assert st2['validity']['init_core_fp_mismatch_seeds'] == [1], st2

        # No digest recorded: unverified (None), never True, and no pair is
        # silently certified as matched.
        unv = {k: {kk: vv for kk, vv in v.items() if kk != 'init_core_fp'}
               for k, v in raw.items()}
        st3 = _run(unv, 'unverified')
        c3 = st3['comparisons']['c1']
        assert c3['same_init_core_fp'] is None and c3['init_core_fp_recorded'] is False, c3
        assert st3['validity']['same_init_core_fp'] is None, st3['validity']
        assert st3['validity']['init_core_fp_recorded'] is False, st3['validity']
        # One-sided evidence is also unverified, not equality.
        one = {k: dict(v) for k, v in unv.items()}
        one['a_arm::s1']['init_core_fp'] = 'ONLY_A'
        st4 = _run(one, 'one_sided')
        c4 = st4['comparisons']['c1']
        assert c4['init_core_fp']['unverified_seeds'] == [0, 1], c4['init_core_fp']
        assert c4['same_init_core_fp'] is None, c4


def test_scale_uniform_segment_and_oom_fallback():
    # The uniform-stride shortcut used to build one Python list entry per
    # block even on a long sequence.  The m=1 / block_size=1 arm makes stride
    # one, so that is exactly the scale path that must stay on the large-row
    # NumPy contract (the general walk already does).
    n = 10000
    bids, hon, counts = L._segment(n, np.zeros(n - 1, dtype=bool), 1, 1)
    assert isinstance(counts, np.ndarray) and counts.dtype == np.int64, type(counts)
    assert counts.size == n and int(counts.min()) == 1 and int(counts.max()) == 1
    assert np.array_equal(bids, np.arange(n, dtype=np.int64)), bids[:8]
    assert hon == {}
    # The all-cuts shortcut also ends with the partial tail block; its values
    # and total must be identical to the repeated-count form.
    n2 = 10001
    bids2, hon2, counts2 = L._segment(n2, np.ones(n2 - 1, dtype=bool), 2, 2)
    assert isinstance(counts2, np.ndarray) and counts2.dtype == np.int64
    exp = np.full((n2 + 1) // 2, 2, dtype=np.int64)
    exp[-1] = 1
    assert np.array_equal(counts2, exp), (counts2[-3:], exp[-3:])
    assert np.array_equal(bids2, np.arange(n2, dtype=np.int64) // 2)
    # Small rows keep the historical list contract (and the segment cache).
    assert L._segment(3, [True, True], 1, 2)[2] == [1, 1, 1]

    # Host pooling geometry is an optimization with a documented fallback:
    # "None when the host form cannot serve; the caller then falls back to the
    # on-device form".  An allocation failure is a MemoryError, which the
    # builders must translate to None instead of crashing a long-context run.
    import unittest.mock as mock
    with mock.patch.object(L.np, 'empty', side_effect=MemoryError):
        assert L._pool_parts_flat([17, 4, 9], 3, 3, 30) is None
        assert L._pool_geom_flat_var([17, 4, 9], 3, 3, 30, 4, 2, 5, 3) is None
        assert L._pool_geom_flat_single([17, 4, 9], 3, 30, 4, 5) is None


def test_fixed_blocks_and_prefix_starts_stay_numpy_at_scale():
    # `blocks_fixed` is the fixed-stride twin of `_segment`'s uniform
    # shortcut.  It used to build one Python list entry per block, so the
    # block_size=1 arm allocated O(n) Python pointers on a long sequence.
    # The values stay identical, but a large-host table must be a NumPy row
    # and the historical list shape must remain for tiny tables.
    n = 10000
    bid = L.blocks_fixed(n, 1, 'cpu')
    counts = getattr(bid, '_pool_cpu', None)
    assert counts is not None, 'blocks_fixed lost its host count attachment'
    assert isinstance(counts[0], np.ndarray) and counts[0].dtype == np.int64, type(counts[0])
    assert counts[0].size == n
    assert np.array_equal(counts[0], np.ones(n, dtype=np.int64))
    assert counts[1] == 1 and counts[2] == 1
    assert L.blocks_fixed(16, 4, 'cpu')._pool_cpu[0] == [4, 4, 4, 4]
    assert L.blocks_fixed(15, 4, 'cpu')._pool_cpu[0] == [4, 4, 4, 3]
    # The attached host counts are retained by the cache and must be charged:
    # at block_size=1 they are the same size as the cached int64 tensor.
    _tensor_bytes = bid.numel() * bid.element_size()
    _charged = L._cache_item_bytes(('blk',), bid, L._idx_cache_total)
    assert _charged >= _tensor_bytes + int(counts[0].nbytes), (_charged, _tensor_bytes)
    assert L._idx_cache_total({('blk',): bid}) == _charged

    # Prefix-invariance diagnostics must not turn a large segmenter counts row
    # back into a Python list just to prepend the zero start.
    cuts = np.zeros(1024, dtype=bool)
    cuts[::11] = True
    rep = L.segmentation_prefix_invariance(cuts, 1024, 1, 4, prefixes=8)
    assert rep['ok'], rep

    # Run-order audits must stay linear in the panel size: many blocks with
    # many cells still classify a blocked execution and an arm-major one.
    variants = ['arm%02d' % i for i in range(8)]
    seeds = list(range(500))
    cells = [(v, s) for s in seeds for v in variants]
    plan = L.build_run_order_plan(cells, block_key=lambda c: c[1], seed=7)
    rep = L.run_order_balance(plan, plan['planned_cells'])
    assert rep['status'] == 'ok' and rep['actual']['complete'] is True, rep
    arm_major = [f'{v}::seed{s}' for v in variants for s in seeds]
    rep = L.run_order_balance(plan, arm_major)
    assert rep['status'] == 'fail' and rep['arm_major'] is True, rep


def test_object_row_keys_reject_complex_per_hash_block():
    # Object-dtype keys used to flatten the WHOLE table just to find a complex
    # cell.  Check the rejection on a hash-block boundary, where a per-block
    # scan must still find a complex value in the first and in a later block.
    import data_guard as DG
    saved = DG._hash_row_block
    DG._hash_row_block = lambda width, cap: 1
    try:
        for arr in (np.array([[1 + 2j], [2]], dtype=object),
                    np.array([[1], [2 + 3j]], dtype=object)):
            try:
                DG._row_keys(arr)
            except TypeError:
                pass
            else:
                raise AssertionError(f'complex object rows accepted: {arr!r}')
        assert len(DG._row_keys(np.array([[1], [2]], dtype=object))) == 2
    finally:
        DG._hash_row_block = saved


def test_ladder_tier_discovery_is_numeric_and_complete():
    # The parameter audit used to iterate a literal S1-S4 tuple.  A newly
    # declared SCALE_LADDER tier must be discovered (and numeric order must
    # not turn S10 into the second tier).
    assert M0._ladder_tier_names({'S10': {}, 'S2': {}, 'X': {}}) == ['S2', 'S10', 'X']
    assert M0._ladder_tier_names({}) == []
    assert {'S1', 'S2', 'S3', 'S4'}.issubset(set(M0._ladder_tier_names()))


def test_run_order_is_blocked_randomized_and_position_balanced():
    # The frozen P5C lenphase used to run every seed of arm A before arm B,
    # which makes arm and calendar time perfectly confounded.  The design
    # helper must be deterministic, complete, seed-blocked and position
    # balanced without changing the (arm, seed) set.
    variants = ['csa_fixed_rope', 'full_rope', 'full_rope_sink', 'csa_fixed_rope_nosink']
    seeds = [0, 1, 2, 3]
    cells = [(v, s) for s in seeds for v in variants]
    order = L.blocked_run_order(cells, block_key=lambda c: c[1], seed=11)
    assert len(order) == len(cells) and set(order) == set(cells)
    assert order == L.blocked_run_order(cells, block_key=lambda c: c[1], seed=11), 'order must be deterministic'
    # Every seed block is contiguous and contains every arm exactly once.
    blocks = []
    for cell in order:
        if not blocks or blocks[-1][0] != cell[1]:
            blocks.append((cell[1], []))
        blocks[-1][1].append(cell[0])
    assert [b for b, _ in blocks] == seeds
    for _b, vs in blocks:
        assert sorted(vs) == sorted(variants), vs
    # Position balance: across the four blocks every arm occupies each slot once.
    for v in variants:
        slots = [vs.index(v) for _b, vs in blocks]
        assert sorted(slots) == list(range(len(blocks))), (v, slots)
    assert L.blocked_run_order(cells, block_key=lambda c: c[1], seed=12) != order, 'design seed must matter'
    # Malformed cells fail closed instead of silently training a duplicate.
    for bad in ([('a', 0), ('a', 0)], [{'variant': 'a', 'seed': 0}]):
        try:
            L.blocked_run_order(bad, block_key=lambda c: c[1] if isinstance(c, tuple) else 0)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f'bad run-order cells accepted: {bad!r}')
    try:
        L.blocked_run_order([('a', 0)], block_key=lambda c: c[1], seed='0')
    except TypeError:
        pass
    else:
        raise AssertionError('a string run-order seed must be rejected')
    plan = L.build_run_order_plan(cells, block_key=lambda c: c[1], seed=11)
    assert plan['kind'] == 'run_order' and plan['schema'] == L.RUN_ORDER_SCHEMA
    assert plan['design']['randomized'] is True
    assert len(plan['planned_cells']) == len(cells)
    with tempfile.TemporaryDirectory() as d:
        path = L.write_run_order(d, plan)
        with open(path, encoding='utf-8') as f:
            assert json.load(f)['planned_cells'] == plan['planned_cells']
    rep = L.run_order_balance(plan)
    assert rep['status'] == 'ok' and rep['design_ok'] is True, rep
    assert rep['complete_blocks'] == len(seeds) and rep['arm_major'] is False
    assert rep['position_balanced'] is True
    # A complete actual execution in the planned order is accepted.
    rep = L.run_order_balance(plan, plan['planned_cells'])
    assert rep['status'] == 'ok', rep
    # An incomplete/resumed panel is still describable.
    rep = L.run_order_balance(plan, plan['planned_cells'][:6])
    assert rep['status'] == 'ok' and rep['actual']['complete'] is False, rep
    # Arm-major design (one arm per block) must be a hard failure.
    arm_major = {
        'kind': 'run_order', 'schema': 1,
        'design': {'name': 'randomized_complete_block_balanced', 'randomized': True},
        'variants': variants,
        'blocks': {str(i): [v] for i, v in enumerate(variants)},
        'planned_cells': [], 'cell_blocks': {}, 'cell_variants': {},
    }
    rep = L.run_order_balance(arm_major)
    assert rep['status'] == 'fail' and rep['arm_major'] is True, rep
    # A missing variant in one block breaks the complete-block contract.
    broken = dict(plan)
    broken['blocks'] = dict(plan['blocks'])
    first = next(iter(broken['blocks']))
    broken['blocks'][first] = broken['blocks'][first][:-1]
    assert L.run_order_balance(broken)['status'] == 'fail'


def test_run_order_audit_detects_temporal_confounding():
    # The zero-GPU audit must (a) accept a blocked plan executed in blocks,
    # (b) fail a completed arm-major legacy run, and (c) say
    # not_verifiable_locally when no design artifact exists.
    variants = ['a', 'b', 'c', 'd']
    seeds = [0, 1, 2, 3]
    cells = [(v, s) for s in seeds for v in variants]
    plan = L.build_run_order_plan(cells, block_key=lambda c: c[1], seed=3)
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, 'run_order.json'), 'w', encoding='utf-8') as f:
            json.dump(plan, f)
        # Execute exactly the plan (serial start order).
        with open(os.path.join(d, 'experiments.jsonl'), 'w', encoding='utf-8') as f:
            for i, lab in enumerate(plan['planned_cells']):
                v, s = lab.rsplit('::seed', 1)
                f.write(json.dumps({'kind': 'cell', 'status': 'ok', 'variant': v,
                                    'seed': int(s),
                                    'started_at': f'2026-01-01T00:{i:02d}:00'}) + '\n')
        rep = A.audit_run_order([d])
        rec = rep['panels'][os.path.basename(d)]
        assert rep['status'] == 'ok' and rec['status'] == 'ok', rec
        assert rec['actual']['complete'] is True
        # A resumed/retrained cell appears twice; the latest execution wins
        # instead of being reported as a duplicate order.
        with open(os.path.join(d, 'experiments.jsonl'), 'a', encoding='utf-8') as f:
            lab = plan['planned_cells'][0]
            v, s = lab.rsplit('::seed', 1)
            f.write(json.dumps({'kind': 'cell', 'status': 'ok', 'variant': v,
                                'seed': int(s),
                                'started_at': '2026-01-01T00:59:00'}) + '\n')
        rep = A.audit_run_order([d])
        assert rep['panels'][os.path.basename(d)]['status'] == 'ok'
        # Legacy arm-major: all seeds of arm A, then all seeds of arm B.
        with open(os.path.join(d, 'experiments.jsonl'), 'w', encoding='utf-8') as f:
            i = 0
            for v in variants:
                for s in seeds:
                    f.write(json.dumps({'kind': 'cell', 'status': 'ok', 'variant': v,
                                        'seed': s,
                                        'started_at': f'2026-01-02T00:{i:02d}:00'}) + '\n')
                    i += 1
        rep = A.audit_run_order([d])
        rec = rep['panels'][os.path.basename(d)]
        assert rep['status'] == 'fail' and rec['status'] == 'fail', rec
        assert rec['arm_major'] is True
        assert any('arm-major' in p for p in rec['problems'])
        # No plan: the legacy order is classified but the design is not claimed.
        os.remove(os.path.join(d, 'run_order.json'))
        rep = A.audit_run_order([d])
        rec = rep['panels'][os.path.basename(d)]
        assert rec['status'] == 'fail' and rec['arm_major'] is True, rec
        # No plan and no events: explicitly unverifiable, never "ok".
        os.remove(os.path.join(d, 'experiments.jsonl'))
        rep = A.audit_run_order([d])
        assert rep['status'] == 'not_verifiable_locally'
        assert rep['panels'][os.path.basename(d)]['status'] == 'not_verifiable_locally'
        # A file that exists but is not a plan object fails closed.
        with open(os.path.join(d, 'run_order.json'), 'w', encoding='utf-8') as f:
            f.write('[]')
        rep = A.audit_run_order([d])
        assert rep['status'] == 'fail'
        assert rep['panels'][os.path.basename(d)]['status'] == 'fail'


def test_evaluation_order_plan_is_seed_blocked_and_covers_grid():
    # P5T used to evaluate every seed of arm A before arm B.  The final-test
    # order must be precommitted with the same complete-block/position-balance
    # contract as training, and it must cover the frozen (arm, seed) grid
    # exactly.
    variants = ['a', 'b', 'c', 'd']
    seeds = [0, 1, 2, 3]
    cells = [(v, s) for s in seeds for v in variants]
    plan = L.build_evaluation_order_plan(
        cells, task='clean_by_length', seed=9,
        created_at='2026-01-01T00:00:00')
    assert plan['kind'] == 'evaluation_order'
    assert plan['schema'] == L.EVALUATION_ORDER_SCHEMA
    assert plan['task'] == 'clean_by_length'
    assert len(plan['planned_cells']) == len(cells)
    assert all(lab.endswith(tuple(f'::s{s}' for s in seeds))
               for lab in plan['planned_cells'])
    # Every seed block is contiguous and contains every arm exactly once.
    blocks = []
    for lab in plan['planned_cells']:
        v, s = lab.rsplit('::s', 1)
        if not blocks or blocks[-1][0] != s:
            blocks.append((s, []))
        blocks[-1][1].append(v)
    assert [s for s, _vs in blocks] == [str(s) for s in seeds]
    for _s, vs in blocks:
        assert sorted(vs) == sorted(variants), vs
    # Position balance: every arm visits every within-block slot once.
    for v in variants:
        slots = [vs.index(v) for _s, vs in blocks]
        assert sorted(slots) == list(range(len(seeds))), (v, slots)
    assert plan['planned_cells'] == L.build_evaluation_order_plan(
        cells, task='clean_by_length', seed=9,
        created_at='2026-01-01T00:00:00')['planned_cells']
    assert plan['planned_cells'] != L.build_evaluation_order_plan(
        cells, task='clean_by_length', seed=10,
        created_at='2026-01-01T00:00:00')['planned_cells']
    ordered = L.evaluation_cells_in_plan_order(cells, plan)
    assert [f'{v}::s{s}' for v, s in ordered] == plan['planned_cells']
    runs = 1 + sum(a[0] != b[0] for a, b in zip(ordered, ordered[1:]))
    assert runs > len(variants), ordered
    for bad in (cells[:-1], cells + [('e', 0)]):
        try:
            L.evaluation_cells_in_plan_order(bad, plan)
        except ValueError:
            pass
        else:
            raise AssertionError('an evaluation plan with a different grid was accepted')
    for bad_task in ('', '   ', None):
        try:
            L.build_evaluation_order_plan(cells, task=bad_task, seed=9)
        except ValueError:
            pass
        else:
            raise AssertionError(f'bad evaluation task accepted: {bad_task!r}')
    try:
        L.build_evaluation_order_plan([('a', True)], task='x', seed=9)
    except TypeError:
        pass
    else:
        raise AssertionError('a boolean evaluation seed must be rejected')


def test_finaltest_evaluation_order_plan_is_precommitted_and_resume_stable():
    man = {'arms': ['a', 'b', 'c', 'd'], 'seeds': [0, 1, 2, 3],
           'manifest_sha256': 'ab' * 32}
    with tempfile.TemporaryDirectory() as d:
        p1 = V12._ensure_evaluation_order_plan(d, man, 'clean_by_length')
        path = os.path.join(d, 'clean_by_length_order.json')
        assert os.path.exists(path)
        with open(path, encoding='utf-8') as f:
            assert json.load(f)['planned_cells'] == p1['planned_cells']
        # Resume under the same manifest reuses the original plan/timestamp.
        p2 = V12._ensure_evaluation_order_plan(d, man, 'clean_by_length')
        assert p2['created_at'] == p1['created_at']
        assert p2['planned_cells'] == p1['planned_cells']
        # Before the test split is bound, a revised manifest may be planned.
        man2 = dict(man, arms=['a', 'b', 'c', 'd', 'e'])
        p3 = V12._ensure_evaluation_order_plan(d, man2, 'clean_by_length')
        assert set(p3['variants']) == set(man2['arms'])
        with open(os.path.join(d, '.freeze_lock.json'), 'w', encoding='utf-8') as f:
            json.dump({'stage': 'preaccess'}, f)
        man3 = dict(man, seeds=[0, 1, 2, 3, 4])
        p4 = V12._ensure_evaluation_order_plan(d, man3, 'clean_by_length')
        assert p4['seeds'] == [str(s) for s in man3['seeds']]
        # Once the one-shot slice is bound, a mismatching plan is a hard
        # refusal: reshuffling after measurement is not a precommitment.
        with open(os.path.join(d, '.freeze_lock.json'), 'w', encoding='utf-8') as f:
            json.dump({'stage': 'accessed', 'test_ids_fp': 'deadbeef'}, f)
        try:
            V12._ensure_evaluation_order_plan(d, man, 'clean_by_length')
        except SystemExit as e:
            assert 'refusing to reshuffle' in str(e), str(e)
        else:
            raise AssertionError('a post-access plan mismatch was accepted')
        # A missing plan after access cannot be reconstructed post hoc either.
        os.remove(path)
        try:
            V12._ensure_evaluation_order_plan(d, man3, 'clean_by_length')
        except SystemExit as e:
            assert 'plan' in str(e) and 'missing' in str(e), str(e)
        else:
            raise AssertionError('a post-access missing plan was recreated')
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, 'clean_by_length_order.json'), 'w', encoding='utf-8') as f:
            f.write('[]')
        try:
            V12._ensure_evaluation_order_plan(d, man, 'clean_by_length')
        except SystemExit:
            pass
        else:
            raise AssertionError('a non-object evaluation plan was accepted')


def test_evaluation_order_audit_detects_measurement_time_confounding():
    # The zero-GPU audit must (a) accept a plan executed in blocked order,
    # (b) fail arm-major execution, (c) fail a plan written after measurement,
    # (d) fail a clean measurement with no precommitted plan, and (e) mark a
    # plan without matching events as not verifiable.
    variants = ['a', 'b', 'c', 'd']
    seeds = [0, 1, 2, 3]
    cells = [(v, s) for s in seeds for v in variants]
    plan = L.build_evaluation_order_plan(
        cells, task='clean_by_length', seed=3,
        created_at='2026-01-01T00:00:00')
    probe = L.build_evaluation_order_plan(
        cells, task='distractor_probe', seed=4,
        created_at='2026-01-01T00:00:00')

    def _write_plan(d, name, obj):
        with open(os.path.join(d, name), 'w', encoding='utf-8') as f:
            json.dump(obj, f)

    def _write_events(d, kind, order, prefix, append=False):
        mode = 'a' if append else 'w'
        with open(os.path.join(d, 'experiments.jsonl'), mode, encoding='utf-8') as f:
            for i, lab in enumerate(order):
                v, s = lab.rsplit('::s', 1)
                f.write(json.dumps({'kind': kind, 'status': 'ok', 'variant': v,
                                    'seed': int(s),
                                    'started_at': f'{prefix}:{i:02d}'}) + '\n')

    with tempfile.TemporaryDirectory() as d:
        name = os.path.basename(d)
        _write_plan(d, 'clean_by_length_order.json', plan)
        _write_events(d, 'eval', plan['planned_cells'], '2026-01-02T00:00')
        rep = A.audit_evaluation_order([d])
        clean = rep['panels'][name]['tasks']['clean_by_length']
        assert rep['status'] == 'ok' and clean['status'] == 'ok', rep
        assert clean['actual']['complete'] is True
        assert clean['arm_major'] is False
        # Probe plan is optional, but when present it is audited with the
        # probe's own event kind and label convention.
        _write_plan(d, 'distractor_probe_order.json', probe)
        _write_events(d, 'cell', probe['planned_cells'], '2026-01-02T01:00',
                      append=True)
        rep = A.audit_evaluation_order([d])
        assert rep['panels'][name]['tasks']['distractor_probe']['status'] == 'ok'
        assert rep['panels'][name]['status'] == 'ok', rep
        # A completed arm-major clean run is a hard failure.
        arm_major = [f'{v}::s{s}' for v in variants for s in seeds]
        _write_events(d, 'eval', arm_major, '2026-01-02T02:00')
        rep = A.audit_evaluation_order([d])
        clean = rep['panels'][name]['tasks']['clean_by_length']
        assert rep['status'] == 'fail' and clean['arm_major'] is True, rep
        # A plan timestamped after the first measurement is not a precommitment.
        late = dict(plan, created_at='2026-01-03T00:00:00')
        _write_plan(d, 'clean_by_length_order.json', late)
        _write_events(d, 'eval', plan['planned_cells'], '2026-01-02T03:00')
        rep = A.audit_evaluation_order([d])
        assert rep['status'] == 'fail'
        assert any('not recorded before measurement' in p
                   for p in rep['panels'][name]['problems'])
        # Clean measurements without a plan fail closed; the probe's optional
        # missing plan is only a warning.
        os.remove(os.path.join(d, 'clean_by_length_order.json'))
        os.remove(os.path.join(d, 'distractor_probe_order.json'))
        rep = A.audit_evaluation_order([d])
        clean = rep['panels'][name]['tasks']['clean_by_length']
        assert rep['status'] == 'fail' and clean['status'] == 'fail', rep
        assert any('no precommitted evaluation-order plan' in p
                   for p in clean['problems'])
        # Plan present but no matching events: design verified, execution not.
        _write_plan(d, 'clean_by_length_order.json', plan)
        os.remove(os.path.join(d, 'experiments.jsonl'))
        rep = A.audit_evaluation_order([d])
        assert rep['status'] == 'not_verifiable_locally'
        assert rep['panels'][name]['tasks']['clean_by_length']['status'] == 'not_verifiable_locally'
        # A malformed plan file fails closed.
        _write_plan(d, 'clean_by_length_order.json', [])
        rep = A.audit_evaluation_order([d])
        assert rep['status'] == 'fail'


def test_target_encoder_groups_keep_duplicate_rows_together():
    # Same feature row in two different groups must not be split into
    # different folds: the twin's label would encode this row's target.
    import data_guard as DG
    X = np.array([['a'], ['a'], ['b'], ['b']], dtype=object)
    y = np.array([0.0, 100.0, 0.0, 0.0])
    groups = np.array(['g0', 'g1', 'g2', 'g3'])
    enc = DG.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2, seed=0)
    oof = enc.fit_transform(X, y, groups=groups)
    assert np.allclose(oof[0], oof[1]), oof
    assert np.allclose(oof[2], oof[3]), oof
    assert not np.allclose(oof[0], 100.0), oof
    # A persisted encoder without the order-insensitive row-set digest would
    # silently lose the reordered-training refusal.
    stats = enc.get_stats()
    stats.pop('train_row_set_fp', None)
    try:
        DG.LeakageSafeTargetEncoder.from_stats(stats)
    except ValueError:
        pass
    else:
        raise AssertionError('from_stats accepted a missing train_row_set_fp')


def test_minmax_overflow_stays_unit_interval():
    import data_guard as DG
    X = np.array([[-1.0e308], [0.0], [1.0e308]])
    enc = DG.LeakageSafePreprocessor(scale='minmax', clip_quantiles=None)
    out = enc.fit_transform(X)
    assert np.all(np.isfinite(out)), out
    assert np.allclose(out.ravel(), [0.0, 0.5, 1.0]), out
    restored = DG.LeakageSafePreprocessor.from_stats(enc.get_stats())
    assert np.allclose(restored.transform(X), out), (restored.transform(X), out)
    # Extended precision categories must not collapse through float64.
    a = DG._target_key(np.longdouble('1.0000000000000000001'))
    b = DG._target_key(np.longdouble('1.0000000000000000002'))
    assert a != b, (a, b)
    assert DG._target_key(np.longdouble(1.0)) == DG._target_key(1)
    # A finite value beyond float64's range is not the infinity category.
    _huge = np.longdouble('1e4000')
    if np.isfinite(_huge):
        assert DG._target_key(_huge) != DG._target_key(np.longdouble('inf'))


def test_selection_audit_requires_positive_seed_effect():
    A = __import__('audit_gates')
    td = tempfile.mkdtemp(prefix='selection_neg_', dir=REPO)
    try:
        path = os.path.join(td, 'selection.json')
        cell = {'arm': 'a', 'eval_len': 1024, 'k': 32, 'n_blocks': 256,
                'n_queries': 100, 'recall_at_k': 0.5, 'mass_covered': 0.9,
                'random_baseline': 0.1,
                'per_seed': {str(s): {'mass_covered': 0.1,
                                      'random_baseline': 0.9}
                             for s in range(6)}}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'cells': [cell]}, f)
        r = A.audit_selection(path=path)
        c = r['cells'][0]
        # Two-sided p can be significant in the negative direction; a positive
        # cell aggregate cannot certify it.
        assert c['seed_test']['p_exact_signflip'] < 0.05, c
        assert c['seed_test']['mean_delta'] < 0.0, c
        assert c['claim_allowed'] is False, c
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_ppl_below_one_is_not_a_measurement():
    assert L.ppl_is_usable(1.0) is True
    assert L.ppl_is_usable(0.5) is False
    assert L.ppl_is_usable(-1.0) is False
    nested = {'ppl': 100.0, 'ppl_estimation_rows': 'nested_validation_holdout',
              'ppl_holdout': -5.0}
    assert L._record_primary_ppl(nested) is None, nested
    nested['ppl_holdout'] = 0.5
    assert L._record_primary_ppl(nested) is None, nested
    nested['ppl_holdout'] = 10.0
    assert L._record_primary_ppl(nested) == 10.0
    assert L._pooled_ppl_from_records(
        [{'nll_sum': -100.0, 'n_tokens': 10}]) == (None, None, None)
    import build_report as BR
    assert BR._ppl_ok(0.5) is False and BR._ppl_ok(1.0) is True
    assert BR._primary_ppl(nested) == 10.0
    nested['ppl_holdout'] = 0.5
    assert BR._primary_ppl(nested) is None
    import fuse_analysis as FA
    assert math.isnan(FA.sample_std([])) and math.isnan(FA.sample_std([1.0]))
    assert np.isnan(FA.sample_std(np.ones((1, 2)), axis=0)).all()
    # A one-seed arm has no measured spread; the report must not print ±0.
    assert FA._fmt_sd(float('nan')) == '—'
    assert FA._mean_pm(1.0, float('nan')) == '1.0000±—'
    assert FA._mean_pm(1.0, 0.0) == '1.0000±0.0000'


def test_paired_error_guard_checks_identity_not_only_vocab_size():
    import error_analysis as EA
    EA._paired_vocab_guard({'vocab': 8192}, {'vocab': 8192})
    for field in ('train_fp', 'eval_fp', 'val_fp', 'tokenizer_vocab_sha256'):
        try:
            EA._paired_vocab_guard({'vocab': 8192, field: 'A'},
                                   {'vocab': 8192, field: 'B'})
        except SystemExit as e:
            assert field in str(e), str(e)
        else:
            raise AssertionError(f'differing {field} was accepted at equal vocab size')
        # One-sided legacy identity stays accepted; the guard only refuses a
        # definite mismatch.
        EA._paired_vocab_guard({'vocab': 8192, field: 'A'}, {'vocab': 8192})


def test_ce_loss_denominator_matches_scored_columns():
    torch.manual_seed(0)
    ids = torch.randint(0, 16, (2, 2))
    wide = torch.randint(0, 16, (2, 4))
    model = L.SmallGPT(16, 16, 1, 2, 8, 8, L.make_layer_cfgs(1, 'full'),
                       mlp_ratio=2)
    got = float(model.ce_loss(ids, wide, 16).detach())
    with torch.no_grad():
        lg = model.head(model.trunk(ids))
        want = float(torch.nn.functional.cross_entropy(
            lg.reshape(-1, 16), wide[:, :2].reshape(-1),
            reduction='mean'))
    assert math.isclose(got, want, rel_tol=1e-5, abs_tol=1e-6), (got, want)
    # The forward(targets=...) path must use the same denominator.
    got2 = float(model(ids, targets=wide, vocab=16).detach())
    assert math.isclose(got2, want, rel_tol=1e-5, abs_tol=1e-6), (got2, want)


def test_kernel_geometry_caches_are_not_inference_tensors():
    import csa_kernels as C
    with torch.inference_mode():
        geom = C._pool_geometry_cached([[2, 3]], 2, 5,
                                       torch.device('cpu'), 128)
        ar = C._full_arange_cached(1, 4, 3, torch.device('cpu'),
                                   contiguous=False)
    assert geom.last_tok.is_inference() is False
    assert ar.is_inference() is False
    # The lazy gather/byte caches must also survive an eval-first access.
    with torch.inference_mode():
        g_a, av = geom.gather_geom()
        b_off = geom.b_off(8)
    assert g_a.is_inference() is False and av.is_inference() is False
    assert b_off.is_inference() is False


def test_kernel_reference_fp64_stays_finite():
    import csa_kernels as C
    T, w = 2, 1
    q = torch.full((1, T, 1, 1), 1e20, dtype=torch.float64)
    kb = torch.empty((1, 0, 1, 1), dtype=torch.float64)
    vb = torch.empty((1, 0, 1, 1), dtype=torch.float64)
    ks = torch.full((1, T, 1, 1), 1e20, dtype=torch.float64)
    vs = torch.ones((1, T, 1, 1), dtype=torch.float64)
    idx = torch.empty((1, T, 0), dtype=torch.long)
    lt = torch.full((1, 0), -1, dtype=torch.long)
    out = C._attn_fwd_ref(q, kb, vb, ks, vs, idx, None, lt, None, None, w, 1.0)
    assert torch.isfinite(out).all(), out
    sink = torch.tensor([1e300], dtype=torch.float64)
    out2 = C._attn_fwd_ref(q, kb, vb, ks, vs, idx, None, lt, None, sink, w, 1.0)
    assert torch.isfinite(out2).all(), out2


def test_open_block_cell_receives_the_panel_deadline():
    # --max-hours was only checked before each cell; train_variant then ran
    # without a deadline and one cell could overrun the whole budget.
    import explib.open_block as OB
    with tempfile.TemporaryDirectory() as td:
        cfg = dict(OB.BASE_CFG)
        cfg.update(outdir=td, steps=1)
        seen = {}

        class _Stop(Exception):
            pass

        def _fake(*args, **kwargs):
            seen.update(kwargs)
            raise _Stop

        _orig = L.train_variant
        L.train_variant = _fake
        try:
            try:
                OB._run_one(
                    cfg, variant='v', seed=0,
                    train_ids=np.arange(8, dtype=np.int64),
                    val_batch=np.zeros((2, 4), dtype=np.int64),
                    vocab=97, val_bnd=None, outdir=td, fp='fp',
                    device=torch.device('cpu'), max_hours=1.0,
                    deadline_ts=123.0)
            except _Stop:
                pass
            else:
                raise AssertionError('the fake trainer should have stopped')
        finally:
            L.train_variant = _orig
        assert seen.get('deadline_ts') == 123.0, seen


def test_finaltest_zero_admission_is_not_a_completed_phase():
    # A P5T pass whose every cell returned skipped_* used to write a stats
    # file with admission.n_admitted=0 and still return normally, so run_full
    # appended P5T to `completed` after the one-shot test read was consumed.
    man = {'arms': ['a', 'b'], 'seeds': [0, 1]}
    assert 'admitted 0/' in (V12._finaltest_admission_problem(
        {'admission': {'n_admitted': 0}}, man) or '')
    assert V12._finaltest_admission_problem(
        {'admission': {'n_admitted': 4}}, man) is None
    assert V12._finaltest_admission_problem(
        {'admission': {'n_admitted': 1}}, man) is not None
    assert V12._finaltest_admission_problem(
        {'admission': {'n_admitted': 1}}, man, allow_partial=True) is None
    # Smoke fixtures are intentionally allowed to be smaller than the frozen
    # grid; the explicit smoke flag is the only exemption.
    assert V12._finaltest_admission_problem(
        {'admission': {'n_admitted': 0}}, man, smoke=True) is None
    # A malformed/missing admission report is not evidence of success.
    assert V12._finaltest_admission_problem({}, man) is not None
    assert V12._finaltest_admission_problem(
        {'admission': {'n_admitted': 'x'}}, man) is not None


def test_open_block_corrupt_summary_fails_closed():
    # Starting from {} after a JSON parse error silently replaced the whole
    # completed panel with the next single record.
    import explib.open_block as OB
    with tempfile.TemporaryDirectory() as td:
        assert OB._load_summary(td) == {}
        for payload in ('{"a": ', '[]', 'null'):
            with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
                f.write(payload)
            try:
                OB._load_summary(td)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f'corrupt summary {payload!r} was accepted')


def test_archive_manifest_covers_bytes_actually_written():
    # A file changing between the manifest hash and tf.add used to produce a
    # release that failed its own `verify`.  The reader hashes the bytes fed
    # to tar and the make path refuses a source that changed while packing.
    import archive_release as AR
    with tempfile.TemporaryDirectory(dir=REPO) as td:
        rel = os.path.basename(td)
        live = os.path.join(td, 'live.log')
        with open(live, 'wb') as f:
            f.write(b'x' * 4096)
        _orig = AR._HashingReader

        class _Mutating(_orig):
            def __init__(self, stream):
                super().__init__(stream)
                self._once = True

            def read(self, size=-1):
                data = super().read(size)
                if data and self._once:
                    self._once = False
                    with open(live, 'ab') as f:
                        f.write(b'changed')
                return data

        AR._HashingReader = _Mutating
        try:
            try:
                AR.cmd_make('race_test', [rel],
                            outdir=os.path.join(rel, 'archive'))
            except RuntimeError:
                pass
            else:
                raise AssertionError(
                    'archiving a file that changed while packed succeeded')
        finally:
            AR._HashingReader = _orig
        assert not os.path.exists(
            os.path.join(td, 'archive', 'release_race_test.tar.gz'))

