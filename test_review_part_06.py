"""CPU review checks (split group 06; t_holm_mixed_n_and_invalid_family_size .. t_par_watchdog_override_validation)."""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)

def test_holm_mixed_n_and_invalid_family_size():
    # A mixed-n family is capable of a significant result through its
    # best-powered member; the family floor must use max(n), not min(n).
    comps = {'A': {'p_exact_signflip': 0.02, 'n': 6},
             'B': {'p_exact_signflip': 0.0005, 'n': 12}}
    fam = L.attach_holm(comps, family='mixed n')
    assert fam['family_resolution_limited'] is False, fam
    assert abs(fam['family_min_attainable_p_holm'] - 0.0009765625) < 1e-15, fam
    # An omitted high-n row cannot carry the family: only tested rows count.
    om = {'A': {'p_exact_signflip': 0.02, 'n': 6},
          'B': {'omitted': 'no measurement', 'n': 12}}
    fam_om = L.attach_holm(om, family='omitted high n')
    assert fam_om['family_resolution_limited'] is True, fam_om
    # An unmeasurable p still occupies its declared family slot; dropping it
    # would shrink m and make the remaining p anti-conservatively small.
    h = L.holm_adjust([('a', 0.04), ('b', float('nan'))], alpha=0.05)
    assert abs(h['a']['p_holm'] - 0.08) < 1e-15, h
    assert h['a']['reject'] is False and h['a']['m_effective'] == 2, h
    assert h['b']['p_holm'] is None and h['b']['m_effective'] == 2, h
    # Fractional seed counts are not silently truncated into a different
    # resolution floor; they are unverifiable.
    assert math.isnan(L.signflip_p_floor(6.5))
    assert math.isnan(L.holm_min_attainable_p(6.5, 4))
    assert math.isnan(L.signflip_p_floor(True))


def test_meta_prediction_interval_df():
    pts = [{'label': f'S{i}', 'scale': i + 1,
            'effects': [0.1 + 0.02 * i, 0.2 + 0.02 * i, 0.3 + 0.02 * i],
            'seeds': [0, 1, 2]} for i in range(3)]
    m = L.meta_analyze_effects(pts)
    assert m['status'] == 'ok' and m['prediction_interval'] is not None, m
    lo, hi = m['prediction_interval']
    clo, chi = m['random_effects']['ci']
    # Prediction interval for a new scale is wider than the mean CI.
    assert lo < clo and hi > chi, m
    # Two usable scales cannot estimate both variance components.
    m2 = L.meta_analyze_effects(pts[:2])
    assert m2['status'] == 'ok' and m2['prediction_interval'] is None, m2
    # A scale point below the pre-registered seed floor is still reported, but
    # it must block the pooled "common effect" claim: pooling would silently
    # promote an underpowered scale to equal evidence.
    under = [{'label': 'U%d' % i, 'scale': i + 1,
              'effects': [0.1 + 0.01 * j for j in range(2)], 'seeds': [0, 1]}
             for i in range(3)]
    mu = L.meta_analyze_effects(under, min_seeds=3)
    assert mu['status'] == 'ok' and mu['poolable'] is False, mu
    assert mu['underpowered_points'] == ['U0', 'U1', 'U2'], mu
    assert 'underpowered' in mu['note'], mu


def test_tost_conf_controls_decision():
    d = [0.054382, 0.064971, 0.075123, 0.086877, 0.097029, 0.107618]
    r90 = L.tost_paired(d, margin=0.1, conf=0.90)
    assert r90['equivalent'] is True and r90['ci_within_margin'] is True, r90
    r95 = L.tost_paired(d, margin=0.1, conf=0.95)
    assert r95['equivalent'] is False, r95
    assert r95['ci_within_margin'] is False, r95
    assert abs(r95['alpha_one_sided'] - 0.025) < 1e-15, r95


def test_data_quality_exact_integer_and_complex_rows():
    # float64 has 53 bits of mantissa; distinct int64 rows above 2**53 must
    # not be reported as exact duplicates.
    big = np.array([[2 ** 53], [2 ** 53 + 1]], dtype=np.int64)
    assert L.duplicate_mask(big).tolist() == [False, False]
    rep = L.data_quality_report(big, train_idx=[0], val_idx=[1])
    assert rep['checks']['duplicate_rows_across_splits']['train__val'] == 0, rep
    cleaned = L.clean_training_rows(big, train_idx=[0, 1])
    assert cleaned['n_duplicate_rows_dropped'] == 0 and cleaned['n_train_rows_out'] == 2, cleaned
    # The same precision requirement applies to object columns of Python
    # integers (a common dtype after pandas/None handling): they must not be
    # rounded through float64 before duplicate detection.
    obj_big = np.array([[2 ** 53], [2 ** 53 + 1]], dtype=object)
    assert L.duplicate_mask(obj_big).tolist() == [False, False]
    obj_rep = L.data_quality_report(obj_big, train_idx=[0], val_idx=[1])
    assert obj_rep['checks']['duplicate_rows_across_splits']['train__val'] == 0, obj_rep
    # Mixed object int/float rows keep the integer's bits while still
    # comparing an integral float equal to the same integer.
    obj_mixed = np.array([[2 ** 53], [2 ** 53 + 2.0], [2 ** 53 + 2]], dtype=object)
    assert L.duplicate_mask(obj_mixed).tolist() == [False, False, True], obj_mixed
    assert L.duplicate_mask(np.array([[float('nan')], [None]], dtype=object)).tolist() == [False, True]
    # Complex rows were previously cast to float64 (dropping the imaginary
    # part); reject them instead of manufacturing duplicates.
    try:
        L.duplicate_mask(np.array([1 + 2j, 1 + 3j]))
    except TypeError:
        pass
    else:
        raise AssertionError('complex duplicate detection must be rejected')
    # Object/categorical string rows must be hashable without the numeric
    # fallback crashing, and a numeric spelling is not its numeric twin.
    cats = np.array([['a'], ['b'], ['a']], dtype=object)
    assert L.duplicate_mask(cats).tolist() == [False, False, True], cats
    mixed = np.array([[1], ['1'], [1.0]], dtype=object)
    assert L.duplicate_mask(mixed).tolist() == [False, False, True], mixed
    assert L.duplicate_mask(np.array([['1'], ['1.0']], dtype=object)).tolist() == [False, False]
    # Native unicode/bytes categorical dtypes (pandas factor columns, not
    # only dtype=object) must also be hashable, and remain type-distinct.
    assert L.duplicate_mask(np.array(['a', 'b', 'a'])).tolist() == [False, False, True]
    assert L.duplicate_mask(np.array([['a'], ['b'], ['a']])).tolist() == [False, False, True]
    assert L.duplicate_mask(np.array([b'a', b'b', b'a'])).tolist() == [False, False, True]
    assert L.duplicate_mask(np.array(['1', 1], dtype=object)).tolist() == [False, False]
    # Structured-array rows (common for grouped feature records) are also
    # well-defined categorical identities, not a float-cast crash.
    _rec = np.array([(1, 2), (1, 2), (3, 4)],
                    dtype=[('a', '<i4'), ('b', '<i4')])
    assert L.duplicate_mask(_rec).tolist() == [False, True, False]
    import data_guard as _DG
    assert _DG._matrix_fingerprint(_rec) == _DG._matrix_fingerprint(_rec.copy())


def test_exact_numeric_contracts_and_fingerprint():
    from decimal import Decimal
    import data_guard as DG
    # Fractional Decimal token counts must not be truncated into a different
    # pooled denominator; integral ones are ordinary counts.
    ok = L._nll_doc_ci([1.0, 2.0, 3.0],
                       [Decimal('10'), Decimal('20'), Decimal('30')])
    assert ok['n_docs'] == 3 and abs(ok['ppl'] - math.exp(0.1)) < 1e-12, ok
    bad = L._nll_doc_ci([1.0, 2.0, 3.0],
                        [Decimal('10'), Decimal('20.5'), Decimal('30')])
    assert bad['n_docs'] == 2 and abs(bad['ppl'] - math.exp(0.1)) < 1e-12, bad
    # Integer-only contracts accept Decimal('17') but reject Decimal('17.5')
    # instead of silently training on a different row count.
    sel, ev, meta = L.validation_row_split(Decimal('17'), seed=Decimal('3'))
    assert len(sel) + len(ev) == 17 and meta['seed'] == 3, meta
    try:
        L.validation_row_split(Decimal('17.5'))
    except ValueError:
        pass
    else:
        raise AssertionError('fractional n_rows must be rejected')
    try:
        L.tuning_row_split_spec({'selection_row_split': {'n_folds': Decimal('2.5')}})
    except ValueError:
        pass
    else:
        raise AssertionError('fractional n_folds must be rejected')
    assert L.tuning_row_split_spec({'selection_row_split': {'n_folds': Decimal('2')}})['n_folds'] == 2
    # A fractional vocab is not a tokenisation width.
    assert L._scale_points([{'scale': 1.0, 'effects': [0.1], 'seeds': [0],
                             'vocab': Decimal('3.5')}])[0]['vocab'] is None
    assert L._scale_points([{'scale': 1.0, 'effects': [0.1], 'seeds': [0],
                             'vocab': Decimal('3')}])[0]['vocab'] == 3
    try:
        L.simple_baseline_fit([0, 1, 0, 1], [0, 1], 2,
                              protocol=dict(L.SIMPLE_BASELINE_POLICY,
                                            max_fit_tokens=Decimal('2.5')))
    except ValueError:
        pass
    else:
        raise AssertionError('fractional max_fit_tokens must be rejected')
    # Matrix fingerprints keep exact integer bits above 2**53 while an
    # exactly representable value keeps the same digest as its float twin.
    big = np.array([[2 ** 53], [2 ** 53 + 1]], dtype=np.int64)
    assert len({DG._matrix_fingerprint(np.array([[v]], dtype=np.int64))
                for v in big.ravel().tolist()}) == 2
    assert DG._matrix_fingerprint(np.array([[2 ** 53]], dtype=np.int64)) == \
        DG._matrix_fingerprint(np.array([[float(2 ** 53)]]))
    assert DG._matrix_fingerprint(np.array([[0.0]])) == \
        DG._matrix_fingerprint(np.array([[-0.0]]))
    obj_big = np.array([[2 ** 53], [2 ** 53 + 1]], dtype=object)
    assert DG._matrix_fingerprint(obj_big[:1]) != DG._matrix_fingerprint(obj_big[1:])
    assert DG._matrix_fingerprint(obj_big) == DG._matrix_fingerprint(obj_big.copy())
    # The rounded float64 image of an inexact row can equal an exact row's
    # image (large int next to its 2**53 neighbour, or an uncoercible string
    # next to a missing value).  Swapping such rows must still change the
    # digest: the exact tail has to carry the row position, not just the key.
    _exact = np.array([[2 ** 53], [2 ** 53 + 1]], dtype=np.int64)
    _swapped = _exact[::-1].copy()
    assert DG._matrix_fingerprint(_exact) != DG._matrix_fingerprint(_swapped)
    _obj_exact = _exact.astype(object)
    assert DG._matrix_fingerprint(_obj_exact) != DG._matrix_fingerprint(_obj_exact[::-1].copy())
    _obj_mixed = np.array([['b'], [None]], dtype=object)
    assert DG._matrix_fingerprint(_obj_mixed) != DG._matrix_fingerprint(_obj_mixed[::-1].copy())
    # Floats wider than float64 must not collide after the historical
    # float64 hash; distinct longdouble values get distinct exact keys.
    if np.finfo(np.longdouble).eps < np.finfo(np.float64).eps:
        _ld1 = np.longdouble('1.0000000000000000001')
        _ld2 = np.longdouble('1.0000000000000000002')
        _ld = np.array([[_ld1], [_ld2]], dtype=np.longdouble)
        assert _ld1 != _ld2
        assert DG.duplicate_mask(_ld).tolist() == [False, False]
        assert DG._matrix_fingerprint(_ld[:1]) != DG._matrix_fingerprint(_ld[1:])
        _same = np.array([[_ld1], [_ld1]], dtype=np.longdouble)
        assert DG.duplicate_mask(_same).tolist() == [False, True]
        assert DG._matrix_fingerprint(_same[:1]) == DG._matrix_fingerprint(_same[1:])
        # A native longdouble table must fingerprint like its object twin:
        # exact rows need no tail, inexact/non-finite rows get the same
        # position-qualified key as every other path.
        assert DG._matrix_fingerprint(_ld) == DG._matrix_fingerprint(_ld.astype(object))
    # Exact rationals (Decimal/Fraction) are not numbers.Real; they must not be
    # rounded through float64 into false duplicates or false uniqueness.
    from fractions import Fraction
    assert L.duplicate_mask(np.array([[Decimal('1.5')], [1.5]],
                                     dtype=object)).tolist() == [False, True]
    assert L.duplicate_mask(np.array([[Decimal(2) ** 80], [2 ** 80]],
                                     dtype=object)).tolist() == [False, True]
    assert L.duplicate_mask(np.array([[Fraction(1, 3)], [1 / 3]],
                                     dtype=object)).tolist() == [False, False]
    # A signaling NaN is still a numeric NaN, not a new categorical value:
    # duplicate detection must group it with None/NaN rather than keying its
    # repr as ordinary data.
    _sn = np.array([[None], [float('nan')], [Decimal('NaN')],
                    [Decimal('sNaN')]], dtype=object)
    assert L.duplicate_mask(_sn).tolist() == [False, True, True, True], \
        L.duplicate_mask(_sn)
    # The matrix digest follows the same missing-like contract for object
    # NaN spellings (None vs Decimal('NaN') must not be two content classes).
    assert DG._matrix_fingerprint(np.array([[None]], dtype=object)) == \
        DG._matrix_fingerprint(np.array([[Decimal('NaN')]], dtype=object))
    # An exactly representable object value keeps the native float digest
    # (dtype-insensitive fingerprint), while an inexact integer still gets an
    # exact tail.
    assert DG._matrix_fingerprint(np.array([[2 ** 53 + 1]], dtype=np.int64)) == \
        DG._matrix_fingerprint(np.array([[2 ** 53 + 1]], dtype=object))
    # Native unicode/bytes categorical tables now use the same canonical
    # object-row keys (previously the float64 cast raised).
    _cats_u = np.array([['a'], ['b']])
    _cats_o = _cats_u.astype(object)
    assert DG._matrix_fingerprint(_cats_u) == DG._matrix_fingerprint(_cats_o)
    assert DG._matrix_fingerprint(_cats_u[:1]) != DG._matrix_fingerprint(_cats_u[1:])
    assert DG._matrix_fingerprint(np.array([[b'a'], [b'b']])) != \
        DG._matrix_fingerprint(np.array([['a'], ['b']]))
    assert DG._matrix_fingerprint(np.array([[1.5]])) == \
        DG._matrix_fingerprint(np.array([[Decimal('1.5')]], dtype=object))
    assert DG._matrix_fingerprint(np.array([[2 ** 53 + 1]], dtype=np.int64)) != \
        DG._matrix_fingerprint(np.array([[2 ** 53]], dtype=np.int64))
    # An exactly representable uint64 above int64's maximum must keep the same
    # digest as its object/float twin; the unsigned round-trip used to mark
    # every such value inexact merely because it did not fit int64.
    _u_exact = np.array([[2 ** 63], [2 ** 63 + 2048]], dtype=np.uint64)
    assert DG._matrix_fingerprint(_u_exact) == \
        DG._matrix_fingerprint(_u_exact.astype(object))
    assert DG._matrix_fingerprint(np.array([[2 ** 63]], dtype=np.uint64)) == \
        DG._matrix_fingerprint(np.array([[float(2 ** 63)]]))
    assert DG._matrix_fingerprint(np.array([[2 ** 63 + 1]], dtype=np.uint64)) != \
        DG._matrix_fingerprint(np.array([[2 ** 63]], dtype=np.uint64))
    # A corpus fingerprint is a function of token values, not storage width.
    _toks = np.array([0, 1, 2, 250, 255], dtype=np.int64)
    _fps = {L._ids_fp(_toks.astype(_dt)) for _dt in
            (np.uint8, np.uint16, np.int32, np.int64)}
    assert len(_fps) == 1, _fps
    # Booleans are flags, not measurements, in the scale helpers too.
    assert L.paired_log_ratio([True, 2.0], [2.0, 1.0])['n'] == 1
    assert L.loglog_regression([(True, 2), (2, 4), (4, 8)])['n_points'] == 2
    assert L.crossover_scaling_fit([{'d': True, 'crossover_tokens': 1,
                                     'train_tokens': 1}])['status'] == 'insufficient_points'
    assert math.isnan(L._scale_points([{'scale': True, 'effects': [1],
                                        'seeds': [0]}])[0]['scale'])
    # Decimal is a real numeric scalar even though it is not numbers.Real;
    # NaN/Infinity and complex values are still not measurements.
    assert L._finite_scalar(Decimal('1.5')) == 1.5
    assert L._finite_scalar(Decimal('2')) == 2.0
    assert L._finite_scalar(Decimal('NaN')) is None
    assert L._finite_scalar(1 + 2j) is None
    assert BR._as_finite(Decimal('2.5')) == 2.5
    # Missing-value detection also covers exact numeric objects.
    assert DG._category_is_missing(Decimal('NaN')) is True
    assert DG._category_is_missing(Decimal('sNaN')) is True
    assert DG._category_is_missing(Decimal('1.5')) is False
    assert DG._label_key(Decimal('NaN')) is None
    assert DG._label_key(Decimal('sNaN')) is None
    assert DG._label_key(Decimal('2')) == Decimal('2')
    assert DG._group_key(Decimal('NaN')) == ('__nan__',)
    assert DG._group_key(Decimal('sNaN')) == ('__nan__',)
    # Decimal('Infinity') equals float('inf') in Python; both must be one
    # category identity instead of falling through to a Decimal-specific repr.
    assert DG._target_key(Decimal('Infinity')) == DG._target_key(float('inf'))
    assert DG._target_key(Decimal('-Infinity')) == DG._target_key(float('-inf'))
    # The digest is a content identity, not a function of the row-block size.
    # Exact-row tails used to be appended immediately after their block, so
    # changing the memory cap interleaved them differently and silently
    # changed a fitted artifact's fingerprint.
    _cap = DG._HASH_BLOCK_BYTES
    _block_cases = [
        (np.arange(40, dtype=np.uint64) + 2 ** 63).reshape(10, 4),
        np.array([[2 ** 53 + i] for i in range(10)], dtype=np.int64),
        np.array([['a'], ['b'], [None], [2 ** 53 + 1]] * 3, dtype=object),
    ]
    if np.finfo(np.longdouble).eps < np.finfo(np.float64).eps:
        _block_cases.append(
            (np.arange(12, dtype=np.longdouble) / 7).reshape(6, 2))
    try:
        for _case in _block_cases:
            _fps = set()
            for _cap_bytes in (1, 7, 64, 1 << 20):
                DG._HASH_BLOCK_BYTES = _cap_bytes
                _fps.add(DG._matrix_fingerprint(_case))
            assert len(_fps) == 1, (_case.dtype, _fps)
    finally:
        DG._HASH_BLOCK_BYTES = _cap


def test_batch_iter_minimal_corpus():
    import torch as _torch
    it = L.batch_iter(np.arange(8, dtype=np.int64), 7, 1, _torch.device('cpu'), seed=0)
    x, y = next(it)
    assert x.shape == (1, 7) and y.shape == (1, 7), (x.shape, y.shape)
    assert x[0, 0].item() == 0 and y[0, -1].item() == 7, (x, y)
    try:
        next(L.batch_iter(np.arange(7, dtype=np.int64), 7, 1, _torch.device('cpu')))
    except ValueError:
        pass
    else:
        raise AssertionError('a corpus shorter than seq_len+1 must be rejected')
    for _seq_len, _batch in ((0, 1), (1, 0), (1, -1), (1.5, 1), (True, 1)):
        try:
            next(L.batch_iter(np.arange(10, dtype=np.int64), _seq_len, _batch,
                              _torch.device('cpu')))
        except ValueError:
            pass
        else:
            raise AssertionError(f'seq_len={_seq_len!r}, batch_size={_batch!r} must be refused')


def test_indexer_invalid_soft_lanes_are_zero():
    import torch as _torch
    n, B, nIH, hd, d = 2, 2, 2, 2, 4
    H = _torch.zeros(n, d)
    comp = _torch.zeros(B, d)
    W_DQ = _torch.zeros(d, nIH * hd)
    W_DK = _torch.zeros(d, nIH * hd)
    W_w = _torch.zeros(nIH, d)
    last_tok = _torch.tensor([-1, 0], dtype=_torch.long)
    box = []
    _m, _idx, soft = L.lightning_indexer(
        H, comp, last_tok, W_DQ, W_DK, W_w, nIH, 2,
        return_mask=False, out_valid=box, need_soft=True)
    valid = box[0]
    assert valid[0].tolist() == [True, False], valid
    assert float(soft[~valid].abs().max()) == 0.0, (soft, valid)


def test_attention_backward_partial_head_group():
    import torch as _torch
    import csa_kernels as _C
    old = os.environ.get('CSA_ATTN_GRAD_SLAB_MB')
    os.environ['CSA_ATTN_GRAD_SLAB_MB'] = '1'
    try:
        R, T, nh, hd = 1, 4, 5, 64
        Bm, nb, w = 1000, 2, 1
        q = _torch.randn(R, T, nh, hd, requires_grad=True)
        k_blk = _torch.randn(R, Bm, nh, hd, requires_grad=True)
        v_blk = _torch.randn(R, Bm, nh, hd, requires_grad=True)
        k_sw = _torch.randn(R, T, nh, hd, requires_grad=True)
        v_sw = _torch.randn(R, T, nh, hd, requires_grad=True)
        dO = _torch.randn(R, T, nh, hd)
        topk_idx = _torch.tensor([[[0, 1], [0, 1], [0, 1], [0, 1]]], dtype=_torch.long)
        last_tok = _torch.full((R, Bm), -1, dtype=_torch.long)
        out = _C._attn_bwd_ref(
            q, k_blk, v_blk, k_sw, v_sw, topk_idx, None, last_tok,
            None, None, w, float(hd) ** -0.5, dO,
            True, True, True, True, True, False, False)
        assert len(out) == 7 and out[1].shape == (R, Bm, nh, hd), out
    finally:
        if old is None:
            os.environ.pop('CSA_ATTN_GRAD_SLAB_MB', None)
        else:
            os.environ['CSA_ATTN_GRAD_SLAB_MB'] = old


def test_checkpoint_stamp_matches_the_actual_backend():
    class _M:
        def state_dict(self):
            return {}
    _old = L._kernel_stamp
    try:
        L._kernel_stamp = lambda: '-ktest'
        _p = L._checkpoint_payload(_M(), variant='v', seed=0, step=0,
                                   val_ppl=1.0, train_ce=1.0, health={})
        assert _p['code'] == L.code_stamp() == L.CODE_SEMANTICS + '-ktest', _p.get('code')
        assert _p['code'] != L.CKPT_CODE
    finally:
        L._kernel_stamp = _old


def test_indexer_chunked_final_tile_uses_active_rows():
    # The Bm>8192 fallback allocates a [R, T_c, Bm] score tile but the last
    # chunk may fill only tc < T_c rows.  The caller must pass the active rows
    # to the chunked top-k helper, or the [R,T_c,k] result cannot be assigned
    # into idx[:, t0:t0+tc] (and stale rows become fake queries).
    import types
    import csa_kernels as C
    R, T, nIH, hd, Bm, k = 1, 3, 1, 2, 8193, 2

    class _Launch:
        def __getitem__(self, _grid):
            return self
        def __call__(self, *args, **kwargs):
            return None

    fake = types.SimpleNamespace(
        triton=types.SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
        idx_score=_Launch())
    seen = []

    def _helper(scores, last_tok, t0, T_, Bm_, k_out, need_soft, out_dtype):
        seen.append((int(scores.shape[1]), int(t0)))
        _r, _tc = int(scores.shape[0]), int(scores.shape[1])
        return (torch.zeros((_r, _tc, k_out), dtype=torch.int32),
                torch.zeros((_r, _tc, k_out), dtype=torch.bool), None)

    _old = (C._kernels, C._IDX_SCORES_CAP_BYTES, C._indexer_topk_chunked)
    try:
        C._kernels = lambda: fake
        C._IDX_SCORES_CAP_BYTES = 2 * R * Bm * 4  # T_c = 2 for T = 3
        C._indexer_topk_chunked = _helper
        idx, valid, soft = C._indexer_fwd_triton(
            torch.randn(R, T, nIH, hd), torch.randn(R, Bm, nIH, hd),
            torch.randn(nIH, hd), torch.zeros(R, Bm, dtype=torch.long),
            k, need_soft=False)
    finally:
        C._kernels, C._IDX_SCORES_CAP_BYTES, C._indexer_topk_chunked = _old
    assert seen == [(2, 0), (1, 2)], seen
    assert idx.shape == (R, T, k) and valid.shape == (R, T, k), (idx.shape, valid.shape)


def test_stats_accept_numpy_scalars():
    # Public numeric helpers are routinely fed NumPy arrays/scalars.  An:
    # isinstance(int, float) filter would drop every np.int64 / np.float32
    # value, turning a real panel into "no data".
    d = np.array([1, 2, 3, 4], dtype=np.int64)
    r = L.paired_effect_stats(d)
    assert r['n'] == 4 and math.isfinite(r['mean']) and math.isfinite(r['p_exact_signflip']), r
    w = L.wilcoxon_signed_rank_p(d)
    assert w['n_nonzero'] == 4 and math.isfinite(w['p']), w
    t = L.tost_paired(d, margin=10.0)
    assert t['n'] == 4 and t['equivalent'] is True, t
    assert L.bootstrap_mean_ci(np.arange(10)) is not None
    assert L.paired_bootstrap_ci(np.arange(10)) is not None
    assert L.unpaired_bootstrap_ci(np.arange(10), np.arange(1, 11)) is not None
    lo, hi, dropped = L.finite_range([
        ('int', np.int64(1)), ('f32', np.float32(2.5)),
        ('nan', np.float64('nan')), ('bool', True), ('npbool', np.bool_(True))])
    assert (lo, hi, dropped) == (1.0, 2.5, ['nan', 'bool', 'npbool']), (lo, hi, dropped)
    assert BR.exact_signflip([np.bool_(True), np.bool_(False)]) is None
    assert L.exact_signflip_p([True, False])['n'] == 0
    assert BR.exact_signflip([True, False]) is None
    meta = L.meta_analyze_effects([
        {'label': 'A', 'scale': 1, 'effects': np.array([1, 2, 3], dtype=np.int64), 'seeds': [0, 1, 2]},
        {'label': 'B', 'scale': 2, 'effects': np.array([2, 3, 4], dtype=np.int64), 'seeds': [0, 1, 2]},
    ])
    assert meta['status'] == 'ok' and all(p['n'] == 3 for p in meta['per_point']), meta
    # The zero-GPU report builder has its own exact test copy; it must use the
    # same scalar contract instead of silently dropping NumPy values.
    for _d in (np.array([1, 2, 3, 4], dtype=np.int64),
               np.array([1, 2, 3, 4], dtype=np.float32),
               [np.int64(1), np.int64(2), np.int64(3), np.int64(4)]):
        _r = BR.exact_signflip(_d)
        assert _r is not None and _r['n'] == 4 and math.isfinite(_r['p_exact_signflip']), _r


def test_apply_rope_preserves_width():
    # Over-long rope_dim must rotate only complete pairs that exist and must
    # never widen the output.
    for width in (0, 1, 2, 3, 4, 5, 8, 16):
        for rope_dim in range(0, 2 * width + 4):
            x = torch.randn(2, 3, width, dtype=torch.float64)
            half = rope_dim // 2
            cos = torch.randn(1, 3, half, dtype=torch.float64)
            sin = torch.randn(1, 3, half, dtype=torch.float64)
            xg = x.clone().requires_grad_(True)
            y = V7.apply_rope(xg, cos, sin, rope_dim)
            assert tuple(y.shape) == tuple(x.shape), (width, rope_dim, y.shape)
            y.sum().backward()
            assert torch.isfinite(xg.grad).all(), (width, rope_dim)


def test_label_key_keeps_numeric_strings():
    # Only real numeric scalars are checked for NaN/Inf.  A categorical label
    # literally spelled "nan" or "inf" must remain a label, not disappear as a
    # corrupted target.
    X = np.arange(24, dtype=float).reshape(12, 2)
    y = ['nan', 'inf', 'a', 'b', 'nan', 'a', 'b', 'a', 'b', 'a', 'b', 'nan']
    rep = L.data_quality_report(X, y=y, train_idx=list(range(6)),
                                val_idx=list(range(6, 9)), test_idx=list(range(9, 12)))
    assert rep['checks']['missing_labels'] == 0 and rep['checks']['nonfinite_labels'] == 0, rep
    assert rep['checks']['label_distribution'].get('nan') == 3, rep
    # Actual NaN/Inf numeric targets are still non-finite.
    ynum = [0, 1, np.nan, np.inf, 2, 3, 4, 5, 6, 7, 8, 9]
    rep2 = L.data_quality_report(X, y=ynum, train_idx=list(range(6)),
                                 val_idx=list(range(6, 9)), test_idx=list(range(9, 12)))
    assert rep2['checks']['missing_labels'] == 0 and rep2['checks']['nonfinite_labels'] == 2, rep2


def test_locked_json_and_csv_generator():
    root = tempfile.mkdtemp(prefix='atomic_', dir=REPO)
    try:
        p = os.path.join(root, 'missing', 'sub', 'x.json')
        assert L.locked_json_read(p) == {}
        got = L.locked_json_update(p, lambda obj: (obj.update(k=1) or obj))
        assert got == {'k': 1} and L.locked_json_read(p) == {'k': 1}, got
        # A parsed list is not a JSON object: an updater would crash on
        # .get()/__setitem__, and overwriting it would destroy the operator's
        # file.  Refuse and leave the bytes untouched.
        bad = os.path.join(root, 'bad.json')
        with open(bad, 'w', encoding='utf-8') as f:
            f.write('[1, 2]')
        for _fn in (L.locked_json_read,
                    lambda _p: L.locked_json_update(_p, lambda obj: obj)):
            try:
                _fn(bad)
            except ValueError:
                pass
            else:
                raise AssertionError('a non-object locked JSON file must be refused')
        with open(bad, encoding='utf-8') as f:
            assert f.read().strip() == '[1, 2]'
        c = os.path.join(root, 'rows.csv')
        L.atomic_write_csv(c, None, ({'a': i, 'b': i * 2} for i in range(3)))
        with open(c, encoding='utf-8') as f:
            lines = f.read().splitlines()
        assert lines == ['a,b', '0,0', '1,2', '2,4'], lines
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_data_guard_preprocessor_option_guards():
    for bad in ({'missing': True}, {'missing': np.bool_(False)},
                {'eps': -1.0}, {'eps': float('nan')}, {'eps': float('inf')},
                {'eps': True}, {'clip_z': True}, {'clip_z': float('nan')},
                {'clip_quantiles': (True, 1.0)},
                {'clip_quantiles': (0.0, np.bool_(True))}):
        try:
            L.LeakageSafePreprocessor(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'preprocessor accepted invalid options {bad!r}')
    prep = L.LeakageSafePreprocessor(scale='standard', clip_quantiles=None,
                                     eps=0.0).fit(np.array([[1.0], [1.0]]))
    out = prep.transform(np.array([[1.0], [2.0]]))
    assert np.all(np.isfinite(out)), out
    assert prep.get_stats()['eps'] == 0.0
    assert L.LeakageSafePreprocessor(missing=np.float32(0.0)).missing == 0.0
    # clip_z was silently ignored outside standard/robust scaling.  It must
    # apply to minmax (where a holdout can still leave [0, 1]) and must be
    # refused for an unscaled feature space where "scaled units" do not exist.
    Xcl = np.array([[0.0], [1.0], [2.0], [100.0]])
    mm = L.LeakageSafePreprocessor(scale='minmax', clip_quantiles=None,
                                   clip_z=1.0).fit(Xcl[:3])
    assert mm.transform(np.array([[100.0]])).item() == 1.0
    try:
        L.LeakageSafePreprocessor(scale='none', clip_z=1.0)
    except ValueError as e:
        assert 'scaled feature space' in str(e), e
    else:
        raise AssertionError("clip_z with scale='none' must be refused")


def test_categorical_shift_requires_two_bins():
    for bad in (0, -1, 1, True):
        try:
            L.categorical_shift_report([1, 2], [1, 2], max_categories=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'max_categories={bad!r} must be rejected')
    r = L.categorical_shift_report([1, 1, 2], [1, 2], max_categories=2)
    assert r['n_reference'] == 3 and r['n_holdout'] == 2, r
    # Truncating the histogram to top categories must not make a category that
    # WAS in the reference look unseen.
    rt = L.categorical_shift_report([1, 1, 1, 2, 3], [2], max_categories=2)
    assert rt['unseen_rate'] == 0.0, rt


def test_data_quality_unhashable_labels():
    import data_guard as DG
    X = np.array([[0.0], [0.0], [1.0], [2.0]])
    y = np.empty(4, dtype=object)
    y[0], y[1], y[2], y[3] = ['a'], ['b'], ['c'], ['d']
    rep = L.data_quality_report(X, y=y, train_idx=[0, 2], val_idx=[1],
                                test_idx=[3])
    assert rep['checks']['label_conflicts_on_duplicate_rows'] == 1, rep
    assert rep['status'] == 'fail', rep
    # The unhashable-key sentinel must not collide with a real tuple label.
    # A list label and a tuple with the old sentinel spelling are different
    # classes; the duplicate-row conflict audit must still see the conflict.
    _sentinel_tuple = ('__unhashable__', 'list', "['x']")
    assert DG._hashable_label(['x']) != DG._hashable_label(_sentinel_tuple)
    assert DG._group_key(['x']) != DG._group_key(_sentinel_tuple)
    Xd = np.array([[0.0], [0.0]])
    yd = np.empty(2, dtype=object)
    yd[0], yd[1] = ['x'], _sentinel_tuple
    rd = L.data_quality_report(Xd, y=yd, train_idx=[0, 1])
    assert rd['checks']['label_conflicts_on_duplicate_rows'] == 1, rd


def test_categorical_encoder_train_only():
    train = np.array([['a', 'x'], ['b', None], ['a', np.nan]], dtype=object)
    holdout = np.array([['z', 'x'], ['a', 'y'], [None, 'z']], dtype=object)
    enc = L.LeakageSafeCategoricalEncoder()
    x_train = enc.fit_transform(train)
    stats = enc.get_stats()
    x_hold = enc.transform(holdout)
    assert np.array_equal(enc.transform(holdout), x_hold)
    assert enc.get_stats() == stats, 'transform must not add holdout categories'
    assert stats['features'][0]['categories'] == ['a', 'b'], stats
    assert stats['features'][1]['categories'] == ['x'], stats
    assert x_train[:, 0].tolist() == [0.0, 1.0, 0.0], x_train
    assert x_hold[:, 0].tolist() == [2.0, 0.0, 3.0], x_hold
    assert x_hold[:, 1].tolist() == [0.0, 1.0, 1.0], x_hold
    restored = L.LeakageSafeCategoricalEncoder.from_stats(stats)
    assert np.array_equal(restored.transform(holdout), x_hold)
    bundle = L.prepare_categorical_features(train, holdout)
    assert np.array_equal(bundle['train'], x_train), bundle
    assert np.array_equal(bundle['val'], x_hold), bundle
    assert bundle['encoder'].get_stats() == stats, bundle
    # A tuple level must survive JSON persistence: json turns it into a list,
    # so the catalog stores a canonical key rather than rebuilding identity
    # from the round-tripped value (which would make the level unseen).
    _tc = np.array([[('a', 'b')], [('a,s:b',)], [('a', 'b')]], dtype=object)
    _tc_h = np.array([[('a', 'b')], [('a,s:b',)]], dtype=object)
    for _enc_kw in ({'encoding': 'ordinal'}, {'encoding': 'onehot'}):
        _tce = L.LeakageSafeCategoricalEncoder(**_enc_kw).fit(_tc)
        _tcr = L.LeakageSafeCategoricalEncoder.from_stats(
            json.loads(json.dumps(_tce.get_stats())))
        assert np.array_equal(_tcr.transform(_tc_h), _tce.transform(_tc_h)), _enc_kw
    # Non-JSON-native levels (bytes, datetime64) must not make get_stats
    # unserializable; their canonical keys still drive transform after reload.
    _tx = np.array([[b'a'], [b'a'], [np.datetime64('2020-01-01')],
                    [np.datetime64('2020-01-01')]], dtype=object)
    _tx_h = np.array([[b'a'], [np.datetime64('2020-01-01')]], dtype=object)
    for _enc_kw in ({'encoding': 'ordinal'}, {'encoding': 'onehot'}):
        _txe = L.LeakageSafeCategoricalEncoder(**_enc_kw).fit(_tx)
        _txr = L.LeakageSafeCategoricalEncoder.from_stats(
            json.loads(json.dumps(_txe.get_stats())))
        assert np.array_equal(_txr.transform(_tx_h), _txe.transform(_tx_h)), _enc_kw
    onehot = L.LeakageSafeCategoricalEncoder(encoding='onehot').fit(train)
    oh_hold = onehot.transform(holdout)
    assert oh_hold.shape == (3, 4 + 3), oh_hold.shape
    assert np.allclose(oh_hold[:, :4].sum(axis=1), 1.0), oh_hold
    assert np.allclose(oh_hold[:, 4:].sum(axis=1), 1.0), oh_hold
    assert len(onehot.get_feature_names()) == oh_hold.shape[1]
    strict = L.LeakageSafeCategoricalEncoder(handle_unknown='error').fit(train)
    try:
        strict.transform(holdout)
    except ValueError:
        pass
    else:
        raise AssertionError('handle_unknown="error" must reject a new level')
    try:
        L.LeakageSafeCategoricalEncoder().transform(train)
    except RuntimeError:
        pass
    else:
        raise AssertionError('transform before fit must fail')
    try:
        enc.transform(np.zeros((1, 3), dtype=object))
    except ValueError:
        pass
    else:
        raise AssertionError('feature-width mismatch must fail')


def test_frequency_encoder_train_only():
    train = np.array([['a', 'x'], ['b', None], ['a', np.nan], ['c', 'x']],
                     dtype=object)
    holdout = np.array([['z', 'x'], ['a', 'y'], [None, 'z']], dtype=object)
    enc = L.LeakageSafeFrequencyEncoder()
    xtr = enc.fit_transform(train)
    stats = enc.get_stats()
    # Counts are learned from training rows only; missing values are not a
    # category and do not inflate the denominator.
    assert stats['features'][0]['n_observed'] == 4, stats
    assert [e['count'] for e in stats['features'][0]['entries']] == [2, 1, 1], stats
    assert stats['features'][1]['n_observed'] == 2, stats
    assert stats['features'][1]['n_missing'] == 2, stats
    assert xtr[:, 0].tolist() == [0.5, 0.25, 0.5, 0.25], xtr
    assert xtr[:, 1].tolist() == [1.0, 0.0, 0.0, 1.0], xtr
    # An unseen/missing holdout category maps to the reserved zero level and
    # never changes the fitted frequency table.
    xh = enc.transform(holdout)
    assert xh.tolist() == [[0.0, 1.0], [0.5, 0.0], [0.0, 0.0]], xh
    assert enc.get_stats() == stats, 'transform must not update frequencies'
    # Raw counts are available when normalisation is disabled.
    raw = L.LeakageSafeFrequencyEncoder(normalize=False).fit(train)
    assert raw.transform(train)[:, 0].tolist() == [2.0, 1.0, 2.0, 1.0]
    restored = L.LeakageSafeFrequencyEncoder.from_stats(stats)
    assert np.array_equal(restored.transform(holdout), xh)
    # Tuple categories are JSON-serialized as lists; the persisted canonical
    # key must keep them as the SAME learned level after a round-trip.
    _tf = np.array([[('a', 'b')], [('a,s:b',)], [('a', 'b')]], dtype=object)
    _tf_h = np.array([[('a', 'b')], [('a,s:b',)], ['z']], dtype=object)
    _tfe = L.LeakageSafeFrequencyEncoder().fit(_tf)
    _tfr = L.LeakageSafeFrequencyEncoder.from_stats(
        json.loads(json.dumps(_tfe.get_stats())))
    assert np.array_equal(_tfr.transform(_tf_h), _tfe.transform(_tf_h)), _tfe.get_stats()
    _tfx = np.array([[b'a'], [b'a'], [np.datetime64('2020-01-01')]], dtype=object)
    _tfx_h = np.array([[b'a'], [np.datetime64('2020-01-01')]], dtype=object)
    _tfxe = L.LeakageSafeFrequencyEncoder().fit(_tfx)
    _tfxr = L.LeakageSafeFrequencyEncoder.from_stats(
        json.loads(json.dumps(_tfxe.get_stats())))
    assert np.array_equal(_tfxr.transform(_tfx_h), _tfxe.transform(_tfx_h)), _tfxe.get_stats()
    bundle = L.prepare_frequency_features(train, holdout)
    assert np.array_equal(bundle['train'], xtr), bundle
    assert np.array_equal(bundle['val'], xh), bundle
    assert bundle['encoder'].get_stats() == stats, bundle
    assert enc.get_feature_names() == ['x0_freq', 'x1_freq']
    strict = L.LeakageSafeFrequencyEncoder(handle_unknown='error').fit(train)
    try:
        strict.transform(holdout)
    except ValueError:
        pass
    else:
        raise AssertionError('handle_unknown="error" must reject a new level')
    # Missing values are not a learned level either; the strict contract must
    # not silently encode them as zero (only the default contract does that).
    _strict_missing = L.LeakageSafeFrequencyEncoder(handle_unknown='error').fit(
        np.array([['a'], [None]], dtype=object))
    try:
        _strict_missing.transform(np.array([[None]], dtype=object))
    except ValueError:
        pass
    else:
        raise AssertionError('handle_unknown="error" must reject missing values')
    try:
        L.LeakageSafeFrequencyEncoder().transform(train)
    except RuntimeError:
        pass
    else:
        raise AssertionError('transform before fit must fail')
    try:
        enc.transform(np.zeros((1, 3), dtype=object))
    except ValueError:
        pass
    else:
        raise AssertionError('feature-width mismatch must fail')
    try:
        L.LeakageSafeFrequencyEncoder().fit(np.empty((0, 2), dtype=object))
    except ValueError:
        pass
    else:
        raise AssertionError('fitting on zero rows must fail')
    for bad in ({'normalize': 'yes'}, {'normalize': 1},
                {'handle_unknown': 'warn'}):
        try:
            L.LeakageSafeFrequencyEncoder(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'frequency encoder accepted {bad!r}')


def test_target_encoder_prevents_self_and_group_leakage():
    X = np.array([['a'], ['b'], ['a'], ['b'], ['c'], ['a']], dtype=object)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    enc = L.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2)
    oof = enc.fit_transform(X, y)
    stats = enc.get_stats()
    assert stats['oof_policy'] == 'row_oof', stats
    # A training row's own label must not enter its out-of-fold encoding.
    for i in range(len(y)):
        y2 = y.copy()
        y2[i] += 1000.0
        other = L.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2)
        assert np.allclose(other.fit_transform(X, y2)[i], oof[i]), i
    # Holdout transform never reads labels (there is no y argument) and unseen
    # levels fall back to the training prior rather than becoming a new level.
    hold = np.array([['a'], ['z'], [None]], dtype=object)
    before = enc.get_stats()
    xh = enc.transform(hold)
    assert enc.get_stats() == before, 'transform must not update statistics'
    assert np.allclose(xh[1], enc.prior_), (xh, enc.prior_)
    assert np.allclose(enc.transform(hold), xh)
    restored = L.LeakageSafeTargetEncoder.from_stats(before)
    assert np.allclose(restored.transform(hold), xh)
    # Applying the full-table lookup to the training rows would encode each
    # row with its own label; refuse unless the caller explicitly asks for a
    # diagnostic in-sample transform.
    for _enc_obj in (enc, restored):
        try:
            _enc_obj.transform(X)
        except ValueError as e:
            assert 'exact training rows' in str(e), e
        else:
            raise AssertionError('in-sample target encoding must be refused')
        assert np.allclose(_enc_obj.transform(X, allow_train=True),
                           _enc_obj.transform(X, allow_train=True))
        # Reordering the same training rows must not bypass the refusal: the
        # reordered matrix still contains every row's own label.
        try:
            _enc_obj.transform(X[::-1])
        except ValueError as e:
            assert 'possibly reordered' in str(e), e
        else:
            raise AssertionError(
                'a reordered training matrix must still be refused')
    # Missing categories are a learned training level, not silently a prior.
    Xm = np.array([['a'], [None], ['a'], [None], ['b'], ['a']], dtype=object)
    em = L.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2).fit(Xm, y)
    assert 'x0_target' in em.get_feature_names()
    # Persisted stats must be JSON-safe even when missing values are a learned
    # level, and the round-trip must encode holdouts identically.  A tuple
    # lookup key used to make json.dumps fail outright.
    _h = np.array([[None], ['a'], ['z']], dtype=object)
    _stats = em.get_stats()
    _restored = L.LeakageSafeTargetEncoder.from_stats(json.loads(json.dumps(_stats)))
    assert np.allclose(_restored.transform(_h), em.transform(_h))
    # A real category equal to the old tuple sentinel is not a missing value.
    _Xt = np.array([['a'], [('__missing__',)], [('__missing__',)], ['a']],
                   dtype=object)
    _et = L.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2).fit(_Xt, y[:4])
    _look = _et.get_stats()['features'][0]['lookup']
    assert '__missing__' not in _look, _look
    # Tuple keys must be self-delimiting: ('a', 'b') and ('a,s:b',) are
    # different categories even though the naive comma-join renders them the
    # same.  A collision would share one target mean across both levels.
    import data_guard as _DG
    assert _DG._target_key(('a', 'b')) != _DG._target_key(('a,s:b',))
    _Xtuple = np.array([[('a', 'b')], [('a,s:b',)], [('a', 'b')], [('a,s:b',)]],
                       dtype=object)
    _etuple = L.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2).fit(
        _Xtuple, np.array([1.0, 100.0, 1.0, 100.0]))
    _tlook = _etuple.get_stats()['features'][0]['lookup']
    assert len(_tlook) == 2, _tlook
    _trestored = L.LeakageSafeTargetEncoder.from_stats(
        json.loads(json.dumps(_etuple.get_stats())))
    _thold = np.array([[('a', 'b')], [('a,s:b',)]], dtype=object)
    assert np.allclose(_trestored.transform(_thold), _etuple.transform(_thold))
    # Groups share a fold, so changing a whole group's label cannot move the
    # out-of-fold encoding of another row in that group.
    groups = np.array([0, 0, 1, 1, 2, 2])
    ge = L.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2)
    g_oof = ge.fit_transform(X, y, groups=groups)
    assert ge.get_stats()['oof_policy'] == 'grouped_oof', ge.get_stats()
    yg = y.copy()
    yg[groups == 0] += 1000.0
    ge2 = L.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2)
    g_oof2 = ge2.fit_transform(X, yg, groups=groups)
    assert np.allclose(g_oof2[groups == 0], g_oof[groups == 0]), (g_oof, g_oof2)
    # Time folds are expanding: changing a later label must not change an
    # earlier row's encoding.
    t = np.arange(len(y))
    te = L.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2)
    t_oof = te.fit_transform(X, y, time=t)
    assert te.get_stats()['oof_policy'] == 'chronological_expanding_oof', te.get_stats()
    yt = y.copy()
    yt[4] += 1000.0
    te2 = L.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2)
    t_oof2 = te2.fit_transform(X, yt, time=t)
    assert np.allclose(t_oof2[:3], t_oof[:3]), (t_oof, t_oof2)
    # With both groups and time, folds stay whole-group AND expanding: changing
    # a later group's label cannot change an earlier group's encoding.
    ct = L.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2)
    c_oof = ct.fit_transform(X, y, groups=groups, time=np.arange(len(y)))
    assert ct.get_stats()['oof_policy'] == 'grouped_chronological_expanding_oof', ct.get_stats()
    yct = y.copy()
    yct[groups == 2] += 1000.0
    ct2 = L.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2)
    c_oof2 = ct2.fit_transform(X, yct, groups=groups, time=np.arange(len(y)))
    assert np.allclose(c_oof2[groups < 2], c_oof[groups < 2]), (c_oof, c_oof2)
    # Overlapping group time windows cannot be ordered into expanding folds
    # without leaking a later timestamp from an earlier group; refuse them.
    try:
        L.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2).fit_transform(
            X, y, groups=np.array([0, 0, 1, 1, 2, 2]),
            time=np.array([0, 10, 5, 6, 20, 21]))
    except ValueError as e:
        assert 'overlapping group time windows' in str(e), e
    else:
        raise AssertionError('overlapping group time windows must be refused')
    bundle = L.prepare_target_features(X, y, hold, groups=groups,
                                       smoothing=0.0, n_folds=2)
    assert np.array_equal(bundle['train'], g_oof), bundle
    assert np.array_equal(bundle['val'], ge.transform(hold)), bundle
    # Non-numeric targets, wrong shapes and malformed options fail closed.
    try:
        L.LeakageSafeTargetEncoder().fit_transform(X, ['a'] * len(X))
    except TypeError:
        pass
    else:
        raise AssertionError('a categorical target must be refused')
    try:
        L.LeakageSafeTargetEncoder().fit_transform(X, y.reshape(-1, 1))
    except ValueError:
        pass
    else:
        raise AssertionError('a 2-D target must be refused')
    try:
        L.LeakageSafeTargetEncoder().fit_transform(X[:-1], y)
    except ValueError:
        pass
    else:
        raise AssertionError('a short target must be refused')
    try:
        L.LeakageSafeTargetEncoder().transform(X)
    except RuntimeError:
        pass
    else:
        raise AssertionError('transform before fit must fail')
    for bad in ({'smoothing': -1}, {'smoothing': float('nan')},
                {'smoothing': True}, {'n_folds': 1}, {'n_folds': 2.5},
                {'n_folds': True}, {'seed': -1}):
        try:
            L.LeakageSafeTargetEncoder(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'target encoder accepted {bad!r}')
    # With one group, out-of-fold encoding would have no reference rows.
    try:
        L.LeakageSafeTargetEncoder().fit_transform(X, y, groups=np.zeros(len(y)))
    except ValueError:
        pass
    else:
        raise AssertionError('one group cannot define out-of-fold folds')
    # A single training row has no out-of-fold reference at all; emitting an
    # all-zero target feature would pretend the encoding was estimated.
    try:
        L.LeakageSafeTargetEncoder().fit_transform(
            np.array([['a']], dtype=object), np.array([1.0]))
    except ValueError as e:
        assert 'at least 2 training rows' in str(e), e
    else:
        raise AssertionError('one row cannot define an out-of-fold encoding')
    # fit (full-table transform for holdouts) is still valid with one row.
    # Re-encoding that lone training row is an explicit in-sample diagnostic,
    # so it must go through allow_train just like any other training matrix.
    _one = L.LeakageSafeTargetEncoder().fit(
        np.array([['a']], dtype=object), np.array([1.0]))
    assert np.allclose(
        _one.transform(np.array([['a']], dtype=object), allow_train=True), 1.0)


def test_clean_training_rows_and_split():
    X = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0],
                  [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    y = np.array([0, 1, 2, 2, 3, 4])
    clean = L.clean_training_rows(X, y, train_idx=[0, 1, 2, 3])
    assert clean['train_idx'].tolist() == [2], clean
    assert sorted(clean['dropped_idx'].tolist()) == [0, 1, 3], clean
    assert clean['n_label_conflict_clusters'] == 1, clean
    assert clean['n_duplicate_rows_dropped'] == 1, clean
    assert clean['n_label_conflict_rows_dropped'] == 2, clean
    keep = L.clean_training_rows(X, y, train_idx=[0, 1, 2, 3],
                                 label_conflicts='keep')
    assert keep['train_idx'].tolist() == [0, 1, 2], keep
    nodrop = L.clean_training_rows(X, y, train_idx=[0, 1, 2, 3],
                                   drop_duplicates=False,
                                   label_conflicts='drop')
    assert nodrop['train_idx'].tolist() == [2, 3], nodrop
    try:
        L.clean_training_rows(X, y, train_idx=[0, 1, 2, 3],
                              label_conflicts='raise')
    except ValueError:
        pass
    else:
        raise AssertionError('label_conflicts="raise" must refuse ambiguity')

    Xs = np.array([[0.0, 1.0], [2.0, 1.0], [np.nan, 1.0],
                   [0.0, 1.0], [1000.0, 1.0], [2000.0, 1.0], [3000.0, 1.0]])
    ys = np.array([0, 1, 1, 0, 1, 0, 1])
    out = L.split_and_prepare(Xs, ys, train_frac=4 / 7, val_frac=1 / 7,
                              test_frac=2 / 7, clean_train=True,
                              quality_report=True)
    idx = out['indices']
    assert idx['train'].tolist() == [0, 1, 2], idx
    assert out['cleaning']['dropped_idx'].tolist() == [3], out['cleaning']
    assert idx['val'].tolist() == [4] and idx['test'].tolist() == [5, 6], idx
    assert abs(out['preprocessor'].get_stats()['features'][0]['center'] - 1.0) < 1e-12
    assert out['quality']['n_train_reference'] == 3, out['quality']


def test_clean_training_invalid_labels():
    X = np.arange(8, dtype=float).reshape(4, 2)
    y = [0.0, np.nan, 2.0, np.inf]
    kept = L.clean_training_rows(X, y, train_idx=[0, 1, 2, 3])
    assert kept['train_idx'].tolist() == [0, 1, 2, 3], kept
    clean = L.clean_training_rows(X, y, train_idx=[0, 1, 2, 3],
                                  drop_invalid_labels=True)
    assert clean['train_idx'].tolist() == [0, 2], clean
    assert np.array_equal(np.sort(clean['dropped_idx']), [1, 3]), clean
    assert clean['n_invalid_label_rows_dropped'] == 2, clean
    # The one-call split helper must clean the same training rows and still
    # leave validation/test untouched.
    Xs = np.arange(12, dtype=float).reshape(6, 2)
    ys = [0.0, np.nan, 1.0, 1.0, 2.0, 3.0]
    out = L.split_and_prepare(Xs, ys, train_frac=3 / 6, val_frac=1 / 6,
                              test_frac=2 / 6, clean_train=True,
                              drop_invalid_labels=True)
    assert out['indices']['train'].tolist() == [0, 2], out['indices']
    assert out['indices']['val'].tolist() == [3], out['indices']
    assert out['indices']['test'].tolist() == [4, 5], out['indices']
    assert out['cleaning']['n_invalid_label_rows_dropped'] == 1, out['cleaning']


def test_clean_training_nonfinite_and_outliers():
    # Non-finite features and train-only outliers can be removed before any
    # preprocessing statistic is fitted.  Holdout rows are not passed to the
    # cleaner, so a large holdout value cannot change which train rows survive.
    X = np.array([[0.0], [1.0], [2.0], [1000.0],
                  [3.0], [4.0], [np.nan], [5.0]])
    clean = L.clean_training_rows(X, train_idx=[0, 1, 2, 3],
                                  drop_nonfinite_features=True,
                                  outlier_iqr=1.5)
    assert clean['train_idx'].tolist() == [0, 1, 2], clean
    assert clean['n_nonfinite_feature_rows_dropped'] == 0, clean
    assert clean['n_outlier_rows_dropped'] == 1, clean
    assert clean['dropped_idx'].tolist() == [3], clean
    # A heldout extreme does not enter the fence: only train row 4 is used as
    # the upper reference.  Row 6 (NaN) is in train and is dropped explicitly.
    clean2 = L.clean_training_rows(X, train_idx=[0, 1, 2, 4, 6],
                                   drop_nonfinite_features=True,
                                   outlier_iqr=1.5)
    assert 6 not in clean2['train_idx'].tolist(), clean2
    assert clean2['n_nonfinite_feature_rows_dropped'] == 1, clean2
    out = L.split_and_prepare(X, train_frac=0.5, val_frac=0.25,
                              test_frac=0.25, clean_train=True,
                              drop_nonfinite_features=True, outlier_iqr=1.5)
    assert out['cleaning']['n_outlier_rows_dropped'] == 1, out['cleaning']
    assert out['cleaning']['train_idx'].tolist() == [0, 1, 2], out['cleaning']
    # The NaN row is in the untouched test split, not in train, so it is not
    # counted as a dropped training row.
    assert out['cleaning']['n_nonfinite_feature_rows_dropped'] == 0, out['cleaning']
    for bad in (0, -1, float('nan'), float('inf'), True):
        try:
            L.clean_training_rows(X, train_idx=[0, 1], outlier_iqr=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'outlier_iqr={bad!r} must be refused')


def test_clean_training_target_outliers():
    # A corrupted regression label in the training split can be removed with a
    # fence learned from the training labels only.  The holdout's extreme label
    # must not move that fence or decide which training row survives.
    X = np.arange(16, dtype=float).reshape(8, 2)
    y = np.array([0.0, 1.0, 2.0, 100.0, 3.0, 4.0, 5.0, 1000.0])
    clean = L.clean_training_rows(X, y, train_idx=[0, 1, 2, 3, 4, 5, 6],
                                  target_outlier_iqr=1.5)
    assert clean['train_idx'].tolist() == [0, 1, 2, 4, 5, 6], clean
    assert clean['dropped_idx'].tolist() == [3], clean
    assert clean['n_target_outlier_rows_dropped'] == 1, clean
    assert clean['target_outlier_fence']['lo'] < 100.0, clean
    # The holdout label (row 7 = 1000) is outside the same fence but is never
    # passed to the cleaner, so it cannot cause a training drop.
    no_leak = L.clean_training_rows(
        X, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1000.0],
        train_idx=[0, 1, 2, 3, 4, 5, 6], target_outlier_iqr=1.5)
    assert no_leak['n_target_outlier_rows_dropped'] == 0, no_leak
    assert no_leak['train_idx'].tolist() == [0, 1, 2, 3, 4, 5, 6], no_leak
    # Categorical labels have no numeric target fence and must not be coerced.
    cats = L.clean_training_rows(
        X, ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
        train_idx=[0, 1, 2, 3, 4, 5, 6], target_outlier_iqr=1.5)
    assert cats['n_target_outlier_rows_dropped'] == 0, cats
    assert cats['target_outlier_fence'] is None, cats
    # A requested fence that cannot be estimated (categorical/multi-output
    # target, or too few finite labels) must be visible, not look like a
    # successful cleanup that found no outliers.
    assert cats['target_outlier_fence_estimated'] is False, cats
    assert cats['n_target_reference_labels'] == 0, cats
    multi = L.clean_training_rows(
        X, [(1, 2)] * 8, train_idx=[0, 1, 2, 3, 4, 5, 6],
        target_outlier_iqr=1.5)
    assert multi['target_outlier_fence_estimated'] is False, multi
    numeric = L.clean_training_rows(
        X, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1000.0],
        train_idx=[0, 1, 2, 3, 4, 5, 6], target_outlier_iqr=1.5)
    assert numeric['target_outlier_fence_estimated'] is True, numeric
    assert numeric['n_target_reference_labels'] == 7, numeric
    no_request = L.clean_training_rows(X, train_idx=[0, 1, 2])
    assert no_request['target_outlier_fence_estimated'] is None, no_request
    # The one-call split helper forwards the option and keeps holdouts intact.
    out = L.split_and_prepare(
        X, y, train_frac=5 / 8, val_frac=1 / 8, test_frac=2 / 8,
        clean_train=True, target_outlier_iqr=1.5)
    assert out['indices']['train'].tolist() == [0, 1, 2, 4], out['indices']
    assert out['indices']['val'].tolist() == [5], out['indices']
    assert out['indices']['test'].tolist() == [6, 7], out['indices']
    assert out['cleaning']['n_target_outlier_rows_dropped'] == 1, out['cleaning']
    # The option requires labels, and malformed multipliers are refused.
    try:
        L.clean_training_rows(X, train_idx=[0, 1, 2], target_outlier_iqr=1.5)
    except ValueError as e:
        assert 'requires y' in str(e), e
    else:
        raise AssertionError('a target fence without y must fail closed')
    for bad in (0, -1, float('nan'), float('inf'), True):
        try:
            L.clean_training_rows(X, y, train_idx=[0, 1], target_outlier_iqr=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'target_outlier_iqr={bad!r} must be refused')
    # A repeated training index would double-count drops and let the same
    # physical row appear twice in the cleaned split; refuse it.
    try:
        L.clean_training_rows(X, y, train_idx=[0, 1, 1])
    except ValueError as e:
        assert 'duplicate row indices' in str(e), e
    else:
        raise AssertionError('duplicate train_idx entries must be refused')


def test_known_labels_audit():
    X = np.arange(8, dtype=float).reshape(4, 2)
    y = [0, 1, 2, 9]
    rep = L.data_quality_report(X, y=y, train_idx=[0, 1], val_idx=[2],
                                test_idx=[3], known_labels=[0, 1, 2])
    assert rep['checks']['unknown_labels_by_split']['test'] == {'9': 1}, rep
    assert rep['status'] == 'warnings', rep
    hard = L.data_quality_report(
        X, y=y, train_idx=[0, 1], val_idx=[2], test_idx=[3],
        known_labels=[0, 1, 2], fail_on_unknown_labels=True)
    assert hard['status'] == 'fail', hard
    try:
        L.data_quality_report(X, y=y, train_idx=[0, 1], known_labels='012')
    except ValueError:
        pass
    else:
        raise AssertionError('a single string must not be read as a label set')
    try:
        L.data_quality_report(X, train_idx=[0, 1], known_labels=[0, 1])
    except ValueError:
        pass
    else:
        raise AssertionError('known_labels without y must fail closed')


def test_data_guard_nat_and_label_identity():
    # NaT timedelta in an object ordering column is still a missing timestamp:
    # it cannot define a temporal split or a monotone partition.
    _t = np.array([np.timedelta64(5, 'ns'), np.timedelta64('NaT', 'ns'),
                   np.timedelta64(1, 'ns')], dtype=object)
    for call in (lambda: L.split_indices(3, time=_t, train_frac=1 / 3,
                                         val_frac=1 / 3, test_frac=1 / 3),
                 lambda: L.assert_no_split_leakage([0], [1], [2], time=_t),
                 lambda: L.data_quality_report(
                     np.arange(6, dtype=float).reshape(3, 2), time=_t,
                     train_idx=[0], val_idx=[1], test_idx=[2])):
        try:
            call()
        except ValueError as e:
            assert 'NaT' in str(e), e
        else:
            raise AssertionError('an object-dtype NaT timedelta must be rejected')
    # A None ordering key is missing too; an all-None object column must not
    # silently become a valid chronology.
    _none_time = np.array([None, None, None], dtype=object)
    for call in (lambda: L.split_indices(3, time=_none_time, train_frac=1 / 3,
                                         val_frac=1 / 3, test_frac=1 / 3),
                 lambda: L.assert_no_split_leakage([0], [1], [2],
                                                  time=_none_time)):
        try:
            call()
        except ValueError as e:
            assert 'missing' in str(e), e
        else:
            raise AssertionError('a None ordering key must be rejected')
    # Categorical encoding: NaT is a missing value, not a category level.
    _cat = np.empty((3, 1), dtype=object)
    _cat[:, 0] = [np.timedelta64('NaT', 'ns'), np.timedelta64(1, 'ns'),
                  np.timedelta64(1, 'ns')]
    _enc = L.LeakageSafeCategoricalEncoder().fit(_cat)
    assert len(_enc.get_stats()['features'][0]['categories']) == 1, _enc.get_stats()
    # Missing NaT labels are reported as non-finite label noise.
    _rep = L.data_quality_report(
        np.arange(8, dtype=float).reshape(4, 2),
        y=np.array([0, 1, np.datetime64('NaT'), 2], dtype=object),
        train_idx=[0, 1], val_idx=[2], test_idx=[3])
    assert _rep['checks']['nonfinite_labels'] == 1, _rep
    # Numeric and string labels that merely share a spelling are different
    # classes; a complete shift between them must not be hidden by str().
    _rep2 = L.data_quality_report(
        np.arange(8, dtype=float).reshape(4, 2), y=[1, 1, '1', '1'],
        train_idx=[0, 1], val_idx=[2, 3])
    _shift = _rep2['checks']['label_distribution_shift_vs_train']['val']
    assert _shift['js_divergence'] > 0.5, _rep2


def test_native_temporal_row_keys_are_unit_canonical():
    # A native datetime/timedelta column stores an instant in a declared unit.
    # Hashing the raw dtype bytes keyed the SAME instant differently by unit,
    # so an exact duplicate held at a different resolution (e.g. a day-stamped
    # train cache and a second-stamped holdout) was reported as zero overlap.
    import data_guard as DG
    days = np.array([[np.datetime64('2020-01-01', 'D')],
                     [np.datetime64('2020-01-02', 'D')]])
    seconds = np.array([[np.datetime64('2020-01-01', 's')],
                        [np.datetime64('2020-01-03', 's')]])
    nanos = np.array([[np.datetime64('2020-01-01', 'ns')]])
    assert DG._row_keys(days)[0] == DG._row_keys(seconds)[0]
    assert DG._row_keys(days)[0] == DG._row_keys(nanos)[0]
    assert DG._row_keys(days)[0] != DG._row_keys(days)[1]
    # Native and object spellings of the same instant share one row key, just
    # as the object column path already canonicalised them.
    obj = np.array([[np.datetime64('2020-01-01', 's')]], dtype=object)
    assert DG._row_keys(days[:1])[0] == DG._row_keys(obj)[0]
    # Outside the int64-nanosecond range the exact unit scaling must still
    # collapse equal instants instead of falling back to dtype bytes.
    far_d = np.array([[np.datetime64('2500-01-01', 'D')]])
    far_s = np.array([[np.datetime64('2500-01-01', 's')]])
    assert DG._row_keys(far_d)[0] == DG._row_keys(far_s)[0]
    # Timedeltas use their own identity and never collide with datetimes.
    td_d = np.array([[np.timedelta64(1, 'D')]])
    td_s = np.array([[np.timedelta64(86400, 's')]])
    assert DG._row_keys(td_d)[0] == DG._row_keys(td_s)[0]
    assert DG._row_keys(td_d)[0] != DG._row_keys(days[:1])[0]
    # NaT is one missing-like identity per temporal kind, across units.
    nat_d = np.array([[np.datetime64('NaT', 'D')]])
    nat_s = np.array([[np.datetime64('NaT', 's')]])
    assert DG._row_keys(nat_d)[0] == DG._row_keys(nat_s)[0]
    assert DG._row_keys(nat_d)[0] != DG._row_keys(
        np.array([[np.timedelta64('NaT', 's')]]))[0]
    # The cross-array overlap audit is the caller that compares separately
    # stored tables, so exercise it end to end.
    import audit_gates as A
    assert A._row_overlap_count(days, seconds) == 1
    assert A._row_overlap_count(days, seconds[1:]) == 0


def test_object_temporal_duplicate_identity():
    # Adjacent nanosecond timestamps have the same float64 image.  An object
    # column must not collapse them into false duplicates (which would drop
    # distinct training rows or manufacture a cross-split leakage alarm).
    import data_guard as DG
    stamps = [np.datetime64('2020-01-01T00:00:00.000000001'),
              np.datetime64('2020-01-01T00:00:00.000000002'),
              np.datetime64('2020-01-01T00:00:00.000000003')]
    X = np.asarray(stamps, dtype=object).reshape(-1, 1)
    assert not DG.duplicate_mask(X).any(), DG.duplicate_mask(X)
    assert len({DG._matrix_fingerprint(X[i:i + 1]) for i in range(3)}) == 3
    # A true duplicate is still detected, and NaT is not conflated with None.
    dup = np.asarray([stamps[0], stamps[0], np.datetime64('NaT'),
                      None], dtype=object).reshape(-1, 1)
    assert DG.duplicate_mask(dup).tolist() == [False, True, False, False],         DG.duplicate_mask(dup)
    # The same instant expressed in different units keeps one identity.
    mixed = np.asarray([np.datetime64('2020-01-01T00:00:01', 's'),
                        np.datetime64('2020-01-01T00:00:01.000000000', 'ns')],
                       dtype=object).reshape(-1, 1)
    assert DG.duplicate_mask(mixed).tolist() == [False, True],         DG.duplicate_mask(mixed)
    # Timedelta values use their own exact identity (and do not collide with
    # the integer with the same value).
    deltas = np.asarray([np.timedelta64(1, 'ns'), np.timedelta64(2, 'ns'),
                         np.timedelta64(1, 'ns'), 1], dtype=object).reshape(-1, 1)
    assert DG.duplicate_mask(deltas).tolist() == [False, False, True, False],         DG.duplicate_mask(deltas)
    # Outside the int64-nanosecond range the key used to include the raw unit
    # and dtype, so the same instant written in two units looked distinct.
    far = np.asarray([np.datetime64('2500-01-01', 'D'),
                      np.datetime64('2500-01-01', 's'),
                      np.datetime64('2500-01-02', 'D')],
                     dtype=object).reshape(-1, 1)
    assert DG.duplicate_mask(far).tolist() == [False, True, False], DG.duplicate_mask(far)
    big = np.asarray([np.timedelta64(200000, 'D'),
                      np.timedelta64(200000 * 86400, 's')],
                     dtype=object).reshape(-1, 1)
    assert DG.duplicate_mask(big).tolist() == [False, True], DG.duplicate_mask(big)
    assert DG._target_key(np.datetime64('2500-01-01', 'D')) == \
        DG._target_key(np.datetime64('2500-01-01', 's'))
    # End-to-end: a clean chronological split of distinct timestamps must not
    # be refused as a duplicate-feature leak, while cleaning must keep them.
    clean_obj = np.empty((6, 1), dtype=object)
    clean_obj[:, 0] = stamps + [np.datetime64('2020-01-01T00:00:00.000000004'),
                                np.datetime64('2020-01-01T00:00:00.000000005'),
                                np.datetime64('2020-01-01T00:00:00.000000006')]
    out = DG.split_and_prepare(clean_obj, preprocess=False,
                               train_frac=0.5, val_frac=0.25, test_frac=0.25)
    assert out['cross_split_duplicates'] == {
        'train__val': 0, 'train__test': 0, 'val__test': 0}, out
    cleaned = DG.clean_training_rows(
        clean_obj, train_idx=np.arange(4), drop_duplicates=True)
    assert cleaned['n_duplicate_rows_dropped'] == 0, cleaned


def test_object_temporal_category_identity():
    # np.timedelta64 is a numbers.Integral, so the generic integral branch
    # either collided with the integer of the same raw count or raised for a
    # larger unit.  Target/category identity must use the exact temporal value.
    import data_guard as DG
    ns = np.timedelta64(1, 'ns')
    one_day = np.timedelta64(1, 'D')
    one_day_h = np.timedelta64(24, 'h')
    assert DG._target_key(ns) != DG._target_key(1), DG._target_key(ns)
    assert DG._target_key(one_day) == DG._target_key(one_day_h)
    assert DG._target_key(one_day) != DG._target_key(np.timedelta64(1, 'ns'))
    assert DG._target_key(np.datetime64('2020-01-01')) != \
        DG._target_key(np.datetime64('2020-01-01T00:00:00.000000001'))
    assert DG._target_key(np.timedelta64('NaT')) == '__missing__'
    # Outside the int64-nanosecond range NumPy WRAPS instead of returning
    # NaT, so a raw ns cast is not proof of identity.  The same instant in
    # D/Y/2Y units must share one exact key (and a 2Y unit must not be
    # multiplied a second time when it is converted to days).
    _far = [np.datetime64('3000-01-01'), np.datetime64('3000', 'Y'),
            np.datetime64('3000', '2Y')]
    assert len({DG._target_key(v) for v in _far}) == 1, _far
    assert len({DG._group_key(v) for v in _far}) == 1, _far
    assert DG._target_key(np.datetime64('3000', '2Y')) != \
        DG._target_key(np.datetime64('3002', '2Y'))
    # End-to-end: a timedelta category and the integer with the same raw count
    # are different levels and must not share one target mean.
    X = np.asarray([[ns], [1], [ns], [1]], dtype=object)
    y = np.asarray([10.0, 20.0, 10.0, 20.0])
    enc = DG.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2).fit(X, y)
    lookup = enc.get_stats()['features'][0]['lookup']
    assert lookup[DG._target_key(ns)] == 10.0, lookup
    assert lookup['i:1'] == 20.0, lookup
    # The persisted keys round-trip through JSON and still separate them.
    import json
    restored = DG.LeakageSafeTargetEncoder.from_stats(
        json.loads(json.dumps(enc.get_stats())))
    hold = np.asarray([[ns], [1]], dtype=object)
    assert np.allclose(restored.transform(hold), enc.transform(hold))
    # The same identity rule is used by the frequency encoder.
    freq = DG.LeakageSafeFrequencyEncoder().fit(X)
    f = freq.get_stats()['features'][0]['entries']
    assert len(f) == 2, f
    # A timedelta category is persisted by the categorical encoder without
    # int()-ing it (which raises for day-or-larger units) and still round-trips
    # to the same ordinal codes.
    cat_X = np.asarray([[one_day], [np.timedelta64(2, 'D')], [one_day],
                        [np.timedelta64(2, 'D')]], dtype=object)
    cat = DG.LeakageSafeCategoricalEncoder().fit(cat_X)
    cat_stats = json.loads(json.dumps(cat.get_stats()))
    cat_restored = DG.LeakageSafeCategoricalEncoder.from_stats(cat_stats)
    assert np.array_equal(cat.transform(cat_X),
                          cat_restored.transform(cat_X)), cat_stats
    # A timedelta is a numeric duration target, not a missing/unsupported
    # label.  NumPy's float(timedelta) works only for ns-like units, so the
    # raw fallback silently dropped day/hour labels; the encoding must use
    # one canonical duration and keep equal instants in different units equal.
    import datetime as _dt
    _td_X = np.array([['a'], ['a'], ['a']], dtype=object)
    _td_y = np.array([one_day, one_day_h, np.timedelta64(2, 'D')],
                     dtype=object)
    _td_enc = DG.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2).fit(
        _td_X, _td_y)
    _td_got = _td_enc.transform(np.array([['a']], dtype=object))[0, 0]
    assert math.isclose(_td_got, (4.0 / 3.0) * 86400e9), _td_got
    _py_y = np.array([_dt.timedelta(days=1), _dt.timedelta(days=2)],
                     dtype=object)
    _py_enc = DG.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2).fit(
        np.array([['a'], ['a']], dtype=object), _py_y)
    _py_got = _py_enc.transform(np.array([['a']], dtype=object))[0, 0]
    assert math.isclose(_py_got, 1.5 * 86400e9), _py_got


def test_persisted_stats_validate_feature_count():
    # transform() writes one output column per persisted feature stat.  A
    # truncated/corrupt artifact would otherwise leave columns uninitialised
    # (or write past the declared width), silently changing the feature table.
    import data_guard as DG
    import json
    X = np.arange(12.0).reshape(4, 3)
    prep = DG.LeakageSafePreprocessor().fit(X)
    valid = json.loads(json.dumps(prep.get_stats()))
    assert np.allclose(DG.LeakageSafePreprocessor.from_stats(valid).transform(X),
                       prep.transform(X))
    for mutate in (
            lambda s: s.update(features=[]),
            lambda s: s.update(features=s['features'][:2], n_features_in=3),
            lambda s: s.update(n_features_in=4),
            lambda s: s.update(n_features_in=True),
            lambda s: s['features'][0].update(factor=0.0),
            lambda s: s['features'][0].update(fill=float('nan'))):
        bad = json.loads(json.dumps(valid))
        mutate(bad)
        try:
            DG.LeakageSafePreprocessor.from_stats(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'malformed preprocessor stats accepted: {bad}')
    Xc = np.asarray([['a'], ['b']], dtype=object)
    for cls, fit_args in (
            (DG.LeakageSafeCategoricalEncoder, (Xc,)),
            (DG.LeakageSafeFrequencyEncoder, (Xc,)),
            (DG.LeakageSafeTargetEncoder, (Xc, np.asarray([1.0, 2.0])))):
        fitted = cls().fit(*fit_args)
        good = json.loads(json.dumps(fitted.get_stats()))
        restored = cls.from_stats(good)
        kw = {'allow_train': True} if cls is DG.LeakageSafeTargetEncoder else {}
        assert np.array_equal(restored.transform(Xc, **kw),
                              fitted.transform(Xc, **kw))
        for bad_stats in (
                dict(good, features=[]),
                dict(good, features=good['features'] * 2)):
            try:
                cls.from_stats(bad_stats)
            except ValueError:
                pass
            else:
                raise AssertionError(
                    f'{cls.__name__}.from_stats accepted a mismatched feature list')


def test_temporal_group_identity_matches_python_equality():
    # NumPy equality and hashing disagree for date-like ids (a datetime.date
    # equals some np.datetime64 values but hashes differently).  Returning the
    # raw objects as set keys let the same id land on both sides of a split
    # without the group-overlap check firing.
    import datetime
    import data_guard as DG
    same_day = [np.datetime64('2020-01-01'), datetime.date(2020, 1, 1)]
    assert DG._group_key(same_day[0]) == DG._group_key(same_day[1])
    # The canonical key is an opaque sentinel, so a real tuple group id that
    # spells the same components cannot impersonate it.
    forged = ('__temporal__', b'DI1577836800000000000')
    assert DG._group_key(same_day[0]) != forged
    try:
        DG.assert_no_split_leakage([0], [1], [], groups=np.asarray(same_day, dtype=object))
    except ValueError as e:
        assert 'share group ids' in str(e), e
    else:
        raise AssertionError('equal temporal group ids must not cross splits')
    same_delta = [np.timedelta64(1, 'D'), datetime.timedelta(days=1)]
    assert DG._group_key(same_delta[0]) == DG._group_key(same_delta[1])
    # Distinct nanosecond timestamps stay distinct, including in the fast
    # native-dtype path (tolist() would turn them into bare integers).
    g = np.asarray(['2020-01-01T00:00:00.000000001',
                    '2020-01-01T00:00:00.000000002'], dtype='datetime64[ns]')
    keys = DG._group_index_fast(g)[0]
    assert keys[0] != keys[1], keys
    idx = DG.grouped_chronological_split_indices(2, g, 0.5, 0.5, 0.0)
    assert idx['train'].tolist() == [0] and idx['val'].tolist() == [1], idx
    # A string/object ``order`` forces the per-row fallback.  Its member map
    # must use the same native-scalar temporal keys as the fast path; using
    # tolist() turned datetime64[ns] into bare integers and raised KeyError.
    g4 = np.asarray(['2020-01-01', '2020-01-02', '2020-01-01',
                     '2020-01-03'], dtype='datetime64[ns]')
    idx4 = DG.grouped_chronological_split_indices(
        4, g4, train_size=2, val_size=1, test_size=1,
        order=np.asarray(['a', 'b', 'a', 'c']))
    assert idx4['train'].tolist() == [0, 2], idx4
    assert idx4['val'].tolist() == [1] and idx4['test'].tolist() == [3], idx4
    # NaT still collapses into one missing group for both temporal kinds.
    nat = np.asarray([np.datetime64('NaT', 'ns'), np.datetime64('NaT', 'ns'),
                      np.datetime64(1, 'ns')])
    try:
        DG.assert_no_split_leakage([0], [1], [2], groups=nat)
    except ValueError as e:
        assert 'share group ids' in str(e), e
    else:
        raise AssertionError('NaT group ids must collapse into one group')
    # Timezone-aware datetimes have one instant even when written with
    # different offsets.  They must share a group id (Python equality/hash do)
    # while staying distinct from a naive datetime with the same wall clock.
    tz = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)
    tz_west = datetime.datetime(
        2019, 12, 31, 19,
        tzinfo=datetime.timezone(datetime.timedelta(hours=-5)))
    assert tz == tz_west
    assert DG._group_key(tz) == DG._group_key(tz_west)
    assert DG._group_key(tz) != DG._group_key(datetime.datetime(2020, 1, 1))
    try:
        DG.assert_no_split_leakage(
            [0], [1], [], groups=np.asarray([tz, tz_west], dtype=object))
    except ValueError as e:
        assert 'share group ids' in str(e), e
    else:
        raise AssertionError('the same aware instant must not cross splits')


def test_python_temporal_row_and_category_identity():
    # datetime.date/datetime/timedelta compare equal to their NumPy temporal
    # counterparts but hash differently.  Duplicate detection, category keys
    # and group keys must all use one canonical identity.
    import datetime
    import json
    import data_guard as DG
    day = datetime.date(2020, 1, 1)
    midpoint = datetime.datetime(2020, 1, 1)
    npday = np.datetime64('2020-01-01')
    assert DG._group_key(day) == DG._group_key(midpoint) == DG._group_key(npday)
    assert DG._target_key(day) == DG._target_key(midpoint) == DG._target_key(npday)
    for pair in ((day, npday), (midpoint, npday),
                 (datetime.timedelta(days=1), np.timedelta64(1, 'D'))):
        rows = np.asarray(pair, dtype=object).reshape(-1, 1)
        assert DG.duplicate_mask(rows).tolist() == [False, True], pair
    # Python datetime only has microsecond resolution: adjacent values stay
    # distinct instead of collapsing through a rounded float key.
    micro = np.asarray([midpoint, midpoint + datetime.timedelta(microseconds=1),
                        midpoint], dtype=object).reshape(-1, 1)
    assert DG.duplicate_mask(micro).tolist() == [False, False, True]
    # An aware value is not the same row as a naive one with the same clock,
    # but the same instant in two zones is.
    utc = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)
    west = datetime.datetime(
        2019, 12, 31, 19,
        tzinfo=datetime.timezone(datetime.timedelta(hours=-5)))
    aware_rows = np.asarray([utc, west, midpoint], dtype=object).reshape(-1, 1)
    assert DG.duplicate_mask(aware_rows).tolist() == [False, True, False]
    # Categorical/frequency encoding: equal spellings are one learned level,
    # and the persisted key round-trips to the same codes.
    X = np.asarray([[day], [npday], [datetime.date(2020, 1, 2)]],
                   dtype=object)
    cat = DG.LeakageSafeCategoricalEncoder().fit(X)
    stats = cat.get_stats()['features'][0]
    assert stats['n_categories'] == 2, stats
    assert len(set(stats['keys'])) == 2, stats
    codes = cat.transform(X).ravel()
    assert codes[0] == codes[1] != codes[2], codes
    cat2 = DG.LeakageSafeCategoricalEncoder.from_stats(
        json.loads(json.dumps(cat.get_stats())))
    assert np.array_equal(cat2.transform(X), cat.transform(X))
    freq = DG.LeakageSafeFrequencyEncoder().fit(X)
    entries = freq.get_stats()['features'][0]['entries']
    assert len(entries) == 2, entries
    assert entries[0]['count'] == 2, entries
    f = freq.transform(X).ravel()
    assert f[0] == f[1] > f[2], f
    enc = DG.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2).fit(
        X, np.asarray([10.0, 10.0, 20.0]))
    lookup = enc.get_stats()['features'][0]['lookup']
    assert lookup[DG._target_key(day)] == 10.0, lookup
    # Labels follow the same equality: equal temporal spellings are one class,
    # so duplicated features are not a false label conflict and a train/val
    # swap between the two spellings is not a fabricated 100% label shift.
    pair_X = np.arange(8, dtype=float).reshape(4, 2)
    conflict = DG.data_quality_report(
        pair_X[:2], y=np.asarray([day, npday], dtype=object),
        train_idx=[0, 1])
    assert conflict['checks']['label_conflicts_on_duplicate_rows'] == 0, conflict
    shift = DG.data_quality_report(
        pair_X, y=np.asarray([day, npday, npday, day], dtype=object),
        train_idx=[0, 1], val_idx=[2, 3])
    assert shift['checks']['label_distribution_shift_vs_train']['val'][
        'js_divergence'] == 0.0, shift
    # Metric label inference uses the same identity: NumPy may fail to merge a
    # Python datetime with the equal np.datetime64, which previously left two
    # declared classes and rejected the observed label as unknown.
    _y = np.asarray([day, npday, datetime.date(2020, 1, 2),
                     npday, day, datetime.date(2020, 1, 2)], dtype=object)
    _m = L.classification_metrics(_y, y_pred=_y)
    assert len(_m['labels']) == 2, _m['labels']
    assert _m['accuracy'] == 1.0, _m
    _same = L.classification_metrics(
        np.asarray([utc, west], dtype=object),
        y_pred=np.asarray([utc, west], dtype=object))
    assert len(_same['labels']) == 1, _same['labels']
    # The categorical fingerprint drives the target encoder's training-row
    # refusal.  Equal temporal categories written as different types must
    # fingerprint identically, or transform() would re-enable the full-table
    # leak on what is really the training matrix.
    day2 = datetime.date(2020, 1, 2)
    train_dates = np.asarray([[day], [day2]], dtype=object)
    holdout_np = np.asarray([[npday],
                             [np.datetime64('2020-01-02')]], dtype=object)
    assert DG._categorical_fingerprint(train_dates) == \
        DG._categorical_fingerprint(holdout_np)
    # The encoder-fit audit compares canonical persisted keys, not readable
    # reprs: the same categories in NumPy spelling must still verify.
    assert DG.audit_encoder_fit(
        DG.LeakageSafeCategoricalEncoder().fit(train_dates),
        holdout_np)['status'] == 'train_only_verified'
    assert DG.audit_encoder_fit(
        DG.LeakageSafeFrequencyEncoder().fit(train_dates),
        holdout_np)['status'] == 'train_only_verified'
    leak_enc = DG.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2).fit(
        train_dates, np.asarray([1.0, 2.0]))
    try:
        leak_enc.transform(holdout_np)
    except ValueError as e:
        assert 'training rows' in str(e), e
    else:
        raise AssertionError(
            'an equal-temporal training matrix must be refused by transform')


def test_validation_row_split_group_aware():
    # The balanced row boundary (4) would cut the group spanning rows 3-4.
    # A group-aware split must move it to a document boundary instead.
    groups = np.array([0, 0, 0, 1, 1, 2, 2, 3])
    sel, ev, meta = L.validation_row_split(8, groups=groups)
    assert meta['groups_aware'] is True and meta['n_groups'] == 4, meta
    assert sorted(sel + ev) == list(range(8)), (sel, ev)
    assert not ({groups[i] for i in sel} & {groups[i] for i in ev}), (sel, ev)
    assert 3 not in sel or 4 not in ev, (sel, ev)
    # Without groups the historical balanced split is unchanged.
    sel2, ev2, meta2 = L.validation_row_split(8)
    assert (sel2, ev2) == (list(range(4)), list(range(4, 8))), (sel2, ev2)
    assert meta2['groups_aware'] is False, meta2
    # A group larger than a fold cannot be cut; refuse instead of leaking it.
    try:
        L.validation_row_split(6, n_folds=3, groups=[0, 0, 0, 1, 1, 1])
    except ValueError as e:
        assert 'without cutting a group' in str(e), e
    else:
        raise AssertionError('a fold boundary inside a group must be refused')
    try:
        L.validation_row_split(4, groups=[0, 0])
    except ValueError as e:
        assert 'groups has length' in str(e), e
    else:
        raise AssertionError('groups length mismatch must be refused')
    # A boundary between adjacent group ids is not enough when the same group
    # id reappears later in the stream: the two returned folds would still
    # share a document.  Refuse the leaking split.
    try:
        L.validation_row_split(8, groups=[0, 1, 0, 1, 0, 1, 0, 1])
    except ValueError as e:
        assert 'contributes rows to both' in str(e), e
    else:
        raise AssertionError('a non-contiguous group spanning both folds must be refused')
    # Missing group ids collapse into one group (NaT is not a unique key).
    _nat = np.array([np.timedelta64('NaT', 'ns'), np.timedelta64('NaT', 'ns'),
                     np.timedelta64(1, 'ns'), np.timedelta64(1, 'ns')],
                    dtype=object)
    _s, _e, _m = L.validation_row_split(4, groups=_nat)
    assert _m['n_groups'] == 2, _m
    assert not ({str(_nat[i]) for i in _s} & {str(_nat[i]) for i in _e}), (_s, _e)


def test_cached_validation_row_groups_merge_shared_documents():
    # The token-level sidecar stores a document id per token.  Fixed row
    # windows can straddle a document, so rows 1-3 must become one group; a
    # row-only split would otherwise be able to select on part of a document
    # and evaluate on another part.
    _d = tempfile.mkdtemp(prefix='valgroups_')
    _sl, _ntr, _vb, _vs = 2, 1, 10, 4
    _tag = f'v5_sl{_sl}_cap{_ntr}_v{_vb}_vs{_vs}'
    _docs = np.array([1, 1, 2, 2, 2, 3, 3, 3], dtype=np.uint64)
    np.save(os.path.join(_d, f'val_token_doc_ids_{_tag}.npy'), _docs)
    with open(os.path.join(_d, f'meta_{_tag}.json'), 'w', encoding='utf-8') as fh:
        json.dump({'doc_ids': {'validation_token_count': int(_docs.size),
                               'validation_token_fp': L._ids_fp(_docs)}}, fh)
    _groups, _meta = L.cached_validation_row_groups(
        _sl, _ntr, vocab_size=_vb, val_seqs=_vs, n_rows=4, cache_dir=_d)
    assert _groups == [0, 1, 1, 1], (_groups, _meta)
    assert _meta['groups_aware'] is True and _meta['n_groups'] == 2, _meta
    _s, _e, _m = L.validation_row_split(4, groups=_groups)
    assert _m['groups_aware'] is True, _m
    assert not ({_groups[i] for i in _s} & {_groups[i] for i in _e}), (_s, _e)
    # A group-aware simple-baseline artifact must pass its own audit.  The
    # audit cannot re-derive these folds from (n_rows, seed, folds) alone: a
    # row-only replay gives different boundaries, so treating that replay as
    # the expected split rejected every valid group-aware floor.
    _rng = np.random.default_rng(7)
    _proto = dict(L.SIMPLE_BASELINE_POLICY,
                  arms={'unigram': {'order': 1}},
                  grids={'unigram': {'alpha': [0.1, 1.0, 2.0]}})
    _art = L.simple_baseline_fit(_rng.integers(0, _vb, size=200),
                                 _rng.integers(0, _vb, size=(4, 5)), _vb,
                                 protocol=_proto, selection_rows=_s,
                                 evaluation_rows=_e, row_split=_m)
    _aud = L.simple_baseline_audit(_art, protocol=_proto)
    assert _aud['baseline_verified'] is True, _aud
    # A missing sidecar is an explicit fallback, not a silent claim.
    _g, _m = L.cached_validation_row_groups(
        _sl, _ntr, vocab_size=_vb, val_seqs=_vs, n_rows=4, cache_dir=tempfile.mkdtemp())
    assert _g is None and _m['groups_aware'] is False and 'absent' in _m['reason'], (_g, _m)
    # A legacy document-level sidecar cannot be expanded to token groups; it
    # is an explicit fallback (not a crash and not a fabricated split).
    _d_legacy = tempfile.mkdtemp(prefix='valgroups_legacy_')
    np.save(os.path.join(_d_legacy, f'val_doc_ids_{_tag}.npy'),
            np.array([1, 2, 3], dtype=np.uint64))
    _gl, _ml = L.cached_validation_row_groups(
        _sl, _ntr, vocab_size=_vb, val_seqs=_vs, n_rows=4, cache_dir=_d_legacy)
    assert _gl is None and _ml['groups_aware'] is False, (_gl, _ml)
    assert 'legacy' in _ml['reason'], _ml
    # Metadata that declares the token-level sidecar but no longer has the
    # file is corrupt evidence: falling back to the legacy document file (or
    # to row-only folds) would silently change the group identity.
    _d_missing = tempfile.mkdtemp(prefix='valgroups_missing_declared_')
    np.save(os.path.join(_d_missing, f'val_doc_ids_{_tag}.npy'),
            np.array([1, 2, 3], dtype=np.uint64))
    with open(os.path.join(_d_missing, f'meta_{_tag}.json'), 'w',
              encoding='utf-8') as fh:
        json.dump({'doc_ids': {'validation_token_count': 8,
                               'validation_token_fp': 'deadbeef'}}, fh)
    try:
        L.cached_validation_row_groups(
            _sl, _ntr, vocab_size=_vb, val_seqs=_vs, n_rows=4,
            cache_dir=_d_missing)
    except ValueError as e:
        assert 'is missing' in str(e), e
    else:
        raise AssertionError('a declared-but-missing token sidecar must be refused')
    # A present but too-short TOKEN sidecar must fail closed.
    _d2 = tempfile.mkdtemp(prefix='valgroups_bad_')
    _tag2 = f'v5_sl{_sl}_cap{_ntr}_v{_vb}_vs{_vs}'
    _docs_short = np.array([1, 1, 2, 2], dtype=np.uint64)
    np.save(os.path.join(_d2, f'val_token_doc_ids_{_tag2}.npy'), _docs_short)
    with open(os.path.join(_d2, f'meta_{_tag2}.json'), 'w', encoding='utf-8') as fh:
        json.dump({'doc_ids': {'validation_token_count': int(_docs_short.size),
                               'validation_token_fp': L._ids_fp(_docs_short)}}, fh)
    try:
        L.cached_validation_row_groups(_sl, _ntr, vocab_size=_vb, val_seqs=_vs,
                                       n_rows=4, cache_dir=_d2)
    except ValueError as e:
        assert 'need' in str(e), e
    else:
        raise AssertionError('a short token-level document-id sidecar must be refused')
    # A float sidecar must not be silently truncated into document groups:
    # int(1.5) == 1 would merge two distinct documents.
    _d3 = tempfile.mkdtemp(prefix='valgroups_float_')
    _tag3 = f'v5_sl{_sl}_cap{_ntr}_v{_vb}_vs{_vs}'
    _docs_float = np.array([1.0, 1.5, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0])
    np.save(os.path.join(_d3, f'val_token_doc_ids_{_tag3}.npy'), _docs_float)
    with open(os.path.join(_d3, f'meta_{_tag3}.json'), 'w', encoding='utf-8') as fh:
        json.dump({'doc_ids': {'validation_token_count': int(_docs_float.size),
                               'validation_token_fp': L._ids_fp(_docs_float)}}, fh)
    try:
        L.cached_validation_row_groups(_sl, _ntr, vocab_size=_vb, val_seqs=_vs,
                                       n_rows=4, cache_dir=_d3)
    except ValueError as e:
        assert 'integers' in str(e), e
    else:
        raise AssertionError('a non-integer document-id sidecar must be refused')
    # Negative ids are not document identities, even when their fingerprint
    # matches the metadata.
    _d4 = tempfile.mkdtemp(prefix='valgroups_negative_')
    _docs_neg = np.array([1, 1, -2, -2, 3, 3, 3, 3], dtype=np.int64)
    np.save(os.path.join(_d4, f'val_token_doc_ids_{_tag3}.npy'), _docs_neg)
    with open(os.path.join(_d4, f'meta_{_tag3}.json'), 'w', encoding='utf-8') as fh:
        json.dump({'doc_ids': {'validation_token_count': int(_docs_neg.size),
                               'validation_token_fp': L._ids_fp(_docs_neg)}}, fh)
    try:
        L.cached_validation_row_groups(_sl, _ntr, vocab_size=_vb, val_seqs=_vs,
                                       n_rows=4, cache_dir=_d4)
    except ValueError as e:
        assert 'negative' in str(e), e
    else:
        raise AssertionError('a negative document-id sidecar must be refused')
    # A malformed meta shape (list instead of object) is fail-closed evidence,
    # not an AttributeError from a list.get call.
    _d5 = tempfile.mkdtemp(prefix='valgroups_meta_')
    np.save(os.path.join(_d5, f'val_token_doc_ids_{_tag3}.npy'), _docs)
    with open(os.path.join(_d5, f'meta_{_tag3}.json'), 'w', encoding='utf-8') as fh:
        json.dump([], fh)
    try:
        L.cached_validation_row_groups(_sl, _ntr, vocab_size=_vb, val_seqs=_vs,
                                       n_rows=4, cache_dir=_d5)
    except ValueError as e:
        assert 'JSON object' in str(e), e
    else:
        raise AssertionError('a non-object metadata file must be refused')
    # A fractional token count must not be truncated to the real sidecar
    # length (int(8.5) == 8) and blessed as a match.
    _d6 = tempfile.mkdtemp(prefix='valgroups_count_')
    np.save(os.path.join(_d6, f'val_token_doc_ids_{_tag3}.npy'), _docs)
    with open(os.path.join(_d6, f'meta_{_tag3}.json'), 'w', encoding='utf-8') as fh:
        json.dump({'doc_ids': {'validation_token_count': float(_docs.size) + 0.5,
                               'validation_token_fp': L._ids_fp(_docs)}}, fh)
    try:
        L.cached_validation_row_groups(_sl, _ntr, vocab_size=_vb, val_seqs=_vs,
                                       n_rows=4, cache_dir=_d6)
    except ValueError as e:
        assert 'length' in str(e), e
    else:
        raise AssertionError('a fractional token-id count must be refused')
    # A multi-dimensional sidecar would be silently flattened into a different
    # token->document mapping; require the documented one-dimensional layout.
    _d8 = tempfile.mkdtemp(prefix='valgroups_2d_')
    np.save(os.path.join(_d8, f'val_token_doc_ids_{_tag3}.npy'), _docs.reshape(2, 4))
    try:
        L.cached_validation_row_groups(_sl, _ntr, vocab_size=_vb, val_seqs=_vs,
                                       n_rows=4, cache_dir=_d8)
    except ValueError as e:
        assert 'one-dimensional' in str(e), e
    else:
        raise AssertionError('a multi-dimensional document-id sidecar must be refused')
    # The tuning wrapper may fall back only when a VALID group table cannot
    # form the requested folds.  A corrupt sidecar must propagate: silently
    # treating it as "no split" would select on the final-estimate rows.
    _base = {'dataset': 'wikitext', 'seq_len': _sl, 'n_train_tokens': _ntr,
             'vocab_size': _vb, 'val_seqs': _vs, 'cache_dir': _d3}
    _spec = {'seed': 20261020, 'n_folds': 2,
             'selection_fold': 0, 'evaluation_fold': 1}
    try:
        BT._nested_row_split_checked(_base, 4, _spec)
    except ValueError as e:
        assert 'integers' in str(e), e
    else:
        raise AssertionError('a corrupt group sidecar must not become a full-validation fallback')
    # A valid table whose single document spans every row cannot be cut into
    # folds; that is an explicit fallback with a reason, not a crash.
    _d7 = tempfile.mkdtemp(prefix='valgroups_one_')
    _docs_one = np.ones(8, dtype=np.uint64)
    np.save(os.path.join(_d7, f'val_token_doc_ids_{_tag3}.npy'), _docs_one)
    with open(os.path.join(_d7, f'meta_{_tag3}.json'), 'w', encoding='utf-8') as fh:
        json.dump({'doc_ids': {'validation_token_count': int(_docs_one.size),
                               'validation_token_fp': L._ids_fp(_docs_one)}}, fh)
    _base_one = dict(_base, cache_dir=_d7)
    _sel1, _ev1, _m1, _why1 = BT._nested_row_split_checked(_base_one, 4, _spec)
    assert _sel1 is None and _ev1 is None and _why1, (_sel1, _ev1, _m1, _why1)
    assert _m1['groups_aware'] is False and 'unavailable' in _m1['reason'], _m1
    # Cache metadata is untrusted input: a non-object/corrupt meta must send
    # the loader to its rebuild path (or a hard refusal for the frozen test
    # cache), never crash later on meta.get().
    _bad_meta = os.path.join(tempfile.mkdtemp(prefix='badmeta_'), 'meta.json')
    for _payload in ('[]', '{not json', 'null'):
        with open(_bad_meta, 'w', encoding='utf-8') as fh:
            fh.write(_payload)
        assert L._read_cache_meta(_bad_meta, 'unit') is None, _payload
    with open(_bad_meta, 'w', encoding='utf-8') as fh:
        json.dump({'ok': 1}, fh)
    assert L._read_cache_meta(_bad_meta, 'unit') == {'ok': 1}


def test_build_report_primary_estimand():
    # The zero-GPU report must pair on the same held-out rows as the nested
    # primary PPL; reading the full-validation `ppl` restores selection rows.
    def _rec(full, hold, seed=0):
        return {'variant': 'a', 'seed': seed, 'ppl': full, 'ppl_holdout': hold,
                'ppl_estimation_rows': 'nested_validation_holdout',
                'nll_sum_holdout': math.log(hold) * 40, 'n_tokens_holdout': 40,
                'run_cfg': 'cfg', 'steps': 1, 'tokens_seen': 40,
                'data_fp': 'd', 'val_fp': 'v', 'eval_fp': 'e'}
    a = _rec(10.0, 9.0)
    b = _rec(10.5, 10.5)
    assert BR._primary_ppl(a) == 9.0 and BR._primary_ppl(b) == 10.5
    got = BR.paired_row({('', 0): a}, {('', 0): b})
    assert got is not None and got[1] == [9.0 - 10.5], got
    # Holdout per-row accounting is an alternative to ppl_holdout.
    rows = dict(a)
    for _k in ('ppl_holdout', 'nll_sum_holdout', 'n_tokens_holdout'):
        rows.pop(_k)
    rows['nll_per_row_holdout'] = [math.log(9.0) * 40]
    rows['n_tokens_per_row_holdout'] = [40]
    assert abs(BR._primary_ppl(rows) - 9.0) < 1e-12, rows
    # A declared-nested record without any holdout accounting is unverifiable,
    # and the pair is omitted rather than quoting the full-validation value.
    bad = dict(a)
    for _k in ('ppl_holdout', 'nll_sum_holdout', 'n_tokens_holdout'):
        bad.pop(_k)
    assert BR._primary_ppl(bad) is None
    assert BR.paired_row({('', 0): bad}, {('', 0): b}) is None
    # A holdout CE beyond the float exponent range is a non-measurement, not a
    # crash in the report builder.
    huge = dict(bad, ppl_estimation_rows='nested_validation_holdout',
                nll_sum_holdout=1e6, n_tokens_holdout=1)
    assert BR._primary_ppl(huge) is None
    # The two arms must declare the SAME estimand.  A full-validation record
    # paired against a holdout record produces a difference on neither sample.
    full = {'variant': 'b', 'seed': 0, 'ppl': 10.0, 'run_cfg': 'cfg',
            'steps': 1, 'tokens_seen': 40, 'data_fp': 'd', 'val_fp': 'v',
            'eval_fp': 'e'}
    assert BR.paired_row({('', 0): a}, {('', 0): full}) is None
    # Nor may different seeds contribute deltas from different estimands to
    # one sign-flip test.
    a1 = _rec(10.0, 9.0, seed=1)
    b1 = _rec(10.5, 10.5, seed=1)
    _mix = BR.paired_row({('', 0): full, ('', 1): a1},
                         {('', 0): dict(full, ppl=11.0), ('', 1): b1})
    assert _mix is None, _mix


def test_build_report_exact_seed_and_params():
    # The zero-GPU report reader must share exp_lib's exact-integer contract:
    # int(Decimal('1.5')) would silently pair the wrong run, and a fractional
    # parameter count is not a capacity measurement.
    from decimal import Decimal
    _orig_summary = BR.summary
    try:
        BR._LOADED.clear()
        BR.summary = lambda _outdir: {'d::x': {'variant': 'x', 'seed': Decimal('1.5'),
                                               'ppl': 2.0, 'run_cfg': 'r',
                                               'steps': 1, 'tokens_seen': 1}}
        assert BR.per_seed_records('/nonexistent-panel') == {}
        BR._LOADED.clear()
        BR.summary = lambda _outdir: {'d::x': {'variant': 'x', 'seed': Decimal('2'),
                                               'ppl': 2.0, 'run_cfg': 'r',
                                               'steps': 1, 'tokens_seen': 1}}
        got = BR.per_seed_records('/nonexistent-panel')
        assert list(got.get('x', {})) == [('', 2)], got
        BR._LOADED.clear()
        BR.summary = lambda _outdir: {'d::x': {'variant': 'x', 'seed': '1.0',
                                               'ppl': 2.0, 'run_cfg': 'r',
                                               'steps': 1, 'tokens_seen': 1}}
        assert BR.per_seed_records('/nonexistent-panel') == {}
        # Malformed/non-string arm ids are excluded, not used as (unhashable)
        # dict keys or silently pooled under a coerced spelling.
        for _bad_variant in ({'x': 1}, ['x'], '', None, 3):
            BR._LOADED.clear()
            BR.summary = lambda _outdir, _v=_bad_variant: {
                'd::x': {'variant': _v, 'seed': 0, 'ppl': 2.0, 'run_cfg': 'r',
                         'steps': 1, 'tokens_seen': 1}}
            assert BR.per_seed_records('/nonexistent-panel') == {}, _bad_variant
    finally:
        BR.summary = _orig_summary
        BR._LOADED.clear()
    assert BR.param_gap({'params': 1000}, {'params': 1000.5}) is None
    assert BR.param_gap({'params': 1000}, {'params': True}) is None
    assert BR.param_gap({'params': 1000}, {'params': 1010}) is not None


def test_par_watchdog_override_validation():
    import par
    ctx = {'workers': 1, 'slots': [''], 'on_cuda': False}
    assert par._job_watchdog_seconds({'kind': 'x', 'watchdog_seconds': 42}, ctx) == 42.0
    assert par._job_watchdog_seconds({'kind': 'x'}, {'watchdog_seconds': 7}) == 7.0
    assert par._job_watchdog_seconds({'kind': 'x', 'watchdog_seconds': 0}, ctx) == 12.0 * 3600.0
    for bad in (-1, 0.5 and -0.5, float('nan'), float('inf'), 'not-a-number',
                True, False):
        try:
            par._job_watchdog_seconds({'kind': 'x', 'watchdog_seconds': bad}, ctx)
        except ValueError:
            pass
        else:
            raise AssertionError(f'watchdog_seconds={bad!r} must be rejected')
    # NumPy scalars are normal numeric payloads, not strings.
    assert par._norm({'seed': np.int64(3)}) == {'seed': 3}
    assert par._norm(np.array([[1, 2]], dtype=np.int64)) == [[1, 2]]

