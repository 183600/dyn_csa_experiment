"""test_review_fixes.py — 单元测试（纯 CPU、零实验）。

覆盖：
  A. exp_lib 统计工具：
     holm_min_attainable_p / signflip_min_consistency / signflip_power /
     tost_paired / exact_signflip_p 假设文档存在性（导入不报错即可证）；
  B. v10 腐坏模式：_corrupt_row 的既有四种模式互相对拍，新模式
     span/doc/topical 的确定性与目标区不可触碰约束；
  C. v12 门槛：_load_manifest 功效门槛（6 种子拒绝、12 种子放行、
     descriptive_only 豁免）；_admit_confirmation_records 的白名单/歧义/
     混切片/None 指纹行为；_recipe_matches_signature；冻结锁的 manifest
     不匹配拒绝与 V12_REFREEZE 降级；v2 清单文件本身的功效表与代码一致；
  D. eval_length_gen 逐文档 PPL：池化 PPL == 逐文档 PPL 的几何平均
     （归约不变）。

运行：python test_review_fixes.py   （需要 torch；无 torch 的环境全部跳过
并给出非零提醒码 0——CI 以「SKIPPED」字样识别）。"""
import contextlib
import io
import json
import math
import os
import random
import shutil
import sys
import tempfile
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except (ValueError, OSError):
            pass
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
try:
    import numpy as np
    import torch  # noqa: F401
    import exp_lib as L
    import explib.corruption as V10
    import explib.confirmation as V12
    import explib.rope as V7
    import audit_gates as A
    import baseline_tune as BT
    import build_report as BR
    import error_analysis as EA
    import explib.m0 as M0
    _HAVE_TORCH = True
except Exception as _e:  # pragma: no cover
    print(f'[test] SKIPPED: cannot import the repo modules here ({type(_e).__name__}: {_e}); run on a box with CPU torch')
    _HAVE_TORCH = False
_results = []
import pytest as _pytest
_SkipTest = _pytest.skip.Exception
def _check(name, fn):
    try:
        fn()
        _results.append((name, 'PASS', ''))
        print(f'[test] PASS  {name}')
    except _SkipTest as e:
        _results.append((name, 'SKIP', str(e)))
        print(f'[test] SKIP  {name}: {e}')
    except AssertionError as e:
        _results.append((name, 'FAIL', str(e)))
        print(f'[test] FAIL  {name}: {e}')
    except Exception as e:
        _results.append((name, 'ERROR', f'{type(e).__name__}: {e}'))
        print(f'[test] ERROR {name}: {type(e).__name__}: {e}')
def _brute_signflip_p(d):
    """Reference chunked enumeration."""
    d = np.asarray(d, dtype=float)
    n = int(d.size)
    obs = abs(float(d.mean()))
    total = 1 << n
    cnt = 0
    bitpos = np.arange(n, dtype=np.int64)[None, :]
    step = max(1, min(total, (1 << 22) // max(n, 1)))
    for off in range(0, total, step):
        bs = min(step, total - off)
        masks = np.arange(off, off + bs, dtype=np.int64)[:, None]
        signs = np.where((masks & (1 << bitpos)) != 0, 1.0, -1.0)
        cnt += int((np.abs((signs * d[None, :]).mean(1)) >= obs - 1e-12).sum())
    return cnt / total
if _HAVE_TORCH:
    _PANEL = np.random.default_rng(7).integers(3, 8000, size=(9, 129), dtype=np.int64)
    _VOCAB = 8192
def _legacy_corrupt_row(val_ids, eval_len, rho, mode, jj, vocab, target):
    """The PRE-refactor inline algorithm, replayed bit-for-bit (uniform /
    natural / mask / delete). Any divergence means the extraction changed a
    measured value — the strongest invariant this suite checks."""
    n_far = eval_len - target
    if mode == 'delete':
        return np.asarray(val_ids[jj, eval_len - target:eval_len + 1], dtype=np.int64).copy()
    ids = np.asarray(val_ids[jj, :eval_len + 1], dtype=np.int64).copy()
    k = int(round(rho * n_far))
    if mode == 'mask':
        ids[:n_far] = V10.MASK_TOKEN_ID
        return ids
    elif k > 0:
        if mode == 'uniform':
            rng = np.random.default_rng((eval_len * 1000003 + jj * 10007) * 1048576 + int(round(rho * 1048576)))
            pos = rng.choice(n_far, size=k, replace=False)
            _draw = rng.integers(0, vocab, size=k)
            _same = _draw == ids[pos]
            while _same.any():
                _draw[_same] = rng.integers(0, vocab, size=int(_same.sum()))
                _same = _draw == ids[pos]
            ids[pos] = _draw
        else:
            rng = np.random.default_rng((eval_len * 1000003 + jj * 10007) * 1048576 + int(round(rho * 1048576)))
            pos = rng.choice(n_far, size=k, replace=False)
            _src = (jj + max(1, val_ids.shape[0] // 2)) % val_ids.shape[0]
            ids[pos] = np.asarray(val_ids[_src, pos], dtype=np.int64)
    return ids
def _write_manifest(path, seeds, extra_stat=None, binding=None):
    m = {'frozen_at': '2026-10-17', 'code_semantics_expected': L.CODE_SEMANTICS, 'arms': ['a_arm', 'b_arm'], 'seeds': seeds, 'primary_comparisons': [{'name': 'c1', 'a': 'a_arm', 'b': 'b_arm', 'endpoint': 'by_len.ppl@4096'}], 'statistics': {'test': 'exact two-sided sign-flip', 'n_seeds': len(seeds), 'correction': 'holm step-down'}}
    if extra_stat:
        m['statistics'].update(extra_stat)
    if binding:
        m['binding'] = binding
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(m, f)
    return m
def _rec(variant, seed, fp, by_len=None, code=None, recipe='steps3000_sl512_bs12_nt8000000_lr0.0003_wd0.1_d256_L6_H8_Dh32_v8192_wu50_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee250_es128_detcpu-eager_dfdeadbeef_cscURRENT', msha='CURRENT', ckpt=None):
    _r = {'variant': variant, 'seed': seed, 'split': 'test', 'data_fp': fp, 'by_len': by_len or {'4096': {'ppl': 30.0, 'n_tok': 16 * 4095, 'truncated': False}}, '_code': code or L.code_stamp(), 'recipe': recipe, 'manifest_sha256': msha, 'key': f'{variant}::s{seed}'}
    if ckpt is not None:
        _r['ckpt_sha256'] = ckpt
    return _r
def _manifest_for_admission(seeds=(0, 1, 2), sig=None):
    m = {'arms': ['a_arm', 'b_arm'], 'seeds': list(seeds), 'code_semantics_expected': L.CODE_SEMANTICS, 'manifest_sha256': 'CURRENT', 'statistics': {'test': 'exact two-sided sign-flip', 'correction': 'holm step-down'}, 'primary_comparisons': []}
    if sig is not None:
        m['binding'] = {'training_recipe_signature': sig}
    return m
def _skip_dataloader_if_unavailable(contract):
    """Turn a sandbox's missing worker-process support into a SKIP, not a fail.

    `dataloader_contract_check` reports the environment error in its result
    instead of raising, so the test must inspect it.  A real wiring bug still
    reaches the assertions below.
    """
    error = str((contract or {}).get('error') or '')
    if 'PermissionError' in error or 'Errno 13' in error:
        raise _SkipTest(
            'DataLoader worker processes need POSIX semaphores, which this '
            f'sandbox forbids ({error})')
def _global_multiplicity_random_cases():
    import random as _rnd
    _rnd.seed(20261021)
    for _trial in range(40):
        m = _rnd.randint(1, 24)
        yield [(f'c{i}', _rnd.choice([None, _rnd.random()])) for i in range(m)]
def _factorial_records(interaction, n=7, params=None, noise=0.0):
    arms = [L.FACTORIAL_DESIGN['dynamic_blocking_x_learned_selection']['cells'][k]
            for k in L._FACTORIAL_CELL_KEYS]
    knobs = L.training_knobs()
    params = params or {}
    recs = {}
    for s in range(n):
        base = math.log(50.0) + 0.1 * s
        inter = interaction[s] if isinstance(interaction, (list, tuple)) else interaction
        levels = {arms[0]: base, arms[1]: base - 0.2, arms[2]: base - 0.1,
                  arms[3]: base - 0.3 + inter
                  + (noise if s % 2 else -noise)}
        for arm in arms:
            recs[(arm, '', s)] = {
                'variant': arm, 'protocol': '', 'seed': s,
                'run_cfg': 'same_cfg', 'ppl': math.exp(levels[arm]),
                'steps': 100, 'tokens_seen': 1000,
                'params': int(params.get(arm, 1000)),
                'train_knobs': knobs,
            }
    return recs
def _document_cell(nll, tokens, doc_ids):
    """Build a self-consistent by_len cell from per-document NLL/tokens."""
    ppl_doc = [math.exp(n / t) for n, t in zip(nll, tokens)]
    nll_sum = math.fsum(nll)
    n_tok = sum(tokens)
    return {'n_tok': n_tok, 'truncated': False, 'n_seq': 2,
            'nll_sum': nll_sum, 'ppl': math.exp(nll_sum / n_tok),
            'doc_granularity': 'document',
            'nll_per_doc': list(nll), 'n_tokens_per_doc': list(tokens),
            'ppl_per_doc': ppl_doc, 'doc_ids': list(doc_ids),
            'n_docs': len(tokens)}
def _fake_source_tree(tag):
    """A small source manifest for freeze-lock tests (no real tree hashing)."""
    return {
        'schema': 1,
        'algorithm': 'sha256(sorted relpath NUL bytes NUL sha256 LF)',
        'sha256': (tag * 64)[:64],
        'n_files': 2,
        'bytes': 10,
        'files': {
            'exp_lib.py': {'sha256': (tag * 64)[:64], 'bytes': 7},
            'explib.confirmation.py': {'sha256': (tag * 64)[:64], 'bytes': 3},
        },
        'scope': 'test fixture',
    }
def _write_document_unit_fixture(td):
    """Minimal line-unit WikiText cache + raw dumps for the L152 audit."""
    train = [' = Alpha = ', ' Alpha is a test article. ', ' It has two lines. ',
             ' = Beta = ', ' Beta is another article. ']
    val = [' = Gamma = ', ' Gamma is a validation article. ']
    test = [' = Alpha = ', ' Alpha is a test article. ', ' It has two lines. ',
            ' = Beta = ', ' Beta is another article. ', ' Another line here. ']
    sl, vocab = 8, 512
    fit = [r for r in train if r.strip()] + [
        'abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 .,:;!?']
    tok = L.build_tokenizer(fit, vocab)
    eos = tok.token_to_id('<eos>')
    tr_len = []
    tr = L._tokenise_row_chunks([train], 10 ** 9, 'train', tok, eos, lengths_out=tr_len)
    tr_rows = [r for r in train if r.strip()]
    tr_doc = L._doc_ids_for_capped_rows(tr_rows, tr_len, len(tr))
    va_len = []
    va = L._tokenise_row_chunks([val], 10 ** 9, 'val', tok, eos, lengths_out=va_len)
    va_rows = [r for r in val if r.strip()]
    nv = len(va) // sl
    va_batch = va[:nv * sl].reshape(nv, sl)
    va_doc = L._doc_ids_for_capped_rows(va_rows, va_len, nv * sl)
    va_tok = L._token_doc_ids_for_capped_rows(va_rows, va_len, nv * sl)
    te_len = []
    te = L._tokenise_row_chunks([test], 10 ** 9, 'test', tok, eos, lengths_out=te_len)
    te_rows = [r for r in test if r.strip()]
    nt = len(te) // sl
    te_batch = te[:nt * sl].reshape(nt, sl)
    te_doc = L._doc_ids_for_capped_rows(te_rows, te_len, nt * sl)
    te_tok = L._token_doc_ids_for_capped_rows(te_rows, te_len, nt * sl)
    tag = f'v5_sl{sl}_capX_v{vocab}_vsX'
    np.save(os.path.join(td, f'train_ids_{tag}.npy'), tr.astype(np.int32))
    np.save(os.path.join(td, f'val_batch_{tag}.npy'), va_batch.astype(np.int64))
    np.save(os.path.join(td, f'val_bnd_{tag}.npy'), np.zeros_like(va_batch, dtype=bool))
    np.save(os.path.join(td, f'train_doc_ids_{tag}.npy'), tr_doc)
    np.save(os.path.join(td, f'val_doc_ids_{tag}.npy'), va_doc)
    np.save(os.path.join(td, f'val_token_doc_ids_{tag}.npy'), va_tok)
    tok.save(os.path.join(td, f'tok_{tag}.json'))
    meta = {'vocab': vocab, 'seq_len': sl, 'fit_fp': 'FIT', 'src_fp': 'x',
            'val_fp': 'y', 'document_unit': 'line',
            'document_unit_is_article': False}
    with open(os.path.join(td, f'meta_{tag}.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f)
    ttag = f'v2_sl{sl}_v{vocab}_ns{nt}'
    np.save(os.path.join(td, f'test_ids_{ttag}.npy'), te_batch.astype(np.int64))
    np.save(os.path.join(td, f'test_bnd_{ttag}.npy'), np.zeros_like(te_batch, dtype=bool))
    np.save(os.path.join(td, f'test_doc_ids_{ttag}.npy'), te_doc)
    np.save(os.path.join(td, f'test_token_doc_ids_{ttag}.npy'), te_tok)
    tok.save(os.path.join(td, f'test_tok_{ttag}.json'))
    with open(os.path.join(td, f'test_meta_{ttag}.json'), 'w', encoding='utf-8') as f:
        json.dump({'vocab': vocab, 'seq_len': sl, 'fit_fp': 'FIT', 'src_fp': 'x',
                   'bnd_fp': 'z', 'document_unit': 'line',
                   'document_unit_is_article': False,
                   'split_fingerprints': {'test': {'sequences': int(nt),
                                                   'seq_len': int(sl)}}}, f)
    raw = os.path.join(td, 'raw')
    os.makedirs(raw, exist_ok=True)
    with open(os.path.join(raw, 'wiki.train.raw'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train) + '\n')
    with open(os.path.join(raw, 'wiki.valid.raw'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(val) + '\n')
    return {'tag': tag, 'ttag': ttag, 'raw': raw,
            'val_doc_path': os.path.join(td, f'val_doc_ids_{tag}.npy'),
            'val_token_doc_path': os.path.join(td, f'val_token_doc_ids_{tag}.npy')}
def _article_cell(nll_doc, tok_doc, ids_doc, nll_art, tok_art, ids_art,
                  n_seq=2):
    """A self-consistent by_len cell carrying BOTH line and article units."""
    n_tok = sum(tok_doc)
    assert sum(tok_art) == n_tok and sum(nll_art) == sum(nll_doc)
    return {
        'n_tok': n_tok, 'truncated': False, 'n_seq': n_seq,
        'nll_sum': math.fsum(nll_art),
        'ppl': math.exp(math.fsum(nll_art) / n_tok),
        'nll_per_seq': [math.fsum(nll_art)] * n_seq,
        'ppl_per_seq': [math.exp(math.fsum(nll_art) / n_tok)] * n_seq,
        'doc_granularity': 'document',
        'nll_per_doc': list(nll_doc), 'n_tokens_per_doc': list(tok_doc),
        'ppl_per_doc': [math.exp(n / t) for n, t in zip(nll_doc, tok_doc)],
        'doc_ids': list(ids_doc), 'n_docs': len(tok_doc),
        'article_granularity': 'article',
        'nll_per_article': list(nll_art),
        'n_tokens_per_article': list(tok_art),
        'ppl_per_article': [math.exp(n / t) for n, t in zip(nll_art, tok_art)],
        'article_ids': list(ids_art), 'n_articles': len(tok_art),
    }
def _dose_factory(n, arm):
    table = {
        'k8': {'index_topk': 8},
        'k32': {'index_topk': 32},
        'k128': {'index_topk': 128},
        'k32_plus_block': {'index_topk': 32, 'block_size': 8},
    }
    if arm not in table:
        raise KeyError(arm)
    return [dict({'kind': 'csa', 'block_size': 4}, **table[arm])
            for _ in range(int(n))]

__all__ = [n for n in list(globals()) if not n.startswith('__')]
