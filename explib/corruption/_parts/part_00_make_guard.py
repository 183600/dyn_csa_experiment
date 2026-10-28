#!/usr/bin/env python3
import copy
import gc
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except (ValueError, OSError):
            pass
import numpy as np
import torch
import torch.nn.functional as F
import explib.resume as V9
V = V9.V
L = V9.L
V8 = V9.V8
REPO = V9.REPO
DEVICE = L.DEVICE
RUN_CAP_V10 = dict(L.RUN_CAP)
RUN_CAP_V10.update(total_hours=float(os.environ.get('V10_CAP_HOURS', 25.0)), state_path='run_time_state_v10.json', already_hours=0.0)

def make_guard():
    g = L.TimeGuard(RUN_CAP_V10)
    if not g.state.get('sps_by_class'):
        for src in ('run_time_state_v9.json', 'run_time_state_v8.json', 'run_time_state_v7.json'):
            if os.path.exists(src):
                try:
                    st = json.load(open(src, encoding='utf-8'))
                    sbc = st.get('sps_by_class') or {}
                    norm = st.get('norm_sps')
                    if not sbc and norm is None:
                        print(f'[v10] NOTE: {src} holds no calibration data — leaving the guard uncalibrated rather than recording an empty calibration')
                        continue
                    g.state['sps_by_class'] = sbc
                    g.state['norm_sps'] = norm
                    g._save()
                    print(f'[v10] TimeGuard calibrated from {src}')
                    break
                except Exception as e:
                    print(f'[v10] WARNING: cannot read {src} ({type(e).__name__}: {e}) — this run starts UNCALIBRATED (the first admission uses the conservative default)')
    return g
P3MT_PAYLOAD = dict(outdir='results_v10_mech', variants=['csa_fixed_rope', 'full_rope'], eval_lens=[512, 1024, 2048, 4096], steps=3000, train_len=512, vocab=8192)
# 从头训练的选择器对照：与 csa_fixed_rope / full_rope 完全同一配方
# （3000 步、同数据、同种子）训练，唯一差异是选择方式本身。P3MT 的 variants
# 扩入它们后，既有 ckpt 的 recipe 逐字节不变（recipe 只含 per-variant 的 mr 与
# 语料指纹），resume 不会重训已完成的臂。
SCRATCH_CONTROLS = ['csa_fixed_rope_randidx', 'csa_fixed_rope_allblocks']
P3MT_PAYLOAD_CONTROLS = dict(P3MT_PAYLOAD, variants=SCRATCH_CONTROLS)
# 干扰注入的腐坏模式：
#   uniform —— 既有协议：远端 ρ 比例 token 替换为词表均匀随机 token（既有
#              cell 键不带 mode 后缀，完全向后兼容）；
#   natural —— 真实 token 同位置换：保持
#              unigram/词频统计，但逐位置拼接破坏句法、语义与篇章结构——
#              只支持「对真实语料 token 替换的鲁棒性」，不支持「对自然
#              长文档干扰的鲁棒性」（报告标签 realistic-token，§15 修订）；
#   mask    —— 远端整体屏蔽为 <pad>（保留长度与位置，移除信息）：测「干净
#              远端上下文到底带来多少收益」。<pad> 对远端是分布外输入，
#              只作互补探针、与 delete 成对解读（§15 修订）；
#   delete  —— 远端整体删除（输入只剩目标区，RoPE 相对位置自洽）：同上
#              问题的另一口径，同时改变长度与上下文结构；
#   span    —— 连续句段替换：远端 k 个 token 换成
#              供体行的连续片段（段长 PROBE_SPAN_LEN，默认 32）——段内保
#              持局部句法，段间与跨位置连贯性被破坏；ρ 为名义值，实际
#              替换量 = 段数 × 段长（可由同一确定性公式复算）；
#   doc     —— 完整无关文档上下文替换（ρ 恒 1.0，新增）：远端整段换成
#              供体行的连续等长切片（协议不允许改变 eval_len，故以等长
#              替换实现「插入无关文档」的科学问题，长度与位置受控）；
#   topical —— 主题相近供体的同位置替换（新增）：供体行 = 与本行远端
#              unigram Jaccard 相似度最高的其他行（确定性、无 RNG），
#              再做 natural 式同位置替换——「主题相近但答案无关」的
#              最廉价近似。
#   全部模式的腐坏只依赖 (eval_len, 行号, ρ, mode, span_len)——与臂/模型
#   种子无关，同一 (行, ρ, mode) 在臂间逐字节相同，严格配对成立。
PROBE_MODES = {'uniform': [0.0, 0.125, 0.25, 0.5], 'natural': [0.125, 0.25, 0.5], 'mask': [1.0], 'delete': [1.0], 'span': [0.125, 0.25, 0.5], 'doc': [1.0], 'topical': [0.25, 0.5]}
PROBE_SPAN_LEN = 32
NIAH_NATURAL = dict(summary='results_v10_mech/niah_natural.json', n_pairs=8, n_seq=16, conditions=['clean', 'distract'], rho=0.5)
PROBE = dict(outdir='results_v10_mech', ckpt_dir='results_v10_mech/ckpt', summary='results_v10_mech/distractor.json', variants=['csa_fixed_rope', 'full_rope'], seeds=[0, 1, 2], eval_lens=[2048, 4096], rhos=[0.0, 0.125, 0.25, 0.5], modes=dict(PROBE_MODES), target=512, n_seq=16, chunk=4, arms=['dense', 'learned', 'randidx', 'allblocks'], arm_overrides={}, niah=dict(NIAH_NATURAL))

def _arm_of(variant, arm, cfg=None):
    _ov = (cfg or {}).get('arm_overrides') or {}
    if variant in _ov:
        return arm in _ov[variant]
    if variant in ('full_rope', 'full_rope_sink'):
        return arm == 'dense'
    return arm in ('learned', 'randidx', 'allblocks')

MASK_TOKEN_ID = 0  # <pad>：mask 模式的远端屏蔽填充 token（分布外输入，§15 修订已声明）

_CORRUPT_MODES = ('uniform', 'natural', 'mask', 'delete', 'span', 'doc', 'topical')


class _BoundedRowCache:
    """Small byte-bounded cache for deterministic probe rows.

    FIFO eviction bounds the table; a miss just rebuilds the row from its
    (eval_len, row, rho, mode) seed, which is the same value.
    """

    def __init__(self, max_bytes=512 << 20):
        self.max_bytes = int(max_bytes)
        self._d = {}
        self._nbytes = 0

    @staticmethod
    def _bytes(v):
        try:
            return int(np.asarray(v).nbytes)
        except Exception:
            return 0

    def get(self, key, default=None):
        return self._d.get(key, default)

    def __contains__(self, key):
        return key in self._d

    def __getitem__(self, key):
        return self._d[key]

    def __setitem__(self, key, value):
        if key in self._d:
            self._nbytes -= self._bytes(self._d[key])
        self._d[key] = value
        self._nbytes += self._bytes(value)
        while self._d and self._nbytes > self.max_bytes:
            k0 = next(iter(self._d))
            self._nbytes -= self._bytes(self._d.pop(k0))

    def clear(self):
        self._d.clear()
        self._nbytes = 0



_TOPICAL_SETS = {}
_TOPICAL_DONORS = {}


def _topical_row_sets(val_ids, n_far):
    """Per-row unigram sets for one (panel, eval_len), built once per process."""
    key = (id(val_ids), tuple(val_ids.shape), str(val_ids.dtype), int(n_far))
    ent = _TOPICAL_SETS.get(key)
    if ent is not None and ent[0] is val_ids:
        return ent[1]
    sets = [set(np.asarray(val_ids[r, :n_far], dtype=np.int64).tolist())
            for r in range(val_ids.shape[0])]
    if len(_TOPICAL_SETS) >= 2:
        _TOPICAL_SETS.pop(next(iter(_TOPICAL_SETS)))
    _TOPICAL_SETS[key] = (val_ids, sets)
    return sets


def _topical_donor(val_ids, jj, n_far, exclude):
    """Deterministic most-similar donor row for the `topical` mode.

    供体 = 与第 jj 行远端 unigram 集合 Jaccard 相似度最高的其他行（排除
    自身与 `exclude`）。纯整数/集合运算、无 RNG、无臂依赖——同一 (面板,
    行号) 在臂间给出同一供体，严格配对不受影响。平手时取行号最小者
    （集合运算的确定性顺序）。行集合与最终供体都按 (panel, eval_len)
    缓存，跨 seed/variant/cell 复用。"""
    _ck = (id(val_ids), tuple(val_ids.shape), str(val_ids.dtype), int(n_far),
           tuple(sorted(int(_x) for _x in exclude)), int(jj))
    _hit = _TOPICAL_DONORS.get(_ck)
    if _hit is not None and _hit[0] is val_ids:
        return _hit[1]
    rows = val_ids.shape[0]
    _sets = _topical_row_sets(val_ids, n_far)
    _base = _sets[jj]
    best, best_sim = None, -1.0
    for cand in range(rows):
        if cand == jj or cand in exclude:
            continue
        _c = _sets[cand]
        if not _base and not _c:
            sim = 0.0
        else:
            inter = len(_base & _c)
            union = len(_base | _c)
            sim = (inter / union) if union else 0.0
        if sim > best_sim:
            best, best_sim = cand, sim
    if len(_TOPICAL_DONORS) >= 65536:
        _TOPICAL_DONORS.pop(next(iter(_TOPICAL_DONORS)))
    _TOPICAL_DONORS[_ck] = (val_ids, best)
    return best


def _corrupt_row(val_ids, eval_len, rho, mode, jj, vocab, target, span_len=PROBE_SPAN_LEN):
    """Build ONE corrupted evaluation row (pure, arm/model-seed independent).

    输入 val_ids 为 (n_rows, >=eval_len+1) 的 token 面板；返回长度
    eval_len+1 的 int64 行。四种既有模式（uniform/natural/mask/delete）与
    既有实现逐字节一致（同一 RNG 播种、同一调用顺序——见 _distractor_ppls
    的行缓存键）。三种新模式（span/doc/topical）同样只
    依赖 (eval_len, jj, rho, mode, span_len)，保证臂间严格配对。"""
    if mode not in _CORRUPT_MODES:
        raise ValueError(f'_corrupt_row: unknown corruption mode {mode!r} (expected one of {_CORRUPT_MODES})')
    if isinstance(target, (bool, np.bool_)) or not isinstance(target, (int, np.integer)):
        raise ValueError(f'_corrupt_row: target must be an integer, got {target!r}')
    target = int(target)
    if target < 1:
        raise ValueError(f'_corrupt_row: target must be positive, got {target}')
    if isinstance(eval_len, (bool, np.bool_)) or not isinstance(eval_len, (int, np.integer)):
        raise ValueError(f'_corrupt_row: eval_len must be an integer, got {eval_len!r}')
    eval_len = int(eval_len)
    if eval_len < target:
        raise ValueError(
            f'_corrupt_row: eval_len={eval_len} is shorter than the scored '
            f'target zone ({target}); the far context would be negative and '
            'the corruption geometry is undefined')
    if isinstance(vocab, (bool, np.bool_)) or not isinstance(vocab, (int, np.integer)):
        raise ValueError(f'_corrupt_row: vocab must be a positive integer, got {vocab!r}')
    vocab = int(vocab)
    if vocab < 1:
        raise ValueError(f'_corrupt_row: vocab must be positive, got {vocab}')
    if isinstance(rho, (bool, np.bool_)):
        raise ValueError(f'_corrupt_row: rho must be a finite number in [0,1], got {rho!r}')
    try:
        rho = float(rho)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'_corrupt_row: rho must be a finite number in [0,1], got {rho!r}') from None
    if not (math.isfinite(rho) and 0.0 <= rho <= 1.0):
        raise ValueError(f'_corrupt_row: rho must be finite and in [0,1], got {rho!r}')
    _panel = np.asarray(val_ids)
    if (_panel.ndim != 2 or not (0 <= int(jj) < _panel.shape[0])
            or _panel.shape[1] < eval_len + 1):
        raise ValueError(
            f'_corrupt_row: val_ids must be a 2-D panel with row {jj!r} and '
            f'>= {eval_len + 1} tokens, got shape {_panel.shape}')
    n_far = eval_len - target
    if mode in ('mask', 'delete', 'doc') and abs(float(rho) - 1.0) > 1e-9:
        raise ValueError(f"mode {mode!r} is an all-or-nothing condition and only admits rho=1.0 (got {rho!r}); partial masking is a different question and is not this probe")
    if mode == 'delete':
        # the far context is REMOVED, not filled: the model sees only
        # the final target+1 tokens (RoPE is relative, so the kept
        # tokens' pairwise structure is unchanged)
        return np.asarray(val_ids[jj, eval_len - target:eval_len + 1], dtype=np.int64).copy()
    ids = np.asarray(val_ids[jj, :eval_len + 1], dtype=np.int64).copy()
    k = int(round(rho * n_far))
    if mode == 'mask':
        ids[:n_far] = MASK_TOKEN_ID
        return ids
    if mode == 'doc':
        # whole far context replaced by ONE contiguous slice of another row
        # (length- and position-controlled "unrelated document" context)
        rng = np.random.default_rng((eval_len * 1000003 + jj * 10007) * 1048576 + int(round(rho * 1048576)))
        _src = (jj + max(1, val_ids.shape[0] // 2)) % val_ids.shape[0]
        _row_len = min(val_ids.shape[1], eval_len + 1)
        _max_off = _row_len - n_far
        _off = int(rng.integers(0, max(1, _max_off))) if _max_off > 0 else 0
        ids[:n_far] = np.asarray(val_ids[_src, _off:_off + n_far], dtype=np.int64)
        return ids
    if k <= 0:
        return ids
    if mode == 'uniform':
        # With a one-token vocabulary there is no other token to draw: the
        # historical rejection loop would redraw the same value forever (the
        # configuration is degenerate, not a corruption condition).
        if int(vocab) <= 1:
            return ids
        rng = np.random.default_rng((eval_len * 1000003 + jj * 10007) * 1048576 + int(round(rho * 1048576)))
        pos = rng.choice(n_far, size=k, replace=False)
        _draw = rng.integers(0, vocab, size=k)
        _same = _draw == ids[pos]
        while _same.any():
            _draw[_same] = rng.integers(0, vocab, size=int(_same.sum()))
            _same = _draw == ids[pos]
        ids[pos] = _draw
        return ids
    if mode in ('natural', 'topical'):
        # natural: paired other-row realistic tokens at the same seeded
        # position grid; collisions with the original token are KEPT (they
        # are part of the realistic-token statistics).
        # topical: same protocol, but the donor row is the most unigram-similar
        # row (topic proxy) instead of the fixed-offset row.
        rng = np.random.default_rng((eval_len * 1000003 + jj * 10007) * 1048576 + int(round(rho * 1048576)))
        pos = rng.choice(n_far, size=k, replace=False)
        if mode == 'topical':
            _nat = (jj + max(1, val_ids.shape[0] // 2)) % val_ids.shape[0]
            _src = _topical_donor(val_ids, jj, n_far, exclude={_nat})
            if _src is None:
                _src = _nat
        else:
            _src = (jj + max(1, val_ids.shape[0] // 2)) % val_ids.shape[0]
        ids[pos] = np.asarray(val_ids[_src, pos], dtype=np.int64)
        return ids
    if mode == 'span':
        # contiguous sentence-like segments from the donor row: local syntax
        # inside each span survives, cross-span coherence is destroyed.
        # actual coverage = len(starts) * seg (recomputable from the same
        # deterministic formula — rho is nominal for this mode).
        rng = np.random.default_rng((eval_len * 1000003 + jj * 10007) * 1048576 + int(round(rho * 1048576)))
        seg = max(1, int(span_len))
        n_seg = max(1, int(math.ceil(k / seg)))
        seg = min(seg, n_far)
        _row_len = min(val_ids.shape[1], eval_len + 1)
        _starts_hi = n_far - seg + 1
        starts = rng.choice(_starts_hi, size=min(n_seg, _starts_hi), replace=False)
        _src = (jj + max(1, val_ids.shape[0] // 2)) % val_ids.shape[0]
        _off_hi = _row_len - seg + 1
        for s in starts:
            o = int(rng.integers(0, _off_hi)) if _off_hi > 1 else 0
            ids[int(s):int(s) + seg] = np.asarray(val_ids[_src, o:o + seg], dtype=np.int64)
        return ids
    raise ValueError(f'_corrupt_row: unknown corruption mode {mode!r} (expected one of {_CORRUPT_MODES})')


@torch.inference_mode()
def _distractor_ppls(model, val_ids, eval_len, rho, n_seq, cfg, row_cache=None, chunk_mem=None, mode='uniform'):
    """Score the clean 512-token target zone under a corrupted far context.

    mode='uniform'（默认）与既有实现逐字节一致：同一 (eval_len, 行号, ρ)
    播种的均匀随机替换、同 token 重采样、同上传与同评分路径。其余模式见
    PROBE_MODES 注释。所有模式的腐坏对同一 (行, ρ, mode) 在臂间逐字节相同
    （只依赖行号与配置，不依赖臂/模型种子），保持严格配对。"""
    target = cfg['target']
    _vraw = cfg.get('vocab') or P3MT_PAYLOAD['vocab']
    if isinstance(_vraw, (bool, np.bool_)):
        raise ValueError(f'_distractor_ppls: vocab must be a positive integer, got {_vraw!r}')
    _v = L._as_seed_int(_vraw)
    if _v is None or _v < 1:
        raise ValueError(f'_distractor_ppls: vocab must be a positive integer, got {_vraw!r}')
    vocab = int(_v)
    # corruptible context positions: input positions outside the scored
    # target zone; the first scored prediction reads input position
    # eval_len - target (it predicts ids[eval_len - target + 1]), so the
    # clean context spans ids[0 : eval_len - target]
    if isinstance(target, (bool, np.bool_)) or not isinstance(target, (int, np.integer)):
        raise ValueError(f'_distractor_ppls: target must be an integer, got {target!r}')
    target = int(target)
    if target < 1:
        raise ValueError(f'_distractor_ppls: target must be positive, got {target}')
    if isinstance(eval_len, (bool, np.bool_)) or not isinstance(eval_len, (int, np.integer)):
        raise ValueError(f'_distractor_ppls: eval_len must be an integer, got {eval_len!r}')
    eval_len = int(eval_len)
    if eval_len < target:
        raise ValueError(
            f'_distractor_ppls: eval_len={eval_len} is shorter than the scored '
            f'target zone ({target}); the far context would be negative')
    try:
        rho = float(rho)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'_distractor_ppls: rho must be a finite number in [0,1], got {rho!r}') from None
    if not (math.isfinite(rho) and 0.0 <= rho <= 1.0):
        raise ValueError(f'_distractor_ppls: rho must be finite and in [0,1], got {rho!r}')
    n_far = eval_len - target
    nll_sum = []
    n_tok = []
    _sum_parts = []
    n_ch = max(1, int(cfg['chunk']))
    if chunk_mem is not None:
        n_ch = max(1, min(n_ch, int(chunk_mem.get(eval_len, n_ch))))
    if mode not in _CORRUPT_MODES:
        raise ValueError(f'_distractor_ppls: unknown corruption mode {mode!r} (expected one of {_CORRUPT_MODES})')
    if mode in ('mask', 'delete', 'doc') and abs(float(rho) - 1.0) > 1e-9:
        raise ValueError(f"mode {mode!r} is an all-or-nothing condition and only admits rho=1.0 (got {rho!r}); partial masking is a different question and is not this probe")
    _span_len = int(cfg.get('span_len') or PROBE_SPAN_LEN)
    i = 0
    while i < n_seq:
        j = min(i + n_ch, n_seq)
        rows = []
        for jj in range(i, j):
            _rk = (eval_len, jj, float(rho), vocab, int(cfg['target']), mode)
            _row = row_cache.get(_rk) if row_cache is not None else None
            if _row is None:
                _row = _corrupt_row(val_ids, eval_len, rho, mode, jj, vocab, target, span_len=_span_len)
                if row_cache is not None:
                    row_cache[_rk] = _row
            rows.append(_row)
        # same bit-exact staged upload as batch_iter (design notes §6.1)
        ids = L._h2d_async(np.stack(rows), torch.int64, DEVICE)
        try:
            logits = model(ids[:, :-1], logits_tail=target)
        except RuntimeError as _oe:
            del ids
            if 'out of memory' in str(_oe).lower() and n_ch > 1:
                if DEVICE.type == 'cuda':
                    torch.cuda.empty_cache()
                n_ch = max(1, n_ch // 2)
                if chunk_mem is not None:
                    chunk_mem[eval_len] = n_ch
                print(f'[p3mp] eval device-memory exhaustion; retrying with chunk={n_ch}')
                continue
            raise
        tgt = ids[:, 1:]
        _ce = F.cross_entropy(logits.reshape(-1, vocab), tgt[:, -target:].reshape(-1), reduction='none').view(len(rows), target)
        # keep each chunk's fp64 row-sums on device (fp64 values unchanged).
        _sum_parts.append(_ce.sum(dim=1, dtype=torch.float64))
        n_tok.extend([target] * len(rows))
        del ids, logits, tgt, _ce
        if chunk_mem is not None:
            chunk_mem[eval_len] = n_ch
        i = j
    if _sum_parts:
        nll_sum.extend(map(float, torch.cat(_sum_parts).tolist()))
    return (nll_sum, n_tok)

def _cell_ppl(nll_sum, n_tok):
    if not nll_sum:
        return float('nan')
    total = math.fsum((float(x) for x in nll_sum))
    nt = int(sum(n_tok))
    if nt <= 0:
        return float('nan')
    if total / nt > 709.0:
        return float('inf')
    return math.exp(total / nt)

def _probe_fingerprint(cfg):
    _need = max((int(_x) for _x in cfg['eval_lens'])) + 1
    _tl = int(cfg.get('train_len') or P3MT_PAYLOAD['train_len'])
    return {'n_seq': int(cfg['n_seq']), 'target': int(cfg['target']), 'vocab': int(cfg.get('vocab') or P3MT_PAYLOAD['vocab']), 'val_seq_len': max(_tl, _need), 'stat': 'ppl_pooled_nll_v5'}

def _probe_fp_for_mode(fp, cfg, mode):
    """Add the parameters a corruption mode actually consumes.

    `span` corruption changes with `span_len`; the existing probe_params for
    every other mode are kept byte-identical so their resume stays valid.
    Without this, changing span_len would silently reuse measurements made
    with a different span geometry."""
    if mode == 'span':
        return dict(fp, span_len=int(cfg.get('span_len') or PROBE_SPAN_LEN))
    return fp

def _fp_norm(p):
    if not isinstance(p, dict):
        return p
    return {k: v for k, v in p.items() if k not in ('chunk', 'mode')}

def _recipe_norm(recipe):
    if not isinstance(recipe, str):
        return recipe
    # Normalise ONLY the per-arm parameter-matching ``mr`` field.  A run_cfg
    # (exp_lib._run_fp) puts the mlp_match_ref value there, and those values
    # contain underscores (``csa_dynamic``, ``csa_fixed_rope``), so the old
    # ``[^_]*`` stopped at the first underscore and left the rest of the field
    # in the key: two arms that differed only in mr then never paired.
    if '_ee' in recipe:
        # run_cfg layout: _mr<mlp_match_ref>_ee<eval_every>_...
        return re.sub(r'_mr.*?(?=_ee)', '_mr*', recipe, count=1)
    if recipe.startswith('p1l_'):
        # P1L layout: _mr<numeric ratio>-<fixed match-ref suffix>_ns...
        return re.sub(r'_mr[^-]*(?=-)', '_mr*', recipe, count=1)
    # Legacy/standalone recipe: the value carries no underscore.
    return re.sub(r'_mr[^_]*', '_mr*', recipe, count=1)

def _clean_probe_params(pp):
    """probe_params used for the ρ=0 clean-baseline lookup.

    `span` carries an extra `span_len` (it changes corruption geometry); the
    clean uniform ρ=0 cell does not.  The lookup therefore strips only this
    mode-specific key."""
    if not isinstance(pp, dict):
        return pp
    out = dict(pp)
    out.pop('span_len', None)
    return out

def _probe_params_current(rec, fp, mode='uniform'):
    """Mode-aware probe-param currency.  uniform cells keep the existing
    fingerprint exactly (records without a 'mode' field are uniform), so the
    v10/v11 uniform panels stay resumable; non-uniform cells additionally carry
    their mode inside probe_params and only match records of the same mode."""
    if not isinstance(rec, dict) or not isinstance(rec.get('probe_params'), dict):
        return False
    if rec['probe_params'].get('mode', 'uniform') != mode:
        return False
    return _fp_norm(rec['probe_params']) == _fp_norm(fp)

def _holm_family(mapping):
    """Declared contrast family for Holm correction.

    Every DECLARED key keeps one slot.  A missing/non-finite p is entered as a
    dummy p=1, never filtered out: dropping the unmeasured cells would shrink
    m and make the surviving contrasts look better powered than the study
    design allows (same fail-closed rule as exp_lib.holm_adjust)."""
    fam = []
    if not isinstance(mapping, dict):
        return fam
    for _k, _v in mapping.items():
        _p = _v.get('p_exact_signflip') if isinstance(_v, dict) else None
        _valid = False
        if _p is not None and not isinstance(_p, (bool, np.bool_)):
            try:
                _pf = float(_p)
                _valid = math.isfinite(_pf) and 0.0 <= _pf <= 1.0
            except (TypeError, ValueError, OverflowError):
                _valid = False
        fam.append((str(_k), _pf if _valid else 1.0))
    return fam


def _probe_defs(cfg):
    """The (eval_len, rho, mode) grid of a probe config.  Absent 'modes' means
    the legacy uniform-only protocol (cfg['rhos'])."""
    _modes = cfg.get('modes') or {'uniform': list(cfg['rhos'])}
    out = []
    for _mode, _rhos in _modes.items():
        for Ln in cfg['eval_lens']:
            for rho in _rhos:
                out.append((int(Ln), float(rho), str(_mode)))
    return out

def _niah_cell_key(v, seed, arm, Ln, cond):
    return f'{v}::s{seed}::{arm}::L{Ln}::niah::{cond}'

def _attach_niah_factor_metadata(record, tag, factor_report):
    _decl = L.NIAH_CONTRAST_FACTORS.get(tag)
    if not isinstance(record, dict) or not isinstance(_decl, dict):
        return record
    for _key in ('single_factor', 'factors', 'factors_note'):
        if _key in _decl:
            record[_key] = _decl[_key]
    _pair = (factor_report.get('pairs') or {}).get(tag) if isinstance(factor_report, dict) else None
    _verified = isinstance(_pair, dict) and _pair.get('verified') is True
    record['factor_declaration_verified'] = _verified
    if not _verified and record.get('single_factor') is True:
        record['declared_single_factor'] = True
        record['single_factor'] = False
    return record

def _niah_fingerprint(cfg, condition=None):
    _n = cfg.get('niah') or {}
    _need = max((int(_x) for _x in cfg['eval_lens'])) + 1
    _tl = int(cfg.get('train_len') or P3MT_PAYLOAD['train_len'])
    fp = {'n_seq': int(_n.get('n_seq', cfg['n_seq'])), 'target': int(cfg['target']), 'vocab': int(cfg.get('vocab') or P3MT_PAYLOAD['vocab']), 'n_pairs': int(_n.get('n_pairs', 8)), 'val_seq_len': max(_tl, _need), 'stat': 'niah_natural_v1'}
    if condition == 'distract':
        # The distractor condition's corruption density is part of the
        # measurement; the clean condition does not read rho.
        fp['dist_rho'] = float(_n.get('rho', 0.5))
    return fp

def _build_niah_natural(val_ids, eval_len, target, n_pairs, vocab, n_seq, seed, condition, rho=0.5):
    """Far-necessary-information task on NATURAL text.

    每条序列 = [远端自然文本（内嵌 n_pairs 个 key→value 耦合）] + [目标区：
    自然文本收尾 + n_pairs 个查询 key]。查询 key 之后紧跟的正确 token 是
    远端出现过的配对 value——正确预测必须依赖远端信息。`distract` 条件在
    腐坏网格（与 uniform 模式同一播种）上把远端自然 token 换成另一行的
    自然 token（自然语言干扰），但**绝不腐坏耦合位置**——针保持干净，
    干扰只加在草料上（realistic-token 替换，不是主题/语义干扰）。所有随机性由 (seed, 行号) 决定，臂间逐字节一致。
    Returns (ids[n_seq, eval_len+1], q_cols, q_vals, stats)：q_cols 为
    **目标尾区内相对的 logits 列号**（预测 answer 的那一列），q_vals 为正确
    答案 token。"""
    rng = np.random.default_rng(seed)
    n_far = eval_len - target
    if n_far < 4 * n_pairs + 8:
        raise ValueError(f'NIAH-natural: far region ({n_far} tokens) too small for {n_pairs} couples at eval_len={eval_len}')
    if 2 * n_pairs > target - 1:
        raise ValueError(f'NIAH-natural: {n_pairs} query couples do not fit inside the {target}-token target zone')
    key_hi = max(4, vocab // 4)
    val_lo = 3 * vocab // 4
    rows = np.asarray(val_ids[:n_seq, :eval_len + 1], dtype=np.int64).copy()
    q_cols = np.zeros((n_seq, n_pairs), dtype=np.int64)
    q_vals = np.zeros((n_seq, n_pairs), dtype=np.int64)
    dists = np.zeros((n_seq, n_pairs), dtype=np.int64)
    _src_rows = (np.arange(n_seq) + max(1, val_ids.shape[0] // 2)) % val_ids.shape[0]
    for i in range(n_seq):
        keys = rng.choice(np.arange(4, key_hi), size=n_pairs, replace=False)
        vals = rng.choice(np.arange(val_lo, vocab), size=n_pairs, replace=False)
        band = n_far // n_pairs
        couple_pos = set()
        for jp in range(n_pairs):
            kp = jp * band + (band // 2 if band > 2 else 1)
            kp = min(kp, n_far - 2)
            vp = kp + 1
            rows[i, kp] = keys[jp]
            rows[i, vp] = vals[jp]
            couple_pos.update((kp, vp))
        if condition == 'distract':
            k = int(round(rho * n_far))
            if k > 0:
                crng = np.random.default_rng((eval_len * 1000003 + i * 10007) * 1048576 + int(round(rho * 1048576)))
                pos = crng.choice(n_far, size=k, replace=False)
                pos = np.array([p for p in pos if int(p) not in couple_pos], dtype=np.int64)
                if pos.size:
                    rows[i, pos] = np.asarray(val_ids[_src_rows[i], pos], dtype=np.int64)
        order = rng.permutation(n_pairs)
        qz = eval_len + 1 - 2 * n_pairs
        for j, pj in enumerate(order):
            kp = qz + 2 * j
            rows[i, kp] = keys[pj]
            rows[i, kp + 1] = vals[pj]  # the answer slot holds the gold token
            # logits columns are RELATIVE to the target tail: the model returns
            # logits for input positions (eval_len - target)..(eval_len - 1),
            # so the column that predicts the answer (position kp+1) is
            # kp - (eval_len - target).
            q_cols[i, j] = kp - (eval_len - target)
            q_vals[i, j] = vals[pj]
            _far_kp = pj * band + (band // 2 if band > 2 else 1)
            _far_kp = min(_far_kp, n_far - 2)
            dists[i, j] = kp - (_far_kp + 1)
    stats = {'n_pairs': int(n_pairs), 'eval_len': int(eval_len), 'target': int(target), 'condition': str(condition), 'rho': float(rho) if condition == 'distract' else 0.0, 'dist_mean': float(dists.mean()), 'dist_min': int(dists.min()), 'dist_max': int(dists.max()), 'couple_positions_excluded_from_distract': True}
    return (rows, q_cols, q_vals, stats)

@torch.inference_mode()
def _niah_ppls(model, val_ids, eval_len, n_seq, cfg, condition, row_cache=None, chunk_mem=None):
    """Evaluate the NIAH-natural task: top-1 accuracy + mean answer NLL."""
    _n = cfg.get('niah') or {}
    n_pairs = int(_n.get('n_pairs', 8))
    vocab = int(cfg.get('vocab') or P3MT_PAYLOAD['vocab'])
    target = int(cfg['target'])
    rho = float(_n.get('rho', 0.5)) if condition == 'distract' else 0.0
    _built = _build_niah_natural(val_ids, eval_len, target, n_pairs, vocab, n_seq, seed=(eval_len * 7919 + 104729), condition=condition, rho=rho)
    ids_np, q_cols, q_vals, stats = _built
    # Deterministic lexical floor: nearest-previous-key copy.  This is the
    # simplest rule that can solve a key->value retrieval task, so a learned
    # indexer must be read against it (and against the uniform/majority floors)
    # rather than against chance alone.
    _copy_rule = L.niah_copy_rule_baseline(ids_np, q_cols, q_vals, target=target)
    n_ch = max(1, int(cfg['chunk']))
    if chunk_mem is not None:
        n_ch = max(1, min(n_ch, int(chunk_mem.get((eval_len, 'niah'), n_ch))))
    n_correct = 0
    n_q = 0
    nll_parts = []
    _true_parts, _pred_parts = [], []
    i = 0
    while i < n_seq:
        j = min(i + n_ch, n_seq)
        ids = L._h2d_async(ids_np[i:j], torch.int64, DEVICE)
        try:
            logits = model(ids[:, :-1], logits_tail=target)
        except RuntimeError as _oe:
            del ids
            if 'out of memory' in str(_oe).lower() and n_ch > 1:
                if DEVICE.type == 'cuda':
                    torch.cuda.empty_cache()
                n_ch = max(1, n_ch // 2)
                if chunk_mem is not None:
                    chunk_mem[(eval_len, 'niah')] = n_ch
                print(f'[niah-nat] eval device-memory exhaustion; retrying with chunk={n_ch}')
                continue
            raise
        _lc = torch.as_tensor(q_cols[i:j], dtype=torch.long, device=logits.device)
        _lv = L._h2d_async(q_vals[i:j], torch.int64, DEVICE)
        _lg = torch.gather(logits, 1, _lc.unsqueeze(-1).expand(-1, -1, logits.size(-1)))
        _nll = F.cross_entropy(_lg.reshape(-1, vocab), _lv.reshape(-1), reduction='none').view(j - i, n_pairs)
        _pred = _lg.argmax(-1)
        n_correct += int((_pred == _lv).sum())
        n_q += int(_lv.numel())
        nll_parts.append(_nll)
        _true_parts.append(_lv.reshape(-1).cpu().tolist())
        _pred_parts.append(_pred.reshape(-1).cpu().tolist())
        del ids, logits, _lg, _lv, _nll
        if chunk_mem is not None:
            chunk_mem[(eval_len, 'niah')] = n_ch
        i = j
    if not nll_parts:
        return {'acc': float('nan'), 'nll': float('nan'), 'n': 0, **stats}
    _nll_all = torch.cat(nll_parts)
    _y_true = [v for part in _true_parts for v in part]
    _y_pred = [v for part in _pred_parts for v in part]
    _rm = L.retrieval_metrics(_y_true, _y_pred)
    # The count and the rate must describe the same observations.  The
    # retrieval helper also supplies the uniform/majority floors and macro
    # recall, so a top-1 accuracy cannot be read without its trivial baseline.
    if int(_rm['n_correct']) != int(n_correct) or int(_rm['n']) != int(n_q):
        raise RuntimeError(f'NIAH: retrieval accounting mismatch ({_rm["n_correct"]}/{_rm["n"]} vs {n_correct}/{n_q})')
    _keys = ('n_correct', 'acc', 'top1_accuracy', 'balanced_accuracy', 'macro_recall',
             'n_classes', 'answer_support', 'per_class_recall', 'per_class',
             'chance_accuracy', 'majority_accuracy', 'majority_label',
             'acc_lift_over_chance', 'acc_lift_over_majority',
             'averaging', 'note')
    return {'nll': float(_nll_all.mean()), 'n': int(n_q),
            **{k: _rm[k] for k in _keys}, 'copy_rule_baseline': _copy_rule,
            **stats}

def run_probe_cell(cfg, v, seed, guard=None, val_ids=None, mech_summ=None):
    """Run exactly ONE (variant, seed) probe cell: load the trained checkpoint
    (refusing stale/unprovenanced weights), probe every (arm, eval_len, rho)
    cell that is stale, and merge each record into the shared distractor.json
    under the lock.  Shared by the serial loop and by par.py children."""
    outdir = cfg['outdir']
    os.makedirs(outdir, exist_ok=True)
    spath = cfg['summary']
    out = {'status': 'ok', 'key': f'{v}::s{seed}', 'variant': v, 'seed': seed}
    _pc_run_id = L.new_experiment_id('probe')
    _pc_started = time.strftime('%Y-%m-%dT%H:%M:%S')
    _pc_t0 = time.time()
    _log_metrics = {}
    need = max(cfg['eval_lens']) + 1
    train_len = int(cfg.get('train_len') or P3MT_PAYLOAD['train_len'])
    if val_ids is None:
        L.set_seed(int(seed))
        val_ids = L.load_wikitext(max(train_len, need), 8000000)[1]
    # Fingerprint of the corpus this cell actually SCORES (probe_params records
    # only the shape/config: n_seq, target, vocab, val_seq_len).  Stored on the
    # record and checked forward-compatibly below: a stored record that carries
    # a val_fp differing from the current slice is stale and re-measured, while
    # legacy records without the field keep their existing resume behaviour.
    # v12 reuses this cell with val_ids=test_ids, so the frozen test slice gets
    # the same per-record fingerprint.
    _val_fp = L._ids_fp(np.ascontiguousarray(val_ids).reshape(-1))
    if mech_summ is None:
        mech_summ = {}
        _msp = os.path.join(cfg.get('outdir') or P3MT_PAYLOAD['outdir'], 'summary.json')
        if os.path.exists(_msp):
            try:
                mech_summ = json.load(open(_msp, encoding='utf-8'))
            except Exception as _e:
                print(f'[p3mp] FATAL: {_msp} exists but cannot be parsed ({type(_e).__name__}: {_e}) — without it no checkpoint can be attributed to a training recipe, so the probe would score unknown weights.  Move it aside to re-train.')
                raise
    ck = os.path.join(cfg['ckpt_dir'], f'{v}_seed{seed}.pt')
    if not os.path.exists(ck):
        print(f'[p3mp] SKIP {v} s{seed}: no checkpoint {ck} (run phase P3MT first)')
        out['status'] = 'skipped_no_ckpt'
        return out
    _defs = _probe_defs(cfg)
    try:
        d = V._load_ckpt(ck)
    except Exception as _cle:
        print(f'[p3mp] SKIP {v} s{seed}: checkpoint unreadable ({type(_cle).__name__}: {_cle}) — re-run the training phase (P3MT/P4MT) to regenerate it')
        out['status'] = 'skipped_bad_ckpt'
        return out
    if d.get('code') != V.CKPT_CODE:
        print(f'[p3mp] REFUSE {v} s{seed}: checkpoint predates the current code stamp (code={d.get('code')!r}) — re-run the training phase (P3MT/P4MT) first')
        del d
        out['status'] = 'skipped_stale_ckpt'
        return out
    _tr = (mech_summ or {}).get(f'{v}::seed{seed}') or {}
    if not _tr:
        print(f"[p3mp] REFUSE {v} s{seed}: {ck} exists but the mech summary has no '{v}::seed{seed}' training record — the weights have no traceable provenance, so they cannot be attributed to the current recipe")
        del d
        out['status'] = 'skipped_no_provenance'
        return out
    if not L.result_is_current(_tr, V.CKPT_CODE, 'by_len') or _tr.get('synthesized'):
        print(f"[p3mp] REFUSE {v} s{seed}: the training record '{v}::seed{seed}' is not a current measurement under this code semantics (stale stamp, unusable ppl, or synthesized) — re-run the training phase (P3MT/P4MT) so the weights are backed by a fresh measured record")
        del d
        out['status'] = 'skipped_stale_record'
        return out
    _ck_recipe = d.get('recipe')

    def _cell_stale(_key, _fp_x, _summary=None, _mode='uniform'):
        _rec = (_summary if _summary is not None else L.locked_json_read(spath)).get(_key)
        return not (L.result_is_current(_rec, V.CKPT_CODE, 'ppl_mean') and L.ppl_is_usable((_rec or {}).get('ppl_mean')) and _probe_params_current(_rec, _fp_x, _mode) and (_rec or {}).get('recipe') == _ck_recipe and ((_rec or {}).get('val_fp') is None or (_rec or {}).get('val_fp') == _val_fp))
    if d.get('vocab') is None:
        print(f'[p3mp] REFUSE {v} s{seed}: checkpoint declares no vocabulary width — refusing to substitute a current-scale default')
        del d
        out['status'] = 'skipped_missing_vocab'
        return out
    _vocab_ck = int(d['vocab'])
    if cfg.get('vocab') is not None and int(cfg['vocab']) != _vocab_ck:
        print(f"[p3mp] {v} s{seed}: probe cfg vocab={int(cfg['vocab'])} differs from the checkpoint's vocab={_vocab_ck} — using the checkpoint's (the model is what is being scored)")
    _pcfg = dict(cfg, vocab=_vocab_ck)
    _fp = _probe_fingerprint(_pcfg)
    _niah_cfg = cfg.get('niah') or None
    _nspath = _niah_cfg.get('summary') if _niah_cfg else None
    _nconds = list(_niah_cfg.get('conditions', ['clean', 'distract'])) if _niah_cfg else []

    def _niah_fp_of(_cond):
        return _niah_fingerprint(_pcfg, condition=_cond)

    def _niah_stale(_key, _summary_n, _fp_n=None):
        _rec = (_summary_n or {}).get(_key)
        return not (isinstance(_rec, dict) and (_rec.get('task') == 'niah_natural') and (isinstance(_rec.get('acc'), (int, float)) and (not isinstance(_rec.get('acc'), bool)) and math.isfinite(float(_rec['acc']))) and (isinstance(_rec.get('nll'), (int, float)) and (not isinstance(_rec.get('nll'), bool)) and math.isfinite(float(_rec['nll']))) and _rec.get('_code') == V.CKPT_CODE and _rec.get('probe_params') == (_niah_fp_of('clean') if _fp_n is None else _fp_n) and _rec.get('recipe') == _ck_recipe and (_rec.get('val_fp') is None or _rec.get('val_fp') == _val_fp))
    _summary = L.locked_json_read(spath)
    _summary_n = L.locked_json_read(_nspath) if _nspath else {}
    _todo = False
    for arm in cfg['arms']:
        if not _arm_of(v, arm, cfg):
            continue
        for Ln, rho, mode in _defs:
            if _cell_stale(f'{v}::s{seed}::{arm}::L{Ln}::r{rho}' + (f'::{mode}' if mode != 'uniform' else ''), _probe_fp_for_mode(_fp, _pcfg, mode), _summary, mode):
                _todo = True
                break
        if (not _todo) and _nspath:
            for Ln in cfg['eval_lens']:
                for cond in _nconds:
                    if _niah_stale(_niah_cell_key(v, seed, arm, Ln, cond), _summary_n, _niah_fp_of(cond)):
                        _todo = True
                        break
                if _todo:
                    break
        if _todo:
            break
    if not _todo:
        del d
        out['status'] = 'skipped_resume'
        return out
    model = None
    _chunk_mem = {}
    for arm in cfg['arms']:
        if not _arm_of(v, arm, cfg):
            continue
        _row_cache = _BoundedRowCache()
        if guard is not None and not guard.can_start(0):
            print(f'[cap] SKIP probe {v} s{seed} arm {arm}: headroom exhausted ({max(guard.remaining_hours(), 0.0):.2f} h left) — re-run the phase to continue')
            if model is None:
                out['status'] = 'skipped_cap'
            break

        def _cell_key(_Ln, _rho, _mode='uniform', _arm=arm, _v=v, _seed=seed):
            _k = f'{_v}::s{_seed}::{_arm}::L{_Ln}::r{_rho}'
            if _mode != 'uniform':
                _k += f'::{_mode}'
            return _k
        _stale_param = [f'L{Ln}::r{rho}::{mode}' for Ln, rho, mode in _defs if (_summary.get(_cell_key(Ln, rho, mode)) or {}).get('probe_params') is not None and (not _probe_params_current(_summary.get(_cell_key(Ln, rho, mode)), _probe_fp_for_mode(_fp, _pcfg, mode), mode))]
        if _stale_param:
            print(f'[p3mp] {v} s{seed} {arm}: re-probing {len(_stale_param)} cell(s) whose stored probe parameters differ from the current config ({', '.join(_stale_param[:3])}{('…' if len(_stale_param) > 3 else '')})')
        _stale_w = [f'L{Ln}::r{rho}::{mode}' for Ln, rho, mode in _defs if (_summary.get(_cell_key(Ln, rho, mode)) or {}).get('ppl_mean') is not None and (_summary.get(_cell_key(Ln, rho, mode)) or {}).get('recipe') != _ck_recipe]
        if _stale_w:
            print(f'[p3mp] {v} s{seed} {arm}: re-probing {len(_stale_w)} cell(s) that were measured under different weights (stored recipe does not match the current checkpoint) — reusing them would quote the old weights')
        cells = [(Ln, rho, mode) for Ln, rho, mode in _defs if _cell_stale(_cell_key(Ln, rho, mode), _probe_fp_for_mode(_fp, _pcfg, mode), _summary, mode)]
        _niah_cells = [(Ln, cond) for Ln in cfg['eval_lens'] for cond in _nconds if _niah_stale(_niah_cell_key(v, seed, arm, Ln, cond), _summary_n, _niah_fp_of(cond))] if _nspath else []
        if not cells and not _niah_cells:
            continue
        if model is None:
            _arch_missing = [k for k in ('d', 'n_layers', 'n_heads', 'd_head')
                             if d.get(k) is None]
            if _arch_missing:
                print(f'[p3mp] REFUSE {v} s{seed}: checkpoint omits model geometry {_arch_missing} — refusing to substitute a current-scale default')
                out['status'] = 'skipped_missing_geometry'
                return out
            _arch = (int(d['d']), int(d['n_layers']), int(d['n_heads']), int(d['d_head']))
            _probe_arch = tuple(int(x) for x in (cfg.get('probe_arch') or (256, 6, 8, 32)))
            if _arch != _probe_arch:
                print(f"[p3mp] REFUSE {v} s{seed}: checkpoint architecture {_arch} differs from the probe's declared architecture {_probe_arch} — not running the probe on unverified weights")
                out['status'] = 'skipped_arch'
                return out
            _train_len_ck = d.get('train_len')
            if _train_len_ck is None:
                print(f'[p3mp] REFUSE {v} s{seed}: checkpoint declares no training context length — refusing to substitute a current-scale default')
                out['status'] = 'skipped_missing_train_len'
                return out
            model = L.SmallGPT(_vocab_ck, *_arch, int(_train_len_ck), [copy.deepcopy(c) for c in d['cfg']], mlp_ratio=d['mlp_ratio']).to(DEVICE)
            model.load_state_dict(d.pop('sd'))
            model.eval()
        for blk, c0 in zip(model.blocks, d['cfg']):
            L.apply_niah_inference_arm(blk.attn.cfg, c0, arm)
        t0 = time.time()
        _pending = {}
        for _ci, (Ln, rho, mode) in enumerate(cells, 1):
            key = _cell_key(Ln, rho, mode)
            nll_sum, n_tok = _distractor_ppls(model, val_ids, Ln, rho, _pcfg['n_seq'], _pcfg, row_cache=_row_cache, chunk_mem=_chunk_mem, mode=mode)
            cell_ppl = _cell_ppl(nll_sum, n_tok)
            per_seq = [float('inf') if s / t > 709.0 else math.exp(s / t) for s, t in zip(nll_sum, n_tok)]
            _fp_base = _probe_fp_for_mode(_fp, _pcfg, mode)
            _fp_rec = dict(_fp_base, mode=mode) if mode != 'uniform' else _fp_base
            _pending[key] = {'variant': v, 'seed': seed, 'arm': arm, 'eval_len': Ln, 'rho': rho, 'mode': mode, 'ppl_mean': float(cell_ppl), 'ppl_std_per_seq': float(np.std(per_seq, ddof=1)) if len(per_seq) > 1 else 0.0, 'ppls': [float(p) for p in per_seq], 'nll_sum': [float(x) for x in nll_sum], 'n_tok': [int(x) for x in n_tok], 'n_seq': len(per_seq), 'target': cfg['target'], 'val_fp': _val_fp, '_code': V.CKPT_CODE, 'probe_params': _fp_rec, 'recipe': _ck_recipe, 'train_knobs': _tr.get('train_knobs'), 'train_objective': _tr.get('train_objective'), 'param_breakdown': _tr.get('param_breakdown'), 'rng_fp': _tr.get('rng_fp'), 'train_fp': _tr.get('data_fp')}
            _log_metrics[key] = {'ppl_mean': float(cell_ppl), 'n_seq': len(per_seq)}
            print(f'  [p3mp] {key:44s} PPL={cell_ppl:8.2f}', flush=True)
            if _ci % 8 == 0:
                L.locked_json_update(spath, (lambda s, _p=dict(_pending): (s.update(_p), s)[1]), indent=1)
                _pending.clear()
        if _pending:
            L.locked_json_update(spath, (lambda s, _p=dict(_pending): (s.update(_p), s)[1]), indent=1)
        for Ln, cond in _niah_cells:
            res = _niah_ppls(model, val_ids, Ln, _pcfg['n_seq'], _pcfg, cond, row_cache=_row_cache, chunk_mem=_chunk_mem)
            key = _niah_cell_key(v, seed, arm, Ln, cond)
            rec = {'variant': v, 'seed': seed, 'arm': arm, 'eval_len': int(Ln), 'task': 'niah_natural', 'condition': cond, 'acc': float(res['acc']), 'nll': float(res['nll']), 'n_queries': int(res['n']), 'n_correct': int(res['n_correct']), 'balanced_accuracy': res.get('balanced_accuracy'), 'macro_recall': res.get('macro_recall'), 'f1_macro': res.get('f1_macro'), 'f1_micro': res.get('f1_micro'), 'f1_weighted': res.get('f1_weighted'), 'f1_averaging': res.get('f1_averaging'), 'n_undeclared_predictions': res.get('n_undeclared_predictions'), 'n_classes': int(res.get('n_classes') or 0), 'answer_support': res.get('answer_support'), 'per_class_recall': res.get('per_class_recall'), 'per_class': res.get('per_class'), 'chance_accuracy': res.get('chance_accuracy'), 'majority_accuracy': res.get('majority_accuracy'), 'majority_label': res.get('majority_label'), 'acc_lift_over_chance': res.get('acc_lift_over_chance'), 'acc_lift_over_majority': res.get('acc_lift_over_majority'), 'minority_prevalence': res.get('minority_prevalence'), 'imbalance_ratio': res.get('imbalance_ratio'), 'imbalance_warning': res.get('imbalance_warning'), 'recommended_primary_metric': res.get('recommended_primary_metric'), 'recommended_primary_reason': res.get('recommended_primary_reason'), 'recommended_primary_available': res.get('recommended_primary_available'), 'copy_rule_baseline': res.get('copy_rule_baseline'), 'n_pairs': int(res.get('n_pairs', (_niah_cfg or {}).get('n_pairs', 8))), 'dist_mean': float(res.get('dist_mean', float('nan'))), 'dist_min': int(res.get('dist_min', -1)), 'dist_max': int(res.get('dist_max', -1)), 'rho': float(res.get('rho', 0.0)), 'val_fp': _val_fp, '_code': V.CKPT_CODE, 'probe_params': _niah_fp_of(cond), 'recipe': _ck_recipe}
            L.locked_json_update(_nspath, (lambda s, _p=dict({key: rec}): (s.update(_p), s)[1]), indent=1)
            _log_metrics[key] = {'acc': float(rec['acc']), 'nll': float(rec['nll']), 'n_queries': int(rec['n_queries'])}
            print(f"  [niah] {key:48s} acc={rec['acc']:.3f} nll={rec['nll']:.3f}", flush=True)
        if guard is not None:
            guard.record_run(time.time() - t0, 0, 0, 0, 0, 0)
    if _log_metrics:
        L.log_experiment_event(outdir, {
            'kind': 'cell', 'tracking_schema': 2, 'run_id': _pc_run_id,
            'status': 'ok', 'started_at': _pc_started,
            'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'elapsed_s': time.time() - _pc_t0, 'key': f'{v}::s{seed}',
            'variant': v, 'seed': int(seed), 'run_cfg': _ck_recipe,
            'code': V.CKPT_CODE,
            'provenance': _tr.get('provenance'),
            'hyperparams': _tr.get('hyperparams'),
            'data_fp': _tr.get('data_fp'), 'val_fp': _tr.get('val_fp'),
            'rng_fp': _tr.get('rng_fp'),
            # the corpus THIS probe scored (training fingerprints above
            # describe the weights; this one describes the measurement input)
            'probe_val_fp': _val_fp,
            'metrics': {'n_probe_cells': len(_log_metrics),
                        'probe_results': _log_metrics},
            'losses': _tr.get('losses'), 'ppl_history': _tr.get('ppl_history'),
        })
    if model is not None:
        del model
        gc.collect()
    del d
    return out


def run_probe_cell_job(job):
    """Child-process entry for one probe cell (invoked by par.py)."""
    guard = L.TimeGuard(job['guard_cap']) if job.get('guard_cap') else None
    return run_probe_cell(job['cfg'], job['v'], job['seed'], guard=guard)


def run_probe(cfg=PROBE, guard=None, label='v10 P3MP', par=None):
    outdir = cfg['outdir']
    os.makedirs(outdir, exist_ok=True)
    spath = cfg['summary']
    L.locked_json_read(spath)
    need = max(cfg['eval_lens']) + 1
    train_len = int(cfg.get('train_len') or P3MT_PAYLOAD['train_len'])
    _t_data = time.time()
    L.set_seed(int((list(cfg.get('seeds') or [0]) or [0])[0]))
    _, val_ids, _, _, _ = L.load_wikitext(max(train_len, need), 8000000)
    if guard is not None:
        guard.record_run(time.time() - _t_data, 0, 0, 0, 0, 0)
    mech_summ = {}
    _trdir = cfg.get('outdir') or P3MT_PAYLOAD['outdir']
    _msp = os.path.join(_trdir, 'summary.json')
    if os.path.exists(_msp):
        try:
            mech_summ = json.load(open(_msp, encoding='utf-8'))
        except Exception as _e:
            print(f'[p3mp] FATAL: {_msp} exists but cannot be parsed ({type(_e).__name__}: {_e}) — without it no checkpoint can be attributed to a training recipe, so the probe would score unknown weights.  Move it aside to re-train.')
            raise
    print(f'[p3mp] val slice {val_ids.shape}, arms={cfg['arms']}, modes={cfg.get('modes') or {'uniform': cfg['rhos']}}' + (f", niah={cfg['niah']}" if cfg.get('niah') else ''))
    _panel_started = time.strftime('%Y-%m-%dT%H:%M:%S')
    _panel_prov = L.experiment_provenance({'dataset': 'wikitext', 'n_train_tokens': cfg.get('n_train_tokens') or 8000000, 'seq_len': train_len}, val_batch=val_ids, vocab=cfg.get('vocab') or 8192)
    L.log_experiment_event(outdir, {
        'kind': 'panel_start', 'tracking_schema': 2, 'label': label or outdir,
        'phase': 'probe', 'started_at': _panel_started,
        'config': L._jsonable(cfg), 'seeds': list(cfg['seeds']),
        'variants': list(cfg['variants']), 'code': V.CKPT_CODE,
        'provenance': _panel_prov,
    })
    cells = [(v, seed) for v in cfg['variants'] for seed in cfg['seeds']]
    _ran_parallel = False
    if len(cells) > 1:
        import par as _par
        _par_ctx = par if isinstance(par, dict) else _par.resolve_ctx(par, label=label or outdir)
        if _par_ctx and _par.ctx_active(_par_ctx, len(cells)):
            module = _par_ctx.get('module') or (_caller if (_caller := sys._getframe(1).f_globals.get('__name__')) and _caller != '__main__' else 'explib.corruption')
            jobs = [dict(kind='probe', module=module, label=label, cfg=cfg, v=v, seed=seed, guard_cap=(guard.cap_spec() if guard is not None else None)) for v, seed in cells]
            _par.run_jobs(jobs, _par_ctx, phase_label=label or outdir)
            _ran_parallel = True
    if not _ran_parallel:
        for v, seed in cells:
            run_probe_cell(cfg, v, seed, guard=guard, val_ids=val_ids, mech_summ=mech_summ)
    summary = L.locked_json_read(spath)
    _measured = len({(_r.get('variant'), _r.get('seed')) for _r in summary.values()
                     if isinstance(_r, dict)
                     and (L.ppl_is_usable(_r.get('ppl_mean')) or _r.get('task') == 'niah_natural')})
    L.log_experiment_event(outdir, {
        'kind': 'panel_finish', 'tracking_schema': 2, 'label': label or outdir,
        'phase': 'probe', 'started_at': _panel_started,
        'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'n_cells_planned': len(cells), 'n_cells_measured': _measured,
        'n_cells_missing': max(0, len(cells) - _measured),
        'summary_ref': os.path.basename(spath), 'code': V.CKPT_CODE,
        'provenance': _panel_prov,
    })
    print(f'[p3mp] wrote {spath} ({len(summary)} cells)')
    return summary

P3SS_CFG = dict(L.RUN, outdir='results_lm_v10_scale_s', d=128, n_layers=4, n_heads=4, d_head=32, seq_len=512, batch_size=12, n_train_tokens=40000000, steps=8000, warmup=100, eval_every=250, eval_subset=64, variants=['csa_fixed', 'full'])
P3SL_CFG = dict(L.RUN, outdir='results_lm_v10_scale_l', d=512, n_layers=10, n_heads=16, d_head=32, seq_len=512, batch_size=12, n_train_tokens=40000000, steps=4000, warmup=100, eval_every=250, eval_subset=64, variants=['csa_fixed', 'full'])
# ---- 规模对照臂：等训练 token 与参数量代理规模的大模型臂 ----
# P3SL_TOK 与 P3SS 对齐累计 token；P3SL_FLOP 按参数量比例缩放步数
# （steps 运行期由 _flop_matched_steps 确定性计算，避免在 import 期
# 构建模型——§7 导入零副作用）。
P3SL_TOK_CFG = dict(P3SL_CFG, outdir='results_lm_v10_scale_l8k', steps=8000)
P3SL_FLOP_CFG = dict(P3SL_CFG, outdir='results_lm_v10_scale_lflop', steps=None)

def _flop_matched_steps(small_cfg, large_cfg, min_steps=250):
    """Deterministic equal-total-compute PROXY step count for the large arm.

    这里的「等计算」是 6·N·D 参数量代理（N=参数量，D=token 数），不是实测
    等 FLOPs——稀疏选择、池化、注意力与激活重算的计算量不与参数量成比例；
    若触发步数下限，也会偏离原目标规模。面板报告必须把它称为「6ND 参数
    量代理规模臂」，并分别陈述等 token 与等计算量。步数推导：先在 CPU 上
    按面板配置构建两个面板的模型（不碰 DEVICE、无 RNG 副作用），取两臂
    （csa/full）参数量的平均值，large_steps = small_steps ·
    N_small / N_large。结果打印完整推导以供审计；确定性：同一代码下重复
    调用逐位相同（count_params 是形状的纯函数）。步数下限 250（warmup=100
    时余下学习率调度仍有意义）——触发下限时必须如实报告「代理值被下限
    偏离」，不得写成严格匹配。"""
    counts = {}
    deriv = []
    for _tag, _cfg in (('small', small_cfg), ('large', large_cfg)):
        _per = {}
        for _v in _cfg['variants']:
            _cfgs = L.make_layer_cfgs(_cfg['n_layers'], _v)
            _m = L.SmallGPT(8192, _cfg['d'], _cfg['n_layers'], _cfg['n_heads'], _cfg['d_head'], _cfg['seq_len'], _cfgs)
            _n = L.count_params(_m)
            _per[_v] = int(_n)
            del _m
        counts[_tag] = float(np.mean(list(_per.values())))
        deriv.append(f'{_tag}: { { _v: int(_n) for _v, _n in _per.items() } } (mean {counts[_tag]:.0f})')
    _ratio = counts['small'] / counts['large']
    _raw = int(round(small_cfg['steps'] * _ratio))
    steps = max(int(min_steps), _raw)
    print(f'[v10] 6ND-param-proxy-matched steps (a PROXY, not measured FLOPs): {"; ".join(deriv)}; ratio={_ratio:.6g} -> large_steps={steps}' + (f'  [NOTE: floored at {min_steps} (raw {_raw}) — the proxy target is deviated by the floor, report it as such]' if steps > _raw else ''))
    return steps

PHASES = [('P3MT', 'lenphase', P3MT_PAYLOAD, [0, 1, 2]), ('P3MTC', 'lenphase', P3MT_PAYLOAD_CONTROLS, [0, 1, 2]), ('P3MP', 'probe', PROBE, None), ('P3T', 'run', V9.P2T_CFG, [2, 3]), ('P3SS', 'run', P3SS_CFG, [0, 1, 2]), ('P3SL', 'run', P3SL_CFG, [0, 1, 2]), ('P3SL2', 'run', P3SL_TOK_CFG, [0, 1, 2]), ('P3SL3', 'run', P3SL_FLOP_CFG, [0, 1, 2])]

def run_phase(name, guard, par=None):
    V9._install_ckpt_patch()
    import par as _par
    _ctx = _par.resolve_ctx(par, label=f'v10 {name}')
    for pname, kind, payload, seeds in PHASES:
        if pname != name:
            continue
        if seeds:
            payload = dict(payload, seeds=list(seeds))
        if pname == 'P3SL3' and payload.get('steps') is None:
            # proxy-compute arm: steps derived deterministically at run time
            payload = dict(payload, steps=_flop_matched_steps(P3SS_CFG, payload))
        if isinstance(_ctx, dict):
            _ctx = dict(_ctx, module='explib.corruption', ckpt_install=True, ckpt_flag=bool(_ctx.get('ckpt_flag')) or pname == 'P3T')
        if kind == 'lenphase':
            return V.run_lenphase(payload, guard=guard, label=f'v10 {pname}', par=_ctx)
        if kind == 'probe':
            return run_probe(payload, guard=guard, label=f'v10 {pname}', par=_ctx)
        ckpt_on = pname == 'P3T'
        try:
            V9._V9_CKPT['on'] = ckpt_on
            s, _a = L.run(payload, seeds=seeds, guard=guard, label=f'v10 {pname}', par=_ctx)
        finally:
            V9._V9_CKPT['on'] = False
        return s
    raise SystemExit(f'unknown phase {name}')

def _ppl_by_seed(outdir, full=False):
    return V9._ppl_by_seed(outdir, full=full)

def _hist_by_seed(outdir, variant, raw=None):
    sp = os.path.join(outdir, 'summary.json')

    class _Curves(dict):

        def __init__(self):
            super().__init__()
            self.per_seed_synth = {}
    out = _Curves()
    synth_of = {}
    if raw is None:
        if not os.path.exists(sp):
            return out
        try:
            raw = json.load(open(sp, encoding='utf-8'))
        except Exception as e:
            print(f'[stats] WARNING: cannot read {sp} ({type(e).__name__}: {e}) — no curve for {variant}')
            return out
    seen = {}
    cfg_of = {}
    for _k, r in raw.items():
        if not (isinstance(r, dict) and r.get('variant') == variant and r.get('ppl_history')):
            continue
        _s_raw = r.get('seed')
        if isinstance(_s_raw, bool):
            _s_raw = None
        if isinstance(_s_raw, float):
            s = int(_s_raw) if math.isfinite(_s_raw) and _s_raw.is_integer() else None
        else:
            try:
                s = int(_s_raw)
            except (TypeError, ValueError, OverflowError):
                s = None
        if s is None:
            print(f'[v10 stats] {_k} has no usable `seed`; its curve cannot be attributed and is skipped')
            continue
        try:
            fp = tuple(((int(t), type(v).__name__, round(float(v), 9)) for t, v in r['ppl_history'] if isinstance(v, (int, float)) and (not isinstance(v, bool)) and (v == v)))
        except (TypeError, ValueError, OverflowError):
            fp = None
        if fp is None:
            print(f'[v10 stats] {_k} has an unreadable ppl_history entry; its curve fingerprint cannot be de-duplicated reliably')
            fp = ('__unreadable__', id(r))
        if fp in seen:
            _s0 = seen[fp]
            _new_real = not r.get('synthesized')
            _cur_sfx = f'_cs{L.code_stamp()}'
            _new_cur = isinstance(r.get('run_cfg'), str) and r['run_cfg'].endswith(_cur_sfx)
            _old_cur = isinstance(cfg_of.get(_s0), str) and cfg_of[_s0].endswith(_cur_sfx)
            if _new_real and (synth_of.get(_s0) or (_new_cur and not _old_cur)):
                if s != _s0 and s in out:
                    print(f'[stats] {sp}: seed {s} of `{variant}` matches the curve of seed {_s0} but a DIFFERENT record already holds seed {s} — the panel is ambiguous; keeping the existing admissions and dropping this record')
                    continue
                seen.pop(fp)
                out.pop(_s0, None)
                synth_of.pop(_s0, None)
                cfg_of.pop(_s0, None)
                if _new_cur and not _old_cur:
                    print(f'[stats] {sp}: seed {s} of `{variant}` is a REAL record stamped with the CURRENT code semantics whose curve matches the older one admitted as seed {_s0}; keeping the current measurement (seed {s}) and dropping the older record.')
                else:
                    print(f'[stats] {sp}: seed {s} of `{variant}` is a REAL record whose curve matches the SYNTHESIZED reconstruction admitted as seed {_s0}; keeping the real measurement (seed {s}) and dropping the reconstruction.')
            else:
                if not r.get('synthesized'):
                    print(f'[stats] {sp}: seed {s} of `{variant}` is a REAL record but its curve is IDENTICAL to the one already admitted for seed {seen[fp]}; keeping seed {seen[fp]} (the first admission) and dropping seed {s} so the curve is not counted twice.')
                continue
        if s in out:
            _new_cur2 = isinstance(r.get('run_cfg'), str) and r['run_cfg'].endswith(f'_cs{L.code_stamp()}')
            _old_cur2 = isinstance(cfg_of.get(s), str) and cfg_of[s].endswith(f'_cs{L.code_stamp()}')
            if not r.get('synthesized') and synth_of.get(s):
                out.pop(s, None)
                synth_of.pop(s, None)
                cfg_of.pop(s, None)
                for _fp_old, _s_old in list(seen.items()):
                    if _s_old == s:
                        seen.pop(_fp_old)
                print(f'[stats] {sp}: a REAL record for ({variant}, seed {s}) arrives after a SYNTHESIZED reconstruction of the same seed whose curve differs; keeping the real measurement and dropping the reconstruction.')
            elif _new_cur2 and not _old_cur2:
                out.pop(s, None)
                synth_of.pop(s, None)
                cfg_of.pop(s, None)
                for _fp_old, _s_old in list(seen.items()):
                    if _s_old == s:
                        seen.pop(_fp_old)
                print(f'[stats] {sp}: two records with DIFFERENT curves both claim ({variant}, seed {s}); keeping the one stamped with the CURRENT code semantics and dropping the older-stamped one (the older reading would be excluded as stale anyway).')
            else:
                print(f'[stats] {sp}: two records with DIFFERENT curves both claim ({variant}, seed {s}) — the gap trajectory is ambiguous; the first record is kept and this one is dropped')
                continue
        seen[fp] = s
        synth_of[s] = bool(r.get('synthesized'))
        cfg_of[s] = r.get('run_cfg')
        _clean_hist = []
        _n_badpt = 0
        for _pt in r['ppl_history']:
            try:
                _t = int(_pt[0])
                _v2 = float(_pt[1])
            except (TypeError, ValueError, IndexError):
                _n_badpt += 1
                continue
            if not (math.isfinite(_v2) and _v2 > 0):
                _n_badpt += 1
                continue
            _clean_hist.append([_t, _v2])
        if _n_badpt:
            print(f'[v10 stats] {_k}: {_n_badpt} non-finite/non-positive curve point(s) dropped before the crossover math')
        out[s] = _clean_hist
    if len(out) > 1:
        _groups = {}
        for s in out:
            _groups.setdefault(json.dumps(cfg_of.get(s), sort_keys=True, default=str), []).append(s)
        if len(_groups) > 1:
            _cur_sfx = f'_cs{L.code_stamp()}'
            _ranked = sorted(_groups.values(), key=lambda g: (not any((isinstance(cfg_of.get(_s), str) and cfg_of[_s].endswith(_cur_sfx) for _s in g)), -len(g)))
            _keep = _ranked[0]
            _dropped = sorted((s for s in out if s not in set(_keep)))
            print(f'[stats] {sp}: `{variant}` curves span {len(_groups)} distinct run_cfg groups — pooling across configurations is not allowed, so the trajectory keeps only one group ({len(_keep)}/{len(out)} seeds, preferring the one stamped with the current code semantics) and drops seeds {_dropped}')
            if sum((1 for g in _groups.values() if len(g) == len(_keep))) > 1:
                print(f'[stats] {sp}: `{variant}` run_cfg groups TIE at {len(_keep)} seed(s) each — the kept group is {sorted(_keep)}; the tied alternatives are dropped, so this panel contributes fewer seeds than were measured')
            for s in _dropped:
                del out[s]
                synth_of.pop(s, None)
                cfg_of.pop(s, None)
    _cur_sfx = f'_cs{L.code_stamp()}'
    _stale = sorted((s for s in out if not (isinstance(cfg_of.get(s), str) and cfg_of[s].endswith(_cur_sfx))))
    if _stale:
        print(f'[stats] {sp}: `{variant}` curve(s) at seed(s) {_stale} are stamped with an OLDER code semantics than the current run_cfg suffix — they are kept on disk but excluded from every crossover/pairing statistic; re-run this cell to measure them with the current code')
    out.stale_cfg_seeds = set(_stale)
    out.per_seed_synth = synth_of
    out.per_seed_cfg = cfg_of
    return out

def _crossover_step(gap_steps, gap_vals, smooth=1):
    g = np.asarray(gap_vals, dtype=float)
    s = np.asarray(gap_steps, dtype=float)
    if len(g) != len(s):
        raise ValueError(f'_crossover_step: gap_steps and gap_vals must be the same length ({len(s)} vs {len(g)}) — the returned step indexes the two together, so a silent misalignment would report the crossover at the wrong STEP while looking perfectly well-formed.')
    if smooth > 0 and len(g) > 2 * smooth:
        g = np.array([g[max(0, i - smooth):i + smooth + 1].mean() for i in range(len(g))])
    if len(g):
        _suf_ok = (np.minimum.accumulate(g[::-1])[::-1] > 0) & (g > 0)
        if _suf_ok.any():
            return (float(s[int(np.argmax(_suf_ok))]), True)
    return (float(s[len(g) - 1]) if len(g) else float('nan'), False)

def _tokens_per_step(outdir, fallback=None, raw=None):
    sp = os.path.join(outdir, 'summary.json')
    if raw is None:
        if not os.path.exists(sp):
            return None
        try:
            raw = json.load(open(sp, encoding='utf-8'))
        except Exception as e:
            print(f'[stats] WARNING: cannot read {sp} ({type(e).__name__}: {e}) — no tokens/step for this panel')
            return None
    rates = []
    n_bad = 0
    for _k, r in raw.items():
        if not isinstance(r, dict) or r.get('synthesized'):
            continue
        _ts, _st = (r.get('tokens_seen'), r.get('steps'))
        if isinstance(_ts, bool) or isinstance(_st, bool):
            n_bad += 1
            continue
        if not (isinstance(_ts, (int, float)) and isinstance(_st, (int, float))):
            if _ts is not None or _st is not None:
                n_bad += 1
            continue
        if not (math.isfinite(_ts) and math.isfinite(_st) and (_st > 0)):
            n_bad += 1
            continue
        rates.append(_ts / _st)
    if n_bad:
        print(f'[stats] WARNING: {outdir}: {n_bad} record(s) state a non-numeric or non-positive tokens_seen/steps and are ignored when establishing the token axis')
    if not rates:
        if fallback is not None:
            print(f'[stats] {outdir}: no measured tokens/step rate on disk (every record is synthesized or unmeasured) — falling back to the panel-declared rate {float(fallback):g} (= batch x seq, exact for this panel)')
            return float(fallback)
        return None
    if len(set(rates)) > 1:
        print(f'[stats] WARNING: {outdir} reports {len(set(rates))} distinct tokens/step rates {sorted(set(rates))} — the token axis is ambiguous, so no crossover_tokens will be published')
        return None
    return float(rates[0])

def _scale_crossovers():
    panels = {'d128_L4': dict(outdir='results_lm_v10_scale_s', d=128, n_layers=4, batch=12, seq=512), 'd256_L6': dict(outdir='results_lm_v3_long', d=256, n_layers=6, batch=12, seq=512), 'd384_L8': dict(outdir='results_lm_v5_scale', d=384, n_layers=8, batch=8, seq=1024), 'd512_L10': dict(outdir='results_lm_v10_scale_l', d=512, n_layers=10, batch=12, seq=512), 'd512_L10_tok': dict(outdir='results_lm_v10_scale_l8k', d=512, n_layers=10, batch=12, seq=512), 'd512_L10_flop': dict(outdir='results_lm_v10_scale_lflop', d=512, n_layers=10, batch=12, seq=512)}
    out = {}
    for label, p in panels.items():
        _raw = None
        _sp = os.path.join(p['outdir'], 'summary.json')
        if os.path.exists(_sp):
            try:
                with open(_sp, encoding='utf-8') as _f:
                    _raw = json.load(_f)
            except Exception:
                _raw = None
        ha = _hist_by_seed(p['outdir'], 'csa_fixed', raw=_raw)
        hb = _hist_by_seed(p['outdir'], 'full', raw=_raw)
        _stale_seeds = sorted(getattr(ha, 'stale_cfg_seeds', set()) | getattr(hb, 'stale_cfg_seeds', set()))
        if _stale_seeds:
            print(f'[stats] {label}: seed(s) {_stale_seeds} were measured under an older code-stamp — a code change must invalidate and recompute, not be reused, so they are excluded from this {label} crossover fit')
        common = sorted((set(ha) & set(hb)) - set(_stale_seeds))
        _synth_seeds = sorted(s for s in common
                              if getattr(ha, 'per_seed_synth', {}).get(s) or getattr(hb, 'per_seed_synth', {}).get(s))
        if _synth_seeds:
            # A reconstructed curve is a copy of an OLD aggregate, not an
            # independent run.  It may be displayed as provenance, but it must
            # not enter a crossover position, a mean trajectory or a fit.
            print(f'[stats] {label}: seed(s) {_synth_seeds} carry a SYNTHESIZED reconstruction — excluded from every crossover statistic (not an independent run)')
            common = [s for s in common if s not in set(_synth_seeds)]
        if not common:
            out[label] = {'status': 'reconstruction_only' if _synth_seeds else 'missing', **p,
                          'excluded_reconstructed': _synth_seeds}
            continue
        _cfg_a = getattr(ha, 'per_seed_cfg', {})
        _cfg_b = getattr(hb, 'per_seed_cfg', {})
        _cfg_bad = [s for s in common if _cfg_a.get(s) != _cfg_b.get(s)]
        if _cfg_bad:
            print(f'[stats] {label}: seed(s) {_cfg_bad} pair a `csa_fixed` curve against a `full` curve measured under a DIFFERENT run_cfg — cross-configuration pairing is not allowed, so these seeds enter no crossover statistic and are excluded from the reported n')
            common = [s for s in common if s not in set(_cfg_bad)]
            if not common:
                out[label] = {'status': 'missing', **p}
                continue
        _paired = []
        _no_overlap = []
        for s in common:
            if set(dict(ha[s])) & set(dict(hb[s])):
                _paired.append(s)
            else:
                _no_overlap.append(s)
        if _no_overlap:
            print(f'[stats] {label}: seed(s) {_no_overlap} have NO eval step in common between `csa_fixed` and `full` — they enter no crossover statistic and are excluded from the reported n')
        if not _paired:
            print(f'[stats] {label}: no seed pairs `csa_fixed` against `full` on any shared eval step — the panel carries no crossover trajectory')
            out[label] = {'status': 'missing', **p}
            continue
        common = _paired
        tps = _tokens_per_step(p['outdir'], fallback=p['batch'] * p['seq'], raw=_raw)
        if tps is not None and tps != p['batch'] * p['seq']:
            print(f'[stats] {label}: panel tokens/step is {tps:g} but its declared batch*seq is {p['batch'] * p['seq']} — using the calibrated rate')
        per_seed = {}
        das = {s: dict(ha[s]) for s in common}
        dbs = {s: dict(hb[s]) for s in common}
        for s in common:
            da, db = (das[s], dbs[s])
            steps = sorted(set(da) & set(db))
            gap = [da[t] - db[t] for t in steps]
            xo, crossed = _crossover_step(steps, gap)
            rec = {'crossover_step': xo, 'crossed': crossed, 'n_evals': len(steps), 'last_step': steps[-1], 'final_gap': gap[-1]}
            if tps is not None and crossed and math.isfinite(xo):
                rec['crossover_tokens'] = xo * tps
            per_seed[s] = rec
        steps = sorted(set.intersection(*[set(das[s]) for s in common], *[set(dbs[s]) for s in common]))
        gap_mean = [float(np.mean([das[s][t] for s in common]) - np.mean([dbs[s][t] for s in common])) for t in steps]
        xo_m, crossed_m = _crossover_step(steps, gap_mean)
        entry = {'status': 'ok', **p, 'seeds': common, 'excluded_reconstructed': _synth_seeds, 'tokens_per_step': tps, 'per_seed': per_seed, 'per_seed_synth': {str(s): bool(getattr(ha, 'per_seed_synth', {}).get(s) or getattr(hb, 'per_seed_synth', {}).get(s)) for s in common}, 'mean_crossover_step': xo_m, 'mean_crossed': crossed_m, 'mean_traj': {'steps': steps, 'gap': gap_mean},
                 # Training-volume comparability: the scale fit needs the
                 # panel's actual training length, not just its width.  The
                 # last common eval step is the measured training horizon.
                 'last_step': (steps[-1] if steps else None),
                 'censored': bool(not crossed_m),
                 'train_tokens': ((steps[-1] * tps) if (steps and tps is not None) else None)}
        if tps is not None and crossed_m and math.isfinite(xo_m):
            entry['mean_crossover_tokens'] = xo_m * tps
        elif tps is None:
            print(f'[stats] {label}: no tokens/step on this panel — `crossover_tokens` omitted (step positions still published)')
        else:
            print(f'[stats] {label}: the mean trajectory is censored — `mean_crossover_tokens` omitted (a censored step is a lower bound, not a crossover position)')
        out[label] = entry
    return out

_SCALE_PANEL_SPECS = {'P3SS': dict(outdir='results_lm_v10_scale_s', d=128, n_layers=4, batch=12, seq=512, steps=8000, note='small arm, 8000 steps (~49.15M tokens)'), 'P3SL': dict(outdir='results_lm_v10_scale_l', d=512, n_layers=10, batch=12, seq=512, steps=4000, note='large arm as originally run, 4000 steps (~24.58M tokens — HALF the small arm training length)'), 'P3SL_TOK': dict(outdir='results_lm_v10_scale_l8k', d=512, n_layers=10, batch=12, seq=512, steps=8000, note='token-matched large arm, 8000 steps (~49.15M tokens)'), 'P3SL_FLOP': dict(outdir='results_lm_v10_scale_lflop', d=512, n_layers=10, batch=12, seq=512, steps=None, note='compute-matched large arm (steps = 8000 * N_small/N_large, floored at 250)')}

def _panel_endpoint(hists, tps):
    """Final-eval gap + token count of a panel: csa_fixed vs full, seed-paired."""
    common = sorted(set(hists['csa_fixed']) & set(hists['full']))
    if not common:
        return None
    # The sign-flip test must compare every seed at the SAME training step, so
    # take the last step present in both arms for EVERY compared seed (the
    # intersection, not each seed's own last step — a seed with a shorter eval
    # grid would otherwise contribute a delta measured at a different scale).
    common_steps = None
    for s in common:
        da, db = (dict(hists['csa_fixed'][s]), dict(hists['full'][s]))
        steps = set(da) & set(db)
        common_steps = steps if common_steps is None else (common_steps & steps)
    if not common_steps:
        return None
    last = max(common_steps)
    per_seed = {}
    for s in common:
        da, db = (dict(hists['csa_fixed'][s]), dict(hists['full'][s]))
        per_seed[s] = {'step': int(last), 'gap': float(da[last] - db[last]), 'ppl_csa': float(da[last]), 'ppl_full': float(db[last])}
    if not per_seed:
        return None
    dl = [per_seed[s]['gap'] for s in sorted(per_seed)]
    out = V.exact_sign_permutation(dl)
    out['seeds'] = sorted(per_seed)
    out['tokens'] = (float(last) * tps) if tps else None
    out['per_seed'] = per_seed
    return out

def _panel_convergence(hists):
    """Plateau-in-training-window diagnostic per (variant, seed) — NOT a convergence
    proof.

    相对 PPL 改进 < 1%（最后四分之一 eval 窗口）只能排除「仍在快速下降」
    的情形；学习率衰减、平台期、评测噪声或优化停滞都可能满足该条件。
    因此本诊断只允许谈论「规模内平台期后」的读数（crossing-after-plateau-
    in-window），不得写成「收敛后」或「渐近优势反转」——后者需要更长
    训练、合理学习率续训与重复验证（design_notes §16 修订）。保留
    `plateaued` 键以兼容既有分析产物；新增 `claim_allowed` 明示允许的
    主张级别。"""
    out = {}
    for v, h in hists.items():
        per_seed = {}
        for s, hist in h.items():
            pts = sorted((int(t), float(x)) for t, x in dict(hist).items())
            if len(pts) < 4:
                per_seed[s] = {'rel_improve_last_quarter': float('nan'), 'plateaued': None}
                continue
            q = pts[max(0, int(len(pts) * 0.75) - 1)]
            last = pts[-1]
            denom = max(abs(last[1]), 1e-9)
            rel = (q[1] - last[1]) / denom
            per_seed[s] = {'rel_improve_last_quarter': float(rel), 'plateaued': bool(rel < 0.01)}
        _vals = [x['rel_improve_last_quarter'] for x in per_seed.values() if math.isfinite(x['rel_improve_last_quarter'])]
        out[v] = {'per_seed': per_seed, 'median_rel_improve': float(np.median(_vals)) if _vals else float('nan'), 'n_plateaued': int(sum((1 for x in per_seed.values() if x['plateaued']))), 'n': len(per_seed), 'diagnostic': 'plateau-in-training-window (NOT proof of convergence: lr decay, stalls or eval noise can also satisfy it)', 'claim_allowed': 'crossing-after-plateau-in-window only; asymptotic/post-convergence claims need longer training + lr-continuation + replication'}
    return out

def _equal_token_slices():
    """The scale panels' endpoints confound model size with
    cumulative training tokens.  This analysis re-reads the SAME curves at
    matched token counts (and, when the matched arms exist, at matched total
    compute), so a cross-scale endpoint difference can no longer be silently
    attributed to scale alone."""
    conf = {}
    _panel_hists = {}
    for name, spec in _SCALE_PANEL_SPECS.items():
        _sp = os.path.join(spec['outdir'], 'summary.json')
        _raw = None
        if os.path.exists(_sp):
            try:
                with open(_sp, encoding='utf-8') as _f:
                    _raw = json.load(_f)
            except Exception:
                _raw = None
        ha = _hist_by_seed(spec['outdir'], 'csa_fixed', raw=_raw)
        hb = _hist_by_seed(spec['outdir'], 'full', raw=_raw)
        _stale = sorted(getattr(ha, 'stale_cfg_seeds', set()) | getattr(hb, 'stale_cfg_seeds', set()))
        for _h in (ha, hb):
            for s in _stale:
                _h.pop(s, None)
        # Same admission rules as `_scale_crossovers`: a synthesized
        # reconstruction is not an independent scale point, and a seed may not
        # pair two curves measured under different run_cfg values.
        _synth = sorted(s for s in (set(ha) & set(hb))
                        if getattr(ha, 'per_seed_synth', {}).get(s)
                        or getattr(hb, 'per_seed_synth', {}).get(s))
        if _synth:
            print(f'[stats] {name}: seed(s) {_synth} carry a SYNTHESIZED reconstruction — excluded from equal-token slices')
            for _h in (ha, hb):
                for s in _synth:
                    _h.pop(s, None)
        _cfg_a = getattr(ha, 'per_seed_cfg', {})
        _cfg_b = getattr(hb, 'per_seed_cfg', {})
        _cfg_bad = [s for s in (set(ha) & set(hb))
                    if _cfg_a.get(s) != _cfg_b.get(s)]
        if _cfg_bad:
            print(f'[stats] {name}: seed(s) {_cfg_bad} pair curves measured under a DIFFERENT run_cfg — excluded from equal-token slices')
            for _h in (ha, hb):
                for s in _cfg_bad:
                    _h.pop(s, None)
        _panel_hists[name] = {'csa_fixed': ha, 'full': hb}
        hists = _panel_hists[name]
        tps = _tokens_per_step(spec['outdir'], fallback=spec['batch'] * spec['seq'], raw=_raw)
        entry = {'status': 'ok', **{k: v for k, v in spec.items() if k != 'steps'}, 'tokens_per_step': tps, 'convergence': _panel_convergence(hists) if (ha or hb) else {}}
        if spec['steps'] is not None:
            entry['declared_steps'] = spec['steps']
        _ep = _panel_endpoint(hists, tps)
        if _ep is not None:
            entry['endpoint'] = _ep
        else:
            entry['status'] = 'reconstruction_only' if _synth else 'missing'
        if _synth:
            entry['excluded_reconstructed'] = _synth
        conf[name] = entry

    def _gap_at(name, step):
        h = _panel_hists.get(name) or {}
        if not h:
            return {}
        per_seed = {}
        for s in sorted(set(h['csa_fixed']) & set(h['full'])):
            da, db = (dict(h['csa_fixed'][s]), dict(h['full'][s]))
            if step in da and step in db:
                per_seed[s] = float(da[step] - db[step])
        return per_seed

    # equal-token paired slices: small vs each large arm on the SHARED step grid
    # (all v10 scale panels share batch*seq = 12*512, so equal steps == equal
    # tokens; the calibrated tokens/step of each panel is respected anyway)
    slices = {}
    _h_s = _panel_hists.get('P3SS') or {}
    _tps_s = (conf.get('P3SS') or {}).get('tokens_per_step')
    if _h_s and _tps_s:
        _common_steps = sorted(set.intersection(*[set(dict(_h_s['csa_fixed'][s])) for s in _h_s['csa_fixed']], *[set(dict(_h_s['full'][s])) for s in _h_s['full']])) if (_h_s['csa_fixed'] and _h_s['full']) else []
        for name in ('P3SL', 'P3SL_TOK', 'P3SL_FLOP'):
            _tps_l = (conf.get(name) or {}).get('tokens_per_step')
            if not _tps_l:
                continue
            for step in _common_steps:
                _g_s = _gap_at('P3SS', step)
                _g_l = _gap_at(name, step)
                common = sorted(set(_g_s) & set(_g_l))
                if not common:
                    continue
                dg = [_g_l[s] - _g_s[s] for s in common]
                res = V.exact_sign_permutation(dg)
                tokens = step * _tps_s
                slices[f'{name} - P3SS @ {tokens / 1000000.0:.2f}M tokens (gap_L - gap_S)'] = {'n': res['n'], 'mean': res['mean'], 'std': res.get('std', 0.0), 'p_exact_signflip': res['p_exact_signflip'], 'per_seed': res.get('per_seed'), 'p_floor': res.get('p_floor'), 'cohens_dz': res.get('cohens_dz'), 'seeds': common, 'gap_L': {str(s): _g_l[s] for s in common}, 'gap_S': {str(s): _g_s[s] for s in common}, 'note': 'positive = the CSA-dense gap is larger in the large arm AT THE SAME cumulative tokens'}
    return {'panels': conf, 'equal_token_slices': slices}
