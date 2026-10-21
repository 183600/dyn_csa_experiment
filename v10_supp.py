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
import v9_supp as V9
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
    if variant == 'full_rope':
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
    vocab = int(cfg.get('vocab') or P3MT_PAYLOAD['vocab'])
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
            **{k: _rm[k] for k in _keys}, **stats}

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
    _vocab_ck = int(d.get('vocab', 8192))
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
            _arch = (int(d.get('d', 256)), int(d.get('n_layers', 6)), int(d.get('n_heads', 8)), int(d.get('d_head', 32)))
            if _arch != (256, 6, 8, 32):
                print(f"[p3mp] REFUSE {v} s{seed}: checkpoint architecture {_arch} differs from the recipe this probe was validated against (256, 6, 8, 32) — not running the probe on unverified weights")
                out['status'] = 'skipped_arch'
                return out
            model = L.SmallGPT(_vocab_ck, *_arch, d.get('train_len', 512), [copy.deepcopy(c) for c in d['cfg']], mlp_ratio=d['mlp_ratio']).to(DEVICE)
            model.load_state_dict(d.pop('sd'))
            model.eval()
        for blk, c0 in zip(model.blocks, d['cfg']):
            ac = blk.attn.cfg
            ac.indexer_mode = 'random' if arm == 'randidx' else getattr(c0, 'indexer_mode', 'learned')
            ac.index_topk = 10 ** 6 if arm == 'allblocks' else getattr(c0, 'index_topk', 32)
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
            rec = {'variant': v, 'seed': seed, 'arm': arm, 'eval_len': int(Ln), 'task': 'niah_natural', 'condition': cond, 'acc': float(res['acc']), 'nll': float(res['nll']), 'n_queries': int(res['n']), 'n_correct': int(res['n_correct']), 'balanced_accuracy': res.get('balanced_accuracy'), 'macro_recall': res.get('macro_recall'), 'n_classes': int(res.get('n_classes') or 0), 'answer_support': res.get('answer_support'), 'per_class_recall': res.get('per_class_recall'), 'per_class': res.get('per_class'), 'chance_accuracy': res.get('chance_accuracy'), 'majority_accuracy': res.get('majority_accuracy'), 'majority_label': res.get('majority_label'), 'acc_lift_over_chance': res.get('acc_lift_over_chance'), 'acc_lift_over_majority': res.get('acc_lift_over_majority'), 'n_pairs': int(res.get('n_pairs', (_niah_cfg or {}).get('n_pairs', 8))), 'dist_mean': float(res.get('dist_mean', float('nan'))), 'dist_min': int(res.get('dist_min', -1)), 'dist_max': int(res.get('dist_max', -1)), 'rho': float(res.get('rho', 0.0)), 'val_fp': _val_fp, '_code': V.CKPT_CODE, 'probe_params': _niah_fp_of(cond), 'recipe': _ck_recipe}
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
            module = _par_ctx.get('module') or (_caller if (_caller := sys._getframe(1).f_globals.get('__name__')) and _caller != '__main__' else 'v10_supp')
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

PHASES = [('P3MT', 'lenphase', P3MT_PAYLOAD, [0, 1, 2]), ('P3MTC', 'lenphase', P3MT_PAYLOAD_CONTROLS, [0, 1, 2]), ('P3MP', 'probe', PROBE, None), ('P3T', 'run', V9.P2T_CFG, [2, 3]), ('P3SS', 'run', P3SS_CFG, [0, 1]), ('P3SL', 'run', P3SL_CFG, [0, 1]), ('P3SL2', 'run', P3SL_TOK_CFG, [0, 1]), ('P3SL3', 'run', P3SL_FLOP_CFG, [0, 1])]

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
            _ctx = dict(_ctx, module='v10_supp', ckpt_install=True, ckpt_flag=bool(_ctx.get('ckpt_flag')) or pname == 'P3T')
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

def v10_analysis(out='analysis_v10/stats.json'):
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    probe = {}
    sp = PROBE['summary']
    if os.path.exists(sp):
        try:
            raw = json.load(open(sp, encoding='utf-8'))
        except Exception as _e:
            print(f'[stats] FATAL: {sp} exists but cannot be parsed ({type(_e).__name__}: {_e}) — refusing to compute probe statistics from a truncated file; move it aside to re-probe.')
            raise
        _usable, _bad_cells = ({}, [])
        for _k, r in raw.items():
            if not isinstance(r, dict):
                _bad_cells.append((_k, 'not a dict'))
                continue
            if r.get('variant') is None or r.get('seed') is None:
                _bad_cells.append((_k, 'no variant/seed'))
                continue
            if not L.ppl_is_usable(r.get('ppl_mean')):
                _bad_cells.append((_k, 'ppl_mean is not a usable measurement'))
                continue
            if r.get('arm') is None or r.get('eval_len') is None or r.get('rho') is None:
                _bad_cells.append((_k, 'no arm/eval_len/rho'))
                continue
            _usable[_k] = r
        if _bad_cells:
            print(f'[v10 stats] {len(_bad_cells)} record(s) in {sp} carry no variant/seed/ppl_mean and are skipped (they are error or truncation stubs, not measurements): ' + ', '.join((f'{k}({why})' for k, why in _bad_cells[:4])))
        _n_bad_pre = len(_bad_cells)
        cells, _dup_cells = ({}, [])
        for k, r in _usable.items():
            if r.get('probe_params') is None:
                _bad_cells.append((k, 'no probe_params'))
                continue
            if r.get('synthesized'):
                _bad_cells.append((k, 'synthesized reconstruction is not an independent measurement'))
                continue
            if r.get('_code') != L.code_stamp():
                _bad_cells.append((k, 'stale code stamp'))
                continue
            if r.get('val_fp') in (None, ''):
                _bad_cells.append((k, 'no validation fingerprint'))
                continue
            _mode = str(r.get('mode', 'uniform'))
            # the grouping key carries the mode EXPLICITLY (probe_params' mode
            # field is normalised out by _fp_norm): uniform and natural cells
            # at the same rho must never merge into one pooled cell.  val_fp is
            # part of the identity: different evaluation corpora are not one
            # cell.
            key = (r['variant'], r['arm'], r['eval_len'], r['rho'], _mode, json.dumps(_fp_norm(r.get('probe_params')), sort_keys=True), str(r.get('_code')), str(r.get('recipe')), str(r.get('val_fp')))
            grp = cells.setdefault(key, {})
            if r['seed'] in grp:
                _dup_cells.append((k, key[:5], r['seed']))
                continue
            grp[r['seed']] = r
        if _dup_cells:
            print(f'[v10 stats] {len(_dup_cells)} probe record(s) share a (cell, seed) identity — their PPL is ambiguous and they are dropped from the pooling: ' + ', '.join((f'{k}' for k, _c, _s in _dup_cells[:4])))
        _new_bad = _bad_cells[_n_bad_pre:]
        if _new_bad:
            print(f'[v10 stats] {len(_new_bad)} record(s) in {sp} lack `probe_params` and are dropped from every mean/paired statistic: ' + ', '.join((f'{k}({why})' for k, why in _new_bad[:4])))
        probe_cells = []
        for (v, arm, Ln, rho, mode, _pp, _cd, _recipe, _vfp), by_seed in sorted(cells.items(), key=lambda kv: kv[0]):
            means = {s: r['ppl_mean'] for s, r in by_seed.items()}
            assert means, (v, arm, Ln, rho, mode)
            probe_cells.append({'variant': v, 'arm': arm, 'eval_len': Ln, 'rho': rho, 'mode': mode, 'seeds': sorted(means), 'ppl_by_seed': means, 'probe_params': next(iter(by_seed.values())).get('probe_params'), '_code': _cd, 'recipe': _recipe, 'val_fp': _vfp, 'mean': float(np.mean(list(means.values()))), 'n': len(means)})
        contrasts = {}

        # The grouped lists are exactly what a scan of probe_cells would have
        # collected, so the ambiguity/unambiguous outcomes (and the picked
        # cell) are unchanged.
        _cell_idx = {}
        for c in probe_cells:
            _cell_idx.setdefault((c['variant'], c['arm'], c['eval_len'], c['rho'], c['mode']), []).append(c)

        def cell_mean(v, arm, Ln, rho, mode='uniform'):
            hits = _cell_idx.get((v, arm, Ln, rho, mode), [])
            if len(hits) > 1:
                print(f'[v10 stats] ({v}/{arm}/L{Ln}/r{rho}/{mode}) holds {len(hits)} distinct probe parameterisations/recipes/code stamps — no unambiguous cell, so the contrast is omitted rather than pooled across them')
                return {}
            return hits[0] if hits else {}

        def _pairable(a, b, Ln, rho, tag, mode='uniform'):
            if _fp_norm(a.get('probe_params')) != _fp_norm(b.get('probe_params')):
                print(f'[v10 stats] L{Ln} r{rho} {mode} {tag}: the two arms were probed under DIFFERENT probe_params — pairing them would difference two different measurements, so the contrast is omitted')
                return False
            if a.get('_code') != b.get('_code') or _recipe_norm(a.get('recipe')) != _recipe_norm(b.get('recipe')):
                print(f"[v10 stats] L{Ln} r{rho} {mode} {tag}: the two arms carry DIFFERENT code/recipe stamps (learned {a.get('_code')}/{a.get('recipe')} vs {b.get('_code')}/{b.get('recipe')}) — pairing them would mix two measurement semantics, so the contrast is omitted; re-run the stale arm's phase to refresh it")
                return False
            return True

        def _add_contrast(name, a, b):
            common = sorted(set(a['ppl_by_seed']) & set(b['ppl_by_seed']))
            if len(common) >= 2:
                dl = [b['ppl_by_seed'][s] - a['ppl_by_seed'][s] for s in common]
                contrasts[name] = V.exact_sign_permutation(dl)
                contrasts[name]['seeds'] = common
        # ---- legacy uniform-mode contrasts (names unchanged for report compat) ----
        for Ln in PROBE['eval_lens']:
            for rho in PROBE['rhos']:
                learned = cell_mean('csa_fixed_rope', 'learned', Ln, rho, 'uniform')
                for arm, tag in (('dense', 'full_rope(dense)'), ('randidx', 'csa+randidx'), ('allblocks', 'csa+allblocks')):
                    other = cell_mean('full_rope', 'dense', Ln, rho, 'uniform') if arm == 'dense' else cell_mean('csa_fixed_rope', arm, Ln, rho, 'uniform')
                    if not learned or not other:
                        continue
                    if not _pairable(learned, other, Ln, rho, tag):
                        continue
                    _add_contrast(f'L{Ln} r{rho}: {tag} - csa+learned', learned, other)
        # ---- from-scratch selector controls ----
        for Ln in PROBE['eval_lens']:
            for rho in PROBE['rhos']:
                learned = cell_mean('csa_fixed_rope', 'learned', Ln, rho, 'uniform')
                if not learned:
                    continue
                for tag, ov, oarm in (('csa_randtrain', 'csa_fixed_rope_randidx', 'learned'), ('csa_alltrain', 'csa_fixed_rope_allblocks', 'learned')):
                    other = cell_mean(ov, oarm, Ln, rho, 'uniform')
                    if not other or not _pairable(learned, other, Ln, rho, tag):
                        continue
                    _add_contrast(f'L{Ln} r{rho}: {tag} - csa+learned', learned, other)
                # intervention-vs-retrain gap (NOT a causal
                # mismatch decomposition): test-time intervention minus the
                # trained-with-that-selection control at the same (L, rho).
                # For losses the positive direction means the intervention
                # scores WORSE than retraining that selection — a gap that MAY
                # itself contain train/test-mismatch effects (plus optimisation
                # adaptation and parameter differences), so it must be read as
                # a raw difference, never as 'the part that cannot be
                # attributed to mismatch'.
                for tag, tt_arm, sv in (('retrain-gap(random): csa+randidx - csa_randtrain', 'randidx', 'csa_fixed_rope_randidx'), ('retrain-gap(allblocks): csa+allblocks - csa_alltrain', 'allblocks', 'csa_fixed_rope_allblocks')):
                    tt = cell_mean('csa_fixed_rope', tt_arm, Ln, rho, 'uniform')
                    sc = cell_mean(sv, 'learned', Ln, rho, 'uniform')
                    if not tt or not sc or not _pairable(tt, sc, Ln, rho, tag):
                        continue
                    _add_contrast(f'L{Ln} r{rho}: {tag}', sc, tt)
                # architecture-aligned dense: full_rope minus
                # the all-blocks CSA twin (same projections/norm/RoPE/sink)
                dense = cell_mean('full_rope', 'dense', Ln, rho, 'uniform')
                allt = cell_mean('csa_fixed_rope_allblocks', 'learned', Ln, rho, 'uniform')
                if dense and allt and _pairable(dense, allt, Ln, rho, 'arch-aligned'):
                    _add_contrast(f'L{Ln} r{rho}: arch-aligned: full_rope - csa_alltrain', allt, dense)
        # ---- delta-NLL on the corruption modes ----
        delta_nll = {}

        def _nll_tok(rec):
            try:
                ns = [float(x) for x in (rec.get('nll_sum') or [])]
                nt = [int(x) for x in (rec.get('n_tok') or [])]
            except (TypeError, ValueError, OverflowError):
                return float('nan')
            tot = sum(nt)
            if not nt or tot <= 0:
                return float('nan')
            v = math.fsum(ns) / tot
            return v if math.isfinite(v) else float('nan')

        for (v, arm, Ln, rho, mode, _pp, _cd, _recipe, _vfp), by_seed in sorted(cells.items()):
            if mode == 'uniform' and abs(float(rho)) < 1e-12:
                continue
            # the clean baseline is the SAME arm's rho=0 uniform cell under the
            # same parameterisation/code/recipe; deltas are per-seed paired.
            # `span` carries an extra span_len in probe_params (it changes the
            # corruption geometry), but the clean ρ=0 cell does not.  Strip only
            # the mode-specific key for the lookup; the measured cells keep it.
            # `_pp` is the canonical JSON string used in the grouping key, so
            # parse it before normalising (a string is not a dict).
            _pp_clean = _pp
            if isinstance(_pp, str):
                try:
                    _pp_obj = json.loads(_pp)
                except (TypeError, ValueError, OverflowError):
                    _pp_obj = None
                if isinstance(_pp_obj, dict):
                    _pp_clean = json.dumps(_clean_probe_params(_pp_obj), sort_keys=True)
            clean_key = (v, arm, Ln, 0.0, 'uniform', _pp_clean, _cd, _recipe, _vfp)
            clean_grp = cells.get(clean_key) or {}
            if not clean_grp:
                continue
            per_seed = {}
            for s, r in by_seed.items():
                cr = clean_grp.get(s)
                if cr is None:
                    continue
                a, b = (_nll_tok(r), _nll_tok(cr))
                if math.isfinite(a) and math.isfinite(b):
                    per_seed[s] = a - b
            if len(per_seed) >= 1:
                means = {s: per_seed[s] for s in sorted(per_seed)}
                _dl = list(means.values())
                entry = {'variant': v, 'arm': arm, 'eval_len': Ln, 'rho': rho, 'mode': mode, 'n': len(_dl), 'mean': float(np.mean(_dl)), 'std': float(np.std(_dl, ddof=1)) if len(_dl) > 1 else 0.0, 'per_seed': means, 'clean_nll_by_seed': {s: _nll_tok(r) for s, r in clean_grp.items() if math.isfinite(_nll_tok(r))}, 'note': 'delta = NLL_corrupted - NLL_clean, nats per target token'}
                delta_nll[f'{v}::{arm}::L{Ln}::{mode}::r{rho}'] = entry
        # paired delta-NLL contrasts between arms at the same (L, mode, rho)
        for Ln in PROBE['eval_lens']:
            _modes = (PROBE.get('modes') or {}).keys() or ['uniform']
            for mode in [m for m in _modes if m != 'uniform'] + ['uniform']:
                _rhos = ((PROBE.get('modes') or {}).get(mode) or ([r for r in PROBE['rhos'] if r > 0] if mode == 'uniform' else []))
                for rho in [r for r in _rhos if r > 0]:
                    learned = delta_nll.get(f'csa_fixed_rope::learned::L{Ln}::{mode}::r{rho}')
                    for arm, tag in (('dense', 'full_rope::dense'), ('randidx', 'csa_fixed_rope::randidx'), ('allblocks', 'csa_fixed_rope::allblocks')):
                        other = delta_nll.get(f'{tag}::L{Ln}::{mode}::r{rho}')
                        if not learned or not other:
                            continue
                        common = sorted(set(learned['per_seed']) & set(other['per_seed']))
                        if len(common) >= 2:
                            dl = [other['per_seed'][s] - learned['per_seed'][s] for s in common]
                            res = V.exact_sign_permutation(dl)
                            res['seeds'] = common
                            contrasts[f'ΔNLL L{Ln} {mode} r{rho}: {arm} - csa+learned'] = res
        # ---- Holm correction over the whole contrast family ----
        holm = {}
        _fam = [(k, v.get('p_exact_signflip')) for k, v in contrasts.items() if isinstance(v.get('p_exact_signflip'), (int, float)) and math.isfinite(v.get('p_exact_signflip'))]
        if _fam:
            holm = L.holm_adjust(_fam)
            for _k, _v in contrasts.items():
                if isinstance(_v, dict):
                    _v['p_holm'] = (holm.get(_k) or {}).get('p_holm') if _k in holm else None
            _rej = [k for k, v in holm.items() if v.get('reject')]
            print(f'[v10 stats] Holm step-down over the {len(_fam)}-contrast family (EXPLORATORY — not pre-registered): {len(_rej)} survive at alpha=0.05' + (f': {sorted(_rej)[:4]}' if _rej else ''))
        probe = {'cells': probe_cells, 'contrasts': contrasts, 'delta_nll': delta_nll, 'holm_exploratory': holm}
        # ---- NIAH-natural (far-necessity task) ----
        _nsp = (PROBE.get('niah') or {}).get('summary')
        if _nsp and os.path.exists(_nsp):
            try:
                nraw = json.load(open(_nsp, encoding='utf-8'))
            except Exception as _e:
                print(f'[stats] WARNING: {_nsp} exists but cannot be parsed ({type(_e).__name__}: {_e}) — the NIAH-natural panel is skipped')
                nraw = {}
            ncells = {}
            for _k, r in nraw.items():
                if not isinstance(r, dict) or r.get('task') != 'niah_natural':
                    continue
                if r.get('variant') is None or r.get('seed') is None or r.get('arm') is None or r.get('condition') is None:
                    continue
                if r.get('synthesized') or r.get('_code') != L.code_stamp() or r.get('val_fp') in (None, ''):
                    continue
                nkey = (r['variant'], r['arm'], r['eval_len'], r['condition'], json.dumps(_fp_norm(r.get('probe_params')), sort_keys=True), str(r.get('_code')), str(r.get('recipe')), str(r.get('val_fp')))
                grp = ncells.setdefault(nkey, {})
                if r['seed'] in grp:
                    continue
                grp[r['seed']] = r
            niah_cells = []
            for (v, arm, Ln, cond, _pp, _cd, _recipe, _vfp), by_seed in sorted(ncells.items(), key=lambda kv: kv[0]):
                accs = {s: r['acc'] for s, r in by_seed.items()}
                nlls = {s: r['nll'] for s, r in by_seed.items()}
                bals = {s: r.get('balanced_accuracy') for s, r in by_seed.items()}
                _first = next(iter(by_seed.values()))
                _bal_vals = [float(x) for x in bals.values()
                             if isinstance(x, (int, float)) and not isinstance(x, bool)
                             and math.isfinite(float(x))]
                niah_cells.append({'variant': v, 'arm': arm, 'eval_len': Ln, 'condition': cond, 'seeds': sorted(accs), 'acc_by_seed': accs, 'nll_by_seed': nlls, 'balanced_by_seed': bals, 'acc_mean': float(np.mean(list(accs.values()))), 'balanced_mean': (float(np.mean(_bal_vals)) if _bal_vals else None), 'nll_mean': float(np.mean(list(nlls.values()))), 'chance_accuracy': _first.get('chance_accuracy'), 'majority_accuracy': _first.get('majority_accuracy'), 'n_classes': _first.get('n_classes'), 'n': len(accs), 'n_pairs': _first.get('n_pairs'), 'dist_mean': _first.get('dist_mean'), 'val_fp': _vfp})
            niah_contrasts = {}
            _nidx = {}
            for c in niah_cells:
                _nidx.setdefault((c['variant'], c['arm'], c['eval_len'], c['condition']), []).append(c)

            def _ncell(v, arm, Ln, cond):
                hits = _nidx.get((v, arm, Ln, cond), [])
                return hits[0] if len(hits) == 1 else {}
            for Ln in PROBE['eval_lens']:
                for cond in (PROBE.get('niah') or {}).get('conditions', ['clean', 'distract']):
                    learned = _ncell('csa_fixed_rope', 'learned', Ln, cond)
                    for arm, tag in (('dense', 'full_rope'), ('randidx', 'csa_fixed_rope'), ('allblocks', 'csa_fixed_rope')):
                        other = _ncell('full_rope', 'dense', Ln, cond) if arm == 'dense' else _ncell(tag, arm, Ln, cond)
                        if not learned or not other:
                            continue
                        common = sorted(set(learned['acc_by_seed']) & set(other['acc_by_seed']))
                        if len(common) >= 2:
                            res = V.exact_sign_permutation([other['nll_by_seed'][s] - learned['nll_by_seed'][s] for s in common])
                            res['seeds'] = common
                            niah_contrasts[f'NIAH-nat L{Ln} {cond}: {arm} - learned (answer NLL)'] = res
                    for tag, ov in (('csa_randtrain', 'csa_fixed_rope_randidx'), ('csa_alltrain', 'csa_fixed_rope_allblocks')):
                        other = _ncell(ov, 'learned', Ln, cond)
                        if not learned or not other:
                            continue
                        common = sorted(set(learned['nll_by_seed']) & set(other['nll_by_seed']))
                        if len(common) >= 2:
                            res = V.exact_sign_permutation([other['nll_by_seed'][s] - learned['nll_by_seed'][s] for s in common])
                            res['seeds'] = common
                            niah_contrasts[f'NIAH-nat L{Ln} {cond}: {tag} - learned (answer NLL)'] = res
            probe['niah_natural'] = {'cells': niah_cells, 'contrasts': niah_contrasts}
            # NIAH contrasts are built AFTER the corruption family was
            # corrected.  Leaving them out of the Holm family would report
            # uncorrected p-values under a "Holm" heading, so re-run the
            # correction over the union and overwrite both.
            _comb = dict(contrasts)
            _comb.update(niah_contrasts)
            if _comb:
                _h2 = L.holm_adjust([(k, v.get('p_exact_signflip')) for k, v in _comb.items()
                                     if isinstance(v.get('p_exact_signflip'), (int, float)) and math.isfinite(v.get('p_exact_signflip'))])
                for _k, _v in _comb.items():
                    if isinstance(_v, dict):
                        _e2 = _h2.get(_k)
                        _v['p_holm'] = (_e2.get('p_holm') if isinstance(_e2, dict) else None)
                holm = _h2
                probe['holm_exploratory'] = _h2
                probe['holm_family_note'] = ('Holm over the combined corruption + NIAH contrast families')
                print(f'[v10 stats] Holm re-attached over the combined {len(_comb)}-contrast family (corruption + NIAH)')
    xo = _scale_crossovers()

    def _is_recon(v):
        flags = v.get('per_seed_synth', {})
        return any((flags.get(str(s)) for s in v.get('seeds', [])))
    fit_entries = {}
    for _lab, _v in xo.items():
        if not isinstance(_v, dict):
            continue
        fit_entries[_lab] = {'label': _lab, 'd': _v.get('d'),
                             'crossover_tokens': _v.get('mean_crossover_tokens'),
                             'train_tokens': _v.get('train_tokens'),
                             'censored': bool(_v.get('censored') or _v.get('mean_crossed') is False),
                             'reconstructed': bool(_is_recon(_v))}
    # One point per width, no censored subset, comparable
    # training volume, at least 4 widths, and a slope WITH CI.  The helper
    # refuses instead of fitting whenever those conditions fail.
    fit = L.crossover_scaling_fit(fit_entries, min_points=4)
    for _why in (fit.get('reasons') or []):
        print(f'[stats] crossover fit refused: {_why}')
    if fit.get('deduped_by_d'):
        print(f'[stats] crossover fit: duplicate widths dropped (token-matched panel kept): {fit["deduped_by_d"]}')
    seq2k = _ppl_by_seed('results_lm_v9_seq2k', full=True)
    comparisons = {}

    def add(name, panel, a, b, outdir):
        V9.add_paired(comparisons, name, panel, a, b, who='v10 ', outdir=outdir)
    add('seq2k(bs1): topk8 - m1', seq2k, 'csa_fixed_topk8', 'csa_fix_m1', 'results_lm_v9_seq2k')
    add('seq2k(bs1): topk512 - m1', seq2k, 'csa_fixed_topk512', 'csa_fix_m1', 'results_lm_v9_seq2k')
    add('seq2k(bs1): topk8 - topk512', seq2k, 'csa_fixed_topk8', 'csa_fixed_topk512', 'results_lm_v9_seq2k')
    panel = V8._panel_block
    scale_conf = _equal_token_slices()
    out_d = {'probe': probe, 'crossover_panels': xo, 'crossover_fit': fit, 'scale_confounds': scale_conf, 'seq2k_bs1_panel': panel(seq2k), 'comparisons': comparisons, 'holm': L.attach_holm(comparisons, family='v10 seq2k comparisons')}
    out_d = L.stamped_analysis('v10_analysis', out_d, outdir=(os.path.dirname(out) or '.'), name=os.path.basename(out))
    print(f'[v10 stats] wrote {out}')
    for k, v in comparisons.items():
        if 'omitted' in v:
            print(f'  {k:38s} omitted — {v['omitted']}')
            continue
        print(f'  {k:38s} Δ={v['mean']:+7.2f} ± {v.get('std', 0):5.2f}  p(sign-flip)={v['p_exact_signflip']:.3f}  n={v['n']}')
    for label, v in xo.items():
        if v.get('status') == 'ok':
            if not v['mean_crossed']:
                _stps = v['mean_traj']['steps']
                if _stps:
                    print(f'  crossover {label:10s} mean-traj: CENSORED (> {_stps[-1]:g} steps)')
                else:
                    print(f'  crossover {label:10s} mean-traj: no eval step shared by every seed — no censored bound')
            elif 'mean_crossover_tokens' in v:
                print(f'  crossover {label:10s} mean-traj: step {v['mean_crossover_step']:g} ({v['mean_crossover_tokens']:g} tokens)')
            else:
                print(f'  crossover {label:10s} mean-traj: step {v['mean_crossover_step']:g} (tokens/step unknown)')
    return out_d

def _fmt_pm(cell, std=None):
    if isinstance(cell, dict):
        if cell.get('mean') is None:
            return '—'
        mean, std = (cell['mean'], cell.get('std', 0.0))
    else:
        mean = cell
    return f'{mean:.2f} ±{std or 0.0:.2f}'

def build_report(out='REPORT_v10.md', stats_p='analysis_v10/stats.json'):
    v10_analysis(out=stats_p)
    st = json.load(open(stats_p, encoding='utf-8'))
    probe = st['probe']
    xo = st['crossover_panels']
    fit = st['crossover_fit']
    seq2k = st['seq2k_bs1_panel']
    comp = st['comparisons']
    mech_sum = {}
    mp = os.path.join(P3MT_PAYLOAD['outdir'], 'summary.json')
    if os.path.exists(mp):
        mech_sum = json.load(open(mp, encoding='utf-8'))
    lines = []
    A = lines.append
    A('# CSA / HCA 受控机制研究 — v10 补实验报告（外推机制 + 反转点标度 + topk 补种子）')
    A('> **证据等级：探索性（exploratory）**。本面板的比较/种子是观察 validation 结果后选择的；Holm 校正只限制过度宣称，不等于预注册确认。')
    A('')
    A('')
    A(f'> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}')
    A('> 参考论文：arXiv:2606.19348。**范围声明**：本报告是小规模 WikiText-103 上的受控研究，不是 DeepSeek-V4 的全量复现，也不支持对其规模、语料或生产系统做外推。')
    A('> 说明：本报告全部数字由 `v10_supp.py report` 从 `results_*/`、`analysis_v10/` 的落盘产物计算得到，无手填数值。')
    A('')

    A('---')
    A('')
    A('## P3M 长度外推优势的机制验证（远距干扰注入）')
    A('')
    A('**背景**：v7 P1L 的外推对比只是附带观察（n=3 时精确符号翻转 p 值下限 0.250）。P1L 权重未持久化，P3MT 先按 v7 P1L 原配方（seq 512、3000 步、3 seeds、AdamW lr 3e-4）重训两臂并保存权重——同时作为该外推对比的独立复现。')
    A('')
    if mech_sum:
        _rows_by_v = {}
        _stale_vars = set()
        for v in list(P3MT_PAYLOAD['variants']) + SCRATCH_CONTROLS:
            _rv = [r for r in mech_sum.values() if isinstance(r, dict) and r.get('variant') == v and (r.get('seed') is not None) and ('by_len' in r)]
            _cur = [r for r in _rv if r.get('_code') == L.code_stamp()]
            if _cur:
                if len(_cur) < len(_rv):
                    print(f'[v10 report] {v}: {len(_rv) - len(_cur)} by-length record(s) predate the current code semantics — the table uses only the {len(_cur)} current one(s)')
                _rv = _cur
            elif _rv:
                # Fail closed: a panel with no record under
                # the current code/recipe stamp is excluded from the table and
                # from every conclusion, not quoted with a warning marker.
                print(f'[v10 report] {v}: no by-length record carries the current code semantics ({L.code_stamp()}) — EXCLUDED from the table and conclusions ({len(_rv)} stale record(s) left on disk)')
                _stale_vars.add(v)
                _rv = []
            _rows_by_v[v] = _rv
        n_by_v = {v: {r['seed'] for r in _rv} for v, _rv in _rows_by_v.items() if _rv}
        if len(n_by_v) > 1:
            _common_seeds = set.intersection(*n_by_v.values())
            if len({frozenset(s) for s in n_by_v.values()}) > 1:
                if _common_seeds:
                    print(f'[v10 report] by-length panel is ragged ({ {v: sorted(s) for v, s in n_by_v.items()} }) — restricting the table to the {len(_common_seeds)}-seed intersection {sorted(_common_seeds)} so every column shares one seed set')
                    for _v in list(_rows_by_v):
                        _rows_by_v[_v] = [r for r in _rows_by_v[_v] if r['seed'] in _common_seeds]
                    n_by_v = {v: {r['seed'] for r in _rv} for v, _rv in _rows_by_v.items() if _rv}
                else:
                    print(f'[v10 report] WARNING: by-length arms share NO seed ({ {v: sorted(s) for v, s in n_by_v.items()} }) — the table below quotes per-arm means over DIFFERENT seed sets and is not a paired comparison')
        A(f'**P3MT 复现对照**（train@512 → 同权重 eval；ratio = PPL@L / PPL@512，seed 平均，n={(min((len(s) for s in n_by_v.values())) if n_by_v else 0)}）：')
        A('')
        A('| variant | PPL@512 | PPL@2048 | PPL@4096 | ratio@4096 |')
        A('|---|---|---|---|---|')
        for v in P3MT_PAYLOAD['variants']:
            rows = _rows_by_v.get(v, [])
            if not rows:
                continue

            def _at(Ln, rows=rows):
                cells = L.by_len_cells(rows, Ln)
                ok = [c for c in cells if isinstance(c.get('truncated'), (bool, np.bool_)) and not bool(c.get('truncated'))]
                if (ok if ok else cells) and len(ok if ok else cells) < len(rows):
                    print(f'[v10 report] {v} L{Ln}: the column mean covers {len(ok if ok else cells)}/{len(rows)} seeds — the rest hold no usable cell at this length (error/truncated cell), so the column is a sub-seed mean')
                if ok:
                    return (float(np.mean([c['ppl'] for c in ok])), False)
                return (float(np.mean([c['ppl'] for c in cells])), True) if cells else (float('nan'), False)

            def _paired_ratio(rows=rows):
                m512, m4096 = ({}, {})
                for r in rows:
                    bl = r.get('by_len')
                    if not isinstance(bl, dict):
                        continue
                    for Ln, mp in ((512, m512), (4096, m4096)):
                        c = bl.get(Ln, bl.get(str(Ln)))
                        if isinstance(c, dict) and L.ppl_is_usable(c.get('ppl')) and isinstance(c.get('truncated'), (bool, np.bool_)) and not bool(c.get('truncated')):
                            mp[r.get('seed')] = float(c['ppl'])
                both = sorted(set(m512) & set(m4096))
                if both and len(both) < len(rows):
                    print(f'[v10 report] {v} ratio@4096: computed over {len(both)}/{len(rows)} seeds — the rest lack a usable cell at 512 or 4096')
                if not both:
                    return None
                return float(np.mean([m4096[s] / m512[s] for s in both]))

            def _fmt_cell(Ln):
                val, tr = _at(Ln)
                return f'{val:.1f}~' if tr else f'{val:.1f}'
            p512, _512tr = _at(512)
            p4096, _4096tr = _at(4096)
            if _512tr or _4096tr:
                _ratio = '—（含截断 cell，不可比）'
            else:
                _pr = _paired_ratio()
                _ratio = f'×{_pr:.2f}' if _pr is not None and np.isfinite(_pr) else '—'
            A(f'| `{v}` | {_fmt_cell(512)} | {_fmt_cell(2048)} | {_fmt_cell(4096)} | {_ratio} |')
        A('')
        A('> 后缀 `~` 表示该列**只有位置受限（abs-PE）的截断 cell**：它是在比列名更短的 span 上评出来的分数，**不是**长上下文测量值。任何 `~` 行不可用于「外推是否稳健」的结论。')
        A('')
        if _stale_vars:
            A('> **⚠ 陈旧记录已排除**：' + '、'.join(f'`{x}`' for x in sorted(_stale_vars)) +
              f' 没有任何按长度记录携带当前代码语义戳 `{L.code_stamp()}`；按「旧代码/旧配方/旧数据指纹不得进入主要结论」的规则，'
              '本表不引用这些数值，也不据此计算 ratio 或外推结论。重跑对应训练阶段后再纳入。')
            A('')
    A(f'**P3MP 干扰注入设计**：eval 长度 2048/4096；把**远距上下文**（目标区之前的全部输入位置，即除最后 {PROBE['target']} 个干净目标 token 及其紧邻前置位置以外的位置）中比例为 ρ ∈ {{0, 1/8, 1/4, 1/2}} 的 token 替换为均匀随机 token——同一 (序列, ρ) 的腐坏对所有臂逐字节相同（按 (eval_len, 序列号, ρ) 播种，与臂/模型种子无关），臂间严格配对。只在干净目标区计 PPL。四个臂：')
    A('')
    A('- `full_rope`（dense，必须直面远距噪声）')
    A('- `csa_fixed_rope` 学习选择（原机制）')
    A('- `csa_fixed_rope` + 随机 indexer（选择被打乱，v4 消融开关，eval 时注入）')
    A('- `csa_fixed_rope` + topk=全块（看得见一切，≈ 带压缩的 dense）')
    A('')
    A('「稀疏掩码滤除远距噪声」假设预测：**学习选择臂的 ρ 曲线平坦，其余三臂随 ρ 陡峭退化**。')
    A('')
    cells = probe.get('cells', [])
    if cells:
        lens = sorted({c['eval_len'] for c in cells})
        rhos = PROBE['rhos']
        arm_order = [('full_rope', 'dense'), ('csa_fixed_rope', 'learned'), ('csa_fixed_rope', 'randidx'), ('csa_fixed_rope', 'allblocks'), ('csa_fixed_rope_randidx', 'learned'), ('csa_fixed_rope_allblocks', 'learned')]
        for Ln in lens:
            _n_ln = max([c.get('n', 0) for c in cells if c['eval_len'] == Ln] or [0])
            A(f'**目标区 PPL（eval_len={Ln}，至多 {_n_ln or "?"} seeds 平均；uniform 腐坏模式）**：')
            A('')
            A('| 臂 | ' + ' | '.join((f'ρ={r:g}' for r in rhos)) + ' |')
            A('|---|' + '---|' * len(rhos))
            for v, arm in arm_order:
                row = []
                for rho in rhos:
                    m = [c['mean'] for c in cells if (c['variant'], c['arm'], c['eval_len'], c['rho'], c.get('mode', 'uniform')) == (v, arm, Ln, rho, 'uniform')]
                    row.append(f'{m[0]:.2f}' if len(m) == 1 else '—')
                    if len(m) > 1:
                        print(f'[report] {v}+{arm} L={Ln} rho={rho}: {len(m)} parameterisations of the cell exist — the table omits the cell (the paired test refuses it too)')
                A(f'| `{v}`+{arm} | ' + ' | '.join(row) + ' |')
            A('')
    contr = probe.get('contrasts', {})
    _legacy_keys = [k for k in contr if ' - csa+learned' in k and any((t in k for t in ('full_rope(dense)', 'csa+randidx', 'csa+allblocks'))) and not k.startswith('ΔNLL')]
    _legacy = {k: contr[k] for k in _legacy_keys}
    if _legacy:
        A('**配对检验（exact sign-flip，Δ = 对照臂 − 学习选择臂，>0 表示学习选择更抗噪；uniform 模式）：**')
        A('')
        A('| 比较 | n | Δ mean±std | p (exact) | p_Holm |')
        A('|---|---|---|---|---|')
        for k, v in _legacy.items():
            A(f'| {k} | {v['n']} | {v['mean']:+.2f} ± {v.get('std', 0):.2f} | {v['p_exact_signflip']:.3f} | {v.get('p_holm') if v.get('p_holm') is not None else '—'} |')
        A('')

        def _rng(tag):
            _lo, _hi, _nonf = L.finite_range(((k, v.get('mean')) for k, v in _legacy.items() if tag in k))
            if _nonf:
                print(f'[v10 report] WARNING: {len(_nonf)} contrast(s) in {tag!r} carry a non-finite mean and are excluded from the reported range: {sorted(_nonf)[:3]}')
            return None if _lo is None else (_lo, _hi)

        def _fmt(r, nd=1):
            return 'n/a' if not r else f'{r[0]:+.{nd}f}~{r[1]:+.{nd}f}'

        def _all_sig(tag, alpha=0.05):
            vs = [(k, v) for k, v in _legacy.items() if tag in k]
            if not vs:
                return False
            # Direction, raw p, Holm p, n floor and |mean|>std must ALL hold:
            # the raw-p version could call a resolution-limited or noise-level
            # contrast "strong evidence" while the table printed p_holm=1.0.
            return all((v.get('mean', 0) > 0 and
                        L.reportable_claim(v, family_size=len(_legacy), alpha=alpha,
                                           min_n=6, require_holm=True)['allowed'])
                       for _k, v in vs)
        dn, ab, ri = (_rng('dense'), _rng('allblocks'), _rng('randidx'))
        ri_sig = _all_sig('randidx')
        _n_by_c = [(k, v.get('n', 0)) for k, v in _legacy.items() if 'omitted' not in v]
        n_min = min((n for _k, n in _n_by_c), default=0)
        _bind = [k for k, n in _n_by_c if n == n_min]
        n_min_ri = min((n for k, n in _n_by_c if 'randidx' in k), default=0)
        p_floor = L.signflip_p_floor(n_min) if n_min else float('nan')
        learned_helps = ri_sig
        _dpos = bool(dn) and dn[0] > 0 and bool(ab) and ab[0] > 0
        _dtxt = '正值表示「看得见远距噪声就会受害」，支持限制远距注意力暴露带来抗噪性' if _dpos else '区间并非全正，「远距暴露致害」的方向在本轮产物上不成立，以配对数字为准'
        A(f'**结论（区间实时算自上方配对检验，Δ = 对照臂 − 学习选择臂）：**(1) dense−learned Δ 区间 {_fmt(dn)} PPL，allblocks−learned Δ 区间 {_fmt(ab)} PPL——{_dtxt}；(2) randidx−learned Δ 区间 {_fmt(ri)} PPL——' + ('全为正值且全部对照达到 p<0.05：学习到的检索对抗噪性有**独立贡献**（强版本成立）。' if learned_helps else f'未能在 p<0.05 上成立（randidx 对照的最小 n={n_min_ri}，全表最小 n={n_min}[{', '.join(sorted(_bind)[:2])}{('…' if len(_bind) > 2 else '')}]，精确符号翻转的 p 下限为 {p_floor:.3f}，' + ('该种子数下无法达到 0.05 显著性' if not (p_floor <= 0.05) else '分辨率已足以达到 0.05，未显著是数据结论而非分辨率上限') + '），「抗噪性来自学习到的检索质量」的强版本**不成立**；现有数据只支持「抗噪性主要来自稀疏归纳偏置本身」这一较弱表述。') + '）')
        A('')
    # ---- P3M-2：从头训练的选择器对照 ----
    _new_c = {k: v for k, v in contr.items() if k not in _legacy and not k.startswith('ΔNLL')}
    if _new_c:
        A('### P3M-2 从头训练的选择器对照（同规模）')
        A('')
        A('**背景**：测试时把已训练模型的选择器替换为随机/全块，只能回答「模型是否依赖原选择器」，不能回答「学习式选择是否优于训练充分的随机选择或全块读取」——后两者的测试时干预改变了模型训练时适应的输入分布，效果下降可能来自失配本身。P3MTC 按与 P3MT **完全相同的配方**（3000 步、同数据、同种子）从头训练两个对照：`csa_fixed_rope_randidx`（训练期选择即随机）与 `csa_fixed_rope_allblocks`（训练期就读全部块；与 `csa_fixed_rope` 逐项同构——同 c_kv 压缩、同 qk_norm、同 RoPE、同 sink/窗口，仅选择方式不同，因此它同时是**逐项对齐的 dense 对照**）。')
        A('')
        A('| 比较 | n | Δ mean±std | d_z | p (exact) | p 下限 |')
        A('|---|---|---|---|---|---|')
        for k, v in sorted(_new_c.items()):
            A(f'| {k} | {v['n']} | {v['mean']:+.2f} ± {v.get('std', 0):.2f} | {v.get('cohens_dz', float('nan')):+.2f} | {v['p_exact_signflip']:.3f} | {v.get('p_floor', float('nan')):.3f} |')
        A('')
        A('> `retrain-gap(…)` 行 = 测试时干预 − 同规模从头训练对照（同 L、同 ρ）。对越低越好的损失，正值说明干预比重新训练**更差**，该差距本身**可能包含**训练—测试失配的代价，也混合了优化适应与参数差异——它是「干预与重新训练的效果差」，**不是**已完成的因果失配分解，不得把简单相减解释为「不能归因于失配的部分」。`arch-aligned` 行 = dense − 全块 CSA 孪生（两者仅剩 KV 压缩投影与分块读取方式之差）。')
        A('')
    # ---- ΔNLL：污染增量 ----
    dnll = probe.get('delta_nll') or {}
    if dnll:
        A('### P3M-3 污染增量 ΔNLL（按 NLL 口径，含干净条件基线）')
        A('')
        A('**口径**：ΔNLL = NLL_污染 − NLL_干净（nats/目标 token，逐种子配对）。腐坏模式：`uniform`（均匀随机 token，既有协议）、`natural`（其他真实序列同位置 token）、`mask`（远端整体屏蔽为 `<pad>`）、`delete`（远端整体删除，仅剩目标区）。`mask`/`delete` 回答「干净远端上下文到底带来多少收益」；`natural` 与 `uniform` 之差只回答「真实语料同位置 token 替换是否比均匀随机替换更难过滤」；它不是主题干扰、语义相近干扰、重复文本或对抗干扰，不能据此声称普遍抗噪性。')
        A('')
        A('| 臂 | L | 模式 | ρ | ΔNLL mean±std | n | 干净 NLL |')
        A('|---|---|---|---|---|---|---|')
        _clean_nll_cache = {}
        for k in sorted(dnll):
            e = dnll[k]
            _ck = f"{e['variant']}::{e['arm']}::L{e['eval_len']}"
            if _ck not in _clean_nll_cache:
                _cl = [x for x in dnll.values() if x['variant'] == e['variant'] and x['arm'] == e['arm'] and x['eval_len'] == e['eval_len'] and x['mode'] == 'uniform' and abs(float(x['rho'])) < 1e-12]
                _clean_nll_cache[_ck] = float(np.mean([np.mean(list(x.get('clean_nll_by_seed', {}).values())) for x in _cl])) if _cl else float('nan')
            A(f"| `{e['variant']}::{e['arm']}` | {e['eval_len']} | {e['mode']} | {e['rho']:g} | {e['mean']:+.4f} ± {e.get('std', 0):.4f} | {e['n']} | {_clean_nll_cache[_ck]:.4f} |")
        A('')
        _dnll_c = {k: v for k, v in contr.items() if k.startswith('ΔNLL')}
        if _dnll_c:
            A('| ΔNLL 配对比较（ΔΔNLL = 臂间污染增量之差） | n | Δ mean±std | p (exact) |')
            A('|---|---|---|---|')
            for k, v in sorted(_dnll_c.items()):
                A(f'| {k} | {v['n']} | {v['mean']:+.4f} ± {v.get('std', 0):.4f} | {v['p_exact_signflip']:.3f} |')
            A('')
        A('> **替代解释提示**：某臂 ΔNLL 小也可能因为它本来就很少利用远端上下文（干净 NLL 更高）——读表时必须把 ΔNLL 与干净 NLL 列一起看；`mask`/`delete` 行给出干净远端的收益幅度作为参照系。')
        A('')
    # ---- NIAH-natural：远端必要信息任务 ----
    niah = probe.get('niah_natural') or {}
    if niah.get('cells'):
        A('### P3M-4 远端必要信息任务（NIAH-natural，自然文本 + 自然干扰）')
        A('')
        A('**设计**：序列 = 远端自然文本（内嵌 8 个 key→value 耦合）+ 目标区（自然文本收尾 + 8 个查询 key）。查询 key 的下一个 token 的正确答案只在远端出现——正确预测**必须**利用远距信息；`distract` 条件把 ρ=0.5 的远端草料换成其他真实序列的 token（realistic-token 替换；不是主题/语义干扰），但**绝不腐坏耦合位置**。报告 answer top-1 准确率与 answer NLL（已按各臂 clean/distract 成对）。')
        A('')
        A('| 臂 | L | 条件 | acc (微) | macro-recall | 均匀 chance | 多数类 floor | answer NLL | n |')
        A('|---|---|---|---|---|---|---|---|---|')
        for c in sorted(niah['cells'], key=lambda c: (c['eval_len'], c['condition'], c['variant'], c['arm'])):
            _bal = (f"{c['balanced_mean']:.3f}" if c.get('balanced_mean') is not None else 'n/a')
            _ch = (f"{c['chance_accuracy']:.3f}" if isinstance(c.get('chance_accuracy'), (int, float)) else 'n/a')
            _mj = (f"{c['majority_accuracy']:.3f}" if isinstance(c.get('majority_accuracy'), (int, float)) else 'n/a')
            A(f"| `{c['variant']}::{c['arm']}` | {c['eval_len']} | {c['condition']} | {c['acc_mean']:.3f} | {_bal} | {_ch} | {_mj} | {c['nll_mean']:.3f} | {c['n']} |")
        A('')
        if niah.get('contrasts'):
            A('| NIAH 配对比较（Δ = 对照臂 − 学习选择，answer NLL，>0 = 学习选择更好） | n | Δ mean±std | p (exact) |')
            A('|---|---|---|---|')
            for k, v in sorted(niah['contrasts'].items()):
                A(f'| {k} | {v['n']} | {v['mean']:+.3f} ± {v.get('std', 0):.3f} | {v['p_exact_signflip']:.3f} |')
            A('')
        A('> 该任务直接检验「长距离信息**利用**能力」，与上面的干扰鲁棒性互补：鲁棒性高 + 检索利用不差，才支持「稀疏选择在滤噪的同时保留了有用远距信息」；若只有鲁棒性而检索差，则更符合「本来就少用远端」的替代解释。')
        A('')
        A('> **平凡基线**：`均匀 chance` = 1 / 实际出现的答案类别数；`多数类 floor` = 总是预测出现最多的答案 token。`acc (微)` 会被高频答案主导，`macro-recall` 对每个答案类别等权；读检索收益时必须三列一起看。')
        A('')
    A('### 统计口径')
    A('')
    _n_all = sorted({v.get('n', 0) for v in contr.values() if isinstance(v, dict) and 'omitted' not in v})
    A(f'- 每个配对比较都给出逐种子 Δ、效应量 d_z 与该 n 下的双侧精确符号翻转 p 值下限 2/2^n（本表 n 集合 {_n_all}）。')
    A('- 上表全部 p 值按 **Holm step-down** 在本报告的对比族内做了多重校正（`analysis_v10/stats.json` 的 `probe.holm_exploratory`）；该校正族**未预先注册**，因此校正后的 p 只能防止过度宣称，不能当作确认性证据。')
    A('- v7→v11 的多轮追加实验与既有面板属于**探索性**证据；确认性检验见 `v12_supp.py`（冻结清单 `confirmation_frozen.json`：固定主比较、终点、种子数，用 wiki.test.raw 独立 test split 一次性评估）。')
    A('')
    A('---')
    A('')
    A('## P3S 反转交叉点的标度读数（4 个模型规模）')
    A('')
    A('**判据（预注册）**：对每个规模的 `csa_fixed − full` 验证 PPL 差距轨迹（eval_every=250 或面板既有网格），取 ±1 点平滑后**首次转正且此后不再回穿**的 eval 步为交叉步；轨迹内未交叉记为删失（> 末步）。token 数 = 步数 × 每步 token。')
    A('')
    A('| 规模 | 面板 | seeds | 交叉步（seed 均值轨迹） | 交叉 token | 状态 |')
    A('|---|---|---|---|---|---|')
    _recon = {k for k, v in xo.items() if v.get('status') == 'ok' and any((v.get('per_seed_synth', {}).get(str(s)) for s in v.get('seeds', [])))}
    for label, v in xo.items():
        if v.get('status') != 'ok':
            A(f'| {label} | `{v['outdir']}` | — | — | — | 数据缺失 |')
            continue
        n = len(v['seeds'])
        _ntag = f'{n}（重构）' if label in _recon else f'{n}'
        if v['mean_crossed']:
            _tok = f'{v['mean_crossover_tokens'] / 1000000.0:.1f}M' if 'mean_crossover_tokens' in v else 'n/a'
            A(f'| d={v['d']}/{v['n_layers']}L | `{v['outdir']}` | {_ntag} | {v['mean_crossover_step']:g} | {_tok} | 已交叉 |')
        else:
            _stps = v['mean_traj']['steps']
            if not _stps:
                A(f'| d={v['d']}/{v['n_layers']}L | `{v['outdir']}` | {_ntag} | — | — | 无公共 eval 步 |')
                continue
            last = _stps[-1]
            _tok = f'> {last * v['tokens_per_step'] / 1000000.0:.1f}M' if v.get('tokens_per_step') else '(tokens/step unknown)'
            A(f'| d={v['d']}/{v['n_layers']}L | `{v['outdir']}` | {_ntag} | > {last:g} | {_tok} | 删失（窗内未交叉） |')
    A('')
    ok = [v for v in xo.values() if v.get('status') == 'ok']
    if ok:
        A('逐种子交叉步：')
        A('')
        for label, v in xo.items():
            if v.get('status') != 'ok':
                continue
            parts = []
            for s, ps in sorted(v['per_seed'].items()):
                parts.append(f's{s}: {ps['crossover_step']:g}{('' if ps['crossed'] else '（删失）')}')
            A(f'- d={v['d']}: ' + '；'.join(parts))
        A('')
    _xo256 = xo.get('d256_L6') or next((v for v in xo.values() if v.get('d') == 256), {})
    _xo256_step = _xo256.get('mean_crossover_step')
    _xo256_txt = f'~{_xo256_step:g} 步' if isinstance(_xo256_step, (int, float)) and math.isfinite(_xo256_step) else '步数未知（该规模无有效交叉读数）'
    _xo256_recon = any(((_xo256.get('per_seed_synth') or {}).get(str(s)) for s in _xo256.get('seeds', [])))
    if _xo256_recon:
        A(f'**注意**：d=256 的轨迹取自 `results_lm_v3_long/summary.json`，其中部分记录是从旧聚合表重构的副本（`synthesized`），逐种子轨迹不可独立恢复——该规模的逐种子交叉步互为副本，只有 seed 均值轨迹的交叉步（{_xo256_txt}，eval 网格 1000 步）是有效读数。')
    elif _xo256.get('status') == 'ok':
        A(f'**注意**：d=256 面板（`results_lm_v3_long`）当前不含重构记录，逐种子交叉步（{_xo256_txt} 附近）与 seed 均值轨迹同为有效读数。')
    A('')
    if fit:
        if fit.get('status') != 'ok' or 'slope' not in fit:
            _why = '; '.join(fit.get('reasons') or [fit.get('note', 'no slope')])
            A(f'**趋势拟合：被拒绝**（status={fit.get("status")}）。{_why}。因此本版**不报告** log-log 斜率或幂律。')
            A('')
        else:
            _ci = fit.get('ci')
            _citxt = f'，95% CI [{_ci[0]:.2f}, {_ci[1]:.2f}]' if isinstance(_ci, (list, tuple)) and len(_ci) == 2 else ''
            _ptxt = f'，p={fit["p"]:.3f}' if isinstance(fit.get('p'), (int, float)) else ''
            _r2txt = f'，R²={fit["r2"]:.2f}' if isinstance(fit.get('r2'), (int, float)) else ''
            _lootxt = ''
            _loo = fit.get('leave_one_out_slope_range')
            if isinstance(_loo, (list, tuple)) and len(_loo) == 2:
                _lootxt = f'；留一斜率范围 [{_loo[0]:.2f}, {_loo[1]:.2f}]'
            _vol = fit.get('volume_ratio')
            A(f'**趋势拟合（描述性，非幂律主张）**：{fit.get("n_points")} 个去重后的规模点，'
              f'log10(交叉 token) ~ log10(d) 斜率 {fit["slope"]:.2f}{_citxt}{_ptxt}{_r2txt}{_lootxt}；'
              f'训练量比 {_vol:.2f}（要求 ≤1.25）。这是观测性回归，规模面板未随机化，不能写成 scaling law。')
            A('')
    sc = st.get('scale_confounds') or {}
    if sc:
        A('### P3S-2 规模混杂控制（等训练 token / 6ND 代理规模 / 规模内平台期）')
        A('')
        A('**背景**：P3SS（d=128/4L，8000 步 ≈ 49.15M token）与 P3SL（d=512/10L，4000 步 ≈ 24.58M token）的累计 token 相差一倍——跨规模终点差异不能单独归因于规模。本节把同一批曲线在**相同累计 token**处切片配对（gap_L − gap_S，逐种子），并给出等 token（`P3SL_TOK`，8000 步）与 6ND 参数量代理规模（`P3SL_FLOP`，按参数量比例缩步数——**代理规模，非实测等 FLOPs**）两个大模型臂，以及各面板的规模内平台期诊断（平台期 ≠ 收敛，只支持「规模内平台期后」读数）。')
        A('')
        _panels = sc.get('panels') or {}
        if _panels:
            A('| 面板 | 规模 | 末步 token | 末点 gap (csa−full) mean±std | n | 末点 p (exact) | 平台化种子 |')
            A('|---|---|---|---|---|---|---|')
            for name in ('P3SS', 'P3SL', 'P3SL_TOK', 'P3SL_FLOP'):
                e = _panels.get(name) or {}
                if not e or e.get('status') != 'ok':
                    A(f'| {name} | — | — | 数据缺失 | 0 | — | — |')
                    continue
                _ep = e.get('endpoint') or {}
                _tok = e.get('tokens_per_step')
                _ntok = (f"{(_ep.get('tokens') or 0) / 1000000.0:.2f}M" if _ep and _ep.get('tokens') else 'n/a')
                _conv = e.get('convergence') or {}
                _pl = [f"{_v.get('n_plateaued', 0)}/{_v.get('n', 0)}" for _v in _conv.values()]
                A(f"| {name} | {e.get('note', '')} | {_ntok} | {_ep.get('mean', float('nan')):+.2f} ± {_ep.get('std', 0):.2f} | {_ep.get('n', 0)} | {_ep.get('p_exact_signflip', float('nan')):.3f} | {'/'.join(_pl) or '—'} |")
            A('')
            _conv_warn = [name for name, e in _panels.items() if e.get('status') == 'ok' and any(((e.get('convergence') or {}).get(v) or {}).get('n_plateaued', 0) == 0 for v in ('csa_fixed', 'full'))]
            if _conv_warn:
                A(f'> **规模内平台期证据不足**：{_conv_warn} 面板在最后四分之一 eval 窗口的相对改进 ≥1%（未平台化）——「平台期后交叉」的主张在该面板上**不成立**，只有「窗口内交叉」的读数有效。')
                A('')
            A('> 即使全部种子平台化，也只允许「规模内平台期后交叉」的读数；「渐近优势反转」「收敛后更差」需要更长训练、学习率续训与重复验证，本面板不提供。')
            A('')
        _slices = sc.get('equal_token_slices') or {}
        if _slices:
            A('**等 token 切片（gap_L − gap_S，同累计 token 处逐种子配对；>0 = 大模型在同 token 处 CSA−dense 差距更大）：**')
            A('')
            A('| 切片 | n | Δ mean±std | d_z | p (exact) |')
            A('|---|---|---|---|---|')
            for k, v in sorted(_slices.items()):
                A(f'| {k} | {v.get('n', 0)} | {v.get('mean', float('nan')):+.2f} ± {v.get('std', 0):.2f} | {v.get('cohens_dz', float('nan')):+.2f} | {v.get('p_exact_signflip', float('nan')):.3f} |')
            A('')
            A('> `P3SL_FLOP` 的步数 = 8000 × N_small/N_large（参数量按面板两臂平均，CPU 上确定性计数；下限 250），即**参数量与步数的 6ND 代理对齐**——它不是实测 FLOPs、GPU 时间或能耗；该臂的 gap 只能读作「该代理近似相等时」的规模效应。')
            A('')
    A('---')
    A('')
    A('## P3T seq-2048 topk 扫描补种子（n=2 → n=4）')
    A('')
    A('**背景**：v9 P2T 的完整 5 点扫描只有 2 seeds，精确符号翻转 p 值下限 0.500——v9 收官后唯一仍处该下限的面板。v10 按同一配方（seq 2048、bs 1、1500 步）补 seeds 2/3（同一 outdir，resume 跳过 seed 0/1）。')
    A('')
    A('| variant | PPL (mean±std) | n |')
    A('|---|---|---|')
    _p3t_done = 0
    for v in ['csa_fixed_topk8', 'csa_fixed_topk32', 'csa_fixed_topk128', 'csa_fixed_topk512', 'csa_fix_m1']:
        if v in seq2k:
            p = seq2k[v]
            A(f'| `{v}` | {_fmt_pm(p)} | {p['n']} |')
            _p3t_done += 1
        else:
            A(f'| `{v}` | —（未完成） | 0 |')
    A('')
    A(f'**完成 {_p3t_done}/5 点**。')
    A('')
    A('| 比较 | n | Δ (a−b) mean±std | p (exact) |')
    A('|---|---|---|---|')
    for k, v in comp.items():
        if k.startswith('seq2k'):
            if 'omitted' in v:
                A(f'| {k} | 0 | —（{v['omitted']}） | — |')
                continue
            A(f'| {k} | {v['n']} | {v['mean']:+.2f} ± {v.get('std', 0):.2f} | {v['p_exact_signflip']:.3f} |')
    A('')
    A('**注**：n=4 时精确符号翻转的 p 值分辨率为 0.125——仍跨不过 0.05（需 n=6 才有 0.031）；此行补齐的是与 scale 面板同级的统计分辨率，显著性主张仍以效应量 + 方向同向性表述。')
    A('')
    A('---')
    A('')
    A('## 产物清单')
    A('')
    A('| 路径 | 内容 |')
    A('|---|---|')
    A('| `results_v10_mech/` | P3M：两臂重训权重（ckpt/，*.pt 不入库）+ by-length 复现表 + 干扰注入探针 `distractor.json`（uniform/natural/mask/delete + span/doc/topical 模式；natural = realistic-token，§15 修订）+ NIAH-natural `niah_natural.json` |')
    A('| `results_lm_v10_scale_s/` | P3S：d=128/4L 交叉定位面板（8000 步） |')
    A('| `results_lm_v10_scale_l/` | P3S：d=512/10L 交叉定位面板（4000 步） |')
    A('| `results_lm_v10_scale_l8k/` | P3S-2：等 token 大模型臂（8000 步） |')
    A('| `results_lm_v10_scale_lflop/` | P3S-2：6ND 参数量代理规模大模型臂（非实测等 FLOPs） |')
    A('| `results_lm_v9_seq2k/` | P3T：topk 扫描追加 seeds 2/3 后的 4-seed 面板 |')
    A('| `analysis_v10/stats.json` | 探针对比（含 ΔNLL、Holm）、交叉定位、等 token 切片与配对符号翻转检验 |')
    A('| `v10_supp.py` | 本阶段驱动（smoke/phase/analysis/report，可断点续跑） |')
    A('')
    L.atomic_write_text(out, '\n'.join(lines) + '\n')
    print(f'[v10 report] wrote {out}')
    return out

def git_push(msg):
    if os.environ.get('V10_NO_PUSH'):
        print(f'[git] push skipped (V10_NO_PUSH): {msg}')
        return True
    return V.git_push(msg)

def schedule_shutdown(delay_s=120):
    if os.environ.get('V10_NO_SHUTDOWN'):
        print('[v10] shutdown suppressed (V10_NO_SHUTDOWN)')
        return
    subprocess.Popen(['bash', '-c', f'sleep {delay_s}; shutdown -h now'], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f'[v10] instance shuts down in {delay_s}s.')

def run_full():
    guard = make_guard()
    guard.report()
    print(f'[v10] headroom {max(guard.remaining_hours(), 0.0):.2f} h left (booked {guard.state['booked_seconds'] / 3600:.2f} h)')
    git_push('v10: supplementary driver (P3M distractor-injection mechanism + P3S crossover scaling + P3T topk seeds 2/3 + scratch selector controls, corruption modes/NIAH, token/compute-matched scale arms)')
    import par
    _ctx = par.resolve_ctx(label='v10')
    all_ok = True
    for pname, _kind, _payload, _seeds in PHASES:
        rem = guard.remaining_hours()
        if rem < 0.4166666666666667:
            print(f'[v10] stopping before {pname}: headroom exhausted ({rem:.2f} h)')
            break
        if not V.cuda_healthy():
            print(f'[v10] CUDA context poisoned before {pname} — aborting (re-run resumes).')
            all_ok = False
            break
        print(f'\n===== v10 phase {pname} =====')
        try:
            run_phase(pname, guard, par=_ctx)
        except Exception:
            traceback.print_exc()
            all_ok = False
        all_ok &= git_push(f'v10: phase {pname} results')
    try:
        build_report()
    except Exception:
        traceback.print_exc()
        all_ok = False
    all_ok &= git_push('v10: stats + REPORT_v10.md (analysis_v10)')
    guard.report()
    print('\n[v10] ALL PHASES DONE.')
    schedule_shutdown(120 if all_ok else 2400)

def run_smoke():
    print('[smoke] 1) tiny lenphase ckpts + extended distractor probe roundtrip (all modes + NIAH + a scratch control)')
    L.set_seed(0)
    payload = dict(P3MT_PAYLOAD, outdir='results_smoke_v10', seeds=[0], steps=40, eval_lens=[512, 1024])
    import shutil
    shutil.rmtree('results_smoke_v10', ignore_errors=True)
    V.run_lenphase(dict(payload, variants=['csa_fixed_rope', 'csa_fixed_rope_randidx']), guard=None, label='v10 smoke', par=False)
    pcfg = dict(PROBE, outdir='results_smoke_v10', ckpt_dir='results_smoke_v10/ckpt', summary='results_smoke_v10/distractor.json', variants=['csa_fixed_rope', 'csa_fixed_rope_randidx'], seeds=[0], eval_lens=[1024], rhos=[0.0, 0.5], modes={'uniform': [0.0, 0.5], 'natural': [0.5], 'mask': [1.0], 'delete': [1.0]}, n_seq=4, chunk=2, niah=dict(NIAH_NATURAL, summary='results_smoke_v10/niah_natural.json', n_pairs=2, n_seq=4, conditions=['clean', 'distract']), arm_overrides={'csa_fixed_rope_randidx': ['learned']})
    s = run_probe(pcfg, guard=None, label='v10 smoke', par=False)
    got = {(r.get('mode', 'uniform'), r['arm'], r['rho']) for r in s.values()}
    _want = {('uniform', 'learned', 0.0), ('uniform', 'learned', 0.5), ('uniform', 'randidx', 0.0), ('uniform', 'randidx', 0.5), ('uniform', 'allblocks', 0.0), ('uniform', 'allblocks', 0.5), ('natural', 'learned', 0.5), ('natural', 'randidx', 0.5), ('natural', 'allblocks', 0.5), ('mask', 'learned', 1.0), ('mask', 'randidx', 1.0), ('mask', 'allblocks', 1.0), ('delete', 'learned', 1.0), ('delete', 'randidx', 1.0), ('delete', 'allblocks', 1.0)}
    assert _want <= got, f'probe cells missing: {_want - got} (got {sorted(got)})'
    niah_sum = {}
    _nsp = 'results_smoke_v10/niah_natural.json'
    if os.path.exists(_nsp):
        niah_sum = json.load(open(_nsp, encoding='utf-8'))
    _niah_got = {(r.get('arm'), r.get('condition')) for r in niah_sum.values() if isinstance(r, dict)}
    _niah_want = {('learned', 'clean'), ('learned', 'distract'), ('randidx', 'clean'), ('randidx', 'distract'), ('allblocks', 'clean'), ('allblocks', 'distract')}
    assert _niah_want <= _niah_got, f'NIAH cells missing: {_niah_want - _niah_got} (got {sorted(_niah_got)})'
    print(f'  probe roundtrip OK ({len(s)} corruption cells + {len(niah_sum)} NIAH cells)')
    shutil.rmtree('results_smoke_v10', ignore_errors=True)
    print('[smoke] 2) forward/backward at the two new scales + the scratch-control rope variants')
    for d, nl, nh in ((128, 4, 4), (512, 10, 16)):
        for v in ('csa_fixed', 'full'):
            cfgs = L.make_layer_cfgs(nl, v)
            m = L.SmallGPT(8192, d, nl, nh, 32, 512, cfgs).to(DEVICE)
            x = torch.randint(0, 8192, (2, 512), device=DEVICE)
            out = m(x)
            loss = F.cross_entropy(out.reshape(-1, 8192), x.reshape(-1)) + 0.05 * m.comp_reg
            loss.backward()
            print(f'  d={d} L={nl} {v:10s} out={tuple(out.shape)} loss={loss.item():.3f} params={L.count_params(m) / 1000000.0:.2f}M')
            del m, out, loss, x
            gc.collect()
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
    for v in SCRATCH_CONTROLS:
        cfgs = L.make_layer_cfgs(6, v)
        m = L.SmallGPT(8192, 256, 6, 8, 32, 512, cfgs).to(DEVICE)
        x = torch.randint(0, 8192, (2, 512), device=DEVICE)
        out = m(x)
        loss = F.cross_entropy(out.reshape(-1, 8192), x.reshape(-1)) + 0.05 * m.comp_reg
        loss.backward()
        print(f'  d=256 L=6 {v:32s} out={tuple(out.shape)} loss={loss.item():.3f} params={L.count_params(m) / 1000000.0:.2f}M')
        del m, out, loss, x
        gc.collect()
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    print('[smoke] 2b) flop-matched steps are deterministic')
    _fs1 = _flop_matched_steps(P3SS_CFG, P3SL_CFG)
    _fs2 = _flop_matched_steps(P3SS_CFG, P3SL_CFG)
    assert _fs1 == _fs2 and _fs1 >= 250, f'flop-matched steps not deterministic: {_fs1} vs {_fs2}'
    print(f'  flop-matched steps OK ({_fs1})')
    print('[smoke] 3) zero-GPU analysis/report on current artifacts (scratch outputs, real artifacts untouched)')
    build_report(out='results_smoke_v10/REPORT_v10.md', stats_p='results_smoke_v10/stats.json')
    shutil.rmtree('results_smoke_v10', ignore_errors=True)
    print('\n[smoke] PASSED')
if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'full'
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    _MODES = ('full', 'smoke', 'analysis', 'report', 'phase')
    if mode in ('-h', '--help', 'help'):
        print(__doc__ or f"[v10] modes: {_MODES} (no argument means 'full')")
        raise SystemExit(0)
    if mode not in _MODES:
        print(f"[v10] unknown mode {mode!r}; expected one of {_MODES} (no argument means 'full').  Refusing to start a run.")
        raise SystemExit(2)
    if mode == 'phase' and len(sys.argv) < 3:
        print(f"[v10] mode 'phase' needs a phase name, e.g. `python v10_supp.py phase P3MT`.  Available: {[p[0] for p in PHASES]}")
        raise SystemExit(2)
    print(f'[v10] mode={mode} repo={REPO} device={DEVICE} ({(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')})')
    try:
        if mode == 'smoke':
            run_smoke()
        elif mode == 'analysis':
            v10_analysis()
        elif mode == 'report':
            build_report()
        elif mode == 'phase':
            run_phase(sys.argv[2], make_guard())
            git_push(f'v10: phase {sys.argv[2]} results')
        else:
            run_full()
    except Exception:
        traceback.print_exc()
        try:
            git_push('v10: PARTIAL — crashed, see log (re-run resumes)')
        except Exception:
            traceback.print_exc()
        if mode == 'full':
            schedule_shutdown(2400)
        sys.exit(1)
