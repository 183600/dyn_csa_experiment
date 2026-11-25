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


def _cap_hours(_name, _default):
    try:
        _v = float(os.environ.get(_name, _default))
    except (TypeError, ValueError, OverflowError):
        raise SystemExit(f'[driver] {_name}={os.environ.get(_name)!r} is not a number — set it to finite hours (or unset it for the default)') from None
    if not (_v == _v) or _v in (float('inf'), float('-inf')) or _v <= 0:
        raise SystemExit(f'[driver] {_name}={os.environ.get(_name)!r} must be a positive finite number of hours')
    return _v

RUN_CAP_V10.update(total_hours=_cap_hours('V10_CAP_HOURS', 25.0), state_path='run_time_state_v10.json', already_hours=0.0)

def make_guard():
    g = L.TimeGuard(RUN_CAP_V10)
    if not g.state.get('sps_by_class'):
        for src in ('run_time_state_v9.json', 'run_time_state_v8.json', 'run_time_state_v7.json'):
            if os.path.exists(src):
                try:
                    with open(src, encoding='utf-8') as _jf:
                        st = json.load(_jf)
                    if not isinstance(st, dict):
                        print(f'[v10] WARNING: {src} is not a JSON object — this run starts UNCALIBRATED')
                        continue
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
# （同数据、同种子）训练，唯一差异是选择方式本身。P3MT 的 variants
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
NIAH_NATURAL = dict(summary='results_v10_mech/niah_natural.json', n_pairs=8, n_seq=16, conditions=['clean', 'distract', 'nofar'], rho=0.5)
# The from-scratch selector controls are trained by P3MTC and must also be
# PROBED by P3MP: otherwise `retrain-gap` / `csa_randtrain` contrasts can never
# be measured from a v10-only run.  P5MP already adds them explicitly; keeping
# the same variants/overrides here makes the v10 phase self-contained.
PROBE = dict(outdir='results_v10_mech', ckpt_dir='results_v10_mech/ckpt', summary='results_v10_mech/distractor.json', variants=['csa_fixed_rope', 'full_rope'] + list(SCRATCH_CONTROLS), seeds=[0, 1, 2], eval_lens=[2048, 4096], rhos=[0.0, 0.125, 0.25, 0.5], modes=dict(PROBE_MODES), target=512, n_seq=16, chunk=4, arms=['dense', 'learned', 'randidx', 'allblocks'], arm_overrides={_v: ['learned'] for _v in SCRATCH_CONTROLS}, niah=dict(NIAH_NATURAL))

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


def _corrupt_row(val_ids, eval_len, rho, mode, jj, vocab, target, span_len=PROBE_SPAN_LEN,
                  return_audit=False):
    """Build ONE corrupted evaluation row (pure, arm/model-seed independent).

    输入 val_ids 为 (n_rows, >=eval_len+1) 的 token 面板；返回长度
    eval_len+1 的 int64 行。四种既有模式（uniform/natural/mask/delete）与
    既有实现逐字节一致（同一 RNG 播种、同一调用顺序——见 _distractor_ppls
    的行缓存键）。三种新模式（span/doc/topical）同样只
    依赖 (eval_len, jj, rho, mode, span_len)，保证臂间严格配对。"""
    # The row geometry is exact-integer state; malformed lengths are refused
    # before any mode branch.
    _ints = {}
    for _name, _value in (('eval_len', eval_len), ('target', target),
                          ('jj', jj), ('span_len', span_len)):
        _iv = L._as_seed_int(_value)
        if _iv is None:
            raise ValueError(
                f'_corrupt_row: {_name}={_value!r} must be an exact integer')
        _ints[_name] = _iv
    eval_len, target = _ints['eval_len'], _ints['target']
    jj, span_len = _ints['jj'], _ints['span_len']
    if eval_len < 2 or target < 1 or target >= eval_len:
        raise ValueError(
            f'_corrupt_row needs eval_len>=2 and 1<=target<eval_len '
            f'(got eval_len={eval_len}, target={target}); there must be a '
            'far context outside the scored target zone')
    if jj < 0 or span_len < 1:
        raise ValueError(
            f'_corrupt_row needs jj>=0 and span_len>=1 '
            f'(got jj={jj}, span_len={span_len})')
    _rows = np.asarray(val_ids)
    if _rows.ndim != 2:
        raise ValueError(
            f'_corrupt_row: val_ids must be a 2-D token matrix, got shape {_rows.shape}')
    if jj >= _rows.shape[0]:
        raise ValueError(
            f'_corrupt_row: row index jj={jj} is outside the {_rows.shape[0]}-row panel')
    if eval_len + 1 > _rows.shape[1]:
        raise ValueError(
            f'_corrupt_row: eval_len+1={eval_len + 1} exceeds the panel row '
            f'width {_rows.shape[1]}')
    val_ids = _rows
    n_far = eval_len - target
    if mode not in _CORRUPT_MODES:
        raise ValueError(f'_corrupt_row: unknown corruption mode {mode!r} (expected one of {_CORRUPT_MODES})')
    # The donor-based modes replace far context with tokens from ANOTHER
    # sequence.  With a one-row panel every "donor" lookup wraps back to the
    # scored row itself, so the operation is a silent no-op that would still
    # be recorded as a corrupted condition.  Refuse the meaningless panel
    # instead of measuring a clean row under a corruption label.
    if mode in ('natural', 'topical', 'span', 'doc') and int(val_ids.shape[0]) < 2:
        raise ValueError(
            f'_corrupt_row: mode {mode!r} needs at least two validation rows '
            f'(got {int(val_ids.shape[0])}); the only possible donor is the '
            'scored row itself, so the corruption would be a no-op')
    # rho is a corruption FRACTION.  A boolean is a flag, not a fraction, and
    # a value outside [0,1] would either silently corrupt nothing (negative)
    # or ask rng.choice for more positions than the far context holds (>1),
    # so validate before any mode-specific branch.
    if isinstance(rho, (bool, np.bool_, np.datetime64, np.timedelta64,
                        complex, np.complexfloating)):
        raise ValueError(f'_corrupt_row: rho={rho!r} is a boolean/temporal/complex value, not a corruption fraction')
    try:
        _rho = float(rho)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'_corrupt_row: rho={rho!r} is not a number') from None
    if not math.isfinite(_rho) or not (0.0 <= _rho <= 1.0):
        raise ValueError(f'_corrupt_row: rho={rho!r} must be a finite fraction in [0,1]')
    if mode in ('mask', 'delete', 'doc') and abs(_rho - 1.0) > 1e-9:
        raise ValueError(f"mode {mode!r} is an all-or-nothing condition and only admits rho=1.0 (got {rho!r}); partial masking is a different question and is not this probe")
    # ---- Disclosure-only audit of what the corruption ACTUALLY did ----
    # (review 10 issue 12).  `rho` is a nominal fraction: for `span` the
    # realised coverage is quantised by segment geometry, for natural /
    # topical / doc replacements a donor token can equal the original token
    # (no change), and `overlap_token_slots` measures repeated coverage.  The
    # audit is a deterministic function of the same arguments and consumes no
    # RNG draws, so it never changes the measured ids.
    ids = np.asarray(val_ids[jj, :eval_len + 1], dtype=np.int64).copy()
    _clean = ids.copy()
    _audit = {'mode': mode, 'rho_nominal': float(_rho),
              'far_positions': int(n_far), 'span_len': None,
              'n_span_segments': None, 'donor_row': None,
              'covered_token_slots': 0, 'covered_unique_positions': 0,
              'overlap_token_slots': 0, 'changed_tokens': 0,
              'changed_definition': 'replaced'}

    def _finish(_ids, _covered, *, changed=None, donor=None, span_len=None,
                n_seg=None, changed_definition='replaced'):
        _cov = np.asarray(_covered, dtype=np.int64)
        _uniq = np.unique(_cov) if _cov.size else _cov
        if changed is None:
            changed = int((_ids[_uniq] != _clean[_uniq]).sum()) if _uniq.size else 0
        _audit.update({
            'covered_token_slots': int(_cov.size),
            'covered_unique_positions': int(_uniq.size),
            'overlap_token_slots': int(max(0, _cov.size - _uniq.size)),
            'changed_tokens': int(changed),
            'changed_definition': changed_definition,
            'realised_coverage_fraction': (float(_uniq.size) / n_far) if n_far else 0.0,
            'realised_changed_fraction': (float(changed) / n_far) if n_far else 0.0,
            'changed_fraction_of_covered': (float(changed) / _cov.size) if _cov.size else None,
            'no_op': bool(_cov.size > 0 and changed == 0),
            'donor_row': donor, 'span_len': span_len,
            'n_span_segments': n_seg})
        return (_ids, _audit) if return_audit else _ids
    if mode == 'delete':
        # the far context is REMOVED, not filled: the model sees only
        # the final target+1 tokens (RoPE is relative, so the kept
        # tokens' pairwise structure is unchanged)
        _short = np.asarray(val_ids[jj, eval_len - target:eval_len + 1], dtype=np.int64).copy()
        return _finish(_short, np.arange(n_far, dtype=np.int64),
                       changed=n_far, changed_definition='removed')
    k = int(round(_rho * n_far))
    if _rho > 0.0:
        # Any positive corruption fraction must replace at least one far token.
        k = max(1, k)
    k = min(k, n_far)
    if mode == 'mask':
        ids[:n_far] = MASK_TOKEN_ID
        return _finish(ids, np.arange(n_far, dtype=np.int64))
    if mode == 'doc':
        # whole far context replaced by ONE contiguous slice of another row
        # (length- and position-controlled "unrelated document" context)
        rng = np.random.default_rng((eval_len * 1000003 + jj * 10007) * 1048576 + int(round(_rho * 1048576)))
        _src = (jj + max(1, val_ids.shape[0] // 2)) % val_ids.shape[0]
        _row_len = min(val_ids.shape[1], eval_len + 1)
        _max_off = _row_len - n_far
        _off = int(rng.integers(0, max(1, _max_off))) if _max_off > 0 else 0
        ids[:n_far] = np.asarray(val_ids[_src, _off:_off + n_far], dtype=np.int64)
        return _finish(ids, np.arange(n_far, dtype=np.int64), donor=int(_src))
    if k <= 0:
        return _finish(ids, np.empty(0, dtype=np.int64))
    if mode == 'uniform':
        # With vocab < 2 there is no alternative token, so the resample loop
        # below could never make progress (an infinite loop rather than a
        # measurement).  Refuse the meaningless condition explicitly.
        _vb = L._as_seed_int(vocab)
        if _vb is None or _vb < 2:
            raise ValueError(f'uniform corruption needs an integer vocab >= 2 (got {vocab!r}); no alternative token exists')
        rng = np.random.default_rng((eval_len * 1000003 + jj * 10007) * 1048576 + int(round(_rho * 1048576)))
        pos = rng.choice(n_far, size=k, replace=False)
        _draw = rng.integers(0, _vb, size=k)
        _same = _draw == ids[pos]
        while _same.any():
            _draw[_same] = rng.integers(0, _vb, size=int(_same.sum()))
            _same = _draw == ids[pos]
        ids[pos] = _draw
        return _finish(ids, pos)
    if mode in ('natural', 'topical'):
        # natural: paired other-row realistic tokens at the same seeded
        # position grid; collisions with the original token are KEPT (they
        # are part of the realistic-token statistics).
        # topical: same protocol, but the donor row is the most unigram-similar
        # row (topic proxy) instead of the fixed-offset row.
        rng = np.random.default_rng((eval_len * 1000003 + jj * 10007) * 1048576 + int(round(_rho * 1048576)))
        pos = rng.choice(n_far, size=k, replace=False)
        if mode == 'topical':
            _nat = (jj + max(1, val_ids.shape[0] // 2)) % val_ids.shape[0]
            _src = _topical_donor(val_ids, jj, n_far, exclude={_nat})
            if _src is None:
                # With no row other than the natural donor, the "topical"
                # condition would silently become the natural condition.  It
                # is a different scientific question, so refuse rather than
                # record one mode under the other's name.
                raise ValueError(
                    'topical corruption needs a donor row other than the '
                    f'natural donor (row {_nat}); the panel has only '
                    f'{int(val_ids.shape[0])} row(s)')
        else:
            _src = (jj + max(1, val_ids.shape[0] // 2)) % val_ids.shape[0]
        ids[pos] = np.asarray(val_ids[_src, pos], dtype=np.int64)
        return _finish(ids, pos, donor=int(_src))
    if mode == 'span':
        # contiguous sentence-like segments from the donor row: local syntax
        # inside each span survives, cross-span coherence is destroyed.
        # Spans are disjoint by construction (one span per equal-width tile
        # of the far zone), so the realised coverage is exactly n_seg * seg
        # replaced tokens; rho is nominal beyond that granularity.
        rng = np.random.default_rng((eval_len * 1000003 + jj * 10007) * 1048576 + int(round(_rho * 1048576)))
        seg = max(1, int(span_len))
        seg = min(seg, n_far)
        n_seg = max(1, int(math.ceil(k / seg)))
        # Disjoint spans by construction: split the far zone into n_seg
        # equal tiles and draw one span uniformly inside each tile, so the
        # realised coverage is exactly n_seg * seg replaced tokens.
        max_disjoint = max(1, n_far // seg)
        if n_seg > max_disjoint:
            n_seg = max_disjoint
        tile = n_far // n_seg
        _row_len = min(val_ids.shape[1], eval_len + 1)
        _src = (jj + max(1, val_ids.shape[0] // 2)) % val_ids.shape[0]
        _off_hi = _row_len - seg + 1
        _covered = []
        for i in range(n_seg):
            s = int(i * tile + rng.integers(0, tile - seg + 1))
            o = int(rng.integers(0, _off_hi)) if _off_hi > 1 else 0
            ids[s:s + seg] = np.asarray(val_ids[_src, o:o + seg], dtype=np.int64)
            _covered.append(np.arange(s, s + seg, dtype=np.int64))
        _cov = (np.concatenate(_covered) if _covered
                else np.empty(0, dtype=np.int64))
        return _finish(ids, _cov, donor=int(_src), span_len=int(seg),
                       n_seg=int(n_seg))
    raise ValueError(f'_corrupt_row: unknown corruption mode {mode!r} (expected one of {_CORRUPT_MODES})')


@torch.inference_mode()
def _corruption_audit(val_ids, eval_len, rho, mode, n_seq, vocab, target,
                      span_len=PROBE_SPAN_LEN):
    """Aggregate what a corruption mode ACTUALLY changed over the scored panel.

    Review 10 issue 12: `rho` is a NOMINAL fraction.  For `span` the realised
    coverage is quantised by the segment geometry (`n_seg * span_len` replaced
    slots, capped by the far zone), different modes carry different geometry at
    the same rho, donor tokens can equal the original token (so a covered
    position is not necessarily a changed one), and the same position can in
    principle be covered more than once.  This audit records the realised
    counts so reports can state, per (mode, rho), how much of the far context
    was actually covered/changed and whether the corruption was a no-op.

    The audit is an arm/model independent deterministic function of the same
    arguments as `_corrupt_row`; it consumes no RNG state and does not change
    any measured id.
    """
    _ns = L._as_seed_int(n_seq)
    if _ns is None or _ns < 1:
        raise ValueError(f'_corruption_audit: n_seq={n_seq!r} must be a positive exact integer')
    _rows = np.asarray(val_ids)
    if _rows.ndim != 2:
        raise ValueError(f'_corruption_audit: val_ids must be a 2-D token matrix, got shape {_rows.shape}')
    if _ns > _rows.shape[0]:
        raise ValueError(f'_corruption_audit: n_seq={_ns} exceeds the {_rows.shape[0]}-row panel')
    audits = []
    for jj in range(_ns):
        _, _a = _corrupt_row(val_ids, eval_len, rho, mode, jj, vocab, target,
                             span_len=span_len, return_audit=True)
        audits.append(_a)
    far = audits[0]['far_positions']
    denom = far * len(audits)
    cov_slots = sum(a['covered_token_slots'] for a in audits)
    cov_uniq = sum(a['covered_unique_positions'] for a in audits)
    changed = sum(a['changed_tokens'] for a in audits)
    overlap = sum(a['overlap_token_slots'] for a in audits)
    _donor = sorted({a['donor_row'] for a in audits if a['donor_row'] is not None})
    return {'mode': mode, 'rho_nominal': float(audits[0]['rho_nominal']),
            'n_rows': len(audits), 'far_positions_per_row': int(far),
            'covered_token_slots_total': int(cov_slots),
            'covered_unique_positions_total': int(cov_uniq),
            'overlap_token_slots_total': int(overlap),
            'changed_tokens_total': int(changed),
            'changed_definition': audits[0]['changed_definition'],
            'realised_coverage_fraction': (float(cov_uniq) / denom) if denom else 0.0,
            'realised_changed_fraction': (float(changed) / denom) if denom else 0.0,
            'changed_fraction_of_covered': (float(changed) / cov_slots) if cov_slots else None,
            'no_op_rows': int(sum(1 for a in audits if a['no_op'])),
            'span_len': audits[0]['span_len'],
            'n_span_segments': audits[0]['n_span_segments'],
            'distinct_donor_rows': len(_donor)}


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
    _span_len = _probe_span_len(cfg)
    i = 0
    while i < n_seq:
        j = min(i + n_ch, n_seq)
        rows = []
        for jj in range(i, j):
            _rk = (eval_len, jj, float(rho), vocab, int(cfg['target']), mode, _span_len)
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
        # fp64 row sums preserve the values exactly.
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
    try:
        ce = total / nt
    except OverflowError:
        # An exact token total beyond the binary64 range underflows the
        # per-token NLL to zero (PPL 1.0).
        return 1.0
    # Use the shared exp-for-PPL helper for the binary64 exp boundary.
    _ppl = L._safe_exp_ppl(ce)
    return float('inf') if _ppl is None else _ppl

def _binom_sf_exact(n, k, q):
    """P(Bin(n, q) >= k) in log space (exact, overflow-free for small n)."""
    n_i = L._as_seed_int(n)
    k_i = L._as_seed_int(k)
    if n_i is None or k_i is None or n_i < 1:
        return float('nan')
    try:
        q_f = float(q)
    except (TypeError, ValueError, OverflowError):
        return float('nan')
    if not (math.isfinite(q_f) and 0.0 <= q_f <= 1.0):
        return float('nan')
    if k_i <= 0:
        return 1.0
    if k_i > n_i:
        return 0.0
    if q_f <= 0.0:
        return 0.0
    if q_f >= 1.0:
        return 1.0
    _lq, _l1q = math.log(q_f), math.log1p(-q_f)
    _logs = [math.lgamma(n_i + 1) - math.lgamma(i + 1) - math.lgamma(n_i - i + 1)
             + i * _lq + (n_i - i) * _l1q for i in range(k_i, n_i + 1)]
    _mx = max(_logs)
    _p = math.exp(_mx) * math.fsum(math.exp(x - _mx) for x in _logs)
    return float(min(1.0, max(0.0, _p)))


def _probe_span_len(cfg):
    """Exact positive span geometry for the `span` corruption mode.

    A missing key keeps the shipped default.  An explicit value must be an
    exact integer >= 1: bare int(True)/int(4.9) would silently corrupt a
    different span than the one named in the record fingerprint."""
    raw = cfg.get('span_len') if isinstance(cfg, dict) else None
    if raw is None:
        return int(PROBE_SPAN_LEN)
    iv = L._as_seed_int(raw)
    if iv is None or iv < 1:
        raise ValueError(
            f'probe span_len={raw!r} must be a positive exact integer')
    return iv


def _probe_fingerprint(cfg):
    if not isinstance(cfg, dict):
        raise ValueError(f'_probe_fingerprint: cfg is {type(cfg).__name__}, not an object')
    _lens = cfg.get('eval_lens')
    if not isinstance(_lens, (list, tuple)) or not _lens:
        raise ValueError('_probe_fingerprint: eval_lens must be a non-empty list')
    _need = 0
    for _x in _lens:
        _xi = L._as_seed_int(_x)
        if _xi is None or _xi < 2:
            raise ValueError(
                f'_probe_fingerprint: eval_len={_x!r} must be an exact integer >= 2')
        _need = max(_need, _xi)
    _need += 1
    _tl = L._as_seed_int(cfg.get('train_len') or P3MT_PAYLOAD['train_len'])
    _target = L._as_seed_int(cfg.get('target'))
    _vocab = L._as_seed_int(cfg.get('vocab') or P3MT_PAYLOAD['vocab'])
    _n_seq = L._as_seed_int(cfg.get('n_seq'))
    for _name, _iv in (('train_len', _tl), ('target', _target),
                       ('vocab', _vocab), ('n_seq', _n_seq)):
        if _iv is None or _iv < 1:
            raise ValueError(
                f'_probe_fingerprint: {_name}={_iv!r} must be a positive exact integer')
    return {'n_seq': int(_n_seq), 'target': int(_target), 'vocab': int(_vocab),
            'val_seq_len': max(_tl, _need), 'stat': 'ppl_pooled_nll_v5'}

def _probe_fp_for_mode(fp, cfg, mode):
    """Add the parameters a corruption mode actually consumes.

    `span` corruption changes with `span_len`; the existing probe_params for
    every other mode are kept byte-identical so their resume stays valid.
    Without this, changing span_len would silently reuse measurements made
    with a different span geometry."""
    if mode not in _CORRUPT_MODES:
        raise ValueError(
            f'_probe_fp_for_mode: unknown corruption mode {mode!r} '
            f'(expected one of {_CORRUPT_MODES})')
    if mode == 'span':
        return dict(fp, span_len=_probe_span_len(cfg))
    return fp

def _probe_record_problem(rec):
    """Why a stored probe record is not a usable measurement (None = usable).

    Grouping keys must never contain unhashable JSON values and must never
    describe a geometry the probe cannot produce: variant/arm are text names,
    seed and eval_len are exact integers (eval_len >= 2), rho is a finite
    fraction, mode is implemented, and all-or-nothing modes use rho=1.  The
    record-specific `ppl_mean`/fingerprint checks stay with each reader."""
    if not isinstance(rec, dict):
        return 'not a dict'
    _v = rec.get('variant')
    if not isinstance(_v, str) or not _v:
        return f'variant {_v!r} is not a non-empty text name'
    _arm = rec.get('arm')
    if not isinstance(_arm, str) or not _arm:
        return f'arm {_arm!r} is not a non-empty text name'
    if L._as_seed_int(rec.get('seed')) is None:
        return f'seed {rec.get("seed")!r} is not an exact integer'
    _ln = L._as_seed_int(rec.get('eval_len'))
    if _ln is None or _ln < 2:
        return (f'eval_len {rec.get("eval_len")!r} is not an exact integer '
                '>= 2 (a length-1 cell has no next-token target)')
    _rho_raw = rec.get('rho')
    if isinstance(_rho_raw, (bool, np.bool_)):
        return f'rho {_rho_raw!r} is a boolean flag, not a corruption fraction'
    _rho = L._finite_scalar(_rho_raw)
    if _rho is None or not (0.0 <= _rho <= 1.0):
        return f'rho {_rho_raw!r} is not a finite fraction in [0,1]'
    _mode = rec.get('mode', 'uniform')
    if not isinstance(_mode, str) or _mode not in _CORRUPT_MODES:
        return (f'mode {_mode!r} is not one of {_CORRUPT_MODES}')
    if _mode in ('mask', 'delete', 'doc') and abs(_rho - 1.0) > 1e-9:
        return (f'mode {_mode!r} is all-or-nothing and requires rho=1.0 '
                f'(got {_rho_raw!r})')
    _pp = rec.get('probe_params')
    if not isinstance(_pp, dict) or not _pp:
        return 'missing probe_params geometry (the measurement cannot be paired)'
    return None

def _niah_record_problem(rec):
    """Why a stored NIAH-natural record is not a usable measurement (None = ok).

    The reader builds grouping keys from variant/arm/eval_len/condition, so
    those fields must be hashable text/exact integers.  Geometry follows the
    same contract as the corruption probe: a length-1 cell has no scored
    target.
    """
    if not isinstance(rec, dict):
        return 'not a dict'
    for _f in ('variant', 'arm'):
        _v = rec.get(_f)
        if not isinstance(_v, str) or not _v:
            return f'{_f} {_v!r} is not a non-empty text name'
    if L._as_seed_int(rec.get('seed')) is None:
        return f'seed {rec.get("seed")!r} is not an exact integer'
    _ln = L._as_seed_int(rec.get('eval_len'))
    if _ln is None or _ln < 2:
        return (f'eval_len {rec.get("eval_len")!r} is not an exact integer '
                '>= 2 (a length-1 cell has no scored target)')
    _cond = rec.get('condition')
    if not isinstance(_cond, str) or not _cond:
        return f'condition {_cond!r} is not a non-empty text name'
    _pp = rec.get('probe_params')
    if not isinstance(_pp, dict) or not _pp:
        return 'missing probe_params geometry (the task cells cannot be paired)'
    return None


def _fp_norm(p):
    if not isinstance(p, dict):
        return p
    return {k: v for k, v in p.items() if k not in ('chunk', 'mode')}

def _val_fp_key(value):
    """Hashable grouping key that preserves an invalid val_fp as invalid.

    Probe records are grouped before the pairing gate runs.  An invalid value
    must stay distinct from the literal string 'True'; the content-identity
    check depends on that separation.
    """
    text = L._content_fp_text(value)
    if text is not None:
        return text
    from data_guard import _UnhashableKey
    return _UnhashableKey(value)

def _val_fp_value(key):
    """The fingerprint to publish on a grouped cell (None if invalid)."""
    return key if isinstance(key, str) else None

def _recipe_norm(recipe):
    if not isinstance(recipe, str):
        return recipe
    # recipe fields are '_' separated (see exp_lib run_cfg); wildcard only the
    # mlp_match_ref field, not everything up to the '-' inside the df hashes.
    return re.sub(r'_mr[^_]*', '_mr*', recipe)

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
    the uniform-only protocol (cfg['rhos'])."""
    if not isinstance(cfg, dict):
        raise ValueError(f'_probe_defs: cfg is {type(cfg).__name__}, not an object')
    _modes = cfg.get('modes') or {'uniform': list(cfg['rhos'])}
    if not isinstance(_modes, dict):
        raise ValueError('_probe_defs: modes must be an object of mode -> rho list')
    _lens = cfg.get('eval_lens')
    if not isinstance(_lens, (list, tuple)) or not _lens:
        raise ValueError('_probe_defs: eval_lens must be a non-empty list')
    _lens_i = []
    for Ln in _lens:
        _iv = L._as_seed_int(Ln)
        if _iv is None or _iv < 2:
            raise ValueError(
                f'_probe_defs: eval_len={Ln!r} must be an exact integer >= 2')
        _lens_i.append(_iv)
    out = []
    for _mode, _rhos in _modes.items():
        if not isinstance(_mode, str) or not _mode:
            raise ValueError(f'_probe_defs: mode {_mode!r} must be a non-empty text name')
        if _mode not in _CORRUPT_MODES:
            raise ValueError(
                f'_probe_defs: unknown corruption mode {_mode!r} '
                f'(expected one of {_CORRUPT_MODES})')
        if not isinstance(_rhos, (list, tuple)):
            raise ValueError(f'_probe_defs: rhos for mode {_mode!r} must be a list')
        _rhos_f = []
        for rho in _rhos:
            if isinstance(rho, (bool, np.bool_, np.datetime64, np.timedelta64,
                                complex, np.complexfloating)):
                raise ValueError(f'_probe_defs: rho={rho!r} is a boolean/temporal/complex value, not a density')
            try:
                _rf = float(rho)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(f'_probe_defs: rho={rho!r} is not numeric') from None
            if not (math.isfinite(_rf) and 0.0 <= _rf <= 1.0):
                raise ValueError(f'_probe_defs: rho={rho!r} must be in [0,1]')
            _rhos_f.append(_rf)
        for Ln in _lens_i:
            for rho in _rhos_f:
                out.append((Ln, rho, _mode))
    return out

def _niah_cell_key(v, seed, arm, Ln, cond):
    return f'{v}::s{seed}::{arm}::L{Ln}::niah::{cond}'

def _niah_fingerprint(cfg, condition=None):
    if not isinstance(cfg, dict):
        raise ValueError(f'_niah_fingerprint: cfg is {type(cfg).__name__}, not an object')
    _n = cfg.get('niah') or {}
    if not isinstance(_n, dict):
        raise ValueError('_niah_fingerprint: niah must be an object')
    _lens = cfg.get('eval_lens')
    if not isinstance(_lens, (list, tuple)) or not _lens:
        raise ValueError('_niah_fingerprint: eval_lens must be a non-empty list')
    _need = 0
    for _x in _lens:
        _xi = L._as_seed_int(_x)
        if _xi is None or _xi < 2:
            raise ValueError(
                f'_niah_fingerprint: eval_len={_x!r} must be an exact integer >= 2')
        _need = max(_need, _xi)
    _need += 1
    _tl = L._as_seed_int(cfg.get('train_len') or P3MT_PAYLOAD['train_len'])
    _target = L._as_seed_int(cfg.get('target'))
    _vocab = L._as_seed_int(cfg.get('vocab') or P3MT_PAYLOAD['vocab'])
    _n_seq = L._as_seed_int(_n.get('n_seq', cfg.get('n_seq')))
    _n_pairs = L._as_seed_int(_n.get('n_pairs', 8))
    for _name, _iv in (('train_len', _tl), ('target', _target),
                       ('vocab', _vocab), ('n_seq', _n_seq),
                       ('n_pairs', _n_pairs)):
        if _iv is None or _iv < 1:
            raise ValueError(
                f'_niah_fingerprint: {_name}={_iv!r} must be a positive exact integer')
    if condition not in (None, 'clean', 'distract', 'nofar'):
        raise ValueError(
            f"_niah_fingerprint: condition={condition!r} must be None, 'clean', "
            "'distract' or 'nofar'")
    if condition == 'nofar':
        # nofar is a different task construction (couples destroyed): keep its
        # geometry visibly distinct from clean/distract in probe_params.
        _fp_nf = {'n_seq': int(_n_seq), 'target': int(_target), 'vocab': int(_vocab),
                  'n_pairs': int(_n_pairs), 'val_seq_len': max(_tl, _need),
                  'stat': 'niah_nofar_v1'}
        return _fp_nf
    fp = {'n_seq': int(_n_seq), 'target': int(_target), 'vocab': int(_vocab),
          'n_pairs': int(_n_pairs), 'val_seq_len': max(_tl, _need),
          'stat': 'niah_natural_v1'}
    if condition == 'distract':
        # The distractor condition's corruption density is part of the
        # measurement; the clean condition does not read rho.
        _rho = _n.get('rho', 0.5)
        if isinstance(_rho, (bool, np.bool_, np.datetime64, np.timedelta64,
                             complex, np.complexfloating)):
            raise ValueError('_niah_fingerprint: rho is a boolean/temporal/complex value, not a density')
        try:
            _rho = float(_rho)
        except (TypeError, ValueError, OverflowError):
            raise ValueError('_niah_fingerprint: rho is not numeric') from None
        if not (math.isfinite(_rho) and 0.0 <= _rho <= 1.0):
            raise ValueError('_niah_fingerprint: rho must be in [0,1]')
        fp['dist_rho'] = _rho
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
    # Geometry is declared with exact integers; malformed values are refused
    # up front.
    _ints = {}
    for _name, _value in (('eval_len', eval_len), ('target', target),
                          ('n_pairs', n_pairs), ('vocab', vocab),
                          ('n_seq', n_seq)):
        _iv = L._as_seed_int(_value)
        if _iv is None:
            raise ValueError(
                f'NIAH-natural: {_name}={_value!r} must be an exact integer')
        _ints[_name] = _iv
    eval_len, target = _ints['eval_len'], _ints['target']
    n_pairs, vocab, n_seq = _ints['n_pairs'], _ints['vocab'], _ints['n_seq']
    if eval_len < 2 or target < 1 or n_pairs < 1 or vocab < 4 or n_seq < 1:
        raise ValueError(
            'NIAH-natural needs eval_len>=2, target>=1, n_pairs>=1, '
            f'vocab>=4 and n_seq>=1 (got eval_len={eval_len}, target={target}, '
            f'n_pairs={n_pairs}, vocab={vocab}, n_seq={n_seq})')
    _seed = L._as_seed_int(seed)
    if _seed is None:
        raise ValueError(f'NIAH-natural: seed must be an exact integer, got {seed!r}')
    if condition not in ('clean', 'distract', 'nofar'):
        raise ValueError(
            "NIAH-natural: condition must be 'clean', 'distract' or 'nofar' "
            f'(got {condition!r}); `nofar` deletes the far key->value couples so '
            'the answer is not recoverable from the far context (a no-far-info '
            'control: accuracy there must collapse to chance)')
    if isinstance(rho, (bool, np.bool_, np.datetime64, np.timedelta64,
                        complex, np.complexfloating)):
        raise ValueError(f'NIAH-natural: rho={rho!r} is a boolean/temporal/complex value, not a density')
    try:
        rho = float(rho)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'NIAH-natural: rho={rho!r} is not numeric') from None
    if not (math.isfinite(rho) and 0.0 <= rho <= 1.0):
        raise ValueError(f'NIAH-natural: rho={rho!r} must be in [0,1]')
    _rows = np.asarray(val_ids)
    if _rows.ndim != 2:
        raise ValueError(
            f'NIAH-natural: val_ids must be a 2-D token matrix, got shape {_rows.shape}')
    if n_seq > _rows.shape[0]:
        raise ValueError(
            f'NIAH-natural: n_seq={n_seq} exceeds the available {_rows.shape[0]} '
            'validation rows')
    if eval_len + 1 > _rows.shape[1]:
        raise ValueError(
            f'NIAH-natural: eval_len+1={eval_len + 1} exceeds the validation '
            f'row width {_rows.shape[1]}')
    _key_hi = max(4, vocab // 4)
    if _key_hi - 4 < n_pairs:
        raise ValueError(
            f'NIAH-natural: vocab={vocab} has only {_key_hi - 4} usable key '
            f'tokens for {n_pairs} distinct keys')
    if vocab - (3 * vocab // 4) < n_pairs:
        raise ValueError(
            f'NIAH-natural: vocab={vocab} has only '
            f'{vocab - (3 * vocab // 4)} usable value tokens for {n_pairs} '
            'distinct values')
    val_ids = _rows
    rng = np.random.default_rng(_seed)
    n_far = eval_len - target
    if n_far < 4 * n_pairs + 8:
        raise ValueError(f'NIAH-natural: far region ({n_far} tokens) too small for {n_pairs} couples at eval_len={eval_len}')
    if 2 * n_pairs > target - 1:
        raise ValueError(f'NIAH-natural: {n_pairs} query couples do not fit inside the {target}-token target zone')
    key_hi = max(4, vocab // 4)
    val_lo = 3 * vocab // 4
    # Distractor tokens come from another validation row.  With a one-row
    # panel the donor mapping wraps back to the scored row, so the condition
    # would be a clean no-op recorded as distractor corruption.
    if condition == 'distract' and int(val_ids.shape[0]) < 2:
        raise ValueError(
            'NIAH-natural distract needs at least two validation rows '
            f'(got {int(val_ids.shape[0])}); the only possible donor is the '
            'scored row itself, so the distractor would be a no-op')
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
        if condition == 'nofar':
            # No-far-information control: overwrite BOTH far couple tokens with
            # donor-row natural tokens, so the query key's answer is nowhere in
            # the far context.  The query keys in the target zone are kept.  A
            # model scoring above chance here means the answer leaks from the
            # local target context and the task is invalid as a far-necessity
            # probe.
            for _jp in range(n_pairs):
                _kp = _jp * band + (band // 2 if band > 2 else 1)
                _kp = min(_kp, n_far - 2)
                rows[i, _kp] = np.asarray(val_ids[_src_rows[i], _kp], dtype=np.int64)
                rows[i, _kp + 1] = np.asarray(val_ids[_src_rows[i], _kp + 1], dtype=np.int64)
        if condition == 'distract':
            k = int(round(rho * n_far))
            if rho > 0.0:
                k = max(1, k)
            k = min(k, n_far)
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
    _n_value_tokens = int(vocab - val_lo)
    stats = {'n_pairs': int(n_pairs), 'eval_len': int(eval_len), 'target': int(target),
             'condition': str(condition),
             'rho': float(rho) if condition == 'distract' else 0.0,
             'dist_mean': float(dists.mean()), 'dist_min': int(dists.min()),
             'dist_max': int(dists.max()),
             # Review 10, issue 13: publish the chance level of the task (the
             # answer is one of `_n_value_tokens` value tokens, sampled
             # uniformly) so "chance-level accuracy" can be judged against the
             # actual task, not assumed.  Positive-control solvers are NOT part
             # of the task construction; without one the analysis refuses to
             # treat NIAH as mechanism evidence.
             'chance_acc': (1.0 / _n_value_tokens) if _n_value_tokens > 0 else None,
             'n_value_tokens': _n_value_tokens,
             'dist_per_query': [int(x) for x in dists.reshape(-1).tolist()],
             'couple_positions_excluded_from_distract': True,
             'nofar_control': bool(condition == 'nofar')}
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
    _correct_parts = []
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
        _correct = (_lg.argmax(-1) == _lv)
        n_correct += int(_correct.sum())
        n_q += int(_lv.numel())
        _correct_parts.append(_correct.detach().cpu().reshape(-1).to(torch.bool))
        nll_parts.append(_nll)
        del ids, logits, _lg, _lv, _nll
        if chunk_mem is not None:
            chunk_mem[(eval_len, 'niah')] = n_ch
        i = j
    if not nll_parts:
        return {'acc': float('nan'), 'nll': float('nan'), 'n': 0, **stats}
    _nll_all = torch.cat(nll_parts)
    out = {'acc': float(n_correct) / max(n_q, 1), 'nll': float(_nll_all.mean()),
           'n': int(n_q), **stats}
    # Distance-bucket accuracy (review 10, issue 13): a far-necessity task must
    # show a short-to-long distance gradient; a flat/chance profile means the
    # task does not exercise distance in this model/format.  Buckets are
    # equal-count quantiles of the ACTUAL query distances in this cell.
    try:
        _corr = torch.cat(_correct_parts).numpy()
        _dists = np.asarray(stats.get('dist_per_query') or [], dtype=np.int64)
        if _corr.size == _dists.size and _corr.size:
            _order = np.argsort(_dists, kind='stable')
            _buckets = []
            _nb = 4
            for _bi in range(_nb):
                _lo = (_bi * _corr.size) // _nb
                _hi = ((_bi + 1) * _corr.size) // _nb
                if _hi <= _lo:
                    continue
                _idx = _order[_lo:_hi]
                _buckets.append({'bucket': _bi,
                                 'dist_min': int(_dists[_idx].min()),
                                 'dist_max': int(_dists[_idx].max()),
                                 'n': int(_idx.size),
                                 'acc': float(_corr[_idx].mean())})
            out['acc_by_distance_bucket'] = _buckets
            out['distance_gradient'] = ('unavailable' if len(_buckets) < 2 else
                                        ('positive' if _buckets[0]['acc'] <= _buckets[-1]['acc']
                                         else 'negative'))
    except (TypeError, ValueError, RuntimeError):
        pass
    return out

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
    # records without the field keep their existing resume behaviour.
    # v12 reuses this cell with val_ids=test_ids, so the frozen test slice gets
    # the same per-record fingerprint.
    _val_fp = L._ids_fp(np.ascontiguousarray(val_ids).reshape(-1))
    if mech_summ is None:
        mech_summ = {}
        _msp = os.path.join(cfg.get('outdir') or P3MT_PAYLOAD['outdir'], 'summary.json')
        if os.path.exists(_msp):
            try:
                with open(_msp, encoding='utf-8') as _jf:
                    mech_summ = json.load(_jf)
            except Exception as _e:
                print(f'[p3mp] FATAL: {_msp} exists but cannot be parsed ({type(_e).__name__}: {_e}) — without it no checkpoint can be attributed to a training recipe, so the probe would score unknown weights.  Move it aside to re-train.')
                raise
            if not isinstance(mech_summ, dict):
                print(f'[p3mp] FATAL: {_msp} is valid JSON but not an object ({type(mech_summ).__name__}) — without a record table no checkpoint can be attributed to a training recipe.  Move it aside to re-train.')
                raise SystemExit(1)
    ck = os.path.join(cfg['ckpt_dir'], L.direct_ckpt_name(v, seed))
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

    def _cell_identity_matches(_key, _rec):
        # The summary key names the requested cell; the record's own fields
        # must agree with it.  Without this check a merged/hand-edited record
        # stored under one key could silently skip a different (variant, seed,
        # arm, length, rho, mode) cell.
        if not isinstance(_rec, dict):
            return False
        try:
            _p = str(_key).split('::')
            if (_p[0] != str(v) or L._as_seed_int(_p[1][1:]) != int(seed)
                    or _p[2] != str(_rec.get('arm'))
                    or L._as_seed_int(_p[3][1:]) != L._as_seed_int(_rec.get('eval_len'))):
                return False
            if len(_p) >= 5 and _p[3].startswith('L') and _p[4].startswith('r'):
                _rh = float(_p[4][1:])
                _rr = _rec.get('rho')
                if _rr is None or not math.isfinite(float(_rr)) or abs(float(_rr) - _rh) > 1e-12:
                    return False
                _kmode = _p[5] if len(_p) > 5 else 'uniform'
                if _rec.get('mode') != _kmode:
                    return False
        except (TypeError, ValueError, IndexError, OverflowError):
            return False
        return (_rec.get('variant') == v and L._as_seed_int(_rec.get('seed')) == int(seed))

    _msha_probe = cfg.get('manifest_sha256')

    def _cell_stale(_key, _fp_x, _summary=None, _mode='uniform'):
        _sum = (_summary if _summary is not None else L.locked_json_read(spath)) or {}
        _rec = _sum.get(_key)
        if not _cell_identity_matches(_key, _rec):
            return True
        # Records from a different freeze revision must not be resume-skipped
        # (nor recorded under a plan they were not measured against).
        if _msha_probe is not None and (_rec or {}).get('manifest_sha256') != _msha_probe:
            return True
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
        if not _cell_identity_matches(_key, _rec):
            return True
        if _msha_probe is not None and (not isinstance(_rec, dict) or _rec.get('manifest_sha256') != _msha_probe):
            return True
        _acc = L._finite_scalar(_rec.get('acc')) if isinstance(_rec, dict) else None
        _nll = L._finite_scalar(_rec.get('nll')) if isinstance(_rec, dict) else None
        try:
            _p = str(_key).split('::')
            _cond = _p[5]
        except (IndexError, TypeError):
            return True
        return not (isinstance(_rec, dict) and (_rec.get('task') == 'niah_natural') and (_rec.get('condition') == _cond) and _acc is not None and _nll is not None and _rec.get('_code') == V.CKPT_CODE and _rec.get('probe_params') == (_niah_fp_of('clean') if _fp_n is None else _fp_n) and _rec.get('recipe') == _ck_recipe and (_rec.get('val_fp') is None or _rec.get('val_fp') == _val_fp))
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
            print(f'[cap] SKIP probe {v} s{seed} arm {arm}: headroom exhausted ({max(guard.headroom_hours(), 0.0):.2f} h left) — re-run the phase to continue')
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
            per_seq = [(_ppl if _ppl is not None else float('inf'))
                       for _ppl in (L._safe_exp_ppl(s / t) for s, t in zip(nll_sum, n_tok))]
            _fp_base = _probe_fp_for_mode(_fp, _pcfg, mode)
            _fp_rec = dict(_fp_base, mode=mode) if mode != 'uniform' else _fp_base
            _corr_audit = _corruption_audit(val_ids, Ln, rho, mode, _pcfg['n_seq'], _pcfg['vocab'], _pcfg['target'], span_len=_probe_span_len(_pcfg))
            _pending[key] = {'variant': v, 'seed': seed, 'arm': arm, 'eval_len': Ln, 'rho': rho, 'mode': mode, 'corruption_audit': _corr_audit, 'ppl_mean': float(cell_ppl), 'ppl_std_per_seq': float(np.std(per_seq, ddof=1)) if len(per_seq) > 1 else 0.0, 'ppls': [float(p) for p in per_seq], 'nll_sum': [float(x) for x in nll_sum], 'n_tok': [int(x) for x in n_tok], 'n_seq': len(per_seq), 'target': cfg['target'], 'val_fp': _val_fp, '_code': V.CKPT_CODE, 'probe_params': _fp_rec, 'recipe': _ck_recipe, 'train_knobs': _tr.get('train_knobs'), 'train_objective': _tr.get('train_objective'), 'param_breakdown': _tr.get('param_breakdown'), 'rng_fp': _tr.get('rng_fp'), 'train_fp': _tr.get('data_fp'), 'manifest_sha256': cfg.get('manifest_sha256')}
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
            rec = {'variant': v, 'seed': seed, 'arm': arm, 'eval_len': int(Ln), 'task': 'niah_natural', 'condition': cond, 'acc': float(res['acc']), 'nll': float(res['nll']), 'n_queries': int(res['n']), 'n_pairs': int(res.get('n_pairs', (_niah_cfg or {}).get('n_pairs', 8))), 'dist_mean': float(res.get('dist_mean', float('nan'))), 'dist_min': int(res.get('dist_min', -1)), 'dist_max': int(res.get('dist_max', -1)), 'rho': float(res.get('rho', 0.0)), 'chance_acc': res.get('chance_acc'), 'n_value_tokens': res.get('n_value_tokens'), 'acc_by_distance_bucket': res.get('acc_by_distance_bucket'), 'distance_gradient': res.get('distance_gradient'), 'val_fp': _val_fp, '_code': V.CKPT_CODE, 'probe_params': _niah_fp_of(cond), 'recipe': _ck_recipe, 'manifest_sha256': cfg.get('manifest_sha256')}
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
            with open(_msp, encoding='utf-8') as _jf:
                mech_summ = json.load(_jf)
        except Exception as _e:
            print(f'[p3mp] FATAL: {_msp} exists but cannot be parsed ({type(_e).__name__}: {_e}) — without it no checkpoint can be attributed to a training recipe, so the probe would score unknown weights.  Move it aside to re-train.')
            raise
        if not isinstance(mech_summ, dict):
            print(f'[p3mp] FATAL: {_msp} is valid JSON but not an object ({type(mech_summ).__name__}) — without a record table no checkpoint can be attributed to a training recipe.  Move it aside to re-train.')
            raise SystemExit(1)
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
    _measured_pairs = {(_r.get('variant'), L._as_seed_int(_r.get('seed')))
                       for _r in summary.values()
                       if isinstance(_r, dict) and not _r.get('synthesized')
                       and _r.get('_code') == V.CKPT_CODE
                       and (L.ppl_is_usable(_r.get('ppl_mean')) or _r.get('task') == 'niah_natural')}
    # Only planned cells count: stale or extra summary records do not
    # contribute to the measured total.
    _measured = sum(1 for _v, _sd in cells if (_v, L._as_seed_int(_sd)) in _measured_pairs)
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
# ---- 规模对照臂：大模型配置与运行期步数推导 ----
# steps 运行期由 _flop_matched_steps 确定性计算，避免在 import 期
# 构建模型——§7 导入零副作用。
P3SL_TOK_CFG = dict(P3SL_CFG, outdir='results_lm_v10_scale_l8k', steps=8000)
P3SL_FLOP_CFG = dict(P3SL_CFG, outdir='results_lm_v10_scale_lflop', steps=None)

def _flop_matched_steps(small_cfg, large_cfg, min_steps=250):
    """Deterministic parameter-count-proportional step count for the large arm.

    Build both panel configs on CPU (no DEVICE touch, no RNG side effects),
    take the mean parameter count over the arms, and scale steps by
    N_small / N_large.  The full derivation is printed; repeated calls with
    the same code are bitwise identical.  Steps are floored at 250 so the LR
    schedule still has room after warmup=100.
    """
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
    _ambiguous = set()
    if raw is None:
        if not os.path.exists(sp):
            return out
        try:
            with open(sp, encoding='utf-8') as _jf:
                raw = json.load(_jf)
        except Exception as e:
            print(f'[stats] WARNING: cannot read {sp} ({type(e).__name__}: {e}) — no curve for {variant}')
            return out
    if not isinstance(raw, dict):
        print(f'[stats] WARNING: {sp} is valid JSON but not an object ({type(raw).__name__}) — no curve for {variant}')
        return out
    seen = {}
    cfg_of = {}
    for _k, r in raw.items():
        if not (isinstance(r, dict) and r.get('variant') == variant and r.get('ppl_history')):
            continue
        # Use the shared exact-integer contract: this also rejects np.float32
        # fractions (int() would truncate 1.5 -> 1) and lossy floats >= 2**53.
        _s_raw = r.get('seed')
        s = L._as_seed_int(_s_raw)
        if s is None:
            print(f'[v10 stats] {_k} has no usable `seed` ({_s_raw!r}); its curve cannot be attributed and is skipped')
            continue
        if s in _ambiguous:
            continue
        try:
            _fp_parts = []
            for _t_raw, _v_raw in r['ppl_history']:
                _vf = L._finite_scalar(_v_raw)
                if _vf is None:
                    # A boolean/NaN point is not a measurement and is dropped
                    # by the clean-history pass below; keep the exact step in
                    # the identity so a fractional step cannot alias an int.
                    continue
                _ti = L._as_seed_int(_t_raw)
                _fp_parts.append((_ti if _ti is not None else ('bad-step', repr(_t_raw)),
                                  type(_v_raw).__name__, round(_vf, 9)))
            fp = tuple(_fp_parts)
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
                print(f'[stats] {sp}: two records with DIFFERENT curves both claim ({variant}, seed {s}) — the gap trajectory is ambiguous; the whole seed slot is EXCLUDED rather than choosing an order-dependent winner')
                out.pop(s, None)
                synth_of.pop(s, None)
                cfg_of.pop(s, None)
                for _fp_old, _s_old in list(seen.items()):
                    if _s_old == s:
                        seen.pop(_fp_old)
                _ambiguous.add(s)
                continue
        seen[fp] = s
        synth_of[s] = bool(r.get('synthesized'))
        cfg_of[s] = r.get('run_cfg')
        _clean_hist = []
        _n_badpt = 0
        for _pt in r['ppl_history']:
            try:
                _t = L._as_seed_int(_pt[0])
                _v2 = L._finite_scalar(_pt[1])
            except (TypeError, ValueError, IndexError):
                _n_badpt += 1
                continue
            # Exact integer steps only: int(1.5)==1 would merge two distinct
            # evaluation points, and float(True)==1.0 would turn a flag into a
            # PPL measurement.
            if _t is None or _v2 is None or _t < 0 or not (_v2 > 0):
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
            # Tie-break on the group's serialised run_cfg bytes: summary.json
            # iteration order must not decide which tied group is kept.
            _ranked = sorted(_groups.items(), key=lambda kv: (not any((isinstance(cfg_of.get(_s), str) and cfg_of[_s].endswith(_cur_sfx) for _s in kv[1])), -len(kv[1]), kv[0]))
            _keep = _ranked[0][1]
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
            with open(sp, encoding='utf-8') as _jf:
                raw = json.load(_jf)
        except Exception as e:
            print(f'[stats] WARNING: cannot read {sp} ({type(e).__name__}: {e}) — no tokens/step for this panel')
            return None
    if not isinstance(raw, dict):
        print(f'[stats] WARNING: {sp} is valid JSON but not an object ({type(raw).__name__}) — no tokens/step for this panel')
        return None
    rates = []
    n_bad = 0
    for _k, r in raw.items():
        if not isinstance(r, dict) or r.get('synthesized'):
            continue
        _ts, _st = (r.get('tokens_seen'), r.get('steps'))
        if isinstance(_ts, (bool, np.bool_)) or isinstance(_st, (bool, np.bool_)):
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
            except Exception as _e:
                print(f'[stats] WARNING: {label}: { _sp} exists but cannot be parsed ({type(_e).__name__}: {_e}) — treating as missing/unreadable, not empty')
                out[label] = {'status': 'unreadable', **p,
                              'error': f'{type(_e).__name__}: {_e}'}
                continue
            if not isinstance(_raw, dict):
                print(f'[stats] WARNING: {label}: {_sp} is {type(_raw).__name__}, not an object — treating as unreadable')
                out[label] = {'status': 'unreadable', **p,
                              'error': f'not an object: {type(_raw).__name__}'}
                continue
        ha = _hist_by_seed(p['outdir'], 'csa_fixed', raw=_raw)
        hb = _hist_by_seed(p['outdir'], 'full', raw=_raw)
        _stale_seeds = sorted(getattr(ha, 'stale_cfg_seeds', set()) | getattr(hb, 'stale_cfg_seeds', set()))
        if _stale_seeds:
            print(f'[stats] {label}: seed(s) {_stale_seeds} were measured under an older code-stamp — a code change must invalidate and recompute, not be reused, so they are excluded from this {label} crossover fit')
        common = sorted((set(ha) & set(hb)) - set(_stale_seeds))
        _synth_seeds = sorted(s for s in common
                              if getattr(ha, 'per_seed_synth', {}).get(s) or getattr(hb, 'per_seed_synth', {}).get(s))
        if _synth_seeds:
            # A reconstructed curve is a copy of a prior aggregate, not an
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
        # The first-positive/no-recross rule depends on the smoothing window;
        # publish the raw and +/-1/2-point variants so a crossing that exists
        # only at one smoothing level is visible rather than implicit.
        _smooth_sens = {}
        for _sm in (0, 1, 2):
            _x_s, _c_s = _crossover_step(steps, gap_mean, smooth=_sm)
            _smooth_sens[str(_sm)] = {'step': _x_s, 'crossed': bool(_c_s)}
        entry = {'status': 'ok', **p, 'seeds': common, 'excluded_reconstructed': _synth_seeds, 'tokens_per_step': tps, 'per_seed': per_seed, 'per_seed_synth': {str(s): bool(getattr(ha, 'per_seed_synth', {}).get(s) or getattr(hb, 'per_seed_synth', {}).get(s)) for s in common}, 'mean_crossover_step': xo_m, 'mean_crossed': crossed_m, 'mean_traj': {'steps': steps, 'gap': gap_mean}, 'smoothing_sensitivity': _smooth_sens,
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

_SCALE_PANEL_SPECS = {'P3SS': dict(outdir='results_lm_v10_scale_s', d=128, n_layers=4, batch=12, seq=512, steps=8000, note='small arm'), 'P3SL': dict(outdir='results_lm_v10_scale_l', d=512, n_layers=10, batch=12, seq=512, steps=4000, note='large arm'), 'P3SL_TOK': dict(outdir='results_lm_v10_scale_l8k', d=512, n_layers=10, batch=12, seq=512, steps=8000, note='token-matched large arm'), 'P3SL_FLOP': dict(outdir='results_lm_v10_scale_lflop', d=512, n_layers=10, batch=12, seq=512, steps=None, note='6ND-param-proxy-budget large arm (NOT measured FLOPs)')}

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
    if not (isinstance(out.get('p_floor'), (int, float)) and math.isfinite(float(out['p_floor']))):
        # All-zero deltas make the exact test return p_floor=NaN; the
        # RESOLUTION of the test is still 2/2^n and must be displayed.
        _n = int(out.get('n') or 0)
        out['p_floor'] = math.ldexp(2.0, -_n) if _n > 0 else float('nan')
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
            # Point nearest the 75th-percentile step (ties break to the
            # earlier point): the window stays a true final quarter and
            # never collapses to zero width on short grids.
            _t_mark = pts[0][0] + 0.75 * (pts[-1][0] - pts[0][0])
            q = min(pts, key=lambda p: abs(p[0] - _t_mark))
            last = pts[-1]
            denom = max(abs(last[1]), 1e-9)
            rel = (q[1] - last[1]) / denom
            per_seed[s] = {'rel_improve_last_quarter': float(rel), 'plateaued': bool(rel < 0.01)}
        _vals = [x['rel_improve_last_quarter'] for x in per_seed.values() if math.isfinite(x['rel_improve_last_quarter'])]
        _n_insufficient = sum(1 for x in per_seed.values() if not math.isfinite(x['rel_improve_last_quarter']))
        out[v] = {'per_seed': per_seed, 'median_rel_improve': float(np.median(_vals)) if _vals else float('nan'), 'n_plateaued': int(sum((1 for x in per_seed.values() if x['plateaued']))), 'n': len(per_seed), 'n_considered': len(_vals), 'n_insufficient_evals': int(_n_insufficient), 'diagnostic': 'plateau-in-training-window (NOT proof of convergence: lr decay, stalls or eval noise can also satisfy it)', 'claim_allowed': 'crossing-after-plateau-in-window only; asymptotic/post-convergence claims need longer training + lr-continuation + replication'}
    return out

_PANEL_PARAM_CACHE = {}

def _panel_param_counts(spec):
    """CPU parameter counts for the two panel arms.

    These are structural counts (default mlp_ratio=4), matching the scale
    panels' actual construction.  They are published because the CSA/dense
    capacity gap shrinks with width (P3SS csa +34.6%, P3SL +9.5%), so an
    apparent cross-scale change in the CSA-dense gap cannot be attributed to
    scale before that confound is stated.
    """
    key = (int(spec['d']), int(spec['n_layers']), int(spec['seq']))
    if key in _PANEL_PARAM_CACHE:
        return dict(_PANEL_PARAM_CACHE[key])
    out = {}
    n_heads = max(1, int(spec['d']) // 32)
    for _v in ('csa_fixed', 'full'):
        _cfgs = L.make_layer_cfgs(int(spec['n_layers']), _v)
        _m = L.SmallGPT(8192, int(spec['d']), int(spec['n_layers']),
                        n_heads, 32, int(spec['seq']), _cfgs)
        out[_v] = int(L.count_params(_m))
        del _m
    _PANEL_PARAM_CACHE[key] = dict(out)
    return out

def _equal_token_slices():
    """The scale panels' endpoints confound model size with
    cumulative training tokens.  This analysis re-reads the SAME curves at
    matched token counts (and, when the matched arms exist, at matched total
    compute), so a cross-scale endpoint difference cannot be silently
    attributed to scale alone."""
    conf = {}
    _panel_hists = {}
    try:
        _proxy_steps = _flop_matched_steps(P3SS_CFG, P3SL_CFG)
    except Exception as _e:
        print(f'[stats] WARNING: cannot derive the 6ND-proxy step budget ({type(_e).__name__}: {_e}) — P3SL_FLOP steps will be reported as unknown')
        _proxy_steps = None
    for name, spec in _SCALE_PANEL_SPECS.items():
        _sp = os.path.join(spec['outdir'], 'summary.json')
        _raw = None
        if os.path.exists(_sp):
            try:
                with open(_sp, encoding='utf-8') as _f:
                    _raw = json.load(_f)
                if not isinstance(_raw, dict):
                    print(f'[stats] WARNING: {_sp} is not a JSON object ({type(_raw).__name__}) — ignoring')
                    _raw = None
            except Exception as _e:
                print(f'[stats] WARNING: {_sp} exists but cannot be parsed ({type(_e).__name__}: {_e}) — ignoring')
                _raw = None
        ha = _hist_by_seed(spec['outdir'], 'csa_fixed', raw=_raw)
        hb = _hist_by_seed(spec['outdir'], 'full', raw=_raw)
        _stale = sorted(getattr(ha, 'stale_cfg_seeds', set()) | getattr(hb, 'stale_cfg_seeds', set()))
        for _h in (ha, hb):
            for s in _stale:
                _h.pop(s, None)
        _synth = sorted(s for s in (set(ha) | set(hb))
                        if getattr(ha, 'per_seed_synth', {}).get(s)
                        or getattr(hb, 'per_seed_synth', {}).get(s))
        if _synth:
            # Reconstructed curves are copies of a prior aggregate, not
            # independent runs; they may not enter an endpoint, convergence
            # read or matched-token slice.
            print(f'[stats] {name}: seed(s) {_synth} carry a SYNTHESIZED reconstruction — excluded from the matched-token analysis')
            for _h in (ha, hb):
                for s in _synth:
                    _h.pop(s, None)
        _cfg_a = getattr(ha, 'per_seed_cfg', {})
        _cfg_b = getattr(hb, 'per_seed_cfg', {})
        _cfg_bad = sorted(s for s in (set(ha) & set(hb))
                          if _cfg_a.get(s) != _cfg_b.get(s))
        if _cfg_bad:
            print(f'[stats] {name}: seed(s) {_cfg_bad} pair a `csa_fixed` curve against a `full` curve measured under a DIFFERENT run_cfg — excluded from the matched-token analysis')
            for _h in (ha, hb):
                for s in _cfg_bad:
                    _h.pop(s, None)
        _panel_hists[name] = {'csa_fixed': ha, 'full': hb}
        hists = _panel_hists[name]
        tps = _tokens_per_step(spec['outdir'], fallback=spec['batch'] * spec['seq'], raw=_raw)
        entry = {'status': 'ok', **{k: v for k, v in spec.items() if k != 'steps'}, 'tokens_per_step': tps, 'convergence': _panel_convergence(hists) if (ha or hb) else {}}
        if spec['steps'] is not None:
            entry['declared_steps'] = spec['steps']
            entry['steps_run'] = spec['steps']
            entry['steps_run_source'] = 'declared total steps'
        else:
            entry['steps_run'] = _proxy_steps
            entry['steps_run_source'] = '6ND parameter-ratio proxy (NOT measured FLOPs)'
        if ha or hb:
            try:
                _pc = _panel_param_counts(spec)
                entry['params'] = _pc
                _pf = _pc.get('full')
                entry['param_rel_diff_csa_vs_full'] = ((float(_pc['csa_fixed']) - float(_pf)) / float(_pf)) if _pf else None
            except Exception as _e:
                print(f'[stats] {name}: parameter count failed ({type(_e).__name__}: {_e}) — capacity confound reported as unknown')
                entry['params'] = None
                entry['param_rel_diff_csa_vs_full'] = None
        _ep = _panel_endpoint(hists, tps)
        if _ep is not None:
            entry['endpoint'] = _ep
        else:
            entry['status'] = 'missing'
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
            # Equal TOKEN slices, not equal step indices: when the calibrated
            # tokens/step differs, using the same step compares different
            # cumulative token counts while the label claims equality.
            _hn = _panel_hists.get(name) or {}
            _steps_l = sorted(set().union(*[set(dict(v)) for v in _hn.get('csa_fixed', {}).values()],
                                         *[set(dict(v)) for v in _hn.get('full', {}).values()])) if _hn else []
            for step in _common_steps:
                target_tokens = step * _tps_s
                if _steps_l:
                    step_l = min(_steps_l, key=lambda x: abs(x * _tps_l - target_tokens))
                else:
                    step_l = step
                tokens_large = step_l * _tps_l
                # If the large arm's step grid stops before the small arm's
                # target, the nearest step is its last one; calling that an
                # equal-token slice would compare different totals under an
                # equal label.
                _tok_tol = 0.5 * max(abs(float(_tps_s)), abs(float(_tps_l)))
                if abs(tokens_large - target_tokens) > _tok_tol:
                    print(f'[stats] equal-token slice {name} @ {target_tokens / 1000000.0:.2f}M: the nearest large-arm step {step_l} has {tokens_large / 1000000.0:.2f}M tokens (difference exceeds half a step) — slice OMITTED rather than misattributed')
                    continue
                _g_s = _gap_at('P3SS', step)
                _g_l = _gap_at(name, step_l)
                common = sorted(set(_g_s) & set(_g_l))
                if not common:
                    continue
                dg = [_g_l[s] - _g_s[s] for s in common]
                res = V.exact_sign_permutation(dg)
                if not (isinstance(res.get('p_floor'), (int, float)) and math.isfinite(float(res['p_floor']))):
                    _n_res = int(res.get('n') or 0)
                    res['p_floor'] = math.ldexp(2.0, -_n_res) if _n_res > 0 else float('nan')
                tokens = step * _tps_s
                _steps_s = (conf.get('P3SS') or {}).get('steps_run')
                _steps_l_total = (conf.get(name) or {}).get('steps_run')
                slices[f'{name} - P3SS @ {tokens / 1000000.0:.2f}M tokens (gap_L - gap_S)'] = {'n': res['n'], 'mean': res['mean'], 'std': res.get('std', 0.0), 'p_exact_signflip': res['p_exact_signflip'], 'per_seed': res.get('per_seed'), 'p_floor': res.get('p_floor'), 'n_nonzero': res.get('n_nonzero'), 'cohens_dz': res.get('cohens_dz'), 'seeds': common, 'gap_L': {str(s): _g_l[s] for s in common}, 'gap_S': {str(s): _g_s[s] for s in common}, 'gap_L_mean': float(np.mean([_g_l[s] for s in common])), 'gap_S_mean': float(np.mean([_g_s[s] for s in common])), 'small_step': step, 'large_step': step_l, 'tokens_small': target_tokens, 'tokens_large': tokens_large, 'tokens_per_step_small': _tps_s, 'tokens_per_step_large': _tps_l, 'steps_small': _steps_s, 'steps_large': _steps_l_total, 'lr_schedule_aligned': bool(_steps_s is not None and _steps_l_total is not None and int(_steps_s) == int(_steps_l_total)), 'note': 'positive = the CSA-dense gap is larger in the large arm at (approximately) the same cumulative tokens'}
    return {'panels': conf, 'equal_token_slices': slices}

def v10_analysis(out='analysis_v10/stats.json'):
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    probe = {}
    sp = PROBE['summary']
    if os.path.exists(sp):
        try:
            with open(sp, encoding='utf-8') as _jf:
                raw = json.load(_jf)
        except Exception as _e:
            print(f'[stats] FATAL: {sp} exists but cannot be parsed ({type(_e).__name__}: {_e}) — refusing to compute probe statistics from a truncated file; move it aside to re-probe.')
            raise
        if not isinstance(raw, dict):
            print(f'[v10 stats] WARNING: {sp} is valid JSON but not an object ({type(raw).__name__}) — the probe panel is EXCLUDED')
            raw = {}
        _usable, _bad_cells = ({}, [])
        for _k, r in raw.items():
            if not isinstance(r, dict):
                _bad_cells.append((_k, 'not a dict'))
                continue
            if r.get('synthesized'):
                _bad_cells.append((_k, 'synthesized reconstruction'))
                continue
            _why = _probe_record_problem(r)
            if _why is not None:
                _bad_cells.append((_k, _why))
                continue
            if not L.ppl_is_usable(r.get('ppl_mean')):
                _bad_cells.append((_k, 'ppl_mean is not a usable measurement'))
                continue
            _usable[_k] = dict(
                r, seed=L._as_seed_int(r.get('seed')),
                eval_len=L._as_seed_int(r.get('eval_len')),
                rho=L._finite_scalar(r.get('rho')),
                mode=r.get('mode', 'uniform'))
        if _bad_cells:
            print(f'[v10 stats] {len(_bad_cells)} record(s) in {sp} carry no variant/seed/ppl_mean and are skipped (they are error or truncation stubs, not measurements): ' + ', '.join((f'{k}({why})' for k, why in _bad_cells[:4])))
        _n_bad_pre = len(_bad_cells)
        cells, _dup_cells = ({}, [])
        _dup_keys = set()
        for k, r in _usable.items():
            if r.get('probe_params') is None:
                _bad_cells.append((k, 'no probe_params'))
                continue
            _mode = r['mode']
            # the grouping key carries the mode EXPLICITLY (probe_params' mode
            # field is normalised out by _fp_norm): uniform and natural cells
            # at the same rho must never merge into one pooled cell.
            key = (r['variant'], r['arm'], r['eval_len'], r['rho'], _mode, json.dumps(_fp_norm(r.get('probe_params')), sort_keys=True), str(r.get('_code')), str(r.get('recipe')), _val_fp_key(r.get('val_fp')))
            grp = cells.setdefault(key, {})
            if r['seed'] in grp:
                _dup_cells.append((k, key[:5], r['seed']))
                _dup_keys.add(key)
                continue
            grp[r['seed']] = r
        for _dk in _dup_keys:
            cells.pop(_dk, None)
        if _dup_cells:
            print(f'[v10 stats] {len(_dup_cells)} probe record(s) share a (cell, seed) identity — the whole ambiguous cell is dropped from the pooling: ' + ', '.join((f'{k}' for k, _c, _s in _dup_cells[:4])))
        _new_bad = _bad_cells[_n_bad_pre:]
        if _new_bad:
            print(f'[v10 stats] {len(_new_bad)} record(s) in {sp} lack `probe_params` and are dropped from every mean/paired statistic: ' + ', '.join((f'{k}({why})' for k, why in _new_bad[:4])))
        def _record_audit(by_seed):
            """The corruption audit is arm/model/seed independent; keep it only
            when every seed record of the cell agrees, otherwise mark unknown."""
            _vals = [r.get('corruption_audit') for r in by_seed.values()]
            if not _vals or any(not isinstance(x, dict) for x in _vals):
                return None
            _first = _vals[0]
            try:
                _same = all(json.dumps(x, sort_keys=True) == json.dumps(_first, sort_keys=True)
                            for x in _vals[1:])
            except (TypeError, ValueError):
                return None
            return dict(_first) if _same else None

        def _cell_pooled(by_seed):
            """Token-pooled PPL + median/IQR over the cell's seeds.

            The per-seed PPL is already exp(NLL/tokens) for that seed; the
            cross-seed population estimate is exp(sum NLL / sum tokens), not
            the arithmetic mean of PPLs.  Records store the per-sequence NLL
            and token counts, so the pooled value is computable without a
            re-run; return None when any record lacks them.
            """
            _nll, _tok = [], []
            for _r in by_seed.values():
                _ns, _ts = _r.get('nll_sum'), _r.get('n_tok')
                if not isinstance(_ns, (list, tuple)) or not isinstance(_ts, (list, tuple)) or len(_ns) != len(_ts) or not _ns:
                    return None
                try:
                    _nll.extend(float(x) for x in _ns)
                    _tok.extend(int(t) for t in _ts)
                except (TypeError, ValueError, OverflowError):
                    return None
            if not _nll or sum(_tok) <= 0 or any(t <= 0 for t in _tok):
                return None
            _ce = sum(_nll) / sum(_tok)
            try:
                return float(np.exp(_ce))
            except OverflowError:
                return None

        probe_cells = []
        for (v, arm, Ln, rho, mode, _pp, _cd, _recipe, _vfp), by_seed in sorted(cells.items(), key=lambda kv: kv[0]):
            means = {s: r['ppl_mean'] for s, r in by_seed.items()}
            if not means:
                raise RuntimeError(f'v10 probe aggregation: empty seed set for {(v, arm, Ln, rho, mode)}')
            _vals = np.asarray(list(means.values()), dtype=float)
            _pooled = _cell_pooled(by_seed)
            _med = float(np.median(_vals)) if _vals.size else float('nan')
            _q1 = float(np.percentile(_vals, 25)) if _vals.size >= 2 else _med
            _q3 = float(np.percentile(_vals, 75)) if _vals.size >= 2 else _med
            probe_cells.append({'variant': v, 'arm': arm, 'eval_len': Ln, 'rho': rho, 'mode': mode, 'seeds': sorted(means), 'ppl_by_seed': means, 'probe_params': next(iter(by_seed.values())).get('probe_params'), '_code': _cd, 'recipe': _recipe, 'val_fp': _val_fp_value(_vfp), 'mean': float(np.mean(list(means.values()))), 'ppl_pooled': _pooled, 'ppl_median': _med, 'ppl_iqr': [_q1, _q3], 'n': len(means), 'corruption_audit': _record_audit(by_seed)})
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
            _va = L._content_fp_text(a.get('val_fp'))
            _vb = L._content_fp_text(b.get('val_fp'))
            if _va is None or _vb is None:
                print(f'[v10 stats] L{Ln} r{rho} {mode} {tag}: at least one arm has no valid validation-slice fingerprint (val_fp {a.get("val_fp")!r} vs {b.get("val_fp")!r}) — the slices cannot be shown to match, so the contrast is omitted')
                return False
            if _va != _vb:
                print(f'[v10 stats] L{Ln} r{rho} {mode} {tag}: the two arms were scored on DIFFERENT validation slices (val_fp {_va[:12]}… vs {_vb[:12]}…) — pairing them would difference two different measurements, so the contrast is omitted')
                return False
            if _fp_norm(a.get('probe_params')) != _fp_norm(b.get('probe_params')):
                print(f'[v10 stats] L{Ln} r{rho} {mode} {tag}: the two arms were probed under DIFFERENT probe_params — pairing them would difference two different measurements, so the contrast is omitted')
                return False
            if a.get('_code') != b.get('_code') or _recipe_norm(a.get('recipe')) != _recipe_norm(b.get('recipe')):
                print(f"[v10 stats] L{Ln} r{rho} {mode} {tag}: the two arms carry DIFFERENT code/recipe stamps (learned {a.get('_code')}/{a.get('recipe')} vs {b.get('_code')}/{b.get('recipe')}) — pairing them would mix two measurement semantics, so the contrast is omitted; re-run the stale arm's phase to refresh it")
                return False
            _cur_code = L.code_stamp()
            if a.get('_code') != _cur_code or b.get('_code') != _cur_code:
                print(f"[v10 stats] L{Ln} r{rho} {mode} {tag}: probe stamp {a.get('_code')}/{b.get('_code')} is not the current code semantics {_cur_code!r} — both arms are stale, so the contrast is omitted (re-run P3MP to refresh)")
                return False
            return True

        def _add_contrast(name, a, b):
            common = sorted(set(a['ppl_by_seed']) & set(b['ppl_by_seed']))
            if len(common) >= 2:
                dl = [b['ppl_by_seed'][s] - a['ppl_by_seed'][s] for s in common]
                contrasts[name] = V.exact_sign_permutation(dl)
                contrasts[name]['seeds'] = common
        # ---- uniform-mode contrasts (names unchanged for report compat) ----
        _declared_contrasts = []
        for Ln in PROBE['eval_lens']:
            for rho in PROBE['rhos']:
                learned = cell_mean('csa_fixed_rope', 'learned', Ln, rho, 'uniform')
                for arm, tag in (('dense', 'full_rope(dense)'), ('randidx', 'csa+randidx'), ('allblocks', 'csa+allblocks')):
                    _cname = f'L{Ln} r{rho}: {tag} - csa+learned'
                    _declared_contrasts.append(_cname)
                    other = cell_mean('full_rope', 'dense', Ln, rho, 'uniform') if arm == 'dense' else cell_mean('csa_fixed_rope', arm, Ln, rho, 'uniform')
                    if not learned or not other:
                        continue
                    if not _pairable(learned, other, Ln, rho, tag):
                        continue
                    _add_contrast(_cname, learned, other)
        # ---- from-scratch selector controls ----
        for Ln in PROBE['eval_lens']:
            for rho in PROBE['rhos']:
                learned = cell_mean('csa_fixed_rope', 'learned', Ln, rho, 'uniform')
                for tag, ov, oarm in (('csa_randtrain', 'csa_fixed_rope_randidx', 'learned'), ('csa_alltrain', 'csa_fixed_rope_allblocks', 'learned')):
                    _cname = f'L{Ln} r{rho}: {tag} - csa+learned'
                    _declared_contrasts.append(_cname)
                    other = cell_mean(ov, oarm, Ln, rho, 'uniform')
                    if not learned or not other or not _pairable(learned, other, Ln, rho, tag):
                        continue
                    _add_contrast(_cname, learned, other)
                # intervention-vs-retrain gap (NOT a causal
                # mismatch decomposition): test-time intervention minus the
                # trained-with-that-selection control at the same (L, rho).
                # For losses the positive direction means the intervention
                # scores WORSE than retraining that selection — a gap that MAY
                # itself contain train/test-mismatch effects (plus solver
                # adaptation and parameter differences), so it must be read as
                # a raw difference, never as 'the part that cannot be
                # attributed to mismatch'.
                for tag, tt_arm, sv in (('retrain-gap(random): csa+randidx - csa_randtrain', 'randidx', 'csa_fixed_rope_randidx'), ('retrain-gap(allblocks): csa+allblocks - csa_alltrain', 'allblocks', 'csa_fixed_rope_allblocks')):
                    _cname = f'L{Ln} r{rho}: {tag}'
                    _declared_contrasts.append(_cname)
                    tt = cell_mean('csa_fixed_rope', tt_arm, Ln, rho, 'uniform')
                    sc = cell_mean(sv, 'learned', Ln, rho, 'uniform')
                    if not tt or not sc or not _pairable(tt, sc, Ln, rho, tag):
                        continue
                    _add_contrast(_cname, sc, tt)
                # system-level dense contrast: full_rope minus the all-blocks
                # CSA twin.  This is NOT architecture-aligned: the CSA twin
                # keeps compressed KV, pooling, cosine q/k + scale=1, sink and
                # the sliding-window branch, while full_rope has per-token K/V,
                # RMSNorm-only q/k, scale=1/sqrt(d_head), no sink and a wider
                # MLP.  It may be reported only as a system-level difference.
                dense = cell_mean('full_rope', 'dense', Ln, rho, 'uniform')
                allt = cell_mean('csa_fixed_rope_allblocks', 'learned', Ln, rho, 'uniform')
                _cname = f'L{Ln} r{rho}: system-level: full_rope - csa_alltrain'
                _declared_contrasts.append(_cname)
                if dense and allt and _pairable(dense, allt, Ln, rho, 'system-level'):
                    _add_contrast(_cname, allt, dense)
        # ---- delta-NLL on the corruption modes ----
        delta_nll = {}
        _dn_id = {}
        _dn_ambiguous = set()

        def _nll_tok(rec):
            # nll_sum and n_tok are PER-ROW arrays: they must align, and token
            # counts must be exact non-negative integers.  int(1.9) -> 1 and
            # int(True) -> 1 would silently change the denominator.
            ns_raw, nt_raw = rec.get('nll_sum'), rec.get('n_tok')
            if (not isinstance(ns_raw, (list, tuple))
                    or not isinstance(nt_raw, (list, tuple))
                    or len(ns_raw) != len(nt_raw) or not ns_raw):
                return float('nan')
            try:
                ns = []
                for x in ns_raw:
                    if isinstance(x, (bool, np.bool_, np.datetime64, np.timedelta64,
                                      complex, np.complexfloating)):
                        return float('nan')
                    ns.append(float(x))
                nt = []
                for x in nt_raw:
                    c = L._as_seed_int(x)
                    if c is None or c < 0:
                        return float('nan')
                    nt.append(c)
            except (TypeError, ValueError, OverflowError):
                return float('nan')
            tot = sum(nt)
            if tot <= 0:
                return float('nan')
            v = math.fsum(ns) / tot
            return v if math.isfinite(v) else float('nan')

        for (v, arm, Ln, rho, mode, _pp, _cd, _recipe, _vfp), by_seed in sorted(cells.items()):
            if _cd != L.code_stamp():
                # Stale probe semantics cannot enter the delta-NLL family either
                # (the uniform contrasts already enforce this in _pairable).
                continue
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
                entry = {'variant': v, 'arm': arm, 'eval_len': Ln, 'rho': rho, 'mode': mode, 'n': len(_dl), 'corruption_audit': _record_audit(by_seed), 'mean': float(np.mean(_dl)), 'std': float(np.std(_dl, ddof=1)) if len(_dl) > 1 else 0.0, 'per_seed': means, 'clean_nll_by_seed': {s: _nll_tok(r) for s, r in clean_grp.items() if math.isfinite(_nll_tok(r))}, 'note': 'delta = NLL_corrupted - NLL_clean, nats per target token'}
                _dnk = f'{v}::{arm}::L{Ln}::{mode}::r{rho}'
                _dnident = (v, arm, Ln, rho, mode, _pp, _cd, _recipe, _vfp)
                if _dnk in _dn_ambiguous:
                    continue
                if _dnk in delta_nll:
                    if _dn_id.get(_dnk) != _dnident:
                        # Two distinct parameterisations/code/recipe stamps
                        # share one display key; keeping either would make the
                        # contrast depend on dict order.
                        print(f'[v10 stats] ΔNLL cell {_dnk} has several probe parameterisations — the cell is EXCLUDED rather than overwritten')
                        delta_nll.pop(_dnk, None)
                        _dn_ambiguous.add(_dnk)
                    continue
                delta_nll[_dnk] = entry
                _dn_id[_dnk] = _dnident
        # paired delta-NLL contrasts between arms at the same (L, mode, rho)
        for Ln in PROBE['eval_lens']:
            _modes = (PROBE.get('modes') or {}).keys() or ['uniform']
            for mode in [m for m in _modes if m != 'uniform'] + ['uniform']:
                _rhos = ((PROBE.get('modes') or {}).get(mode) or ([r for r in PROBE['rhos'] if r > 0] if mode == 'uniform' else []))
                for rho in [r for r in _rhos if r > 0]:
                    _lk = f'csa_fixed_rope::learned::L{Ln}::{mode}::r{rho}'
                    learned = delta_nll.get(_lk)
                    for arm, tag in (('dense', 'full_rope::dense'), ('randidx', 'csa_fixed_rope::randidx'), ('allblocks', 'csa_fixed_rope::allblocks')):
                        _cname = f'ΔNLL L{Ln} {mode} r{rho}: {arm} - csa+learned'
                        _declared_contrasts.append(_cname)
                        _ok = f'{tag}::L{Ln}::{mode}::r{rho}'
                        other = delta_nll.get(_ok)
                        if not learned or not other:
                            continue
                        # Both sides must describe the SAME measurement
                        # geometry/recipe/code; the display key alone omits
                        # those identities.
                        if _dn_id.get(_lk, (None,) * 9)[2:] != _dn_id.get(_ok, (None,) * 9)[2:]:
                            print(f'[v10 stats] ΔNLL L{Ln} {mode} r{rho}: {arm} and learned were measured under different probe_params/code/recipe — contrast omitted')
                            continue
                        common = sorted(set(learned['per_seed']) & set(other['per_seed']))
                        if len(common) >= 2:
                            dl = [other['per_seed'][s] - learned['per_seed'][s] for s in common]
                            res = V.exact_sign_permutation(dl)
                            res['seeds'] = common
                            contrasts[_cname] = res
                    # Same delta-NLL contrast for the from-scratch selector
                    # controls.  Without this row P3MTC's controls are only
                    # visible in the raw-PPL table, not under the corruption
                    # increment that the mechanism claim is about.
                    for tag, sv in (('csa_randtrain', 'csa_fixed_rope_randidx'), ('csa_alltrain', 'csa_fixed_rope_allblocks')):
                        _cname = f'ΔNLL L{Ln} {mode} r{rho}: {tag} - csa+learned'
                        _declared_contrasts.append(_cname)
                        _ok = f'{sv}::learned::L{Ln}::{mode}::r{rho}'
                        other = delta_nll.get(_ok)
                        if not learned or not other:
                            continue
                        if _dn_id.get(_lk, (None,) * 9)[2:] != _dn_id.get(_ok, (None,) * 9)[2:]:
                            print(f'[v10 stats] ΔNLL L{Ln} {mode} r{rho}: {tag} and learned were measured under different probe_params/code/recipe — contrast omitted')
                            continue
                        common = sorted(set(learned['per_seed']) & set(other['per_seed']))
                        if len(common) >= 2:
                            dl = [other['per_seed'][s] - learned['per_seed'][s] for s in common]
                            res = V.exact_sign_permutation(dl)
                            res['seeds'] = common
                            contrasts[_cname] = res
        # ---- Holm correction over the whole contrast family ----
        holm = {}
        # Declared-but-unmeasurable contrasts keep a p=1 family slot; filtering
        # them out shrinks m and makes the surviving p_holm too small.
        _fam_names = list(dict.fromkeys(_declared_contrasts + list(contrasts)))
        _fam = [(k, (contrasts.get(k) or {}).get('p_exact_signflip') if k in contrasts else None)
                for k in _fam_names]
        if _fam:
            # `attach_holm` keeps every declared slot (omitted/unmeasurable
            # comparisons enter as p=1) and returns the family-level
            # resolution ceiling.  The manual `holm_adjust` path used here
            # before dropped that metadata and made the report unable to say
            # that the whole family was resolution-limited.
            _fam_slots = {k: (contrasts.get(k) if k in contrasts
                              else {'omitted': 'no measurable paired cell'})
                          for k in _fam_names}
            holm = L.attach_holm(_fam_slots, family='v10 distractor probe contrasts')
            for _k, _v in contrasts.items():
                if isinstance(_v, dict):
                    _v['p_holm'] = _fam_slots.get(_k, {}).get('p_holm')
            _rej = [k for k, v in _fam_slots.items() if v.get('p_holm') is not None and v['p_holm'] < 0.05]
            print(f'[v10 stats] Holm step-down over the {len(_fam_slots)}-contrast family (EXPLORATORY — not pre-registered): {len(_rej)} survive at alpha=0.05' + (f': {sorted(_rej)[:4]}' if _rej else ''))
        # ---- clean far-context benefit (mask / delete controls) ----
        # The reviewer's alternative explanation for a small corruption delta
        # is that the arm simply uses the far context very little.  The
        # benefit of the clean far context is exactly
        # NLL_corrupted - NLL_clean (positive = clean helped); it is the same
        # paired contrast as Delta-NLL, published under an interpretable name
        # and next to the clean-NLL level.  It is NOT the negative of Delta-NLL.
        far_benefit = {}
        for _k, _e in delta_nll.items():
            if _e.get('mode') not in ('mask', 'delete'):
                continue
            _per = {s: float(v) for s, v in (_e.get('per_seed') or {}).items()}
            if not _per:
                continue
            _res = V.exact_sign_permutation(list(_per.values()))
            if not (isinstance(_res.get('p_floor'), (int, float)) and math.isfinite(float(_res['p_floor']))):
                _n_res = int(_res.get('n') or 0)
                _res['p_floor'] = math.ldexp(2.0, -_n_res) if _n_res > 0 else float('nan')
            _res.update({'variant': _e['variant'], 'arm': _e['arm'],
                         'eval_len': _e['eval_len'], 'mode': _e['mode'],
                         'rho': _e['rho'], 'per_seed': _per,
                         'clean_nll_by_seed': _e.get('clean_nll_by_seed'),
                         'note': 'benefit = NLL_corrupted - NLL_clean; positive = clean far context helps'})
            far_benefit[_k] = _res
        probe = {'cells': probe_cells, 'contrasts': contrasts, 'delta_nll': delta_nll,
                 'far_benefit': far_benefit, 'holm_exploratory': holm}
        # ---- NIAH-natural (far-necessity task) ----
        _nsp = (PROBE.get('niah') or {}).get('summary')
        if _nsp and os.path.exists(_nsp):
            try:
                with open(_nsp, encoding='utf-8') as _jf:
                    nraw = json.load(_jf)
            except Exception as _e:
                print(f'[stats] WARNING: {_nsp} exists but cannot be parsed ({type(_e).__name__}: {_e}) — the NIAH-natural panel is skipped')
                nraw = {}
            if not isinstance(nraw, dict):
                print(f'[v10 stats] WARNING: {_nsp} is valid JSON but not an object ({type(nraw).__name__}) — the NIAH-natural panel is EXCLUDED')
                nraw = {}
            ncells = {}
            _dup_nkeys = set()
            _niah_bad = []
            for _k, r in nraw.items():
                if not isinstance(r, dict) or r.get('task') != 'niah_natural':
                    continue
                if r.get('synthesized'):
                    continue
                _why = _niah_record_problem(r)
                if _why is not None:
                    _niah_bad.append((_k, _why))
                    continue
                if r.get('_code') != L.code_stamp():
                    # Same fail-closed rule as the corruption contrasts: stale
                    # NIAH records cannot contribute p-values.
                    continue
                _seed = L._as_seed_int(r.get('seed'))
                _ln = L._as_seed_int(r.get('eval_len'))
                # acc/nll are measurements: a bool/string/NaN is not a result.
                # Preserve the declared cell with a NaN value so the Holm
                # family keeps its slot; a boolean is not an accuracy.
                _acc = L._finite_scalar(r.get('acc'))
                _nll = L._finite_scalar(r.get('nll'))
                _r = dict(r, seed=_seed, eval_len=_ln,
                          condition=r['condition'],
                          acc=(_acc if _acc is not None else float('nan')),
                          nll=(_nll if _nll is not None else float('nan')))
                nkey = (_r['variant'], _r['arm'], _ln, _r['condition'], json.dumps(_fp_norm(_r.get('probe_params')), sort_keys=True), str(_r.get('_code')), str(_r.get('recipe')), _val_fp_key(_r.get('val_fp')))
                grp = ncells.setdefault(nkey, {})
                if _seed in grp:
                    # Two measurements for one (cell, seed) are ambiguous; the
                    # JSON dict order must not decide which one is reported.
                    _dup_nkeys.add(nkey)
                    continue
                grp[_seed] = _r
            for _dk in _dup_nkeys:
                ncells.pop(_dk, None)
            if _dup_nkeys:
                print(f'[v10 stats] {len(_dup_nkeys)} NIAH-natural cell(s) have duplicate (cell, seed) records — the whole ambiguous cell is dropped from the pooling')
            if _niah_bad:
                print(f'[v10 stats] {len(_niah_bad)} NIAH-natural record(s) are malformed and EXCLUDED before grouping: '
                      + ', '.join(f'{_bk}({_bw})' for _bk, _bw in _niah_bad[:4]))
            niah_cells = []
            for (v, arm, Ln, cond, _pp, _cd, _recipe, _vfp), by_seed in sorted(ncells.items(), key=lambda kv: kv[0]):
                accs = {s: r['acc'] for s, r in by_seed.items()}
                nlls = {s: r['nll'] for s, r in by_seed.items()}
                _first = next(iter(by_seed.values()))
                niah_cells.append({'variant': v, 'arm': arm, 'eval_len': Ln, 'condition': cond, 'seeds': sorted(accs), 'acc_by_seed': accs, 'nll_by_seed': nlls, 'acc_mean': float(np.mean(list(accs.values()))), 'nll_mean': float(np.mean(list(nlls.values()))), 'n': len(accs), 'n_pairs': _first.get('n_pairs'), 'n_queries': _first.get('n_queries'), 'chance_acc': _first.get('chance_acc'), 'n_value_tokens': _first.get('n_value_tokens'), 'acc_by_distance_bucket': _first.get('acc_by_distance_bucket'), 'distance_gradient': _first.get('distance_gradient'), 'dist_mean': _first.get('dist_mean'), '_pp': _pp, '_code': _cd, '_recipe': _recipe, 'val_fp': _val_fp_value(_vfp)})
            niah_contrasts = {}
            _nidx = {}
            for c in niah_cells:
                _nidx.setdefault((c['variant'], c['arm'], c['eval_len'], c['condition']), []).append(c)

            def _ncell(v, arm, Ln, cond):
                hits = _nidx.get((v, arm, Ln, cond), [])
                return hits[0] if len(hits) == 1 else {}
            _niah_declared = []
            for Ln in PROBE['eval_lens']:
                for cond in (PROBE.get('niah') or {}).get('conditions', ['clean', 'distract']):
                    learned = _ncell('csa_fixed_rope', 'learned', Ln, cond)
                    for arm, tag in (('dense', 'full_rope'), ('randidx', 'csa_fixed_rope'), ('allblocks', 'csa_fixed_rope')):
                        _cname = f'NIAH-nat L{Ln} {cond}: {arm} - learned (answer NLL)'
                        _niah_declared.append(_cname)
                        other = _ncell('full_rope', 'dense', Ln, cond) if arm == 'dense' else _ncell(tag, arm, Ln, cond)
                        if not learned or not other:
                            continue
                        _lv = L._content_fp_text(learned.get('val_fp'))
                        _ov = L._content_fp_text(other.get('val_fp'))
                        if _lv is None or _ov is None or _lv != _ov:
                            print(f'[v10 stats] NIAH-nat L{Ln} {cond}: {arm} and learned were not scored on a verifiably identical validation slice — contrast omitted')
                            continue
                        if (learned.get('_pp'), learned.get('_code'), learned.get('_recipe')) != (other.get('_pp'), other.get('_code'), other.get('_recipe')):
                            print(f'[v10 stats] NIAH-nat L{Ln} {cond}: {arm} and learned were measured under different probe_params/code/recipe — contrast omitted')
                            continue
                        common = sorted(set(learned['nll_by_seed']) & set(other['nll_by_seed']))
                        if len(common) >= 2:
                            _deltas = [other['nll_by_seed'][s] - learned['nll_by_seed'][s] for s in common]
                            _keep = [s for s, _d in zip(common, _deltas) if L._finite_scalar(_d) is not None]
                            res = V.exact_sign_permutation(_deltas)
                            res['seeds'] = _keep
                            niah_contrasts[_cname] = res
                    for tag, ov in (('csa_randtrain', 'csa_fixed_rope_randidx'), ('csa_alltrain', 'csa_fixed_rope_allblocks')):
                        _cname = f'NIAH-nat L{Ln} {cond}: {tag} - learned (answer NLL)'
                        _niah_declared.append(_cname)
                        other = _ncell(ov, 'learned', Ln, cond)
                        if not learned or not other:
                            continue
                        _lv = L._content_fp_text(learned.get('val_fp'))
                        _ov = L._content_fp_text(other.get('val_fp'))
                        if _lv is None or _ov is None or _lv != _ov:
                            print(f'[v10 stats] NIAH-nat L{Ln} {cond}: {tag} and learned were not scored on a verifiably identical validation slice — contrast omitted')
                            continue
                        if (learned.get('_pp'), learned.get('_code'), learned.get('_recipe')) != (other.get('_pp'), other.get('_code'), other.get('_recipe')):
                            print(f'[v10 stats] NIAH-nat L{Ln} {cond}: {tag} and learned were measured under different probe_params/code/recipe — contrast omitted')
                            continue
                        common = sorted(set(learned['nll_by_seed']) & set(other['nll_by_seed']))
                        if len(common) >= 2:
                            _deltas = [other['nll_by_seed'][s] - learned['nll_by_seed'][s] for s in common]
                            _keep = [s for s, _d in zip(common, _deltas) if L._finite_scalar(_d) is not None]
                            res = V.exact_sign_permutation(_deltas)
                            res['seeds'] = _keep
                            niah_contrasts[_cname] = res
            probe['niah_natural'] = {'cells': niah_cells, 'contrasts': niah_contrasts}
            # Review 10, issue 13: NIAH task-validity gate.  A random-token
            # key->value task pasted into natural text is not automatically a
            # valid far-necessity probe for a small model that was never
            # trained on the format: chance-level accuracy can mean "task
            # format unsupported", not "attention failed".  The gate therefore
            # requires ALL of:
            #   (1) clean accuracy significantly above the published chance
            #       level (exact binomial lower-tail p < alpha);
            #   (2) the no-far-info control (couples destroyed) at chance —
            #       above-chance accuracy there would mean the answer leaks
            #       from the local target context;
            #   (3) a POSITIVE CONTROL solver that demonstrably solves the
            #       task.  No positive control is part of the probe config, so
            #       this condition is false for every current panel and NIAH
            #       cannot support a "long-distance utilization" mechanism
            #       claim until one is added and recorded.
            _alpha_n = 0.05
            _pc = (PROBE.get('niah') or {}).get('positive_control')
            _pc_ok = bool(isinstance(_pc, dict) and L._finite_scalar(_pc.get('acc')) is not None
                          and L._finite_scalar(_pc.get('chance_acc')) is not None
                          and L._finite_scalar(_pc.get('acc')) > L._finite_scalar(_pc.get('chance_acc')))
            _nvl = {}
            for _bv, _barm, _bL in sorted({(c['variant'], c['arm'], c['eval_len']) for c in niah_cells}):
                def _ncellv(_cond):
                    hits = [c for c in niah_cells
                            if c['variant'] == _bv and c['arm'] == _barm
                            and c['eval_len'] == _bL and c['condition'] == _cond]
                    return hits[0] if len(hits) == 1 else None
                _cl = _ncellv('clean')
                if _cl is None:
                    continue
                _chance = L._finite_scalar(_cl.get('chance_acc'))
                _nq = L._as_seed_int(_cl.get('n_queries'))
                _acc = _cl.get('acc_mean')
                _p_bin = None
                _correct = None
                if (_chance is not None and _nq is not None and _nq > 0
                        and L._finite_scalar(_acc) is not None):
                    _correct = int(round(float(_acc) * _nq * len(_cl.get('acc_by_seed') or {})))
                    _p_bin = _binom_sf_exact(_nq * len(_cl.get('acc_by_seed') or {}), _correct, _chance)
                _nof = _ncellv('nofar')
                _nofar_ok = None
                _nofar_p = None
                _nofar_acc = None
                if _nof is not None and _chance is not None and _nq is not None and _nq > 0:
                    _nofar_acc = _nof.get('acc_mean')
                    _nof_n = _nq * len(_nof.get('acc_by_seed') or {})
                    _nof_k = int(round(float(_nofar_acc) * _nof_n)) if L._finite_scalar(_nofar_acc) is not None else 0
                    _nofar_p = _binom_sf_exact(_nof_n, _nof_k, _chance)
                    _nofar_ok = not (L._finite_scalar(_nofar_acc) is not None
                                     and _nofar_p is not None and _nofar_p < _alpha_n
                                     and float(_nofar_acc) > float(_chance))
                _above = bool(_p_bin is not None and _p_bin < _alpha_n
                              and float(_acc) > float(_chance))
                _valid = bool(_above and (_nofar_ok is not False) and _pc_ok)
                _nvl[f'{_bv}::{_barm}::L{_bL}'] = {
                    'chance_acc': _chance, 'clean_acc': _acc,
                    'n_queries': _nq,
                    'binomial_p_above_chance': _p_bin,
                    'above_chance': _above,
                    'nofar_available': _nof is not None,
                    'nofar_acc': _nofar_acc,
                    'nofar_p_above_chance': _nofar_p,
                    'nofar_consistent_with_chance': _nofar_ok,
                    'positive_control_recorded': _pc_ok,
                    'task_valid_for_mechanism_claim': _valid,
                }
            probe['niah_task_validity'] = _nvl
            probe['niah_mechanism_usable'] = bool(_nvl) and all(
                v['task_valid_for_mechanism_claim'] for v in _nvl.values())
            probe['niah_validity_note'] = (
                'NIAH-natural cannot be used as mechanism evidence unless (1) clean accuracy is '
                'above the published chance level, (2) the no-far-info control is at chance, and '
                '(3) a positive-control solver is recorded. Current panels lack a positive control, '
                'so mechanism claims based on NIAH remain unsupported (descriptive only).')
            if not probe['niah_mechanism_usable']:
                print('[v10 stats] NIAH task-validity gate: mechanism claims NOT supported '
                      '(missing/!pass positive control, or chance/nofar checks failed) — '
                      'NIAH stays descriptive only')
            # NIAH contrasts are built AFTER the corruption family was
            # corrected.  Leaving them out of the Holm family would report
            # uncorrected p-values under a "Holm" heading, so re-run the
            # correction over the union and overwrite both.
            _comb = dict(contrasts)
            _comb.update(niah_contrasts)
            _comb_names = list(dict.fromkeys(_fam_names + _niah_declared + list(_comb)))
            if _comb_names:
                _comb_slots = {k: (_comb.get(k) if k in _comb
                                   else {'omitted': 'no measurable paired cell'})
                               for k in _comb_names}
                _h2 = L.attach_holm(_comb_slots,
                                    family='v10 corruption + NIAH contrast family')
                for _k, _v in _comb.items():
                    if isinstance(_v, dict):
                        _v['p_holm'] = _comb_slots.get(_k, {}).get('p_holm')
                holm = _h2
                probe['holm_exploratory'] = _h2
                probe['holm_family_note'] = ('Holm over the combined corruption + NIAH contrast families')
                print(f'[v10 stats] Holm re-attached over the combined {len(_comb_slots)}-contrast family (corruption + NIAH)')
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
                             'censored': bool(_v.get('censored') or (_v.get('mean_crossed') is False or (isinstance(_v.get('mean_crossed'), np.bool_) and bool(_v.get('mean_crossed')) == False))),
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
    # Cross-family multiplicity: the standalone v10 report displays probe,
    # NIAH and seq2k p-values side by side.  Holm is only within one family;
    # a reader quoting "it was significant in the probe and in NIAH" has
    # selected over the union.  Record BH-FDR + Bonferroni over every p-value
    # the report can display, so the report can constrain its wording.
    _gm_named = []
    for _k, _v in (probe.get('contrasts') or {}).items():
        _gm_named.append((f'probe|{_k}', (_v or {}).get('p_exact_signflip'), 'v10 probe'))
    for _k, _v in (((probe.get('niah_natural') or {}).get('contrasts')) or {}).items():
        _gm_named.append((f'niah|{_k}', (_v or {}).get('p_exact_signflip'), 'v10 NIAH'))
    for _k, _v in comparisons.items():
        _gm_named.append((f'seq2k|{_k}', (_v or {}).get('p_exact_signflip'), 'v10 seq2k'))
    # Review 12: the scale-endpoint and equal-token comparisons are DISPLAYED
    # with p-values in this report, so they belong to the global family too.
    # Omitting them made the printed BH/Bonferroni values anti-conservative
    # whenever those panels had data.
    for _name, _e in ((scale_conf.get('panels') or {}) if isinstance(scale_conf, dict) else {}).items():
        _ep = (_e or {}).get('endpoint') or {}
        _pv = _ep.get('p_exact_signflip')
        if _pv is not None:
            _gm_named.append((f'scale|{_name}: endpoint', _pv, 'v10 scale'))
    for _k, _v in ((scale_conf.get('equal_token_slices') or {}) if isinstance(scale_conf, dict) else {}).items():
        _pv = (_v or {}).get('p_exact_signflip')
        if _pv is not None:
            _gm_named.append((f'equal-token|{_k}', _pv, 'v10 equal-token'))
    _gm = L.global_multiplicity_audit(_gm_named) if _gm_named else {'m': 0, 'rows': [], 'n_raw_significant': 0, 'n_bh_rejected': 0, 'n_bonferroni_significant': 0, 'n_untestable': 0}
    out_d = {'probe': probe, 'crossover_panels': xo, 'crossover_fit': fit, 'scale_confounds': scale_conf, 'seq2k_bs1_panel': panel(seq2k), 'comparisons': comparisons, 'holm': L.attach_holm(comparisons, family='v10 seq2k comparisons'), 'global_multiplicity': _gm}
    out_d = L.stamped_analysis('v10_analysis', out_d, outdir=(os.path.dirname(out) or '.'), name=os.path.basename(out))
    print(f'[v10 stats] wrote {out}')
    for k, v in comparisons.items():
        if 'omitted' in v:
            print(f'  {k:38s} omitted — {v['omitted']}')
            continue
        _ptxt = ('' if (v.get('param_controlled') is True or (isinstance(v.get('param_controlled'), np.bool_) and bool(v.get('param_controlled')) == True)) else
                 (f'  params {100.0 * v['param_rel_diff']:+.1f}% [capacity-confounded: system-level]' if (v.get('param_controlled') is False or (isinstance(v.get('param_controlled'), np.bool_) and bool(v.get('param_controlled')) == False)) else '  params unverifiable [system-level]'))
        print(f'  {k:38s} Δ={v['mean']:+7.2f} ± {v.get('std', 0):5.2f}  p(sign-flip)={v['p_exact_signflip']:.3f}  n={v['n']}{_ptxt}')
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

def _panel_synth_seeds(outdir):
    """Seeds carrying `synthesized` records in one panel (review 12).

    The earlier "this panel contains no reconstruction" sentence was derived
    only from the two ADMITTED csa/full curves, so a synthesized record for
    another variant (or one without a ppl_history) was invisible.  Scan the
    raw summary instead; return None when it cannot be read (unknown, not
    "clean").
    """
    try:
        with open(os.path.join(outdir, 'summary.json'), encoding='utf-8') as _f:
            _d = json.load(_f)
    except Exception:
        return None
    if not isinstance(_d, dict):
        return None
    return sorted({str(_r.get('seed')) for _r in _d.values()
                   if isinstance(_r, dict) and _r.get('synthesized')})


def _one_record_per_seed(rows, label=''):
    """Canonicalise seeds, drop synthesized reconstructions, and exclude
    every ambiguous duplicate seed so a by-length column means each seed
    exactly once."""
    by_seed = {}
    dup = set()
    for r in rows:
        if not isinstance(r, dict) or r.get('synthesized'):
            continue
        s = L._as_seed_int(r.get('seed'))
        if s is None:
            continue
        if s in by_seed:
            dup.add(s)
        else:
            by_seed[s] = dict(r, seed=s)
    if dup:
        print(f'[v10 report] {label}: seed(s) {sorted(dup)} hold more than one by-length record — '
              'the duplicate seed(s) are EXCLUDED rather than averaged into the column')
        for s in dup:
            by_seed.pop(s, None)
    return list(by_seed.values())


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
    try:
        with open(stats_p, encoding='utf-8') as _jf:
            st = json.load(_jf)
    except Exception as _pe:
        raise RuntimeError(f'{stats_p} cannot be parsed ({type(_pe).__name__}: {_pe}); refusing to build a report over it') from _pe
    if not isinstance(st, dict):
        raise RuntimeError(f'{stats_p} is valid JSON but not an object ({type(st).__name__}); refusing to build a report over it')
    probe = st['probe']
    xo = st['crossover_panels']
    fit = st['crossover_fit']
    seq2k = st['seq2k_bs1_panel']
    comp = st['comparisons']
    mech_sum = {}
    mp = os.path.join(P3MT_PAYLOAD['outdir'], 'summary.json')
    if os.path.exists(mp):
        try:
            with open(mp, encoding='utf-8') as _jf:
                mech_sum = json.load(_jf)
        except Exception as _pe:
            print(f'[report] WARNING: {mp} cannot be parsed ({type(_pe).__name__}: {_pe}) — the mechanism records are omitted')
            mech_sum = {}
        if not isinstance(mech_sum, dict):
            print(f'[report] WARNING: {mp} is valid JSON but not an object ({type(mech_sum).__name__}) — the mechanism records are omitted')
            mech_sum = {}
    lines = []
    A = lines.append

    def _fnum(x, spec='.3f'):
        """Render a finite number, or an em dash for None/NaN/inf."""
        try:
            xf = float(x)
        except (TypeError, ValueError, OverflowError):
            return '—'
        return format(xf, spec) if math.isfinite(xf) else '—'

    A('# CSA / HCA 受控机制研究 — v10 补实验报告（外推机制 + 反转点标度 + topk 补种子）')
    A('> **证据等级：探索性（exploratory）**。本面板的比较/种子是观察 validation 结果后选择的；Holm 校正只限制过度宣称，不等于预注册确认。')
    A(L.evidence_status_banner())
    A('')
    A('')
    A(f'> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}')
    A('> 参考论文：arXiv:2606.19348。**范围声明**：本报告是小规模 WikiText-103 上的受控研究，不是 DeepSeek-V4 的全量复现，也不支持对其规模、语料或生产系统做外推。')
    A('> 说明：本报告全部数字由 `v10_supp.py report` 从 `results_*/`、`analysis_v10/` 的落盘产物计算得到，无手填数值。')
    A('')

    A('---')
    A('')
    A('## P3M 长度外推的系统级对比（远距干扰注入；非机制归因）')
    A('')
    A('**背景**：v7 P1L 的外推对比只是附带观察（n=3 时精确符号翻转 p 值下限 0.250）。P1L 权重未持久化，P3MT 先按 v7 P1L 原配方（seq 512、AdamW lr 3e-4）重训两臂并保存权重——同时作为该外推对比的独立复现。')
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
            _rows_by_v[v] = _one_record_per_seed(_rv, label=v)
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
        A(f'**P3MT/P3MTC 干净上下文复现对照**（train@512 → 同权重 eval；P3MT 两臂与 P3MTC 从头训练选择器对照并列；PPL = 跨 seed 的 token-pooled 估计，记录缺 NLL 时退回算术平均并标 `*`；ratio = pooled PPL@L / pooled PPL@512；n={(min((len(s) for s in n_by_v.values())) if n_by_v else 0)}）。**注意**：这是干净上下文读数，不是干扰探针；`csa_fixed_rope_allblocks` 仍保留 CSA 压缩结构，不是 dense 对照。')
        A('')
        A('| variant | PPL@512 | PPL@2048 | PPL@4096 | ratio@4096 |')
        A('|---|---|---|---|---|')
        for v in list(P3MT_PAYLOAD['variants']) + list(SCRATCH_CONTROLS):
            rows = _rows_by_v.get(v, [])
            if not rows:
                continue

            def _pool_nll(cells):
                """Token-pooled PPL over seed cells (review 12, issue 12)."""
                _nll, _tok = [], []
                for c in cells:
                    _n = L._finite_scalar(c.get('nll_sum'))
                    _t = L._as_seed_int(c.get('n_tokens') if c.get('n_tokens') is not None else c.get('n_tok'))
                    if _n is None or _t is None or _t <= 0 or _n < 0.0:
                        return None
                    _nll.append(_n)
                    _tok.append(_t)
                if not _nll or sum(_tok) <= 0:
                    return None
                try:
                    return float(np.exp(sum(_nll) / sum(_tok)))
                except OverflowError:
                    return None

            def _at(Ln, rows=rows, _v=v):
                cells = L.by_len_cells(rows, Ln)
                ok = [c for c in cells if isinstance(c.get('truncated'), (bool, np.bool_)) and not bool(c.get('truncated'))]
                if (ok if ok else cells) and len(ok if ok else cells) < len(rows):
                    print(f'[v10 report] {_v} L{Ln}: the column value covers {len(ok if ok else cells)}/{len(rows)} seeds — the rest hold no usable cell at this length (error/truncated cell), so the column is a sub-seed value')
                use = ok if ok else cells
                if not use:
                    return (float('nan'), False, None)
                _p = _pool_nll(use) if ok else None
                _v_ = _p if _p is not None else float(np.mean([c['ppl'] for c in use]))
                # A value over a strict subset is not a complete column: mark
                # it partial so the rendered cell carries `~` too.
                return (_v_, (len(use) != len(rows)) if ok else True, _p)

            def _paired_ratio(rows=rows, _v=v):
                m512, m4096 = ({}, {})
                c512, c4096 = ({}, {})
                for r in rows:
                    bl = r.get('by_len')
                    if not isinstance(bl, dict):
                        continue
                    for Ln, mp, cp in ((512, m512, c512), (4096, m4096, c4096)):
                        # by_len_cells refuses the ambiguous int/str twin of
                        # one length; a single ambiguous record cannot supply
                        # a defensible paired cell.
                        try:
                            _cells = L.by_len_cells([r], Ln)
                        except ValueError:
                            continue
                        c = _cells[0] if _cells else None
                        if isinstance(c, dict) and isinstance(c.get('truncated'), (bool, np.bool_)) and not bool(c.get('truncated')):
                            mp[r.get('seed')] = float(c['ppl'])
                            cp[r.get('seed')] = c
                both = sorted(set(m512) & set(m4096))
                if both and len(both) < len(rows):
                    print(f'[v10 report] {_v} ratio@4096: computed over {len(both)}/{len(rows)} seeds — the rest lack a usable cell at 512 or 4096')
                if not both:
                    return None, None
                # Prefer the ratio of token-pooled PPLs; mixing an arithmetic
                # seed mean into a ratio of exponentiated PPLs is exactly the
                # aggregation error this review asks to remove.
                _p512 = _pool_nll([c512[s] for s in both])
                _p4096 = _pool_nll([c4096[s] for s in both])
                if _p512 is not None and _p4096 is not None:
                    return (float(_p4096 / _p512), 'pooled')
                return (float(np.mean([m4096[s] / m512[s] for s in both])), 'per-seed-mean')

            def _fmt_cell(Ln):
                val, tr, pooled = _at(Ln)
                _mark = '' if pooled is not None else ('*' if np.isfinite(val) else '')
                return (f'{val:.1f}~' if tr else f'{val:.1f}{_mark}')
            p512, _512tr, _p512p = _at(512)
            p4096, _4096tr, _p4096p = _at(4096)
            if _512tr or _4096tr:
                _ratio = '—（含截断 cell，不可比）'
            else:
                _pr, _pr_basis = _paired_ratio()
                _ratio = f'×{_pr:.2f}' if _pr is not None and np.isfinite(_pr) else '—'
            A(f'| `{v}` | {_fmt_cell(512)} | {_fmt_cell(2048)} | {_fmt_cell(4096)} | {_ratio} |')
        A('')
        A('> 后缀 `~` 表示该列**不是完整的非截断测量**：它可能只含位置受限（abs-PE）的截断 cell（在比列名更短的 span 上评出的分数），也可能只有部分 seed 在该长度有可用 cell；都**不是**完整的长上下文测量值。任何 `~` 行不可用于「外推是否稳健」的结论；`*` 表示该格记录缺少 `nll_sum`/`n_tokens`，退回算术平均而非 pooled。该表只是 512 训练长度下的**外推探针**，不是长上下文训练。')
        A('')
        if _stale_vars:
            A('> **⚠ 陈旧记录已排除**：' + '、'.join(f'`{x}`' for x in sorted(_stale_vars)) +
              f' 没有任何按长度记录携带当前代码语义戳 `{L.code_stamp()}`；按「旧代码/旧配方/旧数据指纹不得进入主要结论」的规则，'
              '本表不引用这些数值，也不据此计算 ratio 或外推结论。重跑对应训练阶段后再纳入。')
            A('')
    A(f'**P3MP 干扰注入设计**：eval 长度 2048/4096；腐坏只作用于**远距上下文**（目标区之前的输入位置，即除最后 {PROBE['target']} 个干净目标 token 及其紧邻前置位置以外的位置），同一 (序列, ρ, mode) 的腐坏对所有臂逐字节相同（按 (eval_len, 序列号, ρ, mode) 播种，与臂/模型种子无关），臂间严格配对；只在干净目标区计 PPL/NLL。测试时四臂（dense 基线 + 在已训练 CSA 权重上替换选择方式的三臂）加两个 P3MTC 从头训练对照：')
    A('')
    A('- `full_rope`（dense，必须直面远距噪声）')
    A('- `csa_fixed_rope` 学习选择（原机制）')
    A('- `csa_fixed_rope` + 随机 indexer（选择被打乱，v4 消融开关，eval 时注入）')
    A('- `csa_fixed_rope` + topk=全块（看得见一切，仍是带压缩的 CSA，不是 dense）')
    A('- `csa_fixed_rope_randidx` / `csa_fixed_rope_allblocks`（P3MTC 从头训练的选择器对照，只跑其自身 `learned` 臂）')
    A('')
    A('腐坏模式共 7 种：`uniform` / `natural` / `span` / `doc` / `topical` / `mask` / `delete`；`doc` 是整段真实文档切片替换，最接近「自然语言干扰」，`span`/`natural`/`topical` 只覆盖特定替换几何。')
    A('')
    A('「稀疏掩码滤除远距噪声」假设预测：**学习选择臂的 ρ 曲线平坦，其余对照随 ρ 陡峭退化**；但该预测本身不是证据——只有配对 ΔNLL 与干净远端收益同时支持时才可作系统级读数。')
    A('')
    cells = probe.get('cells', [])
    if cells:
        lens = sorted({c['eval_len'] for c in cells})
        rhos = PROBE['rhos']
        arm_order = [('full_rope', 'dense'), ('csa_fixed_rope', 'learned'), ('csa_fixed_rope', 'randidx'), ('csa_fixed_rope', 'allblocks'), ('csa_fixed_rope_randidx', 'learned'), ('csa_fixed_rope_allblocks', 'learned')]
        for Ln in lens:
            _n_ln = max([c.get('n', 0) for c in cells if c['eval_len'] == Ln] or [0])
            A(f'**目标区 PPL（eval_len={Ln}，至多 {_n_ln or "?"} seeds；跨 seed token-pooled，uniform 腐坏模式）**：')
            A('')
            A('| 臂 | ' + ' | '.join((f'ρ={r:g}' for r in rhos)) + ' |')
            A('|---|' + '---|' * len(rhos))
            for v, arm in arm_order:
                row = []
                for rho in rhos:
                    _cc = [c for c in cells if (c['variant'], c['arm'], c['eval_len'], c['rho'], c.get('mode', 'uniform')) == (v, arm, Ln, rho, 'uniform')]
                    _val = None
                    if len(_cc) == 1:
                        _c0 = _cc[0]
                        _val = _c0.get('ppl_pooled')
                        if _val is None:
                            _val = _c0.get('mean')
                    row.append(f'{_val:.2f}' if _val is not None else '—')
                    if len(_cc) > 1:
                        print(f'[report] {v}+{arm} L={Ln} rho={rho}: {len(_cc)} parameterisations of the cell exist — the table omits the cell (the paired test refuses it too)')
                A(f'| `{v}`+{arm} | ' + ' | '.join(row) + ' |')
            A('')
    else:
        A('> **P3MP 探针 cell 缺失**：`results_v10_mech/distractor.json` 不存在或没有可用 cell。此时本报告不提供干扰鲁棒性读数；测试时干预与从头训练对照的差异**没有被测量**，不能把设计文字当作结果。请先跑 `python v10_supp.py phase P3MTC`（训练对照）与 `phase P3MP`（探针）。')
        A('')
    contr = probe.get('contrasts', {})
    _legacy_keys = [k for k in contr if ' - csa+learned' in k and any((t in k for t in ('full_rope(dense)', 'csa+randidx', 'csa+allblocks'))) and not k.startswith('ΔNLL')]
    _legacy = {k: contr[k] for k in _legacy_keys}
    if _legacy:
        A('**配对检验（exact sign-flip，Δ = 对照臂 − 学习选择臂，>0 表示学习选择更抗噪；uniform 模式）：**')
        A('')
        A('| 比较 | n | Δ mean±std | p (exact) | p 下限 | p_Holm |')
        A('|---|---|---|---|---|---|')
        for k, v in _legacy.items():
            A(f"| {k} | {v['n']} | {_fnum(v.get('mean'), '+.2f')} ± {_fnum(v.get('std'), '.2f')} | {_fnum(v.get('p_exact_signflip'))} | {_fnum(v.get('p_floor'))} | {v.get('p_holm') if v.get('p_holm') is not None else '—'} |")
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
            # Direction, raw p, Holm p and n floor must hold for the word
          # "significant"; |mean|>std is a STABILITY diagnostic reported
          # separately (review 10 issue 8), never a significance verdict:
            # the raw-p version could call a resolution-limited or noise-level
            # contrast "strong evidence" while the table printed p_holm=1.0.
            return all((v.get('mean', 0) > 0 and
                        L.reportable_claim(v, family_size=len(_legacy), alpha=alpha,
                                           min_n=6, require_holm=True)['allowed'])
                       for _k, v in vs)
        dn, ab, ri = (_rng('dense'), _rng('allblocks'), _rng('randidx'))
        ri_sig = _all_sig('randidx')
        def _floor_n(v):
            n = L._as_seed_int(v.get('n'))
            nn = L._as_seed_int(v.get('n_nonzero'))
            return (nn if nn is not None and (n is None or 0 <= nn <= n)
                    else (n or 0))
        _n_by_c = [(k, _floor_n(v)) for k, v in _legacy.items() if 'omitted' not in v]
        n_min = min((n for _k, n in _n_by_c), default=0)
        _bind = [k for k, n in _n_by_c if n == n_min]
        n_min_ri = min((n for k, n in _n_by_c if 'randidx' in k), default=0)
        # `2.0 / 2**n_min` overflows int->float conversion for n_min >= 1024.
        # The canonical floor helper evaluates the same 2/2**n in the float
        # exponent range (ldexp), so report generation still works at scale.
        p_floor = L.signflip_p_floor(n_min)
        learned_helps = ri_sig
        _dpos = bool(dn) and dn[0] > 0 and bool(ab) and ab[0] > 0
        _dtxt = ('正值表示对照臂（dense/allblocks）在污染下损失更大，方向与「限制远距暴露会降低污染损失」一致；'
                 '但这两个臂同时改变读取范围、KV 表示与容量，不能据此单独归因于「稀疏性」或「远端噪声」。'
                 if _dpos else
                 '区间并非全正，「远距暴露致害」的方向在本轮产物上不成立，以配对数字为准。')
        _mech_note = ('> **归因边界**：`randidx` 是固定确定性哈希（不是每次前向重随机），`allblocks` 保留压缩块表示并大幅改变读取范围与 STE 先验，`dense` 同时改变表示/参数/核；'
                      '三者都不是单变量隔离。即使 randidx 全线更差，最多支持「learned 选择优于该固定伪随机连接模式」这一系统级陈述，'
                      '不能支持「学习到的检索质量有独立贡献」或「抗噪性来自稀疏归纳偏置」的机制结论。'
                      '直接机制证据需要 `audit_gates.py selection` 的 selection-attribution 工件（当前仓库没有该工件）。')
        A(f'**系统级对比读数（区间实时算自上方配对检验，Δ = 对照臂 − 学习选择臂；不是单因素机制归因）：**(1) dense−learned Δ 区间 {_fmt(dn)} PPL，allblocks−learned Δ 区间 {_fmt(ab)} PPL——{_dtxt}；(2) randidx−learned Δ 区间 {_fmt(ri)} PPL——' + ('全为正值且全部对照达到 p<0.05：learned 臂与控制臂存在**系统级**差异（统计显著性成立；稳定性见 mean/std 列，|mean|>std 只作稳定性诊断）。' if learned_helps else f'未能在 p<0.05 上成立（randidx 对照的最小 n={n_min_ri}，全表最小 n={n_min}[{', '.join(sorted(_bind)[:2])}{('…' if len(_bind) > 2 else '')}]，精确符号翻转的 p 下限为 {p_floor:.3f}，' + ('该种子数下无法达到 0.05 显著性' if not (p_floor <= 0.05) else '分辨率已足以达到 0.05，未显著是数据结论而非分辨率上限') + '），只能作方向性读数。') + '）')
        A('')
        A(_mech_note)
        A('')
    # ---- P3M-2：从头训练的选择器对照 ----
    _new_c = {k: v for k, v in contr.items() if k not in _legacy and not k.startswith('ΔNLL')}
    if _new_c:
        A('### P3M-2 从头训练的选择器对照（同规模）')
        A('')
        A('**背景**：测试时把已训练模型的选择器替换为随机/全块，只能回答「模型是否依赖原选择器」，不能回答「学习式选择是否优于训练充分的随机选择或全块读取」——后两者的测试时干预改变了模型训练时适应的输入分布，效果下降可能来自失配本身。P3MTC 按与 P3MT **完全相同的配方**（同数据、同种子）从头训练两个对照：`csa_fixed_rope_randidx`（训练期选择即随机）与 `csa_fixed_rope_allblocks`（训练期就读全部块；与 `csa_fixed_rope` 同架构，仅选择方式不同）。**注意**：`allblocks` 仍保留 CSA 的 KV 压缩、池化、q/k 归一化与 scale=1、sink 和窗口分支，因此它只是**选择方式的同架构对照**，不是 dense 对照；把它与 `full_rope` 相减只能作系统级读数，不能把差值归因于单一组件。')
        A('')
        A('| 比较 | n | Δ mean±std | d_z | p (exact) | p 下限 |')
        A('|---|---|---|---|---|---|')
        for k, v in sorted(_new_c.items()):
            A(f"| {k} | {v['n']} | {_fnum(v.get('mean'), '+.2f')} ± {_fnum(v.get('std'), '.2f')} | {_fnum(v.get('cohens_dz'), '+.2f')} | {_fnum(v.get('p_exact_signflip'))} | {_fnum(v.get('p_floor'))} |")
        A('')
        A('> `retrain-gap(…)` 行 = 测试时干预 − 同规模从头训练对照（同 L、同 ρ）。对越低越好的损失，正值说明干预比重新训练**更差**，该差距本身**可能包含**训练—测试失配的代价，也混合了优化适应与参数差异——它是「干预与重新训练的效果差」，**不是**已完成的因果失配分解，不得把简单相减解释为「不能归因于失配的部分」。`system-level` 行 = dense − 全块 CSA 孪生；两者仍差 KV 压缩、池化、q/k 归一化位置与宽度、attention scale、sink、窗口分支与 MLP 宽度，**不是**逐项对齐的对照。')
        A('')
    else:
        A('> **P3M-2 对照未测量**：P3MTC 的从头训练随机/全块选择器没有对应的 probe cell（或这些 cell 因 recipe/code/val 指纹不一致被拒绝），因此「学习式选择 vs 训练充分的替代选择」在本报告中**没有数字**。测试时干预只能回答「模型是否依赖原选择器」，不能替代这一对照。')
        A('')
    # ---- ΔNLL：污染增量 ----
    dnll = probe.get('delta_nll') or {}
    if dnll:
        A('### P3M-3 污染增量 ΔNLL（按 NLL 口径，含干净条件基线）')
        A('')
        A('**口径**：ΔNLL = NLL_污染 − NLL_干净（nats/目标 token，逐种子配对）。腐坏模式共 7 种：`uniform`（均匀随机 token，既有协议）、`natural`（其他真实序列同位置 token）、`span`（连续句段替换）、`doc`（整段换成另一条真实序列的连续切片，最接近自然文档干扰）、`topical`（主题相近供体的同位置替换）、`mask`（远端整体屏蔽为 `<pad>`）、`delete`（远端整体删除，仅剩目标区）。`mask`/`delete` 回答「干净远端上下文到底带来多少收益」；`doc` 是自然文档级干扰，`natural`/`span`/`topical` 只回答特定 token/句段替换下的鲁棒性，不能据此声称普遍抗噪性；`uniform` 与 `natural` 之差也不等于「自然语言干扰」的完整效应。')
        A('')
        # Review 10 issue 12: rho is NOMINAL.  Publish what each mode actually
        # covered/changed so modes are not compared as if the same rho meant
        # the same realised corruption.  The audit is arm-independent; when a
        # legacy record predates it the cell is marked n/a rather than guessed.
        _audit_rows = {}
        for _c in (probe.get('cells') or []):
            _a = _c.get('corruption_audit')
            _akey = (_c['eval_len'], _c['mode'], _c['rho'])
            if isinstance(_a, dict):
                _audit_rows.setdefault(_akey, _a)
        if _audit_rows:
            A('**实际腐坏几何**（ρ 是名义比例；`span` 的实际覆盖受段长量化，供体 token 可能与原文相同，故覆盖 ≠ 改变）：')
            A('')
            A('| L | 模式 | ρ 名义 | 覆盖槽位 | 唯一覆盖位置 | 重叠槽位 | 实际改变 token | 实现覆盖率 | 改变率 | 空操作行 |')
            A('|---|---|---|---|---|---|---|---|---|---|')
            for (_L, _mode, _rho) in sorted(_audit_rows):
                _a = _audit_rows[(_L, _mode, _rho)]
                _cf = _a.get('changed_fraction_of_covered')
                A(f"| {_L} | {_mode} | {_rho:g} | {_a.get('covered_token_slots_total', 0)} | {_a.get('covered_unique_positions_total', 0)} | {_a.get('overlap_token_slots_total', 0)} | {_a.get('changed_tokens_total', 0)} | {_a.get('realised_coverage_fraction', 0.0):.3f} | {_a.get('realised_changed_fraction', 0.0):.3f} | {_a.get('no_op_rows', 0)} |")
            A('')
            A('> 读法：同一 ρ 下不同模式的**实际改变比例**不同，不能把「ρ 相同」当作「腐坏强度相同」；跨模式比较一律以本表的实现覆盖/改变列为准，ΔNLL 只回答腐坏增量。若某行显示「空操作行 > 0」（自然/主题/文档模式的供体 token 与原 token 相同），该行是内容未变的假腐坏，必须单独说明。')
            A('')
        A('| 臂 | L | 模式 | ρ | ΔNLL mean±std | n | 干净 NLL |')
        A('|---|---|---|---|---|---|---|')
        _clean_nll_cache = {}
        for k in sorted(dnll):
            e = dnll[k]
            _ck = f"{e['variant']}::{e['arm']}::L{e['eval_len']}"
            if _ck not in _clean_nll_cache:
                # The clean rho=0 cells are omitted from delta_nll by design;
                # recover the per-seed clean values that every corrupted entry
                # already carries.
                _cl = {}
                for x in dnll.values():
                    if (x['variant'] == e['variant'] and x['arm'] == e['arm']
                            and x['eval_len'] == e['eval_len']):
                        for _s, _v in (x.get('clean_nll_by_seed') or {}).items():
                            _cl.setdefault(_s, _v)
                _clean_nll_cache[_ck] = (float(np.mean(list(_cl.values())))
                                         if _cl else float('nan'))
            A(f"| `{e['variant']}::{e['arm']}` | {e['eval_len']} | {e['mode']} | {e['rho']:g} | {e['mean']:+.4f} ± {e.get('std', 0):.4f} | {e['n']} | {_clean_nll_cache[_ck]:.4f} |")
        A('')
        _dnll_c = {k: v for k, v in contr.items() if k.startswith('ΔNLL')}
        if _dnll_c:
            A('| ΔNLL 配对比较（ΔΔNLL = 臂间污染增量之差） | n | Δ mean±std | p (exact) | p 下限 | p_Holm |')
            A('|---|---|---|---|---|---|')
            for k, v in sorted(_dnll_c.items()):
                A(f"| {k} | {v['n']} | {_fnum(v.get('mean'), '+.4f')} ± {_fnum(v.get('std'), '.4f')} | {_fnum(v.get('p_exact_signflip'))} | {_fnum(v.get('p_floor'))} | {v.get('p_holm') if v.get('p_holm') is not None else '—'} |")
            A('')
        A('> **替代解释提示**：某臂 ΔNLL 小也可能因为它本来就很少利用远端上下文（干净 NLL 更高）——读表时必须把 ΔNLL 与干净 NLL 列一起看；`mask`/`delete` 行给出干净远端的收益幅度作为参照系。')
        A('')
        # ---- clean far-context benefit (mask/delete) ----
        _fb = probe.get('far_benefit') or {}
        if _fb:
            A('### P3M-3b 干净远端上下文的收益（mask/delete）')
            A('')
            A('**口径**：收益 = NLL_污染 − NLL_干净 = ΔNLL；正值表示**干净远端上下文有用**，接近 0 表示该臂几乎不依赖远端。这个量是判断「ΔNLL 小是否只是因为本来就不用远端」的直接参照：收益小 + 污染增量小 = 低依赖，而不是强鲁棒。')
            A('')
            A('| 臂 | L | 条件 | 收益 mean±std (nats/token) | n | 干净 NLL | p (exact) | p 下限 |')
            A('|---|---|---|---|---|---|---|---|')
            for _k in sorted(_fb, key=lambda x: (_fb[x]['eval_len'], _fb[x]['variant'], _fb[x]['arm'], _fb[x]['mode'])):
                _e = _fb[_k]
                _clean = _e.get('clean_nll_by_seed') or {}
                _ct = (float(np.mean(list(_clean.values()))) if _clean else float('nan'))
                A(f"| `{_e['variant']}::{_e['arm']}` | {_e['eval_len']} | {_e['mode']} | {_fnum(_e.get('mean'), '+.4f')} ± {_fnum(_e.get('std'), '.4f')} | {_e['n']} | {_fnum(_ct, '.4f')} | {_fnum(_e.get('p_exact_signflip'))} | {_fnum(_e.get('p_floor'))} |")
            A('')
            A('> `mask`/`delete` 都改变远端输入分布（`<pad>` 是分布外 token；`delete` 改变长度与位置），因此两者的收益幅度不能互相校准；它们只用于回答「该臂是否使用远端」，不用于估计「自然干扰下的鲁棒性」。本表的 p 是**单样本**符号翻转，未进入上面的 Holm 族，只作描述性读数；不得据此单独宣称显著。')
            A('')
    # ---- NIAH-natural：远端必要信息任务 ----
    niah = probe.get('niah_natural') or {}
    if niah.get('cells'):
        A('### P3M-4 远端必要信息任务（NIAH-natural，自然文本 + 自然干扰）')
        A('')
        A('**设计**：序列 = 远端自然文本（内嵌 8 个 key→value 耦合）+ 目标区（自然文本收尾 + 8 个查询 key）。查询 key 的下一个 token 的正确答案只在远端出现。`distract` 条件把 ρ=0.5 的远端草料换成其他真实序列的 token（realistic-token 替换；不是主题/语义干扰），但**绝不腐坏耦合位置**；`nofar` 条件**删除远端耦合**（无远端信息对照，准确率应退回 chance）。报告 answer top-1 准确率与 answer NLL。**注意**：把随机 token 的 key→value 对插入自然文本，不等于模型具备该格式下的远距检索能力；chance 水平也可能只是任务格式不适配。因此本报告还给出任务有效性的机器判据（见下），不满足时该任务只能作描述性读数。')
        A('')
        A('| 臂 | L | 条件 | acc | chance | 高于 chance? | answer NLL | n |')
        A('|---|---|---|---|---|---|---|---|')
        _nval = probe.get('niah_task_validity') or {}
        for c in sorted(niah['cells'], key=lambda c: (c['eval_len'], c['condition'], c['variant'], c['arm'])):
            _ch = c.get('chance_acc')
            _chtxt = '—' if _ch is None else f'{float(_ch):.5f}'
            _v = _nval.get(f"{c['variant']}::{c['arm']}::L{c['eval_len']}")
            _ab = '—'
            if _v is not None and c['condition'] == 'clean':
                _ab = '是' if _v.get('above_chance') else '否'
            A(f"| `{c['variant']}::{c['arm']}` | {c['eval_len']} | {c['condition']} | {c['acc_mean']:.3f} | {_chtxt} | {_ab} | {c['nll_mean']:.3f} | {c['n']} |")
        A('')
        if _nval:
            A('**任务有效性判据（机器门；不满足则 NIAH 不能作为机制证据）**：')
            A('')
            A('| 臂 | L | clean acc | chance | 二项 p(>chance) | nofar acc | nofar 与 chance 一致 | 正对照已记录 | 可用于机制主张 |')
            A('|---|---|---|---|---|---|---|---|---|')
            for _k, _v in sorted(_nval.items()):
                _pp = _v.get('binomial_p_above_chance')
                _ptxt = '—' if _pp is None else f'{float(_pp):.3g}'
                _nfa = _v.get('nofar_acc')
                _nftxt = '—' if _nfa is None else f'{float(_nfa):.3f}'
                A(f"| `{_k}` | — | {_v.get('clean_acc', float('nan')):.3f} | {(_v.get('chance_acc') or 0.0):.5f} | {_ptxt} | {_nftxt} | {_v.get('nofar_consistent_with_chance')} | {_v.get('positive_control_recorded')} | {_v.get('task_valid_for_mechanism_claim')} |")
            A('')
            A('> ' + str(probe.get('niah_validity_note') or '') + (' **结论：当前面板的 NIAH 只能作为描述性读数，不支持「长距离信息利用」的机制主张（缺少可解任务的正对照）。**' if not probe.get('niah_mechanism_usable') else ' 结论：三个条件均满足，NIAH 可作为远距利用的机制证据。'))
            A('')
        # distance buckets (short-to-long gradient) per arm/length/condition
        _nb = {}
        for c in niah['cells']:
            if c.get('acc_by_distance_bucket'):
                _nb.setdefault((c['variant'], c['arm'], c['eval_len'], c['condition']), c)
        if _nb:
            A('**距离分桶准确率（short→long 梯度；平坦或反向说明该任务没有区分度）**：')
            A('')
            A('| 臂 | L | 条件 | 桶0 (dist) | 桶1 | 桶2 | 桶3 (dist) | 梯度 |')
            A('|---|---|---|---|---|---|---|---|')
            for _k, c in sorted(_nb.items()):
                _bs = c['acc_by_distance_bucket']
                def _btxt(i):
                    if i >= len(_bs):
                        return '—'
                    b = _bs[i]
                    return f"acc={b['acc']:.3f} (d={b['dist_min']}-{b['dist_max']})"
                A(f"| `{_k[0]}::{_k[1]}` | {_k[2]} | {_k[3]} | {_btxt(0)} | {_btxt(1)} | {_btxt(2)} | {_btxt(3)} | {c.get('distance_gradient')} |")
            A('')
        if niah.get('contrasts'):
            A('| NIAH 配对比较（Δ = 对照臂 − 学习选择，answer NLL，>0 = 学习选择更好） | n | Δ mean±std | p (exact) | p 下限 | p_Holm |')
            A('|---|---|---|---|---|---|')
            for k, v in sorted(niah['contrasts'].items()):
                A(f"| {k} | {v['n']} | {_fnum(v.get('mean'), '+.3f')} ± {_fnum(v.get('std'), '.3f')} | {_fnum(v.get('p_exact_signflip'))} | {_fnum(v.get('p_floor'))} | {v.get('p_holm') if v.get('p_holm') is not None else '—'} |")
            A('')
        A('> 该任务**在通过上表任务有效性判据后**才可用于「长距离信息利用」的机制读法：鲁棒性高 + 检索利用不差，才支持「稀疏选择在滤噪的同时保留了有用远距信息」；若只有鲁棒性而检索差，则更符合「本来就少用远端」的替代解释。正对照、无远端信息对照、chance 与距离梯度缺一不可；任一缺失时 NIAH 读数只能描述成败，不能解释机制。')
        A('')
    A('### 统计口径')
    A('')
    _n_all = sorted({v.get('n', 0) for v in contr.values() if isinstance(v, dict) and 'omitted' not in v})
    A(f'- 每个配对比较都给出逐种子 Δ、效应量 d_z 与该 n 下的双侧精确符号翻转 p 值下限 2/2^n（本表 n 集合 {_n_all}）。')
    A('- 上表全部 p 值按 **Holm step-down** 在本报告的对比族内做了多重校正（`analysis_v10/stats.json` 的 `probe.holm_exploratory`）；该校正族**未预先注册**，因此校正后的 p 只能防止过度宣称，不能当作确认性证据。')
    _hm = probe.get('holm_exploratory') or {}
    if isinstance(_hm, dict) and _hm.get('family_min_attainable_p_holm') is not None:
        _floor = _hm.get('family_min_attainable_p_holm')
        A(f'- **族级分辨率**：声明槽位 m={_hm.get('declared_slots')}，n_nonzero_max={_hm.get('n_effective_max')}；该族最小可达 Holm p = {_floor:.4g}。'
          + ('⚠ 该值 > α=0.05：**整个族在当前种子数下不可能拒绝任何比较**——p_Holm=1.0 是分辨率上限，不是零效应。' if _hm.get('family_resolution_limited') else '该值 ≤ α，族内至少有一个比较在分辨率上可达显著。'))
    A('- **检验假设边界**：精确符号翻转检验的尖锐零假设是「每个种子的配对差符号关于 0 对称」；同 seed/数据顺序/评测行/腐坏网格只是控制，不提供随机分配意义上的交换性。p 是 seed-level，不是 token/window-level；|mean|>std 只作稳定性诊断，不是显著性判据。')
    A('- v7→v11 的多轮追加实验与既有面板属于**探索性**证据；确认性检验见 `v12_supp.py`（冻结清单 `confirmation_frozen.json`：固定主比较、终点、种子数，用 wiki.test.raw 独立 test split 一次性评估）。')
    _gm = st.get('global_multiplicity') or {}
    if isinstance(_gm, dict) and _gm.get('m'):
        A('')
        A('### 跨面板多重性（BH-FDR + Bonferroni）')
        A('')
        A(f'- 本报告显示的全部 p 值共 m={_gm.get("m")} 个；原始 p<0.05 的 {_gm.get("n_raw_significant", 0)} 个；BH-FDR q<0.05 的 {_gm.get("n_bh_rejected", 0)} 个；Bonferroni p<0.05 的 {_gm.get("n_bonferroni_significant", 0)} 个；无有效 p 的 {_gm.get("n_untestable", 0)} 个（计入族大小，p 槽位=1）。')
        A('- 跨面板引用（「在 probe 与 NIAH 都显著」）必须按该全局视图措辞；族内 Holm 不能替代它。该全局视图仍是探索性纪律，不把任何面板变成预注册证据。')
    A('')
    A('---')
    A('')
    A('## P3S 反转交叉点的标度读数（4 个模型规模）')
    A('')
    A('**判据（分析前声明的探索性判据，不是冻结预注册）**：对每个规模的 `csa_fixed − full` 验证 PPL 差距轨迹（eval_every=250 或面板既有网格），取 ±1 点平滑后**首次转正且此后不再回穿**的 eval 步为交叉步；轨迹内未交叉记为删失（> 末步）。token 数 = 步数 × 每步 token。')
    A('')
    A('| 规模 | 面板 | seeds | 交叉步（seed 均值轨迹） | 交叉 token | 状态 |')
    A('|---|---|---|---|---|---|')
    _recon = {k for k, v in xo.items() if v.get('status') == 'ok' and (v.get('excluded_reconstructed') or any((v.get('per_seed_synth', {}).get(str(s)) for s in v.get('seeds', []))))}
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
    _sens_diff = []
    for _lab, _v in xo.items():
        _ss = _v.get('smoothing_sensitivity') if isinstance(_v, dict) else None
        if not isinstance(_ss, dict) or not _ss:
            continue
        _vals = [(k, x.get('crossed'), x.get('step')) for k, x in _ss.items() if isinstance(x, dict)]
        if _vals and len({(c, None if s is None else round(float(s), 6)) for _k, c, s in _vals}) > 1:
            _sens_diff.append(_lab + ': ' + ', '.join(f'smooth={k}:{"crossed@" + format(float(x.get("step")), "g") if x.get("crossed") else "censored"}' for k, x in _ss.items() if isinstance(x, dict)))
    if _sens_diff:
        A('> **平滑敏感性**：下列面板的「首次转正且不再回穿」交叉步随 ±0/±1/±2 点平滑窗口改变（或从交叉变为删失）；单一平滑口径的交叉点不是稳健读数：' + '；'.join(_sens_diff))
        A('')
    _xo256 = xo.get('d256_L6') or next((v for v in xo.values() if v.get('d') == 256), {})
    _xo256_step = _xo256.get('mean_crossover_step')
    _xo256_txt = f'~{_xo256_step:g} 步' if isinstance(_xo256_step, (int, float)) and math.isfinite(_xo256_step) else '步数未知（该规模无有效交叉读数）'
    _xo256_recon = bool(_xo256.get('excluded_reconstructed')) or any(((_xo256.get('per_seed_synth') or {}).get(str(s)) for s in _xo256.get('seeds', [])))
    _synth_all = _panel_synth_seeds('results_lm_v3_long')
    if _xo256_recon or _synth_all:
        _ss_txt = (f'（存在 reconstructed seed(s): {', '.join(_synth_all[:6])}{'…' if len(_synth_all) > 6 else ''}）' if _synth_all else '')
        A('**注意**：d=256 面板含 `synthesized` 重构记录' + _ss_txt + '；凡依赖这些 seed 的逐种子轨迹、seed 均值轨迹与交叉点拟合均已剔除（拟合状态为 `reconstruction_only`），因此该规模没有可独立复核的交叉点，本节不引用重构数值。')
    elif _synth_all is None:
        A('**注意**：无法读取 d=256 面板的 `summary.json` 来确认是否存在重构记录——该面板的重构状态是「未知」，不得写成「不含重构记录」。')
    elif _xo256.get('status') == 'ok':
        A(f'**注意**：d=256 面板（`results_lm_v3_long`）的 summary.json 不含 `synthesized` 记录，逐种子交叉步（{_xo256_txt} 附近）与 seed 均值轨迹同为有效读数。')
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
              f'训练量比 {_vol:.2f}（要求 ≤1.25）。这是观测性回归，规模面板未随机化，不能写成 scaling law。'
              '**横轴不是单一规模变量**：d 与层数一一绑定（128/4L、256/6L、384/8L、512/10L），各面板的 batch/seq/eval 网格也不同，'
              '因此斜率同时混入深度、优化动态与评测网格差异；它只能读作「这些配置的轨迹交叉点随 d 的描述性变化」。')
            A('')
    A('**规模轴混杂（无论拟合是否被拒都必须读）**：d 与层数一一绑定（128/4L、256/6L、384/8L、512/10L），各面板的 batch/seq/eval 网格与训练 token 数也不同；即使趋势拟合被拒绝，也不能把跨面板差异读作单一「规模」效应。')
    A('')
    sc = st.get('scale_confounds') or {}
    if sc:
        A('### P3S-2 规模混杂控制（等训练 token / 6ND 代理规模 / 规模内平台期）')
        A('')
        A('**背景**：P3SS（d=128/4L，小模型臂）与 P3SL（d=512/10L，大模型臂）的累计训练 token 不同——跨规模终点差异不能单独归因于规模。本节把同一批曲线在**相同累计 token**处切片配对（gap_L − gap_S，逐种子），并给出等 token（`P3SL_TOK`）与 6ND 参数量代理规模（`P3SL_FLOP`，按参数量比例缩步数——**代理规模，非实测等 FLOPs**）两个大模型臂，以及各面板的规模内平台期诊断（平台期 ≠ 收敛，只支持「规模内平台期后」读数）。')
        A('')
        _panels = sc.get('panels') or {}
        _budget = {'P3SS': 'raw 8000 steps', 'P3SL': 'raw 4000 steps',
                   'P3SL_TOK': 'equal-token 8000 steps', 'P3SL_FLOP': '6ND proxy (NOT FLOPs)'}
        if _panels:
            A('| 面板 | 规模 | 预算口径 | 总步数 | 末步 token | 参数 csa/full (csa−full)/full | 末点 gap (csa−full) mean±std | n | 末点 p (exact) | p 下限 | 平台化种子 |')
            A('|---|---|---|---|---|---|---|---|---|---|---|')
            for name in ('P3SS', 'P3SL', 'P3SL_TOK', 'P3SL_FLOP'):
                e = _panels.get(name) or {}
                if not e or e.get('status') != 'ok':
                    A(f'| {name} | — | {_budget.get(name, "—")} | — | — | — | 数据缺失 | 0 | — | — | — |')
                    continue
                _ep = e.get('endpoint') or {}
                _ntok = (f"{(_ep.get('tokens') or 0) / 1000000.0:.2f}M" if _ep and _ep.get('tokens') else 'n/a')
                _pc = e.get('params') or {}
                _ptxt = '—'
                if _pc.get('csa_fixed') and _pc.get('full'):
                    _pd = e.get('param_rel_diff_csa_vs_full')
                    _ptxt = f"{_pc['csa_fixed'] / 1e6:.2f}M/{_pc['full'] / 1e6:.2f}M ({_pd:+.1%})" if isinstance(_pd, (int, float)) else '—'
                _conv = e.get('convergence') or {}
                _pl = []
                for _vv in ('csa_fixed', 'full'):
                    _cc = _conv.get(_vv) or {}
                    _lbl = 'csa' if _vv == 'csa_fixed' else 'full'
                    _pl.append((f'{_lbl} eval不足' if _cc.get('n_considered', 0) == 0
                                else f"{_lbl} {_cc.get('n_plateaued', 0)}/{_cc.get('n_considered', 0)}"))
                _steps = e.get('steps_run')
                _stxt = str(int(_steps)) if isinstance(_steps, (int, float)) else '—'
                A(f"| {name} | {e.get('note', '')} | {_budget.get(name, '')} | {_stxt} | {_ntok} | {_ptxt} | {_fnum(_ep.get('mean'), '+.2f')} ± {_fnum(_ep.get('std'), '.2f')} | {_ep.get('n', 0)} | {_fnum(_ep.get('p_exact_signflip'))} | {_fnum(_ep.get('p_floor'))} | {'; '.join(_pl) or '—'} |")
            A('')
            _conv_insufficient = []
            _conv_unplateaued = []
            for _name, _e in _panels.items():
                if _e.get('status') != 'ok':
                    continue
                for _vv in ('csa_fixed', 'full'):
                    _cc = ((_e.get('convergence') or {}).get(_vv) or {})
                    if _cc.get('n_considered', 0) == 0:
                        _conv_insufficient.append(f'{_name}/{_vv}')
                    elif _cc.get('n_plateaued', 0) == 0:
                        _conv_unplateaued.append(f'{_name}/{_vv}')
            if _conv_insufficient:
                A(f'> **平台期不可诊断**：{_conv_insufficient} 的可用 eval 点少于 4 个（例如短预算臂），不能据此写「未平台化」或「已平台化」；该面板只给端点读数。')
                A('')
            if _conv_unplateaued:
                A(f'> **规模内平台期证据不足**：{_conv_unplateaued} 在最后四分之一 eval 窗口的相对改进 ≥1%（未平台化）——「平台期后交叉」的主张在该面板上**不成立**，只有「窗口内交叉」的读数有效。')
                A('')
            A('> 即使全部种子平台化，也只允许「规模内平台期后交叉」的读数；「渐近优势反转」「收敛后更差」需要更长训练、学习率续训与重复验证，本面板不提供。')
            A('')
            _pd_rows = [(n, (_panels.get(n) or {}).get('param_rel_diff_csa_vs_full')) for n in ('P3SS', 'P3SL', 'P3SL_TOK', 'P3SL_FLOP')]
            _pd_rows = [(n, x) for n, x in _pd_rows if isinstance(x, (int, float))]
            if len(_pd_rows) >= 2:
                A('> **容量混杂**：' + '；'.join(f'{n} 的 csa−full 参数差 {x:+.1%}' for n, x in _pd_rows) +
                  '。跨规模 gap 的变化不能先归因于规模：CSA 相对 dense 的参数优势本身随宽度缩小，若没有参数匹配的 dense 臂，容量差就可以单独产生「gap 随规模变化」。')
                A('')
            A('> **学习率轨迹**：只有 `P3SL_TOK` 与 P3SS 同为 8000 总步（同一 warmup/cosine 形状）；`P3SL`（4000 步）与 `P3SL_FLOP`（659 步）在同一累计 token 处的 LR 不同，因此下面的切片分成「LR 对齐」与「仅 token 对齐」两组。')
            A('')
            A('> **评测口径**：末点 gap 来自 `eval_subset=64` 的曲线点，不是完整 validation 的最终 PPL；`tokens` 是训练**抽样** token 数（不等于去重后的唯一 token 数）。')
            A('')
        _slices = sc.get('equal_token_slices') or {}
        if _slices:
            _aligned = {k: v for k, v in _slices.items() if v.get('lr_schedule_aligned')}
            _misaligned = {k: v for k, v in _slices.items() if not v.get('lr_schedule_aligned')}
            for _label, _grp, _why in (
                    ('等 token 且同 LR 轨迹（唯一无 LR 混杂的切片）', _aligned,
                     'P3SL_TOK 与 P3SS 同为 8000 步'),
                    ('仅等 token（LR 轨迹不同，LR 仍是混杂）', _misaligned,
                     'P3SL=4000 步 / P3SL_FLOP=659 步，与 P3SS=8000 步的 LR 形状不同')):
                if not _grp:
                    continue
                A(f'**{_label}**（{_why}；gap_L − gap_S，逐种子配对；>0 = 大模型在同 token 处 CSA−dense 差距更大）：')
                A('')
                A('| 规模臂 | tokens | small step | large step | gap_S mean | gap_L mean | Δ mean±std | n | p (exact) | p 下限 |')
                A('|---|---|---|---|---|---|---|---|---|---|')
                for k, v in sorted(_grp.items(), key=lambda kv: (kv[1].get('tokens_small', 0.0), kv[0])):
                    A(f"| {k.split(' - P3SS')[0]} | {v.get('tokens_small', 0.0) / 1e6:.2f}M | {v.get('small_step', '—')} | {v.get('large_step', '—')} | {_fnum(v.get('gap_S_mean'), '+.2f')} | {_fnum(v.get('gap_L_mean'), '+.2f')} | {_fnum(v.get('mean'), '+.2f')} ± {_fnum(v.get('std'), '.2f')} | {v.get('n', 0)} | {_fnum(v.get('p_exact_signflip'))} | {_fnum(v.get('p_floor'))} |")
                A('')
            _floors = [v.get('p_floor') for v in _slices.values() if isinstance(v.get('p_floor'), (int, float))]
            if _floors and max(_floors) > 0.05:
                A(f'> **分辨率警告**：这些切片的精确符号翻转 p 下限最高为 {max(_floors):.3f}（当前 scale 臂 n=2 时下限 0.500，v11 补到 n=4 仍为 0.125）——即使方向全部一致，也**不能**达到 p<0.05；表中的 p 只能读作描述性排序证据。')
                A('')
            A('> `P3SL_FLOP` 的步数按面板平均参数量比例确定性缩放（659 步 / 4.05M token），不是实测 FLOPs、GPU 时间或能耗；它的 gap 只能读作「该 6ND 代理近似相等时」的规模效应。')
            A('')
        else:
            A('> **等 token / 等计算控制未产出**：`P3SL_TOK` / `P3SL_FLOP` 没有可用曲线，`equal_token_slices` 为空。因此跨规模端点差异仍然同时混杂训练 token 数、LR 轨迹与容量差，**不能**单独归因于规模；本报告不提供「优势随规模反转」的证据。')
            A('')
        _ps = _panels.get('P3SS') or {}
        _plf = _panels.get('P3SL_FLOP') or {}
        _pcs, _pcl = (_ps.get('params') or {}), (_plf.get('params') or {})
        _ss, _sl = _ps.get('steps_run'), _plf.get('steps_run')
        if (_ps.get('endpoint') and _plf.get('endpoint') and _pcs and _pcl
                and isinstance(_ss, (int, float)) and isinstance(_sl, (int, float)) and _sl > 0):
            A('**等计算代理臂的逐臂 6ND 对齐（P3SS 8000 步 vs P3SL_FLOP 缩放步）**：')
            A('')
            A('| 臂 | P3SS 参数 | P3SL_FLOP 参数 | steps | 6ND 比值 (small/large) | 是否对齐 |')
            A('|---|---|---|---|---|---|')
            for _arm in ('csa_fixed', 'full'):
                _a, _b = _pcs.get(_arm), _pcl.get(_arm)
                if not _a or not _b:
                    continue
                _r = (float(_a) * float(_ss)) / (float(_b) * float(_sl))
                A(f"| {_arm} | {_a / 1e6:.2f}M | {_b / 1e6:.2f}M | {int(_ss)}/{int(_sl)} | {_r:.3f}× | {'是（±5%）' if abs(_r - 1.0) <= 0.05 else '否'} |")
            A('')
            A('> 该臂用**面板平均参数**决定步数，逐臂 6ND 比值并不严格等于 1（见上表）；它既不是逐臂等计算，也不是实测 FLOPs。真正等计算的比较需要测得的 FLOPs 或同 kernel 墙钟时间，当前仓库没有这两个量。')
            A('')
    A('---')
    A('')
    A('## P3T seq-2048 topk 扫描补种子（n=2 → n=4）')
    A('')
    A('**背景**：v9 P2T 的完整 5 点扫描只有 2 seeds，精确符号翻转 p 值下限 0.500——v9 收官后唯一仍处该下限的面板。v10 按同一配方（seq 2048、bs 1）补 seeds 2/3（同一 outdir，resume 跳过 seed 0/1）。')
    A('')
    A('| variant | PPL pooled (`exp(ΣNLL/Σtok)`) | PPL seed mean±std | median [IQR] | n |')
    A('|---|---|---|---|---|')
    _p3t_done = 0
    for v in ['csa_fixed_topk8', 'csa_fixed_topk32', 'csa_fixed_topk128', 'csa_fixed_topk512', 'csa_fix_m1']:
        if v in seq2k:
            p = seq2k[v]
            _pool = p.get('ppl_pooled')
            _pool_txt = f'{_pool:.2f}' if isinstance(_pool, (int, float)) and np.isfinite(_pool) else '—（缺 NLL/token）'
            _med = p.get('ppl_median')
            _iqr = p.get('ppl_iqr')
            _med_txt = (f'{_med:.2f} [{_iqr[0]:.2f}, {_iqr[1]:.2f}]'
                        if isinstance(_med, (int, float)) and isinstance(_iqr, (list, tuple)) and len(_iqr) == 2 else '—')
            A(f'| `{v}` | {_pool_txt} | {_fmt_pm(p)} | {_med_txt} | {p['n']} |')
            _p3t_done += 1
        else:
            A(f'| `{v}` | —（未完成） | — | — | 0 |')
    A('')
    A(f'**完成 {_p3t_done}/5 点**。')
    A('')
    A('| 比较 | n | Δ (a−b) mean±std | p (exact) | p 下限 | p_Holm |')
    A('|---|---|---|---|---|---|')
    for k, v in comp.items():
        if k.startswith('seq2k'):
            if 'omitted' in v:
                A(f'| {k} | 0 | —（{v['omitted']}） | — | — | — |')
                continue
            _nn = L._as_seed_int(v.get('n_nonzero'))
            if _nn is None:
                _nn = L._as_seed_int(v.get('n'))
            _fl = L.signflip_p_floor(_nn) if _nn is not None and _nn > 0 else float('nan')
            A(f'| {k} | {v['n']} | {v['mean']:+.2f} ± {v.get('std', 0):.2f} | {v['p_exact_signflip']:.3f} | {_fl:.4f} | {v.get('p_holm') if v.get('p_holm') is not None else '—'} |')
    A('')
    A('**注**：n=4 时精确符号翻转的 p 值分辨率为 0.125——仍跨不过 0.05（需 n=6 才有 0.031）；此行补齐的是与 scale 面板同级的统计分辨率，显著性主张仍以效应量 + 方向同向性表述。')
    A('')
    A('---')
    A('')
    A('> **效率口径（review 12, issue 14）**：本报告不给出 throughput/latency/显存结论。解析 FLOPs 只覆盖 attention 相关项（不含 MLP/QKV 投影/pooling/indexer/top-k/gather/padding/backward/内核启动），且 dense 与 CSA 使用不同实现路径；端到端 prefill/prefix-prefill/峰值显存测量只存在于 v7 报告的 `Measured efficiency` 节（真实 checkpoint、同实现），energy 未测。任何「更省算力」的读法必须换成端到端同 kernel 测量。')
    A('')
    A('## 产物清单')
    A('')
    A('| 路径 | 内容 |')
    A('|---|---|')
    A('| `results_v10_mech/` | P3M：两臂重训权重（ckpt/，*.pt 不入库）+ by-length 复现表 + 干扰注入探针 `distractor.json`（uniform/natural/mask/delete + span/doc/topical 模式；natural = realistic-token，§15 修订）+ NIAH-natural `niah_natural.json` |')
    A('| `results_lm_v10_scale_s/` | P3S：d=128/4L 交叉定位面板 |')
    A('| `results_lm_v10_scale_l/` | P3S：d=512/10L 交叉定位面板 |')
    A('| `results_lm_v10_scale_l8k/` | P3S-2：等 token 大模型臂 |')
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
    print(f'[v10] headroom {max(guard.headroom_hours(), 0.0):.2f} h left (booked {guard.state['booked_seconds'] / 3600:.2f} h)')
    git_push('v10: supplementary driver (P3M distractor-injection mechanism + P3S crossover scaling + P3T topk seeds 2/3 + scratch selector controls, corruption modes/NIAH, token-matched and 6ND-param-proxy-budget scale arms)')
    import par
    _ctx = par.resolve_ctx(label='v10')
    all_ok = True
    for pname, _kind, _payload, _seeds in PHASES:
        rem = guard.headroom_hours()
        if rem < 0.4166666666666667:
            print(f'[v10] stopping before {pname}: headroom exhausted ({rem:.2f} h usable under the run cap) — earlier phases did not complete; re-run to resume')
            all_ok = False
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
    if all_ok:
        print('\n[v10] ALL PHASES DONE.')
    else:
        print('\n[v10] INCOMPLETE — not every planned phase ran cleanly; re-run to resume (checkpoints allow resume).')
    schedule_shutdown(120 if all_ok else 2400)
    if not all_ok:
        sys.exit(1)

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
    if not (_want <= got):
        raise AssertionError(f'probe cells missing: {_want - got} (got {sorted(got)})')
    niah_sum = {}
    _nsp = 'results_smoke_v10/niah_natural.json'
    if os.path.exists(_nsp):
        with open(_nsp, encoding='utf-8') as _jf:
            niah_sum = json.load(_jf)
    if not isinstance(niah_sum, dict):
        raise AssertionError(f'NIAH summary is {type(niah_sum).__name__}, not a JSON object')
    _niah_got = {(r.get('arm'), r.get('condition')) for r in niah_sum.values() if isinstance(r, dict)}
    _niah_want = {('learned', 'clean'), ('learned', 'distract'), ('randidx', 'clean'), ('randidx', 'distract'), ('allblocks', 'clean'), ('allblocks', 'distract')}
    if not (_niah_want <= _niah_got):
        raise AssertionError(f'NIAH cells missing: {_niah_want - _niah_got} (got {sorted(_niah_got)})')
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
    if not (_fs1 == _fs2 and _fs1 >= 250):
        raise AssertionError(f'flop-matched steps not deterministic: {_fs1} vs {_fs2}')
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
            git_push('v10: PARTIAL — interrupted, see log (re-run resumes)')
        except Exception:
            traceback.print_exc()
        if mode == 'full':
            schedule_shutdown(2400)
        sys.exit(1)
