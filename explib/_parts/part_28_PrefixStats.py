
class _PrefixStats:
    """Diagnostics for the open-block (current-prefix) reading path.

    Accumulates per-query quantities only when the caller enables block stats
    (the validation/eval report), so training keeps its fast path.  Nothing
    here is a loss term or a model input.
    """

    def __init__(self):
        self.n_tok = 0
        self.n_seq = 0
        self.age_sum = 0.0
        self.age_max = 0
        self.age_p95 = []
        self.outside_len_sum = 0.0
        self.outside_nonzero = 0
        self.prefix_len_sum = 0.0
        self.pos_truncated = 0
        self.pref_valid = 0
        self.pref_selected = 0
        self.closed_selected = 0
        self.slots = 0
        self.history_displaced = 0.0
        self.candidate_sum = 0
        self.selected_sum = 0
        self.prefix_mass_sum = 0.0
        self.prefix_mass_n = 0
        self.selected_rows = 0

    def snapshot(self):
        d = dict(self.__dict__)
        d['age_p95'] = list(self.age_p95)
        return d

    def append_from_layer(self, geo, T, topk_idx, sel_valid, cfg, prefix_mass=None, closed_blocks=0):
        if geo is None or T <= 0:
            return
        try:
            t = torch.arange(T, device=geo['offset'].device)
            age = geo['offset'] + 1
            w = int(getattr(cfg, 'sliding_window', 0) or 0)
            # W_t = [max(0, t-w+1), t]; w<=0 means no window, so the whole
            # prefix lies outside it.  U_t length = |[s_t, window_start-1]|.
            w_start = (t - w + 1).clamp_min(0) if w > 0 else (t + 1)
            outside_len = (w_start - geo['start_of_token']).clamp_min(0)
            cap = int(cfg.position_capacity())
            pos_trunc = (geo['offset'] >= cap)
            pref_valid = (geo['valid'] if geo['valid'].ndim == 2 else (geo['plen'][:, None] > 0))
            k = int(topk_idx.shape[1])
            is_pref = topk_idx.long() >= int(closed_blocks)
            pref_sel = (sel_valid & is_pref).sum(1)
            closed_sel = (sel_valid & ~is_pref).sum(1)
            # A closed-only top-k would fill min(k, readable_closed) slots.
            ends = geo['ends']
            readable_closed = torch.searchsorted(ends, t, right=True)
            displaced = (torch.minimum(torch.as_tensor(k, device=t.device),
                                       readable_closed) - closed_sel).clamp_min(0)
            self.n_tok += int(T)
            self.n_seq += 1
            self.age_sum += float(age.sum().item())
            self.age_max = max(self.age_max, int(age.max().item()))
            # Ages are exact integers; int32 halves the retained diagnostic
            # memory versus the int64 device dtype (the summary casts to
            # float32 only when computing the quantiles).
            self.age_p95.append(age.detach().to(torch.int32).cpu())
            self.outside_len_sum += float(outside_len.sum().item())
            self.outside_nonzero += int((outside_len > 0).sum().item())
            self.prefix_len_sum += float(geo['plen'].sum().item())
            self.pos_truncated += int(pos_trunc.sum().item())
            self.pref_valid += int(pref_valid.sum().item())
            self.pref_selected += int(pref_sel.sum().item())
            self.closed_selected += int(closed_sel.sum().item())
            self.slots += int(sel_valid.numel())
            self.candidate_sum += int(int(closed_blocks) + int(geo['token_idx'].shape[1])) * int(T)
            self.selected_sum += int(pref_sel.sum().item()) + int(closed_sel.sum().item())
            self.history_displaced += float(displaced.sum().item())
            self.selected_rows += int((pref_sel > 0).sum().item())
            if prefix_mass is not None:
                pm = prefix_mass.detach().float()
                self.prefix_mass_sum += float(pm.sum().item())
                self.prefix_mass_n += int(pm.numel())
        except Exception:
            # Diagnostics must never take down a training/eval run.
            return

    def summary(self, seq_len=None):
        if self.n_tok <= 0:
            return None
        try:
            flat = torch.cat([x.reshape(-1) for x in self.age_p95]) if self.age_p95 else torch.empty(0)
            p95 = float(torch.quantile(flat.float(), 0.95).item()) if flat.numel() else None
            p50 = float(torch.quantile(flat.float(), 0.50).item()) if flat.numel() else None
        except Exception:
            p95 = p50 = None
        return {
            'n_tokens': int(self.n_tok),
            'n_sequences': int(self.n_seq),
            'prefix_age_mean': self.age_sum / self.n_tok,
            'prefix_age_p50': p50,
            'prefix_age_p95': p95,
            'prefix_age_max': int(self.age_max),
            'outside_window_len_mean': self.outside_len_sum / self.n_tok,
            'outside_window_nonzero_frac': self.outside_nonzero / self.n_tok,
            'prefix_len_mean': self.prefix_len_sum / self.n_tok,
            'position_truncated_frac': self.pos_truncated / self.n_tok,
            'prefix_valid_candidates_mean': self.pref_valid / self.n_tok,
            'prefix_selected_rate': self.pref_selected / max(self.slots, 1),
            'prefix_selected_any_rate': self.selected_rows / self.n_tok,
            'closed_selected_mean': self.closed_selected / self.n_tok,
            'selected_mean': self.selected_sum / self.n_tok,
            'candidates_mean': self.candidate_sum / self.n_tok,
            'unused_slots_mean': (self.slots - self.selected_sum) / self.n_tok,
            'history_displaced_mean': self.history_displaced / self.n_tok,
            'prefix_attention_mass_mean': (self.prefix_mass_sum / self.prefix_mass_n) if self.prefix_mass_n else None,
            'slots': int(self.slots),
        }


def prefix_stats_summary(attn, seq_len=None):
    st = getattr(attn, '_prefix_stats', None)
    if isinstance(st, _PrefixStats):
        return st.summary(seq_len)
    return None

class _BlockStats:
    """Incremental replacement for the per-sequence bincount list.

    compression_report only needs the pooled mean/std/min/max/fractions over
    block lengths.  Each append folds the integer counts into running moments
    and drops the tensor immediately; values are accumulated in float64 so the
    pooled statistics are at least as accurate as the concatenated float32
    form."""

    def __init__(self, min_block=None, max_block=None, position_capacity=None):
        self.min_block = min_block
        self.max_block = max_block
        self.position_capacity = position_capacity
        self.pos_truncated_tokens = 0
        self.pos_truncated_blocks = 0
        self.forced_only = 0
        self.natural_only = 0
        self.forced_and_natural = 0
        self.forced_total = 0
        self.natural_total = 0
        self._lens = []
        self.n = 0
        self.total = 0
        self.blocks = 0
        self.s = 0.0
        self.ss = 0.0
        self.lo = None
        self.hi = None
        self.at_min = 0
        self.at_max = 0
        # GPU counts are queued until a snapshot is requested.
        self._pending = []

    def snapshot(self):
        d = dict(self.__dict__)
        d['_pending'] = list(self._pending)
        d['_lens'] = list(self._lens)
        return d

    def _add(self, a, n_items=1):
        if not a.size:
            return
        a = a.astype(np.float64, copy=False)
        self.n += int(n_items)
        self.blocks += int(a.size)
        self.total += int(a.size)
        self.s += float(a.sum())
        self.ss += float(np.square(a).sum())
        self._lens.append(a.copy())
        self.lo = float(a.min()) if self.lo is None else min(self.lo, float(a.min()))
        self.hi = float(a.max()) if self.hi is None else max(self.hi, float(a.max()))
        if self.min_block is not None:
            self.at_min += int((a <= self.min_block).sum())
        if self.max_block is not None:
            self.at_max += int((a >= self.max_block).sum())
        if self.position_capacity is not None:
            _cap = int(self.position_capacity)
            # A block of length L uses offsets 0..L-1; truncation starts at
            # offset == capacity, i.e. only when L > capacity.
            self.pos_truncated_tokens += int(np.maximum(a - _cap, 0).sum())
            self.pos_truncated_blocks += int((a > _cap).sum())

    def add_segmentation(self, bid, gate_cuts, max_block):
        """Count forced vs natural (and both) non-tail block ends.

        `gate_cuts` is the segmenter's boolean cut row (length n-1).  A block
        end `end < n` is natural when a cut was wanted at `end-1`; it is
        forced when `max_block` is set and the block length equals it.  The
        two categories can overlap.  Only the main dynamic path supplies a
        cut row; callers that cannot distinguish the cause leave these
        counters untouched rather than guessing.
        """
        if gate_cuts is None or bid is None:
            return
        try:
            b = bid.detach() if torch.is_tensor(bid) else torch.as_tensor(bid)
            B = int(b.max().item()) + 1 if b.numel() else 0
            if B <= 0:
                return
            counts = torch.bincount(b.reshape(-1).long(), minlength=B)[:B]
            ends = torch.cumsum(counts, 0)
            g = gate_cuts
            if not torch.is_tensor(g):
                g = torch.as_tensor(g)
            g = g.reshape(-1).to(torch.bool)
            n = int(b.numel())
            non_tail = ends < n
            if int(ends.max().item()) > 0:
                idx = (ends - 1).clamp(0, max(n - 2, 0))
                natural = ((ends - 1) < max(n - 1, 0)) & (idx < g.numel())
                nat = torch.zeros_like(non_tail)
                if g.numel():
                    nat = g[idx] & natural
                nat = nat & non_tail
            else:
                nat = torch.zeros_like(non_tail)
            if max_block is None:
                forced = torch.zeros_like(non_tail)
            else:
                forced = (counts == int(max_block)) & non_tail
            self.forced_only += int((forced & ~nat).sum().item())
            self.natural_only += int((nat & ~forced).sum().item())
            self.forced_and_natural += int((forced & nat).sum().item())
            self.forced_total += int(forced.sum().item())
            self.natural_total += int(nat.sum().item())
        except Exception:
            return

    def append(self, t):
        if torch.is_tensor(t) and t.device.type != 'cpu':
            self._pending.append(t.detach())
            self.n += 1
            return
        if torch.is_tensor(t):
            t = t.detach()
        self._add(np.asarray(t).reshape(-1))

    def _flush(self):
        if not self._pending:
            return
        parts = self._pending
        self._pending = []
        try:
            flat = torch.cat([p.reshape(-1).to(torch.int64) for p in parts])
        except Exception:
            for p in parts:
                self._add(np.asarray(p.detach().cpu()).reshape(-1), n_items=0)
            return
        # `append` already counted each pending sequence in self.n.
        self._add(flat.detach().cpu().numpy(), n_items=0)
        del flat

    def __len__(self):
        self._flush()
        return self.n

    def summary(self, seq_len):
        self._flush()
        if self.n == 0 or self.total == 0:
            return None
        mean = self.s / self.total
        # Match the concatenated form's sample standard deviation
        # (torch.std's default Bessel correction); with a single observation
        # the original reports 0.0.
        if self.total > 1:
            var = max((self.ss - self.s * self.s / self.total) / (self.total - 1), 0.0)
        else:
            var = 0.0
        return {
            'blocks': self.blocks / self.n,
            'avg_len': seq_len / max(self.blocks / self.n, 1.0),
            'len_mean': mean,
            'len_std': math.sqrt(var),
            'len_min': int(self.lo),
            'len_max': int(self.hi),
            'frac_at_min': self.at_min / self.total,
            'frac_at_max': (self.at_max / self.total if self.max_block is not None else None),
            'position_capacity': (int(self.position_capacity) if self.position_capacity is not None else None),
            'position_truncated_tokens': int(self.pos_truncated_tokens),
            'position_truncated_blocks': int(self.pos_truncated_blocks),
            'position_truncated_frac_of_block_tokens': (self.pos_truncated_tokens / max(self.s or 1.0, 1.0)),
            'forced_only': int(self.forced_only),
            'natural_only': int(self.natural_only),
            'forced_and_natural': int(self.forced_and_natural),
            'forced_total': int(self.forced_total),
            'natural_total': int(self.natural_total),
            'len_p50': (float(np.percentile(np.concatenate(self._lens), 50)) if self._lens else mean),
            'len_p95': (float(np.percentile(np.concatenate(self._lens), 95)) if self._lens else mean),
        }

def enable_block_stats(model, on):
    for blk in model.blocks:
        if on:
            _mb = getattr(blk.attn.cfg, 'max_block', None)
            blk.attn._stats = _BlockStats(getattr(blk.attn.cfg, 'min_block', None),
                                          (None if _mb is None else int(_mb)),
                                          position_capacity=getattr(blk.attn.cfg, 'position_capacity', lambda: None)())
            blk.attn._prefix_stats = _PrefixStats()
        else:
            blk.attn._stats = None
            blk.attn._prefix_stats = None
        blk.attn._cuts = [] if on else None


def _snapshot_block_stats(model):
    """Capture every attention layer's block-count / cut state.

    A failed forward may already have appended per-layer counts and cut
    positions.  OOM retries must restore this state together with the NLL
    accumulators, otherwise the partially processed rows are counted twice."""
    snap = []
    for blk in model.blocks:
        attn = blk.attn
        st = getattr(attn, '_stats', None)
        pst = getattr(attn, '_prefix_stats', None)
        cuts = getattr(attn, '_cuts', None)
        snap.append((attn,
                     st.snapshot() if isinstance(st, _BlockStats) else None,
                     list(cuts) if cuts is not None else None,
                     pst.snapshot() if isinstance(pst, _PrefixStats) else None))
    return snap


def _restore_block_stats(snap):
    for attn, state, cuts, pstate in snap:
        if state is not None:
            attn._stats.__dict__.clear()
            attn._stats.__dict__.update(state)
        if cuts is not None:
            attn._cuts[:] = cuts
        if pstate is not None and isinstance(getattr(attn, '_prefix_stats', None), _PrefixStats):
            attn._prefix_stats.__dict__.clear()
            attn._prefix_stats.__dict__.update(pstate)

# Random-cut baseline has two distinct sampling definitions.  They have
# different estimands (a fixed number of cuts per sequence vs a per-position
# independent cut process with a varying count), so their precisions must
# never be averaged, paired or plotted as one series.
BND_RAND_KINDS = ('matched_cut_count', 'per_sequence_independent')
_BND_RAND_ALIASES = {
    'matched_cut_count': 'matched_cut_count',
    'fixed_k': 'matched_cut_count',
    'fixed_block_count': 'matched_cut_count',
    'per_sequence_independent': 'per_sequence_independent',
    'independent': 'per_sequence_independent',
    'independent_cuts': 'per_sequence_independent',
}


def normalize_bnd_rand_kind(kind):
    if kind is None:
        return 'matched_cut_count'
    key = str(kind).strip().lower()
    if key not in _BND_RAND_ALIASES:
        raise ValueError(
            f'unknown bnd_rand kind {kind!r}; expected one of '
            f'{BND_RAND_KINDS} (aliases: fixed_k / independent).  The two '
            'definitions have different estimands and must not be mixed.')
    return _BND_RAND_ALIASES[key]


def boundary_alignment(pred_cuts, gt_mask_row, n, tol=1, provenance=None,
                       detail=None, rand_kind='matched_cut_count'):
    rand_kind = normalize_bnd_rand_kind(rand_kind)
    gt = np.nonzero(np.asarray(gt_mask_row)[:n - 1])[0] + 1
    pred = np.asarray(pred_cuts)
    P, G = (len(pred), len(gt))
    if P == 0 or G == 0:
        if provenance is not None:
            provenance.setdefault('bnd_rand_exact', True)
            provenance.setdefault('bnd_rand_kind', rand_kind)
            provenance.setdefault('bnd_rand_definition',
                                  'fixed_block_count' if rand_kind == 'matched_cut_count'
                                  else 'per_sequence_independent')
        if detail is not None:
            detail.update({'tp': 0, 'fp': P, 'fn': G, 'n_pred': P, 'n_gt': G})
        return (0.0, 0.0, 0.0, 0.0)
    hits, ghits = _greedy_match_counts(pred, gt, tol)
    if detail is not None:
        detail.update({'tp': int(hits), 'fp': int(P - hits), 'fn': int(G - ghits),
                       'n_pred': int(P), 'n_gt': int(G)})
    prec, rec = (hits / P, ghits / G)
    f1 = 2 * prec * rec / max(prec + rec, 1e-09)
    n_pos = max(n - 1, 1)
    _approx = []
    rand_prec = _random_cut_precision(P, None, n_pos, tol, G, gt,
                                      approx_flag=_approx, kind=rand_kind)
    if provenance is not None:
        provenance['bnd_rand_exact'] = provenance.get('bnd_rand_exact', True) and (not _approx)
        provenance.setdefault('bnd_rand_kind', rand_kind)
        provenance.setdefault('bnd_rand_definition',
                              'fixed_block_count' if rand_kind == 'matched_cut_count'
                              else 'per_sequence_independent')
    return (prec, rec, f1, rand_prec)

def _mask_runs(near, n_pos, tol):
    near_arr = np.asarray(near)
    m = near_arr[1:n_pos + 1].astype(bool)
    if not m.any():
        return np.zeros(0, dtype=int)
    if tol == 0:
        # With tol=0 the neighbourhood of a boundary is the boundary itself,
        # so every True lane is exactly one boundary -- including adjacent
        # boundaries, which form a longer run and cannot be split by the
        # run-length rules below (those are only meaningful for tol>0).
        return np.nonzero(m)[0] + 1
    d = np.diff(np.concatenate([[0], m.view(np.int8), [0]]))
    starts = np.nonzero(d == 1)[0]
    ends = np.nonzero(d == -1)[0]
    lens = ends - starts
    left_touch = bool(near_arr[0]) if near_arr.size else False
    out, ok = ([], True)
    for s, e, L in zip(starts, ends, lens):
        if s == 0 and e == m.size:
            # A run covering the whole sequence can come from one boundary at
            # the centre OR from several boundaries whose neighbourhoods tile
            # it; the post-slice mask cannot distinguish them.
            ok = False
            break
        if s == 0:
            if left_touch:
                # Position 0 is inside the neighbourhood, so the leftmost
                # possible boundary is 1; only a length-(tol+1) run pins it.
                if L == tol + 1:
                    out.append(0)
                else:
                    ok = False
                    break
            else:
                # Position 0 is outside every neighbourhood: the unique
                # minimum boundary is tol+1, and only its full 2*tol+1 run is
                # unambiguous.
                if L == 2 * tol + 1:
                    out.append(tol)
                else:
                    ok = False
                    break
        elif e == m.size:
            # The right edge is clipped by the sequence end: only the
            # one-sided length-(tol+1) run pins the final boundary.
            if L == tol + 1:
                out.append(m.size - 1)
            else:
                ok = False
                break
        else:
            # A fully interior neighbourhood has exactly 2*tol+1 lanes.
            if L == 2 * tol + 1:
                out.append((s + e - 1) // 2)
            else:
                ok = False
                break
    if ok:
        return np.asarray(sorted(set(out)), dtype=int) + 1
    return None

def _random_cut_precision(P, near, n_pos, tol, G, gt_bounds=None, approx_flag=None,
                          kind='matched_cut_count'):
    kind = normalize_bnd_rand_kind(kind)
    if P <= 0 or G <= 0 or n_pos <= 0:
        return 0.0
    if gt_bounds is None:
        if near is None:
            print('[stats] WARNING: `_random_cut_precision` got neither a `near` mask nor `gt_bounds`; the baseline is UNCOMPUTABLE for this call (returning 0.0) rather than being scored against an inflated reference set.  Pass `gt_bounds`.')
            print(f'[stats] bnd_rand kind for this call: {kind!r}; this value must not be pooled with the other definition.')
            return 0.0
        _runs = _mask_runs(near, n_pos, tol)
        if _runs is None:
            print('[stats] WARNING: `_random_cut_precision` got a `near` mask it cannot decompose into boundary neighbourhoods and no `gt_bounds`; the baseline is UNCOMPUTABLE for this call (returning 0.0) rather than being scored against an inflated reference set.  Pass `gt_bounds`.')
            return 0.0
        gt = _runs
    else:
        gt = np.asarray(gt_bounds).astype(int)
        gt = gt[(gt >= 1) & (gt <= n_pos)]
    if gt.size == 0:
        return 0.0
    gt = np.unique(gt)
    clusters, cur = ([], [int(gt[0])])
    for g in gt[1:]:
        g = int(g)
        if g - cur[-1] <= 2 * tol + 1:
            cur.append(g)
        else:
            clusters.append(cur)
            cur = [g]
    clusters.append(cur)
    if kind == 'per_sequence_independent':
        # Each eligible slot is cut independently with p so that the EXPECTED
        # number of cuts is P.  The count is therefore random and the estimand
        # differs from the matched-count baseline above; callers must keep the
        # two kinds separate.  The closed form is the expected number of GT
        # boundaries whose neighbourhood contains at least one cut, divided by
        # the expected number of cuts.  It is a coverage probability, not the
        # exact greedy-match expectation, so mark it as sampled/non-exact for
        # pooling purposes.
        if approx_flag is not None:
            approx_flag.append(('per_sequence_independent', BND_RAND_KINDS[0]))
        p = min(1.0, float(P) / float(n_pos))
        expected = 0.0
        for cl in clusters:
            lo, hi = (max(min(cl) - tol, 1), min(max(cl) + tol, n_pos))
            w = max(1, hi - lo + 1)
            expected += len(cl) * (1.0 - (1.0 - p) ** w)
        # p is clamped at 1 when P > n_pos; the expected number of cuts is
        # then n_pos, not P.  Dividing by P would understate the coverage
        # probability of the definition the caller asked for.  For the normal
        # P <= n_pos case keep the historical denominator (bit-identical to
        # the pre-existing formula rather than merely mathematically equal).
        expected_cuts = float(P) if P <= n_pos else float(n_pos)
        return float(min(expected / max(expected_cuts, 1e-12), 1.0))
    if P >= n_pos:
        return float(min(gt.size, P) / P)
    expect_frac = 0.0
    for cl in clusters:
        lo, hi = (max(min(cl) - tol, 1), min(max(cl) + tol, n_pos))
        win = list(range(lo, hi + 1))
        expect_frac += _cluster_expected_matches(win, cl, tol, P, n_pos, approx_flag)
    return float(expect_frac / P)

def _log_ways_ratio(w, r, n_pos, P):
    a, b, c = (n_pos - w, P - r, P)
    if r < 0 or a < 0 or b < 0 or (c < 0) or (r > w) or (b > a):
        return float('-inf')
    return _lgamma_comb(w, r) + _lgamma_comb(a, b) - _lgamma_comb(n_pos, c)

def _lgamma_comb(n, k):
    if k < 0 or k > n:
        return float('-inf')
    if k == 0 or k == n:
        return 0.0
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
_AMP_WORK_CAP = (1 << 12) - 1
_AMP_FALLBACK_DRAWS = 2048

def _cluster_expected_matches(win, cl, tol, P, n_pos, approx_flag=None):
    w = len(win)
    if w <= 0 or P <= 0:
        return 0.0
    cl_t = tuple((int(b) for b in cl))
    A = [int(p) for p in win if any((abs(int(p) - b) <= tol for b in cl_t))]
    a = len(A)
    if a == 0:
        return 0.0
    rmax = min(P, a)
    _cl_arr = np.asarray(cl_t)
    _ord = np.argsort(_cl_arr, kind='stable')
    _pre = (_cl_arr[_ord], _ord)
    T = [0] * (rmax + 1)
    _comp_T = _component_match_tables(A, cl_t, tol, rmax, _pre)
    a_max = max((a_c for _tbl, a_c in _comp_T.values()), default=0)
    any_sampled = any((tbl == [] for tbl, _a in _comp_T.values()))
    if not any_sampled:
        T = _convolve_component_T(_comp_T, rmax)
    else:
        _work = _subset_work(a_max, rmax)
        if approx_flag is not None:
            approx_flag.append((a_max, _AMP_WORK_CAP, _work))
        print(f'[stats] WARNING: `_random_cut_precision` is APPROXIMATING: a single candidate component holds {a_max} positions and needs {_work} subset scorings, past the enumeration limit of {_AMP_WORK_CAP}, so `bnd_rand` (and hence `bnd_excess`) for this sample is a Monte-Carlo estimate ({_AMP_FALLBACK_DRAWS} draws), NOT the exact probability the other panels report.  Record it as `bnd_rand_exact: False` and do not pool it with the exact readings.')
        seed = (int(n_pos) * 1000003 + int(P) * 10007 + int(tol) * 101 + int(a) * 17 + cl_t[0]) % 2 ** 32
        rng = np.random.default_rng(seed)
        wts = []
        for r in range(1, rmax + 1):
            outside = P - r
            if outside < 0 or outside > n_pos - a:
                continue
            lw = _log_ways_ratio(a, r, n_pos, P)
            if lw == float('-inf'):
                continue
            wts.append((r, lw))
        if not wts:
            return 0.0
        nb_r = len(wts)
        order = sorted(range(nb_r), key=lambda i: -wts[i][1])
        if nb_r > _AMP_FALLBACK_DRAWS:
            # More candidate sizes than the draw limit: keep the top-weight
            # sizes, one draw each, so the actual number of draws never
            # exceeds the declared _AMP_FALLBACK_DRAWS limit.
            _keep = set(order[:_AMP_FALLBACK_DRAWS])
            draws = [1 if i in _keep else 0 for i in range(nb_r)]
        else:
            per = max(1, _AMP_FALLBACK_DRAWS // nb_r)
            draws = [per] * nb_r
            for i in range(_AMP_FALLBACK_DRAWS - per * nb_r):
                draws[order[i % nb_r]] += 1
        total = 0.0
        for i, (r, lw) in enumerate(wts):
            n_draw = draws[i]
            if n_draw <= 0:
                continue
            subs = [_sample_r_subset(rng, A, r) for _ in range(n_draw)]
            s = _score_subsets_components(subs, _comp_T, cl_t, tol, _pre)
            total += math.exp(lw) * (s / n_draw)
        return float(total)
    total = 0.0
    for r in range(0, rmax + 1):
        outside = P - r
        if outside < 0 or outside > n_pos - a:
            continue
        if not r:
            continue
        if not (comb(a, r) and comb(n_pos - a, outside)):
            continue
        mean_r = T[r] / comb(a, r)
        if mean_r == 0.0:
            continue
        total += math.exp(_log_ways_ratio(a, r, n_pos, P)) * mean_r
    return float(total)

def _match_components(A, cl_t, tol):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = (root, parent[x])
        return root

    def union(x, y):
        rx, ry = (find(x), find(y))
        if rx != ry:
            parent[rx] = ry
    if A:
        A_arr = np.asarray(A)
        for gi, b in enumerate(cl_t):
            lo = int(np.searchsorted(A_arr, b - tol, side='left'))
            hi = int(np.searchsorted(A_arr, b + tol, side='right'))
            for pi in range(lo, hi):
                union(('c', pi), ('g', gi))
    comps = {}
    for pi, p in enumerate(A):
        comps.setdefault(find(('c', pi)), ([], []))[0].append(int(p))
    for gi, b in enumerate(cl_t):
        comps.setdefault(find(('g', gi)), ([], []))[1].append(int(b))
    out = []
    for key in sorted(comps, key=repr):
        pos, bnd = comps[key]
        if pos:
            out.append((pos, bnd))
    return out

def _sorted_pre(b_t):
    if not b_t:
        z = np.zeros(0, dtype=int)
        return (z, z)
    arr = np.asarray(b_t)
    o = np.argsort(arr, kind='stable')
    return (arr[o], o)

def _subset_work(w, rmax):
    if w <= 0 or rmax <= 0:
        return 0
    r_hi = min(rmax, w)
    total = 0
    c = 1
    for r in range(0, r_hi + 1):
        if r:
            c = c * (w - r + 1) // r
        if r:
            total += c
    return total

def _component_match_tables(A, cl_t, tol, rmax, _pre):
    tables = {}
    for pos, bnd in _match_components(A, cl_t, tol):
        r_c = min(rmax, len(pos))
        b_t = tuple(bnd)
        c_pre = _sorted_pre(b_t)
        if _subset_work(len(pos), r_c) <= _AMP_WORK_CAP:
            tbl = [0] * (r_c + 1)
            for rr in range(1, r_c + 1):
                subs = list(_subsets(pos, rr))
                if subs:
                    tbl[rr] = int(sum(_greedy_match_count_batch(subs, c_pre[0], c_pre[1], tol)))
            tables[tuple(pos), b_t] = (tbl, len(pos))
        else:
            tables[tuple(pos), b_t] = ([], len(pos))
    return tables

def _convolve_component_T(tables, rmax):
    items = [(tbl, a_c) for tbl, a_c in tables.values() if tbl]
    if not items:
        return [0] * (rmax + 1)
    choice = [0.0] * (rmax + 1)
    choice[0] = 1.0
    wsum = [0.0] * (rmax + 1)
    for tbl, a_c in items:
        rc_max = min(rmax, a_c)
        term = [0.0] * (rmax + 1)
        mean = [0.0] * (rmax + 1)
        for r in range(0, rc_max + 1):
            term[r] = float(comb(a_c, r))
            if r:
                mean[r] = tbl[r] / comb(a_c, r)
        new_choice = [0.0] * (rmax + 1)
        new_wsum = [0.0] * (rmax + 1)
        for i in range(rmax + 1):
            if choice[i] == 0.0 and wsum[i] == 0.0:
                continue
            for r in range(0, rc_max + 1):
                if i + r > rmax or term[r] == 0.0:
                    continue
                new_choice[i + r] += choice[i] * term[r]
                new_wsum[i + r] += wsum[i] * term[r] + choice[i] * term[r] * mean[r]
        choice, wsum = (new_choice, new_wsum)
    return [int(round(wsum[r])) for r in range(rmax + 1)]

def _sample_r_subset(rng, A, r):
    return tuple(np.sort(rng.choice(A, size=r, replace=False)))

def _score_subsets_components(subs, tables, cl_t, tol, _pre):
    _owner_cache = getattr(_score_subsets_components, '_owner_cache', None)
    if _owner_cache is None:
        _owner_cache = _score_subsets_components._owner_cache = {}
    if len(_owner_cache) >= 64 and id(tables) not in _owner_cache:
        _owner_cache.clear()
    cached = _owner_cache.get(id(tables))
    if cached is None:
        owner = {}
        for pos, bnd in tables:
            b_t = tuple(bnd)
            c_pre = _sorted_pre(b_t)
            for p in pos:
                owner[p] = (b_t, c_pre)
        cached = (owner, tables)
        _owner_cache[id(tables)] = cached
    owner = cached[0]
    by_comp = {}
    for si, sub in enumerate(subs):
        for p in sub:
            hit = owner.get(p)
            if hit is None:
                continue
            b_t, c_pre = hit
            per_sub = by_comp.get(b_t)
            if per_sub is None:
                per_sub = by_comp[b_t] = (c_pre, {})
            pts = per_sub[1].get(si)
            if pts is None:
                per_sub[1][si] = [p]
            else:
                pts.append(p)
    total = 0
    for _b_t, (c_pre, per_sub) in by_comp.items():
        sids = sorted(per_sub)
        pieces = [tuple(sorted(per_sub[si])) for si in sids]
        for v in _greedy_match_count_batch(pieces, c_pre[0], c_pre[1], tol):
            total += v
    return total

def _subsets(items, r):
    return itertools.combinations(items, r)

def _greedy_match_counts(pred, gt, tol, _pre=None):
    pred = np.sort(np.asarray(pred))
    P, G = (len(pred), len(gt))
    if P == 0 or G == 0:
        return (0, 0)
    if _pre is not None:
        gt_sorted, order = _pre
    else:
        gt_arr = np.asarray(gt)
        order = np.argsort(gt_arr, kind='stable')
        gt_sorted = gt_arr[order]
    lo_all = np.searchsorted(gt_sorted, np.asarray(pred) - tol, side='left')
    hi_all = np.searchsorted(gt_sorted, np.asarray(pred) + tol, side='right')
    cand = []
    for pi in range(P):
        p = pred[pi]
        for j in range(int(lo_all[pi]), int(hi_all[pi])):
            gi = int(order[j])
            cand.append((abs(p - gt[gi]), pi, gi))
    cand.sort()
    used_p, used_g = (set(), set())
    for _, pi, gi in cand:
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
    return (len(used_p), len(used_g))

def _greedy_match_count_batch(subsets, gt_sorted, order, tol):
    # `_greedy_match_counts` sorts the predicate before its rank tie-break, so
    # the batched form must canonicalise every subset the same way.  Using the
    # caller's order made the match count depend on how a subset happened to
    # be constructed (its rank, not its value, broke distance ties).
    subs = [tuple(sorted(s)) for s in subsets]
    n_sub = len(subs)
    if n_sub == 0:
        return []
    counts = np.zeros(n_sub, dtype=np.int64)
    sizes = np.fromiter((len(s) for s in subs), dtype=np.int64, count=n_sub)
    if sizes.sum() == 0:
        return counts.tolist()
    flat = np.fromiter((int(p) for s in subs for p in s), dtype=np.int64, count=int(sizes.sum()))
    cut_rank = np.concatenate([np.arange(int(k), dtype=np.int64) for k in sizes]) if n_sub else np.zeros(0, dtype=np.int64)
    owner = np.repeat(np.arange(n_sub, dtype=np.int64), sizes)
    lo = np.searchsorted(gt_sorted, flat - tol, side='left')
    hi = np.searchsorted(gt_sorted, flat + tol, side='right')
    cnt = (hi - lo).astype(np.int64)
    total = int(cnt.sum())
    if total:
        row_owner = np.repeat(owner, cnt)
        starts = np.repeat(lo, cnt)
        offs = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        bnd_idx = starts + offs
        gi = order[bnd_idx]
        pi_local = np.repeat(flat, cnt)
        pi_rank = np.repeat(cut_rank, cnt)
        dist = np.abs(pi_local - gt_sorted[bnd_idx])
        order_rows = np.lexsort((gi, pi_rank, dist, row_owner))
        used_p, used_g = (set(), set())
        prev_sid = -1
        for idx in order_rows:
            sid = int(row_owner[idx])
            if sid != prev_sid:
                if prev_sid >= 0:
                    counts[prev_sid] = len(used_p)
                used_p, used_g = (set(), set())
                prev_sid = sid
            p_i = int(pi_rank[idx])
            g_i = int(gi[idx])
            if p_i not in used_p and g_i not in used_g:
                used_p.add(p_i)
                used_g.add(g_i)
        if prev_sid >= 0:
            counts[prev_sid] = len(used_p)
    return counts.tolist()

def compression_report(model, val_batch, device, n_sample=64, val_bnd=None,
                       bnd_tol=1, bnd_rand_kind='matched_cut_count'):
    bnd_rand_kind = normalize_bnd_rand_kind(bnd_rand_kind)
    was_training = model.training
    model.eval()
    enable_block_stats(model, True)
    need_reg_layers = [blk.attn for blk in model.blocks if hasattr(blk.attn, 'need_reg')]
    need_reg_prior = [_a.need_reg for _a in need_reg_layers]
    for _a in need_reg_layers:
        _a.need_reg = False
    try:
        return _compression_report_impl(model, val_batch, device, n_sample,
                                        val_bnd, bnd_tol,
                                        bnd_rand_kind=bnd_rand_kind)
    finally:
        for _a, _prev in zip(need_reg_layers, need_reg_prior):
            _a.need_reg = _prev
        enable_block_stats(model, False)
        model.train(was_training)

def _compression_report_summary(model, seq_len, val_bnd=None, bnd_tol=1,
                                bnd_rand_kind='matched_cut_count'):
    bnd_rand_kind = normalize_bnd_rand_kind(bnd_rand_kind)
    report = {}
    # every dynamic layer's delta_logit (same scalars, same floats).
    _dl_layers = [(li, blk.attn.delta_logit) for li, blk in enumerate(model.blocks)
                  if blk.attn.cfg.dynamic and getattr(blk.attn, 'delta_logit', None) is not None]
    _dl_vals = {}
    if _dl_layers:
        _dl_vals = {li: float(v) for (li, _), v in zip(_dl_layers, torch.stack([p.detach() for _, p in _dl_layers]).tolist())}
    for li, blk in enumerate(model.blocks):
        attn = blk.attn
        st = attn._stats
        kind = attn.cfg.kind
        if st is None or kind == 'full' or len(st) == 0:
            continue
        if isinstance(st, _BlockStats):
            entry = st.summary(seq_len)
            if entry is None:
                continue
        else:
            lens = torch.cat([t if torch.is_tensor(t) else torch.as_tensor(t) for t in st]).float()
            n_blocks = sum((t.numel() for t in st)) / len(st)
            _mb = getattr(attn.cfg, 'max_block', None)
            _frac_max = (float('nan') if _mb is None else (lens >= int(_mb)).float().mean())
            _sc = torch.stack([lens.mean(), lens.std() if lens.numel() > 1 else torch.zeros((), device=lens.device, dtype=lens.dtype), lens.min().float(), lens.max().float(), (lens <= attn.cfg.min_block).float().mean(), torch.as_tensor(_frac_max, device=lens.device, dtype=lens.dtype)])
            _vals = _sc.tolist()
            entry = {'blocks': n_blocks, 'avg_len': seq_len / max(n_blocks, 1.0), 'len_mean': _vals[0], 'len_std': _vals[1], 'len_min': int(_vals[2]), 'len_max': int(_vals[3]), 'frac_at_min': _vals[4], 'frac_at_max': _vals[5]}
            del _sc, _vals
        _ps = prefix_stats_summary(attn, seq_len)
        if getattr(attn.cfg, 'dynamic', False) or getattr(attn.cfg, 'chunking', 'fixed') != 'fixed':
            # The chunking DATA path (similarity signal / strict cut rule /
            # geometry bounds) must be comparable across the arms in a
            # dynamic-vs-fixed or ablation contrast; record it per layer.
            entry['chunking_protocol'] = chunking_protocol(attn.cfg)
        _ob_cfg = {
            'open_block_mode': getattr(attn.cfg, 'open_block_mode', 'closed'),
            'open_block_scope': getattr(attn.cfg, 'open_block_scope', 'full_prefix'),
            'selection_budget': getattr(attn.cfg, 'selection_budget', 'shared_topk'),
            'max_block': getattr(attn.cfg, 'max_block', None),
            'max_block_tokens': getattr(attn.cfg, 'max_block', None),
            'max_block_effective': (int(getattr(attn.cfg, 'max_block')) if getattr(attn.cfg, 'max_block', None) is not None else seq_len),
            'pool_position_capacity': int(getattr(attn.cfg, 'position_capacity', lambda: 0)()),
            'pool_position_policy': getattr(attn.cfg, 'pool_position_policy', 'truncate'),
        }
        if _ob_cfg['open_block_mode'] != 'closed':
            entry['open_block_config'] = _ob_cfg
            if _ps is not None:
                entry['open_block'] = _ps
        if attn.cfg.dynamic and getattr(attn, 'delta_logit', None) is not None:
            entry['delta'] = _dl_vals[li]
        if val_bnd is not None and attn._cuts:
            n_seq_ref = min(len(attn._cuts), val_bnd.shape[0])
            if n_seq_ref < len(attn._cuts):
                print(f'[compression_report] WARNING: {len(attn._cuts)} sequences of cuts but only {val_bnd.shape[0]} reference masks — scoring the first {n_seq_ref} only')
            ps, rs, fs, rps = ([], [], [], [])
            _tp = _fp = _fn = _n_pred = _n_gt = 0
            _cuts = attn._cuts
            _prov = {}
            # The integers transferred are exactly the per-sequence values
            # (index transport, design notes §6.1), segmented on the host.
            _tcut_ix = [si for si in range(n_seq_ref) if torch.is_tensor(_cuts[si])]
            _tcut_host = {}
            if _tcut_ix:
                _tlens = [int(_cuts[si].numel()) for si in _tcut_ix]
                _tflat = torch.cat([_cuts[si].reshape(-1) for si in _tcut_ix]).cpu().numpy()
                _toff = 0
                for si, _tln in zip(_tcut_ix, _tlens):
                    _tcut_host[si] = _tflat[_toff:_toff + _tln]
                    _toff += _tln
                del _tflat
            for si in range(n_seq_ref):
                c = _tcut_host.get(si) if si in _tcut_host else _cuts[si]
                _d = {}
                p, r_, f_, rp = boundary_alignment(
                    c, val_bnd[si], seq_len, tol=bnd_tol,
                    provenance=_prov, detail=_d,
                    rand_kind=bnd_rand_kind)
                ps.append(p)
                rs.append(r_)
                fs.append(f_)
                rps.append(rp)
                _tp += int(_d.get('tp', 0))
                _fp += int(_d.get('fp', 0))
                _fn += int(_d.get('fn', 0))
                _n_pred += int(_d.get('n_pred', 0))
                _n_gt += int(_d.get('n_gt', 0))
            if n_seq_ref:
                entry['bnd_prec'] = float(np.mean(np.asarray(ps)))
                entry['bnd_rec'] = float(np.mean(np.asarray(rs)))
                entry['bnd_f1'] = float(np.mean(np.asarray(fs)))
                entry['bnd_f1_average'] = 'macro_over_sequences'
                entry['bnd_prec_micro'] = float(_tp / _n_pred) if _n_pred else 0.0
                entry['bnd_rec_micro'] = float(_tp / _n_gt) if _n_gt else 0.0
                _pm, _rm = entry['bnd_prec_micro'], entry['bnd_rec_micro']
                entry['bnd_f1_micro'] = (float(2 * _pm * _rm / (_pm + _rm))
                                         if (_pm + _rm) else 0.0)
                entry['bnd_tp'] = int(_tp)
                entry['bnd_fp'] = int(_fp)
                entry['bnd_fn'] = int(_fn)
                entry['bnd_rand'] = float(np.mean(np.asarray(rps)))
                entry['bnd_excess'] = entry['bnd_prec'] - entry['bnd_rand']
                entry['bnd_rand_exact'] = bool(_prov.get('bnd_rand_exact', True))
                entry['bnd_rand_kind'] = str(_prov.get('bnd_rand_kind') or bnd_rand_kind)
                entry['bnd_rand_definition'] = str(
                    _prov.get('bnd_rand_definition')
                    or ('fixed_block_count' if entry['bnd_rand_kind'] == 'matched_cut_count'
                        else 'per_sequence_independent'))
                if not entry['bnd_rand_exact']:
                    print(f'[compression_report] WARNING: `bnd_rand` for L{li}_{kind} is a SAMPLED estimate, not the exact expectation, so `bnd_excess` carries Monte-Carlo error — recorded as `bnd_rand_exact: False` and it must not be pooled with the exact panels.')
            else:
                print('[compression_report] WARNING: no reference sequence to score (n_seq_ref=0) — boundary metrics OMITTED rather than published as nan')
            entry['bnd_n_seq'] = n_seq_ref
        tag = 'dyn' if attn.cfg.dynamic else 'fix'
        report[f'L{li}_{kind}_{tag}'] = entry
    return report


@torch.inference_mode()
def _compression_report_impl(model, val_batch, device, n_sample, val_bnd, bnd_tol,
                             *, bnd_rand_kind='matched_cut_count'):
    report = {}
    seq_len = val_batch.shape[1]
    if int(n_sample) < 1:
        raise ValueError(f'compression_report: n_sample must be >= 1, got {n_sample!r}; the report is an average over the sequences it scores, so there is nothing to measure with no sequence scored')
    n_used = min(n_sample, val_batch.shape[0])
    _row_chunk = _eval_adaptive_chunk(model, np.asarray(val_batch), device,
                                      min(16, _EVAL_TOKENS_PER_CHUNK // max(seq_len, 1)))
    _row_chunk = max(1, min(_row_chunk, n_used))
    # inference_mode (the decorator) already covers the forward pass.  The
    # sampled sequences are independent, so the row grouping does not change
    # any value.
    _r0 = 0
    while _r0 < n_used:
        _cur = min(_row_chunk, n_used - _r0)
        # A failed forward may already have appended per-layer block counts
        # (the stats hook runs layer by layer).  Snapshot them so the retry
        # cannot count the partially-processed rows twice.
        _stats_snap = _snapshot_block_stats(model)
        # as_tensor accepts numpy arrays AND torch tensors (eval_ppl already
        # uses it).
        ids = None
        try:
            ids = torch.as_tensor(np.asarray(val_batch[_r0:_r0 + _cur])).to(device)
            _ = model(ids, logits_tail=1)
        except RuntimeError as _oe:
            del ids
            if ('out of memory' in str(_oe).lower()) and _cur > 1:
                _restore_block_stats(_stats_snap)
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                _row_chunk = max(1, _cur // 2)
                print(f'[compression_report] device-memory exhaustion; retrying with row_chunk={_row_chunk}')
                continue
            raise
        del ids
        _r0 += _cur
    return _compression_report_summary(model, int(seq_len), val_bnd, bnd_tol,
                                        bnd_rand_kind=bnd_rand_kind)

def is_no_decay(name):
    if any((k in name for k in ('delta_logit', 'B_pos', 'W_aZ', 'W_bZ', 'sink', 'tok.weight', 'pos.weight', 'fuse_conv', 'q_norm', 'kv_norm'))):
        return True
    return name == 'g' or name.endswith('.g')

def is_delta_param(name):
    return 'delta_logit' in name

# ---- 受控变量描述符：优化器 / 调度 / 裁剪 / 数据增强 ----
# 对比实验要求「除待验证模块外，优化器、学习率策略、训练轮数、数据增强等设置
# 必须完全一致」。这些设置是 train_variant 内的模块常量，对所有臂天然一致，
# 但 run_cfg 指纹并不显式记录它们——一旦扩规模时为某条臂单独改优化器/调度
# （或将来引入数据增强），两条记录仍可能通过 `pair_reason` 的同 run_cfg 门槛
# 而被静默配对。这里把同一组常量导出成 `training_knobs()` 描述符，随每条记录
# 落盘；`pair_reason` 对两条都带该字段的记录额外要求逐字段相等（缺字段的记录
# 不受影响），审计则把该描述符打印出来「证明」两臂共享优化器与调度，而不是
# 只靠断言。
OPT_ADAMW_BETAS = (0.9, 0.999)
OPT_ADAMW_EPS = 1e-08
GRAD_CLIP_NORM = 1.0
LR_FLOOR_MULT = 0.1
BOUNDARY_PG_COEF = 1.0
LR_COSINE_AMP = 0.45

def optimizer_param_groups(model):
    """Split trainable parameters exactly as the production trainer does."""
    decay = [p for n_, p in model.named_parameters()
             if p.requires_grad and (not is_no_decay(n_)) and (not is_delta_param(n_))]
    no_decay = [p for n_, p in model.named_parameters()
                if p.requires_grad and is_no_decay(n_) and (not is_delta_param(n_))]
    delta = [p for n_, p in model.named_parameters()
             if p.requires_grad and is_delta_param(n_)]
    return decay, no_decay, delta

def build_optimizer(model, *, lr, weight_decay, delta_lr_mult=10.0):
    """Build the production AdamW optimizer (three groups, foreach=True).

    The overfit/reproducibility pre-flights use this function too, so a
    parameter-grouping change can no longer pass a hand-rolled probe while
    breaking the real training loop.
    """
    decay, no_decay, delta = optimizer_param_groups(model)
    groups = [
        {'params': decay, 'weight_decay': float(weight_decay), 'lr_scale': 1.0},
        {'params': no_decay, 'weight_decay': 0.0, 'lr_scale': 1.0},
        {'params': delta, 'weight_decay': 0.0, 'lr_scale': float(delta_lr_mult)},
    ]
    return torch.optim.AdamW(groups, lr=float(lr), betas=OPT_ADAMW_BETAS,
                             eps=OPT_ADAMW_EPS, foreach=True)

def training_knobs(*, amp_dtype='auto', eager_batch='auto'):
    """Canonical, JSON-safe descriptor of the optimisation / schedule /
    gradient-clipping / augmentation / precision settings shared by every arm
    of a controlled panel.

    ``amp_dtype`` and ``eager_batch`` default to the mode that `train_variant`
    will actually use on this machine.  Callers that implement their own
    training loop (for example v7's fp32 warm-up loop) pass their concrete
    values so the record describes what ran instead of what the default would
    have been.  Keeping both fields in this descriptor makes them enter the
    resume fingerprint (`training_knobs_fingerprint`) and the per-record
    pairing gate, so an old fp32 / row-by-row cell can never be silently
    reused by a bf16 / batched-twin run.
    """
    if amp_dtype == 'auto':
        amp_dtype = _active_amp()
    if eager_batch == 'auto':
        eager_batch = _eager_batch_active()
    return {
        'optimizer': 'AdamW',
        'foreach': True,
        'betas': list(OPT_ADAMW_BETAS),
        'eps': OPT_ADAMW_EPS,
        'param_groups': 'decay (wd) | no-decay (wd=0) | delta_logit (wd=0, lr x delta_lr_mult)',
        'grad_clip_norm': GRAD_CLIP_NORM,
        'lr_schedule': 'linear_warmup_then_cosine',
        'lr_floor_mult': LR_FLOOR_MULT,
        'lr_cosine_amp': LR_COSINE_AMP,
        'augmentation': 'none',
        'amp_dtype': amp_dtype,
        'amp_grad_scaler': amp_dtype == 'fp16',
        'eager_batch': bool(eager_batch),
    }

def training_knobs_fingerprint(*, amp_dtype='auto', eager_batch='auto'):
    """Short, stable digest of the shared optimisation/training descriptor.

    `run_cfg` already pins lr / wd / warmup / steps / tokens / architecture /
    data, but it deliberately does not carry the optimizer identity, betas,
    eps, LR-schedule shape, grad-clip, augmentation or the numerical
    implementation mode (AMP dtype / batched eager twin).  Those live in the
    per-record `train_knobs` copy, which `pair_reason` checks at analysis
    time.  Putting this digest in the resume fingerprint makes a changed
    descriptor invalidate earlier cells just like a changed lr.
    Unset/legacy fingerprints simply decode without a `tk` field.
    """
    if amp_dtype == 'auto' and eager_batch == 'auto':
        _knobs = training_knobs()
    else:
        _knobs = training_knobs(amp_dtype=amp_dtype, eager_batch=eager_batch)
    blob = json.dumps(_knobs, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:12]

_CKPT_PEAK_MEMO = {}


def _ckpt_geom_key(n_param, n_layers, d, mlp_ratio, batch_size, seq_len, cfgs, n_heads=8, d_head=32):
    kinds = ','.join(str(getattr(c, 'kind', 'full')) for c in (cfgs or []))
    return (hex(int(n_param)), int(n_layers), int(d), round(float(mlp_ratio), 4),
            int(batch_size), int(seq_len), int(n_heads), int(d_head), kinds)


def _auto_ckpt_mode(device, n_param, n_layers, d, mlp_ratio, batch_size, seq_len, cfgs, n_heads=8, d_head=32):
    # Explicit CSA_CKPT always wins.  Otherwise, on a CUDA box, fall back to
    # activation recomputation when the estimated working set is large.  The
    # recomputed forward is the same pure function, so values and gradients
    # are unchanged (the contract documented at the top).
    if os.environ.get('CSA_CKPT', '').strip():
        return None, None
    if device.type != 'cuda' or not torch.cuda.is_available():
        return None, None
    try:
        free, _total = torch.cuda.mem_get_info(device)
    except Exception:
        return None, None
    tok = max(1, int(batch_size)) * max(1, int(seq_len))
    hidden = max(1, int(round(int(d) * float(mlp_ratio))))
    act = int(n_layers) * tok * (hidden + 8 * int(d)) * 4
    seq = max(1, int(seq_len))
    batch = max(1, int(batch_size))
    extra = 0
    nh = max(1, int(n_heads))
    hd = max(1, int(d_head))
    for c in cfgs:
        kind = getattr(c, 'kind', 'full')
        if kind in ('csa', 'hca'):
            step = max(1, int(getattr(c, 'block_size', 0) or getattr(c, 'target_block_tokens', 0) or getattr(c, 'min_block', 1) or 1))
            Bm = (seq + step - 1) // step
            nih = max(1, int(getattr(c, 'n_index_heads', 1) or 1))
            topk = max(1, int(getattr(c, 'index_topk', 1) or 1))
            win = max(0, int(getattr(c, 'sliding_window', 0) or 0))
            nb_eff = Bm if kind == 'hca' else min(topk, Bm)
            M = nb_eff + win
            index_bytes = seq * Bm * nih * 4
            gather_bytes = (4 * M * nh * hd + 2 * M * nh) * 4
            extra += batch * max(index_bytes, gather_bytes)
            if kind == 'hca':
                extra += batch * seq * Bm * 4
        elif kind == 'full':
            win = max(0, int(getattr(c, 'window', 0) or 0))
            if getattr(c, 'full_sink', False):
                span = seq
            elif 0 < win < seq:
                span = win
            else:
                span = min(seq, 1024)
            extra += batch * 3 * nh * seq * span * 4
    need = act + int(extra) + 16 * int(n_param)
    _peak = _CKPT_PEAK_MEMO.get(_ckpt_geom_key(n_param, n_layers, d, mlp_ratio, batch_size, seq_len, cfgs, n_heads=n_heads, d_head=d_head))
    if _peak:
        need = max(need, int(_peak))
    if free >= need + need // 3:
        return None, need
    has_gate = any(getattr(c, 'chunking', '') == 'cosine_learnable' for c in cfgs)
    # Block-level recompute cannot cover the gate-mean side channel; those
    # arms use the MLP-level fallback.
    mode = 'mlp' if (has_gate or free >= (need * 3) // 4) else 'block'
    return mode, need

_CURVE_MAX_POINTS = 200000
_CURVE_SIDECAR_INLINE_PPL = 4096


def _file_sha256(path, *, chunk=1 << 20):
    """SHA-256 of an artifact file, or None when it cannot be read.

    Tracking currently records a *path* for a curve sidecar.  A path alone
    cannot distinguish the curve written by this cell from a later edit or a
    file copied in from another run.  Storing the digest turns the reference
    into an auditable pointer; the verifier can then recompute it.
    """
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            while True:
                b = f.read(int(chunk))
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except (OSError, TypeError, ValueError):
        return None


def _curve_sidecar_rel(key):
    safe = ''.join(ch if (ch.isalnum() or ch in '._:-') else '_' for ch in str(key))
    return f'curves/{safe}.json'


def _write_cell_curve(outdir, key, rec):
    """Move per-cell curve arrays out of the shared summary file.

    The summary is rewritten (and re-parsed) under the panel lock after every
    cell; embedding a 200k-point loss curve makes that work grow with
    cells x curve length.  Each cell's arrays go to its own sidecar and the
    summary keeps only scalar metadata plus a reference."""
    losses = rec.get('losses') if isinstance(rec.get('losses'), list) else []
    delta = rec.get('delta_trace') if isinstance(rec.get('delta_trace'), dict) else {}
    ph = rec.get('ppl_history') if isinstance(rec.get('ppl_history'), list) else []
    diag = rec.get('opt_diag') if isinstance(rec.get('opt_diag'), dict) else {}
    move_ph = len(ph) > _CURVE_SIDECAR_INLINE_PPL
    payload = {}
    if losses:
        payload['losses'] = losses
    if delta:
        payload['delta_trace'] = delta
    if move_ph:
        payload['ppl_history'] = ph
    if diag:
        payload['opt_diag'] = diag
    if not payload:
        return None, losses, ph, move_ph
    rel = _curve_sidecar_rel(key)
    try:
        atomic_write_json(os.path.join(outdir, rel), payload, indent=0)
    except Exception as _e:
        print(f'[track] WARNING: could not write curve sidecar {rel} ({type(_e).__name__}: {_e}); keeping this cell\'s curves inline')
        return None, losses, ph, False
    if diag:
        rec.pop('opt_diag', None)
        rec['opt_diag_ref'] = rel
        rec['opt_diag_n_points'] = len(diag.get('step') or [])
    return rel, losses, ph, move_ph


def _cell_curve_data(outdir, rec):
    ref = rec.get('curve_ref') if isinstance(rec, dict) else None
    if not ref:
        return {}
    try:
        with open(os.path.join(outdir, ref), encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _reduce_curve_values(vals, max_points):
    """Downsample a loss curve while preserving spikes and non-finite points.

    A plain ``linspace`` stride can drop a one-step NaN/explosion between two
    retained samples, making a diverged long run look clean in the artifact.
    Each bin therefore keeps its first/last index, its finite min/max and the
    first non-finite index.  The result is sorted and bounded by
    ``max_points`` (7 slots per bin).
    """
    n = len(vals)
    max_points = int(max_points)
    if max_points < 1:
        # A zero-point budget cannot retain any sample.  Returning the
        # historical fixed set would silently exceed the caller's bound.
        return []
    if n <= max_points:
        return list(vals)
    nb = max(1, max_points // 7)
    first = [-1] * nb
    last = [-1] * nb
    mini = [-1] * nb
    maxi = [-1] * nb
    badi = [-1] * nb
    minv = [0.0] * nb
    maxv = [0.0] * nb
    for i, v in enumerate(vals):
        b = min(nb - 1, (i * nb) // n)
        if first[b] < 0:
            first[b] = i
        last[b] = i
        try:
            f = float(v)
        except (TypeError, ValueError, OverflowError):
            f = None
        if f is None or not math.isfinite(f):
            if badi[b] < 0:
                badi[b] = i
            continue
        if mini[b] < 0 or f < minv[b]:
            mini[b] = i
            minv[b] = f
        if maxi[b] < 0 or f > maxv[b]:
            maxi[b] = i
            maxv[b] = f
    keep = set()
    for i in range(nb):
        for _ix in (first[i], mini[i], maxi[i], badi[i], last[i]):
            if _ix >= 0:
                keep.add(int(_ix))
    if n:
        keep.add(n - 1)
    _keep = sorted(keep)
    if len(_keep) > max_points:
        # The bin heuristic keeps up to five landmarks per bin, which is more
        # than a very small budget allows.  Keep the last point and the first
        # non-finite/extreme points when they fit, then fill chronologically.
        _bad = _mini = _maxi = None
        _minv = _maxv = None
        for _i, _v in enumerate(vals):
            try:
                _f = float(_v)
            except (TypeError, ValueError, OverflowError):
                _f = None
            if _f is None or not math.isfinite(_f):
                if _bad is None:
                    _bad = _i
                continue
            if _mini is None or _f < _minv:
                _mini, _minv = _i, _f
            if _maxi is None or _f > _maxv:
                _maxi, _maxv = _i, _f
        _sel = []
        for _ix in (n - 1, _bad, _mini, _maxi, 0):
            if _ix is not None and 0 <= _ix < n and _ix not in _sel:
                _sel.append(int(_ix))
                if len(_sel) >= max_points:
                    break
        if len(_sel) < max_points:
            for _ix in _keep:
                if _ix not in _sel:
                    _sel.append(_ix)
                    if len(_sel) >= max_points:
                        break
        _keep = sorted(_sel)
    return [vals[i] for i in _keep]


def _curve_extend(dst, new):
    dst.extend(new)
    if len(dst) > 2 * _CURVE_MAX_POINTS:
        dst[:] = _reduce_curve_values(dst, _CURVE_MAX_POINTS)


def _reduce_curve_points(points, max_points):
    """Downsample ``[step, value]`` trace points without losing spikes/NaN.

    ``_reduce_curve_values`` scores plain scalars; a per-layer delta trace stores
    point pairs, so the spike test must look at the value column.  Bins keep
    their first/last index, finite min/max and first non-finite index.
    """
    n = len(points)
    max_points = int(max_points)
    if max_points < 1:
        return []
    if n <= max_points:
        return list(points)
    nb = max(1, max_points // 7)

    def _value(point):
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            return point[1]
        return point

    first = [-1] * nb
    last = [-1] * nb
    mini = [-1] * nb
    maxi = [-1] * nb
    badi = [-1] * nb
    minv = [0.0] * nb
    maxv = [0.0] * nb
    for i, point in enumerate(points):
        b = min(nb - 1, (i * nb) // n)
        if first[b] < 0:
            first[b] = i
        last[b] = i
        try:
            f = float(_value(point))
        except (TypeError, ValueError, OverflowError):
            f = None
        if f is None or not math.isfinite(f):
            if badi[b] < 0:
                badi[b] = i
            continue
        if mini[b] < 0 or f < minv[b]:
            mini[b] = i
            minv[b] = f
        if maxi[b] < 0 or f > maxv[b]:
            maxi[b] = i
            maxv[b] = f
    keep = set()
    for i in range(nb):
        for _ix in (first[i], mini[i], maxi[i], badi[i], last[i]):
            if _ix >= 0:
                keep.add(int(_ix))
    if n:
        keep.add(n - 1)
    _keep = sorted(keep)
    if len(_keep) > max_points:
        _bad = _mini = _maxi = None
        _minv = _maxv = None
        for _i, _point in enumerate(points):
            try:
                _f = float(_value(_point))
            except (TypeError, ValueError, OverflowError):
                _f = None
            if _f is None or not math.isfinite(_f):
                if _bad is None:
                    _bad = _i
                continue
            if _mini is None or _f < _minv:
                _mini, _minv = _i, _f
            if _maxi is None or _f > _maxv:
                _maxi, _maxv = _i, _f
        _sel = []
        for _ix in (n - 1, _bad, _mini, _maxi, 0):
            if _ix is not None and 0 <= _ix < n and _ix not in _sel:
                _sel.append(int(_ix))
                if len(_sel) >= max_points:
                    break
        if len(_sel) < max_points:
            for _ix in _keep:
                if _ix not in _sel:
                    _sel.append(_ix)
                    if len(_sel) >= max_points:
                        break
        _keep = sorted(_sel)
    return [points[i] for i in _keep]


_LEDGER_CURVE_MAX_POINTS = 20000
