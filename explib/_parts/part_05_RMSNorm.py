
class RMSNorm(nn.Module):

    def __init__(self, d, eps=1e-06):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, self.g.shape, weight=self.g, eps=self.eps)

class MLP(nn.Module):

    def __init__(self, d, hidden, init_gen=None):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d, bias=False)
        # The shared-panel design expects identical MLP weights across variants;
        # drawing them from `init_gen` keeps that true no matter how many RNG
        # draws the attention branch consumed beforehand.
        if init_gen is not None:
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5), generator=init_gen)
                nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5), generator=init_gen)

    def forward(self, x):
        if _CKPT['on'] in ('mlp', 'block') and _CKPT_NEST[0] == 0 and torch.is_grad_enabled() and x.requires_grad:
            # 非重入 checkpoint（模块顶 CSA_CKPT 说明）：重算同一纯函数，
            # 前向值与梯度逐位一致。_CKPT_NEST > 0 时外层块已被 checkpoint
            # 覆盖，内层不再包裹。
            return torch.utils.checkpoint.checkpoint(self._fwd, x, use_reentrant=False, preserve_rng_state=False)
        return self._fwd(x)

    def _fwd(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class Block(nn.Module):

    def __init__(self, d, n_heads, d_head, cfg: AttnCfg, mlp_ratio=4, mlp_gen=None):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.attn = HybridAttention(d, n_heads, d_head, cfg)
        self.n2 = RMSNorm(d)
        self.mlp = MLP(d, int(round(d * mlp_ratio)), init_gen=mlp_gen)

    def forward(self, x):
        if _CKPT['on'] == 'block' and torch.is_grad_enabled() and x.requires_grad:
            if self.attn.cfg.chunking != 'cosine_learnable':
                return torch.utils.checkpoint.checkpoint(self._fwd_ckpt, x, use_reentrant=False, preserve_rng_state=False)
            if 'cosine' not in _CKPT_WARNED:
                _CKPT_WARNED.add('cosine')
                print('[CSA_CKPT] block 级重算不覆盖 cosine_learnable 臂（gate_mean 经属性逃逸出块，初向前向的 no_grad 会剥掉它的图——v9 _V9_CKPT 的同一限制）；这些臂回退到 mlp 级，其余臂整块重算')
        return self._fwd(x)

    def _fwd(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x

    def _fwd_ckpt(self, x):
        # checkpoint 的初向前向与反向重算都经这里：置位期间内层 MLP 不再
        # 重复包裹。
        _CKPT_NEST[0] += 1
        try:
            return self._fwd(x)
        finally:
            _CKPT_NEST[0] -= 1

class SmallGPT(nn.Module):

    def __init__(self, vocab, d, n_layers, n_heads, d_head, max_seq, layer_cfgs, mlp_ratio=4):
        super().__init__()
        _rng_state = torch.get_rng_state()
        _cuda_rng_state = None
        if torch.cuda.is_available():
            try:
                _cuda_rng_state = torch.cuda.get_rng_state_all()
            except Exception:
                _cuda_rng_state = None
        shared = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
        g = torch.Generator().manual_seed(shared)
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_seq, d)
        self.head = nn.Linear(d, vocab, bias=False)
        with torch.no_grad():
            self.tok.weight.normal_(0, 0.02, generator=g)
            self.pos.weight.normal_(0, 0.02, generator=g)
            self.head.weight.normal_(0, 0.02, generator=g)
        self.blocks = nn.ModuleList()
        for i, c in enumerate(layer_cfgs):
            torch.manual_seed(shared + 104729 * (i + 1))
            mlp_g = torch.Generator().manual_seed(shared + 1299709 * (i + 1))
            self.blocks.append(self._make_block(d, n_heads, d_head, c, mlp_ratio, mlp_g))
        # Per-block init temporarily reseeds the GLOBAL torch RNG above; restore
        # the caller's state so model construction never perturbs data shuffling /
        # dropout / later model construction (which would otherwise couple the
        # data order to n_layers and break paired-experiment isolation).
        # `torch.manual_seed` also reseeds every CUDA device, so the CPU state
        # alone is not enough on a GPU box: restore the CUDA streams too.
        torch.set_rng_state(_rng_state)
        if _cuda_rng_state is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(_cuda_rng_state)
            except Exception:
                pass
        self.norm = RMSNorm(d)
        self.max_seq = max_seq
        self.boundary_logp = None
        self.boundary_decisions = 0

    def _make_block(self, d, n_heads, d_head, cfg, mlp_ratio, mlp_gen=None):
        return Block(d, n_heads, d_head, cfg, mlp_ratio, mlp_gen=mlp_gen)

    def forward(self, ids, logits_tail=None, targets=None, vocab=None):
        # When `targets` is supplied this returns the training CE loss instead
        # of logits.  The DDP wrapper only proxies `forward`, so routing the
        # loss through it is what lets DDP install its reducer hooks; calling
        # `ce_loss` directly on the inner module would silently skip the
        # gradient all-reduce.
        if targets is not None:
            if vocab is None:
                raise ValueError('forward(targets=...) requires vocab')
            return self.ce_loss(ids, targets, vocab)
        x = self.trunk(ids)
        if logits_tail:
            x = x[:, -logits_tail:]
        return self.head(x)

    def trunk(self, ids):
        # Full forward minus the LM head; sets self.comp_reg exactly as forward
        # did (the reg terms are accumulated inside the single block loop — the
        # same addition order as the full forward).
        T = ids.shape[1]
        _max_pos = int(self.pos.weight.shape[0])
        if T > _max_pos:
            raise RuntimeError(
                f'absolute position table covers {_max_pos} tokens but the input has {T}; '
                'a dense absolute-position arm cannot evaluate beyond its training length. '
                'Use a RoPE arm, or pass an explicit max_pos so the evaluation span is '
                'truncated and recorded as truncated=True (never silently score a shorter span).')
        x = self.tok(ids) + self.pos.weight[:T]
        reg = None
        blp = None
        bdec = 0
        for blk in self.blocks:
            x = blk(x)
            gm = blk.attn.last_gate_mean
            if gm is not None:
                Lb = max(blk.attn.cfg.target_block_tokens, 1)
                mult = getattr(blk.attn.cfg, 'comp_lambda_mult', 1.0)
                if isinstance(gm, (list, tuple)):
                    tot = gm[0]
                    for _g in gm[1:]:
                        tot = tot + _g
                    gm = tot / len(gm)
                term = mult * (gm * Lb - 1.0) ** 2
                reg = term if reg is None else reg + term
            _lp = getattr(blk.attn, 'last_boundary_logp', None)
            if _lp is not None:
                blp = _lp if blp is None else blp + _lp
                bdec += int(getattr(blk.attn, 'last_boundary_decisions', 0) or 0)
        if reg is None:
            reg = torch.zeros((), device=ids.device)
        self.comp_reg = reg
        self.boundary_logp = blp
        self.boundary_decisions = int(bdec)
        return self.norm(x)

    def _ce_head_chunk(self, h_chunk, tgt_chunk, vocab, ignore_index=-100):
        lg = self.head(h_chunk)
        return F.cross_entropy(lg.reshape(-1, vocab), tgt_chunk.reshape(-1), reduction='sum', ignore_index=ignore_index)

    def ce_loss(self, ids, targets, vocab, chunk=64, ignore_index=None):
        # LM-head cross-entropy computed in time chunks: the chunk sums add to
        # the same total as the full-batch form (up to fp reduction order), so
        # the scalar and its gradients are unchanged.
        # 训练时每个 chunk 包一层非重入 checkpoint：反向按 chunk 重算同一份
        # logits（同一 GEMM、无 RNG，值逐位相同）；chunk 反传的执行次序由
        # 求和链的拓扑结构决定，两种形式相同，因此梯度逐位一致。
        h = self.trunk(ids)
        # When targets contain ignored positions, the mean must be taken over
        # the valid targets only (F.cross_entropy's own reduction), exactly as
        # the unchunked form did.  The default None means every target is valid.
        _ign = -100 if ignore_index is None else int(ignore_index)
        # The trunk must see the full input (block geometry depends on T), but
        # only the first `n_cols` target columns are scored.  Counting the
        # denominator over ALL target columns made a wider target table shrink
        # the loss (and its gradient) by n_cols/targets.shape[1].
        n_cols = min(int(ids.shape[1]), int(targets.shape[1]))
        if n_cols <= 0:
            raise ValueError(
                'ce_loss needs at least one input position with a target; '
                f'got ids{tuple(ids.shape)} targets{tuple(targets.shape)}')
        n_tok = int((targets[:, :n_cols] != _ign).sum().item())
        n_tok = max(n_tok, 1)
        ce = None
        _ck = torch.is_grad_enabled() and h.requires_grad
        if _ck and h.device.type != 'cuda':
            # Every chunk's logits stay alive in the graph until backward, so
            # the limit covers the whole sequence, not just one chunk.
            _saved = 3 * int(ids.shape[0]) * int(ids.shape[1]) * int(vocab) * 4
            _ck = not _cpu_saved_ok(_saved, _CPU_CE_CKPT_BYTES)
        # A next-token target table is one column shorter; score only the
        # rows that have a target, exactly like the original logits[:, :-1]
        # form.  `n_cols` (and its denominator) is fixed above.
        for s in range(0, n_cols, chunk):
            e = min(s + chunk, n_cols)
            if _ck:
                c = torch.utils.checkpoint.checkpoint(self._ce_head_chunk, h[:, s:e], targets[:, s:e], vocab, _ign, use_reentrant=False, preserve_rng_state=False)
            else:
                c = self._ce_head_chunk(h[:, s:e], targets[:, s:e], vocab, _ign)
            ce = c if ce is None else ce + c
        return ce / n_tok
OVERLAP = 8
_WARM_KEY_RE = re.compile('.*::w(\\d+)::seed\\d+$')

# Open-block panel variant names.  The maximum block length and the reading
# rule are the two experimental factors; everything else is the frozen
# csa_dynamic recipe.  `_ow` is the auxiliary outside-window scope (U_t).
OPEN_BLOCK_MAX_VALUES = (32, 64, 128, None)
OPEN_BLOCK_MODES_SHORT = ('closed', 'summary', 'singletons')
_OPEN_VARIANT_RE = re.compile(r'^csa_dynamic_mb(32|64|128|none)_(closed|summary|singletons)(_ow)?$')


def open_block_variant_name(max_block, mode, scope='full_prefix'):
    """Canonical variant name for one cell of the open-block matrix."""
    if mode not in OPEN_BLOCK_MODES_SHORT:
        raise ValueError(f'unknown open-block mode {mode!r}; expected {OPEN_BLOCK_MODES_SHORT}')
    mb = 'none' if max_block is None else str(int(max_block))
    if max_block is not None and int(max_block) not in (32, 64, 128):
        raise ValueError(f'unsupported max_block {max_block!r} for the open-block variant names')
    if scope not in OPEN_BLOCK_SCOPES:
        raise ValueError(f'unknown scope {scope!r}')
    return f'csa_dynamic_mb{mb}_{mode}' + ('_ow' if scope == 'outside_window' else '')


def open_block_variants(max_values=OPEN_BLOCK_MAX_VALUES, modes=OPEN_BLOCK_MODES_SHORT,
                        scope='full_prefix'):
    return [open_block_variant_name(mb, mode, scope) for mb in max_values for mode in modes]


def _open_variant_cfg(variant):
    m = _OPEN_VARIANT_RE.match(str(variant))
    if m is None:
        return None
    mb = None if m.group(1) == 'none' else int(m.group(1))
    short = m.group(2)
    mode = {'closed': 'closed', 'summary': 'prefix_summary',
            'singletons': 'prefix_singletons'}[short]
    scope = 'outside_window' if m.group(3) else 'full_prefix'
    return AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable',
                   target_block_tokens=4, overlap=OVERLAP, index_topk=32,
                   temperature=0.1, min_block=2, max_block=mb,
                   pool_position_capacity=129, pool_position_policy='truncate',
                   open_block_mode=mode, open_block_scope=scope,
                   selection_budget='shared_topk')


def make_layer_cfgs(n_layers, variant):
    _open_cfg = _open_variant_cfg(variant)
    if _open_cfg is not None:
        return [copy.deepcopy(_open_cfg) for _ in range(int(n_layers))]
    cfgs = []
    for i in range(n_layers):
        if variant == 'full':
            cfgs.append(AttnCfg(kind='full'))
        elif variant == 'full_cos':
            cfgs.append(AttnCfg(kind='full', full_cosine=True))
        elif variant == 'full_matched':
            cfgs.append(AttnCfg(kind='full'))
        elif variant == 'full_sw128':
            cfgs.append(AttnCfg(kind='full', window=128))
        elif variant == 'full_sw128_matched':
            cfgs.append(AttnCfg(kind='full', window=128))
        elif variant == 'csa_fixed':
            cfgs.append(AttnCfg(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=32))
        elif variant == 'csa_dynamic':
            cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1))
        elif variant == 'hybrid_fixed':
            if i % 2 == 0:
                cfgs.append(AttnCfg(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=32))
            else:
                cfgs.append(AttnCfg(kind='hca', dynamic=False, block_size=64))
        elif variant == 'hybrid_dynamic':
            if i % 2 == 0:
                cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1))
            else:
                cfgs.append(AttnCfg(kind='hca', dynamic=True, chunking='cosine_learnable', target_block_tokens=64, temperature=0.1, comp_lambda_mult=4.0))
        elif variant == 'full_sink':
            cfgs.append(AttnCfg(kind='full', full_sink=True))
        elif variant == 'full_randomsink':
            cfgs.append(AttnCfg(kind='full', full_sink=True, sink_mode='random'))
        elif variant == 'hybrid_csa_dyn':
            if i % 2 == 0:
                cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1))
            else:
                cfgs.append(AttnCfg(kind='hca', dynamic=False, block_size=64))
        elif variant == 'hybrid_hca_dyn':
            if i % 2 == 0:
                cfgs.append(AttnCfg(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=32))
            else:
                cfgs.append(AttnCfg(kind='hca', dynamic=True, chunking='cosine_learnable', target_block_tokens=64, temperature=0.1, comp_lambda_mult=4.0))
        elif variant == 'csa_dyn_fuse':
            cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1, fuse_mode='learned'))
        elif variant == 'hybrid_csa_dyn_fuse':
            if i % 2 == 0:
                cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1, fuse_mode='learned'))
            else:
                cfgs.append(AttnCfg(kind='hca', dynamic=False, block_size=64))
        elif variant == 'csa_fix_randidx':
            cfgs.append(AttnCfg(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=32, indexer_mode='random'))
        elif variant == 'csa_fix_zerocont':
            cfgs.append(AttnCfg(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=32, content_mode='zero'))
        elif variant == 'csa_fix_shiftcont':
            cfgs.append(AttnCfg(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=32, content_mode='shift'))
        elif variant == 'csa_fix_nosink':
            cfgs.append(AttnCfg(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=32, use_sink=False))
        elif variant == 'csa_fix_randomsink':
            cfgs.append(AttnCfg(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=32, sink_mode='random'))
        elif variant == 'csa_fix_topk8':
            cfgs.append(AttnCfg(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=8))
        elif variant == 'csa_fix_topk64':
            cfgs.append(AttnCfg(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=64))
        elif variant == 'csa_dyn_randidx':
            cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1, indexer_mode='random'))
        elif variant == 'csa_dyn_zerocont':
            cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1, content_mode='zero'))
        elif variant == 'csa_dyn_shiftcont':
            cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1, content_mode='shift'))
        elif variant == 'csa_dyn_nosink':
            cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1, use_sink=False))
        elif variant == 'csa_dyn_randomsink':
            cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1, sink_mode='random'))
        elif variant == 'csa_dynamic_task':
            # Opt-in end-to-end boundary arm: identical to
            # csa_dynamic except the hard cuts are sampled and trained by the
            # LM loss through a score-function term.  This is a NEW arm; the
            # csa_dynamic remains threshold/comp_reg-only and must
            # not be described as task-learned.
            cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1, boundary_policy='reinforce'))
        elif variant == 'csa_dyn_nocomp':
            # Loss-term ablation C: identical architecture, optimizer, LR
            # schedule, scale and data as `csa_dynamic`; only the compression
            # regularizer's multiplier is zeroed.  Keeping run_cfg's `cl`
            # unchanged is what lets `pair_reason` pair the two arms while the
            # actual loss term differs — the thing under test.
            cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1, comp_lambda_mult=0.0))
        elif variant == 'csa_fixed_fullpov':
            cfgs.append(AttnCfg(kind='csa', dynamic=False, block_size=4, overlap=128, index_topk=32))
        elif variant == 'csa_dynamic_fullpov':
            cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=128, index_topk=32, temperature=0.1))
        else:
            raise ValueError(variant)
    return cfgs

def build_tokenizer(texts, vocab_size=8192):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers import decoders as _decoders
    tok = Tokenizer(BPE(unk_token='<unk>'))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=True)
    # The BPE is byte-level, so its inverse must be the byte-level decoder.
    # Without this, `decode` returns the raw BPE surface ("Ġthe") instead of
    # the original text: the persisted tokenizer is then not round-trippable,
    # `boundary_token_ids` misses every space-prefixed punctuation token
    # (Ġ. Ġ, Ġ: …), and no audit can map a token id back to the raw line it
    # came from.  The decoder is the exact inverse of the pre-tokenizer and
    # does not change `encode`/token ids by construction.
    tok.decoder = _decoders.ByteLevel()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=['<pad>', '<unk>', '<bos>', '<eos>'])
    tok.train_from_iterator(texts, trainer=trainer)
    return tok
PUNCT_CHARS = set('.,!?;:')

def boundary_token_ids(tok, eos_id):
    ids = set()
    _n = int(tok.get_vocab_size())
    try:
        # One batched decode instead of a Python-level call per vocabulary
        # entry (32k entries on the B-tier tokenizer).
        decoded = tok.decode_batch([[i] for i in range(_n)])
    except (AttributeError, TypeError, ValueError):
        decoded = [tok.decode([i]) for i in range(_n)]
    for i, text in enumerate(decoded):
        s = text.strip()
        if s and all((ch in PUNCT_CHARS for ch in s)):
            ids.add(i)
    ids.add(eos_id)
    return ids

class _LocalTextSplit:

    def __init__(self, texts):
        self._texts = texts

    def __len__(self):
        return len(self._texts)

    def select(self, idx):
        return _LocalTextSplit([self._texts[i] for i in idx])

    def __getitem__(self, key):
        if isinstance(key, slice):
            return {'text': self._texts[key]}
        if key == 'text':
            return self._texts
        raise TypeError(f'unsupported key {key!r}')

def local_wikitext_if_available(repo_cache_dir):
    raw_txt = os.environ.get('WT103_RAW_TXT')
    candidates = []
    if raw_txt:
        candidates.append(raw_txt)
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(repo_cache_dir)), 'wt103_raw'))
    candidates.append('/root/wt103_raw')
    for cand in candidates:
        train_f = os.path.join(cand, 'wiki.train.raw')
        valid_f = os.path.join(cand, 'wiki.valid.raw')
        if os.path.exists(train_f) and os.path.exists(valid_f):
            train = _read_raw_rows(train_f)
            valid = _read_raw_rows(valid_f)
            print(f'[data] local wikitext-103 raw dump at {cand} ({len(train)} train rows, {len(valid)} val rows)')
            return {'train': _LocalTextSplit(train), 'validation': _LocalTextSplit(valid)}
    return None

def _read_raw_rows(path):
    with open(path, encoding='utf-8') as fh:
        return [ln.rstrip('\n') for ln in fh]

_IDS_FP_MEMO = {}
_IDS_FP_MEMO_CAP = 4
# Bump when the canonical hashing form changes so an old on-disk sidecar is
# not trusted for a digest produced by a different algorithm.
_IDS_FP_ALGO = 'int64-canonical-v2'

def _array_file_path(arr):
    """Resolve the on-disk file backing a memmap (or a whole-file view of one).

    The huge FineWeb caches are served as read-only memmap views; a sidecar
    fingerprint next to that file lets every worker reuse the parent's digest,
    while remaining bound to the file's size+mtime so a rebuilt cache can never
    reuse a stale digest.

    Only the root logical memmap (the deepest memmap in arr's base chain) may
    supply the sidecar.  `arr` must then have that root's exact layout and data
    start.  A suffix slice also ends at EOF but starts elsewhere, so its bytes
    differ from the whole-file digest and must be hashed itself."""
    root = None
    cur = arr
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, np.memmap):
            root = cur
        cur = getattr(cur, 'base', None)
    if root is None:
        return None
    fn = getattr(root, 'filename', None)
    if not isinstance(fn, (str, bytes, os.PathLike)):
        return None
    try:
        fn = os.fspath(fn)
        st = os.stat(fn)
        off = int(getattr(root, 'offset', 0) or 0)
        nb = int(getattr(root, 'nbytes', -1))
        arr_nbytes = int(getattr(arr, 'nbytes', -1))
        root_ptr = root.__array_interface__['data'][0]
        arr_ptr = arr.__array_interface__['data'][0]
        same = (off >= 0 and nb >= 0 and off + nb == int(st.st_size)
                and arr_nbytes == nb
                and arr_ptr == root_ptr
                and tuple(getattr(arr, 'shape', ())) == tuple(root.shape)
                and getattr(arr, 'strides', None) == root.strides
                and np.dtype(arr.dtype) == np.dtype(root.dtype))
    except (OSError, TypeError, ValueError, KeyError, AttributeError):
        return None
    return fn if same else None

def _cache_file_sigs(paths):
    """size+mtime+ctime+inode signature for the files a cache entry is built from.

    size+mtime alone can be preserved when a file is replaced by a same-size
    copy (or copied with ``cp -p``).  ``ctime`` cannot be set by user code and
    the inode changes on replacement, so a stale/mixed file does not match its
    recorded signature; the content fingerprint sidecars
    (``*.idsfp.json``) are keyed on the same fields, so they are invalidated
    too.
    """
    out = {}
    for _k, _p in paths.items():
        try:
            _st = os.stat(_p)
            out[_k] = [int(_st.st_size), int(_st.st_mtime_ns),
                       int(getattr(_st, 'st_ctime_ns', int(_st.st_ctime * 1e9))),
                       int(getattr(_st, 'st_ino', 0))]
        except OSError:
            return None
    return out


def _cache_file_sigs_match(meta, sigs):
    rec = meta.get('file_sigs') if isinstance(meta, dict) else None
    return bool(sigs) and isinstance(rec, dict) and all(rec.get(_k) == _v for _k, _v in sigs.items())


def _read_cache_meta(path, label):
    """Parse cache metadata, returning ``None`` when it cannot be trusted.

    Every cache-hit path verifies content fingerprints, so a malformed meta
    must send the caller down the rebuild path rather than crash with an
    AttributeError halfway through verification (which would also leave a
    stale/incompletely verified cache in place).
    """
    try:
        with open(path, encoding='utf-8') as fh:
            obj = json.load(fh)
    except Exception as e:
        print(f'[data] WARNING: cached {label} metadata {path} cannot be parsed '
              f'({type(e).__name__}: {e}) — the cache is not verifiable')
        return None
    if not isinstance(obj, dict):
        print(f'[data] WARNING: cached {label} metadata {path} is a '
              f'{type(obj).__name__}, not a JSON object — the cache is not verifiable')
        return None
    return obj


def _memo_file_sig(paths):
    """In-process memo key component: includes ctime/inode, so replacing a file
    (even with preserved size/mtime) invalidates the memo entry."""
    s = _cache_file_sigs(paths)
    return tuple(tuple(v) for v in s.values()) if s else None


def _array_id_range(arr):
    a = np.asarray(arr)
    if a.size == 0:
        return None
    return [int(a.min()), int(a.max())]


def _ids_fp(train_ids):
    # Pure function of the token values, memoised on (data pointer, layout)
    # with a weak reference to the array: a dead entry can never alias a
    # recycled pointer, and large corpora are not retained by the cache.
    # Same tokens in -> same digest out, always.  Every integer width up to
    # 64 bit other than uint64 is hashed through its canonical int64 form, so
    # a corpus keeps the same fingerprint regardless of storage width (the
    # uint16 FineWeb cache, int32 views and int64 arrays all agree).
    #
    # The pointer+layout key identifies an ARRAY OBJECT, not its current
    # contents.  An in-place write (e.g. a cleaning pass over a validation
    # array) would otherwise leave the memo returning the pre-write digest,
    # and that stale digest then becomes a resume/data identity.  OWNDATA does
    # not help: a writable view of the same buffer is just as mutable.  So a
    # writable array is never trusted from the memo or from the on-disk
    # sidecar; it is hashed fresh every call.  The production corpora are
    # read-only memmaps (writeable=False), which keep the fast path.
    import weakref as _weakref
    _flags = getattr(train_ids, 'flags', None)

    def _readonly_chain(arr):
        # A read-only view can still alias writable storage (``base[:]`` with
        # ``writeable=False``), so trusting only the view's own flag left a
        # stale memo/sidecar digest after the base was mutated.  Walk the base
        # chain; memmaps expose an mmap (non-ndarray) base and stay on the
        # fast path.
        seen = 0
        cur = arr
        while isinstance(cur, np.ndarray):
            if cur.flags.writeable:
                return False
            seen += 1
            if seen > 64:  # defensive: a cycle is not immutable evidence
                return False
            nxt = getattr(cur, 'base', None)
            if nxt is cur:
                break
            cur = nxt
        return True

    _immutable = bool(_flags is not None and not _flags.writeable
                      and _readonly_chain(train_ids))
    try:
        _key = (train_ids.__array_interface__['data'][0], train_ids.shape, train_ids.strides, train_ids.dtype.str)
    except (AttributeError, TypeError):
        _key = None
    if _key is not None and _immutable:
        _hit = _IDS_FP_MEMO.get(_key)
        if _hit is not None:
            _arr = _hit[0]() if _hit[0] is not None else None
            if _arr is train_ids:
                return _hit[1]
            _IDS_FP_MEMO.pop(_key, None)
    _path = _array_file_path(train_ids)
    _stat = None
    _side = None
    if _path is not None and _immutable:
        try:
            _st = os.stat(_path)
            _stat = (int(getattr(_st, 'st_size', 0)),
                     int(getattr(_st, 'st_mtime_ns', int(_st.st_mtime * 1e9))),
                     int(getattr(_st, 'st_ctime_ns', int(_st.st_ctime * 1e9))),
                     int(getattr(_st, 'st_ino', 0)))
            _side = _path + '.idsfp.json'
            _tshape = tuple(int(_x) for _x in np.shape(train_ids))
            _tdtype = np.dtype(np.asarray(train_ids).dtype).str
            with open(_side, encoding='utf-8') as _sf:
                _rec = json.load(_sf)
            if not isinstance(_rec, dict):
                # A valid-JSON list/scalar is not an id-fingerprint sidecar;
                # ignore it and hash the array instead of crashing on .get().
                raise ValueError('fingerprint sidecar is not a JSON object')
            if ((int(_rec.get('size', -1)), int(_rec.get('mtime_ns', -1)),
                 int(_rec.get('ctime_ns', -1)), int(_rec.get('inode', -1))) == _stat
                    and isinstance(_rec.get('fp'), str)
                    and _rec.get('algo') == _IDS_FP_ALGO
                    and _rec.get('dtype') == _tdtype
                    and tuple(_rec.get('shape') or ()) == _tshape):
                _fp = _rec['fp']
                if _key is not None and _immutable:
                    try:
                        _ref = _weakref.ref(train_ids)
                    except TypeError:
                        _ref = None
                    if len(_IDS_FP_MEMO) >= _IDS_FP_MEMO_CAP and _key not in _IDS_FP_MEMO:
                        _IDS_FP_MEMO.pop(next(iter(_IDS_FP_MEMO)))
                    _IDS_FP_MEMO[_key] = (_ref, _fp)
                return _fp
        except (OSError, ValueError, TypeError):
            pass
    _h = hashlib.sha256()
    _arr = np.asarray(train_ids)
    _kind = _arr.dtype.kind
    # uint64 keeps its own width (values above int64_max have no canonical
    # int64 representation); every other integer width is widened to int64.
    _canon = (np.dtype(np.uint64) if (_kind == 'u' and _arr.dtype.itemsize == 8)
              else np.dtype(np.int64) if _kind in 'iu' else None)
    _cs = 1 << 24
    for _off in range(0, len(_arr), _cs):
        _part = _arr[_off:_off + _cs]
        if _canon is not None:
            _part = _part.astype(_canon, copy=False)
        if not _part.flags.c_contiguous:
            _part = np.ascontiguousarray(_part)
        _h.update(memoryview(_part).cast('B'))
    _fp = _h.hexdigest()[:16]
    if _key is not None and _immutable:
        try:
            _ref = _weakref.ref(train_ids)
        except TypeError:
            _ref = None
        if len(_IDS_FP_MEMO) >= _IDS_FP_MEMO_CAP and _key not in _IDS_FP_MEMO:
            _IDS_FP_MEMO.pop(next(iter(_IDS_FP_MEMO)))
        _IDS_FP_MEMO[_key] = (_ref, _fp)
    if _side is not None and _stat is not None and _immutable:
        try:
            _rec_out = {'size': _stat[0], 'mtime_ns': _stat[1],
                        'ctime_ns': _stat[2], 'inode': _stat[3], 'fp': _fp,
                        'algo': _IDS_FP_ALGO,
                        'dtype': np.dtype(np.asarray(train_ids).dtype).str,
                        'shape': [int(_x) for _x in np.shape(train_ids)]}
            atomic_write_json(_side, _rec_out)
        except (OSError, TypeError, ValueError):
            pass
    return _fp

def _validate_token_ids(ids, vocab, label='token ids'):
    """Hard range check for a token-id array before it enters a cache/batch.

    A negative or out-of-vocabulary id would either index an embedding row
    outside the model (Python/NumPy wraparound) or silently mask corruption in
    the split.  It runs on every cold build and every cache hit.
    """
    arr = np.asarray(ids)
    if arr.size == 0:
        return arr
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f'{label}: expected integer token ids, got dtype {arr.dtype}')
    mn, mx = int(arr.min()), int(arr.max())
    if mn < 0 or mx >= int(vocab):
        raise ValueError(f'{label}: token id range [{mn}, {mx}] is outside vocab [0, {int(vocab)})')
    return arr


def _cached_split_problems(meta, train_ids, val_batch, val_bnd, vocab, seq_len=None,
                           fingerprints_trusted=False, ranges_trusted=False):
    """Reasons a cached (train, val) pair cannot be trusted.

    The metadata carries content fingerprints for BOTH sides.  Checking only
    the train fingerprint lets a stale/mixed validation file be served under a
    valid train cache — exactly the situation in which validation (or test)
    rows leak into the tuning split.  New caches always record ``val_fp``;
    legacy caches without it are accepted with an explicit warning so existing
    results remain reproducible, but their validation bytes are not silently
    blessed as verified.
    """
    problems = []
    try:
        vocab = int(vocab)
    except (TypeError, ValueError, OverflowError):
        return ['metadata has no usable vocab size']
    if not fingerprints_trusted and meta.get('src_fp') != _ids_fp(train_ids):
        problems.append('train id fingerprint does not match the meta file')
    vb = np.asarray(val_batch)
    if vb.ndim != 2:
        problems.append(f'validation batch is not 2-D (shape {vb.shape})')
    elif vb.shape[0] == 0:
        problems.append('validation batch is empty')
    elif seq_len is not None and vb.shape[1] != int(seq_len):
        problems.append(f'validation sequence length {vb.shape[1]} != requested {seq_len}')
    if np.asarray(val_bnd).shape != vb.shape:
        problems.append(f'validation boundary mask shape {np.asarray(val_bnd).shape} != {vb.shape}')
    if (not fingerprints_trusted and 'val_fp' in meta
            and meta.get('val_fp') != _ids_fp(vb.reshape(-1))):
        problems.append('validation id fingerprint does not match the meta file')
    _sf_all = meta.get('split_fingerprints')
    _sfp = _sf_all.get('validation') if isinstance(_sf_all, dict) else None
    if _sfp is not None and not isinstance(_sfp, dict):
        problems.append('split_fingerprints.validation is not an object')
        _sfp = {}
    _sfp = _sfp or {}
    if (not fingerprints_trusted and _sfp.get('ids_fp') is not None
            and str(_sfp['ids_fp']) != str(_ids_fp(vb.reshape(-1)))):
        problems.append('split_fingerprints.validation.ids_fp does not match the validation ids')
    if (not fingerprints_trusted and _sfp.get('boundary_fp') is not None
            and str(_sfp['boundary_fp']) != str(_ids_fp(np.ascontiguousarray(val_bnd).reshape(-1)))):
        problems.append('split_fingerprints.validation.boundary_fp does not match the validation boundary mask')
    if meta.get('shuffle_at_split') is True:
        problems.append('metadata records a shuffled split; time/group order cannot be trusted')
    if meta.get('tokenizer_fit_split') not in (None, 'train'):
        problems.append(f"tokenizer was fitted on {meta.get('tokenizer_fit_split')!r}, not train")
    # The unit of the persisted ids is part of the cache's meaning.  A cache
    # may declare "line" (current WikiText rows) or "article" (a future
    # regrouping); an article claim without article-level sidecars would make
    # every per-document statistic silently line-level, so it is refused here.
    _du = meta.get('document_unit')
    if _du is not None and _du not in ('line', 'article'):
        problems.append(
            f'document_unit={_du!r} is not one of "line"/"article"')
    if _du == 'article' and meta.get('document_unit_is_article') is not True:
        problems.append(
            'cache declares document_unit=article but does not attest that '
            'the persisted ids are article-level')
    if _du == 'article' and meta.get('article_sidecars') is not True:
        problems.append(
            'cache declares document_unit=article but records no article-level '
            'sidecar attestation (article_sidecars=true)')
    if _du == 'line' and meta.get('document_unit_is_article') is True:
        problems.append(
            'cache declares line document_unit but document_unit_is_article=true')
    if not problems and not ranges_trusted:
        for _arr, _label in ((train_ids, 'cached train_ids'), (vb, 'cached val_batch')):
            try:
                _validate_token_ids(_arr, vocab, _label)
            except (TypeError, ValueError, OverflowError) as _e:
                problems.append(str(_e))
    return problems


def _cache_warning(what, problems):
    print(f'[data] WARNING: the cached {what} cannot be verified ({"; ".join(problems)}) — REBUILDING it so a mixed/stale split is never silently served')


_MEMO_MAX_BYTES = 1 << 30


def _memmap_backed(v):
    cur = v
    for _ in range(16):
        if isinstance(cur, np.memmap) or getattr(cur, 'filename', None) is not None:
            return True
        cur = getattr(cur, 'base', None)
        if cur is None or cur is v:
            return False
    return False


def _memo_value_bytes(v):
    """Rough resident size of a memo value; mapped arrays are not charged."""
    if isinstance(v, np.ndarray):
        if _memmap_backed(v):
            return 0
        return int(getattr(v, 'nbytes', 0) or 0)
    if isinstance(v, (tuple, list)):
        return sum(_memo_value_bytes(_x) for _x in v)
    return 0


def _memo_put(memo, key, value, cap_items, cap_bytes=_MEMO_MAX_BYTES):
    """Insert under a byte ceiling plus the item cap (LRU order is insertion
    order).  A single value larger than the ceiling is not cached."""
    b = _memo_value_bytes(value)
    if b > cap_bytes:
        return False
    total = sum(_memo_value_bytes(_v) for _v in memo.values())
    while memo and (len(memo) >= int(cap_items) or total + b > cap_bytes):
        total -= _memo_value_bytes(memo.pop(next(iter(memo))))
    memo[key] = value
    return True


_WT103_MEMO = {}
_WT103_MEMO_CAP = 4

def _wt_cache_dir(cache_dir=None):
    """Resolve the WikiText cache directory, honouring ``WT103_CACHE_DIR``.

    The loaders and the audit/confirmation driver share this resolver, so the
    leakage gate always inspects the same directory the evaluation reads and
    there is a single cache identity.
    """
    if cache_dir not in (None, ''):
        return str(cache_dir)
    return os.environ.get('WT103_CACHE_DIR') or './wt103_cache'


def _unique_cached_vocab(cache_dir, *, label='loader'):
    """The one vocabulary width declared by the train caches, or an error.

    Used to replace the historical ``vocab_size=8192`` default: a held-out
    test cache must be looked up at the vocabulary the training cache actually
    used, not at the scale this repository happened to run first.  Multiple
    train metas are accepted only when they agree on the width (the normal
    case: several sequence-length variants of one tokenizer).
    """
    import glob as _glob
    widths = set()
    for mp in _glob.glob(os.path.join(cache_dir, 'meta_v5_*.json')):
        try:
            with open(mp, encoding='utf-8') as fh:
                m = json.load(fh)
        except Exception:
            continue
        if not isinstance(m, dict) or m.get('vocab') is None:
            continue
        try:
            widths.add(int(m['vocab']))
        except (TypeError, ValueError, OverflowError):
            continue
    if len(widths) == 1:
        return next(iter(widths))
    if not widths:
        raise RuntimeError(
            f'{label}: no train/validation cache declares its vocabulary width; '
            'pass vocab_size explicitly rather than falling back to a '
            'repository default (build the frozen train cache first)')
    raise RuntimeError(
        f'{label}: train caches disagree on vocabulary width {sorted(widths)}; '
        'pass the intended vocab_size explicitly so the held-out cache cannot '
        'be silently paired with the wrong tokenizer')


def _tokenizer_vocab_sha256(tok):
    """Identity of the full id->token map (size alone is not identity)."""
    try:
        vocab = dict(tok.get_vocab())
    except Exception:
        return None
    blob = json.dumps(vocab, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def tokenizer_identity(tok, path=None):
    """Serialisable tokenizer identity for cache metadata and pairing.

    ``vocab_sha256`` binds the id space; ``file_sha256`` (when the tokenizer is
    persisted) additionally binds the concrete merge/byte serialisation.  Both
    are content identities, so two arms that agree on them cannot be using
    tokenizers that only happen to share a vocabulary size.
    """
    out = {'vocab_size': int(tok.get_vocab_size()) if tok is not None else None,
           'vocab_sha256': _tokenizer_vocab_sha256(tok) if tok is not None else None,
           'file_sha256': None}
    if path is not None and os.path.exists(path):
        try:
            out['file_sha256'] = file_sha256(path)
        except Exception:
            out['file_sha256'] = None
    return out


def load_wikitext_test_tokenizer(cache_dir=None, *, n_seqs, seq_len, vocab_size):
    """Load the persisted TEST tokenizer WITHOUT reading the test split.

    The article-unit reconstruction needs the exact byte-level decoder that
    produced the frozen slice.  ``load_wikitest`` writes
    ``test_tok_v2_sl{seq_len}_v{vocab_size}_ns{n_seqs}.json`` at build time;
    reading that file is not a test access.  Returns ``None`` when the cache
    has no persisted tokenizer (a legacy cache), so the caller can fall back
    explicitly instead of guessing an id space.
    """
    n_seqs = _protocol_pos_int('n_seqs', n_seqs)
    seq_len = _protocol_pos_int('seq_len', seq_len)
    vocab_size = _protocol_pos_int('vocab_size', vocab_size)
    cache_dir = _wt_cache_dir(cache_dir)
    tag = f'v2_sl{seq_len}_v{vocab_size}_ns{n_seqs}'
    path = os.path.join(cache_dir, f'test_tok_{tag}.json')
    if not os.path.exists(path):
        return None
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(path)
    except Exception as e:
        raise RuntimeError(
            f'load_wikitext_test_tokenizer: cached test tokenizer {path} is '
            f'unreadable ({type(e).__name__}: {e}); cannot verify the article '
            'unit without the exact id space') from e
    if int(tok.get_vocab_size()) != int(vocab_size):
        raise RuntimeError(
            f'load_wikitext_test_tokenizer: tokenizer width '
            f'{int(tok.get_vocab_size())} != requested {int(vocab_size)}')
    return tok


def _tokenizer_file_problems(meta, tok_path, label, *, require=False):
    """Reasons a persisted tokenizer file does not match its cache metadata.

    A cache that records no tokenizer identity is legacy (``require=False``:
    warning only), but a RECORDED identity that no longer matches is a hard
    problem: the id space or merge set was swapped after the cache was built.
    """
    problems = []
    if not isinstance(meta, dict):
        return [f'{label}: metadata is not an object']
    recorded = meta.get('tokenizer_file_sha256')
    if recorded is None:
        if require:
            problems.append(f'{label}: cache does not record the tokenizer file identity')
        return problems
    if not os.path.exists(tok_path):
        problems.append(f'{label}: tokenizer file {os.path.basename(tok_path)} is missing')
        return problems
    try:
        actual = file_sha256(tok_path)
    except Exception as e:
        problems.append(f'{label}: tokenizer file unreadable ({type(e).__name__}: {e})')
        return problems
    if str(actual) != str(recorded):
        problems.append(
            f'{label}: tokenizer file sha256 {actual[:16]}… does not match the '
            f'cache metadata {str(recorded)[:16]}… (the tokenizer was replaced)')
    return problems


def load_wikitext(seq_len, n_train_tokens, vocab_size=None, val_seqs=512,
                  cache_dir=None):
    if vocab_size is None:
        vocab_size = int(DATA_PROTOCOL_DEFAULTS['wikitext']['vocab_size'])
    vocab_size = _protocol_pos_int('vocab_size', vocab_size)
    from datasets import load_dataset
    cache_dir = _wt_cache_dir(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    tag = f'v5_sl{seq_len}_cap{n_train_tokens}_v{vocab_size}_vs{val_seqs}'
    tr_path = os.path.join(cache_dir, f'train_ids_{tag}.npy')
    va_path = os.path.join(cache_dir, f'val_batch_{tag}.npy')
    vp_path = os.path.join(cache_dir, f'val_bnd_{tag}.npy')
    me_path = os.path.join(cache_dir, f'meta_{tag}.json')
    trd_path = os.path.join(cache_dir, f'train_doc_ids_{tag}.npy')
    vad_path = os.path.join(cache_dir, f'val_doc_ids_{tag}.npy')
    # Only validation needs token-level groups; a 5B-token TRAIN sidecar
    # would be tens of GB and has no consumer.  Keep train document ids at the
    # document level.
    vad_tok_path = os.path.join(cache_dir, f'val_token_doc_ids_{tag}.npy')
    if all((os.path.exists(p) for p in (tr_path, va_path, vp_path, me_path))):
        _sig = _memo_file_sig({'train': tr_path, 'val': va_path,
                               'bnd': vp_path, 'meta': me_path,
                               'tokenizer': os.path.join(cache_dir, f'tok_{tag}.json')})
        _mkey = (os.path.abspath(cache_dir), tag, _sig)
        _hit = _WT103_MEMO.get(_mkey)
        if _hit is not None:
            print('[data] serving token ids from the in-process memo (same cache files, already verified) ...')
            return _hit
        print('[data] loading cached token ids ...')
        # Serve the train ids as a read-only mapping so every process (and the
        # parallel workers' corpus cache) shares the OS page cache.
        train_ids = np.load(tr_path, mmap_mode='r')
        meta = _read_cache_meta(me_path, 'WikiText train/validation')
        if meta is None:
            print('[data] WARNING: the WikiText train/validation metadata is '
                  'unreadable — REBUILDING the cache so a mixed or stale split '
                  'is never silently served')
            meta = {}
        val_batch = np.load(va_path)
        val_bnd = np.load(vp_path)
        _sigs = _cache_file_sigs({'train': tr_path, 'val': va_path, 'bnd': vp_path})
        _fast = _cache_file_sigs_match(meta, _sigs)
        _problems = _cached_split_problems(meta, train_ids, val_batch, val_bnd,
                                           meta.get('vocab'), seq_len=seq_len,
                                           fingerprints_trusted=False,
                                           ranges_trusted=_fast)
        # The persisted tokenizer is part of the split identity: the test
        # loader compares its id space to this file on every hit, so a swapped
        # train tokenizer must invalidate this cache rather than silently
        # defining a different id space for the train/validation ids.
        _problems.extend(_tokenizer_file_problems(
            meta, os.path.join(cache_dir, f'tok_{tag}.json'),
            'WikiText train/validation tokenizer'))
        if meta.get('tokenizer_fit_protocol') != 'train_prefix_token_budget':
            _problems.append('cache does not attest the frozen train-prefix tokenizer fit')
        if not meta.get('fit_fp'):
            _problems.append(
                'cache does not record the tokenizer fit-corpus fingerprint; '
                'tokenizer provenance cannot be bound to the train split')
        if not meta.get('val_fp'):
            _problems.append(
                'cache does not record a validation fingerprint; holdout '
                'content cannot be bound to the split it was audited with')
        _dd_meta = meta.get('dedup')
        if not isinstance(_dd_meta, dict):
            _problems.append('cache does not attest whole-document exact/near duplicate filtering')
        else:
            # Validation is used for every model-selection decision, so it is
            # not enough that the *documents* are disjoint: exact/near copies
            # of a training document must also have been removed from
            # validation at build time.  A cache that does not record this
            # cannot prove the selection split is uncontaminated.
            _vc = _dd_meta.get('validation_cleanup')
            if (not isinstance(_vc, dict)
                    or _vc.get('policy') != 'validation_docs_deduped_against_fit_and_train'):
                _problems.append(
                    'cache does not attest exact/near-duplicate cleanup of the '
                    'validation split against the tokenizer-fit and training documents')
            _splits = _dd_meta.get('splits')
            _vs = (_splits.get('validation')
                   if isinstance(_splits, dict) else None)
            if not isinstance(_vs, dict) or not {'exact', 'near'} <= set(_vs):
                _problems.append(
                    'cache does not record validation duplicate-removal counts '
                    '(exact/near)')
            # The near-duplicate geometry is part of the data identity.  A
            # cache that records a protocol must be self-consistent (digest
            # matches every parameter, redundant fields agree).  A legacy
            # cache without the object is accepted only when its recorded
            # threshold matches the declared default; otherwise the cleaning
            # semantics are unknown and the split must be rebuilt.
            _proto = _dd_meta.get('protocol')
            if _proto is not None:
                _pp = dedup_protocol_problems(_proto)
                if _pp:
                    _problems.append(
                        'cache dedup protocol is invalid: ' + '; '.join(_pp))
                else:
                    if _dd_meta.get('protocol_fp') != dedup_protocol_fp(_proto):
                        _problems.append(
                            'cache dedup protocol fingerprint does not match '
                            'its recorded parameters (mixed or edited metadata)')
                    for _pk, _pv in (
                            ('near_hamming_threshold', _proto['near_hamming_threshold']),
                            ('bands', _proto['bands']),
                            ('shingle_words', _proto['shingle_words']),
                            ('max_shingles', _proto['max_shingles'])):
                        _recorded = _dd_meta.get(_pk)
                        if _recorded is None:
                            continue
                        _recorded_i = _as_seed_int(_recorded)
                        if _recorded_i is None:
                            _problems.append(
                                f'cache dedup field {_pk}={_recorded!r} is not '
                                'an exact integer; the redundant protocol field '
                                'is unverifiable — rebuild the cache')
                            continue
                        if _recorded_i != int(_pv):
                            _problems.append(
                                f'cache dedup field {_pk}={_recorded!r} '
                                f'disagrees with its protocol ({_pv!r})')
            else:
                _partial = [k for k in ('bands', 'shingle_words',
                                        'max_shingles', 'protocol_fp')
                            if _dd_meta.get(k) is not None]
                if _partial:
                    _problems.append(
                        'cache records partial dedup geometry fields '
                        f'{_partial} but no full protocol; the cleaning '
                        'identity is incomplete — rebuild the cache')
                _legacy_thr = _dd_meta.get('near_hamming_threshold')
                if _legacy_thr is not None:
                    try:
                        _legacy_thr = int(_legacy_thr)
                    except (TypeError, ValueError, OverflowError):
                        _problems.append(
                            'cache dedup near_hamming_threshold is not an integer')
                    else:
                        if _legacy_thr != int(DEDUP_PROTOCOL_DEFAULTS['near_hamming_threshold']):
                            _problems.append(
                                'legacy cache records a non-default near-duplicate '
                                'threshold but no full protocol; its cleaning '
                                'geometry (shingle width/bands) is unknowable — '
                                'rebuild the cache')
        # Document ids are the group identity.  When the cache carries them,
        # prove the groups are disjoint instead of trusting the build-time
        # deduplicator; a sidecar pair that disagrees with the metadata is a
        # mixed cache and must be rebuilt.
        _has_doc_meta = bool(meta.get('doc_ids'))
        _has_doc_files = os.path.exists(trd_path) and os.path.exists(vad_path)
        if _has_doc_files:
            try:
                _trd = np.load(trd_path)
                _vad = np.load(vad_path)
                _shared_docs = int(np.intersect1d(_trd, _vad).size)
                if _shared_docs:
                    _problems.append(
                        f'{_shared_docs} document ids occur in both train and validation')
                _mdoc = meta.get('doc_ids') or {}
                if not _mdoc:
                    _problems.append(
                        'cache has document-id sidecars but no document-disjointness '
                        'attestation in its metadata')
                if _mdoc and (_mdoc.get('disjoint') is not True
                              or int(_mdoc.get('overlap', 0) or 0) != 0):
                    _problems.append(
                        'cache metadata does not attest document-disjoint train/validation splits')
                # Fingerprints make the sidecars part of the cache identity:
                # a sidecar swapped in from another build can be disjoint by
                # accident without describing these token arrays.
                if _mdoc.get('train_fp') and _mdoc['train_fp'] != _ids_fp(_trd):
                    _problems.append(
                        'train document-id sidecar does not match the fingerprint in the cache metadata')
                if _mdoc.get('validation_fp') and _mdoc['validation_fp'] != _ids_fp(_vad):
                    _problems.append(
                        'validation document-id sidecar does not match the fingerprint in the cache metadata')
            except Exception as _de:
                _problems.append(f'document-id sidecars unreadable ({type(_de).__name__}: {_de})')
        elif _has_doc_meta:
            _problems.append('cache metadata records document ids but the sidecar files are missing')
        else:
            # A cache that predates document-id sidecars cannot prove whole-
            # document group isolation.  Fail closed: rebuild it instead of
            # silently serving a split whose groups may cross train/validation.
            _problems.append(
                'cache has no document-id sidecars; whole-document group '
                'isolation cannot be verified — rebuild the corpus cache')
        _src_problems, _src_warnings = _dataset_source_problems(
            meta, wikitext_dataset_source(), 'WikiText train/validation')
        _problems.extend(_src_problems)
        _warn_dataset_source('WikiText train/validation', _src_warnings)
        if _problems:
            _cache_warning('WikiText train/validation split', _problems)
        else:
            if 'val_fp' not in meta:
                print('[data] NOTE: this is a legacy cache without a validation fingerprint; its file bytes are range/shape-checked but its split membership cannot be independently re-derived. New caches record val_fp.')
            audit_corpus_split(train_ids, val_batch, int(meta['vocab']),
                               dataset='wikitext', meta=meta)
            _out = (train_ids, val_batch, int(meta['vocab']), None, val_bnd)
            _memo_put(_WT103_MEMO, _mkey, _out, _WT103_MEMO_CAP)
            return _out
    ds = local_wikitext_if_available(cache_dir)
    _dataset_source = local_dataset_source('wikitext_raw')
    if ds is None:
        _wt_source = wikitext_dataset_source()
        last_err = None
        for repo in (WIKITEXT_HF_REPO, WIKITEXT_HF_REPO_ALIAS):
            try:
                # Request train/validation EXPLICITLY.  Calling without
                # `split=` materialises all three splits, including
                # wiki.test.raw; test is only ever requested by load_wikitest.
                # `revision=` pins the immutable Hub commit (see
                # wikitext_dataset_source): a moving branch would silently
                # change a fresh build's corpus.
                ds = {
                    'train': load_dataset(repo, WIKITEXT_HF_CONFIG, split='train', cache_dir=cache_dir, revision=_wt_source['revision']),
                    'validation': load_dataset(repo, WIKITEXT_HF_CONFIG, split='validation', cache_dir=cache_dir, revision=_wt_source['revision']),
                }
                _dataset_source = wikitext_dataset_source(repo=repo)
                break
            except Exception as e:
                last_err = e
        if ds is None:
            raise last_err if last_err is not None else RuntimeError('load_wikitext: no dataset repository could be loaded')
    if not hasattr(ds, '__getitem__') or 'train' not in ds or 'validation' not in ds:
        raise TypeError(f"load_wikitext: the loaded corpus object does not expose the ['train']/['validation'] splits (got {type(ds).__name__}); refusing to tokenise a corpus we cannot identify")
    # Tokenizer fit is a frozen TRAIN prefix with an explicit token cap:
    # every document it sees is also inside the model's training region (for
    # caps below the global cap), never a validation/test row.  The cap
    # is fixed across the scale ladder so S1/S2/S3 share one tokenizer version.
    _fit_cap = min(int(TOKENIZER_FIT_TOKENS), int(n_train_tokens))
    # Only the frozen prefix is needed for the tokenizer fit; the rest of the
    # train split's text does not enter the fit.
    _fit_rows_all = []
    _n_train_rows = len(ds['train'])
    _row_step = 5000
    for _start in range(0, _n_train_rows, _row_step):
        if len(_fit_rows_all) >= 50000:
            break
        _rows = ds['train'][_start:min(_start + _row_step, _n_train_rows)]['text']
        _fit_rows_all.extend(r for r in _rows if isinstance(r, str) and r.strip())
        if len(_fit_rows_all) > 50000:
            del _fit_rows_all[50000:]
    fit_text, tok, n_fit_tokens = _wikitext_fit_rows(
        _fit_rows_all, vocab_size, _fit_cap, max_rows=50000)
    if tok is None or not fit_text:
        raise RuntimeError('load_wikitext: the train split has no non-empty text rows to fit the tokenizer')
    vocab = tok.get_vocab_size()
    fit_fp = _text_fp(fit_text)
    _dedup = _DocDeduplicator()
    # Seed the deduplicator with the FULL rows of the fit prefix.  The token
    # cap may truncate the text of the final fit row; seeding with the
    # truncated view would leave that training document absent from the exact
    # duplicate index, so a full copy in validation/test could slip through.
    _fit_doc_prefix = list(_fit_rows_all[:len(fit_text)])
    for _r in _fit_doc_prefix:
        _dedup.is_duplicate(_r)
    _fit_dedup = _dedup.stats()
    _n_fit_docs = len(_fit_doc_prefix)
    del fit_text
    del _fit_rows_all
    gc.collect()
    eos_id = tok.token_to_id('<eos>')
    _dedup_by_split = {}

    def stream_tokenise(split, cap_tokens, label, _ds=ds, _tok=tok, _eos=eos_id,
                        _dtype=np.int32, want_token_doc_ids=False, stats=None):
        cap_tokens = max(0, int(cap_tokens))
        arr = np.empty(cap_tokens, dtype=_dtype)
        cursor = 0
        _rows_considered = 0
        _doc_ids = array.array('Q')
        # Per-token document ids are the only representation that can map a
        # fixed validation ROW back to the documents it overlaps.  The legacy
        # sidecar stored one id per document, which cannot be expanded without
        # per-document token lengths.  Emit both, so the leakage audit keeps
        # its document-level identity and the fold helper gets token-level
        # groups from the SAME stream.
        _token_doc_ids = array.array('Q') if want_token_doc_ids else None
        _skipped = {'exact': 0, 'near': 0}
        _dedup_by_split[split] = _skipped
        _n_nonempty_seen = 0
        row_chunk = 10000
        _stop = False
        for start in range(0, len(_ds[split]), row_chunk):
            rows = _ds[split][start:start + row_chunk]['text']
            rows = [r for r in rows if r.strip()]
            kept = []
            for _j, r in enumerate(rows):
                _rows_considered += 1
                # The tokenizer-fit documents are the frozen TRAIN prefix and
                # are part of the model's training stream; do not filter them
                # out as duplicates of themselves on the first pass.
                if split == 'train' and (_n_nonempty_seen + _j) < _n_fit_docs:
                    kept.append((r, None))
                    continue
                # The deduplicator hashes the exact document anyway; reuse
                # that fingerprint for the document-id sidecar instead of
                # hashing every kept row a second time.
                _fp = _doc_exact_fp(r)
                _dup, _kind = _dedup.is_duplicate(r, fp=_fp)
                if _dup:
                    _skipped[_kind] += 1
                else:
                    kept.append((r, _fp))
            _n_nonempty_seen += len(rows)
            if kept:
                for (_row, _fp), enc in zip(kept, _tok.encode_batch([_r for _r, _ in kept])):
                    ids = enc.ids
                    if _fp is None:
                        _fp = _doc_exact_fp(_row)
                    _doc_u = int(_doc_id_u64(_fp))
                    _doc_ids.append(_doc_u)
                    take = min(len(ids), cap_tokens - cursor)
                    if take > 0:
                        arr[cursor:cursor + take] = ids[:take]
                        if _token_doc_ids is not None:
                            # `array.extend` consumes an iterator in place, so
                            # a long document does not allocate a temporary
                            # Python list with one pointer per token (the same
                            # O(tokens) hazard the standalone capped-row helper
                            # avoids with np.repeat).
                            _token_doc_ids.extend(
                                itertools.repeat(_doc_u, int(take)))
                        cursor += take
                    if cursor >= cap_tokens:
                        _stop = True
                        break
                    arr[cursor] = _eos
                    if _token_doc_ids is not None:
                        _token_doc_ids.append(_doc_u)
                    cursor += 1
                    if cursor >= cap_tokens:
                        _stop = True
                        break
            if _stop or cursor >= cap_tokens:
                break
            if start and start % 50000 == 0:
                print(f'    [{label}] {start} rows, {cursor} tokens, '
                      f'{sum(_skipped.values())} duplicate docs skipped')
        _doc_arr = (np.frombuffer(_doc_ids, dtype=np.uint64).copy()
                    if _doc_ids else np.empty(0, dtype=np.uint64))
        _tok_doc_arr = (np.frombuffer(_token_doc_ids, dtype=np.uint64).copy()
                        if _token_doc_ids else np.empty(0, dtype=np.uint64))
        if stats is not None:
            stats.update({
                'source_rows_total': int(len(_ds[split])),
                'nonempty_rows_scanned': int(_rows_considered),
                'rows_served': int(_doc_arr.size),
                'tokens_served': int(cursor),
                'token_cap': int(cap_tokens),
                'cap_reached': bool(cursor >= cap_tokens),
                'duplicate_documents_skipped': {
                    'exact': int(_skipped.get('exact', 0)),
                    'near': int(_skipped.get('near', 0)),
                },
            })
        if cursor < arr.size:
            exact = arr[:cursor].copy()
            del arr
            return exact, _doc_arr, _tok_doc_arr
        return arr, _doc_arr, _tok_doc_arr
    print(f'[data] tokenising train (cap {n_train_tokens} tokens) ...')
    _train_stream_stats = {}
    train_ids, train_doc_ids, train_token_doc_ids = stream_tokenise(
        'train', n_train_tokens + seq_len, 'train', _dtype=np.int32,
        stats=_train_stream_stats)
    print(f'[data] train tokens = {len(train_ids)}  vocab = {vocab}')
    if len(train_ids) < n_train_tokens + seq_len:
        print(f'[data] WARNING: the train split ran dry at {len(train_ids)} tokens (< requested {n_train_tokens + seq_len}); runs configured for the advertised token target will train on fewer tokens than advertised')
    print('[data] tokenising validation ...')
    _val_stream_stats = {}
    val_ids, val_doc_ids, val_token_doc_ids = stream_tokenise(
        'validation', val_seqs * seq_len + seq_len, 'val', _dtype=np.int64,
        want_token_doc_ids=True, stats=_val_stream_stats)
    n_val = min(len(val_ids) // seq_len, val_seqs)
    val_batch = val_ids[:n_val * seq_len].reshape(n_val, seq_len)
    _validate_token_ids(train_ids, vocab, 'wikitext train_ids')
    _validate_token_ids(val_batch, vocab, 'wikitext val_batch')
    print(f'[data] val sequences = {n_val} (seq_len={seq_len})')
    if n_val < int(val_seqs):
        print(f'[data] WARNING: the validation split filled only {n_val} of '
              f'{int(val_seqs)} requested sequence(s) ({len(val_ids)} tokens '
              f'for seq_len={seq_len}); downstream panels configured for '
              'val_seqs rows will see a smaller holdout — enlarge the '
              'validation source or lower val_seqs and record the change')
    _shared_docs = int(np.intersect1d(train_doc_ids, val_doc_ids).size)
    if _shared_docs:
        raise RuntimeError(
            f'load_wikitext: {_shared_docs} document ids occur in both train and '
            'validation — refusing to write a leaking split')
    audit_corpus_split(train_ids, val_batch, vocab, dataset='wikitext')
    bnd_ids = sorted(boundary_token_ids(tok, eos_id))
    val_bnd = np.isin(val_batch, bnd_ids)
    print(f'[data] boundary tokens: {int(val_bnd.sum())} ({val_bnd.mean() * 100:.1f}% of val tokens)')
    for _p, _a in ((tr_path, train_ids), (va_path, val_batch), (vp_path, val_bnd),
                   (trd_path, train_doc_ids), (vad_path, val_doc_ids),
                   (vad_tok_path, val_token_doc_ids)):
        _fd, _tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(_p)), prefix=os.path.basename(_p) + '.', suffix='.tmp')
        try:
            with os.fdopen(_fd, 'wb') as _fh:
                np.save(_fh, _a)
            os.replace(_tmp, _p)
        except BaseException:
            try:
                os.unlink(_tmp)
            except OSError:
                pass
            raise
    _file_sigs = _cache_file_sigs({'train': tr_path, 'val': va_path, 'bnd': vp_path})
    # Persist the exact tokenizer BEFORE the metadata so the metadata can bind
    # its bytes.  The held-out test loader compares full vocab identity and the
    # file sha256 against this cache; a tokenizer swapped afterwards must not
    # keep certifying a different id space.
    _tok_path = os.path.join(cache_dir, f'tok_{tag}.json')
    try:
        tok.save(_tok_path)
    except Exception as _te:
        print(f'[data] NOTE: could not persist the train tokenizer ({type(_te).__name__}: {_te})')
    _tok_ident = tokenizer_identity(tok, _tok_path if os.path.exists(_tok_path) else None)
    atomic_write_json(me_path, {
        'vocab': vocab,
        'tokenizer_file': os.path.basename(_tok_path),
        'tokenizer_file_sha256': _tok_ident.get('file_sha256'),
        'tokenizer_vocab_sha256': _tok_ident.get('vocab_sha256'),
        # Data-construction semantics (not the kernel variant): the held-out
        # test cache must be built under the SAME split/cleanup semantics.
        'code_semantics': CODE_SEMANTICS,
        # Immutable upstream identity (Hub commit) or an explicit local-dump
        # record.  A token fingerprint alone cannot tell a fresh build on a
        # changed upstream revision from the original corpus; the audit checks
        # this field and the loaders refuse a mismatched rebuild.
        'dataset_source': _dataset_source,
        'src_fp': _ids_fp(train_ids),
        'val_fp': _ids_fp(np.ascontiguousarray(val_batch).reshape(-1)),
        'fit_fp': fit_fp,
        'file_sigs': _file_sigs,
        'id_range': {'train': _array_id_range(train_ids), 'val': _array_id_range(val_batch)},
        'split_strategy': 'official_train_validation_test_deduplicated',
        # Sampling design is recorded separately from token content.  The
        # current loader takes a source-order prefix, so the served slice is
        # NOT a probability sample of its upstream split; downstream claims
        # must not be generalised to unseen articles without this caveat.
        # A future freeze can build the same cache from a documented
        # document_stratified_plan and record design='stratified_document'.
        'sampling': {
            'schema': 'wikitext_slice_sampling_v1',
            'train': {
                'design': 'source_order_prefix',
                'selection_rule': 'first_nonempty_rows_in_source_order_until_token_cap',
                'random_seed': None,
                'representativeness_claim_allowed': False,
                'population_claim_allowed': False,
                'source_rows': int(_train_stream_stats.get('source_rows_total', len(ds['train']))),
                'nonempty_rows_scanned': int(_train_stream_stats.get('nonempty_rows_scanned', 0)),
                'rows_served': int(_train_stream_stats.get('rows_served', train_doc_ids.size)),
                'tokens_served': int(_train_stream_stats.get('tokens_served', len(train_ids))),
                'token_cap': int(_train_stream_stats.get('token_cap', n_train_tokens + seq_len)),
                'cap_reached': bool(_train_stream_stats.get('cap_reached', False)),
                'duplicate_documents_skipped': _train_stream_stats.get(
                    'duplicate_documents_skipped', {}),
                'claim_reason': (
                    'the served rows are the first non-empty source rows, not a '
                    'random sample; coverage of later articles is unverified'),
            },
            'validation': {
                'design': 'source_order_prefix',
                'selection_rule': 'first_nonempty_rows_in_source_order_until_token_cap',
                'random_seed': None,
                'representativeness_claim_allowed': False,
                'population_claim_allowed': False,
                'source_rows': int(_val_stream_stats.get('source_rows_total', len(ds['validation']))),
                'nonempty_rows_scanned': int(_val_stream_stats.get('nonempty_rows_scanned', 0)),
                'rows_served': int(_val_stream_stats.get('rows_served', val_doc_ids.size)),
                'tokens_served': int(_val_stream_stats.get('tokens_served', len(val_ids))),
                'token_cap': int(_val_stream_stats.get('token_cap', val_seqs * seq_len + seq_len)),
                'cap_reached': bool(_val_stream_stats.get('cap_reached', False)),
                'sequences_served': int(val_batch.shape[0]),
                'sequences_requested': int(val_seqs),
                'duplicate_documents_skipped': _val_stream_stats.get(
                    'duplicate_documents_skipped', {}),
                'claim_reason': (
                    'the selection split is a prefix of the source order, not a '
                    'random sample; it is valid for selection under the frozen '
                    'protocol but not for population inference'),
            },
        },
        # A WikiText row is one raw line (a heading or a paragraph), not one
        # article.  The ids below group by that row; downstream "per-document"
        # statistics must either name this unit or regroup by top-level
        # headings (audit_gates.audit_document_unit).  Recorded explicitly so a
        # later reader cannot mistake the unit for an article by convention.
        'document_unit': 'line',
        'document_unit_is_article': False,
        'document_unit_definition': (
            'one non-empty raw WikiText row (heading/paragraph line); an '
            'article spans multiple rows and is delimited by top-level '
            '" = Title = " headings'),
        'split_fingerprints': {
            'fingerprint_algorithm': _IDS_FP_ALGO,
            'train': {'file': os.path.basename(tr_path), 'ids_fp': _ids_fp(train_ids),
                      'documents_fp': _ids_fp(train_doc_ids),
                      'tokens': int(len(train_ids))},
            'validation': {'file': os.path.basename(va_path),
                           'ids_fp': _ids_fp(np.ascontiguousarray(val_batch).reshape(-1)),
                           'boundary_fp': _ids_fp(np.ascontiguousarray(val_bnd).reshape(-1)),
                           'documents_fp': _ids_fp(val_doc_ids),
                           'token_documents_fp': _ids_fp(val_token_doc_ids),
                           'sequences': int(val_batch.shape[0]),
                           'seq_len': int(seq_len)},
        },
        'shuffle_at_split': False,
        'tokenizer_fit_split': 'train',
        'tokenizer_fit_protocol': 'train_prefix_token_budget',
        'tokenizer_fit_token_cap': int(min(TOKENIZER_FIT_TOKENS, int(n_train_tokens))),
        'tokenizer_fit_max_rows': 50000,
        'tokenizer_fit_tokens': int(n_fit_tokens),
        'tokenizer_fit_rows': int(_n_fit_docs),
        'tokenizer_fit_docs_in_train_prefix': True,
        'doc_ids': {
            'train_count': int(train_doc_ids.size),
            'validation_count': int(val_doc_ids.size),
            'train_fp': _ids_fp(train_doc_ids),
            'validation_fp': _ids_fp(val_doc_ids),
            'validation_token_count': int(val_token_doc_ids.size),
            'validation_token_fp': _ids_fp(val_token_doc_ids),
            'overlap': _shared_docs,
            'disjoint': True,
            'policy': 'whole_document_exact_and_near_duplicate_filtered',
        },
        'dedup': {
            'accounting_schema': 'wikitext_dedup_accounting_v1',
            # The cleaning geometry is part of the data identity.  `protocol`
            # is the complete object; `protocol_fp` binds it into the cache so
            # a train/test pair built with different shingle/band parameters
            # cannot be served as if it were one dataset.
            'protocol': _dedup.protocol(),
            'protocol_fp': _dedup.protocol_fp(),
            'fit': _fit_dedup,
            'splits': _dedup_by_split,
            'policy': 'exact_and_near_duplicate_documents_filtered_across_fit_train_validation',
            'near_hamming_threshold': int(_dedup.near_threshold),
            'bands': int(_dedup.bands),
            'shingle_words': int(_dedup.shingle_words),
            'max_shingles': int(_dedup.max_shingles),
            'validation_cleanup': {
                'protocol': _dedup.protocol(),
                'protocol_fp': _dedup.protocol_fp(),
                'policy': 'validation_docs_deduped_against_fit_and_train',
                'exact_removed': int(_dedup_by_split.get('validation', {}).get('exact', 0)),
                'near_removed': int(_dedup_by_split.get('validation', {}).get('near', 0)),
            },
        },
        'group_isolation': 'whole_documents_exact_and_near_duplicate_filtered',
        'train_tokens': int(len(train_ids)),
        'val_sequences': int(val_batch.shape[0]),
        'seq_len': int(seq_len),
    }, indent=0)
    print('[data] cached token ids to disk (resume-safe)')
    del ds
    gc.collect()
    return (train_ids, val_batch, vocab, tok, val_bnd)

# ---- 独立最终 test split（wiki.test.raw，冻结后一次性评估，见 design_notes §17）
_WT103_TEST_MEMO = {}
_WT103_TEST_MEMO_CAP = 2

def _text_fp(rows):
    """sha256 of a list of text rows (fit-corpus provenance for the test cache)."""
    h = hashlib.sha256()
    for r in rows:
        h.update(r.encode('utf-8', errors='replace'))
        h.update(b'\n')
    return h.hexdigest()[:16]
