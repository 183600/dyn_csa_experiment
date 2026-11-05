# ---------------------------------------------------------------------------
# 148. 同一份声明环境不等于同一套算术：可重放的数值 canary
#
# `env_fp_v2`/`env_fp_v3` 证明两次执行声明了同一套依赖、解释器、OS 与数值
# 后端；它们不能证明两个进程真的执行了同一条浮点路径。同一个 wheel 在
# 不同 CPU dispatch、BLAS 线程数、驱动/内核或 GPU 型号上仍可能给出不同的
# 低位结果。审计一个 PPL 差异时，若把这种「声明相同、算术不同」误当成
# 测量复现，后续所有配对推断都会被污染。
#
# 这里不引入新的统计检验，也不改变任何训练/分析判定。它把一组固定输入
# 上的固定 torch 算子输出按原始字节哈希成一个 `arithmetic_fingerprint`，
# 并把生成它的参数、代码身份和环境身份一起落盘。重放时只比较算术块：
# 算术变了是 drift，元数据变了只是可解释的 warning。canary 不是「训练
# 逐位确定」的证明；它只回答「当前进程的实际算术是否还是记录的那一套」。
# ---------------------------------------------------------------------------

NC_SCHEMA = 1
NC_ARTIFACT_KIND = 'numerical_canary'
NC_OP_NAMES = ('matmul', 'softmax', 'rmsnorm', 'attention',
               'cross_entropy', 'reduction', 'rng', 'topk', 'solve')
NC_MAX_SIZE = 64
NC_DEFAULT_SIZE = 8
NC_DEFAULT_DTYPE = 'float32'
NC_DTYPES = {'float32': torch.float32, 'float64': torch.float64}

NC_RESULT_KEYS = ('kind', 'schema', 'status', 'params', 'arithmetic',
                  'environment', 'warnings')
NC_PARAM_KEYS = ('seed', 'size', 'dtype', 'device')
NC_ARITHMETIC_KEYS = ('schema', 'fingerprint', 'input_fingerprint', 'ops')
NC_OP_KEYS = ('sha256', 'shape', 'dtype', 'finite')
NC_ARTIFACT_KEYS = ('kind', 'schema', 'created_at', 'inputs', 'result',
                    'provenance', 'content_sha256')
NC_PROVENANCE_KEYS = ('code_semantics', 'code_stamp', 'source_tree',
                      'env_fp_v2', 'env_fp_v3', 'lock_sha256')


def _nc_sha_json(obj):
    """Canonical JSON digest; never emits NaN/Infinity."""
    return _sha256_json(obj)  # noqa: F821 (part_00)


def _nc_tensor_bytes(t):
    """Raw CPU bytes of one tensor (bfloat16 is widened to float32)."""
    out = t.detach().to(device='cpu').contiguous()
    if out.dtype == torch.bfloat16:
        out = out.to(torch.float32)
    return out.numpy().tobytes()


def _nc_device(value):
    """Resolve a CPU/CUDA device and refuse an unavailable device.

    ``None`` means CPU: a canary that silently picked CUDA would make a
    laptop and a GPU host look comparable when they are not.
    """
    if value is None:
        value = 'cpu'
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (str, torch.device)):
        raise ValueError(
            f'numerical_canary: device must be "cpu" or "cuda[:N]", got {value!r}')
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError, ValueError):
        raise ValueError(
            f'numerical_canary: invalid device {value!r}') from None
    if device.type not in ('cpu', 'cuda'):
        raise ValueError(
            f'numerical_canary: only cpu/cuda devices are supported, got '
            f'{device.type!r}')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError(
                'numerical_canary: device=cuda requested but CUDA is not '
                'available; a CPU canary must not stand in for a GPU one')
        index = device.index
        if index is None:
            index = int(torch.cuda.current_device())
        if index < 0 or index >= int(torch.cuda.device_count()):
            raise ValueError(
                f'numerical_canary: cuda device index {index} is outside the '
                f'visible device count {torch.cuda.device_count()}')
        device = torch.device('cuda', index)
    return device


def _nc_dtype(value):
    if isinstance(value, torch.dtype):
        for name, dtype in NC_DTYPES.items():
            if value == dtype:
                return name, dtype
        raise ValueError(
            f'numerical_canary: unsupported dtype {value}; expected '
            f'{sorted(NC_DTYPES)}')
    if not isinstance(value, str) or value not in NC_DTYPES:
        raise ValueError(
            f'numerical_canary: dtype must be one of {sorted(NC_DTYPES)}, '
            f'got {value!r}')
    return value, NC_DTYPES[value]


def _nc_size(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, numbers.Integral):
        raise ValueError(
            f'numerical_canary: size must be an integer in [2, {NC_MAX_SIZE}], '
            f'got {value!r}')
    size = int(value)
    if size < 2 or size > NC_MAX_SIZE:
        raise ValueError(
            f'numerical_canary: size must be in [2, {NC_MAX_SIZE}], got {size}')
    return size


def _nc_inputs(size, dtype, device, seed):
    """Fixed, seed-independent inputs plus one seeded RNG draw.

    The matrix values are built in float64 from integers and then cast, so the
    input bytes do not depend on a platform RNG.  The seed is carried by the
    explicit ``rng`` op instead, which keeps the arithmetic probe independent
    of how a particular torch build initialises models.
    """
    base = torch.arange(size * size, dtype=torch.float64).reshape(size, size)
    a64 = ((base % 17) - 8) / 7.0
    b64 = ((base.t().contiguous() % 13) - 6) / 5.0
    a = a64.to(dtype=dtype, device=device)
    b = b64.to(dtype=dtype, device=device)
    targets = (torch.arange(size, dtype=torch.long, device=device) % size)
    eye = torch.eye(size, dtype=dtype, device=device)
    try:
        gen = torch.Generator(device=device).manual_seed(int(seed))
    except (RuntimeError, TypeError):
        # Older backends do not accept a CUDA generator in this constructor;
        # falling back to CPU would silently change what the op measures, so
        # fail closed instead.
        raise ValueError(
            f'numerical_canary: this torch build cannot create a generator '
            f'on {device}') from None
    return a, b, targets, eye, gen


def _nc_op_record(t):
    u = t.detach().to(device='cpu').contiguous()
    if u.dtype == torch.bfloat16:
        u = u.to(torch.float32)
    try:
        finite = bool(torch.isfinite(t).all().item())
    except (RuntimeError, TypeError):
        finite = True
    return {
        'sha256': hashlib.sha256(u.numpy().tobytes()).hexdigest(),
        'shape': [int(x) for x in u.shape],
        'dtype': str(u.dtype).replace('torch.', ''),
        'finite': finite,
    }


def _nc_environment(device):
    import platform
    threads = {
        'torch': int(torch.get_num_threads()),
        'interop': (int(torch.get_num_interop_threads())
                    if hasattr(torch, 'get_num_interop_threads') else None),
    }
    env = {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'torch': torch.__version__,
        'torch_git_version': getattr(torch.version, 'git_version', None),
        'numpy': np.__version__,
        'cuda_version': getattr(torch.version, 'cuda', None),
        'threads': threads,
        'determinism': determinism_label(),  # noqa: F821 (part_26)
        'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
        'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
        'deterministic_algorithms': bool(
            torch.are_deterministic_algorithms_enabled()),
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
    }
    if device.type == 'cuda':
        try:
            env['gpu_name'] = torch.cuda.get_device_name(device)
            props = torch.cuda.get_device_properties(device)
            env['gpu_capability'] = [int(props.major), int(props.minor)]
        except Exception as exc:  # pragma: no cover - driver dependent
            env['gpu_name'] = None
            env['gpu_error'] = f'{type(exc).__name__}: {exc}'
    return env


def numerical_canary(*, seed=0, size=NC_DEFAULT_SIZE, dtype='float32',
                     device=None):
    """Run the fixed arithmetic probe and return a replayable JSON record.

    The probe covers the operators whose low bits matter to this project:
    GEMM, softmax, RMS-norm, scaled-dot-product attention, cross-entropy,
    reductions, the torch RNG stream, top-k ordering and a LAPACK solve.
    Every output is hashed from its raw bytes; the arithmetic fingerprint
    binds the dtype/device class/size and those per-op digests.  Two calls in
    the same process are expected to agree bit for bit; a disagreement is
    itself a finding (non-deterministic arithmetic), not a reason to round.
    """
    seed = _seed_value(seed, where='numerical_canary seed')  # noqa: F821
    size = _nc_size(size)
    dtype_name, dtype_obj = _nc_dtype(dtype)
    dev = _nc_device(device)
    a, b, targets, eye, gen = _nc_inputs(size, dtype_obj, dev, seed)
    with torch.no_grad():
        matmul = a @ b
        softmax = torch.softmax(a / 3.0, dim=-1)
        rmsnorm = a * torch.rsqrt(a.square().mean(dim=-1, keepdim=True) + 1e-6)
        attention = torch.softmax(
            (a @ b.transpose(-1, -2)) / math.sqrt(float(size)), dim=-1) @ b
        ce = torch.nn.functional.cross_entropy(a, targets)
        reduction = torch.stack(
            (a.sum(dim=-1), a.mean(dim=-1),
             a.std(dim=-1, unbiased=False)), dim=-1)
        rng = torch.rand(size, generator=gen, dtype=dtype_obj, device=dev)
        k = min(3, size)
        topk = torch.topk(a.abs(), k=k, dim=-1).indices
        solve = torch.linalg.solve(a + eye * float(size), b)
    op_tensors = {
        'matmul': matmul,
        'softmax': softmax,
        'rmsnorm': rmsnorm,
        'attention': attention,
        'cross_entropy': ce,
        'reduction': reduction,
        'rng': rng,
        'topk': topk,
        'solve': solve,
    }
    ops = {name: _nc_op_record(op_tensors[name]) for name in NC_OP_NAMES}
    input_fp = _nc_sha_json({
        'schema': 1,
        'size': size,
        'dtype': dtype_name,
        'seed': int(seed),
        'a': hashlib.sha256(_nc_tensor_bytes(a)).hexdigest(),
        'b': hashlib.sha256(_nc_tensor_bytes(b)).hexdigest(),
    })
    arithmetic = {
        'schema': 1,
        'fingerprint': _nc_sha_json({
            'schema': 1,
            'size': size,
            'dtype': dtype_name,
            'device_class': dev.type,
            'input_fingerprint': input_fp,
            'ops': ops,
        }),
        'input_fingerprint': input_fp,
        'ops': ops,
    }
    warnings = []
    nonfinite = sorted(name for name, row in ops.items() if not row['finite'])
    if nonfinite:
        warnings.append(
            'non-finite arithmetic output(s): ' + ', '.join(nonfinite))
    if dev.type == 'cuda' and not (
            torch.are_deterministic_algorithms_enabled()
            and torch.backends.cudnn.deterministic):
        warnings.append(
            'cuda canary ran without deterministic-algorithms/cudnn settings; '
            'two records may differ for reasons outside the declared stack')
    return {
        'kind': 'numerical_canary',
        'schema': NC_SCHEMA,
        'status': 'ok',
        'params': {
            'seed': int(seed),
            'size': size,
            'dtype': dtype_name,
            'device': str(dev),
        },
        'arithmetic': arithmetic,
        'environment': _nc_environment(dev),
        'warnings': warnings,
    }


def numerical_canary_provenance():
    """Code/environment identity stored beside a canary artifact.

    The arithmetic block is compared on replay; this block explains which
    checkout and declared stack produced it.  ``source_tree`` uses the same
    digest as every run record, so a canary cannot be attached to a different
    measurement tree without the difference being visible.
    """
    try:
        env = environment_snapshot()  # noqa: F821 (part_27)
    except Exception as exc:  # pragma: no cover - defensive
        env = {'error': f'{type(exc).__name__}: {exc}'}
    contract = env.get('dependency_contract')
    contract = contract if isinstance(contract, dict) else {}
    out = {
        'code_semantics': CODE_SEMANTICS,  # noqa: F821
        'code_stamp': code_stamp(),  # noqa: F821
        'source_tree': source_tree_summary(),  # noqa: F821
        'env_fp_v2': env.get('env_fp_v2'),
        'env_fp_v3': env.get('env_fp_v3'),
        'lock_sha256': contract.get('lock_sha256'),
    }
    return out


def _nc_content_sha(artifact):
    body = {key: value for key, value in artifact.items()
            if key not in ('created_at', 'content_sha256')}
    return _nc_sha_json(body)


def build_numerical_canary_artifact(*, seed=0, size=NC_DEFAULT_SIZE,
                                    dtype='float32', device=None,
                                    created_at=None, provenance=None,
                                    result=None):
    """Run/accept one canary and return a content-addressed artifact.

    ``created_at`` is excluded from ``content_sha256`` so a rebuild of the
    same arithmetic is recognised as the same artifact.  The provenance block
    is *inside* the digest: changing the recorded checkout/environment is a
    different record even when the arithmetic happens to agree.
    """
    if result is None:
        result = numerical_canary(seed=seed, size=size, dtype=dtype,
                                  device=device)
    if not isinstance(result, dict) or result.get('kind') != 'numerical_canary':
        raise TypeError('build_numerical_canary_artifact: result must be a '
                        'numerical_canary record')
    params = result.get('params')
    if not isinstance(params, dict):
        raise ValueError('build_numerical_canary_artifact: result.params is '
                         'missing')
    inputs = {key: params.get(key) for key in NC_PARAM_KEYS}
    artifact = {
        'kind': NC_ARTIFACT_KIND,
        'schema': NC_SCHEMA,
        'created_at': (created_at if created_at is not None
                       else reproducible_timestamp()),  # noqa: F821 (part_46)
        'inputs': inputs,
        'result': result,
        'provenance': (provenance if provenance is not None
                       else numerical_canary_provenance()),
    }
    artifact['content_sha256'] = _nc_content_sha(artifact)
    return artifact


def _nc_validate_result(result):
    """Structural validation of a canary result; no recomputation."""
    problems = []
    if not isinstance(result, dict):
        return ['result must be a JSON object']
    if result.get('kind') != 'numerical_canary':
        problems.append(
            f"result.kind must be 'numerical_canary', got {result.get('kind')!r}")
    if result.get('schema') != NC_SCHEMA:
        problems.append(
            f'result.schema must be {NC_SCHEMA}, got {result.get("schema")!r}')
    if result.get('status') != 'ok':
        problems.append(
            f'result.status must be "ok", got {result.get("status")!r}')
    extra = sorted(set(result) - set(NC_RESULT_KEYS))
    if extra:
        problems.append('result carries field(s) outside the schema: '
                        + ', '.join(map(str, extra)))
    params = result.get('params')
    if not isinstance(params, dict):
        problems.append('result.params must be an object')
    else:
        extra = sorted(set(params) - set(NC_PARAM_KEYS))
        missing = [k for k in NC_PARAM_KEYS if k not in params]
        if extra:
            problems.append('result.params carries field(s) outside the '
                            'schema: ' + ', '.join(map(str, extra)))
        if missing:
            problems.append(f'result.params is missing {missing}')
        if 'seed' in params:
            try:
                _seed_value(params['seed'], where='canary params.seed')  # noqa: F821
            except (TypeError, ValueError) as exc:
                problems.append(str(exc))
        if 'size' in params:
            try:
                _nc_size(params['size'])
            except (TypeError, ValueError) as exc:
                problems.append(str(exc))
        if 'dtype' in params and params['dtype'] not in NC_DTYPES:
            problems.append(
                f'result.params.dtype must be one of {sorted(NC_DTYPES)}, '
                f'got {params["dtype"]!r}')
        if 'device' in params:
            dev = params['device']
            if not isinstance(dev, str) or not dev.startswith(('cpu', 'cuda')):
                problems.append(
                    f'result.params.device must be a cpu/cuda string, got {dev!r}')
    arithmetic = result.get('arithmetic')
    if not isinstance(arithmetic, dict):
        problems.append('result.arithmetic must be an object')
    else:
        extra = sorted(set(arithmetic) - set(NC_ARITHMETIC_KEYS))
        missing = [k for k in NC_ARITHMETIC_KEYS if k not in arithmetic]
        if extra:
            problems.append('result.arithmetic carries field(s) outside the '
                            'schema: ' + ', '.join(map(str, extra)))
        if missing:
            problems.append(f'result.arithmetic is missing {missing}')
        fp = arithmetic.get('fingerprint')
        if not isinstance(fp, str) or not re.fullmatch(r'[0-9a-f]{64}', fp):
            problems.append(
                'result.arithmetic.fingerprint must be a 64-hex string')
        ifp = arithmetic.get('input_fingerprint')
        if not isinstance(ifp, str) or not re.fullmatch(r'[0-9a-f]{64}', ifp):
            problems.append(
                'result.arithmetic.input_fingerprint must be a 64-hex string')
        ops = arithmetic.get('ops')
        if not isinstance(ops, dict) or not ops:
            problems.append('result.arithmetic.ops must be a non-empty object')
        else:
            for name in NC_OP_NAMES:
                if name not in ops:
                    problems.append(f'result.arithmetic.ops is missing {name!r}')
            extra_ops = sorted(set(ops) - set(NC_OP_NAMES))
            if extra_ops:
                problems.append(
                    'result.arithmetic.ops carries op(s) outside the schema: '
                    + ', '.join(map(str, extra_ops)))
            for name, row in sorted(ops.items(), key=lambda kv: str(kv[0])):
                if not isinstance(row, dict):
                    problems.append(f'result.arithmetic.ops[{name!r}] must be '
                                    'an object')
                    continue
                extra = sorted(set(row) - set(NC_OP_KEYS))
                missing = [k for k in NC_OP_KEYS if k not in row]
                if extra:
                    problems.append(
                        f'result.arithmetic.ops[{name!r}] carries field(s) '
                        'outside the schema: ' + ', '.join(map(str, extra)))
                if missing:
                    problems.append(
                        f'result.arithmetic.ops[{name!r}] is missing {missing}')
                sha = row.get('sha256')
                if not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{64}', sha):
                    problems.append(
                        f'result.arithmetic.ops[{name!r}].sha256 must be 64-hex')
                shape = row.get('shape')
                if (not isinstance(shape, list)
                        or any(isinstance(x, (bool, np.bool_))
                               or not isinstance(x, numbers.Integral)
                               or int(x) < 0 for x in shape)):
                    problems.append(
                        f'result.arithmetic.ops[{name!r}].shape must be a '
                        'list of non-negative integers')
                if not isinstance(row.get('dtype'), str):
                    problems.append(
                        f'result.arithmetic.ops[{name!r}].dtype must be a string')
                if not isinstance(row.get('finite'), bool):
                    problems.append(
                        f'result.arithmetic.ops[{name!r}].finite must be a boolean')
        # The fingerprint must be a *function* of the recorded arithmetic, not
        # an independently editable label.  Without this check, changing one
        # op digest while keeping the old fingerprint made both the artifact
        # digest and the replay look valid.
        if (isinstance(params, dict) and isinstance(fp, str)
                and isinstance(ifp, str) and isinstance(ops, dict)
                and all(k in params for k in ('size', 'dtype', 'device'))):
            try:
                expected_fp = _nc_sha_json({
                    'schema': 1,
                    'size': params.get('size'),
                    'dtype': params.get('dtype'),
                    'device_class': str(params.get('device')).split(':')[0],
                    'input_fingerprint': ifp,
                    'ops': ops,
                })
            except Exception as exc:
                problems.append(
                    'result.arithmetic.fingerprint cannot be recomputed: '
                    f'{type(exc).__name__}: {exc}')
            else:
                if expected_fp != fp:
                    problems.append(
                        'result.arithmetic.fingerprint does not recompute from '
                        'the recorded params/input_fingerprint/ops')
    env = result.get('environment')
    if not isinstance(env, dict):
        problems.append('result.environment must be an object')
    warnings = result.get('warnings')
    if warnings is not None and (not isinstance(warnings, list)
                                 or any(not isinstance(x, str) for x in warnings)):
        problems.append('result.warnings must be a list of strings')
    return problems


def _nc_result_identity(result):
    params = result.get('params') if isinstance(result, dict) else {}
    arithmetic = result.get('arithmetic') if isinstance(result, dict) else {}
    params = params if isinstance(params, dict) else {}
    arithmetic = arithmetic if isinstance(arithmetic, dict) else {}
    return {
        'params': {k: params.get(k) for k in NC_PARAM_KEYS},
        'fingerprint': arithmetic.get('fingerprint'),
        'input_fingerprint': arithmetic.get('input_fingerprint'),
        'ops': arithmetic.get('ops') if isinstance(arithmetic.get('ops'), dict) else {},
    }


def compare_numerical_canaries(original, rerun):
    """Compare two canary results (or artifacts) by observed arithmetic.

    Missing/malformed records are ``unverified``; a different device class,
    dtype or size is ``not_comparable`` rather than a spurious mismatch.  The
    environment block is compared separately: a changed thread count or torch
    build is reported as ``environment_changes`` even when the probe agrees.
    """
    def _unwrap(obj):
        if (isinstance(obj, dict) and obj.get('kind') == NC_ARTIFACT_KIND
                and isinstance(obj.get('result'), dict)):
            try:
                digest_ok = _nc_content_sha(obj) == obj.get('content_sha256')
            except Exception:
                digest_ok = False
            if not digest_ok:
                return None
            return obj['result']
        return obj

    a, b = _unwrap(original), _unwrap(rerun)
    out = {
        'kind': 'numerical_canary_comparison',
        'schema': 1,
        'status': 'unverified',
        'arithmetic_match': False,
        'not_comparable': None,
        'differing_ops': [],
        'environment_changes': [],
        'problems': [],
    }
    pa, pb = _nc_validate_result(a), _nc_validate_result(b)
    if pa or pb:
        out['problems'] = ([f'original: {p}' for p in pa]
                           + [f'rerun: {p}' for p in pb])
        return out
    ia, ib = _nc_result_identity(a), _nc_result_identity(b)
    for key in ('device', 'dtype', 'size'):
        va, vb = ia['params'].get(key), ib['params'].get(key)
        if key == 'device':
            va_cls = str(va).split(':')[0] if isinstance(va, str) else None
            vb_cls = str(vb).split(':')[0] if isinstance(vb, str) else None
            if va_cls != vb_cls:
                out['status'] = 'not_comparable'
                out['not_comparable'] = {
                    'field': 'device_class', 'original': va, 'rerun': vb}
                return out
        elif va != vb:
            out['status'] = 'not_comparable'
            out['not_comparable'] = {
                'field': key, 'original': va, 'rerun': vb}
            return out
    if ia['fingerprint'] == ib['fingerprint']:
        out['status'] = 'match'
        out['arithmetic_match'] = True
    else:
        out['status'] = 'mismatch'
        names = sorted(set(ia['ops']) | set(ib['ops']))
        for name in names:
            ra, rb = ia['ops'].get(name), ib['ops'].get(name)
            if ra != rb:
                out['differing_ops'].append(name)
    ea, eb = a.get('environment'), b.get('environment')
    ea = ea if isinstance(ea, dict) else {}
    eb = eb if isinstance(eb, dict) else {}
    for key in sorted(set(ea) | set(eb)):
        if ea.get(key) != eb.get(key):
            out['environment_changes'].append(key)
    return out


def validate_numerical_canary_artifact(artifact, *, replay=True):
    """Replay a stored canary artifact and report arithmetic drift.

    ``ok``       — the digest is intact and the recorded arithmetic replays;
    ``drift``    — the digest is intact but the current process produces a
                   different arithmetic fingerprint;
    ``not_verifiable_locally`` — the recorded device is unavailable here, so
                   the replay cannot be attempted;
    ``fail``     — malformed/edited artifact or an unsupported schema.
    """
    out = {
        'kind': 'numerical_canary_validation',
        'schema': 1,
        'status': 'fail',
        'problems': [],
        'warnings': [],
        'environment_changes': [],
        'provenance_changes': [],
        'drifted_ops': [],
        'arithmetic_match': None,
        'content_sha256': None,
        'result': None,
        'replay': None,
    }
    if not isinstance(artifact, dict):
        out['problems'].append('artifact must be a JSON object')
        return out
    if artifact.get('kind') != NC_ARTIFACT_KIND:
        out['problems'].append(
            f'artifact.kind must be {NC_ARTIFACT_KIND!r}, got '
            f'{artifact.get("kind")!r}')
    if artifact.get('schema') != NC_SCHEMA:
        out['problems'].append(
            f'artifact.schema must be {NC_SCHEMA}, got '
            f'{artifact.get("schema")!r}')
    extra = sorted(set(artifact) - set(NC_ARTIFACT_KEYS))
    missing = [k for k in NC_ARTIFACT_KEYS if k not in artifact]
    if extra:
        out['problems'].append('artifact carries field(s) outside the '
                               'schema: ' + ', '.join(map(str, extra)))
    if missing:
        out['problems'].append(f'artifact is missing {missing}')
    inputs = artifact.get('inputs')
    if not isinstance(inputs, dict):
        out['problems'].append('artifact.inputs must be an object')
        inputs = {}
    else:
        extra = sorted(set(inputs) - set(NC_PARAM_KEYS))
        missing_in = [k for k in NC_PARAM_KEYS if k not in inputs]
        if extra:
            out['problems'].append('artifact.inputs carries field(s) outside '
                                   'the schema: ' + ', '.join(map(str, extra)))
        if missing_in:
            out['problems'].append(f'artifact.inputs is missing {missing_in}')
    if not isinstance(artifact.get('created_at'), str):
        out['problems'].append('artifact.created_at must be a string')
    provenance = artifact.get('provenance')
    if not isinstance(provenance, dict):
        out['problems'].append('artifact.provenance must be an object')
        provenance = {}
    else:
        missing_pv = [k for k in ('code_semantics', 'source_tree')
                      if k not in provenance]
        if missing_pv:
            out['problems'].append(
                f'artifact.provenance is missing {missing_pv}')
        extra_pv = sorted(set(provenance) - set(NC_PROVENANCE_KEYS))
        if extra_pv:
            out['problems'].append(
                'artifact.provenance carries field(s) outside the schema: '
                + ', '.join(map(str, extra_pv)))
    recorded = artifact.get('content_sha256')
    out['content_sha256'] = recorded if isinstance(recorded, str) else None
    if not isinstance(recorded, str) or not re.fullmatch(r'[0-9a-f]{64}', recorded):
        out['problems'].append(
            'artifact.content_sha256 must be a 64-hex string')
    else:
        try:
            replay_sha = _nc_content_sha(artifact)
        except Exception as exc:
            out['problems'].append(
                f'artifact cannot be hashed for replay: '
                f'{type(exc).__name__}: {exc}')
        else:
            if replay_sha != recorded:
                out['problems'].append(
                    'artifact.content_sha256 does not match the recorded body '
                    '(the artifact was edited after it was built)')
    result = artifact.get('result')
    result_problems = _nc_validate_result(result)
    if result_problems:
        out['problems'].extend(result_problems)
        return out
    out['result'] = result
    params = result.get('params')
    if isinstance(params, dict) and isinstance(inputs, dict) and inputs:
        mismatch = [k for k in NC_PARAM_KEYS
                    if k in inputs and inputs.get(k) != params.get(k)]
        if mismatch:
            out['problems'].append(
                'artifact.inputs disagree with result.params on '
                + ', '.join(mismatch))
    if out['problems']:
        return out
    if not replay:
        out['status'] = 'ok'
        return out
    try:
        replay_result = numerical_canary(
            seed=inputs.get('seed'), size=inputs.get('size'),
            dtype=inputs.get('dtype'), device=inputs.get('device'))
    except (TypeError, ValueError, RuntimeError) as exc:
        out['status'] = 'not_verifiable_locally'
        out['warnings'].append(
            f'cannot replay the recorded device here: {type(exc).__name__}: {exc}')
        return out
    out['replay'] = replay_result
    stored_id = _nc_result_identity(result)
    replay_id = _nc_result_identity(replay_result)
    out['arithmetic_match'] = (
        stored_id['fingerprint'] == replay_id['fingerprint']
        and stored_id['input_fingerprint'] == replay_id['input_fingerprint'])
    if not out['arithmetic_match']:
        out['status'] = 'drift'
        names = sorted(set(stored_id['ops']) | set(replay_id['ops']))
        for name in names:
            if stored_id['ops'].get(name) != replay_id['ops'].get(name):
                out['drifted_ops'].append(name)
        out['problems'].append(
            'the current process does not reproduce the recorded arithmetic '
            'fingerprint; drifted op(s): '
            + (', '.join(out['drifted_ops']) or 'fingerprint metadata'))
    stored_env = result.get('environment')
    replay_env = replay_result.get('environment')
    stored_env = stored_env if isinstance(stored_env, dict) else {}
    replay_env = replay_env if isinstance(replay_env, dict) else {}
    for key in sorted(set(stored_env) | set(replay_env)):
        if stored_env.get(key) != replay_env.get(key):
            out['environment_changes'].append(key)
    if out['environment_changes']:
        out['warnings'].append(
            'the recorded environment metadata differs on: '
            + ', '.join(out['environment_changes']))
    current_prov = numerical_canary_provenance()
    for key in NC_PROVENANCE_KEYS:
        if key in ('source_tree', 'env_fp_v2', 'env_fp_v3', 'lock_sha256'):
            old, new = provenance.get(key), current_prov.get(key)
            if key == 'source_tree':
                old = old.get('sha256') if isinstance(old, dict) else old
                new = new.get('sha256') if isinstance(new, dict) else new
            if old != new:
                out['provenance_changes'].append(key)
        elif provenance.get(key) != current_prov.get(key):
            out['provenance_changes'].append(key)
    if out['provenance_changes']:
        out['warnings'].append(
            'the recorded provenance differs from this checkout on: '
            + ', '.join(out['provenance_changes']))
    if out['status'] != 'drift':
        out['status'] = 'ok'
    return out


def write_numerical_canary_artifact(outdir, artifact, *, overwrite=False):
    """Atomically persist ``numerical_canary.json``; never silently replace."""
    if isinstance(outdir, os.PathLike):
        outdir = os.fspath(outdir)
    if not isinstance(outdir, str) or not outdir.strip():
        raise ValueError('outdir must be a non-empty path')
    verdict = validate_numerical_canary_artifact(artifact)
    path = os.path.join(outdir, 'numerical_canary.json')
    out = {'kind': 'write_numerical_canary', 'path': path, 'written': False,
           'unchanged': False, 'conflict': False, 'status': verdict['status'],
           'problems': list(verdict.get('problems') or [])}
    if verdict['status'] != 'ok':
        return out
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as handle:
                old = json.load(handle)
        except Exception:
            old = None
        if (isinstance(old, dict)
                and old.get('content_sha256') == artifact.get('content_sha256')):
            out['unchanged'] = True
            return out
        if not overwrite:
            out['conflict'] = True
            out['problems'].append(
                'a different numerical_canary.json already exists; pass '
                '--overwrite only after confirming the arithmetic change')
            return out
    os.makedirs(outdir, exist_ok=True)
    atomic_write_json(path, artifact, indent=1)  # noqa: F821 (part_08)
    out['written'] = True
    return out


def _nc_cli(args):
    """CLI for ``python -m explib.reproduce canary`` (see usage)."""
    import argparse
    parser = argparse.ArgumentParser(prog='python -m explib.reproduce canary',
                                     add_help=False)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--out', default=os.path.join('analysis_v12'))
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verify', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--size', type=int, default=NC_DEFAULT_SIZE)
    parser.add_argument('--dtype', default=NC_DEFAULT_DTYPE)
    parser.add_argument('--device', default=None)
    try:
        ns = parser.parse_args(args)
    except SystemExit:
        return 2
    if ns.verify is not None:
        path = os.path.abspath(ns.verify)
        try:
            with open(path, encoding='utf-8') as handle:
                artifact = json.load(handle)
        except (OSError, ValueError) as exc:
            print(json.dumps({
                'kind': 'numerical_canary_validation', 'status': 'fail',
                'problems': [f'{path}: {type(exc).__name__}: {exc}']},
                ensure_ascii=False, indent=2))
            return 2
        verdict = validate_numerical_canary_artifact(artifact)
        print(json.dumps(verdict, ensure_ascii=False, indent=2, sort_keys=True,
                         allow_nan=False, default=float))
        if verdict['status'] == 'ok':
            return 0
        return 1 if verdict['status'] in ('fail', 'drift') else 2
    try:
        result = numerical_canary(seed=ns.seed, size=ns.size,
                                  dtype=ns.dtype, device=ns.device)
    except (TypeError, ValueError, RuntimeError) as exc:
        print(json.dumps({'kind': 'numerical_canary', 'status': 'fail',
                          'error': f'{type(exc).__name__}: {exc}'},
                         ensure_ascii=False, indent=2))
        return 2
    if not ns.write:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True,
                         allow_nan=False, default=float))
        return 0
    artifact = build_numerical_canary_artifact(result=result)
    writer = write_numerical_canary_artifact(ns.out, artifact,
                                             overwrite=ns.overwrite)
    print(json.dumps(writer, ensure_ascii=False, indent=2, sort_keys=True,
                     allow_nan=False, default=float))
    return 0 if writer['status'] == 'ok' and not writer['conflict'] else 1
