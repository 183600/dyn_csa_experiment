
# Distributions that are packaging/development tooling, not part of the
# numerical runtime.  They are excluded from the "is every installed
# distribution pinned?" check so installing a linter does not make a healthy
# experiment environment look partial.  Keep this list explicit: an unpinned
# package that is NOT here is treated as a possible numerical dependency and
# enters `environment_fingerprint_v3`.
_TOOLING_DISTRIBUTIONS = frozenset({
    'pip', 'setuptools', 'wheel', 'pkg-resources', 'distribute',
    'ruff', 'mypy', 'black', 'flake8', 'pyflakes', 'isort', 'coverage',
    'pytest-cov', 'tox', 'nox', 'pre-commit', 'build', 'twine',
    # The project itself, when a local wheel/install-check has registered it
    # (dyn_csa_experiment.egg-info): the code under test, not a dependency.
    'dyn-csa-experiment',
})


def dependency_contract(lock_path=None):
    """Compare the declared lock with the versions actually installed.

    A lock that is present but incomplete or stale is the reproducibility
    failure mode a release must expose: the same git commit then runs on a
    different numerical stack.  ``partial`` means every pinned entry matches
    but extra installed distributions are not pinned, or the lock does not
    declare the interpreter/platform (the lock is not a full environment
    record).  ``mismatch`` also covers a declared interpreter/platform that
    differs from the running one.  Tooling (pip/setuptools/wheel and the
    editors/linters in ``_TOOLING_DISTRIBUTIONS``) is not a runtime
    dependency and is excluded from the coverage check.  ``unpinned_versions``
    keeps the version of every remaining unpinned distribution so the strict
    ``environment_fingerprint_v3`` can bind it.
    """
    try:
        parsed = parse_requirements_lock(lock_path)
    except OSError as e:
        # Provenance must never crash a training run because a lock file is
        # unreadable; report it as an explicit contract failure instead.
        return {'kind': 'dependency_contract',
                'lock_path': os.path.abspath(str(lock_path or 'requirements.lock')),
                'lock_sha256': None, 'n_pinned': 0, 'options': [],
                'mismatches': [], 'missing': [], 'unpinned_installed': [],
                'unpinned_versions': {}, 'pinned_installed': {},
                'interpreter': _interpreter_contract(),
                'invalid': [f'could not read the lock: {type(e).__name__}: {e}'],
                'status': 'invalid',
                'note': 'the lock file exists but could not be read'}
    out = {'kind': 'dependency_contract', 'lock_path': parsed['path'],
           'lock_sha256': parsed['sha256'], 'n_pinned': len(parsed['pins']),
           'options': parsed['options'], 'mismatches': [], 'missing': [],
           'pinned_installed': {},
           'interpreter': _interpreter_contract(parsed.get('environment')),
           'invalid': list(parsed['invalid']), 'unpinned_installed': [],
           'unpinned_versions': {}, 'status': 'ok', 'note': ''}
    if not parsed['exists']:
        out['status'] = 'missing_lock'
        out['note'] = 'requirements.lock is absent; the environment is not pinned'
        return out
    if parsed['invalid']:
        out['status'] = 'invalid'
        out['note'] = 'the lock has unparseable/ambiguous entries'
        return out
    installed = {}
    try:
        from importlib import metadata as _md
        for dist in _md.distributions():
            try:
                name = dist.metadata['Name']
                if name:
                    installed[re.sub(r'[-_.]+', '-', str(name)).lower()] = str(dist.version)
            except Exception:
                continue
    except Exception as e:
        out['status'] = 'invalid'
        out['unpinned_versions'] = {}
        out['invalid'].append(f'importlib.metadata unavailable: {type(e).__name__}: {e}')
        return out
    for name, ent in sorted(parsed['pins'].items()):
        got = installed.get(name)
        # `pinned_installed` is the exact lock-declared stack this process
        # actually has.  It is the material `environment_fingerprint_v2`
        # hashes: the legacy `exec_fp` covers only six direct packages, so a
        # transitive numerical dependency (or a transitive pin resolving to a
        # different wheel) could otherwise collide while the environment is
        # different.  A missing pin is recorded as null, never dropped: an
        # absent distribution is part of the environment identity.
        out['pinned_installed'][name] = None if got is None else str(got)
        if got is None:
            out['missing'].append({'name': ent['name'], 'expected': ent['version']})
        elif str(got) != str(ent['version']):
            out['mismatches'].append({'name': ent['name'], 'expected': ent['version'],
                                      'installed': str(got)})
    out['unpinned_installed'] = sorted(
        n for n in installed if n not in parsed['pins']
        and n not in _TOOLING_DISTRIBUTIONS)
    # The version of every unpinned runtime distribution is part of the
    # environment identity: two machines can have the same pinned lock and
    # different unpinned numerical packages (or versions of them).  Keep the
    # historical list for readers and add the name -> version mapping for the
    # strict v3 digest.
    out['unpinned_versions'] = {n: installed[n]
                                for n in out['unpinned_installed']}
    out['n_installed'] = len(installed)
    out['n_unpinned_installed'] = len(out['unpinned_installed'])
    _interp = out['interpreter']
    _version_bad = bool(out['mismatches'] or out['missing'])
    _interp_bad = _interp.get('status') == 'mismatch'
    _interp_missing = list(_interp.get('missing') or [])
    if _version_bad or _interp_bad:
        out['status'] = 'mismatch'
        if _version_bad and _interp_bad:
            out['note'] = ('installed versions and the running '
                           'interpreter/platform do not match the lock')
        elif _interp_bad:
            out['note'] = 'the running interpreter/platform does not match the lock'
        else:
            out['note'] = 'installed versions do not match the lock'
    elif (out['unpinned_installed'] or _interp.get('status') == 'unverified'
          or _interp_missing):
        out['status'] = 'partial'
        _why = []
        if out['unpinned_installed']:
            _why.append('the lock does not pin all installed runtime distributions')
        if _interp.get('status') == 'unverified':
            _why.append('the lock does not declare the interpreter/platform')
        elif _interp_missing:
            _why.append('the lock does not declare ' + ', '.join(_interp_missing)
                        + ' (the interpreter/platform contract is incomplete)')
        out['note'] = ('every pinned entry matches, but ' + ' and '.join(_why))
    else:
        out['note'] = ('every installed runtime distribution and the '
                       'interpreter/platform are pinned and match '
                       '(pip/setuptools/wheel excluded as tooling)')
    return out


_INVOCATION_SECRET_RE = re.compile(
    r'(?i)(token|secret|passw(?:or)?d|api[-_]?key|credential|auth)')


def _invocation_identity(argv=None):
    """The exact command that produced a record, with secret values redacted.

    A seed/config/fingerprint does not tell a reader which driver and mode were
    invoked.  Recording argv closes that gap for a rerun, while a value passed
    after a token/password/api-key flag is replaced by ``***`` so provenance
    does not become a credential leak.  Callers that need the unredacted argv
    (e.g. a deliberate audit) can still inspect ``sys.argv`` themselves.
    """
    raw = [str(a) for a in (sys.argv if argv is None else argv)]
    out = []
    redact_next = False
    for text in raw:
        if redact_next:
            out.append('***')
            redact_next = False
            continue
        if text.startswith('-') and '=' in text:
            name = text.partition('=')[0]
            if _INVOCATION_SECRET_RE.search(name):
                out.append(name + '=***')
                continue
        out.append(text)
        if text.startswith('-') and _INVOCATION_SECRET_RE.search(text):
            redact_next = True
    module = None
    if '-m' in raw:
        try:
            module = raw[raw.index('-m') + 1]
        except IndexError:
            module = None
    return {
        'argv': out,
        'entrypoint': out[0] if out else None,
        'python_module': module,
        'n_args': len(out),
        'redacted': out != raw,
    }


ENVIRONMENT_FP_SCHEMA = 2


def _os_stack_identity():
    """OS / kernel / C library / driver identity not captured by wheel pins.

    ``platform.platform()`` is deliberately not enough: two boxes can report
    the same Python and wheel versions while differing in the kernel, libc or
    NVIDIA driver, and the last two can change numerical kernels.  Every probe
    is best-effort and returns ``None`` rather than raising, because this block
    is provenance, not a gate: a missing value must stay visibly unverified
    instead of crashing a run or being filled with a guessed placeholder.
    """
    import platform
    libc_name = libc_version = None
    try:
        libc_name, libc_version = platform.libc_ver()
    except Exception:
        pass
    driver = os.environ.get('NVIDIA_DRIVER_VERSION') or None
    if driver is None:
        # Linux exposes the loaded NVIDIA kernel module here; on other
        # platforms the absence is expected and stays None.
        try:
            with open('/proc/driver/nvidia/version', encoding='utf-8',
                      errors='replace') as fh:
                line = fh.readline().strip()
            driver = line or None
        except OSError:
            driver = None
    cpu_model = None
    try:
        with open('/proc/cpuinfo', encoding='utf-8', errors='replace') as fh:
            for line in fh:
                if line.lower().startswith('model name'):
                    cpu_model = line.split(':', 1)[1].strip() or None
                    break
    except OSError:
        pass
    return {
        'system': platform.system() or None,
        'release': platform.release() or None,
        'version': platform.version() or None,
        'machine': platform.machine() or None,
        'libc_name': libc_name or None,
        'libc_version': libc_version or None,
        'cpu_model': cpu_model or platform.processor() or None,
        'nvidia_driver': driver,
    }


def environment_fingerprint_v2(env=None):
    """Strict identity of the *declared* numerical stack of a process.

    Why a second fingerprint next to ``execution_fingerprint`` (``exec_fp``):
    the legacy digest hashes only the six direct packages in
    ``_dependency_fingerprint``.  ``requirements.lock`` pins the full
    transitive closure, so two runs can differ in a transitive numerical
    package (or in a pin that resolved to a different wheel) and still share
    ``exec_fp``.  The cross-run reproduction comparator would then report the
    environment as "matching" when the actual stack differed.

    ``env_fp_v2`` hashes, without any host/process identity:

    * the lock bytes (``lock_sha256``), so a changed declaration is visible
      even when the current install still satisfies it;
    * every lock-pinned distribution and its installed version (a missing pin
      stays ``null``);
    * the interpreter/platform contract and the OS/kernel/libc/driver block;
    * torch/numpy/CUDA/cuDNN builds, GPU model + compute capability, precision
      flags, determinism flags and CPU thread counts.

    Hostname, process id, working directory, ``CUDA_VISIBLE_DEVICES`` (a
    scheduling choice, not a stack identity) and per-run hash seeds are
    excluded, so two executions on the same declared stack produce the same
    digest.  Unpinned *tooling* is excluded too: installing a linter must not
    look like an environment change.  The result is a 16-hex digest for
    consistency with ``exec_fp``; it is an identity, not a security hash.
    """
    env = environment_snapshot() if env is None else env
    if not isinstance(env, dict):
        raise TypeError(
            f'environment_fingerprint_v2: env must be a mapping, got {type(env).__name__}')
    contract = env.get('dependency_contract')
    if not isinstance(contract, dict):
        contract = dependency_contract()
    pinned = contract.get('pinned_installed')
    if not isinstance(pinned, dict):
        # A caller may hand in an older snapshot whose contract predates
        # `pinned_installed`; recompute rather than silently hashing an empty
        # stack (which would make every such process look identical).  Reuse
        # the recorded lock path so a custom-lock snapshot is not silently
        # re-identified against the default requirements.lock.
        contract = dependency_contract(contract.get('lock_path'))
        pinned = contract.get('pinned_installed')
        if not isinstance(pinned, dict):
            pinned = {}
    gpus = []
    for gpu in env.get('gpu_identity') or []:
        if not isinstance(gpu, dict):
            continue
        gpus.append({k: gpu.get(k) for k in
                     ('name', 'major', 'minor', 'total_memory_bytes')})
    os_identity = env.get('os_identity')
    if not isinstance(os_identity, dict):
        os_identity = _os_stack_identity()
    interpreter = contract.get('interpreter')
    if not isinstance(interpreter, dict):
        interpreter = {}
    missing_or_mismatched = bool(contract.get('missing') or contract.get('mismatches')
                                 or contract.get('invalid'))
    fields = {
        'schema': ENVIRONMENT_FP_SCHEMA,
        'lock_sha256': contract.get('lock_sha256'),
        'lock_complete': bool(contract.get('lock_sha256') and not missing_or_mismatched),
        'n_pinned': contract.get('n_pinned'),
        'pinned': {name: pinned[name] for name in sorted(pinned)},
        'interpreter': {
            'declared': interpreter.get('declared') or {},
            'actual': interpreter.get('actual') or {},
            'status': interpreter.get('status'),
        },
        'os': os_identity,
        'python': env.get('python'),
        'platform': env.get('platform'),
        'machine': env.get('machine'),
        'processor': env.get('processor'),
        'cpu_count': env.get('cpu_count'),
        'torch': env.get('torch'),
        'torch_git_version': env.get('torch_git_version'),
        'numpy': env.get('numpy'),
        'tokenizers': env.get('tokenizers'),
        'datasets': env.get('datasets'),
        'tensorflow': env.get('tensorflow'),
        'cuda_available': env.get('cuda_available'),
        'cuda_version': env.get('cuda_version'),
        'cudnn_version': env.get('cudnn_version'),
        'cuda_device_count': env.get('cuda_device_count'),
        'gpus': gpus,
        'determinism': env.get('determinism'),
        'cudnn_deterministic': env.get('cudnn_deterministic'),
        'cudnn_benchmark': env.get('cudnn_benchmark'),
        'deterministic_algorithms': env.get('deterministic_algorithms'),
        'tf32_matmul': env.get('tf32_matmul'),
        'tf32_cudnn': env.get('tf32_cudnn'),
        'fp16_reduced_precision_reduction': env.get('fp16_reduced_precision_reduction'),
        'bf16_reduced_precision_reduction': env.get('bf16_reduced_precision_reduction'),
        'float32_matmul_precision': env.get('float32_matmul_precision'),
        'cublas_workspace_config': env.get('cublas_workspace_config'),
        'cpu_threads': env.get('cpu_threads'),
        'cpu_interop_threads': env.get('cpu_interop_threads'),
    }
    blob = json.dumps(fields, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, default=str)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]


ENVIRONMENT_FP_V3_SCHEMA = 3


def _numpy_build_identity():
    """Build/runtime identity of the NumPy numerical backend.

    ``numpy.__version__`` pins the wheel, not the BLAS/LAPACK implementation
    inside it or the CPU dispatch it selects.  A conda build with MKL and a
    pip build with OpenBLAS can both report the same version; their reductions
    and matmuls can differ in the last bits, which is enough to move a PPL
    comparison.  This block records the BLAS/LAPACK name+version, the build's
    SIMD baseline/targets, the runtime dispatched features and a digest of the
    whole build configuration.  Every probe is best-effort: a missing field
    stays ``None`` instead of crashing provenance.
    """
    out = {'version': getattr(np, '__version__', None), 'blas': None,
           'lapack': None, 'simd_baseline': None, 'simd_found': None,
           'simd_runtime_dispatch': None, 'simd_runtime_features': None,
           'build_config_sha256': None}
    try:
        cfg = getattr(getattr(np, '__config__', None), 'CONFIG', None)
        if isinstance(cfg, dict):
            deps = cfg.get('Build Dependencies')
            deps = deps if isinstance(deps, dict) else {}
            for key in ('blas', 'lapack'):
                dep = deps.get(key)
                if not isinstance(dep, dict):
                    continue
                out[key] = {
                    'name': dep.get('name'),
                    'version': dep.get('version'),
                    'found': dep.get('found'),
                    'openblas_config': dep.get('openblas configuration'),
                }
            simd = cfg.get('SIMD Extensions')
            if isinstance(simd, dict):
                if isinstance(simd.get('baseline'), list):
                    out['simd_baseline'] = [str(x) for x in simd['baseline']]
                if isinstance(simd.get('found'), list):
                    out['simd_found'] = [str(x) for x in simd['found']]
            out['build_config_sha256'] = hashlib.sha256(
                json.dumps(cfg, sort_keys=True, separators=(',', ':'),
                           ensure_ascii=True, default=str).encode('utf-8')
            ).hexdigest()[:16]
    except Exception:
        pass
    try:
        try:
            from numpy._core import _multiarray_umath as _mu
        except ImportError:  # NumPy < 2
            from numpy.core import _multiarray_umath as _mu
        features = getattr(_mu, '__cpu_features__', None)
        if isinstance(features, dict):
            out['simd_runtime_features'] = sorted(
                str(k) for k, enabled in features.items() if enabled)
        dispatch = getattr(_mu, '__cpu_dispatch__', None)
        if isinstance(dispatch, (list, tuple, set, frozenset)):
            out['simd_runtime_dispatch'] = sorted(str(x) for x in dispatch)
    except Exception:
        pass
    return out


def _torch_build_identity():
    """Build/runtime identity of the PyTorch numerical backend.

    The wheel pin does not say whether this build uses MKL, oneDNN, OpenMP or
    a particular LAPACK, nor which CPU capability path ATen dispatches to.
    The full build report is hashed, and the human-readable version lines are
    kept so a reader can see *what* differs without re-running torch.
    """
    out = {'build_report_sha256': None, 'mkl': None, 'mkldnn': None,
           'openmp': None, 'mkl_version': None, 'mkldnn_version': None,
           'openmp_version': None, 'blas_info': None, 'lapack_info': None,
           'cpu_capability': None, 'commit_sha': None,
           'cuda_build': getattr(torch.version, 'cuda', None)}
    for name in ('mkl', 'mkldnn', 'openmp'):
        try:
            backend = getattr(torch.backends, name, None)
            out[name] = bool(backend.is_available()) if backend is not None else None
        except Exception:
            out[name] = None
    try:
        import contextlib
        import io
        cfg = getattr(torch, '__config__', None)
        show = getattr(cfg, 'show', None)
        if not callable(show):
            return out
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ret = show()
        text = ret if isinstance(ret, str) else buf.getvalue()
        text = str(text or '')
    except Exception:
        return out
    if not text.strip():
        return out
    out['build_report_sha256'] = hashlib.sha256(
        text.encode('utf-8', 'replace')).hexdigest()[:16]

    def _first(pattern):
        try:
            m = re.search(pattern, text)
            return m.group(1).strip() if m else None
        except Exception:
            return None
    out['mkl_version'] = _first(
        r'oneAPI Math Kernel Library Version ([^\n]+?)\s+for ')
    out['mkldnn_version'] = _first(r'MKL-DNN v([^\s]+)')
    out['openmp_version'] = _first(r'OpenMP (\d+)')
    out['blas_info'] = _first(r'BLAS_INFO=([^,\s]+)')
    out['lapack_info'] = _first(r'LAPACK_INFO=([^,\s]+)')
    out['cpu_capability'] = _first(r'CPU capability usage:\s*([^\n]+)')
    out['commit_sha'] = _first(r'COMMIT_SHA=([0-9a-fA-F]+)')
    try:
        out['cpu_capability'] = out['cpu_capability'] or str(
            torch.backends.cpu.get_cpu_capability())
    except Exception:
        pass
    return out


def _numerical_backend_identity():
    """BLAS/LAPACK/SIMD identity that a wheel version does not capture."""
    return {'numpy': _numpy_build_identity(), 'torch': _torch_build_identity()}


def environment_fingerprint_v3(env=None):
    """Strict numerical-stack identity, extending ``env_fp_v2``.

    ``env_fp_v2`` binds the lock bytes and every lock-pinned distribution, but
    two processes can still share it while differing in

    * an **unpinned runtime distribution** (and its version) that happens to
      be installed next to the lock, and
    * the **numerical backend inside the pinned wheels**: NumPy's BLAS/LAPACK
      build and SIMD dispatch, PyTorch's MKL/oneDNN/OpenMP build and CPU
      capability path.

    Either difference can change floating-point reductions while every
    declared version matches.  v3 therefore hashes the v2 digest, the
    name -> version map of every installed distribution that is neither
    pinned nor declared tooling, and the backend identity above.  It is a
    separate digest rather than a redefinition of v2 so archived records that
    carry only v2 remain comparable: a caller prefers v3 only when both sides
    have it.

    Hostname, PID, cwd, hash seed and scheduling variables are still excluded;
    this is a stack identity, not a process identity.
    """
    env = environment_snapshot() if env is None else env
    if not isinstance(env, dict):
        raise TypeError(
            'environment_fingerprint_v3: env must be a mapping, got '
            f'{type(env).__name__}')
    contract = env.get('dependency_contract')
    if not isinstance(contract, dict):
        contract = dependency_contract()
    unpinned = contract.get('unpinned_versions')
    if not isinstance(unpinned, dict):
        # Older/partial snapshots expose only the name list.  Record null
        # versions rather than dropping the names: presence still matters.
        names = contract.get('unpinned_installed')
        if isinstance(names, list):
            unpinned = {str(n): None for n in names}
        else:
            unpinned = {}
    unpinned = {str(k): (None if v is None else str(v))
                for k, v in unpinned.items()}
    backend = env.get('numerical_backend')
    if not isinstance(backend, dict):
        backend = _numerical_backend_identity()
    fields = {
        'schema': ENVIRONMENT_FP_V3_SCHEMA,
        'env_fp_v2': environment_fingerprint_v2(env),
        'unpinned_versions': {k: unpinned[k] for k in sorted(unpinned)},
        'n_unpinned': len(unpinned),
        'numerical_backend': backend,
    }
    blob = json.dumps(fields, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, default=str)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]


def environment_snapshot():
    """Compact software/hardware snapshot for experiment tracking."""
    import platform
    import socket
    try:
        _hostname = socket.gethostname()
    except Exception:
        _hostname = None
    env = {
        'python': sys.version.split()[0],
        'invocation': _invocation_identity(),
        'hostname': _hostname,
        'process_id': os.getpid(),
        'working_directory': os.getcwd(),
        'executable': sys.executable,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'cpu_count': os.cpu_count(),
        'torch': torch.__version__,
        'torch_git_version': getattr(torch.version, 'git_version', None),
        'numpy': np.__version__,
        'cuda_available': bool(torch.cuda.is_available()),
        'cuda_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        'gpu_names': _gpu_names(),
        'gpu_identity': _gpu_identity(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'determinism': determinism_label(),
        'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
        'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
        'deterministic_algorithms': bool(torch.are_deterministic_algorithms_enabled()),
        'tf32_matmul': bool(torch.backends.cuda.matmul.allow_tf32) if torch.cuda.is_available() else None,
        'tf32_cudnn': bool(torch.backends.cudnn.allow_tf32) if torch.cuda.is_available() else None,
        'fp16_reduced_precision_reduction': bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction) if torch.cuda.is_available() else None,
        'bf16_reduced_precision_reduction': bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction) if torch.cuda.is_available() else None,
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'cpu_threads': int(torch.get_num_threads()),
        'cpu_interop_threads': int(torch.get_num_interop_threads()) if hasattr(torch, 'get_num_interop_threads') else None,
        'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
        'pythonhashseed_at_startup': _PYTHONHASHSEED_AT_STARTUP,
        'pythonhashseed_pinned_at_startup': bool(
            _hashseed_pinned_value(_PYTHONHASHSEED_AT_STARTUP)),
        'hashseed_contract': hashseed_contract(),
        'tensorflow': getattr(sys.modules.get('tensorflow'), '__version__', None),
        'dependencies': _dependency_fingerprint(),
        'dependency_contract': dependency_contract(),
        'flags': {k: os.environ.get(k) for k in ('CSA_KERNEL', 'CSA_CKPT', 'CSA_DETERMINISTIC', 'CSA_CPU_THREADS', 'CSA_SKIP_OVERFIT', 'CSA_REQUIRE_REPRO', 'CSA_SKIP_REPRO', 'CUBLAS_WORKSPACE_CONFIG')},
        # Every run-control variable, not just the historically documented
        # subset: an unrecorded CSA_*/WT103_*/V12_* flag is an unexplained
        # difference between two otherwise identical runs.  Secrets (tokens,
        # passwords) do not use these prefixes and are never collected.
        'env_flags': {k: os.environ.get(k) for k in sorted(os.environ)
                      if k.startswith(('CSA_', 'WT103_', 'FWEDU_', 'M0_', 'V12_'))},
    }
    for _mod, _key in (('tokenizers', 'tokenizers'), ('datasets', 'datasets')):
        try:
            env[_key] = __import__(_mod).__version__
        except Exception:
            env[_key] = None
    # OS-level identity is recorded separately from the digest so an auditor
    # can see which field changed; `env_fp_v2` binds the same block (and the
    # full lock-pinned stack) into one comparable value.  v3 adds the
    # numerical backend and unpinned runtime distributions; both are recorded
    # as structured fields so a differing digest can be explained.
    env['os_identity'] = _os_stack_identity()
    env['numerical_backend'] = _numerical_backend_identity()
    env['env_fp_v2'] = environment_fingerprint_v2(env)
    env['env_fp_v3'] = environment_fingerprint_v3(env)
    env['env_fp_v3_schema'] = ENVIRONMENT_FP_V3_SCHEMA
    return env

def execution_fingerprint(env=None):
    """Fingerprint of the hardware/software/determinism class of a run.

    Same-seed replication is only guaranteed on the same stack: a different
    GPU architecture, CUDA/cuDNN build, precision policy or thread count can
    change every result.  Paired seed contrasts therefore carry this digest,
    and `pair_reason` refuses two records whose digests disagree.  Hostname and
    process id are deliberately excluded so two machines of the same class can
    still be compared; the GPU UUID is excluded for the same reason (hardware
    model and compute capability are the reproducibility-relevant part).
    """
    env = environment_snapshot() if env is None else env
    gpus = []
    for g in env.get('gpu_identity') or []:
        if not isinstance(g, dict):
            continue
        gpus.append({k: g.get(k) for k in ('name', 'major', 'minor', 'total_memory_bytes')})
    deps = env.get('dependencies') or {}
    fields = {
        'python': env.get('python'),
        'platform': env.get('platform'),
        'machine': env.get('machine'),
        'processor': env.get('processor'),
        'cpu_count': env.get('cpu_count'),
        'torch': env.get('torch'),
        'torch_git_version': env.get('torch_git_version'),
        'numpy': env.get('numpy'),
        'cuda_available': env.get('cuda_available'),
        'cuda_version': env.get('cuda_version'),
        'cudnn_version': env.get('cudnn_version'),
        'cuda_device_count': env.get('cuda_device_count'),
        'gpus': gpus,
        'determinism': env.get('determinism'),
        'cudnn_deterministic': env.get('cudnn_deterministic'),
        'cudnn_benchmark': env.get('cudnn_benchmark'),
        'deterministic_algorithms': env.get('deterministic_algorithms'),
        'tf32_matmul': env.get('tf32_matmul'),
        'tf32_cudnn': env.get('tf32_cudnn'),
        'fp16_reduced_precision_reduction': env.get('fp16_reduced_precision_reduction'),
        'bf16_reduced_precision_reduction': env.get('bf16_reduced_precision_reduction'),
        'float32_matmul_precision': env.get('float32_matmul_precision'),
        'cublas_workspace_config': env.get('cublas_workspace_config'),
        'cpu_threads': env.get('cpu_threads'),
        'cpu_interop_threads': env.get('cpu_interop_threads'),
        'packages': deps.get('packages') if isinstance(deps, dict) else None,
    }
    blob = json.dumps(fields, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]

def _environment_fp_v2(rec):
    """Read the strict environment digest from a record, if it carries one."""
    if not isinstance(rec, dict):
        return None
    value = rec.get('env_fp_v2')
    if value is None and isinstance(rec.get('hyperparams'), dict):
        value = rec['hyperparams'].get('env_fp_v2')
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _environment_fp_v3(rec):
    """Read the numerical-backend environment digest, or None.

    v3 is intentionally separate from v2.  A record written before v3 keeps a
    valid v2 identity; the absence of v3 is unverified, never a mismatch.
    """
    if not isinstance(rec, dict):
        return None
    value = rec.get('env_fp_v3')
    if value is None and isinstance(rec.get('hyperparams'), dict):
        value = rec['hyperparams'].get('env_fp_v3')
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def execution_environment_reason(ra, rb):
    """Why two records cannot attest same-environment replication, or None.

    Both-present is required for a verdict; legacy records without the digest
    keep the historical permissive path.  This is only about the execution
    environment, not the seed, data or recipe (those have their own gates).

    Preference order is ``env_fp_v3`` -> ``env_fp_v2`` -> ``exec_fp``.  v3
    adds the BLAS/LAPACK/SIMD backend and unpinned runtime distributions; v2
    covers every lock-pinned distribution, the lock bytes and the OS/driver
    block; both are stricter than the legacy six-package ``exec_fp``.  A
    one-sided newer digest is not a mismatch here (a seed-completion run may
    pair with an older arm), so the comparison falls back to the strongest
    digest both sides carry.  The reproduction comparator still reports the
    missing layer as unverified rather than a match.
    """
    def _fp(rec):
        v = rec.get('exec_fp')
        if v is None and isinstance(rec.get('hyperparams'), dict):
            v = rec['hyperparams'].get('exec_fp')
        return v
    # Prefer v3 when both sides carry it: it includes the v2 digest plus the
    # BLAS/LAPACK/SIMD backend and any unpinned runtime distribution.  A
    # one-sided v3 is a generation difference, not evidence of a different
    # stack, so it falls back to v2 exactly as the v2 introduction did.
    a3, b3 = _environment_fp_v3(ra), _environment_fp_v3(rb)
    if a3 is not None and b3 is not None:
        if a3 != b3:
            return (f'different declared environment (env_fp_v3 {a3[:12]}… vs '
                    f'{b3[:12]}…): lock-pinned packages, numerical backend '
                    '(BLAS/LAPACK/SIMD), unpinned runtime distributions, '
                    'interpreter or OS/driver stack differ; same-seed '
                    'replication is not controlled')
        return None
    a2, b2 = _environment_fp_v2(ra), _environment_fp_v2(rb)
    if a2 is not None and b2 is not None and a2 != b2:
        return (f'different declared environment (env_fp_v2 {a2[:12]}… vs '
                f'{b2[:12]}…): lock-pinned packages, interpreter or OS/driver '
                'stack differ; same-seed replication is not controlled')
    a, b = _fp(ra), _fp(rb)
    ha = a is not None and str(a) != ''
    hb = b is not None and str(b) != ''
    if ha and hb and str(a) != str(b):
        return (f'different execution environment ({str(a)[:12]}… vs '
                f'{str(b)[:12]}…): same-seed replication is not controlled '
                'across hardware/driver/precision classes')
    return None

def _corpus_prefix_fp(train_ids, cap=None):
    """Fingerprint of the tokenizer-fit prefix, independent of training budget.

    `train_fp` hashes the *served* training stream, so a 5B-token tier and a
    12B-token tier over the same corpus necessarily disagree even when they
    share one tokenizer.  The first `TOKENIZER_FIT_TOKENS` ids are exactly the
    prefix the tokenizer was fitted on, and extending the training cap leaves
    that prefix unchanged.  Hash only that prefix so a scale ladder can attest
    "same corpus/tokenizer" without also requiring "same training volume".
    """
    if train_ids is None:
        return None
    try:
        arr = np.asarray(train_ids)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    _cap = TOKENIZER_FIT_TOKENS if cap is None else int(cap)
    if _cap <= 0 or arr.size == 0:
        return None
    return _ids_fp(np.ascontiguousarray(arr[:min(int(arr.size), _cap)]))

def _corpus_fp_from_data(data):
    """Budget-independent corpus/tokenizer identity.

    Deliberately excludes `requested_train_tokens`, `train_tokens`, `train_fp`
    and the split lengths: those change when a scale tier trains longer, but
    they do not change the corpus or tokenizer.  The tokenizer-fit prefix
    fingerprint covers the text actually used to fit the BPE (and therefore
    distinguishes different tokenisations even at equal vocabulary width).
    """
    keys = ('dataset', 'source', 'fw_fit_docs', 'vocab_size',
            'corpus_prefix_fp', 'data_extra')
    payload = {k: data.get(k) for k in keys if data.get(k) is not None}
    if not payload.get('corpus_prefix_fp'):
        return None
    blob = json.dumps(_jsonable(payload), sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]

def _data_version(data):
    """Stable content digest of the corpus identity recorded in a panel.

    The token fingerprints already hash the exact ids seen; this composite
    makes the *data version* a single field a reader can compare across runs
    (dataset/protocol, token counts, vocabulary and both split fingerprints)
    without knowing which individual fields the loader happened to expose.
    `corpus_fp` (when present) is the budget-independent subset used by the
    scale ladder; `data_version` still identifies the exact served bytes.
    """
    keys = ('dataset', 'source', 'fw_fit_docs', 'requested_train_tokens', 'train_tokens', 'seq_len', 'vocab_size', 'val_sequences', 'train_fp', 'val_fp', 'eval_fp', 'cache_val_fp', 'corpus_fp', 'data_extra')
    payload = {k: data.get(k) for k in keys if data.get(k) is not None}
    blob = json.dumps(_jsonable(payload), sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]

def code_provenance():
    """Code + environment provenance block, without any data section.

    Shared by `experiment_provenance` (training/panel records, which append the
    data fingerprints) and by `stamped_analysis` (derived/audit artifacts,
    which have no data of their own — their inputs are other stamped records
    or on-disk panels, so a `data` block here would be misleading)."""
    code = {'code_semantics': CODE_SEMANTICS, 'code_stamp': code_stamp(),
            'kernel_stamp': _kernel_stamp(), **_git_provenance()}
    # Content identity that survives a source tarball / dirty work tree.  It is
    # recorded after the git fields so a reader can compare both: the commit
    # answers "which revision", the tree digest answers "exactly which bytes".
    code['source_tree'] = source_tree_summary()
    return {'code': code, 'environment': environment_snapshot()}

def experiment_provenance(cfg, *, train_ids=None, val_batch=None, vocab=None):
    """Everything needed to identify a panel: code version (semantic stamp
    + git commit + dirty-worktree digest), environment, and the exact data
    bytes seen (token fingerprints + a composite data version)."""
    data = {'dataset': cfg.get('dataset') or 'wikitext', 'requested_train_tokens': cfg.get('n_train_tokens'), 'seq_len': cfg.get('seq_len')}
    for _k in ('cache_dir', 'fw_source', 'fw_fit_docs'):
        if cfg.get(_k) is not None:
            data[_k if _k != 'fw_source' else 'source'] = cfg.get(_k)
    if train_ids is not None:
        data['train_tokens'] = int(len(train_ids))
        data['train_fp'] = _ids_fp(train_ids)
        data['corpus_prefix_fp'] = _corpus_prefix_fp(train_ids)
    if val_batch is not None:
        _vb = np.ascontiguousarray(val_batch).reshape(-1)
        data['val_sequences'] = int(np.asarray(val_batch).shape[0])
        data['val_fp'] = _ids_fp(_vb)
    if vocab is not None:
        data['vocab_size'] = int(vocab)
    _extra = cfg.get('data_extra')
    if isinstance(_extra, dict):
        # Synthetic/task-specific generators (NIAH recipe, corruption modes)
        # must be part of the data identity, not only the human note.
        data['data_extra'] = _jsonable(_extra)
    # Record the source the loader is PINNED to.  The cache metadata records
    # what the bytes were actually built from; this field makes the intent
    # visible in every run/checkpoint record, and `audit_gates data_source`
    # checks the two against each other.  It is deliberately not added to
    # `_data_version` (the content fingerprints already bind the bytes).
    try:
        if data['dataset'] == 'wikitext':
            data['configured_dataset_source'] = wikitext_dataset_source()
        elif data['dataset'] == 'fineweb':
            data['configured_dataset_source'] = fineweb_cache_source(
                data.get('source') or 'sample-10BT')
    except ValueError:
        raise
    except Exception as _dse:
        data['configured_dataset_source'] = {
            'status': 'unresolved',
            'error': f'{type(_dse).__name__}: {_dse}',
        }
    # Compute the composite fingerprint AFTER data_extra is present:
    # `_corpus_fp_from_data` deliberately includes that key, so computing it
    # first made two distinct task/corruption corpora carry the same corpus_fp
    # (the field used by cross-scale corpus binding).
    _cfp = _corpus_fp_from_data(data)
    if _cfp is not None:
        data['corpus_fp'] = _cfp
    data['data_version'] = _data_version(data)
    prov = code_provenance()
    prov['data'] = data
    return prov

def run_hyperparameters(cfg, *, seed, d, n_layers, n_heads, d_head, vocab, mlp_ratio, data_version=None, amp_dtype='auto', eager_batch='auto'):
    """Structured hyper-parameter record (the run_cfg string stays the
    pairing gate; this is the human/machine-readable copy).

    ``amp_dtype`` / ``eager_batch`` must be the values the cell ACTUALLY ran,
    not a fresh inference from the current process's global device.  Callers
    that already have the training record pass its two fields here."""
    hp = {str(k): _jsonable(v) for k, v in cfg.items() if k not in ('seeds', 'variants', 'outdir')}
    hp.update({
        'seed': int(seed),
        'data_seed': int(seed),
        'rng': {
            'python': int(seed),
            'numpy': int(seed),
            'torch': int(seed),
            'cuda': int(seed),
            'tensorflow': int(seed),
            'data': int(seed),
            'eval_sampling': 0,
            'dataloader_generator': int(seed),
            'worker_seed_policy': 'torch.initial_seed() % 2**32',
        },
        'd': int(d),
        'n_layers': int(n_layers),
        'n_heads': int(n_heads),
        'd_head': int(d_head),
        'vocab_size': int(vocab),
        'mlp_ratio': float(mlp_ratio),
        'dataset': cfg.get('dataset') or 'wikitext',
        'exec_fp': execution_fingerprint(),
        'env_fp_v2': environment_fingerprint_v2(),
        'env_fp_v3': environment_fingerprint_v3(),
        'env_fp_v3_schema': ENVIRONMENT_FP_V3_SCHEMA,
        'train_knobs': training_knobs(amp_dtype=amp_dtype, eager_batch=eager_batch),
        'determinism': {
            'level': determinism_label(),
            'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
            'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
            'deterministic_algorithms': bool(torch.are_deterministic_algorithms_enabled()),
            'tf32_matmul': (bool(torch.backends.cuda.matmul.allow_tf32) if torch.cuda.is_available() else None),
            'fp16_reduced_precision_reduction': (bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction) if torch.cuda.is_available() else None),
            'bf16_reduced_precision_reduction': (bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction) if torch.cuda.is_available() else None),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'cpu_threads': int(torch.get_num_threads()),
            'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
        },
    })
    _knobs = hp['train_knobs']
    hp['amp_dtype'] = _knobs.get('amp_dtype')
    hp['eager_batch'] = _knobs.get('eager_batch')
    if data_version is not None:
        hp['data_version'] = str(data_version)
    return hp

def write_run_manifest(outdir, cfg, run_cfg, prov, summary, sanity=None, reproducibility=None, *, data_contract=None, seed_contract=None, status=None, planned_cells=None, started_at=None, cell_records=None, tracking_verification=None):
    """Merge one panel's provenance/config/outcome summary into
    `outdir/run_manifest.json` under the same flock used by summary.json.

    The per-record `losses` / `ppl_history` curves stay in summary.json (to
    avoid duplicating large arrays); each manifest cell points at its key.
    `sanity` (the small-batch overfit reports), `reproducibility` (the
    duplicate-run seed check), `seed_contract` (the explicit RNG-source
    audit) and `data_contract` (the real batch iterator check) are embedded
    when supplied, or read back from `outdir/overfit_sanity.json` /
    `outdir/reproducibility.json` / `outdir/seed_contract.json` /
    `outdir/data_contract.json` so an auto-preflight is traceable without a
    second manual record.

    `status` / `planned_cells` make the manifest a crash-safe tracking record:
    `run()` writes it with status='running' before the first cell, so a killed
    process still leaves the code version, config, data version and the list
    of planned cells on disk; the final call marks it complete and lists any
    cell that produced no measurement."""
    os.makedirs(outdir, exist_ok=True)

    def _evidence_for_this_panel(doc, path, what):
        """Only evidence stamped with THIS panel's run_cfg may enter the
        manifest.  A run_cfg-less (legacy/hand-written) doc must say
        ``legacy: true`` explicitly; otherwise a stale pre-flight from a reused
        outdir could silently certify a new configuration."""
        if doc is None:
            return None
        if not isinstance(doc, dict):
            print(f'[track] WARNING: {what} evidence is not a JSON object ({type(doc).__name__}) — not embedding it in this panel')
            return None
        _rc = doc.get('run_cfg')
        if _rc is None:
            if doc.get('legacy') is True:
                print(f'[track] WARNING: {what} evidence ({path}) has no run_cfg and is explicitly marked legacy — embedding it as unverified legacy evidence')
                return doc
            print(f'[track] WARNING: {what} evidence ({path}) has NO run_cfg — refusing to embed it (it cannot be tied to this panel; mark legacy=True only for a deliberately imported old record)')
            return None
        if _rc != run_cfg:
            print(f'[track] WARNING: {what} evidence ({path}) belongs to a DIFFERENT run_cfg ({_rc!r} != {run_cfg!r}) — not embedding it in this panel (a stale pre-flight must not certify a new config)')
            return None
        return doc

    if sanity is None:
        _san_path = os.path.join(outdir, 'overfit_sanity.json')
        if os.path.exists(_san_path):
            try:
                with open(_san_path, encoding='utf-8') as _sf:
                    sanity = json.load(_sf)
            except Exception as _se:
                print(f'[track] WARNING: could not read {_san_path} for the run manifest ({type(_se).__name__}: {_se})')
    sanity = _evidence_for_this_panel(sanity, os.path.join(outdir, 'overfit_sanity.json'), 'sanity')
    if reproducibility is None:
        _rep_path = os.path.join(outdir, 'reproducibility.json')
        if os.path.exists(_rep_path):
            try:
                with open(_rep_path, encoding='utf-8') as _rf:
                    reproducibility = json.load(_rf)
            except Exception as _re:
                print(f'[track] WARNING: could not read {_rep_path} for the run manifest ({type(_re).__name__}: {_re})')
    reproducibility = _evidence_for_this_panel(reproducibility, os.path.join(outdir, 'reproducibility.json'), 'reproducibility')
    if data_contract is None:
        _bc_path = os.path.join(outdir, 'data_contract.json')
        if os.path.exists(_bc_path):
            try:
                with open(_bc_path, encoding='utf-8') as _bf:
                    data_contract = json.load(_bf)
            except Exception as _be:
                print(f'[track] WARNING: could not read {_bc_path} for the run manifest ({type(_be).__name__}: {_be})')
    data_contract = _evidence_for_this_panel(data_contract, os.path.join(outdir, 'data_contract.json'), 'data contract')
    if seed_contract is None:
        _sc_path = os.path.join(outdir, 'seed_contract.json')
        if os.path.exists(_sc_path):
            try:
                with open(_sc_path, encoding='utf-8') as _scf:
                    seed_contract = json.load(_scf)
            except Exception as _sce:
                print(f'[track] WARNING: could not read {_sc_path} for the run manifest ({type(_sce).__name__}: {_sce})')
    seed_contract = _evidence_for_this_panel(seed_contract, os.path.join(outdir, 'seed_contract.json'), 'seed contract')
    if cell_records is not None:
        # Custom phases (warmup/NIAH/lenphase) have their own summary keys and
        # metrics (PPL with warm-grid keys, NIAH accuracy, by-length PPL).
        # They build already-normalised cell records and use the same
        # crash-safe manifest/provenance/integrity machinery as run().
        cells = {str(k): dict(v) for k, v in dict(cell_records).items() if isinstance(v, dict)}
    else:
        cells = {}
        for seed in cfg.get('seeds') or []:
            for v in cfg.get('variants') or []:
                key = f'{v}::seed{seed}'
                rec = summary.get(key)
                if not isinstance(rec, dict) or not ppl_is_usable(rec.get('ppl')) or rec.get('synthesized'):
                    continue
                _curves = _cell_curve_data(outdir, rec)
                _losses = rec.get('losses') if isinstance(rec.get('losses'), list) else (_curves.get('losses') if isinstance(_curves.get('losses'), list) else [])
                _loss_nums = [float(x) for x in _losses if isinstance(x, (int, float)) and not isinstance(x, bool)]
                _ph = rec.get('ppl_history') if isinstance(rec.get('ppl_history'), list) else (_curves.get('ppl_history') if isinstance(_curves.get('ppl_history'), list) else [])
                if not _losses and rec.get('curve_n_points') is not None:
                    _loss_nums = []
                cells[key] = {
                    'variant': v,
                    'seed': int(seed),
                    'run_id': rec.get('run_id'),
                    'run_cfg': rec.get('run_cfg'),
                    'started_at': rec.get('started_at'),
                    'finished_at': rec.get('finished_at'),
                    'ppl': rec.get('ppl'),
                    'steps': rec.get('steps'),
                    'tokens_seen': rec.get('tokens_seen'),
                    'params': rec.get('params'),
                    'train_time_s': rec.get('train_time_s'),
                    'final_loss_smoothed': rec.get('final_loss_smoothed'),
                    'rng_fp': rec.get('rng_fp'),
                    'exec_fp': rec.get('exec_fp'),
                    'env_fp_v2': rec.get('env_fp_v2'),
                    'env_fp_v3': rec.get('env_fp_v3') or (
                        (rec.get('hyperparams') or {}).get('env_fp_v3')
                        if isinstance(rec.get('hyperparams'), dict) else None),
                    'data_fp': rec.get('data_fp'),
                    'val_fp': rec.get('val_fp'),
                    'data_version': rec.get('data_version') or prov.get('data', {}).get('data_version'),
                    'corpus_fp': rec.get('corpus_fp') or prov.get('data', {}).get('corpus_fp'),
                    'code_stamp': rec.get('_code'),
                    'hyperparams': rec.get('hyperparams'),
                    'curve_summary': {'n_points': (len(_losses) if _losses else int(rec.get('curve_n_points') or 0)),
                                      'first': _loss_nums[0] if _loss_nums else rec.get('curve_first'),
                                      'last': _loss_nums[-1] if _loss_nums else rec.get('curve_last'),
                                      'best': min(_loss_nums) if _loss_nums else rec.get('curve_best'),
                                      'ppl_history_points': (len(_ph) if _ph else int(rec.get('ppl_history_n_points') or 0))},
                    'curve_ref': rec.get('curve_ref') or f'summary.json#{key}.losses',
                    'curve_sha256': rec.get('curve_sha256'),
                    'ppl_history_ref': rec.get('ppl_history_ref') or f'summary.json#{key}.ppl_history',
                    'ppl_history_sha256': rec.get('ppl_history_sha256'),
                    'sanity_ref': rec.get('sanity_ref') or ('overfit_sanity.json' if sanity else None),
                }
    if planned_cells is None:
        planned_cells = [f'{v}::seed{seed}' for seed in cfg.get('seeds') or [] for v in cfg.get('variants') or []]
    planned_cells = [str(_k) for _k in dict.fromkeys(planned_cells)]

    def _curve_ref_ok(cell):
        for _key in ('curve_ref', 'ppl_history_ref'):
            _ref = cell.get(_key) if isinstance(cell, dict) else None
            if not _ref or '#' in str(_ref):
                continue
            if not os.path.exists(os.path.join(outdir, str(_ref))):
                return False
        return True
    # Machine-checkable completeness of the per-cell trace.  A record without
    # provenance/hyperparameters/RNG fingerprint/curve is still a measurement,
    # but it must not silently look as traceable as a complete one.
    #
    # Seed metadata is checked against the RESUME KEY, not against itself: a
    # record copied/mis-keyed under `seed3` with `seed=2` would otherwise count
    # as a third independent seed while actually duplicating the second.
    _seed_mismatch = []
    _seed_ok = 0
    _run_cfg_mismatch = []
    _data_version_mismatch = []
    for _key, _c in cells.items():
        try:
            _expected = int(str(_key).rsplit('::seed', 1)[1])
        except (ValueError, IndexError):
            _seed_mismatch.append(_key)
            continue
        _hp = _c.get('hyperparams') if isinstance(_c.get('hyperparams'), dict) else {}
        _rng = _hp.get('rng') if isinstance(_hp.get('rng'), dict) else {}
        try:
            _vals = [int(_hp.get('seed'))] if _hp.get('seed') is not None else []
            if _hp.get('data_seed') is not None:
                _vals.append(int(_hp.get('data_seed')))
            _vals.extend(int(_rng[k]) for k in ('python', 'numpy', 'torch', 'cuda', 'tensorflow',
                                                'data', 'dataloader_generator') if _rng.get(k) is not None)
            _ok = (bool(_c.get('seed') is not None) and int(_c.get('seed')) == _expected
                   and all(v == _expected for v in _vals))
        except (TypeError, ValueError, OverflowError):
            _ok = False
        if _ok:
            _seed_ok += 1
        else:
            _seed_mismatch.append(_key)
        if _c.get('run_cfg') not in (None, run_cfg):
            _run_cfg_mismatch.append(_key)
        _cell_dv = _c.get('data_version')
        _hp_dv = _hp.get('data_version')
        if _cell_dv is not None and _hp_dv is not None and str(_cell_dv) != str(_hp_dv):
            _data_version_mismatch.append(_key)
    try:
        _declared_seeds = [int(x) for x in (cfg.get('seeds') or [])]
    except (TypeError, ValueError, OverflowError):
        _declared_seeds = []
    _integrity = {
        'n_cells': len(cells),
        'seeds_declared': _declared_seeds,
        'seeds_unique': bool(len(_declared_seeds) == len(set(_declared_seeds))),
        'with_seed_metadata': _seed_ok,
        'seed_metadata_mismatch': _seed_mismatch[:20],
        'with_provenance': sum(1 for c in cells.values() if c.get('data_fp') and c.get('code_stamp')),
        'with_hyperparams': sum(1 for c in cells.values() if c.get('hyperparams')),
        'with_rng_fp': sum(1 for c in cells.values() if c.get('rng_fp')),
        'with_exec_fp': sum(1 for c in cells.values() if c.get('exec_fp')),
        # Additive coverage count, deliberately NOT part of `complete`: an
        # older panel without the strict digest is still a legacy measurement,
        # not an incomplete one.  New panels must expose the count so a silent
        # loss of the field is visible.
        'with_env_fp_v2': sum(1 for c in cells.values() if c.get('env_fp_v2')),
        # v3 is additive, like v2: an older panel is a legacy measurement,
        # not an incomplete one.  New panels must expose the count so a
        # silent loss of the stricter digest is visible.
        'with_env_fp_v3': sum(1 for c in cells.values() if c.get('env_fp_v3')),
        'with_data_version': sum(1 for c in cells.values() if c.get('data_version')),
        'with_curve': sum(1 for c in cells.values() if (c.get('curve_summary') or {}).get('n_points')),
        'with_curve_hash': sum(1 for c in cells.values() if c.get('curve_sha256')),
        'with_sanity_ref': sum(1 for c in cells.values() if c.get('sanity_ref')),
        'with_data_contract': int(bool(isinstance(data_contract, dict) and data_contract.get('passed'))),
        'with_seed_contract': int(bool(isinstance(seed_contract, dict) and seed_contract.get('passed'))),
    }
    # Counts alone let a cell that is missing its code stamp / hyperparameters /
    # data version / RNG fingerprint / curve still look like a normal member of
    # a "complete" panel.  Name the incomplete cells and make panel completion
    # a single machine-readable verdict, so an automated reader cannot mistake
    # a partially traced panel for a fully reproducible one.
    _missing_cells = [str(k) for k in planned_cells if k not in cells]
    _incomplete_cells = []
    _missing_curve_refs = []
    for _k, _c in cells.items():
        _hp2 = _c.get('hyperparams') if isinstance(_c.get('hyperparams'), dict) else {}
        _curve_n = (_c.get('curve_summary') or {}).get('n_points')
        _ref_ok = _curve_ref_ok(_c)
        if not _ref_ok:
            _missing_curve_refs.append(str(_k))
        if not (_c.get('code_stamp') and _hp2 and _c.get('data_version')
                and _c.get('rng_fp') and _curve_n and _ref_ok):
            _incomplete_cells.append(str(_k))
    _integrity['complete_cells'] = len(cells) - len(_incomplete_cells)
    _integrity['incomplete_cells'] = _incomplete_cells[:20]
    _integrity['missing_cells'] = _missing_cells[:20]
    _integrity['missing_curve_refs'] = _missing_curve_refs[:20]
    _integrity['run_cfg_mismatch'] = _run_cfg_mismatch[:20]
    _integrity['data_version_mismatch'] = _data_version_mismatch[:20]
    _integrity['complete'] = bool(not _incomplete_cells and not _missing_cells
                                  and not _seed_mismatch and not _missing_curve_refs
                                  and not _run_cfg_mismatch and not _data_version_mismatch)
    _declared_status = status or 'complete'
    if _declared_status in ('running', 'failed', 'error', 'incomplete'):
        _completion_status = _declared_status
    else:
        _completion_status = 'complete' if _integrity['complete'] else 'incomplete'
    _completion = {
        'status': _completion_status,
        # A panel explicitly declared incomplete/failed/running is not complete
        # just because its recorded cells happen to be trace-complete.  The
        # boolean is a conjunction of the declared terminal state and the
        # per-cell integrity verdict.
        'complete': bool(_integrity['complete'] and _completion_status == 'complete'),
        'n_planned': len(planned_cells),
        'n_measured': len(cells),
        'n_missing': len(_missing_cells),
        'missing_cells': _missing_cells[:20],
        'incomplete_cells': _incomplete_cells[:20],
        'incomplete_reasons': ('missing measurement for every planned cell; '
                               'each measured cell carries code/hyperparameters/'
                               'data_version/rng_fp/curve and seed metadata'
                               if _integrity['complete'] else
                               'panel is not fully traced: see missing_cells / '
                               'incomplete_cells / missing_curve_refs / '
                               'seed_metadata_mismatch / run_cfg_mismatch / '
                               'data_version_mismatch'),
    }
    _prod_evidence = None
    if isinstance(sanity, dict) and isinstance(sanity.get('production_loop'), dict):
        _prod_evidence = bool((sanity.get('production_loop') or {}).get('passed'))
    _evidence = {
        'small_batch_overfit': (bool(sanity.get('passed')) if isinstance(sanity, dict) else None),
        'production_loop_overfit': _prod_evidence,
        'same_machine_reproducibility': (bool(reproducibility.get('passed')) if isinstance(reproducibility, dict) else None),
        'rng_seed_contract': (bool(seed_contract.get('passed')) if isinstance(seed_contract, dict) else None),
        'batch_iterator_contract': (bool(data_contract.get('passed')) if isinstance(data_contract, dict) else None),
    }
    entry = {
        'run_cfg': run_cfg,
        'status': status or 'complete',
        'created_at': started_at or time.strftime('%Y-%m-%dT%H:%M:%S'),
        'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': _jsonable(cfg),
        'provenance': prov,
        'data_version': prov.get('data', {}).get('data_version'),
        'corpus_fp': prov.get('data', {}).get('corpus_fp'),
        'sanity': sanity,
        'reproducibility': reproducibility,
        'seed_contract': seed_contract,
        'data_contract': data_contract,
        'tracking_verification': tracking_verification,
        'evidence': _evidence,
        'experiment_log': 'experiments.jsonl',
        'tracking_schema': 2,
        'planned_cells': planned_cells,
        'n_planned': len(planned_cells),
        'n_completed': len(cells),
        'missing_cells': _missing_cells,
        'completion': _completion,
        'tracking_integrity': _integrity,
        'cells': cells,
    }
    path = os.path.join(outdir, 'run_manifest.json')

    def _merge(m):
        if not isinstance(m, dict):
            m = {}
        m['manifest_version'] = m.get('manifest_version') or 1
        _prev = m.setdefault('panels', {}).get(run_cfg)
        if isinstance(_prev, dict) and _prev.get('created_at') and status != 'running':
            entry['created_at'] = _prev['created_at']
        m['panels'][run_cfg] = entry
        return m
    locked_json_update(path, _merge)
    return path

def verify_run_tracking(outdir, run_cfg=None, *, require_ppl=True):
    """Independently verify one panel's on-disk trace.

    `write_run_manifest` builds the trace; this function reads it back from
    disk and re-derives the checks a reader would otherwise have to trust:
    every planned cell has a record, a code stamp, hyperparameters including
    the full RNG map, a data version, a final result, and a curve whose
    recorded digest still matches the file.  It is read-only and never touches
    the test split, so it can run at the end of every panel (and by hand on an
    archived outdir).

    Returns a JSON-safe verdict.  Missing legacy fields are reported as
    problems for a current panel; `warnings` carries non-fatal notes.
    """
    problems = []
    warnings = []
    outdir = os.path.abspath(outdir)
    manifest_path = os.path.join(outdir, 'run_manifest.json')
    summary_path = os.path.join(outdir, 'summary.json')
    if not os.path.exists(manifest_path):
        return {'kind': 'run_tracking_verification', 'outdir': outdir,
                'run_cfg': run_cfg, 'passed': False,
                'problems': ['run_manifest.json is missing'],
                'warnings': [], 'n_checked': 0}
    try:
        with open(manifest_path, encoding='utf-8') as f:
            manifest = json.load(f)
    except Exception as e:
        return {'kind': 'run_tracking_verification', 'outdir': outdir,
                'run_cfg': run_cfg, 'passed': False,
                'problems': [f'run_manifest.json cannot be parsed: {type(e).__name__}: {e}'],
                'warnings': [], 'n_checked': 0}
    panels = manifest.get('panels') if isinstance(manifest, dict) else None
    if not isinstance(panels, dict) or not panels:
        return {'kind': 'run_tracking_verification', 'outdir': outdir,
                'run_cfg': run_cfg, 'passed': False,
                'problems': ['run_manifest.json has no panels'], 'warnings': [],
                'n_checked': 0}
    if run_cfg is None:
        # Newest panel by created_at / updated_at; insertion order is the
        # fallback for legacy manifests without timestamps.
        def _stamp(entry):
            return str((entry or {}).get('updated_at') or (entry or {}).get('created_at') or '')
        run_cfg = max(panels, key=lambda k: _stamp(panels.get(k)))
    entry = panels.get(run_cfg)
    if not isinstance(entry, dict):
        return {'kind': 'run_tracking_verification', 'outdir': outdir,
                'run_cfg': run_cfg, 'passed': False,
                'problems': [f'panel {run_cfg!r} is absent from run_manifest.json'],
                'warnings': [], 'n_checked': 0}
    try:
        with open(summary_path, encoding='utf-8') as f:
            summary = json.load(f)
    except Exception as e:
        summary = {}
        problems.append(f'summary.json cannot be parsed: {type(e).__name__}: {e}')
    if not isinstance(summary, dict):
        problems.append('summary.json is not a JSON object')
        summary = {}
    planned = [str(k) for k in (entry.get('planned_cells') or [])]
    if not planned:
        problems.append('panel records no planned_cells')
    cells = entry.get('cells') if isinstance(entry.get('cells'), dict) else {}
    prov_dv = ((entry.get('provenance') or {}).get('data') or {}).get('data_version')
    checked = 0
    for key in planned:
        checked += 1
        rec = summary.get(key)
        cell = cells.get(key)
        if not isinstance(rec, dict):
            problems.append(f'{key}: no summary record')
            continue
        if not isinstance(cell, dict):
            problems.append(f'{key}: no manifest cell entry')
            cell = {}
        if not rec.get('_code'):
            problems.append(f'{key}: record has no code stamp')
        if cell.get('code_stamp') and rec.get('_code') and cell['code_stamp'] != rec.get('_code'):
            problems.append(f'{key}: manifest code_stamp disagrees with the record')
        hp = rec.get('hyperparams') if isinstance(rec.get('hyperparams'), dict) else {}
        if not hp:
            problems.append(f'{key}: record has no hyperparameters')
        if not cell.get('hyperparams'):
            problems.append(f'{key}: manifest cell has no hyperparameters')
        rng = hp.get('rng') if isinstance(hp.get('rng'), dict) else {}
        if not rng:
            problems.append(f'{key}: hyperparameters have no RNG map')
        try:
            _expected = int(str(key).rsplit('::seed', 1)[1])
        except (ValueError, IndexError):
            _expected = None
        if _expected is None:
            problems.append(f'{key}: key does not name a seed')
        else:
            _seed_vals = [hp.get('seed'), hp.get('data_seed')]
            _seed_vals += [rng.get(k) for k in ('python', 'numpy', 'torch', 'cuda', 'tensorflow', 'data', 'dataloader_generator')]
            if any(v is None for v in _seed_vals):
                problems.append(f'{key}: RNG map is incomplete')
            else:
                # A hand-edited/archived manifest can contain a non-numeric
                # seed field (for example ``rng.python='bad'``).  ``int()``
                # used to raise out of the verifier, aborting the whole audit
                # instead of recording a failed check.  Normalise through the
                # exact-integer reader and report malformed values as a
                # problem; the returned verdict must stay JSON-safe.
                _seed_norm = [_as_seed_int(v) for v in _seed_vals]
                if any(v is None for v in _seed_norm):
                    problems.append(
                        f'{key}: seed metadata is not an exact integer '
                        f'({_seed_vals!r})')
                elif any(v != _expected for v in _seed_norm):
                    problems.append(f'{key}: seed metadata disagrees with the key')
        dv = rec.get('data_version') or cell.get('data_version')
        if not dv:
            problems.append(f'{key}: record has no data_version')
        elif prov_dv and str(dv) != str(prov_dv):
            problems.append(f'{key}: data_version disagrees with the panel provenance')
        if not rec.get('rng_fp'):
            problems.append(f'{key}: record has no rng_fp')
        if not rec.get('exec_fp'):
            warnings.append(f'{key}: record has no exec_fp (legacy record)')
        if require_ppl and not ppl_is_usable(rec.get('ppl')):
            problems.append(f'{key}: no usable final PPL/result')
        ref = rec.get('curve_ref')
        losses = rec.get('losses') if isinstance(rec.get('losses'), list) else []
        if ref and '#' not in str(ref):
            path = os.path.join(outdir, str(ref))
            if not os.path.exists(path):
                problems.append(f'{key}: curve sidecar {ref!r} is missing')
            else:
                want = rec.get('curve_sha256') or cell.get('curve_sha256')
                got = _file_sha256(path)
                if want and got != want:
                    problems.append(f'{key}: curve sidecar digest no longer matches the record')
                elif not want:
                    warnings.append(f'{key}: curve sidecar has no recorded digest')
                try:
                    with open(path, encoding='utf-8') as f:
                        payload = json.load(f)
                    if not isinstance(payload, dict):
                        problems.append(f'{key}: curve sidecar is not a JSON object')
                    elif not (payload.get('losses') or payload.get('ppl_history')):
                        problems.append(f'{key}: curve sidecar contains no curve data')
                except Exception as e:
                    problems.append(f'{key}: curve sidecar cannot be parsed ({type(e).__name__}: {e})')
        elif not losses and not rec.get('curve_n_points'):
            problems.append(f'{key}: no training curve (inline or sidecar)')
        ph_ref = rec.get('ppl_history_ref')
        if ph_ref and '#' not in str(ph_ref):
            ph_path = os.path.join(outdir, str(ph_ref))
            if not os.path.exists(ph_path):
                problems.append(f'{key}: ppl_history sidecar {ph_ref!r} is missing')
            else:
                want = rec.get('ppl_history_sha256') or cell.get('ppl_history_sha256')
                if want and _file_sha256(ph_path) != want:
                    problems.append(f'{key}: ppl_history sidecar digest no longer matches the record')
    verdict = {
        'kind': 'run_tracking_verification',
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'outdir': outdir,
        'run_cfg': run_cfg,
        'panel_status': entry.get('status'),
        'panel_completion': entry.get('completion'),
        'n_planned': len(planned),
        'n_checked': checked,
        'problems': problems,
        'warnings': warnings,
        'passed': bool(not problems),
    }
    verdict['diagnosis'] = ('ok' if not problems else
                            f'{len(problems)} tracking problem(s); first: {problems[0]}')
    return verdict

_EVAL_TOKENS_PER_CHUNK = 1 << 20


def _eval_row_working_bytes(model, T, d, elem_size, bytes_per_value=8):
    """Conservative per-row activation/workspace estimate for one eval row.

    The dense `T*d` term is only the hidden-state floor.  Long-sequence
    workspace is usually set by the attention workspace instead: a full/sink
    head materialises [nh, rows, T] logits, the CSA indexer scores [T, B_blocks,
    nIH], and HCA's arange expansion is [T, B_blocks].  Estimate the largest of
    those per layer (layers run one at a time under inference_mode) rather than
    the hidden state alone."""
    T = max(1, int(T))
    d = max(1, int(d))
    elem = max(1, int(elem_size))
    per_row = max(1, T * d * elem * max(1, int(bytes_per_value)))
    for blk in getattr(model, 'blocks', []):
        attn = getattr(blk, 'attn', None)
        cfg = getattr(attn, 'cfg', None)
        if cfg is None:
            continue
        nh = max(1, int(getattr(attn, 'nh', 1) or 1))
        kind = getattr(cfg, 'kind', 'full')
        if kind == 'full':
            win = max(0, int(getattr(cfg, 'window', 0) or 0))
            if getattr(attn, 'sink', None) is not None or getattr(cfg, 'full_sink', False):
                span = T
            elif 0 < win < T:
                span = win
            else:
                span = min(T, 1024)
            attn_bytes = 3 * nh * T * span * elem
        else:
            bs = int(getattr(cfg, 'block_size', 0) or 0)
            tgt = int(getattr(cfg, 'target_block_tokens', 0) or 0)
            mn = max(1, int(getattr(cfg, 'min_block', 1) or 1))
            step = max(1, bs or tgt or mn)
            Bm = (T + step - 1) // step
            nih = max(1, int(getattr(cfg, 'n_index_heads', 1) or 1))
            topk = max(1, int(getattr(cfg, 'index_topk', 1) or 1))
            hd = max(1, int(getattr(attn, 'hd', 1) or 1))
            win = max(0, int(getattr(cfg, 'sliding_window', 0) or 0))
            # HCA reads every block (arange expansion), so its effective
            # selection width is Bm; CSA only scores min(topk, Bm) blocks.
            nb_eff = Bm if kind == 'hca' else min(topk, Bm)
            M = nb_eff + win
            gather_bytes = (4 * M * nh * hd + 2 * M * nh) * elem
            index_bytes = min(2 * T * Bm * nih * elem, 384 << 20) if kind != 'hca' else 0
            attn_bytes = max(index_bytes, gather_bytes)
            if kind == 'hca':
                # _arange_exp_cache is a stride-0 expand view: its physical
                # storage is one int32 [Bm] row, not [T, Bm].
                attn_bytes = max(attn_bytes, 4 * Bm)
        per_row = max(per_row, int(attn_bytes))
    return per_row


def _eval_adaptive_chunk(model, val_batch, device, chunk, _free_frac=0.5, _bytes_per_value=8):
    """Bound an evaluation chunk by both token count and currently-free device capacity.

    Rows are independent, so the partition changes no value; it only keeps a
    long-sequence eval from launching a chunk whose attention/indexer
    workspace cannot fit.  The estimate uses the model's actual parameter
    element size (bf16 halves it) and the real per-layer attention shape, not
    just `T*d`; the existing OOM retry remains the final backstop."""
    rows = int(val_batch.shape[0])
    T = int(val_batch.shape[1])
    span = max(1, T - 1)
    chunk = min(max(1, int(chunk)), max(1, _EVAL_TOKENS_PER_CHUNK // span))
    if device.type == 'cuda' and rows > 0:
        try:
            free, _total = torch.cuda.mem_get_info(device)
            d = int(model.tok.weight.shape[1])
            elem = int(model.tok.weight.element_size())
            per_row = _eval_row_working_bytes(model, T, d, elem, _bytes_per_value)
            chunk = min(chunk, max(1, int(free * float(_free_frac)) // per_row))
        except Exception:
            pass
    return max(1, min(chunk, rows))


def _det_segment_sum(n_rows, indices, values, dtype=None, sorted_indices=False):
    """Deterministic scatter-add by stable sort + segmented prefix sums.

    CUDA's ``index_add_``/``index_put_(accumulate=True)`` use atomics, so the
    addition order (and the last-bit result) can depend on thread scheduling.
    Sorting by (index, original position) fixes the order; segment sums are
    then formed with a cumsum and written through ``index_copy_`` (unique
    destinations, no atomics).  Mirrors ``csa_kernels._det_scatter_add``: the
    output keeps the caller's dtype, but floating (and complex) sums accumulate
    in float64/complex128 before the final cast.  A shared prefix from an early
    segment otherwise rounds away every later small segment in fp32, and the
    loss grows with the number of rows/segments.

    ``sorted_indices=True`` promises the flattened indices are already
    non-decreasing (the sampled-eval path maps a sorted global-row table, so
    its per-chunk rows are sorted by construction).  The stable sort is then
    an identity permutation and is skipped; every element keeps the same
    position the sort would have given it, so the segmented sums are
    bit-identical while the sort's time and workspace disappear."""
    idx = indices.reshape(-1).to(torch.int64)
    vals = values.reshape(-1)
    out_dtype = dtype if dtype is not None else vals.dtype
    out = torch.zeros(int(n_rows), device=vals.device, dtype=out_dtype)
    if idx.numel() == 0:
        return out
    valid = (idx >= 0) & (idx < int(n_rows))
    if not bool(valid.all()):
        idx = idx[valid]
        vals = vals[valid]
        if idx.numel() == 0:
            return out
    if out_dtype.is_complex:
        _acc = torch.complex128
    elif out_dtype.is_floating_point:
        _acc = torch.float64
    else:
        # Integer destinations (token counts) stay exact; int64 is the widest
        # representation the destination can require and avoids an unnecessary
        # float round-trip.
        _acc = torch.int64
    if sorted_indices:
        idx_s = idx
        val_s = vals.to(_acc)
    else:
        order = torch.argsort(idx, stable=True)
        idx_s = idx.index_select(0, order)
        val_s = vals.index_select(0, order).to(_acc)
    end = torch.ones_like(idx_s, dtype=torch.bool)
    if idx_s.numel() > 1:
        end[:-1] = idx_s[1:] != idx_s[:-1]
    seg_end = torch.nonzero(end, as_tuple=False).reshape(-1)
    csum = torch.cumsum(val_s, dim=0)
    hi = csum.index_select(0, seg_end)
    lo = torch.zeros_like(hi)
    if seg_end.numel() > 1:
        lo[1:] = csum.index_select(0, seg_end[:-1])
    seg_sum = hi - lo
    seg_id = idx_s.index_select(0, seg_end)
    out.index_copy_(0, seg_id, seg_sum.to(out_dtype))
    return out


UNIT_ORDER_POLICY = {
    'version': 'unit-order-1',
    'schema': 'unit_order_v1',
    'basis': 'first_occurrence_in_scored_targets',
    'unit_kinds': ('document', 'article'),
    'algorithm': (
        'sha256 over the analyzed sequence: for each unit id, the ASCII '
        'decimal UTF-8 byte length, a colon, the UTF-8 bytes, and a newline'),
    'role': (
        'record the order actually used by an order-dependent readout, '
        'together with the permutation from the stored per-unit vectors to '
        'that order'),
    'boundary': (
        'The attestation proves which order the evaluator observed and that '
        'the analysis used it.  It does not prove that the upstream extractor '
        'itself preserved corpus order, nor that first occurrence is a '
        'probability sample.  A sorted/shuffled vector without a matching '
        'attestation is refused rather than silently analyzed.'),
}


def _uo_unit_ids(unit_ids, name='unit_ids'):
    """Normalise one ordered unit-id sequence, refusing ambiguous inputs."""
    if isinstance(unit_ids, (str, bytes, bytearray)):
        raise ValueError(f'{name} must be an iterable of ids, not a string')
    try:
        raw = list(unit_ids)
    except TypeError:
        raise ValueError(f'{name} must be an iterable') from None
    if not raw:
        raise ValueError(f'{name} must not be empty')
    out = []
    for value in raw:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f'{name} contains a boolean; a unit id is a '
                             'string or an integer-like value')
        try:
            text = str(value)
        except Exception:
            raise ValueError(f'{name} contains an unrenderable id') from None
        if text == '':
            raise ValueError(f'{name} contains an empty id')
        out.append(text)
    if len(set(out)) != len(out):
        raise ValueError(f'{name} must be unique; a duplicated unit cannot be '
                         'ordered unambiguously')
    return out


def unit_order_sha256(unit_ids):
    """Stable digest of an ORDERED unit-id sequence.

    Length-prefixing prevents ``['a', 'bc']`` and ``['ab', 'c']`` from sharing
    a digest.  The function deliberately does not sort: the order is the
    measurement.
    """
    import hashlib as _uo_hashlib

    ids = _uo_unit_ids(unit_ids)
    h = _uo_hashlib.sha256()
    for text in ids:
        raw = text.encode('utf-8')
        h.update(str(len(raw)).encode('ascii'))
        h.update(b':')
        h.update(raw)
        h.update(b'\n')
    return h.hexdigest()


def unit_order_attestation(unit_ids, *, unit_kind, permutation=None,
                           first_positions=None, n_tokens=None):
    """Attest that ``unit_ids`` is the order an order-dependent readout used.

    ``unit_ids`` is in the analyzed (first-occurrence/corpus) order.
    ``permutation`` maps the canonical stored per-unit vectors (the sorted
    ``doc_ids``/``article_ids`` vectors) to this order, so an auditor can
    reconstruct the exact sequence from the stored cell.  ``first_positions``
    are the flattened target positions where each ordered unit first appears;
    they must be strictly increasing.
    """
    ids = _uo_unit_ids(unit_ids)
    kind = str(unit_kind)
    if kind not in UNIT_ORDER_POLICY['unit_kinds']:
        raise ValueError(f'unit_kind must be one of '
                         f'{UNIT_ORDER_POLICY["unit_kinds"]}, got {unit_kind!r}')
    n_units = len(ids)
    perm = None
    if permutation is not None:
        if isinstance(permutation, (str, bytes, bytearray)):
            raise ValueError('permutation must be an iterable of integers')
        try:
            raw_perm = list(permutation)
        except TypeError:
            raise ValueError('permutation must be an iterable of integers') from None
        if len(raw_perm) != n_units:
            raise ValueError(f'permutation has {len(raw_perm)} entries for '
                             f'{n_units} units')
        perm = []
        for value in raw_perm:
            if isinstance(value, (bool, np.bool_)):
                raise ValueError('permutation contains a boolean')
            if not isinstance(value, (int, np.integer)):
                raise ValueError('permutation must contain exact integers')
            perm.append(int(value))
        if sorted(perm) != list(range(n_units)):
            raise ValueError('permutation must be a permutation of '
                             f'range({n_units})')
    positions = None
    if first_positions is not None:
        if isinstance(first_positions, (str, bytes, bytearray)):
            raise ValueError('first_positions must be an iterable of integers')
        try:
            raw_pos = list(first_positions)
        except TypeError:
            raise ValueError('first_positions must be an iterable of integers') from None
        if len(raw_pos) != n_units:
            raise ValueError(f'first_positions has {len(raw_pos)} entries '
                             f'for {n_units} units')
        positions = []
        for value in raw_pos:
            if isinstance(value, (bool, np.bool_)):
                raise ValueError('first_positions contains a boolean')
            if not isinstance(value, (int, np.integer)):
                raise ValueError('first_positions must contain exact integers')
            ivalue = int(value)
            if ivalue < 0:
                raise ValueError('first_positions must be non-negative')
            positions.append(ivalue)
        if any(b <= a for a, b in zip(positions, positions[1:])):
            raise ValueError('first_positions must be strictly increasing '
                             'under the analyzed order')
    if n_tokens is not None:
        if isinstance(n_tokens, (bool, np.bool_)):
            raise ValueError('n_tokens must be a positive integer, not a boolean')
        if not isinstance(n_tokens, (int, np.integer)):
            raise ValueError('n_tokens must be a positive integer')
        n_tokens = int(n_tokens)
        if n_tokens <= 0:
            raise ValueError('n_tokens must be a positive integer')
    return {
        'schema': UNIT_ORDER_POLICY['schema'],
        'policy_version': UNIT_ORDER_POLICY['version'],
        'basis': UNIT_ORDER_POLICY['basis'],
        'algorithm': UNIT_ORDER_POLICY['algorithm'],
        'unit_kind': kind,
        'n_units': n_units,
        'sha256': unit_order_sha256(ids),
        'permutation': perm,
        'first_positions': positions,
        'n_tokens': n_tokens,
        'note': UNIT_ORDER_POLICY['boundary'],
    }


def unit_order_problems(attestation, unit_ids, *, unit_kind=None,
                        expected_sha256=None):
    """Return a list of reasons an attestation does not describe ``unit_ids``.

    ``unit_ids`` must be the analyzed order.  An empty list means the digest,
    basis, declared size and (when present) permutation/positions are
    internally consistent with that sequence.  The caller still owns the
    question of whether the upstream evaluation preserved corpus order.
    """
    problems = []
    if not isinstance(attestation, dict):
        return [f'unit_order is a {type(attestation).__name__}, not an object']
    try:
        ids = _uo_unit_ids(unit_ids)
    except ValueError as exc:
        return [str(exc)]
    if attestation.get('schema') != UNIT_ORDER_POLICY['schema']:
        problems.append('unit_order.schema is not '
                        f'{UNIT_ORDER_POLICY["schema"]!r}')
    if attestation.get('policy_version') != UNIT_ORDER_POLICY['version']:
        problems.append('unit_order.policy_version is not '
                        f'{UNIT_ORDER_POLICY["version"]!r} (got '
                        f'{attestation.get("policy_version")!r})')
    if attestation.get('basis') != UNIT_ORDER_POLICY['basis']:
        problems.append('unit_order.basis is not '
                        f'{UNIT_ORDER_POLICY["basis"]!r} (got '
                        f'{attestation.get("basis")!r}); a sorted id vector is '
                        'not a corpus-order attestation')
    if attestation.get('algorithm') != UNIT_ORDER_POLICY['algorithm']:
        problems.append('unit_order.algorithm is not the documented '
                        'length-prefixed SHA-256')
    if unit_kind is not None and attestation.get('unit_kind') != str(unit_kind):
        problems.append('unit_order.unit_kind '
                        f'{attestation.get("unit_kind")!r} != {unit_kind!r}')
    n_units = attestation.get('n_units')
    if isinstance(n_units, bool) or not isinstance(n_units, (int, np.integer)):
        problems.append('unit_order.n_units is not an integer')
    elif int(n_units) != len(ids):
        problems.append(f'unit_order.n_units={n_units!r} != {len(ids)} ids')
    stored_sha = attestation.get('sha256')
    if not isinstance(stored_sha, str) or len(stored_sha) != 64:
        problems.append('unit_order.sha256 is not a 64-character digest')
    else:
        try:
            recomputed = unit_order_sha256(ids)
        except ValueError as exc:
            problems.append(str(exc))
        else:
            if stored_sha.lower() != recomputed:
                problems.append('unit_order.sha256 does not recompute from the '
                                'analyzed id sequence')
    if expected_sha256 is not None:
        if not isinstance(stored_sha, str) or stored_sha.lower() != str(
                expected_sha256).lower():
            problems.append('unit_order.sha256 disagrees with the expected '
                            'order fingerprint')
    if 'permutation' in attestation:
        perm = attestation.get('permutation')
        if perm is not None:
            if not isinstance(perm, list) or len(perm) != len(ids):
                problems.append('unit_order.permutation is not a list aligned '
                                'with n_units')
            else:
                norm = []
                for value in perm:
                    if isinstance(value, bool) or not isinstance(
                            value, (int, np.integer)):
                        norm = None
                        break
                    norm.append(int(value))
                if norm is None or sorted(norm) != list(range(len(ids))):
                    problems.append('unit_order.permutation is not a '
                                    f'permutation of range({len(ids)})')
    if 'first_positions' in attestation:
        pos = attestation.get('first_positions')
        if pos is not None:
            if not isinstance(pos, list) or len(pos) != len(ids):
                problems.append('unit_order.first_positions is not a list '
                                'aligned with n_units')
            else:
                norm = []
                for value in pos:
                    if isinstance(value, bool) or not isinstance(
                            value, (int, np.integer)) or int(value) < 0:
                        norm = None
                        break
                    norm.append(int(value))
                if norm is None:
                    problems.append('unit_order.first_positions contains a '
                                    'non-integer or negative value')
                elif any(b <= a for a, b in zip(norm, norm[1:])):
                    problems.append('unit_order.first_positions is not '
                                    'strictly increasing')
    if 'n_tokens' in attestation and attestation.get('n_tokens') is not None:
        ntok = attestation.get('n_tokens')
        if isinstance(ntok, bool) or not isinstance(ntok, (int, np.integer)) \
                or int(ntok) <= 0:
            problems.append('unit_order.n_tokens is not a positive integer')
    return problems


def _unit_order_from_index(keys, present, inv, n_tokens_total, unit_kind):
    """Build the first-occurrence attestation for the RETURNED vectors.

    ``keys`` is the full sorted unique id array, ``present`` are the indices
    into it that the caller actually returned, and ``inv`` maps every scored
    target to its index in ``keys``.  The returned vectors are
    ``keys[present]`` in ascending-index (sorted-id) order, so ``permutation``
    indexes into those vectors.  First-occurrence order is computed from the
    target stream, never from the id values.
    """
    arr = np.asarray(inv).reshape(-1)
    if arr.size == 0:
        raise ValueError('_unit_order_from_index: empty target map')
    present = [int(x) for x in present]
    present_set = set(present)
    if len(present_set) != len(present):
        raise ValueError('_unit_order_from_index: duplicate present indices')
    pos_in_present = {full_idx: pos for pos, full_idx in enumerate(present)}
    present_units, first_idx = np.unique(arr, return_index=True)
    if any(int(u) not in present_set for u in present_units):
        raise ValueError('_unit_order_from_index: a scored unit is absent from '
                         'the returned vectors')
    first_for_present = [0] * len(present)
    seen = [False] * len(present)
    for unit, pos in zip(present_units.tolist(), first_idx.tolist()):
        idx = pos_in_present[int(unit)]
        first_for_present[idx] = int(pos)
        seen[idx] = True
    if not all(seen):
        raise ValueError('_unit_order_from_index: a returned unit has no '
                         'scored target')
    order = sorted(range(len(present)),
                   key=lambda i: (first_for_present[i], i))
    keys = np.asarray(keys)
    ordered_ids = [format(int(keys[present[i]]), '016x') for i in order]
    positions = [int(first_for_present[i]) for i in order]
    return unit_order_attestation(
        ordered_ids, unit_kind=unit_kind, permutation=order,
        first_positions=positions, n_tokens=int(n_tokens_total))


def document_index(doc_ids):
    """Compact, deterministic index for an arbitrary per-token document-id array.

    Returns ``(keys, inverse)`` where ``keys`` is the sorted unique document
    id array and ``inverse`` maps every element back to its position in
    ``keys``.  The same ``(keys, inverse)`` pair can be sliced for several
    evaluation lengths/chunks, so the index cannot change between them.  The
    sorted order is a canonical vector layout, NOT corpus order; the
    first-occurrence order is recorded separately in the cell's ``unit_order``
    attestation.

    Document ids are uint64 hashes; converting them to a dense index avoids
    materialising a huge [n_docs] tensor indexed by the raw hash value.  The
    function is intentionally NumPy-only and rejects non-integer / negative
    ids, because a float id such as ``1.5`` would silently become document 1.
    """
    arr = np.asarray(doc_ids)
    if arr.size == 0:
        raise ValueError('document_index: document ids are empty')
    if arr.dtype.kind not in 'iu':
        raise ValueError(
            f'document_index: document ids must be integers, got dtype {arr.dtype}')
    if arr.dtype.kind == 'i' and int(arr.min()) < 0:
        raise ValueError('document_index: document ids must be non-negative')
    keys, inv = np.unique(arr.reshape(-1), return_inverse=True)
    return keys, inv.reshape(arr.shape)


def aggregate_nll_by_document(nll, doc_ids, *, document_index_map=None):
    """Aggregate a per-target NLL matrix into per-DOCUMENT NLL / PPL.

    ``nll`` is a ``[n_rows, span]`` (or flat) tensor/array of next-token NLLs;
    ``doc_ids`` has exactly the same shape and gives the document identity of
    each predicted target token.  Fixed evaluation rows are arbitrary windows,
    so a document can span several rows and a row can mix several documents;
    summing by row and calling each row a document double-counts documents in
    the bootstrap.  This function groups tokens by document identity instead.

    ``document_index_map`` is the ``(keys, inverse)`` returned by
    :func:`document_index` for the FULL target slice.  Supplying it makes
    chunked evaluation use one global document order; when omitted it is
    computed here.  Documents with no target in this slice (the map came from a
    longer prefix) are omitted from the returned vectors rather than reported
    with a zero token count and a NaN PPL: a document that has not started yet
    is not an evaluation unit of this cell.
    """
    if not torch.is_tensor(nll):
        nll = torch.as_tensor(nll)
    arr = np.asarray(doc_ids)
    if tuple(arr.shape) != tuple(nll.shape):
        raise ValueError(
            f'aggregate_nll_by_document: doc_ids shape {arr.shape} != nll shape '
            f'{tuple(nll.shape)}')
    if document_index_map is None:
        keys, inv = document_index(arr)
    else:
        try:
            keys, inv = document_index_map
        except (TypeError, ValueError):
            raise ValueError(
                'aggregate_nll_by_document: document_index_map must be a '
                '(keys, inverse) pair') from None
        keys = np.asarray(keys)
        inv = np.asarray(inv)
        if keys.ndim != 1 or keys.size == 0:
            raise ValueError(
                'aggregate_nll_by_document: document_index_map keys must be a '
                f'non-empty 1-D array, got shape {keys.shape}')
        if keys.dtype.kind not in 'iu':
            raise ValueError(
                'aggregate_nll_by_document: document_index_map keys must be '
                f'integers, got dtype {keys.dtype}')
        if keys.dtype.kind == 'i' and int(keys.min()) < 0:
            raise ValueError(
                'aggregate_nll_by_document: document_index_map keys must be '
                'non-negative')
        if keys.size > 1 and not bool(np.all(keys[1:] > keys[:-1])):
            # `document_index` returns a sorted unique key array; the stored
            # per-unit vectors are documented as that canonical layout.  An
            # unsorted/duplicated map would make the returned layout depend on
            # caller order and break the permutation contract.
            raise ValueError(
                'aggregate_nll_by_document: document_index_map keys must be '
                'strictly increasing (the canonical sorted-id layout)')
        if inv.dtype.kind not in 'iu':
            raise ValueError(
                'aggregate_nll_by_document: document_index_map inverse must be '
                f'integers, got dtype {inv.dtype}')
        if inv.size != int(nll.numel()):
            raise ValueError(
                f'aggregate_nll_by_document: index covers {inv.size} targets, '
                f'expected {int(nll.numel())}')
        if inv.size:
            _inv_min = int(inv.min())
            _inv_max = int(inv.max())
            if _inv_min < 0 or _inv_max >= int(keys.size):
                raise ValueError(
                    'aggregate_nll_by_document: document_index_map inverse '
                    f'spans [{_inv_min}, {_inv_max}] but there are only '
                    f'{int(keys.size)} documents')
        inv = inv.reshape(arr.shape)
        # A map for a longer prefix is valid here, but it must still describe
        # THIS slice's identities.  Without this check a stale/mismatched map
        # would silently regroup the same tokens into different documents.
        if not np.array_equal(keys[inv.reshape(-1)], arr.reshape(-1)):
            raise ValueError(
                'aggregate_nll_by_document: document_index_map does not match '
                'doc_ids; refusing to regroup tokens under a different map')
    n_docs = int(keys.size)
    idx = torch.as_tensor(inv.reshape(-1).astype(np.int64, copy=False),
                          dtype=torch.long, device=nll.device)
    vals = nll.reshape(-1)
    nll_doc = _det_segment_sum(n_docs, idx, vals,
                               dtype=torch.float64).detach().cpu().tolist()
    tok_doc = _det_segment_sum(
        n_docs, idx, torch.ones_like(idx), dtype=torch.long
    ).detach().cpu().tolist()
    n_tokens = [int(x) for x in tok_doc]
    nll_values = [float(x) for x in nll_doc]
    # Drop documents that have no target in this slice.  A global map (built
    # on the longest requested prefix) otherwise carries documents that start
    # later, giving them 0 tokens and NaN PPL; that inflates n_docs and makes
    # the per-document vectors length-inconsistent for no measurement reason.
    _present = [i for i, n in enumerate(n_tokens) if n > 0]
    keys_present = np.asarray(keys)[_present]
    nll_values = [nll_values[i] for i in _present]
    n_tokens = [n_tokens[i] for i in _present]
    ppl = [float(_safe_exp_ppl(nll_values[i] / n_tokens[i]))
           for i in range(len(_present))]
    return {
        'doc_granularity': 'document',
        'n_docs': len(_present),
        'nll_per_doc': nll_values,
        'n_tokens_per_doc': n_tokens,
        'ppl_per_doc': ppl,
        'doc_ids': [format(int(k), '016x') for k in keys_present.tolist()],
        'doc_ids_fp': _ids_fp(keys_present),
        'nll_per_doc_total': float(sum(nll_values)),
        'n_tokens_total': int(sum(n_tokens)),
        # The per-document vectors above stay in the canonical sorted-id
        # layout (backward compatible).  Corpus/first-occurrence order is a
        # SEPARATE measurement: `permutation` maps those vectors to it, and
        # `sha256` binds the ordered id sequence.  Order-dependent readouts
        # must use this permutation or refuse; `np.unique` sorting is not
        # corpus order.
        'unit_order': _unit_order_from_index(
            np.asarray(keys), _present, inv, int(sum(n_tokens)), 'document'),
    }


def aggregate_nll_by_article(nll, article_ids, *, document_index_map=None):
    """Aggregate per-target NLL into per-ARTICLE NLL / PPL.

    WikiText rows are lines, not articles; a line-level bootstrap resamples
    paragraphs of the same article and therefore understates the sampling
    uncertainty of an article-level claim.  This is the same deterministic
    segment sum as :func:`aggregate_nll_by_document`, with article-named
    fields so a consumer cannot mistake one unit for the other.

    The pooled totals are identical to the document aggregation for the SAME
    targets (only the grouping differs); the caller keeps both vectors so the
    frozen line-level contract is untouched and the article-level readout is
    additive evidence.
    """
    doc = aggregate_nll_by_document(
        nll, article_ids, document_index_map=document_index_map)
    n_articles = int(doc['n_docs'])
    return {
        'article_granularity': 'article',
        'unit_definition': ('top-level " = Title = " article; rows are lines '
                            'and are not independent of their article'),
        'n_articles': n_articles,
        'nll_per_article': list(doc['nll_per_doc']),
        'n_tokens_per_article': list(doc['n_tokens_per_doc']),
        'ppl_per_article': list(doc['ppl_per_doc']),
        'article_ids': list(doc['doc_ids']),
        'article_ids_fp': doc['doc_ids_fp'],
        'nll_per_article_total': float(doc['nll_per_doc_total']),
        'n_tokens_article_total': int(doc['n_tokens_total']),
        'article_source': 'per_token_article_map',
        'unit_order': dict(doc['unit_order'], unit_kind='article'),
    }


@torch.inference_mode()
def eval_ppl(model, val_batch, device, chunk=8, eval_rows=None, eval_seed=0, return_stats=False,
             collect_block_stats=False, val_bnd=None, bnd_tol=1):
    # `collect_block_stats=True` enables the compression-report hooks for this
    # same forward pass, so block-length/boundary diagnostics come from it too.
    # A chunk is a set of independent sequences (no cross-row op in the
    # model), so capping it by token count does not change any per-row value.
    _span = max(1, int(val_batch.shape[1]) - 1)
    chunk = _eval_adaptive_chunk(model, val_batch, device, chunk)
    was_training = model.training
    model.eval()
    need_reg_layers = [blk.attn for blk in model.blocks if hasattr(blk.attn, 'need_reg')]
    need_reg_prior = [_a.need_reg for _a in need_reg_layers]
    nll = torch.zeros((), dtype=torch.float64, device=device)
    # Per-ROW NLL (eval-subset uncertainty must be visible, not only the
    # seed-to-seed spread).  A row is a fixed evaluation WINDOW, not
    # necessarily one document: use `eval_length_gen` with a per-token
    # document mapping when a document-level bootstrap is required.  With
    # eval_rows sampling, a row with no sampled target gets 0 tokens and is
    # excluded by the bootstrap consumer.
    _row_nll = torch.zeros(int(val_batch.shape[0]), dtype=torch.float64, device=device) if return_stats else None
    _row_ntok = torch.zeros(int(val_batch.shape[0]), dtype=torch.long, device=device) if return_stats else None
    for _a in need_reg_layers:
        _a.need_reg = False
    if collect_block_stats:
        enable_block_stats(model, True)
    _compression_out = None
    try:
        ntok = 0
        V = None
        _sel = None
        if eval_rows is not None and int(eval_rows) > 0:
            _n_rows = int(val_batch.shape[0])
            _n_tgt = max(0, int(val_batch.shape[1]) - 1)
            _flat = _n_rows * _n_tgt
            _k = min(int(eval_rows), _flat)
            if _k < _flat:
                _g = torch.Generator().manual_seed(int(eval_seed))
                # A full randperm costs O(n_rows*T) memory even for a tiny
                # eval_rows sample (e.g. 1 GB at 4096x32768).  Sampling with
                # replacement and de-duplicating gives the same uniform
                # k-subset distribution with O(k) memory; only near-full
                # samples fall back to the historical permutation.
                if _k * 4 >= _flat:
                    _sel = torch.randperm(_flat, generator=_g)[:_k]
                else:
                    _sel = torch.empty(0, dtype=torch.long)
                    while int(_sel.numel()) < _k:
                        _draw = torch.randint(0, _flat,
                                              (max(16, 2 * int(_k - _sel.numel())),),
                                              generator=_g)
                        _sel = torch.unique(torch.cat([_sel, _draw]), sorted=True)
                    if int(_sel.numel()) > _k:
                        _sel = _sel[torch.randperm(int(_sel.numel()), generator=_g)[:_k]]
                # keep the sorted host copy: per-chunk membership bounds are
                # integer range queries on a sorted table — searchsorted on the
                # host gives the same slice the boolean-mask form selected
                # (host-side integer equivalence, design notes §6.2).
                _sel_host = torch.sort(_sel).values
                _sel_sorted = _sel_host.to(device)
                print(f'[eval_ppl] scoring a uniform sample of {_k}/{_flat} target rows (eval_rows={int(eval_rows)}, eval_seed={int(eval_seed)}) — this is a SAMPLED estimate, not the full-set PPL')
            else:
                print(f'[eval_ppl] eval_rows={int(eval_rows)} covers the whole {_flat}-row target space; scoring the FULL set (this is the population statistic, not a sample)')
        _vhost = None
        if device.type == 'cuda':
            _vdev = _dev_str(device)
            _vc = _PINNED_HOST_CACHE.get('val')
            # In-place mutation of the same tensor keeps its id/shape but
            # changes `_version`; without it the pinned copy served stale rows.
            _vver = getattr(val_batch, '_version', None)
            if (_vc is not None and _vc[0] is val_batch and _vc[1] == tuple(val_batch.shape)
                    and _vc[3] == _vdev and _vc[4] == _vver):
                _vhost = _vc[2]
            else:
                try:
                    _vhost = torch.as_tensor(val_batch).pin_memory()
                    _PINNED_HOST_CACHE['val'] = (val_batch, tuple(val_batch.shape), _vhost,
                                                 _vdev, _vver)
                except RuntimeError:
                    _vhost = torch.as_tensor(val_batch)
        else:
            _vhost = torch.as_tensor(val_batch)
        _async = _vhost.is_pinned()
        _span = max(0, int(val_batch.shape[1]) - 1)

        def _run_rows(i, cur):
            nonlocal ntok, V, nll, _row_nll, _row_ntok
            _lo = _hi = 0
            if _sel is not None:
                # Host-side range count on the sorted selection: chunks with
                # no sampled target contribute nothing to nll/ntok.
                _base = i * _span
                _wide = min(cur, int(val_batch.shape[0]) - i) * _span
                if _wide > 0:
                    _lo = int(torch.searchsorted(_sel_host, _base))
                    _hi = int(torch.searchsorted(_sel_host, _base + _wide))
                if _hi <= _lo:
                    return
            elif _span == 0:
                return
            ids = _vhost[i:i + cur].to(device, non_blocking=_async)
            # Trunk + per-slice LM head; comp_reg / need_reg handling is
            # identical to model(ids) because forward() itself is exactly
            # trunk() followed by head().
            h = model.trunk(ids)
            if _sel is not None:
                _local = _sel_sorted[_lo:_hi] - _base
                _ri = torch.div(_local, _span, rounding_mode='floor')
                _ci = _local - _ri * _span
                # `_row_nll` / `_row_ntok` are indexed by the GLOBAL document
                # row, while `_ri` is chunk-local because it indexes this
                # chunk's `h`/`ids`.  Offset it before the scatters.
                _src_rows = _ri + i
                # gather the selected hidden rows BEFORE the head: each output
                # row of the Linear is an independent function of one hidden
                # row, so head(h[sel]) == head(h)[sel]
                _rows = model.head(h[_ri, _ci, :])
                if V is None:
                    V = _rows.size(-1)
                _tgts = ids[_ri, _ci + 1]
                del _ri, _ci
                _ce = F.cross_entropy(_rows, _tgts, reduction='none').double()
                nll += _ce.sum()
                if return_stats:
                    # Deterministic segmented sum (no CUDA atomics): the
                    # per-document NLL feeding the document bootstrap must be
                    # reproducible on the same machine.
                    # `_src_rows` is non-decreasing: it is derived from the
                    # sorted global selection table, so the stable sort in the
                    # accumulator would be an identity.
                    _row_nll += _det_segment_sum(_row_nll.numel(), _src_rows, _ce, dtype=torch.float64, sorted_indices=True)
                    _row_ntok += _det_segment_sum(_row_ntok.numel(), _src_rows, torch.ones_like(_src_rows), dtype=torch.long, sorted_indices=True)
                ntok += int(_tgts.numel())
                del _rows, _tgts, _ce, _src_rows
            else:
                _tgts = ids[:, 1:]
                for _s0 in range(0, _span, 64):
                    _sl = slice(_s0, min(_s0 + 64, _span))
                    _lg = model.head(h[:, _sl, :])
                    if V is None:
                        V = _lg.size(-1)
                    _ce = F.cross_entropy(_lg.transpose(1, 2), _tgts[:, _sl], reduction='none').double()
                    nll += _ce.sum()
                    if return_stats:
                        _b = int(_ce.shape[0])
                        _row_nll[i:i + _b] += _ce.sum(dim=1)
                        _row_ntok[i:i + _b] += int(_ce.shape[1])
                    del _lg, _ce
                ntok += int(_tgts.numel())
                del _tgts
            del ids, h

        i = 0
        _n_rows = int(val_batch.shape[0])
        while i < _n_rows:
            _cur = min(chunk, _n_rows - i)
            _nll_snap = nll.clone()
            _rn_snap = _row_nll.clone() if return_stats else None
            _rt_snap = _row_ntok.clone() if return_stats else None
            _stats_snap = _snapshot_block_stats(model) if collect_block_stats else None
            try:
                _run_rows(i, _cur)
            except RuntimeError as _oe:
                nll.copy_(_nll_snap)
                if return_stats:
                    _row_nll.copy_(_rn_snap)
                    _row_ntok.copy_(_rt_snap)
                if _stats_snap is not None:
                    _restore_block_stats(_stats_snap)
                if ('out of memory' in str(_oe).lower()) and _cur > 1:
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    chunk = max(1, _cur // 2)
                    print(f'[eval_ppl] device-memory exhaustion; retrying with chunk={chunk}')
                    continue
                raise
            i += _cur
    finally:
        if collect_block_stats:
            try:
                _compression_out = _compression_report_summary(model, int(val_batch.shape[1]), val_bnd, bnd_tol)
            except Exception as _ce:
                print(f'[eval_ppl] WARNING: compression diagnostics failed ({type(_ce).__name__}: {_ce})')
                _compression_out = None
            enable_block_stats(model, False)
        for _a, _prev in zip(need_reg_layers, need_reg_prior):
            _a.need_reg = _prev
        model.train(was_training)
    if not ntok:
        if not return_stats:
            return float('nan')
        _empty = {'ppl': float('nan'), 'nll_sum': float('nan'), 'n_tokens': 0,
                  'nll_per_row': [], 'n_tokens_per_row': [], 'eval_rows': int(eval_rows) if eval_rows is not None else None,
                  'eval_seed': int(eval_seed), 'eval_subset': int(val_batch.shape[0])}
        if collect_block_stats:
            _empty['compression'] = _compression_out
        return _empty
    try:
        ppl = math.exp(nll.item() / ntok)
    except OverflowError:
        ppl = float('inf')
    if not return_stats:
        return ppl
    _out = {
        'ppl': float(ppl),
        'nll_sum': float(nll.item()),
        'n_tokens': int(ntok),
        'nll_per_row': [float(x) for x in _row_nll.detach().cpu().tolist()],
        'n_tokens_per_row': [int(x) for x in _row_ntok.detach().cpu().tolist()],
        'eval_rows': int(eval_rows) if eval_rows is not None else None,
        'eval_seed': int(eval_seed),
        'eval_subset': int(val_batch.shape[0]),
    }
    if collect_block_stats:
        _out['compression'] = _compression_out
    return _out
