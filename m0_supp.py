#!/usr/bin/env python3
"""M0/S1 驱动 — B 档扩规模路线图（已选定）的代码入口。

B 档落在两处代码支撑上（均在 exp_lib.py）：
  1. FineWeb-Edu 语料与 32k BPE 分词缓存（load_fineweb_edu：npy 缓存 +
     sha256 内容指纹 + 原子写，纪律与 load_wikitext 一致；离线机器可用
     FWEDU_LOCAL_JSONL 指向本地 jsonl/txt 语料）；
  2. S1–S4 规模阶梯与 S1 主面板配置（SCALE_LADDER / RUN_S1，全部为
     规划值，由 SCALE_LADDER / RUN_S1 统一定义）。

子命令：
  params   零 GPU：按 SmallGPT 实例化统计 S1–S4 的 csa_dynamic 参数量
           （32k 词表；S4 按规划的解析外推口径），对照规划值；
           写入 results_m0_params/params_audit.json（含代码/环境/git 溯源
           戳与 experiments.jsonl 记录），同时打印。
  data     构建 / 校验 FineWeb-Edu 缓存（默认 cap = S1 的 5B token；
           M0_TOKENS 可覆盖，例如 M0_TOKENS=15000000000 先落全量缓存）。
  smoke    端到端自检：微型模型 @ seq 8192 走完 fineweb 数据路径、
           稠密 / 固定稀疏 / 动态稀疏三条注意力路径与 8k 长度的
           掩码 / 分段 / 索引几何（M0_SMOKE_* 可调规模）。
  kernel   G0 验证套件：三算子（块压缩 /
           索引 top-k / 稀疏注意力）的 kernel/eager 同种子数值对照（值 +
           全部梯度）、模型级前反向对照（csa_fixed / csa_dynamic /
            hybrid_dynamic），以及 CUDA 机器上的稀疏注意力基准
            （判据见本模块的容差说明）。任意设备可跑——
           CPU / 无 triton 时验证 torch 参考后端（数值语义层）。
  analysis 零 GPU：读取 results_b_s1/（及可选的 results_b_s2/）的 summary.json，
           输出描述性配对统计、Holm、分辨率上限与跨规模 log-PPL 效应
           （analysis_m0/scale_analysis.json）。S1×4 / S2×3 一律标为描述性。
  full     S1 主面板（5 臂 × 4 seeds × 5B token，results_b_s1/）。
           CSA_PAR=auto 一卡一实验（par.py 零改动，§7 排班口径）。
           M0_PANEL=S2 切换到 S2 主面板（RUN_S2：同 B_ARMS、同
           seq8192/bs8 几何，5 臂 × 3 seeds × 12B token，results_b_s2/，
           规划值口径同 S1 主面板）；面板选择只影响 run 配置
           与记账，data / smoke / kernel 与面板无关。M0_AUTO_CHAIN=S1,S2
           可让 full 顺序串行多个面板（各自的 cap/状态文件，中间不关机）。
  auto     无人值守入口（「G0 结果出来后人工决定 B 档是否继续」的自动化）：
           kernel（G0 套件，fail-closed：数值失败即退出、基准/TOST 未达
           门限不写证书）→ require_g0 证书复核（绑定当前 git 树/代码/内核
           戳）→ 自动启动面板链。面板选择同 full（默认 S1；M0_AUTO_CHAIN=
           S1,S2 串行；M0_PANEL 单选）。证书已对本树有效时跳过重验证直接
           续跑；G0 通过后面板默认 CSA_KERNEL=1（调用方显式设置优先）。
           任一门失败：不启动面板、不 push、留 40 分钟诊断窗口后关机——
           「证书存在 = 双判据全过 = 继续」由机器执行，剩余人工点只有
           平台侧的会话重启。

运行时上限：按面板取 M0_CAP / M2_CAP（可用 M0_CAP_HOURS /
M2_CAP_HOURS 覆盖），状态文件 run_time_state_m0.json /
run_time_state_m2.json（内核模式 .k 后缀，两面板互不混账）；超限单元
拒绝启动、中途到达上限则硬截断且不留半条测量。`full` 启动前先用
`require_g0` 校验 `results_m0_g0/g0_certificate.json`（内核数值、训练级
TOST 等检查都由 `kernel` 子命令写出并绑定代码/内核戳/git 树）；
没有有效证书时拒绝启动，只能用 CSA_ALLOW_UNVERIFIED_EAGER=1 做显式调试。
`full` 还会在真实面板几何上做短校准，把各臂的实测 s/step 记入 TimeGuard；
未校准的解析外推在 S1 尺度上会高估到使全部单元被 cap 跳过的程度。
"""
import gc, json, math, os, re, shutil, subprocess, sys, time, traceback
import numpy as np
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except (ValueError, OSError):
            pass
REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
sys.path.insert(0, REPO)
import exp_lib as L
import torch

def git_push(msg):
    if os.environ.get('M0_NO_PUSH'):
        print(f'[git] push skipped (M0_NO_PUSH): {msg}')
        return True
    run = lambda *a: subprocess.run(a, cwd=REPO, capture_output=True, text=True, encoding='utf-8', errors='replace')
    ra = run('git', 'add', '-A')
    if ra.returncode != 0:
        print(f'[git] add FAILED: {(ra.stdout + ra.stderr)[-300:]} — NOT pushing (a commit could miss unstaged work); leaving the tree for inspection')
        return False
    r = run('git', 'commit', '-m', msg)
    committed = r.returncode == 0
    if not committed and 'nothing to commit' not in r.stdout + r.stderr:
        print(f'[git] commit problem: {(r.stdout + r.stderr)[-300:]}')
        if run('git', 'status', '--porcelain').stdout.strip():
            print('[git] the commit failed while changes are still uncommitted — NOT pushing (a push would publish stale history and report success); leaving the tree for inspection')
            return False
    _branch = run('git', 'rev-parse', '--abbrev-ref', 'HEAD').stdout.strip()
    if _branch != 'HEAD':
        rb = run('git', 'pull', '--rebase', '--autostash', 'origin', _branch)
        if rb.returncode != 0:
            print(f'[git] rebase onto origin failed: {(rb.stdout + rb.stderr)[-300:]}')
            run('git', 'rebase', '--abort')
        r = run('git', 'push', 'origin', 'HEAD')
    else:
        _up = run('git', 'symbolic-ref', '-q', 'refs/remotes/origin/HEAD').stdout.strip()
        _dst = _up[len('refs/remotes/origin/'):] if _up.startswith('refs/remotes/origin/') else ''
        if not _dst or run('git', 'rev-parse', '--verify', '-q', f'refs/remotes/origin/{_dst}').returncode != 0:
            _dst = 'main'
        print(f'[git] detached HEAD — pushing explicitly to origin/{_dst}')
        r = run('git', 'push', 'origin', f'HEAD:{_dst}')
    ok = r.returncode == 0
    tail = (r.stdout + r.stderr).strip()[-300:]
    print(f'[git] push {('OK' if ok else 'FAILED')}: {msg}' + ('' if ok else f'  ({tail})'))
    if not ok and committed:
        print('[git] WARNING: the commit is only on this instance — the work is NOT on the remote.  Re-push before shutting down.')
    return ok

def schedule_shutdown(delay_s=90):
    if os.environ.get('M0_NO_SHUTDOWN'):
        print('[m0] shutdown suppressed (M0_NO_SHUTDOWN)')
        return
    marker = 'm0autoshutdown'
    subprocess.Popen(['bash', '-c', f'sleep {delay_s}; echo {marker}; shutdown -h now'], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f'[m0] instance will SHUT DOWN in {delay_s}s. (cancel: pkill -f {marker})')

def _ladder_params(name):
    """csa_dynamic 参数量：一层实例化 + 均匀层线性组合（make_layer_cfgs 对
    csa_dynamic 逐层同构，权重形状只依赖 d/H/dh 与 AttnCfg 常量，不依赖
    层数）；tok/pos/head 词表与位置嵌入及最终 RMSNorm 单独计入。外推
    恒等式在 run_params 里用 S1 的全量实例化对拍验证。"""
    sc = L.SCALE_LADDER[name]
    seq = L.RUN_S1['seq_len']
    with torch.random.fork_rng():
        cfgs = L.make_layer_cfgs(1, 'csa_dynamic')
        m1 = L.SmallGPT(L.B_VOCAB, sc['d'], 1, sc['n_heads'], sc['d_head'], seq, cfgs)
        one_layer = L.count_params(m1)
    del m1
    gc.collect()
    emb = 2 * L.B_VOCAB * sc['d'] + seq * sc['d'] + sc['d']
    per_layer = one_layer - emb
    return emb + sc['n_layers'] * per_layer, per_layer

_PLAN_PARAMS = {'S1': 0.14e9, 'S2': 0.26e9, 'S3': 0.62e9, 'S4': 1.3e9}

def run_params(outdir='results_m0_params'):
    """S1–S4 参数量审计：SmallGPT 实例化统计（S1 全量对拍，S2–S4 均匀层组合，
    S4 即规划的解析外推口径），对照规划值。

    结果连同代码/环境/git 溯源戳写入 `outdir/params_audit.json` 并登记
    experiments.jsonl——参数量是可引用的数字，只打印不留痕
    就无法追溯是哪一版代码算出的。"""
    print(f'[params] SmallGPT(csa_dynamic) @ vocab={L.B_VOCAB} seq={L.RUN_S1["seq_len"]} — 实例化统计（S1 全量对拍，S2–S4 均匀层组合，S4 即规划的解析外推口径）\n')
    sc1 = L.SCALE_LADDER['S1']
    with torch.random.fork_rng():
        cfgs = L.make_layer_cfgs(sc1['n_layers'], 'csa_dynamic')
        full = L.SmallGPT(L.B_VOCAB, sc1['d'], sc1['n_layers'], sc1['n_heads'], sc1['d_head'], L.RUN_S1['seq_len'], cfgs)
        n_full = L.count_params(full)
    del full
    gc.collect()
    n_s1, _ = _ladder_params('S1')
    if n_s1 != n_full:
        raise AssertionError(f'[params] 均匀层组合 {n_s1:,} != S1 全量实例化 {n_full:,} — 外推恒等式被破坏，停止输出（修 make_layer_cfgs / SmallGPT 的逐层同构性之后再算）')
    print(f'[params] self-check: S1 full instantiation = {n_full:,} == uniform-layer composition  OK\n')
    print(f"{'档':<4}{'配置':<28}{'参数量':>16}{'规划值':>14}{'比值':>8}")
    tiers = []
    for name in ('S1', 'S2', 'S3', 'S4'):
        sc = L.SCALE_LADDER[name]
        n, _pl = _ladder_params(name)
        cfg_txt = f"d={sc['d']}/{sc['n_layers']}L/{sc['n_heads']}H/dh={sc['d_head']}"
        print(f"{name:<4}{cfg_txt:<28}{n:>16,}{_PLAN_PARAMS[name] / 1e9:>12.2f}B{n / _PLAN_PARAMS[name]:>8.2f}")
        tiers.append({'tier': name, 'config': {'d': sc['d'], 'n_layers': sc['n_layers'], 'n_heads': sc['n_heads'], 'd_head': sc['d_head']}, 'params': int(n), 'plan_params': _PLAN_PARAMS[name], 'ratio_to_plan': n / _PLAN_PARAMS[name]})
    _budget_tiers = []
    for _name, _run in (('S1', L.RUN_S1), ('S2', L.RUN_S2)):
        _p = next((t['params'] for t in tiers if t['tier'] == _name), None)
        _budget_tiers.append({'name': _name, 'cfg': dict(_run, vocab=_run.get('vocab_size')), 'params': _p})
    _budget = L.scale_ladder_audit(_budget_tiers)
    print(f"\n[params] scale-transfer audit: {_budget['status']}  "
          f"budget_matched={_budget.get('budget_matched')}  "
          f"tokens/param ratio={_budget.get('tokens_per_param_ratio')}")
    for _w in _budget.get('warnings') or []:
        print(f'[params] WARNING: {_w}')
    print('\n[params] 规划值约 0.14B / 0.26B / 0.62B / 1.3B；差异超过两成即需要回改阶梯配置或修订规划。')
    os.makedirs(outdir, exist_ok=True)
    L.stamped_analysis('m0_params_audit', {
        'vocab': int(L.B_VOCAB), 'seq_len': int(L.RUN_S1['seq_len']),
        'method': 'SmallGPT instantiation count; S1 verified against a full instantiation, S2-S4 by uniform-layer composition (S4 = the planned analytic extrapolation)',
        'self_check': {'s1_full_instantiation': int(n_full), 's1_uniform_layer_composition': int(n_s1), 'passed': bool(n_s1 == n_full)},
        'plan_source': 'SCALE_LADDER planning values', 'tiers': tiers,
        'scale_ladder_audit': _budget,
    }, outdir=outdir, name='params_audit.json')
    print(f'[params] wrote {outdir}/params_audit.json (stamped: code + git provenance + environment)')

def run_data():
    tokens = int(os.environ.get('M0_TOKENS') or L.RUN_S1['n_train_tokens'])
    t0 = time.time()
    train_ids, val_batch, vocab, _tok, val_bnd = L.load_fineweb_edu(L.RUN_S1['seq_len'], tokens, vocab_size=L.B_VOCAB, val_seqs=L.RUN_S1.get('val_seqs', 512))
    dt = time.time() - t0
    print(f'\n[m0-data] cache READY: train tokens = {len(train_ids):,}  vocab = {vocab}  val sequences = {val_batch.shape[0]}  (built/verified in {dt / 60:.1f} min)')
    print('[m0-data] 供 full / smoke 直接复用；离线重建用 FWEDU_LOCAL_JSONL，换 cap 用 M0_TOKENS（新 cap = 新缓存文件，旧面板指纹不受影响）')

def run_overfit_check(variants=None, steps=250, arch=None, seq_len=None, vocab=256, mlp_ratio=None, cfg=None):
    """Pre-flight small-batch overfit check on synthetic ids (independent of
    the corpus, so it cannot be blocked by a 5B-token cache build).  A failure
    is a hard stop before any panel is allowed to start.

    `arch` optionally probes the architecture family the panel is about to
    train (S1: d=768/12L/12H/dh64): a head/layer-geometry mismatch that the tiny
    default probe (d=64/2L/4H) can never see is caught before the panel
    starts.  `seq_len` should exercise the real block geometry (see
    `L.probe_seq_len`), `mlp_ratio` carries the panel's per-variant MLP widths.
    Unset = the default tiny probe."""
    variants = list(variants or L.B_ARMS)
    cfg = dict(cfg or {})
    if seq_len is None:
        seq_len = L.probe_seq_len(cfg.get('seq_len', 512))
    _arch = arch or None
    print(f'\n[overfit] small-batch sanity check: {len(variants)} variant(s), wd=0, comp_lambda=0, no augmentation, arch={_arch or "tiny"}, seq_len={seq_len}')
    _prov = L.experiment_provenance({'dataset': 'synthetic', 'n_train_tokens': 0, 'seq_len': int(seq_len)}, vocab=int(vocab))
    _pfp = L.preflight_fingerprint(cfg, stage='m0_synthetic_overfit', vocab=int(vocab), probe_cfg={'arch': _arch, 'seq_len': int(seq_len), 'mlp_ratio': mlp_ratio})
    os.makedirs('results_overfit_sanity_m0', exist_ok=True)
    try:
        reports = L.overfit_guard(variants, steps=int(steps), vocab=int(vocab),
                                  seq_len=int(seq_len), arch=_arch,
                                  mlp_ratio=mlp_ratio)
    except RuntimeError as _fe:
        _reports = [dict(r) for k, r in L._OVERFIT_REPORTS.items() if k[0] in set(variants)]
        L.atomic_write_json('results_overfit_sanity_m0/overfit_sanity.json', {
            'kind': 'small_batch_overfit_sanity_panel', 'stage': 'm0_preflight',
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'run_cfg': _pfp,
            'passed': False, 'error': f'{_fe}', 'provenance': _prov,
            'reports': _reports,
        })
        L.log_experiment_event('results_overfit_sanity_m0', {'kind': 'preflight', 'tracking_schema': 2, 'status': 'failed', 'stage': 'm0_overfit', 'run_cfg': _pfp, 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'code': L.code_stamp(), 'provenance': _prov, 'variants': variants, 'error': f'{_fe}', 'sanity': {'passed': False, 'n_reports': len(_reports)}})
        raise
    if not reports:
        return []
    ok = all(r['passed'] for r in reports)
    L.atomic_write_json('results_overfit_sanity_m0/overfit_sanity.json', {
        'kind': 'small_batch_overfit_sanity_panel',
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'run_cfg': _pfp,
        'passed': ok,
        'provenance': _prov,
        'reports': reports,
    })
    _seeds = [int(x) for x in (cfg.get('seeds') or [0])]
    _repro_seeds = list(dict.fromkeys([_seeds[0], _seeds[-1]]))
    try:
        _repro = L.reproducibility_guard(variants, arch=_arch, vocab=int(vocab),
                                         mlp_ratio=mlp_ratio, seq_len=int(seq_len),
                                         batch_size=2, seeds=_repro_seeds,
                                         strict=L._is_confirmatory_cfg(cfg))
    except RuntimeError as _fe:
        _reports = [dict(r) for k, r in L._REPRO_REPORTS.items() if k[0] in set(variants)]
        L.atomic_write_json('results_overfit_sanity_m0/reproducibility.json', {
            'kind': 'duplicate_run_reproducibility_panel', 'stage': 'm0_preflight',
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'run_cfg': _pfp,
            'passed': False, 'error': f'{_fe}', 'provenance': _prov,
            'reports': _reports,
        })
        L.log_experiment_event('results_overfit_sanity_m0', {'kind': 'preflight', 'tracking_schema': 2, 'status': 'failed', 'stage': 'm0_repro', 'run_cfg': _pfp, 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'code': L.code_stamp(), 'provenance': _prov, 'variants': variants, 'error': f'{_fe}', 'reproducibility': {'passed': False, 'n_reports': len(_reports)}})
        raise
    if _repro:
        L.atomic_write_json('results_overfit_sanity_m0/reproducibility.json', {
            'kind': 'duplicate_run_reproducibility_panel',
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'run_cfg': _pfp,
            'passed': all(bool(r.get('passed')) for r in _repro),
            'provenance': _prov,
            'reports': _repro,
        })
    L.log_experiment_event('results_overfit_sanity_m0', {'kind': 'preflight', 'tracking_schema': 2, 'status': ('ok' if ok and all(bool(r.get('passed')) for r in _repro) else 'failed'), 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'code': L.code_stamp(), 'provenance': _prov, 'variants': variants, 'run_cfg': _pfp, 'sanity': {'passed': ok, 'n_reports': len(reports)}, 'reproducibility': {'passed': all(bool(r.get('passed')) for r in _repro), 'n_reports': len(_repro)}})
    if not ok:
        _bad = [r['variant'] for r in reports if not r['passed']]
        raise RuntimeError(f'small-batch overfit sanity check FAILED for {_bad} — do NOT start training; inspect results_overfit_sanity_m0/overfit_sanity.json')
    print('[overfit] PASSED (necessary, not sufficient)')
    return reports

def run_smoke():
    seq = int(os.environ.get('M0_SMOKE_SEQ') or L.RUN_S1['seq_len'])
    steps = int(os.environ.get('M0_SMOKE_STEPS') or 12)
    smoke_cfg = dict(L.RUN_S1, d=64, n_layers=2, n_heads=4, d_head=16, batch_size=1, seq_len=seq, steps=steps, warmup=2, eval_every=max(steps // 2, 1), eval_subset=2, val_seqs=2, n_train_tokens=int(os.environ.get('M0_SMOKE_TOKENS') or 200000), fw_fit_docs=int(os.environ.get('M0_SMOKE_FIT_DOCS') or 400), seeds=[0], outdir='results_smoke_m0', variants=['full', 'csa_fixed', 'csa_dynamic'])
    _smoke_ratios = {}
    for _v in smoke_cfg['variants']:
        try:
            _smoke_ratios[_v] = L.variant_mlp_ratio(_v, int(smoke_cfg.get('vocab_size') or 1024), d=64, n_layers=2, n_heads=4, d_head=16, seq_len=int(seq), matched=smoke_cfg.get('matched'), ref_variant=smoke_cfg.get('mlp_match_ref', 'csa_dynamic'))
        except Exception:
            _smoke_ratios[_v] = None
    run_overfit_check(variants=smoke_cfg['variants'],
                      arch={'d': 64, 'n_layers': 2, 'n_heads': 4, 'd_head': 16},
                      seq_len=L.probe_seq_len(seq),
                      vocab=int(smoke_cfg.get('vocab_size') or 1024),
                      mlp_ratio=_smoke_ratios, cfg=smoke_cfg)
    L.run(smoke_cfg, seeds=[0], guard=None, label=f'SMOKE M0 (B 档数据路径 @ seq{seq})', par=False)
    _sp = os.path.join('results_smoke_m0', 'summary.json')
    _got = {}
    if os.path.exists(_sp):
        try:
            _got = json.load(open(_sp, encoding='utf-8'))
        except Exception:
            _got = {}
    _bad = [f'{v}::seed0' for v in smoke_cfg['variants'] if not L.ppl_is_usable((_got.get(f'{v}::seed0') or {}).get('ppl'))]
    if _bad:
        print(f'\n[smoke] FAILED: no usable PPL was measured for {_bad} — the B-tier path (fineweb corpus / seq-{seq} geometry) is degraded; results_smoke_m0/ is KEPT for inspection instead of being deleted.')
        sys.exit(1)
    shutil.rmtree('results_smoke_m0', ignore_errors=True)
    shutil.rmtree('results_overfit_sanity_m0', ignore_errors=True)
    print(f'\n[smoke] PASSED — fineweb 数据路径与 seq-{seq} 的稠密/固定/动态三条注意力路径端到端可用；smoke 输出已删除，未触碰任何实验目录。')

G0_TOST_REL_MARGIN = 0.01


def _training_tost_check(ck, *, seeds=(0, 1, 2, 3, 4, 5), steps=40,
                         variants=('csa_fixed', 'csa_dynamic')):
    """Training-level kernel/eager equivalence (G0 layer 3).

    Operator and model checks certify one forward/backward; they do not prove
    that the two implementations follow the same optimisation trajectory.  Run
    the same tiny model on the same fixed tokens with the torch reference and
    the Triton backend, then apply the pre-registered paired TOST: the mean PPL
    difference must lie inside a +/-1% relative margin at 90% confidence.  The
    `tost_paired` helper uses a Student-t interval (df=n-1), not a z interval.
    """
    vocab, d, nl, nh, dh, seq, bs = 256, 64, 2, 4, 16, 256, 4
    rng = np.random.default_rng(20261019)
    train_ids = rng.integers(0, vocab, size=vocab * 4000, dtype=np.int64)
    val_batch = rng.integers(0, vocab, size=(8, seq), dtype=np.int64)
    out = {'margin_relative': float(G0_TOST_REL_MARGIN), 'steps': int(steps),
           'seeds': [int(x) for x in seeds], 'geometry': {
               'vocab': vocab, 'd': d, 'n_layers': nl, 'n_heads': nh,
               'd_head': dh, 'seq_len': seq, 'batch_size': bs},
           'variants': {}, 'passed': True}
    for variant in variants:
        ref, tri = [], []
        try:
            for seed in seeds:
                with ck.kernel_env('ref'):
                    r = L.train_variant(variant, train_ids, val_batch, vocab, seed=int(seed),
                                        d=d, n_layers=nl, n_heads=nh, d_head=dh,
                                        seq_len=seq, batch_size=bs, steps=int(steps),
                                        eval_every=0, eval_subset=8, log_every=0,
                                        save_ckpt='none', measure_eff=False)
                with ck.kernel_env('triton'):
                    t = L.train_variant(variant, train_ids, val_batch, vocab, seed=int(seed),
                                        d=d, n_layers=nl, n_heads=nh, d_head=dh,
                                        seq_len=seq, batch_size=bs, steps=int(steps),
                                        eval_every=0, eval_subset=8, log_every=0,
                                        save_ckpt='none', measure_eff=False)
                ref.append(float(r['ppl']))
                tri.append(float(t['ppl']))
            deltas = [a - b for a, b in zip(ref, tri)]
            margin = abs(float(np.mean(ref))) * G0_TOST_REL_MARGIN
            st = L.tost_paired(deltas, margin, conf=0.90)
            st['ref_ppl'] = ref
            st['triton_ppl'] = tri
            st['deltas'] = deltas
            st['passed'] = bool(st.get('equivalent') is True)
        except Exception as e:
            st = {'passed': False, 'error': f'{type(e).__name__}: {e}'}
        out['variants'][variant] = st
        out['passed'] = bool(out['passed'] and st.get('passed'))
        print(f"[kernel] 训练级 TOST {variant:<14}: {'PASS' if st.get('passed') else 'FAIL'}"
              + (f"  ({st.get('error')})" if st.get('error') else
                 f"  mean Δ={st.get('mean', float('nan')):+.4f} margin=±{st.get('margin', float('nan')):.4f} "
                 f"p={st.get('p_tost', float('nan')):.3g}"))
    return out


def _write_g0_certificate(ck, *, numeric_pass, benchmark_pass, best_tflops,
                          training_tost):
    """Write the machine-checkable G0 certificate consumed by `require_g0`.

    Every field is an observed value; the file is stamped with the same code
    semantics / kernel stamp / git tree that `_g0_mismatch` re-checks at panel
    start.
    """
    prov = L._git_provenance()
    cert = {
        'status': 'pass' if (numeric_pass and benchmark_pass and training_tost.get('passed')) else 'fail',
        'numeric_pass': bool(numeric_pass),
        'benchmark_pass': bool(benchmark_pass),
        'training_tost_pass': bool(training_tost.get('passed')),
        'code_semantics': L.CODE_SEMANTICS,
        'kernel_stamp': L._kernel_stamp(),
        'git_commit': prov.get('git_commit'),
        'git_diff_sha256': prov.get('git_diff_sha256'),
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'device': str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'cpu',
        'triton_version': getattr(__import__('triton'), '__version__', None) if ck.triton_importable() else None,
        'benchmark_best_tflops': float(best_tflops) if best_tflops else None,
        'training_tost': training_tost,
    }
    path = L.g0_certificate_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    L.atomic_write_json(path, cert)
    print(f'[kernel] G0 certificate {"WRITTEN" if cert["status"] == "pass" else "written as FAIL"} -> {path}')
    return cert


def run_kernel():
    """G0 验证套件。

    判据：
    - 数值一致性：三算子 + 模型级的 kernel/eager 同种子对照（值与全部
      梯度；误差度量 csa_kernels._rel 的分母下限按被比较精度选取——
      fp32 floor_frac=1e-2 / bf16 1.0，纯舍入噪声在该口径下为
      ~1e-4(f32)/~2e-2(bf16) 量级；容差按后端分级——ref 为同算法张量
      表达，fp32 1e-3 / bf16 1e-1；triton 为独立内核实现，fp32 1e-2 /
      bf16 3e-1，属归约分组不同的浮点噪声口径，见 design_notes §5 的
      同类漂移讨论）；
    - 基准（仅 CUDA + triton）：判据见下方的门限常量。
    """
    import csa_kernels as ck
    dev = L.DEVICE
    print('[kernel] G0 验证套件 — 判据: 数值一致性（算子级+模型级）与基准判据（仅 CUDA）')
    print(f'[kernel] device = {dev}   triton 可导入 = {ck.triton_importable()}   CSA_KERNEL = {os.environ.get("CSA_KERNEL") or "(unset)"}')
    backends = ['ref']
    if dev.type == 'cuda' and ck.triton_importable():
        backends.append('triton')
    elif dev.type == 'cuda':
        print('[kernel] triton 不可导入——只验证 torch 参考后端（数值语义层）；安装与 CUDA 匹配的 triton 后重跑可得完整对照')
    dtypes = [torch.float32] + ([torch.bfloat16] if dev.type == 'cuda' else [])
    fail = False
    for impl in backends:
        for dt in dtypes:
            # 容差按后端分级：ref 与 eager 是同一算法的张量表达，残差只剩
            # 归约重结合；triton 为独立内核实现（在线 softmax/分块归约的
            # 结构性重排），另加一个量级。门限必须高于纯舍入噪声（见
            # _rel 的 floor_frac 说明），否则噪声会淹没真实差异。
            tol = {(False, False): 1e-3, (False, True): 1e-1,
                   (True, False): 1e-2, (True, True): 3e-1}[(impl == 'triton', dt != torch.float32)]
            for seed in (0, 1):
                res = ck.selfcheck_operators(device=dev, dtype=dt, impl=impl, seed=seed, verbose=False)
                worst_k = max(res, key=res.get)
                worst = res[worst_k]
                ok = worst < tol
                fail |= not ok
                print(f'[kernel] 算子对照 impl={impl:<6} dtype={str(dt).split(".")[-1]:<8} seed={seed}: worst={worst_k}={worst:.2e}  {"OK" if ok else "FAIL (tol %.0e)" % tol}')
    T_m = int(os.environ.get('M0_KERNEL_T') or 128)
    B_m = int(os.environ.get('M0_KERNEL_B') or 2)
    for impl in backends:
        for variant in ('csa_fixed', 'csa_dynamic', 'hybrid_dynamic'):
            res = ck.selfcheck_model(variant=variant, impl=impl, seed=0, T=T_m, B=B_m, device=dev)
            # 梯度判据：范数相对误差 < 1e-4（主）；逐条目相对误差按后端
            # 分级（ref 1e-3 / triton 5e-2——同算子级的分级理由；前向 ce
            # 逐位一致是更强的主判据）。
            gmax_tol = 5e-2 if impl == 'triton' else 1e-3
            ok = res['ce_abs_diff'] < 1e-5 and res['grad_norm_rel'] < 1e-4 and res['grad_max_rel'] < gmax_tol
            fail |= not ok
            print(f'[kernel] 模型对照 impl={impl:<6} {variant:<14}: {"OK" if ok else "FAIL"}')
    tflops = {}
    if dev.type == 'cuda' and 'triton' in backends:
        for dt in (torch.float32, torch.bfloat16):
            tflops[str(dt)] = ck.bench_attention(dev, impl='triton', dtype=dt,
                                                 iters=int(os.environ.get('M0_KERNEL_ITERS') or 20))
    print()
    best = max(tflops.values()) if tflops else None
    if tflops:
        print(f'[kernel] G0 基准判据：{"PASS" if best >= 150.0 else "NOT YET"}（门限 150 TFLOPS）')
    else:
        print('[kernel] 基准需要 CUDA + triton（本机跳过；G0 基准判据未测量，不算通过）')
    print(f'[kernel] 数值判据：{"PASS" if not fail else "FAIL"}')
    print('[kernel] 说明：内核路径是独立变体（CSA_KERNEL 开启后 run_cfg 指纹追加 -tk1:<backend>）；S1 主面板在 G0 双判据通过前仍被 TimeGuard 拒绝启动')
    if fail:
        sys.exit(1)
    # The training-level layer is the one that can certify a whole panel; run
    # it only on a machine that actually has the Triton backend, and never
    # advertise a pass when the benchmark itself did not reach the gate.
    benchmark_pass = bool(best is not None and best >= 150.0)
    if 'triton' in backends:
        print('[kernel] G0 layer 3: training-level kernel/eager TOST')
        _tost = _training_tost_check(ck)
    else:
        _tost = {'passed': False, 'error': 'CUDA+triton not available; training-level TOST not measured'}
    if benchmark_pass and _tost.get('passed') and not fail:
        _write_g0_certificate(ck, numeric_pass=True, benchmark_pass=True,
                              best_tflops=best, training_tost=_tost)
    else:
        print('[kernel] G0 certificate NOT written/overwritten: ' +
              ('numeric checks failed; ' if fail else '') +
              ('benchmark below threshold; ' if not benchmark_pass else '') +
              ('training-level TOST did not pass' if not _tost.get('passed') else ''))


def _calibrate_panel(run_cfg, guard, panel):
    """Measure the per-step rate at the panel geometry before planning.

    TimeGuard's uncalibrated default is a base-geometry rate extrapolated with
    an analytic factor; on the S1/S2 ladder that extrapolation is many orders
    of magnitude too pessimistic (seq 512 -> 8192 and d 256 -> 768/1024 are far
    outside the fitted range), so a fresh state file would mark every cell
    `skipped_cap` without ever training.  A short real run at the EXACT panel
    geometry gives the guard an exact `sps_by_class` entry per variant, and the
    per-variant key keeps a dense arm from being priced as a sparse one.

    Calibration seconds are booked to the same cap (they are real GPU time).
    `M0_CALIB_STEPS` (default 20) controls the probe length; set it to 0 to
    skip when a prior state file already holds every variant's calibration.
    """
    try:
        calib_steps = int(os.environ.get('M0_CALIB_STEPS', '20'))
    except ValueError:
        print(f"[m0-calib] M0_CALIB_STEPS={os.environ.get('M0_CALIB_STEPS')!r} is not an integer; using 20")
        calib_steps = 20
    variants = list(run_cfg['variants'])
    state = guard._load_state()
    by_class = state.get('sps_by_class') or {}
    missing = [v for v in variants
               if guard._class_key(run_cfg['d'], run_cfg['n_layers'], run_cfg['seq_len'],
                                   run_cfg['batch_size'], v) not in by_class]
    if calib_steps <= 0:
        print(f'[m0-calib] skipped (M0_CALIB_STEPS<=0); {len(missing)}/{len(variants)} variant(s) still uncalibrated: {missing or "none"}')
        return
    if not missing:
        print(f'[m0-calib] all {len(variants)} variant(s) already have an exact panel-geometry calibration')
        return
    print(f'[m0-calib] {len(missing)}/{len(variants)} variant(s) uncalibrated at the panel geometry; '
          f'running {calib_steps} real steps each (booked to the cap) to replace the analytic extrapolation')
    t_data = time.time()
    train_ids, val_batch, vocab, _tok, val_bnd = L.load_run_corpus(run_cfg)
    guard.record_run(time.time() - t_data, 0, 0, 0, 0, 0)
    ok = 0
    for v in missing:
        try:
            ratio = L.variant_mlp_ratio(v, int(vocab), d=int(run_cfg['d']),
                                        n_layers=int(run_cfg['n_layers']),
                                        n_heads=int(run_cfg['n_heads']),
                                        d_head=int(run_cfg['d_head']),
                                        seq_len=int(run_cfg['seq_len']),
                                        matched=run_cfg.get('matched'),
                                        ref_variant=run_cfg.get('mlp_match_ref', 'csa_dynamic'))
            t0 = time.time()
            rec = L.train_variant(v, train_ids, val_batch, vocab, seed=0,
                                  d=int(run_cfg['d']), n_layers=int(run_cfg['n_layers']),
                                  n_heads=int(run_cfg['n_heads']), d_head=int(run_cfg['d_head']),
                                  seq_len=int(run_cfg['seq_len']),
                                  batch_size=int(run_cfg['batch_size']),
                                  steps=calib_steps, lr=run_cfg['lr'],
                                  weight_decay=run_cfg['weight_decay'],
                                  warmup=run_cfg['warmup'],
                                  comp_lambda=run_cfg['comp_lambda'],
                                  delta_lr_mult=run_cfg.get('delta_lr_mult', 10.0),
                                  eval_every=0, eval_subset=1, log_every=0,
                                  val_bnd=val_bnd, mlp_ratio=ratio,
                                  save_ckpt='none', measure_eff=False,
                                  amp_dtype=run_cfg.get('amp_dtype', 'auto'),
                                  eager_batch=run_cfg.get('eager_batch', 'auto'))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.time() - t0
            guard.record_run(dt, calib_steps, int(run_cfg['d']), int(run_cfg['n_layers']),
                             int(run_cfg['seq_len']), int(run_cfg['batch_size']),
                             calib_seconds=rec.get('train_time_s'), variant=v)
            print(f'[m0-calib] {v:<20} {dt / calib_steps:.3f} s/step  ({dt:.1f}s for {calib_steps} steps)')
            ok += 1
        except Exception as e:
            print(f'[m0-calib] {v}: calibration FAILED ({type(e).__name__}: {e}) — the panel will still run it, '
                  'but its projection falls back to the geometry/generic rate')
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    state = guard._load_state()
    by_class = state.get('sps_by_class') or {}
    total = 0.0
    print(f'[m0-calib] projected {panel} cost at the measured rates (includes the 1.15 safety factor):')
    for v in variants:
        sps = by_class.get(guard._class_key(run_cfg['d'], run_cfg['n_layers'],
                                            run_cfg['seq_len'], run_cfg['batch_size'], v))
        est = guard.estimate_seconds(run_cfg['steps'], d=run_cfg['d'], n_layers=run_cfg['n_layers'],
                                     seq_len=run_cfg['seq_len'], batch_size=run_cfg['batch_size'], variant=v)
        cell_h = est / 3600.0
        total += cell_h * len(run_cfg['seeds'])
        print(f'  {v:<20} {("calibrated" if sps is not None else "FALLBACK"):<10} {cell_h:8.2f} h/run  x {len(run_cfg["seeds"])} seeds = {cell_h * len(run_cfg["seeds"]):8.2f} h')
    cap_h = guard.cap_hours()
    print(f'[m0-calib] panel total projection {total:.1f} h vs usable cap {cap_h:.1f} h '
          f'(booked so far {state.get("booked_seconds", 0.0) / 3600:.2f} h)')
    if total + state.get('booked_seconds', 0.0) / 3600.0 > cap_h:
        print('[m0-calib] WARNING: the measured projection exceeds the usable cap. The TimeGuard will skip or hard-truncate '
              'cells that no longer fit (a truncated cell leaves no partial measurement). Raise the cap, reduce the seed/arm '
              'list, or use a faster kernel — do not read the resulting incomplete panel as a full-panel result.')


B_SCALE_DESIGN = {
    'dynamic_blocking': [
        ('csa_dynamic', 'csa_fixed', 'B-tier dynamic vs fixed blocking (CSA)'),
    ],
    'hybrid_hierarchy': [
        ('hybrid_dynamic', 'csa_dynamic', 'B-tier CSA+HCA hybrid vs CSA-only'),
    ],
}


def _records_by_seed(summary, variant, *, primary=False):
    """Measurable records for one arm, keyed by seed.

    ``primary=True`` selects the record's PRIMARY estimand: a nested tuning
    record's full-validation ``ppl`` contains the rows that selected its
    hyper-parameters, so cross-scale pairing must use the holdout PPL (or its
    held-out NLL accounting) exactly like the panel aggregate does.  A record
    that declares the nested estimand but lacks holdout accounting is skipped,
    never silently replaced by the leaky full-validation number.
    """
    out = {}
    for rec in summary.values():
        if not isinstance(rec, dict) or rec.get('variant') != variant:
            continue
        if rec.get('synthesized'):
            # A reconstructed aggregate is provenance, not an independent
            # training draw; pairing it across scales would fabricate a seed.
            continue
        _value = L._record_primary_ppl(rec) if primary else rec.get('ppl')
        if not L.ppl_is_usable(_value):
            continue
        seed = L._as_seed_int(rec.get('seed'))
        if seed is None:
            continue
        if seed in out:
            # aggregate would reject this; keep the analysis explicit too.
            raise ValueError(f'{variant}: duplicate measurable seed {seed}')
        out[seed] = rec
    return out


def _ppl_by_seed(summary, variant):
    """Backward-compatible name: raw per-seed PPL for one arm.

    Cross-scale code must use `_paired_ppl_by_seed`; this helper is kept for
    callers that only need the arm's own values and is not a pairing gate.
    """
    return {s: float(rec['ppl']) for s, rec in _records_by_seed(summary, variant).items()}


def _paired_ppl_by_seed(summary, variant_a, variant_b):
    """Per-seed paired PPLs that pass the SAME control gate as the audit.

    Cross-scale meta-analysis is only defined on controlled within-scale
    contrasts.  Reading `ppl` by (variant, seed) alone can pair two runs with
    different optimizers, steps or data merely because the seed number
    matches.  `exp_lib.pair_reason` is the project-wide gate (run_cfg,
    train_knobs, steps/tokens); any seed it refuses is dropped and reported,
    not silently averaged into a scale point.
    """
    ra = _records_by_seed(summary, variant_a, primary=True)
    rb = _records_by_seed(summary, variant_b, primary=True)
    paired, gated = {}, []
    for seed in sorted(set(ra) & set(rb)):
        reason = L.pair_reason(ra[seed], rb[seed])
        if reason:
            gated.append((seed, reason))
            continue
        _a = L._record_primary_ppl(ra[seed])
        _b = L._record_primary_ppl(rb[seed])
        if not (L.ppl_is_usable(_a) and L.ppl_is_usable(_b)):
            gated.append((seed, 'primary estimand missing (nested record '
                                'without holdout PPL/NLL accounting)'))
            continue
        paired[seed] = (float(_a), float(_b))
    return paired, gated


def _panel_corpus_fp(summary, variant=None):
    """Budget-independent corpus/tokenizer identity of one (panel, arm).

    New records carry `corpus_fp`: the tokenizer-fit prefix plus source and
    vocabulary, deliberately excluding the training cap and served `train_fp`.
    A 5B-token tier and a 12B-token tier over the same corpus therefore attest
    the SAME corpus/tokenizer (the cross-scale contrast is defined) instead of
    being refused as "mixed".  Genuinely different corpora/tokenisations still
    produce different fingerprints.

    Legacy records without `corpus_fp` fall back to `data_version` (which is
    training-budget dependent), so their cross-scale binding stays
    conservative.  A panel that mixes provenance generations is refused rather
    than letting the newer subset speak for the whole panel.
    """
    stable, legacy = set(), set()
    missing_provenance = 0
    for rec in summary.values():
        if not isinstance(rec, dict) or rec.get('synthesized'):
            # The corpus identity must be attested by real measurements; a
            # reconstructed record's copied fingerprint cannot vouch for it.
            continue
        if variant is not None and rec.get('variant') != variant:
            continue
        _prov = rec.get('provenance')
        _pd = _prov.get('data') if isinstance(_prov, dict) else None
        prov_data = _pd if isinstance(_pd, dict) else {}
        fp = rec.get('corpus_fp')
        if fp is None:
            fp = prov_data.get('corpus_fp')
        if fp not in (None, ''):
            stable.add(str(fp))
            continue
        dv = rec.get('data_version')
        if dv is None:
            dv = prov_data.get('data_version')
        if dv not in (None, ''):
            legacy.add(str(dv))
        elif L.ppl_is_usable(rec.get('ppl')):
            missing_provenance += 1
    if stable and legacy:
        raise ValueError(
            f'{variant or "panel"}: records mix corpus_fp-provenance and legacy '
            'data_version records; the stable corpus identity cannot be attested '
            'for the legacy subset (re-run or migrate them)')
    if len(stable) > 1:
        raise ValueError(f'{variant or "panel"}: records carry {len(stable)} different corpus_fp fingerprints {sorted(stable)}')
    if len(legacy) > 1:
        raise ValueError(f'{variant or "panel"}: records carry {len(legacy)} different data_version fingerprints {sorted(legacy)}')
    if missing_provenance:
        # One fingerprinted record cannot attest the corpus identity of the
        # others; the point stays unattested rather than borrowing a subset's
        # fingerprint.
        print(f'[m0-analysis] {variant or "panel"}: {missing_provenance} measured '
              'record(s) carry neither corpus_fp nor data_version; the corpus '
              'identity is not attested')
        return None
    if stable:
        return sorted(stable)[0]
    return (sorted(legacy)[0] if legacy else None)


def _extreme_scale_labels(points):
    """Labels of the smallest and largest scale in a cross-scale point set.

    Choosing by label (`sorted(raw)[0], sorted(raw)[-1]`) breaks as soon as a
    ladder has more than nine tiers: lexicographic order puts `S10` before
    `S2`, so the "extreme" interaction would compare the wrong two scales.
    Order by the declared scale (parameters) instead.  Numeric NumPy scalars
    and integer JSON values are accepted; booleans/non-finite scales are not.
    """
    usable = []
    for p in (points or []):
        if p.get('label') is None:
            continue
        sc = L._finite_scalar(p.get('scale'))
        if sc is None:
            continue
        usable.append((sc, p))
    if len(usable) < 2:
        return None
    order = sorted(usable, key=lambda item: (item[0], str(item[1]['label'])))
    return str(order[0][1]['label']), str(order[-1][1]['label'])


def _flatten_contrasts(audit):
    rows = []
    for family in ('components', 'baselines'):
        for name, entries in (audit.get(family) or {}).items():
            for e in entries:
                rows.append({
                    'family': family, 'name': name,
                    'with': e.get('with'), 'without': e.get('without'),
                    'n_paired': e.get('n_paired'), 'mean': e.get('mean'),
                    'std': e.get('std'), 'relative_effect_pct': e.get('relative_effect_pct'),
                    'p_exact_signflip': e.get('p_exact_signflip'),
                    'p_holm': e.get('p_holm_global', e.get('p_holm_component')),
                    'effect_vs_noise': e.get('effect_vs_noise'),
                    'resolution_limited': e.get('resolution_limited'),
                    'underpowered': e.get('underpowered'),
                    'controlled': e.get('controlled'),
                    'run_cfg_equal': e.get('run_cfg_equal'),
                    'param_controlled': e.get('param_controlled'),
                })
    return rows


_RUN_CFG_ATTR = re.compile(r'^RUN_([A-Z][A-Z0-9_]*)$')

def _tier_run_cfgs(extra=None):
    """Discover the declared scale-tier run configs from exp_lib.

    The analysis used to hard-code S1/S2 in three places (default summary
    discovery, planned seed counts, and the budget ladder).  Adding S3/S4 then
    silently left the new tiers out of the planned-vs-observed check.  Discover
    every ``RUN_<TIER>`` dict that carries an ``outdir`` and ``seeds`` instead;
    ``extra`` can supply test or notebook tiers explicitly.
    """
    out = {}
    _ladder_names = set((getattr(L, 'SCALE_LADDER', {}) or {}).keys())
    for _name in dir(L):
        _m = _RUN_CFG_ATTR.match(_name)
        if not _m or _m.group(1) not in _ladder_names:
            # RUN_ABL / RUN_LONG / RUN_SCALE are earlier exploratory panels,
            # not B-tier scale-ladder tiers; they must not enter this ladder.
            continue
        _cfg = getattr(L, _name)
        if (isinstance(_cfg, dict) and _cfg.get('outdir')
                and _cfg.get('seeds') is not None):
            out[_m.group(1)] = dict(_cfg)
    if extra:
        out.update({str(k): dict(v) for k, v in dict(extra).items()})
    return out

def _declared_seed_list(cfg):
    """Unique declared seeds in numeric order (invalid ids are ignored here;
    the ladder audit reports them as problems)."""
    vals = []
    for _s in ((cfg or {}).get('seeds') or []):
        _n = L._as_seed_int(_s)
        if _n is not None and _n not in vals:
            vals.append(_n)
    return sorted(vals)

def _panel_result_diagnostics(summary):
    """Curve, NaN-triage and cost diagnostics for one scale panel.

    A scale run is not a single PPL number: the same final value can hide a
    diverged curve, a NaN-triaged run, or a tiny accuracy gain bought with a
    10x latency/parameter increase.  This block is descriptive and reuses the
    trainer's recorded fields; it never changes a PPL or a statistical test.
    """
    def _num(v):
        return L._finite_scalar(v)

    def _median(vals):
        return float(np.median(vals)) if vals else None

    counts = {}
    train_counts = {}
    nonfinite, overfitting, stalled, early, above_best, unmonitored = [], [], [], [], [], []
    train_bad = []
    costs = {}
    n_records = 0
    for key, rec in (summary or {}).items():
        if not isinstance(rec, dict) or rec.get('synthesized') or rec.get('error'):
            continue
        n_records += 1
        h = rec.get('curve_health') if isinstance(rec.get('curve_health'), dict) else {}
        status = str(h.get('status') or 'missing')
        counts[status] = int(counts.get(status, 0)) + 1
        th = rec.get('train_curve_health') if isinstance(rec.get('train_curve_health'), dict) else {}
        tstatus = str(th.get('status') or 'missing')
        train_counts[tstatus] = int(train_counts.get(tstatus, 0)) + 1
        if tstatus in ('nonfinite', 'diverging', 'unstable_spikes'):
            train_bad.append(str(key))
        if status == 'nonfinite':
            nonfinite.append(str(key))
        elif status == 'overfitting':
            overfitting.append(str(key))
        elif status == 'underfitting_or_stalled':
            stalled.append(str(key))
        elif status in ('unmonitored', 'insufficient_data'):
            unmonitored.append(str(key))
        _fmb = _num(h.get('final_minus_best'))
        if _fmb is not None and _fmb > 0.0:
            above_best.append(str(key))
        es = rec.get('early_stop')
        if isinstance(es, dict) and es.get('stopped'):
            early.append(str(key))
        v = rec.get('variant') or str(key).split('::')[0]
        c = costs.setdefault(str(v), {'ppl': [], 'params': [], 'train_tps': [],
                                      'train_time_s': [], 'peak_mem_mb': [],
                                      'prefill_ms': [], 'tokens': []})
        _ppl = L._record_primary_ppl(rec)
        if L.ppl_is_usable(_ppl):
            c['ppl'].append(float(_ppl))
        for _key, _dst in (('params', 'params'),
                           ('train_tokens_per_s', 'train_tps'),
                           ('train_time_s', 'train_time_s'),
                           ('tokens_seen', 'tokens')):
            _x = _num(rec.get(_key))
            if _x is not None and _x >= 0.0:
                c[_dst].append(_x)
        _pm = rec.get('peak_memory') if isinstance(rec.get('peak_memory'), dict) else {}
        _pmv = _num(_pm.get('max_memory_allocated_mb'))
        if _pmv is not None and _pmv >= 0.0:
            c['peak_mem_mb'].append(_pmv)
        eff = rec.get('efficiency') if isinstance(rec.get('efficiency'), dict) else {}
        if eff.get('status') == 'ok':
            _lat = _num(eff.get('prefill_ms_p50'))
            if _lat is not None and _lat > 0.0:
                c['prefill_ms'].append(_lat)
            _mem = _num(eff.get('peak_mem_allocated_mb'))
            if _mem is not None and _mem >= 0.0:
                c['peak_mem_mb'].append(_mem)
            _tps = _num(eff.get('tokens_per_s_prefill'))
            if _tps is not None and _tps > 0.0:
                c['eff_tokens_per_s'] = c.get('eff_tokens_per_s', []) + [_tps]
    arms = {}
    for v, c in costs.items():
        _ppl_mean = (sum(c['ppl']) / len(c['ppl'])) if c['ppl'] else None
        _ppl_std = (float(np.std(c['ppl'], ddof=1)) if len(c['ppl']) > 1 else None)
        _ppl_sem = (_ppl_std / math.sqrt(len(c['ppl']))
                    if (_ppl_std is not None and c['ppl']) else None)
        arms[v] = {
            'n': int(max(len(c['ppl']), len(c['params']), len(c['train_tps']))),
            'n_ppl': len(c['ppl']),
            'ppl_median': _median(c['ppl']),
            'ppl_mean': _ppl_mean,
            'ppl_std': _ppl_std,
            'ppl_sem': _ppl_sem,
            'params_median': _median(c['params']),
            'train_tokens_per_s_median': _median(c['train_tps']),
            'train_time_s_median': _median(c['train_time_s']),
            'peak_mem_mb_max': (max(c['peak_mem_mb']) if c['peak_mem_mb'] else None),
            'prefill_ms_median': _median(c['prefill_ms']),
            'prefill_tokens_per_s_median': _median(c.get('eff_tokens_per_s', [])),
            'tokens_seen_median': _median(c['tokens']),
            'flags': [],
        }
    best = None
    _rank = [(r['ppl_median'], v) for v, r in arms.items() if r['ppl_median'] is not None]
    if _rank:
        best = min(_rank)[1]
    if best is not None:
        ref = arms[best]
        for v, r in arms.items():
            if v == best or r['ppl_median'] is None:
                continue
            _d = (r['ppl_median'] - ref['ppl_median']) / ref['ppl_median'] * 100.0
            r['ppl_delta_pct_vs_best'] = _d
            r['latency_ratio_vs_best'] = (r['prefill_ms_median'] / ref['prefill_ms_median']
                                          if (r['prefill_ms_median'] and ref['prefill_ms_median']) else None)
            r['params_ratio_vs_best'] = (r['params_median'] / ref['params_median']
                                         if (r['params_median'] and ref['params_median']) else None)
            r['memory_ratio_vs_best'] = (r['peak_mem_mb_max'] / ref['peak_mem_mb_max']
                                         if (r['peak_mem_mb_max'] and ref['peak_mem_mb_max']) else None)
            if (r['ppl_mean'] is not None and ref['ppl_mean'] is not None
                    and r['ppl_sem'] is not None and ref['ppl_sem'] is not None):
                r['noise_2se'] = 2.0 * math.sqrt(r['ppl_sem'] ** 2 + ref['ppl_sem'] ** 2)
                r['distinguishable'] = (None if r['noise_2se'] == 0.0
                                        else abs(r['ppl_mean'] - ref['ppl_mean']) > r['noise_2se'])
            else:
                r['noise_2se'] = None
                r['distinguishable'] = None
            _cost_ratios = [x for x in (r['latency_ratio_vs_best'], r['params_ratio_vs_best'],
                                        r['memory_ratio_vs_best']) if x is not None]
            _cost_2x = any(x >= 2.0 for x in _cost_ratios)
            if abs(_d) < 0.5:
                r['flags'].append('ppl_gain_below_0.5pct')
            if r['distinguishable'] is False:
                r['flags'].append('ppl_gain_within_seed_noise')
            elif r['distinguishable'] is None:
                r['flags'].append('ppl_uncertainty_unverifiable')
            if _cost_2x and (abs(_d) < 0.5 or r['distinguishable'] is False):
                r['flags'].append('cost_increase_without_clear_gain')
            if r['latency_ratio_vs_best'] is not None and r['latency_ratio_vs_best'] >= 2.0:
                r['flags'].append('latency_2x_or_more')
            if r['params_ratio_vs_best'] is not None and r['params_ratio_vs_best'] >= 2.0:
                r['flags'].append('params_2x_or_more')
            if r['memory_ratio_vs_best'] is not None and r['memory_ratio_vs_best'] >= 2.0:
                r['flags'].append('memory_2x_or_more')
    return {'n_records': int(n_records), 'curve_health_counts': counts,
            'train_curve_health_counts': train_counts,
            'best_ppl_arm': best,
            'records': {
                'nonfinite': nonfinite[:50],
                'overfitting': overfitting[:50],
                'underfitting_or_stalled': stalled[:50],
                'unmonitored_or_insufficient': unmonitored[:50],
                'final_above_best': above_best[:50],
                'early_stopped': early[:50],
                'train_curve_bad': train_bad[:50],
            },
            'cost': arms,
            'note': ('curve health is diagnostic, not a model-selection rule; '
                     'cost ratios compare medians within this panel only and are '
                     'not a speed claim when device/shape provenance differs')}


def scale_analysis(summaries=None, outdir='analysis_m0'):
    """Zero-GPU descriptive analysis for the B-tier S1/S2 panels.

    The scale ladder uses 4 seeds (S1) and 3 seeds (S2), so the exact sign-flip
    floor is 0.125 / 0.25: NO comparison can reach alpha=0.05 regardless of
    effect size.  This entry point therefore reports paired effects, family
    correction and resolution limits explicitly, and never promotes a
    descriptive readout to a confirmatory claim.  When both panels are present
    it also reports the per-scale log-PPL effects and the S1->S2 interaction.
    """
    _tier_cfgs = _tier_run_cfgs()
    if summaries is None:
        summaries = {}
        for label, cfg in sorted(_tier_cfgs.items()):
            path = os.path.join(cfg['outdir'], 'summary.json')
            if os.path.exists(path):
                try:
                    summaries[label] = json.load(open(path, encoding='utf-8'))
                except Exception as e:
                    print(f'[m0-analysis] WARNING: {path} unreadable ({type(e).__name__}: {e}); panel skipped')
    summaries = {k: v for k, v in (summaries or {}).items() if isinstance(v, dict) and v}
    report = {'kind': 'm0_scale_analysis', 'panels': {}, 'cross_scale': {},
              'tier_plan': {k: {'outdir': v.get('outdir'),
                                'declared_seeds': _declared_seed_list(v)}
                            for k, v in sorted(_tier_cfgs.items())},
              'notes': [
                  'Every B-tier panel is DESCRIPTIVE unless its exact sign-flip floor reaches alpha: the floor is 2/2^n per tier and m*2/2^n for a Holm family of size m.',
                  'Cross-scale effects are paired log PPL ratios; they are scale-free, but a two-point ladder is not a scaling law.',
              ]}
    for label, summary in sorted(summaries.items()):
        try:
            agg = L.aggregate(summary)
        except ValueError as e:
            print(f'[m0-analysis] {label}: aggregate REFUSED the panel ({e})')
            report['panels'][label] = {'error': str(e)}
            continue
        n_seeds = max((e.get('n_seeds') or 0) for e in agg.values()) if agg else 0
        variants = sorted({e.get('variant') for e in agg.values() if e.get('variant')})
        audit = L.methodology_audit(summary, design=B_SCALE_DESIGN,
                                    baselines=L.BASELINE_DESIGN,
                                    verbose=False, write=False)
        seed_plan = L.methodology_seed_plan(n_seeds, variants=variants) if variants else {}
        contrasts = _flatten_contrasts(audit)
        measured = [c for c in contrasts if c.get('n_paired')]
        report['panels'][label] = {
            'n_seeds': int(n_seeds),
            'variants': variants,
            'arms': {v: {k: e.get(k) for k in
                         ('n_seeds', 'ppl_primary', 'ppl_pooled', 'ppl_mean',
                          'ppl_std', 'params', 'tokens_seen')}
                     for v, e in agg.items()},
            'contrasts': contrasts,
            'descriptive_only': bool(measured and all(
                c.get('resolution_limited') or c.get('underpowered') for c in measured)),
            'verified_components': audit.get('verified_components'),
            'resolution_limited_components': audit.get('resolution_limited_components'),
            'underpowered_components': audit.get('underpowered_components'),
            'incomplete_components': audit.get('incomplete_components'),
            'seed_plan': {k: seed_plan.get(k) for k in
                          ('n_seeds', 'declared_contrasts',
                           'min_seeds_for_family_significance',
                           'sufficient_for_family_rejection')},
            'warnings': audit.get('warnings'),
            'result_diagnostics': _panel_result_diagnostics(summary),
        }
        _diag = report['panels'][label]['result_diagnostics']
        print(f'[m0-analysis] {label}: {len(measured)} contrast(s), '
              f'n={n_seeds}, descriptive_only={report["panels"][label]["descriptive_only"]}')
        print(f'[m0-analysis] {label}: curve_health={_diag["curve_health_counts"]} '
              f'nonfinite={len(_diag["records"]["nonfinite"])} '
              f'final>best={len(_diag["records"]["final_above_best"])} '
              f'best_ppl_arm={_diag.get("best_ppl_arm")}')
        for _v, _c in sorted((_diag.get('cost') or {}).items()):
            if _c.get('flags'):
                print(f'[m0-analysis] {label}/{_v}: cost flags {_c["flags"]}')
        report['notes'].append(
            '%s: n=%d seed(s), descriptive_only=%s; the raw exact sign-flip '
            'floor and the declared-family floor are both reported in its '
            'seed_plan/contrast fields.'
            % (label, int(n_seeds),
               bool(report['panels'][label]['descriptive_only'])))
    # Cross-scale readout over the contrasts the panels share.
    _declared_runs = _tier_cfgs
    pairs = [('csa_dynamic', 'csa_fixed', 'dynamic_vs_fixed'),
             ('hybrid_dynamic', 'csa_dynamic', 'hybrid_vs_csa'),
             ('csa_dynamic', 'full_matched', 'dynamic_vs_dense')]
    for a, b, name in pairs:
        points = []
        raw = {}
        for label, summary in sorted(summaries.items()):
            try:
                paired, gated = _paired_ppl_by_seed(summary, a, b)
            except ValueError as e:
                print(f'[m0-analysis] {name}/{label}: {e}')
                continue
            if gated:
                print(f'[m0-analysis] {name}/{label}: {len(gated)} seed(s) refused by '
                      'the control gate (run_cfg/train_knobs/steps differ); '
                      'the scale point uses only controlled pairs')
            common = sorted(paired)
            if len(common) < 2:
                continue
            lr = L.paired_log_ratio([paired[s][0] for s in common],
                                    [paired[s][1] for s in common], seeds=common)
            raw[label] = lr
            try:
                _cfp = _panel_corpus_fp(summary, a)
            except ValueError as e:
                print(f'[m0-analysis] {name}/{label}: {e}')
                continue
            _e = (report['panels'].get(label) or {}).get('arms', {}).get(a) or {}
            # The planned seed count comes from the DECLARED run config for a
            # known tier, not from however many records happened to survive.
            # Using the observed max would silently certify a truncated panel
            # as complete (the exact attrition bug the scale gates exist for).
            _declared_seeds = _declared_seed_list(_declared_runs.get(label))
            _declared_n = len(_declared_seeds)
            _observed_n = (report['panels'].get(label) or {}).get('n_seeds')
            _panel_n = (_declared_n or _observed_n)
            _point = {'label': label, 'scale': _e.get('params'),
                      'effects': lr['effects'], 'seeds': lr['seeds'],
                      'n_planned': _panel_n, 'corpus_fp': _cfp}
            if _declared_seeds:
                # An exact plan lets the scale gates distinguish a missing seed
                # from an unexpected extra one; a bare count cannot.
                _point['planned_seeds'] = _declared_seeds
            points.append(_point)
        if len(points) < 2:
            if points:
                report['cross_scale'][name] = {'status': 'single_scale_only', 'per_scale': raw}
            continue
        meta = L.meta_analyze_effects(points, min_seeds=3)
        entry = {'per_scale': raw, 'meta': meta}
        if str(meta.get('status') or '').startswith('refused'):
            # A log-PPL interaction across two tokenisations/corpora is a
            # units error, exactly like the pooled effect the meta layer
            # already refused.
            entry['interaction'] = {'status': meta.get('status'),
                                    'note': meta.get('note')}
            report['cross_scale'][name] = entry
            print(f'[m0-analysis] cross-scale {name}: interaction refused ({meta.get("status")})')
            continue
        # Paired smallest->largest interaction on the seeds present in BOTH
        # panels.  Ordering is by scale, never by label string.
        _ext = _extreme_scale_labels(points)
        low_lbl, high_lbl = _ext if _ext is not None else (None, None)
        lo = dict(zip(raw[low_lbl]['seeds'], raw[low_lbl]['effects'])) if low_lbl in raw else {}
        hi = dict(zip(raw[high_lbl]['seeds'], raw[high_lbl]['effects'])) if high_lbl in raw else {}
        common = sorted(set(lo) & set(hi))
        if len(common) >= 2:
            entry['interaction'] = L.scale_interaction(
                [lo[s] for s in common], [hi[s] for s in common],
                low_label=low_lbl, high_label=high_lbl, seeds=common, min_n=3)
        else:
            entry['interaction'] = {'status': 'insufficient_common_seeds',
                                    'low': low_lbl, 'high': high_lbl,
                                    'n_common': len(common)}
        report['cross_scale'][name] = entry
        print(f'[m0-analysis] cross-scale {name}: {meta.get("status")} '
              f'(poolable={meta.get("poolable")}, points={meta.get("n_points")})')
    _budget_tiers = []
    for _label, _cfg in sorted(_tier_cfgs.items()):
        _pan = report['panels'].get(_label) or {}
        _arms = _pan.get('arms') or {}
        _pick = _arms.get('csa_dynamic') or (next(iter(_arms.values())) if _arms else {})
        _params = _pick.get('params')
        # Bind the tier to the corpus/tokenizer identity its records attest
        # (data_version), so the ladder audit can refuse a cross-tier
        # comparison that silently mixed tokenisations.
        try:
            _cfp = _panel_corpus_fp(summaries.get(_label) or {})
        except ValueError as e:
            print(f'[m0-analysis] {_label}: {e}')
            _cfp = None
        _budget_tiers.append({'name': _label, 'cfg': dict(_cfg, corpus_fp=_cfp),
                              'params': _params})
    report['scale_ladder_audit'] = L.scale_ladder_audit(_budget_tiers)
    for _w in report['scale_ladder_audit'].get('warnings') or []:
        print(f'[m0-analysis] WARNING: {_w}')
    for _p in report['scale_ladder_audit'].get('problems') or []:
        print(f'[m0-analysis] PROBLEM: {_p}')
    os.makedirs(outdir, exist_ok=True)
    L.stamped_analysis('m0_scale_analysis', report, outdir=outdir, name='scale_analysis.json')
    print(f'[m0-analysis] wrote {outdir}/scale_analysis.json')
    return report


def _scale_panel_names():
    """Every declared ``RUN_S<n>`` panel, discovered instead of hard-coded.

    Adding S3/S4 (or an S10) means adding the config in exp_lib; this driver
    then accepts it without another edit.  The previous S1/S2 tuple silently
    refused any later tier and made the scale-up path look complete when it
    was not.
    """
    names = []
    for _n in dir(L):
        _m = re.fullmatch(r'RUN_S(\d+)', _n)
        if _m and isinstance(getattr(L, _n, None), dict):
            names.append('S' + _m.group(1))
    return sorted(names, key=lambda s: int(s[1:]))


def _panel_chain_from_env():
    """Resolve the panel chain: M0_AUTO_CHAIN (comma list) > M0_PANEL > S1.

    `full` keeps its single-panel semantics unless M0_AUTO_CHAIN is set;
    `auto` uses the same resolver so both entry points agree on panel choice.
    Each panel keeps its own cap / state file; chaining only sequences them.
    """
    raw = (os.environ.get('M0_AUTO_CHAIN') or os.environ.get('M0_PANEL') or 'S1').strip()
    panels = [p.strip().upper() for p in raw.split(',') if p.strip()]
    _valid = _scale_panel_names()
    bad = [p for p in panels if p not in _valid]
    if bad or not panels:
        print(f"[m0] unknown panel(s) {bad or raw!r} in M0_AUTO_CHAIN/M0_PANEL={raw!r}; "
              f"declared scale panels are {_valid or '(none)'}.  Nothing was started.")
        raise SystemExit(2)
    if len(panels) > 1:
        print(f'[m0] panel chain: {" -> ".join(panels)} (sequential; each panel keeps its own cap/state file; no shutdown between panels)')
    return panels


def _run_panel(panel):
    """One scale panel: G0 gate -> overfit guard -> calibration -> training.

    No shutdown decision here: the caller (run_full / run_auto) owns the
    end-of-run shutdown so a multi-panel chain never powers off mid-way.
    Returns the final push status.
    """
    run_cfg = getattr(L, f'RUN_{panel}', None)
    if not isinstance(run_cfg, dict):
        print(f'[m0] panel {panel!r} has no RUN_{panel} config; declared panels: '
              f'{_scale_panel_names()}.  Nothing was started.')
        raise SystemExit(2)
    # Geometry is parameterised for future scale: a longer sequence or a card
    # with less free memory can be handled by explicit env overrides, and the
    # override is copied into the panel cfg (hence run_cfg/manifest).
    run_cfg = L.apply_scale_geometry_overrides(run_cfg)
    if run_cfg.get('geometry_overrides'):
        print(f'[m0] geometry overrides: {run_cfg["geometry_overrides"]}')
    _caps = {'S1': (getattr(L, 'M0_CAP', None), 'M0_CAP_HOURS'),
             'S2': (getattr(L, 'M2_CAP', None), 'M2_CAP_HOURS')}
    cap, cap_env = _caps.get(panel, (None, f'{panel}_CAP_HOURS'))
    if not isinstance(cap, dict):
        cap = getattr(L, f'{panel}_CAP', None)
        if not isinstance(cap, dict):
            cap = getattr(L, f'M{panel[1:]}_CAP', None)
    if not isinstance(cap, dict):
        # A future tier without a declared cap still gets its own accounting
        # file instead of silently sharing (and corrupting) S1's budget.
        cap = dict(L.M0_CAP, state_path=f'run_time_state_{panel.lower()}.json')
    # The compute plan makes G0 a hard precondition for the scale panels, so
    # check it before the pre-flight work starts.
    L.require_g0(f'M0/{panel} 主面板')
    _full_ratios = {}
    for _v in run_cfg['variants']:
        try:
            _full_ratios[_v] = L.variant_mlp_ratio(_v, int(run_cfg.get('vocab_size') or L.B_VOCAB), d=int(run_cfg['d']), n_layers=int(run_cfg['n_layers']), n_heads=int(run_cfg['n_heads']), d_head=int(run_cfg['d_head']), seq_len=int(run_cfg['seq_len']), matched=run_cfg.get('matched'), ref_variant=run_cfg.get('mlp_match_ref', 'csa_dynamic'))
        except Exception:
            _full_ratios[_v] = None
    run_overfit_check(variants=run_cfg['variants'],
                      arch={'d': run_cfg['d'], 'n_layers': run_cfg['n_layers'],
                            'n_heads': run_cfg['n_heads'], 'd_head': run_cfg['d_head']},
                      seq_len=L.probe_seq_len(run_cfg['seq_len']),
                      vocab=int(run_cfg.get('vocab_size') or L.B_VOCAB),
                      mlp_ratio=_full_ratios, cfg=run_cfg)
    cap = dict(cap)
    _env = os.environ.get(cap_env)
    if _env:
        print(f'[m0] {cap_env}={_env} 覆盖默认上限 {cap["total_hours"]} 小时')
        cap['total_hours'] = float(_env)
    guard = L.TimeGuard(cap)
    guard.report()
    _what = f'{panel} 主面板（B 档，{len(run_cfg["variants"])} 臂 × {len(run_cfg["seeds"])} seeds × {run_cfg["n_train_tokens"] / 1e9:g}B token）'
    _calibrate_panel(run_cfg, guard, panel)
    guard.report()
    push_ok = git_push(f'm0: {_what}启动（产出目录 {run_cfg["outdir"]}/）')
    L.run(run_cfg, guard=guard, label=f'M0/{_what}')
    push_ok &= git_push(f'm0: {_what}完成一轮（{run_cfg["outdir"]} 为本地实验产物，不入库；量值以 summary.json 现算为准）')
    guard.report()
    print(f'\n[m0] {panel} panel round DONE.')
    return push_ok


def run_full():
    panels = _panel_chain_from_env()
    push_ok = True
    for panel in panels:
        push_ok &= _run_panel(panel)
    print('\n[m0] DONE.')
    schedule_shutdown(90 if push_ok else 2400)
    if not push_ok:
        print('[m0] WARNING: final push failed — instance stays up 40 min for a retry, then shuts down. Results are safe on the data disk either way.')


def _g0_already_valid():
    """True when a G0 certificate binding THIS tree/code/kernel stamp exists.

    Mirrors the accept/reject logic of `L.require_g0` (load + mismatch)
    without its SystemExit, so `auto` can use it as a fast-path probe: a
    resume after an interrupted panel must not re-burn the kernel suite.
    The explicit debug override env is deliberately NOT consulted here —
    `auto` never treats an override as a green gate.
    """
    cert = L.load_g0_certificate(L.g0_certificate_path())
    return cert is not None and L._g0_mismatch(cert) is None


def run_auto():
    """Unattended entry: G0 kernel gate -> certificate re-check -> panel chain.

    Automates the old human decision point "once the G0 numbers are in,
    decide whether the B-tier panels may start":
      * `run_kernel()` is fail-closed — a numeric mismatch exits 1, and a
        benchmark/TOST below the gate writes NO certificate;
      * `L.require_g0('M0/auto')` then re-verifies the certificate binds this
        tree, so a missing/stale certificate stops the chain BEFORE any panel
        GPU time (the only bypass stays the explicit debug override, which
        `auto` never sets on its own);
      * a certificate already valid for this tree skips the (minutes-long)
        re-verification — a resume goes straight back to the panel;
      * after a green gate the panels default to kernel mode (CSA_KERNEL=1)
        because that is the path G0 certifies as fast enough for the ladder;
        an explicit CSA_KERNEL from the caller always wins.
    A gate failure starts no panel, pushes nothing, leaves a 40-minute
    diagnosis window, and exits non-zero.
    """
    if _g0_already_valid():
        print('[m0-auto] a G0 certificate valid for this tree already exists — skipping the kernel re-verification (fast path)')
    else:
        print('[m0-auto] gate 1/2: running the G0 kernel/eager equivalence + benchmark suite (fail-closed)')
        try:
            run_kernel()
            print('[m0-auto] gate 2/2: re-verifying the G0 certificate for this tree')
            L.require_g0('M0/auto')
        except SystemExit:
            print('[m0-auto] G0 gate FAILED — the B-tier panel(s) are NOT started. This was the automated decision '
                  'point: no certificate, no scale run. Instance stays up 40 min for diagnosis (cancel: pkill -f m0autoshutdown).')
            schedule_shutdown(2400)
            raise
    if 'CSA_KERNEL' not in os.environ:
        os.environ['CSA_KERNEL'] = '1'
        print('[m0-auto] CSA_KERNEL=1 (kernel mode — G0 certified this path); set CSA_KERNEL explicitly to override')
    panels = _panel_chain_from_env()
    push_ok = True
    for panel in panels:
        push_ok &= _run_panel(panel)
    print('\n[m0-auto] DONE.')
    schedule_shutdown(90 if push_ok else 2400)
    if not push_ok:
        print('[m0-auto] WARNING: final push failed — instance stays up 40 min for a retry, then shuts down. Results are safe on the data disk either way.')

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode in ('-h', '--help'):
        print(__doc__ or '')
        raise SystemExit(0)
    if mode not in ('params', 'data', 'smoke', 'kernel', 'full', 'auto', 'analysis'):
        print(f"[m0] unknown or missing mode {mode!r}; expected one of 'params' / 'data' / 'smoke' / 'kernel' / 'full' / 'auto' / 'analysis' (all phases are explicit — nothing starts by default).  Nothing was started.")
        raise SystemExit(2)
    try:
        _gpu = torch.cuda.get_device_name(0)
    except Exception:
        _gpu = 'cpu'
    print(f'[m0] mode = {mode}   repo = {REPO}   device = {L.DEVICE} ({_gpu})')
    try:
        if mode == 'params':
            run_params()
        elif mode == 'data':
            run_data()
        elif mode == 'smoke':
            run_smoke()
        elif mode == 'kernel':
            run_kernel()
        elif mode == 'full':
            run_full()
        elif mode == 'auto':
            run_auto()
        elif mode == 'analysis':
            scale_analysis()
    except Exception:
        traceback.print_exc()
        if mode in ('full', 'auto'):
            print('\n[m0] CRASHED — attempting a partial-results push, then the instance stays up 40 min for remote diagnosis before auto-shutdown (cancel with: pkill -f m0autoshutdown).')
            try:
                git_push('m0: PARTIAL — run crashed, see run_m0.log (resume by re-running: completed runs are skipped)')
            except Exception:
                traceback.print_exc()
            schedule_shutdown(2400)
        else:
            print(f'\n[m0] CRASHED in `{mode}` mode — the tree can hold half-written throwaway output, so NOTHING was committed or pushed and no shutdown was scheduled.  Repair the defect and re-run; `smoke` exists to catch exactly this.')
        sys.exit(1)
