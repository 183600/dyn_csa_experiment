#!/usr/bin/env python3
"""M0/S1 驱动 — B 档算力规划（已选定）的代码入口，见 docs/compute_plan.md §4–§6。

B 档落在两处代码支撑上（均在 exp_lib.py）：
  1. FineWeb-Edu 语料与 32k BPE 分词缓存（load_fineweb_edu：npy 缓存 +
     sha256 内容指纹 + 原子写，纪律与 load_wikitext 一致；离线机器可用
     FWEDU_LOCAL_JSONL 指向本地 jsonl/txt 语料）；
  2. S1–S4 规模阶梯与 S1 主面板配置（SCALE_LADDER / RUN_S1，全部为
     规划值，G0 出口的吞吐校准与 M1 配方冻结后滚动修正）。

子命令：
  params   零 GPU：按 SmallGPT 实例化统计 S1–S4 的 csa_dynamic 参数量
           （32k 词表；S4 按规划的解析外推口径），对照 §5 的规划值；
           只打印，不写任何文件。
  data     构建 / 校验 FineWeb-Edu 缓存（默认 cap = S1 的 5B token；
           M0_TOKENS 可覆盖，例如 M0_TOKENS=15000000000 先落全量缓存）。
  smoke    端到端自检：微型模型 @ seq 8192 走完 fineweb 数据路径、
           稠密 / 固定稀疏 / 动态稀疏三条注意力路径与 8k 长度的
           掩码 / 分段 / 索引几何（M0_SMOKE_* 可调规模）。
  full     S1 主面板（5 臂 × 4 seeds × 5B token，results_b_s1/）。
           CSA_PAR=auto 一卡一实验（par.py 零改动，§7 排班口径）。

运行时上限：M0_CAP（可用 M0_CAP_HOURS 覆盖），状态文件
run_time_state_m0.json；超限单元拒绝启动、中途到达上限则硬截断且不留
半条测量。注意（compute_plan §3 的前提判断，不是缺陷）：Triton 内核
（G0 判据）就绪之前，eager 路径的预计耗时会让 TimeGuard 直接拒绝启动
S1 单元——此时应先完成内核与 kernel/eager 数值对照。
"""
import gc, json, os, shutil, subprocess, sys, time, traceback
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

def run_params():
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
    print(f"{'档':<4}{'配置':<28}{'参数量':>16}{'规划值(§5)':>14}{'比值':>8}")
    for name in ('S1', 'S2', 'S3', 'S4'):
        sc = L.SCALE_LADDER[name]
        n, _pl = _ladder_params(name)
        cfg_txt = f"d={sc['d']}/{sc['n_layers']}L/{sc['n_heads']}H/dh={sc['d_head']}"
        print(f"{name:<4}{cfg_txt:<28}{n:>16,}{_PLAN_PARAMS[name] / 1e9:>12.2f}B{n / _PLAN_PARAMS[name]:>8.2f}")
    print('\n[params] 规划值来自 docs/compute_plan.md §5（约 0.14B / 0.26B / 0.62B / 1.3B）；差异超过两成即需要回改阶梯配置或修订规划。')

def run_data():
    tokens = int(os.environ.get('M0_TOKENS') or L.RUN_S1['n_train_tokens'])
    t0 = time.time()
    train_ids, val_batch, vocab, _tok, val_bnd = L.load_fineweb_edu(L.RUN_S1['seq_len'], tokens, vocab_size=L.B_VOCAB, val_seqs=L.RUN_S1.get('val_seqs', 512))
    dt = time.time() - t0
    print(f'\n[m0-data] cache READY: train tokens = {len(train_ids):,}  vocab = {vocab}  val sequences = {val_batch.shape[0]}  (built/verified in {dt / 60:.1f} min)')
    print(f'[m0-data] 供 full / smoke 直接复用；离线重建用 FWEDU_LOCAL_JSONL，换 cap 用 M0_TOKENS（新 cap = 新缓存文件，旧面板指纹不受影响）')

def run_smoke():
    seq = int(os.environ.get('M0_SMOKE_SEQ') or L.RUN_S1['seq_len'])
    steps = int(os.environ.get('M0_SMOKE_STEPS') or 12)
    smoke_cfg = dict(L.RUN_S1, d=64, n_layers=2, n_heads=4, d_head=16, batch_size=1, seq_len=seq, steps=steps, warmup=2, eval_every=max(steps // 2, 1), eval_subset=2, val_seqs=2, n_train_tokens=int(os.environ.get('M0_SMOKE_TOKENS') or 200000), fw_fit_docs=int(os.environ.get('M0_SMOKE_FIT_DOCS') or 400), seeds=[0], outdir='results_smoke_m0', variants=['full', 'csa_fixed', 'csa_dynamic'])
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
    print(f'\n[smoke] PASSED — fineweb 数据路径与 seq-{seq} 的稠密/固定/动态三条注意力路径端到端可用；smoke 输出已删除，未触碰任何实验目录。')

def run_full():
    cap = dict(L.M0_CAP)
    _env = os.environ.get('M0_CAP_HOURS')
    if _env:
        cap['total_hours'] = float(_env)
        print(f'[m0] M0_CAP_HOURS={_env} 覆盖默认上限 {L.M0_CAP["total_hours"]} 卡时')
    guard = L.TimeGuard(cap)
    guard.report()
    print('[m0] 提示（compute_plan §3）：内核（G0）就绪前 eager 预计耗时会被 TimeGuard 拒绝启动——先完成三算子 Triton 化与 kernel/eager 数值对照，再回到这里跑 S1 主面板。')
    push_ok = git_push('m0: S1 主面板启动（B 档，5 臂 × 4 seeds × 5B token）')
    L.run(L.RUN_S1, guard=guard, label='M0/S1 B 档主面板（5 臂 × 4 seeds × 5B token）')
    push_ok &= git_push('m0: S1 主面板完成一轮（results_b_s1 为本地实验产物，不入库；量值以 summary.json 现算为准）')
    guard.report()
    print('\n[m0] DONE.')
    schedule_shutdown(90 if push_ok else 2400)
    if not push_ok:
        print('[m0] WARNING: final push failed — instance stays up 40 min for a retry, then shuts down. Results are safe on the data disk either way.')

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode in ('-h', '--help'):
        print(__doc__ or '')
        raise SystemExit(0)
    if mode not in ('params', 'data', 'smoke', 'full'):
        print(f"[m0] unknown or missing mode {mode!r}; expected one of 'params' / 'data' / 'smoke' / 'full' (all phases are explicit — nothing starts by default).  Nothing was started.")
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
        elif mode == 'full':
            run_full()
    except Exception:
        traceback.print_exc()
        if mode == 'full':
            print('\n[m0] CRASHED — attempting a partial-results push, then the instance stays up 40 min for remote debugging before auto-shutdown (cancel with: pkill -f m0autoshutdown).')
            try:
                git_push('m0: PARTIAL — run crashed, see run_m0.log (resume by re-running: completed runs are skipped)')
            except Exception:
                traceback.print_exc()
            schedule_shutdown(2400)
        else:
            print(f'\n[m0] CRASHED in `{mode}` mode — the tree can hold half-written throwaway output, so NOTHING was committed or pushed and no shutdown was scheduled.  Fix the defect and re-run; `smoke` exists to catch exactly this.')
        sys.exit(1)
