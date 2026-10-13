#!/usr/bin/env python3
"""par.py — task-level parallel executor (one GPU card per independent run).

The experiment drivers (v6-v11) schedule their work as independent cells of
the form (variant, seed) — one full training/eval run each.  This module adds
an opt-in task-level parallel mode on top of them:

    CSA_PAR=auto python v11_supp.py full      # one worker per visible GPU
    CSA_PAR=2     python v11_supp.py phase P4F
    CSA_PAR_GPUS=0,2                           # restrict the GPU pool
    (unset CSA_PAR) python v11_supp.py full    # 100% serial, old behaviour

Design contracts (see docs/design_notes.md "Task-level parallelism"):
  * A cell runs in a CHILD PROCESS with CUDA_VISIBLE_DEVICES pinned to exactly
    one card, so `DEVICE` in exp_lib resolves to that card and model code is
    untouched.  On CPU-only boxes the slots are CPU workers (useful for tests).
  * Shared on-disk state is updated under an flock (summary.json via
    L.locked_json_update, TimeGuard via its own lock): parallel children can
    never lose each other's records.
  * The child pre-books its projected run time into the shared TimeGuard state
    before training and refunds the difference afterwards, so concurrently
    launching children still respect the run-time cap (the cap meters
    GPU-seconds, i.e. N parallel runs consume it N times faster in wall
    clock).
  * Aggregation, plotting, git push and shutdown live ONLY in the parent.
  * If a child dies (rc != 0) the parent raises, exactly like an exception in
    the serial loop; completed cells stay on disk and re-runs resume.
"""
import json
import os
import re
import signal
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
REPO = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(REPO, 'par_jobs')
_KILL = {'on': False}

_OFF = ('', '0', 'no', 'off', 'false', 'none', 'serial')


def _num_or_none(v):
    try:
        n = int(str(v).strip())
        return n if n > 0 else None
    except (TypeError, ValueError):
        return None


def _torch():
    try:
        import torch
        return torch
    except Exception:
        return None


def gpu_slots(want_gpus=None):
    """Parent-relative GPU indices that may be handed to children, one per slot.

    Returns (slots, on_cuda): slots is a list of CUDA_VISIBLE_DEVICES override
    values ('0', '1', ...); on_cuda is False when there is no CUDA at all, in
    which case each slot is a CPU worker (CUDA_VISIBLE_DEVICES='')."""
    torch = _torch()
    n = 0
    if torch is not None:
        try:
            n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            n = 0
    if n > 0:
        slots = [str(i) for i in range(n)]
    else:
        return [], False
    if want_gpus:
        keep = []
        for g in str(want_gpus).split(','):
            g = g.strip()
            if g and g in slots:
                keep.append(g)
            elif g:
                print(f'[par] WARNING: CSA_PAR_GPUS entry {g!r} is not a visible GPU here (visible: 0-{n - 1}) — ignored')
        slots = keep or slots
    return slots, True


def resolve_ctx(par=None, label=''):
    """Turn the `par` argument / CSA_PAR env into an execution context dict.

    None/False/'0'/'off' -> None (serial, the default and the old behaviour).
    'auto'/True -> one worker per GPU slot.  Integer N -> N workers.
    A dict is a driver-decorated context and passes through unchanged.
    The context is a plain dict so it can cross the job-file boundary."""
    if isinstance(par, dict):
        return par or None
    if par is None:
        par = os.environ.get('CSA_PAR')
    if par is None or par is False or str(par).strip().lower() in _OFF:
        return None
    par = str(par).strip().lower() if not isinstance(par, (int, bool)) else par
    if par is True:
        par = 'auto'
    want = None
    if isinstance(par, int):
        want = par
    elif par.isdigit():
        want = int(par)
    elif par != 'auto':
        print(f'[par] WARNING: CSA_PAR={par!r} not understood (use auto / N / off) — running serial')
        return None
    slots, on_cuda = gpu_slots(os.environ.get('CSA_PAR_GPUS'))
    if on_cuda:
        if not slots:
            return None
        workers = len(slots) if want is None else min(want, len(slots))
        if want is not None and want > len(slots):
            print(f'[par] NOTE: requested {want} workers but only {len(slots)} GPU slot(s) are usable — capping at {len(slots)}')
    else:
        if want is None:
            print('[par] NOTE: no CUDA device is visible — CSA_PAR on a CPU box runs CPU-worker children (mainly useful for testing)')
            want = 1
        workers = want
        slots = [''] * workers
    ctx = {'workers': workers, 'slots': slots, 'on_cuda': on_cuda, 'label': label}
    print(f"[par] task-level parallelism ON: {workers} worker(s), one independent run per slot ({'GPUs ' + ','.join(str(s) for s in slots[:workers]) if on_cuda else 'CPU workers'}) — serial behaviour is restored by unsetting CSA_PAR")
    return ctx


def ctx_active(ctx, n_cells=1):
    return bool(ctx) and ctx.get('workers', 0) > 1 and n_cells > 1


def _norm(obj):
    """JSON-safe normalisation for job payloads (set -> sorted list)."""
    if isinstance(obj, dict):
        return {str(k): _norm(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_norm(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((_norm(v) for v in obj), key=lambda x: json.dumps(x, sort_keys=True, default=str))
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _denorm(obj):
    if isinstance(obj, dict):
        return {str(k): _denorm(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_denorm(v) for v in obj]
    return obj


def _slug(s):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(s))[:80]


def _child_env(slot):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(slot) if str(slot) != '' else ''
    env['PYTHONUNBUFFERED'] = '1'
    return env


def run_jobs(jobs, ctx, phase_label='phase'):
    """Run `jobs` (list of dicts from _norm) on the ctx's worker slots.

    Each job becomes a child process: `python par.py __child__ <job> <result>`,
    stdout/stderr teed to its own log file under par_jobs/.  Returns the list
    of result dicts (one per job, in job order).  Raises RuntimeError when a
    child infrastructure-fails (rc != 0) — matching the serial semantics where
    an exception in the phase loop aborts the phase (completed cells stay)."""
    workers = max(1, min(int(ctx.get('workers', 1)), len(ctx['slots'])))
    slots = ctx['slots'][:workers]
    os.makedirs(JOBS_DIR, exist_ok=True)
    run_dir = os.path.join(JOBS_DIR, _slug(phase_label) + '_' + time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)
    print(f'[par] dispatching {len(jobs)} cell(s) to {workers} worker(s) — job dir: {os.path.relpath(run_dir, REPO)}')
    entries = []
    for i, job in enumerate(jobs):
        name = f"{i:03d}_{_slug(job.get('kind'))}_{_slug(job.get('v'))}_s{job.get('seed')}" + (f"_w{job['extra']['warm']}" if job.get('extra') and job.get('extra', {}).get('warm') is not None else '')
        d = os.path.join(run_dir, name)
        os.makedirs(d, exist_ok=True)
        jpath = os.path.join(d, 'job.json')
        with open(jpath, 'w', encoding='utf-8') as f:
            json.dump(_norm(job), f, indent=1)
        entries.append({'idx': i, 'job': job, 'dir': d, 'job_path': jpath, 'result_path': os.path.join(d, 'result.json'), 'log_path': os.path.join(d, 'log.txt'), 'name': name, 'proc': None, 'slot': None, 'done': False, 'result': None})
    pending = list(entries)
    running = []
    results = [None] * len(entries)
    t0 = time.time()
    try:
        while pending or running:
            while pending and len(running) < workers:
                _busy = {r['slot'] for r in running}
                _free = [s for s in slots if s not in _busy]
                if not _free:
                    break
                slot = _free[0]
                e = pending.pop(0)
                e['slot'] = slot
                log_f = open(e['log_path'], 'w', encoding='utf-8')
                e['log_f'] = log_f
                e['proc'] = subprocess.Popen([sys.executable, os.path.abspath(__file__), '__child__', e['job_path'], e['result_path']], cwd=REPO, env=_child_env(slot), stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True)
                where = f"GPU {slot}" if ctx.get('on_cuda') and str(slot) != '' else 'CPU'
                print(f"[par] start {e['name']} on {where}  (log: {os.path.relpath(e['log_path'], REPO)})", flush=True)
                running.append(e)
            time.sleep(2.0)
            still = []
            for e in running:
                rc = e['proc'].poll()
                if rc is None:
                    still.append(e)
                    continue
                e['done'] = True
                e['log_f'].close()
                e['result'] = _read_result(e['result_path'])
                took = time.time() - t0
                results[e['idx']] = e['result']
                status = (e['result'] or {}).get('status', f'rc={rc}')
                print(f"[par] done  {e['name']} rc={rc} status={status} ({took / 60:.1f} min since dispatch)", flush=True)
                if rc != 0:
                    _dump_log_tail(e, 40)
                    raise RuntimeError(f"[par] child {e['name']} exited rc={rc} (see {os.path.relpath(e['log_path'], REPO)}) — aborting the phase like a serial exception; completed cells are kept, re-run resumes")
            running = still
            if running and int(time.time() - t0) % 60 < 2:
                ndone = sum(1 for x in entries if x['done'])
                names = ', '.join(x['name'] for x in running)
                print(f'[par] {len(running)} running / {ndone} done / {len(pending)} pending — {names}', flush=True)
    except BaseException:
        _KILL['on'] = True
        for e in running:
            try:
                os.killpg(os.getpgid(e['proc'].pid), signal.SIGTERM)
            except Exception:
                try:
                    e['proc'].terminate()
                except Exception:
                    pass
        for e in running:
            try:
                e['proc'].wait(timeout=30)
            except Exception:
                try:
                    os.killpg(os.getpgid(e['proc'].pid), signal.SIGKILL)
                except Exception:
                    pass
            try:
                e['log_f'].close()
            except Exception:
                pass
        print('[par] children terminated — a cell killed mid-training leaves NO record (same semantics as a serial crash); re-run resumes')
        raise
    for e in entries:
        if e['result'] is None:
            e['result'] = _read_result(e['result_path'])
            results[e['idx']] = e['result']
    return results


def _read_result(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception as _e:
        return {'status': 'unreadable_result', 'error': f'{type(_e).__name__}: {_e}'}


def _dump_log_tail(e, n=40):
    try:
        with open(e['log_path'], encoding='utf-8', errors='replace') as f:
            tail = f.readlines()[-n:]
        print(f"[par] ---- tail of {os.path.relpath(e['log_path'], REPO)} ----")
        sys.stdout.writelines('    ' + ln for ln in tail)
        print('[par] ------------------------------------------')
    except Exception:
        pass


# ---------------------------------------------------------------- child side

def run_child(job_path, result_path):
    job = _denorm(json.load(open(job_path, encoding='utf-8')))
    result = {'status': 'error', 'kind': job.get('kind'), 'v': job.get('v'), 'seed': job.get('seed')}
    try:
        module = job.get('module') or 'exp_lib'
        mod = __import__(module)
        if job.get('ckpt_install') and hasattr(mod, 'V9'):
            mod.V9._install_ckpt_patch()
        if job.get('ckpt_flag') and hasattr(mod, 'V9'):
            mod.V9._V9_CKPT['on'] = True
        kind = job['kind']
        if kind == 'run':
            res = getattr(mod, 'L', mod).run_cell_job(job)
        elif kind == 'lenphase':
            res = __import__('v7_supp').run_lenphase_cell_job(job)
        elif kind == 'probe':
            res = __import__('v10_supp').run_probe_cell_job(job)
        elif kind == 'warm':
            res = __import__('v7_supp').run_warmup_cell_job(job)
        else:
            raise ValueError(f'unknown job kind {kind!r}')
        result.update(res or {})
        result.setdefault('status', 'ok')
    except SystemExit as e:
        result['status'] = f'exit_{e.code}'
        raise
    except BaseException:
        result['status'] = 'fatal'
        result['error'] = traceback.format_exc()
        traceback.print_exc()
        raise
    finally:
        try:
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(_norm(result), f, indent=1)
        except Exception:
            pass


def main(argv):
    if len(argv) > 3 and argv[1] == '__child__':
        run_child(argv[2], argv[3])
        return 0
    print(__doc__)
    print('[par] this module is imported by the drivers; set CSA_PAR=auto|N to enable task-level parallel runs (one GPU per run).')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
