#!/usr/bin/env python3
import json
import math
import os
import subprocess
import time
BASE = os.path.dirname(os.path.abspath(__file__))
PHASES = [('results_lm_v7_rope', 12, 'P0R  RoPE+QK-norm 4 variants x 3 seeds @1500'), ('results_lm_v7_warmup', 6, 'P0W  dense->sparse warmup csa_fixed x w{0,5k,10k} x 2 seeds'), ('results_niah', None, 'P1N  NIAH retrieval probe'), ('results_lm_v7_long40', 4, 'P0E  long horizon 40k: csa_fixed+full x 2 seeds'), ('results_lm_v7_seq2k', 10, 'P1T  seq2048 topk sweep 5 variants x 2 seeds + m=1'), ('results_lm_v5_scale', None, 'P2S  param-matched scale seeds 2,3')]
BUDGET_STATE = os.path.join(BASE, 'autodl_budget_state_v7.json')

def load(p):
    try:
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def proc_alive():
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                cmd = f.read().decode('utf-8', 'replace')
        except Exception:
            continue
        if 'v7_supp.py' in cmd and 'python' in cmd:
            return int(pid)
    return None

def gpu():
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw', '--format=csv,noheader'], stderr=subprocess.DEVNULL).decode().strip()
        return out
    except Exception:
        return 'n/a'

def main():
    print(f'time: {time.strftime('%Y-%m-%d %H:%M:%S')}   GPU: {gpu()}')
    pid = proc_alive()
    if pid:
        try:
            with open(f'/proc/{pid}/stat', encoding='utf-8') as f:
                fields = f.read().split()
            starttime = int(fields[21])
            hz = os.sysconf(os.sysconf_names['SC_CLK_TCK'])
            with open('/proc/uptime', encoding='utf-8') as f:
                uptime = float(f.read().split()[0])
            elapsed = uptime - starttime / hz
            print(f'driver: PID {pid} alive, running {elapsed / 60:.1f} min')
        except Exception:
            print(f'driver: PID {pid} alive')
    else:
        print('driver: NOT RUNNING')
    st = load(BUDGET_STATE)
    if st:
        hrs = st.get('booked_seconds', 0) / 3600.0
        print(f'budget: {hrs:.2f} h booked, {st.get('runs', '?')} runs logged')
    for d, exp, desc in PHASES:
        p = os.path.join(BASE, d)
        if not os.path.isdir(p):
            print(f'\n--- {desc}: (dir absent)')
            continue
        s = load(os.path.join(p, 'summary.json'))
        n = len(s) if isinstance(s, dict) else 0
        tag = f'{n}/{exp}' if exp else f'{n}'
        print(f'\n--- {desc}  [{tag}]')
        if not isinstance(s, dict):
            continue
        rows = []
        for k, r in s.items():
            if not isinstance(r, dict):
                continue
            rows.append((k, r.get('ppl'), r.get('warm_steps', r.get('warm')), r.get('ppl_at_switch'), (r.get('train_time_s') / 60.0) if isinstance(r.get('train_time_s'), (int, float)) else None))
        rows.sort(key=lambda x: x[0])
        for k, ppl, warm, sw, mins in rows:
            ppl_s = f'{ppl:8.2f}' if isinstance(ppl, (int, float)) and (not isinstance(ppl, bool)) and math.isfinite(ppl) else '     n/a'
            try:
                warm_s = f'{int(warm):5d}' if warm is not None else '    -'
            except (TypeError, ValueError):
                warm_s = '    -'
            sw_s = '      -' if not isinstance(sw, (int, float)) else f'{sw:7.2f}'
            m_s = '' if mins is None else f'{mins:8.1f}min'
            print(f'  {k:34s} ppl={ppl_s} warm={warm_s} switch={sw_s} {m_s}')
if __name__ == '__main__':
    main()
