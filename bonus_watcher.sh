#!/bin/bash
cd /root/autodl-tmp/dyn_csa_experiment || exit 1
export V7_NO_SHUTDOWN=1
export PYTHONUNBUFFERED=1
PY=/root/miniconda3/bin/python
START_TS=$(date +%s)
export V7_BUDGET_YUAN="${V7_BUDGET_YUAN:-107.0}"
export V7_PRICE_PER_HOUR="${V7_PRICE_PER_HOUR:-2.4}"
while pgrep -f "bash driver_v7.sh" > /dev/null; do sleep 300; done
echo "=== [bonus] $(date '+%F %T') main driver exited ==="
if [ ! -f driver_v7.log ]; then
  echo "=== [bonus] driver_v7.log absent -> cannot confirm a clean finish; no bonus ==="
  exit 0
fi
if [ "$(stat -c %Y driver_v7.log)" -lt "$START_TS" ]; then
  echo "=== [bonus] driver_v7.log predates this watcher -> stale log, no bonus ==="
  exit 0
fi
if ! grep -q "\[driver\] ALL DONE" driver_v7.log; then
  echo "=== [bonus] main driver did not finish cleanly -> no bonus, box stays up ==="
  exit 0
fi
BUDGET_STATE=autodl_budget_state_v7.json
if [ ! -f "$BUDGET_STATE" ]; then
  echo "=== [bonus] $BUDGET_STATE absent -> budget unknown; no bonus, box stays up ==="
  exit 0
fi
$PY - <<'PYEOF'
import json, os, sys
try:
    st = json.load(open("autodl_budget_state_v7.json"))
except Exception as e:                       # unreadable / corrupt ledger
    print(f"[bonus] ledger unreadable ({e}) -> no bonus")
    sys.exit(2)
if not isinstance(st, dict) or "booked_seconds" not in st:
    print("[bonus] ledger has no booked_seconds -> no bonus")
    sys.exit(2)
booked = st.get("booked_seconds", 0) / 3600.0
wallet_h = float(os.environ["V7_BUDGET_YUAN"]) / float(os.environ["V7_PRICE_PER_HOUR"])
if booked > wallet_h - 2.0:
    print(f"[bonus] booked {booked:.1f}h leaves <2h headroom under wallet "
          f"{wallet_h:.1f}h -> skip")
    sys.exit(1)
print(f"[bonus] booked {booked:.1f}h, wallet {wallet_h:.1f}h -> go")
PYEOF
_rc=$?
if [ $_rc -eq 2 ]; then
  echo "=== [bonus] skipped (ledger unreadable) ==="
  git add -A; git commit -m "v7: bonus skipped (budget ledger unreadable)" || true
  git push origin HEAD 2>&1 | tail -2
  shutdown
  exit 0
fi
if [ $_rc -ne 0 ]; then
  echo "=== [bonus] skipped (budget) ==="
  git add -A; git commit -m "v7: bonus skipped (wallet headroom too small)" || true
  git push origin HEAD 2>&1 | tail -2
  shutdown
  exit 0
fi
echo "=== [bonus] $(date '+%F %T') P0W seed2 (warm grid n=3) START ==="
$PY - <<'PYEOF'
import v7_supp as V
import exp_lib as L
V.install_patches()
V.run_warmup(dict(L.RUN_LONG, outdir="results_lm_v7_warmup",
                  variants=["csa_fixed"]),
             seeds=[2], guard=L.CostGuard(V.BUDGET_V7),
             label="v7 bonus P0W-seed2",
             warm_grid=(0, 5000, 10000))
PYEOF
_bonus_rc=$?
echo "=== [bonus] $(date '+%F %T') P0W seed2 END rc=$_bonus_rc ==="
$PY v7_supp.py analysis
$PY build_report.py
git add -A
git commit -m "v7 bonus: P0W seed2 complete (warm grid n=3) + final report" || true
git push origin HEAD 2>&1 | tail -2
echo "=== [bonus] ALL DONE $(date '+%F %T'); shutting down to stop billing ==="
shutdown
