#!/bin/bash
cd /root/autodl-tmp/dyn_csa_experiment || exit 1
export V7_NO_SHUTDOWN=1
export PYTHONUNBUFFERED=1
PY=/root/miniconda3/bin/python
START_TS=$(date +%s)
export V7_BUDGET_YUAN="${V7_BUDGET_YUAN:-107.0}"
export V7_PRICE_PER_HOUR="${V7_PRICE_PER_HOUR:-2.4}"
GIVE_UP_AFTER=$((36*3600))
SAW_DRIVER=0
while true; do
  if pgrep -f "driver_v7\.sh" > /dev/null; then
    SAW_DRIVER=1
    sleep 300
    continue
  fi
  [ "$SAW_DRIVER" -eq 1 ] && break
  if [ -f driver_v7.log ] && grep -q "\[driver\] ALL DONE" driver_v7.log; then
    break
  fi
  if [ $(( $(date +%s) - START_TS )) -gt $GIVE_UP_AFTER ]; then
    echo "=== [bonus] no driver process and no ALL-DONE log within ${GIVE_UP_AFTER}s -> giving up; shutting down to stop billing ==="
    shutdown
    exit 0
  fi
  sleep 300
done
echo "=== [bonus] $(date '+%F %T') main driver exited ==="
if [ ! -f driver_v7.log ]; then
  echo "=== [bonus] driver_v7.log absent -> cannot confirm a clean finish; no bonus, box stays up ==="
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
try:
    wallet_h = float(os.environ["V7_BUDGET_YUAN"]) * 0.93 / float(os.environ["V7_PRICE_PER_HOUR"])
except Exception as e:
    print(f"[bonus] wallet env unreadable ({e}) -> unexpected state, box stays up")
    sys.exit(3)
if booked > wallet_h - 2.0:
    print(f"[bonus] booked {booked:.1f}h leaves <2h headroom under the spendable wallet "
          f"{wallet_h:.1f}h -> skip")
    sys.exit(1)
print(f"[bonus] booked {booked:.1f}h, wallet {wallet_h:.1f}h -> go")
PYEOF
_rc=$?
if [ "$_rc" -ne 0 ] && [ "$_rc" -ne 1 ] && [ "$_rc" -ne 2 ]; then
  echo "=== [bonus] ledger check crashed unexpectedly (rc=$_rc); box stays up for inspection ==="
  exit 1
fi
commit_and_push() {
  git add -A
  if ! git commit -m "$1"; then
    if [ -n "$(git status --porcelain)" ]; then
      echo "=== [bonus] commit FAILED with staged changes; box stays up ==="
      return 1
    fi
  fi
  set -o pipefail
  _branch=$(git rev-parse --abbrev-ref HEAD)
  if [ "$_branch" = "HEAD" ]; then
    _up=$(git symbolic-ref -q refs/remotes/origin/HEAD 2>/dev/null)
    _dst="${_up#refs/remotes/origin/}"
    if [ -z "$_dst" ] || ! git rev-parse --verify -q "refs/remotes/origin/${_dst}" >/dev/null; then
      _dst=main
    fi
    _push_ref="HEAD:${_dst}"
  else
    _push_ref="HEAD"
  fi
  if ! git push origin "$_push_ref" 2>&1 | tail -2; then
    echo "=== [bonus] push FAILED — nothing reached the remote; box stays up ==="
    return 1
  fi
  return 0
}
if [ "$_rc" -eq 2 ]; then
  echo "=== [bonus] ledger unreadable -> no bonus, box stays up for inspection ==="
  exit 1
fi
if [ "$_rc" -ne 0 ]; then
  echo "=== [bonus] skipped (budget) ==="
  commit_and_push "v7: bonus skipped (wallet headroom too small)" || exit 1
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
if [ "$_bonus_rc" -ne 0 ]; then
  echo "=== [bonus] P0W seed2 run FAILED (rc=$_bonus_rc) -> skipping analysis/report/commit; box stays up for inspection ==="
  exit 1
fi
if ! $PY v7_supp.py analysis || ! $PY build_report.py; then
  echo "=== [bonus] analysis or report FAILED -> no commit, box stays up ==="
  exit 1
fi
commit_and_push "v7 bonus: P0W seed2 complete (warm grid n=3) + final report" || exit 1
echo "=== [bonus] ALL DONE $(date '+%F %T'); shutting down to stop billing ==="
shutdown
