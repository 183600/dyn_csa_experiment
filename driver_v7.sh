#!/bin/bash
cd /root/autodl-tmp/dyn_csa_experiment || exit 1
export V7_NO_SHUTDOWN=1
export PYTHONUNBUFFERED=1
PY=/root/miniconda3/bin/python
all_ok=1
for phase in P0E P2S P1L P1T; do
  echo "=== [driver] $(date '+%F %T') phase $phase START ==="
  $PY v7_supp.py phase "$phase"
  rc=$?
  echo "=== [driver] $(date '+%F %T') phase $phase END rc=$rc ==="
  $PY mon3.py 2>/dev/null | tail -20 || true
  if [ $rc -ne 0 ]; then
    echo "=== [driver] phase $phase failed (rc=$rc) -> stopping before spending on later phases ==="
    all_ok=0
    break
  fi
done
if [ $all_ok -ne 1 ]; then
  echo "=== [driver] $(date '+%F %T') aborted: a phase failed; skipping analysis/report/push ==="
  exit 1
fi
echo "=== [driver] $(date '+%F %T') final analysis + report rebuild ==="
$PY v7_supp.py analysis || all_ok=0
$PY build_report.py || all_ok=0
if [ $all_ok -ne 1 ]; then
  echo "=== [driver] $(date '+%F %T') aborted: analysis or report failed; skipping commit/push and NOT printing ALL DONE ==="
  exit 1
fi
git add -A
git commit -m "v7: remaining phases complete (P0E/P2S/P1L/P1T) + rebuilt report + stats" || true
git push origin HEAD 2>&1 | tail -3
echo "=== [driver] ALL DONE $(date '+%F %T') ==="
