#!/usr/bin/env bash
# Добивка: e06 (исправленная оценка памяти) и e07 (важность по ключам),
# затем полная сборка отчёта.
set -u
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
export PYTHONPATH=src
mkdir -p logs
run() {
  local name="$1"; shift
  printf '=== %-16s %s ===\n' "$name" "$(date '+%H:%M:%S')"
  local t0=$SECONDS
  if "$PY" "$@" > "logs/$name.log" 2>&1; then
    printf '    готово за %d мин %02d с\n' $(((SECONDS-t0)/60)) $(((SECONDS-t0)%60))
  else
    printf '    ОШИБКА\n'; tail -12 "logs/$name.log" | sed 's/^/    /'
  fi
}
run e06 experiments/e06_scaling.py
run e07 experiments/e07_budget.py
printf '=== сборка отчёта  %s ===\n' "$(date '+%H:%M:%S')"
"$PY" -m afe.plots
"$PY" scripts/render_console.py --all
"$PY" scripts/render_code.py
"$PY" scripts/parse_bench_log.py
"$PY" scripts/fill_results.py
"$PY" scripts/build_pdf.py
printf '=== ВСЁ ГОТОВО  %s ===\n' "$(date '+%H:%M:%S')"
