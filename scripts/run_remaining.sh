#!/usr/bin/env bash
# Последовательный прогон оставшихся экспериментов с логами по одному файлу
# на эксперимент. Запускать из корня репозитория.
set -u
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
export PYTHONPATH=src
mkdir -p logs

run() {
  local name="$1"; shift
  echo "=== $name  $(date '+%H:%M:%S') ==="
  if "$PY" "$@" > "logs/$name.log" 2>&1; then
    echo "    готово  $(date '+%H:%M:%S')"
    tail -3 "logs/$name.log" | sed 's/^/    /'
  else
    echo "    ОШИБКА — см. logs/$name.log"
    tail -15 "logs/$name.log" | sed 's/^/    /'
  fi
}

run e01 experiments/e01_window_vs_decay.py
run e02 experiments/e02_timescales.py
run e05 experiments/e05_leak_audit.py
run e06 experiments/e06_scaling.py
run e07 experiments/e07_budget.py
run e04 experiments/e04_vs_agent.py
run e03_synthetic experiments/e03_benchmark.py --dataset synthetic \
    --n-negatives-eval 49 --prediction-gap-days 1 --n-trials 8

echo "=== графики и отчёт  $(date '+%H:%M:%S') ==="
"$PY" -m afe.plots
"$PY" scripts/fill_results.py
"$PY" scripts/build_pdf.py
echo "=== всё готово  $(date '+%H:%M:%S') ==="
