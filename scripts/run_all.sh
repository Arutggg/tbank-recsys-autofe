#!/usr/bin/env bash
# Полный последовательный прогон на исправленном коде.
#
# ПОСЛЕДОВАТЕЛЬНО, а не параллельно: замерено, что два процесса по восемь
# потоков на восьмиядерной пассивно охлаждаемой машине загоняют её в
# троттлинг (kernel_task до 180% CPU) и дают вчетверо меньше полезной
# работы, чем те же задачи по очереди.
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
    printf '    ОШИБКА (см. logs/%s.log)\n' "$name"
    tail -12 "logs/$name.log" | sed 's/^/    /'
  fi
}

run tests          -m pytest tests/ -q
run e01            experiments/e01_window_vs_decay.py
run e08            experiments/e08_protocol_diagnostics.py
run e02            experiments/e02_timescales.py
run e05            experiments/e05_leak_audit.py 400000
run e06            experiments/e06_scaling.py
run e07            experiments/e07_budget.py
run e04            experiments/e04_vs_agent.py
run e03_synthetic  experiments/e03_benchmark.py --dataset synthetic \
                     --n-negatives-eval 49 --prediction-gap-days 1 --n-trials 8
run e03_retailrocket experiments/e03_benchmark.py --dataset retailrocket \
                     --n-positives 100000 --n-positives-train 100000 \
                     --n-negatives-eval 49 --prediction-gap-days 1 --n-trials 8

printf '=== сборка отчёта  %s ===\n' "$(date '+%H:%M:%S')"
"$PY" -m afe.plots
"$PY" scripts/render_console.py --all
"$PY" scripts/render_code.py
"$PY" scripts/fill_results.py
"$PY" scripts/build_pdf.py
printf '=== ВСЁ ГОТОВО  %s ===\n' "$(date '+%H:%M:%S')"
