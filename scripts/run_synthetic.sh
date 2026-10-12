#!/usr/bin/env bash
# Пересчёт экспериментов, зависящих от синтетического генератора.
#
# Нужен после правки генератора: бренд стал нести сигнал (и НЕ входит в
# ключи ручного baseline), ценовой сегмент намеренно оставлен шумом как
# отрицательный контроль. Эксперименты на реальных данных пересчитывать
# не нужно — они от генератора не зависят.
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
    printf '    ОШИБКА (см. logs/%s.log)\n' "$name"; tail -12 "logs/$name.log" | sed 's/^/    /'
  fi
}
run e02 experiments/e02_timescales.py
run e05 experiments/e05_leak_audit.py 400000
run e06 experiments/e06_scaling.py
run e07 experiments/e07_budget.py
run e04 experiments/e04_vs_agent.py
printf '=== сборка отчёта  %s ===\n' "$(date '+%H:%M:%S')"
"$PY" -m afe.plots; "$PY" scripts/render_console.py --all
"$PY" scripts/render_code.py; "$PY" scripts/fill_results.py; "$PY" scripts/build_pdf.py
printf '=== ВСЁ ГОТОВО  %s ===\n' "$(date '+%H:%M:%S')"
