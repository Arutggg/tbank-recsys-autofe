"""
Слияние результатов частичного прогона бенчмарка в полную таблицу.

Нужен, потому что прогон с --methods считает подмножество методов и пишет
в отдельный файл. Сливать надо с проверкой: таблицы сравнимы только если у
них совпадает конфигурация (датасет, протокол негативов, разрыв
предсказания, число негативов, сид). Иначе в одной таблице окажутся числа,
полученные на разных постановках, и сравнение будет бессмысленным.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPORTS = Path(__file__).resolve().parents[1] / "reports"

#: Поля конфигурации, которые обязаны совпадать, чтобы строки были сравнимы.
MUST_MATCH = ("dataset", "neg_source", "prediction_gap_days",
              "n_negatives", "n_negatives_eval", "n_positives_train", "seed")


def merge(base_path: Path, add_path: Path, out_path: Path | None = None) -> dict:
    base = json.loads(base_path.read_text())
    add = json.loads(add_path.read_text())

    bc, ac = base.get("config", {}), add.get("config", {})
    mismatch = [k for k in MUST_MATCH if bc.get(k) != ac.get(k)]
    if mismatch:
        raise SystemExit(
            "конфигурации не совпадают по полям: " + ", ".join(mismatch) +
            "\nсливать нельзя: строки получены на разных постановках\n" +
            "\n".join(f"  {k}: {bc.get(k)!r} против {ac.get(k)!r}" for k in mismatch))

    by_name = {r["name"]: r for r in base["results"]}
    added, replaced = [], []
    for r in add["results"]:
        (replaced if r["name"] in by_name else added).append(r["name"])
        by_name[r["name"]] = r

    # порядок как в METHODS, чтобы таблица читалась сверху вниз по сложности
    order = ["seen", "popularity", "hand", "hand_keys_decay", "compiler_L0",
             "compiler", "compiler_selected", "compiler_covisit"]
    merged = sorted(by_name.values(),
                    key=lambda r: order.index(r["name"])
                    if r["name"] in order else 99)

    out = {"config": bc, "meta": base.get("meta", {}), "results": merged}
    out_path = out_path or base_path
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"  добавлено: {', '.join(added) or '—'}")
    print(f"  заменено:  {', '.join(replaced) or '—'}")
    print(f"  всего методов: {len(merged)} -> {out_path}")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("использование: merge_bench.py <базовый.json> <добавляемый.json>")
    merge(Path(sys.argv[1]), Path(sys.argv[2]))
