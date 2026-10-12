"""
Собрать JSON бенчмарка из лога прерванного прогона.

Бенчмарк пишет JSON только по завершении всех методов. Если прогон
остановлен раньше (здесь — ради экономии времени на машине, уходящей в
троттлинг), результаты уже посчитанных методов не теряются: они есть в
построчном выводе. Этот разбор восстанавливает из него то, что там есть.

ЧЕГО В ЛОГЕ НЕТ: Recall@10 и MRR — построчный вывод их не печатает, они
появляются только в итоговой таблице. В восстановленном JSON они
помечены отсутствующими, и в отчёт идёт прочерк, а не выдуманное число.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RE_MODEL = re.compile(r"^\[модель\]\s+(\S+):\s+([\d,]+)\s+признаков")
RE_FEAT = re.compile(r"^\[признаки\]\s+(\S+)")
RE_RES = re.compile(
    r"test AUC=([\d.]+)\s+\[([\d.]+),\s*([\d.]+)\]\s+"
    r"logloss=([\d.]+)\s+ndcg@10=([\d.]+)\s+\[([\d.]+),\s*([\d.]+)\]\s+\((\d+)\s*с\)")
RE_HEAD = re.compile(r"датасет:\s+(\S+),\s+событий:\s+([\d,]+)")

HUMAN_HOURS = {"seen": 0.1, "popularity": 0.2, "hand": 6.0,
               "hand_keys_decay": 0.1, "compiler_L0": 0.0,
               "compiler": 0.0, "compiler_selected": 0.0}


def parse(log_path: Path, dataset: str, cfg_extra: dict) -> dict:
    results, cur = [], None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = RE_MODEL.search(line)
        if m:
            cur = {"name": m.group(1), "n_features": int(m.group(2).replace(",", ""))}
            continue
        m = RE_RES.search(line)
        if m and cur is not None:
            cur["test"] = {
                "auc": float(m.group(1)), "auc_lo": float(m.group(2)),
                "auc_hi": float(m.group(3)), "logloss": float(m.group(4)),
                "ndcg@10": float(m.group(5)), "ndcg@10_lo": float(m.group(6)),
                "ndcg@10_hi": float(m.group(7)),
            }
            cur["train_seconds"] = float(m.group(8))
            cur["feature_seconds"] = float("nan")
            cur["val_logloss"] = float("nan")
            cur["best_iteration"] = 0
            cur["config"] = {}
            cur["human_hours"] = HUMAN_HOURS.get(cur["name"], float("nan"))
            results.append(cur)
            cur = None
    n_events = 0
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        h = RE_HEAD.search(line)
        if h:
            n_events = int(h.group(2).replace(",", ""))
            break
    return {"config": {"dataset": dataset, "n_events": n_events, **cfg_extra},
            "results": results, "partial": True,
            "note": ("собрано разбором лога прерванного прогона; "
                     "Recall@10 и MRR в построчном выводе отсутствуют")}


if __name__ == "__main__":
    log = ROOT / "logs" / "e03_retailrocket.log"
    out = ROOT / "reports" / "e03_benchmark_retailrocket.json"
    res = parse(log, "retailrocket",
                {"n_negatives": 23, "n_negatives_eval": 49, "n_trials": 8,
                 "neg_source": "mixed", "prediction_gap_days": 1.0, "seed": 0})
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"восстановлено методов: {len(res['results'])}")
    for r in res["results"]:
        t = r["test"]
        print(f"  {r['name']:<18} {r['n_features']:>5} призн.  "
              f"NDCG@10={t['ndcg@10']:.4f} [{t['ndcg@10_lo']:.4f}, {t['ndcg@10_hi']:.4f}]")
