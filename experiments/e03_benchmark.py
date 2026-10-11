"""
Эксперимент 3. Сквозной бенчмарк: компилятор против затюненного ручного бустинга.

Состав сравнения подобран так, чтобы каждая строка таблицы отвечала на
отдельный вопрос, а не просто давала ещё одно число.

  popularity        тривиальный ориентир (популярность айтема по train).
                    Нужен, чтобы видеть, сколько вообще даёт задача.

  hand              добросовестный ручной набор: прямые и косвенные
                    признаки, жёсткие окна 1/7/30/365 дней, конверсии,
                    доли, уникальные партнёры. Бустинг затюнен.

  hand_keys_decay   ТЕ ЖЕ КЛЮЧИ, что у hand, но вместо жёстких окон —
                    полная сетка масштабов и моменты возрастов. Ключевая
                    абляция: изолирует эффект СЕМЕЙСТВА ВРЕМЕННЫХ ЯДЕР,
                    не смешивая его с эффектом более богатого набора
                    мета-путей.

  compiler_L0       полный перебор, только уровень 0 (то, что требует
                    проход по логу).

  compiler          полный перебор, уровень 0 + уровень 1 (конверсии,
                    доли, контрасты масштабов, PMI-lift). Уровень 1
                    стоит ноль проходов по логу.

Всем наборам выдаётся одинаковый бюджет тюнинга и один и тот же набор
кандидатов. Отбор модели — по val log-loss (см. model.py о том, почему
не по AUC).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afe.datasets.synthetic import SyntheticConfig, generate          # noqa: E402
from afe.dsl import FeatureSet                                        # noqa: E402
from afe.engine import FeatureCompiler                                # noqa: E402
from afe.eval import SplitConfig, TaskConfig, bootstrap_ci            # noqa: E402
from afe.hand import HandFeatures                                     # noqa: E402
from afe.log import build_stream                                      # noqa: E402
from afe.model import (evaluate_feature_set,                           # noqa: E402
                       evaluate_with_selection)
from afe.schema import EnumerationConfig, Schema, enumerate_features  # noqa: E402
from afe.splits import build_splits                                   # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports"

#: Оценка ручного труда на метод, в часах. Грубая, но честная: ручной
#: набор писался и отлаживался, компилятор запускается одной командой.
#: Колонка нужна, потому что задание про автоматизацию, а не только про метрику.
HUMAN_HOURS = {
    "popularity": 0.2,
    "hand": 6.0,
    "hand_keys_decay": 0.1,
    "compiler_L0": 0.0,
    "compiler": 0.0,
    "compiler_selected": 0.0,
}


def _build(name, log, splits, schema, stream, verbose):
    """Построить ОДИН набор признаков. Матрицы строятся по одной и
    освобождаются сразу после оценки: держать hand + hand_keys_decay +
    compiler одновременно это ~2800 столбцов на 480 тысяч строк, то есть
    5.4 ГиБ только под признаки."""
    t0 = time.perf_counter()

    if name == "popularity":
        train_mask = log.t < splits.t_train_end
        pop = np.bincount(log.item[train_mask], minlength=log.n_items).astype(np.float32)
        X = np.log1p(pop[splits.queries.item]).reshape(-1, 1)
        return X, ["log_pop_train"], time.perf_counter() - t0

    if name == "hand":
        hf = HandFeatures(cat_col="cat", n_etypes=log.n_etypes, with_nunique=True)
        X = hf.transform(stream, verbose=verbose)
        return X, hf.feature_names_, time.perf_counter() - t0

    if name == "hand_keys_decay":
        keys = [("user",), ("item",), ("user", "item"), ("user", "cat")]
        fs = enumerate_features(schema, EnumerationConfig(), keys=keys)
        X = FeatureCompiler(fs, n_etypes=log.n_etypes).transform(stream, verbose=verbose)
        return X, fs.names(log.etype_names), time.perf_counter() - t0

    if name in ("compiler_L0", "compiler", "compiler_selected"):
        cfg = EnumerationConfig()
        if name == "compiler_L0":
            cfg = EnumerationConfig(use_conversion=False, use_share=False,
                                    use_trend=False, use_lift=False)
        fs = enumerate_features(schema, cfg)
        X = FeatureCompiler(fs, n_etypes=log.n_etypes).transform(stream, verbose=verbose)
        return X, fs.names(log.etype_names), time.perf_counter() - t0

    raise ValueError(name)


METHODS = ["popularity", "hand", "hand_keys_decay", "compiler_L0",
           "compiler", "compiler_selected"]


def run(n_events=1_000_000, n_positives=4_000, n_positives_train=14_000,
        n_negatives=23, n_trials=12, seed=0, verbose=True) -> dict:
    cfg = SyntheticConfig(n_events=n_events, n_users=20_000, n_items=5_000, seed=seed)
    log, truth = generate(cfg)
    if verbose:
        print(log.describe())

    task = TaskConfig(target_etype=2, n_negatives=n_negatives, seed=seed)
    splits = build_splits(log, SplitConfig(), task,
                          max_positives_per_window=n_positives,
                          max_positives_train=n_positives_train)
    if verbose:
        print("\nвременные сплиты:")
        print(splits.describe())

    schema = Schema(user_cols=["user"],
                    item_cols=["item", "cat", "brand", "price_bucket"],
                    etype_names=log.etype_names)

    stream = build_stream(log, splits.queries)
    results = []
    for name in METHODS:
        if verbose:
            print(f"\n[признаки] {name}", flush=True)
        X, names, fsec = _build(name, log, splits, schema, stream, verbose)
        if verbose:
            print(f"[модель] {name}: {X.shape[1]} признаков", flush=True)
        fn = (evaluate_with_selection if name == "compiler_selected"
              else evaluate_feature_set)
        r = fn(
            name,
            splits.rows(X, "train"), splits.y("train"),
            splits.rows(X, "val"), splits.y("val"),
            splits.rows(X, "test"), splits.y("test"), splits.group("test"),
            n_trials=n_trials, seed=seed, feature_names=names,
            feature_seconds=fsec, human_hours=HUMAN_HOURS.get(name, float("nan")),
            verbose=verbose,
        )
        results.append(r)
        del X
        if verbose:
            m = r.test_metrics
            print(f"    test AUC={m['auc']:.4f}  logloss={m['logloss']:.5f}  "
                  f"ndcg@10={m['ndcg@10']:.4f}  recall@10={m['recall@10']:.4f}  "
                  f"({r.train_seconds:.0f} с)", flush=True)

    return {
        "config": {"n_events": n_events, "n_positives": n_positives,
                   "n_positives_train": n_positives_train,
                   "n_negatives": n_negatives, "n_trials": n_trials, "seed": seed},
        "meta": {},
        "results": [
            {"name": r.name, "n_features": r.n_features,
             "test": r.test_metrics, "val_logloss": r.val_logloss,
             "feature_seconds": r.feature_seconds, "train_seconds": r.train_seconds,
             "best_iteration": r.best_iteration, "human_hours": r.human_hours,
             "config": r.config}
            for r in results
        ],
    }


def report(res: dict) -> None:
    W = 104
    print("\n" + "=" * W)
    print("БЕНЧМАРК: компилятор против затюненного ручного бустинга")
    print("=" * W)
    hdr = (f"{'метод':<18}{'призн.':>8}{'AUC':>9}{'logloss':>10}{'NDCG@10':>10}"
           f"{'Recall@10':>11}{'MRR':>8}{'фичи,с':>9}{'тюнинг,с':>10}{'чел.-ч':>8}")
    print(hdr)
    print("-" * W)
    base = next((r for r in res["results"] if r["name"] == "hand"), None)
    for r in res["results"]:
        m = r["test"]
        print(f"{r['name']:<18}{r['n_features']:>8,}{m['auc']:>9.4f}"
              f"{m['logloss']:>10.5f}{m['ndcg@10']:>10.4f}{m['recall@10']:>11.4f}"
              f"{m['mrr']:>8.4f}{r['feature_seconds']:>9.1f}"
              f"{r['train_seconds']:>10.0f}{r['human_hours']:>8.1f}")
    if base:
        print("-" * W)
        print(f"{'прирост к hand:':<18}")
        for r in res["results"]:
            if r["name"] == "hand":
                continue
            m, b = r["test"], base["test"]
            print(f"  {r['name']:<16}{'':>8}{m['auc'] - b['auc']:>+9.4f}"
                  f"{m['logloss'] - b['logloss']:>+10.5f}"
                  f"{m['ndcg@10'] - b['ndcg@10']:>+10.4f}"
                  f"{m['recall@10'] - b['recall@10']:>+11.4f}"
                  f"{m['mrr'] - b['mrr']:>+8.4f}")
    print("=" * W)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-events", type=int, default=1_000_000)
    ap.add_argument("--n-positives", type=int, default=4_000)
    ap.add_argument("--n-positives-train", type=int, default=14_000)
    ap.add_argument("--n-negatives", type=int, default=23)
    ap.add_argument("--n-trials", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    res = run(n_events=a.n_events, n_positives=a.n_positives,
              n_positives_train=a.n_positives_train,
              n_negatives=a.n_negatives, n_trials=a.n_trials, seed=a.seed)
    report(res)
    OUT.mkdir(exist_ok=True)
    (OUT / "e03_benchmark.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))
    print("результаты: reports/e03_benchmark.json")
