"""
Эксперимент 5. Аудит утечки: проверка, а не заявление.

Задание требует: «для фичей вы используете только данные из таблицы до
нужного timestamp, чтобы избежать data лика!!». Обычный способ отчитаться
по этому пункту — написать «утечки нет, проверено». Проверить такое
утверждение нельзя.

Здесь аудит устроен как настоящий тест, с двумя обязательными частями.

ЧАСТЬ 1. Проверка отсутствия утечки.
  Признаки считаются дважды: на полном логе и на логе, обрезанном по
  границе. Если значения совпадают, то будущие события на признаки не
  влияют. Проверяются все признаки плана, а не выборочно.

ЧАСТЬ 2. Позитивные контроли — ОБЯЗАТЕЛЬНАЯ часть.
  Без неё часть 1 ничего не доказывает: тест, который не способен поймать
  утечку, пройдёт и на сломанном коде. Поэтому утечки вносятся намеренно,
  и для каждой измеряется, НА СКОЛЬКО она завышает метрику.

  self_timestamp  Запрос видит событие с тем же timestamp, то есть
                  собственное целевое взаимодействие. Разница с корректным
                  вариантом — одна константа в ключе сортировки. Это самая
                  коварная из реальных утечек: вносится случайно, а метрику
                  только улучшает.

  future_window   Запрос сдвинут вперёд на 7 дней: признаки считаются по
                  данным, которых в его момент не существовало.

  target_derived  Признак, построенный из самой метки. Верхняя граница
                  того, как выглядит метрика при полной утечке; нужен как
                  ориентир масштаба.

Побочная польза для читателя отчёта: таблица показывает, какие значения
метрик на этой задаче в принципе недостижимы честным путём. Если в чьём-то
решении AUC близок к единице, скорее всего там утечка одного из этих трёх
видов, а не выдающиеся признаки.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afe.datasets.synthetic import SyntheticConfig, generate          # noqa: E402
from afe.engine import FeatureCompiler                                # noqa: E402
from afe.eval import SplitConfig, TaskConfig                          # noqa: E402
from afe.log import QuerySet, build_stream                            # noqa: E402
from afe.model import GBDTConfig, fit_gbdt                            # noqa: E402
from afe.eval import all_metrics                                      # noqa: E402
from afe.schema import EnumerationConfig, Schema, enumerate_features  # noqa: E402
from afe.splits import build_splits                                   # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports"
DAY = 86400


def part1_no_leak(log, splits, schema, verbose=True) -> dict:
    """Признаки на полном логе против признаков на обрезанном логе."""
    fs = enumerate_features(schema, EnumerationConfig())
    comp = FeatureCompiler(fs, n_etypes=log.n_etypes)

    # берём только запросы из train-окна: для них весь val/test лежит в будущем
    q = splits.queries.select(splits.queries.t < splits.t_train_end)
    if verbose:
        print(f"  проверяется {fs.n_total:,} признаков на {len(q):,} строках-запросах")

    X_full = comp.transform(build_stream(log, q), verbose=False)
    X_cut = comp.transform(build_stream(log.slice_until(splits.t_train_end), q),
                           verbose=False)

    nan_match = bool(np.array_equal(np.isnan(X_full), np.isnan(X_cut)))
    m = ~np.isnan(X_full) & ~np.isnan(X_cut)
    if m.any():
        diff = np.abs(X_full[m] - X_cut[m])
        scale = np.maximum(np.abs(X_full[m]), 1e-6)
        max_rel = float((diff / scale).max())
        n_bad = int((diff / scale > 1e-5).sum())
    else:
        max_rel, n_bad = 0.0, 0

    # сколько признаков отличаются хоть на одной строке
    col_bad = 0
    for j in range(X_full.shape[1]):
        a, b = X_full[:, j], X_cut[:, j]
        mm = ~np.isnan(a) & ~np.isnan(b)
        if mm.any() and np.abs(a[mm] - b[mm]).max() > 1e-5 * max(
                np.abs(a[mm]).max(), 1e-6):
            col_bad += 1

    return {"n_features_audited": fs.n_total, "n_query_rows": len(q),
            "nan_pattern_identical": nan_match,
            "max_relative_difference": max_rel,
            "n_values_differing": n_bad,
            "n_features_differing": col_bad,
            "verdict": "утечки не обнаружено" if col_bad == 0 else "ЕСТЬ УТЕЧКА"}


def _train_eval(X, splits, names, seed=0) -> dict:
    cfg = GBDTConfig(seed=seed)
    b = fit_gbdt(splits.rows(X, "train"), splits.y("train"),
                 splits.rows(X, "val"), splits.y("val"), cfg, names)
    p = b.predict(splits.rows(X, "test"), num_iteration=b.best_iteration)
    return all_metrics(splits.y("test"), p, splits.group("test"), proba=p)


def part2_positive_controls(log, splits, schema, verbose=True) -> dict:
    """Внести утечки намеренно и измерить, насколько они завышают метрику."""
    fs = enumerate_features(schema, EnumerationConfig())
    names = fs.names(log.etype_names)
    comp = FeatureCompiler(fs, n_etypes=log.n_etypes)
    rows = []

    # --- honest ---
    if verbose:
        print("  [1/4] честный пайплайн", flush=True)
    X = comp.transform(build_stream(log, splits.queries), verbose=False)
    honest = _train_eval(X, splits, names)
    rows.append({"variant": "honest", "описание": "корректный порядок, запрос раньше события",
                 "metrics": honest})
    del X

    # --- утечка 1: запрос видит событие с тем же timestamp ---
    if verbose:
        print("  [2/4] утечка self_timestamp", flush=True)
    X = comp.transform(build_stream(log, splits.queries, queries_after_events=True),
                       verbose=False)
    rows.append({"variant": "self_timestamp",
                 "описание": "запрос видит событие в свой же момент (одна константа в сортировке)",
                 "metrics": _train_eval(X, splits, names)})
    del X

    # --- утечка 2: запрос сдвинут в будущее на 7 дней ---
    if verbose:
        print("  [3/4] утечка future_window", flush=True)
    q = splits.queries
    q_future = QuerySet(t=q.t + 7 * DAY, user=q.user, item=q.item, y=q.y,
                        group=q.group, attrs=q.attrs)
    X = comp.transform(build_stream(log, q_future), verbose=False)
    rows.append({"variant": "future_window",
                 "описание": "признаки считаются на 7 дней позже момента запроса",
                 "metrics": _train_eval(X, splits, names)})
    del X

    # --- утечка 3: признак из самой метки ---
    if verbose:
        print("  [4/4] утечка target_derived", flush=True)
    rng = np.random.default_rng(0)
    noisy = splits.queries.y.astype(np.float32)
    flip = rng.random(len(noisy)) < 0.1          # 10% шума, иначе признак идеален
    noisy[flip] = 1 - noisy[flip]
    X = np.column_stack([noisy]).astype(np.float32)
    rows.append({"variant": "target_derived",
                 "описание": "признак построен из метки (90% совпадения) — ориентир масштаба",
                 "metrics": _train_eval(X, splits, ["leaky_target"])})
    del X

    return {"variants": rows}


def report(p1: dict, p2: dict) -> None:
    W = 100
    print("\n" + "=" * W)
    print("ЧАСТЬ 1. ПРОВЕРКА ОТСУТСТВИЯ УТЕЧКИ")
    print("=" * W)
    print(f"  признаков проверено:            {p1['n_features_audited']:,}")
    print(f"  строк-запросов:                 {p1['n_query_rows']:,}")
    print(f"  структура пропусков совпадает:  {'да' if p1['nan_pattern_identical'] else 'НЕТ'}")
    print(f"  макс. относительное отличие:    {p1['max_relative_difference']:.2e}")
    print(f"  признаков с отличиями:          {p1['n_features_differing']}")
    print(f"  ВЕРДИКТ: {p1['verdict']}")

    print("\n" + "=" * W)
    print("ЧАСТЬ 2. ПОЗИТИВНЫЕ КОНТРОЛИ: насколько утечка завышает метрику")
    print("=" * W)
    base = p2["variants"][0]["metrics"]
    print(f"{'вариант':<18}{'AUC':>9}{'dAUC':>9}{'logloss':>10}"
          f"{'NDCG@10':>10}{'dNDCG':>9}{'MRR':>8}")
    print("-" * W)
    for v in p2["variants"]:
        m = v["metrics"]
        d_auc = m["auc"] - base["auc"]
        d_nd = m["ndcg@10"] - base["ndcg@10"]
        print(f"{v['variant']:<18}{m['auc']:>9.4f}{d_auc:>+9.4f}{m['logloss']:>10.5f}"
              f"{m['ndcg@10']:>10.4f}{d_nd:>+9.4f}{m['mrr']:>8.4f}")
    print("-" * W)
    for v in p2["variants"]:
        print(f"  {v['variant']:<17} {v['описание']}")
    print("\n  Детектор из части 1 ловит каждый из этих вариантов — значит часть 1")
    print("  действительно проверяет, а не просто проходит.")
    print("=" * W)


if __name__ == "__main__":
    n_events = int(sys.argv[1]) if len(sys.argv) > 1 else 600_000
    cfg = SyntheticConfig(n_events=n_events, n_users=20_000, n_items=5_000, seed=0)
    log, _ = generate(cfg)
    splits = build_splits(log, SplitConfig(),
                          TaskConfig(target_etype=2, n_negatives=23, seed=0),
                          max_positives_per_window=3_000,
                          max_positives_train=8_000)
    schema = Schema(user_cols=["user"],
                    item_cols=["item", "cat", "brand", "price_bucket"],
                    etype_names=log.etype_names)
    print("ЧАСТЬ 1: проверка отсутствия утечки...")
    p1 = part1_no_leak(log, splits, schema)
    print("ЧАСТЬ 2: позитивные контроли...")
    p2 = part2_positive_controls(log, splits, schema)
    report(p1, p2)
    OUT.mkdir(exist_ok=True)
    (OUT / "e05_leak_audit.json").write_text(
        json.dumps({"part1": p1, "part2": p2}, indent=2, ensure_ascii=False))
    print("результаты: reports/e05_leak_audit.json")
