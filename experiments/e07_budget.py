"""
Эксперимент 7. Отбор под бюджет: что отправлять в прод.

Сгенерировать две тысячи признаков легко. Инженерный вопрос другой: какие
из них держать, если на состояние счётчиков есть ограниченная память, а на
инференс — ограниченное время. Без ответа автоматическая генерация
остаётся демонстрацией.

Постановка — ГРУППОВОЙ рюкзак, а не обычный. Стоимость не аддитивна:
сотня признаков на ключе user x brand делит одно состояние размером
(число ключей) x (типов событий) x (ядер) x 3 числа. Заплатив за первый
признак на ключе, остальные получаешь почти даром. Поэтому решение
принимается на уровне КЛЮЧЕЙ: «заводить ли таблицу счётчиков user x brand»
— именно так этот вопрос звучит в проде.

Процедура:
  1. посчитать полный перебор, обучить модель, взять важность по gain;
  2. упорядочить ключи жадно по отношению «полезность группы / память группы»
     (производный признак засчитывается только когда доступны ВСЕ его
     зависимости — иначе стоимость уровня 1 недооценивается);
  3. для каждого префикса набора ключей переобучить и ИЗМЕРИТЬ метрику.

Важность используется только для порядка рассмотрения. Отчётная метрика в
каждой точке измеряется реальным обучением, а не предсказывается.
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
from afe.engine import FeatureCompiler, _factorize_key                # noqa: E402
from afe.eval import SplitConfig, TaskConfig, all_metrics             # noqa: E402
from afe.hand import HandFeatures, _build_csr                         # noqa: E402
from afe.log import build_stream                                      # noqa: E402
from afe.model import GBDTConfig, fit_gbdt                            # noqa: E402
from afe.schema import EnumerationConfig, Schema, enumerate_features  # noqa: E402
from afe.select import (features_for_keys, gain_importance,           # noqa: E402
                        greedy_key_order, key_costs)
from afe.splits import build_splits                                   # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports"


def run(n_events=600_000, neg_source="mixed", seed=0, verbose=True) -> dict:
    cfg = SyntheticConfig(n_events=n_events, n_users=20_000, n_items=5_000, seed=seed)
    log, _ = generate(cfg)
    splits = build_splits(
        log, SplitConfig(),
        TaskConfig(target_etype=2, n_negatives=23, neg_source=neg_source, seed=seed),
        max_positives_per_window=3_000, max_positives_train=10_000)
    schema = Schema(user_cols=["user"],
                    item_cols=["item", "cat", "brand", "price_bucket"],
                    etype_names=log.etype_names)
    fs = enumerate_features(schema, EnumerationConfig())
    names = fs.names(log.etype_names)
    stream = build_stream(log, splits.queries)
    X = FeatureCompiler(fs, n_etypes=log.n_etypes).transform(stream, verbose=verbose)

    # фактические мощности ключей — от них зависит реальная стоимость
    cards = {}
    for key in {p.key for p in fs.primitives}:
        _, n_keys = _factorize_key(stream, key)
        cards[key] = n_keys
    costs = key_costs(fs, cards, n_etypes=log.n_etypes)

    base = GBDTConfig(seed=seed)
    b_all = fit_gbdt(splits.rows(X, "train"), splits.y("train"),
                     splits.rows(X, "val"), splits.y("val"), base, names)
    imp = gain_importance(b_all, fs.n_total)

    order = greedy_key_order(fs, imp, costs)
    if verbose:
        print("\n  порядок добавления ключей (полезность / память):")
        for i, k in enumerate(order):
            c = costs[k]
            print(f"    {i + 1:>2}. {c.name:<22} {c.n_keys:>9,} ключей  "
                  f"{c.state_bytes / 2**20:>8.1f} МиБ")

    pareto = []
    for i in range(1, len(order) + 1):
        chosen = set(order[:i])
        mask = features_for_keys(fs, chosen)
        if mask.sum() == 0:
            continue
        nm = [names[j] for j in np.flatnonzero(mask)]
        b = fit_gbdt(splits.rows(X, "train")[:, mask], splits.y("train"),
                     splits.rows(X, "val")[:, mask], splits.y("val"), base, nm)
        p = b.predict(splits.rows(X, "test")[:, mask], num_iteration=b.best_iteration)
        m = all_metrics(splits.y("test"), p, splits.group("test"), proba=p)
        state = sum(costs[k].state_bytes for k in chosen)
        upd = sum(costs[k].updates_per_event for k in chosen)
        pareto.append({
            "n_keys_selected": i,
            "keys": [costs[k].name for k in order[:i]],
            "added": costs[order[i - 1]].name,
            "label": costs[order[i - 1]].name,
            "state_mib": state / 2**20,
            "updates_per_event": upd,
            "n_features": int(mask.sum()),
            "metrics": m,
        })
        if verbose:
            print(f"  +{costs[order[i - 1]].name:<22} "
                  f"{state / 2**20:>8.1f} МиБ  {int(mask.sum()):>5} призн.  "
                  f"AUC={m['auc']:.4f}  NDCG@10={m['ndcg@10']:.4f}", flush=True)

    # --- ориентир: ручной набор, его память и его метрика ---
    hf = HandFeatures(cat_col="cat", n_etypes=log.n_etypes, with_nunique=True)
    t0 = time.perf_counter()
    Xh = hf.transform(stream, verbose=False)
    hand_keys = [("user",), ("item",), ("user", "item"), ("user", "cat")]
    hand_bytes = 0
    for key in hand_keys:
        kid, nk = _factorize_key(stream, key)
        off, idx = _build_csr(kid, stream.is_query, nk, stream.size)
        hand_bytes += off.nbytes + idx.nbytes
    bh = fit_gbdt(splits.rows(Xh, "train"), splits.y("train"),
                  splits.rows(Xh, "val"), splits.y("val"), base, hf.feature_names_)
    ph = bh.predict(splits.rows(Xh, "test"), num_iteration=bh.best_iteration)
    hand_ref = {
        "state_mib": hand_bytes / 2**20,
        "n_features": Xh.shape[1],
        "metrics": all_metrics(splits.y("test"), ph, splits.group("test"), proba=ph),
        "seconds": time.perf_counter() - t0,
    }
    return {"config": {"n_events": n_events, "neg_source": neg_source, "seed": seed},
            "pareto": pareto, "hand_reference": hand_ref}


def report(res: dict) -> None:
    W = 96
    print("\n" + "=" * W)
    print("ФРОНТ ПАРЕТО: качество против бюджета памяти состояния")
    print("=" * W)
    print(f"{'добавлен ключ':<24}{'память':>12}{'обновл/соб':>12}"
          f"{'признаков':>11}{'AUC':>9}{'NDCG@10':>10}")
    print("-" * W)
    for p in res["pareto"]:
        print(f"{p['added']:<24}{p['state_mib']:>9.1f} МиБ{p['updates_per_event']:>12}"
              f"{p['n_features']:>11}{p['metrics']['auc']:>9.4f}"
              f"{p['metrics']['ndcg@10']:>10.4f}")
    h = res["hand_reference"]
    print("-" * W)
    print(f"{'ручной набор (окна)':<24}{h['state_mib']:>9.1f} МиБ{'':>12}"
          f"{h['n_features']:>11}{h['metrics']['auc']:>9.4f}"
          f"{h['metrics']['ndcg@10']:>10.4f}")
    print("-" * W)

    # при каком бюджете компилятор догоняет ручной набор
    target = h["metrics"]["ndcg@10"]
    hit = next((p for p in res["pareto"] if p["metrics"]["ndcg@10"] >= target), None)
    if hit:
        print(f"  компилятор достигает качества ручного набора на {hit['state_mib']:.1f} МиБ "
              f"({hit['n_keys_selected']} ключей), ручной набор требует "
              f"{h['state_mib']:.1f} МиБ")
        print(f"  отношение памяти: {h['state_mib'] / max(hit['state_mib'], 1e-9):.1f}x")
    else:
        print(f"  компилятор не достигает NDCG@10={target:.4f} ни на одном бюджете")
    print("=" * W)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-events", type=int, default=600_000)
    ap.add_argument("--neg-source", default="mixed")
    a = ap.parse_args()
    res = run(n_events=a.n_events, neg_source=a.neg_source)
    report(res)
    OUT.mkdir(exist_ok=True)
    (OUT / "e07_budget.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("результаты: reports/e07_budget.json")
