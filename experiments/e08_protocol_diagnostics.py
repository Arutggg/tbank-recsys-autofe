"""
Эксперимент 8. Диагностика протокола: три замера, определившие постановку.

Этот эксперимент не про метод, а про ТО, КАК ВООБЩЕ МЕРИТЬ. Три замера
ниже заставили переделать протокол оценки, и без них остальные числа
читать нельзя.

A. ЦЕНА ТИХОЙ УТЕЧКИ В ДЖОЙНЕ АТРИБУТОВ.
   Свойства айтема в RetailRocket временные: категория и цена меняются.
   Джойн «айтем -> последнее известное значение» подставляет событию из
   мая категорию, присвоенную в сентябре. Эта утечка НЕ ловится обычной
   проверкой агрегатов, потому что сидит не в них, а в джойне.

B. ВЫРОЖДЕННОСТЬ ПРОТОКОЛА НЕГАТИВОВ.
   Если негативы сэмплировать по популярности, пользователь их почти
   никогда не видел, и задача решается одним бинарным признаком «видел ли
   юзер этот айтем». Тогда все разумные наборы признаков упираются в один
   потолок и сравнивать нечего.

C. ВНУТРИСЕССИОННАЯ ТРИВИАЛЬНОСТЬ ЦЕЛИ.
   Перед покупкой пользователь почти всегда только что смотрел этот же
   товар. Без разрыва предсказания «предсказание покупки» означает
   «юзер минуту назад открыл карточку товара» — это детекция намерения,
   а не рекомендация.

Все три замера выполняются на реальных данных и печатают числа, на
которые ссылается отчёт.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afe.datasets.retailrocket import RetailRocketConfig, load  # noqa: E402
from afe.dsl import ANY_ETYPE, FeatureSet, Primitive  # noqa: E402
from afe.engine import FeatureCompiler  # noqa: E402
from afe.eval import SplitConfig, TaskConfig, auc, build_queries, ranking_metrics  # noqa: E402
from afe.log import build_stream  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports"


def part_a_asof_vs_leaky() -> dict:
    """Насколько протекающий джойн «знает» больше корректного."""
    rows = []
    for asof in (True, False):
        lg = load(RetailRocketConfig(use_asof_properties=asof), verbose=False)
        rows.append({
            "mode": "as-of (корректно)" if asof else "последнее значение (УТЕЧКА)",
            "cat_coverage": float((lg.attrs["cat"] >= 0).mean()),
            "price_coverage": float((lg.attrs["price_bucket"] >= 0).mean()),
        })
    gap_cat = rows[1]["cat_coverage"] - rows[0]["cat_coverage"]
    return {"rows": rows, "extra_knowledge_cat": gap_cat}


def part_b_negative_protocol(log) -> dict:
    """Что даёт ОДИН бинарный признак при разных источниках негативов."""
    t1, t2 = SplitConfig().boundaries(log)
    prims = [Primitive(("user", "item"), ANY_ETYPE, "dcount", float("inf"))]
    rows = []
    for src in ("popularity", "mixed", "user_history"):
        q = build_queries(
            log, t2, int(log.t[-1]) + 1,
            TaskConfig(target_etype=2, n_negatives=49, max_positives=4_000,
                       neg_source=src, seed=0),
            train_t_end=t1)
        X = np.nan_to_num(
            FeatureCompiler(FeatureSet(prims, []), n_etypes=log.n_etypes)
            .transform(build_stream(log, q), verbose=False), nan=0.0)
        seen = (X[:, 0] > 0).astype(np.float64)
        m = ranking_metrics(q.y, seen, q.group)
        rows.append({
            "neg_source": src,
            "auc_seen": auc(q.y, seen),
            "ndcg10_seen": m["ndcg@10"],
            "share_negatives_seen": float((X[q.y == 0, 0] > 0).mean()),
        })
    return {"rows": rows}


def part_c_session_triviality(log) -> dict:
    """Разрыв между покупкой и предыдущим касанием ТОГО ЖЕ айтема."""
    order = np.lexsort((log.t, log.item, log.user))
    u, i, t, e = log.user[order], log.item[order], log.t[order], log.etype[order]
    change = np.empty(len(u), bool)
    change[0] = True
    change[1:] = (u[1:] != u[:-1]) | (i[1:] != i[:-1])
    prev_t = np.where(change, -1, np.roll(t, 1))
    buy = e == 2
    has_prev = prev_t[buy] >= 0
    g = (t[buy] - prev_t[buy]).astype(float)[has_prev]

    buckets = [("< 1 минуты", 60), ("< 5 минут", 300), ("< 1 часа", 3600),
               ("< 1 дня", 86400), ("< 7 дней", 7 * 86400)]
    dist = [{"label": lbl, "share_of_all_purchases": float((g < s).sum() / buy.sum())}
            for lbl, s in buckets]

    # эффект разрыва предсказания на тривиальный признак
    t1, t2 = SplitConfig().boundaries(log)
    prims = [Primitive(("user", "item"), ANY_ETYPE, "dcount", float("inf"))]
    gaps = []
    for gap_d in (0, 1, 7):
        q = build_queries(
            log, t2, int(log.t[-1]) + 1,
            TaskConfig(target_etype=2, n_negatives=49, max_positives=3_000,
                       neg_source="mixed", prediction_gap_s=int(gap_d * 86400),
                       seed=0),
            train_t_end=t1)
        X = np.nan_to_num(
            FeatureCompiler(FeatureSet(prims, []), n_etypes=log.n_etypes)
            .transform(build_stream(log, q), verbose=False), nan=0.0)
        seen = (X[:, 0] > 0).astype(np.float64)
        gaps.append({
            "gap_days": gap_d,
            "auc_seen": auc(q.y, seen),
            "ndcg10_seen": ranking_metrics(q.y, seen, q.group)["ndcg@10"],
            "positives_with_history": float((X[q.y == 1, 0] > 0).mean()),
        })
    return {
        "n_purchases": int(buy.sum()),
        "share_with_prior_touch": float(has_prev.mean()),
        "median_gap_minutes": float(np.median(g) / 60),
        "distribution": dist,
        "gap_effect": gaps,
    }


def report(a, b, c) -> None:
    W = 84
    print("=" * W)
    print("A. ЦЕНА ТИХОЙ УТЕЧКИ: as-of джойн против «последнего значения»")
    print("=" * W)
    print(f"{'режим джойна':<34}{'категория':>14}{'цена':>12}")
    for r in a["rows"]:
        print(f"{r['mode']:<34}{r['cat_coverage']:>13.1%}{r['price_coverage']:>12.1%}")
    print(f"\n  Протекающий джойн «знает» категорию на "
          f"{a['extra_knowledge_cat']:.1%} событий больше.")
    print("  Это чистая информация из будущего, и обычный детектор утечки её")
    print("  не видит: она в джойне атрибутов, а не в агрегатах.")

    print("\n" + "=" * W)
    print("B. ВЫРОЖДЕННОСТЬ ПРОТОКОЛА: что даёт ОДИН признак «видел ли айтем»")
    print("=" * W)
    print(f"{'негативы':<18}{'AUC':>10}{'NDCG@10':>11}{'негативов знакомо юзеру':>26}")
    for r in b["rows"]:
        print(f"{r['neg_source']:<18}{r['auc_seen']:>10.4f}{r['ndcg10_seen']:>11.4f}"
              f"{r['share_negatives_seen']:>25.1%}")
    pop = b["rows"][0]
    print("\n  При негативах по популярности один бинарный признак даёт")
    print(f"  NDCG@10 = {pop['ndcg10_seen']:.4f}. Любое число рядом с этим на")
    print("  таком протоколе не говорит о качестве модели ничего.")

    print("\n" + "=" * W)
    print("C. ВНУТРИСЕССИОННАЯ ТРИВИАЛЬНОСТЬ: покупка и предыдущее касание")
    print("=" * W)
    print(f"  покупок: {c['n_purchases']:,}")
    print(f"  из них было предыдущее касание ТОГО ЖЕ айтема: "
          f"{c['share_with_prior_touch']:.1%}")
    print(f"  МЕДИАННЫЙ РАЗРЫВ: {c['median_gap_minutes']:.1f} минут\n")
    for d in c["distribution"]:
        print(f"    {d['label']:<14}{d['share_of_all_purchases']:>7.1%} всех покупок")
    print("\n  Эффект разрыва предсказания на тот же тривиальный признак:")
    print(f"{'разрыв':>10}{'AUC':>10}{'NDCG@10':>11}{'у позитива есть история':>26}")
    for g in c["gap_effect"]:
        print(f"{g['gap_days']:>8} дн.{g['auc_seen']:>10.4f}{g['ndcg10_seen']:>11.4f}"
              f"{g['positives_with_history']:>25.1%}")
    print("\n  При разрыве в сутки ярлык исчезает полностью: AUC становится")
    print("  случайным. Только тогда задача превращается в рекомендацию.")
    print("=" * W)


if __name__ == "__main__":
    print("A: as-of против протекающего джойна...", flush=True)
    a = part_a_asof_vs_leaky()
    log = load(RetailRocketConfig(), verbose=False)
    print("B: протокол негативов...", flush=True)
    b = part_b_negative_protocol(log)
    print("C: внутрисессионная тривиальность...", flush=True)
    c = part_c_session_triviality(log)
    report(a, b, c)
    OUT.mkdir(exist_ok=True)
    (OUT / "e08_protocol_diagnostics.json").write_text(
        json.dumps({"part_a": a, "part_b": b, "part_c": c},
                   indent=2, ensure_ascii=False))
    print("результаты: reports/e08_protocol_diagnostics.json")
