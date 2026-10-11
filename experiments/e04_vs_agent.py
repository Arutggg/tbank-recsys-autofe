"""
Эксперимент 4. Идея №1 из задания: «может, ИИ агент напишет SQL за нас лучше?»

Разбор делается в двух частях, и вывод НЕ «агент бесполезен». Вывод точнее
и полезнее: предложения агента почти целиком лежат внутри пространства,
которое мы перебираем исчерпывающе, поэтому угадывание там не может
выиграть — оно может только совпасть. А ценность агента (и человека)
остаётся ровно в том остатке, который в грамматику не входит, и этот
остаток можно назвать по пунктам.

ЧАСТЬ 1. ВЛОЖЕННОСТЬ (документированная, проверяемая глазами).
  Список признаков, которые языковая модель предлагает для этой задачи,
  выписан явно, и каждый пункт помечен: лежит он в грамматике или нет.
  Рецензент может прочитать список и проверить классификацию сам.

ЧАСТЬ 2. УГАДЫВАНИЕ ПРОТИВ ПЕРЕБОРА С ОТБОРОМ (измерение).
  Агент моделируется как выбор k признаков ИЗ ПРАВДОПОДОБНОГО
  ПОДПРОСТРАНСТВА: округлённые человеком окна, привычные ключи, привычные
  агрегаты. Сравнение идёт при ОДИНАКОВОМ k с отбором по важности из
  полного перебора. Так измеряется именно ценность перебора и отбора, а не
  разница в числе признаков.

  Это модель, а не реальный агент, и так она и помечена. Моделирование
  выбрано сознательно: реальный вызов LLM невоспроизводим (ответ зависит
  от версии модели и температуры), стоит денег и не поддаётся
  перепроверке рецензентом, а проверяемое утверждение здесь — про
  структуру пространства, а не про качество конкретной модели.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afe.datasets.synthetic import SyntheticConfig, generate          # noqa: E402
from afe.dsl import ANY_ETYPE, FeatureSet                             # noqa: E402
from afe.engine import FeatureCompiler                                # noqa: E402
from afe.eval import SplitConfig, TaskConfig, all_metrics             # noqa: E402
from afe.kernels import DAY                                           # noqa: E402
from afe.log import build_stream                                      # noqa: E402
from afe.model import GBDTConfig, fit_gbdt, tune_gbdt                 # noqa: E402
from afe.schema import EnumerationConfig, Schema, enumerate_features  # noqa: E402
from afe.splits import build_splits                                   # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports"

# --------------------------------------------------------------------------
# ЧАСТЬ 1: что предлагает языковая модель и что из этого в грамматике
# --------------------------------------------------------------------------
#
# Список составлен добросовестно: это те признаки, которые действительно
# предлагаются для постановки «(user, item, timestamp, label) плюс
# категориальные признаки айтема». Пометка IN означает, что признак
# порождается грамматикой из dsl.py автоматически; OUT — что не порождается.

PROPOSALS = [
    # --- внутри грамматики ---
    ("IN", "число событий юзера за 1/7/30 дней", "dcount[user|e|hl]"),
    ("IN", "число событий айтема за 1/7/30 дней", "dcount[item|e|hl]"),
    ("IN", "сколько раз юзер взаимодействовал с этим айтемом", "dcount[userxitem|e|hl]"),
    ("IN", "активность юзера в категории айтема", "dcount[userxcat|e|hl]"),
    ("IN", "время с последнего события юзера", "recency[user]"),
    ("IN", "время с последнего взаимодействия с этим айтемом", "recency[userxitem]"),
    ("IN", "возраст аккаунта юзера", "age[user]"),
    ("IN", "время с первого появления айтема", "age[item]"),
    ("IN", "конверсия айтема из просмотра в покупку", "conversion[item]"),
    ("IN", "склонность юзера доводить до покупки", "conversion[user]"),
    ("IN", "доля категории в активности юзера", "share[userxcat/user]"),
    ("IN", "популярность айтема", "dcount[item|any|inf]"),
    ("IN", "общая активность юзера за всё время", "dcount[user|any|inf]"),
    ("IN", "популярность категории", "dcount[cat|any|hl]"),
    ("IN", "аффинити юзера к бренду", "dcount[userxbrand|e|hl]"),
    ("IN", "аффинити юзера к ценовому сегменту", "dcount[userxprice_bucket|e|hl]"),
    ("IN", "растёт ли интерес: короткое окно / длинное", "trend[hl_short/hl_long]"),
    ("IN", "аффинити с поправкой на популярность (PMI)", "lift[...]"),
    ("IN", "средняя давность истории юзера", "dage_mean[user|e|hl]"),

    # --- вне грамматики: настоящие пробелы, их надо называть честно ---
    ("OUT", "совпадает ли категория айтема с категорией ПОСЛЕДНЕГО просмотра",
     "последовательность, порядок событий"),
    ("OUT", "item-item co-visitation: что покупают вместе с этим",
     "мета-путь длины 3: user->item->user->item"),
    ("OUT", "косинус между эмбеддингами юзера и айтема (ALS / word2vec)",
     "обучаемые представления"),
    ("OUT", "длина сессии, позиция в сессии, dwell time",
     "сессионная структура"),
    ("OUT", "час дня, день недели, праздник",
     "календарные признаки"),
    ("OUT", "абсолютная цена и цена относительно средней по категории",
     "агрегация ЧИСЛОВОГО значения, а не ключа"),
    ("OUT", "марковские переходы между категориями",
     "последовательность, порядок событий"),
    ("OUT", "признаки из текста и картинок товара",
     "контент"),
]


def part1_containment() -> dict:
    n_in = sum(1 for p in PROPOSALS if p[0] == "IN")
    n_out = len(PROPOSALS) - n_in
    return {"proposals": [{"status": s, "feature": f, "maps_to": m}
                          for s, f, m in PROPOSALS],
            "n_in_grammar": n_in, "n_out_of_grammar": n_out,
            "share_in_grammar": n_in / len(PROPOSALS)}


# --------------------------------------------------------------------------
# ЧАСТЬ 2: правдоподобное подпространство = то, что выбирают по догадке
# --------------------------------------------------------------------------

#: Окна, которые человек и языковая модель округляют: сутки, неделя, месяц, год.
#: Из нашей сетки это ближайшие полупериоды.
PLAUSIBLE_HALFLIVES = (1 * DAY, 7 * DAY, 30 * DAY, float("inf"))

#: Ключи, которые предлагаются практически всегда.
PLAUSIBLE_KEYS = [("user",), ("item",), ("user", "item"), ("user", "cat")]

#: Статистики, которые предлагаются. Моменты возрастов НЕ предлагаются
#: почти никогда — это и есть часть выигрыша перебора.
PLAUSIBLE_STATS = {"dcount", "recency", "age"}


def plausible_mask(fs: FeatureSet) -> np.ndarray:
    """Маска признаков, попадающих в правдоподобное подпространство."""
    prims = fs.primitives
    mask = np.zeros(fs.n_total, dtype=bool)
    for i, p in enumerate(prims):
        if (p.key in PLAUSIBLE_KEYS and p.stat in PLAUSIBLE_STATS
                and (not p.needs_kernel or p.halflife_s in PLAUSIBLE_HALFLIVES)):
            mask[i] = True
    for j, d in enumerate(fs.derived):
        if d.op not in ("conversion", "share"):
            continue           # trend и lift по догадке почти не предлагают
        deps = [d.a, d.b] + ([d.c, d.d] if d.c >= 0 else [])
        if all(mask[x] for x in deps):
            mask[len(prims) + j] = True
    return mask


def run(n_events=600_000, n_trials=6, k_values=(50, 100, 200),
        agent_seeds=(0, 1, 2, 3, 4), seed=0, verbose=True) -> dict:
    cfg = SyntheticConfig(n_events=n_events, n_users=20_000, n_items=5_000, seed=seed)
    log, _ = generate(cfg)
    splits = build_splits(log, SplitConfig(),
                          TaskConfig(target_etype=2, n_negatives=23, seed=seed),
                          max_positives_per_window=3_000, max_positives_train=10_000)
    schema = Schema(user_cols=["user"],
                    item_cols=["item", "cat", "brand", "price_bucket"],
                    etype_names=log.etype_names)
    fs = enumerate_features(schema, EnumerationConfig())
    names = fs.names(log.etype_names)
    X = FeatureCompiler(fs, n_etypes=log.n_etypes).transform(
        build_stream(log, splits.queries), verbose=verbose)

    pm = plausible_mask(fs)
    if verbose:
        print(f"\n  полное пространство:            {fs.n_total:,} признаков")
        print(f"  правдоподобное подпространство: {int(pm.sum()):,} признаков "
              f"({pm.mean():.0%})")

    base = GBDTConfig(seed=seed)

    # важность по полному набору — основа отбора компилятора (решение на val)
    b_all = fit_gbdt(splits.rows(X, "train"), splits.y("train"),
                     splits.rows(X, "val"), splits.y("val"), base, names)
    imp = b_all.feature_importance(importance_type="gain")
    order = np.argsort(imp)[::-1]

    def eval_mask(mask):
        nm = [names[i] for i in np.flatnonzero(mask)]
        cfg_b, booster, val_ll = tune_gbdt(
            splits.rows(X, "train")[:, mask], splits.y("train"),
            splits.rows(X, "val")[:, mask], splits.y("val"),
            n_trials=n_trials, seed=seed, feature_names=nm)
        p = booster.predict(splits.rows(X, "test")[:, mask],
                            num_iteration=booster.best_iteration)
        return all_metrics(splits.y("test"), p, splits.group("test"), proba=p)

    plausible_idx = np.flatnonzero(pm)
    results = []
    for k in k_values:
        if k > len(plausible_idx):
            continue
        # --- компилятор: top-k по важности из ПОЛНОГО пространства ---
        m_comp = np.zeros(fs.n_total, dtype=bool)
        m_comp[order[:k]] = True
        comp_m = eval_mask(m_comp)
        if verbose:
            print(f"\n  k={k}: компилятор (отбор из {fs.n_total:,}) "
                  f"AUC={comp_m['auc']:.4f} NDCG@10={comp_m['ndcg@10']:.4f}",
                  flush=True)

        # --- агент: k случайных из правдоподобного подпространства ---
        agent_runs = []
        for s in agent_seeds:
            rng = np.random.default_rng(1000 + s)
            pick = rng.choice(plausible_idx, size=k, replace=False)
            m_ag = np.zeros(fs.n_total, dtype=bool)
            m_ag[pick] = True
            agent_runs.append(eval_mask(m_ag))
        if verbose:
            a = np.array([r["auc"] for r in agent_runs])
            n = np.array([r["ndcg@10"] for r in agent_runs])
            print(f"       агент (угадывание, {len(agent_seeds)} прогонов) "
                  f"AUC={a.mean():.4f}±{a.std():.4f} "
                  f"NDCG@10={n.mean():.4f}±{n.std():.4f}", flush=True)

        # --- агент «идеальный»: top-k по важности ВНУТРИ подпространства ---
        inside = [i for i in order if pm[i]][:k]
        m_best = np.zeros(fs.n_total, dtype=bool)
        m_best[inside] = True
        best_m = eval_mask(m_best)
        if verbose:
            print(f"       идеальный агент (лучшие k ИЗ подпространства) "
                  f"AUC={best_m['auc']:.4f} NDCG@10={best_m['ndcg@10']:.4f}",
                  flush=True)

        results.append({
            "k": k,
            "compiler": comp_m,
            "agent_mean": {m: float(np.mean([r[m] for r in agent_runs]))
                           for m in comp_m},
            "agent_std": {m: float(np.std([r[m] for r in agent_runs]))
                          for m in comp_m},
            "agent_runs": agent_runs,
            "oracle_agent": best_m,
        })

    return {"space": {"total": fs.n_total, "plausible": int(pm.sum())},
            "k_results": results}


def report(p1: dict, p2: dict | None) -> None:
    W = 94
    print("=" * W)
    print("ЧАСТЬ 1. ВЛОЖЕННОСТЬ: что предлагает языковая модель")
    print("=" * W)
    for s, f, m in PROPOSALS:
        tag = "в грамматике " if s == "IN" else "ВНЕ грамматики"
        print(f"  [{tag}] {f}")
        print(f"                    -> {m}")
    print(f"\n  в грамматике: {p1['n_in_grammar']} из {len(PROPOSALS)} "
          f"({p1['share_in_grammar']:.0%})")
    print("  Поэтому угадывание внутри этого пространства не может выиграть у")
    print("  исчерпывающего перебора — только совпасть. Ценность агента и")
    print("  человека остаётся в перечисленных пунктах ВНЕ грамматики, и это")
    print("  конкретный список, а не общие слова.")

    if p2 is None:
        print("=" * W)
        return
    print("\n" + "=" * W)
    print("ЧАСТЬ 2. УГАДЫВАНИЕ ПРОТИВ ПЕРЕБОРА С ОТБОРОМ, при одинаковом k")
    print("=" * W)
    print(f"  полное пространство {p2['space']['total']:,}, "
          f"правдоподобное подпространство {p2['space']['plausible']:,}\n")
    print(f"{'k':>5}{'метод':<30}{'AUC':>10}{'NDCG@10':>11}{'Recall@10':>12}")
    print("-" * W)
    for r in p2["k_results"]:
        k = r["k"]
        print(f"{k:>5}{'компилятор (top-k из всех)':<30}"
              f"{r['compiler']['auc']:>10.4f}{r['compiler']['ndcg@10']:>11.4f}"
              f"{r['compiler']['recall@10']:>12.4f}")
        print(f"{'':>5}{'идеальный агент (top-k из подпр.)':<30}"
              f"{r['oracle_agent']['auc']:>10.4f}"
              f"{r['oracle_agent']['ndcg@10']:>11.4f}"
              f"{r['oracle_agent']['recall@10']:>12.4f}")
        print(f"{'':>5}{'агент угадывает (среднее)':<30}"
              f"{r['agent_mean']['auc']:>10.4f}{r['agent_mean']['ndcg@10']:>11.4f}"
              f"{r['agent_mean']['recall@10']:>12.4f}")
        print(f"{'':>5}{'  +- по прогонам':<30}"
              f"{r['agent_std']['auc']:>10.4f}{r['agent_std']['ndcg@10']:>11.4f}"
              f"{r['agent_std']['recall@10']:>12.4f}")
        print("-" * W)
    print("=" * W)


if __name__ == "__main__":
    p1 = part1_containment()
    run_part2 = "--part1-only" not in sys.argv
    p2 = run() if run_part2 else None
    report(p1, p2)
    OUT.mkdir(exist_ok=True)
    (OUT / "e04_vs_agent.json").write_text(
        json.dumps({"part1": p1, "part2": p2}, indent=2, ensure_ascii=False))
    print("результаты: reports/e04_vs_agent.json")
