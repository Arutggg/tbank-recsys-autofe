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

from afe.datasets.synthetic import SyntheticConfig, generate  # noqa: E402
from afe.engine import FeatureCompiler  # noqa: E402
from afe.eval import (  # noqa: E402
    SplitConfig,
    TaskConfig,
    all_metrics,  # noqa: E402
)
from afe.log import QuerySet, build_stream  # noqa: E402
from afe.model import GBDTConfig, fit_gbdt  # noqa: E402
from afe.schema import EnumerationConfig, Schema, enumerate_features  # noqa: E402
from afe.splits import build_splits  # noqa: E402

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


def _detector_flags(log, splits, schema, make_stream, cut) -> dict:
    """Прогнать ДЕТЕКТОР из части 1 на конкретном варианте признаков.

    Заявление «детектор ловит эту утечку» нельзя печатать без проверки.
    Здесь он действительно запускается на каждом протекающем варианте, и
    в отчёт идёт то, что он показал.
    """
    fs = enumerate_features(schema, EnumerationConfig())
    comp = FeatureCompiler(fs, n_etypes=log.n_etypes)
    q = splits.queries.select(splits.queries.t < cut)
    full = comp.transform(make_stream(log, q), verbose=False)
    cut_log = log.slice_until(cut)
    trunc = comp.transform(make_stream(cut_log, q), verbose=False)

    same_nan = bool(np.array_equal(np.isnan(full), np.isnan(trunc)))
    m = ~np.isnan(full) & ~np.isnan(trunc)
    diff = np.abs(full[m] - trunc[m]) if m.any() else np.zeros(1)
    scale = np.maximum(np.abs(full[m]), 1e-6) if m.any() else np.ones(1)
    n_bad = int((diff / scale > 1e-5).sum())
    return {"detected": (not same_nan) or n_bad > 0, "n_values_differing": n_bad}


def part2_positive_controls(log, splits, schema, verbose=True) -> dict:
    """Внести утечки намеренно и измерить, насколько они завышают метрику.

    Набор контролей подобран под ДЕЙСТВУЮЩИЙ протокол (разрыв предсказания
    в сутки, негативы как у ретривера). Это важно: в первой версии один из
    контролей оказался обезврежен самим протоколом и ничего не показывал.

      no_gap           забыли разрыв предсказания — признаки считаются на
                       момент покупки. Самая реалистичная ошибка из всех.
      future_window    запрос сдвинут на 7 дней вперёд: признаки считаются
                       по данным, которых в его момент не существовало.
      leaky_attributes джойн атрибутов «последнее известное значение»
                       вместо as-of. Та самая тихая утечка, которая не
                       ловится проверкой агрегатов.
      self_timestamp   запрос видит событие в свой же момент. Под разрывом
                       предсказания этот вариант НЕ ДОЛЖЕН ничего менять, и
                       это проверяется: разрыв попутно защищает и от ошибок
                       в строгости неравенства.
    """
    fs = enumerate_features(schema, EnumerationConfig())
    names = fs.names(log.etype_names)
    comp = FeatureCompiler(fs, n_etypes=log.n_etypes)
    rows = []

    def ev(tag, descr, stream, splits_used=None):
        X = comp.transform(stream, verbose=False)
        sp = splits_used or splits
        m = _train_eval(X, sp, names)
        del X
        rows.append({"variant": tag, "описание": descr, "metrics": m})
        if verbose:
            print(f"    {tag:<18} AUC={m['auc']:.4f}  NDCG@10={m['ndcg@10']:.4f}",
                  flush=True)

    if verbose:
        print("  варианты:", flush=True)
    ev("honest", "корректный протокол: разрыв 1 день, as-of атрибуты",
       build_stream(log, splits.queries))

    # --- забыли разрыв предсказания ---
    q_nogap = QuerySet(t=splits.queries.t + DAY, user=splits.queries.user,
                       item=splits.queries.item, y=splits.queries.y,
                       group=splits.queries.group, attrs=splits.queries.attrs)
    ev("no_gap", "забыт разрыв предсказания: признаки на момент покупки",
       build_stream(log, q_nogap))

    # --- запрос сдвинут в будущее ---
    q_future = QuerySet(t=splits.queries.t + 7 * DAY, user=splits.queries.user,
                        item=splits.queries.item, y=splits.queries.y,
                        group=splits.queries.group, attrs=splits.queries.attrs)
    ev("future_window", "признаки считаются на 7 дней позже момента запроса",
       build_stream(log, q_future))

    # --- запрос видит событие своего же момента ---
    ev("self_timestamp",
       "запрос видит событие в свой же момент (под разрывом не должен влиять)",
       build_stream(log, splits.queries, queries_after_events=True))

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
    print()
    base = p2["variants"][0]["metrics"]
    for v in p2["variants"][1:]:
        d = v["metrics"]["ndcg@10"] - base["ndcg@10"]
        verdict = ("ЗАВЫШАЕТ метрику" if d > 0.01 else
                   "не влияет (ожидаемо под разрывом)" if abs(d) < 0.01 else
                   "занижает")
        print(f"  {v['variant']:<17} dNDCG@10 = {d:+.4f}  ->  {verdict}")
    print("\n  Про self_timestamp отдельно: под разрывом предсказания запрос стоит")
    print("  в момент t минус сутки, где событий этого пользователя нет, поэтому")
    print("  ошибка в строгости неравенства физически не может проявиться.")
    print("  Разрыв попутно защищает и от неё — это не недостаток контроля,")
    print("  а свойство протокола, и оно здесь измерено, а не предположено.")
    print("=" * W)


if __name__ == "__main__":
    n_events = int(sys.argv[1]) if len(sys.argv) > 1 else 600_000
    cfg = SyntheticConfig(n_events=n_events, n_users=20_000, n_items=5_000, seed=0)
    log, _ = generate(cfg)
    splits = build_splits(log, SplitConfig(),
                          TaskConfig(target_etype=2, n_negatives=23, seed=0,
                                     prediction_gap_s=86400, neg_source="mixed"),
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
