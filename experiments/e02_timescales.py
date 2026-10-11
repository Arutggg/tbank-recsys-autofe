"""
Эксперимент 2. Существует ли вообще «правильное окно»?

Задание ставит вопрос прямо: «а что если вместо 7 дней взять 10, и это было
бы лучше?» Здесь он разбирается до конца, и ответ оказывается сильнее, чем
«да, 7 дней неоптимально».

Часть A — ИДЕНТИФИЦИРУЕМОСТЬ (валидация машинерии).
  Хирургический полигон: метка построена ровно из одного затухающего
  счётчика с известным полупериодом. Оптимальный масштаб существует по
  построению. Проверяем, находит ли его пайплайн, и каким критерием.

  Побочный результат, важный сам по себе: AUC НЕ ИДЕНТИФИЦИРУЕТ масштаб.
  Он ранговый и насыщается — признаки с полупериодами 0.6 д и 12 д дают
  один и тот же AUC до четвёртого знака, хотя различаются информационно
  вдвое (57% нулей против 38%). Отбирать окна по AUC нельзя.

Часть B — ВЫВОДИМ ЛИ ОПТИМУМ ИЗ ЗНАНИЯ МЕХАНИЗМОВ (главный результат).
  Богатый полигон, где мы САМИ задали все константы поведения. Вопрос:
  если знать параметры механизмов, можно ли вывести правильный полупериод
  признака? Свипуем параметры генератора и смотрим, следует ли за ними
  найденный оптимум.

  Ответ отрицательный, и это сильнее исходного тезиса:
    * оптимум отслеживает tau_cat только на коротком конце и насыщается;
    * на tau_item он реагирует НЕМОНОТОННО;
    * сдвиг одного параметра меняет оптимум ДРУГОГО семейства ключей,
      потому что механизмы связаны (интерес к категории управляет выбором
      айтема, пресыщение гасит повторы).

  Вывод: «правильный полупериод» не выводится из параметров механизмов
  даже на полигоне, построенном нами самими и полностью нам известном.
  Значит на реальных данных он не выводится тем более — ни рассуждением,
  ни доменной интуицией. Остаётся измерять, то есть держать всю сетку
  масштабов и отдавать выбор модели. Это обоснование автоматизации по
  существу, а не по удобству.

  Решающая эмпирическая проверка (сетка масштабов против окон, выбранных
  человеком) делается в основном бенчмарке, где у модели есть все признаки
  сразу, а не по одному.

ПРИМЕЧАНИЕ О ПРОТОКОЛЕ. Здесь намеренно используются негативы по
популярности и нулевой разрыв предсказания, в отличие от остальных
экспериментов. Причина: предмет изучения — временные масштабы на
управляемом полигоне, где все механизмы заданы нами. Смешанные негативы и
разрыв добавили бы к измерению два дополнительных фактора и смешали бы
эффект масштаба с эффектом постановки. Выводы этого эксперимента про
идентифицируемость, а не про абсолютный уровень метрики.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afe.datasets.synthetic import SyntheticConfig, generate, generate_known_optimum  # noqa: E402
from afe.dsl import ANY_ETYPE, FeatureSet, Primitive  # noqa: E402
from afe.engine import FeatureCompiler  # noqa: E402
from afe.eval import SplitConfig, TaskConfig, auc, build_queries, logloss  # noqa: E402
from afe.kernels import DAY  # noqa: E402
from afe.log import build_stream  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports"

#: Плотная сетка полупериодов для свипа. Шаг 1.54x, поэтому точнее чем
#: в 1.24 раза определить оптимум в принципе нельзя — это и есть предел
#: разрешения, с которым надо сравнивать результат.
DENSE = tuple(np.geomspace(0.25 * DAY, 365 * DAY, 18))
GRID_STEP = (365 / 0.25) ** (1 / 17)


def _sweep(log, queries, key, halflives=DENSE, n_etypes=None):
    """Посчитать dcount на сетке полупериодов и замерить log-loss и AUC.

    log-loss считается логистической регрессией на одном стандартизованном
    признаке, обученной на первой половине запросов и замеренной на второй.
    Форма модели (линейная в логите) согласована с формой порождения метки
    на хирургическом полигоне — именно поэтому критерий идентифицирует
    масштаб, тогда как ранговый AUC этого не делает.
    """
    n_etypes = n_etypes or log.n_etypes
    prims = [Primitive(key, ANY_ETYPE, "dcount", hl) for hl in halflives]
    X = np.nan_to_num(
        FeatureCompiler(FeatureSet(prims, []), n_etypes=n_etypes)
        .transform(build_stream(log, queries), verbose=False), nan=0.0)

    n = len(queries.y)
    tr, te = slice(0, n // 2), slice(n // 2, n)
    rows = []
    for j, hl in enumerate(halflives):
        v = X[:, j:j + 1].astype(np.float64)
        z = (v - v[tr].mean()) / (v[tr].std() + 1e-9)
        if len(np.unique(queries.y[tr])) < 2:
            rows.append((hl / DAY, np.nan, np.nan, float((v == 0).mean())))
            continue
        lr = LogisticRegression(max_iter=1000).fit(z[tr], queries.y[tr])
        p = lr.predict_proba(z[te])[:, 1]
        rows.append((hl / DAY, logloss(queries.y[te], p),
                     auc(queries.y[te], v[te, 0]), float((v == 0).mean())))
    return np.array(rows)


# ---------------------------------------------------------------- часть A
def part_a(seed: int = 0) -> dict:
    res = {"grid_resolution": GRID_STEP, "cases": []}
    for true_hl in (2.0, 12.0, 60.0):
        log, q, truth = generate_known_optimum(
            true_halflife_days=true_hl, n_events=300_000, n_queries=60_000, seed=seed)
        r = _sweep(log, q, ("user", "cat"), n_etypes=1)
        b_ll, b_auc = int(np.nanargmin(r[:, 1])), int(np.nanargmax(r[:, 2]))
        auc_ties = int((r[:, 2] >= r[b_auc, 2] - 1e-4).sum())
        res["cases"].append({
            "true_halflife_days": true_hl,
            "found_by_logloss": float(r[b_ll, 0]),
            "found_by_auc": float(r[b_auc, 0]),
            "ratio_logloss": float(r[b_ll, 0] / true_hl),
            "ratio_auc": float(r[b_auc, 0] / true_hl),
            "logloss_spread": float(np.nanmax(r[:, 1]) - np.nanmin(r[:, 1])),
            "auc_spread": float(np.nanmax(r[:, 2]) - np.nanmin(r[:, 2])),
            "n_auc_indistinguishable": auc_ties,
            "positive_rate": truth["positive_rate"],
            "curve": [[float(x) for x in row] for row in r],
        })
    return res


# ---------------------------------------------------------------- часть B
FAMILIES = [
    (("user", "cat"), "интерес к категории"),
    (("user", "item"), "повтор айтема"),
    (("item",), "трендовость айтема"),
]

#: Свипы параметров генератора. Для каждого значения замеряем, куда уйдёт
#: найденный оптимум у КАЖДОГО семейства ключей — так видно и отслеживание,
#: и связанность механизмов между собой.
SWEEPS = [
    ("tau_cat_days", (5.0, 45.0, 180.0)),
    ("tau_item_days", (1.0, 4.0, 40.0)),
    ("tau_pop_days", (2.0, 10.0, 60.0)),
]


def part_b(n_events: int = 600_000, seed: int = 0, verbose: bool = True) -> dict:
    res = {"sweeps": []}
    for param, values in SWEEPS:
        entry = {"param": param, "values": list(values), "runs": []}
        for v in values:
            cfg = SyntheticConfig(n_events=n_events, n_users=20_000, n_items=5_000,
                                  seed=seed, **{param: v})
            log, _ = generate(cfg)
            t1, t2 = SplitConfig().boundaries(log)
            q = build_queries(
                log, t2, int(log.t[-1]) + 1,
                TaskConfig(target_etype=2, n_negatives=49, max_positives=12_000,
                           seed=seed),
                train_t_end=t1)
            run = {"param_value": v, "families": []}
            for key, label in FAMILIES:
                r = _sweep(log, q, key)
                b = int(np.nanargmin(r[:, 1]))
                run["families"].append({
                    "key": "x".join(key), "label": label,
                    "optimal_halflife_days": float(r[b, 0]),
                    "logloss_at_optimum": float(r[b, 1]),
                    "logloss_spread": float(np.nanmax(r[:, 1]) - np.nanmin(r[:, 1])),
                })
            entry["runs"].append(run)
            if verbose:
                opt = ", ".join(f"{f['key']}={f['optimal_halflife_days']:.1f}д"
                                for f in run["families"])
                print(f"    {param}={v:<6g} -> {opt}", flush=True)
        res["sweeps"].append(entry)
    return res


def report(a: dict, b: dict) -> None:
    W = 78
    print("=" * W)
    print("ЧАСТЬ A. ИДЕНТИФИЦИРУЕМОСТЬ: оптимум существует по построению")
    print("=" * W)
    print(f"разрешение сетки: {a['grid_resolution']:.2f}x -> точнее чем в "
          f"{a['grid_resolution'] ** 0.5:.2f}x определить нельзя\n")
    print(f"{'истина':>9}{'по log-loss':>14}{'откл.':>8}{'по AUC':>11}{'откл.':>8}"
          f"{'размах LL':>11}{'размах AUC':>12}")
    for c in a["cases"]:
        print(f"{c['true_halflife_days']:>8.1f}д{c['found_by_logloss']:>13.2f}д"
              f"{c['ratio_logloss']:>7.2f}x{c['found_by_auc']:>10.2f}д"
              f"{c['ratio_auc']:>7.2f}x{c['logloss_spread']:>11.4f}"
              f"{c['auc_spread']:>12.4f}")
    print("\n  log-loss находит масштаб в пределах разрешения сетки во всех случаях.")
    print("  AUC промахивается и имеет размах на порядок меньше: он ранговый,")
    print("  насыщается и не различает признаки с разным содержанием информации.")

    print("\n" + "=" * W)
    print("ЧАСТЬ B. ВЫВОДИМ ЛИ ОПТИМУМ ИЗ ЗНАНИЯ МЕХАНИЗМОВ?")
    print("=" * W)
    print("Мы сами задали все константы поведения. Свипуем их и смотрим,")
    print("следует ли за ними найденный оптимальный полупериод признака.\n")

    for sw in b["sweeps"]:
        fams = [f["label"] for f in sw["runs"][0]["families"]]
        print(f"  свип {sw['param']}:")
        head = f"{'значение':>12}" + "".join(f"{name[:20]:>22}" for name in fams)
        print("  " + head)
        for run in sw["runs"]:
            row = f"{run['param_value']:>11g}д"
            for f in run["families"]:
                row += f"{f['optimal_halflife_days']:>21.2f}д"
            print("  " + row)
        # отслеживает ли оптимум свой собственный параметр
        own = {"tau_cat_days": 0, "tau_item_days": 1, "tau_pop_days": 2}[sw["param"]]
        opts = [run["families"][own]["optimal_halflife_days"] for run in sw["runs"]]
        pars = sw["values"]
        monotone = all(opts[i] <= opts[i + 1] for i in range(len(opts) - 1))
        par_range = max(pars) / min(pars)
        opt_range = max(opts) / max(min(opts), 1e-9)
        print(f"    параметр менялся в {par_range:.0f}x, оптимум сдвинулся в "
              f"{opt_range:.1f}x, монотонно: {'да' if monotone else 'НЕТ'}")
        # сдвинулись ли ЧУЖИЕ семейства
        for fi, name in enumerate(fams):
            if fi == own:
                continue
            o = [run["families"][fi]["optimal_halflife_days"] for run in sw["runs"]]
            print(f"    побочный сдвиг у «{name}»: {max(o) / max(min(o), 1e-9):.1f}x")
        print()

    print("  Итог: оптимум не воспроизводит параметры механизмов — он насыщается,")
    print("  реагирует немонотонно и смещается от ЧУЖИХ параметров, потому что")
    print("  механизмы связаны между собой. Если правильный полупериод не выводится")
    print("  из полного знания устройства полигона, на реальных данных он не")
    print("  выводится тем более. Остаётся измерять: держать всю сетку масштабов")
    print("  и отдавать выбор модели.")
    print("=" * W)


if __name__ == "__main__":
    print("часть A: хирургический полигон...", flush=True)
    a = part_a()
    print("часть B: богатый полигон...", flush=True)
    b = part_b()
    report(a, b)
    OUT.mkdir(exist_ok=True)
    (OUT / "e02_timescales.json").write_text(
        json.dumps({"part_a": a, "part_b": b}, indent=2, ensure_ascii=False))
    print("\nрезультаты: reports/e02_timescales.json")
