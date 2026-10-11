"""
Отбор признаков под бюджет: что именно отправлять в прод.

Сгенерировать 15 тысяч признаков легко. Продакшен-вопрос другой: какие из
них держать, если на инференс есть 20 мс и ограниченная память под
состояние счётчиков. Без ответа на него автоматическая генерация остаётся
демонстрацией, а не инженерным решением.

Почему это НЕ обычный рюкзак
----------------------------
Стоимость признаков не аддитивна. Все признаки на ключе user x brand
делят одно и то же состояние: массив моментов размером
(число ключей) x (типов событий) x (ядер) x 3. Заплатив за первый признак
на этом ключе, остальные на том же ключе получаешь почти бесплатно.

Значит решение принимается на уровне КЛЮЧЕЙ, а не отдельных признаков:
«заводить ли таблицу счётчиков user x brand» — именно так этот вопрос и
звучит в проде. Формально это групповой рюкзак (set-cover с весами групп),
и жадность по отношению «суммарная полезность группы / стоимость группы»
для такой структуры — естественный и дешёвый приближённый алгоритм.

Что считается стоимостью
------------------------
  память состояния — (число ключей) x (типов) x (ядер) x 3 x sizeof(float32)
                     плюс массивы времён; доминирующий член для ключей
                     высокой кардинальности вроде user x item
  стоимость прохода — число обновлений на событие, пропорционально числу
                     ядер у ключа

Полезность берётся из важности признаков обученной модели (gain), и затем
каждая точка кривой ПЕРЕПРОВЕРЯЕТСЯ реальным обучением на отобранном
подмножестве. Важность — только порядок рассмотрения; отчётная метрика
всегда измеренная, а не предсказанная.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dsl import Derived, FeatureSet, Primitive


@dataclass
class KeyCost:
    """Стоимость одного ключа агрегации."""

    key: tuple[str, ...]
    n_keys: int
    n_lambdas: int
    n_etypes: int
    state_bytes: int
    updates_per_event: int

    @property
    def name(self) -> str:
        return "global" if not self.key else "x".join(self.key)


def key_costs(
    fs: FeatureSet, cardinalities: dict[tuple[str, ...], int], n_etypes: int,
    state_itemsize: int = 4,
) -> dict[tuple[str, ...], KeyCost]:
    """Посчитать стоимость каждого ключа из плана признаков."""
    groups: dict[tuple[str, ...], set[float]] = {}
    for p in fs.primitives:
        groups.setdefault(p.key, set())
        if p.needs_kernel:
            groups[p.key].add(p.halflife_s)

    out = {}
    for key, hls in groups.items():
        n_l = max(len(hls), 1)
        n_k = cardinalities.get(key, 1)
        mom = n_k * n_etypes * n_l * 3 * state_itemsize
        times = n_k * n_etypes * 8 * 3          # t_touch / t_last / t_first (float64)
        out[key] = KeyCost(key=key, n_keys=n_k, n_lambdas=n_l, n_etypes=n_etypes,
                           state_bytes=mom + times, updates_per_event=n_l * 3)
    return out


def feature_key_map(fs: FeatureSet) -> list[tuple[tuple[str, ...], ...]]:
    """Для каждого признака — набор ключей, от которых он зависит.

    Производный признак зависит от ключей ВСЕХ своих слагаемых: отношение
    share[user x item / user] нельзя посчитать, не держа оба состояния.
    Это принципиально: иначе отбор недооценит стоимость уровня 1.
    """
    prims = fs.primitives
    mapping: list[tuple] = [(p.key,) for p in prims]
    for d in fs.derived:
        deps = {prims[d.a].key, prims[d.b].key}
        if d.c >= 0:
            deps.add(prims[d.c].key)
        if d.d >= 0:
            deps.add(prims[d.d].key)
        mapping.append(tuple(sorted(deps)))
    return mapping


@dataclass
class ParetoPoint:
    n_keys_selected: int
    keys: list[str]
    state_mib: float
    updates_per_event: int
    n_features: int
    metrics: dict = field(default_factory=dict)


def greedy_key_order(
    fs: FeatureSet, importance: np.ndarray,
    costs: dict[tuple[str, ...], KeyCost],
) -> list[tuple[str, ...]]:
    """Порядок добавления ключей по отношению полезность/стоимость.

    Полезность ключа — суммарная важность признаков, которые становятся
    доступны при его добавлении. Производный признак засчитывается ключу
    только когда доступны ВСЕ его зависимости, поэтому порядок строится
    итеративно: на каждом шаге пересчитывается, что именно откроется.
    """
    fkm = feature_key_map(fs)
    remaining = set(costs)
    chosen: list[tuple[str, ...]] = []
    chosen_set: set[tuple[str, ...]] = set()

    while remaining:
        best, best_ratio = None, -np.inf
        for key in remaining:
            trial = chosen_set | {key}
            gain = 0.0
            for fi, deps in enumerate(fkm):
                if key in deps and all(d in trial for d in deps):
                    gain += float(importance[fi])
            cost_mib = costs[key].state_bytes / 2**20
            ratio = gain / max(cost_mib, 1e-3)
            if ratio > best_ratio:
                best, best_ratio = key, ratio
        chosen.append(best)
        chosen_set.add(best)
        remaining.discard(best)
    return chosen


def features_for_keys(
    fs: FeatureSet, keys: set[tuple[str, ...]]
) -> np.ndarray:
    """Маска признаков, вычислимых при заданном наборе ключей."""
    fkm = feature_key_map(fs)
    return np.array([all(d in keys for d in deps) for deps in fkm], dtype=bool)


def gain_importance(booster, n_features: int) -> np.ndarray:
    """Важность по gain, выровненная по длине набора признаков."""
    imp = booster.feature_importance(importance_type="gain")
    out = np.zeros(n_features, dtype=np.float64)
    out[: len(imp)] = imp
    return out
