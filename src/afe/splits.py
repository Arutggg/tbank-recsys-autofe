"""
Построение train/val/test наборов кандидатов одним согласованным проходом.

Все три окна строятся как ОДИН объединённый QuerySet, признаки считаются
за один проход, и затем матрица режется по индексам. Это и быстрее, и
безопаснее: исключается возможность случайно посчитать train и test
разными версиями кода или разными настройками движка.

Корректность от объединения не страдает: строка-запрос читает состояние на
свой момент t и состояние не мутирует, поэтому наличие в том же потоке
запросов из будущего ни на что не влияет.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .eval import SplitConfig, TaskConfig, build_queries
from .log import EventLog, QuerySet


@dataclass
class Splits:
    queries: QuerySet
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray
    t_train_end: int
    t_val_end: int

    def y(self, which: str) -> np.ndarray:
        return self.queries.y[getattr(self, f"idx_{which}")]

    def group(self, which: str) -> np.ndarray:
        return self.queries.group[getattr(self, f"idx_{which}")]

    def rows(self, X: np.ndarray, which: str) -> np.ndarray:
        return X[getattr(self, f"idx_{which}")]

    def describe(self) -> str:
        gs = {w: len(np.unique(self.group(w))) for w in ("train", "val", "test")}
        return "\n".join(
            f"  {w:<6} строк={len(getattr(self, f'idx_{w}')):>9,}  групп={gs[w]:>8,}  "
            f"позитивов={self.y(w).mean():.2%}"
            for w in ("train", "val", "test")
        )


def build_splits(
    log: EventLog,
    split: SplitConfig | None = None,
    task: TaskConfig | None = None,
    max_positives_per_window: int | None = None,
    max_positives_train: int | None = None,
    n_negatives_eval: int | None = None,
) -> Splits:
    """Три временных окна кандидатов, склеенных в один QuerySet.

    Негативы во ВСЕХ окнах сэмплируются из распределения, построенного
    только по train — иначе в val/test просачивается информация о том,
    какие айтемы станут популярны.
    """
    split = split or SplitConfig()
    task = task or TaskConfig()
    t1, t2 = split.boundaries(log)
    t_end = int(log.t[-1]) + 1

    # train получает больше позитивов: при тысячах признаков объём обучающей
    # выборки прямо определяет, выживет ли широкий набор признаков. val и
    # test достаточно сделать такими, чтобы метрика была устойчивой.
    # Число негативов на обучении и на оценке разделено намеренно.
    #
    # Обучению много негативов не нужно: градиентный бустинг и так видит
    # достаточно отрицательных примеров, а память растёт линейно по строкам.
    # А вот ТЕСТОВЫЙ набор кандидатов должен быть реалистичным: при 23
    # негативах Recall@10 считается по топ-10 из 24 кандидатов, то есть по
    # почти половине списка, и метрика насыщается (замерено: ручной набор
    # на RetailRocket давал Recall@10 = 0.9911, что уже ничего не
    # различает). Настоящий ретривер отдаёт сотни кандидатов.
    windows = [(int(log.t[0]), t1), (t1, t2), (t2, t_end)]
    caps = [max_positives_train or max_positives_per_window,
            max_positives_per_window, max_positives_per_window]
    negs = [task.n_negatives,
            n_negatives_eval or task.n_negatives,
            n_negatives_eval or task.n_negatives]
    parts = []
    for (lo, hi), cap, nneg in zip(windows, caps, negs):
        tk = TaskConfig(**{**task.__dict__})
        tk.n_negatives = nneg
        if cap is not None:
            tk.max_positives = cap
        parts.append(build_queries(log, lo, hi, tk, train_t_end=t1))

    gshift, idx = 0, []
    cur = 0
    t_all, u_all, i_all, y_all, g_all = [], [], [], [], []
    attrs_all: dict[str, list] = {k: [] for k in parts[0].attrs}
    for qs in parts:
        n = len(qs)
        idx.append(np.arange(cur, cur + n))
        cur += n
        t_all.append(qs.t); u_all.append(qs.user); i_all.append(qs.item)
        y_all.append(qs.y); g_all.append(qs.group + gshift)
        gshift += int(qs.group.max()) + 1
        for k in attrs_all:
            attrs_all[k].append(qs.attrs[k])

    merged = QuerySet(
        t=np.concatenate(t_all), user=np.concatenate(u_all),
        item=np.concatenate(i_all), y=np.concatenate(y_all),
        group=np.concatenate(g_all),
        attrs={k: np.concatenate(v) for k, v in attrs_all.items()},
    )
    return Splits(queries=merged, idx_train=idx[0], idx_val=idx[1],
                  idx_test=idx[2], t_train_end=t1, t_val_end=t2)
