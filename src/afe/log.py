"""
Канонический формат лога взаимодействий и склейка «лог + запросы» в один поток.

Ключевая идея модуля — в `build_stream`. Признаки считаются однопроходным
сканированием по времени: на строке-событии состояние ОБНОВЛЯЕТСЯ, на
строке-запросе состояние ЧИТАЕТСЯ и ничего не обновляется. При равных
timestamp запрос стоит строго раньше события, поэтому значение признака в
момент t физически не может зависеть от событий с t' >= t.

Это и есть «отсутствие утечки по построению»: свойство порядка операций,
а не свойство правильно написанного WHERE t < ...
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

UNKNOWN = np.int32(-1)  # код отсутствующего значения категориального признака


@dataclass
class EventLog:
    """Лог взаимодействий, отсортированный по времени.

    Все идентификаторы и категориальные признаки — плотные int32-коды
    (dense ids), это требование движка: состояние хранится в плоских
    массивах, индексируемых кодом, без хеш-таблиц в горячем цикле.

    attrs — категориальные признаки, уже разрешённые AS OF времени события.
    Для датасетов, где свойства айтема меняются во времени (RetailRocket),
    разрешение as-of делает загрузчик; брать «последнее известное значение»
    нельзя — это утечка из будущего.
    """

    t: np.ndarray          # int64[N]  unix seconds, неубывающий
    user: np.ndarray       # int32[N]
    item: np.ndarray       # int32[N]
    etype: np.ndarray      # int8[N]   тип события (0=view, 1=cart, 2=purchase, ...)
    attrs: dict[str, np.ndarray] = field(default_factory=dict)  # name -> int32[N]
    etype_names: list[str] = field(default_factory=list)
    n_users: int = 0
    n_items: int = 0
    attr_cardinality: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.t)
        assert self.user.shape == self.item.shape == self.etype.shape == (n,)
        assert self.t.dtype == np.int64, self.t.dtype
        assert self.user.dtype == np.int32 and self.item.dtype == np.int32
        assert self.etype.dtype == np.int8
        assert np.all(np.diff(self.t) >= 0), "лог должен быть отсортирован по времени"
        for name, col in self.attrs.items():
            assert col.shape == (n,), f"attr {name}: {col.shape} != {(n,)}"
            assert col.dtype == np.int32, f"attr {name}: dtype {col.dtype}"

    @property
    def n_events(self) -> int:
        return len(self.t)

    @property
    def n_etypes(self) -> int:
        return len(self.etype_names)

    def column(self, name: str) -> np.ndarray:
        """Доступ к любому столбцу по имени — так движок разрешает ключи фичей."""
        if name == "user":
            return self.user
        if name == "item":
            return self.item
        return self.attrs[name]

    def cardinality(self, name: str) -> int:
        if name == "user":
            return self.n_users
        if name == "item":
            return self.n_items
        return self.attr_cardinality[name]

    @property
    def column_names(self) -> list[str]:
        return ["user", "item", *self.attrs.keys()]

    def time_quantile(self, q: float) -> int:
        """Момент времени, отсекающий долю q событий. Для временных сплитов."""
        return int(np.quantile(self.t, q))

    def slice_until(self, t_end: int) -> EventLog:
        """Префикс лога строго до t_end. Нужен для аудита утечки."""
        k = int(np.searchsorted(self.t, t_end, side="left"))
        return EventLog(
            t=self.t[:k].copy(),
            user=self.user[:k].copy(),
            item=self.item[:k].copy(),
            etype=self.etype[:k].copy(),
            attrs={k_: v[:k].copy() for k_, v in self.attrs.items()},
            etype_names=list(self.etype_names),
            n_users=self.n_users,
            n_items=self.n_items,
            attr_cardinality=dict(self.attr_cardinality),
        )

    def describe(self) -> str:
        lines = [
            f"события:      {self.n_events:,}",
            f"пользователи: {self.n_users:,}",
            f"айтемы:       {self.n_items:,}",
            f"период:       {pd.to_datetime(self.t[0], unit='s').date()} .. "
            f"{pd.to_datetime(self.t[-1], unit='s').date()}",
        ]
        cnt = np.bincount(self.etype, minlength=self.n_etypes)
        for i, name in enumerate(self.etype_names):
            lines.append(f"  {name:<12} {cnt[i]:>12,}  ({cnt[i] / self.n_events:6.2%})")
        for name in self.attrs:
            lines.append(f"attr {name:<9} уникальных: {self.attr_cardinality[name]:,}")
        return "\n".join(lines)


@dataclass
class QuerySet:
    """Строки, для которых нужны признаки и есть целевая переменная.

    Это (user, item, t) пары-кандидаты. Запрос НЕ обновляет состояние движка,
    поэтому кандидат может быть как реальным взаимодействием, так и
    негативом, которого в логе нет.

    group — идентификатор группы для ранжирующих метрик (обычно (user, t)).
    """

    t: np.ndarray        # int64[M]
    user: np.ndarray     # int32[M]
    item: np.ndarray     # int32[M]
    y: np.ndarray        # int8[M]  целевая переменная 0/1
    group: np.ndarray    # int64[M] id группы для NDCG/Recall
    attrs: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        m = len(self.t)
        for name, arr in (("user", self.user), ("item", self.item),
                          ("y", self.y), ("group", self.group)):
            assert arr.shape == (m,), f"{name}: {arr.shape} != {(m,)}"
        assert self.t.dtype == np.int64

    def __len__(self) -> int:
        return len(self.t)

    def column(self, name: str) -> np.ndarray:
        if name == "user":
            return self.user
        if name == "item":
            return self.item
        return self.attrs[name]

    def select(self, mask: np.ndarray) -> QuerySet:
        return QuerySet(
            t=self.t[mask], user=self.user[mask], item=self.item[mask],
            y=self.y[mask], group=self.group[mask],
            attrs={k: v[mask] for k, v in self.attrs.items()},
        )


@dataclass
class Stream:
    """Слитый по времени поток событий и запросов — вход движка признаков.

    is_query[i] == True  -> строка-запрос: читаем состояние, не обновляем.
    is_query[i] == False -> строка-события: обновляем состояние после чтения.

    qpos[i] — индекс строки в исходном QuerySet (или -1), чтобы вернуть
    матрицу признаков в исходном порядке запросов.
    """

    t: np.ndarray            # int64[N+M]
    is_query: np.ndarray     # bool[N+M]
    qpos: np.ndarray         # int32[N+M]
    etype: np.ndarray        # int8[N+M]  для запросов -1
    columns: dict[str, np.ndarray]  # name -> int32[N+M]
    n_queries: int
    cardinality: dict[str, int]

    @property
    def size(self) -> int:
        return len(self.t)


def build_stream(
    log: EventLog,
    queries: QuerySet | None = None,
    queries_after_events: bool = False,
) -> Stream:
    """Склеить лог и запросы в один неубывающий по времени поток.

    При равных timestamp запрос ставится РАНЬШЕ события. Это гарантирует,
    что признаки запроса в момент t зависят только от событий с t' < t,
    в том числе когда само целевое взаимодействие лежит в логе на том же t.

    queries_after_events=True переворачивает этот порядок и ВНОСИТ УТЕЧКУ:
    запрос начинает видеть событие, которое произошло в тот же момент, то
    есть в том числе собственное целевое взаимодействие. Режим нужен
    исключительно как позитивный контроль в аудите утечки
    (experiments/e05_leak_audit.py).

    Разница между корректным и протекающим вариантом — одна константа в
    ключе сортировки. Именно поэтому утечку такого рода легко внести
    случайно и невозможно заметить по метрике: она её только улучшает.
    """
    n = log.n_events
    m = 0 if queries is None else len(queries)

    # secondary key: 0 для запросов, 1 для событий -> при равных t запрос первый
    t_all = log.t if m == 0 else np.concatenate([queries.t, log.t])
    q_rank = np.int8(2) if queries_after_events else np.int8(0)
    sec = np.concatenate([np.full(m, q_rank, np.int8), np.ones(n, np.int8)])
    qpos_all = np.concatenate([np.arange(m, dtype=np.int32),
                               np.full(n, -1, dtype=np.int32)])
    is_q_all = np.concatenate([np.ones(m, bool), np.zeros(n, bool)])
    etype_all = np.concatenate([np.full(m, -1, np.int8), log.etype])

    order = np.lexsort((sec, t_all))

    columns: dict[str, np.ndarray] = {}
    for name in log.column_names:
        log_col = log.column(name)
        if m == 0:
            merged = log_col
        else:
            # признак запроса, которого нет в QuerySet.attrs, берём как UNKNOWN:
            # движок всё равно использует его только как ключ агрегата
            q_col = (queries.column(name) if name in ("user", "item") or name in queries.attrs
                     else np.full(m, UNKNOWN, np.int32))
            merged = np.concatenate([q_col, log_col])
        columns[name] = np.ascontiguousarray(merged[order], dtype=np.int32)

    return Stream(
        t=np.ascontiguousarray(t_all[order], dtype=np.int64),
        is_query=np.ascontiguousarray(is_q_all[order]),
        qpos=np.ascontiguousarray(qpos_all[order], dtype=np.int32),
        etype=np.ascontiguousarray(etype_all[order], dtype=np.int8),
        columns=columns,
        n_queries=m,
        cardinality={name: log.cardinality(name) for name in log.column_names},
    )
