"""
Грамматика признаков: типизированное пространство, которое перебирает компилятор.

Наблюдение, с которого всё начинается. Ручная RecSys-фича — не эвристика,
а программа в маленьком типизированном пространстве:

    фича = СТАТИСТИКА над событиями типа E
           на КЛЮЧЕ K        (user / item / user x item / user x category / ...)
           с ВРЕМЕННЫМ ЯДРОМ (полупериод затухания)
           до момента t

Человек выбирает K, E и ядро — и больше ничего. Все три выбора перечислимы,
поэтому «придумывание фичей» заменяется перебором с отбором.

Два уровня:

  Primitive (уровень 0) — состояние, поддерживаемое движком за O(1) на
      событие. Только эти величины требуют прохода по логу.

  Derived (уровень 1) — арифметика над значениями уровня 0: конверсии,
      доли, контрасты масштабов, PMI-подобный lift. Стоят НОЛЬ
      дополнительных проходов по логу, а покрывают большую часть того, что
      человек пишет руками (CTR, conversion rate, share of wallet, affinity).

Отсюда главный структурный вывод: набор примитивов крошечный, а
порождаемое пространство признаков — огромное.
"""

from __future__ import annotations

from dataclasses import dataclass

from .kernels import format_halflife

#: Статистики уровня 0. Все поддерживаются за O(1) на событие.
STATS = (
    "dcount",     # A: затухающий счётчик событий
    "dage_mean",  # B/A: средний возраст истории (вес = затухание)
    "dage_std",   # sqrt(C/A - (B/A)^2): разброс возрастов
    "recency",    # t - t_last: время с последнего события (ядро не нужно)
    "age",        # t - t_first: время с первого события (ядро не нужно)
)

#: Статистики, которым временное ядро не требуется.
KERNEL_FREE_STATS = frozenset({"recency", "age"})

#: Операции уровня 1.
OPS = (
    "conversion",  # dcount[K,e2,l] / dcount[K,e1,l]      — конверсия между типами событий
    "share",       # dcount[K',e,l] / dcount[K,e,l]       — доля активности (K' уточняет K)
    "trend",       # dcount[K,e,l_short] / dcount[K,e,l_long] — контраст масштабов
    "lift",        # PMI-подобное отношение наблюдаемого к ожидаемому при независимости
)

ANY_ETYPE = -1  # etype = -1 означает «событие любого типа»


@dataclass(frozen=True, slots=True)
class Primitive:
    """Признак уровня 0: статистика на ключе, типе события и временном ядре."""

    key: tuple[str, ...]        # например ('user',) или ('user', 'cat'); () = глобально
    etype: int                  # индекс типа события, либо ANY_ETYPE
    stat: str                   # элемент STATS
    halflife_s: float           # полупериод; игнорируется для KERNEL_FREE_STATS

    def __post_init__(self) -> None:
        if self.stat not in STATS:
            raise ValueError(f"неизвестная статистика: {self.stat}")

    @property
    def needs_kernel(self) -> bool:
        return self.stat not in KERNEL_FREE_STATS

    @property
    def moment_order(self) -> int:
        """Максимальный порядок момента возраста, нужный для этой статистики."""
        return {"dcount": 0, "dage_mean": 1, "dage_std": 2}.get(self.stat, 0)

    @property
    def name(self) -> str:
        k = "global" if not self.key else "x".join(self.key)
        e = "any" if self.etype == ANY_ETYPE else f"e{self.etype}"
        if not self.needs_kernel:
            return f"{self.stat}[{k}|{e}]"
        return f"{self.stat}[{k}|{e}|{format_halflife(self.halflife_s)}]"

    def with_etype_names(self, names: list[str]) -> str:
        """Имя с человекочитаемыми типами событий — для отчёта."""
        k = "global" if not self.key else "x".join(self.key)
        e = "any" if self.etype == ANY_ETYPE else names[self.etype]
        if not self.needs_kernel:
            return f"{self.stat}[{k}|{e}]"
        return f"{self.stat}[{k}|{e}|{format_halflife(self.halflife_s)}]"


@dataclass(frozen=True, slots=True)
class Derived:
    """Признак уровня 1: арифметика над примитивами. Стоит ноль проходов по логу.

    Индексы ссылаются на позиции в списке примитивов. `lift` использует
    четыре слагаемых (наблюдаемое, два маргинала, глобальное), остальные — два.
    """

    op: str
    a: int                      # числитель
    b: int                      # знаменатель
    c: int = -1                 # для lift: маргинал
    d: int = -1                 # для lift: глобальная норма

    def __post_init__(self) -> None:
        if self.op not in OPS:
            raise ValueError(f"неизвестная операция: {self.op}")

    def name(self, prims: list[Primitive]) -> str:
        if self.op == "lift":
            return f"lift[{prims[self.a].name}/({prims[self.b].name}*{prims[self.c].name})]"
        return f"{self.op}[{prims[self.a].name}/{prims[self.b].name}]"


@dataclass
class FeatureSet:
    """Полный план вычислений: что считать на проходе и что вывести арифметикой."""

    primitives: list[Primitive]
    derived: list[Derived]

    @property
    def n_total(self) -> int:
        return len(self.primitives) + len(self.derived)

    def names(self, etype_names: list[str] | None = None) -> list[str]:
        if etype_names is None:
            prim_names = [p.name for p in self.primitives]
        else:
            prim_names = [p.with_etype_names(etype_names) for p in self.primitives]
        return prim_names + [d.name(self.primitives) for d in self.derived]

    def describe(self) -> str:
        from collections import Counter

        by_stat = Counter(p.stat for p in self.primitives)
        by_op = Counter(d.op for d in self.derived)
        by_arity = Counter(len(p.key) for p in self.primitives)
        lines = [
            f"примитивов (уровень 0, требуют проход по логу): {len(self.primitives):,}",
            *(f"    {s:<10} {c:>6,}" for s, c in sorted(by_stat.items())),
            f"  по арности ключа:",
            *(f"    ключ из {a}: {c:>6,}" for a, c in sorted(by_arity.items())),
            f"производных (уровень 1, арифметика, 0 проходов): {len(self.derived):,}",
            *(f"    {o:<10} {c:>6,}" for o, c in sorted(by_op.items())),
            f"ВСЕГО признаков: {self.n_total:,}",
        ]
        return "\n".join(lines)
