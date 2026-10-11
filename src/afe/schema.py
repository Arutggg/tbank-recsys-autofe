"""
Перечисление пространства признаков по схеме данных. Человек не участвует.

Это и есть «автоматизация расчёта фичей» из задания, только не в виде
LLM-агента, который УГАДЫВАЕТ SQL, а в виде компилятора, который
ИСЧЕРПЫВАЮЩЕ перебирает структурированное пространство.

Про ключи и мета-пути. Задание отдельно выделяет «косвенные признаки по
другим связям»: берём категориальный признак айтема (категория, автор,
мерчант) и смотрим активность юзера по нему. В терминах гетерогенного
графа взаимодействий это мета-путь длины 2: user -> item -> category.
«Прямые признаки по user-item» — мета-путь длины 1. То есть деление
задания на прямые и косвенные признаки — это деление по длине мета-пути,
а раз так, перечисляется он механически: берём все пары
(атрибут со стороны пользователя) x (атрибут со стороны айтема).

Именно поэтому идея №2 из задания (собрать граф и свести к link
prediction) здесь не конкурент, а подмножество: message passing по такому
графу агрегирует ровно те же мета-пути, но требует обучения и хранения
эмбеддингов, тогда как счётчик по ключу стоит O(1) на событие.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from .dsl import ANY_ETYPE, Derived, FeatureSet, Primitive
from .kernels import DEFAULT_HALFLIVES


@dataclass
class Schema:
    """Описание данных — единственное, что задаёт человек.

    Обратите внимание: здесь нет ни одной фичи. Человек сообщает, какие
    столбцы относятся к пользователю, какие к айтему, и сколько типов
    событий есть. Что из этого считать — решает перебор.
    """

    user_cols: list[str]          # ['user', 'region', ...]
    item_cols: list[str]          # ['item', 'cat', 'brand', ...]
    etype_names: list[str]        # ['view', 'cart', 'purchase']

    #: Высококардинальные ключи (например user x item) требуют много
    #: состояния. Порог, после которого движок переходит на хеширование.
    hash_threshold: int = 5_000_000

    @property
    def n_etypes(self) -> int:
        return len(self.etype_names)

    def key_specs(self, max_arity: int = 2) -> list[tuple[str, ...]]:
        """Все ключи агрегации = все мета-пути до заданной длины.

        Арность 0: глобальные счётчики (популярность, нормировки).
        Арность 1: признаки стороны пользователя и стороны айтема.
        Арность 2: пары user-side x item-side — прямые (user x item) и
                   косвенные (user x cat, user x brand, region x cat, ...).
        Пары внутри одной стороны не берём: атрибут айтема детерминирован
        самим айтемом, такой ключ не даёт новой информации.
        """
        keys: list[tuple[str, ...]] = [()]
        if max_arity >= 1:
            keys += [(c,) for c in self.user_cols]
            keys += [(c,) for c in self.item_cols]
        if max_arity >= 2:
            keys += [(u, i) for u in self.user_cols for i in self.item_cols]
        return keys


@dataclass
class EnumerationConfig:
    """Что включать в перебор. Все флаги — семейства, не отдельные фичи."""

    halflives: tuple[float, ...] = DEFAULT_HALFLIVES
    max_key_arity: int = 2
    #: статистики уровня 0
    use_dcount: bool = True
    use_dage_mean: bool = True
    use_dage_std: bool = True
    use_recency: bool = True
    use_age: bool = True
    #: типы событий: перебирать каждый по отдельности и/или «любой»
    per_etype: bool = True
    any_etype: bool = True
    #: семейства уровня 1
    use_conversion: bool = True
    use_share: bool = True
    use_trend: bool = True
    use_lift: bool = True
    #: предохранитель: ограничение на размер плана
    max_primitives: int = 20_000
    max_derived: int = 40_000


def enumerate_features(
    schema: Schema,
    cfg: EnumerationConfig | None = None,
    keys: list[tuple[str, ...]] | None = None,
) -> FeatureSet:
    """Построить полный план признаков из одной только схемы данных.

    keys — необязательное ограничение набора ключей. Нужно для абляций:
    например, зафиксировать ровно те ключи, по которым считает ручной
    baseline, и сравнить только семейство временных ядер, не смешивая
    эффект ядер с эффектом более богатого набора мета-путей.
    """
    cfg = cfg or EnumerationConfig()
    keys = list(keys) if keys is not None else schema.key_specs(cfg.max_key_arity)

    etypes: list[int] = []
    if cfg.per_etype:
        etypes += list(range(schema.n_etypes))
    if cfg.any_etype:
        etypes.append(ANY_ETYPE)

    kernel_stats = [
        s for s, on in (
            ("dcount", cfg.use_dcount),
            ("dage_mean", cfg.use_dage_mean),
            ("dage_std", cfg.use_dage_std),
        ) if on
    ]
    free_stats = [
        s for s, on in (("recency", cfg.use_recency), ("age", cfg.use_age)) if on
    ]

    # ---------------- уровень 0 ----------------
    prims: list[Primitive] = []
    index: dict[Primitive, int] = {}

    def add(p: Primitive) -> int:
        i = index.get(p)
        if i is None:
            i = len(prims)
            index[p] = i
            prims.append(p)
        return i

    for key in keys:
        for e in etypes:
            for stat in kernel_stats:
                for hl in cfg.halflives:
                    add(Primitive(key=key, etype=e, stat=stat, halflife_s=hl))
            for stat in free_stats:
                add(Primitive(key=key, etype=e, stat=stat, halflife_s=float("inf")))

    if len(prims) > cfg.max_primitives:
        raise ValueError(
            f"план содержит {len(prims):,} примитивов > max_primitives="
            f"{cfg.max_primitives:,}. Сократите сетку полупериодов или арность ключа."
        )

    # ---------------- уровень 1 ----------------
    derived: list[Derived] = []
    seen: set[tuple] = set()

    def add_derived(d: Derived) -> None:
        sig = (d.op, d.a, d.b, d.c, d.d)
        if sig not in seen and d.a != d.b:
            seen.add(sig)
            derived.append(d)

    def pidx(key, etype, hl, stat="dcount") -> int | None:
        return index.get(Primitive(key=key, etype=etype, stat=stat, halflife_s=hl))

    real_etypes = list(range(schema.n_etypes))

    # (1) КОНВЕРСИЯ: тот же ключ и масштаб, разные типы событий.
    #     purchase/view на (user x item) = «насколько этот юзер склонен
    #     доводить до покупки именно этот айтем».
    if cfg.use_conversion:
        for key in keys:
            for hl in cfg.halflives:
                for e_lo, e_hi in itertools.combinations(real_etypes, 2):
                    a, b = pidx(key, e_hi, hl), pidx(key, e_lo, hl)
                    if a is not None and b is not None:
                        add_derived(Derived("conversion", a, b))

    # (2) ДОЛЯ: ключ-уточнение к ключу-родителю.
    #     dcount[(user,item)]/dcount[(user,)] = доля активности юзера,
    #     пришедшаяся на этот айтем — классическая share-of-wallet фича.
    if cfg.use_share:
        key_set = set(keys)
        for key in keys:
            if len(key) < 2:
                continue
            for parent in (key[:1], key[1:]):
                if parent not in key_set:
                    continue
                for e in etypes:
                    for hl in cfg.halflives:
                        a, b = pidx(key, e, hl), pidx(parent, e, hl)
                        if a is not None and b is not None:
                            add_derived(Derived("share", a, b))

    # (3) КОНТРАСТ МАСШТАБОВ: тот же ключ и событие, короткое ядро / длинное.
    #     Это «растёт или падает интерес» — семейство, которое руками почти
    #     никогда не пишут, потому что для него нужно заранее держать два
    #     согласованных окна.
    if cfg.use_trend:
        hl_sorted = sorted(cfg.halflives)
        for key in keys:
            for e in etypes:
                for short, long in zip(hl_sorted[:-1], hl_sorted[1:]):
                    a, b = pidx(key, e, short), pidx(key, e, long)
                    if a is not None and b is not None:
                        add_derived(Derived("trend", a, b))

    # (4) LIFT: наблюдаемая совместность против ожидаемой при независимости,
    #     PMI-подобно. obs(u,i) * glob / (marg(u) * marg(i)).
    #     Нормирует affinity на популярность айтема и активность юзера —
    #     без этого модель путает «юзеру нравится» и «айтем популярен».
    if cfg.use_lift:
        for key in keys:
            if len(key) != 2:
                continue
            ku, ki = (key[0],), (key[1],)
            for e in etypes:
                for hl in cfg.halflives:
                    a = pidx(key, e, hl)
                    b, c = pidx(ku, e, hl), pidx(ki, e, hl)
                    d = pidx((), e, hl)
                    if None not in (a, b, c, d):
                        add_derived(Derived("lift", a, b, c, d))

    if len(derived) > cfg.max_derived:
        derived = derived[: cfg.max_derived]

    return FeatureSet(primitives=prims, derived=derived)
