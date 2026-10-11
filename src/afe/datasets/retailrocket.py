"""
Загрузчик RetailRocket — публичный датасет, структурно совпадающий с заданием.

Почему именно он. Задание описывает строку лога как (user, item, timestamp,
label), где label — тип взаимодействия 0/1/2, и отдельно упоминает
категориальные признаки айтема (категория, автор, мерчант). RetailRocket
даёт в точности это:

    events.csv            visitorid, itemid, timestamp, event
                          event in {view, addtocart, transaction} = label 0/1/2
    item_properties_*.csv timestamp, itemid, property, value
    category_tree.csv     categoryid, parentid

ЛОВУШКА С УТЕЧКОЙ, на которую легко сесть
------------------------------------------
Свойства айтема в этом датасете ВРЕМЕННЫЕ: одна строка item_properties —
это «такого-то числа у айтема такое-то значение свойства». Категория
айтема со временем меняется, цена меняется, доступность меняется.

Естественное и неверное решение — сделать словарь «айтем -> последнее
известное значение» и приджойнить его ко всем событиям. Это утечка из
будущего: для события в январе будет взята категория, присвоенная в
апреле. Утечка тихая, метрику она поднимает, а детектором обычного вида
(«не смотрим на события после t») не ловится, потому что происходит не в
агрегатах, а в джойне атрибутов.

Здесь делается корректный AS-OF джойн: для каждого события берётся
значение свойства, известное на момент этого события, то есть последняя
запись item_properties с меньшим timestamp. Если на момент события
значение ещё не было известно, ставится UNKNOWN (-1), а не ближайшее
будущее.

Отличие двух вариантов измеряется в experiments/e05_leak_audit.py: там
намеренно собирается «протекающая» версия признаков и показывается,
насколько она завышает метрику.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..log import UNKNOWN, EventLog

EVENT_TO_ETYPE = {"view": 0, "addtocart": 1, "transaction": 2}
ETYPE_NAMES = ["view", "addtocart", "transaction"]


@dataclass
class RetailRocketConfig:
    """Параметры загрузки и фильтрации.

    k-core фильтрация нужна по существу: в сыром RetailRocket около 1.4 млн
    посетителей, и у огромной доли ровно одно событие. На таких строках
    никакая история не определена, поэтому они только разбавляют метрику,
    не меняя сравнения методов. Границы фиксируются явно и указываются в
    отчёте, чтобы результат был воспроизводим.
    """

    root: Path = Path("data/retailrocket")
    min_user_events: int = 5
    min_item_events: int = 5
    kcore_iterations: int = 3
    use_asof_properties: bool = True     # False = заведомо протекающий вариант
    add_price_bucket: bool = True
    n_price_buckets: int = 10
    #: В RetailRocket числовые свойства закодированы строками вида "n123.000".
    #: Свойство 790 в датасете соответствует цене; однозначной документации
    #: на это нет, поэтому имя свойства вынесено в параметр, а не вшито.
    price_property: str = "790"


def _read_events(cfg: RetailRocketConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.root / "events.csv",
                     usecols=["timestamp", "visitorid", "event", "itemid"])
    df["t"] = (df["timestamp"] // 1000).astype("int64")      # мс -> с
    df["etype"] = df["event"].map(EVENT_TO_ETYPE).astype("int8")
    return df[["t", "visitorid", "itemid", "etype"]].sort_values("t", kind="stable")


def _kcore(df: pd.DataFrame, cfg: RetailRocketConfig) -> pd.DataFrame:
    """Итеративная k-core фильтрация: удаление редких юзеров и айтемов."""
    for _ in range(cfg.kcore_iterations):
        before = len(df)
        uc = df["visitorid"].value_counts()
        df = df[df["visitorid"].isin(uc[uc >= cfg.min_user_events].index)]
        ic = df["itemid"].value_counts()
        df = df[df["itemid"].isin(ic[ic >= cfg.min_item_events].index)]
        if len(df) == before:
            break
    return df


def _read_properties(cfg: RetailRocketConfig, prop: str) -> pd.DataFrame:
    """Все записи заданного свойства из обеих частей файла, по времени."""
    frames = []
    for part in ("item_properties_part1.csv", "item_properties_part2.csv"):
        path = cfg.root / part
        if not path.exists():
            continue
        df = pd.read_csv(path)
        frames.append(df[df["property"] == prop])
    if not frames:
        return pd.DataFrame(columns=["t", "itemid", "value"])
    out = pd.concat(frames, ignore_index=True)
    out["t"] = (out["timestamp"] // 1000).astype("int64")
    return out[["t", "itemid", "value"]].sort_values("t", kind="stable")


def _asof_values(
    events: pd.DataFrame, props: pd.DataFrame, cfg: RetailRocketConfig
) -> np.ndarray:
    """AS-OF джойн: значение свойства, известное НА МОМЕНТ события.

    merge_asof с by="itemid" и direction="backward" делает групповой as-of
    джойн векторизованно: для каждого события берётся последняя запись
    свойства ЭТОГО айтема с временем <= времени события. Если таких записей
    нет, значение остаётся пропущенным — и это правильно: на тот момент мы
    его действительно не знали. Подставлять будущее значение нельзя.

    (Первая версия делала цикл groupby по айтемам на Python — для ~230
    тысяч айтемов это десятки минут; параметр by переносит ту же логику
    в C.)
    """
    if props.empty:
        return np.full(len(events), None, dtype=object)

    if not cfg.use_asof_properties:
        # ЗАВЕДОМО ПРОТЕКАЮЩИЙ вариант: последнее известное значение для всех
        # событий, независимо от времени. Оставлен намеренно, чтобы измерить
        # величину утечки в эксперименте по аудиту.
        last = props.groupby("itemid")["value"].last()
        return events["itemid"].map(last).to_numpy(dtype=object)

    ev = events[["t", "itemid"]].reset_index(drop=True)
    ev["_row"] = np.arange(len(ev), dtype=np.int64)
    left = ev.sort_values("t", kind="stable")
    right = props[["t", "itemid", "value"]].sort_values("t", kind="stable")
    merged = pd.merge_asof(left, right, on="t", by="itemid", direction="backward")

    out = np.full(len(ev), None, dtype=object)
    out[merged["_row"].to_numpy()] = merged["value"].to_numpy()
    return out


def _asof_property(
    events: pd.DataFrame, props: pd.DataFrame, cfg: RetailRocketConfig
) -> np.ndarray:
    """Плотные коды значения свойства, разрешённого as-of."""
    return _encode(pd.Series(_asof_values(events, props, cfg)))


def _encode(values: pd.Series) -> np.ndarray:
    """Плотные int32-коды, пропуски -> UNKNOWN (-1)."""
    codes, _ = pd.factorize(values, use_na_sentinel=True)
    return codes.astype(np.int32)


def _parse_numeric(values: pd.Series) -> np.ndarray:
    """Числовые свойства RetailRocket закодированы строками вида 'n123.000'.

    Значение может содержать несколько токенов; берётся первый числовой.
    Нечисловые и пропуски дают NaN.
    """
    def one(v):
        if not isinstance(v, str):
            return np.nan
        for tok in v.split():
            if tok.startswith("n"):
                try:
                    return float(tok[1:])
                except ValueError:
                    continue
        return np.nan
    return values.map(one).to_numpy(dtype=np.float64)


def load(cfg: RetailRocketConfig | None = None, verbose: bool = True) -> EventLog:
    """Загрузить RetailRocket в канонический EventLog.

    Все категориальные признаки айтема разрешаются AS OF времени события
    (если cfg.use_asof_properties), поэтому лог уже готов к однопроходному
    счёту признаков без утечки.
    """
    cfg = cfg or RetailRocketConfig()
    if not (cfg.root / "events.csv").exists():
        raise FileNotFoundError(
            f"не найден {cfg.root / 'events.csv'}. Скачайте датасет:\n"
            f"  kaggle datasets download -d retailrocket/ecommerce-dataset "
            f"-p {cfg.root} --unzip")

    ev = _read_events(cfg)
    n_raw = len(ev)
    ev = _kcore(ev, cfg)
    if verbose:
        print(f"  события: {n_raw:,} -> {len(ev):,} после k-core "
              f"(>= {cfg.min_user_events} у юзера, >= {cfg.min_item_events} у айтема)")

    ev = ev.reset_index(drop=True)

    # --- категория, разрешённая as-of ---
    props_cat = _read_properties(cfg, "categoryid")
    props_cat = props_cat[props_cat["itemid"].isin(ev["itemid"].unique())]
    raw_cat_values = _asof_values(ev, props_cat, cfg)
    cat_codes = _encode(pd.Series(raw_cat_values))
    if verbose:
        known = (cat_codes >= 0).mean()
        print(f"  категория известна на момент события: {known:.1%}"
              f"{'' if cfg.use_asof_properties else '  [ПРОТЕКАЮЩИЙ режим]'}")

    # --- родительская категория по дереву ---
    parent_codes = np.full(len(ev), UNKNOWN, dtype=np.int32)
    tree_path = cfg.root / "category_tree.csv"
    if tree_path.exists() and not props_cat.empty:
        tree = pd.read_csv(tree_path)
        parent_of = dict(zip(tree["categoryid"].astype(str), tree["parentid"]))
        parents = pd.Series([parent_of.get(str(c)) if c is not None else None
                             for c in raw_cat_values])
        parent_codes = _encode(parents)

    attrs = {"cat": cat_codes, "parent_cat": parent_codes}

    # --- ценовой бакет (опционально) ---
    if cfg.add_price_bucket:
        props_price = _read_properties(cfg, cfg.price_property)
        props_price = props_price[props_price["itemid"].isin(ev["itemid"].unique())]
        if not props_price.empty:
            num = _parse_numeric(pd.Series(_asof_values(ev, props_price, cfg)))
            ok = np.isfinite(num)
            bucket = np.full(len(ev), UNKNOWN, dtype=np.int32)
            if ok.sum() > 100:
                qs = np.quantile(num[ok], np.linspace(0, 1, cfg.n_price_buckets + 1)[1:-1])
                bucket[ok] = np.searchsorted(qs, num[ok]).astype(np.int32)
            attrs["price_bucket"] = bucket
            if verbose:
                print(f"  ценовой бакет известен: {(bucket >= 0).mean():.1%}")

    # --- плотные id ---
    u_codes, u_uniq = pd.factorize(ev["visitorid"])
    i_codes, i_uniq = pd.factorize(ev["itemid"])

    log = EventLog(
        t=ev["t"].to_numpy(dtype=np.int64),
        user=u_codes.astype(np.int32),
        item=i_codes.astype(np.int32),
        etype=ev["etype"].to_numpy(dtype=np.int8),
        attrs={k: v.astype(np.int32) for k, v in attrs.items()},
        etype_names=list(ETYPE_NAMES),
        n_users=len(u_uniq),
        n_items=len(i_uniq),
        attr_cardinality={k: int(v.max()) + 1 if (v >= 0).any() else 1
                          for k, v in attrs.items()},
    )
    if verbose:
        print(log.describe())
    return log
