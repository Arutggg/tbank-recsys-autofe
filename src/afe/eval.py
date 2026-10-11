"""
Протокол оценки: временной сплит, построение кандидатов, метрики.

Здесь сосредоточено почти всё, где обычно ломается честность сравнения в
RecSys, поэтому каждое решение зафиксировано явно.

1. СПЛИТ ТОЛЬКО ПО ВРЕМЕНИ. Случайный сплит в RecSys даёт утечку: модель
   видит будущее поведение тех же пользователей. Делим ось времени на
   train / val / test.

2. НЕГАТИВЫ СЭМПЛИРУЮТСЯ ИЗ РАСПРЕДЕЛЕНИЯ, ПОСТРОЕННОГО ТОЛЬКО НА TRAIN.
   Если брать популярность на всём логе, в негативы просачивается
   информация о будущем — редкая, но реальная утечка, которую почти никогда
   не проверяют.

3. НАБОР КАНДИДАТОВ ОДИН И ТОТ ЖЕ ДЛЯ ВСЕХ МЕТОДОВ. Иначе сравниваются не
   модели, а ретриверы. Кандидаты строятся один раз и переиспользуются.

4. ГРУППА ДЛЯ РАНЖИРУЮЩИХ МЕТРИК — одно целевое событие: 1 позитив и K
   негативов. NDCG@k и Recall@k считаются внутри группы.

5. ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ БУТСТРЭПОМ. Разница в 0.002 AUC без интервала
   не значит ничего, а именно такими разницами обычно и отчитываются.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np

from .log import UNKNOWN, EventLog, QuerySet


@numba.njit(cache=True, nogil=True)
def _asof_attr_lookup(q_item, q_t, offsets, idx, ev_t, ev_val, out):
    """Значение атрибута айтема, известное НА МОМЕНТ запроса.

    Для каждой пары (айтем, t) берётся атрибут последнего события ЭТОГО
    айтема строго раньше t. Внутри слайса айтема события уже упорядочены
    по времени, поэтому достаточно бинарного поиска.
    """
    for r in range(q_item.shape[0]):
        it = q_item[r]
        if it < 0:
            out[r] = -1
            continue
        lo, hi = offsets[it], offsets[it + 1]
        t = q_t[r]
        a, b = lo, hi
        while a < b:
            m = (a + b) // 2
            if ev_t[idx[m]] < t:
                a = m + 1
            else:
                b = m
        out[r] = ev_val[idx[a - 1]] if a > lo else -1


@numba.njit(cache=True, nogil=True)
def _fill_csr(item, cursor, idx):
    """Заполнение CSR. В JIT, а не в Python: цикл идёт по всем событиям,
    и на логе в 10 млн строк чистый Python добавляет десятки секунд."""
    for i in range(item.shape[0]):     # лог отсортирован по времени ->
        k = item[i]                    # внутри айтема порядок сохраняется
        idx[cursor[k]] = i
        cursor[k] += 1


def _item_time_csr(log: EventLog):
    """CSR-раскладка событий по айтемам с сохранением временного порядка."""
    counts = np.zeros(log.n_items + 1, dtype=np.int64)
    np.add.at(counts, log.item.astype(np.int64) + 1, 1)
    offsets = np.cumsum(counts)
    cursor = np.ascontiguousarray(offsets[:-1].copy())
    idx = np.empty(int(offsets[-1]), dtype=np.int64)
    _fill_csr(log.item, cursor, idx)
    return offsets, idx


@dataclass
class SplitConfig:
    """Границы временного сплита в квантилях событий."""

    train_frac: float = 0.70
    val_frac: float = 0.15          # test получает остаток

    def boundaries(self, log: EventLog) -> tuple[int, int]:
        t1 = log.time_quantile(self.train_frac)
        t2 = log.time_quantile(self.train_frac + self.val_frac)
        return int(t1), int(t2)


@dataclass
class TaskConfig:
    """Постановка задачи предсказания.

    target_etype — тип события, который считается позитивом (обычно покупка).
    n_negatives  — сколько негативов на один позитив в группе.
    neg_power    — показатель степени для сэмплирования негативов из
                   популярности: 0 = равномерно, 1 = пропорционально
                   частоте. 0.75 — стандартный компромисс (как в word2vec):
                   равномерные негативы слишком легки, пропорциональные
                   слишком сильно давят на хвост.
    """

    target_etype: int = 2
    n_negatives: int = 49
    neg_power: float = 0.75
    max_positives: int | None = None
    #: Источник негативов. "popularity" — продакшен-постановка (негативы из
    #: каталога). "user_history" — ДИАГНОСТИЧЕСКИЙ режим: негативы берутся
    #: из прошлой истории самого пользователя. Нужен, чтобы вопрос стал
    #: «какой из уже виденных айтемов он купит», — только в такой постановке
    #: полупериод повтора вообще идентифицируем. При негативах из
    #: популярности 97% строк имеют нулевую историю user-item, и признак
    #: различает лишь «видел / не видел», что от полупериода не зависит.
    #: "mixed" — ОСНОВНОЙ протокол: имитация настоящего ретривера.
    #: Кандидаты собираются из трёх источников так же, как их собирает
    #: продакшен-ретривер: история пользователя, айтемы той же категории,
    #: популярное. Причина — замерено, что при негативах только по
    #: популярности задача вырождается: один бинарный признак «видел ли
    #: юзер этот айтем» даёт AUC 0.798 при NDCG@10 0.546, потому что лишь
    #: 2% негативов вообще встречались пользователю. На таком протоколе все
    #: разумные наборы признаков упираются в один потолок, и сравнивать
    #: нечего. При негативах из истории тот же признак даёт AUC 0.404,
    #: то есть ярлык исчезает полностью.
    neg_source: str = "popularity"
    #: доли источников для neg_source="mixed": история / категория / популярность
    mix_history: float = 0.4
    mix_category: float = 0.3
    seed: int = 0


def build_queries(
    log: EventLog,
    t_start: int,
    t_end: int,
    task: TaskConfig,
    train_t_end: int,
) -> QuerySet:
    """Построить кандидатов на окне [t_start, t_end).

    train_t_end — граница train. Распределение негативов строится ТОЛЬКО по
    событиям до этой границы, независимо от того, какое окно оценивается.
    Это убирает утечку через выбор негативов.
    """
    rng = np.random.default_rng(task.seed)

    # распределение негативов — только по train
    train_mask = log.t < train_t_end
    counts = np.bincount(log.item[train_mask], minlength=log.n_items).astype(np.float64)
    probs = counts ** task.neg_power
    if probs.sum() <= 0:
        probs = np.ones_like(probs)
    probs /= probs.sum()
    sampleable = np.flatnonzero(probs > 0)
    p_sampleable = probs[sampleable] / probs[sampleable].sum()

    pos_idx = np.flatnonzero(
        (log.t >= t_start) & (log.t < t_end) & (log.etype == task.target_etype)
    )
    if task.max_positives is not None and len(pos_idx) > task.max_positives:
        pos_idx = np.sort(rng.choice(pos_idx, task.max_positives, replace=False))
    if len(pos_idx) == 0:
        raise ValueError(f"нет позитивов типа {task.target_etype} на [{t_start}, {t_end})")

    n_pos = len(pos_idx)
    k = task.n_negatives
    group_size = k + 1

    if task.neg_source == "popularity":
        neg_items = rng.choice(sampleable, size=(n_pos, k), p=p_sampleable).astype(np.int32)
    elif task.neg_source == "user_history":
        neg_items = _negatives_from_history(log, pos_idx, k, sampleable, p_sampleable, rng)
    elif task.neg_source == "mixed":
        neg_items = _negatives_mixed(log, pos_idx, k, sampleable, p_sampleable,
                                     task, train_t_end, rng)
    else:
        raise ValueError(f"неизвестный neg_source: {task.neg_source}")

    t_q = np.repeat(log.t[pos_idx], group_size)
    u_q = np.repeat(log.user[pos_idx], group_size)
    g_q = np.repeat(np.arange(n_pos, dtype=np.int64), group_size)

    i_q = np.empty(n_pos * group_size, dtype=np.int32)
    y_q = np.zeros(n_pos * group_size, dtype=np.int8)
    i_q[::group_size] = log.item[pos_idx]
    y_q[::group_size] = 1
    flat = i_q.reshape(n_pos, group_size)
    flat[:, 1:] = neg_items
    i_q = flat.ravel()

    # Атрибуты айтема-кандидата разрешаются AS OF момента запроса.
    #
    # Наивный вариант «айтем -> последнее известное значение» корректен
    # только когда атрибуты статичны. В RetailRocket они меняются во
    # времени, и тогда кандидату в январе приписывается категория,
    # присвоенная в апреле. Это тихая утечка: она не ловится проверкой
    # агрегатов, потому что происходит в джойне атрибутов, а не в них.
    offsets, idx = _item_time_csr(log)
    attrs = {}
    for name, col in log.attrs.items():
        out = np.empty(len(i_q), dtype=np.int32)
        _asof_attr_lookup(i_q, t_q.astype(np.int64), offsets, idx,
                          log.t, col, out)
        attrs[name] = out

    return QuerySet(t=t_q.astype(np.int64), user=u_q.astype(np.int32), item=i_q,
                    y=y_q, group=g_q, attrs=attrs)


def _negatives_from_history(log, pos_idx, k, sampleable, p_sampleable, rng):
    """Негативы из прошлой истории пользователя (строго до момента позитива).

    Если истории не хватает, добор идёт из популярности, чтобы размер
    группы оставался постоянным.
    """
    order = np.argsort(log.user, kind="stable")
    u_sorted = log.user[order]
    bounds = np.searchsorted(u_sorted, np.arange(log.n_users + 1))
    out = np.empty((len(pos_idx), k), dtype=np.int32)

    for row, ei in enumerate(pos_idx):
        u, t_pos, i_pos = log.user[ei], log.t[ei], log.item[ei]
        sl = order[bounds[u]:bounds[u + 1]]
        hist = log.item[sl[log.t[sl] < t_pos]]
        hist = np.unique(hist)
        hist = hist[hist != i_pos]
        if len(hist) >= k:
            out[row] = rng.choice(hist, k, replace=False)
        else:
            need = k - len(hist)
            filler = rng.choice(sampleable, need, p=p_sampleable)
            out[row] = np.concatenate([hist, filler])
    return out


def _negatives_mixed(log, pos_idx, k, sampleable, p_sampleable, task,
                     train_t_end, rng):
    """Негативы как у настоящего ретривера: история + категория + популярность.

    Доли задаются mix_history / mix_category, остаток идёт на популярность.
    Отображение айтем -> категория строится ТОЛЬКО по train-периоду: иначе
    через выбор негативов просачивается информация о будущем.
    """
    n_hist = int(round(k * task.mix_history))
    n_cat = int(round(k * task.mix_category))
    n_pop = k - n_hist - n_cat

    # айтем -> категория, по train-периоду
    cat_col = log.attrs.get("cat")
    item_cat = np.full(log.n_items, -1, dtype=np.int32)
    if cat_col is not None:
        tm = (log.t < train_t_end) & (cat_col >= 0)
        item_cat[log.item[tm]] = cat_col[tm]
    by_cat: dict[int, np.ndarray] = {}
    if cat_col is not None:
        order_c = np.argsort(item_cat, kind="stable")
        sorted_c = item_cat[order_c]
        for c in np.unique(sorted_c[sorted_c >= 0]):
            lo, hi = np.searchsorted(sorted_c, [c, c + 1])
            by_cat[int(c)] = order_c[lo:hi].astype(np.int32)

    order = np.argsort(log.user, kind="stable")
    u_sorted = log.user[order]
    bounds = np.searchsorted(u_sorted, np.arange(log.n_users + 1))

    out = np.empty((len(pos_idx), k), dtype=np.int32)
    for row, ei in enumerate(pos_idx):
        u, t_pos, i_pos = log.user[ei], log.t[ei], log.item[ei]
        picked: list[np.ndarray] = []

        # 1) из истории пользователя (строго до момента позитива)
        sl = order[bounds[u]:bounds[u + 1]]
        hist = np.unique(log.item[sl[log.t[sl] < t_pos]])
        hist = hist[hist != i_pos]
        take = min(n_hist, len(hist))
        if take:
            picked.append(rng.choice(hist, take, replace=False))

        # 2) из той же категории, что и позитив
        c = int(item_cat[i_pos]) if item_cat[i_pos] >= 0 else -1
        pool = by_cat.get(c)
        if pool is not None and len(pool) > 1:
            cand = pool[pool != i_pos]
            take_c = min(n_cat, len(cand))
            if take_c:
                picked.append(rng.choice(cand, take_c, replace=False))

        # 3) остаток — популярность
        have = sum(len(x) for x in picked)
        need = k - have
        if need > 0:
            picked.append(rng.choice(sampleable, need, p=p_sampleable))
        out[row] = np.concatenate(picked)[:k]
    return out


# --------------------------------------------------------------------------
# метрики
# --------------------------------------------------------------------------

def auc(y: np.ndarray, score: np.ndarray) -> float:
    """ROC AUC через ранги (эквивалентно U-статистике Манна-Уитни)."""
    y = np.asarray(y, dtype=np.int8)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # средние ранги для совпадающих значений
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def logloss(y: np.ndarray, p: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _group_bounds(group: np.ndarray) -> np.ndarray:
    """Границы групп. Требует, чтобы группы шли подряд (так строит build_queries)."""
    change = np.flatnonzero(np.diff(group)) + 1
    return np.concatenate([[0], change, [len(group)]])


def ranking_metrics(
    y: np.ndarray, score: np.ndarray, group: np.ndarray, ks=(1, 3, 5, 10)
) -> dict[str, float]:
    """NDCG@k и Recall@k внутри групп.

    В группе ровно один позитив, поэтому NDCG@k = 1/log2(rank+1) при rank<=k,
    а Recall@k совпадает с HitRate@k.
    """
    bounds = _group_bounds(group)
    max_k = max(ks)
    ranks = np.empty(len(bounds) - 1, dtype=np.float64)
    for gi in range(len(bounds) - 1):
        lo, hi = bounds[gi], bounds[gi + 1]
        s, yy = score[lo:hi], y[lo:hi]
        pos = np.flatnonzero(yy == 1)
        if len(pos) != 1:
            ranks[gi] = np.nan
            continue
        # ранг позитива: сколько негативов строго выше + половина равных
        sp = s[pos[0]]
        higher = np.count_nonzero(s > sp)
        ties = np.count_nonzero(s == sp) - 1
        ranks[gi] = higher + ties / 2.0 + 1.0

    valid = ~np.isnan(ranks)
    r = ranks[valid]
    out: dict[str, float] = {"mrr": float(np.mean(1.0 / r))}
    for k in ks:
        hit = r <= k
        out[f"recall@{k}"] = float(hit.mean())
        out[f"ndcg@{k}"] = float(np.mean(np.where(hit, 1.0 / np.log2(r + 1.0), 0.0)))
    return out


def all_metrics(y, score, group, proba=None, ks=(1, 3, 5, 10)) -> dict[str, float]:
    m = {"auc": auc(y, score)}
    if proba is not None:
        m["logloss"] = logloss(y, proba)
    m.update(ranking_metrics(y, score, group, ks))
    return m


def bootstrap_ci(
    y, score, group, metric: str = "auc", n_boot: int = 200, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Бутстрэп по ГРУППАМ (не по строкам) — строки внутри группы зависимы.

    Возвращает (точечная оценка, нижняя граница, верхняя граница).
    """
    rng = np.random.default_rng(seed)
    bounds = _group_bounds(group)
    n_groups = len(bounds) - 1
    slices = [(bounds[i], bounds[i + 1]) for i in range(n_groups)]

    def compute(idx) -> float:
        yy = np.concatenate([y[lo:hi] for lo, hi in (slices[i] for i in idx)])
        ss = np.concatenate([score[lo:hi] for lo, hi in (slices[i] for i in idx)])
        gg = np.concatenate([np.full(hi - lo, j) for j, (lo, hi) in
                             enumerate(slices[i] for i in idx)])
        if metric == "auc":
            return auc(yy, ss)
        return ranking_metrics(yy, ss, gg)[metric]

    point = compute(np.arange(n_groups))
    vals = np.array([compute(rng.integers(0, n_groups, n_groups)) for _ in range(n_boot)])
    lo = float(np.quantile(vals, alpha / 2))
    hi = float(np.quantile(vals, 1 - alpha / 2))
    return float(point), lo, hi
