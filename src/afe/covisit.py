"""
Мета-путь длины 3: user -> item' -> item. Совстречаемость айтемов.

Задание отдельно спрашивает про графы и link prediction. В грамматике
dsl.py мета-пути ограничены арностью 2 (user x атрибут айтема), и путь
«юзер -> айтемы, которые он трогал -> айтемы, встречающиеся вместе с ними»
туда не входит. Это была самая заметная дыра в работе, и здесь она
закрывается.

ПОЧЕМУ БАТЧЕМ, А НЕ В СТРИМЕ. Точная матрица совстречаемости на
RetailRocket — это 36 184^2 = 1.3 миллиарда пар, в памяти не держится.
Первым планом был затухающий count-min sketch: фиксированная память,
известная граница ошибки, обновление в стриме. От него отказались
сознательно. В продакшене матрицу co-visitation пересчитывают батчем раз
в сутки и кладут как справочник — реального времени тут не требуется, а
батчевый вариант проще, точнее и честнее как инженерное решение.

ДИСЦИПЛИНА ОТСУТСТВИЯ УТЕЧКИ. Матрица строится СТРОГО по событиям до
границы train и дальше является фиксированным артефактом: для val и test
она не содержит ни одного события из их окон. История пользователя на
запросе читается тем же бинарным поиском по CSR, что и в hand.py, с
верхней границей строго t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np

from .log import EventLog, QuerySet

SEC_PER_DAY = 86400.0


@dataclass
class CovisitConfig:
    """Параметры построения матрицы совстречаемости.

    max_events_per_user и position_window ограничивают стоимость: без них
    число пар растёт как сумма квадратов длин историй, и тяжёлые
    пользователи дают десятки миллионов пар на пустом месте.
    """

    top_k: int = 50                  # соседей на айтем
    max_events_per_user: int = 30    # сколько последних событий юзера брать
    position_window: int = 5         # с каким числом соседей по позиции спаривать
    time_decay_days: float = 7.0     # затухание веса пары по разнице во времени
    #: веса типов событий: покупка — более сильный сигнал совстречаемости
    etype_weights: tuple[float, ...] = (1.0, 2.0, 3.0)
    #: сколько последних айтемов юзера смотреть при расчёте признака
    history_window: int = 20


def build_cooccurrence(
    log: EventLog, t_end: int, cfg: CovisitConfig, verbose: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Матрица совстречаемости по событиям СТРОГО до t_end.

    Возвращает (nbr_idx, nbr_w) размера (n_items, top_k): для каждого
    айтема — индексы и веса его топ-k соседей. Память: n_items * k * 8 байт,
    на RetailRocket это около 14 МиБ.
    """
    from scipy.sparse import coo_matrix

    mask = log.t < t_end
    u, i, t, e = log.user[mask], log.item[mask], log.t[mask], log.etype[mask]
    if verbose:
        print(f"  совстречаемость строится по {mask.sum():,} событиям "
              f"(строго до границы train)")

    order = np.lexsort((t, u))
    u, i, t, e = u[order], i[order], t[order], e[order]
    bounds = np.searchsorted(u, np.arange(log.n_users + 1))

    w_et = np.asarray(cfg.etype_weights, dtype=np.float64)
    rows, cols, vals = _pairs_from_histories(
        u, i, t.astype(np.float64) / SEC_PER_DAY, e, bounds,
        cfg.max_events_per_user, cfg.position_window,
        cfg.time_decay_days, w_et, log.n_users)

    if verbose:
        print(f"  пар совстречаемости: {len(rows):,}")

    m = coo_matrix((vals, (rows, cols)), shape=(log.n_items, log.n_items))
    m.sum_duplicates()
    csr = m.tocsr()

    # нормировка на популярность: без неё соседями всего подряд становятся
    # просто популярные айтемы, и признак вырождается в популярность
    pop = np.asarray(csr.sum(axis=1)).ravel() + 1.0
    csr = csr.multiply(1.0 / np.sqrt(pop)[:, None]).tocsr()
    csr = csr.multiply(1.0 / np.sqrt(pop)[None, :]).tocsr()

    nbr_idx = np.full((log.n_items, cfg.top_k), -1, dtype=np.int32)
    nbr_w = np.zeros((log.n_items, cfg.top_k), dtype=np.float32)
    indptr, indices, data = csr.indptr, csr.indices, csr.data
    for it in range(log.n_items):
        lo, hi = indptr[it], indptr[it + 1]
        if hi <= lo:
            continue
        d = data[lo:hi]
        k = min(cfg.top_k, len(d))
        sel = np.argpartition(-d, k - 1)[:k]
        sel = sel[np.argsort(-d[sel])]
        nbr_idx[it, :k] = indices[lo:hi][sel]
        nbr_w[it, :k] = d[sel]
    if verbose:
        filled = (nbr_idx[:, 0] >= 0).mean()
        print(f"  у {filled:.1%} айтемов есть хотя бы один сосед")
    return nbr_idx, nbr_w


@numba.njit(cache=True, nogil=True)
def _pairs_from_histories(u, i, t_days, e, bounds, max_ev, win, decay_days,
                          w_et, n_users):
    """Пары (айтем, айтем) из историй пользователей, с весом.

    Вес пары — произведение весов типов событий, затухающее по разнице во
    времени между ними. Пары берутся в окне position_window по позиции:
    это ограничивает стоимость и соответствует смыслу «встречались рядом».
    """
    cap = 0
    for uu in range(n_users):
        n = min(bounds[uu + 1] - bounds[uu], max_ev)
        if n > 1:
            cap += n * win
    rows = np.empty(cap, dtype=np.int32)
    cols = np.empty(cap, dtype=np.int32)
    vals = np.empty(cap, dtype=np.float64)
    m = 0
    lam = np.log(2.0) / decay_days
    for uu in range(n_users):
        lo, hi = bounds[uu], bounds[uu + 1]
        if hi - lo < 2:
            continue
        if hi - lo > max_ev:
            lo = hi - max_ev          # только последние max_ev событий
        for a in range(lo, hi):
            ia, ta, wa = i[a], t_days[a], w_et[e[a]]
            top = min(a + 1 + win, hi)
            for b in range(a + 1, top):
                ib = i[b]
                if ib == ia:
                    continue
                w = wa * w_et[e[b]] * np.exp(-lam * (t_days[b] - ta))
                if m + 1 >= cap:
                    break
                rows[m] = ia
                cols[m] = ib
                vals[m] = w
                m += 1
                rows[m] = ib          # симметрично
                cols[m] = ia
                vals[m] = w
                m += 1
    return rows[:m], cols[:m], vals[:m]


@numba.njit(cache=True, nogil=True)
def _covisit_features(
    q_user, q_item, q_t, ev_item, ev_t, ev_e, offsets, idx,
    nbr_idx, nbr_w, w_et, lambdas, hist_win, out,
):
    """Признаки совстречаемости для каждой строки-запроса.

    Для запроса (u, i, t) берутся последние hist_win айтемов пользователя
    СТРОГО раньше t, и для каждого проверяется, входит ли кандидат i в его
    топ-k соседей. Верхняя граница по времени ставится бинарным поиском —
    та же дисциплина, что в hand.py.

    Выходные признаки (на каждый полупериод затухания):
      sum   суммарный вес совстречаемости
      max   максимальный вес
      hits  сколько айтемов истории дали попадание
    """
    n_q = q_user.shape[0]
    n_lam = lambdas.shape[0]
    top_k = nbr_idx.shape[1]
    for r in range(n_q):
        uu = q_user[r]
        cand = q_item[r]
        t = q_t[r]
        if uu < 0 or cand < 0:
            continue
        lo, hi = offsets[uu], offsets[uu + 1]

        # верхняя граница: первое событие пользователя с временем >= t
        a, b = lo, hi
        while a < b:
            mid = (a + b) // 2
            if ev_t[idx[mid]] < t:
                a = mid + 1
            else:
                b = mid
        upper = a
        start = upper - hist_win
        if start < lo:
            start = lo

        for li in range(n_lam):
            s = 0.0
            mx = 0.0
            hits = 0.0
            for j in range(start, upper):
                ev = idx[j]
                src = ev_item[ev]
                if src == cand:
                    continue           # прямое повторение — это уже другой признак
                w_hist = w_et[ev_e[ev]] * np.exp(-lambdas[li] * (t - ev_t[ev]))
                for k in range(top_k):
                    nb = nbr_idx[src, k]
                    if nb < 0:
                        break
                    if nb == cand:
                        w = nbr_w[src, k] * w_hist
                        s += w
                        if w > mx:
                            mx = w
                        hits += 1.0
                        break
            out[r, li * 3 + 0] = s
            out[r, li * 3 + 1] = mx
            out[r, li * 3 + 2] = hits


def covisit_features(
    log: EventLog, queries: QuerySet, nbr_idx: np.ndarray, nbr_w: np.ndarray,
    cfg: CovisitConfig, halflives_days=(1.0, 7.0, 30.0), verbose: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Матрица признаков совстречаемости в порядке строк QuerySet."""
    import time as _time

    order = np.lexsort((log.t, log.user))
    offsets = np.searchsorted(log.user[order], np.arange(log.n_users + 1))
    idx = order.astype(np.int64)

    lam = np.array([np.log(2.0) / h for h in halflives_days], dtype=np.float64)
    out = np.zeros((len(queries), len(lam) * 3), dtype=np.float32)

    t0 = _time.perf_counter()
    _covisit_features(
        queries.user, queries.item,
        queries.t.astype(np.float64) / SEC_PER_DAY,
        log.item, log.t.astype(np.float64) / SEC_PER_DAY, log.etype,
        offsets, idx, nbr_idx, nbr_w,
        np.asarray(cfg.etype_weights, dtype=np.float64), lam,
        cfg.history_window, out)
    secs = _time.perf_counter() - t0

    names = [f"covisit_{s}[{h:g}d]" for h in halflives_days
             for s in ("sum", "max", "hits")]
    if verbose:
        nz = (out[:, 0] > 0).mean()
        print(f"  признаков совстречаемости: {out.shape[1]} за {secs:.1f} с, "
              f"ненулевых строк {nz:.1%}")
    return out, names
