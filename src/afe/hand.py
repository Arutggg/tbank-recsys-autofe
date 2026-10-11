"""
Baseline ручных признаков: добросовестная реализация того, что пишет человек.

Это самая важная для честности часть работы. Если baseline слабый, всё
сравнение превращается в соломенное чучело, и проверять его будут первым
делом. Поэтому здесь реализован полный классический набор ровно в том виде,
в каком его пишет ML-инженер в RecSys, и с ТОЧНЫМИ жёсткими окнами, а не с
приближением.

Состав набора (по структуре из задания):
  user       — счётчики по типам событий в окнах, число уникальных айтемов,
               давность последнего события, возраст аккаунта
  item       — счётчики по типам событий в окнах, число уникальных юзеров,
               давность, возраст, конверсии view->cart->purchase
  user-item  — ПРЯМЫЕ признаки: счётчики в окнах, давность последнего
               взаимодействия именно с этим айтемом
  user-cat   — КОСВЕННЫЕ признаки: активность юзера по категории айтема,
               доля категории в активности юзера
  отношения  — конверсии и доли, то есть «ручные» аналоги уровня 1

Окна взяты те, которые в индустрии выбирают по умолчанию: 1, 7, 30, 365
дней. Именно этот выбор задание и ставит под сомнение («а почему 7, а не
10?»), и именно он сравнивается с полной сеткой масштабов.

Про корректность. Окна считаются ТОЧНО: для каждого ключа события
раскладываются в CSR-структуру по времени, и счётчик в окне берётся двумя
бинарными поисками с явной верхней границей t. Поэтому отличие от
компилятора — только в семействе признаков, но не в дисциплине отсутствия
утечки: оба считают строго по событиям с t' < t.
"""

from __future__ import annotations

import numpy as np
import numba

from .log import Stream
from .kernels import DAY

SEC_PER_DAY = 86400.0

#: Окна, которые человек выбирает по умолчанию.
HUMAN_WINDOWS_DAYS: tuple[float, ...] = (1.0, 7.0, 30.0, 365.0)


@numba.njit(cache=True, nogil=True)
def _build_csr(key_id, is_query, n_keys, n_rows):
    """Разложить события (не запросы) по ключам в CSR: offsets + порядок строк.

    Поток отсортирован по времени, поэтому внутри слайса ключа времена
    автоматически неубывающие — это и позволяет искать окно бинарным поиском.
    """
    counts = np.zeros(n_keys + 1, dtype=np.int64)
    for i in range(n_rows):
        if is_query[i]:
            continue
        k = key_id[i]
        if k >= 0:
            counts[k + 1] += 1
    for k in range(n_keys):
        counts[k + 1] += counts[k]
    offsets = counts.copy()
    cursor = counts[:-1].copy()
    idx = np.empty(offsets[n_keys], dtype=np.int64)
    for i in range(n_rows):
        if is_query[i]:
            continue
        k = key_id[i]
        if k >= 0:
            idx[cursor[k]] = i
            cursor[k] += 1
    return offsets, idx


@numba.njit(cache=True, nogil=True)
def _window_counts(
    t, is_query, qpos, etype, key_id, offsets, idx, ev_t, ev_e,
    windows, n_etypes, out, out_col_base,
):
    """Точные счётчики в окнах + давность и возраст, двумя бинарными поисками.

    Для каждой строки-запроса в момент t и каждого окна W берётся число
    событий ключа в [t - W, t). Верхняя граница строго t, поэтому событие
    в сам момент t в признак не попадает — та же дисциплина, что в
    компиляторе.
    """
    n_rows = t.shape[0]
    n_w = windows.shape[0]
    for i in range(n_rows):
        if not is_query[i]:
            continue
        k = key_id[i]
        row = qpos[i]
        if k < 0:
            continue
        lo, hi = offsets[k], offsets[k + 1]
        ti = t[i]

        # верхняя граница: первое событие с временем >= ti
        a, b = lo, hi
        while a < b:
            m = (a + b) // 2
            if ev_t[idx[m]] < ti:
                a = m + 1
            else:
                b = m
        upper = a

        col = out_col_base
        # --- счётчики по типам событий и окнам ---
        for e in range(n_etypes):
            for wi in range(n_w):
                t_lo = ti - windows[wi]
                a2, b2 = lo, upper
                while a2 < b2:
                    m = (a2 + b2) // 2
                    if ev_t[idx[m]] < t_lo:
                        a2 = m + 1
                    else:
                        b2 = m
                c = 0.0
                for j in range(a2, upper):
                    if ev_e[idx[j]] == e:
                        c += 1.0
                out[row, col] = c
                col += 1
        # --- счётчики по ЛЮБОМУ типу события ---
        for wi in range(n_w):
            t_lo = ti - windows[wi]
            a2, b2 = lo, upper
            while a2 < b2:
                m = (a2 + b2) // 2
                if ev_t[idx[m]] < t_lo:
                    a2 = m + 1
                else:
                    b2 = m
            out[row, col] = float(upper - a2)
            col += 1
        # --- давность последнего события и возраст первого ---
        if upper > lo:
            out[row, col] = ti - ev_t[idx[upper - 1]]
            out[row, col + 1] = ti - ev_t[idx[lo]]
        else:
            out[row, col] = np.nan
            out[row, col + 1] = np.nan


@numba.njit(cache=True, nogil=True)
def _nunique_partner(
    t, is_query, qpos, key_id, partner, offsets, idx, ev_t, ev_partner,
    out, out_col,
):
    """Число уникальных партнёров ключа до момента t (айтемов у юзера и наоборот).

    Считается честно, перебором слайса до верхней границы. Это дорого
    (O(событий ключа) на запрос) — и это ровно та причина, по которой такие
    признаки в проде либо не считают, либо считают приближённо. Для
    baseline берём точный вариант, чтобы не ослаблять его искусственно.
    """
    n_rows = t.shape[0]
    for i in range(n_rows):
        if not is_query[i]:
            continue
        k = key_id[i]
        row = qpos[i]
        if k < 0:
            continue
        lo, hi = offsets[k], offsets[k + 1]
        ti = t[i]
        a, b = lo, hi
        while a < b:
            m = (a + b) // 2
            if ev_t[idx[m]] < ti:
                a = m + 1
            else:
                b = m
        upper = a
        if upper <= lo:
            out[row, out_col] = np.nan
            continue
        seen = 0
        # подсчёт уникальных через сортировку слайса партнёров
        vals = np.empty(upper - lo, dtype=np.int32)
        for j in range(lo, upper):
            vals[j - lo] = ev_partner[idx[j]]
        vals.sort()
        prev = -1
        for v in vals:
            if v != prev:
                seen += 1
                prev = v
        out[row, out_col] = float(seen)


class HandFeatures:
    """Классический ручной набор признаков с точными жёсткими окнами."""

    def __init__(
        self,
        cat_col: str = "cat",
        windows_days: tuple[float, ...] = HUMAN_WINDOWS_DAYS,
        n_etypes: int = 3,
        with_nunique: bool = True,
    ):
        self.cat_col = cat_col
        self.windows = np.array(windows_days, dtype=np.float64)
        self.n_etypes = n_etypes
        self.with_nunique = with_nunique
        self.feature_names_: list[str] = []
        self.stats: dict = {}

    # ключи, по которым человек считает агрегаты
    def _keyspecs(self) -> list[tuple[str, ...]]:
        return [("user",), ("item",), ("user", "item"), ("user", self.cat_col)]

    def _block_names(self, key: tuple[str, ...]) -> list[str]:
        kn = "x".join(key)
        names = []
        for e in range(self.n_etypes):
            for w in self.windows:
                names.append(f"cnt[{kn}|e{e}|{w:g}d]")
        for w in self.windows:
            names.append(f"cnt[{kn}|any|{w:g}d]")
        names += [f"recency[{kn}]", f"age[{kn}]"]
        return names

    def transform(self, stream: Stream, verbose: bool = True) -> np.ndarray:
        import time as _time
        from .engine import _factorize_key

        t_days = ((stream.t - stream.t[0]) / SEC_PER_DAY).astype(np.float64)
        ev_e = stream.etype.astype(np.int8)
        n_q = stream.n_queries
        n_w = len(self.windows)
        block = self.n_etypes * n_w + n_w + 2

        keyspecs = self._keyspecs()
        n_extra = 2 if self.with_nunique else 0        # uniq items у юзера, uniq юзеров у айтема
        total = block * len(keyspecs) + n_extra
        out = np.full((n_q, total), np.nan, dtype=np.float32)

        names: list[str] = []
        t0 = _time.perf_counter()
        for gi, key in enumerate(keyspecs):
            key_id, n_keys = _factorize_key(stream, key)
            offsets, idx = _build_csr(key_id, stream.is_query, n_keys, stream.size)
            _window_counts(
                t_days, stream.is_query, stream.qpos, ev_e, key_id,
                offsets, idx, t_days, ev_e, self.windows, self.n_etypes,
                out, gi * block,
            )
            names += self._block_names(key)
            if verbose:
                print(f"  [{gi + 1}/{len(keyspecs)}] ручной блок {'x'.join(key):<16} "
                      f"ключей={n_keys:>9,}  признаков={block}", flush=True)

        if self.with_nunique:
            col = block * len(keyspecs)
            for key, partner_col, nm in (
                (("user",), "item", "nuniq_items[user]"),
                (("item",), "user", "nuniq_users[item]"),
            ):
                key_id, n_keys = _factorize_key(stream, key)
                offsets, idx = _build_csr(key_id, stream.is_query, n_keys, stream.size)
                _nunique_partner(
                    t_days, stream.is_query, stream.qpos, key_id,
                    stream.columns[partner_col], offsets, idx, t_days,
                    stream.columns[partner_col], out, col,
                )
                names.append(nm)
                col += 1

        base_s = _time.perf_counter() - t0

        # --- отношения: ручные аналоги конверсий и долей ---
        out, names = self._add_ratios(out, names)
        self.feature_names_ = names
        self.stats = {"seconds": _time.perf_counter() - t0,
                      "base_seconds": base_s, "n_features": len(names)}
        if verbose:
            print(f"  ручной набор: {len(names)} признаков за "
                  f"{self.stats['seconds']:.1f} с")
        return out

    def _add_ratios(self, out: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
        """Конверсии и доли — то, что человек считает поверх счётчиков."""
        pos = {n: i for i, n in enumerate(names)}
        extra, extra_names = [], []
        n_w = len(self.windows)

        def log_ratio(a_col: int, b_col: int) -> np.ndarray:
            """Лог-отношение, как в компиляторе: формы признаков должны
            совпадать, иначе сравнение смешивает эффект семейства признаков
            с эффектом способа кодирования отношений."""
            la = np.log1p(np.maximum(out[:, a_col], 0.0))
            lb = np.log1p(np.maximum(out[:, b_col], 0.0))
            return la - lb

        for key in self._keyspecs():
            kn = "x".join(key)
            for w in self.windows:
                # конверсия view -> purchase
                a = pos.get(f"cnt[{kn}|e{self.n_etypes - 1}|{w:g}d]")
                b = pos.get(f"cnt[{kn}|e0|{w:g}d]")
                if a is not None and b is not None:
                    extra.append(log_ratio(a, b))
                    extra_names.append(f"conv[{kn}|{w:g}d]")
            # доля активности: user-item / user, user-cat / user
            if len(key) == 2:
                for w in self.windows:
                    a = pos.get(f"cnt[{kn}|any|{w:g}d]")
                    b = pos.get(f"cnt[user|any|{w:g}d]")
                    if a is not None and b is not None:
                        extra.append(log_ratio(a, b))
                        extra_names.append(f"share[{kn}|{w:g}d]")
        if extra:
            out = np.hstack([out, np.vstack(extra).T.astype(np.float32)])
            names = names + extra_names
        return out, names
