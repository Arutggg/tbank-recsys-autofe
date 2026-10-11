"""
Компилятор признаков: один проход по времени, O(1) на событие, без утечки.

Устройство прохода
------------------
Лог и запросы склеены в один неубывающий по времени поток (`log.build_stream`).

  строка-СОБЫТИЕ  -> состояние обновляется;
  строка-ЗАПРОС   -> состояние читается, НЕ обновляется и НЕ мутируется.

При равных timestamp запрос стоит раньше события, поэтому признак в момент t
зависит только от событий с t' < t. Утечка невозможна не потому, что мы
аккуратно написали WHERE, а потому что в этот момент нужных данных ещё нет
в состоянии. Это свойство порядка операций.

Ленивое затухание
-----------------
Затухать все ключи на каждом событии нельзя — это O(число ключей) на шаг.
Поэтому для каждой пары (ключ, тип события) хранится момент последнего
касания t_touch, и затухание применяется только к тем слотам, которые мы
прямо сейчас читаем или пишем. Затухание мультипликативно и композируется
(dt1, затем dt2 == dt1+dt2), поэтому ленивое применение точно эквивалентно
немедленному.

Асимметрия, которая даёт скорость: строка-событие трогает слоты только
своего типа события (L слотов), а строка-запрос читает все (n_e * L), но
при чтении ничего не записывает обратно.

Слот «любой тип события» не хранится: моменты аддитивны по типам событий,
поэтому A_any = sum_e A_e считается на чтении точно, без лишней памяти.

Единицы времени
---------------
Внутри движка время — ДНИ (float64), а не секунды. Моменты второго порядка
содержат квадрат возраста; в секундах это ~1e14 и при хранении во float32
дисперсия возрастов теряется на вычитании (C/A - (B/A)^2). В днях величины
порядка 1e4..1e5 и float32 достаточно.
"""

from __future__ import annotations

import numpy as np
import numba

from .dsl import ANY_ETYPE, Derived, FeatureSet, Primitive
from .kernels import halflives_to_lambdas
from .log import Stream

SEC_PER_DAY = 86400.0

#: коды статистик для JIT-ядра
KIND_DCOUNT = 0
KIND_DAGE_MEAN = 1
KIND_DAGE_STD = 2
KIND_RECENCY = 3
KIND_AGE = 4

_KIND_OF = {
    "dcount": KIND_DCOUNT,
    "dage_mean": KIND_DAGE_MEAN,
    "dage_std": KIND_DAGE_STD,
    "recency": KIND_RECENCY,
    "age": KIND_AGE,
}


@numba.njit(cache=True, nogil=True)
def _scan_keyspec(
    t, is_query, qpos, etype, key_id,          # поток
    lam, n_e,                                  # ядра и число типов событий
    mom, t_touch, t_last, t_first,             # состояние
    out, out_col, out_le, out_ll, out_kind,    # выход
):
    """Однопроходное сканирование для ОДНОГО ключа агрегации.

    mom[k, e, l, m] — момент порядка m для ключа k, типа события e, ядра l.
    t_touch[k, e]   — время последнего применения затухания к этим слотам.
    t_last / t_first[k, e] — для recency / age (ядро не нужно).
    Значение -1.0 в t_last/t_first означает «событий не было» -> на выходе NaN.
    """
    n_stream = t.shape[0]
    n_l = lam.shape[0]
    n_out = out_col.shape[0]

    for i in range(n_stream):
        k = key_id[i]
        if k < 0:              # неизвестное значение категориального признака
            continue
        ti = t[i]

        if is_query[i]:
            # ---- ЧТЕНИЕ: состояние не мутируется ----
            row = qpos[i]
            for j in range(n_out):
                kind = out_kind[j]
                le = out_le[j]

                if kind == KIND_RECENCY or kind == KIND_AGE:
                    src = t_last if kind == KIND_RECENCY else t_first
                    if le >= 0:
                        tv = src[k, le]
                        val = np.nan if tv < 0.0 else ti - tv
                    else:
                        # «любой тип»: recency -> самое свежее, age -> самое давнее
                        best = -1.0
                        for e in range(n_e):
                            tv = src[k, e]
                            if tv < 0.0:
                                continue
                            if best < 0.0:
                                best = tv
                            elif kind == KIND_RECENCY:
                                if tv > best:
                                    best = tv
                            else:
                                if tv < best:
                                    best = tv
                        val = np.nan if best < 0.0 else ti - best
                    out[row, out_col[j]] = val
                    continue

                # моментные статистики: затухаем на лету, в состояние не пишем
                ll = out_ll[j]
                lmb = lam[ll]
                A = 0.0
                B = 0.0
                C = 0.0
                if le >= 0:
                    dt = ti - t_touch[k, le]
                    if dt < 0.0:
                        dt = 0.0
                    d = np.exp(-lmb * dt)
                    a0 = mom[k, le, ll, 0]
                    b0 = mom[k, le, ll, 1]
                    c0 = mom[k, le, ll, 2]
                    A = d * a0
                    B = d * (b0 + dt * a0)
                    C = d * (c0 + 2.0 * dt * b0 + dt * dt * a0)
                else:
                    for e in range(n_e):          # аддитивность по типам событий
                        dt = ti - t_touch[k, e]
                        if dt < 0.0:
                            dt = 0.0
                        d = np.exp(-lmb * dt)
                        a0 = mom[k, e, ll, 0]
                        b0 = mom[k, e, ll, 1]
                        c0 = mom[k, e, ll, 2]
                        A += d * a0
                        B += d * (b0 + dt * a0)
                        C += d * (c0 + 2.0 * dt * b0 + dt * dt * a0)

                if kind == KIND_DCOUNT:
                    out[row, out_col[j]] = A
                elif A <= 0.0:
                    out[row, out_col[j]] = np.nan
                elif kind == KIND_DAGE_MEAN:
                    out[row, out_col[j]] = B / A
                else:                              # KIND_DAGE_STD
                    m1 = B / A
                    var = C / A - m1 * m1
                    out[row, out_col[j]] = np.sqrt(var) if var > 0.0 else 0.0
        else:
            # ---- ОБНОВЛЕНИЕ: трогаем только слоты своего типа события ----
            le = etype[i]
            for ll in range(n_l):
                dt = ti - t_touch[k, le]
                if dt < 0.0:
                    dt = 0.0
                d = np.exp(-lam[ll] * dt)
                a0 = mom[k, le, ll, 0]
                b0 = mom[k, le, ll, 1]
                c0 = mom[k, le, ll, 2]
                # затухание, затем приход события возраста 0: A += 1, B и C без изменений
                mom[k, le, ll, 0] = d * a0 + 1.0
                mom[k, le, ll, 1] = d * (b0 + dt * a0)
                mom[k, le, ll, 2] = d * (c0 + 2.0 * dt * b0 + dt * dt * a0)
            t_touch[k, le] = ti
            if t_first[k, le] < 0.0:
                t_first[k, le] = ti
            t_last[k, le] = ti


def _factorize_key(stream: Stream, key: tuple[str, ...]) -> tuple[np.ndarray, int]:
    """Плотные целые id для кортежа столбцов.

    Движок индексирует состояние плоскими массивами, поэтому ключ-кортеж
    нужно свернуть в один int32. UNKNOWN (-1) в любом компоненте делает
    ключ невалидным (-1) — такая строка просто пропускается.
    """
    n = stream.size
    if not key:
        return np.zeros(n, dtype=np.int32), 1

    packed = np.zeros(n, dtype=np.int64)
    valid = np.ones(n, dtype=bool)
    for col in key:
        v = stream.columns[col]
        card = stream.cardinality[col]
        valid &= v >= 0
        packed = packed * np.int64(card) + v.astype(np.int64)

    packed[~valid] = -1
    codes, uniq = pd_factorize(packed)
    codes[~valid] = -1
    return codes.astype(np.int32), int(uniq)


def pd_factorize(packed: np.ndarray) -> tuple[np.ndarray, int]:
    """Факторизация int64 -> плотные коды. Выделена для тестируемости."""
    uniq, codes = np.unique(packed, return_inverse=True)
    return codes.astype(np.int64), len(uniq)


class FeatureCompiler:
    """Считает план признаков (FeatureSet) за один проход по потоку."""

    def __init__(
        self,
        feature_set: FeatureSet,
        n_etypes: int,
        dtype=np.float32,
        state_dtype=np.float32,
    ):
        """dtype — точность выходной матрицы признаков.

        state_dtype — точность внутреннего состояния моментов. float32 вдвое
        экономит память (а состояние для ключа user x item и есть главный
        потребитель RAM), ценой относительной погрешности ~1e-7. Для
        признаков бустинга это не имеет значения, но тесты на точное
        соответствие определению требуют float64.
        """
        self.fs = feature_set
        self.n_etypes = n_etypes
        self.dtype = dtype
        self.state_dtype = state_dtype
        self.stats: dict = {}

    # -- оценка памяти до запуска: 16 ГБ RAM требуют считать это заранее --
    def estimate_state_bytes(self, stream: Stream) -> dict[tuple[str, ...], int]:
        groups = self._group_primitives()
        est = {}
        for key, items in groups.items():
            n_keys = self._key_cardinality(stream, key)
            n_l = len(items["lambdas"])
            mom = n_keys * self.n_etypes * n_l * 3 * np.dtype(self.state_dtype).itemsize
            times = n_keys * self.n_etypes * 8 * 3              # t_touch/last/first float64
            est[key] = mom + times
        return est

    def _key_cardinality(self, stream: Stream, key: tuple[str, ...]) -> int:
        """Верхняя оценка числа ключей без факторизации (произведение мощностей)."""
        if not key:
            return 1
        prod = 1
        for col in key:
            prod *= max(stream.cardinality[col], 1)
        return prod

    def _group_primitives(self) -> dict[tuple[str, ...], dict]:
        """Сгруппировать примитивы по ключу: один проход на ключ."""
        groups: dict[tuple[str, ...], dict] = {}
        for idx, p in enumerate(self.fs.primitives):
            g = groups.setdefault(p.key, {"items": [], "lambdas": []})
            g["items"].append((idx, p))
        for key, g in groups.items():
            hls = sorted({p.halflife_s for _, p in g["items"] if p.needs_kernel})
            if not hls:
                hls = [float("inf")]
            g["lambdas"] = hls
        return groups

    def transform(self, stream: Stream, verbose: bool = True) -> np.ndarray:
        """Вернуть матрицу признаков (n_queries, n_total) в порядке QuerySet."""
        import time as _time

        n_q = stream.n_queries
        n_prim = len(self.fs.primitives)
        out = np.full((n_q, self.fs.n_total), np.nan, dtype=self.dtype)

        t_days = ((stream.t - stream.t[0]) / SEC_PER_DAY).astype(np.float64)
        groups = self._group_primitives()

        peak_bytes = 0
        t0 = _time.perf_counter()
        for gi, (key, g) in enumerate(groups.items()):
            key_id, n_keys = _factorize_key(stream, key)
            lam = halflives_to_lambdas(g["lambdas"]) * SEC_PER_DAY  # 1/день
            n_l = len(lam)
            hl_pos = {hl: i for i, hl in enumerate(g["lambdas"])}

            mom = np.zeros((n_keys, self.n_etypes, n_l, 3), dtype=self.state_dtype)
            t_touch = np.zeros((n_keys, self.n_etypes), dtype=np.float64)
            t_last = np.full((n_keys, self.n_etypes), -1.0, dtype=np.float64)
            t_first = np.full((n_keys, self.n_etypes), -1.0, dtype=np.float64)
            peak_bytes = max(peak_bytes, mom.nbytes + t_touch.nbytes
                             + t_last.nbytes + t_first.nbytes)

            cols, les, lls, kinds = [], [], [], []
            for idx, p in g["items"]:
                cols.append(idx)
                les.append(p.etype if p.etype != ANY_ETYPE else -1)
                lls.append(hl_pos[p.halflife_s] if p.needs_kernel else 0)
                kinds.append(_KIND_OF[p.stat])

            _scan_keyspec(
                t_days, stream.is_query, stream.qpos, stream.etype, key_id,
                lam, self.n_etypes, mom, t_touch, t_last, t_first,
                out,
                np.array(cols, dtype=np.int32), np.array(les, dtype=np.int8),
                np.array(lls, dtype=np.int8), np.array(kinds, dtype=np.int8),
            )
            del mom, t_touch, t_last, t_first
            if verbose:
                kname = "global" if not key else "x".join(key)
                print(f"  [{gi + 1}/{len(groups)}] ключ {kname:<24} "
                      f"уникальных={n_keys:>9,}  ядер={n_l}  "
                      f"признаков={len(cols):>4}", flush=True)

        scan_s = _time.perf_counter() - t0

        t1 = _time.perf_counter()
        self._apply_derived(out, n_prim)
        derived_s = _time.perf_counter() - t1

        self.stats = {
            "scan_seconds": scan_s,
            "derived_seconds": derived_s,
            "stream_rows": stream.size,
            "n_queries": n_q,
            "rows_per_sec": stream.size / max(scan_s, 1e-9),
            "peak_state_bytes": peak_bytes,
            "n_keyspecs": len(groups),
        }
        if verbose:
            print(f"  проход: {scan_s:.1f} с  ({self.stats['rows_per_sec']:,.0f} строк/с), "
                  f"арифметика уровня 1: {derived_s:.2f} с, "
                  f"пик состояния: {peak_bytes / 2**20:,.0f} МиБ")
        return out

    def _apply_derived(self, out: np.ndarray, n_prim: int) -> None:
        """Уровень 1: арифметика над уже посчитанными столбцами. Без проходов по логу.

        Знаменатели сглажены +1 (аддитивное сглаживание): иначе отношение
        при нулевом знаменателе даёт inf, а при малом — шум, на котором
        бустинг охотно переобучается.
        """
        eps = 1.0
        for j, d in enumerate(self.fs.derived):
            col = n_prim + j
            a = out[:, d.a].astype(np.float64)
            b = out[:, d.b].astype(np.float64)
            if d.op == "lift":
                c = out[:, d.c].astype(np.float64)
                dd = out[:, d.d].astype(np.float64)
                val = (a * np.maximum(dd, eps)) / ((b + eps) * (c + eps))
            else:
                val = a / (b + eps)
            out[:, col] = val.astype(self.dtype)
