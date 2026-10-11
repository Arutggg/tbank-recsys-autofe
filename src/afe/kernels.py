"""
Базис временных ядер: почему затухание, а не окно.

Классическая ручная фича — «сколько событий за последние W дней»:

    f_W(t) = int_0^inf  1[s < W]  dN(t - s)

то есть интеграл процесса событий по ИНДИКАТОРНОМУ ядру. Человек выбирает W.
Задание прямо спрашивает: а почему 7 дней, а не 10? Ответ этого модуля —
окно вообще не та параметризация. Берём базис экспоненциальных ядер:

    g_l(t) = int_0^inf  exp(-l * s)  dN(t - s)

Три свойства, из которых всё и следует:

1. ЛИНЕЙНЫЙ ПЕРЕНОС. Если 1[s<W] ~= sum_k c_k exp(-l_k s) как функции от s,
   то f_W ~= sum_k c_k g_{l_k} для ЛЮБОЙ истории событий N. Приближение в
   пространстве ядер переносится в пространство признаков для всех
   пользователей и айтемов сразу, а не «в среднем по датасету».
   `fit_window_as_decay_mix` считает ошибку этого приближения численно.

2. O(1) НА СОБЫТИЕ И O(1) ПАМЯТИ НА КЛЮЧ. Рекуррента
        S <- S * exp(-l * dt) + 1
   обновляет счётчик за константное время, не храня сами события. Жёсткое
   окно требует либо буфера событий внутри окна, либо повторного join по
   таблице. Именно это свойство делает метод применимым к логам на
   миллиарды интеракций — ровно та претензия, которую задание предъявляет
   к трансформерам.

3. ЧЕЛОВЕК НЕ ВЫБИРАЕТ НИЧЕГО. Вместо одного W берём геометрическую сетку
   полупериодов (час .. год .. бесконечность) целиком. Сетка покрывает всё
   семейство окон, а какой масштаб релевантен — решает модель по данным.

Важная оговорка про честность утверждения: GBDT делает осевые разбиения и
не формирует линейные комбинации признаков напрямую. Поэтому корректная
формулировка — не «дерево воспроизведёт любое окно», а «информация о любом
окне присутствует в базисе» (отображение из счётчиков в оконные суммы —
фиксированное линейное). Выигрывает ли дерево на практике — вопрос
эмпирический, и он проверяется экспериментом, а не этим рассуждением.
"""

from __future__ import annotations

import numpy as np

HOUR = 3600.0
DAY = 24 * HOUR
WEEK = 7 * DAY

#: Сетка полупериодов по умолчанию. Геометрическая, от часа до года,
#: плюс inf = счётчик за всё время (классическая lifetime-фича).
DEFAULT_HALFLIVES: tuple[float, ...] = (
    1 * HOUR,
    6 * HOUR,
    1 * DAY,
    3 * DAY,
    7 * DAY,
    30 * DAY,
    180 * DAY,
    float("inf"),
)

#: Жёсткие окна, которые типично выбирает человек. Нужны только как baseline
#: для сравнения — в самом компиляторе не используются.
TYPICAL_HUMAN_WINDOWS: tuple[float, ...] = (1 * DAY, 7 * DAY, 30 * DAY, 365 * DAY)


def halflife_to_lambda(halflife_s: float) -> float:
    """Полупериод -> скорость затухания. l = ln2 / T, при T=inf даёт l=0."""
    if not np.isfinite(halflife_s):
        return 0.0
    if halflife_s <= 0:
        raise ValueError(f"полупериод должен быть > 0, получено {halflife_s}")
    return float(np.log(2.0) / halflife_s)


def halflives_to_lambdas(halflives_s) -> np.ndarray:
    return np.array([halflife_to_lambda(h) for h in halflives_s], dtype=np.float64)


def geometric_halflives(
    t_min: float = HOUR, t_max: float = 180 * DAY, k: int = 7, with_inf: bool = True
) -> tuple[float, ...]:
    """Геометрическая сетка из k полупериодов на [t_min, t_max] (+ inf)."""
    grid = list(np.geomspace(t_min, t_max, k))
    if with_inf:
        grid.append(float("inf"))
    return tuple(float(x) for x in grid)


def format_halflife(halflife_s: float) -> str:
    """Человекочитаемое имя масштаба — попадает в имена фичей и в отчёт."""
    if not np.isfinite(halflife_s):
        return "inf"
    if halflife_s < DAY:
        return f"{halflife_s / HOUR:.3g}h"
    if halflife_s < 30 * DAY:
        return f"{halflife_s / DAY:.3g}d"
    return f"{halflife_s / DAY:.0f}d"


# --------------------------------------------------------------------------
# Эксперимент 1: насколько базис затуханий покрывает семейство жёстких окон
# --------------------------------------------------------------------------

def fit_window_as_decay_mix(
    window_s: float,
    halflives_s=DEFAULT_HALFLIVES,
    s_max: float | None = None,
    n_grid: int = 4000,
    ridge: float = 1e-10,
) -> dict:
    """Приблизить индикатор окна [0, W] смесью экспонент из базиса.

    Решает  min_c || 1[s<W] - sum_k c_k exp(-l_k s) ||_2  на сетке s.
    Коэффициенты знаковые: без отрицательных слагаемых ступеньку не построить.

    Возвращает коэффициенты и ошибки. rel_l2 — относительная L2-ошибка,
    sup — поточечный максимум |невязки|; именно sup мажорирует ошибку
    признака: |f_W - sum c_k g_k| <= sup * (число событий в истории).
    """
    lambdas = halflives_to_lambdas(halflives_s)
    if s_max is None:
        s_max = 4.0 * window_s
    s = np.linspace(0.0, s_max, n_grid)

    target = (s < window_s).astype(np.float64)
    design = np.exp(-np.outer(s, lambdas))  # (n_grid, k)

    gram = design.T @ design + ridge * np.eye(len(lambdas))
    coef = np.linalg.solve(gram, design.T @ target)

    approx = design @ coef
    resid = target - approx
    denom = float(np.linalg.norm(target)) or 1.0
    return {
        "window_s": float(window_s),
        "halflives_s": tuple(float(h) for h in halflives_s),
        "coef": coef,
        "rel_l2": float(np.linalg.norm(resid) / denom),
        "sup": float(np.max(np.abs(resid))),
        "coef_l1": float(np.abs(coef).sum()),
        "s": s,
        "target": target,
        "approx": approx,
    }


def window_coverage_table(
    windows_s=TYPICAL_HUMAN_WINDOWS, halflives_s=DEFAULT_HALFLIVES
) -> list[dict]:
    """Таблица «окно -> ошибка приближения базисом» для отчёта."""
    rows = []
    for w in windows_s:
        fit = fit_window_as_decay_mix(w, halflives_s)
        rows.append(
            {
                "window": format_halflife(w),
                "rel_l2": fit["rel_l2"],
                "sup": fit["sup"],
                "coef_l1": fit["coef_l1"],
                "n_kernels": len(halflives_s),
            }
        )
    return rows


# --------------------------------------------------------------------------
# Референсная реализация — эталон для тестов JIT-движка
# --------------------------------------------------------------------------

def decayed_count_reference(
    t_events: np.ndarray, t_query: np.ndarray, halflife_s: float
) -> np.ndarray:
    """Прямое (медленное) вычисление затухающего счётчика по определению.

    Для каждого t_q считает sum_{t_e < t_q} exp(-l (t_q - t_e)) двойным
    циклом. Нужна только чтобы проверить, что JIT-рекуррента в engine.py
    даёт то же самое: тест на эквивалентность определения и рекурренты.
    """
    lmbd = halflife_to_lambda(halflife_s)
    out = np.zeros(len(t_query), dtype=np.float64)
    for i, tq in enumerate(t_query):
        past = t_events[t_events < tq]
        if len(past):
            out[i] = np.exp(-lmbd * (tq - past)).sum()
    return out


def windowed_count_reference(
    t_events: np.ndarray, t_query: np.ndarray, window_s: float
) -> np.ndarray:
    """Жёсткое окно по определению — baseline «как считает человек»."""
    out = np.zeros(len(t_query), dtype=np.float64)
    for i, tq in enumerate(t_query):
        out[i] = np.count_nonzero((t_events < tq) & (t_events >= tq - window_s))
    return out


# --------------------------------------------------------------------------
# Моменты затухающего распределения возрастов — O(1) альтернатива окнам
# --------------------------------------------------------------------------
#
# Эксперимент выше показал: восстановить жёсткое окно из базиса затуханий
# устойчиво невозможно (||c||_1 ~ 1e3..1e5, R^2 признака ~ -1e8). Значит
# имитировать окно — тупиковый путь.
#
# Но чем вообще является окно? Это один срез РАСПРЕДЕЛЕНИЯ ВОЗРАСТОВ
# событий: «какая доля истории моложе W». Человек, выбирая набор окон,
# грубо сэмплирует функцию распределения возрастов в нескольких точках.
#
# Вместо срезов возьмём моменты этого распределения, взвешенного затуханием:
#
#     A = sum_j exp(-l s_j)            (порядок 0: затухающий счётчик)
#     B = sum_j s_j   exp(-l s_j)      (порядок 1)
#     C = sum_j s_j^2 exp(-l s_j)      (порядок 2)
#
# где s_j = t - t_j — возраст j-го события на момент t.
#
# Ключевой факт (проверен численно до 1e-16, см. tests/test_kernels.py):
# при сдвиге времени на dt все три поддерживаются за O(1):
#
#     A <- e^{-l dt} A
#     B <- e^{-l dt} (B + dt A)
#     C <- e^{-l dt} (C + 2 dt B + dt^2 A)
#
# а приход нового события даёт A += 1 (его возраст равен нулю, поэтому
# B и C не меняются). Обобщение на порядок n — биномиальная свёртка.
#
# Производные интерпретируемые величины:
#     B/A                      — средний возраст истории (с весом затухания)
#     sqrt(C/A - (B/A)^2)      — разброс возрастов: «кучно» или «размазано»
#
# Содержательно: A отвечает на «сколько», B/A — на «когда», и ни то, ни
# другое не требует от человека выбирать границу окна.

def age_moment_decay_step(
    A: float, B: float, C: float, lmbd: float, dt: float
) -> tuple[float, float, float]:
    """Один шаг затухания моментов на интервал dt. Эталон для тестов движка."""
    e = np.exp(-lmbd * dt)
    return e * A, e * (B + dt * A), e * (C + 2.0 * dt * B + dt * dt * A)


def age_moments_reference(
    t_events: np.ndarray, t_query: np.ndarray, halflife_s: float
) -> np.ndarray:
    """Моменты возраста по определению (прямой двойной цикл). Эталон."""
    lmbd = halflife_to_lambda(halflife_s)
    out = np.zeros((len(t_query), 3), dtype=np.float64)
    for i, tq in enumerate(t_query):
        s = tq - t_events[t_events < tq]
        if len(s):
            w = np.exp(-lmbd * s)
            out[i] = (w.sum(), (s * w).sum(), (s * s * w).sum())
    return out
