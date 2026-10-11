"""
Эксперимент 1. Можно ли заменить жёсткое окно базисом затуханий? Нет.

Это первая гипотеза, которую я проверил, и она оказалась ЛОЖНОЙ. Результат
сохранён как отдельный воспроизводимый эксперимент, потому что из него
следует вся дальнейшая конструкция.

Гипотеза была такая. Ручная фича «событий за последние W дней» — это
интеграл процесса событий по индикаторному ядру 1[s < W]. Если индикатор
приближается смесью экспонент, то

    f_W ~= sum_k c_k * g_{lambda_k}   для ЛЮБОЙ истории событий,

потому что приближение в пространстве ядер переносится в пространство
признаков линейно. Тогда достаточно держать сетку затухающих счётчиков
(каждый O(1) на событие), и любое окно получается бесплатно.

Что измерено:

  A. Приближение ЯДРА. Индикатор окна разрывен в точке s = W, а сумма
     гладких экспонент разрыв воспроизвести не может. sup-ошибка остаётся
     около 0.5 при любом числе ядер — это предел, а не недостаток
     оптимизации. Норма коэффициентов при этом взрывается.

  B. Приближение ПРИЗНАКА. Главная проверка: даже если ядро приближено
     плохо, признак — это интеграл, и ошибка могла бы усредниться. Не
     усредняется: на пуассоновских историях корреляция около нуля, R^2
     порядка -1e8. Плохая обусловленность (восстановление окна из базиса
     затуханий — это обратная задача Лапласа) усиливает шум катастрофически.

Вывод: окно и затухающий счётчик — два разных линейных функционала
истории, и ни один не выражается через другой устойчиво. Значит
«что лучше» не решается теоретически, только экспериментом. Отсюда
управляемый синтетический полигон (эксперимент 2) и сквозной бенчмарк
(эксперимент 3).

И отсюда же правильный примитив: не имитировать окно, а перейти к
МОМЕНТАМ затухающего распределения возрастов (kernels.py), которые тоже
поддерживаются за O(1) и описывают форму истории, а не один её срез.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afe.kernels import (DAY, DEFAULT_HALFLIVES, TYPICAL_HUMAN_WINDOWS,   # noqa: E402
                         fit_window_as_decay_mix, format_halflife,
                         geometric_halflives, halflives_to_lambdas)

OUT = Path(__file__).resolve().parents[1] / "reports"


def part_a_kernel() -> dict:
    """Ошибка приближения ЯДРА и её поведение при росте числа экспонент."""
    coverage = []
    for w in TYPICAL_HUMAN_WINDOWS:
        f = fit_window_as_decay_mix(w, DEFAULT_HALFLIVES)
        # где именно сидит ошибка: у разрыва или всюду
        s, res = f["s"], np.abs(f["target"] - f["approx"])
        near = np.abs(s - w) < 0.15 * w
        coverage.append({
            "window": format_halflife(w),
            "rel_l2": f["rel_l2"], "sup": f["sup"], "coef_l1": f["coef_l1"],
            "sup_away_from_jump": float(res[~near].max()),
            "l2_share_at_jump": float((res[near] ** 2).sum() / (res ** 2).sum()),
        })

    convergence = []
    for k in (2, 3, 4, 6, 8, 12, 16, 24, 32):
        hl = geometric_halflives(3600.0, 180 * DAY, k=k, with_inf=True)
        f = fit_window_as_decay_mix(7 * DAY, hl)
        convergence.append({"n_kernels": len(hl), "rel_l2": f["rel_l2"],
                            "sup": f["sup"], "coef_l1": f["coef_l1"]})
    return {"coverage": coverage, "convergence": convergence}


def part_b_feature(n_hist: int = 4000, seed: int = 0) -> dict:
    """Ошибка приближения САМОГО ПРИЗНАКА на пуассоновских историях."""
    rng = np.random.default_rng(seed)
    W = 7 * DAY
    f = fit_window_as_decay_mix(W, DEFAULT_HALFLIVES)
    lam, c = halflives_to_lambdas(DEFAULT_HALFLIVES), f["coef"]

    rows = []
    for rate in (0.05, 0.2, 1.0, 5.0):
        true_w, approx = [], []
        for _ in range(n_hist):
            n = rng.poisson(rate * 365)
            if n == 0:
                continue
            ages = rng.uniform(0, 365 * DAY, n)
            true_w.append(np.count_nonzero(ages < W))
            S = np.array([np.exp(-l * ages).sum() for l in lam])
            approx.append(float(c @ S))
        tw, ap = np.array(true_w, float), np.array(approx, float)
        rows.append({
            "rate_per_day": rate,
            "corr": float(np.corrcoef(tw, ap)[0, 1]),
            "rel_mae": float(np.abs(tw - ap).mean() / max(tw.mean(), 1e-9)),
            "r2": float(1 - ((tw - ap) ** 2).sum() / ((tw - tw.mean()) ** 2).sum()),
            "n_histories": len(tw),
        })
    return {"coef_l1": f["coef_l1"], "rows": rows}


def report(a: dict, b: dict) -> None:
    W = 86
    print("=" * W)
    print("ЧАСТЬ A. ПРИБЛИЖЕНИЕ ЯДРА: индикатор окна смесью экспонент")
    print("=" * W)
    print(f"{'окно':>8}{'rel_L2':>10}{'sup':>9}{'sup вне разрыва':>18}"
          f"{'L2 у разрыва':>15}{'||c||_1':>12}")
    for r in a["coverage"]:
        print(f"{r['window']:>8}{r['rel_l2']:>10.4f}{r['sup']:>9.4f}"
              f"{r['sup_away_from_jump']:>18.4f}{r['l2_share_at_jump']:>14.0%}"
              f"{r['coef_l1']:>12.1f}")
    print("\n  сходимость по числу экспонент (окно 7 дней):")
    print(f"  {'K':>4}{'rel_L2':>10}{'sup':>9}{'||c||_1':>14}")
    for r in a["convergence"]:
        print(f"  {r['n_kernels']:>4}{r['rel_l2']:>10.4f}{r['sup']:>9.4f}"
              f"{r['coef_l1']:>14.1f}")
    sups = [r["sup"] for r in a["convergence"]]
    print(f"\n  sup-ошибка при K от {a['convergence'][0]['n_kernels']} до "
          f"{a['convergence'][-1]['n_kernels']}: от {sups[0]:.4f} до {sups[-1]:.4f}")
    print("  Она НЕ убывает: индикатор разрывен в s = W, гладкие экспоненты")
    print("  разрыв не воспроизводят. Это предел, а не недостаток оптимизации.")

    print("\n" + "=" * W)
    print("ЧАСТЬ B. ПРИБЛИЖЕНИЕ ПРИЗНАКА: главная проверка")
    print("=" * W)
    print("  Ядро приближено плохо, но признак — это интеграл по ядру.")
    print("  Могла ли ошибка усредниться на реальных историях событий?\n")
    print(f"{'событий/день':>14}{'corr':>9}{'отн.MAE':>12}{'R^2':>16}{'историй':>10}")
    for r in b["rows"]:
        print(f"{r['rate_per_day']:>14.2f}{r['corr']:>9.4f}{r['rel_mae']:>12.1f}"
              f"{r['r2']:>16.3g}{r['n_histories']:>10,}")
    print(f"\n  Не усредняется. При ||c||_1 = {b['coef_l1']:.0f} восстановление")
    print("  окна из базиса затуханий — плохо обусловленная обратная задача")
    print("  Лапласа, и шум усиливается катастрофически.")
    print("\n  ВЫВОД: окно и затухающий счётчик не выражаются друг через друга")
    print("  устойчиво. Сравнивать их можно только экспериментально, что и")
    print("  делается в экспериментах 2 и 3. А правильный примитив — не")
    print("  имитация окна, а моменты затухающего распределения возрастов.")
    print("=" * W)


if __name__ == "__main__":
    a = part_a_kernel()
    b = part_b_feature()
    report(a, b)
    OUT.mkdir(exist_ok=True)
    (OUT / "e01_window_vs_decay.json").write_text(
        json.dumps({"part_a": a, "part_b": b}, indent=2, ensure_ascii=False))
    print("результаты: reports/e01_window_vs_decay.json")
