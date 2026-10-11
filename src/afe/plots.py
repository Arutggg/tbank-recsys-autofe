"""Графики для отчёта. Каждый читает JSON соответствующего эксперимента."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORTS = Path(__file__).resolve().parents[2] / "reports"
FIGS = REPORTS / "figures"

# единый стиль: без лишних рамок, читаемые подписи, без цветового шума
plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False, "figure.autolayout": False,
})
C_MAIN, C_ALT, C_BAD, C_MUTE = "#1f4e79", "#c0504d", "#8c8c8c", "#bfbfbf"


def _load(name: str):
    p = REPORTS / name
    return json.loads(p.read_text()) if p.exists() else None


def fig_window_vs_decay() -> Path | None:
    """Эксперимент 1: почему базис затуханий не заменяет жёсткое окно."""
    d = _load("e01_window_vs_decay.json")
    if d is None:
        return None
    from .kernels import DAY, DEFAULT_HALFLIVES, fit_window_as_decay_mix

    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))

    f = fit_window_as_decay_mix(7 * DAY, DEFAULT_HALFLIVES)
    s_d = f["s"] / DAY
    ax[0].plot(s_d, f["target"], color=C_MAIN, lw=1.8, label="окно 7 дней: 1[s<W]")
    ax[0].plot(s_d, f["approx"], color=C_BAD, lw=1.2, ls="--",
               label="лучшая смесь 8 экспонент")
    ax[0].axvline(7, color=C_MUTE, lw=0.8, ls=":")
    ax[0].set_xlim(0, 20)
    ax[0].set_xlabel("возраст события s, дней")
    ax[0].set_ylabel("вес ядра")
    ax[0].set_title("A. Разрыв не воспроизводится")
    ax[0].legend(fontsize=7.5, loc="upper right")

    cv = d["part_a"]["convergence"]
    k = [r["n_kernels"] for r in cv]
    ax[1].plot(k, [r["sup"] for r in cv], "o-", color=C_BAD, ms=3.5,
               label="sup-ошибка")
    ax[1].plot(k, [r["rel_l2"] for r in cv], "s-", color=C_MAIN, ms=3.5,
               label="относительная L2")
    ax[1].set_xlabel("число экспонент K")
    ax[1].set_ylabel("ошибка")
    ax[1].set_ylim(0, 0.75)
    ax[1].set_title("B. sup-ошибка не убывает с K")
    ax[1].legend(fontsize=7.5)

    ax2 = ax[2]
    ax2.plot(k, [r["coef_l1"] for r in cv], "o-", color=C_BAD, ms=3.5)
    ax2.set_yscale("log")
    ax2.set_xlabel("число экспонент K")
    ax2.set_ylabel(r"$\|c\|_1$")
    ax2.set_title("C. Обусловленность разрушается")
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "e01_window_vs_decay.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_timescales() -> Path | None:
    """Эксперимент 2A: log-loss идентифицирует масштаб, AUC — нет."""
    d = _load("e02_timescales.json")
    if d is None:
        return None
    cases = d["part_a"]["cases"]
    fig, axes = plt.subplots(1, len(cases), figsize=(3.4 * len(cases), 3.0))
    if len(cases) == 1:
        axes = [axes]
    for ax, c in zip(axes, cases):
        cur = np.array(c["curve"])          # [полупериод, logloss, auc, доля нулей]
        hl, ll, au = cur[:, 0], cur[:, 1], cur[:, 2]
        ax.plot(hl, ll, "o-", color=C_MAIN, ms=3, label="log-loss")
        ax.axvline(c["true_halflife_days"], color=C_MUTE, lw=1.2, ls="--",
                   label=f"истина {c['true_halflife_days']:g}д")
        ax.set_xscale("log")
        ax.set_xlabel("полупериод, дней")
        ax.set_ylabel("log-loss")
        ax.set_title(f"истина {c['true_halflife_days']:g}д -> "
                     f"найдено {c['found_by_logloss']:.1f}д")
        a2 = ax.twinx()
        a2.plot(hl, au, "s-", color=C_BAD, ms=2.5, lw=1.0, alpha=0.75)
        a2.set_ylabel("AUC", color=C_BAD)
        a2.tick_params(axis="y", labelcolor=C_BAD, labelsize=7)
        a2.grid(False)
        ax.legend(fontsize=7, loc="upper center")
    fig.suptitle("Минимум log-loss (синий) совпадает с истиной; "
                 "AUC (красный) почти плоский", fontsize=9.5, y=1.03)
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "e02_timescales.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_benchmark() -> Path | None:
    """Эксперимент 3: сквозное сравнение методов."""
    d = _load("e03_benchmark.json")
    if d is None:
        return None
    rs = d["results"]
    names = [r["name"] for r in rs]
    fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.2))
    y = np.arange(len(names))
    for a, key, title in (
        (ax[0], "auc", "test AUC"),
        (ax[1], "ndcg@10", "test NDCG@10"),
    ):
        v = [r["test"][key] for r in rs]
        cols = [C_MAIN if n.startswith("compiler") else
                (C_ALT if n == "hand" else C_MUTE) for n in names]
        a.barh(y, v, color=cols, height=0.62)
        a.set_yticks(y)
        a.set_yticklabels(names, fontsize=7.5)
        a.set_xlim(min(v) - 0.02 * (max(v) - min(v) + 1e-9), max(v) + 0.005)
        a.set_title(title)
        a.invert_yaxis()
        for yi, vi in zip(y, v):
            a.text(vi, yi, f" {vi:.4f}", va="center", fontsize=7)

    nf = [r["n_features"] for r in rs]
    au = [r["test"]["auc"] for r in rs]
    for n, x, yv in zip(names, nf, au):
        c = C_MAIN if n.startswith("compiler") else (C_ALT if n == "hand" else C_MUTE)
        ax[2].scatter(max(x, 1), yv, s=38, color=c)
        ax[2].annotate(n, (max(x, 1), yv), fontsize=6.5,
                       textcoords="offset points", xytext=(4, 3))
    ax[2].set_xscale("log")
    ax[2].set_xlabel("число признаков")
    ax[2].set_ylabel("test AUC")
    ax[2].set_title("качество против размера набора")
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "e03_benchmark.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_scaling() -> Path | None:
    """Эксперимент 6: плато состояния против линейного роста."""
    d = _load("e06_scaling.json")
    if d is None:
        return None
    p = d["points"]
    n = np.array([r["n_events"] for r in p], float)
    sc = np.array([r["compiler_peak_state_bytes"] for r in p], float) / 2**20
    hb = np.array([r["hand_window_state_bytes"] for r in p], float) / 2**20
    t = np.array([r["compiler_scan_seconds"] for r in p], float)

    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.2))
    ax[0].plot(n, sc, "o-", color=C_MAIN, ms=4,
               label="состояние компилятора (затухание)")
    ax[0].plot(n, hb, "s-", color=C_BAD, ms=4,
               label="состояние жёстких окон (CSR событий)")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("событий в логе")
    ax[0].set_ylabel("память состояния, МиБ")
    ax[0].set_title("Каталог фиксирован: состояние выходит на плато")
    ax[0].legend(fontsize=7.5)

    ax[1].plot(n, t, "o-", color=C_MAIN, ms=4, label="время прохода")
    ref = t[0] * n / n[0]
    ax[1].plot(n, ref, ":", color=C_MUTE, lw=1.2, label="линейный рост")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("событий в логе")
    ax[1].set_ylabel("секунды")
    ax[1].set_title("Время прохода линейно по длине лога")
    ax[1].legend(fontsize=7.5)
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "e06_scaling.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_pareto() -> Path | None:
    """Эксперимент 7: качество против бюджета памяти."""
    d = _load("e07_budget.json")
    if d is None:
        return None
    pts = d["pareto"]
    mem = [p["state_mib"] for p in pts]
    val = [p["metrics"]["ndcg@10"] for p in pts]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(mem, val, "o-", color=C_MAIN, ms=4)
    for p, m, v in zip(pts, mem, val):
        if p.get("label"):
            ax.annotate(p["label"], (m, v), fontsize=6.5,
                        textcoords="offset points", xytext=(5, -2))
    if d.get("hand_reference"):
        h = d["hand_reference"]
        ax.scatter([h["state_mib"]], [h["metrics"]["ndcg@10"]], marker="*",
                   s=140, color=C_ALT, zorder=5, label="ручной набор")
        ax.legend(fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("память состояния, МиБ")
    ax.set_ylabel("test NDCG@10")
    ax.set_title("Фронт Парето: качество против бюджета памяти")
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "e07_pareto.png"
    fig.savefig(out)
    plt.close(fig)
    return out


ALL = [fig_window_vs_decay, fig_timescales, fig_benchmark, fig_scaling, fig_pareto]


def build_all(verbose: bool = True) -> list[Path]:
    made = []
    for fn in ALL:
        try:
            p = fn()
        except Exception as e:                      # график не должен ронять сборку
            if verbose:
                print(f"  {fn.__name__}: ошибка {type(e).__name__}: {e}")
            continue
        if p is None:
            if verbose:
                print(f"  {fn.__name__}: нет данных эксперимента, пропуск")
        else:
            made.append(p)
            if verbose:
                print(f"  {p.relative_to(REPORTS.parent)}")
    return made


if __name__ == "__main__":
    build_all()
