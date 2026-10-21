"""
Графики для отчёта.

Правила, которым подчинены все фигуры здесь:

  ОДНА ОСЬ НА ГРАФИК. Двойная ось (две шкалы y) запрещена: выравнивание
  шкал произвольно, и график начинает показывать корреляцию, которой в
  данных нет. Первая версия e02 именно этим и грешила — log-loss и AUC на
  двух автомасштабированных осях выглядели одинаково изменчивыми, хотя
  весь смысл был в том, что размах AUC на порядок меньше. Две величины
  разного масштаба приводятся к общей базе и рисуются на одной оси.

  ПОДПИСИ ВЫБОРОЧНО И БЕЗ НАЛОЖЕНИЙ. Число у каждой точки — шум. Там, где
  подписей много (фронт Парето, сравнение методов), категории выносятся на
  собственную ось горизонтальными полосами, и наложиться физически не могут.

  ПАЛИТРА ПРОВЕРЕНА, А НЕ ПОДОБРАНА НА ГЛАЗ. Слоты взяты из валидированного
  набора: синий #2a78d6 и оранжевый #eb6834 дают разделение CVD ΔE 24.7 при
  пороге 8. Прежняя пара (#1f4e79, #c0504d) проверку проваливала: синий
  выходил из диапазона светлоты и читался как серый.

  ТОНКИЕ ШТРИХИ, СПЛОШНАЯ СЕТКА НА ТОН ОТ ФОНА, текст нейтральными
  чернилами, а не цветом серии.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPORTS = Path(__file__).resolve().parents[2] / "reports"
FIGS = REPORTS / "figures"

#: слоты категориальной палитры (проверены validate_palette.js)
C1 = "#2a78d6"   # слот 1, синий
C2 = "#eb6834"   # слот 2, оранжевый
C3 = "#1baf7a"   # слот 3, аква
INK = "#1a1a1a"       # основной текст
INK_MUTED = "#6b7280"  # вторичный текст
GRID = "#e5e7eb"      # сетка, на тон от фона
MARK_MUTED = "#b9c0c9"  # приглушённый штрих

#: Ширина фигуры в дюймах, равная печатной ширине A4 с полями (178 мм).
#: Если рисовать шире, PDF ужмёт картинку, и подписи 8pt превратятся в
#: эффективные 5pt — на печати нечитаемо.
PAGE_W = 7.3

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "grid.linestyle": "-", "grid.alpha": 1.0,
    "legend.frameon": False, "legend.fontsize": 7,
    "lines.linewidth": 2.0, "lines.markersize": 4.5,
})


def _load(name: str):
    p = REPORTS / name
    return json.loads(p.read_text()) if p.exists() else None


def _save(fig, name: str) -> Path:
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / name
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_window_vs_decay() -> Path | None:
    """Эксперимент 1: почему базис затуханий не заменяет жёсткое окно."""
    d = _load("e01_window_vs_decay.json")
    if d is None:
        return None
    from .kernels import DAY, DEFAULT_HALFLIVES, fit_window_as_decay_mix

    fig, ax = plt.subplots(1, 3, figsize=(PAGE_W, 2.7))

    f = fit_window_as_decay_mix(7 * DAY, DEFAULT_HALFLIVES)
    s_d = f["s"] / DAY
    ax[0].plot(s_d, f["target"], color=C1, lw=2.2, label="окно 7 дн.")
    ax[0].plot(s_d, f["approx"], color=C2, lw=1.6, ls="--",
               label="8 экспонент")
    ax[0].axvline(7, color=MARK_MUTED, lw=0.8, zorder=0)
    ax[0].set_xlim(0, 20)
    ax[0].set_ylim(-0.25, 1.35)
    ax[0].set_xlabel("возраст события, дней")
    ax[0].set_ylabel("вес ядра")
    ax[0].set_title("A. Разрыв не воспроизводится")
    # справа по центру пусто: после 10 дней обе кривые лежат у нуля
    ax[0].legend(loc="upper right", borderaxespad=0.3, handlelength=1.6)

    cv = d["part_a"]["convergence"]
    k = [r["n_kernels"] for r in cv]
    ax[1].plot(k, [r["sup"] for r in cv], "o-", color=C2, label="sup-ошибка")
    ax[1].plot(k, [r["rel_l2"] for r in cv], "s-", color=C1,
               label="относительная L2")
    ax[1].set_ylim(0, 0.8)
    ax[1].set_xlabel("число экспонент")
    ax[1].set_ylabel("ошибка приближения")
    ax[1].set_title("B. sup-ошибка не убывает")
    ax[1].legend(loc="upper right", borderaxespad=0.4)
    # подпись только у крайней точки, а не у каждой
    ax[1].annotate(f"{cv[-1]['sup']:.2f}", (k[-1], cv[-1]["sup"]),
                   xytext=(0, 7), textcoords="offset points",
                   fontsize=7.5, color=INK_MUTED, ha="center")

    ax[2].plot(k, [r["coef_l1"] for r in cv], "o-", color=C1)
    ax[2].set_yscale("log")
    ax[2].set_xlabel("число экспонент")
    ax[2].set_ylabel("норма коэффициентов")
    ax[2].set_title("C. Обусловленность разрушается")
    ax[2].annotate("$%.1f\\cdot10^{%d}$" % (cv[-1]["coef_l1"] / 10 ** int(np.log10(cv[-1]["coef_l1"])),
                                          int(np.log10(cv[-1]["coef_l1"]))),
                   (k[-1], cv[-1]["coef_l1"]),
                   xytext=(0, -14), textcoords="offset points",
                   fontsize=7.5, color=INK_MUTED, ha="center")

    fig.tight_layout(w_pad=2.2)
    return _save(fig, "e01_window_vs_decay.png")


def fig_timescales() -> Path | None:
    """Эксперимент 2A: log-loss идентифицирует масштаб, AUC — нет.

    ОДНА ОСЬ. Обе метрики приведены к общей базе — «насколько процентов
    хуже лучшего значения». Только в таком виде видно то, ради чего график
    и нужен: размах log-loss около 25%, размах AUC около 1.5%, то есть на
    порядок меньше. На двух автомасштабированных осях (как было в первой
    версии) обе кривые выглядели одинаково изменчивыми, и график
    опровергал собственную подпись.
    """
    d = _load("e02_timescales.json")
    if d is None:
        return None
    cases = d["part_a"]["cases"]
    fig, axes = plt.subplots(1, len(cases), figsize=(PAGE_W, 2.9),
                             sharey=True)
    if len(cases) == 1:
        axes = [axes]

    ll_spreads, au_spreads = [], []
    for ci, (ax, c) in enumerate(zip(axes, cases, strict=False)):
        cur = np.array(c["curve"])          # [полупериод, logloss, auc, нулей]
        hl, ll, au = cur[:, 0], cur[:, 1], cur[:, 2]
        ll_rel = (ll - np.nanmin(ll)) / np.nanmin(ll) * 100
        au_rel = (np.nanmax(au) - au) / np.nanmax(au) * 100
        ll_spreads.append(ll_rel.max())
        au_spreads.append(au_rel.max())

        ax.plot(hl, ll_rel, "o-", color=C1, label="log-loss")
        ax.plot(hl, au_rel, "s-", color=C2, label="AUC")
        ax.axvline(c["true_halflife_days"], color=MARK_MUTED, lw=1.0, zorder=0)

        # где каждая метрика ставит оптимум — это и есть суть графика
        i_ll, i_au = int(np.nanargmin(ll_rel)), int(np.nanargmin(au_rel))
        ax.plot(hl[i_ll], ll_rel[i_ll], "o", color=C1, ms=11, mfc="none", mew=1.8)
        ax.plot(hl[i_au], au_rel[i_au], "s", color=C2, ms=11, mfc="none", mew=1.8)

        # найденные оптимумы — текстом в свободном верхнем правом углу:
        # подписи у самих точек налезали на кривые у края оси
        ax.text(0.97, 0.97, f"log-loss: {hl[i_ll]:.1f} дн.",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color=C1, weight="bold")
        ax.text(0.97, 0.88, f"AUC: {hl[i_au]:.1f} дн.",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color=C2, weight="bold")
        ax.set_xscale("log")
        ax.set_xticks([1, 10, 100])
        ax.set_xticklabels(["1", "10", "100"])
        ax.set_xlabel("полупериод, дней")
        ax.set_title(f"истинный полупериод {c['true_halflife_days']:g} дн.")
        if ci == 0:
            ax.set_ylabel("хуже лучшего значения, %")
        ax.set_ylim(-2, 33)

    h1, l1 = axes[0].get_legend_handles_labels()
    fig.legend(h1, l1, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(w_pad=1.6, rect=(0, 0, 1, 0.92))
    return _save(fig, "e02_timescales.png")


#: читаемые названия методов и группа, определяющая цвет
METHOD_LABEL = {
    "seen": "ярлык «видел айтем»",
    "popularity": "популярность",
    "hand": "ручные фичи, окна",
    "hand_keys_decay": "те же ключи, сетка ядер",
    "compiler_L0": "компилятор, ур. 0",
    "compiler": "компилятор, ур. 0+1",
    "compiler_selected": "компилятор + отбор",
    "compiler_covisit": "+ совстречаемость (протекает)",
    "compiler_covisit_honest": "+ совстречаемость (честная)",
}
METHOD_GROUP = {
    "seen": "trivial", "popularity": "trivial",
    "hand": "hand", "hand_keys_decay": "hand",
}
GROUP_COLOR = {"trivial": MARK_MUTED, "hand": C2, "compiler": C1}
GROUP_LABEL = {"trivial": "тривиальные ориентиры",
               "hand": "ручной набор и его ключи",
               "compiler": "компилятор"}


def _benchmark_panel(ax, data, title: str, show_legend: bool) -> None:
    """Горизонтальные полосы NDCG@10 с доверительными интервалами.

    Категории вынесены на собственную ось — подписи наложиться не могут.
    Значение печатается за концом уса, а не поверх полосы.
    """
    rs = sorted(data["results"], key=lambda r: r["test"]["ndcg@10"])
    y = np.arange(len(rs))
    vals = np.array([r["test"]["ndcg@10"] for r in rs])
    lo = np.array([r["test"].get("ndcg@10_lo", v) for r, v in zip(rs, vals, strict=False)])
    hi = np.array([r["test"].get("ndcg@10_hi", v) for r, v in zip(rs, vals, strict=False)])
    groups = [METHOD_GROUP.get(r["name"], "compiler") for r in rs]
    colors = [GROUP_COLOR[g] for g in groups]

    ax.barh(y, vals, color=colors, height=0.62, zorder=2)
    ax.errorbar(vals, y, xerr=[vals - lo, hi - vals], fmt="none",
                ecolor=INK_MUTED, elinewidth=1.1, capsize=3, zorder=3)
    # отступ подписи пропорционален диапазону оси: у панелей он разный,
    # и абсолютная константа вплотную прижимала бы текст к усу
    pad = max(hi) * 0.025
    for yi, v, h in zip(y, vals, hi, strict=False):
        ax.text(h + pad, yi, f"{v:.4f}", va="center", fontsize=7.5, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABEL.get(r["name"], r["name"]) for r in rs],
                       fontsize=8)
    ax.set_xlim(0, max(hi) * 1.22)
    ax.set_xlabel("test NDCG@10  (ус — 95% доверительный интервал)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)

    if show_legend:
        from matplotlib.patches import Patch
        seen_groups = list(dict.fromkeys(groups))
        ax.legend(handles=[Patch(facecolor=GROUP_COLOR[g], label=GROUP_LABEL[g])
                           for g in seen_groups],
                  loc="lower right", borderaxespad=0.6)


def fig_benchmark() -> Path | None:
    """Эксперимент 3: сравнение методов на реальных данных и на синтетике.

    Первая версия читала устаревший e03_benchmark.json с числами первого,
    давно отменённого прогона, и печатала значение поверх названия метода.
    Здесь берутся актуальные таблицы, значения вынесены за ус, а категории
    стоят на собственной оси.
    """
    rr = _load("e03_benchmark_retailrocket.json")
    syn = _load("e03_benchmark_synthetic.json")
    panels = [(rr, "RetailRocket: методы расходятся"),
              (syn, "Синтетика: все упираются в потолок")]
    panels = [(d, t) for d, t in panels if d is not None]
    if not panels:
        return None

    # панели ОДНА ПОД ДРУГОЙ, а не рядом: на ширине A4 две панели в ряд
    # сжимаются так, что подписи методов становятся нечитаемыми
    fig, axes = plt.subplots(len(panels), 1, figsize=(PAGE_W, 3.3 * len(panels)))
    if len(panels) == 1:
        axes = [axes]
    for i, (ax, (d, title)) in enumerate(zip(axes, panels, strict=False)):
        _benchmark_panel(ax, d, title, show_legend=(i == 0))
    fig.tight_layout(h_pad=2.0)
    return _save(fig, "e03_benchmark.png")


def fig_scaling() -> Path | None:
    """Эксперимент 6: две оси масштабирования, каждая на своём графике.

    Левая панель — длина лога, где предсказание о плато НЕ подтвердилось.
    Правая — горизонт признака, где различие принципиальное. Обе величины
    на правой панели нормированы на первую точку, поэтому сравниваются
    на одной оси, без второй шкалы.
    """
    d = _load("e06_scaling.json")
    if d is None:
        return None
    p = d["points"]
    n = np.array([r["n_events"] for r in p], float)
    sc = np.array([r["compiler_peak_state_bytes"] for r in p], float) / 2**20
    hb = np.array([r["hand_window_state_bytes"] for r in p], float) / 2**20
    t = np.array([r["compiler_scan_seconds"] for r in p], float)

    pb = d.get("part_b")
    ncol = 3 if pb else 2
    fig, ax = plt.subplots(1, ncol, figsize=(PAGE_W, 2.8))

    ax[0].plot(n, sc, "o-", color=C1, label="компилятор")
    ax[0].plot(n, hb, "s-", color=C2, label="жёсткие окна")
    # явные деления: автоматические минорные метки лог-шкалы слипаются
    # у правого края ("3x10^6 4x10^6")
    ticks = [f"{v / 1e6:g}M" for v in n]
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xticks(n)
    ax[0].set_xticklabels(ticks)
    ax[0].minorticks_off()
    ax[0].set_yticks([20, 50, 100, 200, 400])
    ax[0].set_yticklabels(["20", "50", "100", "200", "400"])
    ax[0].set_ylim(15, 500)
    ax[0].set_xlabel("событий в логе")
    ax[0].set_ylabel("память состояния, МиБ")
    ax[0].set_title("A. Длина лога: плато нет")
    ax[0].legend(loc="lower right", borderaxespad=0.4)

    ax[1].plot(n, t, "o-", color=C1, label="замерено")
    ax[1].plot(n, t[0] * n / n[0], "-", color=MARK_MUTED, lw=1.2,
               label="линейный рост")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xticks(n)
    ax[1].set_xticklabels(ticks)
    ax[1].minorticks_off()
    ax[1].set_yticks([1, 2, 4, 8])
    ax[1].set_yticklabels(["1", "2", "4", "8"])
    ax[1].set_ylim(0.7, 10)
    ax[1].set_xlabel("событий в логе")
    ax[1].set_ylabel("время прохода, с")
    ax[1].set_title("B. Время прохода линейно")
    ax[1].legend(loc="upper left", borderaxespad=0.4)

    if pb:
        h = np.array([r["horizon_days"] for r in pb["rows"]], float)
        cb = np.array([r["compiler_state_bytes"] for r in pb["rows"]], float)
        wb = np.array([r["window_state_bytes"] for r in pb["rows"]], float)
        ax[2].plot(h, cb / cb[0], "o-", color=C1, label="компилятор")
        ax[2].plot(h, wb / wb[0], "s-", color=C2, label="жёсткие окна")
        ax[2].set_xscale("log")
        ax[2].set_xticks(h)
        ax[2].set_xticklabels([f"{v:g}" for v in h])
        ax[2].set_ylim(0.6, 6.3)
        ax[2].minorticks_off()
        ax[2].set_xlabel("горизонт признака, дней")
        ax[2].set_ylabel("память, × от горизонта 1 день")
        ax[2].set_title("C. Горизонт признака")
        ax[2].legend(loc="upper left", borderaxespad=0.4)
        ax[2].annotate(f"×{wb[-1] / wb[0]:.1f}", (h[-1], wb[-1] / wb[0]),
                       xytext=(-30, 2), textcoords="offset points",
                       fontsize=8, color=INK)
        ax[2].annotate(f"×{cb[-1] / cb[0]:.2f}", (h[-1], cb[-1] / cb[0]),
                       xytext=(-32, 8), textcoords="offset points",
                       fontsize=8, color=INK)

    fig.tight_layout(w_pad=2.4)
    return _save(fig, "e06_scaling.png")


def fig_pareto() -> Path | None:
    """Эксперимент 7: что даёт каждый добавленный ключ.

    Первая версия рисовала точки в осях «память против качества» с
    подписью у каждой. Четыре последних ключа оказывались в одной точке, и
    пять подписей наложились в нечитаемую кляксу. Здесь ключи вынесены на
    собственную ось в порядке добавления: наложиться невозможно, а
    стоимость каждого шага указана в подписи строки.
    """
    d = _load("e07_budget.json")
    if d is None:
        return None
    pts = d["pareto"]
    y = np.arange(len(pts))
    vals = np.array([p["metrics"]["ndcg@10"] for p in pts])
    mem = [p["state_mib"] for p in pts]
    labels = [f"{p['added']}   ({m:,.0f} МиБ)".replace(",", " ")
              for p, m in zip(pts, mem, strict=False)]

    fig, ax = plt.subplots(figsize=(PAGE_W, 3.9))
    ax.barh(y, vals, color=C1, height=0.6, zorder=2)
    pad = max(vals) * 0.02
    for yi, v in zip(y, vals, strict=False):
        ax.text(v + pad, yi, f"{v:.4f}", va="center", fontsize=7.5, color=INK)

    h = d.get("hand_reference")
    if h:
        hv = h["metrics"]["ndcg@10"]
        ax.axvline(hv, color=C2, lw=1.6, zorder=1)
        # подпись слева от линии в верхней строке: там полоса нулевая,
        # и текст ничего не перекрывает (легенда справа перечёркивалась линией)
        ax.text(hv - 0.012, 0, f"ручной набор: {hv:.4f} при {h['state_mib']:.0f} МиБ",
                ha="right", va="center", fontsize=7.5, color=C2)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.15)
    ax.set_xlabel("test NDCG@10 после добавления ключа")
    ax.set_title("Ключи в порядке добавления жадностью «польза / память»")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, "e07_pareto.png")


ALL = [fig_window_vs_decay, fig_timescales, fig_benchmark, fig_scaling,
       fig_pareto]


def build_all(verbose: bool = True) -> list[Path]:
    made = []
    for fn in ALL:
        try:
            p = fn()
        except Exception as e:                  # график не должен ронять сборку
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
