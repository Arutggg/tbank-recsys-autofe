"""
Подстановка результатов экспериментов в README.md.

Числа в отчёте не переписываются руками. Каждый раздел с результатами
генерируется из JSON, который сохранил соответствующий эксперимент, и
подставляется между маркерами:

    <!-- RESULTS:e03 -->  ...сгенерировано...  <!-- /RESULTS:e03 -->

Так отчёт не может разойтись с тем, что реально посчитано: если
эксперимент перезапущен с другими параметрами, таблица в README меняется
вместе с ним, а не остаётся от прошлого прогона.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
README = ROOT / "README.md"


def _load(name: str):
    p = REPORTS / name
    return json.loads(p.read_text()) if p.exists() else None


def _fmt(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}"


# ---------------------------------------------------------------- e03
METHOD_RU = {
    "popularity": "популярность (ориентир)",
    "hand": "ручные фичи, жёсткие окна",
    "hand_keys_decay": "те же ключи, сетка масштабов",
    "compiler_L0": "компилятор, уровень 0",
    "compiler": "компилятор, уровни 0+1",
    "compiler_selected": "компилятор + отбор на val",
}


def section_e03(dataset: str) -> str:
    d = _load(f"e03_benchmark_{dataset}.json")
    if d is None:
        return "_эксперимент ещё не запускался_\n"
    c = d["config"]
    lines = [
        f"Датасет: **{dataset}**, событий {c.get('n_events', 0):,}. "
        f"Негативов: {c['n_negatives']} на обучении, "
        f"{c.get('n_negatives_eval', c['n_negatives'])} на оценке. "
        f"Протокол негативов: `{c.get('neg_source')}`. "
        f"Бюджет тюнинга: {c['n_trials']} испытаний каждому методу.",
        "",
        "| метод | признаков | AUC | logloss | NDCG@10 | 95% ДИ для NDCG@10 | Recall@10 | фичи, с | чел.-ч |",
        "|---|---:|---:|---:|---:|:---:|---:|---:|---:|",
    ]
    base = next((r for r in d["results"] if r["name"] == "hand"), None)
    for r in d["results"]:
        m = r["test"]
        nm = METHOD_RU.get(r["name"], r["name"])
        best = "**" if base and r["name"] != "hand" and m["ndcg@10"] > base["test"]["ndcg@10"] else ""
        ci = (f"[{m['ndcg@10_lo']:.4f}, {m['ndcg@10_hi']:.4f}]"
              if "ndcg@10_lo" in m else "—")
        lines.append(
            f"| {nm} | {r['n_features']:,} | {_fmt(m['auc'])} | {_fmt(m['logloss'], 5)} | "
            f"{best}{_fmt(m['ndcg@10'])}{best} | {ci} | {_fmt(m['recall@10'])} | "
            f"{r['feature_seconds']:.1f} | {r['human_hours']:.1f} |")
    if base:
        b = base["test"]
        lines += ["", "Прирост к ручному набору. Колонка «значимо» отвечает на "
                      "вопрос, различимы ли методы вообще: различие считается "
                      "значимым, только если доверительные интервалы NDCG@10 "
                      "не пересекаются.", "",
                  "| метод | ΔAUC | ΔNDCG@10 | ΔRecall@10 | значимо |",
                  "|---|---:|---:|---:|:---:|"]
        for r in d["results"]:
            if r["name"] == "hand":
                continue
            m = r["test"]
            if "ndcg@10_lo" in m and "ndcg@10_lo" in b:
                disjoint = (m["ndcg@10_lo"] > b["ndcg@10_hi"]
                            or m["ndcg@10_hi"] < b["ndcg@10_lo"])
                sig = "да" if disjoint else "нет"
            else:
                sig = "—"
            lines.append(
                f"| {METHOD_RU.get(r['name'], r['name'])} | {m['auc'] - b['auc']:+.4f} | "
                f"{m['ndcg@10'] - b['ndcg@10']:+.4f} | "
                f"{m['recall@10'] - b['recall@10']:+.4f} | {sig} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- e05
def section_e05() -> str:
    d = _load("e05_leak_audit.json")
    if d is None:
        return "_эксперимент ещё не запускался_\n"
    p1, p2 = d["part1"], d["part2"]
    lines = [
        "**Часть 1 — проверка отсутствия утечки.** Признаки считаются дважды: "
        "на полном логе и на логе, обрезанном по границе train.",
        "",
        "| проверено признаков | строк-запросов | признаков с отличиями | макс. отн. отличие |",
        "|---:|---:|---:|---:|",
        f"| {p1['n_features_audited']:,} | {p1['n_query_rows']:,} | "
        f"{p1['n_features_differing']} | {p1['max_relative_difference']:.1e} |",
        "",
        f"Вердикт: **{p1['verdict']}**.",
        "",
        "**Часть 2 — позитивные контроли.** Без них часть 1 ничего не "
        "доказывает: тест, не способный поймать утечку, пройдёт и на сломанном коде.",
        "",
        "| вариант | AUC | ΔAUC | NDCG@10 | ΔNDCG@10 | что это |",
        "|---|---:|---:|---:|---:|---|",
    ]
    base = p2["variants"][0]["metrics"]
    for v in p2["variants"]:
        m = v["metrics"]
        lines.append(
            f"| `{v['variant']}` | {_fmt(m['auc'])} | {m['auc'] - base['auc']:+.4f} | "
            f"{_fmt(m['ndcg@10'])} | {m['ndcg@10'] - base['ndcg@10']:+.4f} | "
            f"{v['описание']} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- e06
def section_e06() -> str:
    d = _load("e06_scaling.json")
    if d is None:
        return "_эксперимент ещё не запускался_\n"
    import numpy as np
    pts = d["points"]
    c = d["config"]
    lines = [
        f"Каталог зафиксирован ({c['n_users']:,} юзеров, {c['n_items']:,} айтемов), "
        "растёт только длина лога. Предсказания двух подходов расходятся "
        "качественно: состояние компилятора не зависит от числа событий, "
        "а жёсткие окна обязаны хранить сами события.",
        "",
        "| событий | проход, с | тыс. строк/с | состояние компилятора | состояние окон | отношение |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in pts:
        ratio = r["hand_window_state_bytes"] / max(r["compiler_peak_state_bytes"], 1)
        lines.append(
            f"| {r['n_events']:,} | {r['compiler_scan_seconds']:.1f} | "
            f"{r['compiler_rows_per_sec'] / 1e3:.0f} | "
            f"{r['compiler_peak_state_bytes'] / 2**20:.0f} МиБ | "
            f"{r['hand_window_state_bytes'] / 2**20:.0f} МиБ | {ratio:.1f}× |")
    n = np.array([r["n_events"] for r in pts], float)
    t = np.array([r["compiler_scan_seconds"] for r in pts], float)
    sc = np.array([r["compiler_peak_state_bytes"] for r in pts], float)
    hb = np.array([r["hand_window_state_bytes"] for r in pts], float)
    sl = lambda y: float(np.polyfit(np.log(n), np.log(np.maximum(y, 1)), 1)[0])
    lines += ["", f"Показатели степени в `y ~ N^a`: время прохода **N^{sl(t):.2f}** "
                  f"(линейность = 1.00), состояние компилятора **N^{sl(sc):.2f}** "
                  f"(плато = 0.00), состояние жёстких окон **N^{sl(hb):.2f}**."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- e07
def section_e07() -> str:
    d = _load("e07_budget.json")
    if d is None:
        return "_эксперимент ещё не запускался_\n"
    lines = [
        "| добавлен ключ | память состояния | обновлений/событие | признаков | AUC | NDCG@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for p in d["pareto"]:
        lines.append(
            f"| `{p['added']}` | {p['state_mib']:.1f} МиБ | {p['updates_per_event']} | "
            f"{p['n_features']} | {_fmt(p['metrics']['auc'])} | "
            f"{_fmt(p['metrics']['ndcg@10'])} |")
    h = d.get("hand_reference")
    if h:
        lines.append(
            f"| _ручной набор (окна)_ | {h['state_mib']:.1f} МиБ | — | "
            f"{h['n_features']} | {_fmt(h['metrics']['auc'])} | "
            f"{_fmt(h['metrics']['ndcg@10'])} |")
        tgt = h["metrics"]["ndcg@10"]
        hit = next((p for p in d["pareto"] if p["metrics"]["ndcg@10"] >= tgt), None)
        if hit:
            lines += ["", f"Компилятор достигает качества ручного набора на "
                          f"**{hit['state_mib']:.1f} МиБ** состояния "
                          f"({hit['n_keys_selected']} ключей) против "
                          f"**{h['state_mib']:.1f} МиБ** у жёстких окон — "
                          f"в {h['state_mib'] / max(hit['state_mib'], 1e-9):.1f} раза меньше."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- e04
def section_e04() -> str:
    d = _load("e04_vs_agent.json")
    if d is None:
        return "_эксперимент ещё не запускался_\n"
    p1 = d["part1"]
    lines = [
        f"**Вложенность.** Из {len(p1['proposals'])} признаков, которые "
        f"предлагает языковая модель для этой постановки, **{p1['n_in_grammar']} "
        f"({p1['share_in_grammar']:.0%}) уже порождаются грамматикой** "
        f"автоматически. Угадывание внутри перечисляемого пространства не может "
        f"выиграть — только совпасть.",
        "",
        "Что остаётся вне грамматики (и где агент с человеком полезны):",
        "",
    ]
    for pr in p1["proposals"]:
        if pr["status"] == "OUT":
            lines.append(f"- {pr['feature']} — _{pr['maps_to']}_")
    p2 = d.get("part2")
    if p2:
        lines += ["", f"**Угадывание против перебора с отбором** при одинаковом k "
                      f"(полное пространство {p2['space']['total']:,}, "
                      f"правдоподобное подпространство {p2['space']['plausible']:,}):",
                  "", "| k | метод | AUC | NDCG@10 |", "|---:|---|---:|---:|"]
        for r in p2["k_results"]:
            k = r["k"]
            lines.append(f"| {k} | компилятор (top-k из всех) | "
                         f"{_fmt(r['compiler']['auc'])} | {_fmt(r['compiler']['ndcg@10'])} |")
            lines.append(f"| | идеальный агент (top-k из подпространства) | "
                         f"{_fmt(r['oracle_agent']['auc'])} | "
                         f"{_fmt(r['oracle_agent']['ndcg@10'])} |")
            lines.append(f"| | агент угадывает (среднее ± σ) | "
                         f"{_fmt(r['agent_mean']['auc'])} ± {r['agent_std']['auc']:.4f} | "
                         f"{_fmt(r['agent_mean']['ndcg@10'])} ± "
                         f"{r['agent_std']['ndcg@10']:.4f} |")
    return "\n".join(lines) + "\n"


SECTIONS = {
    "e03_synthetic": lambda: section_e03("synthetic"),
    "e03_retailrocket": lambda: section_e03("retailrocket"),
    "e04": section_e04,
    "e05": section_e05,
    "e06": section_e06,
    "e07": section_e07,
}


def fill(verbose: bool = True) -> None:
    text = README.read_text(encoding="utf-8")
    for key, fn in SECTIONS.items():
        pat = re.compile(rf"(<!-- RESULTS:{key} -->)(.*?)(<!-- /RESULTS:{key} -->)", re.S)
        if not pat.search(text):
            if verbose:
                print(f"  маркер RESULTS:{key} не найден в README — пропуск")
            continue
        body = fn()
        text = pat.sub(lambda m, b=body: f"{m.group(1)}\n{b}{m.group(3)}", text)
        if verbose:
            print(f"  RESULTS:{key} подставлено ({len(body.splitlines())} строк)")
    README.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    fill()
