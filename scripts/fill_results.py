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
    """Прочерк вместо числа, если метрики нет.

    Таблица RetailRocket восстановлена разбором лога прерванного прогона, и
    Recall@10 с MRR там отсутствуют: построчный вывод их не печатает. В
    отчёт идёт прочерк, а не выдуманное значение.
    """
    if x is None:
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


# ---------------------------------------------------------------- e03
METHOD_RU = {
    "seen": "ярлык «видел айтем» (ориентир)",
    "popularity": "популярность (ориентир)",
    "hand": "ручные фичи, жёсткие окна",
    "hand_keys_decay": "те же ключи, сетка масштабов",
    "compiler_L0": "компилятор, уровень 0",
    "compiler": "компилятор, уровни 0+1",
    "compiler_selected": "компилятор + отбор на val",
    "compiler_covisit": "+ совстречаемость (протекающая)",
    "compiler_covisit_honest": "+ совстречаемость (честная)",
}


def section_e03(dataset: str) -> str:
    d = _load(f"e03_benchmark_{dataset}.json")
    if d is None:
        return "_эксперимент ещё не запускался_\n"
    c = d["config"]
    partial = d.get("partial")
    lines = [
        f"Датасет: **{dataset}**, событий {c.get('n_events', 0):,}. "
        f"Негативов: {c['n_negatives']} на обучении, "
        f"{c.get('n_negatives_eval', c['n_negatives'])} на оценке. "
        f"Протокол негативов: `{c.get('neg_source')}`. "
        f"Бюджет тюнинга: {c['n_trials']} испытаний каждому методу. "
        f"Разрыв предсказания: {c.get('prediction_gap_days', 0):g} дн.",
        "",
        *(["> Таблица восстановлена разбором лога прерванного прогона "
           "(счёт остановлен ради экономии времени на перегревавшейся машине). "
           "`Recall@10`, `MRR` и время расчёта признаков в построчном выводе "
           "отсутствуют и показаны прочерком; метод `compiler_selected` "
           "не досчитан.", ""] if partial else []),
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
        fs_ = r.get("feature_seconds")
        fs_txt = "—" if fs_ is None or fs_ != fs_ else f"{fs_:.1f}"
        lines.append(
            f"| {nm} | {r['n_features']:,} | {_fmt(m.get('auc'))} | "
            f"{_fmt(m.get('logloss'), 5)} | "
            f"{best}{_fmt(m.get('ndcg@10'))}{best} | {ci} | "
            f"{_fmt(m.get('recall@10'))} | {fs_txt} | {r['human_hours']:.1f} |")
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
            def _d(key, m=m, b=b):     # связываем явно: иначе позднее связывание
                if key in m and key in b:
                    return f"{m[key] - b[key]:+.4f}"
                return "—"
            lines.append(
                f"| {METHOD_RU.get(r['name'], r['name'])} | {_d('auc')} | "
                f"{_d('ndcg@10')} | {_d('recall@10')} | {sig} |")
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
        f"**Часть A — длина лога.** Каталог зафиксирован ({c['n_users']:,} юзеров, "
        f"{c['n_items']:,} айтемов), растёт только число событий. "
        "Предсказывалось плато у компилятора — **предсказание не подтвердилось**: "
        "при фиксированном каталоге число уникальных *пар* user×item всё равно "
        "растёт с длиной лога.",
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
    def sl(y):
        return float(np.polyfit(np.log(n), np.log(np.maximum(y, 1)), 1)[0])

    lines += ["", f"Показатели степени в `y ~ N^a`: время прохода **N^{sl(t):.2f}** "
                  f"(линейность = 1.00), состояние компилятора **N^{sl(sc):.2f}** "
                  f"(плато было бы 0.00), состояние жёстких окон **N^{sl(hb):.2f}**.",
              "", "На этих масштабах компилятор занимает **больше** памяти, чем "
                  "жёсткие окна: он держит 72 числа на ключ, а CSR окон — 8 байт на "
                  "событие. Утверждение «O(1) на ключ против O(N) на события» верно "
                  "асимптотически, но здесь перевешивает константа."]

    pb = d.get("part_b")
    if pb:
        lines += ["", "**Часть B — горизонт признака.** Вот ось, на которой разница "
                      "принципиальна. Лог зафиксирован, растёт только самый длинный "
                      "временной масштаб.", "",
                  "| горизонт | компилятор | жёсткие окна |", "|---:|---:|---:|"]
        for r in pb["rows"]:
            lines.append(f"| {r['horizon_days']:.0f} дн. | "
                         f"{r['compiler_state_bytes'] / 2**20:.0f} МиБ | "
                         f"{r['window_state_bytes'] / 2**20:.0f} МиБ |")
        c0 = pb["rows"][0]["compiler_state_bytes"]
        cN = pb["rows"][-1]["compiler_state_bytes"]
        w0 = pb["rows"][0]["window_state_bytes"]
        wN = pb["rows"][-1]["window_state_bytes"]
        lines += ["", f"Горизонт вырос в 365×: память компилятора изменилась в "
                      f"**{cN / max(c0, 1):.2f}×**, память жёстких окон — в "
                      f"**{wN / max(w0, 1):.2f}×**. Счётчик с полупериодом в год "
                      "занимает ровно столько же, сколько с полупериодом в час, "
                      "а окну в год нужно держать события за год."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- e07
def section_e07() -> str:
    d = _load("e07_budget.json")
    if d is None:
        return "_эксперимент ещё не запускался_\n"
    lines = []
    ibk = d.get("importance_by_key")
    if ibk:
        br = next((r["share"] for r in ibk if r["key"] == "userxbrand"), None)
        pr = next((r["share"] for r in ibk if r["key"] == "userxprice_bucket"), None)
        top = ibk[0]
        lines += [
            "**Какие мета-пути метод реально использует** (суммарный gain, "
            "без поправки на стоимость):", "",
            "| ключ | доля важности |", "|---|---:|",
            *(f"| `{r['key']}` | {r['share']:.1%} |" for r in ibk), "",
            f"Ключ `{top['key']}` забирает **{top['share']:.1%}** всей важности. "
            "Остальные девять делят остаток, и всё ниже `item` лежит на уровне шума.",
        ]
        if br is not None and pr is not None:
            lines += [
                "", "**Проверка полигона, которая не прошла.** В генераторе бренд "
                "несёт сигнал (τ = 20 дней), а ценовой сегмент — чистый шум. "
                "Предсказывалось, что важность `userxbrand` заметно превысит "
                f"`userxprice_bucket`. Замер: **{br:.1%} против {pr:.1%}**, "
                f"отношение {br / max(pr, 1e-9):.1f}× — **различия нет**.",
                "", "Причина видна в той же таблице: доминирование `userxitem` "
                "оставляет вторичным мета-путям уровень шума, и на этом полигоне "
                "они неразличимы в принципе. Это ограничение синтетики, а не "
                "результат метода; доказательство пользы перебора получено на "
                "реальных данных, где есть запас по метрике.", "",
            ]
    lines += [
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


# ---------------------------------------------------------------- e08
def section_e08() -> str:
    d = _load("e08_protocol_diagnostics.json")
    if d is None:
        return "_эксперимент ещё не запускался_\n"
    a, b, c = d["part_a"], d["part_b"], d["part_c"]
    L = ["**A. Цена тихой утечки в джойне атрибутов.**", "",
         "| режим джойна | известна категория | известна цена |",
         "|---|---:|---:|"]
    for r in a["rows"]:
        L.append(f"| {r['mode']} | {r['cat_coverage']:.1%} | {r['price_coverage']:.1%} |")
    L += ["", f"Протекающий джойн «знает» категорию на **{a['extra_knowledge_cat']:.1%}** "
              "событий больше — это чистая информация из будущего, и обычный "
              "детектор утечки её не видит.", ""]

    L += ["**B. Вырожденность протокола негативов.** Что даёт ОДИН бинарный "
          "признак «взаимодействовал ли юзер с этим айтемом раньше»:", "",
          "| негативы | AUC | NDCG@10 | негативов знакомо пользователю |",
          "|---|---:|---:|---:|"]
    for r in b["rows"]:
        L.append(f"| `{r['neg_source']}` | {r['auc_seen']:.4f} | "
                 f"{r['ndcg10_seen']:.4f} | {r['share_negatives_seen']:.1%} |")
    pop = b["rows"][0]
    L += ["", f"При негативах по популярности тривиальный признак даёт NDCG@10 = "
              f"**{pop['ndcg10_seen']:.4f}**. Любое число рядом с этим на таком "
              "протоколе не говорит о качестве модели ничего.", ""]

    L += ["**C. Внутрисессионная тривиальность цели.** "
          f"Из {c['n_purchases']:,} покупок у **{c['share_with_prior_touch']:.1%}** "
          "было предыдущее касание ТОГО ЖЕ айтема. "
          f"Медианный разрыв — **{c['median_gap_minutes']:.1f} минут**.", "",
          "| разрыв между покупкой и предыдущим касанием | доля всех покупок |",
          "|---|---:|"]
    for x in c["distribution"]:
        L.append(f"| {x['label']} | {x['share_of_all_purchases']:.1%} |")
    L += ["", "Эффект разрыва предсказания на тот же признак:", "",
          "| разрыв | AUC | NDCG@10 | у позитива есть история по айтему |",
          "|---|---:|---:|---:|"]
    for g in c["gap_effect"]:
        L.append(f"| {g['gap_days']} дн. | {g['auc_seen']:.4f} | "
                 f"{g['ndcg10_seen']:.4f} | {g['positives_with_history']:.1%} |")
    L += ["", "При разрыве в сутки ярлык исчезает полностью: AUC становится "
              "случайным. Только тогда задача превращается в рекомендацию."]
    return "\n".join(L) + "\n"


SECTIONS = {
    "e08": section_e08,
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
