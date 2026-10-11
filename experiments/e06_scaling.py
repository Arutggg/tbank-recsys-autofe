"""
Эксперимент 6. Масштабирование: откуда берётся применимость к большим логам.

Задание прямо формулирует претензию к идее с трансформером: академические
датасеты — миллион интеракций, а в проде миллиард и больше, поэтому
«выглядит интересно, но с большими вопросами по масштабируемости».

Здесь это проверяется измерением, а не рассуждением. Дизайн выбран так,
чтобы ответ был однозначным: ЧИСЛО ЮЗЕРОВ И АЙТЕМОВ ФИКСИРОВАНО, растёт
только число событий.

При таком дизайне предсказания двух подходов расходятся качественно:

  компилятор   состояние = (число ключей) x (типов) x (ядер) x 3 чисел.
               От числа событий НЕ зависит. Значит память должна выйти на
               ПЛАТО, а время — расти линейно.

  жёсткие окна события внутри окна нужно хранить, чтобы уметь их
               посчитать. Память должна расти ЛИНЕЙНО по событиям.

Если плато есть — метод применим к логу любой длины при фиксированном
каталоге, и это свойство структуры, а не результат оптимизации кода.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afe.datasets.synthetic import SyntheticConfig, generate  # noqa: E402
from afe.engine import FeatureCompiler, _factorize_key  # noqa: E402
from afe.eval import TaskConfig, build_queries  # noqa: E402
from afe.hand import _build_csr  # noqa: E402
from afe.log import build_stream  # noqa: E402
from afe.schema import EnumerationConfig, Schema, enumerate_features  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports"

#: Каталог и аудитория ФИКСИРОВАНЫ во всех точках — меняется только длина лога.
N_USERS = 50_000
N_ITEMS = 10_000
N_QUERIES_POSITIVES = 2_000      # мало: измеряем стоимость прохода, не вывода


def hand_state_bytes(stream, keys) -> int:
    """Память, которую требуют жёсткие окна: события надо хранить.

    CSR-раскладка (offsets + индексы событий) — это минимум, необходимый,
    чтобы уметь посчитать окно произвольной ширины. Отсюда линейный рост.
    """
    total = 0
    for key in keys:
        key_id, n_keys = _factorize_key(stream, key)
        offsets, idx = _build_csr(key_id, stream.is_query, n_keys, stream.size)
        total += offsets.nbytes + idx.nbytes
        del key_id, offsets, idx
        gc.collect()
    return total


def run_point(n_events: int, verbose: bool = True) -> dict:
    cfg = SyntheticConfig(n_events=n_events, n_users=N_USERS, n_items=N_ITEMS,
                          span_days=365.0, seed=0)
    t0 = time.perf_counter()
    log, _ = generate(cfg)
    gen_s = time.perf_counter() - t0

    task = TaskConfig(target_etype=2, n_negatives=9,
                      max_positives=N_QUERIES_POSITIVES, seed=0)
    t_cut = log.time_quantile(0.8)
    q = build_queries(log, t_cut, int(log.t[-1]) + 1, task, train_t_end=t_cut)

    t0 = time.perf_counter()
    stream = build_stream(log, q)
    stream_s = time.perf_counter() - t0

    schema = Schema(user_cols=["user"],
                    item_cols=["item", "cat", "brand", "price_bucket"],
                    etype_names=log.etype_names)
    fs = enumerate_features(schema, EnumerationConfig())
    comp = FeatureCompiler(fs, n_etypes=log.n_etypes)

    t0 = time.perf_counter()
    X = comp.transform(stream, verbose=False)
    comp_s = time.perf_counter() - t0
    st = dict(comp.stats)
    del X
    gc.collect()

    hand_keys = [("user",), ("item",), ("user", "item"), ("user", "cat")]
    hb = hand_state_bytes(stream, hand_keys)

    # фактическое число различных ключей: от него зависит состояние
    n_pairs = len(np.unique(log.user.astype(np.int64) * log.n_items + log.item))

    row = {
        "n_events": n_events,
        "n_distinct_user_item_pairs": int(n_pairs),
        "generate_seconds": gen_s,
        "build_stream_seconds": stream_s,
        "compiler_total_seconds": comp_s,
        "compiler_scan_seconds": st["scan_seconds"],
        "compiler_rows_per_sec": st["rows_per_sec"],
        "compiler_peak_state_bytes": st["peak_state_bytes"],
        "hand_window_state_bytes": hb,
        "n_features": fs.n_total,
    }
    if verbose:
        print(f"  {n_events:>12,} событий | проход {st['scan_seconds']:>6.1f} с "
              f"({st['rows_per_sec'] / 1e3:>6.0f} тыс.строк/с) | "
              f"состояние компилятора {st['peak_state_bytes'] / 2**20:>7.0f} МиБ | "
              f"состояние окон {hb / 2**20:>7.0f} МиБ | "
              f"пар user-item {n_pairs / 1e6:>5.2f} млн", flush=True)
    del log, q, stream
    gc.collect()
    return row


def run(sizes=(500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000),
        verbose=True) -> dict:
    """Верхняя точка 8 млн событий подобрана под 16 ГБ RAM.

    На 8 млн: лог ~260 МиБ, поток вдвое больше, состояние компилятора
    ~1.2 ГиБ (ключ user x item даёт несколько миллионов пар), CSR жёстких
    окон ~260 МиБ. С запасом помещается; 10 млн уже упирается в предел,
    а вывод о плато виден и так — важен не абсолютный размер, а показатель
    степени в зависимости состояния от длины лога.
    """
    if verbose:
        print(f"каталог фиксирован: {N_USERS:,} юзеров, {N_ITEMS:,} айтемов, "
              f"период 365 дней\nрастёт только длина лога\n")
    rows = [run_point(n, verbose) for n in sizes]
    return {"config": {"n_users": N_USERS, "n_items": N_ITEMS,
                       "n_query_positives": N_QUERIES_POSITIVES},
            "points": rows}


def report(res: dict) -> None:
    rows = res["points"]
    W = 96
    print("\n" + "=" * W)
    print("МАСШТАБИРОВАНИЕ: каталог фиксирован, растёт только длина лога")
    print("=" * W)
    print(f"{'события':>13}{'проход,с':>11}{'тыс.стр/с':>11}"
          f"{'состояние комп.':>18}{'состояние окон':>17}{'отношение':>11}")
    print("-" * W)
    for r in rows:
        ratio = r["hand_window_state_bytes"] / max(r["compiler_peak_state_bytes"], 1)
        print(f"{r['n_events']:>13,}{r['compiler_scan_seconds']:>11.1f}"
              f"{r['compiler_rows_per_sec'] / 1e3:>11.0f}"
              f"{r['compiler_peak_state_bytes'] / 2**20:>15.0f} МиБ"
              f"{r['hand_window_state_bytes'] / 2**20:>14.0f} МиБ"
              f"{ratio:>10.1f}x")
    print("-" * W)

    n = np.array([r["n_events"] for r in rows], dtype=np.float64)
    t = np.array([r["compiler_scan_seconds"] for r in rows])
    sc = np.array([r["compiler_peak_state_bytes"] for r in rows], dtype=np.float64)
    hb = np.array([r["hand_window_state_bytes"] for r in rows], dtype=np.float64)

    def slope(x, y):
        """Показатель степени в y ~ x^a: наклон в логарифмических осях."""
        return float(np.polyfit(np.log(x), np.log(np.maximum(y, 1)), 1)[0])

    print(f"  время прохода:           ~ N^{slope(n, t):.2f}   "
          f"(линейность = 1.00)")
    print(f"  состояние компилятора:   ~ N^{slope(n, sc):.2f}   "
          f"(плато = 0.00)")
    print(f"  состояние жёстких окон:  ~ N^{slope(n, hb):.2f}   "
          f"(линейный рост = 1.00)")
    print()
    growth_sc = sc[-1] / sc[0]
    growth_hb = hb[-1] / hb[0]
    growth_n = n[-1] / n[0]
    print(f"  при росте лога в {growth_n:.0f}x:")
    print(f"    состояние компилятора выросло в {growth_sc:.2f}x")
    print(f"    состояние жёстких окон выросло в {growth_hb:.2f}x")
    print("=" * W)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000])
    a = ap.parse_args()
    res = run(sizes=tuple(a.sizes))
    report(res)
    OUT.mkdir(exist_ok=True)
    (OUT / "e06_scaling.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("результаты: reports/e06_scaling.json")
