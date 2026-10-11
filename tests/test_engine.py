"""Движок против определения: рекуррента должна совпадать с прямой суммой."""
from __future__ import annotations

import numpy as np
import pytest

from afe.dsl import ANY_ETYPE, FeatureSet, Primitive
from afe.engine import SEC_PER_DAY, FeatureCompiler
from afe.kernels import DAY, age_moments_reference, decayed_count_reference
from afe.log import EventLog, QuerySet, build_stream

N_ETYPES = 3


def make_log(n=4000, n_users=40, n_items=60, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(0, 200 * 86400, n)).astype(np.int64)
    user = rng.integers(0, n_users, n).astype(np.int32)
    item = rng.integers(0, n_items, n).astype(np.int32)
    etype = rng.integers(0, N_ETYPES, n).astype(np.int8)
    cat = (item % 7).astype(np.int32)
    return EventLog(
        t=t, user=user, item=item, etype=etype,
        attrs={"cat": cat},
        etype_names=["view", "cart", "purchase"],
        n_users=n_users, n_items=n_items, attr_cardinality={"cat": 7},
    )


def make_queries(log, m=300, seed=1):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(log.t), m, replace=False)
    return QuerySet(
        t=log.t[idx].astype(np.int64),
        user=log.user[idx].astype(np.int32),
        item=log.item[idx].astype(np.int32),
        y=rng.integers(0, 2, m).astype(np.int8),
        group=np.arange(m, dtype=np.int64),
        attrs={"cat": log.attrs["cat"][idx].astype(np.int32)},
    )


def compute(log, queries, prims):
    fs = FeatureSet(primitives=prims, derived=[])
    comp = FeatureCompiler(fs, n_etypes=N_ETYPES, dtype=np.float64,
                           state_dtype=np.float64)
    return comp.transform(build_stream(log, queries), verbose=False)


@pytest.mark.parametrize("halflife", [1 * DAY, 7 * DAY, 60 * DAY])
@pytest.mark.parametrize("etype", [0, 2])
def test_dcount_matches_definition(halflife, etype):
    """Затухающий счётчик == прямая сумма exp(-l*возраст) по прошлым событиям."""
    log, = (make_log(),)
    q = make_queries(log)
    p = Primitive(key=("user",), etype=etype, stat="dcount", halflife_s=halflife)
    got = compute(log, q, [p])[:, 0]

    for u in np.unique(q.user)[:8]:
        qm = q.user == u
        em = (log.user == u) & (log.etype == etype)
        # движок работает в днях -> эталон тоже в днях
        exp = decayed_count_reference(
            log.t[em] / SEC_PER_DAY, q.t[qm] / SEC_PER_DAY, halflife / SEC_PER_DAY
        )
        np.testing.assert_allclose(got[qm], exp, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("halflife", [3 * DAY, 30 * DAY])
def test_age_moments_match_definition(halflife):
    """dage_mean и dage_std == моменты, посчитанные по определению."""
    log = make_log()
    q = make_queries(log)
    prims = [
        Primitive(("user",), 1, "dcount", halflife),
        Primitive(("user",), 1, "dage_mean", halflife),
        Primitive(("user",), 1, "dage_std", halflife),
    ]
    got = compute(log, q, prims)

    for u in np.unique(q.user)[:8]:
        qm = q.user == u
        em = (log.user == u) & (log.etype == 1)
        mom = age_moments_reference(
            log.t[em] / SEC_PER_DAY, q.t[qm] / SEC_PER_DAY, halflife / SEC_PER_DAY
        )
        A, B, C = mom[:, 0], mom[:, 1], mom[:, 2]
        ok = A > 0
        np.testing.assert_allclose(got[qm, 0], A, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(got[qm, 1][ok], (B / A)[ok], rtol=1e-8, atol=1e-8)
        var = np.maximum(C[ok] / A[ok] - (B[ok] / A[ok]) ** 2, 0.0)
        np.testing.assert_allclose(got[qm, 2][ok], np.sqrt(var), rtol=1e-6, atol=1e-7)


def test_any_etype_equals_sum_over_etypes():
    """Слот 'любой тип' не хранится — он должен точно равняться сумме по типам."""
    log = make_log()
    q = make_queries(log)
    hl = 7 * DAY
    prims = [Primitive(("user", "cat"), e, "dcount", hl) for e in range(N_ETYPES)]
    prims.append(Primitive(("user", "cat"), ANY_ETYPE, "dcount", hl))
    got = compute(log, q, prims)
    np.testing.assert_allclose(got[:, :N_ETYPES].sum(axis=1), got[:, N_ETYPES],
                               rtol=1e-9, atol=1e-9)


def test_recency_and_age():
    """recency = t - последнее событие, age = t - первое событие, строго до t."""
    log = make_log()
    q = make_queries(log)
    prims = [Primitive(("item",), 0, "recency", float("inf")),
             Primitive(("item",), 0, "age", float("inf"))]
    got = compute(log, q, prims)
    for i in np.unique(q.item)[:10]:
        qm = q.item == i
        te = log.t[(log.item == i) & (log.etype == 0)] / SEC_PER_DAY
        for row, tq in zip(np.where(qm)[0], q.t[qm] / SEC_PER_DAY):
            past = te[te < tq]
            if len(past) == 0:
                assert np.isnan(got[row, 0]) and np.isnan(got[row, 1])
            else:
                assert got[row, 0] == pytest.approx(tq - past.max(), abs=1e-9)
                assert got[row, 1] == pytest.approx(tq - past.min(), abs=1e-9)


def test_no_leak_future_events_do_not_change_features():
    """ГЛАВНЫЙ ТЕСТ. Признаки запроса в момент t не должны меняться от того,
    что после t в лог дописали ещё события. Если меняются — есть утечка."""
    log = make_log(n=4000)
    cut = log.time_quantile(0.6)
    q_all = make_queries(log, m=400)
    q = q_all.select(q_all.t < cut)
    assert len(q) > 50

    prims = [
        Primitive(("user",), 0, "dcount", 7 * DAY),
        Primitive(("user", "item"), 2, "dcount", 30 * DAY),
        Primitive(("user", "cat"), ANY_ETYPE, "dage_mean", 3 * DAY),
        Primitive(("item",), 1, "recency", float("inf")),
    ]
    full = compute(log, q, prims)                 # признаки на полном логе
    truncated = compute(log.slice_until(cut), q, prims)  # только прошлое

    np.testing.assert_array_equal(np.isnan(full), np.isnan(truncated))
    m = ~np.isnan(full)
    np.testing.assert_allclose(full[m], truncated[m], rtol=1e-12, atol=1e-12)


def test_leak_detector_actually_detects_leak():
    """Позитивный контроль: если утечку внести намеренно, тест выше обязан
    её поймать. Иначе он ничего не проверяет.

    Вносим утечку самым типичным способом: сдвигаем запрос на час ВПЕРЁД,
    то есть позволяем ему увидеть события, которых в его момент не было.
    """
    log = make_log(n=4000)
    cut = log.time_quantile(0.6)
    q_all = make_queries(log, m=400)
    q = q_all.select(q_all.t < cut)
    prims = [Primitive(("user",), 0, "dcount", 7 * DAY)]

    honest = compute(log.slice_until(cut), q, prims)
    q_leaky = QuerySet(t=q.t + 7 * 86400, user=q.user, item=q.item, y=q.y,
                       group=q.group, attrs=q.attrs)
    leaky = compute(log, q_leaky, prims)
    assert not np.allclose(np.nan_to_num(honest), np.nan_to_num(leaky)), \
        "детектор утечки не видит заведомую утечку"


def test_hand_baseline_is_also_leak_free():
    """Ручной baseline должен быть так же свободен от утечки, как компилятор.

    Иначе сравнение нечестно в любую из сторон: если у baseline есть
    утечка, он завышен; если дисциплина у него строже, он занижен.
    """
    from afe.hand import HandFeatures

    log = make_log(n=4000)
    cut = log.time_quantile(0.6)
    q_all = make_queries(log, m=400)
    q = q_all.select(q_all.t < cut)

    def feats(lg):
        hf = HandFeatures(cat_col="cat", n_etypes=N_ETYPES, with_nunique=True)
        return hf.transform(build_stream(lg, q), verbose=False)

    full = feats(log)
    truncated = feats(log.slice_until(cut))
    np.testing.assert_array_equal(np.isnan(full), np.isnan(truncated))
    m = ~np.isnan(full)
    np.testing.assert_allclose(full[m], truncated[m], rtol=1e-6, atol=1e-6)


def _derived_reference(out, fs, n_prim):
    """Прямолинейный расчёт уровня 1 — эталон для блочной реализации.

    Медленный и кэш-недружественный, но очевидно соответствующий
    определению: именно с ним сверяется оптимизированный путь.
    """
    ref = np.empty((out.shape[0], len(fs.derived)), dtype=np.float64)
    for j, d in enumerate(fs.derived):
        la = np.log1p(np.maximum(out[:, d.a].astype(np.float64), 0.0))
        lb = np.log1p(np.maximum(out[:, d.b].astype(np.float64), 0.0))
        if d.op == "lift":
            lc = np.log1p(np.maximum(out[:, d.c].astype(np.float64), 0.0))
            ld = np.log1p(np.maximum(out[:, d.d].astype(np.float64), 0.0))
            ref[:, j] = la + ld - lb - lc
        else:
            ref[:, j] = la - lb
    return ref


def test_blocked_derived_matches_naive_reference():
    """Блочный уровень 1 обязан совпасть с прямолинейным расчётом.

    Переписывание было ради скорости: на реальном прогоне уровень 1 занимал
    4417 секунд против 29 секунд на сам проход по логу, потому что чтение
    столбцов из матрицы, хранящейся по строкам, било мимо кэша. После
    блочной перестановки отношение стало меньше единицы. Такая оптимизация
    обязана быть покрыта сверкой с определением, иначе ошибка в ней прошла
    бы незамеченной и испортила бы почти половину признаков.
    """
    from afe.engine import FeatureCompiler
    from afe.schema import EnumerationConfig, Schema, enumerate_features

    log = make_log(n=3000, n_users=60, n_items=80)
    q = make_queries(log, m=500)
    schema = Schema(user_cols=["user"], item_cols=["item", "cat"],
                    etype_names=log.etype_names)
    fs = enumerate_features(schema, EnumerationConfig())
    assert len(fs.derived) > 100

    comp = FeatureCompiler(fs, n_etypes=N_ETYPES, dtype=np.float64,
                           state_dtype=np.float64)
    comp.DERIVED_BLOCK = 128          # много блоков: проверяются и границы
    X = comp.transform(build_stream(log, q), verbose=False)

    n_prim = len(fs.primitives)
    ref = _derived_reference(X, fs, n_prim)
    got = X[:, n_prim:]
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10)


def test_derived_block_size_does_not_change_values():
    """Результат не должен зависеть от размера блока."""
    from afe.engine import FeatureCompiler
    from afe.schema import EnumerationConfig, Schema, enumerate_features

    log = make_log(n=2000, n_users=40, n_items=50)
    q = make_queries(log, m=300)
    schema = Schema(user_cols=["user"], item_cols=["item", "cat"],
                    etype_names=log.etype_names)
    fs = enumerate_features(schema, EnumerationConfig())
    stream = build_stream(log, q)

    outs = []
    for block in (64, 1000, 10_000_000):
        comp = FeatureCompiler(fs, n_etypes=N_ETYPES, dtype=np.float64,
                               state_dtype=np.float64)
        comp.DERIVED_BLOCK = block
        outs.append(comp.transform(stream, verbose=False))
    np.testing.assert_allclose(outs[0], outs[1], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(outs[0], outs[2], rtol=1e-12, atol=1e-12)
