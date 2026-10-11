"""
Тесты метрик и бутстрэпа.

Главное здесь — что быстрый путь бутстрэпа не разошёлся с определением.
NDCG, Recall и MRR считаются как средние по группам, поэтому их бутстрэп
сводится к ресэмплингу готовых чисел. Эта оптимизация даёт ускорение в
тысячи раз, и ровно поэтому её надо проверять: ошибка в ней была бы
незаметна и испортила бы все доверительные интервалы в отчёте.
"""
from __future__ import annotations

import numpy as np
import pytest

from afe.eval import auc, bootstrap_ci, per_group_values, ranking_metrics


def make_groups(n_groups=300, k=20, seed=0, signal=0.6):
    rng = np.random.default_rng(seed)
    group = np.repeat(np.arange(n_groups), k)
    y = np.zeros(n_groups * k, np.int8)
    y[::k] = 1
    score = rng.random(n_groups * k) + y * signal
    return y, score, group


@pytest.mark.parametrize("metric", ["ndcg@10", "ndcg@3", "recall@5", "mrr"])
def test_per_group_mean_equals_direct_metric(metric):
    """Среднее по группам обязано совпасть с прямым расчётом метрики."""
    y, score, group = make_groups()
    direct = ranking_metrics(y, score, group)[metric]
    fast = per_group_values(y, score, group, metric).mean()
    assert fast == pytest.approx(direct, abs=1e-12)


def test_bootstrap_point_estimate_matches_metric():
    """Точечная оценка бутстрэпа — это сама метрика, а не среднее выборок."""
    y, score, group = make_groups()
    pt, lo, hi = bootstrap_ci(y, score, group, metric="ndcg@10", n_boot=100)
    assert pt == pytest.approx(ranking_metrics(y, score, group)["ndcg@10"], abs=1e-12)
    assert lo < pt < hi


def test_bootstrap_auc_point_estimate_matches():
    y, score, group = make_groups(n_groups=120, k=10)
    pt, lo, hi = bootstrap_ci(y, score, group, metric="auc", n_boot=40)
    assert pt == pytest.approx(auc(y, score), abs=1e-12)
    assert lo < pt < hi


def test_interval_narrows_with_more_groups():
    """Больше групп — уже интервал. Иначе бутстрэп считает не то."""
    _, _, _ = make_groups()
    widths = []
    for n in (100, 1600):
        y, score, group = make_groups(n_groups=n, k=20, seed=1)
        _, lo, hi = bootstrap_ci(y, score, group, metric="ndcg@10", n_boot=200)
        widths.append(hi - lo)
    assert widths[1] < widths[0] * 0.6


def test_ranking_metrics_handle_ties():
    """При полном совпадении скоров позитив должен получать средний ранг."""
    y = np.array([1, 0, 0, 0], np.int8)
    score = np.ones(4)
    m = ranking_metrics(y, score, np.zeros(4, np.int64), ks=(1, 2, 4))
    # ранг = 1 + 3/2 = 2.5 -> в топ-1 не попадает, в топ-4 попадает
    assert m["recall@1"] == 0.0
    assert m["recall@4"] == 1.0
    assert m["mrr"] == pytest.approx(1 / 2.5)


def test_auc_is_half_for_constant_score():
    y = np.array([1, 0, 1, 0], np.int8)
    assert auc(y, np.ones(4)) == pytest.approx(0.5)
