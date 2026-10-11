"""
Тесты AS-OF джойна атрибутов айтема.

Это отдельная и неочевидная форма утечки. В RetailRocket свойства айтема
временные: категория, цена и доступность меняются. Естественный джойн
«айтем -> последнее известное значение» подставляет событиям из января
категорию, присвоенную в апреле.

Такая утечка не ловится обычной проверкой агрегатов («не смотрим события
после t»), потому что происходит не в агрегатах, а в джойне атрибутов.
Поэтому на неё нужен свой тест.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afe.datasets.retailrocket import (RetailRocketConfig, _asof_values,
                                       _encode, _parse_numeric)


def _events(ts, items):
    return pd.DataFrame({"t": ts, "itemid": items})


def test_asof_takes_value_known_at_event_time():
    """Событие берёт последнее значение с временем <= своего, и ничего позже."""
    ev = _events([10, 20, 30, 40], [1, 1, 1, 2])
    props = pd.DataFrame({"t": [5, 25], "itemid": [1, 1], "value": ["A", "B"]})
    got = _asof_values(ev, props, RetailRocketConfig())
    assert got[0] == "A"            # t=10: известно только A из t=5
    assert got[1] == "A"            # t=20: B из t=25 ещё НЕ известно
    assert got[2] == "B"            # t=30: B уже известно
    assert got[3] is None or (isinstance(got[3], float) and np.isnan(got[3]))


def test_asof_returns_missing_before_first_record():
    """До первой записи свойства значение неизвестно — а не «ближайшее будущее»."""
    ev = _events([1, 2], [7, 7])
    props = pd.DataFrame({"t": [100], "itemid": [7], "value": ["X"]})
    got = _asof_values(ev, props, RetailRocketConfig())
    for v in got:
        assert v is None or (isinstance(v, float) and np.isnan(v))


def test_leaky_mode_actually_leaks():
    """Позитивный контроль: протекающий режим обязан показывать будущее.

    Без этой проверки тест выше не доказывает, что мы вообще способны
    отличить корректный джойн от протекающего.
    """
    ev = _events([10, 20, 30], [1, 1, 1])
    props = pd.DataFrame({"t": [5, 25], "itemid": [1, 1], "value": ["A", "B"]})
    leaky = _asof_values(ev, props, RetailRocketConfig(use_asof_properties=False))
    assert list(leaky) == ["B", "B", "B"], "протекающий режим не воспроизводит утечку"
    honest = _asof_values(ev, props, RetailRocketConfig())
    assert honest[0] != leaky[0], "корректный и протекающий режимы совпали"


def test_asof_does_not_mix_items():
    """Значения не должны перетекать между айтемами (проверка параметра by)."""
    ev = _events([10, 10], [1, 2])
    props = pd.DataFrame({"t": [5, 5], "itemid": [1, 2], "value": ["ONE", "TWO"]})
    got = _asof_values(ev, props, RetailRocketConfig())
    assert got[0] == "ONE" and got[1] == "TWO"


def test_asof_preserves_event_order():
    """Результат возвращается в исходном порядке строк, а не в порядке сортировки."""
    ev = _events([30, 10, 20], [1, 1, 1])
    props = pd.DataFrame({"t": [5, 25], "itemid": [1, 1], "value": ["A", "B"]})
    got = _asof_values(ev, props, RetailRocketConfig())
    assert list(got) == ["B", "A", "A"]


def test_encode_maps_missing_to_minus_one():
    codes = _encode(pd.Series(["a", None, "b", "a", np.nan]))
    assert codes.dtype == np.int32
    assert codes[1] == -1 and codes[4] == -1
    assert codes[0] == codes[3] and codes[0] >= 0 and codes[2] >= 0


@pytest.mark.parametrize("raw,expected", [
    ("n123.000", 123.0),
    ("1116713 n277.200 424566", 277.2),
    ("не число", None),
    (None, None),
])
def test_parse_numeric_properties(raw, expected):
    """Числовые свойства RetailRocket закодированы токенами вида 'n123.000'."""
    got = _parse_numeric(pd.Series([raw]))[0]
    if expected is None:
        assert np.isnan(got)
    else:
        assert got == pytest.approx(expected)
