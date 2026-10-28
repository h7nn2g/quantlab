import pandas as pd
import pytest
import numpy as np

from quantlab.analytics.returns import simple_returns, log_returns


def test_simple_returns():
    prices = pd.Series([50, 60, 54])

    result = simple_returns(prices)

    expected = pd.Series(
        [0.20, -0.10],
        index=[1,2]
    )

    pd.testing.assert_series_equal(result, expected)

def test_zero_prices():
    prices = pd.Series([0, 12, 15])

    with pytest.raises(ValueError, match="Prices are incorrect"):
        simple_returns(prices)

def test_empty_prices():
    prices = pd.Series(dtype=float)

    with pytest.raises(ValueError, match="Prices are incorrect"):
        simple_returns(prices)

def test_simple_returns_index():
    prices = pd.Series(
        [100, 105, 102],
        index=pd.to_datetime([
            "2026-01-01",
            "2026-01-02",
            "2026-01-03"
        ])
    )

    result = simple_returns(prices)
    assert result.index.equals(prices.index[1:])


def test_single_price():
    prices = pd.Series([100])

    result = simple_returns(prices)
    assert result.empty

def test_log_returns():
    prices = pd.Series([100, 110, 99])

    result = log_returns(prices)

    expected = pd.Series(
        [np.log(1.1), np.log(0.9)],
        index=[1, 2]
    )

    pd.testing.assert_series_equal(result, expected)


def test_log_returns_constant_prices():
    prices = pd.Series([100, 100, 100])

    result = log_returns(prices)

    expected = pd.Series([0.0, 0.0], index=[1, 2])

    pd.testing.assert_series_equal(result, expected)


def test_log_returns_index():
    prices = pd.Series(
        [100, 110, 99],
        index=pd.to_datetime([
            "2026-01-01",
            "2026-01-02",
            "2026-01-03"
        ])
    )

    result = log_returns(prices)

    assert result.index.equals(prices.index[1:])


def test_log_returns_empty():
    prices = pd.Series(dtype=float)

    with pytest.raises(ValueError, match="Prices are incorrect"):
        log_returns(prices)


def test_log_returns_zero_price():
    prices = pd.Series([100, 0, 110])

    with pytest.raises(ValueError, match="Prices are incorrect"):
        log_returns(prices)


def test_log_returns_negative_price():
    prices = pd.Series([100, -50, 110])

    with pytest.raises(ValueError, match="Prices are incorrect"):
        log_returns(prices)


def test_log_returns_nan():
    prices = pd.Series([100, np.nan, 110])

    with pytest.raises(ValueError, match="Prices are incorrect"):
        log_returns(prices)


def test_log_returns_single_price():
    prices = pd.Series([100])

    result = log_returns(prices)

    assert result.empty


def test_log_returns_does_not_modify_input():
    prices = pd.Series([100, 110, 99])
    original = prices.copy(deep=True)

    log_returns(prices)

    pd.testing.assert_series_equal(prices, original)


def test_log_returns_additivity():
    prices = pd.Series([100, 120, 110, 125])

    result = log_returns(prices)

    expected_total = np.log(125 / 100)

    assert np.isclose(result.sum(), expected_total)


def test_log_returns_relation_to_simple():
    prices = pd.Series([100, 110, 99])

    log_result = log_returns(prices)
    simple_result = prices.pct_change().dropna()

    expected = np.log1p(simple_result)

    pd.testing.assert_series_equal(
        log_result,
        expected,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12
    )