
import numpy as np
import pandas as pd
import pytest

from quantlab.strategies.momentum import momentum_signal, generate_positions


def test_momentum_positive():
    prices = pd.Series([100, 102, 105, 103, 110])

    result = momentum_signal(prices, lookback=2)
    expected = prices.pct_change(periods=2).dropna()

    pd.testing.assert_series_equal(result, expected)


def test_momentum_negative():
    prices = pd.Series([110, 105, 100, 95])

    result = momentum_signal(prices, lookback=2)

    assert (result < 0).all()


def test_momentum_constant_prices():
    prices = pd.Series([100, 100, 100, 100])

    result = momentum_signal(prices, lookback=2)

    assert (result == 0).all()


def test_momentum_default_lookback():
    prices = pd.Series(np.arange(100, 130, dtype=float))

    result = momentum_signal(prices)

    assert len(result) == 10
    assert result.iloc[0] == pytest.approx(120 / 100 - 1)


def test_momentum_preserves_dates():
    dates = pd.date_range("2025-01-01", periods=5)
    prices = pd.Series([100, 102, 105, 103, 110], index=dates)

    result = momentum_signal(prices, lookback=2)

    assert result.index.equals(dates[2:])


@pytest.mark.parametrize("lookback", [
    0, -1, 2.5, True, False, "20", None
])
def test_momentum_invalid_lookback(lookback):
    prices = pd.Series([100, 101, 102, 103])

    with pytest.raises(ValueError):
        momentum_signal(prices, lookback=lookback)


@pytest.mark.parametrize("prices", [
    [],
    [100],
    [100, 101],
])
def test_momentum_insufficient_prices(prices):
    series = pd.Series(prices, dtype=float)

    with pytest.raises(ValueError):
        momentum_signal(series, lookback=2)


@pytest.mark.parametrize("values", [
    [100, np.nan, 105, 110],
    [100, np.inf, 105, 110],
    [100, -np.inf, 105, 110],
    [100, 0, 105, 110],
    [100, -5, 105, 110],
])
def test_momentum_invalid_prices(values):
    prices = pd.Series(values, dtype=float)

    with pytest.raises(ValueError):
        momentum_signal(prices, lookback=2)


def test_momentum_does_not_modify_input():
    prices = pd.Series([100, 102, 105, 103, 110])
    original = prices.copy(deep=True)

    momentum_signal(prices, lookback=2)

    pd.testing.assert_series_equal(prices, original)

def test_generate_positions():
    momentum = pd.Series([0.05, 0.02, -0.03, 0.0, 0.04])

    result = generate_positions(momentum)
    expected = pd.Series([1, 1, 0, 0, 1])

    pd.testing.assert_series_equal(result, expected)


def test_positions_preserve_dates():
    dates = pd.date_range("2025-01-01", periods=3)
    momentum = pd.Series([0.05, -0.02, 0.01], index=dates)

    result = generate_positions(momentum)

    assert result.index.equals(dates)


def test_positions_empty():
    momentum = pd.Series([], dtype=float)

    with pytest.raises(ValueError):
        generate_positions(momentum)


@pytest.mark.parametrize("values", [
    [0.05, np.nan, 0.01],
    [0.05, np.inf, 0.01],
    [0.05, -np.inf, 0.01],
])
def test_positions_invalid_values(values):
    momentum = pd.Series(values)

    with pytest.raises(ValueError):
        generate_positions(momentum)


def test_positions_does_not_modify_input():
    momentum = pd.Series([0.05, -0.02, 0.01])
    original = momentum.copy(deep=True)

    generate_positions(momentum)

    pd.testing.assert_series_equal(momentum, original)
