
import math

import numpy as np
import pandas as pd
import pytest

from quantlab.analytics.statistics import (
    daily_volatility,
    annualised_volatility,
    covariance,
    correlation,
)

def test_daily_volatility():
    returns = pd.Series([0.01, -0.02, 0.03])

    expected = returns.std(ddof=1)
    result = daily_volatility(returns)

    assert result == pytest.approx(expected)


def test_daily_volatility_constant_returns():
    returns = pd.Series([0.01, 0.01, 0.01])

    assert daily_volatility(returns) == pytest.approx(0.0)


def test_daily_volatility_negative_returns():
    returns = pd.Series([-0.01, -0.03, -0.02])

    expected = returns.std(ddof=1)

    assert daily_volatility(returns) == pytest.approx(expected)


@pytest.mark.parametrize(
    "values",
    [
        [],
        [0.01],
    ],
)
def test_daily_volatility_insufficient_data(values):
    returns = pd.Series(values, dtype=float)

    with pytest.raises(ValueError, match="Not enough returns"):
        daily_volatility(returns)


@pytest.mark.parametrize(
    "values",
    [
        [0.01, np.nan, 0.03],
        [0.01, np.inf, 0.03],
        [0.01, -np.inf, 0.03],
    ],
)
def test_daily_volatility_invalid_values(values):
    returns = pd.Series(values)

    with pytest.raises(ValueError):
        daily_volatility(returns)


def test_daily_volatility_two_observations():
    returns = pd.Series([0.01, -0.01])

    expected = returns.std(ddof=1)

    assert daily_volatility(returns) == pytest.approx(expected)


def test_daily_volatility_does_not_modify_input():
    returns = pd.Series([0.01, -0.02, 0.03])
    original = returns.copy(deep=True)

    daily_volatility(returns)

    pd.testing.assert_series_equal(returns, original)


def test_annualised_volatility():
    returns = pd.Series([0.01, -0.02, 0.03])

    expected = returns.std(ddof=1) * math.sqrt(252)
    result = annualised_volatility(returns)

    assert result == pytest.approx(expected)


def test_annualised_volatility_custom_days():
    returns = pd.Series([0.01, -0.02, 0.03])

    expected = returns.std(ddof=1) * math.sqrt(365)

    assert annualised_volatility(
        returns,
        trading_days=365
    ) == pytest.approx(expected)


def test_annualised_volatility_one_day():
    returns = pd.Series([0.01, -0.02, 0.03])

    expected = returns.std(ddof=1)

    assert annualised_volatility(
        returns,
        trading_days=1
    ) == pytest.approx(expected)


def test_annualised_volatility_constant_returns():
    returns = pd.Series([0.02, 0.02, 0.02])

    assert annualised_volatility(returns) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "trading_days",
    [
        0,
        -1,
        -252,
        2.5,
        True,
        False,
        "252",
        None,
    ],
)
def test_annualised_volatility_invalid_days(trading_days):
    returns = pd.Series([0.01, -0.02, 0.03])

    with pytest.raises(ValueError):
        annualised_volatility(
            returns,
            trading_days=trading_days
        )


@pytest.mark.parametrize(
    "values",
    [
        [],
        [0.01],
        [0.01, np.nan],
        [0.01, np.inf],
        [0.01, -np.inf],
    ],
)
def test_annualised_volatility_invalid_returns(values):
    returns = pd.Series(values, dtype=float)

    with pytest.raises(ValueError):
        annualised_volatility(returns)

def test_covariance_positive():
    a = pd.Series([0.01, 0.02, 0.03])
    b = pd.Series([0.02, 0.04, 0.06])

    assert covariance(a, b) == pytest.approx(0.0002)


def test_covariance_negative():
    a = pd.Series([0.01, 0.02, 0.03])
    b = pd.Series([0.06, 0.04, 0.02])

    assert covariance(a, b) == pytest.approx(-0.0002)


def test_covariance_zero():
    a = pd.Series([0.01, 0.02, 0.03])
    b = pd.Series([0.02, 0.02, 0.02])

    assert covariance(a, b) == pytest.approx(0.0)


def test_covariance_matches_pandas():
    a = pd.Series([0.01, -0.02, 0.03, 0.015])
    b = pd.Series([-0.01, 0.04, 0.02, -0.005])

    expected = a.cov(b)

    assert covariance(a, b) == pytest.approx(expected)


def test_covariance_symmetric():
    a = pd.Series([0.01, -0.03, 0.02])
    b = pd.Series([0.02, 0.01, -0.04])

    assert covariance(a, b) == pytest.approx(covariance(b, a))


def test_covariance_with_itself():
    a = pd.Series([0.01, -0.02, 0.03])

    assert covariance(a, a) == pytest.approx(a.var(ddof=1))


def test_covariance_two_observations():
    a = pd.Series([0.01, 0.03])
    b = pd.Series([0.02, 0.06])

    assert covariance(a, b) == pytest.approx(a.cov(b))


@pytest.mark.parametrize(
    "values_a, values_b",
    [
        ([], []),
        ([0.01], [0.02]),
        ([0.01, 0.02], [0.03]),
        ([0.01], [0.02, 0.03]),
    ],
)
def test_covariance_insufficient_data(values_a, values_b):
    a = pd.Series(values_a, dtype=float)
    b = pd.Series(values_b, dtype=float)

    with pytest.raises(ValueError):
        covariance(a, b)


@pytest.mark.parametrize(
    "values_a, values_b",
    [
        ([0.01, np.nan], [0.02, 0.03]),
        ([0.01, 0.02], [np.nan, 0.03]),
        ([0.01, np.inf], [0.02, 0.03]),
        ([0.01, 0.02], [-np.inf, 0.03]),
    ],
)
def test_covariance_invalid_values(values_a, values_b):
    a = pd.Series(values_a)
    b = pd.Series(values_b)

    with pytest.raises(ValueError):
        covariance(a, b)


def test_covariance_mismatched_dates():
    a = pd.Series(
        [0.01, 0.02, 0.03],
        index=pd.date_range("2026-01-01", periods=3)
    )

    b = pd.Series(
        [0.02, 0.03, 0.04],
        index=pd.date_range("2026-01-02", periods=3)
    )

    with pytest.raises(ValueError):
        covariance(a, b)


def test_covariance_reordered_dates():
    dates = pd.date_range("2026-01-01", periods=3)

    a = pd.Series([0.01, 0.02, 0.03], index=dates)
    b = pd.Series([0.03, 0.02, 0.01], index=dates[::-1])

    with pytest.raises(ValueError):
        covariance(a, b)


def test_covariance_does_not_modify_inputs():
    a = pd.Series([0.01, -0.02, 0.03])
    b = pd.Series([0.04, 0.01, -0.01])

    original_a = a.copy(deep=True)
    original_b = b.copy(deep=True)

    covariance(a, b)

    pd.testing.assert_series_equal(a, original_a)
    pd.testing.assert_series_equal(b, original_b)

def test_correlation_perfect_positive():
    a = pd.Series([0.01, 0.02, 0.03])
    b = pd.Series([0.02, 0.04, 0.06])

    assert correlation(a, b) == pytest.approx(1.0)


def test_correlation_perfect_negative():
    a = pd.Series([0.01, 0.02, 0.03])
    b = pd.Series([0.06, 0.04, 0.02])

    assert correlation(a, b) == pytest.approx(-1.0)


def test_correlation_zero():
    a = pd.Series([-0.01, 0.0, 0.01])
    b = pd.Series([0.01, -0.02, 0.01])

    assert correlation(a, b) == pytest.approx(0.0)


def test_correlation_matches_pandas():
    a = pd.Series([0.01, -0.02, 0.03, 0.015])
    b = pd.Series([-0.01, 0.04, 0.02, -0.005])

    assert correlation(a, b) == pytest.approx(a.corr(b))


def test_correlation_symmetric():
    a = pd.Series([0.01, 0.03, -0.02])
    b = pd.Series([-0.01, 0.02, 0.04])

    assert correlation(a, b) == pytest.approx(correlation(b, a))


def test_correlation_with_itself():
    a = pd.Series([0.01, -0.02, 0.03])

    assert correlation(a, a) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "values_a, values_b",
    [
        ([0.01, 0.01, 0.01], [0.02, 0.03, 0.04]),
        ([0.01, 0.02, 0.03], [0.04, 0.04, 0.04]),
        ([0.01, 0.01, 0.01], [0.02, 0.02, 0.02]),
    ],
)
def test_correlation_zero_volatility(values_a, values_b):
    a = pd.Series(values_a)
    b = pd.Series(values_b)

    with pytest.raises(ValueError):
        correlation(a, b)


@pytest.mark.parametrize(
    "values_a, values_b",
    [
        ([], []),
        ([0.01], [0.02]),
        ([0.01, np.nan], [0.02, 0.03]),
        ([0.01, np.inf], [0.02, 0.03]),
    ],
)
def test_correlation_invalid_data(values_a, values_b):
    a = pd.Series(values_a, dtype=float)
    b = pd.Series(values_b, dtype=float)

    with pytest.raises(ValueError):
        correlation(a, b)


def test_correlation_mismatched_indices():
    a = pd.Series(
        [0.01, 0.02, 0.03],
        index=pd.date_range("2026-01-01", periods=3)
    )

    b = pd.Series(
        [0.02, 0.03, 0.04],
        index=pd.date_range("2026-01-02", periods=3)
    )

    with pytest.raises(ValueError):
        correlation(a, b)


def test_correlation_does_not_modify_inputs():
    a = pd.Series([0.01, 0.02, 0.03])
    b = pd.Series([0.03, -0.01, 0.02])

    original_a = a.copy(deep=True)
    original_b = b.copy(deep=True)

    correlation(a, b)

    pd.testing.assert_series_equal(a, original_a)
    pd.testing.assert_series_equal(b, original_b)
