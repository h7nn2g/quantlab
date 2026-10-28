
import numpy as np
import pandas as pd
import pytest

from unittest.mock import patch

from quantlab.research.market_analysis import (
    load_assets,
    calculate_returns,
    volatility_report,
    correlation_matrix,
    covariance_matrix,
)

@pytest.fixture
def sample_prices():
    dates = pd.date_range("2025-01-01", periods=4)

    return pd.DataFrame(
        {
            "SPY": [100.0, 102.0, 101.0, 105.0],
            "QQQ": [200.0, 210.0, 205.0, 215.0],
            "TSM": [150.0, 153.0, 156.0, 154.0],
        },
        index=dates,
    )


@patch("quantlab.research.market_analysis.load_market_data")
def test_load_assets(mock_loader):
    dates = pd.date_range("2025-01-01", periods=3)

    def fake_loader(ticker, start, end):
        data = {
            "SPY": [100.0, 101.0, 102.0],
            "QQQ": [200.0, 205.0, 210.0],
            "TSM": [150.0, 152.0, 154.0],
        }

        return pd.DataFrame(
            {"Close": data[ticker]},
            index=dates,
        )

    mock_loader.side_effect = fake_loader

    result = load_assets(
        ["SPY", "QQQ", "TSM"],
        "2025-01-01",
        "2025-02-01",
    )

    assert result.shape == (3, 3)
    assert list(result.columns) == ["SPY", "QQQ", "TSM"]
    assert result.loc[dates[0], "SPY"] == 100.0
    assert result.loc[dates[2], "TSM"] == 154.0

    assert mock_loader.call_count == 3


@patch("quantlab.research.market_analysis.load_market_data")
def test_load_assets_duplicate_tickers(mock_loader):
    mock_loader.return_value = pd.DataFrame(
        {"Close": [100.0, 101.0]},
        index=pd.date_range("2025-01-01", periods=2),
    )

    result = load_assets(
        ["SPY", "QQQ", "SPY"],
        "2025-01-01",
        "2025-02-01",
    )

    assert list(result.columns) == ["SPY", "QQQ"]
    assert mock_loader.call_count == 2


@patch("quantlab.research.market_analysis.load_market_data")
def test_load_assets_aligns_dates(mock_loader):
    dates_a = pd.date_range("2025-01-01", periods=3)
    dates_b = pd.date_range("2025-01-02", periods=3)

    mock_loader.side_effect = [
        pd.DataFrame(
            {"Close": [100.0, 101.0, 102.0]},
            index=dates_a,
        ),
        pd.DataFrame(
            {"Close": [200.0, 201.0, 202.0]},
            index=dates_b,
        ),
    ]

    result = load_assets(
        ["SPY", "QQQ"],
        "2025-01-01",
        "2025-02-01",
    )

    expected_dates = pd.date_range("2025-01-02", periods=2)

    assert result.index.equals(expected_dates)
    assert result.shape == (2, 2)
    assert not result.isna().any().any()


@patch("quantlab.research.market_analysis.load_market_data")
def test_load_assets_empty_tickers(mock_loader):
    with pytest.raises(ValueError):
        load_assets([], "2025-01-01", "2025-02-01")

    mock_loader.assert_not_called()


@patch("quantlab.research.market_analysis.load_market_data")
def test_load_assets_no_overlapping_dates(mock_loader):
    mock_loader.side_effect = [
        pd.DataFrame(
            {"Close": [100.0, 101.0]},
            index=pd.date_range("2025-01-01", periods=2),
        ),
        pd.DataFrame(
            {"Close": [200.0, 201.0]},
            index=pd.date_range("2025-02-01", periods=2),
        ),
    ]

    with pytest.raises(ValueError):
        load_assets(
            ["SPY", "QQQ"],
            "2025-01-01",
            "2025-03-01",
        )


@patch("quantlab.research.market_analysis.load_market_data")
def test_load_assets_preserves_ticker_order(mock_loader):
    mock_loader.return_value = pd.DataFrame(
        {"Close": [100.0, 101.0]},
        index=pd.date_range("2025-01-01", periods=2),
    )

    result = load_assets(
        ["TSM", "SPY", "QQQ"],
        "2025-01-01",
        "2025-02-01",
    )

    assert list(result.columns) == ["TSM", "SPY", "QQQ"]

def test_calculate_returns(sample_prices):
    result = calculate_returns(sample_prices)

    expected = sample_prices.pct_change().dropna()

    pd.testing.assert_frame_equal(result, expected)


def test_calculate_returns_shape(sample_prices):
    result = calculate_returns(sample_prices)

    assert result.shape == (3, 3)
    assert list(result.columns) == ["SPY", "QQQ", "TSM"]


def test_calculate_returns_preserves_dates(sample_prices):
    result = calculate_returns(sample_prices)

    assert result.index.equals(sample_prices.index[1:])


def test_calculate_returns_no_nan(sample_prices):
    result = calculate_returns(sample_prices)

    assert not result.isna().any().any()


def test_calculate_returns_does_not_modify_input(sample_prices):
    original = sample_prices.copy(deep=True)

    calculate_returns(sample_prices)

    pd.testing.assert_frame_equal(sample_prices, original)


def test_calculate_returns_constant_prices():
    prices = pd.DataFrame(
        {
            "SPY": [100.0, 100.0, 100.0],
            "QQQ": [200.0, 200.0, 200.0],
        }
    )

    result = calculate_returns(prices)

    assert (result == 0.0).all().all()


def test_calculate_returns_negative_return():
    prices = pd.DataFrame(
        {"SPY": [100.0, 90.0, 81.0]}
    )

    result = calculate_returns(prices)

    assert result["SPY"].iloc[0] == pytest.approx(-0.10)
    assert result["SPY"].iloc[1] == pytest.approx(-0.10)

def test_volatility_report(sample_prices):
    returns = sample_prices.pct_change().dropna()

    result = volatility_report(returns)

    assert list(result.columns) == ["Annualised Volatility"]
    assert list(result.index) == list(returns.columns)

    for ticker in returns.columns:
        expected = returns[ticker].std(ddof=1) * np.sqrt(252)
        assert result.loc[ticker, "Annualised Volatility"] == pytest.approx(expected)


def test_correlation_matrix(sample_prices):
    returns = sample_prices.pct_change().dropna()

    result = correlation_matrix(returns)
    expected = returns.corr()

    pd.testing.assert_frame_equal(
        result, expected, check_exact=False, atol=1e-12
    )


def test_covariance_matrix(sample_prices):
    returns = sample_prices.pct_change().dropna()

    result = covariance_matrix(returns)
    expected = returns.cov()

    pd.testing.assert_frame_equal(
        result, expected, check_exact=False, atol=1e-12
    )


def test_matrices_are_symmetric(sample_prices):
    returns = sample_prices.pct_change().dropna()

    for matrix in [
        correlation_matrix(returns),
        covariance_matrix(returns),
    ]:
        assert np.allclose(matrix.values, matrix.values.T)


def test_covariance_diagonal_equals_variance(sample_prices):
    returns = sample_prices.pct_change().dropna()

    result = covariance_matrix(returns)

    for ticker in returns.columns:
        assert result.loc[ticker, ticker] == pytest.approx(
            returns[ticker].var(ddof=1)
        )


def test_correlation_diagonal_is_one(sample_prices):
    returns = sample_prices.pct_change().dropna()

    result = correlation_matrix(returns)

    assert np.allclose(np.diag(result), 1.0)
