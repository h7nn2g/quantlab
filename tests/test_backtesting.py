
import numpy as np
import pandas as pd
import pytest

from quantlab.backtesting.engine import strategy_returns, equity_curve, maximum_drawdown, sharpe_ratio, performance_report

def test_strategy_returns():
    asset_returns = pd.Series([0.02, -0.03, 0.01, 0.04])
    positions = pd.Series([1, 0, 1, 1])

    result = strategy_returns(asset_returns, positions)
    expected = pd.Series([0.0, -0.03, 0.0, 0.04])

    pd.testing.assert_series_equal(result, expected)


def test_strategy_returns_all_cash():
    asset_returns = pd.Series([0.02, -0.03, 0.01])
    positions = pd.Series([0, 0, 0])

    result = strategy_returns(asset_returns, positions)

    assert (result == 0).all()


def test_strategy_returns_all_long():
    asset_returns = pd.Series([0.02, -0.03, 0.01])
    positions = pd.Series([1, 1, 1])

    result = strategy_returns(asset_returns, positions)

    expected = pd.Series([0.0, -0.03, 0.01])

    pd.testing.assert_series_equal(result, expected)


def test_strategy_returns_first_day_zero():
    asset_returns = pd.Series([0.05, 0.02])
    positions = pd.Series([1, 1])

    result = strategy_returns(asset_returns, positions)

    assert result.iloc[0] == pytest.approx(0.0)


def test_strategy_returns_preserves_dates():
    dates = pd.date_range("2025-01-01", periods=4)
    asset_returns = pd.Series(
        [0.02, -0.03, 0.01, 0.04], index=dates
    )
    positions = pd.Series([1, 0, 1, 1], index=dates)

    result = strategy_returns(asset_returns, positions)

    assert result.index.equals(dates)

def test_strategy_returns_uses_previous_position():
    asset_returns = pd.Series([0.10, -0.20, 0.30])
    positions = pd.Series([1, 0, 1])

    result = strategy_returns(asset_returns, positions)

    expected = pd.Series([0.0, -0.20, 0.0])

    pd.testing.assert_series_equal(result, expected)


def test_strategy_returns_no_same_day_execution():
    asset_returns = pd.Series([0.05, 0.10, -0.05])
    positions = pd.Series([0, 1, 0])

    result = strategy_returns(asset_returns, positions)

    expected = pd.Series([0.0, 0.0, -0.05])

    pd.testing.assert_series_equal(result, expected)

@pytest.mark.parametrize("values", [
    [],
    [0.01, np.nan, 0.02],
    [0.01, np.inf, 0.02],
    [0.01, -np.inf, 0.02],
])
def test_strategy_returns_invalid_asset_returns(values):
    asset_returns = pd.Series(values, dtype=float)
    positions = pd.Series([1] * len(values), dtype=int)

    with pytest.raises(ValueError):
        strategy_returns(asset_returns, positions)


@pytest.mark.parametrize("values", [
    [],
    [1, np.nan, 0],
    [1, np.inf, 0],
    [1, -np.inf, 0],
    [1, -1, 0],
    [1, 2, 0],
    [1, 0.5, 0],
])
def test_strategy_returns_invalid_positions(values):
    positions = pd.Series(values, dtype=float)
    asset_returns = pd.Series([0.01] * len(values))

    with pytest.raises(ValueError):
        strategy_returns(asset_returns, positions)

def test_strategy_returns_mismatched_dates():
    asset_returns = pd.Series(
        [0.01, 0.02, 0.03],
        index=pd.date_range("2025-01-01", periods=3)
    )
    positions = pd.Series(
        [1, 0, 1],
        index=pd.date_range("2025-01-02", periods=3)
    )

    with pytest.raises(ValueError):
        strategy_returns(asset_returns, positions)


def test_strategy_returns_reordered_dates():
    dates = pd.date_range("2025-01-01", periods=3)

    asset_returns = pd.Series(
        [0.01, 0.02, 0.03], index=dates
    )
    positions = pd.Series([1, 0, 1], index=dates[::-1])

    with pytest.raises(ValueError):
        strategy_returns(asset_returns, positions)

def test_strategy_returns_does_not_modify_inputs():
    asset_returns = pd.Series([0.01, -0.02, 0.03])
    positions = pd.Series([1, 0, 1])

    original_returns = asset_returns.copy(deep=True)
    original_positions = positions.copy(deep=True)

    strategy_returns(asset_returns, positions)

    pd.testing.assert_series_equal(
        asset_returns, original_returns
    )
    pd.testing.assert_series_equal(
        positions, original_positions
    )

def test_equity_curve_basic():
    returns = pd.Series([0.05, -0.10, 0.03])

    result = equity_curve(returns)

    expected = pd.Series([
        10500.0,
        9450.0,
        9733.5
    ])

    pd.testing.assert_series_equal(
        result, expected, check_exact=False
    )


def test_equity_curve_compounding():
    returns = pd.Series([0.10, -0.10])

    result = equity_curve(returns, 10000)

    assert result.iloc[-1] == pytest.approx(9900.0)


def test_equity_curve_constant_positive():
    returns = pd.Series([0.01, 0.01, 0.01])

    expected = 10000 * (1.01 ** 3)

    assert equity_curve(returns).iloc[-1] == pytest.approx(expected)


def test_equity_curve_zero_returns():
    returns = pd.Series([0.0, 0.0, 0.0])

    result = equity_curve(returns)

    assert (result == 10000.0).all()


def test_equity_curve_custom_capital():
    returns = pd.Series([0.05, 0.10])

    result = equity_curve(returns, initial_capital=5000)

    assert result.iloc[-1] == pytest.approx(5775.0)


def test_equity_curve_total_loss():
    returns = pd.Series([0.05, -1.0, 0.20])

    result = equity_curve(returns)

    assert result.iloc[1] == pytest.approx(0.0)
    assert result.iloc[2] == pytest.approx(0.0)


@pytest.mark.parametrize("values", [
    [],
    [0.01, np.nan, 0.02],
    [0.01, np.inf, 0.02],
    [0.01, -np.inf, 0.02],
    [0.01, -1.01, 0.02],
    [0.01, -2.0, 0.02],
])
def test_equity_curve_invalid_returns(values):
    returns = pd.Series(values, dtype=float)

    with pytest.raises(ValueError):
        equity_curve(returns)

@pytest.mark.parametrize("capital", [
    0,
    -100,
    -10000,
    np.nan,
    np.inf,
    -np.inf,
    True,
    False,
    "10000",
    None,
])
def test_equity_curve_invalid_capital(capital):
    returns = pd.Series([0.01, -0.02, 0.03])

    with pytest.raises(ValueError):
        equity_curve(returns, initial_capital=capital)


def test_equity_curve_preserves_dates():
    dates = pd.date_range("2025-01-01", periods=3)
    returns = pd.Series([0.05, -0.10, 0.03], index=dates)

    result = equity_curve(returns)

    assert result.index.equals(dates)


def test_equity_curve_does_not_modify_input():
    returns = pd.Series([0.05, -0.10, 0.03])
    original = returns.copy(deep=True)

    equity_curve(returns)

    pd.testing.assert_series_equal(returns, original)


def test_equity_curve_matches_pandas():
    returns = pd.Series([0.01, -0.02, 0.04, 0.03, -0.01])

    expected = 10000 * (1 + returns).cumprod()
    result = equity_curve(returns)

    pd.testing.assert_series_equal(
        result, expected, check_exact=False
    )

def test_maximum_drawdown_basic():
    equity = pd.Series([
        10000, 11000, 10500, 8800, 9600, 11500
    ])

    result = maximum_drawdown(equity)

    assert result == pytest.approx(-0.20)


def test_maximum_drawdown_no_drawdown():
    equity = pd.Series([10000, 11000, 12000, 13000])

    assert maximum_drawdown(equity) == pytest.approx(0.0)


def test_maximum_drawdown_constant_equity():
    equity = pd.Series([10000, 10000, 10000])

    assert maximum_drawdown(equity) == pytest.approx(0.0)


def test_maximum_drawdown_total_loss():
    equity = pd.Series([10000, 11000, 0, 0])

    assert maximum_drawdown(equity) == pytest.approx(-1.0)


def test_maximum_drawdown_recovery():
    equity = pd.Series([10000, 8000, 9000, 11000])

    assert maximum_drawdown(equity) == pytest.approx(-0.20)


def test_maximum_drawdown_multiple_peaks():
    equity = pd.Series([
        10000, 12000, 10800, 13000, 10400, 14000
    ])

    assert maximum_drawdown(equity) == pytest.approx(-0.20)


def test_maximum_drawdown_single_value():
    equity = pd.Series([10000.0])

    assert maximum_drawdown(equity) == pytest.approx(0.0)


@pytest.mark.parametrize("values", [
    [],
    [10000, np.nan, 9000],
    [10000, np.inf, 9000],
    [10000, -np.inf, 9000],
    [10000, -100, 9000],
    [0, 0, 0],
    [0, 10000, 9000],
])
def test_maximum_drawdown_invalid_equity(values):
    equity = pd.Series(values, dtype=float)

    with pytest.raises(ValueError):
        maximum_drawdown(equity)


def test_maximum_drawdown_preserves_dates():
    dates = pd.date_range("2025-01-01", periods=4)

    equity = pd.Series(
        [10000, 11000, 8800, 12000],
        index=dates
    )

    result = maximum_drawdown(equity)

    assert result == pytest.approx(-0.20)


def test_maximum_drawdown_does_not_modify_input():
    equity = pd.Series([10000, 11000, 8800, 12000])
    original = equity.copy(deep=True)

    maximum_drawdown(equity)

    pd.testing.assert_series_equal(equity, original)


def test_maximum_drawdown_matches_manual_calculation():
    equity = pd.Series([
        10000, 10500, 9800, 12000, 9000, 11500
    ])

    running_peak = equity.cummax()
    expected = (equity / running_peak - 1).min()

    assert maximum_drawdown(equity) == pytest.approx(expected)

def test_sharpe_ratio_basic():
    returns = pd.Series([0.01, -0.02, 0.03, 0.015])

    expected = (
        returns.mean() / returns.std(ddof=1)
    ) * np.sqrt(252)

    assert sharpe_ratio(returns) == pytest.approx(expected)


def test_sharpe_ratio_with_risk_free_rate():
    returns = pd.Series([0.01, -0.02, 0.03, 0.015])
    risk_free_rate = 0.05

    expected = (
        (returns.mean() - risk_free_rate / 252)
        / returns.std(ddof=1)
    ) * np.sqrt(252)

    assert sharpe_ratio(
        returns, risk_free_rate=risk_free_rate
    ) == pytest.approx(expected)


def test_sharpe_ratio_negative():
    returns = pd.Series([-0.01, -0.02, 0.005, -0.015])

    result = sharpe_ratio(returns)

    assert result < 0


def test_sharpe_ratio_zero_mean():
    returns = pd.Series([-0.01, 0.01, -0.02, 0.02])

    assert sharpe_ratio(returns) == pytest.approx(0.0)


def test_sharpe_ratio_custom_trading_days():
    returns = pd.Series([0.01, -0.02, 0.03, 0.015])

    expected = (
        returns.mean() / returns.std(ddof=1)
    ) * np.sqrt(365)

    assert sharpe_ratio(
        returns, trading_days=365
    ) == pytest.approx(expected)


def test_sharpe_ratio_negative_risk_free_rate():
    returns = pd.Series([0.01, -0.02, 0.03])
    risk_free_rate = -0.01

    expected = (
        (returns.mean() - risk_free_rate / 252)
        / returns.std(ddof=1)
    ) * np.sqrt(252)

    assert sharpe_ratio(
        returns, risk_free_rate=risk_free_rate
    ) == pytest.approx(expected)


@pytest.mark.parametrize("values", [
    [],
    [0.01],
    [0.01, np.nan, 0.02],
    [0.01, np.inf, 0.02],
    [0.01, -np.inf, 0.02],
])
def test_sharpe_ratio_invalid_returns(values):
    returns = pd.Series(values, dtype=float)

    with pytest.raises(ValueError):
        sharpe_ratio(returns)


@pytest.mark.parametrize("values", [
    [0.01, 0.01, 0.01],
    [0.0, 0.0, 0.0],
    [-0.02, -0.02, -0.02],
])
def test_sharpe_ratio_zero_volatility(values):
    returns = pd.Series(values)

    with pytest.raises(ValueError):
        sharpe_ratio(returns)


@pytest.mark.parametrize("trading_days", [
    0,
    -1,
    2.5,
    True,
    False,
    "252",
    None,
])
def test_sharpe_ratio_invalid_trading_days(trading_days):
    returns = pd.Series([0.01, -0.02, 0.03])

    with pytest.raises(ValueError):
        sharpe_ratio(returns, trading_days=trading_days)


@pytest.mark.parametrize("risk_free_rate", [
    np.nan,
    np.inf,
    -np.inf,
    True,
    False,
    "0.05",
    None,
])
def test_sharpe_ratio_invalid_risk_free_rate(risk_free_rate):
    returns = pd.Series([0.01, -0.02, 0.03])

    with pytest.raises(ValueError):
        sharpe_ratio(returns, risk_free_rate=risk_free_rate)


def test_sharpe_ratio_two_observations():
    returns = pd.Series([0.01, -0.02])

    expected = (
        returns.mean() / returns.std(ddof=1)
    ) * np.sqrt(252)

    assert sharpe_ratio(returns) == pytest.approx(expected)


def test_sharpe_ratio_preserves_input():
    returns = pd.Series([0.01, -0.02, 0.03])
    original = returns.copy(deep=True)

    sharpe_ratio(returns)

    pd.testing.assert_series_equal(returns, original)

def test_performance_report_basic():
    returns = pd.Series([0.05, -0.10, 0.03])

    report = performance_report(returns)

    assert report["Initial_capital"] == pytest.approx(10000.0)
    assert report["Final_equity"] == pytest.approx(9733.5)
    assert report["Total_return"] == pytest.approx(-0.02665)
    assert report["Annualised_return"] == pytest.approx(
        0.97335 ** (252 / 3) - 1
    )
    assert report["Maximum_drawdown"] == pytest.approx(-0.10)


def test_performance_report_keys():
    returns = pd.Series([0.05, -0.10, 0.03])

    report = performance_report(returns)

    expected_keys = {
        "Initial_capital",
        "Final_equity",
        "Total_return",
        "Annualised_return",
        "Annualised_volatility",
        "Sharpe_ratio",
        "Maximum_drawdown",
    }

    assert set(report.keys()) == expected_keys


def test_performance_report_volatility():
    returns = pd.Series([0.01, -0.02, 0.03, 0.015])

    report = performance_report(returns)

    expected = returns.std(ddof=1) * np.sqrt(252)

    assert report["Annualised_volatility"] == pytest.approx(expected)


def test_performance_report_sharpe():
    returns = pd.Series([0.01, -0.02, 0.03, 0.015])

    report = performance_report(returns)

    expected = (
        returns.mean() / returns.std(ddof=1)
    ) * np.sqrt(252)

    assert report["Sharpe_ratio"] == pytest.approx(expected)


def test_performance_report_custom_capital():
    returns = pd.Series([0.10, -0.05, 0.02])

    report = performance_report(
        returns,
        initial_capital=5000.0
    )

    expected_final = 5000 * 1.10 * 0.95 * 1.02

    assert report["Initial_capital"] == pytest.approx(5000)
    assert report["Final_equity"] == pytest.approx(expected_final)


def test_performance_report_custom_trading_days():
    returns = pd.Series([0.01, -0.02, 0.03, 0.015])

    report = performance_report(
        returns,
        trading_days=365
    )

    total_growth = (1 + returns).prod()

    expected_cagr = total_growth ** (365 / len(returns)) - 1
    expected_vol = returns.std(ddof=1) * np.sqrt(365)

    assert report["Annualised_return"] == pytest.approx(expected_cagr)
    assert report["Annualised_volatility"] == pytest.approx(expected_vol)


def test_performance_report_drawdown_from_initial_capital():
    returns = pd.Series([-0.20, 0.10, 0.05])

    report = performance_report(returns)

    # Initial 10000 -> 8000 -> 8800 -> 9240
    assert report["Maximum_drawdown"] == pytest.approx(-0.20)


def test_performance_report_matches_compounding():
    returns = pd.Series([0.03, -0.02, 0.04, -0.01])

    report = performance_report(returns)

    expected_final = 10000 * (1 + returns).prod()
    expected_total = expected_final / 10000 - 1

    assert report["Final_equity"] == pytest.approx(expected_final)
    assert report["Total_return"] == pytest.approx(expected_total)


def test_performance_report_preserves_input():
    returns = pd.Series([0.01, -0.02, 0.03])
    original = returns.copy(deep=True)

    performance_report(returns)

    pd.testing.assert_series_equal(returns, original)


@pytest.mark.parametrize("values", [
    [],
    [0.01],
    [0.01, np.nan, 0.03],
    [0.01, np.inf, 0.03],
    [0.01, -np.inf, 0.03],
    [0.01, -1.5, 0.03],
])
def test_performance_report_invalid_returns(values):
    returns = pd.Series(values, dtype=float)

    with pytest.raises(ValueError):
        performance_report(returns)


@pytest.mark.parametrize("capital", [
    0,
    -100,
    np.nan,
    np.inf,
    True,
    "10000",
    None,
])
def test_performance_report_invalid_capital(capital):
    returns = pd.Series([0.01, -0.02, 0.03])

    with pytest.raises(ValueError):
        performance_report(returns, initial_capital=capital)


@pytest.mark.parametrize("trading_days", [
    0,
    -1,
    2.5,
    True,
    "252",
    None,
])
def test_performance_report_invalid_trading_days(trading_days):
    returns = pd.Series([0.01, -0.02, 0.03])

    with pytest.raises(ValueError):
        performance_report(returns, trading_days=trading_days)