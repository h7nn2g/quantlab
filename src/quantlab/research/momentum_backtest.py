
import pandas as pd

from quantlab.research.market_analysis import load_assets
from quantlab.analytics.returns import simple_returns
from quantlab.strategies.momentum import (
    momentum_signal,
    generate_positions,
)
from quantlab.backtesting.engine import (
    strategy_returns,
    performance_report,
)


def run_momentum_backtest(
    ticker: str,
    start: str,
    end: str,
    lookback: int = 20,
    initial_capital: float = 10000.0,
) -> dict:

    prices_df = load_assets([ticker], start, end)
    prices = prices_df[ticker]

    momentum = momentum_signal(prices, lookback)
    positions = generate_positions(momentum)

    asset_returns = simple_returns(prices)

    asset_returns = asset_returns.loc[positions.index]

    strategy_daily_returns = strategy_returns(
        asset_returns,
        positions
    )

    momentum_report = performance_report(
        strategy_daily_returns,
        initial_capital=initial_capital
    )

    buy_and_hold_report = performance_report(
        asset_returns,
        initial_capital=initial_capital
    )

    return {
        "ticker": ticker,
        "momentum": momentum_report,
        "buy_and_hold": buy_and_hold_report,
        "momentum_signal": momentum,
        "strategy_returns": strategy_daily_returns,
        "asset_returns": asset_returns,
        "positions": positions,
    }


if __name__ == "__main__":
    result = run_momentum_backtest(
        ticker="SPY",
        start="2023-01-01",
        end="2026-01-01",
        lookback=20,
        initial_capital=10000.0,
    )

    print("\n===== MOMENTUM STRATEGY =====")
    for metric, value in result["momentum"].items():
        print(f"{metric}: {value:.4f}")

    print("\n===== BUY & HOLD =====")
    for metric, value in result["buy_and_hold"].items():
        print(f"{metric}: {value:.4f}")

    print("\n===== POSITION COUNTS =====")
    print(result["positions"].value_counts())
