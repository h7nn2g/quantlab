import pandas as pd
import numpy as np
from math import sqrt

from quantlab.analytics.statistics import daily_volatility, annualised_volatility

def strategy_returns(asset_returns: pd.Series, positions: pd.Series) -> pd.Series:
    if asset_returns.empty or not np.isfinite(asset_returns).all():
        raise ValueError("Invalid asset returns")

    if positions.empty or not np.isfinite(positions).all():
        raise ValueError("Invalid positions")

    if not positions.isin([0, 1]).all():
        raise ValueError("Positions must be either 0 or 1")

    if not asset_returns.index.equals(positions.index):
        raise ValueError("Asset returns and positions indices do not match")

    returns = asset_returns * positions.shift(1).fillna(0)
    return returns

def equity_curve(returns: pd.Series, initial_capital: float = 10000.0) -> pd.Series:
    if returns.empty or not np.isfinite(returns).all():
        raise ValueError("Invalid returns")

    if (type(initial_capital) not in (int, float)) or initial_capital <= 0 or not np.isfinite(initial_capital):
        raise ValueError("Invalid Initial capital")

    if (returns < -1).any():
        raise ValueError("Returns cannot be below -100%")

    growth_factors = 1 + returns
    growth_factors = growth_factors.cumprod()
    equity_curve = growth_factors * initial_capital
    return equity_curve

def maximum_drawdown(equity: pd.Series) -> float:
    if equity.empty or not np.isfinite(equity).all():
        raise ValueError("Invalid equity")
    if (equity < 0).any() or equity.iloc[0] <= 0:
        raise ValueError("Equity must start positive and remain non-negative")
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1
    
    return float(drawdown.min())

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, trading_days: int = 252) -> float:
    if type(trading_days) is not int or trading_days <= 0:
        raise ValueError("Invalid number of trading days")

    if len(returns) < 2:
        raise ValueError("Not enough returns")

    if type(risk_free_rate) not in (float, int) or not np.isfinite(risk_free_rate):
        raise ValueError("Invalid risk free rate")

    mean_return = returns.mean()
    dv = daily_volatility(returns)

    if dv == 0:
        raise ValueError("Sharpe Ratio undefined for zero volatility")

    sharpe_ratio = (mean_return - risk_free_rate / trading_days) / dv * sqrt(trading_days)
    return float(sharpe_ratio)

def performance_report(returns: pd.Series, initial_capital: float = 10000.0, trading_days: int = 252) -> dict:
    report = {}
    equity = equity_curve(returns, initial_capital)
    final_equity = equity.iloc[-1]

    volatility = annualised_volatility(returns, trading_days)

    total_return = final_equity / initial_capital - 1
    annualised_return = (final_equity / initial_capital) ** (trading_days / len(returns)) - 1

    drawdown_equity = maximum_drawdown(pd.concat([pd.Series([initial_capital]), equity.reset_index(drop=True)], ignore_index=True))

    sharpe = sharpe_ratio(returns, 0, trading_days)

    report["Initial_capital"] = initial_capital
    report["Final_equity"] = final_equity
    report["Total_return"] = total_return
    report["Annualised_return"] = annualised_return
    report["Annualised_volatility"] = volatility
    report["Sharpe_ratio"] = sharpe
    report["Maximum_drawdown"] = drawdown_equity

    return report

returns = pd.Series([0.05, -0.10, 0.03])
report = performance_report(returns)

print(report)