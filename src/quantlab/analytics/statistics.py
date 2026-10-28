import pandas as pd
import numpy as np
import math

def daily_volatility(returns: pd.Series) -> float:
    if len(returns) < 2:
        raise ValueError("Not enough returns")

    if not np.isfinite(returns).all():
        raise ValueError("NaN or infite values were found")

    mean = returns.mean()

    squared_deviations = (returns - mean) ** 2
    variance = squared_deviations.sum() / (len(returns) - 1)

    return math.sqrt(variance)

def annualised_volatility(returns: pd.Series, trading_days: int = 252) -> float:
    if type(trading_days) is not int or trading_days <= 0:
        raise ValueError("Amount of traiding days is incorrect")
    return math.sqrt(trading_days) * daily_volatility(returns)

def covariance(returns_a: pd.Series, returns_b: pd.Series) -> float:
    if len(returns_a) < 2 or len(returns_b) < 2:
            raise ValueError("Not enough returns")

    if not np.isfinite(returns_a).all() or not np.isfinite(returns_b).all():
            raise ValueError("NaN or infite values were found")

    if not returns_a.index.equals(returns_b.index):
        raise ValueError("Returns indices do not match")

    mean_a = returns_a.mean()
    mean_b = returns_b.mean()

    deviations = (returns_a - mean_a) * (returns_b - mean_b)
    cov = deviations.sum() / (len(deviations) - 1)

    return float(cov)

def correlation(returns_a: pd.Series, returns_b: pd.Series) -> float:
    vol_a = daily_volatility(returns_a)
    vol_b = daily_volatility(returns_b)

    if vol_a == 0 or vol_b == 0:
        raise ValueError("Correlation undefined for zero volatility")

    corr = covariance(returns_a, returns_b) / (vol_a * vol_b)
    return corr
