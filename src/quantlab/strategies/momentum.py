import pandas as pd
import numpy as np

def momentum_signal(prices: pd.Series, lookback: int = 20) -> pd.Series:
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Invalid prices")
    
    if type(lookback) is not int or lookback <= 0:
        raise ValueError("Invalid lookback")

    if len(prices) <= lookback:
        raise ValueError("Not enough price observations")

    momentum = prices / prices.shift(lookback) - 1
    return momentum.dropna()

def generate_positions(momentum: pd.Series) -> pd.Series:
    if momentum.empty or not np.isfinite(momentum).all():
        raise ValueError("Invalid momentums")
    # positions = []
    # for moment in momentum:
    #     if moment > 0:
    #         positions.append(1)
    #     else:
    #         positions.append(0)
    # return pd.Series(positions, index=momentum.index)
    # Vectorised operation
    return (momentum > 0).astype(int)