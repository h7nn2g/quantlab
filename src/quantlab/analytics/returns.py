import pandas as pd
import numpy as np

def simple_returns(prices: pd.Series) -> pd.Series:
    if prices.empty or prices.isna().any() or (prices <= 0).any():
        raise ValueError("Prices are incorrect")

    previous_prices = prices.shift(1)
    returns = pd.Series(prices / previous_prices - 1)

    returns = returns.dropna()
    return returns

def log_returns(prices: pd.Series) -> pd.Series:
    if prices.empty or prices.isna().any() or (prices <= 0).any():
        raise ValueError("Prices are incorrect")

    previous_prices = prices.shift(1)

    returns = np.log(prices / previous_prices)
    
    returns = returns.dropna()
    return returns