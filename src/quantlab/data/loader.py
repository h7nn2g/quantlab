import yfinance as yf
import pandas as pd

def validate_market_data(df: pd.DataFrame) -> None:
    if df.index.duplicated().any():
        raise ValueError("Found duplicated date")
    
    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Found non-positive OHLC prices")

    if (df["High"] < df[["Open", "Low", "Close"]].max(axis=1)).any():
        raise ValueError("High value is incorrect")

    if (df["Low"] > df[["Open", "High", "Close"]].min(axis=1)).any():
        raise ValueError("Low value is incorrect")

    if (df["Volume"] < 0).any():
        raise ValueError("Found invalid Volume")

def load_market_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    
    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True
    )

    if isinstance(data.columns, pd.MultiIndex):

        required = {"Open", "High", "Low", "Close", "Volume"}

        level_0 = set(data.columns.get_level_values(0))
        level_1 = set(data.columns.get_level_values(1))

        if required.issubset(level_0):
            data.columns = data.columns.droplevel(1)
            
        elif required.issubset(level_1):
            data.columns = data.columns.droplevel(0)

        else:
            raise ValueError("Invalid MultiIndex format")

    df = data.sort_index(ascending=True)

    if df.empty:
        raise ValueError(f"No data found for {ticker}")

    if df.isnull().values.any():
        raise ValueError(f"Some missing data for {ticker}")

    validate_market_data(df)

    return df

def save_as_csv(data: pd.DataFrame):
    data.to_csv("data/raw/spy.csv")