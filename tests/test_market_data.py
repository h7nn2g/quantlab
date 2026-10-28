import pandas as pd
import pytest

from unittest.mock import patch
from quantlab.data.loader import validate_market_data, load_market_data

@pytest.fixture
def sample_market_data():
    return pd.DataFrame(
        {
            "Open": [100, 103, 106],
            "High": [105, 108, 110],
            "Low": [99, 101, 104],
            "Close": [103, 106, 107],
            "Volume": [10000, 12000, 11000],
        },
        index=pd.to_datetime([
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
        ])
    )



def test_valid_market_data(sample_market_data):
    validate_market_data(sample_market_data)

def test_duplicated_dates(sample_market_data):
    df = sample_market_data.copy()
    df.index = pd.to_datetime([
        "2026-01-01",
        "2026-01-01",
        "2026-01-03",
    ])

    with pytest.raises(ValueError, match="Found duplicated date"):
        validate_market_data(df)

def test_invalid_high(sample_market_data):
    df = sample_market_data.copy()
    df.loc["2026-01-02", "High"] = 105

    with pytest.raises(ValueError, match="High value is incorrect"):
        validate_market_data(df)

def test_invalid_low(sample_market_data):
    df = sample_market_data.copy()
    df.loc["2026-01-03", "Low"] = 108

    with pytest.raises(ValueError, match="Low value is incorrect"):
        validate_market_data(df)

def test_invalid_volume(sample_market_data):
    df = sample_market_data.copy()
    df.loc["2026-01-02", "Volume"] = -100000

    with pytest.raises(ValueError, match="Found invalid Volume"):
        validate_market_data(df)

def test_non_positive_price(sample_market_data):
    df = sample_market_data.copy()

    df.loc["2026-01-02", "Low"] = 0

    with pytest.raises(ValueError, match="Found non-positive OHLC prices"):
        validate_market_data(df)

def test_load_market_data(sample_market_data):
    with patch("quantlab.data.loader.yf.download") as mock_download:
        mock_download.return_value = sample_market_data

        df = load_market_data("SPY","2026-01-01", "2026-01-04")
        assert len(df) == 3
        mock_download.assert_called_once()
        mock_download.assert_called_once_with("SPY",start="2026-01-01", end="2026-01-04", auto_adjust=True)

def test_load_market_data_rejects_invalid_data(sample_market_data):
    df = sample_market_data.copy()
    df.loc["2026-01-02", "Low"] = 0

    with patch("quantlab.data.loader.yf.download") as mock_download:
        mock_download.return_value = df

        with pytest.raises(ValueError, match="Found non-positive OHLC prices"):
            load_market_data("SPY","2026-01-01", "2026-01-04")

def test_load_market_data_empty():
    df = pd.DataFrame()

    with patch("quantlab.data.loader.yf.download") as mock_download:
        mock_download.return_value = df

        with pytest.raises(ValueError, match="No data found for SPY"):
            load_market_data("SPY", "2026-01-01", "2026-01-04")


def test_load_market_data_multiindex(sample_market_data):
    df = sample_market_data.copy()

    df.columns = pd.MultiIndex.from_product(
        [df.columns, ["SPY"]]
    )

    with patch("quantlab.data.loader.yf.download") as mock_download:
        mock_download.return_value = df

        result = load_market_data("SPY", "2026-01-01", "2026-01-04")
        assert not isinstance(result.columns, pd.MultiIndex)

def test_load_market_data_reversed_multiindex(sample_market_data):
    df = sample_market_data.copy()
    
    df.columns = pd.MultiIndex.from_product(
        [["SPY"], df.columns]
    )
    
    with patch("quantlab.data.loader.yf.download") as mock_download:
        mock_download.return_value = df
    
        result = load_market_data("SPY", "2026-01-01", "2026-01-04")
        assert not isinstance(result.columns, pd.MultiIndex)