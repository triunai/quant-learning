"""Shared pytest fixtures for quant-learning tests."""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch


@pytest.fixture
def sample_price_data():
    """Generate sample OHLCV price data for testing."""
    dates = pd.date_range(start='2020-01-01', periods=252, freq='D')
    np.random.seed(42)
    
    close_prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 252)))
    
    data = pd.DataFrame({
        'Open': close_prices * (1 + np.random.uniform(-0.01, 0.01, 252)),
        'High': close_prices * (1 + np.random.uniform(0, 0.02, 252)),
        'Low': close_prices * (1 - np.random.uniform(0, 0.02, 252)),
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, 252)
    }, index=dates)
    
    return data


@pytest.fixture
def sample_returns():
    """Generate sample return series for testing."""
    np.random.seed(42)
    return pd.Series(np.random.normal(0.0005, 0.015, 252))


@pytest.fixture
def mock_yfinance_download(sample_price_data):
    """Mock yfinance download to return sample data."""
    def _download(*args, **kwargs):
        return sample_price_data.copy()
    return _download


@pytest.fixture
def mock_empty_yfinance():
    """Mock yfinance download that returns empty DataFrame."""
    def _download(*args, **kwargs):
        return pd.DataFrame()
    return _download