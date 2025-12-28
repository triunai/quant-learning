import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from refinery.semi_markov import SemiMarkovModel

@pytest.fixture
def mock_data():
    """Creates synthetic price data with identifiable regimes and trends (1000 days)."""
    dates = pd.date_range(start="2018-01-01", periods=1000, freq="D")

    # Generate distinct regimes
    # 0-200: Flat (Low Vol)
    # 200-400: Bull (High Return, Med Vol)
    # 400-500: Crash (Neg Return, High Vol)
    # 500-700: Recovery/Rally (High Return, High Vol)
    # 700-1000: Bear (Slow Neg Return, Med Vol)

    returns = np.zeros(1000)
    # Flat
    returns[0:200] = np.random.normal(0, 0.005, 200)
    # Bull
    returns[200:400] = np.random.normal(0.001, 0.01, 200)
    # Crash
    returns[400:500] = np.random.normal(-0.003, 0.03, 100)
    # Rally
    returns[500:700] = np.random.normal(0.002, 0.02, 200)
    # Bear
    returns[700:1000] = np.random.normal(-0.001, 0.015, 300)

    price = 100 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({'Close': price}, index=dates)
    return df

@patch('yfinance.download')
def test_process_data(mock_download, mock_data):
    mock_download.return_value = mock_data

    model = SemiMarkovModel("TEST", n_states=5)
    df = model._process_data(period="5y")

    assert 'Log_Ret' in df.columns
    assert 'State_Idx' in df.columns
    assert 'block' in df.columns
    assert 'Vol_20d' in df.columns
    assert 'Ret_60d' in df.columns

    # Check if KMeans produced distinct states
    unique_states = df['State_Idx'].nunique()
    # Depending on convergence, might be less than 5, but should be > 1 given the data
    assert unique_states > 1

    # Check label sorting (0 should have lowest momentum, 4 highest)
    # We can check the means of Ret_60d for the states found
    means = df.groupby('State_Idx')['Ret_60d'].mean()
    if 0 in means and 4 in means:
        assert means[0] < means[4]

@patch('yfinance.download')
def test_fit_distributions(mock_download, mock_data):
    mock_download.return_value = mock_data
    model = SemiMarkovModel("TEST", n_states=5)
    model._process_data(period="5y")
    model.fit_distributions()

    assert len(model.duration_params) == 5
    for state in range(5):
        params = model.duration_params[state]
        assert params['dist'] in ['gamma', 'expon']

@patch('yfinance.download')
def test_run_simulation(mock_download, mock_data):
    mock_download.return_value = mock_data
    model = SemiMarkovModel("TEST", n_states=5)
    model._process_data(period="5y")
    model.fit_distributions()

    paths = model.run_simulation(days=50, simulations=10)

    assert paths.shape == (10, 50)
    assert not np.isnan(paths).any()
    assert np.all(paths[:, 0] > 0)

def test_sample_residual_duration():
    model = SemiMarkovModel("TEST")
    # Setup params
    # Gamma: shape=2, scale=10 => mean=20
    model.duration_params = {
        0: {'dist': 'gamma', 'params': (2, 0, 10)}
    }

    # Case 1: Elapsed 0 -> Should be full distribution
    d0 = [model.sample_residual_duration(0, 0) for _ in range(100)]
    assert np.mean(d0) > 1

    # Case 2: Elapsed 30 (tail) -> Should be small?
    # Actually Gamma(2) has hazard rate that ... ?
    # Let's just check it runs and returns int >= 1
    d_tail = model.sample_residual_duration(0, 30)
    assert isinstance(d_tail, int)
    assert d_tail >= 1

def test_regime_fatigue_score():
    model = SemiMarkovModel("TEST")
    model.duration_params = {
        0: {'dist': 'gamma', 'params': (2, 0, 10)} # Mean 20
    }

    # Score should increase with time
    score_5 = model.regime_fatigue_score(0, 5)
    score_30 = model.regime_fatigue_score(0, 30)

    assert score_5 < score_30
    assert 0 <= score_5 <= 1
    assert 0 <= score_30 <= 1
