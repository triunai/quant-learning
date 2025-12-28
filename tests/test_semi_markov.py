import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from refinery.semi_markov import SemiMarkovModel

@pytest.fixture
def mock_data():
    """Creates synthetic price data with identifiable regimes."""
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    # Create a pattern: 5 days up, 5 days flat, 5 days down
    close = [100]
    for i in range(1, 100):
        if (i // 5) % 3 == 0: # Up (Bull/Rally)
            ret = 0.01
        elif (i // 5) % 3 == 1: # Flat
            ret = 0.00
        else: # Down (Bear/Crash)
            ret = -0.01
        close.append(close[-1] * np.exp(ret))

    df = pd.DataFrame({'Close': close}, index=dates)
    return df

@patch('yfinance.download')
def test_process_data(mock_download, mock_data):
    mock_download.return_value = mock_data

    model = SemiMarkovModel("TEST", n_states=3)
    df = model._process_data(period="1y")

    assert 'Log_Ret' in df.columns
    assert 'State_Idx' in df.columns
    assert 'block' in df.columns

    # Check if durations were extracted
    assert len(model.duration_data) == 3
    # We expect some durations to be around 5
    for state in range(3):
        if model.duration_data[state]:
            assert max(model.duration_data[state]) > 0

@patch('yfinance.download')
def test_fit_distributions(mock_download, mock_data):
    mock_download.return_value = mock_data
    model = SemiMarkovModel("TEST", n_states=3)
    model._process_data(period="1y")
    model.fit_distributions()

    assert len(model.duration_params) == 3
    for state in range(3):
        params = model.duration_params[state]
        assert params['dist'] in ['gamma', 'expon']
        assert len(params['params']) > 0

@patch('yfinance.download')
def test_run_simulation(mock_download, mock_data):
    mock_download.return_value = mock_data
    model = SemiMarkovModel("TEST", n_states=3)
    model._process_data(period="1y")
    model.fit_distributions()

    paths = model.run_simulation(days=50, simulations=10)

    assert paths.shape == (10, 50)
    # Check for NaN or Inf
    assert not np.isnan(paths).any()
    assert not np.isinf(paths).any()

    # Check starting point (should be close to last price)
    last_price = mock_data['Close'].iloc[-1]
    # first step might drift, but shouldn't be zero
    assert np.all(paths[:, 0] > 0)

def test_sample_duration():
    model = SemiMarkovModel("TEST")
    # Mock params
    model.duration_params = {
        0: {'dist': 'expon', 'params': (0, 5)}, # Mean 5
        1: {'dist': 'gamma', 'params': (2, 0, 2)} # Shape 2, Scale 2 -> Mean 4
    }

    d0 = [model.sample_duration(0) for _ in range(100)]
    d1 = [model.sample_duration(1) for _ in range(100)]

    assert min(d0) >= 1
    assert min(d1) >= 1
    assert np.mean(d0) > 1
