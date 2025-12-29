import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from refinery.semi_markov import SemiMarkovModel

@pytest.fixture
def mock_data():
    """Creates synthetic price data with identifiable regimes and trends (1000 days)."""
    dates = pd.date_range(start="2018-01-01", periods=1000, freq="D")

    returns = np.zeros(1000)
    # Flat
    returns[0:200] = np.random.normal(0, 0.005, 200)
    # Bull
    returns[200:400] = np.random.normal(0.001, 0.01, 200)
    # Crash (High Vol, Neg Return)
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
    assert 'Cluster' in df.columns
    assert 'State_Idx' in df.columns

    # Check Risk Ordering
    # State 0 should be Crash (High Vol, Neg Ret)
    # State 4 should be Bull/Rally (Low Risk)

    stats_0 = df[df['State_Idx'] == 0]
    stats_4 = df[df['State_Idx'] == 4]

    # Check if State 0 has higher vol or worse return than State 4
    # Just need to confirm ordering logic worked broadly
    # Risk Score = Vol*2 - Ret - DD*1.5

    def get_risk(subset):
        return (subset['Vol_20d'].mean() * 2) - subset['Ret_60d'].mean() - (subset['DD'].mean() * 1.5)

    risk_0 = get_risk(stats_0)
    risk_4 = get_risk(stats_4)

    assert risk_0 > risk_4

@patch('yfinance.download')
def test_sample_regime_returns_contiguous(mock_download, mock_data):
    mock_download.return_value = mock_data
    model = SemiMarkovModel("TEST", n_states=5)
    model._process_data(period="5y")

    # Test sampling
    # We want to make sure it returns an array of length n_days
    # and that the values actually exist in the original data (sanity check)

    ret_sample = model.sample_regime_returns(0, 10)
    assert len(ret_sample) == 10

    # Check if these returns exist in the source data
    # (Exact match might fail due to float precision, use approx)
    source_rets = model.data[model.data['State_Idx'] == 0]['Log_Ret'].values
    for r in ret_sample:
        assert np.isclose(source_rets, r).any()

def test_get_position_size():
    model = SemiMarkovModel("TEST")
    model.duration_params = {
        0: {'dist': 'gamma', 'params': (2, 0, 10)}, # Risky
        2: {'dist': 'gamma', 'params': (2, 0, 10)}  # Normal
    }

    # State 0 (Crash) -> Multiplier 0.3 + (1-Fatigue)*0.3
    # Fatigue=0 -> 0.6. Fatigue=1 -> 0.3.
    pos_crash_low_fatigue = model.get_position_size(0, 1) # ~0 fatigue
    pos_crash_high_fatigue = model.get_position_size(0, 100) # ~1 fatigue

    assert pos_crash_low_fatigue > pos_crash_high_fatigue
    assert 0.3 <= pos_crash_high_fatigue <= 0.6

    # State 2 (Normal) -> Multiplier 0.5 + (1-Fatigue)*0.5
    # Fatigue=0 -> 1.0. Fatigue=1 -> 0.5.
    pos_norm_low_fatigue = model.get_position_size(2, 1)
    pos_norm_high_fatigue = model.get_position_size(2, 100)

    assert pos_norm_low_fatigue > pos_norm_high_fatigue
    assert 0.5 <= pos_norm_high_fatigue <= 1.0

@patch('yfinance.download')
def test_validate_model(mock_download, mock_data):
    mock_download.return_value = mock_data
    model = SemiMarkovModel("TEST", n_states=5)
    model._process_data(period="5y")
    model.fit_distributions()

    paths = model.run_simulation(days=50, simulations=10)

    report = model.validate_model(paths)
    assert 'real_vol_acf_lag1' in report
    assert 'sim_vol_acf_lag1' in report
    assert isinstance(report['vol_clustering_error'], float)
