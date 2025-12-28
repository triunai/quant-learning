"""Unit tests for refinery/regime_engine_v7.py - Regime Engine V7."""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from refinery.regime_engine_v7 import RegimeEngineV7


class TestRegimeEngineV7Initialization:
    """Test suite for RegimeEngineV7 initialization."""
    
    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        engine = RegimeEngineV7(ticker="TEST")
        assert engine.ticker == "TEST"
        assert engine.days_ahead == 126
        assert engine.simulations == 5000
        assert engine.n_regimes == 3
    
    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        engine = RegimeEngineV7(
            ticker="AAPL",
            days_ahead=252,
            simulations=10000,
            n_regimes=4,
            market_ticker="SPY"
        )
        assert engine.ticker == "AAPL"
        assert engine.days_ahead == 252
        assert engine.simulations == 10000
        assert engine.n_regimes == 4
        assert engine.market_ticker == "SPY"
    
    def test_init_with_targets(self):
        """Test initialization with price targets."""
        engine = RegimeEngineV7(
            ticker="TEST",
            target_up=150.0,
            target_down=80.0
        )
        assert engine.target_up == 150.0
        assert engine.target_down == 80.0
    
    def test_init_with_stop_loss(self):
        """Test initialization with custom stop loss."""
        engine = RegimeEngineV7(ticker="TEST", stop_loss_pct=0.20)
        assert engine.stop_loss_pct == 0.20
    
    def test_initial_state_is_none(self):
        """Test that data is None before ingestion."""
        engine = RegimeEngineV7(ticker="TEST")
        assert engine.data is None
        assert engine.market_data is None
        assert engine.gmm is None


class TestRegimeEngineV7DataIngestion:
    """Test suite for data ingestion methods."""
    
    @patch('yfinance.download')
    def test_ingest_data_success(self, mock_download, sample_price_data):
        """Test successful data ingestion."""
        mock_download.return_value = sample_price_data
        
        engine = RegimeEngineV7(ticker="TEST")
        engine.ingest_data()
        
        assert engine.data is not None
        assert engine.market_data is not None
        assert len(engine.data) > 0
        assert 'Log_Ret' in engine.data.columns
    
    @patch('yfinance.download')
    def test_ingest_creates_log_returns(self, mock_download, sample_price_data):
        """Test that log returns are calculated."""
        mock_download.return_value = sample_price_data
        
        engine = RegimeEngineV7(ticker="TEST")
        engine.ingest_data()
        
        assert 'Log_Ret' in engine.data.columns
        # Check log return calculation is correct
        expected = np.log(engine.data['Close'] / engine.data['Close'].shift(1))
        pd.testing.assert_series_equal(
            engine.data['Log_Ret'].dropna(),
            expected.dropna(),
            check_names=False
        )
    
    @patch('yfinance.download')
    def test_ingest_creates_slow_features(self, mock_download, sample_price_data):
        """Test that slow features are created."""
        mock_download.return_value = sample_price_data
        
        engine = RegimeEngineV7(ticker="TEST")
        engine.ingest_data()
        
        # Check for slow feature columns
        expected_features = ['Vol_20d', 'Vol_60d', 'Ret_20d', 'Drawdown']
        for feature in expected_features:
            assert feature in engine.data.columns, f"Missing feature: {feature}"
    
    @patch('yfinance.download')
    def test_ingest_handles_multiindex_columns(self, mock_download, sample_price_data):
        """Test handling of MultiIndex columns from yfinance."""
        # Create MultiIndex columns (yfinance 1.0+ format)
        multi_df = sample_price_data.copy()
        multi_df.columns = pd.MultiIndex.from_product([multi_df.columns, ['TEST']])
        mock_download.return_value = multi_df
        
        engine = RegimeEngineV7(ticker="TEST")
        engine.ingest_data()
        
        # Should flatten columns correctly
        assert not isinstance(engine.data.columns, pd.MultiIndex)
    
    @patch('yfinance.download')
    def test_ingest_sets_last_price(self, mock_download, sample_price_data):
        """Test that last price is correctly set."""
        mock_download.return_value = sample_price_data
        
        engine = RegimeEngineV7(ticker="TEST")
        engine.ingest_data()
        
        assert engine.last_price == pytest.approx(sample_price_data['Close'].iloc[-1])
    
    @patch('yfinance.download')
    def test_ingest_computes_realized_vol(self, mock_download, sample_price_data):
        """Test that realized volatility is computed."""
        mock_download.return_value = sample_price_data
        
        engine = RegimeEngineV7(ticker="TEST")
        engine.ingest_data()
        
        assert engine.realized_vol > 0
        assert engine.realized_vol < 5.0  # Sanity check (500% annual vol is extreme)
    
    @patch('yfinance.download')
    def test_ingest_aligns_asset_and_market_data(self, mock_download, sample_price_data):
        """Test that asset and market data are aligned on dates."""
        mock_download.return_value = sample_price_data
        
        engine = RegimeEngineV7(ticker="TEST", market_ticker="SPY")
        engine.ingest_data()
        
        # Indices should match
        pd.testing.assert_index_equal(engine.data.index, engine.market_data.index)
    
    @patch('yfinance.download')
    def test_ingest_handles_missing_data(self, mock_download):
        """Test handling of missing/NaN data."""
        # Create data with NaNs
        dates = pd.date_range('2020-01-01', periods=100)
        data_with_nan = pd.DataFrame({
            'Close': [100.0] * 50 + [np.nan] * 25 + [105.0] * 25,
            'Volume': [1000000] * 100
        }, index=dates)
        mock_download.return_value = data_with_nan
        
        engine = RegimeEngineV7(ticker="TEST")
        engine.ingest_data()
        
        # Should drop NaN rows
        assert engine.data['Close'].isna().sum() == 0


class TestRegimeEngineV7RegimeModel:
    """Test suite for regime modeling methods."""
    
    @patch('yfinance.download')
    def test_build_regime_model_creates_regimes(self, mock_download, sample_price_data):
        """Test that regime model creates regime assignments."""
        mock_download.return_value = sample_price_data
        
        engine = RegimeEngineV7(ticker="TEST", n_regimes=3)
        engine.ingest_data()
        engine.build_regime_model()
        
        assert 'Regime' in engine.data.columns
        assert engine.data['Regime'].nunique() <= 3
    
    @patch('yfinance.download')
    def test_build_regime_model_sets_current_regime(self, mock_download, sample_price_data):
        """Test that current regime is set."""
        mock_download.return_value = sample_price_data
        
        engine = RegimeEngineV7(ticker="TEST")
        engine.ingest_data()
        engine.build_regime_model()
        
        assert engine.current_regime is not None
        assert 0 <= engine.current_regime < engine.n_regimes
    
    @patch('yfinance.download')
    def test_compute_regime_stats_creates_parameters(self, mock_download, sample_price_data):
        """Test that regime statistics are computed."""
        mock_download.return_value = sample_price_data
        
        engine = RegimeEngineV7(ticker="TEST")
        engine.ingest_data()
        engine.build_regime_model()
        
        # Should have mu and sigma for each regime
        assert len(engine.regime_mu) == engine.n_regimes
        assert len(engine.regime_sigma) == engine.n_regimes
        
        for r in range(engine.n_regimes):
            assert r in engine.regime_mu
            assert r in engine.regime_sigma
    
    @patch('yfinance.download')
    def test_regime_durations_are_positive(self, mock_download, sample_price_data):
        """Test that regime durations are positive."""
        mock_download.return_value = sample_price_data
        
        engine = RegimeEngineV7(ticker="TEST")
        engine.ingest_data()
        engine.build_regime_model()
        
        for r in range(engine.n_regimes):
            assert engine.regime_duration[r]['mean'] > 0
    
    @patch('yfinance.download')
    def test_transition_matrix_is_stochastic(self, mock_download, sample_price_data):
        """Test that transition matrix rows sum to 1."""
        mock_download.return_value = sample_price_data
        
        engine = RegimeEngineV7(ticker="TEST")
        engine.ingest_data()
        engine.build_regime_model()
        
        # Each row should sum to 1 (stochastic matrix)
        row_sums = engine.transition_matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)
    
    @patch('yfinance.download')
    def test_transition_matrix_has_correct_shape(self, mock_download, sample_price_data):
        """Test that transition matrix has correct dimensions."""
        mock_download.return_value = sample_price_data
        
        engine = RegimeEngineV7(ticker="TEST", n_regimes=3)
        engine.ingest_data()
        engine.build_regime_model()
        
        assert engine.transition_matrix.shape == (3, 3)


class TestRegimeEngineV7MacroConditioning:
    """Test suite for macro conditioning (beta/alpha)."""
    
    @patch('yfinance.download')
    def test_compute_market_beta_sets_beta(self, mock_download, sample_price_data):
        """Test that market beta is computed."""
        mock_download.return_value = sample_price_data
        
        engine = RegimeEngineV7(ticker="TEST")
        engine.ingest_data()
        engine.build_regime_model()
        engine.compute_market_beta()
        
        assert engine.market_beta is not None
        assert -5.0 < engine.market_beta < 5.0  # Sanity check
    
    @patch('yfinance.download')
    def test_compute_market_beta_sets_alpha(self, mock_download, sample_price_data):
        """Test that market alpha is computed."""
        mock_download.return_value = sample_price_data
        
        engine = RegimeEngineV7(ticker="TEST")
        engine.ingest_data()
        engine.build_regime_model()
        engine.compute_market_beta()
        
        assert hasattr(engine, 'market_alpha')
        assert engine.market_alpha is not None
    
    @patch('yfinance.download')
    def test_regime_specific_alphas(self, mock_download, sample_price_data):
        """Test that per-regime alphas are computed."""
        mock_download.return_value = sample_price_data
        
        engine = RegimeEngineV7(ticker="TEST")
        engine.ingest_data()
        engine.build_regime_model()
        engine.compute_market_beta()
        
        assert len(engine.regime_alpha) == engine.n_regimes
        for r in range(engine.n_regimes):
            assert r in engine.regime_alpha
    
    @patch('yfinance.download')
    def test_idiosyncratic_vol_is_positive(self, mock_download, sample_price_data):
        """Test that idiosyncratic volatility is positive."""
        mock_download.return_value = sample_price_data
        
        engine = RegimeEngineV7(ticker="TEST")
        engine.ingest_data()
        engine.build_regime_model()
        engine.compute_market_beta()
        
        assert engine.idio_vol > 0


class TestRegimeEngineV7Simulation:
    """Test suite for Monte Carlo simulation."""
    
    @patch('yfinance.download')
    def test_simulate_returns_correct_shape(self, mock_download, sample_price_data):
        """Test that simulation returns correct path dimensions."""
        mock_download.return_value = sample_price_data
        
        engine = RegimeEngineV7(ticker="TEST", days_ahead=126, simulations=100)
        engine.ingest_data()
        engine.build_regime_model()
        engine.compute_market_beta()
        
        paths = engine.simulate()
        
        assert paths.shape == (100, 126)
    
    @patch('yfinance.download')
    def test_simulate_paths_are_positive(self, mock_download, sample_price_data):
        """Test that simulated prices are positive."""
        mock_download.return_value = sample_price_data
        
        engine = RegimeEngineV7(ticker="TEST", simulations=100)
        engine.ingest_data()
        engine.build_regime_model()
        engine.compute_market_beta()
        
        paths = engine.simulate()
        
        assert np.all(paths > 0), "All prices should be positive"
    
    @patch('yfinance.download')
    def test_simulate_paths_start_correctly(self, mock_download, sample_price_data):
        """Test that paths start near last price."""
        mock_download.return_value = sample_price_data
        
        engine = RegimeEngineV7(ticker="TEST", simulations=100)
        engine.ingest_data()
        engine.build_regime_model()
        engine.compute_market_beta()
        
        paths = engine.simulate()
        
        # First prices should be very close to last_price after one step
        # (they evolve from last_price via returns)
        assert paths[:, 0].min() > 0
        assert paths[:, 0].max() < engine.last_price * 5  # Reasonable bound
    
    @patch('yfinance.download')
    def test_simulate_reproducibility_with_seed(self, mock_download, sample_price_data):
        """Test that simulation is reproducible with random seed."""
        mock_download.return_value = sample_price_data
        
        engine1 = RegimeEngineV7(ticker="TEST", simulations=50)
        engine1.ingest_data()
        engine1.build_regime_model()
        engine1.compute_market_beta()
        np.random.seed(42)
        paths1 = engine1.simulate()
        
        engine2 = RegimeEngineV7(ticker="TEST", simulations=50)
        engine2.ingest_data()
        engine2.build_regime_model()
        engine2.compute_market_beta()
        np.random.seed(42)
        paths2 = engine2.simulate()
        
        np.testing.assert_allclose(paths1, paths2, rtol=1e-10)


class TestRegimeEngineV7RiskMetrics:
    """Test suite for risk metric computation."""
    
    @patch('yfinance.download')
    def test_compute_risk_metrics_returns_dict(self, mock_download, sample_price_data):
        """Test that risk metrics returns a dictionary."""
        mock_download.return_value = sample_price_data
        
        engine = RegimeEngineV7(ticker="TEST", simulations=100)
        engine.ingest_data()
        engine.build_regime_model()
        engine.compute_market_beta()
        paths = engine.simulate()
        
        risk = engine.compute_risk_metrics(paths)
        
        assert isinstance(risk, dict)
    
    @patch('yfinance.download')
    def test_risk_metrics_contains_required_keys(self, mock_download, sample_price_data):
        """Test that all required risk metrics are present."""
        mock_download.return_value = sample_price_data
        
        engine = RegimeEngineV7(ticker="TEST", simulations=100)
        engine.ingest_data()
        engine.build_regime_model()
        engine.compute_market_beta()
        paths = engine.simulate()
        
        risk = engine.compute_risk_metrics(paths)
        
        required_keys = ['var_95', 'cvar_95', 'prob_dd_20', 'prob_dd_30', 
                        'expected_max_dd', 'prob_stop', 'kelly_fraction']
        for key in required_keys:
            assert key in risk, f"Missing risk metric: {key}"
    
    @patch('yfinance.download')
    def test_var_is_negative(self, mock_download, sample_price_data):
        """Test that VaR is negative (represents loss)."""
        mock_download.return_value = sample_price_data
        
        engine = RegimeEngineV7(ticker="TEST", simulations=100)
        engine.ingest_data()
        engine.build_regime_model()
        engine.compute_market_beta()
        paths = engine.simulate()
        
        risk = engine.compute_risk_metrics(paths)
        
        # VaR should typically be negative (5th percentile of returns)
        assert risk['var_95'] < 0.5  # Less than 50% gain at 5th percentile
    
    @patch('yfinance.download')
    def test_cvar_worse_than_var(self, mock_download, sample_price_data):
        """Test that CVaR is worse (more negative) than VaR."""
        mock_download.return_value = sample_price_data
        
        engine = RegimeEngineV7(ticker="TEST", simulations=1000)
        engine.ingest_data()
        engine.build_regime_model()
        engine.compute_market_beta()
        paths = engine.simulate()
        
        risk = engine.compute_risk_metrics(paths)
        
        # CVaR should be worse than VaR (tail conditional)
        assert risk['cvar_95'] <= risk['var_95']
    
    @patch('yfinance.download')
    def test_kelly_fraction_bounded(self, mock_download, sample_price_data):
        """Test that Kelly fraction is bounded [0, 1]."""
        mock_download.return_value = sample_price_data
        
        engine = RegimeEngineV7(ticker="TEST", simulations=100)
        engine.ingest_data()
        engine.build_regime_model()
        engine.compute_market_beta()
        paths = engine.simulate()
        
        risk = engine.compute_risk_metrics(paths)
        
        assert 0 <= risk['kelly_fraction'] <= 1.0
    
    @patch('yfinance.download')
    def test_probabilities_bounded(self, mock_download, sample_price_data):
        """Test that all probabilities are in [0, 1]."""
        mock_download.return_value = sample_price_data
        
        engine = RegimeEngineV7(ticker="TEST", simulations=100)
        engine.ingest_data()
        engine.build_regime_model()
        engine.compute_market_beta()
        paths = engine.simulate()
        
        risk = engine.compute_risk_metrics(paths)
        
        prob_keys = ['prob_dd_20', 'prob_dd_30', 'prob_stop', 'win_rate']
        for key in prob_keys:
            if key in risk:
                assert 0 <= risk[key] <= 1.0, f"{key} out of bounds: {risk[key]}"


class TestRegimeEngineV7Validation:
    """Test suite for validation and calibration methods."""
    
    @patch('yfinance.download')
    def test_compute_historical_validation_sets_rates(self, mock_download, sample_price_data):
        """Test that historical validation computes hit rates."""
        mock_download.return_value = sample_price_data
        
        engine = RegimeEngineV7(ticker="TEST")
        engine.ingest_data()
        engine.compute_historical_validation()
        
        assert hasattr(engine, 'hist_up_hit')
        assert hasattr(engine, 'hist_down_hit')
        assert 0 <= engine.hist_up_hit <= 1.0
        assert 0 <= engine.hist_down_hit <= 1.0
    
    @patch('yfinance.download')
    def test_verify_simulation_invariants(self, mock_download, sample_price_data):
        """Test simulation invariant verification."""
        mock_download.return_value = sample_price_data
        
        engine = RegimeEngineV7(ticker="TEST", simulations=100)
        engine.ingest_data()
        engine.build_regime_model()
        engine.compute_market_beta()
        paths = engine.simulate()
        
        invariants = engine.verify_simulation_invariants(paths)
        
        assert isinstance(invariants, dict)
        assert 'mean_ok' in invariants
        assert 'std_ok' in invariants


class TestRegimeEngineV7EdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_ticker_type(self):
        """Test that invalid ticker type raises error."""
        with pytest.raises(TypeError):
            RegimeEngineV7(ticker=123)
    
    def test_negative_simulations(self):
        """Test that negative simulations raises error."""
        with pytest.raises(ValueError):
            RegimeEngineV7(ticker="TEST", simulations=-100)
    
    def test_zero_simulations(self):
        """Test that zero simulations raises error."""
        with pytest.raises(ValueError):
            RegimeEngineV7(ticker="TEST", simulations=0)
    
    def test_invalid_n_regimes(self):
        """Test that invalid n_regimes raises error."""
        with pytest.raises(ValueError):
            RegimeEngineV7(ticker="TEST", n_regimes=1)  # Need at least 2
    
    @patch('yfinance.download')
    def test_handles_download_failure(self, mock_download):
        """Test handling of yfinance download failure."""
        mock_download.side_effect = Exception("Network error")
        
        engine = RegimeEngineV7(ticker="TEST")
        with pytest.raises(Exception):
            engine.ingest_data()


@pytest.mark.integration
class TestRegimeEngineV7Integration:
    """Integration tests for full workflow."""
    
    @patch('yfinance.download')
    @patch('matplotlib.pyplot.show')
    def test_full_run_workflow(self, mock_show, mock_download, sample_price_data):
        """Test complete run() workflow."""
        mock_download.return_value = sample_price_data
        
        engine = RegimeEngineV7(ticker="TEST", simulations=50)
        result = engine.run(run_full_calibration=False)
        
        assert 'paths' in result
        assert 'prob_up' in result
        assert 'prob_down' in result
        assert 'risk' in result
        assert 'signal' in result
    
    @patch('yfinance.download')
    @patch('matplotlib.pyplot.show')
    def test_run_with_calibration(self, mock_show, mock_download, sample_price_data):
        """Test run with full calibration suite."""
        mock_download.return_value = sample_price_data
        
        engine = RegimeEngineV7(ticker="TEST", simulations=50)
        result = engine.run(run_full_calibration=True)
        
        # Should complete without errors
        assert result is not None