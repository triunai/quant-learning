"""Unit tests for refinery/regime_engine_v7.py - Regime Engine V7 Wrapper.

Tests the RegimeRiskEngineV7 wrapper class that provides a Streamlit-friendly
interface to the battle-tested RegimeRiskPlatform.
"""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def sample_price_data():
    """Generate realistic sample OHLCV price data for testing."""
    np.random.seed(42)
    n_days = 504  # 2 years of trading days
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='B')
    
    # Generate realistic price path with drift and volatility
    daily_returns = np.random.normal(0.0005, 0.02, n_days)
    close_prices = 100 * np.exp(np.cumsum(daily_returns))
    
    data = pd.DataFrame({
        'Open': close_prices * (1 + np.random.uniform(-0.005, 0.005, n_days)),
        'High': close_prices * (1 + np.random.uniform(0.005, 0.02, n_days)),
        'Low': close_prices * (1 - np.random.uniform(0.005, 0.02, n_days)),
        'Close': close_prices,
        'Adj Close': close_prices,
        'Volume': np.random.randint(1_000_000, 50_000_000, n_days)
    }, index=dates)
    
    return data


@pytest.fixture
def mock_yfinance(sample_price_data):
    """Create a mock for yfinance.download that returns sample data."""
    def _download(*args, **kwargs):
        return sample_price_data.copy()
    return _download


# ==============================================================================
# TEST: INITIALIZATION
# ==============================================================================

class TestRegimeEngineV7Initialization:
    """Test suite for RegimeRiskEngineV7 initialization."""
    
    @patch('refinery.regime_engine_v7.RegimeRiskPlatform')
    def test_init_with_defaults(self, mock_platform_class):
        """Test initialization with default parameters."""
        from refinery.regime_engine_v7 import RegimeRiskEngineV7
        
        engine = RegimeRiskEngineV7(ticker="TEST")
        
        assert engine.ticker == "TEST"
        assert engine.days_ahead == 126
        assert engine.simulations == 5000
        mock_platform_class.assert_called_once()
    
    @patch('refinery.regime_engine_v7.RegimeRiskPlatform')
    def test_init_with_custom_parameters(self, mock_platform_class):
        """Test initialization with custom parameters."""
        from refinery.regime_engine_v7 import RegimeRiskEngineV7
        
        engine = RegimeRiskEngineV7(
            ticker="AAPL",
            days_ahead=252,
            simulations=10000,
            n_regimes=4,
            market_ticker="SPY"
        )
        
        assert engine.ticker == "AAPL"
        assert engine.days_ahead == 252
        assert engine.simulations == 10000
        
        # Verify platform was initialized with correct params
        call_kwargs = mock_platform_class.call_args[1]
        assert call_kwargs['ticker'] == "AAPL"
        assert call_kwargs['days_ahead'] == 252
        assert call_kwargs['simulations'] == 10000
        assert call_kwargs['n_regimes'] == 4
        assert call_kwargs['market_ticker'] == "SPY"
    
    @patch('refinery.regime_engine_v7.RegimeRiskPlatform')
    def test_init_with_targets(self, mock_platform_class):
        """Test initialization with price targets."""
        from refinery.regime_engine_v7 import RegimeRiskEngineV7
        
        engine = RegimeRiskEngineV7(
            ticker="TEST",
            target_up=150.0,
            target_down=80.0
        )
        
        assert engine.target_up == 150.0
        assert engine.target_down == 80.0
    
    @patch('refinery.regime_engine_v7.RegimeRiskPlatform')
    def test_init_with_stop_loss(self, mock_platform_class):
        """Test initialization with custom stop loss."""
        from refinery.regime_engine_v7 import RegimeRiskEngineV7
        
        engine = RegimeRiskEngineV7(ticker="TEST", stop_loss_pct=0.20)
        
        call_kwargs = mock_platform_class.call_args[1]
        assert call_kwargs['stop_loss_pct'] == 0.20
    
    @patch('refinery.regime_engine_v7.RegimeRiskPlatform')
    def test_initial_state(self, mock_platform_class):
        """Test that initial state is correctly set."""
        from refinery.regime_engine_v7 import RegimeRiskEngineV7
        
        engine = RegimeRiskEngineV7(ticker="TEST")
        
        assert engine.last_price == 0
        assert engine.current_regime == 0
        assert engine.regime_names == {}
        assert engine.market_beta == 0
        assert engine.transition_matrix is None


# ==============================================================================
# TEST: DATA INGESTION
# ==============================================================================

class TestRegimeEngineV7DataIngestion:
    """Test suite for data ingestion via the wrapper."""
    
    @patch('refinery.regime_engine_v7.RegimeRiskPlatform')
    def test_ingest_data_calls_platform(self, mock_platform_class):
        """Test that ingest_data delegates to platform."""
        from refinery.regime_engine_v7 import RegimeRiskEngineV7
        
        mock_platform = MagicMock()
        mock_platform.last_price = 100.0
        mock_platform.target_up = 120.0
        mock_platform.target_down = 80.0
        mock_platform.realized_vol = 0.25
        mock_platform_class.return_value = mock_platform
        
        engine = RegimeRiskEngineV7(ticker="TEST")
        engine.ingest_data()
        
        mock_platform.ingest_data.assert_called_once()
    
    @patch('refinery.regime_engine_v7.RegimeRiskPlatform')
    def test_ingest_syncs_attributes(self, mock_platform_class):
        """Test that ingest_data syncs key attributes from platform."""
        from refinery.regime_engine_v7 import RegimeRiskEngineV7
        
        mock_platform = MagicMock()
        mock_platform.last_price = 150.0
        mock_platform.target_up = 180.0
        mock_platform.target_down = 120.0
        mock_platform.realized_vol = 0.35
        mock_platform_class.return_value = mock_platform
        
        engine = RegimeRiskEngineV7(ticker="TEST")
        engine.ingest_data()
        
        assert engine.last_price == 150.0
        assert engine.target_up == 180.0
        assert engine.target_down == 120.0
        assert engine.realized_vol == 0.35


# ==============================================================================
# TEST: RUN METHOD
# ==============================================================================

class TestRegimeEngineV7Run:
    """Test suite for the main run() method."""
    
    @patch('refinery.regime_engine_v7.RegimeRiskPlatform')
    def test_run_returns_tuple(self, mock_platform_class):
        """Test that run() returns (results, signal) tuple."""
        from refinery.regime_engine_v7 import RegimeRiskEngineV7
        
        # Setup mock platform
        mock_platform = MagicMock()
        mock_platform.n_regimes = 3
        mock_platform.current_regime = 1
        mock_platform.regime_names = {0: "Crash", 1: "Normal", 2: "Rally"}
        mock_platform.market_beta = 1.2
        mock_platform.market_alpha = 0.0002
        mock_platform.regime_alpha = {0: -0.001, 1: 0.0, 2: 0.001}
        mock_platform.regime_mu = {0: -0.001, 1: 0.0005, 2: 0.002}
        mock_platform.regime_sigma = {0: 0.03, 1: 0.015, 2: 0.025}
        mock_platform.regime_duration = {0: {'mean': 10}, 1: {'mean': 50}, 2: {'mean': 20}}
        mock_platform.transition_matrix = np.array([[0.9, 0.08, 0.02], [0.05, 0.9, 0.05], [0.02, 0.08, 0.9]])
        mock_platform.last_price = 100.0
        mock_platform.target_up = 120.0
        mock_platform.target_down = 80.0
        mock_platform.vix_level = 18.0
        mock_platform.garch_vol = 0.22
        mock_platform.realized_vol = 0.25
        mock_platform.idio_vol = 0.15
        mock_platform.hist_up_hit = 0.6
        mock_platform.hist_down_hit = 0.3
        mock_platform.data = pd.DataFrame({
            'Regime': [0, 1, 1, 2, 1],
            'Drawdown': [-0.05, -0.02, -0.01, -0.03, -0.02],
            'Vol_20d': [0.03, 0.015, 0.018, 0.025, 0.016]
        })
        
        mock_platform.simulate.return_value = np.random.rand(100, 126) * 100 + 80
        mock_platform.analyze_fpt.return_value = 0.65
        mock_platform.compute_risk_metrics.return_value = {'var_95': -0.15, 'cvar_95': -0.22}
        mock_platform.generate_signal.return_value = {'signal': 'HOLD', 'confidence': 0.7}
        
        mock_platform_class.return_value = mock_platform
        
        engine = RegimeRiskEngineV7(ticker="TEST", simulations=100)
        results, signal = engine.run()
        
        assert isinstance(results, dict)
        assert isinstance(signal, dict)
    
    @patch('refinery.regime_engine_v7.RegimeRiskPlatform')
    def test_run_results_structure(self, mock_platform_class):
        """Test that results dict has expected keys."""
        from refinery.regime_engine_v7 import RegimeRiskEngineV7
        
        # Setup mock platform
        mock_platform = MagicMock()
        mock_platform.n_regimes = 3
        mock_platform.current_regime = 1
        mock_platform.regime_names = {0: "Crash", 1: "Normal", 2: "Rally"}
        mock_platform.market_beta = 1.2
        mock_platform.market_alpha = 0.0002
        mock_platform.regime_alpha = {0: -0.001, 1: 0.0, 2: 0.001}
        mock_platform.regime_mu = {0: -0.001, 1: 0.0005, 2: 0.002}
        mock_platform.regime_sigma = {0: 0.03, 1: 0.015, 2: 0.025}
        mock_platform.regime_duration = {0: {'mean': 10}, 1: {'mean': 50}, 2: {'mean': 20}}
        mock_platform.transition_matrix = np.array([[0.9, 0.08, 0.02], [0.05, 0.9, 0.05], [0.02, 0.08, 0.9]])
        mock_platform.last_price = 100.0
        mock_platform.target_up = 120.0
        mock_platform.target_down = 80.0
        mock_platform.vix_level = 18.0
        mock_platform.garch_vol = 0.22
        mock_platform.realized_vol = 0.25
        mock_platform.idio_vol = 0.15
        mock_platform.hist_up_hit = 0.6
        mock_platform.hist_down_hit = 0.3
        mock_platform.data = pd.DataFrame({
            'Regime': [0, 1, 1, 2, 1],
            'Drawdown': [-0.05, -0.02, -0.01, -0.03, -0.02],
            'Vol_20d': [0.03, 0.015, 0.018, 0.025, 0.016]
        })
        
        mock_platform.simulate.return_value = np.random.rand(100, 126) * 100 + 80
        mock_platform.analyze_fpt.return_value = 0.65
        mock_platform.compute_risk_metrics.return_value = {'var_95': -0.15, 'cvar_95': -0.22}
        mock_platform.generate_signal.return_value = {'signal': 'HOLD', 'confidence': 0.7}
        
        mock_platform_class.return_value = mock_platform
        
        engine = RegimeRiskEngineV7(ticker="TEST", simulations=100)
        results, signal = engine.run()
        
        # Check required keys
        assert 'paths' in results
        assert 'prob_up' in results
        assert 'prob_down' in results
        assert 'risk' in results
        assert 'regime_diagnostics' in results
        assert 'sanity' in results
        assert 'macro' in results
    
    @patch('refinery.regime_engine_v7.RegimeRiskPlatform')
    def test_run_calls_build_methods(self, mock_platform_class):
        """Test that run() calls all the build/compute methods on platform."""
        from refinery.regime_engine_v7 import RegimeRiskEngineV7
        
        # Setup mock platform
        mock_platform = MagicMock()
        mock_platform.n_regimes = 3
        mock_platform.current_regime = 1
        mock_platform.regime_names = {}
        mock_platform.market_beta = 1.0
        mock_platform.market_alpha = 0.0
        mock_platform.regime_alpha = {}
        mock_platform.regime_mu = {}
        mock_platform.regime_sigma = {}
        mock_platform.regime_duration = {}
        mock_platform.transition_matrix = np.eye(3)
        mock_platform.last_price = 100.0
        mock_platform.target_up = 120.0
        mock_platform.target_down = 80.0
        mock_platform.vix_level = 18.0
        mock_platform.garch_vol = 0.22
        mock_platform.realized_vol = 0.25
        mock_platform.idio_vol = 0.15
        mock_platform.hist_up_hit = 0.5
        mock_platform.hist_down_hit = 0.5
        mock_platform.data = pd.DataFrame({'Regime': [], 'Drawdown': [], 'Vol_20d': []})
        
        mock_platform.simulate.return_value = np.random.rand(100, 126) * 100
        mock_platform.analyze_fpt.return_value = 0.5
        mock_platform.compute_risk_metrics.return_value = {}
        mock_platform.generate_signal.return_value = {}
        
        mock_platform_class.return_value = mock_platform
        
        engine = RegimeRiskEngineV7(ticker="TEST", simulations=100)
        engine.run()
        
        # Verify all methods were called
        mock_platform.build_regime_model.assert_called_once()
        mock_platform.compute_market_beta.assert_called_once()
        mock_platform.check_macro_context.assert_called_once()
        mock_platform.run_garch.assert_called_once()
        mock_platform.compute_historical_validation.assert_called_once()
        mock_platform.simulate.assert_called_once()
        mock_platform.verify_simulation_invariants.assert_called_once()
    
    @patch('refinery.regime_engine_v7.RegimeRiskPlatform')
    def test_run_with_calibration(self, mock_platform_class):
        """Test that run(run_full_calibration=True) calls calibration methods."""
        from refinery.regime_engine_v7 import RegimeRiskEngineV7
        
        # Setup mock platform
        mock_platform = MagicMock()
        mock_platform.n_regimes = 3
        mock_platform.current_regime = 1
        mock_platform.regime_names = {}
        mock_platform.market_beta = 1.0
        mock_platform.market_alpha = 0.0
        mock_platform.regime_alpha = {}
        mock_platform.regime_mu = {}
        mock_platform.regime_sigma = {}
        mock_platform.regime_duration = {}
        mock_platform.transition_matrix = np.eye(3)
        mock_platform.last_price = 100.0
        mock_platform.target_up = 120.0
        mock_platform.target_down = 80.0
        mock_platform.vix_level = 18.0
        mock_platform.garch_vol = 0.22
        mock_platform.realized_vol = 0.25
        mock_platform.idio_vol = 0.15
        mock_platform.hist_up_hit = 0.5
        mock_platform.hist_down_hit = 0.5
        mock_platform.data = pd.DataFrame({'Regime': [], 'Drawdown': [], 'Vol_20d': []})
        
        mock_platform.simulate.return_value = np.random.rand(100, 126) * 100
        mock_platform.analyze_fpt.return_value = 0.5
        mock_platform.compute_risk_metrics.return_value = {}
        mock_platform.generate_signal.return_value = {}
        
        mock_platform_class.return_value = mock_platform
        
        engine = RegimeRiskEngineV7(ticker="TEST", simulations=100)
        engine.run(run_full_calibration=True)
        
        # Verify calibration methods were called
        mock_platform.bucket_asymmetry_diagnostics.assert_called_once()
        mock_platform.multi_threshold_calibration.assert_called_once()
        mock_platform.walk_forward_validation.assert_called_once()
    
    @patch('refinery.regime_engine_v7.RegimeRiskPlatform')
    def test_run_without_calibration(self, mock_platform_class):
        """Test that run() without calibration skips calibration methods."""
        from refinery.regime_engine_v7 import RegimeRiskEngineV7
        
        # Setup mock platform
        mock_platform = MagicMock()
        mock_platform.n_regimes = 3
        mock_platform.current_regime = 1
        mock_platform.regime_names = {}
        mock_platform.market_beta = 1.0
        mock_platform.market_alpha = 0.0
        mock_platform.regime_alpha = {}
        mock_platform.regime_mu = {}
        mock_platform.regime_sigma = {}
        mock_platform.regime_duration = {}
        mock_platform.transition_matrix = np.eye(3)
        mock_platform.last_price = 100.0
        mock_platform.target_up = 120.0
        mock_platform.target_down = 80.0
        mock_platform.vix_level = 18.0
        mock_platform.garch_vol = 0.22
        mock_platform.realized_vol = 0.25
        mock_platform.idio_vol = 0.15
        mock_platform.hist_up_hit = 0.5
        mock_platform.hist_down_hit = 0.5
        mock_platform.data = pd.DataFrame({'Regime': [], 'Drawdown': [], 'Vol_20d': []})
        
        mock_platform.simulate.return_value = np.random.rand(100, 126) * 100
        mock_platform.analyze_fpt.return_value = 0.5
        mock_platform.compute_risk_metrics.return_value = {}
        mock_platform.generate_signal.return_value = {}
        
        mock_platform_class.return_value = mock_platform
        
        engine = RegimeRiskEngineV7(ticker="TEST", simulations=100)
        engine.run(run_full_calibration=False)
        
        # Verify calibration methods were NOT called
        mock_platform.bucket_asymmetry_diagnostics.assert_not_called()
        mock_platform.multi_threshold_calibration.assert_not_called()
        mock_platform.walk_forward_validation.assert_not_called()


# ==============================================================================
# TEST: ATTRIBUTE SYNCING
# ==============================================================================

class TestRegimeEngineV7AttributeSync:
    """Test that wrapper correctly syncs attributes from platform."""
    
    @patch('refinery.regime_engine_v7.RegimeRiskPlatform')
    def test_run_syncs_regime_attributes(self, mock_platform_class):
        """Test that run() syncs regime-related attributes."""
        from refinery.regime_engine_v7 import RegimeRiskEngineV7
        
        mock_platform = MagicMock()
        mock_platform.n_regimes = 3
        mock_platform.current_regime = 2
        mock_platform.regime_names = {0: "Bear", 1: "Neutral", 2: "Bull"}
        mock_platform.market_beta = 1.5
        mock_platform.market_alpha = 0.0003  # Daily
        mock_platform.regime_alpha = {0: -0.002, 1: 0.0, 2: 0.002}  # Daily
        mock_platform.regime_mu = {0: -0.001, 1: 0.0005, 2: 0.002}  # Daily
        mock_platform.regime_sigma = {0: 0.025, 1: 0.015, 2: 0.02}  # Daily
        mock_platform.regime_duration = {0: {'mean': 15}, 1: {'mean': 60}, 2: {'mean': 25}}
        mock_platform.transition_matrix = np.array([[0.85, 0.1, 0.05], [0.1, 0.8, 0.1], [0.05, 0.1, 0.85]])
        mock_platform.last_price = 125.0
        mock_platform.target_up = 150.0
        mock_platform.target_down = 100.0
        mock_platform.vix_level = 22.0
        mock_platform.garch_vol = 0.28
        mock_platform.realized_vol = 0.30
        mock_platform.idio_vol = 0.18
        mock_platform.hist_up_hit = 0.55
        mock_platform.hist_down_hit = 0.35
        mock_platform.data = pd.DataFrame({'Regime': [1], 'Drawdown': [-0.02], 'Vol_20d': [0.02]})
        
        mock_platform.simulate.return_value = np.random.rand(100, 126) * 100 + 100
        mock_platform.analyze_fpt.return_value = 0.6
        mock_platform.compute_risk_metrics.return_value = {}
        mock_platform.generate_signal.return_value = {}
        
        mock_platform_class.return_value = mock_platform
        
        engine = RegimeRiskEngineV7(ticker="TEST", simulations=100)
        engine.run()
        
        # Check synced attributes
        assert engine.current_regime == 2
        assert engine.regime_names == {0: "Bear", 1: "Neutral", 2: "Bull"}
        assert engine.market_beta == 1.5
        assert engine.last_price == 125.0
        assert engine.target_up == 150.0
        assert engine.target_down == 100.0
        assert engine.n_regimes == 3
        
        # Check annualized values
        assert engine.market_alpha == pytest.approx(0.0003 * 252, rel=0.01)


# ==============================================================================
# TEST: REGIME DIAGNOSTICS
# ==============================================================================

class TestRegimeEngineV7Diagnostics:
    """Test regime diagnostics generation."""
    
    @patch('refinery.regime_engine_v7.RegimeRiskPlatform')
    def test_regime_diagnostics_structure(self, mock_platform_class):
        """Test that regime diagnostics have correct structure."""
        from refinery.regime_engine_v7 import RegimeRiskEngineV7
        
        mock_platform = MagicMock()
        mock_platform.n_regimes = 2
        mock_platform.current_regime = 0
        mock_platform.regime_names = {0: "Low Vol", 1: "High Vol"}
        mock_platform.market_beta = 1.0
        mock_platform.market_alpha = 0.0001
        mock_platform.regime_alpha = {0: 0.0001, 1: -0.0001}
        mock_platform.regime_mu = {0: 0.001, 1: -0.0005}
        mock_platform.regime_sigma = {0: 0.01, 1: 0.03}
        mock_platform.regime_duration = {0: {'mean': 40}, 1: {'mean': 20}}
        mock_platform.transition_matrix = np.array([[0.95, 0.05], [0.1, 0.9]])
        mock_platform.last_price = 100.0
        mock_platform.target_up = 120.0
        mock_platform.target_down = 80.0
        mock_platform.vix_level = 15.0
        mock_platform.garch_vol = 0.18
        mock_platform.realized_vol = 0.20
        mock_platform.idio_vol = 0.12
        mock_platform.hist_up_hit = 0.6
        mock_platform.hist_down_hit = 0.25
        
        # Create realistic regime data
        mock_platform.data = pd.DataFrame({
            'Regime': [0, 0, 0, 1, 1, 0, 0, 1],
            'Drawdown': [-0.01, -0.02, -0.015, -0.08, -0.06, -0.01, -0.02, -0.05],
            'Vol_20d': [0.01, 0.012, 0.011, 0.035, 0.03, 0.015, 0.013, 0.028]
        })
        
        mock_platform.simulate.return_value = np.random.rand(50, 126) * 100 + 80
        mock_platform.analyze_fpt.return_value = 0.55
        mock_platform.compute_risk_metrics.return_value = {}
        mock_platform.generate_signal.return_value = {}
        
        mock_platform_class.return_value = mock_platform
        
        engine = RegimeRiskEngineV7(ticker="TEST", simulations=50)
        results, _ = engine.run()
        
        diagnostics = results['regime_diagnostics']
        
        assert len(diagnostics) == 2
        
        for diag in diagnostics:
            assert 'name' in diag
            assert 'n_samples' in diag
            assert 'avg_dd' in diag
            assert 'avg_vol' in diag
            assert 'mu' in diag
            assert 'sigma' in diag
            assert 'alpha' in diag
            assert 'avg_duration' in diag


# ==============================================================================
# TEST: VIX TOGGLE
# ==============================================================================

class TestRegimeEngineV7VixToggle:
    """Test VIX enable/disable functionality."""
    
    @patch('refinery.regime_engine_v7.RegimeRiskPlatform')
    def test_vix_enabled_by_default(self, mock_platform_class):
        """Test that VIX checking is enabled by default."""
        from refinery.regime_engine_v7 import RegimeRiskEngineV7
        
        mock_platform = MagicMock()
        mock_platform.n_regimes = 3
        mock_platform.current_regime = 1
        mock_platform.regime_names = {}
        mock_platform.market_beta = 1.0
        mock_platform.market_alpha = 0.0
        mock_platform.regime_alpha = {}
        mock_platform.regime_mu = {}
        mock_platform.regime_sigma = {}
        mock_platform.regime_duration = {}
        mock_platform.transition_matrix = np.eye(3)
        mock_platform.last_price = 100.0
        mock_platform.target_up = 120.0
        mock_platform.target_down = 80.0
        mock_platform.vix_level = 18.0
        mock_platform.garch_vol = 0.22
        mock_platform.realized_vol = 0.25
        mock_platform.idio_vol = 0.15
        mock_platform.hist_up_hit = 0.5
        mock_platform.hist_down_hit = 0.5
        mock_platform.data = pd.DataFrame({'Regime': [], 'Drawdown': [], 'Vol_20d': []})
        mock_platform.simulate.return_value = np.random.rand(50, 126) * 100
        mock_platform.analyze_fpt.return_value = 0.5
        mock_platform.compute_risk_metrics.return_value = {}
        mock_platform.generate_signal.return_value = {}
        
        mock_platform_class.return_value = mock_platform
        
        engine = RegimeRiskEngineV7(ticker="TEST")
        assert engine.enable_vix is True
        
        engine.run()
        mock_platform.check_macro_context.assert_called_once()
    
    @patch('refinery.regime_engine_v7.RegimeRiskPlatform')
    def test_vix_disabled(self, mock_platform_class):
        """Test that VIX checking can be disabled."""
        from refinery.regime_engine_v7 import RegimeRiskEngineV7
        
        mock_platform = MagicMock()
        mock_platform.n_regimes = 3
        mock_platform.current_regime = 1
        mock_platform.regime_names = {}
        mock_platform.market_beta = 1.0
        mock_platform.market_alpha = 0.0
        mock_platform.regime_alpha = {}
        mock_platform.regime_mu = {}
        mock_platform.regime_sigma = {}
        mock_platform.regime_duration = {}
        mock_platform.transition_matrix = np.eye(3)
        mock_platform.last_price = 100.0
        mock_platform.target_up = 120.0
        mock_platform.target_down = 80.0
        mock_platform.vix_level = 0
        mock_platform.garch_vol = 0.22
        mock_platform.realized_vol = 0.25
        mock_platform.idio_vol = 0.15
        mock_platform.hist_up_hit = 0.5
        mock_platform.hist_down_hit = 0.5
        mock_platform.data = pd.DataFrame({'Regime': [], 'Drawdown': [], 'Vol_20d': []})
        mock_platform.simulate.return_value = np.random.rand(50, 126) * 100
        mock_platform.analyze_fpt.return_value = 0.5
        mock_platform.compute_risk_metrics.return_value = {}
        mock_platform.generate_signal.return_value = {}
        
        mock_platform_class.return_value = mock_platform
        
        engine = RegimeRiskEngineV7(ticker="TEST", enable_vix=False)
        assert engine.enable_vix is False
        
        engine.run()
        mock_platform.check_macro_context.assert_not_called()


# ==============================================================================
# TEST: BACKWARD COMPATIBILITY
# ==============================================================================

class TestBackwardCompatibility:
    """Test backward compatibility exports."""
    
    def test_regime_risk_engine_alias(self):
        """Test that RegimeRiskEngine is an alias for RegimeRiskEngineV7."""
        from refinery.regime_engine_v7 import RegimeRiskEngine, RegimeRiskEngineV7
        
        assert RegimeRiskEngine is RegimeRiskEngineV7


# ==============================================================================
# TEST: MACRO DICT STRUCTURE
# ==============================================================================

class TestMacroDictStructure:
    """Test the macro conditioning dict structure in results."""
    
    @patch('refinery.regime_engine_v7.RegimeRiskPlatform')
    def test_macro_dict_keys(self, mock_platform_class):
        """Test macro dict has all expected keys."""
        from refinery.regime_engine_v7 import RegimeRiskEngineV7
        
        mock_platform = MagicMock()
        mock_platform.n_regimes = 3
        mock_platform.current_regime = 1
        mock_platform.regime_names = {}
        mock_platform.market_beta = 1.25
        mock_platform.market_alpha = 0.0002
        mock_platform.regime_alpha = {}
        mock_platform.regime_mu = {}
        mock_platform.regime_sigma = {}
        mock_platform.regime_duration = {}
        mock_platform.transition_matrix = np.eye(3)
        mock_platform.last_price = 100.0
        mock_platform.target_up = 120.0
        mock_platform.target_down = 80.0
        mock_platform.vix_level = 19.5
        mock_platform.garch_vol = 0.24
        mock_platform.realized_vol = 0.26
        mock_platform.idio_vol = 0.14
        mock_platform.hist_up_hit = 0.55
        mock_platform.hist_down_hit = 0.30
        mock_platform.data = pd.DataFrame({'Regime': [], 'Drawdown': [], 'Vol_20d': []})
        mock_platform.simulate.return_value = np.random.rand(50, 126) * 100
        mock_platform.analyze_fpt.return_value = 0.55
        mock_platform.compute_risk_metrics.return_value = {}
        mock_platform.generate_signal.return_value = {}
        
        mock_platform_class.return_value = mock_platform
        
        engine = RegimeRiskEngineV7(ticker="TEST", simulations=50)
        results, _ = engine.run()
        
        macro = results['macro']
        
        assert 'beta' in macro
        assert 'alpha_ann' in macro
        assert 'idio_vol' in macro
        assert 'vix' in macro
        assert 'garch_vol' in macro
        assert 'realized_vol' in macro
        
        assert macro['beta'] == 1.25
        assert macro['vix'] == 19.5
        assert macro['garch_vol'] == 0.24
        assert macro['realized_vol'] == 0.26
        assert macro['idio_vol'] == 0.14


# ==============================================================================
# TEST: SANITY DICT STRUCTURE
# ==============================================================================

class TestSanityDictStructure:
    """Test the sanity check dict structure in results."""
    
    @patch('refinery.regime_engine_v7.RegimeRiskPlatform')
    def test_sanity_dict_keys(self, mock_platform_class):
        """Test sanity dict has historical hit rates."""
        from refinery.regime_engine_v7 import RegimeRiskEngineV7
        
        mock_platform = MagicMock()
        mock_platform.n_regimes = 3
        mock_platform.current_regime = 1
        mock_platform.regime_names = {}
        mock_platform.market_beta = 1.0
        mock_platform.market_alpha = 0.0
        mock_platform.regime_alpha = {}
        mock_platform.regime_mu = {}
        mock_platform.regime_sigma = {}
        mock_platform.regime_duration = {}
        mock_platform.transition_matrix = np.eye(3)
        mock_platform.last_price = 100.0
        mock_platform.target_up = 120.0
        mock_platform.target_down = 80.0
        mock_platform.vix_level = 18.0
        mock_platform.garch_vol = 0.22
        mock_platform.realized_vol = 0.25
        mock_platform.idio_vol = 0.15
        mock_platform.hist_up_hit = 0.62
        mock_platform.hist_down_hit = 0.28
        mock_platform.data = pd.DataFrame({'Regime': [], 'Drawdown': [], 'Vol_20d': []})
        mock_platform.simulate.return_value = np.random.rand(50, 126) * 100
        mock_platform.analyze_fpt.return_value = 0.5
        mock_platform.compute_risk_metrics.return_value = {}
        mock_platform.generate_signal.return_value = {}
        
        mock_platform_class.return_value = mock_platform
        
        engine = RegimeRiskEngineV7(ticker="TEST", simulations=50)
        results, _ = engine.run()
        
        sanity = results['sanity']
        
        assert 'hist_up_hit' in sanity
        assert 'hist_down_hit' in sanity
        assert sanity['hist_up_hit'] == 0.62
        assert sanity['hist_down_hit'] == 0.28