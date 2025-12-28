"""Unit tests for run_risk.py entry point script.

Tests the main entry point that runs the Regime Risk Engine.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import importlib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


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


# ==============================================================================
# TEST: RUN_RISK MODULE
# ==============================================================================

class TestRunRiskModule:
    """Test suite for run_risk.py module structure."""
    
    def test_module_has_main_function(self):
        """Test that run_risk module has a main function."""
        # We need to mock the import to avoid running the actual engine
        with patch.dict('sys.modules', {'refinery.regime_engine': MagicMock()}):
            # Force reimport
            if 'run_risk' in sys.modules:
                del sys.modules['run_risk']
            
            import run_risk
            assert hasattr(run_risk, 'main')
            assert callable(run_risk.main)
    
    def test_module_imports_regime_engine(self):
        """Test that run_risk imports RegimeRiskEngine."""
        # Check that the import statement exists in the source
        run_risk_path = Path(__file__).parent.parent / 'run_risk.py'
        source = run_risk_path.read_text()
        
        assert 'RegimeRiskEngine' in source
        assert 'refinery.regime_engine' in source


class TestMainFunction:
    """Test suite for the main() function."""
    
    @patch('refinery.regime_engine.RegimeRiskEngine')
    def test_main_creates_engine_with_pltr(self, mock_engine_class):
        """Test that main() creates an engine for PLTR."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        mock_engine.run.return_value = (
            {'prob_up': 0.65, 'prob_down': 0.25},
            {'signal': 'HOLD', 'confidence': 50, 'reasoning': []},
            None  # fig
        )
        
        # Force reimport with mocked engine
        if 'run_risk' in sys.modules:
            del sys.modules['run_risk']
        
        import run_risk
        run_risk.main()
        
        # Verify engine was created with PLTR ticker
        mock_engine_class.assert_called_once()
        call_kwargs = mock_engine_class.call_args[1]
        assert call_kwargs['ticker'] == 'PLTR'
    
    @patch('refinery.regime_engine.RegimeRiskEngine')
    def test_main_calls_ingest_data(self, mock_engine_class):
        """Test that main() calls ingest_data() on the engine."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        mock_engine.run.return_value = (
            {'prob_up': 0.65, 'prob_down': 0.25},
            {'signal': 'HOLD', 'confidence': 50, 'reasoning': []},
            None
        )
        
        if 'run_risk' in sys.modules:
            del sys.modules['run_risk']
        
        import run_risk
        run_risk.main()
        
        mock_engine.ingest_data.assert_called_once()
    
    @patch('refinery.regime_engine.RegimeRiskEngine')
    def test_main_calls_run_with_plot(self, mock_engine_class):
        """Test that main() calls run(plot=True)."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        mock_engine.run.return_value = (
            {'prob_up': 0.65, 'prob_down': 0.25},
            {'signal': 'HOLD', 'confidence': 50, 'reasoning': []},
            None
        )
        
        if 'run_risk' in sys.modules:
            del sys.modules['run_risk']
        
        import run_risk
        run_risk.main()
        
        mock_engine.run.assert_called_once_with(plot=True)
    
    @patch('refinery.regime_engine.RegimeRiskEngine')
    @patch('builtins.open', create=True)
    def test_main_saves_summary_file(self, mock_open, mock_engine_class):
        """Test that main() saves a summary file."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        mock_engine.run.return_value = (
            {'prob_up': 0.65, 'prob_down': 0.25},
            {'signal': 'BUY', 'confidence': 75, 'reasoning': ['Good setup']},
            None
        )
        
        mock_file = MagicMock()
        mock_open.return_value.__enter__ = Mock(return_value=mock_file)
        mock_open.return_value.__exit__ = Mock(return_value=False)
        
        if 'run_risk' in sys.modules:
            del sys.modules['run_risk']
        
        import run_risk
        run_risk.main()
        
        # Verify file was opened for writing
        mock_open.assert_called_with("risk_summary.txt", "w")
    
    @patch('refinery.regime_engine.RegimeRiskEngine')
    def test_main_handles_run_error(self, mock_engine_class):
        """Test that main() handles errors from run() gracefully."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        mock_engine.run.side_effect = ValueError("Simulation failed")
        
        if 'run_risk' in sys.modules:
            del sys.modules['run_risk']
        
        import run_risk
        
        # Should exit with error code
        with pytest.raises(SystemExit) as exc_info:
            run_risk.main()
        
        assert exc_info.value.code == 1


class TestRunRiskOutputStructure:
    """Test the expected output structure from run_risk."""
    
    @patch('refinery.regime_engine.RegimeRiskEngine')
    @patch('builtins.open', create=True)
    def test_writes_signal_to_file(self, mock_open, mock_engine_class):
        """Test that signal information is written to file."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        mock_engine.run.return_value = (
            {'prob_up': 0.65, 'prob_down': 0.25},
            {'signal': 'BUY', 'confidence': 80, 'reasoning': ['Bullish pattern']},
            None
        )
        
        mock_file = MagicMock()
        mock_open.return_value.__enter__ = Mock(return_value=mock_file)
        mock_open.return_value.__exit__ = Mock(return_value=False)
        
        if 'run_risk' in sys.modules:
            del sys.modules['run_risk']
        
        import run_risk
        run_risk.main()
        
        # Check that signal was written
        write_calls = [str(call) for call in mock_file.write.call_args_list]
        output = ''.join(write_calls)
        
        assert 'BUY' in output or 'Signal' in output
    
    @patch('refinery.regime_engine.RegimeRiskEngine')
    def test_handles_missing_figure(self, mock_engine_class):
        """Test that main() handles when no figure is returned."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        mock_engine.run.return_value = (
            {'prob_up': 0.5, 'prob_down': 0.5},
            {'signal': 'HOLD', 'confidence': 50, 'reasoning': []},
            None  # No figure
        )
        
        if 'run_risk' in sys.modules:
            del sys.modules['run_risk']
        
        import run_risk
        
        # Should not raise
        with patch('builtins.open', create=True):
            run_risk.main()
    
    @patch('refinery.regime_engine.RegimeRiskEngine')
    def test_saves_figure_when_provided(self, mock_engine_class):
        """Test that main() saves figure when one is returned."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        
        mock_fig = MagicMock()
        mock_engine.run.return_value = (
            {'prob_up': 0.6, 'prob_down': 0.3},
            {'signal': 'HOLD', 'confidence': 55, 'reasoning': []},
            mock_fig
        )
        
        if 'run_risk' in sys.modules:
            del sys.modules['run_risk']
        
        import run_risk
        
        with patch('builtins.open', create=True):
            run_risk.main()
        
        # Verify savefig was called
        mock_fig.savefig.assert_called_once_with("pltr_risk.png")


class TestImportError:
    """Test import error handling."""
    
    def test_source_has_import_error_handling(self):
        """Test that run_risk.py has import error handling."""
        run_risk_path = Path(__file__).parent.parent / 'run_risk.py'
        source = run_risk_path.read_text()
        
        assert 'ImportError' in source
        assert 'sys.exit' in source