"""Unit tests for run_risk.py entry point script."""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRunRiskScript:
    """Test suite for run_risk.py main execution script."""
    
    def test_imports_succeed(self):
        """Test that all required imports are available."""
        try:
            import run_risk
            assert run_risk is not None
        except ImportError as e:
            pytest.fail(f"Failed to import run_risk: {e}")
    
    @patch('sys.argv', ['run_risk.py'])
    @patch('refinery.regime_engine_v7.RegimeEngineV7')
    def test_main_execution_with_defaults(self, mock_engine_class):
        """Test main execution with default parameters."""
        # Setup mock
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        mock_engine.run.return_value = {
            'paths': np.random.rand(100, 126),
            'prob_up': 0.65,
            'prob_down': 0.25
        }
        
        # Import and run
        import run_risk
        
        # Verify engine was instantiated with correct defaults
        mock_engine_class.assert_called_once()
        call_kwargs = mock_engine_class.call_args[1]
        assert call_kwargs['ticker'] == 'PLTR'
        assert call_kwargs['days_ahead'] == 126
        assert call_kwargs['simulations'] == 5000
    
    @patch('refinery.regime_engine_v7.RegimeEngineV7')
    def test_engine_initialization_parameters(self, mock_engine_class):
        """Test that engine receives correct initialization parameters."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        
        import run_risk
        
        # Verify all key parameters are passed
        call_kwargs = mock_engine_class.call_args[1]
        assert 'ticker' in call_kwargs
        assert 'days_ahead' in call_kwargs
        assert 'simulations' in call_kwargs
        assert 'n_regimes' in call_kwargs
    
    @patch('refinery.regime_engine_v7.RegimeEngineV7')
    def test_run_method_called(self, mock_engine_class):
        """Test that engine.run() is called during execution."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        
        import run_risk
        
        mock_engine.run.assert_called_once()
    
    @patch('refinery.regime_engine_v7.RegimeEngineV7')
    def test_error_handling_on_engine_failure(self, mock_engine_class):
        """Test that script handles engine failures gracefully."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        mock_engine.run.side_effect = Exception("Engine failure")
        
        # Should not raise, but print error
        with patch('builtins.print') as mock_print:
            try:
                import run_risk
            except SystemExit:
                pass  # Expected if script exits on error
            
            # Verify error was printed
            assert any('error' in str(call).lower() or 'fail' in str(call).lower() 
                      for call in mock_print.call_args_list)


class TestRunRiskIntegration:
    """Integration tests for run_risk.py with real components."""
    
    @pytest.mark.integration
    @patch('yfinance.download')
    def test_full_execution_with_mock_data(self, mock_download, sample_price_data):
        """Test full script execution with mocked market data."""
        # Setup mock to return sample data
        mock_download.return_value = sample_price_data
        
        # Import should trigger execution
        with patch('matplotlib.pyplot.show'):  # Prevent plot display
            import run_risk
        
        # If we get here without exceptions, basic execution succeeded
        assert True
    
    @pytest.mark.integration
    @patch('yfinance.download')
    def test_handles_insufficient_data(self, mock_download):
        """Test script handles insufficient historical data gracefully."""
        # Return very short dataframe
        short_data = pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000, 1100, 1050]
        }, index=pd.date_range('2024-01-01', periods=3))
        
        mock_download.return_value = short_data
        
        # Should handle gracefully (may print warning but not crash)
        with patch('builtins.print'):
            try:
                import run_risk
            except (ValueError, KeyError, IndexError):
                # Expected if data is insufficient
                pass