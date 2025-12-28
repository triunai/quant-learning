"""
REGIME ENGINE v7.0 ADAPTER FOR STREAMLIT DASHBOARD
===================================================
This adapter imports the battle-tested RegimeRiskPlatform and wraps it
for clean Streamlit integration without modifying the source.
"""

import sys
import os

# Add battle-tested to path for import
BATTLE_TESTED_PATH = os.path.join(os.path.dirname(__file__), '..', 'battle-tested')
if BATTLE_TESTED_PATH not in sys.path:
    sys.path.insert(0, BATTLE_TESTED_PATH)

# Import the production v7.0 platform
# Note: This file is named with a hyphen, so we use importlib
import importlib.util
spec = importlib.util.spec_from_file_location(
    "pltr_test_2",
    os.path.join(BATTLE_TESTED_PATH, "PLTR-test-2.py")
)
pltr_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pltr_module)

# Export the main class for dashboard use
RegimeRiskPlatform = pltr_module.RegimeRiskPlatform


class RegimeRiskEngineV7:
    """
    Streamlit-friendly wrapper for RegimeRiskPlatform v7.0.
    Provides the same interface as the legacy v6.0 engine for dashboard compatibility.
    """

    def __init__(self, ticker,
                 market_ticker="QQQ",
                 days_ahead=126,
                 simulations=5000,
                 target_up=None,
                 target_down=None,
                 stop_loss_pct=0.15,
                 n_regimes=3,
                 enable_vix=True):

        # Initialize the underlying v7.0 platform
        self.platform = RegimeRiskPlatform(
            ticker=ticker,
            market_ticker=market_ticker,
            days_ahead=days_ahead,
            simulations=simulations,
            target_up=target_up,
            target_down=target_down,
            stop_loss_pct=stop_loss_pct,
            n_regimes=n_regimes
        )
        
        self.ticker = ticker
        self.days_ahead = days_ahead
        self.simulations = simulations
        self.enable_vix = enable_vix

        # Expose key attributes for dashboard access
        self.last_price = 0
        self.current_regime = 0
        self.regime_names = {}
        self.market_beta = 0
        self.realized_vol = 0
        self.target_up = target_up
        self.target_down = target_down
        self.transition_matrix = None

    def ingest_data(self):
        """Load and prepare data via the underlying platform."""
        self.platform.ingest_data()
        
        # Sync exposed attributes
        self.last_price = self.platform.last_price
        self.target_up = self.platform.target_up
        self.target_down = self.platform.target_down
        self.realized_vol = self.platform.realized_vol

    def run(self, plot=False, run_full_calibration=False):
        """
        Execute the v7.0 simulation and return Streamlit-compatible results.
        
        Returns:
            results (dict): Contains paths, prob_up, prob_down, risk, etc.
            signal (dict): Contains signal, confidence, reasoning.
        """
        # Build models
        self.platform.build_regime_model()
        self.platform.compute_market_beta()
        if self.enable_vix:
            self.platform.check_macro_context()
        self.platform.run_garch()
        self.platform.compute_historical_validation()
        
        if run_full_calibration:
            self.platform.bucket_asymmetry_diagnostics()
            self.platform.multi_threshold_calibration()
            self.platform.walk_forward_validation()
        
        # Run simulation
        paths = self.platform.simulate()
        self.platform.verify_simulation_invariants(paths)
        
        # Compute outputs
        prob_up = self.platform.analyze_fpt(paths, self.platform.target_up, "up")
        prob_down = self.platform.analyze_fpt(paths, self.platform.target_down, "down")
        risk = self.platform.compute_risk_metrics(paths)
        signal = self.platform.generate_signal(prob_up, prob_down, risk)
        
        # Sync exposed attributes for dashboard
        self.current_regime = self.platform.current_regime
        self.regime_names = self.platform.regime_names
        self.market_beta = self.platform.market_beta
        self.market_alpha = getattr(self.platform, 'market_alpha', 0) * 252  # Annualized
        self.regime_alpha = {k: v * 252 for k, v in getattr(self.platform, 'regime_alpha', {}).items()}
        self.regime_mu = {k: v * 252 for k, v in getattr(self.platform, 'regime_mu', {}).items()}
        self.regime_sigma = {k: v * (252**0.5) for k, v in getattr(self.platform, 'regime_sigma', {}).items()}
        self.regime_duration = getattr(self.platform, 'regime_duration', {})
        self.transition_matrix = self.platform.transition_matrix
        self.last_price = self.platform.last_price
        self.target_up = self.platform.target_up
        self.target_down = self.platform.target_down
        self.vix_level = getattr(self.platform, 'vix_level', 0)
        self.garch_vol = getattr(self.platform, 'garch_vol', 0)
        self.realized_vol = self.platform.realized_vol
        self.idio_vol = getattr(self.platform, 'idio_vol', 0)
        self.n_regimes = self.platform.n_regimes

        # Build regime diagnostics for UI
        regime_diagnostics = []
        for r in range(self.platform.n_regimes):
            mask = self.platform.data['Regime'] == r
            n_samples = int(mask.sum())
            avg_dd = float(self.platform.data.loc[mask, 'Drawdown'].mean())
            avg_vol = float(self.platform.data.loc[mask, 'Vol_20d'].mean())
            name = self.regime_names.get(r, f"R{r}")
            regime_diagnostics.append({
                'name': name,
                'n_samples': n_samples,
                'avg_dd': avg_dd,
                'avg_vol': avg_vol,
                'mu': self.regime_mu.get(r, 0),
                'sigma': self.regime_sigma.get(r, 0),
                'alpha': self.regime_alpha.get(r, 0),
                'avg_duration': self.regime_duration.get(r, {}).get('mean', 0)
            })

        # Build Streamlit-compatible results dict
        results = {
            'paths': paths,
            'prob_up': prob_up,
            'prob_down': prob_down,
            'risk': risk,
            'times_up': [],
            'times_down': [],
            'regime_diagnostics': regime_diagnostics,
            'sanity': {
                'hist_up_hit': self.platform.hist_up_hit,
                'hist_down_hit': self.platform.hist_down_hit,
            },
            'macro': {
                'beta': self.market_beta,
                'alpha_ann': self.market_alpha,
                'idio_vol': self.idio_vol,
                'vix': self.vix_level,
                'garch_vol': self.garch_vol,
                'realized_vol': self.realized_vol,
            }
        }
        
        return results, signal


# For backward compatibility, also expose as RegimeRiskEngine
RegimeRiskEngine = RegimeRiskEngineV7
