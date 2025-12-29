"""
PROJECT PLTR - CONSOLIDATED STREAMLIT RISK DASHBOARD
=====================================================
Consolidates:
1. v7.0 Regime Risk Platform (GMM + Alpha/Beta Factor Model)
2. Semi-Markov Model with full log capture
3. Signals Factory integration

All outputs captured + exportable to JSON.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import io
import sys
import logging
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timedelta
from scipy import stats
# Import engines - use absolute path based on file location
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Parent of to_refine/

# Add project root and to_refine to path
# Add project root and to_refine to path
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

print(f"DEBUG: PROJECT_ROOT={PROJECT_ROOT}")
print(f"DEBUG: sys.path={sys.path}")

from regime_engine_v7 import RegimeRiskEngineV7
from refinery.semi_markov import SemiMarkovModel
from signals_factory.signal_vol_compression import VolCompressionSignal
from signals_factory.aggregator import SignalAggregator
from stationary_bootstrap import StationaryBootstrap, BootstrapConfig

# =============================================================================
# LOGGING CAPTURE UTILITY
# =============================================================================
class LogCapture:
    """Captures print statements and logging to a string buffer."""
    
    def __init__(self):
        self.logs = []
        self.start_time = None
        
    def start(self):
        self.start_time = datetime.now()
        self.logs = []
        self.logs.append(f"╔═══════════════════════════════════════════════════════════════╗")
        self.logs.append(f"║  EXECUTION LOG - Started at {self.start_time.isoformat()}    ║")
        self.logs.append(f"╚═══════════════════════════════════════════════════════════════╝\n")
    
    def log(self, msg: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        prefix = {
            "INFO": "ℹ️ ",
            "SUCCESS": "✅",
            "WARNING": "⚠️ ",
            "ERROR": "❌",
            "DEBUG": "🔍",
        }.get(level, "  ")
        self.logs.append(f"[{timestamp}] {prefix} {msg}")
    
    def section(self, title: str):
        self.logs.append(f"\n{'─'*60}")
        self.logs.append(f"▶ {title}")
        self.logs.append(f"{'─'*60}")
    
    def finish(self):
        elapsed = datetime.now() - self.start_time if self.start_time else timedelta(0)
        self.logs.append(f"\n{'═'*60}")
        self.logs.append(f"✅ COMPLETE | Elapsed: {elapsed.total_seconds():.2f}s")
        self.logs.append(f"{'═'*60}")
    
    def get_log_text(self) -> str:
        return "\n".join(self.logs)


# =============================================================================
# STREAMLIT CONFIG
# =============================================================================
st.set_page_config(
    layout="wide", 
    page_title="Project PLTR - Consolidated Dashboard",
    page_icon="🦅"
)
plt.style.use('dark_background')

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.header("🎛️ Mission Control")
st.sidebar.caption("Consolidated: v7.0 + Semi-Markov + Signals Factory")
st.sidebar.divider()

# Core Parameters
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY")
market_ticker = st.sidebar.text_input("Market Benchmark", value="QQQ")
days_ahead = st.sidebar.slider("Simulation Days", 63, 252, 126)
simulations = st.sidebar.slider("Monte Carlo Paths", 500, 5000, 1000)
n_regimes = st.sidebar.selectbox("GMM Regimes", [2, 3, 4, 5], index=2)
enable_vix = st.sidebar.checkbox("Enable VIX Filter", value=True)

st.sidebar.divider()
st.sidebar.subheader("🔧 Module Selection")
run_mode_b = st.sidebar.checkbox("🎯 Run Mode B (Stationary Bootstrap)", value=True, help="Ground truth benchmark - block samples historical data with no model assumptions.")
mode_b_config = st.sidebar.selectbox("Mode B Config", ["debug", "fast", "production"], index=1, help="debug=1k sims, fast=5k sims, production=50k sims") if run_mode_b else "fast"
run_v7 = st.sidebar.checkbox("Run v7.0 Regime Engine (Mode A)", value=False)
run_full_calibration = st.sidebar.checkbox("⚡ Full Calibration Suite", value=False, help="Runs multi-threshold calibration, walk-forward validation, and bucket asymmetry diagnostics. Takes longer but provides rigorous validation.")
run_semi_markov = st.sidebar.checkbox("Run Semi-Markov Model", value=False)
run_signals = st.sidebar.checkbox("Run Signals Factory", value=False)

st.sidebar.divider()
run_btn = st.sidebar.button("🚀 RUN ALL SELECTED", type="primary")

# =============================================================================
# MAIN EXECUTION FUNCTIONS (with logging)
# =============================================================================

def run_regime_engine_v7(ticker, market_ticker, days_ahead, simulations, n_regimes, enable_vix, run_full_calibration, log: LogCapture):
    """Run v7.0 Regime Risk Platform with FULL logging of all diagnostics."""
    log.section("V7.0 REGIME RISK PLATFORM (BATTLE-TESTED)")
    log.log(f"Ticker: {ticker} | Benchmark: {market_ticker}")
    log.log(f"Params: {days_ahead} days, {simulations} sims, {n_regimes} regimes, VIX={enable_vix}")
    log.log(f"Full Calibration Suite: {run_full_calibration}")
    
    calibration_results = {}
    
    try:
        log.log("Initializing RegimeRiskEngineV7...")
        engine = RegimeRiskEngineV7(
            ticker=ticker,
            market_ticker=market_ticker,
            days_ahead=days_ahead,
            simulations=simulations,
            n_regimes=n_regimes,
            enable_vix=enable_vix,
        )
        
        log.log("Ingesting market data...")
        engine.ingest_data()
        log.log(f"Last price: ${engine.last_price:.2f}", "SUCCESS")
        log.log(f"Target Up: ${engine.target_up:.0f} (+{((engine.target_up/engine.last_price)-1)*100:.0f}%)")
        log.log(f"Target Down: ${engine.target_down:.0f} ({((engine.target_down/engine.last_price)-1)*100:.0f}%)")
        log.log(f"Realized Vol: {engine.realized_vol:.1%}")
        
        # Access the underlying platform for full diagnostics
        platform = engine.platform
        
        log.log("Building GMM regime model...")
        platform.build_regime_model()
        
        # Log regime diagnostics
        log.log("Regime Diagnostics:", "SUCCESS")
        for r in range(platform.n_regimes):
            mask = platform.data['Regime'] == r
            n_samples = int(mask.sum())
            avg_dd = float(platform.data.loc[mask, 'Drawdown'].mean())
            avg_vol = float(platform.data.loc[mask, 'Vol_20d'].mean())
            mu = platform.regime_mu.get(r, 0) * 252
            sigma = platform.regime_sigma.get(r, 0) * (252**0.5)
            name = platform.regime_names.get(r, f"R{r}")
            duration = platform.regime_duration.get(r, {})
            avg_dur = duration.get('mean', 0)
            log.log(f"  {name}: n={n_samples}, DD={avg_dd:.1%}, Vol={avg_vol:.1%}, μ={mu:+.1%}, σ={sigma:.1%}, AvgDur={avg_dur:.1f}d", "DEBUG")
        
        log.log("Computing market alpha/beta...")
        platform.compute_market_beta()
        log.log(f"Beta: {platform.market_beta:.2f} | Alpha (ann): {platform.market_alpha*252:+.1%}", "SUCCESS")
        log.log(f"Idiosyncratic Vol: {platform.idio_vol:.1%}")
        
        # Per-regime alpha
        for r in range(platform.n_regimes):
            name = platform.regime_names.get(r, f"R{r}")
            alpha = platform.regime_alpha.get(r, 0) * 252
            log.log(f"  {name} alpha: {alpha:+.1%}", "DEBUG")
        
        if enable_vix:
            log.log("Checking macro context (VIX + Anomaly)...")
            platform.check_macro_context()
            log.log(f"VIX: {platform.vix_level:.1f} | Jump Prob: {platform.jump_prob:.0%}", "SUCCESS")
            log.log(f"Anomaly Detected: {'YES!' if platform.is_anomaly else 'No'}")
        
        log.log("Running GARCH volatility forecast...")
        platform.run_garch()
        log.log(f"GARCH Vol: {platform.garch_vol:.1%} | Realized: {platform.realized_vol:.1%}")
        
        log.log("Computing historical validation (first-passage)...")
        platform.compute_historical_validation()
        log.log(f"Historical Up Hit: {platform.hist_up_hit:.1%}")
        log.log(f"Historical Down Hit: {platform.hist_down_hit:.1%}")
        
        # === FULL CALIBRATION SUITE ===
        if run_full_calibration:
            log.section("FULL CALIBRATION SUITE")
            
            # Bucket Asymmetry Diagnostics
            log.log("Running bucket asymmetry diagnostics...", "INFO")
            for r in range(platform.n_regimes):
                mask = platform.data['Regime'] == r
                returns = platform.data.loc[mask, 'Log_Ret'].values
                if len(returns) >= 10:
                    name = platform.regime_names.get(r, f"R{r}")
                    mean = np.mean(returns) * 252 * 100
                    std = np.std(returns) * np.sqrt(252) * 100
                    skew = float(stats.skew(returns))
                    p5 = np.percentile(returns, 5) * 100
                    p95 = np.percentile(returns, 95) * 100
                    log.log(f"  {name}: μ={mean:+.1f}%, σ={std:.1f}%, skew={skew:+.2f}, 5%={p5:+.1f}%, 95%={p95:+.1f}%", "DEBUG")
            
            # Multi-Threshold Calibration
            log.log("Running multi-threshold calibration...", "INFO")
            closes = platform.data['Close'].values
            n = len(closes)
            thresholds = [0.10, 0.20, 0.30, 0.50]
            horizons = [21, 63, 126]
            
            mt_results = []
            for horizon in horizons:
                for thresh in thresholds:
                    up_hits = 0
                    down_hits = 0
                    windows = 0
                    for t in range(n - horizon):
                        start_price = closes[t]
                        path = closes[t:t + horizon]
                        if np.max(path) >= start_price * (1 + thresh):
                            up_hits += 1
                        if np.min(path) <= start_price * (1 - thresh):
                            down_hits += 1
                        windows += 1
                    if windows > 0:
                        mt_results.append({
                            'horizon': horizon,
                            'threshold': thresh,
                            'up_hit': up_hits / windows,
                            'down_hit': down_hits / windows
                        })
            
            log.log(f"  {'Horizon':<10} {'Thresh':<10} {'Up Hit':<12} {'Down Hit':<12}", "DEBUG")
            for r in mt_results:
                log.log(f"  {r['horizon']:<10} {r['threshold']*100:+.0f}%{'':<6} {r['up_hit']:.1%}{'':<6} {r['down_hit']:.1%}", "DEBUG")
            
            calibration_results['multi_threshold'] = mt_results
            
            # Walk-Forward Validation
            log.log("Running walk-forward validation (5 folds)...", "INFO")
            wf_result = platform.walk_forward_validation(n_folds=5)
            if wf_result:
                log.log(f"  Brier Score: {wf_result['brier']:.3f} (lower=better, 0.25=random)", "SUCCESS")
                log.log(f"  Calibration Error: {wf_result['cal_error']:.3f}")
                log.log(f"  Mean Predicted: {np.mean(wf_result['predictions']):.1%}")
                log.log(f"  Mean Actual: {np.mean(wf_result['actuals']):.1%}")
                calibration_results['walk_forward'] = {
                    'brier': float(wf_result['brier']),
                    'cal_error': float(wf_result['cal_error']),
                    'mean_predicted': float(np.mean(wf_result['predictions'])),
                    'mean_actual': float(np.mean(wf_result['actuals']))
                }
        
        log.section("MONTE CARLO SIMULATION")
        log.log(f"Running {simulations:,} paths x {days_ahead} days...")
        paths = platform.simulate()
        log.log(f"Simulation complete: {paths.shape}", "SUCCESS")
        
        # Verify simulation invariants
        log.log("Verifying simulation invariants (Sim vs Historical)...", "INFO")
        sim_returns = np.diff(np.log(paths), axis=1).flatten()
        hist_returns = platform.data['Log_Ret'].values
        
        sim_mean = np.mean(sim_returns)
        hist_mean = np.mean(hist_returns)
        sim_std = np.std(sim_returns)
        hist_std = np.std(hist_returns)
        sim_skew = float(stats.skew(sim_returns))
        hist_skew = float(stats.skew(hist_returns))
        sim_kurt = float(stats.kurtosis(sim_returns))
        hist_kurt = float(stats.kurtosis(hist_returns))
        
        mean_ok = abs(sim_mean - hist_mean) < 0.001
        std_ok = abs(sim_std - hist_std) / hist_std < 0.3
        skew_ok = abs(sim_skew - hist_skew) < 1.0
        kurt_ok = abs(sim_kurt - hist_kurt) < 3.0
        
        log.log(f"  {'Metric':<12} {'Sim':<12} {'Hist':<12} {'Match?'}", "DEBUG")
        log.log(f"  {'Mean':<12} {sim_mean*100:+.3f}%{'':<5} {hist_mean*100:+.3f}%{'':<5} {'[OK]' if mean_ok else '[FAIL]'}", "DEBUG")
        log.log(f"  {'Std':<12} {sim_std*100:.3f}%{'':<6} {hist_std*100:.3f}%{'':<6} {'[OK]' if std_ok else '[FAIL]'}", "DEBUG")
        log.log(f"  {'Skew':<12} {sim_skew:+.2f}{'':<8} {hist_skew:+.2f}{'':<8} {'[OK]' if skew_ok else '[FAIL]'}", "DEBUG")
        log.log(f"  {'Kurtosis':<12} {sim_kurt:.2f}{'':<9} {hist_kurt:.2f}{'':<9} {'[OK]' if kurt_ok else '[FAIL]'}", "DEBUG")
        
        if not (mean_ok and std_ok and skew_ok and kurt_ok):
            log.log("WARNING: Simulation invariants don't match! Check simulation logic.", "WARNING")
        else:
            log.log("All invariants passed!", "SUCCESS")
        
        calibration_results['invariants'] = {
            'sim_mean': float(sim_mean),
            'hist_mean': float(hist_mean),
            'sim_std': float(sim_std),
            'hist_std': float(hist_std),
            'sim_skew': sim_skew,
            'hist_skew': hist_skew,
            'sim_kurtosis': sim_kurt,
            'hist_kurtosis': hist_kurt,
            'all_passed': mean_ok and std_ok and skew_ok and kurt_ok
        }
        
        # Analyze first-passage times
        log.log("Analyzing first-passage times...")
        prob_up = platform.analyze_fpt(paths, platform.target_up, "up")
        prob_down = platform.analyze_fpt(paths, platform.target_down, "down")
        
        log.log(f"Prob Up (${platform.target_up:.0f}): {prob_up:.1%} (hist: {platform.hist_up_hit:.1%})", "SUCCESS")
        log.log(f"Prob Down (${platform.target_down:.0f}): {prob_down:.1%} (hist: {platform.hist_down_hit:.1%})", "SUCCESS")
        
        # Compute risk metrics
        log.log("Computing risk metrics...")
        risk = platform.compute_risk_metrics(paths)
        log.log(f"VaR(95): {risk['var_95']*100:+.1f}%")
        log.log(f"CVaR(95): {risk['cvar_95']*100:+.1f}%")
        log.log(f"P(MaxDD > 20%): {risk['prob_dd_20']:.1%}")
        log.log(f"P(MaxDD > 30%): {risk['prob_dd_30']:.1%}")
        log.log(f"P(Stop Breach): {risk['prob_stop']:.1%}")
        log.log(f"Kelly Fraction: {risk['kelly_fraction']:.0%}")
        log.log(f"Win Rate: {risk['win_rate']:.1%}")
        
        # Generate signal
        log.log("Generating signal...")
        signal = platform.generate_signal(prob_up, prob_down, risk)
        log.log(f"SIGNAL: {signal['signal']} ({signal['confidence']}% confidence)", "SUCCESS")
        for reason in signal['reasoning']:
            log.log(f"  • {reason}", "DEBUG")
        
        # Sync engine attributes
        engine.current_regime = platform.current_regime
        engine.regime_names = platform.regime_names
        engine.market_beta = platform.market_beta
        engine.market_alpha = getattr(platform, 'market_alpha', 0) * 252
        engine.regime_alpha = {k: v * 252 for k, v in getattr(platform, 'regime_alpha', {}).items()}
        engine.regime_mu = {k: v * 252 for k, v in getattr(platform, 'regime_mu', {}).items()}
        engine.regime_sigma = {k: v * (252**0.5) for k, v in getattr(platform, 'regime_sigma', {}).items()}
        engine.regime_duration = getattr(platform, 'regime_duration', {})
        engine.transition_matrix = platform.transition_matrix
        engine.vix_level = getattr(platform, 'vix_level', 0)
        engine.garch_vol = getattr(platform, 'garch_vol', 0)
        engine.idio_vol = getattr(platform, 'idio_vol', 0)
        engine.n_regimes = platform.n_regimes
        
        # Build regime diagnostics
        regime_diagnostics = []
        for r in range(platform.n_regimes):
            mask = platform.data['Regime'] == r
            n_samples = int(mask.sum())
            avg_dd = float(platform.data.loc[mask, 'Drawdown'].mean())
            avg_vol = float(platform.data.loc[mask, 'Vol_20d'].mean())
            name = engine.regime_names.get(r, f"R{r}")
            regime_diagnostics.append({
                'name': name,
                'n_samples': n_samples,
                'avg_dd': avg_dd,
                'avg_vol': avg_vol,
                'mu': engine.regime_mu.get(r, 0),
                'sigma': engine.regime_sigma.get(r, 0),
                'alpha': engine.regime_alpha.get(r, 0),
                'avg_duration': engine.regime_duration.get(r, {}).get('mean', 0)
            })
        
        # Build full results
        results = {
            'paths': paths,
            'prob_up': prob_up,
            'prob_down': prob_down,
            'risk': risk,
            'regime_diagnostics': regime_diagnostics,
            'sanity': {
                'hist_up_hit': platform.hist_up_hit,
                'hist_down_hit': platform.hist_down_hit,
            },
            'macro': {
                'beta': platform.market_beta,
                'alpha_ann': engine.market_alpha,
                'idio_vol': engine.idio_vol,
                'vix': engine.vix_level,
                'garch_vol': engine.garch_vol,
                'realized_vol': platform.realized_vol,
            },
            'calibration': calibration_results
        }
        
        return engine, results, signal
        
    except Exception as e:
        log.log(f"V7.0 Engine failed: {str(e)}", "ERROR")
        import traceback
        log.log(traceback.format_exc(), "ERROR")
        return None, None, None


def run_semi_markov_model(ticker, log: LogCapture):
    """Run Semi-Markov model with logging."""
    log.section("SEMI-MARKOV MODEL")
    log.log(f"Ticker: {ticker}")
    
    try:
        log.log("Initializing SemiMarkovModel with 5 states...")
        model = SemiMarkovModel(ticker, n_states=5)
        
        log.log("Processing data (5y history)...")
        model._process_data(period="5y")
        log.log(f"Data points: {len(model.data)}", "SUCCESS")
        
        log.log("Fitting duration distributions...")
        model.fit_distributions()
        
        # Log distribution parameters
        for state_idx, state_name in model.state_map.items():
            params = model.duration_params.get(state_idx, {})
            dist_type = params.get('dist', 'unknown')
            log.log(f"  State {state_idx} ({state_name}): {dist_type} distribution", "DEBUG")
        
        log.log("Running Monte Carlo simulation...")
        paths = model.run_simulation(days=126, simulations=500)
        log.log(f"Simulated {paths.shape[0]} paths x {paths.shape[1]} days", "SUCCESS")
        
        log.log("Validating model...")
        validation = model.validate_model(paths)
        log.log(f"Real Vol ACF(1): {validation['real_vol_acf_lag1']:.3f}")
        log.log(f"Sim Vol ACF(1): {validation['sim_vol_acf_lag1']:.3f}")
        log.log(f"Vol Clustering Error: {validation['vol_clustering_error']:.3f}")
        
        # Current state info
        current_state = int(model.data['State_Idx'].iloc[-1])
        last_block = model.data['block'].iloc[-1]
        days_in_state = len(model.data[model.data['block'] == last_block])
        fatigue = model.regime_fatigue_score(current_state, days_in_state)
        position_size = model.get_position_size(current_state, days_in_state)
        
        log.log(f"Current State: {model.state_map.get(current_state, 'Unknown')} (State {current_state})", "SUCCESS")
        log.log(f"Days in State: {days_in_state}")
        log.log(f"Regime Fatigue: {fatigue:.2%}")
        log.log(f"Recommended Position Size: {position_size:.1%}")
        
        return model, paths, validation, {
            'current_state': current_state,
            'state_name': model.state_map.get(current_state, 'Unknown'),
            'days_in_state': days_in_state,
            'fatigue': fatigue,
            'position_size': position_size
        }
        
    except Exception as e:
        log.log(f"Semi-Markov failed: {str(e)}", "ERROR")
        import traceback
        log.log(traceback.format_exc(), "ERROR")
        return None, None, None, None


def run_signals_factory(ticker, log: LogCapture):
    """Run Signals Factory with logging."""
    log.section("SIGNALS FACTORY")
    log.log(f"Ticker: {ticker}")
    
    try:
        import yfinance as yf
        
        log.log("Fetching OHLC data for signal calculation...")
        data = yf.download(ticker, period="2y", progress=False)
        
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        log.log(f"Data loaded: {len(data)} rows", "SUCCESS")
        
        # Initialize signals
        log.log("Initializing VolCompressionSignal...")
        vol_signal = VolCompressionSignal(lookback=20, history_window=252)
        
        log.log("Fitting signal on historical data...")
        vol_signal.fit(data)
        
        log.log("Generating prediction...")
        vol_result = vol_signal.predict()
        
        log.log(f"Compression Score: {vol_result['score']:.1f}/100", "SUCCESS")
        log.log(f"Position Sizing Mult: {vol_result['sizing']:.2f}")
        for reason in vol_result['reasoning']:
            log.log(f"  • {reason}", "DEBUG")
        
        # Create aggregator (with just vol signal for now)
        signals = [vol_signal]
        aggregator = SignalAggregator(signals)
        agg_result = aggregator.aggregate()
        
        log.log(f"Aggregated Final Score: {agg_result['final_score']:.2f}")
        log.log(f"Aggregated Final Sizing: {agg_result['final_sizing']:.2f}")
        
        return vol_signal, agg_result
        
    except Exception as e:
        log.log(f"Signals Factory failed: {str(e)}", "ERROR")
        import traceback
        log.log(traceback.format_exc(), "ERROR")
        return None, None


def run_mode_b_bootstrap(ticker, market_ticker, days_ahead, config_mode, log: LogCapture):
    """Run Mode B: Stationary Bootstrap (Ground Truth Benchmark)."""
    log.section("MODE B: STATIONARY BOOTSTRAP (GROUND TRUTH)")
    log.log(f"Ticker: {ticker} | Benchmark: {market_ticker}")
    log.log(f"Config: {config_mode} | Days: {days_ahead}")
    
    try:
        # Create config
        config = BootstrapConfig(mode=config_mode, seed=42)
        log.log(f"Simulations: {config.simulations:,}")
        log.log(f"Mean block length: {config.mean_block_length}")
        
        log.log("Initializing StationaryBootstrap engine...")
        engine = StationaryBootstrap(
            ticker=ticker,
            market_ticker=market_ticker,
            days_ahead=days_ahead,
            config=config,
        )
        
        log.log("Ingesting market data...")
        engine.ingest_data()
        log.log(f"Last price: ${engine.last_price:.2f}", "SUCCESS")
        log.log(f"Target Up: ${engine.target_up:.0f} (+{((engine.target_up/engine.last_price)-1)*100:.0f}%)")
        log.log(f"Target Down: ${engine.target_down:.0f} ({((engine.target_down/engine.last_price)-1)*100:.0f}%)")
        log.log(f"Beta: {engine.beta:.2f} | Alpha (ann): {engine.alpha * 252:+.1%}")
        log.log(f"Realized Vol: {engine.realized_vol:.1%}")
        
        log.log(f"Running {config.simulations:,} simulation paths...")
        paths = engine.simulate()
        log.log(f"Simulation complete: {paths.shape}", "SUCCESS")
        
        # Compute diagnostics
        log.section("DIAGNOSTICS (Sim vs Historical)")
        diagnostics = engine.compute_diagnostics(paths)
        
        gatekeepers = [d for d in diagnostics if d.is_gatekeeper]
        info_metrics = [d for d in diagnostics if not d.is_gatekeeper]
        
        log.log("GATEKEEPER METRICS (Must Pass):", "INFO")
        for d in gatekeepers:
            status = "✅" if d.passed else "❌"
            log.log(f"  {status} {d.metric}: Sim={d.sim_value:.4f}, Hist={d.hist_value:.4f}", 
                   "SUCCESS" if d.passed else "ERROR")
        
        log.log("DIAGNOSTIC METRICS (Info Only):", "INFO")
        for d in info_metrics:
            status = "✅" if d.passed else "⚠️"
            log.log(f"  {status} {d.metric}: Sim={d.sim_value:.4f}, Hist={d.hist_value:.4f}", "DEBUG")
        
        all_gates_passed = all(d.passed for d in gatekeepers)
        log.log(f"GATEKEEPER STATUS: {'ALL PASSED' if all_gates_passed else 'SOME FAILED'}", 
               "SUCCESS" if all_gates_passed else "ERROR")
        
        # First-passage analysis
        log.section("FIRST-PASSAGE ANALYSIS")
        fpt = engine.analyze_first_passage(paths)
        log.log(f"P(Up First): {fpt['prob_up_first']:.1%}")
        log.log(f"P(Down First): {fpt['prob_down_first']:.1%}")
        log.log(f"P(Neither): {fpt['prob_neither']:.1%}")
        
        if not np.isnan(fpt['mean_tau_up']):
            log.log(f"Mean Time to Up: {fpt['mean_tau_up']:.1f} days")
        if not np.isnan(fpt['mean_tau_down']):
            log.log(f"Mean Time to Down: {fpt['mean_tau_down']:.1f} days")
        
        # Historical validation
        hist_rates = engine.compute_historical_hit_rates()
        log.log(f"Historical Up Hit Rate: {hist_rates['hist_up_hit']:.1%} ({hist_rates['n_windows']} windows)")
        log.log(f"Historical Down Hit Rate: {hist_rates['hist_down_hit']:.1%}")
        
        # Risk metrics
        log.section("RISK METRICS")
        risk = engine.compute_risk_metrics(paths)
        log.log(f"VaR(95): {risk['var_95']*100:+.1f}%")
        log.log(f"CVaR(95): {risk['cvar_95']*100:+.1f}%")
        log.log(f"P(MaxDD > 20%): {risk['prob_dd_20']:.1%}")
        log.log(f"P(MaxDD > 30%): {risk['prob_dd_30']:.1%}")
        log.log(f"Win Rate: {risk['win_rate']:.1%}")
        log.log(f"Kelly Fraction: {risk['kelly_fraction']:.0%}")
        
        # Build results
        results = {
            'paths': paths,
            'diagnostics': diagnostics,
            'first_passage': fpt,
            'risk': risk,
            'historical_validation': hist_rates,
            'all_gates_passed': all_gates_passed,
        }
        
        return engine, results
        
    except Exception as e:
        log.log(f"Mode B failed: {str(e)}", "ERROR")
        import traceback
        log.log(traceback.format_exc(), "ERROR")
        return None, None


# =============================================================================
# MAIN UI
# =============================================================================

if run_btn:
    log = LogCapture()
    log.start()
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Results storage
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "ticker": ticker,
        "market_ticker": market_ticker,
        "parameters": {
            "days_ahead": days_ahead,
            "simulations": simulations,
            "n_regimes": n_regimes,
            "enable_vix": enable_vix
        },
        "modules": {}
    }
    
    v7_engine = None
    v7_results = None
    v7_signal = None
    mode_b_engine = None
    mode_b_results = None
    sm_model = None
    sm_paths = None
    sm_validation = None
    sm_state_info = None
    sf_signal = None
    sf_agg = None
    
    total_steps = sum([run_mode_b, run_v7, run_semi_markov, run_signals])
    current_step = 0
    
    # === Run Mode B (Ground Truth) ===
    if run_mode_b:
        status_text.write("Running Mode B: Stationary Bootstrap (Ground Truth)...")
        mode_b_engine, mode_b_results = run_mode_b_bootstrap(
            ticker, market_ticker, days_ahead, mode_b_config, log
        )
        if mode_b_engine and mode_b_results:
            all_results["modules"]["mode_b"] = {
                "price": mode_b_engine.last_price,
                "beta": mode_b_engine.beta,
                "alpha_ann": mode_b_engine.alpha * 252,
                "realized_vol": mode_b_engine.realized_vol,
                "all_gates_passed": mode_b_results['all_gates_passed'],
                "first_passage": {
                    "prob_up_first": mode_b_results['first_passage']['prob_up_first'],
                    "prob_down_first": mode_b_results['first_passage']['prob_down_first'],
                    "prob_neither": mode_b_results['first_passage']['prob_neither'],
                },
                "risk": {
                    "var_95": mode_b_results['risk']['var_95'],
                    "cvar_95": mode_b_results['risk']['cvar_95'],
                    "kelly_fraction": mode_b_results['risk']['kelly_fraction'],
                },
                "historical_validation": mode_b_results['historical_validation'],
            }
        current_step += 1
        progress_bar.progress(current_step / total_steps if total_steps > 0 else 1.0)
    
    # === Run V7.0 ===
    if run_v7:
        status_text.write("Running v7.0 Regime Risk Platform...")
        v7_engine, v7_results, v7_signal = run_regime_engine_v7(
            ticker, market_ticker, days_ahead, simulations, n_regimes, enable_vix, run_full_calibration, log
        )
        if v7_engine:
            all_results["modules"]["v7_regime"] = {
                "price": v7_engine.last_price,
                "regime": v7_engine.regime_names.get(v7_engine.current_regime, "Unknown"),
                "beta": v7_engine.market_beta,
                "signal": v7_signal['signal'],
                "confidence": v7_signal['confidence'],
                "reasoning": v7_signal['reasoning'],
                "prob_up": v7_results['prob_up'],
                "prob_down": v7_results['prob_down'],
                "risk": {
                    "var_95": v7_results['risk']['var_95'],
                    "cvar_95": v7_results['risk']['cvar_95'],
                    "kelly_fraction": v7_results['risk']['kelly_fraction']
                }
            }
        current_step += 1
        progress_bar.progress(current_step / total_steps)
    
    # === Run Semi-Markov ===
    if run_semi_markov:
        status_text.write("Running Semi-Markov Model...")
        sm_model, sm_paths, sm_validation, sm_state_info = run_semi_markov_model(ticker, log)
        if sm_model:
            all_results["modules"]["semi_markov"] = {
                "state_map": sm_model.state_map,
                "current_state": sm_state_info['state_name'],
                "days_in_state": sm_state_info['days_in_state'],
                "fatigue": sm_state_info['fatigue'],
                "position_size": sm_state_info['position_size'],
                "validation": {
                    "real_vol_acf_lag1": float(sm_validation['real_vol_acf_lag1']),
                    "sim_vol_acf_lag1": float(sm_validation['sim_vol_acf_lag1']),
                    "vol_clustering_error": float(sm_validation['vol_clustering_error'])
                },
                "duration_params": {
                    str(k): {"dist": v['dist']} 
                    for k, v in sm_model.duration_params.items()
                }
            }
        current_step += 1
        progress_bar.progress(current_step / total_steps)
    
    # === Run Signals Factory ===
    if run_signals:
        status_text.write("Running Signals Factory...")
        sf_signal, sf_agg = run_signals_factory(ticker, log)
        if sf_signal:
            all_results["modules"]["signals_factory"] = {
                "vol_compression": {
                    "score": sf_signal.confidence,
                    "sizing_mult": sf_signal.position_sizing_mult,
                    "direction": sf_signal.direction,
                    "reasoning": sf_signal.reasoning,
                    "rv_percentile": sf_signal.rv_percentile,
                    "range_percentile": sf_signal.range_percentile,
                    "vix_percentile": sf_signal.vix_percentile
                },
                "aggregated": {
                    "final_score": sf_agg['final_score'],
                    "final_sizing": sf_agg['final_sizing']
                }
            }
        current_step += 1
        progress_bar.progress(current_step / total_steps)
    
    log.finish()
    progress_bar.progress(1.0)
    status_text.write("✅ All modules complete!")
    
    # Store log in results
    all_results["execution_log"] = log.get_log_text()
    
    # ==========================================================================
    # DISPLAY RESULTS
    # ==========================================================================
    
    st.title(f"🦅 {ticker} - Consolidated Analysis")
    
    # Header metrics - prioritize Mode B as ground truth
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if mode_b_engine:
            st.metric("Price", f"${mode_b_engine.last_price:.2f}")
        elif v7_engine:
            st.metric("Price", f"${v7_engine.last_price:.2f}")
        else:
            st.metric("Price", "N/A")
    
    with col2:
        if mode_b_results:
            status = "✅ PASS" if mode_b_results['all_gates_passed'] else "❌ FAIL"
            st.metric("Mode B Gates", status)
        elif v7_engine:
            regime_name = v7_engine.regime_names.get(v7_engine.current_regime, "Unknown")
            st.metric("V7 Regime", regime_name)
        else:
            st.metric("Mode B Gates", "N/A")
    
    with col3:
        if mode_b_results:
            fpt = mode_b_results['first_passage']
            if fpt['prob_up_first'] > fpt['prob_down_first']:
                edge = f"↑ {fpt['prob_up_first']:.0%}"
            else:
                edge = f"↓ {fpt['prob_down_first']:.0%}"
            st.metric("First-Passage Edge", edge)
        elif sm_state_info:
            st.metric("Semi-Markov State", sm_state_info['state_name'])
        else:
            st.metric("Edge", "N/A")
    
    with col4:
        if mode_b_results:
            kelly = mode_b_results['risk']['kelly_fraction']
            st.metric("Kelly", f"{kelly:.0%}")
        elif v7_signal:
            st.metric("Signal", f"{v7_signal['signal']} ({v7_signal['confidence']}%)")
        else:
            st.metric("Kelly", "N/A")
    
    with col5:
        if mode_b_engine:
            st.metric("Beta", f"{mode_b_engine.beta:.2f}")
        elif sf_signal:
            st.metric("Vol Compression", f"{sf_signal.confidence:.0f}/100")
        else:
            st.metric("Beta", "N/A")
    
    st.divider()
    
    # ==========================================================================
    # TABS
    # ==========================================================================
    tab_mode_b, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Mode B (Ground Truth)",
        "📊 Mode A (V7.0)", 
        "🧬 Semi-Markov", 
        "📡 Signals", 
        "📋 Terminal Log", 
        "💾 Export JSON"
    ])
    
    # --- Tab Mode B: Ground Truth Benchmark ---
    with tab_mode_b:
        if mode_b_results and mode_b_engine:
            st.subheader("🎯 Mode B: Stationary Bootstrap (Ground Truth)")
            st.caption("Block samples historical data with geometric random block lengths. No model assumptions.")
            
            # Gate status banner
            if mode_b_results['all_gates_passed']:
                st.success("✅ ALL GATEKEEPER METRICS PASSED - Simulation is trustworthy")
            else:
                st.error("❌ SOME GATEKEEPER METRICS FAILED - Review diagnostics below")
            
            # Main metrics row
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("Beta", f"{mode_b_engine.beta:.2f}")
                st.metric("Alpha (ann)", f"{mode_b_engine.alpha * 252:+.1%}")
            with col_b:
                st.metric("Realized Vol", f"{mode_b_engine.realized_vol:.1%}")
                st.metric("Target Up", f"${mode_b_engine.target_up:.0f}")
            with col_c:
                st.metric("VaR(95)", f"{mode_b_results['risk']['var_95']*100:+.1f}%")
                st.metric("CVaR(95)", f"{mode_b_results['risk']['cvar_95']*100:+.1f}%")
            with col_d:
                st.metric("Win Rate", f"{mode_b_results['risk']['win_rate']:.1%}")
                st.metric("Kelly", f"{mode_b_results['risk']['kelly_fraction']:.0%}")
            
            st.divider()
            
            # Diagnostics Table
            st.markdown("### 🔬 Diagnostic Metrics")
            
            diag_data = []
            for d in mode_b_results['diagnostics']:
                status = "✅ PASS" if d.passed else "❌ FAIL"
                gate = "🚨 GATEKEEPER" if d.is_gatekeeper else "ℹ️ Info"
                diag_data.append({
                    "Metric": d.metric,
                    "Simulated": f"{d.sim_value:.4f}",
                    "Historical": f"{d.hist_value:.4f}",
                    "Threshold": f"±{d.threshold:.4f}",
                    "Status": status,
                    "Type": gate,
                })
            st.dataframe(pd.DataFrame(diag_data), use_container_width=True)
            
            st.divider()
            
            # First-Passage Analysis
            st.markdown("### 🎲 First-Passage Analysis")
            fpt = mode_b_results['first_passage']
            hist_val = mode_b_results['historical_validation']
            
            col_fp1, col_fp2, col_fp3 = st.columns(3)
            with col_fp1:
                st.markdown("**Simulated Probabilities**")
                st.metric("P(Up First)", f"{fpt['prob_up_first']:.1%}")
                st.metric("P(Down First)", f"{fpt['prob_down_first']:.1%}")
                st.metric("P(Neither)", f"{fpt['prob_neither']:.1%}")
            with col_fp2:
                st.markdown("**Hitting Times (Days)**")
                mean_up = fpt['mean_tau_up'] if not np.isnan(fpt['mean_tau_up']) else "N/A"
                mean_down = fpt['mean_tau_down'] if not np.isnan(fpt['mean_tau_down']) else "N/A"
                st.metric("Mean Time to Up", f"{mean_up:.1f}" if isinstance(mean_up, float) else mean_up)
                st.metric("Mean Time to Down", f"{mean_down:.1f}" if isinstance(mean_down, float) else mean_down)
            with col_fp3:
                st.markdown("**Historical Validation**")
                st.metric("Hist Up Hit Rate", f"{hist_val['hist_up_hit']:.1%}")
                st.metric("Hist Down Hit Rate", f"{hist_val['hist_down_hit']:.1%}")
                st.metric("Windows Used", f"{hist_val['n_windows']}")
            
            st.divider()
            
            # Charts
            st.markdown("### 📊 Simulation Visualization")
            
            paths = mode_b_results['paths']
            
            fig = plt.figure(figsize=(18, 10))
            gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])
            
            # Cone Chart
            ax1 = fig.add_subplot(gs[0, :])
            x = np.arange(paths.shape[1])
            
            p5 = np.percentile(paths, 5, axis=0)
            p25 = np.percentile(paths, 25, axis=0)
            p50 = np.median(paths, axis=0)
            p75 = np.percentile(paths, 75, axis=0)
            p95 = np.percentile(paths, 95, axis=0)
            
            ax1.fill_between(x, p5, p95, color='cyan', alpha=0.1, label='90%')
            ax1.fill_between(x, p25, p75, color='cyan', alpha=0.2, label='50%')
            ax1.plot(x, p50, color='cyan', lw=2.5, label='Median')
            
            ax1.axhline(mode_b_engine.target_up, color='lime', ls=':', lw=2, label=f'Up ${mode_b_engine.target_up:.0f}')
            ax1.axhline(mode_b_engine.target_down, color='red', ls=':', lw=2, label=f'Down ${mode_b_engine.target_down:.0f}')
            ax1.axhline(mode_b_engine.last_price, color='white', lw=1, alpha=0.5)
            
            ax1.set_title(f"{ticker} MODE B | Stationary Bootstrap | Beta={mode_b_engine.beta:.2f}",
                          fontsize=14, fontweight='bold')
            ax1.legend(loc='upper left')
            ax1.grid(alpha=0.2)
            ax1.set_xlabel("Days")
            ax1.set_ylabel("Price")
            
            # Drawdown Distribution
            ax2 = fig.add_subplot(gs[1, 0])
            risk = mode_b_results['risk']
            sns.histplot(risk['max_drawdowns'] * 100, color='red', ax=ax2, bins=50, alpha=0.7)
            ax2.axvline(-20, color='orange', ls='--', label='-20%')
            ax2.axvline(-30, color='red', ls='--', label='-30%')
            ax2.set_title("Max Drawdown Distribution")
            ax2.set_xlabel("Drawdown %")
            ax2.legend()
            
            # Final Price Distribution
            ax3 = fig.add_subplot(gs[1, 1])
            final = paths[:, -1]
            sns.histplot(final, color='cyan', ax=ax3, bins=50, alpha=0.7)
            ax3.axvline(mode_b_engine.last_price, color='white', ls='-', label='Current')
            ax3.axvline(np.median(final), color='yellow', ls='--', label='Median')
            ax3.set_title("Final Price Distribution")
            ax3.legend()
            
            # ACF Comparison (Sim vs Historical)
            ax4 = fig.add_subplot(gs[1, 2])
            
            # Compute ACF for both
            sim_returns = np.diff(np.log(paths), axis=1).flatten()
            hist_returns = mode_b_engine.data['Log_Ret'].values
            
            def compute_acf(returns, max_lag=20):
                r2 = returns ** 2
                n = len(r2)
                mean_r2 = np.mean(r2)
                var_r2 = np.var(r2)
                acf = []
                for lag in range(1, max_lag + 1):
                    if n > lag and var_r2 > 0:
                        cov = np.mean((r2[lag:] - mean_r2) * (r2[:-lag] - mean_r2))
                        acf.append(cov / var_r2)
                    else:
                        acf.append(0)
                return acf
            
            lags = list(range(1, 21))
            hist_acf = compute_acf(hist_returns, 20)
            sim_acf = compute_acf(sim_returns, 20)
            
            ax4.bar(np.array(lags) - 0.2, hist_acf, width=0.4, color='orange', alpha=0.7, label='Historical')
            ax4.bar(np.array(lags) + 0.2, sim_acf, width=0.4, color='cyan', alpha=0.7, label='Simulated')
            ax4.set_title("ACF(r²) Comparison - Vol Clustering")
            ax4.set_xlabel("Lag")
            ax4.set_ylabel("ACF")
            ax4.legend()
            ax4.grid(alpha=0.2)
            
            st.pyplot(fig)
            
        else:
            st.info("Enable and run Mode B (Stationary Bootstrap) to see the ground truth benchmark.")
    
    # --- Tab 1: Visuals (V7.0 charts) ---
    with tab1:
        if v7_results and v7_engine:
            fig = plt.figure(figsize=(18, 10))
            gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])
            
            paths = v7_results['paths']
            
            # Cone Chart
            ax1 = fig.add_subplot(gs[0, :])
            x = np.arange(paths.shape[1])
            
            p5 = np.percentile(paths, 5, axis=0)
            p25 = np.percentile(paths, 25, axis=0)
            p50 = np.median(paths, axis=0)
            p75 = np.percentile(paths, 75, axis=0)
            p95 = np.percentile(paths, 95, axis=0)
            
            ax1.fill_between(x, p5, p95, color='cyan', alpha=0.1, label='90%')
            ax1.fill_between(x, p25, p75, color='cyan', alpha=0.2, label='50%')
            ax1.plot(x, p50, color='cyan', lw=2.5, label='Median')
            
            ax1.axhline(v7_engine.target_up, color='lime', ls=':', lw=2, label=f'Up ${v7_engine.target_up:.0f}')
            ax1.axhline(v7_engine.target_down, color='red', ls=':', lw=2, label=f'Down ${v7_engine.target_down:.0f}')
            ax1.axhline(v7_engine.last_price, color='white', lw=1, alpha=0.5)
            
            ax1.set_title(f"{ticker} v7.0 | {v7_signal['signal']} | {regime_name} | Beta={v7_engine.market_beta:.1f}",
                          fontsize=14, fontweight='bold')
            ax1.legend(loc='upper left')
            ax1.grid(alpha=0.2)
            ax1.set_xlabel("Days")
            ax1.set_ylabel("Price")
            
            # Drawdown Distribution
            ax2 = fig.add_subplot(gs[1, 0])
            risk = v7_results['risk']
            sns.histplot(risk['max_drawdowns'] * 100, color='red', ax=ax2, bins=50, alpha=0.7)
            ax2.axvline(-20, color='orange', ls='--', label='-20%')
            ax2.axvline(-30, color='red', ls='--', label='-30%')
            ax2.set_title("Max Drawdown Distribution")
            ax2.set_xlabel("Drawdown %")
            ax2.legend()
            
            # Final Price Distribution
            ax3 = fig.add_subplot(gs[1, 1])
            final = paths[:, -1]
            sns.histplot(final, color='cyan', ax=ax3, bins=50, alpha=0.7)
            ax3.axvline(v7_engine.last_price, color='white', ls='-')
            ax3.axvline(np.median(final), color='yellow', ls='--')
            ax3.set_title("Final Price Distribution")
            
            # Transition Matrix
            ax4 = fig.add_subplot(gs[1, 2])
            if v7_engine.transition_matrix is not None:
                labels = [v7_engine.regime_names.get(i, f"R{i}")[:6] for i in range(len(v7_engine.transition_matrix))]
                sns.heatmap(v7_engine.transition_matrix, annot=True, fmt='.2f', cmap='magma',
                            xticklabels=labels, yticklabels=labels, ax=ax4)
                ax4.set_title("Transition Matrix")
            
            st.pyplot(fig)
        else:
            st.info("Enable and run v7.0 Regime Engine to see visualizations.")
    
    # --- Tab 2: Semi-Markov Details ---
    with tab2:
        if sm_model and sm_paths is not None:
            st.subheader("🎲 Semi-Markov Monte Carlo")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("### Current State")
                st.metric("State", sm_state_info['state_name'])
                st.metric("Days in State", sm_state_info['days_in_state'])
                st.metric("Regime Fatigue", f"{sm_state_info['fatigue']:.1%}")
                st.metric("Recommended Position", f"{sm_state_info['position_size']:.1%}")
            
            with col_b:
                st.markdown("### Validation")
                st.metric("Real Vol ACF(1)", f"{sm_validation['real_vol_acf_lag1']:.3f}")
                st.metric("Sim Vol ACF(1)", f"{sm_validation['sim_vol_acf_lag1']:.3f}")
                st.metric("Clustering Error", f"{sm_validation['vol_clustering_error']:.3f}")
            
            st.divider()
            
            # Semi-Markov paths plot
            fig_sm, ax_sm = plt.subplots(figsize=(12, 5))
            for i in range(min(100, sm_paths.shape[0])):
                ax_sm.plot(sm_paths[i], alpha=0.1, color='cyan')
            ax_sm.set_title("Semi-Markov Monte Carlo Paths")
            ax_sm.set_xlabel("Days")
            ax_sm.set_ylabel("Price")
            ax_sm.grid(alpha=0.2)
            st.pyplot(fig_sm)
            
            # Duration parameters
            st.markdown("### Duration Distributions")
            dur_data = []
            for state_idx, state_name in sm_model.state_map.items():
                params = sm_model.duration_params.get(state_idx, {})
                dur_data.append({
                    "State": f"{state_idx}: {state_name}",
                    "Distribution": params.get('dist', 'unknown'),
                    "Samples": len(sm_model.duration_data.get(state_idx, []))
                })
            st.dataframe(pd.DataFrame(dur_data), use_container_width=True)
            
        else:
            st.info("Enable and run Semi-Markov Model to see details.")
    
    # --- Tab 3: Signals Factory ---
    with tab3:
        if sf_signal and sf_agg:
            st.subheader("📡 Vol Compression Signal")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Compression Score", f"{sf_signal.confidence:.1f}/100")
            with col_b:
                st.metric("Position Sizing Mult", f"{sf_signal.position_sizing_mult:.2f}")
            with col_c:
                direction_str = {-1: "SHORT", 0: "NEUTRAL", 1: "LONG"}.get(sf_signal.direction, "?")
                st.metric("Direction", direction_str)
            
            st.divider()
            
            st.markdown("### Percentile Breakdown")
            perc_data = {
                "Metric": ["RV (20d)", "Range (Smoothed)", "VIX"],
                "Percentile": [
                    f"{sf_signal.rv_percentile:.1%}",
                    f"{sf_signal.range_percentile:.1%}",
                    f"{sf_signal.vix_percentile:.1%}"
                ],
                "Compression Contribution": [
                    f"{(1 - sf_signal.rv_percentile) * 100:.1f}",
                    f"{(1 - sf_signal.range_percentile) * 100:.1f}",
                    f"{(1 - sf_signal.vix_percentile) * 100:.1f}"
                ]
            }
            st.dataframe(pd.DataFrame(perc_data), use_container_width=True)
            
            st.markdown("### Reasoning")
            for reason in sf_signal.reasoning:
                st.write(f"• {reason}")
            
            st.divider()
            
            st.markdown("### Aggregated Signal")
            st.json({
                "final_score": sf_agg['final_score'],
                "final_sizing": sf_agg['final_sizing'],
                "signals": sf_agg['signals']
            })
        else:
            st.info("Enable and run Signals Factory to see details.")
    
    # --- Tab 4: Terminal Log ---
    with tab4:
        st.subheader("📋 Full Execution Log")
        st.caption("This is the complete terminal output from all modules.")
        
        # Display as monospace code block
        st.code(log.get_log_text(), language="text")
        
        # Download button for log
        st.download_button(
            label="📥 Download Log as TXT",
            data=log.get_log_text(),
            file_name=f"{ticker}_execution_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    # --- Tab 5: Export JSON ---
    with tab5:
        st.subheader("💾 Export Complete Results to JSON")
        st.caption("All analysis results consolidated into a single JSON file.")
        
        # Pretty print JSON
        report_json = json.dumps(all_results, indent=2, default=str)
        
        st.code(report_json, language="json")
        
        st.download_button(
            label="📥 Download Full Report as JSON",
            data=report_json,
            file_name=f"{ticker}_consolidated_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        st.divider()
        
        # Also offer chart download if available
        if v7_results:
            st.subheader("📥 Download Chart")
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0e1117')
            buf.seek(0)
            st.download_button(
                label="Download Chart as PNG",
                data=buf,
                file_name=f"{ticker}_chart_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
                mime="image/png"
            )

else:
    # Landing page
    st.title("🦅 Project PLTR - Consolidated Dashboard")
    st.markdown("""
    ### Welcome to the Unified Analysis Platform
    
    This dashboard consolidates **three powerful analysis engines** into a single interface:
    
    | Module | Description |
    |--------|-------------|
    | **v7.0 Regime Engine** | GMM-based regime clustering with Alpha/Beta factor model |
    | **Semi-Markov Model** | Duration-aware regime simulation with fatigue scoring |
    | **Signals Factory** | Context-first signals (Vol Compression, etc.) |
    
    ---
    
    ### 🚀 Quick Start
    
    1. Enter a **ticker symbol** in the sidebar (default: SPY)
    2. Select which **modules** to run
    3. Click **RUN ALL SELECTED**
    4. View results in the tabs:
       - **📊 Visuals** - Monte Carlo cone charts
       - **🧬 Semi-Markov** - Duration distributions & fatigue
       - **📡 Signals** - Vol compression analysis
       - **📋 Terminal Log** - Full execution output
       - **💾 Export JSON** - Download complete report
    
    ---
    
    ### 📦 Requirements
    
    This dashboard requires the following modules:
    - `refinery.semi_markov` — Semi-Markov chain implementation  
    - `signals_factory` — Signal generators  
    - `regime_engine_v7` — v7.0 GMM engine  
    
    Make sure all dependencies are installed:
    ```bash
    pip install -r requirements.txt
    ```
    """)
