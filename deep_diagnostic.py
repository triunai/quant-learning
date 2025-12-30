"""Deep diagnostic to understand why Momentum 59.6% is lower than expected."""
import numpy as np
import pandas as pd
import yfinance as yf
import sys
import io

# Load the module (suppress output)
old_stdout = sys.stdout
sys.stdout = io.StringIO()

with open('battle-tested/PLTR-test-2.py', encoding='utf-8') as f:
    code = f.read().split('if __name__')[0]
exec(code)

platform = RegimeRiskPlatform(ticker='PLTR', market_ticker='QQQ')
platform.ingest_data()
platform.build_regime_model()
platform.compute_market_beta()

sys.stdout = old_stdout

# Write results to file
with open('deep_diagnostic.txt', 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("DEEP DIAGNOSTIC: WHY IS MOMENTUM PROBABILITY LOW?\n")
    f.write("="*70 + "\n\n")
    
    # Find momentum regime
    momentum_idx = None
    for r, name in platform.regime_names.items():
        if 'Momentum' in str(name):
            momentum_idx = r
            break
    
    if momentum_idx is None:
        f.write("No Momentum regime found!\n")
    else:
        mask = platform.data['Regime'] == momentum_idx
        dates = platform.data.index[mask]
        asset_ret = platform.data.loc[mask, 'Log_Ret'].values
        market_ret = platform.market_data['Log_Ret'].reindex(dates).values
        
        f.write(f"[1] MOMENTUM REGIME DATA\n")
        f.write(f"    Total days: {len(asset_ret)}\n")
        f.write(f"    Date range: {dates.min()} to {dates.max()}\n")
        f.write(f"    Periods: {len(dates)}\n\n")
        
        f.write(f"[2] ACTUAL RETURNS IN MOMENTUM REGIME\n")
        f.write(f"    PLTR mean daily: {np.mean(asset_ret)*100:.3f}%\n")
        f.write(f"    PLTR mean annual: {np.mean(asset_ret)*252*100:.1f}%\n")
        f.write(f"    QQQ mean daily: {np.mean(market_ret)*100:.3f}%\n")
        f.write(f"    QQQ mean annual: {np.mean(market_ret)*252*100:.1f}%\n\n")
        
        # The key question: what implied QQQ return?
        beta = platform.regime_beta[momentum_idx]
        alpha = platform.regime_alpha[momentum_idx]
        beta_contrib = beta * np.mean(market_ret) * 252
        
        f.write(f"[3] FACTOR DECOMPOSITION CHECK\n")
        f.write(f"    Beta: {beta:.2f}\n")
        f.write(f"    Alpha (ann): {alpha*252*100:+.1f}%\n")
        f.write(f"    Actual QQQ annual in this regime: {np.mean(market_ret)*252*100:.1f}%\n")
        f.write(f"    Beta * Actual QQQ = {beta_contrib*100:.1f}%\n")
        f.write(f"    Alpha + Beta*Market = {(alpha*252 + beta_contrib)*100:.1f}%\n")
        f.write(f"    Actual PLTR (should match): {np.mean(asset_ret)*252*100:.1f}%\n\n")
        
        # R-squared analysis
        residuals = platform.regime_residuals[momentum_idx]
        r_squared = 1 - np.var(residuals) / np.var(asset_ret)
        
        f.write(f"[4] R-SQUARED ANALYSIS\n")
        f.write(f"    R-squared: {r_squared:.2f}\n")
        f.write(f"    Factor model explains: {r_squared*100:.0f}% of variance\n")
        f.write(f"    Unexplained (residuals): {(1-r_squared)*100:.0f}%\n")
        f.write(f"    Residual std (ann): {np.std(residuals)*np.sqrt(252)*100:.1f}%\n\n")
        
        # Simplified simulation comparison
        f.write(f"[5] SIMPLIFIED vs FULL SIMULATION\n")
        
        # Simplified: just use drift formula
        np.random.seed(42)
        n_sim = 10000
        horizon = 126
        dt = 1/252
        
        alpha_daily = alpha
        sigma_daily = np.std(asset_ret)
        market_daily = np.mean(market_ret)
        
        # Analytical expected growth
        daily_drift = alpha + beta * market_daily
        annual_drift = daily_drift * 252
        expected_log_return = daily_drift * horizon
        expected_growth = np.exp(expected_log_return)
        
        f.write(f"    Daily drift: {daily_drift*100:.4f}%\n")
        f.write(f"    Annual drift: {annual_drift*100:.1f}%\n")
        f.write(f"    Expected 126-day growth: {expected_growth:.2f}x\n")
        f.write(f"    Expected price: ${184 * expected_growth:.0f}\n\n")
        
        # GBM simulation (no regime switching, no residual sampling)
        paths_simple = np.zeros((n_sim, horizon+1))
        paths_simple[:, 0] = 184
        
        for t in range(1, horizon+1):
            drift = alpha + beta * market_daily
            shock = np.random.randn(n_sim) * sigma_daily
            ret = drift + shock
            paths_simple[:, t] = paths_simple[:, t-1] * np.exp(ret)
        
        prob_simple = np.mean(np.any(paths_simple >= 276, axis=1))
        median_simple = np.median(paths_simple[:, -1])
        
        f.write(f"    SIMPLIFIED (GBM with fixed drift):\n")
        f.write(f"      P($276): {prob_simple:.1%}\n")
        f.write(f"      Median final price: ${median_simple:.0f}\n\n")
        
        # Full simulation from platform
        platform.current_regime = momentum_idx
        platform.simulations = 10000
        paths_full = platform.simulate()
        
        prob_full = np.mean(np.any(paths_full >= 276, axis=1))
        median_full = np.median(paths_full[:, -1])
        
        f.write(f"    FULL PLATFORM SIMULATION:\n")
        f.write(f"      P($276): {prob_full:.1%}\n")
        f.write(f"      Median final price: ${median_full:.0f}\n\n")
        
        f.write(f"[6] WHAT'S DIFFERENT?\n")
        f.write(f"    Simple prob: {prob_simple:.1%}\n")
        f.write(f"    Full prob: {prob_full:.1%}\n")
        f.write(f"    Difference: {(prob_simple - prob_full)*100:.1f} pp\n\n")
        
        if prob_simple > prob_full + 0.05:
            f.write(f"    [!] Full simulation is LOWER than simplified!\n")
            f.write(f"    Possible causes:\n")
            f.write(f"      1. Regime transitions (leaving Momentum into Bear/Crisis)\n")
            f.write(f"      2. Residual sampling vs normal shocks\n")
            f.write(f"      3. Market return sampling (historical vs constant)\n")
        
        # Check regime transitions in simulation
        f.write(f"\n[7] SEMI-MARKOV DURATION IMPACT\n")
        momentum_duration = platform.regime_duration[momentum_idx]
        f.write(f"    Momentum avg duration: {momentum_duration['mean']:.1f} days\n")
        f.write(f"    Momentum median duration: {np.median(momentum_duration['samples']):.1f} days\n")
        f.write(f"    Horizon: {horizon} days\n")
        expected_changes = horizon / momentum_duration['mean']
        f.write(f"    Expected regime changes in horizon: {expected_changes:.1f}\n")
        
        if expected_changes > 1:
            f.write(f"    [!] Multiple regime changes expected -> will leave Momentum\n")
            f.write(f"    This explains lower probability!\n")

print("Results written to deep_diagnostic.txt")
