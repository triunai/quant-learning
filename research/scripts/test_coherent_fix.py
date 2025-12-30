"""Test script to verify the coherent factor model fix."""
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
with open('test_results.txt', 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("COHERENT FACTOR MODEL VALIDATION\n")
    f.write("="*70 + "\n\n")
    
    for r in range(platform.n_regimes):
        name = platform.regime_names.get(r, f"R{r}")
        mask = platform.data['Regime'] == r
        asset_ret = platform.data.loc[mask, 'Log_Ret'].values
        market_ret = platform.market_data['Log_Ret'].reindex(platform.data[mask].index).values
        
        beta = platform.regime_beta.get(r, 1.0)
        alpha = platform.regime_alpha.get(r, 0)
        resid = platform.regime_residuals.get(r, np.array([0]))
        model_type = platform.regime_model_type.get(r, "unknown")
        
        f.write(f"\n[{name}] (model: {model_type})\n")
        f.write(f"  Days: {len(asset_ret)}\n")
        f.write(f"  Beta: {beta:.2f}\n")
        f.write(f"  Alpha (ann): {alpha*252*100:+.1f}%\n")
        f.write(f"  Residual mean (ann): {np.mean(resid)*252*100:.4f}%\n")
        
        # Validate decomposition
        actual_drift = np.mean(asset_ret) * 252 * 100
        market_contrib = beta * np.mean(market_ret) * 252 * 100
        model_drift = alpha * 252 * 100 + market_contrib
        
        f.write(f"\n  DRIFT DECOMPOSITION:\n")
        f.write(f"    Alpha contribution: {alpha*252*100:+.1f}%\n")
        f.write(f"    Beta*Market contrib: {market_contrib:+.1f}%\n")
        f.write(f"    Model total: {model_drift:+.1f}%\n")
        f.write(f"    Actual drift: {actual_drift:+.1f}%\n")
        f.write(f"    ERROR: {abs(model_drift - actual_drift):.2f}%\n")
    
    # What-if momentum test
    f.write("\n" + "="*70 + "\n")
    f.write("WHAT-IF MOMENTUM TEST\n")
    f.write("="*70 + "\n")
    
    momentum_idx = None
    bear_idx = None
    for r, name in platform.regime_names.items():
        if 'Momentum' in str(name):
            momentum_idx = r
        if 'Bear' in str(name):
            bear_idx = r
    
    if momentum_idx is not None and bear_idx is not None:
        # Test Bear regime
        platform.current_regime = bear_idx
        platform.simulations = 1000
        paths_bear = platform.simulate()
        prob_bear = np.mean(np.any(paths_bear >= platform.target_up, axis=1))
        
        # Test Momentum regime
        platform.current_regime = momentum_idx
        paths_momentum = platform.simulate()
        prob_momentum = np.mean(np.any(paths_momentum >= platform.target_up, axis=1))
        
        f.write(f"\n  Bear regime probability of ${platform.target_up:.0f}: {prob_bear:.1%}\n")
        f.write(f"  Momentum regime probability of ${platform.target_up:.0f}: {prob_momentum:.1%}\n")
        
        if prob_momentum > prob_bear:
            f.write(f"\n  [SUCCESS] Momentum ({prob_momentum:.1%}) > Bear ({prob_bear:.1%})\n")
        else:
            f.write(f"\n  [FAIL] Momentum ({prob_momentum:.1%}) <= Bear ({prob_bear:.1%})\n")
            f.write("  BUG STILL PRESENT!\n")

print("Results written to test_results.txt")
