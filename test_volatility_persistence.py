"""Validate hypothesis: Regime persistence inversely correlates with volatility."""
import numpy as np
import pandas as pd
import yfinance as yf
import sys
import io
import matplotlib.pyplot as plt

# Load the module (suppress output)
old_stdout = sys.stdout
sys.stdout = io.StringIO()

with open('battle-tested/PLTR-test-2.py', encoding='utf-8') as f:
    code = f.read().split('if __name__')[0]
exec(code)

sys.stdout = old_stdout

# Diverse set of stocks across sectors and types
stocks = [
    # High Growth / Speculative
    ('PLTR', 'QQQ'),
    ('TSLA', 'QQQ'),
    ('NVDA', 'QQQ'),
    ('AMD', 'QQQ'),
    ('COIN', 'QQQ'),
    
    # Mega-Cap Tech
    ('MSFT', 'QQQ'),
    ('AAPL', 'QQQ'),
    ('GOOGL', 'QQQ'),
    ('META', 'QQQ'),
    ('AMZN', 'QQQ'),
    
    # Defensive / Value
    ('WMT', 'SPY'),
    ('JNJ', 'SPY'),
    ('PG', 'SPY'),
    ('KO', 'SPY'),
    ('PEP', 'SPY'),
    
    # Financials
    ('JPM', 'SPY'),
    ('BAC', 'SPY'),
    ('GS', 'SPY'),
    
    # Energy
    ('XOM', 'SPY'),
    ('CVX', 'SPY'),
    
    # Indices
    ('QQQ', 'SPY'),
    ('SPY', 'QQQ'),
]

results = []

print("Testing volatility-persistence relationship...")
print("="*60)

for ticker, market in stocks:
    try:
        # Suppress platform output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        platform = RegimeRiskPlatform(
            ticker=ticker,
            market_ticker=market,
            n_regimes=3,
            days_ahead=126,
            simulations=100  # Minimal for speed
        )
        platform.ingest_data()
        platform.build_regime_model()
        platform.compute_market_beta()
        
        sys.stdout = old_stdout
        
        # Calculate average regime duration
        all_durations = []
        for r in range(platform.n_regimes):
            samples = platform.regime_duration.get(r, {}).get('samples', [])
            if samples:
                all_durations.extend(samples)
        
        avg_duration = np.mean(all_durations) if all_durations else 0
        median_duration = np.median(all_durations) if all_durations else 0
        
        results.append({
            'ticker': ticker,
            'volatility': platform.realized_vol,
            'beta': platform.market_beta,
            'avg_duration': avg_duration,
            'median_duration': median_duration,
            'n_samples': len(platform.data)
        })
        
        print(f"  {ticker:6} | Vol: {platform.realized_vol:.1%} | Avg Duration: {avg_duration:.1f}d | Median: {median_duration:.1f}d")
        
    except Exception as e:
        sys.stdout = old_stdout
        print(f"  {ticker:6} | ERROR: {str(e)[:40]}")

print("\n" + "="*60)

# Calculate correlation
if len(results) > 5:
    volatilities = [r['volatility'] for r in results]
    avg_durations = [r['avg_duration'] for r in results]
    median_durations = [r['median_duration'] for r in results]
    
    corr_avg = np.corrcoef(volatilities, avg_durations)[0, 1]
    corr_median = np.corrcoef(volatilities, median_durations)[0, 1]
    
    print(f"\nCORRELATION RESULTS:")
    print(f"  Volatility vs Avg Duration:    r = {corr_avg:.3f}")
    print(f"  Volatility vs Median Duration: r = {corr_median:.3f}")
    
    if corr_avg < -0.3:
        print(f"\n  [CONFIRMED] Negative correlation supports hypothesis:")
        print(f"  Higher volatility -> Shorter regime persistence")
    elif corr_avg > 0.3:
        print(f"\n  [REJECTED] Positive correlation contradicts hypothesis")
    else:
        print(f"\n  [INCONCLUSIVE] Weak correlation ({corr_avg:.2f})")
    
    # Save plot
    plt.figure(figsize=(12, 8))
    
    # Main scatter
    plt.scatter([r['volatility']*100 for r in results], 
                [r['avg_duration'] for r in results],
                s=100, alpha=0.7)
    
    # Annotate each point
    for r in results:
        plt.annotate(r['ticker'], 
                    (r['volatility']*100, r['avg_duration']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9)
    
    # Trendline
    z = np.polyfit(volatilities, avg_durations, 1)
    p = np.poly1d(z)
    vol_range = np.linspace(min(volatilities), max(volatilities), 100)
    plt.plot(vol_range*100, p(vol_range), "r--", alpha=0.7, 
             label=f'Trend (r={corr_avg:.2f})')
    
    plt.xlabel('Annualized Volatility (%)', fontsize=12)
    plt.ylabel('Average Regime Duration (days)', fontsize=12)
    plt.title('Regime Persistence vs Stock Volatility\n(Hypothesis: Higher Vol → Shorter Regimes)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('volatility_vs_persistence.png', dpi=150, facecolor='white')
    print(f"\n  Plot saved to: volatility_vs_persistence.png")

# Write detailed results
with open('volatility_persistence_results.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("VOLATILITY VS REGIME PERSISTENCE ANALYSIS\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"{'Ticker':<8} {'Vol':<8} {'Beta':<6} {'Avg Dur':<10} {'Med Dur':<10} {'N Days':<8}\n")
    f.write("-"*70 + "\n")
    
    for r in sorted(results, key=lambda x: x['volatility'], reverse=True):
        f.write(f"{r['ticker']:<8} {r['volatility']:.1%}   {r['beta']:.2f}   {r['avg_duration']:>6.1f}d    {r['median_duration']:>6.1f}d    {r['n_samples']:<8}\n")
    
    f.write("\n" + "="*70 + "\n")
    f.write("STATISTICAL SUMMARY\n")
    f.write("="*70 + "\n")
    f.write(f"  Correlation (Vol vs Avg Duration):    {corr_avg:.3f}\n")
    f.write(f"  Correlation (Vol vs Median Duration): {corr_median:.3f}\n")
    f.write(f"  Hypothesis (negative correlation): {'SUPPORTED' if corr_avg < -0.3 else 'NOT SUPPORTED'}\n")

print("\nResults written to volatility_persistence_results.txt")
