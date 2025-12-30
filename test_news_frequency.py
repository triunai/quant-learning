"""Test if news frequency (via kurtosis/jump frequency) predicts regime duration."""
import numpy as np
import pandas as pd
import yfinance as yf
import sys
import io
from scipy import stats
import matplotlib.pyplot as plt

# Load the module (suppress output)
old_stdout = sys.stdout
sys.stdout = io.StringIO()

with open('battle-tested/PLTR-test-2.py', encoding='utf-8') as f:
    code = f.read().split('if __name__')[0]
exec(code)

sys.stdout = old_stdout

stocks = [
    ('PLTR', 'QQQ'), ('TSLA', 'QQQ'), ('NVDA', 'QQQ'), ('AMD', 'QQQ'), ('COIN', 'QQQ'),
    ('MSFT', 'QQQ'), ('AAPL', 'QQQ'), ('GOOGL', 'QQQ'), ('META', 'QQQ'), ('AMZN', 'QQQ'),
    ('WMT', 'SPY'), ('JNJ', 'SPY'), ('PG', 'SPY'), ('KO', 'SPY'), ('PEP', 'SPY'),
    ('JPM', 'SPY'), ('BAC', 'SPY'), ('GS', 'SPY'), ('XOM', 'SPY'), ('CVX', 'SPY'),
    ('QQQ', 'SPY'), ('SPY', 'QQQ'),
]

results = []

print("Testing NEWS FREQUENCY vs Regime Duration...")
print("="*70)
print(f"{'Ticker':<8} {'Vol':<8} {'Kurt':<8} {'JumpFreq':<10} {'VolClust':<10} {'AvgDur':<10}")
print("-"*70)

for ticker, market in stocks:
    try:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        platform = RegimeRiskPlatform(
            ticker=ticker, market_ticker=market,
            n_regimes=3, days_ahead=126, simulations=100
        )
        platform.ingest_data()
        platform.build_regime_model()
        
        sys.stdout = old_stdout
        
        returns = platform.data['Log_Ret'].values
        
        # NEWS FREQUENCY PROXIES
        # 1. Kurtosis (fat tails = news-driven jumps)
        kurtosis = stats.kurtosis(returns)
        
        # 2. Jump frequency (% of days with >3 std moves)
        std = np.std(returns)
        jump_freq = np.mean(np.abs(returns) > 3 * std) * 100  # Percentage
        
        # 3. Volatility clustering (ACF of squared returns)
        squared_returns = returns ** 2
        vol_cluster = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
        
        # Combined news score
        news_score = (kurtosis / 10) * (1 + jump_freq) * (1 + vol_cluster)
        
        # Regime duration
        all_durations = []
        for r in range(platform.n_regimes):
            samples = platform.regime_duration.get(r, {}).get('samples', [])
            if samples:
                all_durations.extend(samples)
        avg_duration = np.mean(all_durations) if all_durations else 0
        
        results.append({
            'ticker': ticker,
            'volatility': platform.realized_vol,
            'kurtosis': kurtosis,
            'jump_freq': jump_freq,
            'vol_cluster': vol_cluster,
            'news_score': news_score,
            'avg_duration': avg_duration
        })
        
        print(f"{ticker:<8} {platform.realized_vol:.1%}   {kurtosis:>5.1f}   {jump_freq:>6.2f}%   {vol_cluster:>8.2f}   {avg_duration:>6.1f}d")
        
    except Exception as e:
        sys.stdout = old_stdout
        print(f"{ticker:<8} ERROR: {str(e)[:40]}")

print("\n" + "="*70)

if len(results) > 5:
    # Calculate correlations
    kurt = [r['kurtosis'] for r in results]
    jump = [r['jump_freq'] for r in results]
    volc = [r['vol_cluster'] for r in results]
    news = [r['news_score'] for r in results]
    dur = [r['avg_duration'] for r in results]
    vol = [r['volatility'] for r in results]
    
    # Exclude META outlier for robustness check
    results_no_meta = [r for r in results if r['ticker'] != 'META']
    dur_no_meta = [r['avg_duration'] for r in results_no_meta]
    kurt_no_meta = [r['kurtosis'] for r in results_no_meta]
    jump_no_meta = [r['jump_freq'] for r in results_no_meta]
    news_no_meta = [r['news_score'] for r in results_no_meta]
    
    print("\nCORRELATION WITH REGIME DURATION:")
    print("-"*50)
    print(f"  Kurtosis:           r = {np.corrcoef(kurt, dur)[0,1]:+.3f}")
    print(f"  Jump Frequency:     r = {np.corrcoef(jump, dur)[0,1]:+.3f}")
    print(f"  Vol Clustering:     r = {np.corrcoef(volc, dur)[0,1]:+.3f}")
    print(f"  News Score:         r = {np.corrcoef(news, dur)[0,1]:+.3f}")
    print(f"  Volatility:         r = {np.corrcoef(vol, dur)[0,1]:+.3f}")
    
    print(f"\n  (Excluding META outlier):")
    print(f"  Kurtosis:           r = {np.corrcoef(kurt_no_meta, dur_no_meta)[0,1]:+.3f}")
    print(f"  News Score:         r = {np.corrcoef(news_no_meta, dur_no_meta)[0,1]:+.3f}")
    
    # Hypothesis: High news frequency -> shorter regimes (NEGATIVE correlation expected)
    corr_main = np.corrcoef(news, dur)[0,1]
    corr_nometa = np.corrcoef(news_no_meta, dur_no_meta)[0,1]
    
    print("\n" + "="*70)
    print("HYPOTHESIS TEST: High news frequency -> Shorter regimes")
    print("="*70)
    
    if corr_main < -0.3:
        print(f"  [SUPPORTED] (r={corr_main:.2f}) High news frequency correlates with shorter regimes")
    elif corr_main > 0.3:
        print(f"  [OPPOSITE] (r={corr_main:.2f}) High news frequency correlates with LONGER regimes")
    else:
        print(f"  [INCONCLUSIVE] (r={corr_main:.2f}) Weak correlation")
    
    # Create multi-panel plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Kurtosis vs Duration
    ax = axes[0, 0]
    ax.scatter([r['kurtosis'] for r in results], dur, s=80, alpha=0.7)
    for r in results:
        ax.annotate(r['ticker'], (r['kurtosis'], r['avg_duration']), fontsize=8)
    ax.set_xlabel('Kurtosis (Fat Tails)')
    ax.set_ylabel('Avg Regime Duration (days)')
    ax.set_title(f"Kurtosis vs Duration (r={np.corrcoef(kurt, dur)[0,1]:.2f})")
    ax.grid(True, alpha=0.3)
    
    # 2. Jump Frequency vs Duration
    ax = axes[0, 1]
    ax.scatter(jump, dur, s=80, alpha=0.7, color='orange')
    for r in results:
        ax.annotate(r['ticker'], (r['jump_freq'], r['avg_duration']), fontsize=8)
    ax.set_xlabel('Jump Frequency (%)')
    ax.set_ylabel('Avg Regime Duration (days)')
    ax.set_title(f"Jump Frequency vs Duration (r={np.corrcoef(jump, dur)[0,1]:.2f})")
    ax.grid(True, alpha=0.3)
    
    # 3. News Score vs Duration
    ax = axes[1, 0]
    ax.scatter(news, dur, s=80, alpha=0.7, color='green')
    for r in results:
        ax.annotate(r['ticker'], (r['news_score'], r['avg_duration']), fontsize=8)
    ax.set_xlabel('News Score (Kurtosis × JumpFreq × VolClust)')
    ax.set_ylabel('Avg Regime Duration (days)')
    ax.set_title(f"News Score vs Duration (r={np.corrcoef(news, dur)[0,1]:.2f})")
    ax.grid(True, alpha=0.3)
    
    # 4. Volatility vs Duration
    ax = axes[1, 1]
    ax.scatter([v*100 for v in vol], dur, s=80, alpha=0.7, color='red')
    for r in results:
        ax.annotate(r['ticker'], (r['volatility']*100, r['avg_duration']), fontsize=8)
    ax.set_xlabel('Volatility (%)')
    ax.set_ylabel('Avg Regime Duration (days)')
    ax.set_title(f"Volatility vs Duration (r={np.corrcoef(vol, dur)[0,1]:.2f})")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('news_frequency_analysis.png', dpi=150, facecolor='white')
    print(f"\n  Plot saved to: news_frequency_analysis.png")

# Write detailed results
with open('news_frequency_results.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("NEWS FREQUENCY VS REGIME DURATION ANALYSIS\n")
    f.write("="*80 + "\n\n")
    
    f.write("Hypothesis: High news frequency (measured by kurtosis, jump freq, vol clustering)\n")
    f.write("            correlates with SHORTER regime durations.\n\n")
    
    f.write(f"{'Ticker':<8} {'Vol':<8} {'Kurt':<8} {'JumpFreq':<10} {'VolClust':<10} {'NewsScore':<12} {'AvgDur':<10}\n")
    f.write("-"*80 + "\n")
    
    for r in sorted(results, key=lambda x: x['news_score'], reverse=True):
        f.write(f"{r['ticker']:<8} {r['volatility']:.1%}   {r['kurtosis']:>5.1f}   {r['jump_freq']:>6.2f}%   ")
        f.write(f"{r['vol_cluster']:>8.2f}   {r['news_score']:>8.2f}     {r['avg_duration']:>6.1f}d\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("CORRELATION MATRIX\n")
    f.write("="*80 + "\n")
    f.write(f"  Kurtosis vs Duration:       r = {np.corrcoef(kurt, dur)[0,1]:+.3f}\n")
    f.write(f"  Jump Frequency vs Duration: r = {np.corrcoef(jump, dur)[0,1]:+.3f}\n")
    f.write(f"  Vol Clustering vs Duration: r = {np.corrcoef(volc, dur)[0,1]:+.3f}\n")
    f.write(f"  News Score vs Duration:     r = {np.corrcoef(news, dur)[0,1]:+.3f}\n")
    f.write(f"  Volatility vs Duration:     r = {np.corrcoef(vol, dur)[0,1]:+.3f}\n")

print("\nResults written to news_frequency_results.txt")
