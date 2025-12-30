"""Cross-stock regime persistence comparison.
Tests if regime dynamics differ by stock type.
"""
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

sys.stdout = old_stdout

stocks = [
    ('PLTR', 'QQQ', 'Growth/Speculative'),
    ('MSFT', 'QQQ', 'Mega-Cap Tech'),
    ('WMT', 'SPY', 'Defensive Value'),
    ('QQQ', 'SPY', 'Tech Index'),
]

results = {}

with open('regime_persistence_comparison.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("CROSS-STOCK REGIME PERSISTENCE COMPARISON\n")
    f.write("="*80 + "\n\n")
    
    for ticker, market, category in stocks:
        f.write(f"\n{'='*80}\n")
        f.write(f"{ticker} ({category}) vs {market}\n")
        f.write("="*80 + "\n")
        
        try:
            # Suppress platform output
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            platform = RegimeRiskPlatform(
                ticker=ticker,
                market_ticker=market,
                n_regimes=3,
                days_ahead=126,
                simulations=1000
            )
            platform.ingest_data()
            platform.build_regime_model()
            platform.compute_market_beta()
            
            sys.stdout = old_stdout
            
            f.write(f"\n[BASIC STATS]\n")
            f.write(f"  Price: ${platform.last_price:.2f}\n")
            f.write(f"  Realized Vol: {platform.realized_vol:.1%}\n")
            f.write(f"  Global Beta: {platform.market_beta:.2f}\n")
            f.write(f"  Global Alpha (ann): {platform.market_alpha*252:.1%}\n")
            f.write(f"  Current Regime: {platform.regime_names.get(platform.current_regime, 'Unknown')}\n")
            
            # Regime persistence analysis
            f.write(f"\n[REGIME PERSISTENCE]\n")
            
            stock_results = {
                'category': category,
                'vol': platform.realized_vol,
                'beta': platform.market_beta,
                'regimes': {}
            }
            
            for r in range(platform.n_regimes):
                name = platform.regime_names.get(r, f"R{r}")
                duration_info = platform.regime_duration.get(r, {'mean': 0, 'samples': []})
                
                n_periods = len(duration_info.get('samples', []))
                mean_dur = duration_info.get('mean', 0)
                median_dur = np.median(duration_info.get('samples', [0])) if duration_info.get('samples') else 0
                
                # Regime stats
                mask = platform.data['Regime'] == r
                n_days = np.sum(mask)
                pct = n_days / len(platform.data) * 100
                
                returns = platform.data.loc[mask, 'Log_Ret'].values
                ann_return = np.mean(returns) * 252 * 100 if len(returns) > 0 else 0
                ann_vol = np.std(returns) * np.sqrt(252) * 100 if len(returns) > 0 else 0
                
                alpha = platform.regime_alpha.get(r, 0) * 252 * 100
                beta = platform.regime_beta.get(r, 1.0)
                model_type = platform.regime_model_type.get(r, 'unknown')
                
                f.write(f"\n  [{name}] (model: {model_type})\n")
                f.write(f"    Days: {n_days} ({pct:.0f}%)\n")
                f.write(f"    Duration: mean={mean_dur:.1f}d, median={median_dur:.1f}d, periods={n_periods}\n")
                f.write(f"    Return: {ann_return:+.1f}% | Vol: {ann_vol:.1f}%\n")
                f.write(f"    Alpha: {alpha:+.1f}% | Beta: {beta:.2f}\n")
                
                stock_results['regimes'][name] = {
                    'days': n_days,
                    'pct': pct,
                    'mean_duration': mean_dur,
                    'median_duration': median_dur,
                    'n_periods': n_periods,
                    'return': ann_return,
                    'vol': ann_vol,
                    'alpha': alpha,
                    'beta': beta
                }
            
            results[ticker] = stock_results
            
            # Expected regime changes in 6 months
            f.write(f"\n[EXPECTED REGIME DYNAMICS (126-day horizon)]\n")
            total_expected_changes = 0
            for r in range(platform.n_regimes):
                name = platform.regime_names.get(r, f"R{r}")
                duration_info = platform.regime_duration.get(r, {'mean': 10})
                mean_dur = duration_info.get('mean', 10)
                if mean_dur > 0:
                    expected = 126 / mean_dur
                    total_expected_changes += expected * (platform.data['Regime'] == r).mean()
                    f.write(f"    {name}: ~{expected:.1f} exits if starting here\n")
            
            f.write(f"    WEIGHTED TOTAL: ~{total_expected_changes:.1f} transitions in 6 months\n")
            
        except Exception as e:
            sys.stdout = old_stdout
            f.write(f"  ERROR: {str(e)}\n")
    
    # Summary comparison
    f.write("\n\n" + "="*80 + "\n")
    f.write("SUMMARY COMPARISON\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"{'Ticker':<8} {'Category':<20} {'Vol':<8} {'Beta':<6} {'Regimes':<30}\n")
    f.write("-"*80 + "\n")
    
    for ticker, data in results.items():
        regimes_str = ", ".join([f"{name[:3]}:{info['mean_duration']:.0f}d" 
                                  for name, info in data['regimes'].items()])
        f.write(f"{ticker:<8} {data['category']:<20} {data['vol']:.1%}  {data['beta']:.2f}   {regimes_str}\n")
    
    # Key insights
    f.write("\n\n[KEY INSIGHTS]\n")
    
    # Find stock with longest regime durations
    longest_regime = None
    longest_duration = 0
    for ticker, data in results.items():
        for name, info in data['regimes'].items():
            if info['mean_duration'] > longest_duration:
                longest_duration = info['mean_duration']
                longest_regime = (ticker, name)
    
    if longest_regime:
        f.write(f"  Most persistent regime: {longest_regime[0]} {longest_regime[1]} ({longest_duration:.0f} days avg)\n")
    
    # Compare speculative vs stable
    if 'PLTR' in results and 'WMT' in results:
        pltr_volatility = results['PLTR']['vol']
        wmt_volatility = results['WMT']['vol']
        f.write(f"  PLTR vol ({pltr_volatility:.1%}) vs WMT vol ({wmt_volatility:.1%}): {pltr_volatility/wmt_volatility:.1f}x difference\n")

print("Results written to regime_persistence_comparison.txt")
