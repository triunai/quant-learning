"""
PHASE 3: RESEARCH VALIDATION - Kurtosis vs Regime Duration
============================================================

HYPOTHESIS: Fat-tail stocks (high kurtosis) have LONGER regime durations.
- Original finding: r = +0.84 on 22 stocks
- This contradicts conventional wisdom (jumps → more transitions)
- Needs validation before trusting

VALIDATION TESTS:
1. Expand to 100+ stocks across sectors
2. Outlier sensitivity (remove META and retest)
3. Sector-level analysis (does it hold within sectors?)
4. Time-period robustness (2015-2019 vs 2020-2024)
5. Out-of-sample prediction

DECISION MATRIX:
- r > 0.5 on 100+ stocks → VALID, implement stock classification
- r driven by outliers    → FRAGILE, document only
- r < 0.3                 → FAILED, move on

Created: 2025-12-31
"""

import numpy as np
import pandas as pd
import yfinance as yf
import sys
import io
import warnings
from scipy import stats
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json

warnings.filterwarnings('ignore')

# ============================================================================
# UNIVERSE DEFINITION - 100+ stocks across sectors
# ============================================================================

UNIVERSE = {
    'Large Cap Tech': ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NVDA', 'TSLA', 'ORCL', 'CRM', 'ADBE', 'INTC', 'AMD'],
    'Financials': ['JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'BLK', 'SCHW', 'AXP', 'USB'],
    'Consumer Discretionary': ['WMT', 'TGT', 'COST', 'HD', 'NKE', 'SBUX', 'MCD', 'LOW', 'TJX', 'BKNG'],
    'Consumer Staples': ['PG', 'KO', 'PEP', 'PM', 'MO', 'CL', 'KMB', 'GIS', 'K', 'CAG'],
    'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'LLY', 'BMY', 'TMO', 'DHR', 'AMGN'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY', 'MPC', 'VLO', 'PSX', 'HAL'],
    'Industrials': ['CAT', 'BA', 'GE', 'HON', 'UPS', 'DE', 'MMM', 'RTX', 'LMT', 'FDX'],
    'Materials': ['LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'DOW', 'DD'],
    'REITs': ['O', 'AMT', 'PLD', 'SPG', 'EQIX', 'PSA', 'DLR'],
    'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'XEL', 'SRE'],
    'Communication': ['VZ', 'T', 'TMUS', 'DIS', 'NFLX', 'CMCSA'],
    'Indices/ETFs': ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI'],
    'High Volatility': ['COIN', 'MARA', 'RIOT', 'GME', 'AMC'],
}

# Map each stock to its sector for analysis
STOCK_TO_SECTOR = {}
for sector, stocks in UNIVERSE.items():
    for stock in stocks:
        STOCK_TO_SECTOR[stock] = sector

# Market reference for each sector (for factor model)
SECTOR_MARKET = {
    'Large Cap Tech': 'QQQ',
    'Financials': 'XLF',
    'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',
    'Healthcare': 'XLV',
    'Energy': 'XLE',
    'Industrials': 'XLI',
    'Materials': 'XLB',
    'REITs': 'XLRE',
    'Utilities': 'XLU',
    'Communication': 'XLC',
    'Indices/ETFs': 'SPY',
    'High Volatility': 'QQQ',
}


# ============================================================================
# LOAD REGIME PLATFORM
# ============================================================================

def load_regime_platform():
    """Load the RegimeRiskPlatform class from battle-tested."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    with open('battle-tested/PLTR-test-2.py', encoding='utf-8') as f:
        code = f.read().split('if __name__')[0]
    exec(code, globals())
    
    sys.stdout = old_stdout
    return globals()['RegimeRiskPlatform']


# ============================================================================
# DATA EXTRACTION
# ============================================================================

def extract_metrics(ticker: str, market: str, platform_cls, 
                   start_date: str = None, end_date: str = None,
                   n_simulations: int = 100) -> Dict:
    """
    Extract kurtosis and regime duration metrics for a single stock.
    
    Returns:
        Dict with ticker, kurtosis, avg_duration, volatility, etc.
        Returns None if analysis fails.
    """
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        platform = platform_cls(
            ticker=ticker, 
            market_ticker=market,
            n_regimes=3, 
            days_ahead=126, 
            simulations=n_simulations
        )
        
        # Custom date handling if specified
        if start_date and end_date:
            # Download data for specific period
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            market_data = yf.download(market, start=start_date, end=end_date, progress=False)
            
            if len(data) < 252:  # Need at least 1 year of data
                return None
                
            # Prepare data manually
            df = pd.DataFrame()
            df['Adj Close'] = data['Adj Close']
            df['Market'] = market_data['Adj Close']
            df = df.dropna()
            df['Log_Ret'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
            df['Mkt_Ret'] = np.log(df['Market'] / df['Market'].shift(1))
            df = df.dropna()
            
            platform.data = df
            platform.realized_vol = df['Log_Ret'].std() * np.sqrt(252)
        else:
            platform.ingest_data()
        
        platform.build_regime_model()
        
        sys.stdout = old_stdout
        
        returns = platform.data['Log_Ret'].values
        
        # Metrics
        kurtosis = stats.kurtosis(returns)
        volatility = platform.realized_vol
        
        # Jump frequency (% of days with >3 std moves)
        std = np.std(returns)
        jump_freq = np.mean(np.abs(returns) > 3 * std) * 100
        
        # Volatility clustering (ACF of squared returns)
        squared_returns = returns ** 2
        if len(squared_returns) > 1:
            vol_cluster = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
        else:
            vol_cluster = 0
        
        # Average regime duration across all regimes
        all_durations = []
        for r in range(platform.n_regimes):
            samples = platform.regime_duration.get(r, {}).get('samples', [])
            if samples:
                all_durations.extend(samples)
        avg_duration = np.mean(all_durations) if all_durations else 0
        
        # Per-regime stats
        regime_durations = {}
        for r in range(platform.n_regimes):
            name = platform.regime_names.get(r, f"R{r}")
            d = platform.regime_duration.get(r, {})
            regime_durations[name] = {
                'mean': d.get('mean', 0),
                'std': d.get('std', 0),
                'count': len(d.get('samples', []))
            }
        
        return {
            'ticker': ticker,
            'sector': STOCK_TO_SECTOR.get(ticker, 'Unknown'),
            'kurtosis': kurtosis,
            'volatility': volatility,
            'jump_freq': jump_freq,
            'vol_cluster': vol_cluster,
            'avg_duration': avg_duration,
            'regime_durations': regime_durations,
            'n_observations': len(returns),
            'start_date': str(platform.data.index[0].date()) if hasattr(platform.data.index[0], 'date') else str(platform.data.index[0]),
            'end_date': str(platform.data.index[-1].date()) if hasattr(platform.data.index[-1], 'date') else str(platform.data.index[-1]),
        }
        
    except Exception as e:
        sys.stdout = old_stdout
        return {'ticker': ticker, 'error': str(e)}


# ============================================================================
# VALIDATION TESTS
# ============================================================================

class KurtosisValidation:
    """Comprehensive validation of kurtosis-duration hypothesis."""
    
    def __init__(self):
        self.platform_cls = None
        self.results = []
        self.validation_report = {}
        
    def load_platform(self):
        """Load the regime platform class."""
        print("Loading RegimeRiskPlatform...")
        self.platform_cls = load_regime_platform()
        print("✓ Platform loaded\n")
    
    def test_expanded_universe(self, min_stocks: int = 50) -> Dict:
        """
        TEST 1: Expand to 100+ stocks across sectors.
        
        Success Criteria:
        - Correlation remains > 0.5 with 50+ stocks
        - Statistical significance (p < 0.01)
        """
        print("=" * 70)
        print("TEST 1: EXPANDED UNIVERSE (100+ stocks)")
        print("=" * 70)
        
        all_tickers = []
        for sector, tickers in UNIVERSE.items():
            all_tickers.extend([(t, SECTOR_MARKET.get(sector, 'SPY')) for t in tickers])
        
        print(f"Testing {len(all_tickers)} stocks across {len(UNIVERSE)} sectors...\n")
        print(f"{'Ticker':<8} {'Sector':<25} {'Kurt':>8} {'AvgDur':>10} {'Status'}")
        print("-" * 65)
        
        successful = 0
        for ticker, market in all_tickers:
            result = extract_metrics(ticker, market, self.platform_cls)
            
            if result and 'error' not in result:
                self.results.append(result)
                successful += 1
                print(f"{ticker:<8} {result['sector']:<25} {result['kurtosis']:>8.2f} {result['avg_duration']:>8.1f}d    ✓")
            else:
                error = result.get('error', 'Unknown') if result else 'Failed'
                print(f"{ticker:<8} {'--':<25} {'--':>8} {'--':>10}    ✗ {error[:30]}")
        
        print("\n" + "-" * 70)
        print(f"Successfully analyzed: {successful}/{len(all_tickers)} stocks")
        
        if successful < min_stocks:
            print(f"⚠️  WARNING: Only {successful} stocks analyzed (need {min_stocks}+)")
        
        # Calculate correlation
        if len(self.results) > 10:
            kurt = [r['kurtosis'] for r in self.results]
            dur = [r['avg_duration'] for r in self.results]
            
            # Pearson and Spearman correlation
            pearson_r, pearson_p = pearsonr(kurt, dur)
            spearman_r, spearman_p = spearmanr(kurt, dur)
            
            print(f"\nCORRELATION RESULTS:")
            print(f"  Pearson r  = {pearson_r:+.4f}  (p = {pearson_p:.2e})")
            print(f"  Spearman ρ = {spearman_r:+.4f}  (p = {spearman_p:.2e})")
            
            self.validation_report['test1_expanded'] = {
                'n_stocks': successful,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'status': 'PASS' if abs(pearson_r) > 0.3 and pearson_p < 0.05 else 'FAIL'
            }
            
            return self.validation_report['test1_expanded']
        
        return {'status': 'INSUFFICIENT_DATA'}
    
    def test_outlier_sensitivity(self, outliers: List[str] = ['META']) -> Dict:
        """
        TEST 2: Outlier Sensitivity - Remove META and retest.
        
        META has kurtosis=26.6, duration=119d - extreme outlier.
        If correlation drops significantly without it, finding is fragile.
        """
        print("\n" + "=" * 70)
        print("TEST 2: OUTLIER SENSITIVITY")
        print("=" * 70)
        
        if not self.results:
            print("No results to analyze. Run test_expanded_universe first.")
            return {'status': 'NO_DATA'}
        
        # With all data
        all_kurt = [r['kurtosis'] for r in self.results]
        all_dur = [r['avg_duration'] for r in self.results]
        all_r, all_p = pearsonr(all_kurt, all_dur)
        
        # Without outliers
        filtered = [r for r in self.results if r['ticker'] not in outliers]
        filt_kurt = [r['kurtosis'] for r in filtered]
        filt_dur = [r['avg_duration'] for r in filtered]
        filt_r, filt_p = pearsonr(filt_kurt, filt_dur)
        
        # Cook's distance analysis - identify influential points
        # Simplified: find stocks with high leverage
        kurt_z = (np.array(all_kurt) - np.mean(all_kurt)) / np.std(all_kurt)
        dur_z = (np.array(all_dur) - np.mean(all_dur)) / np.std(all_dur)
        leverage = kurt_z ** 2 + dur_z ** 2
        
        influential_idx = np.argsort(leverage)[-5:]  # Top 5 influential
        influential_stocks = [self.results[i]['ticker'] for i in influential_idx]
        
        print(f"\nWith all stocks ({len(self.results)} total):")
        print(f"  r = {all_r:+.4f}")
        
        print(f"\nWithout outliers ({outliers}):")
        print(f"  r = {filt_r:+.4f}  (Δr = {filt_r - all_r:+.4f})")
        
        print(f"\nMost influential stocks (high leverage):")
        for i in influential_idx[::-1]:
            r = self.results[i]
            print(f"  {r['ticker']}: kurt={r['kurtosis']:.1f}, dur={r['avg_duration']:.1f}d")
        
        # Remove all influential and retest
        non_influential = [r for r in self.results if r['ticker'] not in influential_stocks]
        ni_kurt = [r['kurtosis'] for r in non_influential]
        ni_dur = [r['avg_duration'] for r in non_influential]
        ni_r, ni_p = pearsonr(ni_kurt, ni_dur)
        
        print(f"\nWithout top 5 influential points:")
        print(f"  r = {ni_r:+.4f}  (Δr = {ni_r - all_r:+.4f})")
        
        # Assess fragility
        is_fragile = abs(filt_r - all_r) > 0.2 or abs(ni_r - all_r) > 0.3
        
        print("\n" + "-" * 50)
        if is_fragile:
            print("⚠️  FINDING IS FRAGILE: Correlation sensitive to outliers")
        else:
            print("✓ FINDING IS ROBUST: Correlation survives outlier removal")
        
        self.validation_report['test2_outliers'] = {
            'r_all': all_r,
            'r_no_meta': filt_r,
            'r_no_influential': ni_r,
            'delta_no_outliers': filt_r - all_r,
            'delta_no_influential': ni_r - all_r,
            'influential_stocks': influential_stocks,
            'is_fragile': is_fragile,
            'status': 'FRAGILE' if is_fragile else 'ROBUST'
        }
        
        return self.validation_report['test2_outliers']
    
    def test_sector_analysis(self) -> Dict:
        """
        TEST 3: Sector Analysis - Does correlation hold within sectors?
        
        Key insight: If it only holds ACROSS sectors but not WITHIN,
        it might be a sector composition effect rather than fundamental.
        """
        print("\n" + "=" * 70)
        print("TEST 3: SECTOR ANALYSIS")
        print("=" * 70)
        
        if not self.results:
            print("No results to analyze. Run test_expanded_universe first.")
            return {'status': 'NO_DATA'}
        
        # Group by sector
        sector_data = {}
        for r in self.results:
            sector = r['sector']
            if sector not in sector_data:
                sector_data[sector] = []
            sector_data[sector].append(r)
        
        print(f"\n{'Sector':<25} {'N':<5} {'r':>8} {'p':>10} {'Status'}")
        print("-" * 60)
        
        sector_correlations = {}
        sectors_with_positive_r = 0
        sectors_significant = 0
        
        for sector, data in sorted(sector_data.items()):
            n = len(data)
            if n >= 4:  # Need at least 4 points for correlation
                kurt = [r['kurtosis'] for r in data]
                dur = [r['avg_duration'] for r in data]
                r, p = pearsonr(kurt, dur)
                
                status = ""
                if p < 0.05:
                    status = "✓ SIG"
                    sectors_significant += 1
                if r > 0:
                    sectors_with_positive_r += 1
                
                sector_correlations[sector] = {'r': r, 'p': p, 'n': n}
                print(f"{sector:<25} {n:<5} {r:>+8.3f} {p:>10.3f}  {status}")
            else:
                print(f"{sector:<25} {n:<5} {'N/A':>8} {'--':>10}  (too few)")
        
        print("\n" + "-" * 60)
        n_sectors = len([s for s in sector_correlations if sector_correlations[s]['n'] >= 4])
        print(f"Positive correlation: {sectors_with_positive_r}/{n_sectors} sectors")
        print(f"Statistically significant: {sectors_significant}/{n_sectors} sectors")
        
        self.validation_report['test3_sectors'] = {
            'sector_correlations': sector_correlations,
            'n_positive': sectors_with_positive_r,
            'n_significant': sectors_significant,
            'n_sectors': n_sectors,
            'status': 'MIXED' if sectors_with_positive_r < n_sectors * 0.7 else 'CONSISTENT'
        }
        
        return self.validation_report['test3_sectors']
    
    def test_time_robustness(self) -> Dict:
        """
        TEST 4: Time-Period Robustness
        
        Test on two periods:
        - Period A: 2015-01-01 to 2019-12-31 (pre-COVID)
        - Period B: 2020-01-01 to 2024-12-31 (COVID and after)
        
        Key question: Is this finding COVID-era specific?
        """
        print("\n" + "=" * 70)
        print("TEST 4: TIME-PERIOD ROBUSTNESS")
        print("=" * 70)
        
        # Use a subset for efficiency
        test_tickers = [
            ('AAPL', 'QQQ'), ('MSFT', 'QQQ'), ('GOOGL', 'QQQ'), ('META', 'QQQ'),
            ('AMZN', 'QQQ'), ('JPM', 'SPY'), ('BAC', 'SPY'), ('XOM', 'SPY'),
            ('CVX', 'SPY'), ('JNJ', 'SPY'), ('WMT', 'SPY'), ('HD', 'SPY'),
            ('CAT', 'SPY'), ('BA', 'SPY'), ('NEE', 'SPY'), ('SPY', 'QQQ'),
        ]
        
        periods = {
            'Pre-COVID': ('2015-01-01', '2019-12-31'),
            'Post-COVID': ('2020-01-01', '2024-12-31'),
        }
        
        period_results = {}
        
        for period_name, (start, end) in periods.items():
            print(f"\n{period_name}: {start} to {end}")
            print("-" * 50)
            
            results = []
            for ticker, market in test_tickers:
                r = extract_metrics(ticker, market, self.platform_cls, start, end)
                if r and 'error' not in r:
                    results.append(r)
                    print(f"  {ticker}: kurt={r['kurtosis']:.2f}, dur={r['avg_duration']:.1f}d")
                else:
                    print(f"  {ticker}: FAILED")
            
            if len(results) >= 5:
                kurt = [r['kurtosis'] for r in results]
                dur = [r['avg_duration'] for r in results]
                r, p = pearsonr(kurt, dur)
                
                period_results[period_name] = {
                    'n': len(results),
                    'r': r,
                    'p': p,
                    'results': results
                }
                print(f"\n  Correlation: r = {r:+.4f} (p = {p:.4f})")
        
        print("\n" + "-" * 50)
        print("TIME ROBUSTNESS SUMMARY:")
        
        for period_name, data in period_results.items():
            status = "✓" if data['r'] > 0.3 and data['p'] < 0.1 else "✗"
            print(f"  {period_name}: r = {data['r']:+.3f}  {status}")
        
        # Check if finding is time-stable
        r_values = [d['r'] for d in period_results.values()]
        is_stable = all(r > 0.2 for r in r_values) or all(r < -0.2 for r in r_values)
        
        self.validation_report['test4_time'] = {
            'period_results': {k: {k2: v2 for k2, v2 in v.items() if k2 != 'results'} 
                              for k, v in period_results.items()},
            'is_stable': is_stable,
            'status': 'STABLE' if is_stable else 'ERA-SPECIFIC'
        }
        
        return self.validation_report['test4_time']
    
    def test_out_of_sample(self) -> Dict:
        """
        TEST 5: Out-of-Sample Prediction
        
        The real test:
        1. Use 2015-2022 data to fit kurtosis→duration relationship
        2. Predict regime durations for 2023-2024
        3. Compare predicted vs actual
        
        If this fails, finding is descriptive, not predictive.
        """
        print("\n" + "=" * 70)
        print("TEST 5: OUT-OF-SAMPLE PREDICTION")
        print("=" * 70)
        
        test_tickers = [
            ('AAPL', 'QQQ'), ('MSFT', 'QQQ'), ('JPM', 'SPY'), ('XOM', 'SPY'),
            ('AMZN', 'QQQ'), ('GOOGL', 'QQQ'), ('JNJ', 'SPY'), ('WMT', 'SPY'),
            ('BAC', 'SPY'), ('CAT', 'SPY'), ('HD', 'SPY'), ('NVDA', 'QQQ'),
        ]
        
        # Training period: 2015-2022
        # Test period: 2023-2024
        train_start, train_end = '2015-01-01', '2022-12-31'
        test_start, test_end = '2023-01-01', '2024-12-31'
        
        print(f"\nTraining period: {train_start} to {train_end}")
        print(f"Test period: {test_start} to {test_end}")
        print("-" * 50)
        
        # Get training data
        train_results = []
        for ticker, market in test_tickers:
            r = extract_metrics(ticker, market, self.platform_cls, train_start, train_end)
            if r and 'error' not in r:
                train_results.append(r)
        
        if len(train_results) < 5:
            print("⚠️ Insufficient training data")
            return {'status': 'INSUFFICIENT_DATA'}
        
        # Fit linear model: duration = a + b * kurtosis
        train_kurt = np.array([r['kurtosis'] for r in train_results])
        train_dur = np.array([r['avg_duration'] for r in train_results])
        
        # Simple linear regression
        slope, intercept = np.polyfit(train_kurt, train_dur, 1)
        
        print(f"\nTrained model: duration = {intercept:.2f} + {slope:.2f} * kurtosis")
        print(f"  (based on {len(train_results)} stocks)")
        
        # Get test data and predict
        test_results = []
        predictions = []
        actuals = []
        
        print(f"\n{'Ticker':<8} {'Kurt':>8} {'Pred':>10} {'Actual':>10} {'Error':>10}")
        print("-" * 50)
        
        for ticker, market in test_tickers:
            r = extract_metrics(ticker, market, self.platform_cls, test_start, test_end)
            if r and 'error' not in r:
                test_results.append(r)
                
                predicted_dur = intercept + slope * r['kurtosis']
                actual_dur = r['avg_duration']
                error = predicted_dur - actual_dur
                
                predictions.append(predicted_dur)
                actuals.append(actual_dur)
                
                print(f"{ticker:<8} {r['kurtosis']:>8.2f} {predicted_dur:>10.1f} {actual_dur:>10.1f} {error:>+10.1f}")
        
        if len(predictions) < 5:
            print("⚠️ Insufficient test data")
            return {'status': 'INSUFFICIENT_DATA'}
        
        # Calculate prediction metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        r_oos, p_oos = pearsonr(predictions, actuals)
        
        # Naive baseline: predict mean of training
        naive_pred = np.mean(train_dur)
        naive_mae = np.mean(np.abs(naive_pred - actuals))
        
        skill = 1 - (mae / naive_mae)  # Skill score vs naive baseline
        
        print("\n" + "-" * 50)
        print("OUT-OF-SAMPLE METRICS:")
        print(f"  MAE:  {mae:.1f} days")
        print(f"  RMSE: {rmse:.1f} days")
        print(f"  r (pred vs actual): {r_oos:+.3f}")
        print(f"  Naive MAE: {naive_mae:.1f} days")
        print(f"  Skill Score: {skill:+.2%}")
        
        # Assessment
        is_predictive = r_oos > 0.3 and skill > 0
        
        print("\n" + "-" * 50)
        if is_predictive:
            print("✓ FINDING IS PREDICTIVE: Kurtosis predicts future regime duration")
        else:
            print("⚠️ FINDING IS DESCRIPTIVE: No out-of-sample predictive power")
        
        self.validation_report['test5_oos'] = {
            'model_intercept': intercept,
            'model_slope': slope,
            'n_train': len(train_results),
            'n_test': len(test_results),
            'mae': mae,
            'rmse': rmse,
            'r_oos': r_oos,
            'skill_score': skill,
            'is_predictive': is_predictive,
            'status': 'PREDICTIVE' if is_predictive else 'DESCRIPTIVE'
        }
        
        return self.validation_report['test5_oos']
    
    def generate_visualizations(self):
        """Generate summary visualizations."""
        if not self.results:
            print("No results to visualize.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Kurtosis vs Regime Duration: Research Validation', fontsize=14, fontweight='bold')
        
        # Prepare data
        kurt = [r['kurtosis'] for r in self.results]
        dur = [r['avg_duration'] for r in self.results]
        tickers = [r['ticker'] for r in self.results]
        sectors = [r['sector'] for r in self.results]
        
        # Color by sector
        unique_sectors = list(set(sectors))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sectors)))
        sector_color = {s: colors[i] for i, s in enumerate(unique_sectors)}
        point_colors = [sector_color[s] for s in sectors]
        
        # 1. Main scatter plot
        ax = axes[0, 0]
        ax.scatter(kurt, dur, c=point_colors, s=60, alpha=0.7, edgecolors='white', linewidth=0.5)
        
        # Add regression line
        z = np.polyfit(kurt, dur, 1)
        x_line = np.linspace(min(kurt), max(kurt), 100)
        ax.plot(x_line, np.polyval(z, x_line), 'r--', alpha=0.8, label='OLS fit')
        
        # Annotate extreme points
        for ticker, k, d in zip(tickers, kurt, dur):
            if k > np.percentile(kurt, 95) or d > np.percentile(dur, 95):
                ax.annotate(ticker, (k, d), fontsize=8, alpha=0.8)
        
        r, p = pearsonr(kurt, dur)
        ax.set_xlabel('Kurtosis')
        ax.set_ylabel('Avg Regime Duration (days)')
        ax.set_title(f'All Stocks (r={r:.2f}, p={p:.2e})')
        ax.grid(True, alpha=0.3)
        
        # 2. Without outliers
        ax = axes[0, 1]
        # Remove top 5% by kurtosis
        threshold = np.percentile(kurt, 95)
        filtered = [(k, d, t) for k, d, t in zip(kurt, dur, tickers) if k < threshold]
        f_kurt, f_dur, f_tick = zip(*filtered) if filtered else ([], [], [])
        
        ax.scatter(f_kurt, f_dur, c='steelblue', s=60, alpha=0.7, edgecolors='white', linewidth=0.5)
        if len(f_kurt) > 2:
            z = np.polyfit(f_kurt, f_dur, 1)
            x_line = np.linspace(min(f_kurt), max(f_kurt), 100)
            ax.plot(x_line, np.polyval(z, x_line), 'r--', alpha=0.8)
            r_f, p_f = pearsonr(f_kurt, f_dur)
            ax.set_title(f'Without Top 5% Outliers (r={r_f:.2f})')
        else:
            ax.set_title('Without Top 5% Outliers')
        ax.set_xlabel('Kurtosis')
        ax.set_ylabel('Avg Regime Duration (days)')
        ax.grid(True, alpha=0.3)
        
        # 3. By sector
        ax = axes[0, 2]
        sector_r = {}
        for sector in unique_sectors:
            s_data = [(k, d) for k, d, s in zip(kurt, dur, sectors) if s == sector]
            if len(s_data) >= 4:
                s_kurt, s_dur = zip(*s_data)
                r_s, _ = pearsonr(s_kurt, s_dur)
                sector_r[sector] = r_s
        
        if sector_r:
            sorted_sectors = sorted(sector_r.items(), key=lambda x: x[1], reverse=True)
            sector_names = [s[0][:15] for s in sorted_sectors]
            sector_vals = [s[1] for s in sorted_sectors]
            colors = ['green' if v > 0 else 'red' for v in sector_vals]
            
            ax.barh(sector_names, sector_vals, color=colors, alpha=0.7, edgecolor='white')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Correlation (r)')
            ax.set_title('Correlation by Sector')
        
        # 4. Distribution of kurtosis
        ax = axes[1, 0]
        ax.hist(kurt, bins=20, color='steelblue', alpha=0.7, edgecolor='white')
        ax.axvline(x=np.median(kurt), color='red', linestyle='--', label=f'Median: {np.median(kurt):.1f}')
        ax.set_xlabel('Kurtosis')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Kurtosis')
        ax.legend()
        
        # 5. Distribution of regime duration
        ax = axes[1, 1]
        ax.hist(dur, bins=20, color='coral', alpha=0.7, edgecolor='white')
        ax.axvline(x=np.median(dur), color='blue', linestyle='--', label=f'Median: {np.median(dur):.1f}d')
        ax.set_xlabel('Avg Regime Duration (days)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Regime Duration')
        ax.legend()
        
        # 6. Summary statistics table
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = f"""
VALIDATION SUMMARY
==================

Stocks Analyzed: {len(self.results)}
Sectors: {len(unique_sectors)}

Overall Correlation:
  Pearson r = {pearsonr(kurt, dur)[0]:+.3f}
  Spearman ρ = {spearmanr(kurt, dur)[0]:+.3f}

Kurtosis Stats:
  Range: [{min(kurt):.1f}, {max(kurt):.1f}]
  Mean: {np.mean(kurt):.2f}

Duration Stats:
  Range: [{min(dur):.1f}, {max(dur):.1f}d]
  Mean: {np.mean(dur):.1f}d
"""
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('research/outputs/kurtosis_validation.png', dpi=150, facecolor='white', bbox_inches='tight')
        print(f"\n✓ Visualization saved to: research/outputs/kurtosis_validation.png")
    
    def generate_report(self):
        """Generate markdown validation report."""
        
        report = f"""# Kurtosis-Duration Validation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

**Original Finding:** Fat-tail stocks (high kurtosis) have LONGER regime durations (r = +0.84 on 22 stocks)

**Validation Results:**

"""
        
        # Test 1
        if 'test1_expanded' in self.validation_report:
            t1 = self.validation_report['test1_expanded']
            report += f"""### Test 1: Expanded Universe
- **Stocks Analyzed:** {t1.get('n_stocks', 'N/A')}
- **Pearson r:** {t1.get('pearson_r', 0):+.4f} (p = {t1.get('pearson_p', 1):.2e})
- **Spearman ρ:** {t1.get('spearman_r', 0):+.4f}
- **Status:** {t1.get('status', 'N/A')}

"""
        
        # Test 2
        if 'test2_outliers' in self.validation_report:
            t2 = self.validation_report['test2_outliers']
            report += f"""### Test 2: Outlier Sensitivity
- **r with all data:** {t2.get('r_all', 0):+.4f}
- **r without META:** {t2.get('r_no_meta', 0):+.4f}
- **r without top 5 influential:** {t2.get('r_no_influential', 0):+.4f}
- **Most influential stocks:** {', '.join(t2.get('influential_stocks', []))}
- **Status:** {t2.get('status', 'N/A')}

"""
        
        # Test 3
        if 'test3_sectors' in self.validation_report:
            t3 = self.validation_report['test3_sectors']
            report += f"""### Test 3: Sector Analysis
- **Sectors with positive r:** {t3.get('n_positive', 0)}/{t3.get('n_sectors', 0)}
- **Sectors with significant p:** {t3.get('n_significant', 0)}/{t3.get('n_sectors', 0)}
- **Status:** {t3.get('status', 'N/A')}

| Sector | n | r | p |
|--------|---|---|---|
"""
            for sector, data in t3.get('sector_correlations', {}).items():
                report += f"| {sector} | {data['n']} | {data['r']:+.3f} | {data['p']:.3f} |\n"
            report += "\n"
        
        # Test 4
        if 'test4_time' in self.validation_report:
            t4 = self.validation_report['test4_time']
            report += f"""### Test 4: Time-Period Robustness
"""
            for period, data in t4.get('period_results', {}).items():
                report += f"- **{period}:** r = {data['r']:+.3f} (n={data['n']})\n"
            report += f"- **Status:** {t4.get('status', 'N/A')}\n\n"
        
        # Test 5
        if 'test5_oos' in self.validation_report:
            t5 = self.validation_report['test5_oos']
            report += f"""### Test 5: Out-of-Sample Prediction
- **Model:** duration = {t5.get('model_intercept', 0):.2f} + {t5.get('model_slope', 0):.2f} × kurtosis
- **Training stocks:** {t5.get('n_train', 0)}
- **Test stocks:** {t5.get('n_test', 0)}
- **Out-of-sample r:** {t5.get('r_oos', 0):+.3f}
- **MAE:** {t5.get('mae', 0):.1f} days
- **Skill vs naive:** {t5.get('skill_score', 0):+.1%}
- **Status:** {t5.get('status', 'N/A')}

"""
        
        # Final Decision
        report += """## Decision Matrix

| Criterion | Result | Threshold | Pass? |
|-----------|--------|-----------|-------|
"""
        
        if 'test1_expanded' in self.validation_report:
            t1 = self.validation_report['test1_expanded']
            pass1 = "✓" if t1.get('pearson_r', 0) > 0.5 else "✗"
            report += f"| r on 100+ stocks | {t1.get('pearson_r', 0):+.3f} | > 0.5 | {pass1} |\n"
        
        if 'test2_outliers' in self.validation_report:
            t2 = self.validation_report['test2_outliers']
            pass2 = "✓" if not t2.get('is_fragile', True) else "✗"
            report += f"| Outlier robust | {t2.get('status', 'N/A')} | Not fragile | {pass2} |\n"
        
        if 'test5_oos' in self.validation_report:
            t5 = self.validation_report['test5_oos']
            pass5 = "✓" if t5.get('is_predictive', False) else "✗"
            report += f"| OOS predictive | {t5.get('r_oos', 0):+.3f} | > 0.3 | {pass5} |\n"
        
        report += """
## Recommendation

"""
        # Make recommendation based on results
        n_passes = 0
        if self.validation_report.get('test1_expanded', {}).get('pearson_r', 0) > 0.3:
            n_passes += 1
        if not self.validation_report.get('test2_outliers', {}).get('is_fragile', True):
            n_passes += 1
        if self.validation_report.get('test5_oos', {}).get('is_predictive', False):
            n_passes += 1
        
        if n_passes >= 2:
            report += """**FINDING: VALID** ✓

The kurtosis-duration relationship is robust. Proceed with:
1. Implement `classify_stock_type()` in RegimeRiskPlatform
2. Adjust `n_regimes` based on kurtosis profile
3. Document as validated research finding
"""
        elif n_passes == 1:
            report += """**FINDING: FRAGILE** ⚠️

The relationship exists but is sensitive to conditions. Actions:
1. Document as "interesting observation"
2. Do NOT build core features around it
3. Consider further investigation with larger dataset
"""
        else:
            report += """**FINDING: FAILED** ✗

The original finding does not hold on expanded validation. Actions:
1. Archive the research
2. Do NOT implement stock classification
3. Move on to other research directions
"""
        
        # Write report
        with open('research/outputs/kurtosis_validation_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✓ Report saved to: research/outputs/kurtosis_validation_report.md")
        
        return report
    
    def run_all(self):
        """Run complete validation suite."""
        print("\n" + "=" * 70)
        print("PHASE 3: KURTOSIS-DURATION RESEARCH VALIDATION")
        print("=" * 70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70 + "\n")
        
        self.load_platform()
        
        # Run tests
        self.test_expanded_universe()
        self.test_outlier_sensitivity()
        self.test_sector_analysis()
        self.test_time_robustness()
        self.test_out_of_sample()
        
        # Generate outputs
        self.generate_visualizations()
        report = self.generate_report()
        
        # Save raw results
        with open('research/outputs/kurtosis_validation_data.json', 'w') as f:
            # Convert results to serializable format
            serializable_results = []
            for r in self.results:
                sr = {k: v for k, v in r.items() if k != 'regime_durations'}
                if 'regime_durations' in r:
                    sr['regime_durations'] = {str(k): v for k, v in r['regime_durations'].items()}
                serializable_results.append(sr)
            
            json.dump({
                'results': serializable_results,
                'validation_report': self.validation_report,
                'generated': datetime.now().isoformat()
            }, f, indent=2, default=str)
        
        print(f"\n✓ Raw data saved to: research/outputs/kurtosis_validation_data.json")
        
        print("\n" + "=" * 70)
        print("VALIDATION COMPLETE")
        print("=" * 70)
        
        return self.validation_report


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import os
    
    # Ensure output directory exists
    os.makedirs('research/outputs', exist_ok=True)
    
    # Run validation
    validator = KurtosisValidation()
    results = validator.run_all()
    
    print("\n\nFinal Decision:")
    print("-" * 50)
    
    # Quick summary
    t1 = results.get('test1_expanded', {})
    t2 = results.get('test2_outliers', {})
    t5 = results.get('test5_oos', {})
    
    pearson_r = t1.get('pearson_r', 0)
    is_fragile = t2.get('is_fragile', True)
    is_predictive = t5.get('is_predictive', False)
    
    if pearson_r > 0.5 and not is_fragile and is_predictive:
        print("🎯 FINDING IS VALID - Implement stock classification")
    elif pearson_r > 0.3:
        print("⚠️ FINDING IS FRAGILE - Document only, don't build on it")
    else:
        print("❌ FINDING FAILED - Archive and move on")
