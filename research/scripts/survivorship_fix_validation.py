"""
SURVIVORSHIP BIAS FIX + MULTI-PERIOD VALIDATION
=================================================

Finance Expert's Critical Feedback:
1. [FIX] Survivorship bias - Using 2024 winners for 2015 test
2. [FIX] Sharpe calculation inconsistency - Monthly vs Daily
3. [FIX] Missing drawdown metrics
4. [FIX] Single period test insufficient

This script fixes all critical flaws and re-validates.

Author: Project Iris Research Team
Date: 2025-12-31
Status: Implementing Finance Expert's corrections
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# HISTORICAL CONSTITUENTS - FIX SURVIVORSHIP BIAS
# ============================================================================

# Top 10 S&P 500 by market cap in January of each year
# Source: Historical S&P 500 data
MEGA_CAPS_BY_YEAR = {
    2010: ['XOM', 'MSFT', 'AAPL', 'GE', 'WMT', 
           'CVX', 'JNJ', 'PG', 'JPM', 'IBM'],
    2011: ['XOM', 'AAPL', 'MSFT', 'GE', 'CVX', 
           'IBM', 'WMT', 'JNJ', 'PG', 'JPM'],
    2012: ['AAPL', 'XOM', 'MSFT', 'IBM', 'CVX', 
           'GE', 'WMT', 'PG', 'JNJ', 'JPM'],
    2013: ['AAPL', 'XOM', 'MSFT', 'GE', 'CVX', 
           'WMT', 'IBM', 'JNJ', 'PG', 'JPM'],
    2014: ['AAPL', 'XOM', 'MSFT', 'GOOGL', 'GE', 
           'JNJ', 'CVX', 'WMT', 'JPM', 'PG'],
    2015: ['AAPL', 'GOOGL', 'XOM', 'MSFT', 'BRK-B', 
           'JNJ', 'GE', 'WMT', 'JPM', 'CVX'],
    2016: ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'XOM',
           'FB', 'JNJ', 'GE', 'BRK-B', 'JPM'],
    2017: ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB',
           'BRK-B', 'JNJ', 'JPM', 'XOM', 'V'],
    2018: ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB',
           'BRK-B', 'JPM', 'JNJ', 'V', 'XOM'],
    2019: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB',
           'BRK-B', 'JPM', 'JNJ', 'V', 'UNH'],
    2020: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB',
           'BRK-B', 'V', 'JPM', 'JNJ', 'WMT'],
    2021: ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB',
           'TSLA', 'BRK-B', 'NVDA', 'JPM', 'V'],
    2022: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
           'BRK-B', 'NVDA', 'FB', 'UNH', 'JPM'],
    2023: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'BRK-B',
           'NVDA', 'META', 'TSLA', 'UNH', 'XOM'],
    2024: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
           'META', 'BRK-B', 'TSLA', 'LLY', 'V'],
}


def get_constituents_for_period(start_year: int) -> List[str]:
    """
    Get historical top 10 mega-caps as of the START of the period.
    This eliminates survivorship bias.
    """
    if start_year in MEGA_CAPS_BY_YEAR:
        return MEGA_CAPS_BY_YEAR[start_year]
    else:
        # Find closest year
        years = sorted(MEGA_CAPS_BY_YEAR.keys())
        closest = min(years, key=lambda x: abs(x - start_year))
        return MEGA_CAPS_BY_YEAR[closest]


# ============================================================================
# FIXED REGIME DETECTION - Use Return Momentum, Not Sharpe
# ============================================================================

def detect_regime_fixed(returns: pd.Series, lookback: int = 60) -> Tuple[str, float]:
    """
    Finance Expert corrected regime detection.
    
    Uses annualized return thresholds instead of Sharpe:
    - Bull: Annualized return > 10%
    - Bear: Annualized return < -5%
    - Neutral: else
    """
    if len(returns) < lookback:
        return 'Neutral', 0.5
    
    recent = returns.tail(lookback)
    
    # Annualized return
    ann_return = recent.mean() * 252
    ann_vol = recent.std() * np.sqrt(252)
    
    # Finance expert's thresholds
    if ann_return > 0.10:  # >10% annual
        return 'Bull', min(0.9, 0.5 + ann_return)
    elif ann_return < -0.05:  # < -5% annual
        return 'Bear', min(0.9, 0.5 - ann_return)
    else:
        return 'Neutral', 0.5


# ============================================================================
# FIXED DRAWDOWN CALCULATION
# ============================================================================

def calculate_drawdowns(portfolio_values: List[float]) -> Dict:
    """
    Calculate drawdown metrics (expert requirement).
    """
    cumulative = pd.Series(portfolio_values)
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    # Calculate drawdown duration (consecutive days below peak)
    below_peak = drawdown < 0
    duration_groups = (below_peak != below_peak.shift()).cumsum()
    durations = below_peak.groupby(duration_groups).cumsum()
    
    return {
        'max_drawdown': drawdown.min(),
        'avg_drawdown': drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0,
        'max_duration': durations.max(),
        'drawdown_series': drawdown,
    }


# ============================================================================
# FIXED SHARPE CALCULATION - Consistent Daily Returns
# ============================================================================

def calculate_sharpe_daily(portfolio_values: List[float], dates: pd.DatetimeIndex) -> float:
    """
    Calculate Sharpe using DAILY returns (Finance Expert requirement).
    
    This ensures consistency with SPY comparison.
    """
    # Convert to daily returns
    port_series = pd.Series(portfolio_values, index=dates[:len(portfolio_values)])
    
    # Resample to daily and forward-fill
    daily_values = port_series.resample('D').ffill().dropna()
    daily_returns = daily_values.pct_change().dropna()
    
    if len(daily_returns) < 30:
        return 0.0
    
    # Annualize
    ann_return = (1 + daily_returns).prod() ** (252 / len(daily_returns)) - 1
    ann_vol = daily_returns.std() * np.sqrt(252)
    
    return ann_return / ann_vol if ann_vol > 0 else 0


# ============================================================================
# CORRECTED BACKTEST
# ============================================================================

def corrected_backtest(
    start_year: int,
    end_year: int,
    transaction_cost_bps: float = 5,  # More realistic for mega-caps
) -> Dict:
    """
    Corrected backtest with:
    1. Historical constituents (no survivorship bias)
    2. Fixed regime detection (return momentum)
    3. Daily Sharpe calculation
    4. Drawdown metrics
    """
    print(f"\n{'='*70}")
    print(f"CORRECTED BACKTEST: {start_year} to {end_year}")
    print(f"Using historical constituents from {start_year}")
    print(f"{'='*70}")
    
    # Get historical constituents
    tickers = get_constituents_for_period(start_year)
    print(f"\nConstituents (as of Jan {start_year}):")
    print(f"  {', '.join(tickers)}")
    
    # Download data
    print("\nFetching data...")
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    
    prices = pd.DataFrame()
    valid_tickers = []
    
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(data) > 100:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
                prices[ticker] = data[price_col]
                valid_tickers.append(ticker)
        except Exception as e:
            print(f"  Warning: Could not load {ticker}: {e}")
    
    prices = prices.dropna()
    returns = np.log(prices / prices.shift(1)).dropna()
    
    print(f"Loaded {len(prices)} days for {len(valid_tickers)} stocks")
    
    # Monthly rebalancing
    rebal_dates = prices.resample('M').last().index
    
    # Safe lookup
    def safe_lookup(df, date, col=None):
        valid = df.index[df.index <= date]
        if len(valid) == 0:
            return df[col].iloc[0] if col else df.iloc[0]
        return df.loc[valid[-1], col] if col else df.loc[valid[-1]]
    
    # Track portfolio
    portfolio_value = 100000
    portfolio_history = [portfolio_value]
    portfolio_dates = [prices.index[0]]
    holdings = {ticker: 0 for ticker in valid_tickers}
    cash = portfolio_value
    
    total_trades = 0
    total_costs = 0
    
    # Backtest loop
    for i in range(1, len(rebal_dates)):
        rebal_start = rebal_dates[i-1]
        rebal_end = rebal_dates[i]
        
        if rebal_end > prices.index[-1]:
            continue
        
        # Calculate target weights using FIXED detection
        target_weights = {}
        
        for ticker in valid_tickers:
            ticker_returns = returns[ticker].loc[:rebal_start]
            
            # Use corrected regime detection
            regime, prob = detect_regime_fixed(ticker_returns)
            
            if regime == 'Bull':
                target_weights[ticker] = 1.0 / len(valid_tickers)
            else:
                target_weights[ticker] = 0.0
        
        # Normalize weights
        total_weight = sum(target_weights.values())
        if total_weight > 1.0:
            for t in target_weights:
                target_weights[t] /= total_weight
        
        # Calculate current value
        current_value = cash
        for ticker in valid_tickers:
            current_value += holdings[ticker] * safe_lookup(prices, rebal_start, ticker)
        
        # Execute trades
        for ticker in valid_tickers:
            price = safe_lookup(prices, rebal_start, ticker)
            current_pos = holdings[ticker] * price
            target_pos = current_value * target_weights.get(ticker, 0)
            
            trade_value = abs(target_pos - current_pos)
            
            if trade_value > 100:
                cost = trade_value * (transaction_cost_bps / 10000)
                total_costs += cost
                total_trades += 1
                
                if target_pos > current_pos:
                    shares = (target_pos - current_pos - cost) / price
                    holdings[ticker] += shares
                    cash -= (target_pos - current_pos)
                else:
                    shares = (current_pos - target_pos) / price
                    holdings[ticker] -= shares
                    cash += (current_pos - target_pos - cost)
        
        # Mark to market
        end_value = cash
        for ticker in valid_tickers:
            end_value += holdings[ticker] * safe_lookup(prices, rebal_end, ticker)
        
        portfolio_history.append(end_value)
        portfolio_dates.append(rebal_end)
    
    # Calculate SPY benchmark
    spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False)
    if isinstance(spy_data.columns, pd.MultiIndex):
        spy_data.columns = spy_data.columns.get_level_values(0)
    spy_col = 'Adj Close' if 'Adj Close' in spy_data.columns else 'Close'
    spy_returns = spy_data[spy_col].pct_change().dropna()
    spy_final = 100000 * (1 + spy_returns).cumprod().iloc[-1]
    spy_total = (spy_final / 100000) - 1
    spy_ann = (1 + spy_total) ** (252 / len(spy_returns)) - 1
    spy_vol = spy_returns.std() * np.sqrt(252)
    spy_sharpe = spy_ann / spy_vol if spy_vol > 0 else 0
    
    # Calculate strategy metrics (FIXED: Daily Sharpe)
    strategy_total = (portfolio_history[-1] / portfolio_history[0]) - 1
    strategy_sharpe = calculate_sharpe_daily(portfolio_history, pd.DatetimeIndex(portfolio_dates))
    
    # Drawdown calculation (NEW)
    drawdown_stats = calculate_drawdowns(portfolio_history)
    
    # Simple annual return estimate
    years = (end_year - start_year) or 1
    strategy_annual = (1 + strategy_total) ** (1 / years) - 1
    
    # Print results
    print("\n" + "-"*50)
    print("CORRECTED RESULTS:")
    print("-"*50)
    
    print(f"\n{'Metric':<30} {'Strategy':>15} {'SPY':>15}")
    print("-"*60)
    print(f"{'Total Return':<30} {strategy_total*100:>14.1f}% {spy_total*100:>14.1f}%")
    print(f"{'Annualized Return':<30} {strategy_annual*100:>14.1f}% {spy_ann*100:>14.1f}%")
    print(f"{'Sharpe Ratio (Daily)':<30} {strategy_sharpe:>15.2f} {spy_sharpe:>15.2f}")
    print(f"{'Max Drawdown':<30} {drawdown_stats['max_drawdown']*100:>14.1f}% {'N/A':>15}")
    
    print(f"\n{'Trading Activity':}")
    print(f"  Total trades: {total_trades}")
    print(f"  Total costs: ${total_costs:,.0f}")
    
    # Verdict
    print("\n" + "-"*50)
    if strategy_sharpe >= 1.0:
        print(f"[PASS] Sharpe {strategy_sharpe:.2f} >= 1.0")
    else:
        print(f"[FAIL] Sharpe {strategy_sharpe:.2f} < 1.0")
    
    if drawdown_stats['max_drawdown'] >= -0.25:
        print(f"[PASS] Max DD {drawdown_stats['max_drawdown']*100:.1f}% >= -25%")
    else:
        print(f"[FAIL] Max DD {drawdown_stats['max_drawdown']*100:.1f}% < -25%")
    
    return {
        'period': f"{start_year}-{end_year}",
        'constituents': valid_tickers,
        'strategy_return': strategy_total,
        'strategy_sharpe': strategy_sharpe,
        'spy_sharpe': spy_sharpe,
        'max_drawdown': drawdown_stats['max_drawdown'],
        'total_trades': total_trades,
        'portfolio_history': portfolio_history,
    }


# ============================================================================
# MULTI-PERIOD TEST (Finance Expert Requirement)
# ============================================================================

def run_multi_period_test():
    """
    Run on 3 different periods as required by Finance Expert.
    
    Requirement: Strategy must work in at least 2 of 3 periods.
    """
    print("="*70)
    print("MULTI-PERIOD VALIDATION (Survivorship Bias Fixed)")
    print("Requirement: Sharpe > 1.0 in at least 2 of 3 periods")
    print("="*70)
    
    # Three distinct periods
    periods = [
        (2010, 2014),  # Post-GFC recovery
        (2015, 2019),  # Low-vol bull
        (2020, 2024),  # COVID era
    ]
    
    results = {}
    passes = 0
    
    for start, end in periods:
        period_key = f"{start}-{end}"
        results[period_key] = corrected_backtest(start, end)
        
        if results[period_key]['strategy_sharpe'] >= 1.0:
            passes += 1
    
    # Summary
    print("\n" + "="*70)
    print("MULTI-PERIOD SUMMARY")
    print("="*70)
    
    print(f"\n{'Period':<15} {'Sharpe':>10} {'Max DD':>12} {'vs SPY':>12} {'Status':>10}")
    print("-"*60)
    
    for period, res in results.items():
        vs_spy = res['strategy_sharpe'] - res['spy_sharpe']
        status = "[PASS]" if res['strategy_sharpe'] >= 1.0 else "[FAIL]"
        print(f"{period:<15} {res['strategy_sharpe']:>10.2f} {res['max_drawdown']*100:>11.1f}% {vs_spy:>+11.2f} {status:>10}")
    
    # Final verdict
    print("\n" + "-"*50)
    print(f"PERIODS PASSED: {passes} of {len(periods)}")
    
    if passes >= 2:
        print("[SUCCESS] STRATEGY VALIDATED: Works in multiple periods")
        print("          Ready for production with corrections applied")
    else:
        print("[FAILED] STRATEGY FAILED: Does not generalize across periods")
        print("         May be overfitted to specific market conditions")
    
    # Survivorship comparison
    print("\n" + "="*70)
    print("SURVIVORSHIP BIAS IMPACT")
    print("="*70)
    
    print("""
    BEFORE FIX (using 2024 winners in all periods):
      - 2020-2024 Sharpe: 1.40 (inflated)
    
    AFTER FIX (using historical constituents):
      - Results above use point-in-time holdings
      - GE, XOM, IBM included in early periods (they were mega-caps then)
      - NVDA, TSLA excluded in early periods (weren't mega-caps)
    
    If Sharpe dropped significantly, this confirms survivorship bias
    was inflating our original results.
    """)
    
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    results = run_multi_period_test()
