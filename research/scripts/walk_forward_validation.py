"""
WALK-FORWARD VALIDATION
========================

The ONLY test that matters for institutional credibility.

METHODOLOGY:
1. TRAIN: 2015-2019 (pre-COVID) → Learn sector regime duration statistics
2. FREEZE: Lock the statistics, no lookahead bias
3. TEST:  2020-2024 (COVID+) → Apply strategy with frozen stats

SUCCESS CRITERIA:
- OOS Sharpe > 1.5 = AMAZING (product ready)
- OOS Sharpe > 1.3 = GOOD (commercially viable)
- OOS Sharpe > 1.0 = ACCEPTABLE (needs refinement)
- OOS Sharpe < 1.0 = PROBLEM (overfitted)

Author: Project Iris Research Team
Date: 2025-12-31
Finance Expert Approved: Pending walk-forward results
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class RegimeDurationStats:
    """Statistics for regime duration."""
    mean: float
    std: float
    count: int


class RegimePhase(Enum):
    EARLY = 'early'
    MIDDLE = 'middle'
    LATE = 'late'
    VERY_LATE = 'very_late'


# ============================================================================
# STEP 1: CALCULATE SECTOR STATS FROM TRAINING PERIOD ONLY
# ============================================================================

def detect_regime_periods(returns: pd.Series, lookback: int = 60) -> List[Dict]:
    """
    Detect regime periods and their durations.
    
    Returns list of regime periods with start/end dates and durations.
    """
    if len(returns) < lookback:
        return []
    
    regimes = []
    current_regime = None
    regime_start = None
    
    for i in range(lookback, len(returns)):
        recent = returns.iloc[i-lookback:i]
        
        # Calculate Sharpe
        ret = recent.mean() * 252
        vol = recent.std() * np.sqrt(252)
        sharpe = ret / vol if vol > 0 else 0
        
        # Classify
        if sharpe > 0.3:
            regime = 'Bull'
        elif sharpe < -0.3:
            regime = 'Bear'
        else:
            regime = 'Neutral'
        
        if regime != current_regime:
            # Save previous regime
            if current_regime is not None:
                regimes.append({
                    'regime': current_regime,
                    'start_idx': regime_start,
                    'end_idx': i,
                    'duration': i - regime_start,
                })
            
            current_regime = regime
            regime_start = i
    
    # Save final regime
    if current_regime is not None and regime_start is not None:
        regimes.append({
            'regime': current_regime,
            'start_idx': regime_start,
            'end_idx': len(returns),
            'duration': len(returns) - regime_start,
        })
    
    return regimes


def calculate_sector_stats_from_training(
    tickers: List[str],
    train_start: str,
    train_end: str,
) -> Dict[str, Dict[str, RegimeDurationStats]]:
    """
    Calculate sector regime duration statistics from TRAINING period only.
    
    This creates the frozen stats for out-of-sample testing.
    """
    print(f"\nCalculating regime stats from {train_start} to {train_end}...")
    
    # Stock to sector mapping
    stock_sector = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
        'AMZN': 'Consumer', 'META': 'Technology', 'NVDA': 'Technology',
        'JPM': 'Financials', 'V': 'Financials',
        'JNJ': 'Healthcare', 'UNH': 'Healthcare',
        'XOM': 'Energy', 'WMT': 'Consumer',
    }
    
    # Collect regime durations by sector
    sector_regime_durations = {}
    
    for ticker in tickers:
        sector = stock_sector.get(ticker, 'Default')
        
        # Download training data
        data = yf.download(ticker, start=train_start, end=train_end, progress=False)
        if len(data) < 100:
            continue
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
        returns = np.log(data[price_col] / data[price_col].shift(1)).dropna()
        
        # Detect regime periods
        periods = detect_regime_periods(returns)
        
        # Collect by sector and regime type
        if sector not in sector_regime_durations:
            sector_regime_durations[sector] = {'Bull': [], 'Bear': [], 'Neutral': []}
        
        for period in periods:
            regime_type = period['regime']
            duration = period['duration']
            sector_regime_durations[sector][regime_type].append(duration)
    
    # Calculate statistics
    sector_stats = {}
    
    for sector, regime_data in sector_regime_durations.items():
        sector_stats[sector] = {}
        
        for regime_type, durations in regime_data.items():
            if len(durations) > 0:
                sector_stats[sector][regime_type] = RegimeDurationStats(
                    mean=np.mean(durations),
                    std=np.std(durations),
                    count=len(durations),
                )
            else:
                # Default if no data
                sector_stats[sector][regime_type] = RegimeDurationStats(
                    mean=25, std=10, count=0
                )
    
    # Print summary
    print("\n📊 TRAINING PERIOD SECTOR STATISTICS:")
    print("-"*60)
    for sector, regimes in sector_stats.items():
        print(f"\n{sector}:")
        for regime, stats in regimes.items():
            print(f"  {regime}: mean={stats.mean:.1f}d, std={stats.std:.1f}d (n={stats.count})")
    
    return sector_stats


# ============================================================================
# STEP 2: BACKTEST WITH FROZEN STATS (NO LOOKAHEAD BIAS)
# ============================================================================

def get_regime_phase_with_frozen_stats(
    sector: str,
    regime: str,
    days_in_regime: int,
    frozen_stats: Dict[str, Dict[str, RegimeDurationStats]],
) -> Tuple[RegimePhase, float]:
    """
    Determine regime phase using FROZEN training period stats.
    
    This ensures no lookahead bias in the test.
    """
    # Get stats from frozen training data
    if sector in frozen_stats and regime in frozen_stats[sector]:
        stats = frozen_stats[sector][regime]
        expected_duration = stats.mean
    else:
        expected_duration = 25  # Default
    
    # Calculate phase
    progress = days_in_regime / expected_duration if expected_duration > 0 else 1.0
    
    if progress < 0.3:
        return RegimePhase.EARLY, 1.2
    elif progress < 1.0:
        return RegimePhase.MIDDLE, 1.0
    elif progress < 1.5:
        return RegimePhase.LATE, 0.5
    else:
        return RegimePhase.VERY_LATE, 0.0


def walk_forward_backtest(
    tickers: List[str],
    test_start: str,
    test_end: str,
    frozen_stats: Dict[str, Dict[str, RegimeDurationStats]],
    use_lifecycle: bool = True,
    transaction_cost_bps: float = 10,
) -> Dict:
    """
    Run backtest on TEST period using FROZEN training stats.
    
    This is the out-of-sample test that validates the strategy.
    """
    mode = "LIFECYCLE (with frozen stats)" if use_lifecycle else "BINARY"
    
    print(f"\n{'='*70}")
    print(f"OUT-OF-SAMPLE BACKTEST: {test_start} to {test_end}")
    print(f"Mode: {mode}")
    print(f"{'='*70}")
    
    # Stock to sector mapping
    stock_sector = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
        'AMZN': 'Consumer', 'META': 'Technology', 'NVDA': 'Technology',
        'JPM': 'Financials', 'V': 'Financials',
        'JNJ': 'Healthcare', 'UNH': 'Healthcare',
        'XOM': 'Energy', 'WMT': 'Consumer',
    }
    
    # Download test period data
    print("\nFetching test period data...")
    prices = pd.DataFrame()
    
    for ticker in tickers:
        data = yf.download(ticker, start=test_start, end=test_end, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
        prices[ticker] = data[price_col]
    
    prices = prices.dropna()
    returns = np.log(prices / prices.shift(1)).dropna()
    
    print(f"Loaded {len(prices)} days of test data")
    
    # Monthly rebalancing
    rebal_dates = prices.resample('M').last().index
    
    # Safe lookup
    def safe_lookup(df, date, col=None):
        valid = df.index[df.index <= date]
        if len(valid) == 0:
            return df[col].iloc[0] if col else df.iloc[0]
        return df.loc[valid[-1], col] if col else df.loc[valid[-1]]
    
    # Portfolio tracking
    portfolio_value = 100000
    portfolio_history = [portfolio_value]
    holdings = {ticker: 0 for ticker in tickers}
    cash = portfolio_value
    
    total_trades = 0
    total_costs = 0
    phase_counts = {'early': 0, 'middle': 0, 'late': 0, 'very_late': 0}
    
    # Backtest loop
    for i in range(1, len(rebal_dates)):
        rebal_start = rebal_dates[i-1]
        rebal_end = rebal_dates[i]
        
        if rebal_end > prices.index[-1]:
            continue
        
        # Calculate target weights
        target_weights = {}
        
        for ticker in tickers:
            ticker_returns = returns[ticker].loc[:rebal_start]
            sector = stock_sector.get(ticker, 'Default')
            
            # Detect regime
            if len(ticker_returns) < 60:
                regime, prob = 'Neutral', 0.5
                days_in_regime = 0
            else:
                recent = ticker_returns.tail(60)
                ret = recent.mean() * 252
                vol = recent.std() * np.sqrt(252)
                sharpe = ret / vol if vol > 0 else 0
                
                if sharpe > 0.3:
                    regime = 'Bull'
                    prob = min(0.9, 0.5 + sharpe * 0.25)
                elif sharpe < -0.3:
                    regime = 'Bear'
                    prob = min(0.9, 0.5 - sharpe * 0.25)
                else:
                    regime = 'Neutral'
                    prob = 0.6
                
                # Estimate days in regime
                days_in_regime = min(60, len(ticker_returns) // 5)  # Simplified
            
            if use_lifecycle:
                # Get phase using FROZEN stats (no lookahead!)
                phase, position_mult = get_regime_phase_with_frozen_stats(
                    sector=sector,
                    regime=regime,
                    days_in_regime=days_in_regime,
                    frozen_stats=frozen_stats,
                )
                
                phase_counts[phase.value] += 1
                
                if regime == 'Bull':
                    base_weight = 1.0 / len(tickers)
                    target_weights[ticker] = base_weight * position_mult
                else:
                    target_weights[ticker] = 0.0
            else:
                # Binary
                if regime == 'Bull' and prob > 0.5:
                    target_weights[ticker] = 1.0 / len(tickers)
                else:
                    target_weights[ticker] = 0.0
        
        # Normalize weights
        total_weight = sum(target_weights.values())
        if total_weight > 1.0:
            for t in target_weights:
                target_weights[t] /= total_weight
        
        # Execute trades
        current_value = cash
        for ticker in tickers:
            price = safe_lookup(prices, rebal_start, ticker)
            current_value += holdings[ticker] * price
        
        for ticker in tickers:
            price = safe_lookup(prices, rebal_start, ticker)
            current_pos_value = holdings[ticker] * price
            target_pos_value = current_value * target_weights.get(ticker, 0)
            
            trade_value = abs(target_pos_value - current_pos_value)
            
            if trade_value > 100:
                cost = trade_value * (transaction_cost_bps / 10000) * 0.6
                total_costs += cost
                total_trades += 1
                
                if target_pos_value > current_pos_value:
                    shares = (target_pos_value - current_pos_value - cost) / price
                    holdings[ticker] += shares
                    cash -= (target_pos_value - current_pos_value)
                else:
                    shares = (current_pos_value - target_pos_value) / price
                    holdings[ticker] -= shares
                    cash += (current_pos_value - target_pos_value - cost)
        
        # Mark to market
        end_value = cash
        for ticker in tickers:
            price = safe_lookup(prices, rebal_end, ticker)
            end_value += holdings[ticker] * price
        
        portfolio_history.append(end_value)
    
    # Calculate benchmarks
    equal_returns = prices.pct_change().mean(axis=1).dropna()
    benchmark_cum = (1 + equal_returns).cumprod()
    benchmark_final = 100000 * benchmark_cum.iloc[-1]
    
    spy_data = yf.download('SPY', start=test_start, end=test_end, progress=False)
    if isinstance(spy_data.columns, pd.MultiIndex):
        spy_data.columns = spy_data.columns.get_level_values(0)
    spy_col = 'Adj Close' if 'Adj Close' in spy_data.columns else 'Close'
    spy_returns = spy_data[spy_col].pct_change().dropna()
    spy_cum = (1 + spy_returns).cumprod()
    spy_final = 100000 * spy_cum.iloc[-1]
    
    # Statistics
    port_returns = pd.Series(portfolio_history).pct_change().dropna()
    
    strategy_total = (portfolio_history[-1] / portfolio_history[0]) - 1
    benchmark_total = (benchmark_final / 100000) - 1
    spy_total = (spy_final / 100000) - 1
    
    n_periods = len(port_returns)
    strategy_annual = (1 + strategy_total) ** (12 / n_periods) - 1
    benchmark_annual = (1 + benchmark_total) ** (252 / len(equal_returns)) - 1
    spy_annual = (1 + spy_total) ** (252 / len(spy_returns)) - 1
    
    strategy_vol = port_returns.std() * np.sqrt(12)
    benchmark_vol = equal_returns.std() * np.sqrt(252)
    spy_vol = spy_returns.std() * np.sqrt(252)
    
    strategy_sharpe = strategy_annual / strategy_vol if strategy_vol > 0 else 0
    benchmark_sharpe = benchmark_annual / benchmark_vol if benchmark_vol > 0 else 0
    spy_sharpe = spy_annual / spy_vol if spy_vol > 0 else 0
    
    # Print results
    print("\n" + "-"*50)
    print("OUT-OF-SAMPLE RESULTS:")
    print("-"*50)
    
    print(f"\n{'Metric':<30} {'Strategy':>12} {'EW Mega':>12} {'SPY':>12}")
    print("-"*70)
    print(f"{'Total Return':<30} {strategy_total*100:>11.1f}% {benchmark_total*100:>11.1f}% {spy_total*100:>11.1f}%")
    print(f"{'Annualized Return':<30} {strategy_annual*100:>11.1f}% {benchmark_annual*100:>11.1f}% {spy_annual*100:>11.1f}%")
    print(f"{'Annualized Volatility':<30} {strategy_vol*100:>11.1f}% {benchmark_vol*100:>11.1f}% {spy_vol*100:>11.1f}%")
    print(f"{'Sharpe Ratio':<30} {strategy_sharpe:>12.2f} {benchmark_sharpe:>12.2f} {spy_sharpe:>12.2f}")
    
    if use_lifecycle:
        total_phases = sum(phase_counts.values())
        print(f"\n{'Lifecycle Phases (OOS):':}")
        for phase, count in phase_counts.items():
            print(f"  {phase}: {count} ({count*100/max(total_phases,1):.0f}%)")
    
    return {
        'strategy_return': strategy_total,
        'strategy_sharpe': strategy_sharpe,
        'strategy_vol': strategy_vol,
        'benchmark_sharpe': benchmark_sharpe,
        'spy_sharpe': spy_sharpe,
        'total_trades': total_trades,
        'phase_counts': phase_counts,
    }


# ============================================================================
# MAIN WALK-FORWARD TEST
# ============================================================================

def run_walk_forward_validation():
    """
    The definitive walk-forward validation test.
    
    Train: 2015-2019 (pre-COVID)
    Test:  2020-2024 (COVID and beyond)
    """
    print("="*70)
    print("WALK-FORWARD VALIDATION")
    print("The only test that matters for institutional credibility")
    print("="*70)
    
    # Define universe
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'WMT', 'JPM', 'V', 'JNJ', 'UNH', 'XOM']
    
    # STEP 1: Calculate stats from TRAINING period only
    print("\n" + "="*70)
    print("STEP 1: TRAINING PERIOD (2015-2019)")
    print("Learning sector regime duration statistics...")
    print("="*70)
    
    frozen_stats = calculate_sector_stats_from_training(
        tickers=tickers,
        train_start='2015-01-01',
        train_end='2019-12-31',
    )
    
    # STEP 2: Test with binary (no lifecycle) for baseline
    print("\n" + "="*70)
    print("STEP 2a: OUT-OF-SAMPLE TEST (2020-2024) - BINARY BASELINE")
    print("="*70)
    
    binary_results = walk_forward_backtest(
        tickers=tickers,
        test_start='2020-01-01',
        test_end='2024-12-31',
        frozen_stats=frozen_stats,
        use_lifecycle=False,
    )
    
    # STEP 3: Test with lifecycle using FROZEN stats
    print("\n" + "="*70)
    print("STEP 2b: OUT-OF-SAMPLE TEST (2020-2024) - LIFECYCLE w/ FROZEN STATS")
    print("="*70)
    
    lifecycle_results = walk_forward_backtest(
        tickers=tickers,
        test_start='2020-01-01',
        test_end='2024-12-31',
        frozen_stats=frozen_stats,
        use_lifecycle=True,
    )
    
    # FINAL VERDICT
    print("\n" + "="*70)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("="*70)
    
    print(f"\n{'Metric':<30} {'Binary (OOS)':>15} {'Lifecycle (OOS)':>15}")
    print("-"*60)
    print(f"{'Sharpe Ratio':<30} {binary_results['strategy_sharpe']:>15.2f} {lifecycle_results['strategy_sharpe']:>15.2f}")
    print(f"{'Total Return':<30} {binary_results['strategy_return']*100:>14.1f}% {lifecycle_results['strategy_return']*100:>14.1f}%")
    print(f"{'Volatility':<30} {binary_results['strategy_vol']*100:>14.1f}% {lifecycle_results['strategy_vol']*100:>14.1f}%")
    
    sharpe_improvement = lifecycle_results['strategy_sharpe'] - binary_results['strategy_sharpe']
    return_improvement = (lifecycle_results['strategy_return'] - binary_results['strategy_return']) * 100
    
    print(f"\n{'Sharpe Improvement':<30} {sharpe_improvement:>+15.2f}")
    print(f"{'Return Improvement':<30} {return_improvement:>+14.1f}%")
    
    # VERDICT
    oos_sharpe = lifecycle_results['strategy_sharpe']
    
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    if oos_sharpe >= 1.5:
        verdict = "🎯 AMAZING: OOS Sharpe >= 1.5 - PRODUCT READY!"
        status = "VALIDATED"
    elif oos_sharpe >= 1.3:
        verdict = "✅ GOOD: OOS Sharpe >= 1.3 - Commercially viable"
        status = "VALIDATED"
    elif oos_sharpe >= 1.0:
        verdict = "⚡ ACCEPTABLE: OOS Sharpe >= 1.0 - Needs refinement"
        status = "BORDERLINE"
    else:
        verdict = "❌ PROBLEM: OOS Sharpe < 1.0 - May be overfitted"
        status = "FAILED"
    
    print(f"""
OUT-OF-SAMPLE (2020-2024) PERFORMANCE:
  Sharpe Ratio: {oos_sharpe:.2f}
  Total Return: {lifecycle_results['strategy_return']*100:.1f}%
  Volatility:   {lifecycle_results['strategy_vol']*100:.1f}%

VERDICT: {verdict}

COMPARISON:
  In-Sample (2019-2024):  Sharpe ~1.71
  Out-of-Sample (2020-2024): Sharpe {oos_sharpe:.2f}
  
  Decay: {(1.71 - oos_sharpe):.2f} Sharpe points
  Decay %: {(1.71 - oos_sharpe)*100/1.71:.0f}%
  
  NOTE: Some decay is NORMAL and expected.
        20-30% decay is acceptable.
        >50% decay indicates overfitting.

STATUS: {status}

NEXT STEPS:
  {'→ Proceed to MVP development' if status == 'VALIDATED' else '→ Investigate sources of decay'}
  {'→ Build Streamlit dashboard' if status == 'VALIDATED' else '→ Recalibrate sector statistics'}
  {'→ Prepare investor materials' if status == 'VALIDATED' else '→ Check for period-specific patterns'}
""")
    
    return {
        'binary_results': binary_results,
        'lifecycle_results': lifecycle_results,
        'frozen_stats': frozen_stats,
        'oos_sharpe': oos_sharpe,
        'status': status,
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    results = run_walk_forward_validation()
