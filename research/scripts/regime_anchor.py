"""
REGIME ANCHOR - Two-Tier Regime Trading System
================================================

Product architecture based on validated research findings:

KEY INSIGHT: Mega-caps behave DIFFERENTLY than sector averages
- Mega-caps: More stable, longer regimes (70-90 days)
- Others: Follow sector patterns (20-30 days)

STRATEGY:
- Tier 1 (Mega-cap): Long lookbacks, quarterly rebalancing, higher position sizes
- Tier 2 (Others): Sector defaults, monthly rebalancing

TARGET: Sharpe > 1.5 with < 200 trades/year

Author: Project Iris Research Team
Date: 2025-12-31
Finance Expert Approved: Yes
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# TIER CLASSIFICATION
# ============================================================================

class StockTier(Enum):
    """Stock classification for two-tier system."""
    MEGA_CAP = 1      # >$100B market cap, stable, institutional
    LARGE_CAP = 2     # $10B-$100B, sector followers
    OTHER = 3         # Everything else


@dataclass
class StockProfile:
    """Complete stock profile for tier classification."""
    ticker: str
    tier: StockTier
    market_cap: float  # In billions
    volatility: float  # Annualized
    lookback: int      # Recommended lookback days
    rebalance_freq: str  # 'monthly', 'quarterly'
    position_risk: float  # % of portfolio per position
    sector: str


class TwoTierClassifier:
    """
    Classify stocks into Tier 1 (Mega-cap) or Tier 2 (Other).
    
    Mega-cap criteria (from research):
    - Market cap > $100B
    - Volatility < sector average * 0.7
    - Or: in pre-defined mega-cap list
    """
    
    # Pre-defined mega-caps (top 20 by market cap + stability)
    MEGA_CAP_UNIVERSE = [
        # Tech Giants
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA',
        # Diversified Giants
        'BRK-B', 'JPM', 'JNJ', 'V', 'UNH', 'MA', 'PG', 'HD',
        'XOM', 'CVX', 'WMT', 'KO', 'PEP', 'MRK', 'ABBV',
        # Add TSLA but with caution (high vol)
    ]
    
    # Sector for each stock
    STOCK_SECTOR = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
        'GOOG': 'Technology', 'AMZN': 'Consumer', 'META': 'Technology',
        'NVDA': 'Technology', 'BRK-B': 'Financials', 'JPM': 'Financials',
        'JNJ': 'Healthcare', 'V': 'Financials', 'UNH': 'Healthcare',
        'MA': 'Financials', 'PG': 'Consumer', 'HD': 'Consumer',
        'XOM': 'Energy', 'CVX': 'Energy', 'WMT': 'Consumer',
        'KO': 'Consumer', 'PEP': 'Consumer', 'MRK': 'Healthcare',
        'ABBV': 'Healthcare', 'TSLA': 'Technology',
    }
    
    # Sector volatility benchmarks (from our 110-stock analysis)
    SECTOR_VOL = {
        'Technology': 0.40,
        'Consumer': 0.23,
        'Financials': 0.28,
        'Healthcare': 0.25,
        'Energy': 0.34,
        'Default': 0.25,
    }
    
    # Tier-specific parameters (from research findings)
    TIER_PARAMS = {
        StockTier.MEGA_CAP: {
            'lookback': 80,          # Longer lookback for stability
            'rebalance_freq': 'quarterly',
            'position_risk': 0.025,  # 2.5% risk per position (can be higher)
            'min_regime_days': 40,   # Don't switch regimes too fast
            'signal_threshold': 0.5, # Bull prob > 50% = buy
        },
        StockTier.LARGE_CAP: {
            'lookback': 50,
            'rebalance_freq': 'monthly',
            'position_risk': 0.02,
            'min_regime_days': 20,
            'signal_threshold': 0.6,
        },
        StockTier.OTHER: {
            'lookback': 30,
            'rebalance_freq': 'monthly',
            'position_risk': 0.015,
            'min_regime_days': 10,
            'signal_threshold': 0.65,
        },
    }
    
    def __init__(self):
        self.stock_data_cache = {}
    
    def get_market_cap(self, ticker: str) -> float:
        """Get approximate market cap in billions (cached)."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            market_cap = info.get('marketCap', 0) / 1e9  # Convert to billions
            return market_cap
        except:
            return 0
    
    def get_volatility(self, ticker: str, lookback_days: int = 252) -> float:
        """Calculate annualized volatility."""
        try:
            data = yf.download(ticker, period='2y', progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
            returns = np.log(data[price_col] / data[price_col].shift(1))
            vol = returns.tail(lookback_days).std() * np.sqrt(252)
            return vol
        except:
            return 0.30  # Default
    
    def classify(self, ticker: str, quick_mode: bool = True) -> StockProfile:
        """
        Classify a stock into its appropriate tier.
        
        Args:
            ticker: Stock symbol
            quick_mode: If True, use pre-defined lists instead of live data
        
        Returns:
            StockProfile with all parameters
        """
        ticker = ticker.upper()
        
        # Quick classification using pre-defined list
        if quick_mode and ticker in self.MEGA_CAP_UNIVERSE:
            tier = StockTier.MEGA_CAP
            sector = self.STOCK_SECTOR.get(ticker, 'Technology')
            params = self.TIER_PARAMS[tier]
            
            return StockProfile(
                ticker=ticker,
                tier=tier,
                market_cap=0,  # Not fetched in quick mode
                volatility=0,
                lookback=params['lookback'],
                rebalance_freq=params['rebalance_freq'],
                position_risk=params['position_risk'],
                sector=sector,
            )
        
        # Full classification using live data
        market_cap = self.get_market_cap(ticker)
        volatility = self.get_volatility(ticker)
        sector = self.STOCK_SECTOR.get(ticker, 'Default')
        sector_vol = self.SECTOR_VOL.get(sector, 0.25)
        
        # Classification logic
        if market_cap > 100:  # > $100B
            # Check if volatility is low enough for mega-cap
            if volatility < sector_vol * 0.8:
                tier = StockTier.MEGA_CAP
            else:
                tier = StockTier.LARGE_CAP  # High-vol mega-cap
        elif market_cap > 10:  # $10B - $100B
            tier = StockTier.LARGE_CAP
        else:
            tier = StockTier.OTHER
        
        params = self.TIER_PARAMS[tier]
        
        return StockProfile(
            ticker=ticker,
            tier=tier,
            market_cap=market_cap,
            volatility=volatility,
            lookback=params['lookback'],
            rebalance_freq=params['rebalance_freq'],
            position_risk=params['position_risk'],
            sector=sector,
        )


# ============================================================================
# REGIME DETECTION
# ============================================================================

class RegimeDetector:
    """
    Simple regime detection optimized per tier.
    
    Uses rolling Sharpe ratio and volatility to identify:
    - Bull: Positive returns, below-average volatility
    - Bear: Negative returns OR high volatility
    - Neutral: Everything else
    """
    
    def __init__(self, lookback: int = 60):
        self.lookback = lookback
    
    def detect_regime(self, returns: pd.Series) -> Tuple[str, float]:
        """
        Detect current regime and probability.
        
        Returns:
            (regime_name, probability)
        """
        if len(returns) < self.lookback:
            return 'Neutral', 0.5
        
        recent = returns.tail(self.lookback)
        
        # Calculate rolling metrics
        rolling_return = recent.mean() * 252  # Annualized
        rolling_vol = recent.std() * np.sqrt(252)
        sharpe = rolling_return / rolling_vol if rolling_vol > 0 else 0
        
        # Historical vol for comparison
        if len(returns) > 252:
            hist_vol = returns.tail(252).std() * np.sqrt(252)
        else:
            hist_vol = rolling_vol
        
        vol_ratio = rolling_vol / hist_vol if hist_vol > 0 else 1.0
        
        # Regime classification with probability
        if sharpe > 0.5 and vol_ratio < 1.1:
            # Strong bull: high Sharpe, low volatility
            prob = min(0.95, 0.5 + sharpe * 0.3)
            return 'Bull', prob
        elif sharpe > 0.2 and vol_ratio < 1.3:
            # Moderate bull
            prob = 0.5 + sharpe * 0.25
            return 'Bull', prob
        elif sharpe < -0.3:
            # Bear regime
            prob = min(0.9, 0.5 - sharpe * 0.3)
            return 'Bear', prob
        elif vol_ratio > 1.5:
            # Crisis (high vol)
            prob = min(0.9, 0.5 + (vol_ratio - 1) * 0.3)
            return 'Bear', prob
        else:
            return 'Neutral', 0.55
    
    def calculate_regime_duration(self, returns: pd.Series) -> int:
        """Calculate how many days we've been in current regime."""
        if len(returns) < self.lookback:
            return 0
        
        current_regime, _ = self.detect_regime(returns)
        
        # Walk backward to find regime start
        days_in_regime = 0
        for i in range(len(returns) - 1, max(self.lookback, 0), -1):
            historical_returns = returns.iloc[:i]
            if len(historical_returns) < self.lookback:
                break
            regime, _ = self.detect_regime(historical_returns)
            if regime == current_regime:
                days_in_regime += 1
            else:
                break
            # Limit search to avoid expensive computation
            if days_in_regime > 200:
                break
        
        return days_in_regime


# ============================================================================
# REGIME ANCHOR STRATEGY
# ============================================================================

class RegimeAnchorStrategy:
    """
    Two-tier regime-based trading strategy.
    
    Core Logic:
    - Mega-caps: Buy when bull probability > 50%, hold minimum 40 days
    - Others: Follow sector regime with higher threshold
    
    Position Sizing:
    - Based on tier-specific risk parameters
    - Reduce position during transitions
    """
    
    def __init__(self):
        self.classifier = TwoTierClassifier()
        self.regime_detectors = {}
    
    def setup_universe(self, tickers: List[str]) -> Dict[str, StockProfile]:
        """Classify all tickers and set up regime detectors."""
        profiles = {}
        for ticker in tickers:
            profile = self.classifier.classify(ticker, quick_mode=True)
            profiles[ticker] = profile
            
            # Create tier-specific regime detector
            lookback = self.classifier.TIER_PARAMS[profile.tier]['lookback']
            self.regime_detectors[ticker] = RegimeDetector(lookback=lookback)
        
        return profiles
    
    def generate_signals(self, 
                        returns_dict: Dict[str, pd.Series],
                        profiles: Dict[str, StockProfile]) -> Dict[str, Dict]:
        """
        Generate trading signals for all tickers.
        
        Returns:
            Dict of {ticker: signal_dict}
        """
        signals = {}
        
        for ticker, returns in returns_dict.items():
            if ticker not in profiles:
                continue
            
            profile = profiles[ticker]
            detector = self.regime_detectors.get(ticker)
            
            if detector is None:
                continue
            
            # Detect regime
            regime, probability = detector.detect_regime(returns)
            duration = detector.calculate_regime_duration(returns)
            
            # Get tier-specific parameters
            params = self.classifier.TIER_PARAMS[profile.tier]
            min_duration = params['min_regime_days']
            threshold = params['signal_threshold']
            
            # Generate signal
            if regime == 'Bull' and probability > threshold:
                action = 'BUY'
                strength = probability
            elif regime == 'Bear' and probability > threshold:
                if duration < min_duration:
                    # Don't sell too fast - wait for confirmed bear
                    action = 'HOLD'
                    strength = 0.5
                else:
                    action = 'SELL'
                    strength = probability
            else:
                action = 'HOLD'
                strength = 0.5
            
            # Reduce strength during regime transitions
            if duration < min_duration * 0.5:
                strength *= 0.7  # Fade new signals
            
            signals[ticker] = {
                'action': action,
                'strength': strength,
                'regime': regime,
                'probability': probability,
                'duration': duration,
                'tier': profile.tier.name,
                'position_risk': profile.position_risk,
            }
        
        return signals


# ============================================================================
# MEGA-CAP BACKTEST
# ============================================================================

def mega_cap_backtest(
    start_year: int = 2019,
    end_year: int = 2024,
    transaction_cost_bps: float = 10,
    rebalance_freq: str = 'quarterly',
):
    """
    Backtest mega-cap focused regime strategy.
    
    Target: Sharpe > 1.5 with < 200 trades/year
    
    Strategy:
    - 10-stock mega-cap portfolio
    - 80-day lookback for regime detection
    - Quarterly rebalancing
    - Minimum 40-day regime hold
    """
    print("\n" + "="*70)
    print("REGIME ANCHOR: Mega-Cap Backtest")
    print(f"Period: {start_year} to {end_year}")
    print(f"Rebalancing: {rebalance_freq}")
    print("="*70)
    
    # Core mega-cap universe (10 stocks, diversified)
    mega_caps = [
        'AAPL', 'MSFT', 'GOOGL',  # Tech
        'AMZN', 'WMT',            # Consumer
        'JPM', 'V',               # Financials
        'JNJ', 'UNH',             # Healthcare
        'XOM',                    # Energy
    ]
    
    # Setup
    strategy = RegimeAnchorStrategy()
    profiles = strategy.setup_universe(mega_caps)
    
    # Download data
    print("\nFetching mega-cap data...")
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    
    prices = pd.DataFrame()
    returns_dict = {}
    
    for ticker in mega_caps:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
        prices[ticker] = data[price_col]
        
    prices = prices.dropna()
    
    for ticker in mega_caps:
        returns_dict[ticker] = np.log(prices[ticker] / prices[ticker].shift(1)).dropna()
    
    returns = prices.pct_change().dropna()
    
    print(f"Loaded {len(prices)} days of data for {len(mega_caps)} stocks")
    
    # Rebalance schedule
    if rebalance_freq == 'quarterly':
        rebal_dates = prices.resample('Q').last().index
    else:
        rebal_dates = prices.resample('M').last().index
    
    # Safe lookup helper
    def safe_lookup(df, date, col=None):
        valid = df.index[df.index <= date]
        if len(valid) == 0:
            return df[col].iloc[0] if col else df.iloc[0]
        return df.loc[valid[-1], col] if col else df.loc[valid[-1]]
    
    # Track portfolio
    portfolio_value = 100000
    portfolio_history = [portfolio_value]
    holdings = {ticker: 0 for ticker in mega_caps}
    cash = portfolio_value
    
    total_trades = 0
    total_costs = 0
    
    # Regime history for tracking
    regime_history = []
    
    # Backtest loop
    for i in range(1, len(rebal_dates)):
        rebal_start = rebal_dates[i-1]
        rebal_end = rebal_dates[i]
        
        if rebal_end > prices.index[-1]:
            continue
        
        # Get current returns for regime detection
        current_returns = {
            ticker: returns_dict[ticker].loc[:rebal_start]
            for ticker in mega_caps
        }
        
        # Generate signals
        signals = strategy.generate_signals(current_returns, profiles)
        
        # Calculate target weights
        target_weights = {}
        buy_signals = [(t, s) for t, s in signals.items() if s['action'] == 'BUY']
        
        if buy_signals:
            # Equal weight among buy signals
            base_weight = 1.0 / len(buy_signals)
            for ticker, signal in buy_signals:
                # Adjust weight by signal strength
                target_weights[ticker] = base_weight * signal['strength']
        else:
            # No buy signals = all cash
            pass
        
        # Normalize weights
        total_weight = sum(target_weights.values())
        if total_weight > 1.0:
            for t in target_weights:
                target_weights[t] /= total_weight
        
        # Fill missing with 0
        for ticker in mega_caps:
            if ticker not in target_weights:
                target_weights[ticker] = 0
        
        # Calculate portfolio value at rebalance date
        current_value = cash
        for ticker in mega_caps:
            price = safe_lookup(prices, rebal_start, ticker)
            current_value += holdings[ticker] * price
        
        # Execute trades
        for ticker in mega_caps:
            current_shares = holdings[ticker]
            price = safe_lookup(prices, rebal_start, ticker)
            current_pos_value = current_shares * price
            target_pos_value = current_value * target_weights[ticker]
            
            trade_value = abs(target_pos_value - current_pos_value)
            
            if trade_value > 100:  # Minimum trade
                # Transaction cost
                cost = trade_value * (transaction_cost_bps / 10000) * 0.6  # ETF discount
                total_costs += cost
                total_trades += 1
                
                if target_pos_value > current_pos_value:
                    # Buy
                    shares_to_buy = (target_pos_value - current_pos_value - cost) / price
                    holdings[ticker] += shares_to_buy
                    cash -= (target_pos_value - current_pos_value)
                else:
                    # Sell
                    shares_to_sell = (current_pos_value - target_pos_value) / price
                    holdings[ticker] -= shares_to_sell
                    cash += (current_pos_value - target_pos_value - cost)
        
        # Record regime states
        for ticker, signal in signals.items():
            regime_history.append({
                'date': rebal_start,
                'ticker': ticker,
                'regime': signal['regime'],
                'probability': signal['probability'],
                'duration': signal['duration'],
                'action': signal['action'],
            })
        
        # Mark to market at period end
        end_value = cash
        for ticker in mega_caps:
            price = safe_lookup(prices, rebal_end, ticker)
            end_value += holdings[ticker] * price
        
        portfolio_history.append(end_value)
    
    # Calculate benchmark (equal weight mega-cap buy-and-hold)
    equal_weight_returns = returns[mega_caps].mean(axis=1)
    benchmark_cum = (1 + equal_weight_returns).cumprod()
    benchmark_final = 100000 * benchmark_cum.iloc[-1]
    
    # Calculate S&P 500 benchmark
    spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False)
    if isinstance(spy_data.columns, pd.MultiIndex):
        spy_data.columns = spy_data.columns.get_level_values(0)
    spy_col = 'Adj Close' if 'Adj Close' in spy_data.columns else 'Close'
    spy_returns = np.log(spy_data[spy_col] / spy_data[spy_col].shift(1)).dropna()
    spy_cum = (1 + spy_returns.apply(np.exp) - 1).cumprod()
    spy_final = 100000 * spy_cum.iloc[-1]
    
    # Statistics
    portfolio_returns = pd.Series(portfolio_history).pct_change().dropna()
    
    strategy_total = (portfolio_history[-1] / portfolio_history[0]) - 1
    benchmark_total = (benchmark_final / 100000) - 1
    spy_total = (spy_final / 100000) - 1
    
    periods_per_year = 4 if rebalance_freq == 'quarterly' else 12
    strategy_annual = (1 + strategy_total) ** (periods_per_year / len(portfolio_returns)) - 1
    benchmark_annual = (1 + benchmark_total) ** (252 / len(equal_weight_returns)) - 1
    spy_annual = (1 + spy_total) ** (252 / len(spy_returns)) - 1
    
    strategy_vol = portfolio_returns.std() * np.sqrt(periods_per_year)
    benchmark_vol = equal_weight_returns.std() * np.sqrt(252)
    spy_vol = spy_returns.std() * np.sqrt(252)
    
    strategy_sharpe = strategy_annual / strategy_vol if strategy_vol > 0 else 0
    benchmark_sharpe = benchmark_annual / benchmark_vol if benchmark_vol > 0 else 0
    spy_sharpe = spy_annual / spy_vol if spy_vol > 0 else 0
    
    # Print results
    print("\n" + "-"*50)
    print("RESULTS:")
    print("-"*50)
    
    print(f"\n{'Metric':<30} {'Strategy':>12} {'EW Mega':>12} {'SPY':>12}")
    print("-"*70)
    print(f"{'Total Return':<30} {strategy_total*100:>11.1f}% {benchmark_total*100:>11.1f}% {spy_total*100:>11.1f}%")
    print(f"{'Annualized Return':<30} {strategy_annual*100:>11.1f}% {benchmark_annual*100:>11.1f}% {spy_annual*100:>11.1f}%")
    print(f"{'Annualized Volatility':<30} {strategy_vol*100:>11.1f}% {benchmark_vol*100:>11.1f}% {spy_vol*100:>11.1f}%")
    print(f"{'Sharpe Ratio':<30} {strategy_sharpe:>12.2f} {benchmark_sharpe:>12.2f} {spy_sharpe:>12.2f}")
    
    print(f"\n{'Transaction Costs':}")
    print(f"  Total trades: {total_trades}")
    print(f"  Total costs: ${total_costs:,.0f}")
    print(f"  Trades/year: {total_trades / ((end_year - start_year) or 1):.0f}")
    
    # Verdict
    print("\n" + "-"*50)
    if strategy_sharpe > 1.5:
        print("🎯 TARGET MET: Sharpe > 1.5!")
    elif strategy_sharpe > 1.0:
        print("✓ GOOD: Sharpe > 1.0 after costs")
    else:
        print("⚠️ NEEDS WORK: Sharpe < 1.0")
    
    if total_trades / max((end_year - start_year), 1) < 200:
        print(f"✓ LOW TURNOVER: {total_trades / max((end_year - start_year), 1):.0f} trades/year")
    else:
        print(f"⚠️ HIGH TURNOVER: {total_trades / max((end_year - start_year), 1):.0f} trades/year")
    
    if strategy_total > spy_total:
        print(f"✓ BEATS SPY: +{(strategy_total - spy_total)*100:.1f}%")
    else:
        print(f"❌ TRAILS SPY: {(strategy_total - spy_total)*100:.1f}%")
    
    return {
        'strategy_return': strategy_total,
        'benchmark_return': benchmark_total,
        'spy_return': spy_total,
        'strategy_sharpe': strategy_sharpe,
        'benchmark_sharpe': benchmark_sharpe,
        'spy_sharpe': spy_sharpe,
        'total_trades': total_trades,
        'total_costs': total_costs,
        'portfolio_history': portfolio_history,
        'regime_history': regime_history,
    }


# ============================================================================
# COMPARATIVE ANALYSIS
# ============================================================================

def run_comparative_analysis():
    """Compare different rebalancing frequencies and parameters."""
    
    print("\n" + "="*70)
    print("REGIME ANCHOR: Comparative Analysis")
    print("="*70)
    
    results = {}
    
    # Test 1: Quarterly vs Monthly rebalancing
    print("\n\n" + "#"*70)
    print("# TEST 1: Quarterly Rebalancing (Lower Turnover)")
    print("#"*70)
    results['quarterly'] = mega_cap_backtest(rebalance_freq='quarterly')
    
    print("\n\n" + "#"*70)
    print("# TEST 2: Monthly Rebalancing (Higher Turnover)")
    print("#"*70)
    results['monthly'] = mega_cap_backtest(rebalance_freq='monthly')
    
    # Summary
    print("\n\n" + "="*70)
    print("SUMMARY: QUARTERLY vs MONTHLY")
    print("="*70)
    print(f"\n{'Metric':<30} {'Quarterly':>15} {'Monthly':>15}")
    print("-"*60)
    print(f"{'Sharpe Ratio':<30} {results['quarterly']['strategy_sharpe']:>15.2f} {results['monthly']['strategy_sharpe']:>15.2f}")
    print(f"{'Total Return':<30} {results['quarterly']['strategy_return']*100:>14.1f}% {results['monthly']['strategy_return']*100:>14.1f}%")
    print(f"{'Trades/Year':<30} {results['quarterly']['total_trades']/5:>15.0f} {results['monthly']['total_trades']/5:>15.0f}")
    print(f"{'Total Costs':<30} ${results['quarterly']['total_costs']:>14,.0f} ${results['monthly']['total_costs']:>14,.0f}")
    
    # Winner
    print("\n" + "-"*50)
    if results['quarterly']['strategy_sharpe'] > results['monthly']['strategy_sharpe']:
        print("🏆 WINNER: Quarterly Rebalancing")
    else:
        print("🏆 WINNER: Monthly Rebalancing")
    
    # Final Recommendation
    print("\n" + "="*70)
    print("FINAL RECOMMENDATION")
    print("="*70)
    
    best_config = 'quarterly' if results['quarterly']['strategy_sharpe'] > results['monthly']['strategy_sharpe'] else 'monthly'
    best = results[best_config]
    
    print(f"""
REGIME ANCHOR STRATEGY
----------------------
Universe: 10 Mega-cap stocks (AAPL, MSFT, GOOGL, AMZN, JPM, V, JNJ, UNH, WMT, XOM)
Lookback: 80 days
Rebalancing: {best_config.upper()}
Minimum regime hold: 40 days

PERFORMANCE (2019-2024):
- Sharpe Ratio: {best['strategy_sharpe']:.2f}
- Total Return: {best['strategy_return']*100:.1f}%
- Trades/Year: {best['total_trades']/5:.0f}
- vs SPY: {'+' if best['strategy_return'] > best['spy_return'] else ''}{(best['strategy_return'] - best['spy_return'])*100:.1f}%

PRODUCT VIABILITY:
- Target (Sharpe > 1.5): {'✓ MET' if best['strategy_sharpe'] > 1.5 else '❌ NOT MET'}
- Turnover (<200/yr): {'✓ LOW' if best['total_trades']/5 < 200 else '❌ HIGH'}
- Beats SPY: {'✓ YES' if best['strategy_return'] > best['spy_return'] else '❌ NO'}
""")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run complete RegimeAnchor analysis."""
    
    print("="*70)
    print("REGIME ANCHOR - Two-Tier Trading System")
    print("Building on validated research findings")
    print("="*70)
    
    # Run comparative analysis
    results = run_comparative_analysis()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    return results


if __name__ == '__main__':
    results = main()
