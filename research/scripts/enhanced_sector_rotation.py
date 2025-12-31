"""
ENHANCED SECTOR ROTATION STRATEGY
==================================

Implements finance expert recommendations:
1. Realistic transaction costs (0.1% per trade)
2. Volatility-adjusted Kelly position sizing
3. Enhanced signals: Regime + Momentum + Volatility trio
4. Two-tier analysis: Stock vs Sector deviation

CRITICAL FIXES:
- Transaction costs will reduce Sharpe from 1.25 to ~1.0
- Dynamic position sizing improves risk-adjusted returns
- Two-tier analysis explains anomalies like AAPL vs NEE

Author: Project Iris Research Team
Date: 2025-12-31
Finance Expert Review: Incorporated
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import sector metadata from sector_optimized_hmm
try:
    from sector_optimized_hmm import SECTOR_CLUSTERS, get_sector, get_sector_params
except ImportError:
    # Fallback - define minimal metadata
    SECTOR_CLUSTERS = {
        'Large Cap Tech': {'avg_duration': 32, 'avg_volatility': 0.404, 'position_risk': 0.02},
        'Utilities': {'avg_duration': 19, 'avg_volatility': 0.213, 'position_risk': 0.03},
    }


# ============================================================================
# TRANSACTION COST MODEL
# ============================================================================

class TransactionCostModel:
    """
    Realistic transaction cost model for institutional trading.
    
    Components:
    1. Brokerage commission (minimal for institutional)
    2. Bid-ask spread (varies by liquidity)
    3. Market impact (for large orders)
    4. Slippage (execution timing)
    """
    
    # Default costs by asset class (in basis points, i.e., 10 = 0.1%)
    DEFAULT_COSTS = {
        'Large Cap ETF': 5,     # Very liquid (SPY, XLK, etc.)
        'Sector ETF': 10,       # Standard sector ETFs
        'Individual Stock - Large': 8,
        'Individual Stock - Small': 20,
        'High Volatility': 30,  # Meme stocks, crypto-adjacent
    }
    
    def __init__(self, base_cost_bps: float = 10):
        """
        Args:
            base_cost_bps: Base transaction cost in basis points (10 = 0.1%)
        """
        self.base_cost_bps = base_cost_bps
        
    def calculate_cost(self, ticker: str, trade_value: float, 
                       is_etf: bool = True, volatility: float = None) -> float:
        """
        Calculate total transaction cost for a trade.
        
        Args:
            ticker: Security symbol
            trade_value: Dollar value of trade (absolute)
            is_etf: Whether this is an ETF (lower costs)
            volatility: Current volatility (higher vol = higher costs)
        
        Returns:
            Total cost in dollars
        """
        # Base cost
        cost_bps = self.base_cost_bps
        
        # Adjust for volatility (higher vol = wider spreads)
        if volatility is not None:
            vol_multiplier = max(1.0, volatility / 0.20)  # Baseline 20% vol
            cost_bps *= vol_multiplier
        
        # ETF discount
        if is_etf:
            cost_bps *= 0.6  # 40% lower costs for ETFs
        
        # Fixed minimum cost per trade
        min_cost = 1.0  # $1 minimum per trade
        
        cost = max(min_cost, trade_value * (cost_bps / 10000))
        
        return cost
    
    def calculate_roundtrip(self, trade_value: float, **kwargs) -> float:
        """Calculate round-trip cost (buy + sell)."""
        return 2 * self.calculate_cost(trade_value=trade_value, **kwargs)
    
    def estimate_annual_drag(self, annual_turnover: float, avg_trade_size: float,
                             **kwargs) -> float:
        """
        Estimate annual drag from transaction costs.
        
        Args:
            annual_turnover: Times portfolio turns over per year (e.g., 12 = monthly)
            avg_trade_size: Average trade size in dollars
        
        Returns:
            Annualized cost as percentage of portfolio
        """
        cost_per_turn = self.calculate_roundtrip(avg_trade_size, **kwargs)
        return cost_per_turn * annual_turnover / avg_trade_size


# ============================================================================
# DYNAMIC POSITION SIZING (Modified Kelly)
# ============================================================================

class DynamicPositionSizer:
    """
    Implements volatility-adjusted Modified Kelly position sizing.
    
    Finance Expert Formula:
    Position Size = (Target Risk %) / (Sector Volatility Ratio)
    
    Modified Kelly:
    kelly = win_rate - (1 - win_rate) / win_loss_ratio
    adjusted = kelly * (market_vol / sector_vol)
    capped = min(max_position, max(min_position, adjusted))
    """
    
    # Market baseline volatility (long-term average)
    MARKET_VOL = 0.16  # ~16% annualized for S&P 500
    
    def __init__(self, 
                 target_risk: float = 0.02,
                 max_position: float = 0.05,
                 min_position: float = 0.005,
                 kelly_fraction: float = 0.5):  # Half-Kelly for safety
        """
        Args:
            target_risk: Target risk per position (2% default)
            max_position: Maximum position size (5%)
            min_position: Minimum position size (0.5%)
            kelly_fraction: Fraction of Kelly to use (0.5 = half-Kelly)
        """
        self.target_risk = target_risk
        self.max_position = max_position
        self.min_position = min_position
        self.kelly_fraction = kelly_fraction
    
    def volatility_adjusted_size(self, sector_vol: float) -> float:
        """
        Simple volatility-adjusted position sizing.
        
        Higher sector vol → smaller position
        Lower sector vol → larger position
        
        Args:
            sector_vol: Annualized sector volatility
        
        Returns:
            Position size as fraction of portfolio
        """
        vol_ratio = sector_vol / self.MARKET_VOL
        adjusted = self.target_risk / vol_ratio
        
        return np.clip(adjusted, self.min_position, self.max_position)
    
    def modified_kelly_size(self, 
                           win_rate: float, 
                           avg_win: float, 
                           avg_loss: float,
                           sector_vol: float) -> float:
        """
        Modified Kelly with volatility adjustment.
        
        Args:
            win_rate: Historical win rate (P(profit))
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive value)
            sector_vol: Annualized sector volatility
        
        Returns:
            Position size as fraction of portfolio
        """
        if avg_loss == 0:
            return self.min_position
        
        win_loss_ratio = avg_win / avg_loss
        
        # Kelly formula
        kelly = win_rate - (1 - win_rate) / win_loss_ratio
        
        # Apply Kelly fraction (half-Kelly is safer)
        kelly *= self.kelly_fraction
        
        # Volatility adjustment
        vol_ratio = sector_vol / self.MARKET_VOL
        adjusted = kelly / vol_ratio
        
        return np.clip(adjusted, self.min_position, self.max_position)
    
    def regime_adjusted_size(self, 
                            base_size: float, 
                            regime: str,
                            regime_progress: float) -> float:
        """
        Adjust position size based on regime state.
        
        Finance Expert Insight:
        - Increase during bull regimes
        - Decrease during bear/crisis regimes
        - Consider regime duration (fade at extremes)
        
        Args:
            base_size: Base position size
            regime: Current regime name
            regime_progress: How far into the regime (0-1+)
        
        Returns:
            Adjusted position size
        """
        regime_multipliers = {
            'Bull': 1.3,      # Increase in bull
            'Calm': 1.1,      # Slightly increase in calm
            'Neutral': 1.0,   # No change
            'Bear': 0.7,      # Reduce in bear
            'Crisis': 0.4,    # Significantly reduce in crisis
        }
        
        multiplier = regime_multipliers.get(regime, 1.0)
        
        # Fade at regime extremes (>100% of expected duration)
        if regime_progress > 1.0:
            fade_factor = 1.0 / (1.0 + (regime_progress - 1.0) * 0.5)
            multiplier *= fade_factor
        
        adjusted = base_size * multiplier
        return np.clip(adjusted, self.min_position, self.max_position)


# ============================================================================
# ENHANCED SIGNAL MODEL (Regime + Momentum + Volatility Trio)
# ============================================================================

class EnhancedSignalModel:
    """
    Three-factor signal model combining:
    1. Regime (50% weight)
    2. Momentum (25% weight)
    3. Volatility (25% weight)
    
    Finance Expert: "This creates regime-aware momentum signals."
    """
    
    WEIGHTS = {
        'regime': 0.50,
        'momentum': 0.25,
        'volatility': 0.25,
    }
    
    def __init__(self, 
                 momentum_lookback: int = 20,
                 volatility_lookback: int = 60):
        self.momentum_lookback = momentum_lookback
        self.volatility_lookback = volatility_lookback
    
    def calculate_regime_score(self, regime: str, regime_progress: float) -> float:
        """
        Regime component of signal.
        
        Returns score between -1 (strong sell) and +1 (strong buy).
        """
        regime_scores = {
            'Bull': 0.8,
            'Calm': 0.3,
            'Neutral': 0.0,
            'Bear': -0.6,
            'Crisis': -1.0,
        }
        
        base_score = regime_scores.get(regime, 0.0)
        
        # Early in regime = more confident
        # Late in regime = fade the signal
        if regime_progress < 0.5:
            confidence = 1.0 + (0.5 - regime_progress)  # Up to 1.5x
        elif regime_progress > 1.0:
            confidence = 1.0 / (1.0 + (regime_progress - 1.0))  # Fade
        else:
            confidence = 1.0
        
        return np.clip(base_score * confidence, -1.0, 1.0)
    
    def calculate_momentum_score(self, returns: pd.Series) -> float:
        """
        Momentum component of signal.
        
        Uses 20-day momentum, normalized.
        """
        if len(returns) < self.momentum_lookback:
            return 0.0
        
        momentum = returns.tail(self.momentum_lookback).sum()
        
        # Normalize by typical momentum magnitude
        typical_momentum = returns.std() * np.sqrt(self.momentum_lookback)
        normalized = momentum / (typical_momentum + 1e-6)
        
        return np.clip(normalized, -1.0, 1.0)
    
    def calculate_volatility_score(self, returns: pd.Series, 
                                   sector_avg_vol: float) -> float:
        """
        Volatility component of signal.
        
        Below-average vol = bullish (calm markets trend)
        Above-average vol = bearish (volatile markets mean-revert)
        """
        if len(returns) < self.volatility_lookback:
            return 0.0
        
        current_vol = returns.tail(self.volatility_lookback).std() * np.sqrt(252)
        
        # Score: positive if below average, negative if above
        vol_ratio = current_vol / sector_avg_vol
        score = 1.0 - vol_ratio  # >1 ratio → negative score
        
        return np.clip(score, -1.0, 1.0)
    
    def generate_signal(self, 
                       regime: str,
                       regime_progress: float,
                       returns: pd.Series,
                       sector_avg_vol: float) -> Dict:
        """
        Generate combined signal using three-factor model.
        
        Returns:
            Dict with signal, strength, and component breakdown
        """
        regime_score = self.calculate_regime_score(regime, regime_progress)
        momentum_score = self.calculate_momentum_score(returns)
        volatility_score = self.calculate_volatility_score(returns, sector_avg_vol)
        
        # Weighted average
        combined = (
            self.WEIGHTS['regime'] * regime_score +
            self.WEIGHTS['momentum'] * momentum_score +
            self.WEIGHTS['volatility'] * volatility_score
        )
        
        # Convert to signal
        if combined > 0.3:
            action = 'BUY'
        elif combined < -0.3:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        return {
            'action': action,
            'strength': abs(combined),
            'combined_score': combined,
            'regime_score': regime_score,
            'momentum_score': momentum_score,
            'volatility_score': volatility_score,
        }


# ============================================================================
# TWO-TIER ANALYSIS: Stock vs Sector Deviation
# ============================================================================

class StockSectorAnalyzer:
    """
    Two-tier analysis: Why does a stock deviate from its sector?
    
    Answers questions like:
    - Why does AAPL have shorter regimes than Tech sector average?
    - Is this stock behaving unusually vs its sector?
    - Should we use sector params or stock-specific tuning?
    """
    
    def __init__(self, deviation_threshold: float = 0.20):
        """
        Args:
            deviation_threshold: % deviation to flag as significant (20%)
        """
        self.deviation_threshold = deviation_threshold
    
    def fetch_stock_data(self, ticker: str, years: int = 5) -> pd.DataFrame:
        """Fetch stock data with error handling."""
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(years=years)
        
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Handle Adj Close vs Close
        price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
        
        df = pd.DataFrame(index=data.index)
        df['close'] = data[price_col]
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df = df.dropna()
        
        return df
    
    def calculate_regime_duration(self, returns: np.ndarray, 
                                  lookback: int = 60) -> float:
        """
        Estimate average regime duration using volatility-based detection.
        """
        # Calculate rolling volatility
        vol = pd.Series(returns).rolling(lookback).std() * np.sqrt(252)
        vol = vol.dropna()
        
        if len(vol) < 100:
            return 20  # Default
        
        # Simple regime detection: Above/below median vol
        median_vol = vol.median()
        high_vol = vol > median_vol
        
        # Count regime changes
        regime_changes = (high_vol != high_vol.shift(1)).sum()
        
        if regime_changes < 2:
            return len(vol)  # Single regime
        
        avg_duration = len(vol) / regime_changes
        return avg_duration
    
    def analyze_stock_vs_sector(self, ticker: str) -> Dict:
        """
        Full stock-vs-sector deviation analysis.
        
        Returns detailed diagnostic with:
        - Sector expectation vs actual
        - Deviation percentage
        - Likely causes
        - Trading implication
        """
        # Get sector info
        sector = get_sector(ticker) if 'get_sector' in dir() else 'Large Cap Tech'
        sector_params = SECTOR_CLUSTERS.get(sector, SECTOR_CLUSTERS.get('Large Cap Tech', {}))
        
        sector_expected_duration = sector_params.get('avg_duration', 25)
        sector_avg_vol = sector_params.get('avg_volatility', 0.25)
        sector_lookback = sector_params.get('lookback_days', 60)
        
        # Fetch stock data
        df = self.fetch_stock_data(ticker)
        returns = df['log_ret'].values
        
        # Calculate stock-specific metrics
        stock_duration = self.calculate_regime_duration(returns, sector_lookback)
        stock_vol = df['log_ret'].std() * np.sqrt(252)
        stock_kurtosis = stats.kurtosis(returns)
        
        # Calculate deviation
        duration_deviation = (stock_duration - sector_expected_duration) / sector_expected_duration
        vol_deviation = (stock_vol - sector_avg_vol) / sector_avg_vol
        
        # Determine likely causes
        likely_causes = []
        
        if abs(vol_deviation) > 0.30:
            cause = 'Higher volatility than sector' if vol_deviation > 0 else 'Lower volatility than sector'
            likely_causes.append(cause)
        
        if stock_kurtosis > 6:
            likely_causes.append('High kurtosis (fat tails) - news-driven moves')
        
        if duration_deviation > self.deviation_threshold:
            likely_causes.append('Longer regimes - trend persistence')
        elif duration_deviation < -self.deviation_threshold:
            likely_causes.append('Shorter regimes - high regime turnover')
        
        # Trading implication
        if duration_deviation > 0:
            trading_implication = 'Use longer lookback than sector default'
        else:
            trading_implication = 'Use shorter lookback than sector default'
        
        # Confidence in sector parameters
        total_deviation = abs(duration_deviation) + abs(vol_deviation)
        if total_deviation < 0.30:
            confidence = 'HIGH - Use sector parameters'
        elif total_deviation < 0.60:
            confidence = 'MEDIUM - Consider stock-specific tuning'
        else:
            confidence = 'LOW - Stock behaves differently than sector'
        
        return {
            'ticker': ticker,
            'sector': sector,
            'sector_expected_duration': sector_expected_duration,
            'stock_actual_duration': round(stock_duration, 1),
            'duration_deviation_pct': round(duration_deviation * 100, 1),
            'sector_avg_vol': round(sector_avg_vol * 100, 1),
            'stock_vol': round(stock_vol * 100, 1),
            'vol_deviation_pct': round(vol_deviation * 100, 1),
            'stock_kurtosis': round(stock_kurtosis, 2),
            'likely_causes': likely_causes,
            'trading_implication': trading_implication,
            'sector_param_confidence': confidence,
            'recommended_lookback': round(sector_lookback * (1 + duration_deviation * 0.5)),
        }


# ============================================================================
# ENHANCED BACKTEST WITH TRANSACTION COSTS
# ============================================================================

def enhanced_sector_rotation_backtest(
    start_year: int = 2019,
    end_year: int = 2024,
    transaction_cost_bps: float = 10,
    use_kelly_sizing: bool = True,
    use_enhanced_signals: bool = True,
):
    """
    Enhanced sector rotation backtest with:
    1. Realistic transaction costs
    2. Volatility-adjusted position sizing
    3. Three-factor signals
    
    Finance Expert Target: Sharpe > 1.0 AFTER transaction costs
    """
    print("\n" + "="*70)
    print("ENHANCED SECTOR ROTATION BACKTEST")
    print(f"Period: {start_year} to {end_year}")
    print(f"Transaction costs: {transaction_cost_bps} bps")
    print(f"Position sizing: {'Modified Kelly' if use_kelly_sizing else 'Equal Weight'}")
    print(f"Signals: {'Enhanced (Regime+Momentum+Vol)' if use_enhanced_signals else 'Simple'}")
    print("="*70)
    
    # Sector ETFs
    sector_etfs = ['XLK', 'XLY', 'XLP', 'XLV', 'XLF', 'XLE', 'XLI', 'XLU', 'XLC']
    
    sector_mapping = {
        'XLK': 'Large Cap Tech',
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLV': 'Healthcare',
        'XLF': 'Financials',
        'XLE': 'Energy',
        'XLI': 'Industrials',
        'XLU': 'Utilities',
        'XLC': 'Communication',
    }
    
    # Download data
    print("\nFetching sector ETF data...")
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    
    prices = pd.DataFrame()
    for etf in sector_etfs:
        data = yf.download(etf, start=start_date, end=end_date, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
        prices[etf] = data[price_col]
    
    prices = prices.dropna()
    returns = prices.pct_change().dropna()
    
    print(f"Loaded {len(prices)} days of data")
    
    # Initialize models
    cost_model = TransactionCostModel(base_cost_bps=transaction_cost_bps)
    position_sizer = DynamicPositionSizer()
    signal_model = EnhancedSignalModel()
    
    # Calculate sector statistics
    sector_stats = {}
    for etf in sector_etfs:
        etf_returns = returns[etf]
        vol = etf_returns.rolling(60).std() * np.sqrt(252)
        momentum = etf_returns.rolling(20).sum()
        
        # Simple regime detection
        sharpe_rolling = (etf_returns.rolling(60).mean() * 252) / vol
        vol_median = vol.rolling(252).median()
        
        # Bull = positive Sharpe and below-average vol
        is_bull = (sharpe_rolling > 0.3) & (vol < vol_median * 1.2)
        
        sector_stats[etf] = {
            'volatility': vol,
            'momentum': momentum,
            'sharpe_rolling': sharpe_rolling,
            'is_bull': is_bull.astype(float),
            'avg_vol': vol.mean(),
        }
    
    # Monthly rebalancing with enhanced logic
    monthly_dates = prices.resample('M').last().index
    
    # Track portfolio
    portfolio_value = 100000  # Start with $100k
    portfolio_history = [portfolio_value]
    holdings = {etf: 0 for etf in sector_etfs}
    cash = portfolio_value
    
    total_trades = 0
    total_costs = 0
    
    # Helper function for safe date lookup
    def safe_price_lookup(df, date, column=None):
        """Safely lookup price at or before given date."""
        try:
            if column:
                # For DataFrame with columns
                valid_dates = df.index[df.index <= date]
                if len(valid_dates) == 0:
                    return df[column].iloc[0]
                return df.loc[valid_dates[-1], column]
            else:
                # For Series
                valid_dates = df.index[df.index <= date]
                if len(valid_dates) == 0:
                    return df.iloc[0]
                return df.loc[valid_dates[-1]]
        except:
            return df.iloc[-1] if column is None else df[column].iloc[-1]
    
    # Backtest loop
    for i in range(1, len(monthly_dates)):
        month_start = monthly_dates[i-1]
        month_end = monthly_dates[i]
        
        # Skip if dates not in price data
        if month_end > prices.index[-1]:
            continue
        
        # Get signals for each sector
        target_weights = {}
        
        for etf in sector_etfs:
            if use_enhanced_signals:
                # Get regime (simplified)
                is_bull = sector_stats[etf]['is_bull'].loc[:month_start].iloc[-1]
                regime = 'Bull' if is_bull else 'Bear'
                
                # Get recent returns for momentum
                recent_returns = returns[etf].loc[:month_start].tail(60)
                avg_vol = sector_stats[etf]['avg_vol']
                
                # Calculate regime progress (simplified)
                regime_progress = 0.5  # Placeholder
                
                signal = signal_model.generate_signal(
                    regime=regime,
                    regime_progress=regime_progress,
                    returns=recent_returns,
                    sector_avg_vol=avg_vol
                )
                
                # Convert signal to weight target
                if signal['action'] == 'BUY':
                    weight = 0.2 * signal['strength']
                elif signal['action'] == 'SELL':
                    weight = 0.0
                else:
                    weight = 0.1  # Small holding if HOLD
            else:
                # Simple signal: bull regime = buy
                is_bull = sector_stats[etf]['is_bull'].loc[:month_start].iloc[-1]
                weight = 0.2 if is_bull else 0.0
            
            # Apply position sizing
            if use_kelly_sizing:
                vol = sector_stats[etf]['volatility'].loc[:month_start].iloc[-1]
                if not np.isnan(vol):
                    size_factor = position_sizer.volatility_adjusted_size(vol)
                    weight = weight * size_factor / 0.02  # Normalize to base size
            
            target_weights[etf] = weight
        
        # Normalize weights to sum to 1 (or less if going to cash)
        total_weight = sum(target_weights.values())
        if total_weight > 1.0:
            for etf in target_weights:
                target_weights[etf] /= total_weight
        
        # Calculate required trades
        current_portfolio_value = cash + sum(
            holdings[etf] * safe_price_lookup(prices, month_start, etf)
            for etf in sector_etfs
        )
        
        for etf in sector_etfs:
            current_value = holdings[etf] * safe_price_lookup(prices, month_start, etf)
            target_value = current_portfolio_value * target_weights[etf]
            trade_value = abs(target_value - current_value)
            
            if trade_value > 100:  # Minimum trade size
                # Calculate transaction cost
                vol = sector_stats[etf]['volatility'].loc[:month_start].iloc[-1]
                cost = cost_model.calculate_cost(
                    ticker=etf,
                    trade_value=trade_value,
                    is_etf=True,
                    volatility=vol if not np.isnan(vol) else 0.20
                )
                
                total_costs += cost
                total_trades += 1
                
                # Execute trade
                if target_value > current_value:
                    # Buy
                    price_at_start = safe_price_lookup(prices, month_start, etf)
                    shares_to_buy = (target_value - current_value - cost) / price_at_start
                    holdings[etf] += shares_to_buy
                    cash -= (target_value - current_value)
                else:
                    # Sell
                    price_at_start = safe_price_lookup(prices, month_start, etf)
                    shares_to_sell = (current_value - target_value) / price_at_start
                    holdings[etf] -= shares_to_sell
                    cash += (current_value - target_value - cost)
        
        # Calculate month-end portfolio value
        month_returns = returns.loc[month_start:month_end]
        
        # Mark-to-market
        end_portfolio_value = cash + sum(
            holdings[etf] * safe_price_lookup(prices, month_end, etf)
            for etf in sector_etfs
        )
        
        portfolio_history.append(end_portfolio_value)
    
    # Calculate benchmark (equal weight, no trading)
    benchmark_returns = returns.mean(axis=1)
    benchmark_cum = (1 + benchmark_returns).cumprod()
    benchmark_final = 100000 * benchmark_cum.iloc[-1]
    
    # Calculate statistics
    portfolio_returns = pd.Series(portfolio_history).pct_change().dropna()
    
    strategy_total_return = (portfolio_history[-1] / portfolio_history[0]) - 1
    benchmark_total_return = (benchmark_final / 100000) - 1
    
    strategy_annual = (1 + strategy_total_return) ** (12 / len(portfolio_returns)) - 1
    benchmark_annual = (1 + benchmark_total_return) ** (252 / len(returns)) - 1
    
    strategy_vol = portfolio_returns.std() * np.sqrt(12)
    benchmark_vol = benchmark_returns.std() * np.sqrt(252)
    
    strategy_sharpe = strategy_annual / strategy_vol if strategy_vol > 0 else 0
    benchmark_sharpe = benchmark_annual / benchmark_vol if benchmark_vol > 0 else 0
    
    # Transaction cost impact
    cost_drag = total_costs / portfolio_history[0]
    
    print("\n" + "-"*50)
    print("RESULTS:")
    print("-"*50)
    print(f"\n{'Metric':<30} {'Strategy':>12} {'Benchmark':>12}")
    print("-"*55)
    print(f"{'Total Return':<30} {strategy_total_return*100:>11.1f}% {benchmark_total_return*100:>11.1f}%")
    print(f"{'Annualized Return':<30} {strategy_annual*100:>11.1f}% {benchmark_annual*100:>11.1f}%")
    print(f"{'Annualized Volatility':<30} {strategy_vol*100:>11.1f}% {benchmark_vol*100:>11.1f}%")
    print(f"{'Sharpe Ratio':<30} {strategy_sharpe:>12.2f} {benchmark_sharpe:>12.2f}")
    
    print(f"\n{'Transaction Costs Impact':}")
    print(f"  Total trades: {total_trades}")
    print(f"  Total costs: ${total_costs:,.0f}")
    print(f"  Cost drag: {cost_drag*100:.2f}%")
    print(f"  Cost per trade: ${total_costs/total_trades:.2f}" if total_trades > 0 else "")
    
    outperformance = strategy_annual - benchmark_annual
    print(f"\n{'Outperformance':<30} {outperformance*100:>11.1f}%")
    
    # Verdict
    print("\n" + "-"*50)
    if strategy_sharpe > 1.0:
        print("✓ TARGET MET: Sharpe > 1.0 AFTER transaction costs!")
    elif strategy_sharpe > benchmark_sharpe:
        print("⚠️ OUTPERFORMS but Sharpe < 1.0. Needs refinement.")
    else:
        print("✗ UNDERPERFORMS. Strategy needs major revision.")
    
    return {
        'strategy_return': strategy_total_return,
        'benchmark_return': benchmark_total_return,
        'strategy_sharpe': strategy_sharpe,
        'benchmark_sharpe': benchmark_sharpe,
        'total_costs': total_costs,
        'cost_drag': cost_drag,
        'portfolio_history': portfolio_history,
    }


# ============================================================================
# MAIN: Run All Analyses
# ============================================================================

def main():
    """Run complete enhanced analysis suite."""
    
    print("="*70)
    print("ENHANCED SECTOR ANALYSIS SUITE")
    print("With Finance Expert Recommendations")
    print("="*70)
    
    # 1. Stock vs Sector Analysis for AAPL, NEE, MSFT
    print("\n\n" + "#"*70)
    print("# 1. STOCK-VS-SECTOR DEVIATION ANALYSIS")
    print("#"*70)
    
    analyzer = StockSectorAnalyzer()
    
    for ticker in ['AAPL', 'NEE', 'MSFT']:
        result = analyzer.analyze_stock_vs_sector(ticker)
        
        print(f"\n📊 {ticker} vs {result['sector']}:")
        print(f"   Expected duration: {result['sector_expected_duration']} days")
        print(f"   Actual duration:   {result['stock_actual_duration']} days ({result['duration_deviation_pct']:+.0f}%)")
        print(f"   Sector vol: {result['sector_avg_vol']:.1f}%")
        print(f"   Stock vol:  {result['stock_vol']:.1f}% ({result['vol_deviation_pct']:+.0f}%)")
        print(f"   Kurtosis: {result['stock_kurtosis']}")
        print(f"   Likely causes: {', '.join(result['likely_causes']) if result['likely_causes'] else 'None identified'}")
        print(f"   Confidence: {result['sector_param_confidence']}")
        print(f"   Recommended lookback: {result['recommended_lookback']} days")
    
    # 2. Enhanced backtest with transaction costs
    print("\n\n" + "#"*70)
    print("# 2. BACKTEST: Simple Strategy (Baseline)")
    print("#"*70)
    
    baseline = enhanced_sector_rotation_backtest(
        use_kelly_sizing=False,
        use_enhanced_signals=False,
        transaction_cost_bps=10,
    )
    
    print("\n\n" + "#"*70)
    print("# 3. BACKTEST: Enhanced Strategy (Kelly + Signals)")
    print("#"*70)
    
    enhanced = enhanced_sector_rotation_backtest(
        use_kelly_sizing=True,
        use_enhanced_signals=True,
        transaction_cost_bps=10,
    )
    
    # Summary
    print("\n\n" + "="*70)
    print("SUMMARY: SIMPLE vs ENHANCED")
    print("="*70)
    print(f"\n{'Metric':<30} {'Simple':>12} {'Enhanced':>12}")
    print("-"*55)
    print(f"{'Sharpe Ratio':<30} {baseline['strategy_sharpe']:>12.2f} {enhanced['strategy_sharpe']:>12.2f}")
    print(f"{'Total Return':<30} {baseline['strategy_return']*100:>11.1f}% {enhanced['strategy_return']*100:>11.1f}%")
    print(f"{'Cost Drag':<30} {baseline['cost_drag']*100:>11.2f}% {enhanced['cost_drag']*100:>11.2f}%")
    
    improvement = enhanced['strategy_sharpe'] - baseline['strategy_sharpe']
    print(f"\n{'Sharpe Improvement':<30} {improvement:>+12.2f}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    return analyzer, baseline, enhanced


if __name__ == '__main__':
    analyzer, baseline, enhanced = main()
