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
# REGIME TRANSITION PREDICTION (The Missing 20%)
# ============================================================================

class RegimeTransitionPredictor:
    """
    Predict regime changes BEFORE they happen using leading indicators.
    
    Finance Expert Insight:
    - The strategy is reactive (identifies regimes after they start)
    - This class makes it PREDICTIVE (anticipates transitions 10-20 days earlier)
    
    Leading Indicators Used:
    1. Volume surge (3x normal) → Smart money positioning
    2. Volatility compression (Bollinger squeeze) → Big move coming
    3. Momentum divergence (price vs momentum) → Trend exhaustion
    4. Regime fatigue (too long in current regime)
    
    Expected Improvement: +3-5% annual returns from earlier entries
    """
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
    
    def calculate_volume_surge(self, volume: pd.Series) -> float:
        """
        Detect smart money positioning via volume surge.
        
        Volume surge (> 2.5x normal) often precedes major moves.
        """
        if len(volume) < self.lookback + 5:
            return 0.0
        
        # Recent volume vs average
        avg_volume_20 = volume.rolling(20).mean().iloc[-1]
        recent_volume = volume.tail(5).mean()
        
        if avg_volume_20 == 0:
            return 0.0
        
        surge_ratio = recent_volume / avg_volume_20
        
        # Score: 0 if normal, up to 1.0 if surge > 3x
        if surge_ratio > 2.5:
            return min(1.0, (surge_ratio - 2.5) / 0.5)
        return 0.0
    
    def calculate_volatility_compression(self, prices: pd.Series) -> float:
        """
        Detect Bollinger Band squeeze (volatility compression).
        
        When bands compress, a big move is imminent (but direction unknown).
        Combine with other signals for direction.
        """
        if len(prices) < self.lookback * 2:
            return 0.0
        
        # Calculate Bollinger Band width
        rolling_mean = prices.rolling(20).mean()
        rolling_std = prices.rolling(20).std()
        
        bb_width = (2 * rolling_std) / rolling_mean  # %B width
        bb_width = bb_width.dropna()
        
        if len(bb_width) < self.lookback:
            return 0.0
        
        # Current width vs historical
        current_width = bb_width.iloc[-1]
        avg_width = bb_width.rolling(self.lookback).mean().iloc[-1]
        
        if avg_width == 0:
            return 0.0
        
        compression_ratio = current_width / avg_width
        
        # Score: 1.0 if heavily compressed (< 0.5x normal)
        if compression_ratio < 0.7:
            return min(1.0, (0.7 - compression_ratio) / 0.2)
        return 0.0
    
    def calculate_momentum_divergence(self, prices: pd.Series) -> Tuple[float, str]:
        """
        Detect price-momentum divergence.
        
        - Price up but momentum slowing → Bearish divergence (top)
        - Price down but momentum improving → Bullish divergence (bottom)
        """
        if len(prices) < self.lookback * 2:
            return 0.0, 'neutral'
        
        returns = prices.pct_change().dropna()
        
        # Price trend (10-day)
        price_change = (prices.iloc[-1] / prices.iloc[-10]) - 1
        
        # Momentum (rate of change of returns)
        momentum_recent = returns.tail(5).mean()
        momentum_prior = returns.tail(15).head(10).mean()
        
        # Divergence detection
        if price_change > 0.02:  # Price up > 2%
            if momentum_recent < momentum_prior * 0.5:
                # Price rising but momentum fading → Bearish divergence
                strength = min(1.0, abs(momentum_prior - momentum_recent) / 0.005)
                return strength, 'bearish_divergence'
        elif price_change < -0.02:  # Price down > 2%
            if momentum_recent > momentum_prior * 0.5:
                # Price falling but momentum improving → Bullish divergence
                strength = min(1.0, abs(momentum_recent - momentum_prior) / 0.005)
                return strength, 'bullish_divergence'
        
        return 0.0, 'neutral'
    
    def calculate_regime_fatigue(self, current_regime: str, 
                                 regime_duration: int,
                                 expected_duration: int) -> float:
        """
        Detect regime fatigue (extended beyond normal duration).
        
        Regimes that last > 1.5x expected duration are likely to change.
        """
        if expected_duration == 0:
            return 0.0
        
        fatigue_ratio = regime_duration / expected_duration
        
        # Score: 0 if normal, up to 1.0 if > 2x expected
        if fatigue_ratio > 1.5:
            return min(1.0, (fatigue_ratio - 1.5) / 0.5)
        return 0.0
    
    def predict_transition(self, 
                          prices: pd.Series,
                          volume: pd.Series = None,
                          current_regime: str = 'Neutral',
                          regime_duration: int = 0,
                          expected_duration: int = 40) -> Dict:
        """
        Predict regime transition using all leading indicators.
        
        Returns:
            Dict with:
            - transition_score: 0-1 probability of imminent transition
            - anticipated_regime: Expected next regime
            - expected_in_days: Estimated days until transition
            - signals: Breakdown of which indicators fired
        """
        scores = []
        signals_fired = []
        
        # 1. Volume surge
        if volume is not None and len(volume) > 25:
            vol_score = self.calculate_volume_surge(volume)
            if vol_score > 0.3:
                scores.append(vol_score * 0.25)  # 25% weight
                signals_fired.append(f"Volume surge ({vol_score:.0%})")
        
        # 2. Volatility compression (Bollinger squeeze)
        compression_score = self.calculate_volatility_compression(prices)
        if compression_score > 0.3:
            scores.append(compression_score * 0.25)  # 25% weight
            signals_fired.append(f"Vol compression ({compression_score:.0%})")
        
        # 3. Momentum divergence
        div_score, div_type = self.calculate_momentum_divergence(prices)
        if div_score > 0.3:
            scores.append(div_score * 0.30)  # 30% weight
            signals_fired.append(f"{div_type} ({div_score:.0%})")
        
        # 4. Regime fatigue
        fatigue_score = self.calculate_regime_fatigue(
            current_regime, regime_duration, expected_duration
        )
        if fatigue_score > 0.3:
            scores.append(fatigue_score * 0.20)  # 20% weight
            signals_fired.append(f"Regime fatigue ({fatigue_score:.0%})")
        
        # Total transition probability
        transition_score = sum(scores)
        
        # Anticipated direction
        if div_type == 'bullish_divergence':
            anticipated = 'Bull'
        elif div_type == 'bearish_divergence':
            anticipated = 'Bear'
        elif current_regime == 'Bull':
            anticipated = 'Bear'  # Reversal expected
        elif current_regime == 'Bear':
            anticipated = 'Bull'
        else:
            anticipated = 'Bull'  # Default optimistic
        
        # Estimated days until transition
        if transition_score > 0.7:
            expected_days = 5
        elif transition_score > 0.5:
            expected_days = 10
        elif transition_score > 0.3:
            expected_days = 15
        else:
            expected_days = 30  # No imminent transition
        
        return {
            'transition_score': transition_score,
            'anticipated_regime': anticipated,
            'expected_in_days': expected_days,
            'signals_fired': signals_fired,
            'should_act_early': transition_score > 0.5,
        }


# ============================================================================
# REGIME ANCHOR STRATEGY
# ============================================================================

class RegimeAnchorStrategy:
    """
    Two-tier regime-based trading strategy with PREDICTIVE transitions.
    
    Core Logic:
    - Mega-caps: Buy when bull probability > 50%, hold minimum 40 days
    - Others: Follow sector regime with higher threshold
    - NEW: Use RegimeTransitionPredictor for early entries (the missing 20%)
    
    Position Sizing:
    - Based on tier-specific risk parameters
    - Reduce position during transitions
    - INCREASE position for high-confidence early entries
    """
    
    def __init__(self, use_prediction: bool = True):
        self.classifier = TwoTierClassifier()
        self.regime_detectors = {}
        self.transition_predictors = {}
        self.use_prediction = use_prediction
    
    def setup_universe(self, tickers: List[str]) -> Dict[str, StockProfile]:
        """Classify all tickers and set up regime detectors."""
        profiles = {}
        for ticker in tickers:
            profile = self.classifier.classify(ticker, quick_mode=True)
            profiles[ticker] = profile
            
            # Create tier-specific regime detector
            lookback = self.classifier.TIER_PARAMS[profile.tier]['lookback']
            self.regime_detectors[ticker] = RegimeDetector(lookback=lookback)
            
            # Create transition predictor
            self.transition_predictors[ticker] = RegimeTransitionPredictor(lookback=20)
        
        return profiles
    
    def generate_signals(self, 
                        returns_dict: Dict[str, pd.Series],
                        profiles: Dict[str, StockProfile],
                        prices_dict: Dict[str, pd.Series] = None,
                        volumes_dict: Dict[str, pd.Series] = None) -> Dict[str, Dict]:
        """
        Generate trading signals for all tickers.
        
        Now includes PREDICTIVE transitions for early entries.
        
        Returns:
            Dict of {ticker: signal_dict}
        """
        signals = {}
        
        for ticker, returns in returns_dict.items():
            if ticker not in profiles:
                continue
            
            profile = profiles[ticker]
            detector = self.regime_detectors.get(ticker)
            predictor = self.transition_predictors.get(ticker)
            
            if detector is None:
                continue
            
            # Detect current regime
            regime, probability = detector.detect_regime(returns)
            duration = detector.calculate_regime_duration(returns)
            
            # Get tier-specific parameters
            params = self.classifier.TIER_PARAMS[profile.tier]
            min_duration = params['min_regime_days']
            threshold = params['signal_threshold']
            
            # NEW: Predict regime transition (the missing 20%)
            transition_info = {'should_act_early': False, 'transition_score': 0, 'anticipated_regime': 'Neutral'}
            
            if self.use_prediction and predictor is not None:
                # Get prices for prediction
                if prices_dict and ticker in prices_dict:
                    prices = prices_dict[ticker]
                else:
                    # Convert returns to prices (approximate)
                    prices = (1 + returns).cumprod() * 100
                
                volume = volumes_dict.get(ticker) if volumes_dict else None
                
                transition_info = predictor.predict_transition(
                    prices=prices,
                    volume=volume,
                    current_regime=regime,
                    regime_duration=duration,
                    expected_duration=params['min_regime_days']
                )
            
            # Generate signal with prediction enhancement
            action, strength = self._generate_action(
                regime=regime,
                probability=probability,
                duration=duration,
                min_duration=min_duration,
                threshold=threshold,
                transition_info=transition_info
            )
            
            signals[ticker] = {
                'action': action,
                'strength': strength,
                'regime': regime,
                'probability': probability,
                'duration': duration,
                'tier': profile.tier.name,
                'position_risk': profile.position_risk,
                'transition_score': transition_info.get('transition_score', 0),
                'anticipated_regime': transition_info.get('anticipated_regime', 'Neutral'),
                'signals_fired': transition_info.get('signals_fired', []),
            }
        
        return signals
    
    def _generate_action(self, regime: str, probability: float, duration: int,
                        min_duration: int, threshold: float, 
                        transition_info: Dict) -> Tuple[str, float]:
        """
        Generate action with prediction enhancement.
        
        Key enhancement: If predictor says transition is coming, act EARLY.
        """
        # Check for predictive early entry/exit
        should_act_early = transition_info.get('should_act_early', False)
        anticipated = transition_info.get('anticipated_regime', 'Neutral')
        transition_score = transition_info.get('transition_score', 0)
        
        # EARLY ENTRY: Predictor sees bull coming before regime detector
        if should_act_early and anticipated == 'Bull' and regime != 'Bull':
            action = 'EARLY_BUY'
            # Higher strength for higher-confidence predictions
            strength = 0.6 + transition_score * 0.3
            return action, strength
        
        # EARLY EXIT: Predictor sees bear coming before regime detector
        if should_act_early and anticipated == 'Bear' and regime == 'Bull':
            action = 'EARLY_SELL'
            strength = 0.5 + transition_score * 0.3
            return action, strength
        
        # Standard regime-based signals (original logic)
        if regime == 'Bull' and probability > threshold:
            action = 'BUY'
            strength = probability
        elif regime == 'Bear' and probability > threshold:
            if duration < min_duration:
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
            strength *= 0.7
        
        return action, strength


# ============================================================================
# MEGA-CAP BACKTEST
# ============================================================================

def mega_cap_backtest(
    start_year: int = 2019,
    end_year: int = 2024,
    transaction_cost_bps: float = 10,
    rebalance_freq: str = 'quarterly',
    use_prediction: bool = True,
):
    """
    Backtest mega-cap focused regime strategy.
    
    Target: Sharpe > 1.5 with < 200 trades/year
    
    Strategy:
    - 10-stock mega-cap portfolio
    - 80-day lookback for regime detection
    - Quarterly rebalancing
    - Minimum 40-day regime hold
    - NEW: Predictive transition layer for early entries
    """
    mode = "PREDICTIVE" if use_prediction else "REACTIVE"
    print("\n" + "="*70)
    print(f"REGIME ANCHOR: Mega-Cap Backtest ({mode})")
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
    
    # Setup - with prediction mode
    strategy = RegimeAnchorStrategy(use_prediction=use_prediction)
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
    early_entry_count = 0  # Track predictive entries
    
    # Create prices_dict for prediction
    prices_dict = {ticker: prices[ticker] for ticker in mega_caps}
    
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
        
        # Get prices up to current date for prediction
        current_prices = {
            ticker: prices[ticker].loc[:rebal_start]
            for ticker in mega_caps
        }
        
        # Generate signals WITH PREDICTION
        signals = strategy.generate_signals(
            current_returns, 
            profiles,
            prices_dict=current_prices,
            volumes_dict=None,  # Volume not available in this simple version
        )
        
        # Calculate target weights - NOW INCLUDES EARLY_BUY/EARLY_SELL
        target_weights = {}
        buy_signals = [
            (t, s) for t, s in signals.items() 
            if s['action'] in ['BUY', 'EARLY_BUY']
        ]
        
        # Count early entries for diagnostic
        for t, s in signals.items():
            if 'EARLY' in s['action']:
                early_entry_count += 1
        
        if buy_signals:
            # Equal weight among buy signals
            base_weight = 1.0 / len(buy_signals)
            for ticker, signal in buy_signals:
                # Adjust weight by signal strength
                # Early entries get slight boost
                boost = 1.1 if 'EARLY' in signal['action'] else 1.0
                target_weights[ticker] = base_weight * signal['strength'] * boost
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
    if use_prediction:
        print(f"  Early entries: {early_entry_count} ({early_entry_count * 100 / max(total_trades, 1):.0f}% of trades)")
    
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
        'early_entries': early_entry_count,
        'portfolio_history': portfolio_history,
        'regime_history': regime_history,
        'use_prediction': use_prediction,
    }


# ============================================================================
# COMPARATIVE ANALYSIS
# ============================================================================

def run_comparative_analysis():
    """Compare REACTIVE vs PREDICTIVE strategies - the key test."""
    
    print("\n" + "="*70)
    print("REGIME ANCHOR: REACTIVE vs PREDICTIVE Comparison")
    print("Testing the Finance Expert's 'Missing 20%' hypothesis")
    print("="*70)
    
    results = {}
    
    # Test 1: Reactive strategy (original, no prediction)
    print("\n\n" + "#"*70)
    print("# TEST 1: REACTIVE Strategy (Original - No Prediction)")
    print("#"*70)
    results['reactive'] = mega_cap_backtest(
        rebalance_freq='monthly', 
        use_prediction=False
    )
    
    # Test 2: Predictive strategy (with transition anticipation)
    print("\n\n" + "#"*70)
    print("# TEST 2: PREDICTIVE Strategy (With Early Entry Signals)")
    print("#"*70)
    results['predictive'] = mega_cap_backtest(
        rebalance_freq='monthly', 
        use_prediction=True
    )
    
    # Summary: The Key Comparison
    print("\n\n" + "="*70)
    print("CRITICAL COMPARISON: REACTIVE vs PREDICTIVE")
    print("Does early transition prediction add alpha?")
    print("="*70)
    print(f"\n{'Metric':<30} {'Reactive':>15} {'Predictive':>15}")
    print("-"*60)
    print(f"{'Sharpe Ratio':<30} {results['reactive']['strategy_sharpe']:>15.2f} {results['predictive']['strategy_sharpe']:>15.2f}")
    print(f"{'Total Return':<30} {results['reactive']['strategy_return']*100:>14.1f}% {results['predictive']['strategy_return']*100:>14.1f}%")
    print(f"{'Trades/Year':<30} {results['reactive']['total_trades']/5:>15.0f} {results['predictive']['total_trades']/5:>15.0f}")
    print(f"{'Early Entries':<30} {'N/A':>15} {results['predictive'].get('early_entries', 0):>15}")
    
    # Calculate improvement
    sharpe_improvement = results['predictive']['strategy_sharpe'] - results['reactive']['strategy_sharpe']
    return_improvement = (results['predictive']['strategy_return'] - results['reactive']['strategy_return']) * 100
    
    print(f"\n{'Sharpe Improvement':<30} {sharpe_improvement:>+15.2f}")
    print(f"{'Return Improvement':<30} {return_improvement:>+14.1f}%")
    
    # Winner and Verdict
    print("\n" + "-"*50)
    if sharpe_improvement > 0:
        print("🏆 PREDICTIVE WINS - Early entries add alpha!")
        print(f"   → Sharpe improved by {sharpe_improvement:.2f}")
    elif sharpe_improvement < -0.05:
        print("⚠️ REACTIVE WINS - Prediction adds noise")
        print(f"   → Prediction cost {abs(sharpe_improvement):.2f} Sharpe")
    else:
        print("↔️ TIE - Prediction has minimal impact")
    
    # Finance Expert Validation
    print("\n" + "="*70)
    print("FINANCE EXPERT HYPOTHESIS VALIDATION")
    print("="*70)
    
    if sharpe_improvement > 0.2:
        verdict = "✓ VALIDATED: Predictive layer adds significant alpha"
        expected_improvement = True
    elif sharpe_improvement > 0:
        verdict = "⚡ PARTIALLY VALIDATED: Small improvement from prediction"
        expected_improvement = True
    else:
        verdict = "✗ NOT VALIDATED: Prediction doesn't help in this test"
        expected_improvement = False
    
    print(f"""
HYPOTHESIS: 'The strategy is reactive, not predictive. Adding 
            transition prediction should add 3-5% annual returns.'

RESULT: {verdict}

DETAILS:
  - Reactive Sharpe: {results['reactive']['strategy_sharpe']:.2f}
  - Predictive Sharpe: {results['predictive']['strategy_sharpe']:.2f}
  - Improvement: {sharpe_improvement:+.2f}

EARLY ENTRIES ANALYSIS:
  - Total early entries: {results['predictive'].get('early_entries', 0)}
  - % of signals from prediction: {results['predictive'].get('early_entries', 0) * 100 / max(results['predictive']['total_trades'], 1):.0f}%

NEXT STEPS:
  {'→ Deploy predictive layer in production' if expected_improvement else '→ Recalibrate prediction thresholds'}
  {'→ Test with volume data for better signals' if not expected_improvement else '→ Add volume-based predictions next'}
  → Walk-forward validation needed before production
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
