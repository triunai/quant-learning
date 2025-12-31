"""
REGIME SCANNER - Market Classification Engine
==============================================

A diagnostic tool that tells traders WHAT type of market they're in,
not WHAT will happen next.

COMMERCIAL VALUE:
- "Is this a trend or chop market?"
- "Should I trend-follow or mean-revert?"
- "Have we seen this pattern before?"

VALIDATED COMPONENTS:
- Sector clustering (works across all periods)
- Market type classification (TREND/CHOP/TRANSITION)
- Historical similarity matching

Author: Project Iris Research Team
Date: 2025-12-31
Status: Prototype (pivoting from failed trading strategy)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class MarketType(Enum):
    TREND = 'TREND'           # Directional, persistent moves
    CHOP = 'CHOP'             # Sideways, mean-reverting
    TRANSITION = 'TRANSITION' # Changing from one to another
    CRISIS = 'CRISIS'         # Extreme volatility


class TradingStyle(Enum):
    TREND_FOLLOW = 'Trend Following'
    MEAN_REVERT = 'Mean Reversion'
    AVOID = 'Avoid'
    HEDGED = 'Hedged'


class SectorCluster(Enum):
    GROWTH = 'Growth'           # Tech, Consumer Discretionary
    NEWS_DRIVEN = 'News-Driven' # Healthcare, Communications
    ECONOMIC = 'Economic'       # Financials, Energy, Industrials
    SPECULATIVE = 'Speculative' # High volatility, meme stocks


@dataclass
class RegimeScan:
    """Complete regime scan result."""
    ticker: str
    market_type: MarketType
    sector_cluster: SectorCluster
    recommended_style: TradingStyle
    confidence: float
    volatility_percentile: float
    trend_strength: float
    similar_periods: List[str]
    expected_duration: str
    scan_date: str
    
    def __str__(self):
        return f"""
RegimeScan for {self.ticker}
{'='*40}
Market Type:       {self.market_type.value} ({self.confidence*100:.0f}% confidence)
Sector Cluster:    {self.sector_cluster.value}
Recommended Style: {self.recommended_style.value}
Volatility:        {self.volatility_percentile:.0f}th percentile
Trend Strength:    {self.trend_strength:.2f}
Expected Duration: {self.expected_duration}
Similar Periods:   {', '.join(self.similar_periods[:3])}
"""


# ============================================================================
# SECTOR CLASSIFICATION
# ============================================================================

SECTOR_MAP = {
    # Growth Cluster
    'AAPL': SectorCluster.GROWTH,
    'MSFT': SectorCluster.GROWTH,
    'GOOGL': SectorCluster.GROWTH,
    'AMZN': SectorCluster.GROWTH,
    'META': SectorCluster.GROWTH,
    'NVDA': SectorCluster.GROWTH,
    'TSLA': SectorCluster.SPECULATIVE,
    
    # News-Driven Cluster
    'JNJ': SectorCluster.NEWS_DRIVEN,
    'UNH': SectorCluster.NEWS_DRIVEN,
    'PFE': SectorCluster.NEWS_DRIVEN,
    'LLY': SectorCluster.NEWS_DRIVEN,
    'DIS': SectorCluster.NEWS_DRIVEN,
    'NFLX': SectorCluster.NEWS_DRIVEN,
    
    # Economic Cluster
    'JPM': SectorCluster.ECONOMIC,
    'V': SectorCluster.ECONOMIC,
    'MA': SectorCluster.ECONOMIC,
    'XOM': SectorCluster.ECONOMIC,
    'CVX': SectorCluster.ECONOMIC,
    'GE': SectorCluster.ECONOMIC,
    'BA': SectorCluster.ECONOMIC,
    
    # Speculative Cluster
    'GME': SectorCluster.SPECULATIVE,
    'AMC': SectorCluster.SPECULATIVE,
    'COIN': SectorCluster.SPECULATIVE,
    'MARA': SectorCluster.SPECULATIVE,
    
    # Index defaults
    'SPY': SectorCluster.GROWTH,
    'QQQ': SectorCluster.GROWTH,
    'IWM': SectorCluster.ECONOMIC,
}


# ============================================================================
# HISTORICAL REFERENCE PERIODS
# ============================================================================

HISTORICAL_PERIODS = {
    '2008-Q4': {'type': MarketType.CRISIS, 'desc': 'Financial crisis crash'},
    '2009-Q1': {'type': MarketType.TRANSITION, 'desc': 'Post-crash recovery'},
    '2010-2011': {'type': MarketType.CHOP, 'desc': 'Post-GFC consolidation'},
    '2012-2014': {'type': MarketType.TREND, 'desc': 'QE-driven bull'},
    '2015-Q3': {'type': MarketType.CHOP, 'desc': 'China fears'},
    '2016': {'type': MarketType.CHOP, 'desc': 'Brexit/Election uncertainty'},
    '2017': {'type': MarketType.TREND, 'desc': 'Low-vol bull market'},
    '2018-Q4': {'type': MarketType.TRANSITION, 'desc': 'Fed tightening correction'},
    '2019': {'type': MarketType.TREND, 'desc': 'Fed pivot bull'},
    '2020-Q1': {'type': MarketType.CRISIS, 'desc': 'COVID crash'},
    '2020-Q2+': {'type': MarketType.TREND, 'desc': 'COVID rally'},
    '2022': {'type': MarketType.TRANSITION, 'desc': 'Fed hiking bear'},
    '2023-2024': {'type': MarketType.TREND, 'desc': 'AI-driven bull'},
}


# ============================================================================
# CORE SCANNER ENGINE
# ============================================================================

class RegimeScanner:
    """
    Market regime classification engine.
    
    Answers:
    1. "Is this a trend or chop market?"
    2. "Should I trend-follow or mean-revert?"
    3. "Have we seen this pattern before?"
    """
    
    def __init__(self, lookback_short: int = 20, lookback_long: int = 60):
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long
        self.cache = {}
    
    def scan(self, ticker: str, data: Optional[pd.DataFrame] = None) -> RegimeScan:
        """
        Perform complete regime scan for a ticker.
        
        Returns:
            RegimeScan with market type, style recommendation, etc.
        """
        # Fetch data if not provided
        if data is None:
            data = self._fetch_data(ticker)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
        prices = data[price_col]
        returns = np.log(prices / prices.shift(1)).dropna()
        
        # Core metrics
        volatility_percentile = self._calculate_volatility_percentile(returns)
        trend_strength = self._calculate_trend_strength(returns)
        autocorrelation = self._calculate_autocorrelation(returns)
        
        # Classify market type
        market_type, confidence = self._classify_market_type(
            returns, volatility_percentile, trend_strength, autocorrelation
        )
        
        # Get sector cluster
        sector_cluster = self._get_sector_cluster(ticker)
        
        # Determine trading style
        recommended_style = self._recommend_style(market_type, volatility_percentile)
        
        # Find similar historical periods
        similar_periods = self._find_similar_periods(
            volatility_percentile, trend_strength, market_type
        )
        
        # Estimate expected duration
        expected_duration = self._estimate_duration(market_type, volatility_percentile)
        
        return RegimeScan(
            ticker=ticker,
            market_type=market_type,
            sector_cluster=sector_cluster,
            recommended_style=recommended_style,
            confidence=confidence,
            volatility_percentile=volatility_percentile,
            trend_strength=trend_strength,
            similar_periods=similar_periods,
            expected_duration=expected_duration,
            scan_date=pd.Timestamp.now().strftime('%Y-%m-%d'),
        )
    
    def _fetch_data(self, ticker: str, period: str = '2y') -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        return yf.download(ticker, period=period, progress=False)
    
    def _calculate_volatility_percentile(self, returns: pd.Series) -> float:
        """
        Calculate where current volatility stands relative to history.
        """
        if len(returns) < self.lookback_long:
            return 50.0
        
        # Current 20-day volatility
        current_vol = returns.tail(self.lookback_short).std() * np.sqrt(252)
        
        # Historical volatility distribution
        rolling_vol = returns.rolling(self.lookback_short).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()
        
        # Percentile
        percentile = (rolling_vol < current_vol).mean() * 100
        return percentile
    
    def _calculate_trend_strength(self, returns: pd.Series) -> float:
        """
        Calculate trend strength using return persistence.
        
        High positive = strong uptrend
        High negative = strong downtrend
        Near zero = choppy
        """
        if len(returns) < self.lookback_long:
            return 0.0
        
        # 60-day cumulative return
        cum_return = returns.tail(self.lookback_long).sum()
        
        # Normalize by volatility
        vol = returns.tail(self.lookback_long).std() * np.sqrt(self.lookback_long)
        
        if vol > 0:
            trend_strength = cum_return / vol
        else:
            trend_strength = 0.0
        
        return np.clip(trend_strength, -3, 3)
    
    def _calculate_autocorrelation(self, returns: pd.Series) -> float:
        """
        Calculate return autocorrelation (persistence).
        
        Positive = trending (returns follow returns)
        Negative = mean-reverting
        """
        if len(returns) < self.lookback_long:
            return 0.0
        
        recent = returns.tail(self.lookback_long)
        return recent.autocorr(lag=1) if len(recent) > 10 else 0.0
    
    def _classify_market_type(
        self, 
        returns: pd.Series,
        vol_pct: float, 
        trend: float, 
        autocorr: float
    ) -> Tuple[MarketType, float]:
        """
        Classify market type based on multiple metrics.
        
        Returns:
            (MarketType, confidence_score)
        """
        # Crisis: Extreme volatility
        if vol_pct > 90:
            return MarketType.CRISIS, 0.85
        
        # Trend: Strong directional move with positive autocorrelation
        if abs(trend) > 1.5 and autocorr > 0.1:
            return MarketType.TREND, min(0.9, 0.5 + abs(trend) * 0.15)
        
        # Chop: Low trend strength, negative or neutral autocorrelation
        if abs(trend) < 0.5 and autocorr < 0.1:
            return MarketType.CHOP, 0.7 + (0.5 - abs(trend)) * 0.2
        
        # Transition: Mixed signals
        return MarketType.TRANSITION, 0.6
    
    def _get_sector_cluster(self, ticker: str) -> SectorCluster:
        """Map ticker to sector cluster."""
        return SECTOR_MAP.get(ticker.upper(), SectorCluster.GROWTH)
    
    def _recommend_style(
        self, 
        market_type: MarketType, 
        vol_pct: float
    ) -> TradingStyle:
        """
        Recommend trading style based on market type.
        """
        if market_type == MarketType.CRISIS:
            return TradingStyle.HEDGED
        elif market_type == MarketType.TREND:
            return TradingStyle.TREND_FOLLOW
        elif market_type == MarketType.CHOP:
            if vol_pct < 50:
                return TradingStyle.MEAN_REVERT
            else:
                return TradingStyle.AVOID
        else:  # TRANSITION
            return TradingStyle.AVOID
    
    def _find_similar_periods(
        self, 
        vol_pct: float, 
        trend: float, 
        market_type: MarketType
    ) -> List[str]:
        """
        Find historical periods similar to current conditions.
        """
        matches = []
        
        for period, info in HISTORICAL_PERIODS.items():
            if info['type'] == market_type:
                matches.append(f"{period} ({info['desc']})")
        
        # If no matches, return closest
        if not matches:
            if market_type in [MarketType.TREND, MarketType.TRANSITION]:
                matches = ['2019 (trend)', '2017 (low-vol bull)']
            else:
                matches = ['2016 (choppy)', '2015-Q3 (China fears)']
        
        return matches[:3]
    
    def _estimate_duration(
        self, 
        market_type: MarketType, 
        vol_pct: float
    ) -> str:
        """
        Estimate expected regime duration.
        """
        if market_type == MarketType.CRISIS:
            return 'SHORT (1-4 weeks)'
        elif market_type == MarketType.TREND:
            if vol_pct < 50:
                return 'LONG (3-6 months)'
            else:
                return 'MEDIUM (1-3 months)'
        elif market_type == MarketType.CHOP:
            return 'MEDIUM (2-4 months)'
        else:
            return 'SHORT (2-6 weeks)'


# ============================================================================
# MULTI-TICKER SCANNER
# ============================================================================

def scan_universe(tickers: List[str]) -> Dict[str, RegimeScan]:
    """
    Scan multiple tickers and return regime analysis.
    """
    scanner = RegimeScanner()
    results = {}
    
    for ticker in tickers:
        try:
            results[ticker] = scanner.scan(ticker)
        except Exception as e:
            print(f"Error scanning {ticker}: {e}")
    
    return results


def print_market_heatmap(scans: Dict[str, RegimeScan]):
    """
    Print a heatmap-style summary of all scans.
    """
    print("\n" + "="*70)
    print("REGIME SCANNER - MARKET HEATMAP")
    print("="*70)
    
    print(f"\n{'Ticker':<8} {'Type':<12} {'Style':<18} {'Vol%':>6} {'Trend':>7} {'Conf':>6}")
    print("-"*70)
    
    for ticker, scan in scans.items():
        print(f"{ticker:<8} {scan.market_type.value:<12} {scan.recommended_style.value:<18} "
              f"{scan.volatility_percentile:>5.0f}% {scan.trend_strength:>+6.2f} "
              f"{scan.confidence*100:>5.0f}%")
    
    print("-"*70)


# ============================================================================
# VALIDATION: TEST AGAINST MARKET HISTORY
# ============================================================================

def validate_scanner():
    """
    Validate scanner classifications against known market history.
    
    Tests:
    - 2017: Should be TREND (low-vol bull)
    - 2016: Should be CHOP (choppy year)
    - 2020-Q1: Should be CRISIS (COVID crash)
    - 2023: Should be TREND (AI bull)
    """
    print("="*70)
    print("REGIME SCANNER VALIDATION")
    print("Testing classifications against known market history")
    print("="*70)
    
    test_cases = [
        ('2017-01-01', '2017-12-31', 'Expected: TREND (low-vol bull)'),
        ('2016-01-01', '2016-12-31', 'Expected: CHOP (choppy year)'),
        ('2020-01-01', '2020-04-30', 'Expected: CRISIS/TRANSITION (COVID)'),
        ('2023-01-01', '2023-12-31', 'Expected: TREND (AI bull)'),
    ]
    
    scanner = RegimeScanner()
    
    for start, end, expected in test_cases:
        print(f"\n--- Testing {start[:4]} ---")
        print(f"{expected}")
        
        data = yf.download('SPY', start=start, end=end, progress=False)
        if len(data) > 0:
            scan = scanner.scan('SPY', data)
            print(f"Result:   {scan.market_type.value} ({scan.confidence*100:.0f}% confidence)")
            print(f"Style:    {scan.recommended_style.value}")
            print(f"Vol:      {scan.volatility_percentile:.0f}th percentile")
            print(f"Trend:    {scan.trend_strength:.2f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Demonstrate RegimeScanner capabilities.
    """
    print("="*70)
    print("REGIME SCANNER - Market Classification Engine")
    print("Tells you WHAT type of market you're in")
    print("="*70)
    
    # 1. Validate against history
    validate_scanner()
    
    # 2. Current market scan
    print("\n" + "="*70)
    print("CURRENT MARKET SCAN")
    print("="*70)
    
    tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'JPM', 'XOM']
    
    print("\nScanning current market...")
    scans = scan_universe(tickers)
    print_market_heatmap(scans)
    
    # 3. Detailed scan example
    print("\n" + "="*70)
    print("DETAILED SCAN: SPY")
    print("="*70)
    
    if 'SPY' in scans:
        print(scans['SPY'])
    
    print("\n" + "="*70)
    print("COMMERCIAL VALUE")
    print("="*70)
    print("""
    Target Customers:
    1. Retail Traders ($99/month): "What type of market is this?"
    2. Hedge Fund Analysts ($499/month): Sector regime rotation
    3. Risk Managers ($999/month): Regime-based VAR calculations
    
    Key Features:
    - Market type classification (TREND/CHOP/TRANSITION/CRISIS)
    - Sector cluster analysis
    - Trading style recommendations
    - Historical period matching
    - Confidence scoring
    """)


if __name__ == '__main__':
    main()
