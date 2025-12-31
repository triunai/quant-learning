"""
SECTOR-OPTIMIZED HIDDEN MARKOV MODEL
=====================================

This module implements sector-specific regime detection based on validated findings
from the 110-stock analysis (Phase 3).

KEY DISCOVERY:
- Regime duration varies 2X by sector (33 days for Tech vs 19 days for Utilities)
- 4 distinct behavioral clusters exist with different trading implications
- Sector matters MORE than individual stock characteristics

SECTOR CLUSTERS (from empirical analysis):
1. GROWTH CYCLES (30+ day regimes): Tech, Consumer, Materials
2. NEWS-DRIVEN (20d, jumpy): Healthcare, Industrials, Communication
3. ECONOMIC CYCLES (19-22d, stable): Financials, Energy, REITs, Utilities
4. SPECULATIVE (33d, extreme vol): High Volatility stocks

TRADING IMPLICATIONS:
- Tech: Trend-following, monthly rebalancing, wider stops
- Utilities: Mean-reversion, bi-weekly rebalancing, tighter stops
- High Vol: Event-driven, extreme position sizing discipline

Author: Project Iris Research Team
Date: 2025-12-31
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Try to import sklearn for HMM
try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("⚠️ sklearn not available - using simplified regime detection")


# ============================================================================
# SECTOR METADATA (Validated from 110-stock analysis)
# ============================================================================

SECTOR_CLUSTERS = {
    # Cluster 1: GROWTH CYCLES - Long regimes, trend-following works
    'Large Cap Tech': {
        'cluster': 1,
        'cluster_name': 'Growth Cycles',
        'avg_duration': 32,
        'std_duration': 28.7,
        'avg_kurtosis': 9.7,
        'avg_volatility': 0.404,
        'n_regimes': 4,           # More nuanced states for complex behavior
        'lookback_days': 60,      # Longer memory for monthly cycles
        'volatility_threshold': 0.025,
        'tail_profile': 'fat',
        'trading_style': 'trend_following',
        'rebalance_freq': 'monthly',
        'position_risk': 0.02,    # 2% risk per trade
        'etf': 'XLK',
        'tickers': ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'AMZN', 'NVDA', 'TSLA', 
                   'ORCL', 'CRM', 'ADBE', 'INTC', 'AMD', 'AVGO', 'CSCO', 'QCOM'],
    },
    'Consumer Discretionary': {
        'cluster': 1,
        'cluster_name': 'Growth Cycles',
        'avg_duration': 31,
        'std_duration': 21.6,
        'avg_kurtosis': 10.9,
        'avg_volatility': 0.267,
        'n_regimes': 4,
        'lookback_days': 55,
        'volatility_threshold': 0.022,
        'tail_profile': 'fat',
        'trading_style': 'trend_following',
        'rebalance_freq': 'monthly',
        'position_risk': 0.02,
        'etf': 'XLY',
        'tickers': ['WMT', 'TGT', 'COST', 'HD', 'NKE', 'SBUX', 'MCD', 'LOW', 'TJX', 'BKNG'],
    },
    'Consumer Staples': {
        'cluster': 1,
        'cluster_name': 'Growth Cycles',
        'avg_duration': 31,
        'std_duration': 33.8,
        'avg_kurtosis': 9.2,
        'avg_volatility': 0.192,
        'n_regimes': 3,           # Simpler than tech
        'lookback_days': 50,
        'volatility_threshold': 0.018,
        'tail_profile': 'normal',
        'trading_style': 'trend_following',
        'rebalance_freq': 'monthly',
        'position_risk': 0.025,
        'etf': 'XLP',
        'tickers': ['PG', 'KO', 'PEP', 'PM', 'MO', 'CL', 'KMB', 'GIS', 'K', 'CAG'],
    },
    'Materials': {
        'cluster': 1,
        'cluster_name': 'Growth Cycles',
        'avg_duration': 28,
        'std_duration': 7.5,
        'avg_kurtosis': 7.6,
        'avg_volatility': 0.297,
        'n_regimes': 3,
        'lookback_days': 45,
        'volatility_threshold': 0.022,
        'tail_profile': 'normal',
        'trading_style': 'trend_following',
        'rebalance_freq': 'monthly',
        'position_risk': 0.02,
        'etf': 'XLB',
        'tickers': ['LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'DOW', 'DD'],
    },
    
    # Cluster 2: NEWS-DRIVEN - Short regimes, jump-prone, need faster reactions
    'Healthcare': {
        'cluster': 2,
        'cluster_name': 'News-Driven',
        'avg_duration': 24,
        'std_duration': 16.1,
        'avg_kurtosis': 9.2,
        'avg_volatility': 0.251,
        'n_regimes': 3,
        'lookback_days': 40,
        'volatility_threshold': 0.020,
        'tail_profile': 'fat',
        'trading_style': 'event_driven',
        'rebalance_freq': 'biweekly',
        'position_risk': 0.015,   # Lower because of jump risk
        'etf': 'XLV',
        'tickers': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'LLY', 'BMY', 'TMO', 'DHR', 'AMGN'],
    },
    'Industrials': {
        'cluster': 2,
        'cluster_name': 'News-Driven',
        'avg_duration': 20,
        'std_duration': 4.2,
        'avg_kurtosis': 10.6,
        'avg_volatility': 0.280,
        'n_regimes': 3,
        'lookback_days': 35,
        'volatility_threshold': 0.022,
        'tail_profile': 'fat',
        'trading_style': 'event_driven',
        'rebalance_freq': 'biweekly',
        'position_risk': 0.015,
        'etf': 'XLI',
        'tickers': ['CAT', 'BA', 'GE', 'HON', 'UPS', 'DE', 'MMM', 'RTX', 'LMT', 'FDX'],
    },
    'Communication': {
        'cluster': 2,
        'cluster_name': 'News-Driven',
        'avg_duration': 20,
        'std_duration': 9.7,
        'avg_kurtosis': 14.9,     # HIGHEST kurtosis - extreme fat tails!
        'avg_volatility': 0.280,
        'n_regimes': 4,           # Need more states for this complexity
        'lookback_days': 35,
        'volatility_threshold': 0.025,
        'tail_profile': 'extreme_fat',
        'trading_style': 'event_driven',
        'rebalance_freq': 'biweekly',
        'position_risk': 0.012,   # Lowest because of extreme tails
        'etf': 'XLC',
        'tickers': ['VZ', 'T', 'TMUS', 'DIS', 'NFLX', 'CMCSA'],
    },
    
    # Cluster 3: ECONOMIC CYCLES - Stable behavior, normal tails
    'Financials': {
        'cluster': 3,
        'cluster_name': 'Economic Cycles',
        'avg_duration': 27,
        'std_duration': 13.4,
        'avg_kurtosis': 4.9,
        'avg_volatility': 0.283,
        'n_regimes': 3,
        'lookback_days': 45,
        'volatility_threshold': 0.022,
        'tail_profile': 'normal',
        'trading_style': 'cycle_timing',
        'rebalance_freq': 'monthly',
        'position_risk': 0.02,
        'etf': 'XLF',
        'tickers': ['JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'BLK', 'SCHW', 'AXP', 'USB'],
    },
    'Energy': {
        'cluster': 3,
        'cluster_name': 'Economic Cycles',
        'avg_duration': 22,
        'std_duration': 5.0,
        'avg_kurtosis': 2.9,      # Lowest kurtosis!
        'avg_volatility': 0.338,
        'n_regimes': 3,
        'lookback_days': 40,
        'volatility_threshold': 0.028,
        'tail_profile': 'normal',
        'trading_style': 'cycle_timing',
        'rebalance_freq': 'monthly',
        'position_risk': 0.018,
        'etf': 'XLE',
        'tickers': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY', 'MPC', 'VLO', 'PSX', 'HAL'],
    },
    'REITs': {
        'cluster': 3,
        'cluster_name': 'Economic Cycles',
        'avg_duration': 19,
        'std_duration': 3.0,
        'avg_kurtosis': 3.0,
        'avg_volatility': 0.255,
        'n_regimes': 3,
        'lookback_days': 30,
        'volatility_threshold': 0.020,
        'tail_profile': 'normal',
        'trading_style': 'mean_reversion',
        'rebalance_freq': 'biweekly',
        'position_risk': 0.025,
        'etf': 'XLRE',
        'tickers': ['O', 'AMT', 'PLD', 'SPG', 'EQIX', 'PSA', 'DLR'],
    },
    'Utilities': {
        'cluster': 3,
        'cluster_name': 'Economic Cycles',
        'avg_duration': 19,
        'std_duration': 5.5,
        'avg_kurtosis': 7.4,
        'avg_volatility': 0.213,
        'n_regimes': 3,
        'lookback_days': 28,      # Shortest lookback - fast cycles
        'volatility_threshold': 0.015,
        'tail_profile': 'normal',
        'trading_style': 'mean_reversion',
        'rebalance_freq': 'biweekly',
        'position_risk': 0.03,    # Higher because less volatile
        'etf': 'XLU',
        'tickers': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'XEL', 'SRE'],
    },
    'Indices/ETFs': {
        'cluster': 3,
        'cluster_name': 'Economic Cycles',
        'avg_duration': 22,
        'std_duration': 3.9,
        'avg_kurtosis': 5.5,
        'avg_volatility': 0.189,
        'n_regimes': 3,
        'lookback_days': 35,
        'volatility_threshold': 0.018,
        'tail_profile': 'normal',
        'trading_style': 'trend_following',
        'rebalance_freq': 'monthly',
        'position_risk': 0.02,
        'etf': 'SPY',
        'tickers': ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI'],
    },
    
    # Cluster 4: SPECULATIVE - Extreme behavior, special handling
    'High Volatility': {
        'cluster': 4,
        'cluster_name': 'Speculative',
        'avg_duration': 33,
        'std_duration': 26.4,
        'avg_kurtosis': 8.2,
        'avg_volatility': 0.988,  # 99% annualized vol!
        'n_regimes': 5,           # Need more states for extreme behavior
        'lookback_days': 90,      # Long memory for crashes
        'volatility_threshold': 0.05,
        'tail_profile': 'extreme_fat',
        'trading_style': 'momentum_with_stops',
        'rebalance_freq': 'weekly',
        'position_risk': 0.01,    # Only 1% risk - extreme discipline
        'etf': None,              # No ETF for this cluster
        'tickers': ['COIN', 'MARA', 'RIOT', 'GME', 'AMC'],
    },
}


# ============================================================================
# SECTOR LOOKUP UTILITIES
# ============================================================================

def get_sector(ticker: str) -> Optional[str]:
    """Get sector for a given ticker."""
    ticker = ticker.upper()
    for sector, meta in SECTOR_CLUSTERS.items():
        if ticker in meta['tickers']:
            return sector
    return None

def get_sector_params(sector: str) -> Dict:
    """Get parameters for a given sector."""
    if sector in SECTOR_CLUSTERS:
        return SECTOR_CLUSTERS[sector]
    # Default to Large Cap Tech params
    return SECTOR_CLUSTERS['Large Cap Tech']

def get_cluster(ticker: str) -> Optional[int]:
    """Get behavioral cluster for a ticker (1-4)."""
    sector = get_sector(ticker)
    if sector:
        return SECTOR_CLUSTERS[sector]['cluster']
    return None


# ============================================================================
# SECTOR-OPTIMIZED HMM
# ============================================================================

class SectorOptimizedHMM:
    """
    Hidden Markov Model with sector-specific parameters.
    
    This class implements regime detection that adapts based on sector characteristics:
    - Tech stocks: 4 regimes, 60-day lookback, trend-following signals
    - Utilities: 3 regimes, 28-day lookback, mean-reversion signals
    - High Volatility: 5 regimes, 90-day lookback, extreme stops
    
    The key insight: SECTOR DETERMINES REGIME BEHAVIOR, not individual stock characteristics.
    """
    
    def __init__(self, ticker: str, sector: str = None):
        """
        Initialize sector-optimized HMM.
        
        Args:
            ticker: Stock symbol
            sector: Override sector (if None, will auto-detect)
        """
        self.ticker = ticker.upper()
        
        # Auto-detect sector if not provided
        self.sector = sector or get_sector(self.ticker)
        if self.sector is None:
            print(f"⚠️ Unknown sector for {ticker}, using 'Large Cap Tech' defaults")
            self.sector = 'Large Cap Tech'
        
        # Get sector-specific parameters
        self.params = get_sector_params(self.sector)
        
        # Set model parameters from sector profile
        self.n_regimes = self.params['n_regimes']
        self.lookback_days = self.params['lookback_days']
        self.volatility_threshold = self.params['volatility_threshold']
        
        # State
        self.data = None
        self.regimes = None
        self.regime_stats = {}
        self.current_regime = None
        self.regime_durations = {}
        
        print(f"📊 Initialized {self.ticker}")
        print(f"   Sector: {self.sector} ({self.params['cluster_name']})")
        print(f"   Regimes: {self.n_regimes}, Lookback: {self.lookback_days}d")
        print(f"   Expected duration: {self.params['avg_duration']}d")
        print(f"   Trading style: {self.params['trading_style']}")
    
    def fetch_data(self, years: int = 5) -> pd.DataFrame:
        """Fetch and prepare historical data."""
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(years=years)
        
        data = yf.download(self.ticker, start=start_date, end=end_date, progress=False)
        
        # Handle multi-level columns from yfinance (Price, Ticker)
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten: use just the price level
            data.columns = data.columns.get_level_values(0)
        
        if len(data) < 252:
            raise ValueError(f"Insufficient data for {self.ticker}")
        
        self.data = pd.DataFrame(index=data.index)
        
        # Handle both old (Adj Close) and new (Close with auto_adjust=True) yfinance
        if 'Adj Close' in data.columns:
            self.data['close'] = data['Adj Close']
        else:
            self.data['close'] = data['Close']
        
        self.data['log_ret'] = np.log(self.data['close'] / self.data['close'].shift(1))
        self.data['volatility'] = self.data['log_ret'].rolling(self.lookback_days).std() * np.sqrt(252)
        self.data = self.data.dropna()
        
        print(f"   Loaded {len(self.data)} days of data")
        return self.data
    
    def build_features(self) -> np.ndarray:
        """
        Build feature matrix for regime detection.
        
        Features are sector-aware:
        - All sectors: volatility, momentum
        - Fat-tail sectors: Add jump indicator
        - Growth sectors: Add trend strength
        """
        df = self.data.copy()
        
        # Base features (all sectors)
        df['vol_z'] = (df['volatility'] - df['volatility'].mean()) / df['volatility'].std()
        df['momentum'] = df['log_ret'].rolling(self.lookback_days).mean() * np.sqrt(252)
        df['mom_z'] = (df['momentum'] - df['momentum'].mean()) / df['momentum'].std()
        
        features = ['vol_z', 'mom_z']
        
        # Add jump indicator for fat-tail sectors
        if self.params['tail_profile'] in ['fat', 'extreme_fat']:
            df['jump'] = (np.abs(df['log_ret']) > 3 * df['log_ret'].std()).astype(float)
            df['jump_rolling'] = df['jump'].rolling(self.lookback_days).mean()
            features.append('jump_rolling')
        
        # Add trend strength for growth sectors
        if self.params['cluster'] == 1:  # Growth Cycles
            df['trend'] = df['close'].pct_change(self.lookback_days)
            df['trend_z'] = (df['trend'] - df['trend'].mean()) / df['trend'].std()
            features.append('trend_z')
        
        df = df.dropna()
        self.data = df
        
        X = df[features].values
        return X
    
    def fit_regimes(self) -> np.ndarray:
        """
        Fit regime model using Gaussian Mixture.
        
        Returns array of regime labels.
        """
        if not SKLEARN_OK:
            return self._fallback_regime_detection()
        
        X = self.build_features()
        
        # Fit Gaussian Mixture Model
        gmm = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            n_init=10,
            random_state=42
        )
        
        self.regimes = gmm.fit_predict(X)
        self.data['regime'] = self.regimes
        
        # Name regimes by characteristics
        self._name_regimes()
        
        # Compute regime durations
        self._compute_durations()
        
        return self.regimes
    
    def _fallback_regime_detection(self) -> np.ndarray:
        """Simple volatility-based regime detection if sklearn unavailable."""
        vol = self.data['volatility'].values
        
        # Simple percentile-based regimes
        thresholds = np.percentile(vol, [33, 66])
        
        regimes = np.zeros(len(vol), dtype=int)
        regimes[vol > thresholds[1]] = 2  # High vol
        regimes[(vol > thresholds[0]) & (vol <= thresholds[1])] = 1  # Medium
        # Low vol stays at 0
        
        self.regimes = regimes
        self.data['regime'] = regimes
        
        self.regime_stats = {
            0: {'name': 'Low Vol'},
            1: {'name': 'Normal'},
            2: {'name': 'High Vol'},
        }
        
        return regimes
    
    def _name_regimes(self):
        """Name regimes based on their statistical characteristics."""
        for r in range(self.n_regimes):
            mask = self.data['regime'] == r
            regime_data = self.data[mask]
            
            if len(regime_data) == 0:
                continue
            
            avg_ret = regime_data['log_ret'].mean() * 252  # Annualized
            avg_vol = regime_data['volatility'].mean()
            sharpe = avg_ret / avg_vol if avg_vol > 0 else 0
            
            # Name based on characteristics
            if sharpe > 0.5:
                name = 'Bull'
            elif sharpe < -0.5:
                name = 'Bear'
            elif avg_vol > self.volatility_threshold * 1.5:
                name = 'Crisis'
            elif avg_vol < self.volatility_threshold * 0.7:
                name = 'Calm'
            else:
                name = 'Neutral'
            
            self.regime_stats[r] = {
                'name': name,
                'avg_return': avg_ret,
                'avg_volatility': avg_vol,
                'sharpe': sharpe,
                'days': len(regime_data),
                'pct_of_time': len(regime_data) / len(self.data),
            }
    
    def _compute_durations(self):
        """Compute empirical regime duration distributions."""
        regimes = self.data['regime'].values
        
        for r in range(self.n_regimes):
            durations = []
            current_run = 0
            
            for reg in regimes:
                if reg == r:
                    current_run += 1
                else:
                    if current_run > 0:
                        durations.append(current_run)
                    current_run = 0
            
            if current_run > 0:
                durations.append(current_run)
            
            if durations:
                self.regime_durations[r] = {
                    'mean': np.mean(durations),
                    'std': np.std(durations),
                    'median': np.median(durations),
                    'samples': durations,
                }
    
    def get_current_state(self) -> Dict:
        """Get current regime state and trading signal."""
        if self.regimes is None:
            raise ValueError("Must call fit_regimes() first")
        
        current_regime = self.regimes[-1]
        regime_stats = self.regime_stats.get(current_regime, {})
        
        # Calculate days in current regime
        days_in_regime = 0
        for r in reversed(self.regimes):
            if r == current_regime:
                days_in_regime += 1
            else:
                break
        
        expected_duration = self.params['avg_duration']
        regime_progress = days_in_regime / expected_duration
        
        # Generate trading signal based on sector style
        signal = self._generate_signal(current_regime, days_in_regime, regime_progress)
        
        return {
            'ticker': self.ticker,
            'sector': self.sector,
            'cluster': self.params['cluster_name'],
            'current_regime': current_regime,
            'regime_name': regime_stats.get('name', 'Unknown'),
            'days_in_regime': days_in_regime,
            'expected_duration': expected_duration,
            'regime_progress': regime_progress,
            'signal': signal,
            'position_risk': self.params['position_risk'],
            'trading_style': self.params['trading_style'],
        }
    
    def _generate_signal(self, regime: int, days_in_regime: int, progress: float) -> Dict:
        """Generate trading signal based on sector style and regime state."""
        regime_name = self.regime_stats.get(regime, {}).get('name', 'Unknown')
        trading_style = self.params['trading_style']
        
        signal = {
            'action': 'HOLD',
            'strength': 0.0,
            'reason': '',
        }
        
        if trading_style == 'trend_following':
            # Trend following: Enter early in bull regimes, exit late
            if regime_name == 'Bull' and progress < 0.5:
                signal = {'action': 'BUY', 'strength': 1 - progress, 
                         'reason': 'Early in Bull regime - ride the trend'}
            elif regime_name == 'Bear' and progress > 0.3:
                signal = {'action': 'SELL', 'strength': progress,
                         'reason': 'Bear regime maturing - protect capital'}
            elif regime_name == 'Crisis':
                signal = {'action': 'EXIT', 'strength': 1.0,
                         'reason': 'Crisis regime - risk off'}
        
        elif trading_style == 'mean_reversion':
            # Mean reversion: Buy at regime extremes
            if regime_name == 'Bear' and progress > 0.8:
                signal = {'action': 'BUY', 'strength': progress - 0.5,
                         'reason': 'Bear regime exhausted - expect reversal'}
            elif regime_name == 'Bull' and progress > 0.8:
                signal = {'action': 'TAKE_PROFIT', 'strength': progress - 0.5,
                         'reason': 'Bull regime mature - take profits'}
        
        elif trading_style == 'event_driven':
            # Event-driven: React to regime changes quickly
            if regime_name == 'Bull' and days_in_regime <= 5:
                signal = {'action': 'BUY', 'strength': 0.8,
                         'reason': 'Fresh Bull regime - momentum entry'}
            elif regime_name == 'Bear' and days_in_regime <= 3:
                signal = {'action': 'SELL', 'strength': 0.9,
                         'reason': 'Bear regime started - quick exit'}
        
        return signal
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"\n{'='*60}",
            f"SECTOR-OPTIMIZED HMM: {self.ticker}",
            f"{'='*60}",
            f"Sector: {self.sector} ({self.params['cluster_name']})",
            f"Regimes: {self.n_regimes} | Lookback: {self.lookback_days}d",
            f"",
            "REGIME SUMMARY:",
        ]
        
        for r, stats in sorted(self.regime_stats.items()):
            dur = self.regime_durations.get(r, {})
            lines.append(
                f"  {stats['name']:8s}: {stats['pct_of_time']*100:5.1f}% of time, "
                f"ret={stats['avg_return']*100:+5.1f}%, "
                f"dur={dur.get('mean', 0):.1f}d"
            )
        
        state = self.get_current_state()
        lines.extend([
            "",
            "CURRENT STATE:",
            f"  Regime: {state['regime_name']} (day {state['days_in_regime']})",
            f"  Progress: {state['regime_progress']*100:.0f}% of expected duration",
            f"  Signal: {state['signal']['action']} ({state['signal']['reason']})",
            f"  Position risk: {state['position_risk']*100:.1f}% per trade",
        ])
        
        return '\n'.join(lines)


# ============================================================================
# SECTOR COMPARISON TEST
# ============================================================================

def compare_sector_regimes(ticker1: str, ticker2: str):
    """
    Compare regime behavior between two tickers from different sectors.
    
    This demonstrates that sector determines regime behavior:
    - Tech (AAPL): Should have ~32 day regimes, 4 states, fat tails
    - Utilities (NEE): Should have ~19 day regimes, 3 states, normal tails
    """
    print("\n" + "="*70)
    print("SECTOR COMPARISON: Does Sector Determine Regime Behavior?")
    print("="*70)
    
    # Analyze first ticker
    hmm1 = SectorOptimizedHMM(ticker1)
    hmm1.fetch_data()
    hmm1.fit_regimes()
    
    print(hmm1.summary())
    
    # Analyze second ticker
    hmm2 = SectorOptimizedHMM(ticker2)
    hmm2.fetch_data()
    hmm2.fit_regimes()
    
    print(hmm2.summary())
    
    # Comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    avg_dur1 = np.mean([d['mean'] for d in hmm1.regime_durations.values()])
    avg_dur2 = np.mean([d['mean'] for d in hmm2.regime_durations.values()])
    
    print(f"\n{ticker1} ({hmm1.sector}):")
    print(f"  Avg regime duration: {avg_dur1:.1f} days")
    print(f"  Expected from sector: {hmm1.params['avg_duration']} days")
    print(f"  Kurtosis: {stats.kurtosis(hmm1.data['log_ret']):.2f}")
    
    print(f"\n{ticker2} ({hmm2.sector}):")
    print(f"  Avg regime duration: {avg_dur2:.1f} days")
    print(f"  Expected from sector: {hmm2.params['avg_duration']} days")
    print(f"  Kurtosis: {stats.kurtosis(hmm2.data['log_ret']):.2f}")
    
    # Validate hypothesis
    print("\n" + "-"*50)
    if avg_dur1 > avg_dur2:
        ratio = avg_dur1 / avg_dur2
        print(f"✓ {ticker1} has {ratio:.1f}x LONGER regimes than {ticker2}")
        print(f"  This matches sector expectations (Tech > Utilities)")
    else:
        ratio = avg_dur2 / avg_dur1
        print(f"⚠️ {ticker2} has {ratio:.1f}x LONGER regimes than {ticker1}")
        print(f"  This is OPPOSITE to sector expectations")
    
    return hmm1, hmm2


# ============================================================================
# QUICK SECTOR ROTATION BACKTEST
# ============================================================================

def quick_sector_rotation_backtest(start_year: int = 2019, end_year: int = 2024):
    """
    Quick backtest: Rotate into sectors that are in Bull regimes.
    
    Strategy:
    1. Each month, identify which sector ETFs are in Bull regime
    2. Allocate equally to Bull regime sectors
    3. Hold cash if no sectors in Bull
    
    Compare to: Equal-weight buy-and-hold of all sectors
    """
    print("\n" + "="*70)
    print("SECTOR ROTATION BACKTEST")
    print(f"Period: {start_year} to {end_year}")
    print("="*70)
    
    # Sector ETFs to test
    sector_etfs = ['XLK', 'XLY', 'XLP', 'XLV', 'XLF', 'XLE', 'XLI', 'XLU', 'XLC']
    
    # Download data
    print("\nFetching sector ETF data...")
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    
    prices = pd.DataFrame()
    for etf in sector_etfs:
        data = yf.download(etf, start=start_date, end=end_date, progress=False)
        # Handle multi-level columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        # Use Adj Close if available, otherwise Close (auto_adjust=True)
        if 'Adj Close' in data.columns:
            prices[etf] = data['Adj Close']
        else:
            prices[etf] = data['Close']
    
    prices = prices.dropna()
    returns = prices.pct_change().dropna()
    
    print(f"Loaded {len(prices)} days of data")
    
    # Simple regime detection for backtest (monthly rebalancing)
    # Use 60-day rolling volatility and momentum
    regime_signals = pd.DataFrame(index=returns.index, columns=sector_etfs)
    
    for etf in sector_etfs:
        # Calculate rolling metrics
        vol = returns[etf].rolling(60).std() * np.sqrt(252)
        mom = returns[etf].rolling(60).mean() * 252
        sharpe_rolling = mom / vol
        
        # Bull regime: positive rolling Sharpe, below-average vol
        vol_median = vol.rolling(252).median()
        is_bull = (sharpe_rolling > 0.3) & (vol < vol_median * 1.2)
        
        regime_signals[etf] = is_bull.astype(float)
    
    # Monthly rebalancing
    monthly_signals = regime_signals.resample('M').last()
    
    # Calculate strategy returns
    strategy_returns = []
    
    for i in range(1, len(monthly_signals)):
        month_start = monthly_signals.index[i-1]
        month_end = monthly_signals.index[i]
        
        # Get signals from previous month
        signals = monthly_signals.iloc[i-1]
        n_bull = signals.sum()
        
        if n_bull > 0:
            # Equal weight across Bull sectors
            weights = signals / n_bull
        else:
            # All cash (0 return)
            weights = pd.Series(0, index=sector_etfs)
        
        # Calculate monthly return
        month_returns = returns.loc[month_start:month_end]
        portfolio_return = (month_returns * weights).sum(axis=1).sum()
        strategy_returns.append(portfolio_return)
    
    # Calculate benchmark (equal weight buy-and-hold)
    benchmark_weights = pd.Series(1/len(sector_etfs), index=sector_etfs)
    benchmark_returns = []
    
    for i in range(1, len(monthly_signals)):
        month_start = monthly_signals.index[i-1]
        month_end = monthly_signals.index[i]
        
        month_returns = returns.loc[month_start:month_end]
        portfolio_return = (month_returns * benchmark_weights).sum(axis=1).sum()
        benchmark_returns.append(portfolio_return)
    
    # Statistics
    strategy_cum = np.cumprod(1 + np.array(strategy_returns)) - 1
    benchmark_cum = np.cumprod(1 + np.array(benchmark_returns)) - 1
    
    strategy_annual = np.mean(strategy_returns) * 12
    benchmark_annual = np.mean(benchmark_returns) * 12
    
    strategy_vol = np.std(strategy_returns) * np.sqrt(12)
    benchmark_vol = np.std(benchmark_returns) * np.sqrt(12)
    
    strategy_sharpe = strategy_annual / strategy_vol if strategy_vol > 0 else 0
    benchmark_sharpe = benchmark_annual / benchmark_vol if benchmark_vol > 0 else 0
    
    print("\n" + "-"*50)
    print("RESULTS:")
    print("-"*50)
    print(f"\n{'Metric':<25} {'Strategy':>12} {'Benchmark':>12}")
    print("-"*50)
    print(f"{'Total Return':<25} {strategy_cum[-1]*100:>11.1f}% {benchmark_cum[-1]*100:>11.1f}%")
    print(f"{'Annualized Return':<25} {strategy_annual*100:>11.1f}% {benchmark_annual*100:>11.1f}%")
    print(f"{'Annualized Volatility':<25} {strategy_vol*100:>11.1f}% {benchmark_vol*100:>11.1f}%")
    print(f"{'Sharpe Ratio':<25} {strategy_sharpe:>12.2f} {benchmark_sharpe:>12.2f}")
    
    outperformance = strategy_annual - benchmark_annual
    print(f"\n{'Outperformance':<25} {outperformance*100:>11.1f}%")
    
    if strategy_sharpe > benchmark_sharpe:
        print("\n✓ SECTOR ROTATION OUTPERFORMS on risk-adjusted basis")
    else:
        print("\n⚠️ SECTOR ROTATION UNDERPERFORMS - needs refinement")
    
    return strategy_returns, benchmark_returns


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run complete sector analysis suite."""
    
    print("="*70)
    print("SECTOR-OPTIMIZED REGIME DETECTION SUITE")
    print("Based on 110-Stock Validation Analysis")
    print("="*70)
    
    # Test 1: Compare Tech vs Utilities
    print("\n\n" + "#"*70)
    print("# TEST 1: SECTOR COMPARISON (AAPL vs NEE)")
    print("#"*70)
    hmm_aapl, hmm_nee = compare_sector_regimes('AAPL', 'NEE')
    
    # Test 2: Compare within same cluster
    print("\n\n" + "#"*70)
    print("# TEST 2: WITHIN-CLUSTER COMPARISON (AAPL vs MSFT)")
    print("#"*70)
    hmm_aapl2, hmm_msft = compare_sector_regimes('AAPL', 'MSFT')
    
    # Test 3: Quick backtest
    print("\n\n" + "#"*70)
    print("# TEST 3: SECTOR ROTATION BACKTEST")
    print("#"*70)
    strategy_rets, benchmark_rets = quick_sector_rotation_backtest()
    
    # Summary for finance expert
    print("\n\n" + "="*70)
    print("SUMMARY FOR FINANCE EXPERT")
    print("="*70)
    
    summary = """
KEY FINDINGS FROM SECTOR ANALYSIS:

1. REGIME DURATION VARIES 2X BY SECTOR
   - Tech: 32 days average (monthly cycles)
   - Utilities: 19 days average (3-week cycles)
   - High Vol: 33 days but with extreme variance

2. FOUR BEHAVIORAL CLUSTERS
   - Cluster 1 (Growth): Tech, Consumer - trend-following works
   - Cluster 2 (News): Healthcare, Comms - event-driven, jump-prone
   - Cluster 3 (Economic): Financials, Energy - cycle timing
   - Cluster 4 (Speculative): COIN, GME - extreme discipline needed

3. TRADING IMPLICATIONS
   - Tech: Use 60-day lookback, 4 regimes, monthly rebalancing
   - Utilities: Use 28-day lookback, 3 regimes, bi-weekly rebalancing
   - Position sizing: 2% risk for Tech, 3% for Utilities, 1% for High Vol

4. SECTOR ROTATION POTENTIAL
   - Simple signal: Rotate into sectors with positive rolling Sharpe
   - Monthly rebalancing vs buy-and-hold comparison above

NEXT STEPS:
- Refine sector rotation signals
- Add regime-based position sizing
- Build walk-forward validation framework
"""
    print(summary)
    
    return hmm_aapl, hmm_nee


if __name__ == '__main__':
    hmm_aapl, hmm_nee = main()
