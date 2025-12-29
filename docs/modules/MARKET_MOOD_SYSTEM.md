# 🎯 Market Mood Detection System - 1-Day MVP Plan

> **Purpose**: A focused, battle-tested implementation plan for building a market mood detection system with regime fatigue prediction in **one day**.
>
> **Status**: REVISED after AI research audit. Original scope was over-engineered.

---

## ⚠️ Key Audit Findings (From AI Research)

### 🪤 Traps Identified & Avoided

| Original Plan | Problem | Resolution |
|---------------|---------|------------|
| Fear & Greed Index API | CNN's FGI has no clean free API. `alternative.me` is crypto-focused (404'd). **Double-counting**: CNN FGI already uses VIX + Put/Call internally! | ❌ **DROP IT**. Build internal FGI-like composite from raw signals. |
| Multi-source RSS crawling | Paywalls, ToS issues, brittle scraping. Reuters doesn't expose clean RSS. FT requires login. | ❌ **OPTIONAL**. Treat as degradable modifier, not core signal. |
| FinBERT/Transformers NLP | Heavy, slow, GPU-preferred. Overkill for Day-1. | ❌ **DEFER**. Use keyword-based sentiment if news is added. |
| Academic fatigue calibration | Backtesting + empirical rate tuning is Phase 3+ work. | ⚠️ **SIMPLIFY**. Use heuristic fatigue with honest labeling. |

### ✅ What Survives

- **VIX** → Direct fear/greed proxy (no double-counting)
- **Cboe Put/Call Ratio** → Official source, reliable
- **SPY Returns (1D, 5D)** → Trend context
- **Volume Ratio** → Participation/capitulation signal
- **Regime Fatigue** → Your unique insight (keep as heuristic)

---

## 📐 Revised System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│            MARKET MOOD DETECTION SYSTEM (1-Day MVP)                     │
├─────────────────────────────────────────────────────────────────────────┤
│  P0 SIGNALS (Build Today):                                              │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────────────┐  │
│  │ VIX Level    │ │ Put/Call     │ │ SPY Returns  │ │ Volume Ratio  │  │
│  │ (yfinance)   │ │ (Cboe Daily) │ │ (yfinance)   │ │ (yfinance)    │  │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └───────┬───────┘  │
│         └────────────────┴────────────────┴─────────────────┘          │
│                              ↓                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    SIGNAL AGGREGATOR                              │  │
│  │  • Weighted scoring (no double-counting)                          │  │
│  │  • Regime classification (PANIC → EXTREME_GREED)                  │  │
│  │  • Confidence = coverage + agreement                              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              ↓                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                 REGIME FATIGUE TRACKER                            │  │
│  │  • Days in current regime                                         │  │
│  │  • Heuristic fatigue % (monotonic mapping)                        │  │
│  │  • Shift probability warning                                      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              ↓                                          │
│  OUTPUT: "PANIC (Day 3) - Fatigue: 65% - High shift probability"       │
│                                                                         │
│  P1 OPTIONAL (If Time Allows):                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  News Sentiment (MarketWatch RSS only, keyword-based, degradable) │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 1-Day Success Definition

**If you complete ONLY this, you've won:**

### Must-Have Output (Console or JSON)

```json
{
  "timestamp": "2025-12-29T16:00:00+08:00",
  "signals": {
    "vix": 18.5,
    "put_call_ratio": 0.85,
    "spy_1d_return": 0.3,
    "spy_5d_return": 1.2,
    "volume_ratio": 1.1
  },
  "mood_regime": "NEUTRAL",
  "score": 12,
  "fatigue": {
    "day_count": 5,
    "fatigue_pct": 35,
    "shift_probability": "LOW"
  },
  "confidence": {
    "coverage": "4/4 signals",
    "agreement": "HIGH"
  }
}
```

### Nice-to-Have (Only if time remains)
- `news_sentiment` as a small modifier (-0.1 to +0.1)

---

## 🔥 Implementation Order (Build in This Sequence)

### Step 1: Market Data Fetcher (P0) - ~1 hour

```python
# signals_factory/market_mood/data_fetcher.py

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional

@dataclass
class MarketSignals:
    """Immutable container for market mood signals"""
    timestamp: datetime
    vix: float
    put_call_ratio: Optional[float]  # May be unavailable
    spy_1d_return: float
    spy_5d_return: float
    volume_ratio: float
    
    @property
    def coverage(self) -> str:
        """How many signals are available"""
        available = sum([
            self.vix is not None,
            self.put_call_ratio is not None,
            self.spy_1d_return is not None,
            self.spy_5d_return is not None,
            self.volume_ratio is not None
        ])
        return f"{available}/5 signals"


class MarketDataFetcher:
    """
    Fetches core market signals for mood detection.
    Uses yfinance for VIX, SPY, volume.
    """
    
    def __init__(self, volume_lookback_days: int = 20):
        self.volume_lookback = volume_lookback_days
    
    def fetch_signals(self) -> MarketSignals:
        """Fetch all market signals with same-timestamp alignment"""
        now = datetime.now()
        
        # VIX
        vix = self._fetch_vix()
        
        # SPY returns and volume
        spy_1d, spy_5d, volume_ratio = self._fetch_spy_data()
        
        # Put/Call ratio (optional - may need manual or Cboe scrape)
        put_call = self._fetch_put_call_ratio()
        
        return MarketSignals(
            timestamp=now,
            vix=vix,
            put_call_ratio=put_call,
            spy_1d_return=spy_1d,
            spy_5d_return=spy_5d,
            volume_ratio=volume_ratio
        )
    
    def _fetch_vix(self) -> float:
        """Fetch current VIX level"""
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1d")
            return hist['Close'].iloc[-1] if not hist.empty else 20.0
        except Exception as e:
            print(f"VIX fetch error: {e}")
            return 20.0  # Neutral fallback
    
    def _fetch_spy_data(self) -> tuple[float, float, float]:
        """Fetch SPY returns and volume ratio"""
        try:
            spy = yf.Ticker("SPY")
            hist = spy.history(period="1mo")
            
            if len(hist) < 6:
                return 0.0, 0.0, 1.0
            
            # Returns
            spy_1d = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100
            spy_5d = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-6]) - 1) * 100
            
            # Volume ratio (today vs 20-day average)
            avg_volume = hist['Volume'].iloc[-self.volume_lookback:].mean()
            today_volume = hist['Volume'].iloc[-1]
            volume_ratio = today_volume / avg_volume if avg_volume > 0 else 1.0
            
            return spy_1d, spy_5d, volume_ratio
        except Exception as e:
            print(f"SPY fetch error: {e}")
            return 0.0, 0.0, 1.0
    
    def _fetch_put_call_ratio(self) -> Optional[float]:
        """
        Fetch Put/Call ratio.
        
        NOTE: Cboe publishes daily at https://www.cboe.com/us/options/market_statistics/daily/
        For Day-1: Return None (degradable) or hardcode recent value.
        Phase 2: Implement Cboe daily stats scraper.
        """
        # TODO: Implement Cboe scraper in Phase 2
        # For now, return None (signal will be marked as missing)
        return None
```

### Step 2: Regime Fatigue Calculator (P0) - ~1 hour

```python
# signals_factory/market_mood/fatigue_calculator.py

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import json

@dataclass
class FatigueState:
    """Tracks regime duration and fatigue"""
    current_regime: str
    regime_start: datetime
    day_count: int
    fatigue_pct: int
    shift_probability: str  # LOW, MEDIUM, HIGH, IMMINENT


class RegimeFatigueCalculator:
    """
    YOUR UNIQUE INSIGHT: Markets get tired of extreme states.
    
    This is HEURISTIC fatigue - honestly labeled, not statistically calibrated.
    Calibration is Phase 3+ work after backtesting.
    """
    
    # Heuristic fatigue rates (days to reach 50% fatigue)
    # Lower number = faster fatigue
    FATIGUE_HALF_LIFE = {
        'EXTREME_PANIC': 3,     # Panic burns out fast
        'PANIC': 5,
        'FEAR': 7,
        'NEUTRAL': 15,          # Neutral is stable
        'OPTIMISM': 7,
        'GREED': 5,
        'EXTREME_GREED': 3,     # FOMO burns out fast
    }
    
    def __init__(self, state_file: Optional[str] = None):
        """
        state_file: Path to persist state across restarts (SQLite or JSON)
        """
        self.state_file = state_file
        self._current_regime: Optional[str] = None
        self._regime_start: Optional[datetime] = None
        
        if state_file:
            self._load_state()
    
    def update_regime(self, new_regime: str) -> FatigueState:
        """
        Update current regime and calculate fatigue.
        Call this each time you get new signals.
        """
        now = datetime.now()
        
        # Regime changed?
        if new_regime != self._current_regime:
            self._current_regime = new_regime
            self._regime_start = now
            if self.state_file:
                self._save_state()
        
        # Calculate fatigue
        day_count = (now - self._regime_start).days if self._regime_start else 0
        fatigue_pct = self._calculate_fatigue(new_regime, day_count)
        shift_prob = self._get_shift_probability(fatigue_pct)
        
        return FatigueState(
            current_regime=new_regime,
            regime_start=self._regime_start,
            day_count=day_count,
            fatigue_pct=fatigue_pct,
            shift_probability=shift_prob
        )
    
    def _calculate_fatigue(self, regime: str, day_count: int) -> int:
        """
        Monotonic fatigue mapping.
        Uses exponential approach to 100% based on regime-specific half-life.
        
        fatigue = 100 * (1 - 0.5^(days / half_life))
        """
        half_life = self.FATIGUE_HALF_LIFE.get(regime, 10)
        
        # Exponential saturation formula
        fatigue = 100 * (1 - (0.5 ** (day_count / half_life)))
        
        return min(100, max(0, int(fatigue)))
    
    def _get_shift_probability(self, fatigue_pct: int) -> str:
        """Map fatigue % to human-readable shift probability"""
        if fatigue_pct >= 80:
            return "IMMINENT"
        elif fatigue_pct >= 60:
            return "HIGH"
        elif fatigue_pct >= 40:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _save_state(self):
        """Persist state to file"""
        if not self.state_file:
            return
        state = {
            'current_regime': self._current_regime,
            'regime_start': self._regime_start.isoformat() if self._regime_start else None
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f)
    
    def _load_state(self):
        """Load state from file"""
        if not self.state_file:
            return
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                self._current_regime = state.get('current_regime')
                start_str = state.get('regime_start')
                self._regime_start = datetime.fromisoformat(start_str) if start_str else None
        except (FileNotFoundError, json.JSONDecodeError):
            pass
```

### Step 3: Signal Aggregator & Regime Classifier (P0) - ~1.5 hours

```python
# signals_factory/market_mood/aggregator.py

from dataclasses import dataclass
from typing import Optional
from .data_fetcher import MarketSignals
from .fatigue_calculator import RegimeFatigueCalculator, FatigueState

@dataclass
class MoodReport:
    """Complete market mood report"""
    timestamp: str
    signals: dict
    mood_regime: str
    score: int
    fatigue: FatigueState
    confidence: dict
    summary: str


class MarketMoodAggregator:
    """
    Aggregates signals into a unified mood score and regime classification.
    
    IMPORTANT: No double-counting!
    We're NOT using external FGI (which already contains VIX + Put/Call internally).
    All signals here are raw, independent data.
    """
    
    # VIX thresholds (historical context)
    # <15: Complacent, 15-20: Normal, 20-25: Elevated, 25-35: Fear, >35: Panic
    VIX_THRESHOLDS = {
        'extreme_fear': 35,
        'fear': 25,
        'elevated': 20,
        'normal': 15,
        # Below 15 = complacent/greed
    }
    
    # Put/Call ratio thresholds
    # >1.0: Fear (more puts), <0.8: Greed (more calls)
    PC_THRESHOLDS = {
        'fear': 1.0,
        'neutral_high': 0.9,
        'neutral_low': 0.8,
        # Below 0.8 = greed
    }
    
    # Signal weights (sum to 100)
    WEIGHTS = {
        'vix': 40,            # Primary fear gauge
        'put_call': 20,       # Options market sentiment
        'spy_returns': 25,    # Trend context (1D + 5D combined)
        'volume': 15,         # Participation level
    }
    
    def __init__(self, fatigue_calculator: RegimeFatigueCalculator):
        self.fatigue_calc = fatigue_calculator
    
    def generate_report(self, signals: MarketSignals) -> MoodReport:
        """Generate complete mood report from signals"""
        
        # Calculate component scores
        vix_score = self._score_vix(signals.vix)
        pc_score = self._score_put_call(signals.put_call_ratio)
        returns_score = self._score_returns(signals.spy_1d_return, signals.spy_5d_return)
        volume_score = self._score_volume(signals.volume_ratio, signals.spy_1d_return)
        
        # Weighted aggregate (range: -100 to +100)
        total_score = (
            vix_score * self.WEIGHTS['vix'] +
            pc_score * self.WEIGHTS['put_call'] +
            returns_score * self.WEIGHTS['spy_returns'] +
            volume_score * self.WEIGHTS['volume']
        ) / 100
        
        # Classify regime
        regime = self._classify_regime(total_score)
        
        # Update fatigue tracker
        fatigue = self.fatigue_calc.update_regime(regime)
        
        # Calculate confidence
        confidence = self._calculate_confidence(signals, vix_score, pc_score, returns_score)
        
        # Generate summary
        summary = self._generate_summary(regime, fatigue, signals)
        
        return MoodReport(
            timestamp=signals.timestamp.isoformat(),
            signals={
                'vix': signals.vix,
                'put_call_ratio': signals.put_call_ratio,
                'spy_1d_return': round(signals.spy_1d_return, 2),
                'spy_5d_return': round(signals.spy_5d_return, 2),
                'volume_ratio': round(signals.volume_ratio, 2)
            },
            mood_regime=regime,
            score=int(total_score),
            fatigue=fatigue,
            confidence=confidence,
            summary=summary
        )
    
    def _score_vix(self, vix: float) -> float:
        """
        VIX score: -1 (extreme fear) to +1 (complacent/greed)
        Higher VIX = more fear = negative score
        """
        if vix >= self.VIX_THRESHOLDS['extreme_fear']:
            return -1.0  # Extreme fear
        elif vix >= self.VIX_THRESHOLDS['fear']:
            return -0.7
        elif vix >= self.VIX_THRESHOLDS['elevated']:
            return -0.3
        elif vix >= self.VIX_THRESHOLDS['normal']:
            return 0.0  # Normal
        elif vix >= 12:
            return 0.5  # Complacent
        else:
            return 1.0  # Extreme complacency (potential greed)
    
    def _score_put_call(self, pc_ratio: Optional[float]) -> float:
        """
        Put/Call score: -1 (fear) to +1 (greed)
        Higher P/C = more fear (buying protection)
        """
        if pc_ratio is None:
            return 0.0  # Neutral if unavailable
        
        if pc_ratio >= 1.2:
            return -1.0  # Extreme fear
        elif pc_ratio >= self.PC_THRESHOLDS['fear']:
            return -0.5
        elif pc_ratio >= self.PC_THRESHOLDS['neutral_high']:
            return -0.2
        elif pc_ratio >= self.PC_THRESHOLDS['neutral_low']:
            return 0.0  # Neutral
        elif pc_ratio >= 0.7:
            return 0.5
        else:
            return 1.0  # Extreme greed (no protection buying)
    
    def _score_returns(self, ret_1d: float, ret_5d: float) -> float:
        """
        Returns score: -1 (falling) to +1 (rising)
        Combines short and medium term momentum
        """
        # Weight 1D slightly more (recency)
        combined = (ret_1d * 0.6 + ret_5d * 0.4)
        
        # Normalize to -1 to +1 (assuming ±5% is extreme)
        return max(-1.0, min(1.0, combined / 5.0))
    
    def _score_volume(self, volume_ratio: float, ret_1d: float) -> float:
        """
        Volume score with context:
        - High volume + down day = capitulation (bearish exhaustion)
        - High volume + up day = breakout (bullish)
        - Low volume = muted conviction
        """
        if volume_ratio < 0.7:
            return 0.0  # Low volume = neutral
        elif volume_ratio > 1.5:
            # High volume - direction matters
            return 0.5 if ret_1d > 0 else -0.5
        else:
            return 0.0  # Normal volume
    
    def _classify_regime(self, score: float) -> str:
        """Map aggregate score to regime"""
        if score <= -60:
            return "EXTREME_PANIC"
        elif score <= -40:
            return "PANIC"
        elif score <= -20:
            return "FEAR"
        elif score <= 20:
            return "NEUTRAL"
        elif score <= 40:
            return "OPTIMISM"
        elif score <= 60:
            return "GREED"
        else:
            return "EXTREME_GREED"
    
    def _calculate_confidence(self, signals: MarketSignals, 
                             vix_score: float, pc_score: float, 
                             returns_score: float) -> dict:
        """
        Confidence based on:
        1. Coverage: How many signals are available
        2. Agreement: Do signals point the same direction
        """
        # Coverage
        coverage = signals.coverage
        
        # Agreement (do signals agree on direction)
        scores = [vix_score, pc_score, returns_score]
        non_zero = [s for s in scores if s != 0]
        
        if len(non_zero) < 2:
            agreement = "INSUFFICIENT"
        else:
            # Check if all same sign
            all_positive = all(s > 0 for s in non_zero)
            all_negative = all(s < 0 for s in non_zero)
            
            if all_positive or all_negative:
                agreement = "HIGH"
            elif any(s > 0 for s in non_zero) and any(s < 0 for s in non_zero):
                agreement = "MIXED"
            else:
                agreement = "MODERATE"
        
        return {
            'coverage': coverage,
            'agreement': agreement
        }
    
    def _generate_summary(self, regime: str, fatigue: FatigueState, 
                         signals: MarketSignals) -> str:
        """Human-readable summary"""
        emoji_map = {
            'EXTREME_PANIC': '🚨',
            'PANIC': '🔴',
            'FEAR': '🟠',
            'NEUTRAL': '🟡',
            'OPTIMISM': '🟢',
            'GREED': '🔵',
            'EXTREME_GREED': '💎'
        }
        
        emoji = emoji_map.get(regime, '⚪')
        fatigue_warning = ""
        
        if fatigue.fatigue_pct >= 70:
            fatigue_warning = f" ⚠️ HIGH FATIGUE ({fatigue.fatigue_pct}%)"
        elif fatigue.fatigue_pct >= 50:
            fatigue_warning = f" ↗️ Building fatigue ({fatigue.fatigue_pct}%)"
        
        return (
            f"{emoji} {regime} (Day {fatigue.day_count}) - "
            f"VIX: {signals.vix:.1f} - "
            f"Shift: {fatigue.shift_probability}"
            f"{fatigue_warning}"
        )
```

### Step 4: Main Entry Point (P0) - ~30 mins

```python
# signals_factory/market_mood/main.py

from .data_fetcher import MarketDataFetcher
from .fatigue_calculator import RegimeFatigueCalculator
from .aggregator import MarketMoodAggregator, MoodReport
import json

def run_market_mood(state_file: str = "mood_state.json") -> MoodReport:
    """
    Main entry point for market mood detection.
    
    Returns a complete MoodReport with regime, fatigue, and confidence.
    """
    # Initialize components
    fetcher = MarketDataFetcher()
    fatigue_calc = RegimeFatigueCalculator(state_file=state_file)
    aggregator = MarketMoodAggregator(fatigue_calc)
    
    # Fetch signals
    signals = fetcher.fetch_signals()
    
    # Generate report
    report = aggregator.generate_report(signals)
    
    return report


def print_report(report: MoodReport):
    """Pretty print the mood report"""
    print("\n" + "=" * 60)
    print("📊 MARKET MOOD REPORT")
    print("=" * 60)
    print(f"Timestamp: {report.timestamp}")
    print(f"\n{report.summary}")
    print(f"\nRegime: {report.mood_regime}")
    print(f"Score: {report.score}")
    print(f"\nFatigue:")
    print(f"  - Day count: {report.fatigue.day_count}")
    print(f"  - Fatigue %: {report.fatigue.fatigue_pct}%")
    print(f"  - Shift probability: {report.fatigue.shift_probability}")
    print(f"\nConfidence:")
    print(f"  - Coverage: {report.confidence['coverage']}")
    print(f"  - Agreement: {report.confidence['agreement']}")
    print(f"\nRaw Signals:")
    for k, v in report.signals.items():
        print(f"  - {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    report = run_market_mood()
    print_report(report)
    
    # Also output as JSON for programmatic use
    print("\n📋 JSON Output:")
    print(json.dumps({
        'timestamp': report.timestamp,
        'mood_regime': report.mood_regime,
        'score': report.score,
        'fatigue': {
            'day_count': report.fatigue.day_count,
            'fatigue_pct': report.fatigue.fatigue_pct,
            'shift_probability': report.fatigue.shift_probability
        },
        'confidence': report.confidence,
        'signals': report.signals
    }, indent=2))
```

---

## 🧪 Day-1 Validation (Quick & Dirty)

**Forget historical backtests. Do this instead:**

### Test 1: Sanity Check Current Market
1. Run the system
2. Ask: "Does this match my intuition about today's market?"
3. Check VIX is actually fetched (not fallback 20.0)

### Test 2: Hard-Coded Panic Scenario
```python
# In a test file, override signals
from dataclasses import replace

panic_signals = replace(
    fetcher.fetch_signals(),
    vix=45.0,           # Panic levels
    spy_1d_return=-5.0,  # Big down day
    volume_ratio=2.5     # High volume
)
report = aggregator.generate_report(panic_signals)
assert report.mood_regime in ["PANIC", "EXTREME_PANIC"]
```

### Test 3: Fatigue Progression
```python
# Run multiple days in same regime (simulate with manual state)
# Check that fatigue_pct increases monotonically
```

---

## 📦 Final Package List (Day-1 Only)

| Package | Purpose | Status |
|---------|---------|--------|
| `yfinance` | VIX, SPY, volume | ✅ REQUIRED |
| `pandas` | Data manipulation | ✅ Already have |
| `tenacity` | Retry/backoff for flaky APIs | ✅ RECOMMENDED |

**That's it.** Everything else is Phase 2+.

---

## 🧯 Risk Register

| Risk | Mitigation |
|------|------------|
| **Lookahead/timestamp mismatch** | All signals aligned to same cutoff (last close) |
| **yfinance rate limits** | Add `tenacity` retry logic |
| **Put/Call missing** | Degradable - system works without it |
| **State loss on restart** | JSON file persistence for fatigue state |
| **Weekend/holiday data** | Use last available close (yfinance handles this) |

---

## 📅 What's Deferred to Phase 2+

1. **Cboe Put/Call Scraper** - Parse daily stats page
2. **News Sentiment** - MarketWatch RSS + keyword analyzer
3. **Fatigue Calibration** - Backtest against historical regime changes
4. **Dashboard Integration** - Add to Streamlit
5. **Alerting** - Slack/Discord notifications on regime change

---

## ✅ Day-1 Checklist

- [ ] Create `signals_factory/market_mood/` directory
- [ ] Implement `data_fetcher.py`
- [ ] Implement `fatigue_calculator.py`
- [ ] Implement `aggregator.py`
- [ ] Implement `main.py`
- [ ] Run on current market data
- [ ] Verify output makes intuitive sense
- [ ] Test with hard-coded panic scenario

---

*Document Revised: 2025-12-29*
*Status: 1-DAY MVP PLAN*
*AI Audit: INCORPORATED*
