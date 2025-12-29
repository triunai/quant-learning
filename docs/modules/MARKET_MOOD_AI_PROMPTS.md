# 🤖 AI Research Prompts - Revised (Post-Audit)

> **Status**: Updated after AI research audit. Scope dramatically reduced for 1-day MVP.

---

## 🚫 Prompts NO LONGER NEEDED (Traps Avoided)

| Original Topic | Why Dropped |
|----------------|-------------|
| Fear & Greed Index API | CNN's FGI isn't freely available. **Double-counting trap**: FGI already uses VIX + Put/Call internally! |
| Multi-source RSS crawling | Paywalls, ToS issues, brittle. Not day-1 feasible. |
| FinBERT / Transformers NLP | Heavy, slow, overkill for MVP. |
| Academic fatigue calibration | Phase 3+ work. Day-1 uses heuristic. |

---

## ✅ Remaining Research Prompts (If Needed)

### Prompt 1: Cboe Put/Call Ratio Scraping (Phase 2)

```
I need to fetch the daily Put/Call ratio from Cboe's published statistics.

Context:
- Cboe publishes daily market statistics at:
  https://www.cboe.com/us/options/market_statistics/daily/
- I need the Total Put/Call Ratio (or Equity vs Index separately)

Questions:
1. What is the structure of the Cboe daily stats page?
2. Can I programmatically download the daily data (CSV or API)?
3. What are the legal/ToS considerations for scraping?
4. What Python packages are best for this (requests + BeautifulSoup)?
5. How to handle weekends/holidays (no new data)?

Please provide:
- Example scraping code if legally permissible
- Alternative data sources for Put/Call ratio
- Fallback strategy if scraping fails
```

### Prompt 2: VIX Thresholds & Interpretation

```
I'm using VIX as the primary fear gauge in a market mood system.

Current thresholds I'm using:
- VIX < 15: Complacent/Greed
- VIX 15-20: Normal
- VIX 20-25: Elevated caution
- VIX 25-35: Fear
- VIX > 35: Panic

Questions:
1. Are these thresholds historically accurate?
2. Has VIX "regime" changed over time (e.g., post-2020 structurally higher)?
3. Should I use VIX percentile rank (vs absolute level) for better signals?
4. What is the VIX term structure (VX30/VX90 ratio) and is it useful?

Please provide:
- Historically validated VIX thresholds with dates
- Any academic papers on VIX interpretation
- Whether to use absolute VIX or percentile
```

### Prompt 3: Volume Analysis Interpretation

```
I'm using volume ratio (today's volume / 20-day average) in my mood system.

Current logic:
- Volume ratio > 1.5 + down day = capitulation (bearish exhaustion)
- Volume ratio > 1.5 + up day = breakout (bullish)
- Volume ratio < 0.7 = low conviction

Questions:
1. Are these ratios historically validated?
2. Should I differentiate up-volume vs down-volume (on-balance volume)?
3. What's a good lookback period for average volume (20 days? 50 days?)?
4. How to handle expiration days (OpEx) with artificially high volume?

Please provide:
- Volume ratio thresholds with historical context
- Whether OBV or money flow is better than simple ratio
- OpEx calendar consideration
```

### Prompt 4: Fatigue Calibration (Phase 3)

```
I have a "regime fatigue" concept for market mood prediction.

Core hypothesis: 
- Markets get tired of extreme states (panic/greed)
- Probability of regime change increases with time in regime
- Panic fatigues faster than neutral

Current implementation:
- Exponential saturation: fatigue = 100 * (1 - 0.5^(days / half_life))
- Half-lives: Extreme states = 3 days, Neutral = 15 days

FOR FUTURE CALIBRATION (not day-1):

Questions:
1. How to backtest fatigue predictions against historical regime changes?
2. What historical events should I validate against?
   - COVID crash (2020-03-23): Days of panic before bottom?
   - 2021 peak (Nov): Days of greed before correction?
   - 2022 bear (Oct): Days of panic before bottom?
3. Should half-lives be market-dependent (bull vs bear regimes)?
4. How to measure "regime change lead time" accuracy?

Please provide:
- Backtesting methodology for fatigue
- Suggested historical test cases with dates
- Metrics for evaluating fatigue prediction accuracy
```

---

## 📊 Quick Reference: What's Already Decided

### Signal Sources (Locked In)
| Signal | Source | Package |
|--------|--------|---------|
| VIX | Yahoo Finance | `yfinance` |
| SPY Returns | Yahoo Finance | `yfinance` |
| Volume | Yahoo Finance | `yfinance` |
| Put/Call | Cboe (Phase 2) | TBD |

### Scoring Weights (Locked In)
| Signal | Weight | Rationale |
|--------|--------|-----------|
| VIX | 40% | Primary fear gauge |
| Put/Call | 20% | Options market sentiment |
| SPY Returns | 25% | Trend context |
| Volume | 15% | Participation level |

### Regime Thresholds (Locked In)
| Score Range | Regime |
|-------------|--------|
| ≤ -60 | EXTREME_PANIC |
| -60 to -40 | PANIC |
| -40 to -20 | FEAR |
| -20 to +20 | NEUTRAL |
| +20 to +40 | OPTIMISM |
| +40 to +60 | GREED |
| > +60 | EXTREME_GREED |

### Fatigue Half-Lives (Locked In for Day-1)
| Regime | Half-Life (days) |
|--------|------------------|
| EXTREME_PANIC | 3 |
| PANIC | 5 |
| FEAR | 7 |
| NEUTRAL | 15 |
| OPTIMISM | 7 |
| GREED | 5 |
| EXTREME_GREED | 3 |

---

## 🎯 Day-1 Focus

**You don't need more research for Day-1.** The plan is locked in:

1. Build `MarketDataFetcher` (yfinance)
2. Build `RegimeFatigueCalculator` (heuristic)
3. Build `MarketMoodAggregator` (weighted scoring)
4. Run and validate on current market

**Only revisit these prompts for Phase 2+ refinement.**

---

*Document Revised: 2025-12-29*
*Status: POST-AUDIT - SCOPE REDUCED*
