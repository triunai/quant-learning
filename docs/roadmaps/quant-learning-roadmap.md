# Quant Learning Roadmap: Building Edge & Understanding

**Created:** 2025-12-31  
**Status:** ACTIVE  
**Goal:** Learn quant skills, understand market structure, build real edges

> "Edge isn't a strategy. Edge is understanding something the market doesn't."

---

## Phase 1: Immediate Deep Dives (Week 1-2)

### 1.1 Understand WHY the strategy failed
- [ ] **Deep dive: COVID vs Normal market regimes**
  - Question: Why did binary detection work in COVID but fail in 2010-2019?
  - Method: Compare volatility, trend persistence, autocorrelation across periods
  - File: `research/analysis/covid_regime_failure_analysis.py`

- [ ] **Feature stability analysis**
  - Do the same features predict regimes in all periods?
  - Calculate feature predictive power by period
  - Document which features are stable vs unstable

- [ ] **Threshold sensitivity analysis**
  - Are optimal thresholds (0.3 Sharpe, 60-day lookback) period-specific?
  - Grid search thresholds across all periods
  - Find universal vs period-specific parameters

### 1.2 Tech vs Non-Tech regime investigation
- [ ] **Statistical comparison**
  - Compare Tech regime duration: 32±12 days (your finding)
  - Compare Non-tech regime duration: ~22±8 days
  - T-test for significance
  
- [ ] **Investigate WHY tech regimes are longer**
  - Institutional ownership patterns
  - News cycle frequency
  - Product cycle timing
  - Earnings report impact

---

## Phase 2: Paper Replication (Week 3-4)

### 2.1 Hamilton (1989) Regime-Switching Model
- [ ] **Read the original paper**
  - Title: "A New Approach to the Economic Analysis of Nonstationary Time Series"
  - [Link: JSTOR](https://www.jstor.org/stable/1912559)
  
- [ ] **Implement Markov-Switching GARCH**
  - Use `arch` or `statsmodels` for Python implementation
  - Test on S&P 500 data 1990-2024
  - Compare to your regime detector

- [ ] **Document findings**
  - How does Hamilton's model compare to yours?
  - What does it capture that you missed?

### 2.2 Modern Regime Detection Literature
- [ ] **Read: "Machine Learning for Asset Managers" (Lopez de Prado)**
  - Focus on Chapter 5: Fractionally Differentiated Features
  - Implement feature engineering techniques

- [ ] **Read: "Advances in Financial Machine Learning" (Lopez de Prado)**
  - Chapter 10: Bet Sizing
  - Chapter 17: Structural Breaks

---

## Phase 3: Market Microstructure (Week 5-6)

### 3.1 Book Study
- [ ] **Read: "Trading and Exchanges" by Larry Harris**
  - Priority chapters: Market Structure, Order Flow, Volatility
  - Takes notes on regime-related insights

- [ ] **Read: "Market Microstructure in Practice" by Lehalle & Laruelle**
  - Focus on: Order book dynamics, price formation
  - Application to regime detection

### 3.2 Research Questions
- [ ] **Do regimes end with volatility spikes?**
  - Event study around regime boundaries
  - Calculate average volatility before/after transitions

- [ ] **Do volume patterns predict regime changes?**
  - Volume analysis 5-10 days before regime shift
  - Compare to random periods

- [ ] **Do options flows signal regime shifts?**
  - Put/Call ratio around transitions
  - VIX term structure changes

---

## Phase 4: Cross-Asset Edge (Week 7-8)

### 4.1 Cross-Asset Regime Propagation
- [ ] **Do stock regimes lead bond regimes?**
  - Granger causality test: SPY → TLT
  - Cross-correlation analysis
  - Lead-lag relationship

- [ ] **Does FX volatility predict equity regimes?**
  - DXY volatility vs SPY regime changes
  - EUR/USD as leading indicator

- [ ] **Do commodity regimes correlate with sector regimes?**
  - Oil → Energy sector
  - Gold → Safe haven sectors
  - Copper → Industrials

### 4.2 Build Cross-Asset Database
- [ ] **Collect daily data 2000-2024**
  - Equities: SPY, QQQ, IWM, sector ETFs
  - Bonds: TLT, IEF, HYG, LQD
  - Commodities: GLD, USO, copper
  - FX: DXY, EUR/USD, JPY
  - Volatility: VIX, VVIX, term structure

---

## Phase 5: Behavioral Edge (Week 9-10)

### 5.1 Flow Analysis
- [ ] **Do retail flows increase in TREND regimes?**
  - Source: Robinhood data, Reddit sentiment
  - Compare flow patterns by regime type

- [ ] **Do institutional flows predict regime changes?**
  - 13F filing analysis (quarterly)
  - Dark pool activity patterns

- [ ] **Does sentiment extreme signal regime end?**
  - Fear & Greed Index at regime transitions
  - Put/Call ratio extremes
  - VIX percentile at transitions

### 5.2 Sentiment Data Collection
- [ ] **Build sentiment database**
  - Reddit sentiment (r/wallstreetbets, r/investing)
  - Twitter/X finance sentiment
  - News headline sentiment

---

## Phase 6: Build Real Edge (Week 11-12)

### 6.1 Edge Hypothesis: Regime Persistence by Market Type
- [ ] **Test hypothesis**
  - TREND market + Bull regime → Long duration (40+ days)
  - CHOP market + Bull regime → Short duration (10-20 days)
  - TRANSITION market → Very short (5-10 days)

- [ ] **Calculate conditional probabilities**
  - P(regime ends | market_type, days_in_regime)
  - Build survival curves by market type

- [ ] **Backtest the edge**
  - If proven, test trading implications
  - Walk-forward + multi-period validation

### 6.2 RegimeScanner Enhancement
- [ ] **Tune classification thresholds**
  - Trend strength: 1.5 → 1.0
  - Add autocorrelation weight
  - Validate against known periods

- [ ] **Build Streamlit dashboard**
  - Market heatmap visualization
  - Historical period matching
  - Trading style recommendations

---

## Infrastructure Upgrades

### Research Pipeline
- [ ] **Reorganize codebase**
  ```
  research/
  ├── data_pipeline/           # Clean, reproducible data
  │   ├── download.py
  │   ├── clean.py
  │   └── feature_engineering.py
  ├── hypothesis_tests/        # Systematic testing
  │   ├── test_sector_clusters.py
  │   └── test_regime_persistence.py
  ├── models/                  # Reusable models
  │   ├── regime_classifier.py
  │   └── regime_scanner.py
  └── notebooks/              # Exploration
  ```

- [ ] **Create reproducibility standards**
  - All scripts should have random seeds
  - Version data sources
  - Document all parameters

### Documentation
- [ ] **Research log**
  - Weekly entries on findings
  - Failed hypotheses documented
  - Insights and next steps

---

## Reading List

### Must Read (Priority Order)
1. [ ] Hamilton (1989) - Regime-Switching Models
2. [ ] "Trading and Exchanges" - Larry Harris
3. [ ] "Advances in Financial Machine Learning" - Lopez de Prado
4. [ ] "Machine Learning for Asset Managers" - Lopez de Prado
5. [ ] "Quantitative Risk Management" - McNeil, Frey, Embrechts

### Secondary Reading
- [ ] "Expected Returns" - Antti Ilmanen
- [ ] "Active Portfolio Management" - Grinold & Kahn
- [ ] "The Econometrics of Financial Markets" - Campbell, Lo, MacKinlay

---

## Success Metrics

### Learning Metrics (Not Profit)
- [ ] Can explain Markov-switching models from first principles
- [ ] Can implement 3 different regime detection methods
- [ ] Have tested 10+ hypotheses with proper validation
- [ ] Understand market microstructure basics
- [ ] Have one documented "edge" (even if small)

### Process Metrics
- [ ] Weekly research log entries
- [ ] All code reproducible with documentation
- [ ] Every hypothesis has walk-forward + multi-period test

---

## Session Summary: 2025-12-31

### What We Built Today
- [x] Walk-forward validation framework
- [x] Multi-period testing (caught COVID overfit)
- [x] Survivorship bias correction
- [x] RegimeScanner prototype

### What We Learned
- [x] Binary strategy only worked in COVID era
- [x] Survivorship bias inflated Sharpe by 10%
- [x] Simple strategies can still overfit
- [x] Classification easier than prediction

### Archived (Failed)
- Binary regime trading strategy (COVID-only)

### Created for Future
- RegimeScanner (classification product)
- Research validation framework

---

## Quick Reference: Key Files

| File | Purpose |
|------|---------|
| `research/prototypes/regime_scanner.py` | Market type classifier |
| `research/scripts/walk_forward_validation.py` | Proper backtest |
| `research/scripts/survivorship_fix_validation.py` | Multi-period test |
| `research/archive/covid_only_strategy/README.md` | Failed strategy docs |
| `docs/sessions/2025-12-31-binary-strategy-validated.md` | Today's notes |

---

*"The money will come from understanding. The understanding comes from research."*
