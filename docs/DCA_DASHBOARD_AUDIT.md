# 🎯 DCA Enhancement Dashboard - Feasibility Audit

> **Objective:** Build a Streamlit dashboard for any ticker that provides comprehensive DCA analysis using the three existing systems, with eventual React port.

---

## 📊 Executive Summary

| Aspect | Status | Effort |
|--------|--------|--------|
| **Core Engine (RegimeRiskPlatform)** | ✅ Ready | Low |
| **Semi-Markov Integration** | ✅ Ready | Low |
| **Signals Factory** | ✅ Ready | Low |
| **Existing Streamlit Dashboard** | ✅ Ready (base) | Medium to extend |
| **DCA-Specific Logic** | ❌ Not built | Medium |
| **React Port** | 📋 Planned | High |

**Verdict: HIGHLY FEASIBLE** - You have 80% of the backend built. The main work is:
1. Creating the `DCARegimeEnhancer` integration class
2. Extending the Streamlit dashboard with DCA-specific UI
3. Adding historical DCA backtesting

---

## 🔍 Existing Assets Audit

### 1. Regime Risk Platform (`battle-tested/PLTR-test-2.py`)

| Feature | Status | Notes |
|---------|--------|-------|
| `RegimeRiskPlatform` class | ✅ | Full v7.0 implementation |
| `ingest_data()` | ✅ | Any ticker via yfinance |
| `build_regime_model()` | ✅ | GMM on slow features |
| `compute_market_beta()` | ✅ | Alpha/Beta decomposition |
| `check_macro_context()` | ✅ | VIX + anomaly detection |
| `simulate()` | ✅ | Factor model + Semi-Markov |
| `compute_risk_metrics()` | ✅ | VaR, CVaR, Kelly, DD |
| `generate_signal()` | ✅ | LONG/SHORT/NEUTRAL |
| Walk-forward validation | ✅ | Brier score calibration |

**Gap:** Not packaged as importable module (lives in `battle-tested/`).

**Fix Required:**
```bash
# Move to proper location
mv battle-tested/PLTR-test-2.py to_refine/regime_risk_platform.py
```

### 2. Semi-Markov Model (`refinery/semi_markov.py`)

| Feature | Status | Notes |
|---------|--------|-------|
| `SemiMarkovModel` class | ✅ | Full implementation |
| `_process_data()` | ✅ | KMeans clustering |
| `fit_distributions()` | ✅ | Gamma/Exponential |
| `sample_residual_duration()` | ✅ | Inverse transform |
| `regime_fatigue_score()` | ✅ | CDF-based |
| `get_position_size()` | ✅ | Fatigue-adjusted |
| `run_simulation()` | ✅ | Semi-Markov MC |

**Gap:** Missing `get_days_in_current_state()` method.

**Fix Required:**
```python
def get_days_in_current_state(self):
    """Returns days since last regime change."""
    last_block = self.data['block'].iloc[-1]
    return len(self.data[self.data['block'] == last_block])
```

### 3. Signals Factory (`signals_factory/`)

| Feature | Status | Notes |
|---------|--------|-------|
| `BaseSignal` ABC | ✅ | Clean interface |
| `SignalAggregator` | ✅ | Weighted combination |
| `VolCompressionSignal` | ✅ | RV + Range + VIX |
| `RegimeFatigueSignal` | ❌ | Not built |
| `LiquidityHorizonSignal` | ❌ | Not built |

**Gap:** Only 1 of 5 planned signals implemented.

**Priority Signals to Add:**
1. `RegimeFatigueSignal` - wraps Semi-Markov fatigue
2. `TrendMomentumSignal` - simple price momentum

### 4. Existing Dashboard (`to_refine/dashboard.py`)

| Feature | Status | Notes |
|---------|--------|-------|
| Streamlit setup | ✅ | Dark theme, wide layout |
| Ticker input | ✅ | Sidebar text input |
| Regime analysis | ✅ | Full v7.0 display |
| Risk metrics | ✅ | VaR, CVaR, Kelly |
| Visualization | ✅ | Cone, DD dist, transitions |
| Export | ✅ | JSON + PNG download |

**Gap:** No DCA-specific features.

**Extensions Required:**
1. DCA amount input
2. DCA adjustment calculator
3. Historical DCA backtest chart
4. Monthly decision log

---

## 🏗️ Architecture Plan

### Current State
```
battle-tested/PLTR-test-2.py  ──┐
                                 │
refinery/semi_markov.py        ──┼──> to_refine/dashboard.py (Streamlit)
                                 │
signals_factory/               ──┘
```

### Target State
```
to_refine/
├── regime_risk_platform.py    # Moved from battle-tested
├── regime_engine_v7.py        # Existing wrapper
├── semi_markov.py             # Move from refinery (optional)
├── dashboard.py               # Extend with DCA tab
└── dca_enhancer.py            # NEW: Integration class

signals_factory/
├── base_signal.py             # Existing
├── aggregator.py              # Existing
├── signal_vol_compression.py  # Existing
└── signal_regime_fatigue.py   # NEW

apps/
└── dca_dashboard.py           # NEW: Dedicated DCA Streamlit app
```

---

## 📋 Implementation Roadmap

### Phase 1: Foundation (1-2 hours)
- [ ] Move `PLTR-test-2.py` to `to_refine/regime_risk_platform.py`
- [ ] Add `get_days_in_current_state()` to Semi-Markov
- [ ] Create `DCARegimeEnhancer` integration class
- [ ] Update shim to include new module

### Phase 2: Signals (2-3 hours)
- [ ] Create `RegimeFatigueSignal` (wraps Semi-Markov)
- [ ] Add to `SignalAggregator` weights
- [ ] Test signal pipeline

### Phase 3: Dashboard Extension (3-4 hours)
- [ ] Add "DCA Advisor" tab to existing dashboard
- [ ] DCA amount input + adjustment display
- [ ] Historical DCA backtest visualization
- [ ] Monthly decision log with reasoning

### Phase 4: Standalone DCA App (4-5 hours)
- [ ] Create dedicated `apps/dca_dashboard.py`
- [ ] Multi-ticker comparison view
- [ ] Portfolio-level DCA allocation
- [ ] PDF report generation

### Phase 5: React Port (8-12 hours)
- [ ] Create FastAPI backend exposing JSON endpoints
- [ ] React frontend with charts (Recharts/Visx)
- [ ] State management (Zustand or Redux)
- [ ] Responsive design for mobile

---

## 🔧 Key Integration Points

### DCARegimeEnhancer Class (Core Logic)

```python
class DCARegimeEnhancer:
    def __init__(self, ticker: str, base_monthly_amount: float = 1000):
        self.ticker = ticker
        self.base_amount = base_monthly_amount
        
        # Initialize systems
        self.platform = None  # Lazy load (expensive)
        self.semi_markov = None
        self.signals = None
        
    def run_full_analysis(self) -> dict:
        """Run all three systems and compute DCA adjustment."""
        # 1. Regime Risk Platform
        self.platform = RegimeRiskPlatform(self.ticker)
        self.platform.ingest_data()
        self.platform.build_regime_model()
        results = self.platform.run(run_full_calibration=False)
        
        # 2. Semi-Markov Fatigue
        self.semi_markov = SemiMarkovModel(self.ticker)
        self.semi_markov._process_data()
        self.semi_markov.fit_distributions()
        fatigue = self._compute_fatigue()
        
        # 3. Signals
        vol_signal = VolCompressionSignal()
        vol_signal.fit(self.platform.data)
        signal_result = vol_signal.predict()
        
        # 4. Integration
        adjustment = self._compute_adjustment(results, fatigue, signal_result)
        
        return {
            'base_amount': self.base_amount,
            'adjusted_amount': adjustment['amount'],
            'multiplier': adjustment['multiplier'],
            'regime': self.platform.regime_names[self.platform.current_regime],
            'kelly': results['risk']['kelly_fraction'],
            'fatigue': fatigue,
            'compression': signal_result['score'],
            'reasoning': adjustment['reasons']
        }
```

### Regime Mapping

```python
# Map 3-regime platform to 5-regime semi-markov
REGIME_MAP = {
    'Low Vol': [3, 4],      # Bull, Rally states
    'Normal': [2],          # Neutral state
    'Crisis': [0, 1]        # Crash, Bear states
}

def map_platform_to_semimarkov(platform_regime: int) -> int:
    """Convert platform regime to semi-markov state."""
    name = platform.regime_names[platform_regime]
    return REGIME_MAP.get(name, [2])[0]
```

---

## 📊 Streamlit Dashboard Wireframe

```
┌─────────────────────────────────────────────────────────────────┐
│ 🎯 DCA REGIME ENHANCER                               [PLTR ▼]   │
├─────────────────────────────────────────────────────────────────┤
│ SIDEBAR                    │  MAIN CONTENT                      │
│ ─────────                  │  ──────────────                    │
│ Ticker: [PLTR   ]          │  ┌──────────────────────────────┐  │
│ Base DCA: [$1000 ]         │  │  DECEMBER 2024 DECISION       │  │
│ Benchmark: [QQQ  ]         │  │  ─────────────────────────    │  │
│                            │  │  Base:     $1,000             │  │
│ [RUN ANALYSIS]             │  │  Adjusted: $720 (-28%)        │  │
│                            │  │                               │  │
│ ─────────────              │  │  Regime:   Normal             │  │
│ Quick Stats:               │  │  Fatigue:  Low (20%)          │  │
│ • Price: $78.50            │  │  Kelly:    15%                │  │
│ • Regime: Normal           │  │  Compression: 45/100          │  │
│ • VIX: 18.5                │  │                               │  │
│ • Beta: 1.85               │  │  REASONING:                   │  │
│                            │  │  ✓ Normal regime, stable      │  │
│                            │  │  ⚠ Kelly suggests caution     │  │
│                            │  │  ✓ Low compression            │  │
│                            │  └──────────────────────────────┘  │
│                            │                                    │
│                            │  TABS: [Analysis] [History] [Sim]  │
│                            │  ────────────────────────────────  │
│                            │  [6-Month Cone Chart]              │
│                            │  [Drawdown Distribution]           │
│                            │  [DCA Backtest Chart]              │
└────────────────────────────┴────────────────────────────────────┘
```

---

## 🚀 React Port Strategy

### Backend (FastAPI)

```python
# api/main.py
from fastapi import FastAPI
from dca_enhancer import DCARegimeEnhancer

app = FastAPI()

@app.get("/api/dca/{ticker}")
async def get_dca_analysis(ticker: str, base_amount: float = 1000):
    enhancer = DCARegimeEnhancer(ticker, base_amount)
    return enhancer.run_full_analysis()

@app.get("/api/regime/{ticker}")
async def get_regime_analysis(ticker: str):
    # Return regime details for visualization
    pass
```

### Frontend (React + TypeScript)

```
react-dca/
├── src/
│   ├── components/
│   │   ├── TickerInput.tsx
│   │   ├── DCADecision.tsx
│   │   ├── RegimeGauge.tsx
│   │   ├── FatigueIndicator.tsx
│   │   └── ConeChart.tsx
│   ├── hooks/
│   │   └── useDCAAnalysis.ts
│   └── App.tsx
└── package.json
```

**Key Libraries:**
- `recharts` or `visx` for charts
- `zustand` for state management
- `tanstack-query` for data fetching
- `tailwindcss` for styling

---

## ⚠️ Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Performance**: Full analysis takes 2-3 min | High | Cache results, run heavy ops in background |
| **API Rate Limits**: yfinance throttling | Medium | Add retry logic, cache historical data |
| **Regime Mismatch**: 3 vs 5 state mapping | Low | Use unambiguous mapping table |
| **Cold Start**: First run slow | Medium | Pre-warm with common tickers |
| **VIX Unavailable**: Market closed | Low | Use last known value, default to 20 |

---

## 💰 Cost Estimate

| Phase | Hours | Priority |
|-------|-------|----------|
| Phase 1: Foundation | 2 | 🔴 High |
| Phase 2: Signals | 3 | 🟡 Medium |
| Phase 3: Dashboard | 4 | 🔴 High |
| Phase 4: Standalone | 5 | 🟡 Medium |
| Phase 5: React | 12 | 🟢 Low (later) |
| **Total** | **26** | |

---

## ✅ Recommended Next Steps

1. **Start with Phase 1** – Get `DCARegimeEnhancer` working with a simple CLI script.
2. **Add DCA tab** to existing dashboard – Fastest path to usable product.
3. **Iterate** on signals and UI based on usage.
4. **React port** – Only after Streamlit version is stable and battle-tested.

---

## 🎯 Quick Win: Minimal Viable DCA Dashboard

Want me to build a **minimal viable version** right now? It would:

1. Add a `DCARegimeEnhancer` class to `to_refine/`
2. Extend the existing dashboard with a "DCA Advisor" tab
3. Show adjusted DCA amount + reasoning
4. Include a simple backtest chart

This gets you a working demo in ~2 hours of implementation time.

**Say "go" and I'll start building!**
