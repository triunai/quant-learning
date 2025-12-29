# 🔧 DCA Dashboard - Engineering Planning

> **Status:** Architecture Planning  
> **Last Updated:** 2024-12-29  
> **Purpose:** Document engineering decisions, package choices, pipeline design, and open questions.

---

## 🧭 0) The One Thing We Must Decide First: "What Counts as Truth?"

Because backtests and regimes live or die on data integrity.

### ✅ Core Truth Table

You'll have 3 "classes" of data:

| Class | Data Type | Examples |
|-------|-----------|----------|
| **A) Price + Volume** (primary truth) | OHLCV (daily), dividends + splits | Corporate actions |
| **B) Macro / Cross-Asset** (secondary truth) | VIX, SPY/QQQ, rates, credit spreads | Market context |
| **C) Event Data** (guards/overlays) | Earnings dates, FOMC/CPI | Optional but powerful |

### ⚠️ Engineering Challenge

These sources must be **reproducible and versioned** (or at least cached), otherwise every run "changes the past".

---

## 📦 1) Best Packages (By Purpose)

### 🔥 Data Acquisition

| Level | Package | Notes |
|-------|---------|-------|
| ✅ Minimum Viable | `yfinance` | Easy, but unstable throttling + occasional weirdness |
| ✅ Best Practice "Retail-Grade but Serious" | `pandas-datareader` + `requests-cache` | Game-changer for rate limits |
| ✅ "Quants Don't Mess Around" | `polygon` (paid), `tiingo` (paid), `alpaca` (has data), `stooq` (free-ish) | For macro: `fredapi` (FRED) |

**Recommendation for you (now):**
```python
# Keep yfinance, but wrap it in:
# - requests-cache (automatic caching)
# - retry/backoff (handle rate limits)
# - disk caching (reproducibility)
```

### 🧠 Modelling + Stats

| Package | Purpose |
|---------|---------|
| `numpy`, `pandas` | Core data structures |
| `scipy` | Distributions, KS tests |
| `scikit-learn` | GMM, KMeans clustering |
| `statsmodels` | OLS, diagnostics |
| `arch` | GARCH volatility (if using) |

### 📊 Dashboard Plotting

| Package | Notes |
|---------|-------|
| `plotly` | Interactive, perfect for dashboards |
| `altair` | Nice, but less flexible |
| `matplotlib` | Avoid in Streamlit unless needed (heavy) |

**Recommendation:** Use `plotly` for Streamlit.

### ⚡ Speed + Caching

| Package | Purpose |
|---------|---------|
| `joblib` | Cache objects to disk |
| `diskcache` | Simple persistent cache |
| `@st.cache_data` | Streamlit-specific caching |
| `@st.cache_resource` | For expensive objects (engines) |

---

## 🧬 2) Data Pipeline Practices (The "Adult" Way)

You want a pipeline that is:
- ✅ Deterministic
- ✅ Cacheable
- ✅ Incremental
- ✅ Testable

### Recommended Pipeline Layers

```
┌─────────────────────────────────────────────────────────────────┐
│  (1) FETCH LAYER                                                │
│      • Fetch raw OHLCV + corporate actions                      │
│      • Returns normalized DataFrame                             │
├─────────────────────────────────────────────────────────────────┤
│  (2) CLEAN LAYER                                                │
│      • Fix missing days                                         │
│      • Ensure adjusted vs unadjusted policy is consistent       │
├─────────────────────────────────────────────────────────────────┤
│  (3) FEATURE LAYER                                              │
│      • Slow features (vol, drawdown, returns)                   │
│      • Store features in cache                                  │
├─────────────────────────────────────────────────────────────────┤
│  (4) MODEL LAYER                                                │
│      • Regimes, Semi-Markov params, signals                     │
│      • Save model artifacts (JSON + pickle)                     │
├─────────────────────────────────────────────────────────────────┤
│  (5) DECISION LAYER                                             │
│      • DCA multiplier + reasoning                               │
├─────────────────────────────────────────────────────────────────┤
│  (6) BACKTEST LAYER                                             │
│      • Walk-forward evaluation                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 3) The BIG Engineering Pitfall: "Walk-Forward Correctness"

You already understand leakage — but DCA backtests introduce a special trap.

### ✅ For Each Monthly Decision Date `t`:

You must run:
1. Fit features on `<= t`
2. Fit regimes on `<= t`
3. Fit distributions on `<= t`
4. Simulate forward from `t → t+H`
5. Compute sizing decision at `t`

**Otherwise your backtest cheats by using future regimes.**

### Engineering Implication

Your current engines must support something like:

```python
ingest_data(end_date=decision_date)
build_regime_model(train_end=decision_date)
```

### 📋 Question 1 (Critical)

> **Do you want MVP backtest to be "good enough visualization", or do you want true walk-forward from day one?**

**Best Practice:**
- ✅ MVP can be "fast + approximate", but **label it clearly**
- ✅ Then upgrade to true walk-forward (which is real alpha proof)

---

## 🧰 4) Caching Strategy (So It Doesn't Take 3 Mins Per Run)

### What to Cache (Priority Order)

| Tier | What | Cache Key |
|------|------|-----------|
| **Tier 1** | Raw market data | `ticker + period + interval + source + fetch_timestamp` |
| **Tier 2** | Features DataFrame | Slow features reused in all engines |
| **Tier 3** | Model artifacts | Regime assignments, transition matrix, duration params, residual pools |
| **Tier 4** | Simulation outputs | Optional (big), but can cache summary stats |

### Best Practice Tools

| Context | Tool |
|---------|------|
| Streamlit MVP | `diskcache` or `joblib.Memory` |
| In dashboard | `@st.cache_resource` keyed by inputs |

### 📋 Question 2

> **Are you okay storing cached data on disk (`./.cache/`) inside the repo runtime, or do you want a proper local DB (SQLite/DuckDB)?**

**My default:** DuckDB (fast, simple, "quant-friendly").

---

## 🗃️ 5) Best Storage Formats (So React Port Is Painless)

### Minimum Viable

| Format | Use Case |
|--------|----------|
| JSON | Decision reports |
| CSV | Features (or Parquet) |

### Best Practice

| Format | Use Case |
|--------|----------|
| ✅ Parquet | DataFrames (fast, compressed) |
| ✅ JSON | Decision reports (human-readable) |
| ✅ Pickle/Joblib | sklearn objects (GMM, etc.) |

### 📋 Question 3

> **Do you want artifacts stored per ticker (folder-based), e.g. `artifacts/PLTR/...`, or central DB (DuckDB tables)?**

---

## 📡 6) Data Sources: What's "Best" for Your Exact Use Case?

Let's rank sources by what you need (daily DCA, not HFT):

### For Equities OHLCV + Adjusted Closes

| Source | Notes |
|--------|-------|
| **Best Free** | Stooq (surprisingly stable), Yahoo (via yfinance) |
| **Best Paid** | Polygon, Tiingo |

### For VIX

| Source | Notes |
|--------|-------|
| Yahoo | Okay, but can fail |
| **Better** | Use `^VIX` via Yahoo and cache last known |
| Or | Get VIX from CBOE (not always easy) |

### For Macro/Rates

| Source | Notes |
|--------|-------|
| **FRED** | King for reliability (and free) |

### 📋 Question 4

> **Are you planning to run this on US tickers only, or also Bursa/Malaysia later?**
> 
> (Important because Malaysia data sources are a totally different world.)

---

## 🧠 7) Signal System Design (So You Don't Create Spaghetti)

### Planned Signals

- ✅ `RegimeFatigueSignal`
- ✅ `KellyOverrideSignal`
- ✅ `CrisisOverrideSignal`
- ✅ `DrawdownGuardSignal`

### Rule: Signals Should NOT Re-Run Engines

Signals should only **consume a shared context object**.

### Example Context Payload

```python
context = {
    "platform": platform_report,
    "semi": {"fatigue": 0.62, "days_in_regime": 41},
    "features": df_features.tail(1),
}
```

Then signals read from context, no heavy compute.

### 📋 Question 5

> **Do you want SignalsFactory to accept context explicitly? Or keep it "fit/predict on df" style?**

**Best Practice:** Keep both:
- `fit(df)` for distributions
- `predict(df, context=...)` for overlays

---

## 🧪 8) Testing + Invariants (You Already Love This Stuff)

You should add unit tests for the DCA policy so you never regress.

### Must-Have Tests

| Test | Invariant |
|------|-----------|
| Multiplier bounds | Always within `[0.3, 2.0]` |
| Crisis handling | Crisis regime + high DD risk → reduces DCA |
| Compression | High compression → reduces size |
| Fatigue monotonicity | Fatigue increases → penalty increases |

**Recommendation:** `pytest` + a small `fixtures/` dataset for PLTR.

### 📋 Question 6

> **Are you willing to set up pytest now, or later after MVP?**

---

## 🛠️ 9) React Port Readiness (FastAPI Contract)

If you want React later, you should **lock the API response contract now**.

### ✅ Stable Schema from Day One

```json
{
  "ticker": "PLTR",
  "base_amount": 1000,
  "adjusted_amount": 720,
  "multiplier": 0.72,
  "regime": "Normal",
  "risk": {
    "kelly": 0.15,
    "prob_dd_30": 0.22
  },
  "signals": {
    "compression": 45,
    "fatigue": 0.2
  },
  "reasoning": [
    "Normal regime but Kelly suggests caution",
    "Low volatility compression detected"
  ]
}
```

### 📋 Question 7

> **For React later, do you want:**
> - "One endpoint returns everything"
> - Or "Separate endpoints per tab" (analysis, backtest, sim)?

**Default:** Separate endpoints (prevents mega-slow payloads).

---

## ✅ Proposed Build Plan (Engineering-First)

Here's how I'd sequence it to minimize rewrite:

### Phase 1 (MVP, But Architected Correctly)

- [ ] Build `MarketDataClient` (cached yfinance + retry)
- [ ] Build `FeatureStore` (Parquet or DuckDB)
- [ ] Build `DCARegimeEnhancer` (orchestrator)
- [ ] Add Streamlit DCA tab using cached enhancer
- [ ] Add "MVP backtest" (clearly labeled)

### Phase 2 (Real Proof)

- [ ] Add `train_end_date` support to engines
- [ ] Add true walk-forward DCA backtest
- [ ] Add performance metrics panel (CAGR, DD, Sharpe)

---

## 🎯 7 Engineering Questions to Answer

Reply with quick bullets (no essays). I'll adapt the architecture.

| # | Question | Options |
|---|----------|---------|
| 1 | **Backtest standard** | MVP approximate first, or true walk-forward immediately? |
| 2 | **Caching storage** | Disk folder `.cache/` vs DuckDB? |
| 3 | **Artifact structure** | Per-ticker folders vs centralized DB tables? |
| 4 | **Ticker universe** | US only for now, or Malaysia later? |
| 5 | **Signals interface** | Accept a context payload or remain "df-only"? |
| 6 | **Testing** | Add pytest now or later? |
| 7 | **API contract** | One endpoint vs separate endpoints later? |

---

## 📦 Once Answered, You Get:

1. **The exact repo structure**
2. **The "best packages" lockfile list**
3. **The pipeline skeleton** (data client + cache + feature store + enhancer)

So you can start coding cleanly.

---

## 📝 Answers Log

> *Fill in your answers here as we discuss:*

| Question | Your Answer | Notes |
|----------|-------------|-------|
| 1. Backtest | | |
| 2. Caching | | |
| 3. Artifacts | | |
| 4. Tickers | | |
| 5. Signals | | |
| 6. Testing | | |
| 7. API | | |

---

# 🎯 **Architecture Decisions (Final v1)**

## **1) Commodities v1 Scope**
**✅ ETFs only (GLD/SLV via Yahoo) - defer spot/FRED to v2**

**Why:**
- GLD/SLV have proper OHLCV, volume data
- ETF returns are what you actually earn (includes fees, tracking error)
- FRED spot data needs synthetic OHLC creation → adds complexity now
- Can still get gold/silver exposure via ETFs

**v2 Enhancement:** Add spot data with `OHLCFromSeriesAdapter` class

## **2) Price Truth Policy**  
**✅ Use Adjusted Close for returns, keep Close for price display**

**Implementation:**
```python
class PriceNormalizer:
    def normalize(self, df):
        return {
            'price_display': df['Close'],      # For charts, targets
            'price_returns': df['Adj Close'],  # For all return calculations
            'features': self.compute_features(df['Adj Close'])
        }
```

## **3) Heavy Compute Triggers**
**Besides Monday, trigger full re-run when:**

1. **Regime label changed** (Low Vol → Crisis, etc.)
2. **20d vol percentile moved > 25 points** (e.g., 30th → 60th percentile)
3. **Drawdown crossed -15% threshold** (enter or exit >15% drawdown)

**Plus time-based:** 
- First of month (portfolio rebalancing check)
- After earnings (if we track events)

## **4) Storage**
**✅ Hybrid: SQLite for logs + Parquet for timeseries**

**Structure:**
```
data/
├── sqlite/
│   └── portfolio.db     # Portfolio config, decisions, regime history
└── parquet/
    ├── raw/            # Raw OHLCV by ticker
    ├── features/       # Computed features
    ├── simulations/    # Monte Carlo paths
    └── models/         # Pickled sklearn models
```

## **5) Correlation Penalty Behavior**
**✅ Apply penalty only when multiple assets are >1x (risk-on)**

**Logic:**
```python
def apply_correlation_penalty(decisions, corr_matrix):
    # Find clusters with correlation > 0.7
    clusters = find_correlation_clusters(corr_matrix, threshold=0.7)
    
    for cluster in clusters:
        # Only penalize if ALL in cluster are recommending >1x
        if all(decisions[asset]['multiplier'] > 1.0 for asset in cluster):
            penalty = max(0.6, 1 - 0.15*(len(cluster)-1))
            for asset in cluster:
                decisions[asset]['multiplier'] *= penalty
                decisions[asset]['reasoning'].append(
                    f"Correlation penalty: {len(cluster)}-asset cluster"
                )
    
    return decisions
```

## **6) Strategy Config**
**✅ Per asset-class YAML (stocks/etfs/crypto) with inheritance**

**Structure:**
```
strategies/
├── base.yaml          # Common settings
├── stocks.yaml        # Stock-specific (e.g., earnings guard)
├── etfs.yaml          # ETF-specific (e.g., tracking error check)
└── crypto.yaml        # Crypto-specific (e.g., 24/7 trading)
```

---

# 🏗️ **Final v1 System Contract**

## **Data Schema (Parquet + SQLite)**

### **1. Raw Data Contract (Parquet)**
```python
# Every fetcher must output this DataFrame structure:
{
    'Open': float64,
    'High': float64, 
    'Low': float64,
    'Close': float64,
    'Adj Close': float64,  # Adjusted for splits/dividends
    'Volume': float64,
    'Dividends': float64,   # Optional
    'Stock Splits': float64  # Optional
}

# Metadata stored separately (JSON):
{
    'symbol': 'PLTR',
    'currency': 'USD',
    'timezone': 'America/New_York',
    'source': 'yfinance',
    'last_updated': '2024-12-28',
    'corporate_actions_applied': True
}
```

### **2. Features Contract (Parquet)**
```python
# Computed features (daily frequency):
{
    'log_ret': float64,      # Daily log return
    'vol_20d': float64,      # 20d rolling vol (annualized)
    'vol_60d': float64,      # 60d rolling vol
    'ret_20d': float64,      # 20d cumulative return
    'drawdown': float64,     # Current drawdown from 60d high
    'volume_z': float64,     # Volume z-score (20d)
    'regime': int32,         # Regime label (0, 1, 2)
    'regime_prob_0': float64 # Probability of regime 0
}
```

### **3. Decision Contract (SQLite)**
```sql
CREATE TABLE dca_decisions (
    id INTEGER PRIMARY KEY,
    date DATE NOT NULL,
    symbol TEXT NOT NULL,
    asset_class TEXT NOT NULL,  -- 'stock', 'etf', 'crypto'
    base_amount REAL NOT NULL,
    adjusted_amount REAL NOT NULL,
    multiplier REAL NOT NULL,
    regime TEXT NOT NULL,
    kelly_fraction REAL,
    prob_dd_30 REAL,
    fatigue_score REAL,
    compression_score REAL,
    correlation_penalty REAL,
    reasoning_json TEXT,  -- JSON array of strings
    executed_amount REAL,  -- What was actually bought
    executed_price REAL,
    strategy_version TEXT,  -- YAML hash
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_decisions ON dca_decisions(date, symbol);
```

### **4. Context Schema (Signal Interface)**
```python
Context = {
    'symbol': 'PLTR',
    'date': '2024-12-28',
    
    # From RegimeRiskPlatform
    'platform': {
        'regime': 'Normal',
        'regime_id': 1,
        'regime_probs': [0.1, 0.7, 0.2],
        'risk': {
            'kelly_fraction': 0.15,
            'prob_dd_20': 0.35,
            'prob_dd_30': 0.22,
            'var_95': -0.28,
            'cvar_95': -0.41
        },
        'targets': {
            'prob_up': 0.52,
            'prob_down': 0.19
        },
        'beta': 1.85,
        'alpha_annual': 0.188
    },
    
    # From SemiMarkov
    'semi_markov': {
        'fatigue_score': 0.2,
        'days_in_regime': 15,
        'expected_remaining_days': 30,
        'duration_distribution': 'gamma',
        'duration_params': {'a': 5.2, 'scale': 6.1}
    },
    
    # From Features
    'features': {
        'vol_20d': 0.55,
        'vol_percentile': 0.45,  # vs 1-year history
        'drawdown': -0.08,
        'volume_z': 0.5
    },
    
    # Market Context
    'market': {
        'vix': 18.5,
        'vix_percentile': 0.3,
        'spy_ret_20d': 0.05
    }
}
```

---

# 🔄 **Pipeline Diagram**

## **Weekly Heavy Pipeline (Monday 9 AM)**
```
┌─────────────────────────────────────────────────────────┐
│                     MONDAY HEAVY RUN                     │
├─────────────────────────────────────────────────────────┤
│ 1. Data Refresh (30 min)                                │
│    ├─ Fetch latest for all portfolio assets             │
│    ├─ Compute features (vectorized)                     │
│    └─ Store in Parquet                                  │
│                                                         │
│ 2. Regime Detection (15 min)                            │
│    ├─ Run GMM on features                               │
│    ├─ Assign regime labels                              │
│    └─ Check for regime changes                          │
│                                                         │
│ 3. Heavy Analysis (45 min)                              │
│    ├─ Monte Carlo simulations (5000 paths)              │
│    ├─ Compute risk metrics (VaR, CVaR, Kelly)           │
│    ├─ Fit Semi-Markov distributions                     │
│    └─ Cache results                                     │
│                                                         │
│ 4. Portfolio Optimization (15 min)                      │
│    ├─ Compute correlation matrix                        │
│    ├─ Apply correlation penalties                       │
│    └─ Generate weekly report                            │
└─────────────────────────────────────────────────────────┘
         Total: ~105 minutes for full portfolio
```

## **Daily Light Pipeline (9 AM Every Day)**
```
┌─────────────────────────────────────────────────────────┐
│                     DAILY LIGHT CHECK                    │
├─────────────────────────────────────────────────────────┤
│ 1. Quick Data Update (5 min)                            │
│    ├─ Fetch yesterday's close for all assets            │
│    ├─ Update latest features                            │
│    └─ Check for regime drift                            │
│                                                         │
│ 2. Signal Refresh (2 min)                               │
│    ├─ Update compression scores                         │
│    ├─ Update fatigue scores                             │
│    └─ Check trigger conditions                          │
│                                                         │
│ 3. DCA Decision (for tomorrow's DCA) (3 min)            │
│    ├─ Load cached heavy analysis                        │
│    ├─ Apply latest signals                              │
│    ├─ Generate decision with reasoning                  │
│    └─ Store in SQLite                                   │
│                                                         │
│ 4. Alert Check (1 min)                                  │
│    ├─ Check for regime changes                          │
│    ├─ Check for high compression (>70)                  │
│    ├─ Check for crisis regime                           │
│    └─ Send alerts if needed                             │
└─────────────────────────────────────────────────────────┘
         Total: ~11 minutes
```

---

# 📦 **Package List + Caching Approach**

## **requirements.txt (Production)**
```txt
# Core
python>=3.9
pandas>=2.0
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3
statsmodels>=0.14

# Data
yfinance>=0.2.33
requests-cache>=1.1
pandas-datareader>=0.10
fredapi>=0.5
duckdb>=0.9

# Caching
diskcache>=5.6
joblib>=1.3
polars>=0.19  # Fast parquet ops

# Dashboard
streamlit>=1.28
plotly>=5.17
altair>=5.2

# API (future)
fastapi>=0.104
uvicorn>=0.24
pydantic>=2.5

# Testing
pytest>=7.4
pytest-mock>=3.12

# Utils
pyyaml>=6.0
python-dotenv>=1.0
loguru>=0.7
```

## **Three-Layer Caching Strategy**
```python
import diskcache as dc
import duckdb

class ThreeLayerCache:
    def __init__(self):
        # Layer 1: In-memory LRU (fastest)
        self.memory = dc.Cache('cache/memory', size_limit=1_000_000)
        
        # Layer 2: Disk cache (medium)
        self.disk = dc.Cache('cache/disk')
        
        # Layer 3: DuckDB (structured query)
        self.db = duckdb.connect('cache/duckdb.db')
        
        # Layer 4: Parquet (bulk timeseries)
        self.parquet_dir = 'cache/parquet'
    
    def get_ohlcv(self, symbol, period):
        key = f'ohlcv_{symbol}_{period}'
        
        # Try memory first
        if key in self.memory:
            return self.memory[key]
        
        # Try disk
        if key in self.disk:
            data = self.disk[key]
            self.memory[key] = data  # Promote to memory
            return data
        
        # Fetch and cache at all levels
        data = self.fetch_from_source(symbol, period)
        self.memory[key] = data
        self.disk[key] = data
        self.save_to_parquet(symbol, data)
        
        return data
```

---

# 🚀 **Implementation Order**

## **Phase 1A: Data Foundation (Days 1-2)**
```
1. DataFetcher with multi-source support
   - yfinance wrapper with caching
   - Unified output contract
   
2. FeatureComputer 
   - Compute vol, drawdown, returns
   - Store in Parquet
   
3. DuckDB integration
   - Schema setup
   - Query helpers
```

## **Phase 1B: Core Engines Adaptor (Days 3-4)**
```
4. RegimeRiskPlatform adapter
   - Add train_end_date support
   - Caching of simulations
   
5. SemiMarkov adapter
   - Expose fatigue scoring
   - Cache duration distributions
   
6. SignalFactory v2
   - Context-based signals
   - KellyOverride, CrisisOverride, etc.
```

## **Phase 2: DCA Orchestration (Days 5-7)**
```
7. DCARegimeEnhancer (single asset)
   - Integration logic
   - Reasoning chain
   
8. PortfolioDCAManager
   - Multi-asset optimization
   - Correlation penalties
   
9. Scheduler
   - Heavy/light compute triggers
   - Alert system
```

## **Phase 3: Dashboard (Days 8-10)**
```
10. Streamlit dashboard extension
    - Portfolio input
    - DCA decisions table
    - Backtest visualization
    
11. Report generation
    - PDF/HTML reports
    - Email alerts
```

---

# ✅ **Approval Checklist**

| Item | Status | Notes |
|------|--------|-------|
| System Contract (Data Schema + Context Schema) | ⏳ Pending | |
| Pipeline Design (Weekly/Daily split) | ⏳ Pending | |
| Package List | ⏳ Pending | |
| Implementation Order | ⏳ Pending | |

---

# 📝 **What's Delivered After Approval**

1. **Complete `DataFetcher` implementation**
2. **`FeatureComputer` with caching**
3. **Updated `DCARegimeEnhancer` skeleton**
4. **Streamlit dashboard extension template**
