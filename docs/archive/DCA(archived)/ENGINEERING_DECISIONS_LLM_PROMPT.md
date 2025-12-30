# 🧱 Engineering Decisions v1 — LLM-Ready Prompt Pack

> **Status:** ✅ LOCKED (Copy-paste ready for any LLM)  
> **Last Updated:** 2024-12-29  
> **Purpose:** Single source of truth for v1 DCA Weather Report platform, aligned with Supabase pivot.

---

## ✅ Alignment Summary

We're building **"DCA Weather Report for Any Asset"** with:
- **Multi-asset portfolio input**
- **Per-asset DCA adjustment**
- **Portfolio-level correlation guardrails**

But we keep **v1 scope tight** and **v2 hooks clean**.

**Core pipeline:**
```
Data → Features → Engines → Context → Signals → Decision → Store → UI
```

---

# 🧱 Engineering Decisions v1

## 0) Scope & Truth Policy (Non-Negotiables)

| Decision | Value |
|----------|-------|
| v1 asset scope | **US stocks + ETFs via Yahoo** (including **GLD/SLV**) |
| Spot commodities + crypto | **v2 adapters** |
| Returns truth | Use **Adj Close** for returns + all feature computations |
| Display | **Close** only for display |

### Reproducibility Rule

Every run must be **replayable** by referencing:
- `data_source`
- `asof_timestamp`
- `strategy_version_hash`
- `engine_version`
- `feature_version`

---

## 1) Storage: Supabase Is the "System of Record"

### ✅ Supabase Postgres Stores (Canonical State)

| Table | Contents |
|-------|----------|
| Portfolio config | Symbols, base DCA, asset_class, weights |
| Decisions log | Monthly decisions, reasoning, executed info |
| Regime history | Date → regime label/probabilities |
| Run metadata | What was computed, with what versions |

### ✅ Supabase Storage Stores (Large Artifacts)

| Bucket | Contents |
|--------|----------|
| `raw_ohlcv/` | Parquet snapshots |
| `features/` | Computed feature parquets |
| `models/` | GMM pickle/joblib, semi-markov params JSON |
| `simulations/` | Paths parquet or summary stats |
| `reports/` | Generated PDFs/HTML |

### ✅ DuckDB Stays (Optional But Recommended)

DuckDB is **not competing with Supabase** — it's your **local analytics accelerator**:

| Use Case |
|----------|
| Fast correlation matrices / joins / backtests |
| Cheap "feature store" locally while iterating |
| Lets Streamlit feel instant without hammering Postgres |

> **Supabase = truth & sync. DuckDB = speed & iteration.**

Supabase Python client: `create_client(url, key)` — [Supabase Docs](https://docs.supabase.com/docs/reference/python/initializing)

---

## 2) Data Acquisition: "2-Source Reliability"

| Source | Role |
|--------|------|
| **yfinance** | Primary (fast iteration, broad coverage) |
| **Stooq via pandas-datareader** | Fallback for OHLCV continuity when Yahoo breaks/throttles |

Stooq reader: `StooqDailyReader` — [Pandas DataReader Docs](https://pandas-datareader.readthedocs.io/en/latest/readers/stooq.html)

> ⚠️ Yahoo 429 throttling issues are real → **must wrap fetch with cache + retry/backoff**

### Data Fetch Hard Rules

| Rule |
|------|
| Always output a **normalized OHLCV contract** |
| Always store **metadata**: `symbol`, `source`, `asof_utc`, `timezone`, `currency`, `corporate_actions_applied` |
| Always cache raw pulls (see caching section) |

---

## 3) Caching Strategy (So You Don't DOS Yourself)

### Caching Layers (Practical)

| Layer | What |
|-------|------|
| 1 | **HTTP cache:** `requests-cache` (Yahoo/Stooq calls) |
| 2 | **Data snapshots:** Parquet in Storage (or local while dev) |
| 3 | **Feature cache:** keyed by `(symbol, feature_version, data_asof)` |
| 4 | **Model cache:** keyed by `(symbol, model_version, train_end_date)` |
| 5 | **Decision cache:** in Postgres (canonical) |

### Cache Keys (Must Include Versioning)

```
{symbol}:{source}:{interval}:{period}:{asof_utc}:{feature_version}:{engine_version}
```

---

## 4) Pipeline Design (Heavy/Light Split + Supabase)

### Daily "Light" (Fast, Frequent)

| Step | Action |
|------|--------|
| 1 | Fetch latest bar(s) |
| 2 | Update last row features |
| 3 | Compute/refresh signals |
| 4 | Generate DCA decision **using cached heavy outputs** |
| 5 | Write decision row to Supabase |

### Weekly "Heavy" (Expensive, Scheduled)

| Step | Action |
|------|--------|
| 1 | Recompute features batch |
| 2 | Refit regime model (GMM) + semi-markov params |
| 3 | Run simulations (or risk summary) |
| 4 | Update regime history + risk metrics |
| 5 | Store artifacts to Storage, store pointers + hashes in Postgres |

### Heavy Triggers (Event-Based)

| Trigger |
|---------|
| Regime label changed |
| 20d vol percentile jump > threshold |
| Drawdown crosses -15% boundary |
| First of month |

---

## 5) Engine Interfaces (To Avoid Future Rewrite)

### v1 Requirement

| Requirement |
|-------------|
| Engines must accept an optional **`train_end_date`** even if MVP doesn't fully use it yet |
| MVP backtest may be "illustrative" but MUST be labeled as such |

### v2 Requirement (True Proof)

| Requirement |
|-------------|
| Walk-forward: for each decision date `t`, fit on `<=t`, decide at `t`, evaluate forward |

---

## 6) Context-First Signals (No Spaghetti)

### Rules

| Rule |
|------|
| Signals must **never** refit engines |
| Signals consume a shared `Context` object |

### Context Contains

| Field | Source |
|-------|--------|
| Platform outputs | Regime, kelly, dd probabilities |
| Semi-markov outputs | Fatigue, days in regime |
| Feature snapshot | Vol, drawdown, volume_z |
| Market context | VIX, etc. |

### Signals Interface

| Method | Usage |
|--------|-------|
| `fit(df)` | Optional |
| `predict(context)` | **Mandatory** for DCA overlays |

---

## 7) Portfolio Correlation Guardrail

### Policy: Only Penalize "Risk-On Clusters"

| Step | Action |
|------|--------|
| 1 | Build corr matrix on recent returns |
| 2 | Find clusters where corr > 0.7 |
| 3 | Apply penalty **only if all assets in cluster are recommending > 1.0x** |

---

## 8) Defaults (Ship v1)

| Setting | Default | Notes |
|---------|---------|-------|
| **VIX fallback** | Last cached in DB, else constant **20** | With warning flag |
| **Simulation count** | 1000 (interactive), 3000-5000 (heavy) | |
| **Regime truth** | 3-regime platform is source-of-truth | Semi-markov adds fatigue overlay only |
| **Skip month policy** | Never force skip | Allow per-asset "min floor" = 0.3x |
| **Max multiplier** | Cap at **1.5x** | 2.0x feels aggressive in portfolios |

---

# 🗄️ Supabase Schema

### Tables (Postgres)

| Table | Purpose |
|-------|---------|
| `portfolios` | Portfolio configuration |
| `dca_decisions` | Decision log with reasoning |
| `regime_history` | Historical regime snapshots |
| `analysis_runs` | Run metadata + versions |
| `artifact_index` | Pointers to Storage objects + hashes |

### Buckets (Storage)

| Bucket | Contents |
|--------|----------|
| `raw_ohlcv/` | Raw OHLCV parquets |
| `features/` | Computed feature parquets |
| `models/` | Model artifacts |
| `simulations/` | Simulation outputs |
| `reports/` | Generated reports |

### Security Decision

| Environment | Key Type |
|-------------|----------|
| Streamlit/local runner | **Service role key** (server-side only) |
| React later | **Anon key + RLS** (never expose service role in browser) |

---

# 📦 Package Picks (Tight + Proven)

### Data & Reliability

| Package | Purpose |
|---------|---------|
| `yfinance` | Primary data source |
| `pandas-datareader` | Stooq fallback |
| `requests-cache` | HTTP caching |
| `tenacity` | Retry/backoff |

### Storage

| Package | Purpose |
|---------|---------|
| `supabase` (supabase-py) | System of record |
| `pyarrow` | Parquet I/O |
| `duckdb` | Local analytics accelerator |

### Compute

| Package | Purpose |
|---------|---------|
| `numpy`, `pandas` | Core |
| `scipy`, `scikit-learn` | Stats + GMM |
| `statsmodels` | OLS + diagnostics |

### UI

| Package | Purpose |
|---------|---------|
| `streamlit` | Dashboard |
| `plotly` | Interactive charts |

### Testing

| Package | Purpose |
|---------|---------|
| `pytest` | Invariant tests |

---

# 📝 Single-Paragraph Prompt (For Any LLM)

> Build a multi-asset DCA Weather Report platform. Supabase Postgres is the system-of-record for portfolios, decisions, regime history, and run metadata; Supabase Storage holds large artifacts (parquet snapshots, model pickles, simulation outputs). Use yfinance as primary OHLCV source with requests-cache + retry/backoff; add Stooq via pandas-datareader as fallback. Returns/features use Adj Close; Close is display-only. Pipeline is daily-light (update bars, features, signals, decision using cached heavy outputs) plus weekly-heavy (refit regime + semi-markov, run sims/risk summaries, store artifacts). Signals are context-first and never rerun engines. Regime truth is 3-state platform; semi-markov adds fatigue overlay. Default caps: multiplier in [0.3, 1.5], VIX fallback last-cached else 20, sims=1000 interactive / 3000–5000 heavy, correlation penalty applies only to clusters where all assets recommend >1x.

---

# 🗂️ Ready for Module Layout

When you paste your current repo tree (folders + filenames), I'll map these decisions into a **concrete module layout**:

- Where Supabase client lives
- Where artifacts index lives
- What becomes "interfaces" vs "implementations"

---

# ✅ Status: LOCKED

All decisions finalized. Ready for:
1. **Module layout mapping** (pending repo tree)
2. **Phase 1A implementation** (on your go-ahead)
