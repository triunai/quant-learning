# 🔎 Engineering Decisions v1.1 — Audit & Patch Notes

> **Status:** ✅ PATCHED  
> **Last Updated:** 2024-12-29  
> **Purpose:** Corrections to v1 spec based on correctness audit. All builds must use this patched version.

---

## ✅ Overall Verdict

The spec is **coherent and implementable** as a Supabase-centric v1. The "heavy/light split + context-first signals + correlation guard" is the right spine.

**This document patches:**
- 5 correctness bugs
- 3 missing schema essentials
- 2 performance assumptions

---

# 🟢 Solid / Approved (No Changes Needed)

| Component | Status |
|-----------|--------|
| Supabase as System-of-Record + Storage for artifacts | ✅ Approved |
| v1 scope: US stocks/ETFs only | ✅ Approved |
| "Adjusted Close for returns, Close for display" | ✅ Approved |
| Context-first signals (no refitting inside signals) | ✅ Approved |

---

# 🔴 Must-Fix Issues (Correctness / Spec Contradictions)

## 1) `crisis_floor` Is Semantically Wrong

### ❌ Original (Wrong)
```python
'crisis_floor': 0.5,  # Crisis regime minimum
```

In a crisis you want to **cap** buying (reduce), not enforce a minimum.

### ✅ Patched
```python
MULTIPLIER_BOUNDS = {
    "min": 0.3,
    "max": 1.5,
    "crisis_cap": 0.5,   # In crisis, multiplier must be <= 0.5
}
```

**Rule:** In Crisis regime, enforce `multiplier <= crisis_cap`.

---

## 2) Stooq Fallback May Not Provide "Adj Close"

### ❌ Problem
Your "non-negotiable" contract requires `Adj Close`. Stooq often won't give true dividend/split adjusted series.

### ✅ Patched
Make fallback contract explicit:

| Scenario | Action |
|----------|--------|
| Source lacks adjusted data | Compute split-adjusted if splits available |
| Otherwise | Use Close and set flags |

**Flags to set:**
```python
{
    'corporate_actions_applied': False,
    'returns_basis': 'close_unadjusted',
    'trust_level': 'lower'  # Warn in UI
}
```

> Keep Stooq as a "keep the lights on" fallback, not a truth source.

---

## 3) "Supabase Is Truth" Conflicts with "Yahoo Changes the Past"

### ❌ Problem
Yahoo can revise corporate actions / adjusted series. Without snapshots, backtests become non-reproducible.

### ✅ Patched: Data Snapshot Rule

Every fetch writes a **raw parquet snapshot** keyed by:

```
{symbol}/{interval}/{start_date}_{end_date}/{source}/{asof_utc}.parquet
```

Canonical tables reference `snapshot_id` (or `artifact_hash`) so every decision ties to **immutable inputs**.

**Schema addition:**
```sql
ALTER TABLE dca_decisions ADD COLUMN snapshot_id UUID REFERENCES artifact_index(id);
```

---

## 4) Train/Test Requirement vs v1 "Illustrative Backtest"

### ❌ Problem
You require `train_end_date` everywhere, but v1 backtests are "illustrative" — this creates half-implemented leakage controls.

### ✅ Patched: Define Three Execution Modes

| Mode | Behavior | Label |
|------|----------|-------|
| `mode="live"` | Uses latest cached heavy artifacts | Production |
| `mode="backtest_illustrative"` | Fits once on full history | ⚠️ "Contains look-ahead bias" |
| `mode="backtest_walkforward"` | True walk-forward (v2) | ✅ Proper |

**Implementation:**
```python
def run_analysis(ticker, mode: Literal["live", "backtest_illustrative", "backtest_walkforward"]):
    if mode == "live":
        return use_cached_artifacts(ticker)
    elif mode == "backtest_illustrative":
        return run_full_fit_once(ticker, label_bias=True)
    elif mode == "backtest_walkforward":
        return run_walk_forward(ticker)  # v2
```

---

## 5) Latency Targets Are Unrealistic Without Portfolio Size Cap

### ❌ Problem
`heavy_weekly < 5 minutes` is only plausible if portfolio is small (3-10 assets).

### ✅ Patched: Scale-Aware Targets

```yaml
latency_targets:
  daily_light:
    per_10_assets: < 30s
  heavy_weekly:
    per_10_assets: < 5-10m   # depending on sims + CPU
  max_portfolio_assets_v1: 20  # Hard cap for v1
```

---

# 🟡 Missing Essentials (Schema + Ops)

## 1) Add `user_id` Everywhere (RLS Reality)

### ❌ Problem
RLS example references `user_id`, but most tables don't include it.

### ✅ Patched Tables

| Table | Add Column |
|-------|------------|
| `portfolios` | `user_id UUID NOT NULL` |
| `portfolio_assets` | `user_id UUID` (or join through portfolio) |
| `dca_decisions` | `user_id UUID NOT NULL` |
| `analysis_runs` | `user_id UUID NOT NULL` |
| `artifact_index` | `portfolio_id UUID` (foreign key) |

---

## 2) Add `strategy_hash` + `engine_version` to Every Stored Artifact + Decision

### ✅ Patched: Auditability Columns

| Column | Type | Description |
|--------|------|-------------|
| `strategy_hash` | `TEXT NOT NULL` | Hash of YAML + engine params |
| `engine_version` | `TEXT NOT NULL` | Git SHA or semantic version |

**Add to:**
- `dca_decisions`
- `artifact_index`
- `analysis_runs`

---

## 3) Add `run_id` Foreign Keys

### ✅ Patched: Run Tracking

```sql
CREATE TABLE analysis_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    run_type TEXT NOT NULL,  -- 'daily_light', 'weekly_heavy'
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    status TEXT NOT NULL,  -- 'running', 'completed', 'failed'
    error_message TEXT,
    strategy_hash TEXT NOT NULL,
    engine_version TEXT NOT NULL
);
```

**Tie everything to a run:**
- `artifact_index.run_id`
- `dca_decisions.run_id`
- `regime_history.run_id`

---

# 📦 Package Spec Audit (Version Fixes)

## ❌ Supabase Python Client Version Outdated

**Original:**
```txt
supabase>=2.0
```

**✅ Patched (Dec 2025 latest):**
```txt
supabase>=2.27,<3
```

---

## ❌ yfinance Minimum Should Allow Current Major

**Original:**
```txt
yfinance>=0.2.33
```

**✅ Patched (yfinance 1.0 released):**
```txt
yfinance>=1.0,<2
```

---

## ✅ requests-cache Is Fine

Latest is 1.2.1 (mid-2024). No changes needed.
```txt
requests-cache>=1.2,<2
```

---

# 🔒 Final Patched Engineering Decisions (v1.1)

## Core Decisions (Unchanged)

1. **System of Record:** Supabase Postgres for canonical state; Supabase Storage for large artifacts; DuckDB local accelerator (rebuildable).

2. **Scope v1:** US stocks + US ETFs only; commodities exposure via GLD/SLV ETFs.

3. **Price truth:** Returns/features computed from **Adj Close when available**. If fallback source lacks adjusted series, store `returns_basis="close_unadjusted"` + `corporate_actions_applied=false`.

## Patched Decisions

4. **Reproducibility:** Every fetch produces immutable `snapshot_id` (raw parquet + metadata). Decisions reference snapshot/hash. **(NEW)**

5. **Execution modes:** `live`, `backtest_illustrative` (labeled), `backtest_walkforward` (v2). **(NEW)**

6. **Pipelines:** (Unchanged)
   - Daily Light: fetch latest bars → update last-row features → generate decisions using cached heavy artifacts → persist decisions
   - Weekly Heavy: full refresh → feature recompute → refit regimes + semi-Markov → sims → risk metrics → store artifacts + indexes

7. **Triggers:** monday + first_of_month + regime change + vol percentile jump + drawdown threshold crossing

8. **Signals:** context-only; never refit engines.

9. **Correlation penalty:** apply only if all assets in a cluster recommend >1×, threshold corr > 0.7, penalty `max(0.6, 1 - 0.15*(n-1))`.

10. **Multipliers:** global `[0.3, 1.5]`, plus **`crisis_cap=0.5`** (must be ≤ 0.5 in Crisis). **(PATCHED)**

11. **API readiness:** separate endpoints later; store stable JSON contract now.

12. **Schema requirements:** include `user_id`, `run_id`, `snapshot_id`, `strategy_hash`, `engine_version` in canonical tables for RLS + auditability. **(PATCHED)**

13. **Latency targets:** Scale-aware, per-10-assets basis. Max 20 assets for v1. **(PATCHED)**

14. **Fallback data:** Stooq is "non-adjusted mode" with explicit flags. **(PATCHED)**

## Dependencies (Patched Versions)

```txt
supabase>=2.27,<3
yfinance>=1.0,<2
requests-cache>=1.2,<2
tenacity>=8,<9
pandas>=2.0
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3
duckdb>=0.9
pyarrow>=14.0
streamlit>=1.28
plotly>=5.17
pytest>=7.4
```

---

# 🔥 Top 3 Fixes (If You Only Do These)

| Priority | Fix |
|----------|-----|
| 1 | Rename **`crisis_floor → crisis_cap`** and enforce `multiplier <= crisis_cap` in Crisis |
| 2 | Add **snapshot_id/hash + strategy_hash + engine_version + run_id** threading through decisions/artifacts |
| 3 | Make Stooq fallback **explicitly "non-adjusted mode"** with flags (don't silently pretend it's Adj Close) |

---

# ✅ Status: v1.1 LOCKED

This patched version supersedes v1.0. All implementations must reference this document.

**Document lineage:**
- `ENGINEERING_DECISIONS_v1.md` → Original (deprecated)
- `ENGINEERING_DECISIONS_v1.1_PATCHED.md` → **Current (use this)**
- `ENGINEERING_DECISIONS_LLM_PROMPT.md` → LLM-ready prompt (update pending)
