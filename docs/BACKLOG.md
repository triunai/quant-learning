# 📋 Quant Platform Backlog

> **Status:** Active  
> **Last Updated:** 2025-12-29  
> **Purpose:** Track issues, improvements, and features identified during code review.

---

## 🔴 P0: Critical Fixes (Before Live Use)

### 1. Kurtosis Mismatch in Simulation

**Issue:** Simulated daily returns have kurtosis ~4.14 vs historical ~8.28. This means the simulation **under-represents extreme events** (tail risk is the whole point of this system).

**Impact:** VaR/CVaR estimates will be too optimistic. Real drawdowns will be worse than simulated.

**Investigation:**
- [ ] Check if empirical residual sampling is truly uniform random
- [ ] Verify no smoothing or averaging is applied to residual pool
- [ ] Consider adding jump diffusion back (currently disabled) with proper separation of normal vs jump residuals

**File:** `battle-tested/PLTR-test-2.py` → `simulate()` method (lines 452-552)

---

### 2. Targets Too Aggressive for Asset Class

**Issue:** Default targets are +50% / -35% in 126 days. For SPY (a broad market ETF), historical hit rates are 0%.

**Impact:** The signal generator always says "Sim > 1.5x historical" because targets are unreachable.

**Fix:**
- [ ] Add asset-class-specific target defaults:
  - ETFs (SPY, QQQ): ±10-15%
  - Large Cap (AAPL, MSFT): ±15-25%
  - Growth/Volatile (PLTR, TSLA): ±30-50%
- [ ] Or: Auto-calculate targets from historical volatility percentiles

**File:** `battle-tested/PLTR-test-2.py` → `__init__()` method

---

### 3. GARCH Not Actually Forecasting

**Issue:** GARCH Vol = Realized Vol exactly. This means either:
- `arch` package not installed properly, or
- GARCH is failing silently and falling back to realized vol

**Fix:**
- [ ] Verify `arch` is installed: `pip install arch`
- [ ] Add explicit logging when GARCH falls back
- [ ] GARCH should produce a *different* number than realized (it's forward-looking)

**File:** `battle-tested/PLTR-test-2.py` → `run_garch()` method

---

## 🟡 P1: Important Improvements

### 4. No True Train/Test Split

**Issue:** GMM is fitted on all 5Y of data, then validated on the same data. This is "illustrative backtest" mode, not true out-of-sample.

**Fix:**
- [ ] Implement walk-forward GMM refitting:
  - Fit on trailing 3Y
  - Validate on next 6mo
  - Roll forward
- [ ] Add `mode` parameter from ENGINEERING_DECISIONS_v1.1_PATCHED.md

**Reference:** `docs/ENGINEERING_DECISIONS_v1.1_PATCHED.md` → Execution Modes

---

### 5. Supabase Persistence Not Implemented

**Issue:** Docs describe Supabase as system-of-record, but no actual persistence layer exists.

**Fix:**
- [ ] Create `infrastructure/supabase/` adapter
- [ ] Store `run_id`, `snapshot_id`, `strategy_hash`, `engine_version` per decision
- [ ] Implement raw parquet snapshots for reproducibility

**Reference:** `docs/ARCHITECTURE.md`, `docs/ENGINEERING_DECISIONS_v1.1_PATCHED.md`

---

### 6. Ensemble Signal Aggregation

**Issue:** Three engines (v7.0, Semi-Markov, Signals Factory) run independently but don't combine into a unified recommendation.

**Fix:**
- [ ] Create weighted vote system:
  - v7.0 signal: 40% weight
  - Semi-Markov position size: 30% weight
  - Signals Factory compression: 30% weight
- [ ] Output single `final_position_size` and `final_direction`

**File:** `to_refine/dashboard_consolidated.py` or new `signals_factory/ensemble.py`

---

## 🟢 P2: Nice to Have

### 7. Regime Change Alerts

**Feature:** Notify when `current_regime` switches via Discord/Telegram webhook.

**Implementation:**
- [ ] Add webhook URL to config
- [ ] Trigger on regime transition detected
- [ ] Include: ticker, old regime, new regime, confidence

---

### 8. Blue Color Theme

**Request:** User prefers blue color scheme over current cyan/plasma.

**Fix:**
- [ ] Update `plt.style.use()` and `sns.set_palette()` in dashboard
- [ ] Change cone chart colors from cyan to blue gradient
- [ ] Update heatmaps to use blues

**Files:** `to_refine/dashboard_consolidated.py`, `to_refine/dashboard.py`

---

### 9. Custom Target Inputs in Dashboard

**Feature:** Let user input custom `target_up` and `target_down` prices in sidebar.

**Implementation:**
- [ ] Add number inputs to sidebar
- [ ] Pass to `RegimeRiskEngineV7` constructor
- [ ] Update JSON export to include custom targets

---

### 10. Backtesting Harness

**Feature:** Simulate actual DCA decisions over historical periods, track P&L.

**Implementation:**
- [ ] Create `backtesting/` module
- [ ] Implement walk-forward decision loop
- [ ] Track: entry date, exit date, position size, PnL
- [ ] Compute Sharpe, Sortino, max drawdown

---

## ✅ Completed

| Task | Date | Notes |
|------|------|-------|
| Consolidated dashboard with full logging | 2025-12-29 | `dashboard_consolidated.py` |
| Terminal-style log capture | 2025-12-29 | `LogCapture` class |
| JSON export with all results | 2025-12-29 | Export tab in dashboard |
| Full calibration suite in dashboard | 2025-12-29 | Walk-forward, multi-threshold |
| Semi-Markov integration | 2025-12-29 | Position sizing + fatigue |
| Signals Factory integration | 2025-12-29 | Vol compression signal |

---

## 📊 Priority Matrix

| Priority | Count | Focus |
|----------|-------|-------|
| 🔴 P0 | 3 | Fix before using with real money |
| 🟡 P1 | 3 | Important for production quality |
| 🟢 P2 | 4 | UX and feature improvements |

---

## 📝 Notes

- The architecture is solid (Clean Architecture principles)
- Statistical methodology is correct (GMM, factor model, semi-Markov)
- Main gaps are in parameter tuning and persistence layer
- Dashboard works; just needs the P0 fixes for reliable signals
