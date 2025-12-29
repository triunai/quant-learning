# 📋 Quant Platform Backlog

> **Status:** Active  
> **Last Updated:** 2025-12-29  
> **Purpose:** Track critical engineering tasks for the Regime Risk Platform.

---

## 🔴 P0: Critical Fidelity Fixes (Prioritized)

### P0-A: Simulation Realism (The "Kurtosis & Clustering" Fix)
**Problem:** Statistical verification confirms kurtosis mismatch (4.14 vs 8.28) and lack of volatility clustering in simulations.
**Root Cause 1:** Independent sampling of market/residuals kills tail dependence (panic coupling).
**Root Cause 2:** Independent daily sampling kills serial correlation.
**Fix Strategy:**
1.  **Coupled Pair Sampling:** Sample `(market_ret, residual)` pairs to preserve joint distribution.
2.  **Stationary Block Bootstrap:** Use random block lengths (Geometric, mean L=10-25) to preserve clustering without periodic artifacts.
3.  **Mode Separation (Avoid Double Counting):**
    *   *Mode A (Regime Model):* Semi-Markov controls switching; use coupled pairs within regimes.
    *   *Mode B (Historical Dependent):* Pure stationary bootstrap of blocked pairs (disables Semi-Markov) as a model-free benchmark.
**Success Criteria:**
*   Excess Kurtosis mismatch < 1.0.
*   Tail Dependence: $P(|r_{asset}| > q_{95} \mid r_{mkt} < q_{05})$ matches history ±10%.
*   ACF($r^2$) matches history at lags 1, 5, 10.

### P0-B: Target Realism (Auto-Calibration)
**Problem:** Hardcoded defaults (1.5x) and `sqrt(T)` scaling break for fat-tailed assets.
**Fix Strategy:**
-   **Empirical Rolling Quantiles:** Use actual historical H-day returns.
    -   `Target Up = Last Price * exp(Quantile(Roll_Ret_H, 0.95))`
    -   `Target Down = Last Price * exp(Quantile(Roll_Ret_H, 0.05))`
-   Automatically adapts to asset volatility and non-normal scaling.
**Success Criteria:** Targets adapt ±20% based on asset vol; Hit rates for "Sim vs Hist" are comparable.

### P0-C: GARCH Reliability & Truthfulness
**Problem:** Silent fallback to realized vol makes validation opaque.
**Fix Strategy:**
-   **Status Tracking:** `self.garch_status = "OK" | "MISSING" | "FAILED"`.
-   **Convergence Check:** Warn if `garch_vol == realized_vol` (indicates silent fit failure).
-   **Dashboard:** Explicit "ACTIVE" vs "FALLBACK" indicator.
**Success Criteria:** Dashboard shows status with confidence intervals.

### P0-D: Simulation Calibration Diagnostic
**Problem:** Mismatches happen, but root cause is opaque.
**Fix Strategy:**
-   Implement `diagnose_mismatch()` returning a **"Blame Table"**:
    -   **Unconditional:** Mean, Std, Skew, Excess Kurtosis.
    -   **Clustering:** ACF($r^2$) at lags 1, 5, 20.
    -   **Regime:** Occupancy (Hist vs Sim), Conditional Mean/Vol.
    -   **Tail:** Conditional probability of asset crash given market crash.

---

## 🟡 P0.5: Operational Control & Stability

### 1. Simulation Config & Reproducibility
**Problem:** "Production" runs are slow; Parallel RNG can be flawed.
**Fix:**
-   `SimulationConfig` (debug/fast/production).
-   **RNG Safety:** Use `np.random.SeedSequence(seed).spawn(n_workers)` for independent parallel streams.

### 2. Regime Definition Stability
**Problem:** Regime counts (3 vs 4) can be arbitrary.
**Fix:** Test sensitivity to `n_regimes` and stability of boundaries over time.

---

## 🟡 P1: Robustness & Hygiene

### 1. API Safety (The "Landmine")
**Problem:** `run()` crashes if `ingest_data()` isn't called.
**Fix:** Add state guard `if self.data is None: self.ingest_data()`.

### 2. Walk-Forward Stability & Leakage Prevention
**Problem:** Current validation ignores outlier stability and risks data leakage.
**Fix:**
-   Rolling window GMM refit.
-   **Leakage Guard:** Refit Scaler and GMM on *train* split only; do not leak OOS data into regime definitions.

---

## 🟢 P2: Features & UX

- [ ] **Ensemble Voting:** Combine v7.0 (Macro/Regime) + Semi-Markov (Duration) + Signals (Technical).
- [ ] **Supabase Persistence:** Save `run_id` and results.
- [ ] **Regime Alerts:** Webhook notifications.

---

## 🔬 Deep Dive: Implementation Specs

### 1. Coupled Pair Sampling & Storage (P0-A)
Store pairs as a DataFrame indexable by time for block sampling.
```python
def prepare_coupled_pairs(self):
    # Dataframe structure for easy block slicing
    # Index: Date, Columns: [Market_Ret, Asset_Residual, Regime]
    
    # ... computation details ...
    
    self.pair_db = pd.DataFrame({
        'mkt_ret': market_rets,
        'resid': residuals,
        'regime': regimes
    }, index=common_idx)
    
    # Also keep regime-specific buckets for Mode A
    self.regime_buckets = {r: self.pair_db[self.pair_db['regime'] == r] for r in range(self.n_regimes)}
```

### 2. Empirical Targets (P0-B)
Use rolling H-day returns to capture actual horizon distributions.
```python
def auto_calibrate_targets(self, horizon_days=20, confidence=0.05):
    # Calculate H-day rolling log returns
    log_prices = np.log(self.data['Close'])
    roll_rets = (log_prices.shift(-horizon_days) - log_prices).dropna()
    
    # Empirical quantiles
    ret_up = np.percentile(roll_rets, (1-confidence)*100)
    ret_down = np.percentile(roll_rets, confidence*100)
    
    self.target_up = self.last_price * np.exp(ret_up)
    self.target_down = self.last_price * np.exp(ret_down)
```

### 3. Simulation Diagnostics (P0-D)
Diagnose *why* stats diverge.
```python
def diagnose_simulation_mismatch(self, sim_paths, hist_returns):
    sim_returns = np.diff(np.log(sim_paths), axis=1).flatten()
    
    # 1. Tail Dependence (Panic Coupling)
    mkt_crash_thresh = np.percentile(self.market_returns, 5)
    # Conditional probability: P(Asset < X | Mkt < Y)
    sim_crash_prob = ... 
    
    # 2. Clustering (ACF of squared returns)
    # ...
```

### 4. Parallel RNG Safety (P0.5)
Ensure robust randomness.
```python
from numpy.random import SeedSequence, default_rng

def simulate_parallel(self, n_workers=4):
    ss = SeedSequence(12345)
    child_seeds = ss.spawn(n_workers)
    # pass child_seeds[i] to worker i
```
