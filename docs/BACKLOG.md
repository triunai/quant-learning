# 📋 Quant Platform Backlog

> **Status:** Active  
> **Last Updated:** 2025-12-29  
> **Purpose:** Track critical engineering tasks for the Regime Risk Platform.

---

## 🔴 P0: Critical Fidelity Fixes (Scientific Sanity Check)

### P0-A: Simulation Realism (The "Invariants" Fix)
**Problem:** Simulation fails to reproduce stylized facts: Heavy Tails, Volatility Clustering, and Crisis Feedback.
**FIX STRATEGY:**
1.  **Coupled Pair Sampling:** Sample `(market_ret, residual)` time-ordered vectors to preserve joint distribution.
2.  **Stationary Block Bootstrap:** Random block lengths (Geometric distribution) to preserve clustering without periodic artifacts.
    *   *Tuning:* Select mean block length $L$ by matching ACF($r^2$) decay.
3.  **Mode Separation (Avoid Double Counting):**
    *   *Mode A (Structural):* Semi-Markov with Regime-Dependent Beta (Beta must spike in crisis).
    *   *Mode B (Benchmark):* Pure Stationary Bootstrap of squared/paired vectors (The "Truth Serum").
    *   *Mode C (Vol-Adaptive):* Filtered Historical Simulation (FHS) using GARCH/EGARCH to filter vol, bootstrap residuals, and simulate forward.
**Success Criteria:**
*   **Tail Dependence:** Lower vs Upper dependence matches history ±10%.
*   **Clustering:** Time-series ACF($r^2$) matches history at lags 1, 5, 10.
*   **Leverage Effect:** Correlation($r_t, \sigma^2_{t+1}$) is negative.

### P0-B: Target Realism (First-Passage Calibration)
**Problem:** Calibrating on *terminal* returns mismatches the "hit probability" (Path Extrema) concept.
**FIX STRATEGY:**
-   **Path Extrema Quantiles:** Calibrate targets using the distribution of Max/Min excursions over horizon $H$.
    -   $M^{up}_t = \max_{1\le k \le H} \ln(P_{t+k}/P_t)$
    -   $M^{down}_t = \min_{1\le k \le H} \ln(P_{t+k}/P_t)$
    -   *Correction:* Must use forward-looking window loop.
-   **State-Dependent Targets:** Targets must be conditional on current Regime/Vol bucket.

### P0-C: GARCH Reliability & Truthfulness
**Problem:** Silent fallback obscures model failure.
**FIX STRATEGY:**
-   **Status Class:** `GARCHStatus` (status, confidence, persistence, fallback_reason).
-   **Fallback Hierarchy:** 1. GARCH(1,1) -> 2. EWMA ($\lambda=0.94$) -> 3. Rolling Realized Vol -> 4. Long-term Average.
-   **Roadmap:** Adopt **Filtered Historical Simulation (FHS)** to allow GARCH to influence simulation clustering.

### P0-D: Simulation Calibration Diagnostic (The "Blame Table")
**Problem:** Mismatches happen; root cause is opaque.
**FIX STRATEGY:**
-   **Unconditional:** Mean, Std, Skew, Excess Kurtosis.
-   **Clustering:** Time-series ACF($r^2$) (computed per path, then averaged).
-   **Leverage Effect:** Corr($r_t, r_{t+1}^2$).
-   **Tail Dependence:** Lower vs Upper conditional probabilities.
-   **Regime Fidelity:** Compare Regime Frequency (Occupancy) and Conditional Means/Vols (Hist vs Sim).

---

## 🟡 P0.5: Operational Control & Stability

### 1. Simulation Config & Reproducibility
**Problem:** RNG collisions in parallel runs; slow dev cycles.
**Fix:**
-   `SimulationConfig` with modes:
    -   `debug`: 1000 sims, block_len=1
    -   `fast`: 5000 sims, block_len=3
    -   `production`: 50,000 sims, block_len=geometric, parallel=True
-   **RNG Safety:** Use `np.random.SeedSequence(seed).spawn(n_workers)`.

### 2. Regime Definition Stability
**Problem:** Arbitrary boundaries (3 vs 4 regimes).
**Fix:** Test sensitivity to `n_regimes` and stability of boundaries over time.

---

## 🟡 P1: Model Confidence & Institutional Safety

### 1. Institutional Pitfalls Defense
-   **Leakage Guard:** Refit Scaler/GMM/Beta on *Train* split only.
-   **Beta Instability:** Beta assumes constant; consider Regime-Specific Beta or Vol-dependent Beta.
-   **Scarcity Risk:** Fallback to "Same Stress Quadrant" rather than just "Nearest Vol".

### 2. Walk-Forward Stability Check (The "Leakage" Fix)
**Reference:** *Critical Bug in Walk-Forward Validation*
**Problem:** `walk_forward_validation` incorrectly uses `current` price for historical targets (Look-ahead bias).
**Fix:**
```python
# Use percent moves relative to START of fold
up_pct_move = (self.target_up / self.last_price) - 1
# if path_max >= start_price * (1 + up_pct_move): ...
```
**Enhancement:** Implement rolling window GMM refit to test regime stability out-of-sample.

### 3. Trust Score Metric (0-100)
-   Aggregate fidelity score (Stats 40%, Stability 30%, OOS 20%, Context 10%).

### 4. Defensive Kelly Logic ("Kelly Betrayal")
-   Scale position size by `regime_confidence` (GMM posterior).
-   Hard penalty if VIX > 30.

---

## 🛠️ P2: Engineering Enablers & Optimizations

### 1. Memory Optimization (Sparse Output)
**Problem:** Storing 5000 x 126 paths consumes excessive RAM.
**Fix:**
```python
def simulate(self, use_sparse_output=True):
    # Track hitting times online, do not store full paths
    up_hit_times = np.full(self.simulations, np.inf)
    # ... update in loop ...
    return {'up_hit_times': up_hit_times, 'final_prices': ...}
```

### 2. Parallel Simulation
**Problem:** CPU bound.
**Fix:** Use `ProcessPoolExecutor`.
**Crucial:** Use `np.random.SeedSequence` to ensure RNG safety across workers.

### 3. Caching System
**Problem:** Repeated runs slow down dev.
**Fix:** Disk cache keyed by `(ticker, date, method_args)`.

### 4. API Safety
**Problem:** `run()` crashes if `ingest_data()` isn't called.
**Fix:** Add state guard `if self.data is None: self.ingest_data()`.

---

## 🟢 P3: Features & UX

- [ ] **Ensemble Voting:** Combine v7.0 (Macro/Regime) + Semi-Markov (Duration) + Signals (Technical).
- [ ] **Supabase Persistence:** Save `run_id` and results.
- [ ] **Regime Alerts:** Webhook notifications.
- [ ] **Enhanced Risk Metrics:** Add CDaR (Conditional Drawdown at Risk), Ulcer Index, Omega Ratio.

---

## 🔬 Deep Dive & Implementation Context

### 1. Coupled Pair Sampling (Detailed)
```python
def prepare_coupled_pairs(self):
    # Store aligned pairs: Market Ret, Asset Residual, Regime
    # Important: Store as Time Series (Date Index) for Block Bootstrapping
    self.pair_db = pd.DataFrame({'mkt': ..., 'eps': ..., 'regime': ...}, index=dates)

def get_geometric_block_length(self, mean_length=20):
    # Stationary Bootstrap: Random block length
    return np.random.geometric(1/mean_length)
```

### 2. First-Passage Target Calibration
```python
def auto_calibrate_targets(self, h=20, conf=0.05):
    # Correct Forward-Looking Max/Min
    roll_max = [np.max(log_prices[t:t+h]) - log_prices[t] for t in range(len(log_prices)-h)]
    # ... Quantiles of roll_max ...
```

### 3. Scarcity & Hierarchical Borrowing
```python
def handle_scarce_regime(r, min_samples=80):
    # Fallback: Borrow Global, Scale by Volatility Ratio
    scale = regime_vol / global_vol
    scaled_pairs = [(mkt, eps * scale) for mkt, eps in self.global_pairs]
    return scaled_pairs
```

### 4. Robust/Defensive Kelly
```python
def compute_regime_aware_kelly(self, paths, current_regime):
    # ... Standard Kelly ...
    # Adjust for Regime Confidence & VIX
    if regime_conf < 0.7: kelly *= regime_conf
    if self.vix > 30: kelly *= 0.5
    return kelly
```
