# 📋 Quant Platform Backlog

> **Status:** Active  
> **Last Updated:** 2025-12-30  
> **Purpose:** Track critical engineering tasks for the Regime Risk Platform.  
> **Current Version:** v7.1 (Coherent Factor Model)

---

## ✅ Recently Completed (2025-12-30)

### P0-E: Coherent Factor Model Fix ✅ COMPLETE
**Problem:** Median alpha + asymmetric beta created non-zero mean residuals, causing under-prediction in high-alpha regimes.

**Fix Applied:**
- Standard OLS beta (not asymmetric)
- Mean-based alpha (not median)
- Zero-mean residuals enforced by construction
- R² check for empirical sampling fallback

**Validation:**
- All regimes now have 0.0000% residual mean
- What-if test: Momentum (61.9%) > Bear (45.4%) ✓
- Drift decomposition error: 0.00%

**Reference:** `docs/immediate-tasks/phases/phase-2-mode-a-fundamentals.md`

---

## 🔬 Research Discovery: Kurtosis-Persistence Relationship

**Finding:** Cross-sectional analysis (22 stocks) revealed kurtosis predicts regime duration (r = +0.84).

**Key Insight:** Fat-tail events CREATE persistent regimes (opposite of conventional wisdom).

| Stock | Kurtosis | Avg Duration | Type |
|-------|----------|--------------|------|
| META | 26.6 | 119.5 days | Fat-Tail |
| JPM | 5.0 | 59.8 days | Normal |
| COIN | 2.6 | 18.1 days | Noise |

**Next Steps:**
- [ ] Implement stock classification (Fat-Tail vs Noise)
- [ ] Adaptive n_regimes based on kurtosis
- [ ] Anchor event detection algorithm

**Reference:** `research/papers/kurtosis_regime_persistence/draft.md`

---

## 🔴 P0: Critical Fidelity Fixes (Scientific Sanity Check)

### P0-A: Simulation Realism (The "Invariants" & "Triple Mode" Fix)
**Problem:** Simulation fails to reproduce stylized facts: Heavy Tails, Volatility Clustering, Leverage Effect, and Crisis Feedback.
**FIX STRATEGY:**
1.  **Coupled Pair Sampling:** Sample `(market_ret, residual)` time-ordered vectors to preserve joint distribution (Panic Coupling).
2.  **Stationary Block Bootstrap (Mode B - "The Truth Serum"):** Random block lengths (Geometric distribution) to preserve clustering without periodic artifacts. Acts as the **Null Hypothesis** benchmark.
    *   *Tuning:* Minimize distance between $\text{ACF}^{hist}(r^2)$ and $\text{ACF}^{sim}(r^2)$ at lags 1,5,10,20.
3.  **Mode Separation (Avoid Double Counting):**
    *   *Mode A (Structural):* Semi-Markov with Regime-Dependent Beta.
        *   **Endogenous Feedback:** Crisis probability $P(S_{t+1}=\text{Crisis})$ must depend on market shock ($r_{mkt} < q_{05}$) or VIX level.
    *   *Mode B (Benchmark):* Pure Stationary Bootstrap of squared/paired vectors.
    *   *Mode C (Vol-Adaptive):* **Filtered Historical Simulation (FHS)** using EGARCH/GJR-GARCH to filter vol $\to$ bootstrap standardized residuals $\to$ simulate forward.
**Success Criteria:**
*   **Tail Dependence:** Matches history ±10% at q01/q05/q10 using Lower $\lambda_L$ and Upper $\lambda_U$ (with Bootstrap CI).
*   **Clustering:** Time-series ACF($r^2$) matches history.
*   **Leverage Effect:** Correlation($r_t, \sigma^2_{t+1}$) is negative.

### P0-B: Target Realism (First-Passage Calibration)
**Problem:** Calibrating on *terminal* returns mismatches the "hit probability" (Path Extrema) concept.
**FIX STRATEGY:**
-   **Path Extrema Quantiles:** Calibrate targets using the distribution of Max/Min excursions over horizon $H$.
    -   $M^{up}_t = \max_{1\le k \le H} \ln(P_{t+k}/P_t)$
    -   $M^{down}_t = \min_{1\le k \le H} \ln(P_{t+k}/P_t)$
    -   *Correction:* Must use forward-looking window loop (non-overlapping windows preferred).
-   **State-Dependent Targets:** Targets must be conditional on current Regime/Vol bucket.
-   **First-Passage Ordering:** Replace "Edge" with $P(\tau_{up} < \tau_{down})$ vs $P(\tau_{down} < \tau_{up})$ (and track "Both"/"Neither").
-   **Absorbing Boundaries:** Ensure simulation stops for a path once a barrier is hit to correctly model the stochastic stopping time.

### P0-C: GARCH Reliability & Truthfulness
**Problem:** Silent fallback obscures model failure.
**FIX STRATEGY:**
-   **Status Class:** `GARCHStatus` (status, confidence, persistence, fallback_reason).
-   **Fallback Hierarchy:** 1. EGARCH/GARCH -> 2. EWMA ($\lambda=0.94$) -> 3. Rolling Realized Vol -> 4. Long-term Average.
-   **Automatic Selection:** Score models by OOS Volatility Forecast MSE (Rolling Window).
    -   *Enhancement:* Use **QLIKE Loss** ($log(\hat{\sigma}^2) + r^2/\hat{\sigma}^2$) for volatility scoring (handles asymmetry).
-   **Roadmap:** FHS (Mode C) allows GARCH to actually drive simulation, not just display a number.

### P0-D: Simulation Calibration Diagnostic (The "Blame Table")
**Problem:** Mismatches happen; root cause is opaque.
**FIX STRATEGY:**
-   **Unconditional:** Mean, Std, Skew, Excess Kurtosis.
-   **Clustering:** Time-series ACF($r^2$) (computed per path, then averaged).
-   **Leverage Effect:** Corr($r_t, r_{t+1}^2$).
-   **Tail Dependence:** Lower ($P(Asset < q05 | Mkt < q05)$) vs Upper conditional probabilities.
-   **Regime Fidelity:** Compare Regime Frequency (Occupancy) and Conditional Means/Vols (Hist vs Sim).
-   **Coverage Backtests:** Kupiec (unconditional) + **Christoffersen (independence)** tests for VaR validity.
-   **Cross-Correlation:** Check if correlations go to 1 in crisis (Hist vs Sim).
-   **Extreme Event Clustering:** $P(|r_t|>\theta \mid |r_{t-1}|>\theta)$ for $\theta \in \{95, 99\}$.
-   **Regime Transition Check:** Basic Z-stat test for negative market return precedent.
-   **Statistical Comparison:** KS / Anderson-Darling tests for Hist vs Sim distribution match.

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

### 2. Regime Definition Stability & Semantics
**Problem:** Arbitrary boundaries (3 vs 4 regimes).
**Fix:** Test sensitivity to `n_regimes` and stability of boundaries over time.
**New:** Test parameter stability (Rolling Beta) within regimes.
**New:** Regime Semantic Validation (Centroid Drift, Scarcity check, Interpretability).

### 3. Evaluation Suite Freeze (Anti-Overfit Guardrail)
**Problem:** Simulator tuning can “fit PLTR” and fail to generalize. Triple-mode + many diagnostics creates infinite degrees of freedom.
**Fix:** Define a **frozen, minimal evaluation suite** used for every change.
-   **Asset Panel (3–6 instruments):** 1 broad ETF (SPY), 1 tech beta (QQQ), 1 idiosyncratic (PLTR/TSLA).
-   **Horizon Set:** {20d, 126d}.
-   **Threshold Set:** {10%, 30%}.
-   **Diagnostics Subset (must-pass):** ACF(r²) distance, Tail dependence λL/λU, VaR (Kupiec+Christoffersen), First-passage ordering.
-   **Rule:** No feature “ships” unless it improves or does not degrade scores on the frozen suite. "No tuning to pass metrics; improvements must generalize across ≥ 3 assets."

---

## 🟡 P1: Model Confidence & Institutional Safety

### 1. Data Integrity, Scalability & Reproducibility
**Problem:** Bad data kills good models.
**Fix:**
-   **Corporate Actions:** Verify split adjustments.
-   **Outlier Filters:** Policy for bad ticks.
-   **Consistency Check:** Ensure log-returns used consistently across all modules.
-   **Survivorship Bias:** Warning flag for universe selection.
-   **Experiment Tracking:** Hash(Data + Code + Config) saved with results for exact reproducibility.
-   **Data Provenance:** Store source/vendor, calendar, and Universe definition.

### 2. Institutional Pitfalls Defense
-   **Leakage Guard:** Refit Scaler/GMM/Beta on *Train* split only.
-   **Beta Instability:** Beta assumes constant; consider Regime-Specific Beta or Vol-dependent Beta.
-   **Scarcity Risk:** Fallback to "Same Stress Quadrant" rather than just "Nearest Vol".
-   **Parameter Uncertainty:** Bootstrap parameters (Posterior sampling) to reflect estimation risk in tails.

### 3. Model Risk Quantification
**Problem:** Unknown sources of error.
**Fix:** Decompose variance contribution: Parameter Estimation vs Model Specification vs Data Quality.
**Enhancement:** Propagate parameter uncertainty into forecast distributions.

### 4. Walk-Forward Stability Check (The "Leakage" Fix)
**Reference:** *Critical Bug in Walk-Forward Validation*
**Problem:** `walk_forward_validation` incorrectly uses `current` price for historical targets (Look-ahead bias).
**Fix:**
```python
# Use percent moves relative to START of fold
up_pct_move = (self.target_up / self.last_price) - 1
# if path_max >= start_price * (1 + up_pct_move): ...
```
**Enhancement:** Implement rolling window GMM refit to test regime stability out-of-sample.

### 5. Trust Score Metric (0-100)
-   Aggregate fidelity score (Stats 40%, Stability 30%, OOS 20%, Context 10%).

### 6. Defensive Kelly Logic ("Kelly Betrayal")
-   Scale position size by `regime_confidence` (GMM posterior).
-   Hard penalty if VIX > 30.

### 7. Economic Hypothesis Testing
**Problem:** Models can fit stats but fail economics.
**Fix:** Test specific hypotheses: Leverage Effect, Volatility Feedback, Risk Premium (Higher vol = higher expected return?).

### 8. Out-of-Sample Stress Testing
**Problem:** History may not contain future shocks.
**Fix:** Run model on specific scenarios: 1987 Crash, 2008 Lehman, 2020 COVID, Volmageddon. Evaluate exceedances.

### 9. Parameter Stability Monitoring
**Problem:** Silent drift in Beta or Regime definition.
**Fix:** Monitor rolling parameters. Use Chow Test / CUSUM / Bai-Perron to detect structural breaks.

---

## 🛠️ P2: Engineering Enablers & Optimizations

### 1. Memory Optimization (Sparse Output)
**Problem:** Storing 5000 x 126 paths consumes excessive RAM.
**Fix:**
```python
def simulate(self, use_sparse_output=True):
    # Track hitting times online, do not store full paths
    # Accumulate MaxDD, Worst Drawdown, Hit Times
    up_hit_times = np.full(self.simulations, np.inf)
    # ... update in loop ...
    return {'up_hit_times': up_hit_times, 'max_drawdowns': ...}
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

### 5. Mode Convergence Test
**Problem:** Triple Mode explosion.
**Fix:** Verify Mode A/B/C converge on central usage at large N (Mean/Median), diverge in tails (VaR/CVaR).

### 6. Mode Ensemble / Evidence Weighting (BMA) (New)
**Problem:** Single mode selection implies false certainty.
**Fix:** Average modes (A/B/C) using Bayesian Model Averaging.
-   **Weights:** $\propto \exp(-0.5 \cdot \Delta BIC)$ or WAIC.
-   **Output:** Weighted risk forecasts + Model Uncertainty Band.
-   **Dominance:** Report "Mode Dominance" (e.g., "Mode B is 90% probable").

### 7. Extreme Value Theory (EVT) Bridge (New - P2)
**Problem:** History does not contain Black Swans; Bootstrap is bounded by history.
**Fix:** Use Generalized Pareto Distribution (GPD) to extrapolate tails beyond historical max/min. Bridge historical pairs with synthetic extreme moves.

---

## 🟢 P3: Features & UX

- [ ] **Ensemble Voting:** Combine v7.0 (Macro/Regime) + Semi-Markov (Duration) + Signals (Technical).
- [ ] **Supabase Persistence:** Save `run_id` and results.
- [ ] **Regime Alerts:** Webhook notifications.
- [ ] **Enhanced Risk Metrics:** Add CDaR (Conditional Drawdown at Risk), Ulcer Index, Omega Ratio.
- [ ] **Causal Validation Framework:** Use **Granger Causality** (VAR) to formally validate endogenous feedback (Scientific Advanced Step).

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
    # Tuning: Match ACF(r^2) decay profile
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

### 5. Success Metrics Benchmark
```python
class SuccessMetrics:
    @staticmethod
    def simulation_realism_success(sim_stats, hist_stats):
        return {
            'kurtosis_match': abs(sim_stats['kurtosis'] - hist_stats['kurtosis']) < 1.0,
            'acf_match': all(abs(sim_stats['acf'][lag] - hist_stats['acf'][lag]) < 0.05 
                           for lag in [1, 5, 10]),
            'tail_dependence_match': abs(sim_stats['lower_tail_dep'] - 
                                        hist_stats['lower_tail_dep']) < 0.10,
            'leverage_effect_present': sim_stats['leverage_corr'] < -0.05
        }
```

### 6. Absorbing Boundaries Logic (First-Passage)
```python
def simulate_with_absorbing_boundaries(self, target_up, target_down):
    # Initialize
    active = np.ones(n_paths, dtype=bool)
    tau_up, tau_down = np.full(n_paths, np.inf), np.full(n_paths, np.inf)
    
    for t in range(horizon):
        if not np.any(active): break
        # Strict logic: Once active=False, path stops evolving relative to barrier logic
        # ... generate returns ...
        # Check barriers
        tau_up[(price >= target_up) & active] = t + 1
        active[price >= target_up] = False
        # ... same for down ...
    return tau_up, tau_down
```

### 7. Scientific Validation (Triple Mode Experiment)
```python
class TripleModeValidator:
    def validate_mode_convergence(self):
        # Central tendency check
        central_std = np.std([results[m]['central'] for m in 'ABC'])
        # Tail divergence check
        tail_std = np.std([results[m]['tails'] for m in 'ABC'])
        return {'central_convergence': central_std < 0.02, 'tail_divergence': tail_std > 0.05}
```

### 8. Extreme Value Theory Bridge (Tail Extrapolation)
```python
class EVTBridge:
    def augment_tails(self, historical_pairs, confidence=0.999):
        """
        Bridge the gap between historical extremes and Black Swans.
        Use Generalized Pareto Distribution (GPD) to extrapolate beyond history.
        """
        # Fit GPD to tail exceedances
        # Generate synthetic extreme joint moves
        # Blend with historical pairs
        pass
```

### 9. Bayesian Model Averaging (Uncertainty Quantification)
```python
class BayesianModelAverager:
    def average_modes(self, results, prior_weights):
        """
        Average modes (A/B/C) with uncertainty.
        weights ~ exp(-0.5 * delta_BIC) or posterior probabilities.
        """
        # Compute model evidence (BIC, WAIC)
        # Compute posterior model probabilities
        # Weighted average of predictions
        # Output dominance metrics (e.g., "Mode B is 90% probable")
        pass
```

### 10. Causal Validation Framework (Granger Causality)
```python
class CausalValidator:
    def test_granger_causality(self, regime_changes, market_shocks):
        """
        Test if market shocks Granger-cause regime changes.
        Validates 'endogenous feedback' assumption formally.
        """
        # Granger causality test
        # Vector autoregression with regime dummies
        # Impulse response functions
        pass
```
