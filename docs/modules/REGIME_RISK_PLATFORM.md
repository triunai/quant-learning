# Regime Risk Platform v7.1 Documentation

> **Location:** `battle-tested/PLTR-test-2.py`  
> **Purpose:** Institutional-grade Monte Carlo risk platform with regime conditioning, factor models, and walk-forward validation.
> **Updated:** 2025-12-30 (v7.1 - Coherent Factor Model Fix + Kurtosis Research)

---

## What's New in v7.1

### 🔧 Bug Fixes
1. **Coherent Factor Model** - Fixed non-zero mean residuals caused by median alpha + asymmetric beta
2. **Zero-Mean Residuals** - Now enforced by construction using OLS

### 🔬 Research Discoveries
1. **Kurtosis-Persistence Relationship** - Fat-tail stocks have LONGER regimes (r = +0.84)
2. **Stock Classification** - Stocks can be classified as Fat-Tail vs Noise for optimal regime detection
3. **Anchor Event Hypothesis** - Extreme events create regime-defining boundaries

---

## Overview

The `RegimeRiskPlatform` is a comprehensive quant simulation engine that combines:

- **GMM Regime Clustering** on slow features (not return buckets)
- **Coherent Alpha/Beta Factor Model** with per-regime parameters and zero-mean residuals
- **Semi-Markov Duration Modeling** for regime persistence
- **Empirical Residual Sampling** (preserves fat tails)
- **Full Risk Dashboard** (VaR, CVaR, Kelly, Max Drawdown)
- **Walk-Forward Validation** (no future leakage)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 1: DATA INGESTION                        │
│  ├─ Asset data (5Y)                                             │
│  ├─ Market data (QQQ) for beta calculation                      │
│  └─ Slow Features: Vol_20d, Vol_60d, Ret_20d, Drawdown          │
├─────────────────────────────────────────────────────────────────┤
│                   PHASE 2: REGIME MODEL                          │
│  ├─ GMM Clustering (n_regimes=3 default)                        │
│  ├─ Regime naming by Sharpe: Momentum → Bull → Neutral → Bear   │
│  ├─ Per-regime μ, σ (return distributions)                      │
│  ├─ Semi-Markov duration fitting                                │
│  └─ Transition matrix (with Laplace smoothing)                  │
├─────────────────────────────────────────────────────────────────┤
│                   PHASE 3: COHERENT FACTOR MODEL                 │
│  ├─ r = α_regime + β_regime × r_market + ε                      │
│  ├─ OLS Beta (not asymmetric - usable in simulation)            │
│  ├─ Mean-based Alpha (guarantees zero-mean residuals)           │
│  ├─ Per-regime α and β                                          │
│  ├─ R² check: Fallback to empirical if factor model weak        │
│  └─ Empirical residual pools per regime (fat tails preserved)   │
├─────────────────────────────────────────────────────────────────┤
│                   PHASE 4: SIMULATION                            │
│  ├─ Check model type per regime (factor vs empirical)           │
│  ├─ Factor: α_regime + β_regime × r_market + ε_empirical        │
│  ├─ Empirical: Sample directly from historical regime returns   │
│  ├─ Semi-Markov duration: expire → off-diagonal transition      │
│  └─ Market returns sampled from historical (fat tails)          │
├─────────────────────────────────────────────────────────────────┤
│                   PHASE 5: RISK DASHBOARD                        │
│  ├─ VaR(95), CVaR(95) on simple returns                         │
│  ├─ P(MaxDD > 20%), P(MaxDD > 30%)                              │
│  ├─ Stop-loss breach probability                                │
│  └─ Fractional Kelly with exponential DD penalty                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Innovations

### 1. Real Regimes from Slow Features

**Problem:** Many quant models bucket returns directly (e.g., top/bottom decile), creating look-ahead bias and meaningless "regimes."

**Solution:** Cluster on **slow features** that are observable in real-time:

```python
features = ['Vol_20d', 'Vol_60d', 'Ret_20d', 'Drawdown']
self.gmm = GaussianMixture(n_components=self.n_regimes, covariance_type='full')
self.gmm.fit(X_scaled)
```

| Feature | Why Slow? | What It Captures |
|---------|-----------|------------------|
| Vol_20d | 20-day rolling | Current fear level |
| Vol_60d | 60-day rolling | Longer-term trend |
| Ret_20d | 20-day return | Momentum |
| Drawdown | Peak-to-trough | Pain level |

### 2. Coherent Alpha/Beta Factor Model (v7.1 Fix)

**Problem (v7.0):** Median alpha + asymmetric beta created non-zero mean residuals, causing under-prediction of high-alpha regimes.

**Solution (v7.1):** Use standard OLS with mean-based alpha:

```python
def compute_market_beta(self):
    """Coherent factor model with guaranteed zero-mean residuals."""
    
    # 1. Standard OLS beta (not asymmetric)
    cov_matrix = np.cov(asset_ret, market_ret)
    self.market_beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    
    # 2. Alpha from MEAN (guarantees zero-mean residuals)
    self.market_alpha = np.mean(asset_ret) - self.market_beta * np.mean(market_ret)
    
    # 3. Per-regime calculations
    for r in range(self.n_regimes):
        mask = self.data['Regime'].values == r
        r_asset = asset_ret[mask]
        r_market = market_ret[mask]
        
        # OLS for regime
        beta = cov(r_asset, r_market)[0,1] / var(r_market)
        alpha = mean(r_asset) - beta * mean(r_market)
        
        # Residuals (zero mean BY CONSTRUCTION)
        residuals = r_asset - (alpha + beta * r_market)
        residuals = residuals - mean(residuals)  # Enforce
        
        # R² check: If factor model explains < 5%, use empirical
        r_squared = 1 - var(residuals) / var(r_asset)
        if r_squared < 0.05:
            self.regime_model_type[r] = "empirical"
        else:
            self.regime_model_type[r] = "factor"
```

**Validation:**
```
Regime       Beta    Alpha      Resid Mean  Status
--------------------------------------------------------
Bear         1.59    -56.0%     +0.0000%    [OK]
Crisis       1.92     -3.9%     -0.0000%    [OK]
Momentum     1.80   +208.5%     +0.0000%    [OK]
```

### 3. Empirical Residual Sampling

**Problem:** Normal distribution assumptions destroy fat tails → underestimate tail risk.

**Solution:** Sample residuals **empirically** from historical data:

```python
# In simulation loop
if model_type == "empirical":
    returns[s] = np.random.choice(self.regime_returns[r])
else:
    residual_pool = self.regime_residuals.get(r)
    epsilon = np.random.choice(residual_pool)  # Fat tails preserved!
    returns[s] = alpha + beta * market_ret + epsilon
```

### 4. Semi-Markov with Length-Biased Sampling

**Problem:** Standard Markov chains can't model regime persistence correctly without artificially inflating diagonal probabilities.

**Solution:** Separate duration from transitions:

```python
# Sample initial duration using length-biased distribution
weights = samples_array / samples_array.sum()
full_duration = np.random.choice(samples_array, p=weights)
remaining_duration[s] = np.random.randint(1, max(2, full_duration + 1))

# When duration expires, use OFF-DIAGONAL transitions only
probs = self.transition_matrix[r].copy()
probs[r] = 0  # Zero out self-transition
probs = probs / probs.sum()
new_r = np.random.choice(self.n_regimes, p=probs)
```

### 5. Regime Naming by Sharpe Ratio

**New in v7.1:** Regimes are named by economic characteristics, not arbitrary labels:

```python
def _compute_regime_names_by_sharpe(self):
    """Assign meaningful names based on Sharpe ratio."""
    # Sort regimes by Sharpe
    # Sharpe > 1.5 → "Momentum"
    # Sharpe > 0.5 → "Bull"  
    # Sharpe < -0.5 → "Bear"
    # Else → "Neutral"
```

---

## Research Discovery: Kurtosis-Persistence Relationship

### The Finding

Cross-sectional analysis across 22 stocks revealed:

| Metric | Correlation with Regime Duration |
|--------|----------------------------------|
| **Kurtosis** | **+0.837** |
| Volatility | +0.078 |
| Jump Frequency | -0.054 |

**Interpretation:** Stocks with fat-tail distributions have LONGER regime durations, not shorter.

### Evidence

| Stock | Kurtosis | Avg Duration |
|-------|----------|--------------|
| META | 26.6 | 119.5 days |
| JPM | 5.0 | 59.8 days |
| COIN | 2.6 | 18.1 days |
| TSLA | 2.5 | 15.7 days |

### The "Anchor Event" Hypothesis

Fat-tail events don't destabilize regimes—they **CREATE** them:
- Rare, extreme moves serve as regime boundary markers
- GMM detects clear separation before/after extreme events
- Without another extreme event, regime persists

Noise stocks (constant moderate moves) have regime churn because no single event is significant enough to anchor a regime.

### Stock Classification

```python
def classify_stock_type(returns):
    kurtosis = stats.kurtosis(returns)
    
    if kurtosis > 5.0:
        return "Fat-Tail"   # Fewer regimes, event-driven
    elif kurtosis < 2.5:
        return "Noise"      # More regimes, pattern-driven
    else:
        return "Normal"
```

---

## Risk Dashboard

### VaR/CVaR

```python
# Use SIMPLE returns for interpretable metrics
simple_returns = np.exp(log_returns) - 1
var_95 = np.percentile(simple_returns, 5)
cvar_95 = np.mean(simple_returns[simple_returns <= var_95])
```

### Kelly Fraction with DD Penalty

```python
# Continuous Kelly: f* ~ μ/σ²
raw_kelly = mu / (sigma ** 2)

# Exponential DD penalty
dd_penalty = np.exp(-3 * prob_dd_30)

# Final: 0.5× fractional Kelly, capped at 25%
kelly = max(0, min(0.25, raw_kelly * 0.5 * dd_penalty))
```

---

## Walk-Forward Validation

**Critical:** Never train on future data.

```python
def walk_forward_validation(self, n_folds=5):
    # For each fold:
    # 1. Train on past data only
    # 2. Simulate forward
    # 3. Compare to actual outcome
    # 4. Compute Brier score
```

---

## Usage Example

```python
from battle_tested.PLTR_test_2 import RegimeRiskPlatform

# Initialize
platform = RegimeRiskPlatform(
    ticker="PLTR",
    market_ticker="QQQ",
    days_ahead=126,
    simulations=5000,
    n_regimes=3
)

# Data ingestion
platform.ingest_data()

# Build regime model
platform.build_regime_model()

# Macro conditioning (coherent factor model)
platform.compute_market_beta()

# VIX and anomaly detection
platform.check_macro_context()

# GARCH volatility
platform.run_garch()

# Simulation
paths = platform.simulate()

# Invariant check
platform.verify_simulation_invariants(paths)

# Risk metrics
risk = platform.compute_risk_metrics(paths)
print(f"VaR(95): {risk['var_95']*100:.1f}%")
print(f"Kelly: {risk['kelly_fraction']:.0%}")

# Validation
platform.walk_forward_validation(n_folds=5)
```

---

## Validation Package

Run the one-click validation:

```bash
python validation_package.py --ticker PLTR --market QQQ
```

This validates:
1. Zero-mean residuals for all regimes
2. What-if test: Momentum > Bear probability
3. Kurtosis-persistence relationship
4. Stock type classification

---

## Known Limitations

1. **Jump Diffusion Disabled** – Empirical sampling already captures fat tails
2. **Single-Asset** – No portfolio-level correlation/hedging
3. **No Options Overlay** – Pure delta-one simulation
4. **Stock-Type Adaptation** – Manual; future version will auto-classify

---

## Future Roadmap (v7.2+)

1. **Automatic Stock Classification** – Use kurtosis on `ingest_data()`
2. **Adaptive n_regimes** – Fewer for fat-tail, more for noise stocks
3. **Anchor Event Detection** – Flag extreme events that define regime boundaries
4. **Multi-Asset Extension** – Correlated regime simulation
5. **Streamlit Dashboard** – `to_refine/dashboard.py` integration

---

## References

- Session notes: `docs/sessions/2025-12-30-mode-a-fundamentals.md`
- Research paper: `research/papers/kurtosis_regime_persistence/draft.md`
- Implementation guide: `docs/implementation_guide_v71.md`
