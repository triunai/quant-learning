# Regime Risk Platform v7.0 Documentation

> **Location:** `battle-tested/PLTR-test-2.py`  
> **Purpose:** Institutional-grade Monte Carlo risk platform with regime conditioning, factor models, and walk-forward validation.

---

## Overview

The `RegimeRiskPlatform` is a comprehensive quant simulation engine that combines:

- **GMM Regime Clustering** on slow features (not return buckets)
- **Alpha/Beta Factor Model** with per-regime alphas
- **Semi-Markov Duration Modeling** for regime persistence
- **Jump Diffusion** with VIX-conditioned probabilities
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
│  ├─ Regime naming by volatility: Low Vol → Normal → Crisis      │
│  ├─ Per-regime μ, σ (return distributions)                      │
│  ├─ Semi-Markov duration fitting                                │
│  └─ Transition matrix (with Laplace smoothing)                  │
├─────────────────────────────────────────────────────────────────┤
│                   PHASE 3: MACRO CONDITIONING                    │
│  ├─ r = α_regime + β × r_market + ε                             │
│  ├─ Per-regime alpha (critical for upside probability!)         │
│  ├─ Idiosyncratic volatility                                    │
│  └─ Empirical residual pools per regime                         │
├─────────────────────────────────────────────────────────────────┤
│                   PHASE 4: SIMULATION                            │
│  ├─ Factor model returns: α_regime + β × r_market + ε_empirical │
│  ├─ Semi-Markov duration: expire → off-diagonal transition      │
│  ├─ Length-biased initial duration sampling                     │
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

### 2. Alpha/Beta Factor Model

**Problem:** Pure regime-conditioned returns collapse upside probability because they ignore the asset's systematic outperformance.

**Solution:** Decompose returns into:

```
r_asset = α_regime + β × r_market + ε
```

```python
# OLS decomposition
self.market_alpha = np.mean(asset_ret) - self.market_beta * np.mean(market_ret)

# Per-regime alpha (even better!)
for r in range(self.n_regimes):
    mask = self.data['Regime'].values == r
    r_asset = asset_ret[mask]
    r_market = market_ret[mask]
    self.regime_alpha[r] = np.mean(r_asset) - self.market_beta * np.mean(r_market)
```

**Result:** PLTR's positive alpha is now captured, giving realistic upside probabilities.

### 3. Empirical Residual Sampling

**Problem:** Normal distribution assumptions destroy fat tails → underestimate tail risk.

**Solution:** Sample residuals **empirically** from historical data:

```python
# In simulation loop
residual_pool = self.regime_residuals.get(r, np.array([0]))
epsilon = np.random.choice(residual_pool)  # Fat tails preserved!

# Full factor model
returns[s] = alpha + self.market_beta * market_ret + epsilon
```

### 4. Semi-Markov with Length-Biased Sampling

**Problem:** Standard Markov chains can't model regime persistence correctly without artificially inflating diagonal probabilities.

**Solution:** Separate duration from transitions:

```python
# Sample initial duration using length-biased distribution
# (longer runs are more likely to be "observed in progress")
weights = samples_array / samples_array.sum()
full_duration = np.random.choice(samples_array, p=weights)
remaining_duration[s] = np.random.randint(1, max(2, full_duration + 1))

# When duration expires, use OFF-DIAGONAL transitions only
probs = self.transition_matrix[r].copy()
probs[r] = 0  # Zero out self-transition
probs = probs / probs.sum()
new_r = np.random.choice(self.n_regimes, p=probs)
```

### 5. Invariant Verification

**Problem:** Simulators can drift without anyone noticing.

**Solution:** Compare sim stats to historical after every run:

```python
def verify_simulation_invariants(self, paths):
    sim_mean = np.mean(sim_returns)
    hist_mean = np.mean(hist_returns)
    
    mean_ok = abs(sim_mean - hist_mean) < 0.001
    std_ok = abs(sim_std - hist_std) / hist_std < 0.3
    skew_ok = abs(sim_skew - hist_skew) < 1.0
    kurt_ok = abs(sim_kurt - hist_kurt) < 3.0
    
    # Print [OK] or [FAIL] for each metric
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

**Problem:** Binary Kelly doesn't apply to continuous returns, and ignores drawdown risk.

**Solution:** Continuous Kelly with exponential DD penalty:

```python
# Continuous Kelly: f* ~ μ/σ²
raw_kelly = mu / (sigma ** 2)

# Exponential DD penalty (smoother than linear)
dd_penalty = np.exp(-3 * prob_dd_30)

# Final: 0.5× fractional Kelly, capped at 25%
kelly = max(0, min(0.25, raw_kelly * 0.5 * dd_penalty))
```

| prob_dd_30 | DD Penalty | Effect |
|------------|------------|--------|
| 0% | 1.0 | Full Kelly |
| 20% | 0.55 | Moderate reduction |
| 40% | 0.30 | Aggressive reduction |

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

### Multi-Threshold Calibration

Test multiple targets and horizons:

| Horizon | Threshold | Up Hit | Down Hit |
|---------|-----------|--------|----------|
| 21d (1mo) | +10% | 15% | 12% |
| 63d (3mo) | +20% | 25% | 18% |
| 126d (6mo) | +30% | 35% | 22% |

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

# Macro conditioning (alpha + beta)
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

# Historical validation
platform.compute_historical_validation()

# Full calibration suite
platform.multi_threshold_calibration()
platform.walk_forward_validation(n_folds=5)
```

---

## Output: Risk Report

```json
{
  "ticker": "PLTR",
  "price": 78.50,
  "regime": "Normal",
  "beta": 1.85,
  "signal": "LONG",
  "confidence": 72,
  "targets": {
    "up": {"price": 117.75, "prob_sim": 0.42, "prob_hist": 0.38},
    "down": {"price": 51.03, "prob_sim": 0.18, "prob_hist": 0.15}
  },
  "risk": {
    "var_95": -0.28,
    "cvar_95": -0.41,
    "prob_dd_20": 0.35,
    "prob_dd_30": 0.18,
    "kelly_fraction": 0.15,
    "win_rate": 0.58
  }
}
```

---

## Known Limitations

1. **Jump Diffusion Disabled** – Empirical sampling already captures fat tails; adding jumps would double-count.
2. **Single-Asset** – No portfolio-level correlation/hedging.
3. **No Options Overlay** – Pure delta-one simulation.
4. **VIX Fetch Can Fail** – Defaults to 20 if unavailable.

---

## Future Roadmap

1. **Streamlit Dashboard Integration** – `to_refine/dashboard.py` wraps this engine.
2. **Multi-Asset Extension** – Correlated regime simulation.
3. **Options Pricing** – Monte Carlo for exotic payoffs.
4. **Live Trading Hooks** – Paper trading via broker API.
