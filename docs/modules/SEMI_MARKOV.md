# Semi-Markov Model Documentation

> **Location:** `refinery/semi_markov.py`  
> **Purpose:** Duration-aware regime simulation that preserves autocorrelation and volatility clustering.

---

## Overview

The `SemiMarkovModel` class implements a **Semi-Markov Chain** for financial regime modeling. Unlike standard Markov chains (which assume exponential/memoryless durations), this model:

1. **Clusters market states** using KMeans on slow features (volatility, momentum, drawdown).
2. **Fits duration distributions** per regime (Gamma for high-vol states, Exponential for low-vol).
3. **Samples residual life** correctly when entering mid-regime.
4. **Preserves autocorrelation** by sampling contiguous historical runs.

---

## Key Features

### 1. Risk-Sorted Regime Clustering

```python
# Features used for clustering
features = ['Vol_20d', 'Ret_60d', 'DD']  # Volatility, Momentum, Drawdown
```

States are sorted by a **composite risk score**:
```
Risk Score = Vol × 2 − Return − Drawdown × 1.5
```

This ensures:
- **State 0 = Crash** (highest risk)
- **State 4 = Rally** (lowest risk)

### 2. Duration Modeling

Each regime has a fitted duration distribution:

| Regime | Distribution | Why |
|--------|--------------|-----|
| Crash/Bear | Gamma | Fat-tailed, captures volatility clustering |
| Bull/Rally | Exponential | Memoryless, mean-reverting |

```python
def sample_duration(self, state):
    """Samples a fresh duration for the given state."""
    params = self.duration_params[state]
    if params['dist'] == 'gamma':
        return gamma.rvs(a, loc, scale)
    else:
        return expon.rvs(loc, scale)
```

### 3. Residual Life Sampling

When entering mid-regime (not at the start), we sample the **forward recurrence time**:

```python
def sample_residual_duration(self, state, elapsed_days):
    """Uses Inverse Transform Sampling for conditional duration."""
    cdf_t = gamma.cdf(elapsed_days, a, loc, scale)
    u = np.random.uniform(cdf_t, 1.0)
    total_duration = gamma.ppf(u, a, loc, scale)
    return max(1, int(total_duration - elapsed_days))
```

**Why this matters:** Without proper residual sampling, you'd either:
- Underestimate regime persistence (exiting too early)
- Overestimate duration (ignoring that you're already mid-regime)

### 4. Autocorrelation-Preserving Return Sampling

Instead of sampling IID returns per day, we sample **contiguous historical runs**:

```python
def sample_regime_returns(self, state, n_days):
    """Samples contiguous historical runs of this state."""
    # Group filtered data by 'block' (contiguous runs)
    runs = [group['Log_Ret'].values for _, group in state_data.groupby('block')]
    
    # Pick a run that is at least n_days long
    valid_runs = [r for r in runs if len(r) >= n_days]
    if valid_runs:
        chosen_run = valid_runs[np.random.choice(len(valid_runs))]
        return chosen_run[start_idx : start_idx + n_days]
```

**Benefits:**
- Preserves volatility clustering (ARCH effects)
- Maintains realistic return autocorrelation
- Avoids over-smoothing from IID sampling

---

## Position Sizing with Regime Fatigue

The `get_position_size()` method adjusts exposure based on how "tired" a regime is:

```python
def get_position_size(self, current_state, days_in_state, base_size=1.0):
    fatigue = self.regime_fatigue_score(current_state, days_in_state)
    
    if current_state in [0, 1]:  # Crash/Bear
        multiplier = 0.3 + (1 - fatigue) * 0.3  # 0.3 to 0.6
    else:  # Bull/Rally
        multiplier = 0.5 + (1 - fatigue) * 0.5  # 0.5 to 1.0
    
    return base_size * multiplier
```

| Scenario | Fatigue | Position Size |
|----------|---------|---------------|
| Early in Crash | Low | 60% (still dangerous) |
| Late in Crash | High | 30% (might turn worse) |
| Early in Rally | Low | 100% (ride the wave) |
| Late in Rally | High | 50% (prepare for reversal) |

---

## Validation

The `validate_model()` method checks:

1. **Duration KS Test** – Do simulated regime durations match empirical?
2. **Volatility ACF** – Is vol clustering preserved?

```python
report = model.validate_model(paths)
# Example output:
# {
#   'real_vol_acf_lag1': 0.45,
#   'sim_vol_acf_lag1': 0.42,
#   'vol_clustering_error': 0.03
# }
```

---

## Usage Example

```python
from refinery.semi_markov import SemiMarkovModel

# Initialize
model = SemiMarkovModel("PLTR", n_states=5)

# Process data and fit distributions
model._process_data(period="5y")
model.fit_distributions()

# Run simulation
paths = model.run_simulation(days=126, simulations=1000)

# Get position sizing
current_state = 2  # Bull
days_in_state = 15
size = model.get_position_size(current_state, days_in_state)
print(f"Position size: {size:.2f}")

# Validate
report = model.validate_model(paths)
print(f"Vol clustering error: {report['vol_clustering_error']:.2f}")
```

---

## Why Semi-Markov Over Standard Markov?

| Feature | Standard Markov | Semi-Markov |
|---------|-----------------|-------------|
| Duration Distribution | Exponential (memoryless) | Any (Gamma, Weibull, etc.) |
| Persistence Modeling | Sticky diagonal only | Explicit duration fitting |
| Volatility Clustering | ❌ Lost | ✅ Preserved |
| Forward Recurrence | ❌ Ignored | ✅ Properly sampled |
| Return Autocorrelation | ❌ IID | ✅ Contiguous runs |

---

## Research Discovery: Kurtosis Predicts Duration

**Finding (December 2024):** Cross-sectional analysis revealed that **return kurtosis predicts regime duration** with r = +0.84.

| Stock | Kurtosis | Avg Duration |
|-------|----------|--------------|
| META | 26.6 | 119.5 days |
| JPM | 5.0 | 59.8 days |
| COIN | 2.6 | 18.1 days |

**Interpretation:** Fat-tail events create "anchor points" that define regime boundaries. Stocks with rare but extreme moves have longer-lasting regimes. Stocks with constant moderate noise have regime churn.

### Implications for Semi-Markov

1. **Duration priors:** Use kurtosis to set initial duration distribution parameters
2. **Stock classification:** Fat-tail vs Noise stocks need different n_states
3. **Expected persistence:** High-kurtosis stocks should have longer expected durations

See: `research/papers/kurtosis_regime_persistence/draft.md`

---

## Future Improvements

1. **Return state history from simulation** – Enable full duration validation.
2. **Per-regime beta conditioning** – Factor model integration.
3. **Online learning** – Update duration params as new data arrives.
4. **Jump diffusion overlay** – Separate tail events from normal residuals.
