# Mode B: Stationary Bootstrap Simulator

> **Role:** Ground Truth Benchmark ("Truth Serum")  
> **Status:** ✅ Production Ready  
> **File:** `to_refine/stationary_bootstrap.py`  
> **Last Validated:** 2025-12-30

---

## Overview

Mode B implements the **Stationary Bootstrap** method (Politis & Romano, 1994) for Monte Carlo simulation of financial time series. Unlike parametric models that impose distributional assumptions, Mode B **samples directly from historical data**, preserving the full complexity of real market behavior.

### Why "Ground Truth"?

Mode B is the benchmark against which all other simulation modes (A, C) are validated:

| Property | Mode B Advantage |
|----------|-----------------|
| **No model assumptions** | Uses actual historical returns, not fitted distributions |
| **Preserves fat tails** | Real crisis events sampled directly |
| **Captures dependencies** | Coupled pair sampling maintains crisis correlations |
| **Block structure** | Random block lengths preserve volatility clustering |

**Rule:** If Mode A disagrees with Mode B, Mode A is wrong until proven otherwise.

---

## Key Features

### 1. Coupled Pair Sampling

Unlike naive bootstrap that samples returns i.i.d., Mode B samples `(market_return, residual)` pairs together:

```
r_asset = α + β × r_market + ε
```

**Why this matters:** During crises, asset residuals and market returns are jointly extreme. Sampling them together preserves:
- Panic coupling (when market crashes, residuals spike)
- Leverage effect (negative returns → higher vol)  
- Joint tail dependence

### 2. Geometric Block Lengths

Following Politis & Romano (1994), block lengths are drawn from a geometric distribution:

```
P(block_length = k) = (1-p) × p^(k-1)
```

This creates random restart points, avoiding periodic artifacts while preserving autocorrelation structure.

**Mean block length:** 10 days (configurable)

### 3. Ridge-Regularized Beta

Beta estimation uses Ridge regression instead of OLS:

```python
XTX += ridge_lambda * I  # Regularization
beta = solve(XTX, X.T @ y)
```

**Why:** Small samples or collinear data can cause unstable OLS estimates. Ridge regression (`λ = 0.01`) provides stable beta estimates.

### 4. Numba JIT Compilation

The inner simulation loop is JIT-compiled with Numba for **10x speedup**:

```python
@njit(cache=False)
def _simulate_paths_numba(...):
    # Hot path: 50k paths × 126 days = 6.3M iterations
```

**Note:** `cache=False` required when module is in subdirectory (Numba cache import limitation).

### 5. Data Caching

The `DataCache` class avoids repeated yfinance API calls:

```python
DataCache.get(key)      # Return cached data if available
DataCache.set(key, df)  # Store for future use
DataCache.clear()       # Clear all cached data
```

---

## Gatekeeper Metrics

**Gatekeepers** are metrics that **must pass** for a simulation to be considered valid. Mode B uses:

### ACF(r²) at Lags 1, 5, 10

Measures volatility clustering persistence. Simulated ACF must match historical within **±0.10**.

```
ACF(r², lag) = Cov(r²_t, r²_{t-lag}) / Var(r²)
```

| Lag | Threshold | Typical Value |
|-----|-----------|---------------|
| 1 | ±0.10 | 0.15-0.25 |
| 5 | ±0.10 | 0.10-0.20 |
| 10 | ±0.10 | 0.05-0.15 |

### Lower Tail Dependence (λ_L)

Measures joint crash behavior:

```
λ_L = P(Asset < q5 | Market < q5)
```

Threshold: **±0.15**

**Interpretation:** When the market is in its worst 5% of returns, what fraction of asset returns are also in their worst 5%?

---

## Diagnostic Metrics (Non-Gatekeeper)

These provide additional validation but don't gate pass/fail:

| Metric | Description | Threshold |
|--------|-------------|-----------|
| Leverage Effect | Corr(r_t, r²_{t+1}) | ±0.20 |
| Excess Kurtosis | Fat tail measure | ±3.0 |
| Skewness | Asymmetry | ±0.50 |
| Daily Mean | Average return | ±0.001 |
| Daily Std | Volatility | ±20% relative |

---

## Validation Results

### Frozen Evaluation Suite (2025-12-30)

**Configuration:** `mode='fast'` (5,000 simulations), `seed=42`

| Asset | Horizon | Beta | Vol | ACF Δ | λ_L Δ | Status |
|-------|---------|------|-----|-------|-------|--------|
| SPY | 20d | 0.69 | 17.1% | <0.06 | 0.008 | ✅ PASS |
| SPY | 126d | 0.69 | 17.1% | <0.05 | 0.001 | ✅ PASS |
| QQQ | 20d | 1.17 | 22.7% | <0.08 | 0.008 | ✅ PASS |
| QQQ | 126d | 1.17 | 22.7% | <0.07 | 0.007 | ✅ PASS |
| PLTR | 20d | 1.89 | 66.9% | <0.02 | 0.003 | ✅ PASS |
| PLTR | 126d | 1.89 | 66.9% | <0.02 | 0.004 | ✅ PASS |

**Result:** 6/6 tests passed (100%), exceeding the ≥3/4 requirement.

**Key Finding:** Mode B generalizes across:
- Low-beta ETFs (SPY, β=0.69)
- High-beta tech ETFs (QQQ, β=1.17)
- High-volatility individual stocks (PLTR, β=1.89, vol=66.9%)

---

## Usage

### Basic Usage

```python
from to_refine.stationary_bootstrap import StationaryBootstrap, BootstrapConfig

# Configure
config = BootstrapConfig(
    mode="fast",        # debug (1k), fast (5k), production (50k)
    seed=42,            # Reproducibility
    use_numba=True,     # Enable JIT (10x speedup)
    show_progress=True, # tqdm bars in production mode
    ridge_lambda=0.01,  # Beta regularization
)

# Initialize
engine = StationaryBootstrap(
    ticker="AAPL",
    market_ticker="SPY",
    days_ahead=126,     # 6-month horizon
    config=config,
)

# Run pipeline
engine.ingest_data()
paths = engine.simulate()
diagnostics = engine.compute_diagnostics(paths)

# Check gatekeepers
for d in diagnostics:
    if d.is_gatekeeper:
        print(f"{d.metric}: {'✅' if d.passed else '❌'}")
```

### Full Pipeline with All Metrics

```python
results = engine.run(verbose=True)

# Results contains:
# - paths: (n_sims, n_days) price matrix
# - diagnostics: List[DiagnosticResult]
# - diagnostics_with_ci: Dict with bootstrap CIs
# - path_stats: Path-level validation
# - fpt_analysis: First-passage time analysis
# - risk_metrics: VaR, CVaR, drawdowns
```

### Run Frozen Evaluation Suite

```python
from to_refine.stationary_bootstrap import run_frozen_evaluation_suite, BootstrapConfig

results = run_frozen_evaluation_suite(
    config=BootstrapConfig(mode="fast", seed=42),
    verbose=True
)

print(f"Suite Passed: {results['suite_passed']}")
print(f"Score: {results['passed_count']}/{results['total_count']}")
```

---

## API Reference

### Configuration

```python
@dataclass
class BootstrapConfig:
    mode: str = "fast"           # debug (1k), fast (5k), production (50k)
    seed: Optional[int] = None   # Random seed
    use_numba: bool = True       # Enable JIT compilation
    show_progress: bool = True   # tqdm progress bars
    ridge_lambda: float = 0.01   # Ridge regularization for beta
```

### Key Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `ingest_data(period="5y")` | Fetch and prepare data | None |
| `simulate()` | Run bootstrap simulation | np.ndarray (paths) |
| `compute_diagnostics(paths)` | Gatekeeper + diagnostic metrics | List[DiagnosticResult] |
| `compute_diagnostics_with_ci(paths)` | Metrics with bootstrap CIs | Dict |
| `validate_path_statistics(paths)` | Whole-path validation | Dict |
| `analyze_first_passage(paths)` | Hit probabilities and times | Dict |
| `compute_risk_metrics(paths)` | VaR, CVaR, drawdowns | Dict |
| `calibrate_targets_from_path_extrema()` | Auto-calibrate targets | Dict |
| `run(verbose=True)` | Full pipeline | Dict |

### Custom Exceptions

```python
BootstrapError      # Base exception
DataFetchError      # yfinance failures, empty data
SimulationError     # Invalid simulation results
ConfigurationError  # Invalid config parameters
DiagnosticError     # Diagnostic computation failures
```

---

## Academic Reference

> Politis, D. N., & Romano, J. P. (1994). **The Stationary Bootstrap**. *Journal of the American Statistical Association*, 89(428), 1303-1313.

**Key insight:** Unlike the block bootstrap with fixed block lengths, the stationary bootstrap uses random (geometric) block lengths, making the resampled series stationary while preserving autocorrelation structure.

---

## Backlog References

- **P0.5.3:** Frozen evaluation suite (COMPLETE)
- **P0-B:** Path extrema calibration (COMPLETE)
- **P1.1:** Confidence intervals with block bootstrap (COMPLETE)
- **P1.2:** Bonferroni correction for multiple testing (COMPLETE)

---

## Related Documentation

- [ARCHITECTURE.md](../architecture/ARCHITECTURE.md) - Overall platform architecture
- [phase-1-consolidate.md](../immediate-tasks/phases/phase-1-consolidate.md) - Current phase progress
- [BACKLOG.md](../backlog/BACKLOG.md) - Full backlog with priorities

---

*Last updated: 2025-12-30*
