# Implementation Guide: Regime Risk Platform v7.1

## Overview

This document describes the key fixes and improvements from v7.0 to v7.1 of the Regime Risk Platform, including the critical factor model bug fix and adaptive stock classification.

---

## 1. The Critical Bug Fix

### Problem: Non-Zero Mean Residuals

The v7.0 implementation used:
- **Median-based alpha** (robust to outliers)
- **Asymmetric beta** (max of up/down market betas)

This created a mathematical inconsistency:

```python
# v7.0 (BROKEN)
alpha = np.median(adjusted_returns)
beta = max(abs(beta_up), abs(beta_down))
residuals = returns - (alpha + beta * market_returns)
# residuals.mean() ≠ 0 !!!
```

**Evidence of the bug:**
```
Momentum regime:
  Alpha (ann): +103.6%
  Residual mean (ann): +93.8%  ← SHOULD BE 0!
  Missing drift: 113%
```

### Solution: Coherent OLS Factor Model

```python
# v7.1 (FIXED)
def compute_market_beta(self):
    """
    COHERENT factor model with guaranteed zero-mean residuals.
    """
    asset_ret = self.data['Log_Ret'].values
    market_ret = self.market_data['Log_Ret'].values
    
    # 1. Standard OLS beta (not asymmetric)
    cov_matrix = np.cov(asset_ret, market_ret)
    self.market_beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    
    # 2. Alpha from MEAN (not median)
    self.market_alpha = np.mean(asset_ret) - self.market_beta * np.mean(market_ret)
    
    # 3. Residuals: Zero mean BY CONSTRUCTION
    residuals = asset_ret - (self.market_alpha + self.market_beta * market_ret)
    
    # 4. Enforce zero mean (numerical precision)
    residuals = residuals - np.mean(residuals)
    
    # 5. Per-regime calculations with same approach
    for r in range(self.n_regimes):
        mask = self.data['Regime'].values == r
        if np.sum(mask) < 30:
            continue
        
        r_asset = asset_ret[mask]
        r_market = market_ret[mask]
        
        # OLS for regime
        cov_r = np.cov(r_asset, r_market)
        beta = cov_r[0, 1] / cov_r[1, 1]
        alpha = np.mean(r_asset) - beta * np.mean(r_market)
        
        # Residuals (zero mean enforced)
        regime_residuals = r_asset - (alpha + beta * r_market)
        regime_residuals = regime_residuals - np.mean(regime_residuals)
        
        # Store
        self.regime_beta[r] = beta
        self.regime_alpha[r] = alpha
        self.regime_residuals[r] = regime_residuals
```

### Validation

After the fix:
```
Momentum regime:
  Alpha (ann): +208.5%
  Residual mean (ann): 0.0000%  ✓
  Drift decomposition error: 0.00%  ✓
```

---

## 2. Simulation Update

### Factor vs Empirical Model Type

The simulation now checks model type per regime:

```python
def simulate(self):
    for d in range(self.days_ahead):
        for s in range(self.simulations):
            r = current_regimes[s]
            
            # Check model type
            model_type = self.regime_model_type.get(r, 'factor')
            
            if model_type == "empirical":
                # Pure empirical: sample from historical returns
                returns[s] = np.random.choice(self.regime_returns[r])
            else:
                # Factor model: alpha + beta*market + epsilon
                alpha = self.regime_alpha.get(r, self.market_alpha)
                beta = self.regime_beta.get(r, self.market_beta)
                market_ret = market_returns[s, d]
                epsilon = np.random.choice(self.regime_residuals[r])
                returns[s] = alpha + beta * market_ret + epsilon
```

### R² Check for Fallback

If factor model explains < 5% of variance, fall back to empirical:

```python
r_squared = 1 - np.var(regime_residuals) / np.var(r_asset)

if r_squared < 0.05:
    self.regime_model_type[r] = "empirical"
    self.regime_returns[r] = r_asset  # Store for sampling
else:
    self.regime_model_type[r] = "factor"
```

---

## 3. Stock-Type Classification (New)

### The Discovery

Kurtosis predicts regime persistence (r = +0.84):
- **Fat-tail stocks:** Rare extreme events anchor long regimes
- **Noise stocks:** Constant moderate moves cause regime churn

### Classification Function

```python
def classify_stock_type(returns):
    """Classify stock by information flow pattern."""
    from scipy import stats
    
    kurtosis = stats.kurtosis(returns)
    
    if kurtosis > 5.0:
        return "Fat-Tail"   # Long regimes expected
    elif kurtosis < 2.5:
        return "Noise"      # Short regimes expected
    else:
        return "Normal"     # Medium regimes
```

### Adaptive Parameters

```python
def get_regime_parameters(stock_type):
    """Return optimal regime detection parameters by stock type."""
    
    if stock_type == "Fat-Tail":
        return {
            'n_regimes': 2,
            'features': ['Drawdown', 'Extreme_Return_Flag'],
            'expected_duration': 60,
            'min_regime_days': 30
        }
    elif stock_type == "Noise":
        return {
            'n_regimes': 4,
            'features': ['Vol_20d', 'Vol_60d', 'Trend_Strength'],
            'expected_duration': 15,
            'min_regime_days': 10
        }
    else:  # Normal
        return {
            'n_regimes': 3,
            'features': ['Vol_20d', 'Vol_60d', 'Ret_20d', 'Drawdown'],
            'expected_duration': 25,
            'min_regime_days': 20
        }
```

---

## 4. Regime Naming by Sharpe

Regimes are now named by their economic characteristics:

```python
def _compute_regime_names_by_sharpe(self):
    """Assign meaningful names based on Sharpe ratio."""
    
    regime_stats = []
    for r in range(self.n_regimes):
        mask = self.data['Regime'] == r
        returns = self.data.loc[mask, 'Log_Ret'].values
        
        ann_return = np.mean(returns) * 252
        ann_vol = np.std(returns) * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        regime_stats.append({
            'regime': r,
            'sharpe': sharpe,
            'return': ann_return,
            'vol': ann_vol
        })
    
    # Sort by Sharpe and assign names
    sorted_regimes = sorted(regime_stats, key=lambda x: x['sharpe'])
    
    for i, stat in enumerate(sorted_regimes):
        r = stat['regime']
        if stat['sharpe'] > 1.5:
            self.regime_names[r] = "Momentum"
        elif stat['sharpe'] > 0.5:
            self.regime_names[r] = "Bull"
        elif stat['sharpe'] < -0.5:
            self.regime_names[r] = "Bear"
        else:
            self.regime_names[r] = "Neutral"
```

---

## 5. Validation Checklist

### Mathematical Coherence
- [ ] Residual mean ≈ 0 for all regimes
- [ ] Drift decomposition error < 0.1%
- [ ] Alpha + Beta × Market = Actual drift

### Behavioral Correctness
- [ ] What-if Momentum > Bear probability
- [ ] Higher alpha regime → Higher probability
- [ ] Regime persistence matches historical durations

### Cross-Stock Validation
- [ ] Model runs on multiple tickers without error
- [ ] Stock-type classification matches economic intuition
- [ ] Regime durations vary appropriately by stock type

---

## 6. Test Scripts

| Script | Purpose |
|--------|---------|
| `test_coherent_fix.py` | Validates zero-mean residuals |
| `deep_diagnostic.py` | Analyzes probability damping |
| `compare_regime_persistence.py` | Cross-stock comparison |
| `test_volatility_persistence.py` | Tests vol→duration hypothesis |
| `test_news_frequency.py` | Tests kurtosis→duration hypothesis |

### Running Validation

```bash
# Full validation package
python test_coherent_fix.py
python test_news_frequency.py

# View results
cat test_results.txt
cat news_frequency_results.txt
```

---

## 7. Future Improvements

### Planned for v7.2
1. **Stock-type detection:** Automatic classification on `ingest_data()`
2. **Adaptive n_regimes:** Use kurtosis to set optimal regime count
3. **Anchor event detection:** Flag extreme events that define regime boundaries

### Research Extensions
1. Multi-asset regime synchronization
2. Predictive modeling of regime transitions
3. Real-time regime probability updates
