# Phase 4: Implement Sector-Based Regime Parameters

**Estimated Time:** 3-4 hours  
**Priority:** HIGH - Actionable findings from Phase 3  
**Status:** 🔜 READY TO START  
**Depends On:** Phase 3 (COMPLETED)

---

## Context

Phase 3 validation revealed that the kurtosis-duration hypothesis was **FRAGILE**, but we discovered something more valuable: **sector-specific regime behavior patterns**.

### The Pivot

| Before (Fragile) | After (Validated) |
|------------------|-------------------|
| "Kurtosis predicts duration" | "Sector determines regime behavior" |
| Universal relationship | Sector-specific patterns |
| Build on kurtosis | Build on sector fingerprints |

---

## Validated Findings to Implement

### 1. Sector Regime Durations (from 110-stock analysis)

| Sector | Avg Duration | Std | Recommended Lookback |
|--------|-------------|-----|---------------------|
| High Volatility | 33d | 26.4 | 50-60 days |
| Large Cap Tech | 32d | 28.7 | 45-60 days |
| Consumer Discretionary | 31d | 21.6 | 45-50 days |
| Consumer Staples | 31d | 33.8 | 45-50 days |
| Materials | 28d | 7.5 | 40-45 days |
| Financials | 27d | 13.4 | 40-45 days |
| Healthcare | 24d | 16.1 | 35-40 days |
| Energy | 22d | 5.0 | 30-35 days |
| Indices/ETFs | 22d | 3.9 | 30-35 days |
| Communication | 20d | 9.7 | 30-35 days |
| Industrials | 20d | 4.2 | 30-35 days |
| REITs | 19d | 3.0 | 25-30 days |
| Utilities | 19d | 5.5 | 25-30 days |

### 2. Sector Clusters (4 behavioral groups)

```
Cluster 1: LONG REGIMES
├── Large Cap Tech
├── Consumer Discretionary
├── Consumer Staples
└── Materials

Cluster 2: SHORT REGIMES, JUMPY
├── Healthcare
├── Industrials
└── Communication

Cluster 3: SHORT REGIMES, STABLE
├── Financials
├── Energy
├── REITs
├── Utilities
└── Indices/ETFs

Cluster 4: EXTREME VOLATILITY
└── High Volatility (COIN, MARA, GME, etc.)
```

### 3. Sector Fingerprints (z-scores)

Key patterns to encode:
- **Fat Tails:** Tech, Consumer Discretionary, Communication, Industrials
- **Normal Tails:** Financials, Energy, REITs
- **High Vol:** High Volatility, Large Cap Tech
- **Low Vol:** Consumer Staples, Utilities, Indices

---

## Tasks

### Task 1: Add Sector Metadata to RegimeRiskPlatform (30 min)

**File:** `battle-tested/PLTR-test-2.py`

```python
class RegimeRiskPlatform:
    
    # Add sector database
    SECTOR_METADATA = {
        'Large Cap Tech': {
            'tickers': ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NVDA', 'TSLA', ...],
            'avg_duration': 32,
            'recommended_lookback': 50,
            'volatility_profile': 'high',
            'tail_profile': 'fat',
            'cluster': 1,
        },
        'Utilities': {
            'tickers': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'XEL', 'SRE'],
            'avg_duration': 19,
            'recommended_lookback': 28,
            'volatility_profile': 'low',
            'tail_profile': 'normal',
            'cluster': 3,
        },
        # ... etc
    }
    
    def get_sector(self, ticker: str) -> Optional[str]:
        """Look up sector for a ticker."""
        for sector, meta in self.SECTOR_METADATA.items():
            if ticker in meta['tickers']:
                return sector
        return None
    
    def get_sector_params(self, sector: str) -> Dict:
        """Get recommended parameters for a sector."""
        if sector in self.SECTOR_METADATA:
            return self.SECTOR_METADATA[sector]
        return self.SECTOR_METADATA['Indices/ETFs']  # Default
```

### Task 2: Auto-Adjust Regime Parameters by Sector (45 min)

```python
def __init__(self, ticker, market_ticker="QQQ", ...):
    # ... existing init ...
    
    # Auto-detect sector and adjust parameters
    self.detected_sector = self.get_sector(ticker)
    if self.detected_sector:
        sector_params = self.get_sector_params(self.detected_sector)
        self.regime_lookback = sector_params['recommended_lookback']
        self.expected_duration = sector_params['avg_duration']
        print(f"📊 Detected sector: {self.detected_sector}")
        print(f"   Using lookback: {self.regime_lookback} days")
```

### Task 3: Sector-Aware Position Sizing (45 min)

```python
def compute_sector_adjusted_position(self, base_size: float) -> float:
    """
    Adjust position size based on sector volatility profile.
    
    - High vol sectors: reduce by 30%
    - Normal sectors: keep as-is
    - Low vol sectors: increase by 20%
    """
    if not self.detected_sector:
        return base_size
    
    sector_params = self.get_sector_params(self.detected_sector)
    vol_profile = sector_params['volatility_profile']
    
    adjustments = {
        'high': 0.70,    # Reduce for high vol
        'normal': 1.00,
        'low': 1.20,     # Increase for low vol
    }
    
    return base_size * adjustments.get(vol_profile, 1.0)
```

### Task 4: Sector-Specific Monte Carlo Distributions (60 min)

```python
def select_simulation_distribution(self) -> str:
    """
    Select appropriate distribution for Monte Carlo based on sector.
    
    Fat tail sectors: Use empirical bootstrap + jump diffusion
    Normal tail sectors: Can use fitted normal
    """
    if not self.detected_sector:
        return 'empirical'  # Default safe choice
    
    sector_params = self.get_sector_params(self.detected_sector)
    tail_profile = sector_params['tail_profile']
    
    if tail_profile == 'fat':
        return 'empirical_with_jumps'
    else:
        return 'fitted_t'  # t-distribution, still fat but less extreme
```

### Task 5: Sector Comparison Dashboard (45 min)

Add to Streamlit dashboard:
- Dropdown to compare multiple sectors
- Show fingerprint heatmap
- Display clustering visualization
- Regime duration distribution by sector

---

## Success Criteria

1. [ ] RegimeRiskPlatform auto-detects sector from ticker
2. [ ] Lookback period adjusts based on sector expected duration
3. [ ] Position sizing accounts for sector volatility profile
4. [ ] Monte Carlo uses appropriate distribution for sector tail profile
5. [ ] Dashboard shows sector comparison tools

---

## Files to Modify

| File | Changes |
|------|---------|
| `battle-tested/PLTR-test-2.py` | Add sector metadata, auto-adjust params |
| `to_refine/dashboard_consolidated.py` | Add sector comparison tab |

## Files to Reference

| File | Content |
|------|---------|
| `research/outputs/sector_statistics.csv` | Sector statistics |
| `research/outputs/kurtosis_validation_data.json` | Full 110-stock data |
| `research/scripts/sector_regime_analysis.py` | Sector analysis code |

---

## Definition of Done

Phase 4 is complete when:

1. [ ] Sector metadata embedded in platform
2. [ ] Auto-adjustment of parameters working
3. [ ] Position sizing accounts for sector
4. [ ] Dashboard shows sector comparison
5. [ ] Walk-forward validation tested by sector (Phase 5)

---

*Created: 2025-12-31*
