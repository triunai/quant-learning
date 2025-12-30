# Phase 3: Research Validation (Next Session)

**Estimated Time:** 2-3 hours  
**Priority:** HIGH - Before trusting the kurtosis finding  
**Status:** 🔜 PLANNED

---

## Context

In Phase 2, we discovered a correlation between kurtosis and regime duration (r = +0.84). However, this finding is based on only 22 stocks with potential outlier issues. Before building anything on this insight, we need validation.

---

## Critical Questions to Answer

1. **Is META driving the correlation?** (Remove and retest)
2. **Does it hold across sectors?** (Tech vs Finance vs Consumer)
3. **Is it time-stable?** (2015-2020 vs 2020-2024)
4. **Can we predict out-of-sample?** (Use kurtosis to predict duration)

---

## Tasks

### Task 1: Expand to 100+ Stocks
**Time:** 45 minutes  
**Script:** `validate_kurtosis_finding.py`

```python
# Test on comprehensive stock universe
UNIVERSE = {
    'Large Cap Tech': ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NVDA', 'TSLA'],
    'Financials': ['JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'BLK'],
    'Consumer': ['WMT', 'TGT', 'COST', 'HD', 'MCD', 'NKE', 'SBUX'],
    'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'LLY', 'BMY'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY', 'MPC'],
    'Industrials': ['CAT', 'BA', 'GE', 'HON', 'UPS', 'DE', 'MMM'],
    'Indices': ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI'],
    'Crypto/High Vol': ['COIN', 'MARA', 'RIOT'],
    'REITs': ['O', 'AMT', 'PLD', 'SPG'],
    'Utilities': ['NEE', 'DUK', 'SO', 'D']
}
```

**Success Criteria:**
- [ ] Correlation remains > 0.5 with 50+ stocks
- [ ] Statistical significance (p < 0.01)

---

### Task 2: Outlier Sensitivity
**Time:** 15 minutes

**Actions:**
1. Remove META (most extreme outlier)
2. Recalculate correlation
3. If r drops below 0.5, finding is fragile

**Code:**
```python
results_no_outliers = [r for r in results if r['ticker'] != 'META']
new_correlation = np.corrcoef(...)
```

---

### Task 3: Sector Analysis
**Time:** 30 minutes

**Question:** Does kurtosis→duration hold within sectors, or only across?

| Sector | Expected Correlation |
|--------|---------------------|
| Tech | Should hold (mix of fat-tail and noise) |
| Utilities | Might not hold (all low-kurtosis) |
| Financials | Should hold (crisis events) |

---

### Task 4: Time-Period Robustness
**Time:** 30 minutes

**Test:**
- Period A: 2015-01-01 to 2019-12-31
- Period B: 2020-01-01 to 2024-12-31

**Question:** Is the relationship COVID-era specific or fundamental?

---

### Task 5: Out-of-Sample Prediction
**Time:** 45 minutes

**The Real Test:**
1. Use 2015-2022 data to fit kurtosis→duration relationship
2. Predict regime durations for 2023-2024
3. Compare predicted vs actual

**If this fails:** The finding is not predictive, just descriptive.

---

## Decision Matrix

After completing tasks, choose path:

| If Results Show... | Then... |
|-------------------|---------|
| r > 0.5 on 100+ stocks, holds across sectors, survives OOS | **VALID** - Proceed to stock classification implementation |
| r > 0.5 but driven by outliers | **FRAGILE** - Document as interesting observation only |
| r < 0.3 on expanded sample | **FAILED** - Finding was noise, archive and move on |

---

## Files to Create

| File | Purpose |
|------|---------|
| `validate_kurtosis_finding.py` | Runs full validation suite |
| `research/validation_results.md` | Documents validation outcomes |

---

## Definition of Done

Phase 3 is complete when:

1. [ ] Tested on 50+ stocks across 5+ sectors
2. [ ] Outlier sensitivity analyzed
3. [ ] Time-period robustness checked
4. [ ] Out-of-sample prediction attempted
5. [ ] Decision documented (VALID / FRAGILE / FAILED)

---

## If Finding is VALID

Next steps would be:
- Implement `classify_stock_type()` in platform
- Adjust `n_regimes` based on kurtosis
- Write proper research paper

## If Finding is FRAGILE/FAILED

Next steps would be:
- Document as "interesting observation"
- Focus on other improvements (walk-forward, multi-asset)
- Don't build products around it

---

*Created: 2025-12-30*
