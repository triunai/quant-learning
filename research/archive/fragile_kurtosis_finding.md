# Archived Finding: Kurtosis vs Regime Duration

**Status:** ARCHIVED - Fragile, not universal  
**Date:** 2025-12-31  
**Researcher:** Phase 3 Validation

---

## Original Hypothesis

> Fat-tail stocks (high kurtosis) have LONGER regime durations than noise stocks (low kurtosis).

**Original Evidence:** r = +0.84 on 22 stocks (p < 0.001)

---

## Validation Results

### Expanded Testing: 110 stocks, 13 sectors

| Metric | Original | Expanded | Delta |
|--------|----------|----------|-------|
| Pearson r | +0.84 | +0.35 | -0.49 |
| Sample size | 22 | 110 | +88 |
| Significance | Unknown | p < 0.001 | ✓ |

### Key Findings

1. **Correlation dropped by 58%** when expanded from 22 → 110 stocks
2. **Five outliers drive most of the signal:**
   - K (Kellogg): kurt=28.6, dur=131d
   - META: kurt=26.6, dur=120d
   - NFLX: kurt=53.0, dur=34d
   - COST: kurt=9.4, dur=92d
   - AMC: kurt=19.0, dur=85d
3. **Without top 5 outliers:** r = +0.13 (essentially zero)
4. **Only 2/13 sectors show significant relationship:**
   - Large Cap Tech: r = +0.77 (p = 0.003)
   - Consumer Staples: r = +0.76 (p = 0.011)
5. **Some sectors show NEGATIVE correlation:**
   - Financials: r = -0.20
   - Healthcare: r = -0.30

---

## Why This Finding is FRAGILE

1. **Sample-dependent:** Original 22 stocks were inadvertently biased toward Tech
2. **Outlier-driven:** Remove top 5 influential points and signal disappears
3. **Sector-specific:** Only holds in 2 of 13 sectors tested
4. **Not universal:** Cannot generalize to "all stocks"

---

## What We Learned Instead

The failure of this hypothesis revealed **sector-specific regime behavior**:

### Sectors with Long, Persistent Regimes
- **Large Cap Tech:** 32 days avg duration
- **Consumer Discretionary:** 31 days
- **Consumer Staples:** 31 days

### Sectors with Short, Unstable Regimes
- **Utilities:** 19 days avg duration
- **REITs:** 19 days
- **Industrials:** 20 days

### Sector Clusters (Similar Behavior)
1. **Cluster 1:** Tech, Consumer Discretionary, Consumer Staples, Materials (Long regimes)
2. **Cluster 2:** Healthcare, Industrials, Communication (Short regimes, jumpy)
3. **Cluster 3:** Financials, Energy, REITs, Utilities, Indices (Short regimes, stable)
4. **Cluster 4:** High Volatility (Long regimes, extreme vol)

---

## Valid Research Directions (Pivoted To)

Based on this analysis, we pivoted to:

1. **Sector-based regime modeling** ✅
   - Different sectors need different lookback periods
   - Sector fingerprints can guide parameter selection

2. **Sector clustering for portfolio construction**
   - Group similar-behaving sectors
   - Diversify across behavioral clusters, not just sectors

3. **Outlier-aware regime detection**
   - K, META, NFLX, COST, AMC have special patterns
   - May warrant individual stock models

---

## Do NOT Build

❌ Stock classification by kurtosis  
❌ Trading signals based solely on kurtosis  
❌ Universal kurtosis → duration prediction  

---

## Files Related to This Finding

- `research/scripts/test_news_frequency.py` - Original 22-stock test
- `research/scripts/validate_kurtosis_finding.py` - Full validation
- `research/outputs/kurtosis_validation_data.json` - Raw data
- `research/outputs/kurtosis_validation_report.md` - Validation report
- `research/scripts/sector_regime_analysis.py` - Pivoted analysis

---

## Lessons Learned

1. **Always validate on larger samples** - 22 stocks is not enough
2. **Check outlier sensitivity** - A few extreme cases can create fake signals
3. **Test within groups, not just across** - Cross-sector variance can mask within-sector noise
4. **A failed hypothesis is still progress** - We now understand sector behavior better

---

*Archived: 2025-12-31*
