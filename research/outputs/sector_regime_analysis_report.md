# Sector Regime Analysis Report

Generated: 2025-12-31 09:29:08

## Executive Summary

Analysis of **110 stocks** across **13 sectors** to understand 
sector-level regime behavior patterns.

## Key Findings

### 1. Regime Duration Varies Dramatically by Sector

| Sector | Avg Duration | Std | Volatility |
|--------|-------------|-----|------------|
| High Volatility | 32.9d | 26.4 | 98.8% |
| Large Cap Tech | 31.6d | 28.7 | 40.4% |
| Consumer Discretionary | 30.7d | 21.6 | 26.7% |
| Consumer Staples | 30.5d | 33.8 | 19.2% |
| Materials | 27.8d | 7.5 | 29.7% |

### 2. Sector Clusters (Similar Behavior Patterns)

**Cluster 1:** Large Cap Tech, Consumer Discretionary, Consumer Staples, Materials

**Cluster 2:** Healthcare, Industrials, Communication

**Cluster 3:** Financials, Energy, REITs, Utilities, Indices/ETFs

**Cluster 4:** High Volatility


### 3. Sector Fingerprints

| Sector | Duration | Volatility | Kurtosis | Pattern |
|--------|----------|------------|----------|---------|
| High Volatility | +1.56 | +3.33 | +0.07 | Long Regimes, High Vol |
| Large Cap Tech | +1.30 | +0.39 | +0.53 | Long Regimes, Fat Tails |
| Consumer Discretiona | +1.12 | -0.30 | +0.91 | Long Regimes, Fat Tails |
| Consumer Staples | +1.09 | -0.67 | +0.38 | Long Regimes, Low Vol |
| Materials | +0.54 | -0.14 | -0.14 | Long Regimes |
| Financials | +0.32 | -0.22 | -0.96 | Normal Tails, Jump-prone |
| Healthcare | -0.15 | -0.38 | +0.38 | Jump-prone |
| Energy | -0.53 | +0.06 | -1.59 | Short Regimes, Normal Tails |
| Indices/ETFs | -0.70 | -0.69 | -0.76 | Short Regimes, Low Vol, Normal Tails |
| Communication | -1.04 | -0.23 | +2.11 | Short Regimes, Fat Tails, Jump-prone |
| Industrials | -1.07 | -0.23 | +0.81 | Short Regimes, Fat Tails, Jump-prone |
| REITs | -1.14 | -0.35 | -1.54 | Short Regimes, Normal Tails |
| Utilities | -1.31 | -0.57 | -0.19 | Short Regimes, Low Vol |

## Trading Implications

### For Long Regime Sectors (Materials, Communication, REITs)
- Use longer lookback windows for regime detection
- Trend-following strategies may outperform
- Lower rebalancing frequency

### For Short Regime Sectors (Healthcare, Consumer Staples, Utilities)
- Use shorter lookback windows
- Mean-reversion signals more relevant
- Higher rebalancing frequency may help

### For High Volatility Sectors (High Volatility, Large Cap Tech)
- Reduce position sizes relative to other sectors
- Wider stop-losses needed
- Consider volatility-scaling positions

### For Fat-Tail Sectors (Large Cap Tech, Communication)
- Use fat-tailed distributions in Monte Carlo simulations
- Account for jump risk in VaR calculations
- News-driven events are common

## Actionable Next Steps

1. **Implement sector-specific regime parameters** in RegimeRiskPlatform
2. **Create sector-based position sizing rules**
3. **Build sector rotation signals** based on regime state
4. **Test walk-forward validation** by sector

## Files Generated

- `sector_regime_analysis.png` - Visualization dashboard
- `sector_regime_analysis_report.md` - This report
