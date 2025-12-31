# 📚 Documentation Index

> **Last Updated:** 2025-12-31  
> **Platform Version:** v7.1 (Coherent Factor Model)  
> **LATEST:** 🎯 Binary Regime Strategy OOS Validated (Sharpe 1.40)

---

## Quick Links

| Document | Purpose |
|----------|---------|
| [REGIME_RISK_PLATFORM.md](modules/REGIME_RISK_PLATFORM.md) | Main platform documentation (v7.1) |
| [implementation_guide_v71.md](implementation_guide_v71.md) | Bug fixes and new features |
| [BACKLOG.md](backlog/BACKLOG.md) | Engineering task tracker |
| [ai_validation_workflow.md](workflows/ai_validation_workflow.md) | Validation workflow |
| **[2025-12-31-binary-strategy-validated.md](sessions/2025-12-31-binary-strategy-validated.md)** | **🎯 Latest: OOS Validated Strategy** |

---

## 🎯 Production-Ready Strategy

**RegimeBinary Signals** - Walk-forward validated trading strategy

| Metric | Value | Status |
|--------|-------|--------|
| OOS Sharpe | **1.40** | ✅ Validated |
| Volatility | 8.0% | ✅ 50% less than SPY |
| Rule | Bull=Buy, Bear=Cash | ✅ Simple & Explainable |

**Dashboard:** `research/dashboards/regime_binary_dashboard.py`

---

## Module Documentation

### Core Modules
| Module | File | Description |
|--------|------|-------------|
| Regime Risk Platform | [REGIME_RISK_PLATFORM.md](modules/REGIME_RISK_PLATFORM.md) | v7.1 Monte Carlo engine with regime conditioning |
| Stationary Bootstrap | [MODE_B.md](modules/MODE_B.md) | Ground truth benchmark (non-parametric) |
| Semi-Markov | [SEMI_MARKOV.md](modules/SEMI_MARKOV.md) | Duration-aware regime transitions |
| Signals Factory | [SIGNALS_FACTORY.md](modules/SIGNALS_FACTORY.md) | Technical signal generation |
| Market Mood System | [MARKET_MOOD_SYSTEM.md](modules/MARKET_MOOD_SYSTEM.md) | Sentiment and macro indicators |

---

## Implementation Guides

| Guide | Purpose |
|-------|---------| 
| [implementation_guide_v71.md](implementation_guide_v71.md) | v7.0 → v7.1 upgrade details |

---

## Research

| Document | Topic | Status |
|----------|-------|--------|
| `research/scripts/walk_forward_validation.py` | OOS validation (Binary vs Lifecycle) | ✅ Key File |
| `research/scripts/regime_lifecycle.py` | Lifecycle phases (overfitted) | ⚠️ Research |
| `research/scripts/sector_regime_analysis.py` | Sector fingerprinting | ✅ Complete |
| `research/archive/fragile_kurtosis_finding.md` | Archive of failed hypothesis | 📁 Archived |

---

## Session Notes

| Session | Date | Summary |
|---------|------|---------|
| **[Binary Strategy Validated](sessions/2025-12-31-binary-strategy-validated.md)** | **2025-12-31** | **🎯 OOS Sharpe 1.40 - Simplicity Wins!** |
| [Mode A Fundamentals](sessions/2025-12-30-mode-a-fundamentals.md) | 2025-12-30 | Bug fix + kurtosis discovery |

---

## Phase Tracking

| Phase | Status | Description |
|-------|--------|-------------|
| [Phase 1: Consolidate](immediate-tasks/phases/phase-1-consolidate.md) | ✅ Complete | Mode B validation |
| [Phase 2: Mode A Fundamentals](immediate-tasks/phases/phase-2-mode-a-fundamentals.md) | ✅ Complete | Factor model fix + kurtosis discovery |
| [Phase 3: Research Validation](immediate-tasks/phases/phase-3-research-validation.md) | ✅ Complete | 110-stock validation → FRAGILE → Pivoted |
| [Phase 4: Sector Implementation](immediate-tasks/phases/phase-4-sector-implementation.md) | 🔄 Pivoted | Complex sector sizing overfitted; Binary wins |

---

## Archive

Archived documentation moved to `docs/archive/`:
- `DCA(archived)/` - DCA Dashboard architecture (future project)
- `v7_audit.md`, `v7_deep_dive.md` - Pre-bug-fix PLTR analysis
- `SEMI_MARKOV_KNOWLEDGE_BASE.md` - Superseded by modules/SEMI_MARKOV.md
- Test suite summaries - May be outdated

---

## Validation Tools

| Tool | Usage |
|------|-------|
| `validation_package.py` | `python validation_package.py --ticker PLTR` |
| `research/scripts/walk_forward_validation.py` | **Walk-forward OOS test** |
| `research/dashboards/regime_binary_dashboard.py` | **Live signals dashboard** |
| `test_coherent_fix.py` | Factor model validation |
| `compare_regime_persistence.py` | Cross-stock analysis |

---

## Key Findings (December 2025)

### 🎯 1. Binary Regime Strategy (2025-12-31) - VALIDATED
- **Rule:** Bull regime (Sharpe > 0.3) = Buy, else = Cash
- **OOS Sharpe:** 1.40 (2020-2024, trained on 2015-2019)
- **Volatility:** 8.0% (50% less than SPY)
- **Key Insight:** SIMPLE beats COMPLEX in walk-forward testing

### 2. Kurtosis Hypothesis - FAILED/PIVOTED
- **Original Finding:** r = +0.84 correlation between kurtosis and regime duration
- **Validation Result:** FRAGILE (driven by outliers)
- **Pivot:** Led to sector analysis, eventually to Binary strategy

### 3. Factor Model Bug Fix
- **Issue:** Median alpha + asymmetric beta → non-zero mean residuals
- **Fix:** Standard OLS + mean-based alpha + enforced zero mean
- **Validation:** All regimes now have 0.0000% residual mean

---

*This index is auto-maintained. See individual documents for details.*

