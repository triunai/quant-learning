# 📚 Documentation Index

> **Last Updated:** 2025-12-30  
> **Platform Version:** v7.1 (Coherent Factor Model)

---

## Quick Links

| Document | Purpose |
|----------|---------|
| [REGIME_RISK_PLATFORM.md](modules/REGIME_RISK_PLATFORM.md) | Main platform documentation (v7.1) |
| [implementation_guide_v71.md](implementation_guide_v71.md) | Bug fixes and new features |
| [BACKLOG.md](backlog/BACKLOG.md) | Engineering task tracker |
| [ai_validation_workflow.md](workflows/ai_validation_workflow.md) | Validation workflow |

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

| Document | Topic |
|----------|-------|
| [Kurtosis Paper Draft](../research/papers/kurtosis_regime_persistence/draft.md) | Fat-tail events create persistent regimes |

---

## Session Notes

| Session | Date | Summary |
|---------|------|---------|
| [Mode A Fundamentals](sessions/2025-12-30-mode-a-fundamentals.md) | 2025-12-30 | Bug fix + kurtosis discovery |

---

## Phase Tracking

| Phase | Status | Description |
|-------|--------|-------------|
| [Phase 1: Consolidate](immediate-tasks/phases/phase-1-consolidate.md) | ✅ Complete | Mode B validation |
| [Phase 2: Mode A Fundamentals](immediate-tasks/phases/phase-2-mode-a-fundamentals.md) | ✅ Complete | Factor model fix + kurtosis discovery |
| [Phase 3: Research Validation](immediate-tasks/phases/phase-3-research-validation.md) | 🔜 Planned | Validate kurtosis finding on 100+ stocks |

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
| `test_coherent_fix.py` | Factor model validation |
| `compare_regime_persistence.py` | Cross-stock analysis |
| `test_news_frequency.py` | Kurtosis research test |

---

## Key Findings (December 2025)

### 1. Factor Model Bug Fix
- **Issue:** Median alpha + asymmetric beta → non-zero mean residuals
- **Fix:** Standard OLS + mean-based alpha + enforced zero mean
- **Validation:** All regimes now have 0.0000% residual mean

### 2. Kurtosis-Persistence Discovery
- **Finding:** r = +0.84 correlation between kurtosis and regime duration
- **Insight:** Fat-tail events CREATE persistent regimes (opposite of conventional wisdom)
- **Implication:** Need stock-type-specific regime detection (Fat-Tail vs Noise)

---

*This index is auto-maintained. See individual documents for details.*
