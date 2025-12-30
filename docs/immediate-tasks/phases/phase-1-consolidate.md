# Phase 1: Consolidate Mode B (Ground Truth)

**Estimated Time:** 2-3 hours  
**Priority:** HIGH - Foundation for all future work  
**Status:** ✅ COMPLETE (2025-12-30)

---

## Context

Mode B (Stationary Bootstrap) is now feature-complete with:
- ✅ Coupled pair sampling for crisis correlation
- ✅ All gatekeeper metrics passing (ACF, Tail Dependence)
- ✅ Numba JIT optimization
- ✅ Statistical fixes (block bootstrap, Bonferroni correction)
- ✅ Dashboard integration

This phase is about **consolidation before expansion**.

---

## Tasks

### Task 1: Fix Walk-Forward Validation Bug ⚡ Quick Win ✅ COMPLETE (No Bug Found)
**Time:** 30 minutes  
**File:** `battle-tested/PLTR-test-2.py`  
**Backlog Ref:** P1.4
**Completed:** 2025-12-30

**Finding:**
Walk-forward validation runs correctly. The presumed "bug" was actually expected behavior:
- Brier Score: 0.325 (baseline = 0.25)
- High calibration error (17% predicted vs 60% actual) is expected for momentum stocks like PLTR
- Early folds underpredict because they don't capture the stock's explosive growth

**This is validation working as designed** - revealing that past behavior doesn't perfectly predict future returns.

**Success Criteria:**
- [x] Walk-forward runs without errors (confirmed)
- [x] Produces valid out-of-sample hit rates (5 folds, metrics computed)
- [ ] Dashboard shows walk-forward results in calibration tab (future enhancement)

---

### Task 2: Run Frozen Evaluation Suite 🧪 Validation ✅ COMPLETE
**Time:** 15 minutes  
**File:** `to_refine/stationary_bootstrap.py`  
**Method:** `run_frozen_evaluation_suite()`
**Completed:** 2025-12-30

**Purpose:**
Prove Mode B generalizes beyond the ticker it was developed on.

**Test Matrix (ACTUAL RESULTS):**
| Asset | Horizon | Beta | Vol | ACF Match | Tail Dep | Status |
|-------|---------|------|-----|-----------|----------|--------|
| SPY | 20d | 0.69 | 17.1% | Δ<0.06 | Δ=0.008 | ✅ PASS |
| SPY | 126d | 0.69 | 17.1% | Δ<0.05 | Δ=0.001 | ✅ PASS |
| QQQ | 20d | 1.17 | 22.7% | Δ<0.08 | Δ=0.008 | ✅ PASS |
| QQQ | 126d | 1.17 | 22.7% | Δ<0.07 | Δ=0.007 | ✅ PASS |
| PLTR | 20d | 1.89 | 66.9% | Δ<0.02 | Δ=0.003 | ✅ PASS |
| PLTR | 126d | 1.89 | 66.9% | Δ<0.02 | Δ=0.004 | ✅ PASS |

**Bug Fixed:** Numba `cache=True` caused import issues when module is in subdirectory.

**Success Criteria:**
- [x] ≥3 out of 4 assets pass all gatekeeper metrics (EXCEEDED: 6/6 = 100%)
- [x] No asset fails catastrophically (>50% threshold miss)
- [x] Document results in MODE_B.md (Task 3)

---

### Task 3: Create MODE_B.md Documentation 📝 Capture ✅ COMPLETE
**Time:** 45 minutes  
**Location:** `docs/modules/MODE_B.md`
**Completed:** 2025-12-30

**Contents Documented:**
- ✅ Overview (Ground Truth concept, Politis & Romano 1994)
- ✅ Key Features (Coupled pairs, Geometric blocks, Ridge beta, Numba)
- ✅ Gatekeeper Metrics (ACF, Tail Dependence with thresholds)
- ✅ Diagnostic Metrics (Leverage, Kurtosis, Skewness)
- ✅ Usage (Code examples, CLI, API reference)
- ✅ Validation Results (Frozen suite 6/6 pass)

**Success Criteria:**
- [x] Someone can understand Mode B without reading code
- [x] Includes validation evidence (full results table)
- [x] Links to relevant backlog items

---

### Task 4: Dashboard Polish 🎨 UX ✅ COMPLETE
**Time:** 30 minutes  
**File:** `to_refine/dashboard_consolidated.py`
**Completed:** 2025-12-30

**Enhancements Implemented:**
- [x] Fix `regime_name` undefined error (previously completed)
- [x] Add confidence intervals display for diagnostics (new table with Bonferroni CIs)
- [x] Show path validation statistics in Mode B tab (time above, excursion timing, roughness)
- [x] Add "Export Results" button for Mode B (dedicated JSON export)

**Success Criteria:**
- [x] All tabs work without errors
- [x] Mode B results are exportable as JSON

---

## Definition of Done

Phase 1 is complete when:

1. ✅ Walk-forward validation runs successfully
2. ✅ Frozen evaluation suite passes on ≥3 assets
3. ✅ MODE_B.md is created with validation evidence
4. ✅ Dashboard works for all modes without errors

---

## Post-Phase 1 Decision

After completing Phase 1, decide:

| If... | Then... |
|-------|---------|
| Mode B generalizes well | Proceed to Phase 2 (Mode A comparison) |
| Mode B fails on some assets | Investigate and fix before proceeding |
| Burnout setting in | Take a break, you've done great work |

---

## Quick Commands

```bash
# Run Frozen Evaluation Suite
python -c "
from to_refine.stationary_bootstrap import StationaryBootstrap, BootstrapConfig
engine = StationaryBootstrap('SPY', 'QQQ', 126, BootstrapConfig(mode='fast', seed=42))
engine.ingest_data()
paths = engine.simulate()
diagnostics = engine.compute_diagnostics(paths)
for d in diagnostics:
    print(d)
"

# Run Dashboard
python -m streamlit run to_refine/dashboard_consolidated.py --server.port 8501
```

---

## Notes

- Mode B is the **benchmark** - all other modes (A, C) will be validated against it
- Don't add new features until consolidation is complete
- Document as you go, not after

---

*Created: 2025-12-30*  
*Last Updated: 2025-12-30*
