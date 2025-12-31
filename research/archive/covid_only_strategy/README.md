# COVID-Only Strategy Archive

**Status:** ARCHIVED  
**Date:** 2025-12-31  
**Reason:** Strategy only works in COVID era (2020-2024), fails multi-period validation

---

## Results Summary

| Period | Sharpe | Status |
|--------|--------|--------|
| 2010-2014 | 0.66 | FAIL |
| 2015-2019 | 0.72 | FAIL |
| 2020-2024 | 1.26 | PASS |

**Verdict:** Strategy is overfitted to COVID market conditions.

---

## What Was Built

- Binary regime detection (Bull = Buy, Bear = Cash)
- 60-day Sharpe threshold (>0.3 for Bull)
- Monthly rebalancing
- 10 mega-cap universe

## Why It Failed

1. **COVID regimes were extreme** - clear bull/bear with huge moves
2. **Normal markets are choppy** - binary signals create false entries
3. **Survivorship bias** - original test used 2024 winners for 2015 backtest

## Lessons Learned

1. Walk-forward validation is necessary but NOT sufficient
2. Multi-period testing required (2 of 3 periods minimum)
3. Simple strategies can still overfit to specific market conditions
4. Honest validation caught the problem before building failed product

---

## Files to Reference

- `research/scripts/walk_forward_validation.py` - Original validation
- `research/scripts/survivorship_fix_validation.py` - Corrected validation
- `research/scripts/quick_survivorship_test.py` - Final multi-period test
- `docs/sessions/2025-12-31-binary-strategy-validated.md` - Session notes

---

## Pivot Recommendation

Use validated research components for **RegimeScanner** product:
- Sector clustering (validated across periods)
- Market type classification (TREND/CHOP/TRANSITION)
- Historical similarity matching

See: `research/prototypes/regime_scanner.py`
