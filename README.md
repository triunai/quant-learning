# 👁️ Project Iris: Regime Risk Research Platform

> *Iris — The Egyptian goddess who links heaven and earth, carrying messages between realms. Like her namesake, this platform bridges market regimes, revealing the hidden structure of price dynamics.*

A production-grade quantitative research platform for regime-switching Monte Carlo simulation, featuring GMM-based regime detection, coherent factor models, and cross-asset analysis capabilities.

**Version:** 7.1 (Coherent Factor Model)  
**Status:** Active Research

---

## 🎯 Key Capabilities

| Feature | Description |
|---------|-------------|
| **Regime Detection** | GMM clustering on slow features (volatility, momentum, drawdown) |
| **Factor Model** | Coherent OLS with zero-mean residuals and R² fallback |
| **Semi-Markov** | Duration-aware regime transitions with length-biased sampling |
| **Monte Carlo** | 10,000+ path simulation with empirical residual sampling |
| **Risk Metrics** | VaR, CVaR, Kelly fraction, max drawdown probabilities |
| **Validation** | Walk-forward backtesting, invariant checks, multi-threshold calibration |

---

## 🔬 Recent Research Discovery

**Kurtosis-Regime Persistence Relationship (December 2024)**

Cross-sectional analysis revealed that return kurtosis predicts regime duration (r = +0.84):

| Stock Type | Kurtosis | Regime Duration | Behavior |
|------------|----------|-----------------|----------|
| Fat-Tail (META) | 26.6 | 119 days | Anchor events create persistent regimes |
| Normal (JPM) | 5.0 | 60 days | Mixed behavior |
| Noise (COIN) | 2.6 | 18 days | Constant churn, no anchors |

**Key Insight:** Fat-tail events CREATE persistent regimes (opposite of conventional wisdom).

---

## 📂 Project Structure

```
quant-learning/
├── battle-tested/           # Core platform (v7.1)
│   └── PLTR-test-2.py       # RegimeRiskPlatform class
│
├── docs/                    # Documentation
│   ├── INDEX.md             # Master documentation index
│   ├── modules/             # Module docs (Regime, Semi-Markov, etc.)
│   ├── immediate-tasks/     # Phase tracking
│   └── archive/             # Historical docs
│
├── research/                # Research artifacts
│   ├── papers/              # Research paper drafts
│   ├── scripts/             # Analysis scripts
│   └── outputs/             # Generated plots
│
├── to_refine/               # Dashboard & WIP
│   ├── dashboard_consolidated.py  # Streamlit UI
│   └── stationary_bootstrap.py    # Mode B benchmark
│
├── refinery/                # Legacy modules
├── signals_factory/         # Technical signals
├── tests/                   # Test suite
│
├── validation_package.py    # One-click validation tool
└── run_tests.py             # Test runner
```

---

## 🚀 Quick Start

### Validation Package (Recommended)
```bash
# Run full validation on any ticker
python validation_package.py --ticker PLTR --market QQQ

# Test on different stocks
python validation_package.py --ticker MSFT --market QQQ
python validation_package.py --ticker WMT --market SPY
```

### Streamlit Dashboard
```bash
cd to_refine
streamlit run dashboard_consolidated.py
```

### Run Tests
```bash
python run_tests.py
```

---

## 🧠 Core Architecture (v7.1)

### 1. Coherent Factor Model
```
r_asset = α_regime + β_regime × r_market + ε

Where:
- α_regime = Mean-based alpha (guarantees zero-mean residuals)
- β_regime = Standard OLS beta (not asymmetric)
- ε = Empirically sampled residuals (preserves fat tails)
```

### 2. Regime Detection
- **Features:** `[Vol_20d, Vol_60d, Ret_20d, Drawdown]`
- **Method:** Gaussian Mixture Model (GMM)
- **Naming:** Sharpe-based (Momentum > Bull > Neutral > Bear)

### 3. Semi-Markov Duration
- Models regime persistence explicitly
- Length-biased sampling for current regime
- Off-diagonal transitions when duration expires

### 4. Risk Dashboard
- VaR(95), CVaR(95) on simple returns
- P(MaxDD > 20%), P(MaxDD > 30%)
- DD-aware fractional Kelly sizing

---

## 📊 Validation

The platform includes multiple validation layers:

| Check | Purpose |
|-------|---------|
| Zero-mean residuals | Factor model coherence |
| Drift decomposition | α + β×Market = Actual drift |
| What-if test | Momentum > Bear probability |
| Invariant check | Sim stats match historical |
| Walk-forward | Out-of-sample calibration |

---

## 📚 Documentation

See `docs/INDEX.md` for the full documentation index:
- [Regime Risk Platform v7.1](docs/modules/REGIME_RISK_PLATFORM.md)
- [Implementation Guide](docs/implementation_guide_v71.md)
- [Phase Tracking](docs/immediate-tasks/phases/)
- [Backlog](docs/backlog/BACKLOG.md)

---

## 🔧 Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- `numpy`, `pandas`, `scipy` - Core computation
- `scikit-learn` - GMM clustering
- `yfinance` - Market data
- `streamlit` - Dashboard
- `arch` - GARCH modeling
- `matplotlib`, `seaborn` - Visualization

---

## ⚠️ Disclaimer

This is a **research tool** for modeling tail risks and regime dynamics. It is NOT financial advice. All signals, probabilities, and position sizes are for educational and research purposes only.

---

## 📜 License

Private research project. All rights reserved.
