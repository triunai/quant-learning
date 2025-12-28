# 🦅 Project PLTR: Regime Risk Platform

A production-grade quantitative risk engine designed to model asymmetric market regimes, capture non-linear path risks, and generate high-fidelity trade signals.

---

## 📂 Project Structure

| Directory | Description |
| :--- | :--- |
| **`/battle-tested`** | **Proprietary v7.0 Platform.** GMM-based regime clustering, semi-Markov duration modeling, and alpha/beta macro-conditioning. |
| **`/refinery`** | **Streamlit Dashboard (v6.0)**, experimental scripts, and legacy Markov models. |
| **`/outcome`** | Audit logs, performance reports, and signal deep-dives. |

---

## 🧠 v7.0 Core Architecture (The "Alpha" Layer)

Located in `battle-tested/PLTR-test-2.py`, this is the most advanced iteration of the engine.

### 1. Regime Discovery (GMM Clustering)
Unlike standard quantile-based models, v7.0 uses **Gaussian Mixture Models (GMM)** on multi-dimensional features `[Vol_20d, Vol_60d, Ret_20d, Drawdown]` to identify real market states:
*   **Low Vol**: The "Quiet Bleed" / Underwater grind.
*   **Normal**: High-alpha melt-up / trending states.
*   **Crisis**: Left-tail volatility and regime collapse.

### 2. Semi-Markov Duration Modeling
Captures the **persistence** of regimes. The model doesn't just switch states daily; it models the "run length" of each state, correctly accounting for **residual life** (forward recurrence) when projecting forward.

### 3. Macro-Conditioning (Alpha/Beta Factor Model)
Every simulation is conditioned on market beta and regime-specific alpha:
`r = alpha_regime + beta * r_market + epsilon_empirical`
*   **Idiosyncratic Residuals**: Sampled empirically to preserve "fat tails" that parametric models miss.

---

## 📊 Streamlit Dashboard (v6.0)

Located in `refinery/dashboard.py`, the dashboard provides a visual "Mission Control" for the engine:
*   **Live Simulation**: Run 5,000+ paths in the browser.
*   **Cone Charts**: Probabilistic price paths with quartile shading.
*   **Risk Dashboard**: VaR(95), CVaR(95), and Path-level Drawdown probabilities.
*   **Transition Matrix**: Real-time visualization of regime stickiness.

---

## 📐 Risk & Signal Logic

### Kelly Criterion (DD-Aware)
The platform uses a specialized fractional Kelly formula optimized for continuous returns:
`f* = mu / sigma^2`
*   **Penalty Layer**: Position sizing is automatically slashed based on **P(MaxDD > 30%)**. Even with a strong edge, the engine will size at **0%** if the path risk is unsurvivable.

### Invariant Validation
The "Truth Serum" for the simulator. Every run verifies that simulated daily returns match historical **Mean, Std, Skew, and Kurtosis**.

---

## 🔧 Setup & Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the Dashboard
streamlit run refinery/dashboard.py

# Run the v7.0 Platform (CLI)
python battle-tested/PLTR-test-2.py
```

---

## ⚠️ Disclaimer
This is a research tool for modeling tail risks. It is NOT financial advice. Leverage and sizing are governed by the Kelly fraction which accounts for historical volatility.
