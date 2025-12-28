# 📊 Quantitative Finance: The Intersection of Mathematics, Finance, and Computation

> [!INFO] Summary
> **Quantitative Finance (Quant)** is the use of mathematical models and extremely large datasets to analyze financial markets and securities. It bridges the gap between theoretical finance and practical trading using tools from statistics, physics, and computer science.

---

## 1. Introduction and Scope

Quantitative finance is the field of applied mathematics concerned with financial markets. Unlike traditional finance, which often relies on qualitative analysis (management quality, brand strength), quant finance relies on **quantitative analysis** (price patterns, volatility, correlations).

### 1.1 The "Quant"
A "Quant" is a specialist who applies mathematical and statistical methods to financial and risk management problems.
*   **Desk Quants:** Implement pricing models directly for traders.
*   **Model Validation Quants:** Independently verify that models work as intended.
*   **Research Quants:** Invent new approaches to pricing and trading.
*   **Algorithmic Traders:** Build automated systems to trade based on signals.

---

## 2. Core Mathematical Pillars

### 2.1 Stochastic Calculus
Financial markets are inherently random. Stochastic calculus provides the framework for modeling systems that evolve randomly over time.
*   **Brownian Motion ($W_t$):** A continuous-time stochastic process, serving as the standard building block for asset price models.
*   **Itô's Lemma:** The "Chain Rule" of stochastic calculus, essential for deriving the dynamics of derivative prices.

### 2.2 Linear Algebra
Used for portfolio optimization and handling large datasets.
*   **Covariance Matrix ($\Sigma$):** Captures the correlation structure between multiple assets.
*   **Eigenvalue Decomposition:** Used in Principal Component Analysis (PCA) to identify main drivers of portfolio risk.

### 2.3 Probability & Statistics
The bedrock of inference and prediction.
*   **Distributions:** Normal, Log-Normal, Student-t (for fat tails).
*   **Time Series Analysis:** ARIMA, GARCH models for forecasting volatility.

---

## 3. Fundamental Theories & Models

### 3.1 Efficient Market Hypothesis (EMH)
Proposed by Eugene Fama.
$$ P_t = E[P_{t+1} | \mathcal{F}_t] $$
Where $\mathcal{F}_t$ is the information set at time $t$. Weak, Semi-Strong, and Strong forms dictate what "information" is already priced in.

### 3.2 Modern Portfolio Theory (MPT)
Harry Markowitz introduced the idea of maximizing return for a given level of risk.
*   **Objective:** Minimize portfolio variance $\sigma_p^2$.
$$ \sigma_p^2 = \sum_{i} \sum_{j} w_i w_j \sigma_{ij} $$
Where $w_i$ are weights and $\sigma_{ij}$ is the covariance between asset $i$ and $j$.

### 3.3 Capital Asset Pricing Model (CAPM)
Relates expected return to systematic risk ($\beta$).
$$ E[R_i] = R_f + \beta_i (E[R_m] - R_f) $$
*   $R_f$: Risk-free rate
*   $\beta_i$: Sensitivity to market movements
*   $(E[R_m] - R_f)$: Market Risk Premium

### 3.4 Black-Scholes-Merton Model
The most famous equation in finance for pricing European options.
$$ \frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0 $$
The solution for a Call Option ($C$):
$$ C(S, t) = N(d_1)S_t - N(d_2)K e^{-r(T-t)} $$
Where:
$$ d_1 = \frac{\ln(S_t/K) + (r + \frac{\sigma^2}{2})(T-t)}{\sigma\sqrt{T-t}} $$
$$ d_2 = d_1 - \sigma\sqrt{T-t} $$

---

## 4. Algorithmic Trading & High-Frequency Trading (HFT)

### 4.1 Algorithmic Trading
Using computers to execute orders based on pre-defined instructions (time, price, volume).
*   **VWAP (Volume Weighted Average Price):** Executing orders in line with market volume.
*   **Statistical Arbitrage:** Exploiting mean-reversion properties of pairs or baskets of stocks.

### 4.2 High-Frequency Trading (HFT)
A subset of algo trading characterized by extremely high speeds (microseconds/nanoseconds).
*   **Market Making:** Providing liquidity by placing both buy and sell limit orders.
*   **Latency Arbitrage:** Exploiting tiny time delays between different exchanges.

---

## 5. Risk Management

> "It's not about how much you make, it's about how much you don't lose."

### 5.1 Value at Risk (VaR)
The maximum loss not exceeded with a given confidence level $\alpha$ over a period $T$.
$$ P(L > \text{VaR}_\alpha) = 1 - \alpha $$
*   Example: A 1-day 95% VaR of \$1M means there is a 5% chance of losing more than \$1M tomorrow.

### 5.2 Conditional Value at Risk (CVaR) / Expected Shortfall
VaR doesn't tell you *how bad* the loss is if the threshold is breached. CVaR does.
$$ \text{CVaR}_\alpha = E[L | L \ge \text{VaR}_\alpha] $$
This is the **average** loss in the worst $(1-\alpha)\%$ of cases.

---

## 6. The Quant Workflow

1.  **Hypothesis Generation:** "I believe stock prices revert to the mean after extreme moves."
2.  **Data Collection:** Cleaning OHLCV (Open, High, Low, Close, Volume) data.
3.  **Backtesting:** simulating the strategy on historical data.
    *   *Look-ahead bias:* Using data not available at the time of trade.
    *   *Overfitting:* Tuning parameters to perfectly match history but fail in future.
4.  **Execution:** Implementing the strategy via API.
5.  **Monitoring:** Watching for "Alpha Decay" (strategy stops working).

---
> [!TIP] Further Reading
> *   *Options, Futures, and Other Derivatives* by John Hull
> *   *Active Portfolio Management* by Grinold and Kahn
> *   *The Man Who Solved the Market* (Jim Simons biography)
