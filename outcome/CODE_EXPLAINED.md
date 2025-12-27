# 💻 Code Deep Dive: Blended Markov Chain Forecaster

> [!IMPORTANT] Purpose
> This document explains the architecture and algorithmic logic of `NVO-markov_chain.py`. It serves as the bridge between the theoretical concepts (Quant/Markov Chains) and the Python implementation.

---

## 1. Architecture Overview

The code is structured around a single class, `BlendedMarkovModel`, which encapsulates the entire pipeline:
1.  **Ingestion:** Fetching raw OHLC data from Yahoo Finance.
2.  **Transformation:** Converting prices to log-returns and binning them into states.
3.  **Modeling:** Calculating transition matrices for different time horizons.
4.  **Blending:** Combining matrices to balance long-term structure with short-term momentum.
5.  **Simulation:** Using Monte Carlo methods to project future price paths.

---

## 2. Data Ingestion & Pre-processing

### 2.1 Logarithmic Returns
The code calculates Log Returns instead of simple percentage returns.

```python
df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
```

**Why?**
*   **Additivity:** Log returns are additive over time. $R_{total} = r_1 + r_2 + \dots$
*   **Normality:** Log returns often approximate a normal distribution better than simple returns.
*   **Numerical Stability:** Avoids issues with arithmetic compounding in simulations.

### 2.2 Discretization (Quantile Binning)
We map continuous returns into discrete states using `pd.qcut`.

```python
df['State_Idx'] = pd.qcut(df['Log_Ret'], q=self.n_states, labels=False)
```
*   `q=5`: Splits data into 5 equal-sized buckets (20% of data each).
*   **0 (Crash):** Bottom 20%
*   **2 (Flat):** Middle 20%
*   **4 (Rally):** Top 20%

This transformation converts the continuous time-series problem into a **discrete state space** problem suitable for Markov Chains.

---

## 3. Transition Matrix Estimation

The core "learning" step happens in `pd.crosstab`.

```python
matrix = pd.crosstab(df['State_Idx'], df['Next_State_Idx'], normalize='index')
```

This performs **Maximum Likelihood Estimation (MLE)** for the transition probabilities:
$$ \hat{p}_{ij} = \frac{N_{ij}}{\sum_k N_{ik}} $$
Where $N_{ij}$ is the count of transitions from state $i$ to state $j$.

*   `normalize='index'`: Ensures each row sums to 1.0.
*   `reindex`: Ensures the matrix is always 5x5, even if a rare state (like a crash) never happened in a short window.

---

## 4. The "Blended" Matrix Innovation

This is the "Alpha" of the model. Standard Markov models often fail because:
*   **Long windows (10y):** Too slow to react to new regimes (e.g., a new product launch).
*   **Short windows (5y):** Too noisy and sparse.

The solution is a linear combination:

```python
self.matrix_blended = (self.matrix_5y * weight_5y) + (self.matrix_10y * (1 - weight_5y))
```

$$ P_{\text{Final}} = w \cdot P_{\text{ShortTerm}} + (1-w) \cdot P_{\text{LongTerm}} $$

This allows the model to prioritize recent market "personality" (momentum) while retaining the structural stability of long-term history.

---

## 5. Vectorized Monte Carlo Simulation

The `run_simulation` method is heavily optimized using NumPy vectorization. Instead of looping 5000 times (slow), it updates all 5000 paths simultaneously.

### 5.1 The Algorithm

1.  **State Transition:**
    For each path, select the next state based on the current state's row in $P_{\text{Final}}$.
    ```python
    new_states = np.array([np.random.choice(self.n_states, p=trans_matrix[s]) for s in states])
    ```

2.  **Return Sampling (Empirical Distribution):**
    Instead of assuming a Normal Distribution $N(\mu, \sigma)$, we sample from the **actual historical returns** observed in that state.
    ```python
    returns = np.array([np.random.choice(return_pools[s]) for s in new_states])
    ```
    > [!TIP] Fat Tails
    > This preserves the "Fat Tails" (extreme events) of the market. If the model picks state "Crash", it samples from real historical crashes, not a sanitized mathematical approximation.

3.  **Price Path Construction:**
    ```python
    cumulative_log_returns = np.cumsum(log_returns, axis=1)
    sim_paths = last_price * np.exp(cumulative_log_returns)
    ```

---

## 6. Risk Metrics: CVaR

The code calculates **Conditional Value at Risk (CVaR)**, a superior metric to simple VaR.

```python
cvar_5 = np.mean(final_prices[final_prices <= p5])
```

*   `p5`: The 5th percentile price (Value at Risk).
*   `cvar_5`: The average of all prices *below* `p5`.

This answers the critical question: *"If everything goes wrong (worst 5% scenario), how much do I stand to lose on average?"*

---

## 7. Visualization

The code produces two key visuals:
1.  **Heatmap:** Visualizes the `matrix_blended`. Strong diagonals indicate "trend persistence" (Bull days follow Bull days).
2.  **Histogram:** Shows the final probability density of prices. The skewness (left or right lean) indicates the market's bias.

---
> [!WARNING] Limitations
> *   **Stationarity:** The model assumes past transition probabilities apply to the future.
> *   **No External Data:** It ignores earnings reports, interest rates, or news. It is a pure "Price Action" model.
