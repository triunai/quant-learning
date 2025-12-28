# 🔗 Markov Chains: Stochastic Processes for Predictive Modeling

> [!INFO] Summary
> A **Markov Chain** is a mathematical system that undergoes transitions from one state to another according to certain probabilistic rules. The defining characteristic is that the probability of transitioning to any particular future state depends **solely on the current state** and time elapsed, not on the sequence of events that preceded it.

---

## 1. Formal Definition

A stochastic process $\{X_t, t \in T\}$ is a Markov chain if it satisfies the **Markov Property** (or "memorylessness").

### 1.1 The Markov Property
$$ P(X_{t+1} = j \mid X_t = i, X_{t-1} = i_{t-1}, \dots, X_0 = i_0) = P(X_{t+1} = j \mid X_t = i) $$

In plain English: *The future is independent of the past, given the present.*

---

## 2. Key Components

### 2.1 State Space ($\mathcal{S}$)
The set of all possible conditions the system can be in.
*   In a board game: $\mathcal{S} = \{ \text{Start}, \text{Square 1}, \dots, \text{Finish} \}$
*   In this Stock Model: $\mathcal{S} = \{ \text{Crash}, \text{Bear}, \text{Flat}, \text{Bull}, \text{Rally} \}$

### 2.2 Transition Probability Matrix ($P$)
A square matrix where the element $p_{ij}$ represents the probability of moving from state $i$ to state $j$.

$$
P = \begin{bmatrix}
p_{11} & p_{12} & \dots & p_{1n} \\
p_{21} & p_{22} & \dots & p_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
p_{n1} & p_{n2} & \dots & p_{nn}
\end{bmatrix}
$$

**Properties:**
1.  $0 \le p_{ij} \le 1$
2.  $\sum_{j} p_{ij} = 1$ (Rows must sum to 100%)

---

## 3. The Chapman-Kolmogorov Equations
How do we calculate probabilities $n$ steps into the future?
If $P$ is the one-step transition matrix, then the $n$-step transition matrix $P^{(n)}$ is simply the $n$-th power of $P$.

$$ P^{(n)} = P^n $$

Example: If there is a 70% chance of rain tomorrow if it rains today, what is the chance of rain in 2 days?
$$ \pi_{t+2} = \pi_t \times P^2 $$

---

## 4. Stationary Distribution ($\pi$)

Over a long enough time horizon, many Markov chains settle into a **steady state** where the probabilities of being in any state become constant, regardless of the starting state. This is the stationary distribution $\pi$.

It satisfies the equation:
$$ \pi = \pi P $$

Subject to:
$$ \sum_{i} \pi_i = 1 $$

In financial contexts, this tells us the "long-term average" regime of the market. For instance, the market might be in a "Bull" state 40% of the time over the long run.

---

## 5. Applications in Finance

### 5.1 Regime-Switching Models
Markets behave differently in different economic climates. A simple linear regression might fail because it assumes a single set of parameters.
*   **Regime 1 (Low Volatility):** Steady growth.
*   **Regime 2 (High Volatility):** Panic/Crisis.
Markov Switching models (like Hamilton's model) dynamically update the probability of being in a specific regime.

### 5.2 Credit Rating Migration
Banks use Markov chains to model the migration of corporate bonds between credit ratings (e.g., from AAA to BBB).
$$
P_{\text{Credit}} = \begin{bmatrix}
\text{AAA} \to \text{AAA} & \text{AAA} \to \text{AA} & \dots \\
\text{AA} \to \text{AAA} & \text{AA} \to \text{AA} & \dots \\
\vdots & \vdots & \ddots
\end{bmatrix}
$$

### 5.3 Stock Price Forecasting (Our Model)
By discretizing continuous returns into discrete states (quantiles), we convert a complex continuous problem into a tractable discrete Markov problem.
*   **Advantage:** Captures non-linear dependencies (e.g., "volatility clustering" where crashes follow crashes).
*   **Disadvantage:** Loss of information due to discretization.

---
> [!NOTE] Ergodicity
> A Markov chain is **ergodic** if it is possible to go from every state to every other state (not necessarily in one move). This is a requirement for a unique stationary distribution to exist. In financial data, we usually assume ergodicity (market cycles repeat).
