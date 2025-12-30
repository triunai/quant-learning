# Semi-Markov Regime Model: Solving the Memoryless Paradox

## The Memoryless Paradox
Standard Markov Chains operate on the assumption that "Tomorrow's regime depends ONLY on today's regime."
$$ P(S_{t+1} | S_t, S_{t-1}, ...) = P(S_{t+1} | S_t) $$

This is mathematically elegant but empirically incorrect for financial markets.
**Reality:**
1.  **Volatility Clustering:** High volatility regimes tend to persist longer than a simple transition probability would suggest (the "echo" effect).
2.  **Duration Dependence:** The probability of exiting a regime changes depending on how long you have been in it. Day 30 of a crash is different from Day 1.
3.  **Asymmetry:** Crashes often start abruptly (fast entry) but heal slowly (slow exit).

## The Semi-Markov Solution
A Semi-Markov model explicitly separates the process into two components:
1.  **Transition Probability ($M_{ij}$):** "Given we are leaving state $i$, where do we go next?"
2.  **Duration Distribution ($D_i$):** "How long do we stay in state $i$ before leaving?"

The simulation logic becomes:
1.  Enter state $S$.
2.  Draw a duration $d \sim D_S$ (e.g., from a Gamma distribution).
3.  Stay in $S$ for $d$ days.
4.  Transition to a new state $S'$ based on $M_{S, \cdot}$.

## Implementation Details

### Duration Distributions
We fit different distributions to the duration of each regime:
*   **Gamma Distribution:** Used for high-volatility/trending regimes (Crash, Rally). Gamma distributions can model "aging" effects where the probability of exit increases (or decreases) over time.
    $$ f(x; \alpha, \beta) = \frac{\beta^\alpha x^{\alpha-1} e^{-\beta x}}{\Gamma(\alpha)} $$
*   **Exponential Distribution:** Can be used for "flat" or memoryless regimes, though Gamma generalizes this.

### Conditional Transition Matrix
Unlike a standard Markov matrix where diagonal elements $P_{ii}$ represent the probability of staying, our transition matrix $T$ represents transitions *given* a change.
$$ T_{ii} = 0 $$
$$ \sum_{j \neq i} T_{ij} = 1 $$

## Advantages for Trading
1.  **Realistic Stress Testing:** Simulations produce "clumped" volatility events rather than random scattered bad days.
2.  **Regime Fatigue Signals:** We can calculate the probability of regime exit based on current duration.
    $$ P(\text{Exit} | \text{Duration} > t) $$
3.  **Better Drawdown Estimation:** By modeling the persistence of crash regimes, we get a more accurate tail risk assessment.
