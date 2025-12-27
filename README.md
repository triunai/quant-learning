# 📈 Blended Markov Chain Stock Forecaster

A quantitative finance model that uses **Markov Chains** with a novel blending technique to forecast stock price distributions.

---

## 🎯 What This Model Does

This tool creates a **probabilistic forecast** of future stock prices by:

1. Learning market "regimes" (states) from historical data
2. Building transition probability matrices from two timeframes
3. Blending them for balanced predictions
4. Running Monte Carlo simulations to generate price distributions

---

## 🧠 Core Concepts

### Market States

The model discretizes daily returns into **5 quantile-based states**:

| State | Label | Description |
|-------|-------|-------------|
| 0 | **Crash** | Bottom 20% of daily returns (worst days) |
| 1 | **Bear** | 20-40th percentile returns |
| 2 | **Flat** | 40-60th percentile returns (median days) |
| 3 | **Bull** | 60-80th percentile returns |
| 4 | **Rally** | Top 20% of daily returns (best days) |

### Transition Matrix

A **5×5 matrix** where each cell `[i, j]` represents:

> "Given we're in state `i` today, what's the probability of being in state `j` tomorrow?"

**Example interpretation from your NVO matrix:**
- From **Crash** → **Crash**: 0.25 (25% chance of consecutive crash days)
- From **Bull** → **Flat**: 0.23 (23% chance of cooling off after a bull day)

---

## ⚖️ The Blending Innovation

### Why Blend?

| Single Timeframe Problem | Blended Solution |
|--------------------------|------------------|
| 10Y data captures structure but misses recent trends | ✅ Combines long-term wisdom with recent behavior |
| 5Y data captures current regime but lacks depth | ✅ Uses 70% recent + 30% historical by default |

### Formula

```
Matrix_Blended = (Matrix_5Y × 0.7) + (Matrix_10Y × 0.3)
```

This is particularly useful for stocks undergoing **regime changes** (e.g., Novo Nordisk's "Ozempic Era" since ~2021).

---

## 🎲 Monte Carlo Simulation

### Process (per simulation path)

```
For each of 5000 simulations:
    Start at current price & current state
    
    For each of 126 days:
        1. Sample next state using BLENDED transition probabilities
        2. Sample actual return from 5Y EMPIRICAL data for that state
        3. Apply return: new_price = price × exp(log_return)
```

### Fat Tail Preservation

Unlike parametric models (which assume normal distributions), this model:
- **Samples from actual historical returns** for each state
- **Preserves real volatility clusters and tail events**
- **Captures skewness and kurtosis** naturally

---

## 📊 Output Visualizations

### 1. Transition Matrix Heatmap

![Transition Matrix](./transition_matrix_example.png)

**How to read:**
- **Rows** = Current state
- **Columns** = Next day's state
- **Colors** = Probability intensity (brighter = higher probability)
- **Diagonal strength** = Regime persistence (states tend to repeat)

### 2. Price Distribution Histogram

Shows the distribution of **final prices** after 126 trading days (~6 months):

- **Median line (white)**: Most likely outcome
- **CVaR 5% line (red)**: Expected price in the worst 5% of scenarios

---

## 📐 Risk Metrics

### CVaR (Conditional Value at Risk)

Also known as **Expected Shortfall**, CVaR answers:

> "If we're in the worst 5% of outcomes, what's the average price we'd expect?"

This is more conservative than VaR because it accounts for **tail severity**, not just the threshold.

---

## 🔧 Usage

### Basic Usage

```python
from markov_chain import BlendedMarkovModel

model = BlendedMarkovModel("AAPL")  # Any valid ticker
model.build_models()
model.blend_matrices(weight_5y=0.7)
model.visualize_blended_matrix()

paths = model.run_simulation(days=126, simulations=5000)
```

### Customize Blend Weights

```python
# More weight to recent data (momentum-focused)
model.blend_matrices(weight_5y=0.9)

# More weight to historical data (mean-reversion focused)
model.blend_matrices(weight_5y=0.5)
```

---

## ⚠️ Limitations & Caveats

| Limitation | Implication |
|------------|-------------|
| **Stationarity assumption** | Assumes transition probabilities don't change over time |
| **No exogenous factors** | Ignores earnings, macro events, sector rotation |
| **Quantile boundaries** | State definitions vary by period (not absolute thresholds) |
| **Path independence** | Only considers previous day's state, not longer memory |

---

## 🛠️ Dependencies

```
yfinance>=1.0
pandas
numpy
matplotlib
seaborn
```

### yfinance 1.0 Note

This code uses `auto_adjust=True` for compatibility with yfinance 1.0+, which changed from flat columns (`'Adj Close'`) to MultiIndex columns.

---

## 📚 Further Reading

- [Markov Chains in Finance](https://en.wikipedia.org/wiki/Markov_chain#Use_in_finance)
- [Conditional Value at Risk](https://en.wikipedia.org/wiki/Expected_shortfall)
- [Regime-Switching Models](https://en.wikipedia.org/wiki/Markov_switching_multifractal)

---

## 📝 License

MIT License - Use freely, but this is for educational purposes only. **Not financial advice.**
