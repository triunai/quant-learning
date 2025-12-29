# Signals Factory Documentation

> **Location:** `signals_factory/`  
> **Purpose:** Modular signal framework for combining multiple alpha sources into a unified "Weather Report."

---

## Architecture Overview

```
signals_factory/
├── __init__.py
├── base_signal.py          # Abstract base class for all signals
├── aggregator.py           # Combines multiple signals into final output
└── signal_vol_compression.py  # Concrete implementation: volatility compression detector
```

---

## Core Concepts

### 1. BaseSignal (Abstract Base Class)

All signals inherit from `BaseSignal` and must implement:

```python
class BaseSignal(ABC):
    def __init__(self, name: str):
        self.name = name
        self.direction = 0      # -1 (Short), 0 (Neutral), 1 (Long)
        self.confidence = 0.0   # 0 to 100
        self.position_sizing_mult = 1.0  # 0 to 1
        self.time_horizon = 1   # days
        self.reasoning: list[str] = []

    @abstractmethod
    def fit(self, data: pd.DataFrame):
        """Train the signal model on historical data."""
        pass

    @abstractmethod
    def predict(self, current_data: pd.DataFrame):
        """Generate a signal for the current state."""
        pass
```

**Key Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `direction` | int | -1=Short, 0=Neutral, 1=Long |
| `confidence` | float | 0-100 score |
| `position_sizing_mult` | float | 0-1 multiplier (conservative = lower) |
| `reasoning` | list[str] | Human-readable explanation |

### 2. SignalAggregator

Combines multiple signals into a final "Weather Report":

```python
class SignalAggregator:
    def __init__(self, signals: List[BaseSignal]):
        self.signals = signals
        self.weights = {
            'VolCompression': 0.3,
            'RegimeFatigue': 0.25,
            'LiquidityHorizon': 0.2,
            'CrossAsset': 0.15,
            'EarningsRisk': 0.1
        }

    def aggregate(self) -> Dict:
        """Collect votes from all signals and compute final output."""
        combined_sizing = 1.0  # Multiplicative (conservative)
        weighted_direction = 0.0  # Weighted average
        ...
```

**Aggregation Logic:**

| Metric | Combination Method | Rationale |
|--------|-------------------|-----------|
| `final_sizing` | **Multiplicative** | If ANY signal says "reduce", we reduce |
| `final_score` | **Weighted Average** | Balanced view of direction |

```python
# Sizing is multiplicative (conservative)
for signal in self.signals:
    combined_sizing *= signal.position_sizing_mult

# Direction is weighted average
if signal.direction != 0:
    weighted_direction += signal.direction * weight
    total_weight += weight
```

---

## Implemented Signals

### VolCompressionSignal

**Purpose:** Detect low-volatility periods that often precede large moves (breakouts).

**Features Used:**

| Feature | Calculation | Interpretation |
|---------|-------------|----------------|
| RV Percentile | 20d realized vol vs 1Y history | Low = compressed |
| Range Percentile | (High-Low)/Close smoothed | Low = tight range |
| VIX Percentile | Current VIX vs 1Y history | Low = complacency |

**Score Calculation:**

```python
score_components = [
    1.0 - self.rv_percentile,    # Low RV = high compression
    1.0 - self.range_percentile, # Low range = high compression
    1.0 - self.vix_percentile    # Low VIX = high compression
]
avg_compression = np.mean(score_components)
self.confidence = avg_compression * 100.0
```

**Position Sizing Logic:**

| Compression Score | Action | Rationale |
|-------------------|--------|-----------|
| > 70 | Size = 0.5× | "Calm before the storm" – reduce exposure |
| < 20 | Size = 1.0× | Already volatile – normal sizing |
| 20-70 | Size = 1.0× | Normal conditions |

```python
if self.confidence > 70:
    self.direction = 0  # Neutral (direction unknown)
    self.position_sizing_mult = 0.5
    self.reasoning.append("HIGH COMPRESSION: Reduce size, prepare for breakout.")
```

---

## Usage Example

```python
import pandas as pd
from signals_factory.signal_vol_compression import VolCompressionSignal
from signals_factory.aggregator import SignalAggregator

# Load data
data = pd.read_csv("PLTR_data.csv", index_col=0, parse_dates=True)

# Initialize signal
vol_signal = VolCompressionSignal(lookback=20, history_window=252)

# Fit and predict
vol_signal.fit(data)
result = vol_signal.predict()

print(f"Compression Score: {result['score']:.1f}/100")
print(f"Position Sizing: {result['sizing']:.2f}")
print(f"Reasoning: {result['reasoning']}")

# Aggregate with other signals
signals = [vol_signal]  # Add more signals here
aggregator = SignalAggregator(signals)
report = aggregator.aggregate()

print(f"Final Sizing: {report['final_sizing']:.2f}")
print(f"Final Direction Score: {report['final_score']:.2f}")
```

---

## Adding New Signals

To create a new signal:

1. **Inherit from BaseSignal:**

```python
from signals_factory.base_signal import BaseSignal

class MyNewSignal(BaseSignal):
    def __init__(self):
        super().__init__("MyNewSignal")
```

2. **Implement `fit()` and `predict()`:**

```python
def fit(self, data: pd.DataFrame):
    # Prepare historical distributions
    self.historical_data = data
    # ... calculate features ...

def predict(self, current_data: pd.DataFrame = None):
    # Generate signal
    self.direction = 1  # Long
    self.confidence = 75.0
    self.position_sizing_mult = 0.8
    self.reasoning = ["My signal reasoning here"]
    
    return {
        'direction': self.direction,
        'confidence': self.confidence,
        'sizing': self.position_sizing_mult
    }
```

3. **Add weight to aggregator:**

```python
# In aggregator.py
self.weights = {
    'VolCompression': 0.3,
    'MyNewSignal': 0.2,  # Add your weight
    ...
}
```

---

## Planned Signals

| Signal Name | Description | Status |
|-------------|-------------|--------|
| VolCompression | Low-vol breakout detector | ✅ Implemented |
| RegimeFatigue | Semi-Markov duration exhaustion | 🔄 In Progress |
| LiquidityHorizon | Volume-based holding period | 📋 Planned |
| CrossAsset | Sector/macro correlation | 📋 Planned |
| EarningsRisk | Event calendar overlay | 📋 Planned |

---

## Design Principles

1. **Single Responsibility** – Each signal does ONE thing well.
2. **Explainability** – Every signal provides human-readable `reasoning`.
3. **Conservative Sizing** – Multiplicative combination means "if any signal says reduce, we reduce."
4. **Walk-Forward Ready** – `fit()` + `predict()` separation enables proper backtesting.
5. **Caching** – VIX data is cached to avoid repeated API calls.

---

## Future Improvements

1. **Dynamic Weights** – Adjust weights based on backtest Sharpe/Calmar.
2. **Signal Decay** – Time-weighted confidence (recent signals count more).
3. **Correlation Penalty** – Reduce weight of highly correlated signals.
4. **Event Guards** – Automatic sizing reduction near earnings/FOMC.
