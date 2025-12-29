# 🏛️ DCA Weather Report Platform — Clean Architecture v1.1

> **Status:** 📐 Architecture Specification (Patched)  
> **Last Updated:** 2024-12-29  
> **Principles:** Clean Architecture, Domain-Driven Design, Dependency Inversion  
> **Patches Applied:** 9 P0 fixes, 6 P1 improvements

---

## 🎯 Architecture Vision

This platform follows **Clean Architecture** principles:

1. **Independence of Frameworks** — Business logic doesn't know about Supabase, Streamlit, or yfinance
2. **Testability** — Core can be tested without UI, database, or external services
3. **Independence of UI** — Swap Streamlit for React without touching domain
4. **Independence of Database** — Swap Supabase for SQLite without touching domain
5. **Independence of External Services** — Swap yfinance for Polygon without touching domain

---

## 📐 The Dependency Rule

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRAMEWORKS & DRIVERS                      │
│  (Streamlit, FastAPI, Supabase SDK, yfinance, pandas)           │
│                              ↓                                   │
├─────────────────────────────────────────────────────────────────┤
│                      INTERFACE ADAPTERS                          │
│  (Controllers, Presenters, Gateways, Repositories)              │
│                              ↓                                   │
├─────────────────────────────────────────────────────────────────┤
│                       APPLICATION LAYER                          │
│  (Use Cases, Orchestrators, Application Services)               │
│                              ↓                                   │
├─────────────────────────────────────────────────────────────────┤
│                         DOMAIN LAYER                             │
│  (Entities, Value Objects, Domain Services, Interfaces)         │
└─────────────────────────────────────────────────────────────────┘

RULE: Dependencies point INWARD only. Inner layers know nothing about outer layers.
```

---

## 🔴 Audit Patches Applied (P0 Must-Fix)

| # | Issue | Fix |
|---|-------|-----|
| 1 | Domain uses pandas (impure) | Domain returns own value types, adapters convert |
| 2 | Column name mismatch | Canonical snake_case everywhere + SchemaValidator |
| 3 | Adj Close not guaranteed | Explicit fallback mode with flags |
| 4 | `date.today()` wrong for finance | Use `asof_market_date` from last completed bar |
| 5 | VIX cache returns wrong fallback flag | Freshness window = 1-2 trading days |
| 6 | Unused regime_engine injection | Remove from light use case (only heavy) |
| 7 | Artifact port methods not defined | Full IArtifactStore interface defined |
| 8 | Mutable List in frozen entity | Use `tuple[str, ...]` |
| 9 | Money uses float | Use `Decimal` with currency quantization |

---

## 🗂️ Folder Structure

```
quant-learning/
├── src/
│   ├── domain/                    # 🟢 INNERMOST - Pure business logic (NO PANDAS)
│   │   ├── __init__.py
│   │   ├── entities/              # Core business objects
│   │   │   ├── __init__.py
│   │   │   ├── asset.py           # Asset entity
│   │   │   ├── portfolio.py       # Portfolio aggregate
│   │   │   ├── decision.py        # DCA Decision entity
│   │   │   ├── regime.py          # Regime value object
│   │   │   ├── market_data.py     # MarketData aggregate (replaces DataFrame)
│   │   │   └── price_bar.py       # PriceBar value object
│   │   │
│   │   ├── value_objects/         # Immutable domain primitives
│   │   │   ├── __init__.py
│   │   │   ├── multiplier.py      # Bounded multiplier (0.3-1.5)
│   │   │   ├── money.py           # Decimal-based currency
│   │   │   ├── ticker.py          # Validated ticker symbol
│   │   │   ├── market_date.py     # Trading day (not calendar date)
│   │   │   ├── dca_schedule.py    # Monthly schedule rules
│   │   │   └── timeframe.py       # Date range with validation
│   │   │
│   │   ├── services/              # Domain logic (stateless)
│   │   │   ├── __init__.py
│   │   │   ├── dca_calculator.py  # Core DCA multiplier logic
│   │   │   ├── regime_classifier.py  # Regime assignment logic
│   │   │   └── correlation_guard.py  # Cluster penalty logic
│   │   │
│   │   └── interfaces/            # Ports (abstract contracts) — NO PANDAS
│   │       ├── __init__.py
│   │       ├── market_data_port.py    # IMarketDataProvider
│   │       ├── macro_data_port.py     # IMacroDataProvider (VIX, etc.)
│   │       ├── portfolio_port.py      # IPortfolioRepository
│   │       ├── decision_port.py       # IDecisionRepository
│   │       ├── artifact_port.py       # IArtifactStore (full interface)
│   │       └── analysis_port.py       # IAnalysisResult
│   │
│   ├── application/               # 🟡 USE CASES - Orchestration
│   │   ├── __init__.py
│   │   ├── use_cases/
│   │   │   ├── __init__.py
│   │   │   ├── compute_dca_decision.py      # Single asset DCA (light)
│   │   │   ├── compute_portfolio_dca.py     # Multi-asset with correlation
│   │   │   ├── run_heavy_analysis.py        # Weekly heavy compute
│   │   │   ├── run_light_update.py          # Daily light update
│   │   │   └── run_backtest.py              # Backtest execution
│   │   │
│   │   ├── services/              # Application services
│   │   │   ├── __init__.py
│   │   │   ├── analysis_orchestrator.py     # Heavy/light coordination
│   │   │   ├── signal_aggregator.py         # Signal combination
│   │   │   ├── calendar_service.py          # Trading day resolution
│   │   │   └── cache_manager.py             # Cache invalidation logic
│   │   │
│   │   └── dto/                   # Data Transfer Objects
│   │       ├── __init__.py
│   │       ├── dca_request.py     # Input DTOs
│   │       ├── dca_response.py    # Output DTOs
│   │       └── analysis_result.py # Heavy analysis output
│   │
│   ├── infrastructure/            # 🔴 OUTERMOST - External world (PANDAS LIVES HERE)
│   │   ├── __init__.py
│   │   ├── data_providers/        # Market data adapters
│   │   │   ├── __init__.py
│   │   │   ├── yahoo_provider.py      # yfinance implementation
│   │   │   ├── stooq_provider.py      # Stooq fallback
│   │   │   ├── composite_provider.py  # Fallback chain
│   │   │   ├── vix_provider.py        # VIX-specific provider
│   │   │   └── schema_validator.py    # Canonical schema enforcement
│   │   │
│   │   ├── persistence/           # Database adapters
│   │   │   ├── __init__.py
│   │   │   ├── supabase/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── client.py          # Supabase client singleton
│   │   │   │   ├── portfolio_repo.py  # IPortfolioRepository impl
│   │   │   │   ├── decision_repo.py   # IDecisionRepository impl (with upsert)
│   │   │   │   └── artifact_repo.py   # IArtifactStore impl
│   │   │   │
│   │   │   └── local/                 # Local dev alternatives
│   │   │       ├── __init__.py
│   │   │       ├── sqlite_repo.py     # SQLite fallback
│   │   │       └── parquet_store.py   # Local parquet store
│   │   │
│   │   ├── engines/               # Quantitative engines
│   │   │   ├── __init__.py
│   │   │   ├── regime_risk_platform.py    # v7.0 GMM engine
│   │   │   ├── semi_markov_engine.py      # Duration modeling
│   │   │   └── feature_computer.py        # Feature extraction
│   │   │
│   │   ├── cache/                 # Caching infrastructure
│   │   │   ├── __init__.py
│   │   │   ├── disk_cache.py      # diskcache wrapper
│   │   │   ├── duckdb_cache.py    # DuckDB analytics cache
│   │   │   └── http_cache.py      # requests-cache wrapper
│   │   │
│   │   ├── auth/                  # Authentication
│   │   │   ├── __init__.py
│   │   │   └── user_context.py    # Derive user_id from JWT, not settings
│   │   │
│   │   └── config/                # Configuration
│   │       ├── __init__.py
│   │       ├── settings.py        # Environment-based config
│   │       └── strategies/        # Strategy YAML configs
│   │           ├── base.yaml
│   │           ├── stocks.yaml
│   │           └── etfs.yaml
│   │
│   └── presentation/              # 🔵 UI LAYER - User interfaces
│       ├── __init__.py
│       ├── streamlit/             # Streamlit dashboard
│       │   ├── __init__.py
│       │   ├── app.py             # Main Streamlit entry
│       │   ├── pages/
│       │   │   ├── __init__.py
│       │   │   ├── portfolio.py   # Portfolio input page
│       │   │   ├── dca_advisor.py # DCA decision page
│       │   │   ├── analysis.py    # Deep analysis page
│       │   │   └── backtest.py    # Backtest page
│       │   │
│       │   └── components/        # Reusable UI components
│       │       ├── __init__.py
│       │       ├── regime_gauge.py
│       │       ├── decision_card.py
│       │       └── cone_chart.py
│       │
│       └── api/                   # FastAPI (future)
│           ├── __init__.py
│           ├── main.py            # FastAPI entry
│           ├── routes/
│           │   ├── __init__.py
│           │   ├── dca.py
│           │   ├── analysis.py
│           │   └── portfolio.py
│           │
│           └── schemas/           # Pydantic models
│               ├── __init__.py
│               └── responses.py
│
├── tests/                         # Mirrors src/ structure
│   ├── domain/
│   ├── application/
│   ├── infrastructure/
│   └── integration/
│
├── scripts/                       # CLI entry points
│   ├── run_daily.py               # Daily light pipeline
│   ├── run_weekly.py              # Weekly heavy pipeline
│   └── run_backtest.py            # Backtest runner
│
├── data/                          # Local data (gitignored)
│   ├── cache/
│   ├── parquet/
│   └── models/
│
└── docs/                          # Documentation
    ├── ARCHITECTURE.md            # This file
    ├── ENGINEERING_DECISIONS_v1.1_PATCHED.md
    └── ...
```

---

## 🟢 Domain Layer (The Core) — NO EXTERNAL DEPENDENCIES

The domain layer contains **pure business logic** with **zero external dependencies**.  
**No pandas. No yfinance. No Supabase. Just pure Python + standard library.**

### Value Objects (P0 Patched)

```python
# src/domain/value_objects/money.py

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

@dataclass(frozen=True)
class Money:
    """Decimal-based currency with proper quantization."""
    
    amount: Decimal
    currency: str = "USD"
    
    def __post_init__(self):
        # Enforce Decimal type
        if not isinstance(self.amount, Decimal):
            object.__setattr__(self, 'amount', Decimal(str(self.amount)))
        
        # Quantize to 2 decimal places for USD
        quantized = self.amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        object.__setattr__(self, 'amount', quantized)
    
    def __mul__(self, factor: float) -> "Money":
        return Money(self.amount * Decimal(str(factor)), self.currency)
    
    def __repr__(self) -> str:
        return f"${self.amount:,.2f} {self.currency}"
```

```python
# src/domain/value_objects/market_date.py

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

@dataclass(frozen=True)
class MarketDate:
    """
    Trading day value object.
    
    Represents the last completed trading session, NOT calendar date.
    """
    
    value: date
    timezone: str = "America/New_York"
    is_trading_day: bool = True
    
    @classmethod
    def from_last_completed_bar(cls, bar_timestamp: datetime) -> "MarketDate":
        """Create from the timestamp of the last completed bar."""
        return cls(value=bar_timestamp.date())
    
    @classmethod
    def today_market_close(cls) -> "MarketDate":
        """
        Get today's market date IF market has closed, else previous trading day.
        
        Note: This should be computed by application layer with calendar awareness.
        """
        raise NotImplementedError("Use CalendarService in application layer")
```

```python
# src/domain/value_objects/multiplier.py

from dataclasses import dataclass

@dataclass(frozen=True)
class Multiplier:
    """DCA multiplier with enforced bounds."""
    
    value: float
    
    MIN = 0.3
    MAX = 1.5
    CRISIS_CAP = 0.5
    
    def __post_init__(self):
        if not self.MIN <= self.value <= self.MAX:
            raise ValueError(f"Multiplier must be in [{self.MIN}, {self.MAX}], got {self.value}")
    
    @classmethod
    def clamped(cls, raw: float) -> "Multiplier":
        """Create a multiplier, clamping to bounds."""
        clamped = max(cls.MIN, min(cls.MAX, raw))
        return cls(value=round(clamped, 4))
    
    @classmethod
    def crisis_capped(cls, raw: float) -> "Multiplier":
        """Create a multiplier with crisis cap applied."""
        capped = min(raw, cls.CRISIS_CAP)
        return cls.clamped(capped)
```

### Entities (P0 Patched — Immutable Reasoning)

```python
# src/domain/entities/price_bar.py

from dataclasses import dataclass
from decimal import Decimal
from datetime import date

@dataclass(frozen=True)
class PriceBar:
    """Single OHLCV bar — pure domain type, no pandas."""
    
    date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    adj_close: Decimal
    volume: int
    
    @property
    def is_adjusted(self) -> bool:
        return self.adj_close != self.close
```

```python
# src/domain/entities/market_data.py

from dataclasses import dataclass
from typing import Tuple
from datetime import datetime

from .price_bar import PriceBar


@dataclass(frozen=True)
class MarketDataMeta:
    """Metadata about market data fetch."""
    
    symbol: str
    source: str
    asof_utc: datetime
    timezone: str
    currency: str
    corporate_actions_applied: bool
    returns_basis: str  # 'adj_close' or 'close_unadjusted'
    trust_level: str    # 'high', 'medium', 'lower'


@dataclass(frozen=True)
class MarketData:
    """
    Market data aggregate — DOMAIN TYPE, not DataFrame.
    
    Infrastructure adapters convert DataFrames → MarketData.
    """
    
    bars: Tuple[PriceBar, ...]
    meta: MarketDataMeta
    
    @property
    def last_bar(self) -> PriceBar:
        return self.bars[-1]
    
    @property
    def asof_market_date(self) -> "MarketDate":
        """Date of last completed bar (use this, not date.today())."""
        from ..value_objects import MarketDate
        return MarketDate(value=self.last_bar.date)
```

```python
# src/domain/entities/decision.py

from dataclasses import dataclass
from typing import Tuple
from decimal import Decimal

from ..value_objects import Multiplier, Money, Ticker, MarketDate


@dataclass(frozen=True)
class DCADecision:
    """
    Immutable DCA decision for a single asset.
    
    P0 Patches Applied:
    - reasoning is Tuple (immutable), not List
    - decision_date is MarketDate, not date.today()
    - Money uses Decimal
    """
    
    symbol: Ticker
    decision_date: MarketDate           # Last completed bar date
    asof_utc: str                        # ISO timestamp of computation
    base_amount: Money
    adjusted_amount: Money
    multiplier: Multiplier
    regime: str
    reasoning: Tuple[str, ...]           # P0 FIX: Immutable tuple, not list
    
    # Reproducibility (P1)
    run_id: str
    snapshot_id: str
    strategy_hash: str
    engine_version: str
    
    @property
    def is_reduced(self) -> bool:
        return self.multiplier.value < 1.0
    
    @property
    def is_crisis_capped(self) -> bool:
        return self.regime == "Crisis" and self.multiplier.value <= 0.5
```

### Ports (Interfaces) — NO PANDAS

```python
# src/domain/interfaces/market_data_port.py

from abc import ABC, abstractmethod
from datetime import date

from ..entities import MarketData


class IMarketDataProvider(ABC):
    """
    Port for market data acquisition.
    
    Returns DOMAIN TYPES, not DataFrames.
    Infrastructure adapters convert DataFrames → MarketData.
    """
    
    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> MarketData:
        """Fetch OHLCV data as domain MarketData type."""
        pass
```

```python
# src/domain/interfaces/macro_data_port.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class VIXResult:
    """VIX fetch result with freshness metadata."""
    
    value: float
    as_of_date: date
    is_stale: bool       # True if > 2 trading days old
    is_fallback: bool    # True if using default (20)
    source: str          # 'yahoo', 'cached', 'default'


class IMacroDataProvider(ABC):
    """Port for macro data (VIX, rates, etc.) — separate from OHLCV."""
    
    @abstractmethod
    def get_vix(self) -> VIXResult:
        """
        Get current VIX level with freshness flags.
        
        Freshness rules:
        - Fresh: <= 2 trading days old
        - Stale: > 2 trading days old (is_stale=True)
        - Fallback: No data available (is_fallback=True, value=20)
        """
        pass
```

```python
# src/domain/interfaces/artifact_port.py

from abc import ABC, abstractmethod
from typing import Optional
from datetime import date
from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisArtifact:
    """Heavy analysis results."""
    
    symbol: str
    asof_date: date
    regime: str
    regime_id: int
    regime_probs: tuple  # Immutable
    
    # Risk metrics
    kelly_fraction: float
    prob_dd_20: float
    prob_dd_30: float
    var_95: float
    cvar_95: float
    
    # Semi-Markov
    fatigue_score: float
    days_in_regime: int
    expected_remaining_days: float
    
    # Reproducibility
    run_id: str
    snapshot_id: str
    engine_version: str
    strategy_hash: str
    
    # Monte Carlo reproducibility
    rng_seed: int
    n_paths: int
    horizon_days: int


class IArtifactStore(ABC):
    """
    Port for artifact persistence.
    
    P0 FIX: All methods used by use cases are defined here.
    """
    
    @abstractmethod
    def save_snapshot(self, symbol: str, data: "MarketData") -> str:
        """
        Save immutable data snapshot.
        
        Returns: snapshot_id (content hash)
        """
        pass
    
    @abstractmethod
    def load_snapshot(self, snapshot_id: str) -> Optional["MarketData"]:
        """Load snapshot by ID."""
        pass
    
    @abstractmethod
    def save_analysis(self, analysis: AnalysisArtifact) -> str:
        """
        Save heavy analysis results.
        
        Returns: artifact_id
        """
        pass
    
    @abstractmethod
    def load_latest_analysis(self, symbol: str) -> Optional[AnalysisArtifact]:
        """Load most recent analysis for symbol."""
        pass
    
    @abstractmethod
    def load_analysis_by_date(
        self, 
        symbol: str, 
        asof_date: date,
    ) -> Optional[AnalysisArtifact]:
        """Load analysis for specific date (for backtests)."""
        pass
```

```python
# src/domain/interfaces/decision_port.py

from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import date

from ..entities import DCADecision
from ..value_objects import MarketDate


class IDecisionRepository(ABC):
    """
    Port for decision persistence.
    
    P1 FIX: Upsert semantics + unique constraint awareness.
    """
    
    @abstractmethod
    def save(self, decision: DCADecision) -> str:
        """
        Save decision with upsert semantics.
        
        Unique constraint: (user_id, symbol, decision_date, strategy_hash)
        If exists: update
        If not: insert
        
        Returns: decision_id
        """
        pass
    
    @abstractmethod
    def get_by_date(
        self, 
        symbol: str, 
        decision_date: MarketDate,
    ) -> Optional[DCADecision]:
        """Get decision for symbol on date."""
        pass
    
    @abstractmethod
    def get_history(
        self, 
        symbol: str, 
        limit: int = 12,
    ) -> List[DCADecision]:
        """Get recent decision history."""
        pass
    
    @abstractmethod
    def exists(
        self,
        symbol: str,
        decision_date: MarketDate,
        strategy_hash: str,
    ) -> bool:
        """Check if decision already exists (for idempotency)."""
        pass
```

### Domain Services (P1 Patched — Input Validation)

```python
# src/domain/services/dca_calculator.py

from typing import Tuple
from ..value_objects import Multiplier
from ..entities import Regime


class DCACalculator:
    """
    Pure domain logic for DCA multiplier calculation.
    
    P1 FIX: All inputs validated at entry.
    """
    
    def calculate_multiplier(
        self,
        regime: Regime,
        kelly_fraction: float,
        fatigue_score: float,
        signal_overlay: float,
        prob_dd_30: float,
    ) -> Tuple[Multiplier, Tuple[str, ...]]:
        """
        Calculate DCA multiplier with reasoning chain.
        
        Args:
            regime: Current market regime
            kelly_fraction: Kelly fraction [0, 1]
            fatigue_score: Regime fatigue [0, 1]
            signal_overlay: Signal multiplier [0.5, 1.5]
            prob_dd_30: P(DD > 30%) [0, 1]
        
        Returns:
            (Multiplier, Tuple of reasoning strings)
        
        Raises:
            ValueError: If inputs out of expected bounds
        """
        # P1 FIX: Validate inputs
        self._validate_inputs(kelly_fraction, fatigue_score, signal_overlay, prob_dd_30)
        
        reasoning = []
        
        # Step 1: Kelly-based sizing
        kelly_mult = min(1.0, max(0.3, kelly_fraction * 4))
        reasoning.append(f"Kelly fraction {kelly_fraction:.0%} → {kelly_mult:.2f}x")
        
        # Step 2: Regime adjustment
        if regime.is_crisis:
            base_mult = min(kelly_mult, Multiplier.CRISIS_CAP)
            reasoning.append(f"Crisis regime → capped at {Multiplier.CRISIS_CAP}x")
        else:
            base_mult = kelly_mult
        
        # Step 3: Fatigue penalty (up to -40%)
        fatigue_penalty = 1.0 - min(0.4, fatigue_score * 0.4)
        base_mult *= fatigue_penalty
        reasoning.append(f"Fatigue {fatigue_score:.0%} → penalty {fatigue_penalty:.2f}x")
        
        # Step 4: Signal overlay
        base_mult *= signal_overlay
        reasoning.append(f"Signal overlay → {signal_overlay:.2f}x")
        
        # Step 5: Drawdown guard
        if prob_dd_30 > 0.5:
            dd_penalty = 0.7
            base_mult *= dd_penalty
            reasoning.append(f"High DD risk ({prob_dd_30:.0%}) → penalty {dd_penalty:.2f}x")
        
        # Final clamp
        if regime.is_crisis:
            multiplier = Multiplier.crisis_capped(base_mult)
        else:
            multiplier = Multiplier.clamped(base_mult)
        
        reasoning.append(f"Final multiplier: {multiplier.value:.2f}x")
        
        return multiplier, tuple(reasoning)  # P0 FIX: Return tuple, not list
    
    def _validate_inputs(
        self,
        kelly_fraction: float,
        fatigue_score: float,
        signal_overlay: float,
        prob_dd_30: float,
    ) -> None:
        """P1 FIX: Fail fast on invalid engine outputs."""
        
        if not 0 <= kelly_fraction <= 1:
            raise ValueError(f"kelly_fraction must be in [0, 1], got {kelly_fraction}")
        
        if not 0 <= fatigue_score <= 1:
            raise ValueError(f"fatigue_score must be in [0, 1], got {fatigue_score}")
        
        if not 0.5 <= signal_overlay <= 1.5:
            raise ValueError(f"signal_overlay must be in [0.5, 1.5], got {signal_overlay}")
        
        if not 0 <= prob_dd_30 <= 1:
            raise ValueError(f"prob_dd_30 must be in [0, 1], got {prob_dd_30}")
```

---

## 🔴 Infrastructure Layer (Adapters) — PANDAS LIVES HERE

Infrastructure adapters convert between external world (pandas, yfinance) and domain types.

### Schema Validator (P0 Fix)

```python
# src/infrastructure/data_providers/schema_validator.py

import pandas as pd
from typing import Literal

# Canonical schema: snake_case everywhere
CANONICAL_COLUMNS = ['open', 'high', 'low', 'close', 'adj_close', 'volume']

YAHOO_TO_CANONICAL = {
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Adj Close': 'adj_close',
    'Volume': 'volume',
}


class SchemaValidator:
    """
    Enforces canonical snake_case schema at adapter boundary.
    
    P0 FIX: Single source of truth for column names.
    """
    
    @staticmethod
    def normalize(df: pd.DataFrame, source: Literal["yahoo", "stooq"]) -> pd.DataFrame:
        """Convert any source schema to canonical snake_case."""
        
        if source == "yahoo":
            # Rename Yahoo columns
            df = df.rename(columns=YAHOO_TO_CANONICAL)
        elif source == "stooq":
            # Stooq already uses similar names
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
            })
        
        # Validate required columns exist
        missing = set(CANONICAL_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        return df[CANONICAL_COLUMNS]  # Return only canonical columns in order
```

### Yahoo Provider (P0 Patched)

```python
# src/infrastructure/data_providers/yahoo_provider.py

import yfinance as yf
import pandas as pd
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from src.domain.interfaces import IMarketDataProvider
from src.domain.entities import MarketData, MarketDataMeta, PriceBar
from .schema_validator import SchemaValidator


class YahooDataProvider(IMarketDataProvider):
    """
    yfinance implementation of market data port.
    
    P0 FIXES:
    - Returns domain MarketData, not DataFrame
    - Guarantees adj_close or sets fallback flags
    - Uses canonical snake_case schema
    """
    
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = cache_dir
    
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> MarketData:
        """Fetch OHLCV from Yahoo Finance, return domain type."""
        
        # Fetch with yfinance
        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            actions=True,  # Include dividends/splits for adj_close
            progress=False,
        )
        
        if df.empty:
            raise ValueError(f"No data returned for {symbol}")
        
        # Normalize schema
        df = SchemaValidator.normalize(df, source="yahoo")
        
        # Determine trust level based on adj_close availability
        if 'adj_close' in df.columns and not df['adj_close'].isna().all():
            returns_basis = "adj_close"
            corporate_actions_applied = True
            trust_level = "high"
        else:
            # Fallback: use close, mark as unadjusted
            df['adj_close'] = df['close']
            returns_basis = "close_unadjusted"
            corporate_actions_applied = False
            trust_level = "lower"
        
        # Convert to domain types
        bars = tuple(
            PriceBar(
                date=idx.date() if hasattr(idx, 'date') else idx,
                open=Decimal(str(row['open'])),
                high=Decimal(str(row['high'])),
                low=Decimal(str(row['low'])),
                close=Decimal(str(row['close'])),
                adj_close=Decimal(str(row['adj_close'])),
                volume=int(row['volume']),
            )
            for idx, row in df.iterrows()
        )
        
        meta = MarketDataMeta(
            symbol=symbol,
            source="yahoo",
            asof_utc=datetime.utcnow(),
            timezone="America/New_York",
            currency="USD",
            corporate_actions_applied=corporate_actions_applied,
            returns_basis=returns_basis,
            trust_level=trust_level,
        )
        
        return MarketData(bars=bars, meta=meta)
```

### VIX Provider (P0 Patched — Correct Freshness)

```python
# src/infrastructure/data_providers/vix_provider.py

import yfinance as yf
from datetime import date, datetime
from typing import Optional, Tuple

from src.domain.interfaces import IMacroDataProvider, VIXResult


class VIXProvider(IMacroDataProvider):
    """
    VIX provider with correct freshness flagging.
    
    P0 FIX:
    - Freshness window is 2 trading days, not 7
    - Returns is_stale correctly
    """
    
    FRESHNESS_DAYS = 2  # Trading days
    DEFAULT_VIX = 20.0
    
    def __init__(self):
        self._cache: Optional[Tuple[float, date]] = None
    
    def get_vix(self) -> VIXResult:
        """Get VIX with proper freshness flags."""
        
        # Try fresh fetch
        try:
            vix_ticker = yf.Ticker("^VIX")
            hist = vix_ticker.history(period="5d")
            
            if not hist.empty:
                vix_value = float(hist["Close"].iloc[-1])
                vix_date = hist.index[-1].date()
                
                # Update cache
                self._cache = (vix_value, vix_date)
                
                # Check freshness (trading days, not calendar)
                days_old = self._trading_days_between(vix_date, date.today())
                is_stale = days_old > self.FRESHNESS_DAYS
                
                return VIXResult(
                    value=vix_value,
                    as_of_date=vix_date,
                    is_stale=is_stale,
                    is_fallback=False,
                    source="yahoo",
                )
        except Exception:
            pass
        
        # Try cache
        if self._cache:
            cached_vix, cached_date = self._cache
            days_old = self._trading_days_between(cached_date, date.today())
            
            return VIXResult(
                value=cached_vix,
                as_of_date=cached_date,
                is_stale=days_old > self.FRESHNESS_DAYS,
                is_fallback=False,
                source="cached",
            )
        
        # Ultimate fallback
        return VIXResult(
            value=self.DEFAULT_VIX,
            as_of_date=date.today(),
            is_stale=True,
            is_fallback=True,
            source="default",
        )
    
    def _trading_days_between(self, d1: date, d2: date) -> int:
        """Approximate trading days (exclude weekends)."""
        total_days = (d2 - d1).days
        weeks = total_days // 7
        remainder = total_days % 7
        trading = weeks * 5 + min(remainder, 5)
        return max(0, trading)
```

### Decision Repository (P1 Patched — Upsert + Auth)

```python
# src/infrastructure/persistence/supabase/decision_repo.py

from datetime import date
from typing import List, Optional
import json
from decimal import Decimal

from supabase import Client
from src.domain.interfaces import IDecisionRepository
from src.domain.entities import DCADecision
from src.domain.value_objects import Ticker, Money, Multiplier, MarketDate
from src.infrastructure.auth import UserContext


class SupabaseDecisionRepository(IDecisionRepository):
    """
    Supabase implementation of decision repository.
    
    P1 FIXES:
    - Upsert semantics (unique constraint aware)
    - user_id from auth context, not settings
    """
    
    def __init__(self, client: Client, user_context: UserContext):
        self.client = client
        self.user_id = user_context.get_user_id()  # P1 FIX: From auth, not settings
        self.table = "dca_decisions"
    
    def save(self, decision: DCADecision) -> str:
        """
        Save decision with upsert semantics.
        
        Unique constraint: (user_id, symbol, decision_date, strategy_hash)
        """
        
        data = {
            "user_id": self.user_id,
            "symbol": decision.symbol.value,
            "decision_date": decision.decision_date.value.isoformat(),
            "asof_utc": decision.asof_utc,
            "base_amount": float(decision.base_amount.amount),
            "adjusted_amount": float(decision.adjusted_amount.amount),
            "multiplier": decision.multiplier.value,
            "regime": decision.regime,
            "reasoning_json": json.dumps(decision.reasoning),
            "run_id": decision.run_id,
            "snapshot_id": decision.snapshot_id,
            "strategy_hash": decision.strategy_hash,
            "engine_version": decision.engine_version,
        }
        
        # Upsert: update if exists, insert if not
        result = (
            self.client.table(self.table)
            .upsert(
                data,
                on_conflict="user_id,symbol,decision_date,strategy_hash",
            )
            .execute()
        )
        
        return result.data[0]["id"]
    
    def exists(
        self,
        symbol: str,
        decision_date: MarketDate,
        strategy_hash: str,
    ) -> bool:
        """Check if decision already exists (for idempotency)."""
        
        result = (
            self.client.table(self.table)
            .select("id")
            .eq("user_id", self.user_id)
            .eq("symbol", symbol)
            .eq("decision_date", decision_date.value.isoformat())
            .eq("strategy_hash", strategy_hash)
            .execute()
        )
        
        return len(result.data) > 0
    
    # ... other methods similar to before
```

---

## 🟡 Application Layer (Use Cases) — P0 Patched

Use cases are now cleaner: no unused dependencies, proper date handling.

```python
# src/application/use_cases/compute_dca_decision.py

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from src.domain.entities import DCADecision
from src.domain.value_objects import Ticker, Money, Multiplier, MarketDate
from src.domain.services import DCACalculator
from src.domain.interfaces import (
    IMarketDataProvider,
    IMacroDataProvider,
    IDecisionRepository,
    IArtifactStore,
)
from ..dto import DCARequest, DCAResponse
from ..services import CalendarService


@dataclass
class ComputeDCADecision:
    """
    Use case: Compute DCA decision for a single asset.
    
    P0 FIXES:
    - Removed unused regime_engine (light path uses cached artifacts)
    - Uses asof_market_date from last bar, not date.today()
    - Returns VIX freshness flags
    """
    
    # Injected dependencies
    market_data: IMarketDataProvider
    macro_data: IMacroDataProvider       # Separate VIX provider (ISP)
    decisions: IDecisionRepository
    artifacts: IArtifactStore
    calculator: DCACalculator
    calendar: CalendarService
    
    # Removed: regime_engine (not used in light path)
    
    def execute(self, request: DCARequest) -> DCAResponse:
        """
        Execute the use case.
        
        1. Fetch latest market data
        2. Load cached regime analysis (fail if missing)
        3. Compute signals
        4. Calculate DCA multiplier
        5. Persist decision
        6. Return response
        """
        
        # 1. Fetch data (returns domain type, not DataFrame)
        market = self.market_data.fetch_ohlcv(
            symbol=request.symbol,
            start_date=request.lookback_start,
            end_date=self.calendar.last_trading_day(),
        )
        
        # P0 FIX: Use market date from last completed bar
        decision_date = market.asof_market_date
        
        # 2. Save snapshot (immutable)
        snapshot_id = self.artifacts.save_snapshot(request.symbol, market)
        
        # 3. Load cached heavy analysis
        analysis = self.artifacts.load_latest_analysis(request.symbol)
        if analysis is None:
            raise ValueError(
                f"No cached analysis for {request.symbol}. "
                "Run heavy analysis first or use RunHeavyAnalysis use case."
            )
        
        # 4. Get VIX with freshness
        vix_result = self.macro_data.get_vix()
        
        # 5. Compute signal overlay
        signal_overlay = self._compute_signal_overlay(market, analysis)
        
        # 6. Calculate multiplier (domain service)
        multiplier, reasoning = self.calculator.calculate_multiplier(
            regime=analysis.regime,
            kelly_fraction=analysis.kelly_fraction,
            fatigue_score=analysis.fatigue_score,
            signal_overlay=signal_overlay,
            prob_dd_30=analysis.prob_dd_30,
        )
        
        # 7. Create decision entity
        decision = DCADecision(
            symbol=Ticker(request.symbol),
            decision_date=decision_date,
            asof_utc=datetime.utcnow().isoformat(),
            base_amount=Money(Decimal(str(request.base_amount)), "USD"),
            adjusted_amount=Money(
                Decimal(str(request.base_amount)) * Decimal(str(multiplier.value)),
                "USD",
            ),
            multiplier=multiplier,
            regime=analysis.regime,
            reasoning=reasoning,  # Already tuple from domain service
            run_id=request.run_id,
            snapshot_id=snapshot_id,
            strategy_hash=request.strategy_hash,
            engine_version=analysis.engine_version,
        )
        
        # 8. Persist (upsert)
        decision_id = self.decisions.save(decision)
        
        # 9. Return response with all flags
        return DCAResponse(
            decision_id=decision_id,
            symbol=request.symbol,
            base_amount=request.base_amount,
            adjusted_amount=float(decision.adjusted_amount.amount),
            multiplier=multiplier.value,
            regime=analysis.regime,
            reasoning=list(reasoning),  # Convert to list for JSON
            vix_value=vix_result.value,
            vix_is_stale=vix_result.is_stale,
            vix_is_fallback=vix_result.is_fallback,
            data_trust_level=market.meta.trust_level,
            asof_market_date=decision_date.value.isoformat(),
        )
    
    def _compute_signal_overlay(self, market, analysis) -> float:
        # Signal aggregation logic
        return 1.0  # Placeholder
```

---

## 🧪 Testing Strategy

### Domain Tests (Pure — No Mocks Needed)

```python
# tests/domain/test_multiplier.py

import pytest
from src.domain.value_objects import Multiplier

def test_multiplier_bounded():
    mult = Multiplier.clamped(2.0)
    assert mult.value == 1.5  # Clamped to max

def test_multiplier_crisis_cap():
    mult = Multiplier.crisis_capped(1.0)
    assert mult.value == 0.5  # Capped

def test_multiplier_invalid_raises():
    with pytest.raises(ValueError):
        Multiplier(value=2.0)  # Exceeds max
```

```python
# tests/domain/test_money.py

from decimal import Decimal
from src.domain.value_objects import Money

def test_money_uses_decimal():
    m = Money(100.123456)
    assert m.amount == Decimal("100.12")  # Quantized

def test_money_multiplication():
    m = Money(1000)
    result = m * 0.72
    assert result.amount == Decimal("720.00")
```

```python
# tests/domain/test_dca_calculator.py

from src.domain.services import DCACalculator
from src.domain.entities import Regime

def test_calculator_reasoning_is_tuple():
    calc = DCACalculator()
    mult, reasoning = calc.calculate_multiplier(
        regime=Regime("Normal"),
        kelly_fraction=0.15,
        fatigue_score=0.2,
        signal_overlay=1.0,
        prob_dd_30=0.22,
    )
    
    assert isinstance(reasoning, tuple)  # Not list
    assert 0.3 <= mult.value <= 1.5

def test_calculator_validates_inputs():
    calc = DCACalculator()
    
    with pytest.raises(ValueError) as exc:
        calc.calculate_multiplier(
            regime=Regime("Normal"),
            kelly_fraction=1.5,  # Invalid: > 1
            fatigue_score=0.2,
            signal_overlay=1.0,
            prob_dd_30=0.22,
        )
    
    assert "kelly_fraction" in str(exc.value)
```

---

## ✅ Architecture Checklist (Patched)

| Principle | Status | Implementation |
|-----------|--------|----------------|
| **Pure Domain** | ✅ Fixed | No pandas, uses own value types |
| **Immutability** | ✅ Fixed | Tuple for reasoning, Decimal for money |
| **Canonical Schema** | ✅ Fixed | snake_case everywhere, SchemaValidator |
| **Market Dates** | ✅ Fixed | MarketDate from last bar, not date.today() |
| **VIX Freshness** | ✅ Fixed | 2 trading days, proper flags |
| **Input Validation** | ✅ Fixed | Domain services validate bounds |
| **Upsert Semantics** | ✅ Fixed | Unique constraint + upsert |
| **Auth from Context** | ✅ Fixed | user_id from JWT, not settings |
| **Reproducibility** | ✅ Fixed | run_id, snapshot_id, strategy_hash, engine_version |

---

## 📝 LLM Feed: "Validated Clean Architecture Decisions"

> Implement Clean Architecture with a pure domain (no pandas/clients), using ports returning domain value types (MarketData with PriceBars + metadata). Infrastructure adapters (yfinance/Stooq/Supabase) convert external data into canonical snake_case schemas validated at boundaries. Decisions are reproducible and idempotent: every DCADecision stores run_id, snapshot_id (immutable parquet snapshot), strategy_hash, engine_version, and analysis_asof_date derived from last completed market bar (not date.today). VIX retrieval must return freshness flags correctly (2 trading days). Artifact store port must match use-case calls (save_snapshot/load_latest_analysis). Money uses Decimal; decision reasoning is immutable (tuple). Repos enforce unique constraints (user_id,symbol,decision_date,strategy_hash) with upsert. Monte Carlo outputs persist RNG seed and full config metadata for reproducibility; daily-light consumes cached heavy artifacts unless on-demand heavy is explicitly enabled.

---

## 🚀 Ready to Build

All P0 issues patched. Ready for implementation when you give the go-ahead! 🏗️
