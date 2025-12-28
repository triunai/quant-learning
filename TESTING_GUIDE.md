# Testing Guide for Quant-Learning Project

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=refinery --cov=run_risk --cov-report=html
```

## What's Been Tested

### ✅ New Modules (Current Branch)
- **run_risk.py** - Entry point script (8 tests)
- **refinery/market_noise.py** - Sentiment analysis (36 tests)
- **refinery/regime_engine_v7.py** - Engine wrapper (50+ tests)
- **refinery/integration_snippet.py** - Logic validation (10 tests)

### 📝 Recommended Next: Battle-Tested Module
The main `battle-tested/PLTR-test-2.py` (1086 lines) would benefit from:
- GMM clustering tests
- Semi-Markov duration tests
- Jump diffusion simulation tests
- Calibration suite tests

## Test Organization