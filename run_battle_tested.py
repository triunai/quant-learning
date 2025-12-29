"""Quick runner for battle-tested RegimeRiskPlatform to show ACTUAL terminal output."""
import sys
import os

# Add battle-tested to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'battle-tested'))

# Import the platform
import importlib.util
spec = importlib.util.spec_from_file_location("pltr", os.path.join(os.path.dirname(__file__), 'battle-tested', 'PLTR-test-2.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Run it
if __name__ == "__main__":
    platform = mod.RegimeRiskPlatform(
        ticker="SPY",
        market_ticker="QQQ",
        days_ahead=126,
        simulations=1000,
        n_regimes=3
    )
    platform.ingest_data()
    results = platform.run(run_full_calibration=True)
