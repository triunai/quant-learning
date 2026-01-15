"""
Risk Engine Runner Script
"""
import sys
import matplotlib.pyplot as plt
from refinery.regime_engine import RegimeRiskEngine

# Tee implementation for logging to file and console
class Tee(object):
    def __init__(self, file_stream):
        self.file = file_stream
        self.stdout = sys.stdout
        sys.stdout = self
    def close(self):
        if self.stdout is not None:
            sys.stdout = self.stdout
            self.stdout = None
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()

def main():
    try:
        # Open file with context manager
        with open('risk_summary.txt', 'w') as f:
            # Capture output
            tee = Tee(f)
            try:
                # Standard PLTR analysis
                platform = RegimeRiskEngine(
                    ticker="PLTR",
                    market_ticker="QQQ",
                    target_up=280,
                    target_down=120,
                    stop_loss_pct=0.15,
                    days_ahead=126,
                    simulations=5000,
                    n_regimes=3
                )

                platform.ingest_data()

                # Run execution
                # Note: run returns (results, signal, fig)
                results, signal, fig = platform.run(plot=True)

                # Explicitly print signal for visibility (and tests)
                print(f"\nFinal Signal: {signal['signal']} ({signal['confidence']}%)")

                if fig:
                    fig.savefig('pltr_risk.png')
                    print("[SAVED] pltr_risk.png")

                # Run diagnostic to understand calibration
                print("\n" + "="*70)
                print("CALIBRATION DIAGNOSTICS")
                print("="*70)
                platform.diagnose_under_prediction()

                # Run multi-target walk-forward
                print("\n" + "="*70)
                print("MULTI-TARGET WALK-FORWARD VALIDATION")
                print("="*70)
                for target in [0.10, 0.20, 0.30, 0.50]:
                    platform.walk_forward_validation_fixed(n_folds=8, target_pct=target)

                # CRITICAL TEST: What if we were in Momentum regime?
                platform.what_if_momentum_regime()

            finally:
                tee.close()

    except ImportError:
        print("Critical dependency missing")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
