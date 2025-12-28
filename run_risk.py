
import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

try:
    from refinery.regime_engine import RegimeRiskEngine
except ImportError as e:
    print(f"Error importing RegimeRiskEngine: {e}")
    sys.exit(1)

def main():
    print("Initializing Regime Risk Engine for PLTR...")
    engine = RegimeRiskEngine(ticker="PLTR")

    print("Ingesting data...")
    engine.ingest_data()

    print("Running simulation...")
    # run returns: results, signal, fig
    try:
        results, signal, fig = engine.run(plot=True)
    except (ValueError, TypeError) as e:
        print(f"Error: engine.run() returned unexpected structure: {e}")
        sys.exit(1)

    # Save the plot
    if fig:
        print("Saving plot to pltr_risk.png...")
        fig.savefig("pltr_risk.png")
    else:
        print("Warning: No figure returned.")

    # Save summary
    print("Saving summary to risk_summary.txt...")
    with open("risk_summary.txt", "w") as f:
        signal_value = signal.get('signal', 'UNKNOWN')
        confidence = signal.get('confidence', 0)
        f.write(f"Signal: {signal_value} ({confidence}%)\n")
        f.write("Reasoning:\n")
        for r in signal.get('reasoning', []):
            f.write(f"- {r}\n")
        f.write("\nStats:\n")
        prob_up = results.get('prob_up', 0.0)
        prob_down = results.get('prob_down', 0.0)
        f.write(f"Prob Up: {prob_up:.1%}\n")
        f.write(f"Prob Down: {prob_down:.1%}\n")

    print("Done.")

if __name__ == "__main__":
    main()
