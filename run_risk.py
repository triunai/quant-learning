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
    """
    Run the PLTR RegimeRiskEngine workflow and persist results.
    
    Instantiates a RegimeRiskEngine for ticker "PLTR", ingests data, runs the simulation with plotting enabled, saves the returned figure to "pltr_risk.png" when present, and writes a plain-text summary to "risk_summary.txt" containing the signal, its confidence, the reasoning lines, and the Prob Up / Prob Down statistics.
    """
    print("Initializing Regime Risk Engine for PLTR...")
    engine = RegimeRiskEngine(ticker="PLTR")

    print("Ingesting data...")
    engine.ingest_data()

    print("Running simulation...")
    # run returns: results, signal, fig
    results, signal, fig = engine.run(plot=True)

    # Save the plot
    if fig:
        print("Saving plot to pltr_risk.png...")
        fig.savefig("pltr_risk.png")
    else:
        print("Warning: No figure returned.")

    # Save summary
    print("Saving summary to risk_summary.txt...")
    with open("risk_summary.txt", "w") as f:
        f.write(f"Signal: {signal['signal']} ({signal['confidence']}%)\n")
        f.write("Reasoning:\n")
        for r in signal['reasoning']:
            f.write(f"- {r}\n")
        f.write("\nStats:\n")
        f.write(f"Prob Up: {results['prob_up']:.1%}\n")
        f.write(f"Prob Down: {results['prob_down']:.1%}\n")

    print("Done.")

if __name__ == "__main__":
    main()