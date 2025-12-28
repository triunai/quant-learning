import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from regime_engine import RegimeRiskEngine

st.set_page_config(layout="wide", page_title="Project PLTR - Risk Dashboard")
plt.style.use('dark_background')

# Sidebar
st.sidebar.header("Mission Control")
ticker = st.sidebar.text_input("Ticker Symbol", value="PLTR")
days_ahead = st.sidebar.slider("Days Ahead", 63, 252, 126)
simulations = st.sidebar.slider("Simulations", 1000, 10000, 5000)
enable_vix = st.sidebar.checkbox("Enable VIX Filter", value=True)
run_btn = st.sidebar.button("RUN SIMULATION")

@st.cache_data
def run_analysis(ticker, days_ahead, simulations, enable_vix):
    # Initialize Engine
    engine = RegimeRiskEngine(
        ticker=ticker,
        days_ahead=days_ahead,
        simulations=simulations,
        enable_vix=enable_vix,
    )
    engine.ingest_data()
    results, signal = engine.run(plot=False)

    return engine, results, signal

if run_btn:
    with st.spinner(f"Running v7.0 Engine on {ticker}..."):
        try:
            engine, results, signal = run_analysis(ticker, days_ahead, simulations, enable_vix)

            # Header
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.metric("Ticker", ticker)
            with col2:
                st.metric("Current Price", f"${engine.last_price:.2f}")
            with col3:
                # Format signal from engine
                jj_signal = f"{signal['signal']} ({signal['confidence']}%)"
                st.metric("Jjules Signal", jj_signal)

            st.divider()

            # Metrics
            m1, m2, m3 = st.columns(3)
            with m1:
                st.subheader("Regime & Drift")
                st.info(f"Regime: {engine.state_map[engine.current_state]}")
                st.write(f"Implied Drift: {engine.implied_daily_drift*100:.3f}%")

            with m2:
                st.subheader("Target Up Prob")
                prob_up = results['prob_up']
                st.success(f"{prob_up:.1%}")
                st.caption(f"Target: ${engine.target_up:.0f}")

            with m3:
                st.subheader("Kelly & Risk")
                prob_down = results['prob_down']
                st.error(f"Downside Risk: {prob_down:.1%}")
                st.caption(f"Target: ${engine.target_down:.0f}")

                # Simple Kelly calc approximation based on confidence
                conf = signal['confidence'] / 100.0
                kelly_est = (conf * 2) - 1 # Rough mapping from confidence to Kelly-like exposure
                st.write(f"Confidence: {signal['confidence']}%")

            st.divider()

            # Tabs
            tab1, tab2 = st.tabs(["Visuals", "Deep Data"])

            with tab1:
                # Re-implement plotting logic
                fig = plt.figure(figsize=(18, 10))
                gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])

                paths = results['paths']
                times_up = results['times_up']
                times_down = results['times_down']

                # 1. Cone Chart
                ax1 = fig.add_subplot(gs[0, :])
                x = np.arange(engine.days_ahead)

                # Handle shapes: paths might include day 0
                if paths.shape[1] == engine.days_ahead + 1:
                    x = np.arange(engine.days_ahead + 1)

                p5 = np.percentile(paths, 5, axis=0)
                p25 = np.percentile(paths, 25, axis=0)
                p50 = np.median(paths, axis=0)
                p75 = np.percentile(paths, 75, axis=0)
                p95 = np.percentile(paths, 95, axis=0)

                ax1.fill_between(x, p5, p95, color='cyan', alpha=0.1, label='90%')
                ax1.fill_between(x, p25, p75, color='cyan', alpha=0.2, label='50%')
                ax1.plot(x, p50, color='cyan', lw=2.5, label='Median')

                ax1.axhline(engine.target_up, color='lime', ls=':', lw=2, label=f'Up ${engine.target_up:.0f}')
                ax1.axhline(engine.target_down, color='red', ls=':', lw=2, label=f'Down ${engine.target_down:.0f}')
                ax1.axhline(engine.last_price, color='white', lw=1, alpha=0.5)

                sig_text = signal['signal']
                ax1.set_title(f"Cone Chart - {sig_text}", color='white')
                ax1.legend(loc='upper left')
                ax1.grid(alpha=0.2)

                # 2. Time to Up
                ax2 = fig.add_subplot(gs[1, 0])
                if len(times_up) > 20:
                    sns.histplot(times_up, color='lime', ax=ax2, bins=30, alpha=0.7)
                else:
                    ax2.text(0.5, 0.5, "Rare", ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title("Time to Up")

                # 3. Time to Down
                ax3 = fig.add_subplot(gs[1, 1])
                if len(times_down) > 20:
                    sns.histplot(times_down, color='red', ax=ax3, bins=30, alpha=0.7)
                else:
                    ax3.text(0.5, 0.5, "Rare", ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title("Time to Down")

                # 4. Matrix
                ax4 = fig.add_subplot(gs[1, 2])
                labels = [s[:4] for s in engine.state_map.values()]
                sns.heatmap(engine.markov_matrix, annot=True, fmt='.2f', cmap='magma',
                            xticklabels=labels, yticklabels=labels, ax=ax4)
                ax4.set_title("Transition Matrix")

                st.pyplot(fig)

            with tab2:
                st.subheader("Convexity")
                st.json(results['convexity'])

                st.subheader("Sanity Check")
                st.json(results['sanity'])

                st.subheader("Signal Logic")
                st.write(signal)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)

else:
    st.info("Configure parameters in the sidebar and click RUN SIMULATION to start.")
