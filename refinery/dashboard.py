"""
PROJECT PLTR - STREAMLIT RISK DASHBOARD
========================================
Now powered by the v7.0 Regime Risk Platform (GMM + Alpha/Beta Factor Model)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import io
import base64
from datetime import datetime

# Import the v7.0 engine adapter
from regime_engine_v7 import RegimeRiskEngineV7

st.set_page_config(layout="wide", page_title="Project PLTR - v7.0 Risk Dashboard")
plt.style.use('dark_background')

# Sidebar
st.sidebar.header("Mission Control v7.0")
st.sidebar.caption("GMM Regime Clustering + Alpha/Beta Factor Model")
st.sidebar.divider()

ticker = st.sidebar.text_input("Ticker Symbol", value="PLTR")
market_ticker = st.sidebar.text_input("Market Benchmark", value="QQQ")
days_ahead = st.sidebar.slider("Days Ahead", 63, 252, 126)
simulations = st.sidebar.slider("Simulations", 1000, 10000, 5000)
n_regimes = st.sidebar.selectbox("Number of Regimes", [2, 3, 4], index=1)
enable_vix = st.sidebar.checkbox("Enable VIX Filter", value=True)
run_full_calibration = st.sidebar.checkbox("Run Full Calibration Suite", value=False)
run_btn = st.sidebar.button("RUN SIMULATION", type="primary")


@st.cache_resource(show_spinner=False)
def run_analysis(ticker, market_ticker, days_ahead, simulations, n_regimes, enable_vix, run_full_calibration):
    """Run the v7.0 platform and return results."""
    engine = RegimeRiskEngineV7(
        ticker=ticker,
        market_ticker=market_ticker,
        days_ahead=days_ahead,
        simulations=simulations,
        n_regimes=n_regimes,
        enable_vix=enable_vix,
    )
    engine.ingest_data()
    results, signal = engine.run(plot=False, run_full_calibration=run_full_calibration)
    return engine, results, signal


if run_btn:
    with st.spinner(f"Running v7.0 Platform on {ticker}..."):
        try:
            engine, results, signal = run_analysis(
                ticker, market_ticker, days_ahead, simulations, 
                n_regimes, enable_vix, run_full_calibration
            )

            # Header Metrics
            st.title(f"🦅 {ticker} - Regime Risk Platform v7.0")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Price", f"${engine.last_price:.2f}")
            with col2:
                regime_name = engine.regime_names.get(engine.current_regime, "Unknown")
                st.metric("Regime", regime_name)
            with col3:
                st.metric("Beta", f"{engine.market_beta:.2f}")
            with col4:
                sig_color = "green" if signal['signal'] == 'LONG' else "red" if signal['signal'] == 'SHORT' else "gray"
                st.metric("Signal", f"{signal['signal']} ({signal['confidence']}%)")

            st.divider()

            # Risk Dashboard
            m1, m2, m3 = st.columns(3)
            with m1:
                st.subheader("📈 Upside Probability")
                prob_up = results['prob_up']
                st.success(f"{prob_up:.1%}")
                st.caption(f"Target: ${engine.target_up:.0f}")
                st.caption(f"Historical: {results['sanity']['hist_up_hit']:.1%}")

            with m2:
                st.subheader("📉 Downside Probability")
                prob_down = results['prob_down']
                st.error(f"{prob_down:.1%}")
                st.caption(f"Target: ${engine.target_down:.0f}")
                st.caption(f"Historical: {results['sanity']['hist_down_hit']:.1%}")

            with m3:
                st.subheader("⚠️ Risk Metrics")
                risk = results['risk']
                st.write(f"**VaR(95):** {risk['var_95']*100:+.1f}%")
                st.write(f"**CVaR(95):** {risk['cvar_95']*100:+.1f}%")
                st.write(f"**P(MaxDD > 30%):** {risk['prob_dd_30']:.1%}")
                st.write(f"**Kelly Fraction:** {risk['kelly_fraction']:.0%}")

            st.divider()

            # Tabs for Visuals and Deep Data
            tab1, tab2, tab3 = st.tabs(["📊 Visuals", "🔬 Deep Data", "📋 Export"])

            with tab1:
                fig = plt.figure(figsize=(18, 10))
                gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])

                paths = results['paths']

                # 1. Cone Chart
                ax1 = fig.add_subplot(gs[0, :])
                x = np.arange(paths.shape[1])

                p5 = np.percentile(paths, 5, axis=0)
                p25 = np.percentile(paths, 25, axis=0)
                p50 = np.median(paths, axis=0)
                p75 = np.percentile(paths, 75, axis=0)
                p95 = np.percentile(paths, 95, axis=0)

                ax1.fill_between(x, p5, p95, color='cyan', alpha=0.1, label='90%')
                ax1.fill_between(x, p25, p75, color='cyan', alpha=0.2, label='50%')
                ax1.plot(x, p50, color='cyan', lw=2.5, label='Median')

                ax1.axhline(engine.target_up, color='lime', ls=':', lw=2, label=f'Up ${engine.target_up:.0f} ({prob_up:.0%})')
                ax1.axhline(engine.target_down, color='red', ls=':', lw=2, label=f'Down ${engine.target_down:.0f} ({prob_down:.0%})')
                ax1.axhline(engine.last_price, color='white', lw=1, alpha=0.5)

                sig_colors = {'LONG': 'lime', 'SHORT': 'red', 'NEUTRAL': 'yellow', 'CASH': 'gray'}
                title_color = sig_colors.get(signal['signal'], 'white')
                ax1.set_title(f"{ticker} v7.0 | {signal['signal']} | {regime_name} | Beta={engine.market_beta:.1f}",
                              fontsize=14, color=title_color, fontweight='bold')
                ax1.legend(loc='upper left')
                ax1.grid(alpha=0.2)
                ax1.set_xlabel("Days")
                ax1.set_ylabel("Price")

                # 2. Drawdown Distribution
                ax2 = fig.add_subplot(gs[1, 0])
                sns.histplot(risk['max_drawdowns'] * 100, color='red', ax=ax2, bins=50, alpha=0.7)
                ax2.axvline(-20, color='orange', ls='--', label='-20%')
                ax2.axvline(-30, color='red', ls='--', label='-30%')
                ax2.set_title("Max Drawdown Distribution")
                ax2.set_xlabel("Drawdown %")
                ax2.legend()

                # 3. Final Price Distribution
                ax3 = fig.add_subplot(gs[1, 1])
                final = paths[:, -1]
                sns.histplot(final, color='cyan', ax=ax3, bins=50, alpha=0.7)
                ax3.axvline(engine.last_price, color='white', ls='-')
                ax3.axvline(np.median(final), color='yellow', ls='--')
                ax3.set_title("Final Price Distribution")

                # 4. Transition Matrix (Exit Matrix)
                ax4 = fig.add_subplot(gs[1, 2])
                if engine.transition_matrix is not None:
                    labels = [engine.regime_names.get(i, f"R{i}")[:6] for i in range(len(engine.transition_matrix))]
                    # Plot EXIT matrix (what simulator actually uses)
                    exit_matrix = engine.transition_matrix.copy()
                    np.fill_diagonal(exit_matrix, 0)
                    row_sums = exit_matrix.sum(axis=1, keepdims=True)
                    exit_matrix = np.where(row_sums > 0, exit_matrix / row_sums, exit_matrix)
                    sns.heatmap(exit_matrix, annot=True, fmt='.2f', cmap='magma',
                                xticklabels=labels, yticklabels=labels, ax=ax4)
                    ax4.set_title("Exit Transitions (sim)")
                else:
                    ax4.text(0.5, 0.5, "No Matrix", ha='center', va='center', transform=ax4.transAxes)

                st.pyplot(fig)

            with tab2:
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.subheader("🧬 Regime Diagnostics")
                    if 'regime_diagnostics' in results:
                        for rd in results['regime_diagnostics']:
                            st.markdown(f"""
                            **{rd['name']}** (n={rd['n_samples']})
                            - Avg Drawdown: `{rd['avg_dd']:.1%}`
                            - Avg Vol: `{rd['avg_vol']:.1%}`
                            - Drift (ann): `{rd['mu']:.1%}`
                            - Alpha (ann): `{rd['alpha']:+.1%}`
                            - Avg Duration: `{rd['avg_duration']:.1f} days`
                            """)
                            st.divider()
                
                with col_b:
                    st.subheader("🌍 Macro Conditioning")
                    if 'macro' in results:
                        macro = results['macro']
                        st.write(f"**Market Beta:** {macro['beta']:.2f}")
                        st.write(f"**Alpha (ann):** {macro['alpha_ann']:+.1%}")
                        st.write(f"**Idiosyncratic Vol:** {macro['idio_vol']:.1%}")
                        st.write(f"**VIX Level:** {macro['vix']:.1f}")
                        st.write(f"**GARCH Vol:** {macro['garch_vol']:.1%}")
                        st.write(f"**Realized Vol:** {macro['realized_vol']:.1%}")
                
                st.divider()
                
                st.subheader("📋 Signal Reasoning")
                for reason in signal['reasoning']:
                    st.write(f"- {reason}")

                st.subheader("⚠️ Risk Breakdown")
                st.json({
                    'VaR(95)': f"{risk['var_95']*100:+.1f}%",
                    'CVaR(95)': f"{risk['cvar_95']*100:+.1f}%",
                    'P(MaxDD > 20%)': f"{risk['prob_dd_20']:.1%}",
                    'P(MaxDD > 30%)': f"{risk['prob_dd_30']:.1%}",
                    'P(Stop Breach)': f"{risk['prob_stop']:.1%}",
                    'Kelly Fraction': f"{risk['kelly_fraction']:.0%}",
                    'Win Rate': f"{risk['win_rate']:.1%}"
                })

                st.subheader("✅ Historical Validation")
                st.write(f"**Up Target Hit Rate (Hist):** {results['sanity']['hist_up_hit']:.1%}")
                st.write(f"**Down Target Hit Rate (Hist):** {results['sanity']['hist_down_hit']:.1%}")

            with tab3:
                st.subheader("📋 Copy Full Report to Clipboard")
                st.caption("Click the copy icon in the top-right corner of the code block below.")
                
                # Build comprehensive report JSON
                report = {
                    "timestamp": datetime.now().isoformat(),
                    "ticker": ticker,
                    "market_ticker": market_ticker,
                    "price": engine.last_price,
                    "signal": signal['signal'],
                    "confidence": signal['confidence'],
                    "reasoning": signal['reasoning'],
                    "regime": {
                        "current": engine.regime_names.get(engine.current_regime, "Unknown"),
                        "diagnostics": results.get('regime_diagnostics', [])
                    },
                    "macro": results.get('macro', {}),
                    "targets": {
                        "up": {"price": engine.target_up, "prob_sim": results['prob_up'], "prob_hist": results['sanity']['hist_up_hit']},
                        "down": {"price": engine.target_down, "prob_sim": results['prob_down'], "prob_hist": results['sanity']['hist_down_hit']}
                    },
                    "risk": {
                        "var_95": risk['var_95'],
                        "cvar_95": risk['cvar_95'],
                        "prob_dd_20": risk['prob_dd_20'],
                        "prob_dd_30": risk['prob_dd_30'],
                        "prob_stop": risk['prob_stop'],
                        "kelly_fraction": risk['kelly_fraction'],
                        "win_rate": risk['win_rate']
                    },
                    "distribution": {
                        "p5": float(np.percentile(paths[:, -1], 5)),
                        "median": float(np.median(paths[:, -1])),
                        "p95": float(np.percentile(paths[:, -1], 95))
                    }
                }
                
                # Display as copyable code block
                report_json = json.dumps(report, indent=2, default=str)
                st.code(report_json, language="json")
                
                st.divider()
                
                # Download chart as PNG
                st.subheader("📥 Download Chart")
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0e1117')
                buf.seek(0)
                st.download_button(
                    label="Download Chart as PNG",
                    data=buf,
                    file_name=f"{ticker}_v7_report_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
                    mime="image/png"
                )

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)

else:
    st.info("Configure parameters in the sidebar and click **RUN SIMULATION** to start.")
    st.markdown("""
    ### 🦅 Project PLTR - v7.0 Features
    
    This dashboard is now powered by the **v7.0 Regime Risk Platform**:
    
    *   **GMM Clustering**: Real regimes from slow features (Vol, Trend, Drawdown), not return buckets.
    *   **Alpha/Beta Factor Model**: `r = alpha_regime + beta * r_market + epsilon`
    *   **Semi-Markov Duration**: Captures regime persistence with forward recurrence.
    *   **DD-Aware Kelly**: Position sizing capped by max drawdown probability.
    *   **Invariant Validation**: Ensures sim matches historical stats (Mean, Std, Skew, Kurtosis).
    """)
