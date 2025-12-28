"""
REGIME RISK ENGINE v3.0 - THE TRIAD ARCHITECTURE
=================================================
A complete Quant Stack with three detection layers:

1. THE CORE (Markov):       "What is the probability of each regime?"
2. THE CHECK (VIX Filter):  "Is the macro weather safe for flying?"
3. THE GUARDRAIL (Anomaly): "Are we flying in known airspace?"

Optional: GARCH volatility forecasting for dynamic cones.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import sklearn for Anomaly Detection
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: sklearn not installed. Anomaly detection disabled.")

# Try to import arch for GARCH
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False


plt.style.use('dark_background')
sns.set_palette("plasma")


class TriadRiskEngine:
    """
    The Triad Architecture: Three Detection Layers

    Layer 1 - MARKOV CORE:     Regime transition probabilities
    Layer 2 - VIX FILTER:      Macro fear overlay
    Layer 3 - ANOMALY ENGINE:  Out-of-distribution detection

    Optional: GARCH for dynamic volatility forecasting
    """

    def __init__(self, ticker, days_ahead=126, simulations=5000,
                 target_up=None, target_down=None):
        self.ticker = ticker
        self.days_ahead = days_ahead
        self.simulations = simulations

        # FIXED: Separate upside and downside targets
        self.target_up = target_up      # Moonshot target
        self.target_down = target_down  # Stress test target

        self.data = None
        self.markov_matrix = None
        self.markov_matrix_raw = None  # Before VIX adjustment
        self.mu = 0
        self.sigma = 0
        self.last_price = 0
        self.last_date = None
        self.current_state = 0

        self.n_states = 5
        self.state_map = {0: "Crash", 1: "Bear", 2: "Consolidation", 3: "Bull", 4: "Rally"}

        # Triad Status
        self.vix_level = 0
        self.vix_alert = False
        self.is_anomaly = False
        self.anomaly_score = 0
        self.garch_forecast = None

    # =========================================================================
    # LAYER 0: DATA INGESTION
    # =========================================================================

    def ingest_data(self):
        print(f"[DATA] Loading {self.ticker}...")
        self.data = yf.download(self.ticker, period="5y", auto_adjust=True, progress=False)

        self.data['Log_Ret'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.data['Volume_Norm'] = self.data['Volume'] / self.data['Volume'].rolling(20).mean()
        self.data = self.data.dropna()

        self.last_price = self.data['Close'].iloc[-1].item()
        self.last_date = self.data.index[-1]

        self.mu = self.data['Log_Ret'].mean() * 252
        self.sigma = self.data['Log_Ret'].std() * np.sqrt(252)

        # Auto-set targets if not provided
        if self.target_up is None:
            self.target_up = self.last_price * 1.5  # +50% moonshot
        if self.target_down is None:
            self.target_down = self.last_price * 0.7  # -30% stress

        print(f"    Loaded {len(self.data)} days | Current: ${self.last_price:.2f}")

    # =========================================================================
    # LAYER 1: MARKOV CORE
    # =========================================================================

    def build_markov_core(self):
        print("[LAYER 1] Building Markov Core...")

        self.data['State_Idx'] = pd.qcut(self.data['Log_Ret'], q=self.n_states, labels=False)
        self.current_state = int(self.data['State_Idx'].iloc[-1])

        self.data['Next_State'] = self.data['State_Idx'].shift(-1)
        matrix = pd.crosstab(self.data['State_Idx'], self.data['Next_State'], normalize='index')
        self.markov_matrix_raw = matrix.reindex(
            index=range(self.n_states),
            columns=range(self.n_states),
            fill_value=0
        ).copy()
        self.markov_matrix = self.markov_matrix_raw.copy()

        print(f"    Current Regime: {self.state_map[self.current_state]}")

    # =========================================================================
    # LAYER 2: VIX FILTER (MACRO SENSOR)
    # =========================================================================

    def apply_vix_filter(self):
        print("[LAYER 2] VIX Filter (Macro Sensor)...")

        try:
            vix_data = yf.download("^VIX", period="5d", progress=False)
            self.vix_level = vix_data['Close'].iloc[-1].item()
        except Exception:
            self.vix_level = 20  # Default neutral
            print("    WARNING: Could not fetch VIX, using default")
            return

        print(f"    VIX Level: {self.vix_level:.1f}")

        # VIX Thresholds
        if self.vix_level > 30:
            self.vix_alert = True
            tilt = 1.4  # Extreme fear - 40% boost to downside
            alert_level = "EXTREME FEAR"
        elif self.vix_level > 25:
            self.vix_alert = True
            tilt = 1.2  # High fear - 20% boost to downside
            alert_level = "HIGH FEAR"
        elif self.vix_level < 15:
            self.vix_alert = False
            tilt = 0.9  # Complacency - slight boost to upside
            alert_level = "COMPLACENT"
        else:
            self.vix_alert = False
            tilt = 1.0
            alert_level = "NEUTRAL"

        print(f"    Macro Status: {alert_level}")

        # Apply tilt to matrix
        if tilt != 1.0:
            matrix = self.markov_matrix.values.copy()

            # Boost crash/bear states, reduce bull/rally
            matrix[:, 0] *= tilt  # Crash column
            matrix[:, 1] *= tilt  # Bear column
            matrix[:, 3] *= (2 - tilt)  # Bull column (inverse)
            matrix[:, 4] *= (2 - tilt)  # Rally column (inverse)

            # Re-normalize rows
            matrix = matrix / matrix.sum(axis=1, keepdims=True)
            self.markov_matrix = pd.DataFrame(
                matrix,
                index=self.markov_matrix.index,
                columns=self.markov_matrix.columns
            )
            print(f"    Matrix TILTED by VIX factor: {tilt:.2f}")

    # =========================================================================
    # LAYER 3: ANOMALY ENGINE (ISOLATION FOREST)
    # =========================================================================

    def run_anomaly_detection(self):
        print("[LAYER 3] Anomaly Engine (Isolation Forest)...")

        if not SKLEARN_AVAILABLE:
            print("    SKIPPED: sklearn not available")
            return

        # Features: Returns, Volume, Volatility
        features = pd.DataFrame({
            'Log_Ret': self.data['Log_Ret'],
            'Volume_Norm': self.data['Volume_Norm'],
            'Vol_5d': self.data['Log_Ret'].rolling(5).std()
        }).dropna()

        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(features)

        # Train Isolation Forest
        clf = IsolationForest(contamination=0.02, random_state=42)  # Top 2% weirdest
        features['Anomaly'] = clf.fit_predict(X)

        # Get anomaly score for latest point
        latest = X[-1].reshape(1, -1)
        self.anomaly_score = clf.score_samples(latest)[0]
        self.is_anomaly = clf.predict(latest)[0] == -1

        # Count recent anomalies
        recent_anomalies = (features['Anomaly'].tail(20) == -1).sum()

        if self.is_anomaly:
            print("    ALERT: Current behavior is ANOMALOUS!")
            print(f"    Anomaly Score: {self.anomaly_score:.3f} (lower = weirder)")
        else:
            print("    Status: Normal behavior")
            print(f"    Anomaly Score: {self.anomaly_score:.3f}")

        print(f"    Recent Anomalies (20d): {recent_anomalies}")

    # =========================================================================
    # OPTIONAL: GARCH VOLATILITY FORECAST
    # =========================================================================

    def run_garch_forecast(self):
        print("[OPTIONAL] GARCH Volatility Forecast...")

        if not ARCH_AVAILABLE:
            print("    SKIPPED: arch package not available (pip install arch)")
            return

        try:
            returns = self.data['Log_Ret'].dropna() * 100  # Scale for GARCH
            model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
            result = model.fit(disp='off')

            # Forecast next day volatility
            forecast = result.forecast(horizon=5)
            self.garch_forecast = np.sqrt(forecast.variance.iloc[-1].values) / 100

            print(f"    5-Day Vol Forecast: {self.garch_forecast[0]*np.sqrt(252):.1%} -> {self.garch_forecast[-1]*np.sqrt(252):.1%}")
        except Exception as e:
            print(f"    GARCH failed: {e}")

    # =========================================================================
    # SIMULATIONS (WITH FIXED DIRECTION LOGIC)
    # =========================================================================

    def simulate_markov(self):
        """Markov simulation with empirical fat-tail sampling."""
        return_pools = {
            s: self.data[self.data['State_Idx'] == s]['Log_Ret'].values
            for s in range(self.n_states)
        }

        matrix_vals = self.markov_matrix.values
        log_returns = np.zeros((self.simulations, self.days_ahead))
        current_states = np.full(self.simulations, self.current_state, dtype=int)

        for d in range(self.days_ahead):
            random_draws = np.random.rand(self.simulations)
            cum_probs = matrix_vals[current_states].cumsum(axis=1)
            next_states = (cum_probs >= random_draws[:, None]).argmax(axis=1)

            sampled_returns = np.zeros(self.simulations)
            for s in range(self.n_states):
                mask = (next_states == s)
                if np.any(mask):
                    pool = return_pools[s]
                    sampled_returns[mask] = np.random.choice(pool, size=np.sum(mask))

            log_returns[:, d] = sampled_returns
            current_states = next_states

        cumulative_log_returns = np.cumsum(log_returns, axis=1)
        return self.last_price * np.exp(cumulative_log_returns)

    def analyze_stress_hit_rate(self, paths, target, direction="up"):
        """
        FIXED: Proper hit rate calculation with direction.
        direction="up"   -> checks if price breaks ABOVE target (moonshots)
        direction="down" -> checks if price drops BELOW target (stress tests)
        """
        if direction == "down":
            hits = np.any(paths <= target, axis=1)
        else:
            hits = np.any(paths >= target, axis=1)

        # First passage times (for paths that hit)
        if direction == "down":
            raw_times = np.argmax(paths <= target, axis=1)
        else:
            raw_times = np.argmax(paths >= target, axis=1)

        valid_times = raw_times[hits]
        return np.mean(hits), valid_times

    # =========================================================================
    # MAIN ANALYSIS
    # =========================================================================

    def run_full_analysis(self):
        """Execute the complete Triad analysis."""

        print("\n" + "="*70)
        print("TRIAD RISK ENGINE v3.0")
        print("="*70)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"Ticker: {self.ticker}")
        print("="*70)

        # Run all layers
        self.build_markov_core()
        self.apply_vix_filter()
        self.run_anomaly_detection()
        self.run_garch_forecast()

        # Triad Status Summary
        print(f"\n{'='*70}")
        print("TRIAD STATUS SUMMARY")
        print("="*70)
        print(f"Current Price:    ${self.last_price:.2f}")
        print(f"Data Through:     {self.last_date.strftime('%Y-%m-%d')}")
        print("\n[LAYER 1] Markov Core:")
        print(f"    Regime:       {self.state_map[self.current_state]}")
        print("\n[LAYER 2] VIX Filter:")
        print(f"    VIX:          {self.vix_level:.1f}")
        print(f"    Alert:        {'YES - DEFENSIVE POSTURE' if self.vix_alert else 'No - Normal'}")
        print("\n[LAYER 3] Anomaly Engine:")
        if SKLEARN_AVAILABLE:
            print(f"    Status:       {'ANOMALY DETECTED - CAUTION' if self.is_anomaly else 'Normal Behavior'}")
            print(f"    Score:        {self.anomaly_score:.3f}")
        else:
            print("    Status:       DISABLED (install sklearn)")

        # Flight check
        print(f"\n{'='*70}")
        print("FLIGHT CHECK: Is it safe to trade?")
        print("="*70)

        if self.is_anomaly:
            print("   [X] ANOMALY: Current behavior is outside known patterns")
            print("       >> RECOMMENDATION: Cash is King. Wait for clarity.")
            can_fly = False
        elif self.vix_alert and self.current_state <= 1:  # VIX high + Bear/Crash
            print("   [!] HIGH RISK: Macro fear + Downside regime")
            print("       >> RECOMMENDATION: Defensive positions only.")
            can_fly = False
        elif self.vix_alert:
            print("   [~] ELEVATED RISK: Macro fear present")
            print("       >> RECOMMENDATION: Reduce position size 50%")
            can_fly = True
        else:
            print("   [OK] CLEAR: Normal conditions")
            print("       >> RECOMMENDATION: Standard position sizing")
            can_fly = True

        # Run simulations
        print(f"\n{'='*70}")
        print("STRESS SCENARIOS")
        print("="*70)

        paths = self.simulate_markov()

        # Upside analysis
        prob_up, times_up = self.analyze_stress_hit_rate(paths, self.target_up, "up")
        print(f"\nUPSIDE Target: ${self.target_up:.0f} ({((self.target_up/self.last_price)-1)*100:+.0f}%)")
        print(f"    Hit Rate:     {prob_up:.1%}")
        if len(times_up) > 0:
            print(f"    Median Time:  {np.median(times_up):.0f} days (if it happens)")

        # Downside analysis
        prob_down, times_down = self.analyze_stress_hit_rate(paths, self.target_down, "down")
        print(f"\nDOWNSIDE Target: ${self.target_down:.0f} ({((self.target_down/self.last_price)-1)*100:+.0f}%)")
        print(f"    Hit Rate:     {prob_down:.1%}")
        if len(times_down) > 0:
            print(f"    Median Time:  {np.median(times_down):.0f} days (if it happens)")

        # Final price distribution
        final = paths[:, -1]
        print(f"\nFINAL PRICE DISTRIBUTION (Day {self.days_ahead}):")
        print(f"    5th %:   ${np.percentile(final, 5):.0f} ({((np.percentile(final, 5)/self.last_price)-1)*100:+.0f}%)")
        print(f"    25th %:  ${np.percentile(final, 25):.0f} ({((np.percentile(final, 25)/self.last_price)-1)*100:+.0f}%)")
        print(f"    Median:  ${np.median(final):.0f}")
        print(f"    75th %:  ${np.percentile(final, 75):.0f} ({((np.percentile(final, 75)/self.last_price)-1)*100:+.0f}%)")
        print(f"    95th %:  ${np.percentile(final, 95):.0f} ({((np.percentile(final, 95)/self.last_price)-1)*100:+.0f}%)")

        print(f"\n{'='*70}")

        # Visualization
        self._plot_triad(paths, prob_up, prob_down, times_up, times_down)

        return {
            'can_fly': can_fly,
            'vix_level': self.vix_level,
            'vix_alert': self.vix_alert,
            'is_anomaly': self.is_anomaly,
            'current_regime': self.state_map[self.current_state],
            'prob_up': prob_up,
            'prob_down': prob_down,
            'paths': paths
        }

    def _plot_triad(self, paths, prob_up, prob_down, times_up, times_down):
        """Visualize the Triad analysis."""

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])

        # 1. Price Scenarios
        ax1 = fig.add_subplot(gs[0, :])
        x = np.arange(self.days_ahead)

        p10 = np.percentile(paths, 10, axis=0)
        p25 = np.percentile(paths, 25, axis=0)
        p50 = np.median(paths, axis=0)
        p75 = np.percentile(paths, 75, axis=0)
        p90 = np.percentile(paths, 90, axis=0)

        ax1.fill_between(x, p10, p90, color='cyan', alpha=0.1, label='80% Range')
        ax1.fill_between(x, p25, p75, color='cyan', alpha=0.2, label='50% Range')
        ax1.plot(x, p50, color='cyan', lw=2, label='Median')

        ax1.axhline(self.target_up, color='lime', ls=':', lw=2, label=f'Upside ${self.target_up:.0f} ({prob_up:.0%})')
        ax1.axhline(self.target_down, color='red', ls=':', lw=2, label=f'Downside ${self.target_down:.0f} ({prob_down:.0%})')
        ax1.axhline(self.last_price, color='white', ls='-', lw=1, alpha=0.5)

        # Triad status indicator
        status_color = 'red' if self.is_anomaly else ('orange' if self.vix_alert else 'lime')
        status_text = 'ANOMALY' if self.is_anomaly else ('VIX ALERT' if self.vix_alert else 'CLEAR')

        ax1.set_title(f"{self.ticker} TRIAD ANALYSIS | Status: {status_text} | "
                      f"VIX: {self.vix_level:.0f} | Regime: {self.state_map[self.current_state]}\n"
                      f"Date: {datetime.now().strftime('%Y-%m-%d')} | Current: ${self.last_price:.2f}",
                      fontsize=13, color=status_color)
        ax1.set_xlabel("Trading Days")
        ax1.set_ylabel("Price (USD)")
        ax1.legend(fontsize=9, loc='upper left')
        ax1.grid(alpha=0.2)

        # 2. Time-to-Target (Upside)
        ax2 = fig.add_subplot(gs[1, 0])
        if len(times_up) > 20:
            sns.histplot(times_up, color='lime', ax=ax2, bins=30, alpha=0.7)
            ax2.axvline(np.median(times_up), color='white', ls='--', label=f'Median: {np.median(times_up):.0f}d')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, f"Upside ${self.target_up:.0f}\nrarely hit ({prob_up:.1%})",
                    ha='center', va='center', fontsize=12, transform=ax2.transAxes)
        ax2.set_title("Time to UPSIDE Target", fontsize=11)
        ax2.set_xlabel("Days")

        # 3. Time-to-Target (Downside)
        ax3 = fig.add_subplot(gs[1, 1])
        if len(times_down) > 20:
            sns.histplot(times_down, color='red', ax=ax3, bins=30, alpha=0.7)
            ax3.axvline(np.median(times_down), color='white', ls='--', label=f'Median: {np.median(times_down):.0f}d')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, f"Downside ${self.target_down:.0f}\nrarely hit ({prob_down:.1%})",
                    ha='center', va='center', fontsize=12, transform=ax3.transAxes)
        ax3.set_title("Time to DOWNSIDE Target", fontsize=11)
        ax3.set_xlabel("Days")

        # 4. Regime Heatmap (with VIX adjustment shown)
        ax4 = fig.add_subplot(gs[1, 2])
        plot_matrix = self.markov_matrix.copy()
        plot_matrix.index = [self.state_map[i][:5] for i in plot_matrix.index]
        plot_matrix.columns = [self.state_map[i][:5] for i in plot_matrix.columns]

        title_suffix = " (VIX TILTED)" if self.vix_alert else ""
        sns.heatmap(plot_matrix, annot=True, fmt=".2f", cmap="magma", ax=ax4)
        ax4.set_title(f"Transition Matrix{title_suffix}", fontsize=11)
        ax4.set_ylabel("Today")
        ax4.set_xlabel("Tomorrow")

        plt.tight_layout()
        plt.show()


# ============================================================================
# EXECUTE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TRIAD RISK ENGINE v3.0 - THE QUANT STACK")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*70)

    # PLTR Analysis with proper targets
    engine = TriadRiskEngine(
        ticker="PLTR",
        target_up=280,      # Moonshot (+48% from ~$188)
        target_down=130,    # Stress (-31% from ~$188)
        days_ahead=126,
        simulations=5000
    )
    engine.ingest_data()
    results = engine.run_full_analysis()
