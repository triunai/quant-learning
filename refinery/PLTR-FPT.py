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
        """
                 Initialize a TriadRiskEngine instance for scenario simulation and risk analysis.
                 
                 Parameters:
                     ticker (str): Security ticker symbol used for data ingestion.
                     days_ahead (int): Simulation horizon in trading days (default 126).
                     simulations (int): Number of Monte Carlo / Markov simulated paths (default 5000).
                     target_up (float | None): Optional upside (moonshot) price target; can be set later if None.
                     target_down (float | None): Optional downside (stress) price target; can be set later if None.
                 
                 Attributes:
                     ticker (str): Provided ticker symbol.
                     days_ahead (int): Simulation horizon.
                     simulations (int): Number of simulated paths.
                     target_up (float | None): Upside target.
                     target_down (float | None): Downside target.
                     data (pandas.DataFrame | None): Placeholder for downloaded price and feature data.
                     markov_matrix (numpy.ndarray | None): VIX-adjusted Markov transition matrix.
                     markov_matrix_raw (numpy.ndarray | None): Raw Markov transition matrix before VIX adjustment.
                     mu (float): Annualized mean log return (initialized to 0).
                     sigma (float): Annualized return volatility (initialized to 0).
                     last_price (float): Most recent adjusted close price (initialized to 0).
                     last_date (datetime.date | None): Date of the most recent price (initialized to None).
                     current_state (int): Index of current regime state (initialized to 0).
                     n_states (int): Number of regime states (5).
                     state_map (dict): Mapping of state indices to regime labels.
                     vix_level (float): Latest VIX level (initialized to 0).
                     vix_alert (bool): Flag indicating VIX alert state.
                     is_anomaly (bool): Flag indicating if latest observation is anomalous.
                     anomaly_score (float): Anomaly score for the latest observation.
                     garch_forecast (float | None): Optional GARCH volatility forecast.
                 """
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
        """
        Load 5 years of price and volume data for the engine's ticker and compute primary return and volatility metrics.
        
        Fetches adjusted daily data via yfinance, computes log returns and a 20-day normalized volume series, drops missing rows, and updates the engine's derived attributes. Attributes set:
            - self.data (pd.DataFrame): downloaded data with 'Log_Ret' and 'Volume_Norm' columns.
            - self.last_price (float): most recent close price.
            - self.last_date (pd.Timestamp): date of the most recent row.
            - self.mu (float): annualized mean log return.
            - self.sigma (float): annualized log return volatility.
            - self.target_up (float): upside target (set to last_price * 1.5 if not provided).
            - self.target_down (float): downside target (set to last_price * 0.7 if not provided).
        """
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
        """
        Constructs the Markov regime core by assigning discrete regime indices and computing the state transition matrix.
        
        Creates a 'State_Idx' column by partitioning 'Log_Ret' into `n_states` quantile buckets, sets a shifted 'Next_State' column, and records the most recent regime index in `current_state`. Builds a row-normalized transition matrix (reindexed to include all states) and stores it in `markov_matrix_raw`, then copies it to `markov_matrix`.
        """
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
        """
        Assess recent VIX and adjust the Markov transition matrix and internal VIX state accordingly.
        
        Fetches recent VIX level and sets self.vix_level and self.vix_alert. Depending on the VIX regime, the method may modify self.markov_matrix to tilt transition probabilities toward downside or upside (adjusting the engine's regime transition tendencies). The method also prints a short status summary. If VIX cannot be fetched, a default neutral level is assigned and the matrix is left unchanged.
        """
        print("[LAYER 2] VIX Filter (Macro Sensor)...")

        try:
            vix_data = yf.download("^VIX", period="5d", progress=False)
            self.vix_level = vix_data['Close'].iloc[-1].item()
        except:
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
        """
        Detects anomalous recent market behavior using an Isolation Forest and updates instance anomaly attributes.
        
        If sklearn is unavailable, the method returns without modifying anomaly-related attributes. When executed, the detector uses Log_Ret, Volume_Norm, and 5-day rolling volatility as features, fits an Isolation Forest (contamination 2%), and evaluates the most recent observation.
        
        Attributes set on the instance:
            anomaly_score (float): model score for the latest observation (lower = more anomalous).
            is_anomaly (bool): True if the latest observation is classified as an anomaly.
        
        Notes:
            The method also computes and prints a count of anomalies in the most recent 20-day window but does not persist that count to an attribute.
        """
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
        """
        Fit a GARCH(1,1) model to historical log returns and store a 5-day volatility forecast on the instance.
        
        Fits a GARCH(1,1) to self.data['Log_Ret'] (scaled by 100), forecasts variance over a 5-day horizon, converts the forecasted variances to standard deviations, rescales them back (divide by 100), and assigns the resulting 1-D NumPy array to self.garch_forecast. If the required `arch` package is not available the method returns without changing state. Any model-fitting errors are caught and reported but not raised.
        """
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
        """
        Simulate future price paths using the Markov regime transition matrix and empirical regime return samples.
        
        Returns:
            simulated_paths (np.ndarray): Array of simulated price paths with shape (simulations, days_ahead) where each row is a future price trajectory starting from `self.last_price` for the configured horizon.
        """
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
        Compute the proportion of simulated price paths that reach a specified target and the first-passage times for those that do.
        
        Parameters:
            paths (array-like, shape (n_paths, n_steps)): Simulated price paths where each row is a single path over time.
            target (float): Price level to test for crossing.
            direction (str, optional): "up" to test for reaching or exceeding the target, "down" to test for falling to or below the target. Defaults to "up".
        
        Returns:
            hit_rate (float): Fraction of paths that hit the target (value between 0 and 1).
            valid_times (ndarray): 1-D array of first-passage time indices for paths that hit the target.
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
        """
        Run the complete Triad analysis pipeline and produce summary results and visualizations.
        
        This method executes all detection layers (Markov Core, VIX Filter, Anomaly Engine, optional GARCH forecast), runs Monte Carlo stress simulations, prints a status summary and recommendations, renders the Triad visualization, and returns the combined results.
        
        Returns:
            result (dict): Summary dictionary with keys:
                - can_fly (bool): Recommendation whether it is advisable to trade under current conditions.
                - vix_level (float): Latest VIX level used for filtering.
                - vix_alert (bool): Whether the VIX-triggered alert/tilt is active.
                - is_anomaly (bool): Whether the latest observation was flagged as an anomaly.
                - current_regime (str): Human-readable label of the current market regime.
                - prob_up (float): Empirical probability that simulated paths reach the upside target within the horizon.
                - prob_down (float): Empirical probability that simulated paths reach the downside target within the horizon.
                - paths (numpy.ndarray): Simulated price path matrix of shape (n_simulations, days_ahead).
        """

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
        """
        Render a four-panel visualization summarizing the Triad analysis results.
        
        Displays:
        - price scenario bands (median, 50% and 80% ranges) with upside/downside target lines and current price;
        - time-to-target histograms for upside and downside (or a summary note when hits are rare);
        - Markov regime transition heatmap (annotated and optionally showing VIX tilt).
        
        Parameters:
            paths (ndarray): Simulated price paths shaped (n_simulations, days_ahead).
            prob_up (float): Empirical probability of reaching the upside target over the horizon (0-1).
            prob_down (float): Empirical probability of reaching the downside target over the horizon (0-1).
            times_up (array-like): First-passage times (in days) for simulations that hit the upside target.
            times_down (array-like): First-passage times (in days) for simulations that hit the downside target.
        """

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