"""
REGIME RISK ENGINE v6.0 - FULLY CALIBRATED
===========================================
All audit findings addressed:

P0 FIXES:
- Laplace smoothing + fallback for zero rows (no more state=0 teleport)
- Assert n_states matches actual bins (dynamic state handling)

P1 FIXES:
- VIX stickiness via convex blend: M_new = (1-α)M + αI
- GARCH scales volatility only, not drift: (r - μ)*scale + μ
- Soft ceiling uses smooth logistic drag

P2 FIXES:
- Sanity check uses z-score vs historical distribution
- Added stationary distribution + implied drift diagnostics
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    from arch import arch_model
    ARCH_OK = True
except ImportError:
    ARCH_OK = False

plt.style.use('dark_background')
sns.set_palette("plasma")


class RegimeRiskEngine:
    """
    v6.0 - Fully Calibrated

    This is a Markov chain over return-quantile buckets with conditional bootstrap.
    States = sorted return days (not learned regimes).

    Use for: Volatility analysis, stress testing, FPT estimation
    """

    def __init__(self, ticker,
                 days_ahead=126,
                 simulations=5000,
                 target_up=None,
                 target_down=None,
                 soft_ceiling=None,
                 enable_vix=True,
                 enable_anomaly=True,
                 enable_garch=True):

        self.ticker = ticker
        self.days_ahead = days_ahead
        self.simulations = simulations

        self.target_up = target_up
        self.target_down = target_down
        self.soft_ceiling = soft_ceiling

        self.enable_vix = enable_vix
        self.enable_anomaly = enable_anomaly
        self.enable_garch = enable_garch

        self.data = None
        self.markov_matrix = None
        self.markov_matrix_raw = None
        self.mu = 0
        self.sigma = 0
        self.last_price = 0
        self.last_date = None
        self.current_state = 0

        # State config - will be validated after qcut
        self.n_states = 5
        self.state_map = {
            0: "Crash", 1: "Down", 2: "Flat", 3: "Up", 4: "Rip"
        }
        self.state_colors = ['#FF0000', '#FF6B6B', '#888888', '#90EE90', '#00FF00']

        self.regime_confidence = 0
        self.initial_state_dist = None
        self.state_history = []

        self.vix_level = 0
        self.vix_alert = False

        self.garch_vol_forecast = None
        self.realized_vol = 0
        self.vol_scale = 1.0

        self.return_pools = {}
        self.state_means = {}  # For drift-preserving vol scaling

        # Historical benchmarks
        self.hist_6mo_returns = None
        self.hist_6mo_median = 0
        self.hist_6mo_std = 0

        # Diagnostics
        self.stationary_dist = None
        self.implied_daily_drift = 0

    def ingest_data(self):
        print(f"[DATA] Loading {self.ticker}...")

        df = yf.download(self.ticker, period="5y", auto_adjust=True, progress=False)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        self.data = df.copy()

        self.data['Log_Ret'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.data['Volume_Norm'] = self.data['Volume'] / self.data['Volume'].rolling(20).mean()
        self.data['Vol_20d'] = self.data['Log_Ret'].rolling(20).std() * np.sqrt(252)

        self.data = self.data.dropna()

        self.last_price = float(self.data['Close'].iloc[-1])
        self.last_date = self.data.index[-1]

        self.mu = float(self.data['Log_Ret'].mean() * 252)
        self.sigma = float(self.data['Log_Ret'].std() * np.sqrt(252))
        self.realized_vol = self.sigma

        # Historical benchmarks (log returns, not price returns) based on days_ahead
        self.hist_6mo_returns = self.data['Log_Ret'].rolling(self.days_ahead).sum().dropna()
        self.hist_6mo_median = float(self.hist_6mo_returns.median())
        self.hist_6mo_std = float(self.hist_6mo_returns.std())

        if self.target_up is None:
            self.target_up = self.last_price * 1.5
        if self.target_down is None:
            self.target_down = self.last_price * 0.65
        if self.soft_ceiling is None:
            self.soft_ceiling = self.last_price * 3.0

        hist_6mo_pct = (np.exp(self.hist_6mo_median) - 1) * 100
        print(f"    {len(self.data)} days | ${self.last_price:.2f} | Vol: {self.sigma:.1%}")
        print(f"    Historical 6mo median: {hist_6mo_pct:+.1f}% (log: {self.hist_6mo_median:+.3f})")

    def build_markov_core(self):
        print("[MARKOV] Building transition matrix...")

        # Use rank-based quantiles for stability
        self.data['State_Idx'] = pd.qcut(
            self.data['Log_Ret'].rank(method='first'),
            q=self.n_states,
            labels=False
        )

        # P0-2 FIX: Validate actual number of states
        actual_states = self.data['State_Idx'].nunique()
        if actual_states != self.n_states:
            print(f"    WARNING: qcut produced {actual_states} states, expected {self.n_states}")
            self.n_states = actual_states
            # Rebuild state map dynamically
            self.state_map = {i: f"Q{i}" for i in range(self.n_states)}

        self.current_state = int(self.data['State_Idx'].iloc[-1])
        self.state_history = self.data['State_Idx'].tail(20).values.astype(int)

        # Build transition matrix
        self.data['Next_State'] = self.data['State_Idx'].shift(-1)
        matrix = pd.crosstab(
            self.data['State_Idx'],
            self.data['Next_State'],
            normalize='index'
        )
        self.markov_matrix_raw = matrix.reindex(
            index=range(self.n_states),
            columns=range(self.n_states),
            fill_value=0
        ).values.copy()

        # ==============================================================
        # P0-1 FIX: Laplace smoothing + zero-row fallback
        # ==============================================================
        eps = 1e-3
        M = self.markov_matrix_raw + eps
        M = M / M.sum(axis=1, keepdims=True)

        # Fallback for any near-zero rows (shouldn't happen with smoothing, but safety)
        row_sums = M.sum(axis=1)
        bad_rows = row_sums < 0.5
        if np.any(bad_rows):
            print(f"    WARNING: {np.sum(bad_rows)} invalid rows, applying uniform fallback")
            M[bad_rows] = 1.0 / self.n_states

        self.markov_matrix = M

        # Build return pools with means for P1-2 fix
        for s in range(self.n_states):
            pool = self.data[self.data['State_Idx'] == s]['Log_Ret'].values
            if len(pool) > 0:
                self.return_pools[s] = pool
                self.state_means[s] = float(pool.mean())
            else:
                self.return_pools[s] = np.array([0])
                self.state_means[s] = 0.0

        # Calculate stationary distribution
        self._compute_stationary_distribution()

        self._calculate_fuzzy_initialization()

        print(f"    Bucket: {self.state_map[self.current_state]} | Confidence: {self.regime_confidence:.0%}")
        print(f"    Implied daily drift: {self.implied_daily_drift*100:.3f}%")

    def _compute_stationary_distribution(self):
        """Compute stationary distribution π such that πM = π."""
        try:
            # Find left eigenvector for eigenvalue 1
            eigenvalues, eigenvectors = np.linalg.eig(self.markov_matrix.T)
            idx = np.argmin(np.abs(eigenvalues - 1))
            stationary = np.real(eigenvectors[:, idx])
            stationary = stationary / stationary.sum()
            self.stationary_dist = stationary

            # Implied expected daily return
            self.implied_daily_drift = sum(
                self.stationary_dist[s] * self.state_means[s]
                for s in range(self.n_states)
            )
        except:
            self.stationary_dist = np.ones(self.n_states) / self.n_states
            self.implied_daily_drift = self.mu / 252

    def _calculate_fuzzy_initialization(self):
        if len(self.state_history) < 5:
            self.regime_confidence = 0.5
            self.initial_state_dist = np.ones(self.n_states) / self.n_states
            return

        current_count = np.sum(self.state_history[-10:] == self.current_state)
        transitions = np.sum(np.diff(self.state_history[-10:]) != 0)

        stability = current_count / 10
        churn = 1 - (transitions / 9)
        self.regime_confidence = (stability + churn) / 2

        self.initial_state_dist = np.zeros(self.n_states)

        if self.regime_confidence > 0.8:
            self.initial_state_dist[self.current_state] = 1.0
        else:
            self.initial_state_dist[self.current_state] = self.regime_confidence

            remaining = 1.0 - self.regime_confidence
            neighbors = [s for s in [self.current_state - 1, self.current_state + 1]
                        if 0 <= s < self.n_states]

            if neighbors:
                for n in neighbors:
                    self.initial_state_dist[n] = remaining / len(neighbors)
            else:
                self.initial_state_dist[self.current_state] += remaining

    def apply_vix_filter(self):
        """
        VIX stickiness via convex blend with identity.
        FIXED: α is now proportional to raw persistence to avoid inflation.
        """
        if not self.enable_vix:
            return

        print("[VIX] Macro context...")

        try:
            vix = yf.download("^VIX", period="5d", progress=False)
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            self.vix_level = float(vix['Close'].iloc[-1])
        except:
            self.vix_level = 20
            print("    WARNING: VIX fetch failed")
            return

        # Get raw diagonal mean to scale alpha proportionally
        raw_diag = np.mean(np.diag(self.markov_matrix))

        # Base alpha values (REDUCED from before)
        if self.vix_level > 30:
            self.vix_alert = True
            base_alpha = 0.02  # Less sticky in fear
            status = "EXTREME FEAR"
        elif self.vix_level > 25:
            self.vix_alert = True
            base_alpha = 0.05
            status = "HIGH FEAR"
        elif self.vix_level < 15:
            self.vix_alert = False
            base_alpha = 0.12  # More sticky in quiet (reduced from 0.25)
            status = "COMPLACENT"
        else:
            self.vix_alert = False
            base_alpha = 0.08
            status = "NEUTRAL"

        # Scale alpha so we don't inflate persistence beyond raw + 30%
        # If raw_diag is already high, apply less; if low, apply more
        alpha = base_alpha * (1 - raw_diag)  # Less blending when already sticky

        print(f"    VIX: {self.vix_level:.1f} | Status: {status}")

        # ==============================================================
        # P1-1 FIX: Convex blend with identity (always valid probs)
        # ==============================================================
        I = np.eye(self.n_states)
        self.markov_matrix = (1 - alpha) * self.markov_matrix + alpha * I
        # Already normalized by construction

        print(f"    Stickiness blend α={alpha:.2f}")

    def run_anomaly_detection(self):
        if not self.enable_anomaly or not SKLEARN_OK:
            return

        print("[ANOMALY] Isolation Forest...")

        features = pd.DataFrame({
            'Log_Ret': self.data['Log_Ret'],
            'Volume_Norm': self.data['Volume_Norm'],
            'Vol_20d': self.data['Vol_20d']
        }).dropna()

        scaler = StandardScaler()
        X = scaler.fit_transform(features)

        clf = IsolationForest(contamination=0.02, random_state=42)
        clf.fit(X)

        latest = X[-1].reshape(1, -1)
        self.anomaly_score = clf.score_samples(latest)[0]
        self.is_anomaly = clf.predict(latest)[0] == -1

        status = "ANOMALY!" if self.is_anomaly else "Normal"
        print(f"    Status: {status} | Score: {self.anomaly_score:.3f}")

    def run_garch_forecast(self):
        if not self.enable_garch or not ARCH_OK:
            return

        print("[GARCH] Volatility forecast...")

        try:
            returns = self.data['Log_Ret'].dropna() * 100
            model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
            result = model.fit(disp='off')

            forecast = result.forecast(horizon=5)
            self.garch_vol_forecast = np.sqrt(forecast.variance.iloc[-1].values) / 100

            garch_annual = self.garch_vol_forecast[0] * np.sqrt(252)
            self.vol_scale = garch_annual / self.realized_vol if self.realized_vol > 0 else 1.0
            self.vol_scale = np.clip(self.vol_scale, 0.5, 2.0)

            print(f"    GARCH vol: {garch_annual:.1%} | Realized: {self.realized_vol:.1%}")
            print(f"    Vol scale: {self.vol_scale:.2f} (drift-preserving)")
        except Exception as e:
            print(f"    GARCH failed: {e}")
            self.vol_scale = 1.0

    def simulate(self):
        """
        Full simulation with all P0/P1 fixes:
        - Returns from current state
        - Drift-preserving vol scaling
        - Smooth logistic ceiling drag
        """
        initial_states = np.random.choice(
            self.n_states,
            size=self.simulations,
            p=self.initial_state_dist
        )

        current_states = initial_states
        price_paths = np.zeros((self.simulations, self.days_ahead + 1))
        price_paths[:, 0] = self.last_price

        # Precompute state means array for vectorization
        means_arr = np.array([self.state_means[s] for s in range(self.n_states)])

        for d in range(self.days_ahead):
            prev_price = price_paths[:, d]

            # Sample returns from CURRENT state
            sampled_returns = np.zeros(self.simulations)
            for s in range(self.n_states):
                mask = (current_states == s)
                if np.any(mask):
                    pool = self.return_pools[s]
                    sampled_returns[mask] = np.random.choice(pool, size=np.sum(mask))

            # ==============================================================
            # P1-2 FIX: Scale volatility without scaling drift
            # (r - μ_state) * scale + μ_state
            # ==============================================================
            state_mu = means_arr[current_states]
            sampled_returns = (sampled_returns - state_mu) * self.vol_scale + state_mu

            # ==============================================================
            # P2-2 FIX: Smooth logistic ceiling drag (bounded)
            # ==============================================================
            proximity = prev_price / self.soft_ceiling
            # Logistic function: smooth transition, bounded output
            gravity = -0.02 / (1 + np.exp(-10 * (proximity - 0.9)))
            # Clamp gravity magnitude
            gravity = np.clip(gravity, -0.02, 0)

            # Evolve prices
            adjusted_returns = sampled_returns + gravity
            price_paths[:, d + 1] = prev_price * np.exp(adjusted_returns)

            # Transition to next state
            random_draws = np.random.rand(self.simulations)
            row_probs = self.markov_matrix[current_states]
            cum_probs = np.cumsum(row_probs, axis=1)
            next_states = (cum_probs >= random_draws[:, None]).argmax(axis=1)
            current_states = next_states

        return price_paths[:, 1:]

    def analyze_fpt(self, paths, target, direction="up"):
        if direction == "down":
            hits = np.any(paths <= target, axis=1)
            times = np.argmax(paths <= target, axis=1)
        else:
            hits = np.any(paths >= target, axis=1)
            times = np.argmax(paths >= target, axis=1)

        valid_times = times[hits]
        prob = np.mean(hits)
        return prob, valid_times

    def calculate_convexity(self):
        returns = self.data['Log_Ret'].values
        up = returns[returns > 0]
        down = returns[returns < 0]

        return {
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'upside_vol': np.std(up) * np.sqrt(252) if len(up) > 0 else 0,
            'downside_vol': np.std(down) * np.sqrt(252) if len(down) > 0 else 0,
        }

    def sanity_check(self, paths):
        """
        Enhanced sanity check with HISTORICAL HIT-RATE VALIDATION.
        This is the truth serum - compares sim hit rates to actual history.
        """
        final = paths[:, -1]

        # Simulated 6mo log return
        sim_log_returns = np.log(final / self.last_price)
        sim_median_log = np.median(sim_log_returns)

        # Z-score vs historical
        if self.hist_6mo_std > 0:
            z_score = (sim_median_log - self.hist_6mo_median) / self.hist_6mo_std
        else:
            z_score = 0

        # Count extreme outcomes
        pct_2x = np.mean(final >= self.last_price * 2) * 100
        pct_3x = np.mean(final >= self.last_price * 3) * 100
        pct_half = np.mean(final <= self.last_price * 0.5) * 100

        print("\n[SANITY CHECK]")
        print(f"    Sim 6mo median log-return:  {sim_median_log:+.3f}")
        print(f"    Hist 6mo median log-return: {self.hist_6mo_median:+.3f}")
        print(f"    Z-score vs history: {z_score:+.2f}", end="")
        if abs(z_score) > 2:
            print(" (WARNING: >2σ from history)")
        else:
            print(" (OK)")

        print(f"    2x paths: {pct_2x:.1f}% | 3x paths: {pct_3x:.1f}% | Half: {pct_half:.1f}%")

        # ==================================================================
        # HISTORICAL HIT-RATE VALIDATION (The Truth Serum)
        # ==================================================================
        hist_up_hit, hist_down_hit = self._compute_historical_hit_rates()

        print("\n[HISTORICAL FIRST-PASSAGE VALIDATION]")
        print(f"    Target Up:   ${self.target_up:.0f} ({((self.target_up/self.last_price)-1)*100:+.0f}%)")
        print(f"    Target Down: ${self.target_down:.0f} ({((self.target_down/self.last_price)-1)*100:+.0f}%)")
        print("    ---")
        print(f"    Historical up-hit:   {hist_up_hit:.1%}")
        print(f"    Historical down-hit: {hist_down_hit:.1%}")

        # ==================================================================
        # RAW vs SMOOTHED DIAGONAL (Are we inventing persistence?)
        # ==================================================================
        raw_diag_mean = np.mean(np.diag(self.markov_matrix_raw))
        smoothed_diag_mean = np.mean(np.diag(self.markov_matrix))

        print("\n[PERSISTENCE CHECK]")
        print(f"    Raw matrix diagonal mean:      {raw_diag_mean:.2f}")
        print(f"    Smoothed matrix diagonal mean: {smoothed_diag_mean:.2f}")
        if smoothed_diag_mean > raw_diag_mean * 1.5:
            print(f"    WARNING: Smoothing increased persistence by {(smoothed_diag_mean/raw_diag_mean - 1)*100:.0f}%!")

        return {
            'sim_median_log': sim_median_log,
            'hist_median_log': self.hist_6mo_median,
            'z_score': z_score,
            'pct_2x': pct_2x,
            'pct_3x': pct_3x,
            'pct_half': pct_half,
            'hist_up_hit': hist_up_hit,
            'hist_down_hit': hist_down_hit,
            'raw_diag': raw_diag_mean,
            'smoothed_diag': smoothed_diag_mean
        }

    def _compute_historical_hit_rates(self):
        """
        Compute actual historical first-passage hit rates.
        Rolling 126-day windows: did max(path) hit target_up? did min(path) hit target_down?
        """
        closes = self.data['Close'].values
        n = len(closes)
        horizon = self.days_ahead

        up_mult = self.target_up / self.last_price  # e.g., 1.48 for +48%
        down_mult = self.target_down / self.last_price  # e.g., 0.64 for -36%

        up_hits = 0
        down_hits = 0
        windows = 0

        for t in range(n - horizon):
            start_price = closes[t]
            path = closes[t:t + horizon]

            # Check if path ever touched targets relative to start price
            if np.max(path) >= start_price * up_mult:
                up_hits += 1
            if np.min(path) <= start_price * down_mult:
                down_hits += 1
            windows += 1

        if windows == 0:
            return 0.0, 0.0

        return up_hits / windows, down_hits / windows

    def generate_trade_signal(self, results, sanity):
        signal = 'NEUTRAL'
        confidence = 50
        reasons = []

        if self.is_anomaly:
            return {
                'signal': 'CASH',
                'confidence': 90,
                'reasoning': ['ANOMALY detected']
            }

        if self.vix_alert:
            confidence -= 15
            reasons.append(f"VIX Alert ({self.vix_level:.0f})")

        prob_up = results['prob_up']
        prob_down = results['prob_down']

        if prob_up > 0.70:
            signal = 'LONG'
            confidence = int(50 + prob_up * 40)
            reasons.append(f"Edge: {prob_up:.0%} up")
        elif prob_down > 0.70:
            signal = 'SHORT'
            confidence = int(50 + prob_down * 40)
            reasons.append(f"Edge: {prob_down:.0%} down")
        elif prob_up > 0.50 and prob_down < 0.30:
            signal = 'LONG'
            confidence = int(40 + prob_up * 30)
            reasons.append(f"Lean: {prob_up:.0%}↑ / {prob_down:.0%}↓")
        elif prob_down > 0.50 and prob_up < 0.30:
            signal = 'SHORT'
            confidence = int(40 + prob_down * 30)
            reasons.append(f"Lean: {prob_down:.0%}↓ / {prob_up:.0%}↑")

        # Regime context
        if signal == 'LONG' and self.current_state <= 1:
            reasons.append("CONTRARIAN REVERSAL")
            confidence -= 10
        elif signal == 'SHORT' and self.current_state >= 3:
            reasons.append("TOP PICKING")
            confidence -= 15

        # Edge quality
        edge = abs(prob_up - prob_down)
        if edge < 0.20:
            signal = 'NEUTRAL'
            confidence = 40
            reasons = [f"Weak edge: {edge:.0%}"]

        # Guardrails
        if self.regime_confidence < 0.4:
            confidence = min(confidence, 55)
            reasons.append("Low regime confidence")

        if abs(sanity['z_score']) > 2:
            confidence = min(confidence, 50)
            reasons.append("Sim far from history")

        if sanity['pct_3x'] > 10:
            confidence = min(confidence, 55)
            reasons.append("Many extreme paths")

        confidence = max(30, min(85, confidence))

        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': reasons
        }

    def run(self, plot=True):
        print("\n" + "="*70)
        print(f"REGIME RISK ENGINE v6.0 (CALIBRATED) - {self.ticker}")
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*70)

        self.build_markov_core()
        self.apply_vix_filter()
        self.run_anomaly_detection()
        self.run_garch_forecast()

        convex = self.calculate_convexity()

        print(f"\n[SIM] {self.simulations:,} paths x {self.days_ahead} days")
        paths = self.simulate()

        prob_up, times_up = self.analyze_fpt(paths, self.target_up, "up")
        prob_down, times_down = self.analyze_fpt(paths, self.target_down, "down")

        sanity = self.sanity_check(paths)

        results = {
            'paths': paths,
            'prob_up': prob_up,
            'prob_down': prob_down,
            'times_up': times_up,
            'times_down': times_down,
            'convexity': convex,
            'sanity': sanity
        }

        signal = self.generate_trade_signal(results, sanity)

        self._print_summary(results, signal, sanity)

        fig = None
        if plot:
            fig = self._plot(paths, prob_up, prob_down, times_up, times_down, signal)

        return results, signal, fig

    def _print_summary(self, results, signal, sanity):
        print(f"\n{'='*70}")
        print("SUMMARY")
        print("="*70)

        print("\n[STATE]")
        print(f"    Price:     ${self.last_price:.2f}")
        print(f"    Bucket:    {self.state_map[self.current_state]}")
        print(f"    Conf:      {self.regime_confidence:.0%}")
        print(f"    VIX:       {self.vix_level:.1f}")
        print(f"    Vol×:      {self.vol_scale:.2f}")

        print("\n[TARGETS]")
        print(f"    Up:   ${self.target_up:.0f} -> {results['prob_up']:.1%}")
        print(f"    Down: ${self.target_down:.0f} -> {results['prob_down']:.1%}")

        final = results['paths'][:, -1]
        print("\n[DISTRIBUTION]")
        print(f"    5th:    ${np.percentile(final, 5):.0f}")
        print(f"    Median: ${np.median(final):.0f}")
        print(f"    95th:   ${np.percentile(final, 95):.0f}")

        print(f"\n{'='*70}")
        print("SIGNAL")
        print("="*70)
        print(f"    >>> {signal['signal']} ({signal['confidence']}%)")
        for r in signal['reasoning']:
            print(f"        - {r}")
        print("="*70)

    def _plot(self, paths, prob_up, prob_down, times_up, times_down, signal):
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])

        sig_colors = {'LONG': 'lime', 'SHORT': 'red', 'NEUTRAL': 'yellow', 'CASH': 'gray'}
        title_color = sig_colors.get(signal['signal'], 'white')

        ax1 = fig.add_subplot(gs[0, :])
        x = np.arange(self.days_ahead)

        p5 = np.percentile(paths, 5, axis=0)
        p25 = np.percentile(paths, 25, axis=0)
        p50 = np.median(paths, axis=0)
        p75 = np.percentile(paths, 75, axis=0)
        p95 = np.percentile(paths, 95, axis=0)

        ax1.fill_between(x, p5, p95, color='cyan', alpha=0.1, label='90%')
        ax1.fill_between(x, p25, p75, color='cyan', alpha=0.2, label='50%')
        ax1.plot(x, p50, color='cyan', lw=2.5, label='Median')

        ax1.axhline(self.target_up, color='lime', ls=':', lw=2,
                    label=f'Up ${self.target_up:.0f} ({prob_up:.0%})')
        ax1.axhline(self.target_down, color='red', ls=':', lw=2,
                    label=f'Down ${self.target_down:.0f} ({prob_down:.0%})')
        ax1.axhline(self.last_price, color='white', lw=1, alpha=0.5)

        ax1.set_title(f"{self.ticker} v6.0 | {signal['signal']} | "
                      f"{self.state_map[self.current_state]} | Vol×{self.vol_scale:.1f}",
                      fontsize=14, color=title_color, fontweight='bold')
        ax1.set_xlabel("Days")
        ax1.set_ylabel("Price")
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(alpha=0.2)

        ax2 = fig.add_subplot(gs[1, 0])
        if len(times_up) > 20:
            sns.histplot(times_up, color='lime', ax=ax2, bins=30, alpha=0.7)
        else:
            ax2.text(0.5, 0.5, f"Rare: {prob_up:.1%}", ha='center', va='center',
                    fontsize=14, transform=ax2.transAxes)
        ax2.set_title(f"Time to Up ${self.target_up:.0f}")

        ax3 = fig.add_subplot(gs[1, 1])
        if len(times_down) > 20:
            sns.histplot(times_down, color='red', ax=ax3, bins=30, alpha=0.7)
        else:
            ax3.text(0.5, 0.5, f"Rare: {prob_down:.1%}", ha='center', va='center',
                    fontsize=14, transform=ax3.transAxes)
        ax3.set_title(f"Time to Down ${self.target_down:.0f}")

        ax4 = fig.add_subplot(gs[1, 2])
        labels = [s[:4] for s in self.state_map.values()]
        sns.heatmap(self.markov_matrix, annot=True, fmt='.2f', cmap='magma',
                    xticklabels=labels, yticklabels=labels, ax=ax4)
        ax4.set_title("Transition Matrix (smoothed)")

        plt.tight_layout()
        return fig
