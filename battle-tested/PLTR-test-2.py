"""
REGIME RISK PLATFORM v7.0 - INSTITUTIONAL GRADE
================================================
Complete overengineered quant platform with:

PHASE 1 - VALIDATION:
- Historical first-passage hit rates
- Walk-forward training (no future leakage)
- Calibration suite (Brier score)

PHASE 2 - REAL REGIMES:
- GMM clustering on slow features (vol, trend, drawdown)
- Regime-conditioned return distributions

PHASE 3 - TAIL + MACRO:
- Jump diffusion simulation
- Semi-Markov duration modeling
- Macro conditioning (market beta)

RISK DASHBOARD:
- VaR/CVaR at horizon
- Max drawdown probability
- Stop-loss breach probability
- Kelly-style position sizing
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

# Core ML
try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# GARCH
try:
    from arch import arch_model
    ARCH_OK = True
except ImportError:
    ARCH_OK = False

plt.style.use('dark_background')
sns.set_palette("plasma")


class RegimeRiskPlatform:
    """
    v7.0 - Institutional Grade Risk Platform

    Features:
    - Real regimes from slow features (not return buckets)
    - Jump diffusion for tail risk
    - Semi-Markov duration modeling
    - Macro conditioning
    - Full risk dashboard
    - Walk-forward validation
    """

    def __init__(self, ticker,
                 market_ticker="QQQ",
                 days_ahead=126,
                 simulations=5000,
                 target_up=None,
                 target_down=None,
                 stop_loss_pct=0.15,
                 n_regimes=3):

        """
                 Initialize the RegimeRiskPlatform instance and set modeling, simulation, and risk-analysis defaults.
                 
                 Parameters:
                     ticker (str): Asset ticker to model.
                     market_ticker (str): Market/index ticker used for beta and market conditioning (default "QQQ").
                     days_ahead (int): Simulation horizon in trading days for scenario generation (default 126).
                     simulations (int): Number of Monte Carlo simulated price paths to generate (default 5000).
                     target_up (float | None): Optional upward price target as a fractional gain (e.g., 0.10 for +10%); if None, target determination is deferred.
                     target_down (float | None): Optional downward price target as a fractional loss (e.g., 0.10 for -10%); if None, target determination is deferred.
                     stop_loss_pct (float): Fractional stop-loss threshold (e.g., 0.15 for 15% stop) used in risk metrics (default 0.15).
                     n_regimes (int): Number of regimes to infer with the regime model (default 3).
                 
                 The constructor also initializes internal containers and default parameters for regime statistics, jump-diffusion, macro conditioning, transition dynamics, GARCH fallbacks, historical benchmarks, and anomaly/VIX state.
                 """
                 self.ticker = ticker
        self.market_ticker = market_ticker
        self.days_ahead = days_ahead
        self.simulations = simulations
        self.target_up = target_up
        self.target_down = target_down
        self.stop_loss_pct = stop_loss_pct
        self.n_regimes = n_regimes

        # Data
        self.data = None
        self.market_data = None
        self.last_price = 0
        self.last_date = None

        # Regime model (GMM on slow features)
        self.gmm = None
        self.regime_features = None
        self.current_regime = 0
        self.regime_probs = None
        self.regime_names = {0: "Low Vol", 1: "Trending", 2: "Crisis"}

        # Per-regime parameters
        self.regime_mu = {}
        self.regime_sigma = {}
        self.regime_duration = {}  # Semi-Markov

        # Jump diffusion params
        self.jump_prob = 0.02  # 2% daily jump probability
        self.jump_mu = 0
        self.jump_sigma = 0.05

        # Macro conditioning
        self.market_beta = 0
        self.idio_vol = 0

        # Transition matrix
        self.transition_matrix = None

        # GARCH
        self.garch_vol = 0
        self.realized_vol = 0

        # Historical benchmarks
        self.hist_up_hit = 0
        self.hist_down_hit = 0

        # Anomaly
        self.is_anomaly = False
        self.vix_level = 0

    # =========================================================================
    # DATA INGESTION
    # =========================================================================

    def ingest_data(self):
        """
        Load historical price data for the asset and market, compute log returns and slow features used for regime detection, align both time series, and set derived state (last price/date, realized volatility, and default up/down targets).
        
        This method fetches five years of adjusted price data for the configured ticker and market ticker, computes log returns, rolling return and volatility features (5/20/60-day aggregates), a 60-day drawdown series, and a 20-day volume z-score; it removes missing values, intersects the asset and market indices, records the most recent close and date, computes annualized realized volatility from log returns, and populates default target_up/target_down values when they are not provided.
        """
        print(f"[DATA] Loading {self.ticker} + {self.market_ticker}...")

        # Asset data
        df = yf.download(self.ticker, period="5y", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        self.data = df.copy()

        # Market data for beta
        mkt = yf.download(self.market_ticker, period="5y", auto_adjust=True, progress=False)
        if isinstance(mkt.columns, pd.MultiIndex):
            mkt.columns = mkt.columns.get_level_values(0)
        self.market_data = mkt.copy()

        # Core returns
        self.data['Log_Ret'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.market_data['Log_Ret'] = np.log(self.market_data['Close'] / self.market_data['Close'].shift(1))

        # =====================================================================
        # SLOW FEATURES for regime detection (not return buckets!)
        # =====================================================================
        self.data['Ret_5d'] = self.data['Log_Ret'].rolling(5).sum()
        self.data['Ret_20d'] = self.data['Log_Ret'].rolling(20).sum()
        self.data['Vol_20d'] = self.data['Log_Ret'].rolling(20).std() * np.sqrt(252)
        self.data['Vol_60d'] = self.data['Log_Ret'].rolling(60).std() * np.sqrt(252)
        self.data['Drawdown'] = (self.data['Close'] / self.data['Close'].rolling(60).max()) - 1
        self.data['Volume_Z'] = (self.data['Volume'] - self.data['Volume'].rolling(20).mean()) / self.data['Volume'].rolling(20).std()

        self.data = self.data.dropna()
        self.market_data = self.market_data.dropna()

        # Align dates
        common_idx = self.data.index.intersection(self.market_data.index)
        self.data = self.data.loc[common_idx]
        self.market_data = self.market_data.loc[common_idx]

        self.last_price = float(self.data['Close'].iloc[-1])
        self.last_date = self.data.index[-1]

        # Stats
        self.realized_vol = float(self.data['Log_Ret'].std() * np.sqrt(252))

        # Targets
        if self.target_up is None:
            self.target_up = self.last_price * 1.5
        if self.target_down is None:
            self.target_down = self.last_price * 0.65

        print(f"    {len(self.data)} days | ${self.last_price:.2f} | Vol: {self.realized_vol:.1%}")

    # =========================================================================
    # PHASE 2: REAL REGIMES (GMM on slow features)
    # =========================================================================

    def build_regime_model(self):
        """
        Derive market regimes from slow-moving features and populate regime state on the instance.
        
        Fits a Gaussian Mixture Model on slow features ['Vol_20d', 'Vol_60d', 'Ret_20d', 'Drawdown'] (or falls back to a simple volatility bucketing if sklearn is unavailable), assigns regime labels and regime probabilities to historical rows, determines the current regime with its probability, and names regimes by their 20-day volatility characteristic. As side effects, this method sets self.gmm, self.data['Regime'], self.regime_probs, self.current_regime, and self.regime_names, prints regime diagnostics, and then computes per-regime statistics, semi-Markov durations, and the regime transition matrix via internal helpers.
        """
        print("[REGIMES] GMM clustering on slow features...")

        if not SKLEARN_OK:
            print("    sklearn not available, using simple vol buckets")
            self._fallback_regimes()
            return

        # Feature matrix
        features = ['Vol_20d', 'Vol_60d', 'Ret_20d', 'Drawdown']
        X = self.data[features].values

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit GMM
        self.gmm = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            random_state=42,
            n_init=5
        )
        self.gmm.fit(X_scaled)

        # Assign regimes
        self.data['Regime'] = self.gmm.predict(X_scaled)
        self.regime_probs = self.gmm.predict_proba(X_scaled)

        # Current regime (with uncertainty)
        latest_features = X_scaled[-1].reshape(1, -1)
        self.current_regime = self.gmm.predict(latest_features)[0]
        current_probs = self.gmm.predict_proba(latest_features)[0]

        # Name regimes by vol characteristic
        regime_vols = {}
        for r in range(self.n_regimes):
            mask = self.data['Regime'] == r
            regime_vols[r] = self.data.loc[mask, 'Vol_20d'].mean()

        sorted_regimes = sorted(regime_vols.items(), key=lambda x: x[1])
        self.regime_names = {
            sorted_regimes[0][0]: "Low Vol",
            sorted_regimes[1][0]: "Normal",
            sorted_regimes[2][0]: "Crisis" if self.n_regimes > 2 else "High Vol"
        }

        print(f"    Current: {self.regime_names.get(self.current_regime, f'R{self.current_regime}')} "
              f"(prob: {current_probs[self.current_regime]:.0%})")

        # FIX: Add regime diagnostics (sample counts, feature centroids)
        print("    Regime diagnostics:")
        for r in range(self.n_regimes):
            mask = self.data['Regime'] == r
            n_samples = np.sum(mask)
            avg_dd = self.data.loc[mask, 'Drawdown'].mean()
            avg_vol = self.data.loc[mask, 'Vol_20d'].mean()
            name = self.regime_names.get(r, f"R{r}")
            print(f"      {name}: n={n_samples}, avg_dd={avg_dd:.1%}, avg_vol={avg_vol:.1%}")

        # Per-regime statistics
        self._compute_regime_stats()

        # Semi-Markov: duration modeling
        self._compute_regime_durations()

        # Transition matrix
        self._compute_transition_matrix()

    def _fallback_regimes(self):
        """Simple vol-based regimes if sklearn unavailable."""
        self.data['Regime'] = pd.qcut(self.data['Vol_20d'], q=3, labels=[0, 1, 2])
        self.current_regime = int(self.data['Regime'].iloc[-1])
        self._compute_regime_stats()
        self._compute_transition_matrix()

    def _compute_regime_stats(self):
        """
        Compute and store per-regime mean and standard deviation of log returns.
        
        For each regime index, sets self.regime_mu[r] to the regime's sample mean of Log_Ret
        and self.regime_sigma[r] to the sample standard deviation when the regime has
        more than 10 observations. If a regime has 10 or fewer observations, sets the
        mean to 0.0 and the sigma to self.realized_vol / sqrt(252) as a fallback.
        This method also prints a brief, annualized summary of each regime's mu and sigma.
        """
        for r in range(self.n_regimes):
            mask = self.data['Regime'] == r
            returns = self.data.loc[mask, 'Log_Ret'].values
            if len(returns) > 10:
                self.regime_mu[r] = float(np.mean(returns))
                self.regime_sigma[r] = float(np.std(returns))
            else:
                self.regime_mu[r] = 0.0
                self.regime_sigma[r] = self.realized_vol / np.sqrt(252)

        print("    Regime stats: ", end="")
        for r in range(self.n_regimes):
            name = self.regime_names.get(r, f"R{r}")
            print(f"{name[:4]}(mu={self.regime_mu[r]*252:.1%}, sig={self.regime_sigma[r]*np.sqrt(252):.1%}) ", end="")
        print()

    def _compute_regime_durations(self):
        """
        Compute and store empirical run-length statistics for each regime based on historical regime labels.
        
        Populates self.regime_duration with a dict for each regime containing:
        - 'mean': average consecutive-day run length,
        - 'std': standard deviation of run lengths,
        - 'samples': list of observed run-lengths.
        
        If a regime has no observed runs, a default entry of {'mean': 5, 'std': 2, 'samples': [5]} is used. The method also prints a one-line summary of average duration and dispersion for each regime.
        """
        print("    Semi-Markov duration modeling...")

        regimes = self.data['Regime'].values

        for r in range(self.n_regimes):
            durations = []
            current_run = 0

            for i, reg in enumerate(regimes):
                if reg == r:
                    current_run += 1
                else:
                    if current_run > 0:
                        durations.append(current_run)
                    current_run = 0

            if current_run > 0:
                durations.append(current_run)

            if len(durations) > 0:
                self.regime_duration[r] = {
                    'mean': np.mean(durations),
                    'std': np.std(durations),
                    'samples': durations
                }
            else:
                self.regime_duration[r] = {'mean': 5, 'std': 2, 'samples': [5]}

        for r in range(self.n_regimes):
            name = self.regime_names.get(r, f"R{r}")
            d = self.regime_duration[r]
            print(f"      {name}: avg duration {d['mean']:.1f} days (+/-{d['std']:.1f})")

    def _compute_transition_matrix(self):
        """
        Builds and stores a smoothed row-stochastic transition matrix of regime-to-regime daily transitions.
        
        Counts transitions between consecutive days' Regime labels in self.data, adds additive smoothing of 0.01 to every cell, normalizes each row to sum to 1, and assigns the result to self.transition_matrix as an (n_regimes x n_regimes) NumPy array.
        """
        regimes = self.data['Regime'].values
        matrix = np.zeros((self.n_regimes, self.n_regimes))

        for i in range(len(regimes) - 1):
            current = int(regimes[i])
            next_r = int(regimes[i + 1])
            matrix[current, next_r] += 1

        # Normalize + smooth
        matrix += 0.01
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        self.transition_matrix = matrix

    # =========================================================================
    # MACRO CONDITIONING (WITH ALPHA!)
    # =========================================================================

    def compute_market_beta(self):
        """
        Estimate the asset's market exposure and intercept relative to the configured market series.
        
        Calculates parameters for the linear model r = alpha + beta * r_market + epsilon and stores:
        - market_beta: estimated beta (market sensitivity).
        - market_alpha: estimated intercept (alpha).
        - idio_vol: annualized standard deviation of residuals.
        - regime_alpha: per-regime intercepts (dict mapping regime -> alpha).
        - regime_residuals: per-regime residual pools (dict mapping regime -> residual array).
        """
        print(f"[MACRO] Computing alpha + beta vs {self.market_ticker}...")

        asset_ret = self.data['Log_Ret'].values
        market_ret = self.market_data['Log_Ret'].values

        # Simple OLS: r_asset = alpha + beta*r_market + epsilon
        cov = np.cov(asset_ret, market_ret)
        self.market_beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 1.0

        # CRITICAL FIX: Compute alpha (intercept)
        self.market_alpha = float(np.mean(asset_ret) - self.market_beta * np.mean(market_ret))

        # Residuals
        residuals = asset_ret - (self.market_alpha + self.market_beta * market_ret)
        self.idio_vol = float(np.std(residuals) * np.sqrt(252))

        # Per-regime alpha (even better!)
        self.regime_alpha = {}
        for r in range(self.n_regimes):
            mask = self.data['Regime'].values == r
            if np.sum(mask) > 10:
                r_asset = asset_ret[mask]
                r_market = market_ret[mask]
                self.regime_alpha[r] = float(np.mean(r_asset) - self.market_beta * np.mean(r_market))
            else:
                self.regime_alpha[r] = self.market_alpha

        # Store per-regime residual pools for empirical sampling
        self.regime_residuals = {}
        for r in range(self.n_regimes):
            mask = self.data['Regime'].values == r
            if np.sum(mask) > 10:
                r_asset = asset_ret[mask]
                r_market = market_ret[mask]
                self.regime_residuals[r] = r_asset - (self.regime_alpha[r] + self.market_beta * r_market)
            else:
                self.regime_residuals[r] = residuals

        print(f"    Beta: {self.market_beta:.2f} | Alpha (ann): {self.market_alpha*252:.1%}")
        print(f"    Idio vol: {self.idio_vol:.1%}")
        for r in range(self.n_regimes):
            name = self.regime_names.get(r, f"R{r}")
            print(f"      {name} alpha: {self.regime_alpha[r]*252:+.1%}")

    # =========================================================================
    # VIX + ANOMALY
    # =========================================================================

    def check_macro_context(self):
        """
        Update VIX level, adjust daily jump probability, and detect an anomaly in the latest observation.
        
        Sets the following attributes on self:
        - vix_level: latest VIX close price (defaults to 20 if the VIX fetch fails).
        - jump_prob: daily jump probability chosen by VIX thresholds (>30 → 0.05, >25 → 0.03, otherwise 0.02).
        - is_anomaly: when scikit-learn is available, True if the most recent (Log_Ret, Vol_20d, Drawdown) observation is flagged as an anomaly by an IsolationForest, False otherwise.
        """
        print("[CONTEXT] VIX + Anomaly...")

        try:
            vix = yf.download("^VIX", period="5d", progress=False)
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            self.vix_level = float(vix['Close'].iloc[-1])
        except:
            self.vix_level = 20

        # Adjust jump probability based on VIX
        if self.vix_level > 30:
            self.jump_prob = 0.05  # 5% daily jump prob in fear
        elif self.vix_level > 25:
            self.jump_prob = 0.03
        else:
            self.jump_prob = 0.02

        print(f"    VIX: {self.vix_level:.1f} | Jump prob: {self.jump_prob:.0%}")

        # Anomaly detection
        if SKLEARN_OK:
            features = self.data[['Log_Ret', 'Vol_20d', 'Drawdown']].dropna()
            scaler = StandardScaler()
            X = scaler.fit_transform(features)
            clf = IsolationForest(contamination=0.02, random_state=42)
            clf.fit(X)
            self.is_anomaly = clf.predict(X[-1].reshape(1, -1))[0] == -1
            print(f"    Anomaly: {'YES!' if self.is_anomaly else 'No'}")

    # =========================================================================
    # GARCH
    # =========================================================================

    def run_garch(self):
        """
        Compute and store an annualized one-step GARCH(1,1) volatility forecast.
        
        Sets self.garch_vol to the forecasted annualized volatility (e.g., 0.20 for 20%). If the ARCH library is unavailable or model fitting/forecasting fails, sets self.garch_vol to self.realized_vol.
        """
        if not ARCH_OK:
            self.garch_vol = self.realized_vol
            return

        print("[GARCH] Volatility forecast...")

        try:
            returns = self.data['Log_Ret'].dropna() * 100
            model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
            result = model.fit(disp='off')
            forecast = result.forecast(horizon=1)
            self.garch_vol = float(np.sqrt(forecast.variance.iloc[-1, 0]) / 100 * np.sqrt(252))
            print(f"    GARCH: {self.garch_vol:.1%} | Realized: {self.realized_vol:.1%}")
        except:
            self.garch_vol = self.realized_vol

    # =========================================================================
    # SIMULATION: JUMP DIFFUSION + SEMI-MARKOV + MACRO
    # =========================================================================

    def simulate(self):
        """
        Simulate future price paths using a regime-conditioned factor model with semi-Markov persistence.
        
        Simulations use a per-regime alpha plus beta times a market return plus an empirically sampled residual; market returns are drawn from historical market log-returns, regime persistence is handled via semi-Markov durations (forward-recurrence for initial remaining time), and occasional jump shocks are applied according to the configured jump parameters.
        
        Returns:
            np.ndarray: Simulated price paths of shape (n_paths, days_ahead) containing future prices for each simulation (initial price at t=0 is excluded).
        """
        print(f"[SIM] {self.simulations:,} paths x {self.days_ahead} days...")

        price_paths = np.zeros((self.simulations, self.days_ahead + 1))
        price_paths[:, 0] = self.last_price

        # Initialize regimes and durations
        current_regimes = np.full(self.simulations, self.current_regime)
        remaining_duration = np.zeros(self.simulations)

        # FIX: Sample initial durations using FORWARD RECURRENCE (residual life)
        # When already in a regime, remaining time is NOT the full run length.
        # Residual life is approximately uniform over [1, sampled_duration]
        for s in range(self.simulations):
            r = current_regimes[s]
            samples = self.regime_duration[r]['samples']
            full_duration = np.random.choice(samples)
            # Forward recurrence: uniform over [1, full_duration]
            remaining_duration[s] = np.random.randint(1, max(2, full_duration + 1))

        # Market returns: sample from HISTORICAL distribution (not Normal!)
        hist_market_ret = self.market_data['Log_Ret'].values
        market_returns = np.random.choice(hist_market_ret, size=(self.simulations, self.days_ahead))

        for d in range(self.days_ahead):
            prev_price = price_paths[:, d]

            # =================================================================
            # FACTOR MODEL WITH ALPHA: r = alpha_regime + beta*r_market + epsilon_empirical
            # =================================================================
            returns = np.zeros(self.simulations)

            for s in range(self.simulations):
                r = current_regimes[s]

                # Get per-regime alpha
                alpha = self.regime_alpha.get(r, self.market_alpha)

                # Market component
                market_ret = market_returns[s, d]

                # Sample residual EMPIRICALLY (preserves fat tails!)
                residual_pool = self.regime_residuals.get(r, np.array([0]))
                epsilon = np.random.choice(residual_pool)

                # Full factor model
                returns[s] = alpha + self.market_beta * market_ret + epsilon

            # Jump component (rare events)
            jump_mask = np.random.rand(self.simulations) < self.jump_prob
            jump_returns = np.random.normal(self.jump_mu, self.jump_sigma, self.simulations)
            returns = np.where(jump_mask, returns + jump_returns, returns)

            # Evolve prices
            price_paths[:, d + 1] = prev_price * np.exp(returns)

            # =================================================================
            # SEMI-MARKOV: Duration handles persistence (NOT transition matrix)
            # =================================================================
            remaining_duration -= 1
            expired = remaining_duration <= 0

            if np.any(expired):
                # Remove diagonal bias - let semi-Markov handle persistence
                # Use OFF-DIAGONAL transition probs only when duration expires
                for s in np.where(expired)[0]:
                    r = current_regimes[s]
                    probs = self.transition_matrix[r].copy()
                    # Zero out self-transition (duration already handled staying)
                    probs[r] = 0
                    if probs.sum() > 0:
                        probs = probs / probs.sum()
                    else:
                        probs = np.ones(self.n_regimes) / self.n_regimes

                    new_r = np.random.choice(self.n_regimes, p=probs)
                    current_regimes[s] = new_r
                    samples = self.regime_duration[new_r]['samples']
                    remaining_duration[s] = np.random.choice(samples)

        return price_paths[:, 1:]

    def verify_simulation_invariants(self, paths):
        """
        Check that simulated daily return statistics match historical daily return statistics and report results.
        
        Compares mean, standard deviation, skewness, and kurtosis between simulated returns (from `paths`) and historical returns stored in the instance, prints a diagnostic table and a warning if any check fails, and returns a dictionary of boolean results.
        
        Parameters:
            paths (ndarray): Simulated price paths with shape (n_simulations, n_days + 1); daily log returns are computed from adjacent price ratios.
        
        Returns:
            dict: Boolean flags with keys 'mean_ok', 'std_ok', 'skew_ok', 'kurt_ok' indicating whether each statistic falls within the internal tolerance:
                - mean_ok: abs(sim_mean - hist_mean) < 0.001
                - std_ok: relative difference abs(sim_std - hist_std) / hist_std < 0.3
                - skew_ok: abs(sim_skew - hist_skew) < 1.0
                - kurt_ok: abs(sim_kurt - hist_kurt) < 3.0
        """
        print("\n[INVARIANT CHECK] Sim vs Historical daily stats:")

        # Compute simulated daily returns
        sim_returns = np.diff(np.log(paths), axis=1).flatten()
        hist_returns = self.data['Log_Ret'].values

        # Stats comparison
        sim_mean = np.mean(sim_returns)
        hist_mean = np.mean(hist_returns)

        sim_std = np.std(sim_returns)
        hist_std = np.std(hist_returns)

        sim_skew = stats.skew(sim_returns)
        hist_skew = stats.skew(hist_returns)

        sim_kurt = stats.kurtosis(sim_returns)
        hist_kurt = stats.kurtosis(hist_returns)

        print(f"    {'Metric':<12} {'Sim':<12} {'Hist':<12} {'Match?':<10}")
        print(f"    {'-'*46}")

        mean_ok = abs(sim_mean - hist_mean) < 0.001
        std_ok = abs(sim_std - hist_std) / hist_std < 0.3
        skew_ok = abs(sim_skew - hist_skew) < 1.0
        kurt_ok = abs(sim_kurt - hist_kurt) < 3.0

        print(f"    {'Mean (daily)':<12} {sim_mean*100:+.3f}%{'':<5} {hist_mean*100:+.3f}%{'':<5} {'[OK]' if mean_ok else '[FAIL]'}")
        print(f"    {'Std (daily)':<12} {sim_std*100:.3f}%{'':<6} {hist_std*100:.3f}%{'':<6} {'[OK]' if std_ok else '[FAIL]'}")
        print(f"    {'Skewness':<12} {sim_skew:+.2f}{'':<8} {hist_skew:+.2f}{'':<8} {'[OK]' if skew_ok else '[FAIL]'}")
        print(f"    {'Kurtosis':<12} {sim_kurt:.2f}{'':<9} {hist_kurt:.2f}{'':<9} {'[OK]' if kurt_ok else '[FAIL]'}")

        all_ok = mean_ok and std_ok and skew_ok and kurt_ok
        if not all_ok:
            print("    WARNING: Invariants don't match! Check simulation logic.")

        return {'mean_ok': mean_ok, 'std_ok': std_ok, 'skew_ok': skew_ok, 'kurt_ok': kurt_ok}

    # =========================================================================
    # RISK DASHBOARD
    # =========================================================================

    def compute_risk_metrics(self, paths):
        """
        Compute a concise risk dashboard from simulated price paths.
        
        Parameters:
            paths (np.ndarray): Simulated price trajectories with shape (simulations, horizon+1), where column 0 is the starting price (self.last_price) and subsequent columns are future prices.
        
        Returns:
            metrics (dict): Dictionary containing:
                var_95 (float): 5th percentile of simple returns across simulations.
                cvar_95 (float): Mean of simple returns that are <= `var_95`.
                prob_dd_20 (float): Fraction of simulations with maximum drawdown <= -20%.
                prob_dd_30 (float): Fraction of simulations with maximum drawdown <= -30%.
                expected_max_dd (float): Average maximum drawdown across simulations.
                prob_stop (float): Fraction of simulations that breached the configured stop price.
                kelly_fraction (float): Fractional Kelly sizing (capped between 0 and 0.5) after a drawdown-aware penalty.
                win_rate (float): Fraction of simulations with positive simple return at horizon.
                max_drawdowns (np.ndarray): Array of per-simulation maximum drawdowns (negative values).
        """
        final = paths[:, -1]

        # FIX: Use SIMPLE returns for interpretable VaR/CVaR (not log returns)
        log_returns = np.log(final / self.last_price)
        simple_returns = np.exp(log_returns) - 1  # Convert: R = e^r - 1

        # VaR / CVaR on simple returns (what humans expect)
        var_95 = np.percentile(simple_returns, 5)
        cvar_95 = np.mean(simple_returns[simple_returns <= var_95])

        # Max drawdown per path
        max_drawdowns = []
        for i in range(self.simulations):
            path = paths[i]
            peak = np.maximum.accumulate(path)
            dd = (path - peak) / peak
            max_drawdowns.append(np.min(dd))

        max_drawdowns = np.array(max_drawdowns)
        prob_dd_20 = np.mean(max_drawdowns <= -0.20)
        prob_dd_30 = np.mean(max_drawdowns <= -0.30)
        expected_max_dd = np.mean(max_drawdowns)

        # Stop-loss breach
        stop_price = self.last_price * (1 - self.stop_loss_pct)
        stop_breach = np.any(paths <= stop_price, axis=1)
        prob_stop = np.mean(stop_breach)

        # FIX: Fractional Kelly with DD penalty (binary Kelly doesn't apply to continuous returns)
        # Kelly for continuous: f* ~ mu/sigma^2, but we use fractional (0.5x) + DD cap
        win_rate = np.mean(simple_returns > 0)
        mu = np.mean(simple_returns)
        sigma = np.std(simple_returns)
        raw_kelly = mu / (sigma ** 2) if sigma > 0 else 0

        # Apply fractional Kelly (0.5x) and DD-aware penalty
        dd_penalty = max(0, 1 - 2 * prob_dd_30)  # Scale down if DD risk > 50%
        kelly = max(0, min(0.5, raw_kelly * 0.5 * dd_penalty))

        return {
            'var_95': var_95,
            'cvar_95': cvar_95,
            'prob_dd_20': prob_dd_20,
            'prob_dd_30': prob_dd_30,
            'expected_max_dd': expected_max_dd,
            'prob_stop': prob_stop,
            'kelly_fraction': kelly,
            'win_rate': win_rate,
            'max_drawdowns': max_drawdowns
        }

    # =========================================================================
    # RIGOROUS CALIBRATION SUITE
    # =========================================================================

    def compute_historical_validation(self):
        """
        Compute historical first-passage hit rates for the configured up and down targets and store them on the instance.
        
        Scans historical closing prices with a sliding window of length `self.days_ahead`. For each window starting at time t, it checks whether the price path within the next `days_ahead` days reaches or exceeds the up target multiplier (target_up relative to the most recent price) or falls to or below the down target multiplier.
        
        Sets:
            self.hist_up_hit (float): Fraction of windows where the up target was hit.
            self.hist_down_hit (float): Fraction of windows where the down target was hit.
        
        Notes:
            Uses `self.data['Close']`, `self.days_ahead`, `self.target_up`, `self.target_down`, and `self.last_price` to compute the results.
        """
        closes = self.data['Close'].values
        n = len(closes)
        horizon = self.days_ahead

        up_mult = self.target_up / self.last_price
        down_mult = self.target_down / self.last_price

        up_hits = 0
        down_hits = 0
        windows = 0

        for t in range(n - horizon):
            start_price = closes[t]
            path = closes[t:t + horizon]

            if np.max(path) >= start_price * up_mult:
                up_hits += 1
            if np.min(path) <= start_price * down_mult:
                down_hits += 1
            windows += 1

        if windows > 0:
            self.hist_up_hit = up_hits / windows
            self.hist_down_hit = down_hits / windows

        print("\n[VALIDATION] Historical hit rates:")
        print(f"    Up {((up_mult-1)*100):+.0f}%:   {self.hist_up_hit:.1%}")
        print(f"    Down {((down_mult-1)*100):+.0f}%: {self.hist_down_hit:.1%}")

    def multi_threshold_calibration(self):
        """
        Calibrates historical probabilities of reaching multiple percentage thresholds over several fixed horizons.
        
        Computes, for each combination of horizon and threshold, the fraction of historical rolling windows in which the asset's price
        reached an upside threshold (start_price * (1 + threshold)) or a downside threshold (start_price * (1 - threshold))
        within the horizon. Also prints a human-readable table of the calibration results.
        
        Returns:
            results (list[dict]): A list of dictionaries, each containing:
                - 'horizon' (int): horizon in trading days used for the window.
                - 'threshold' (float): target percentage (e.g., 0.10 for 10%).
                - 'up_hit' (float): fraction of windows that hit the upside threshold (0.0–1.0).
                - 'down_hit' (float): fraction of windows that hit the downside threshold (0.0–1.0).
                - 'n_windows' (int): number of rolling windows evaluated for that combination.
        """
        print("\n[MULTI-THRESHOLD CALIBRATION]")

        closes = self.data['Close'].values
        n = len(closes)

        thresholds = [0.10, 0.20, 0.30, 0.50]  # +/- percentages
        horizons = [21, 63, 126]  # 1mo, 3mo, 6mo

        results = []

        for horizon in horizons:
            for thresh in thresholds:
                up_hits = 0
                down_hits = 0
                windows = 0

                for t in range(n - horizon):
                    start_price = closes[t]
                    path = closes[t:t + horizon]

                    if np.max(path) >= start_price * (1 + thresh):
                        up_hits += 1
                    if np.min(path) <= start_price * (1 - thresh):
                        down_hits += 1
                    windows += 1

                if windows > 0:
                    results.append({
                        'horizon': horizon,
                        'threshold': thresh,
                        'up_hit': up_hits / windows,
                        'down_hit': down_hits / windows,
                        'n_windows': windows
                    })

        # Print table
        print(f"    {'Horizon':<10} {'Thresh':<10} {'Up Hit':<12} {'Down Hit':<12}")
        print(f"    {'-'*44}")
        for r in results:
            print(f"    {r['horizon']:<10} {r['threshold']*100:+.0f}%{'':<6} "
                  f"{r['up_hit']:.1%}{'':<6} {r['down_hit']:.1%}")

        return results

    def bucket_asymmetry_diagnostics(self):
        """
        Report per-regime return distribution and persistence statistics to diagnose downside/upside asymmetry.
        
        Prints, for each fitted regime with at least 10 observations, the regime name and these metrics:
        - annualized mean return (percent),
        - annualized volatility (percent),
        - sample skewness,
        - 5th and 95th percentile returns (percent),
        - average contiguous run length in days (persistence).
        
        If the Gaussian mixture model has not been fitted or a regime has fewer than 10 observations, that regime is skipped. No value is returned; output is emitted to standard output.
        """
        print("\n[BUCKET ASYMMETRY DIAGNOSTICS]")

        if self.gmm is None:
            print("    GMM not fitted, skipping")
            return

        for r in range(self.n_regimes):
            mask = self.data['Regime'] == r
            returns = self.data.loc[mask, 'Log_Ret'].values

            if len(returns) < 10:
                continue

            name = self.regime_names.get(r, f"R{r}")
            # FIX: Multiply by 100 to convert decimal to percent
            mean = np.mean(returns) * 252 * 100
            std = np.std(returns) * np.sqrt(252) * 100
            skew = stats.skew(returns)
            p5 = np.percentile(returns, 5) * 100
            p95 = np.percentile(returns, 95) * 100

            # Run length analysis
            regimes = self.data['Regime'].values
            runs = []
            current_run = 0
            for reg in regimes:
                if reg == r:
                    current_run += 1
                else:
                    if current_run > 0:
                        runs.append(current_run)
                    current_run = 0
            avg_run = np.mean(runs) if runs else 0

            print(f"    {name:12} | mu={mean:+6.1f}% | sig={std:5.1f}% | "
                  f"skew={skew:+.2f} | 5%={p5:+.1f}% | 95%={p95:+.1f}% | "
                  f"avg_run={avg_run:.1f}d")

    def walk_forward_validation(self, n_folds=5):
        """
        Perform walk-forward (chronological) validation of the historical probability of hitting the configured up target.
        
        Parameters:
            n_folds (int): Number of sequential training/testing folds to run (default 5). Each fold trains on all data prior to the test window and tests on the next horizon window.
        
        Returns:
            dict or None: If validation folds were produced, returns a dictionary with:
                - brier (float): Mean squared Brier score between predicted probabilities and actual binary outcomes.
                - cal_error (float): Absolute difference between mean predicted probability and mean actual hit rate.
                - predictions (ndarray): Predicted probability for each fold.
                - actuals (ndarray): Binary actual outcomes (1 if target hit in test window, 0 otherwise) for each fold.
                - fold_details (list): Per-fold dictionaries containing 'fold' (int), 'pred' (float), and 'actual' (int).
            Returns None if no valid folds could be evaluated.
        """
        print(f"\n[WALK-FORWARD VALIDATION] ({n_folds} folds)")

        closes = self.data['Close'].values
        n = len(closes)
        fold_size = n // (n_folds + 1)

        # FIX: Define multiplier ONCE outside loop (as percentage move)
        # e.g., target_up=280, last_price=189 -> up_mult=1.48 means "hit +48%"
        up_mult = self.target_up / self.last_price
        self.target_down / self.last_price

        predictions = []
        actuals = []
        fold_details = []  # For debugging

        for fold in range(n_folds):
            train_end = (fold + 1) * fold_size
            test_start = train_end
            test_end = min(test_start + self.days_ahead, n)

            if test_end - test_start < self.days_ahead:
                continue

            # Training data only
            train_closes = closes[:train_end]

            # Compute historical hit rate on training data
            # Apply SAME percentage move to each window's start price
            up_hits = 0
            windows = 0
            for t in range(len(train_closes) - self.days_ahead):
                start_price = train_closes[t]
                path = train_closes[t:t + self.days_ahead]
                if np.max(path) >= start_price * up_mult:
                    up_hits += 1
                windows += 1

            if windows > 0:
                predicted_prob = up_hits / windows
            else:
                predicted_prob = 0.5

            # Actual outcome on test period (same percentage move)
            test_start_price = closes[test_start]
            test_path = closes[test_start:test_end]
            actual_hit = 1 if np.max(test_path) >= test_start_price * up_mult else 0

            predictions.append(predicted_prob)
            actuals.append(actual_hit)
            fold_details.append({'fold': fold, 'pred': predicted_prob, 'actual': actual_hit})

        if len(predictions) > 0:
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            brier = np.mean((predictions - actuals) ** 2)
            cal_error = abs(np.mean(predictions) - np.mean(actuals))

            print(f"    Target: +{(up_mult-1)*100:.0f}% move (mult={up_mult:.2f})")
            print(f"    Brier Score: {brier:.3f} (lower = better, 0.25 = random)")
            print(f"    Calibration Error: {cal_error:.3f}")
            print(f"    Mean Predicted: {np.mean(predictions):.1%}")
            print(f"    Mean Actual:    {np.mean(actuals):.1%}")

            return {'brier': brier, 'cal_error': cal_error,
                    'predictions': predictions, 'actuals': actuals,
                    'fold_details': fold_details}

        return None

    def analyze_fpt(self, paths, target, direction="up"):
        """
        Compute the fraction of simulated price paths that hit a specified price target within the simulation horizon.
        
        Parameters:
            paths (ndarray): 2D array of simulated prices with shape (n_paths, n_steps), where each row is one path.
            target (float): Price threshold to test for first-passage.
            direction (str, optional): "up" to test for any price greater than or equal to `target`, "down" to test for any price less than or equal to `target`. Defaults to "up".
        
        Returns:
            float: Proportion of paths that reach the target at least once during the simulated period (value between 0 and 1).
        """
        if direction == "down":
            hits = np.any(paths <= target, axis=1)
        else:
            hits = np.any(paths >= target, axis=1)
        return np.mean(hits)

    # =========================================================================
    # SIGNAL + CONFIDENCE
    # =========================================================================

    def generate_signal(self, prob_up, prob_down, risk):
        """
        Generate a trading signal and a calibrated confidence score from simulated outcome probabilities and risk metrics.
        
        Parameters:
            prob_up (float): Simulated probability of reaching the upside target.
            prob_down (float): Simulated probability of reaching the downside target.
            risk (dict): Risk metrics produced by the simulation (expects keys such as 'prob_dd_30') used to adjust confidence.
        
        Returns:
            dict: {
                'signal': str,        # one of 'LONG', 'SHORT', 'NEUTRAL', or 'CASH' (immediate cash-out on anomaly)
                'confidence': int,    # confidence percentage clamped between 30 and 85
                'reasoning': list     # human-readable reasons that influenced the signal/confidence
            }
        
        Behavior notes:
        - If the platform has flagged an anomaly, returns 'CASH' with high confidence immediately.
        - Signals are driven by the edge (prob_up - prob_down), then adjusted downwards by drawdown risk, elevated VIX, and historical calibration comparisons.
        """
        signal = 'NEUTRAL'
        confidence = 50
        reasons = []

        if self.is_anomaly:
            return {'signal': 'CASH', 'confidence': 90, 'reasoning': ['Anomaly detected']}

        # Probability-driven
        edge = prob_up - prob_down
        if edge > 0.25:
            signal = 'LONG'
            confidence = int(50 + edge * 100)
            reasons.append(f"Strong edge: {edge:.0%}")
        elif edge < -0.25:
            signal = 'SHORT'
            confidence = int(50 - edge * 100)
            reasons.append(f"Negative edge: {edge:.0%}")
        else:
            reasons.append(f"Weak edge: {edge:.0%}")

        # Risk overlay
        if risk['prob_dd_30'] > 0.15:
            confidence = min(confidence, 50)
            reasons.append(f"High DD risk: {risk['prob_dd_30']:.0%}")

        if self.vix_level > 25:
            confidence -= 10
            reasons.append(f"VIX elevated: {self.vix_level:.0f}")

        # Historical calibration
        if prob_up > self.hist_up_hit * 1.5:
            confidence = min(confidence, 55)
            reasons.append("Sim > 1.5x historical")

        confidence = max(30, min(85, confidence))

        return {'signal': signal, 'confidence': confidence, 'reasoning': reasons}

    # =========================================================================
    # MAIN
    # =========================================================================

    def run(self, run_full_calibration=True):
        """
        Run the full regime-risk workflow to produce simulated price scenarios, risk metrics, and a trading signal.
        
        This method executes the end-to-end pipeline: build regimes, compute market conditioning (beta/alpha), check macro context, fit volatility (GARCH), perform historical validation, optionally run the full calibration suite, simulate forward price paths using the regime-conditioned stochastic engine, verify simulation invariants against historical statistics, compute first-passage probabilities for targets, derive risk dashboard metrics, and generate a final trading signal. It also prints a textual summary and produces diagnostic plots.
        
        Parameters:
            run_full_calibration (bool): If True, execute the additional calibration and validation steps (bucket asymmetry diagnostics, multi-threshold calibration, and walk‑forward validation). Defaults to True.
        
        Returns:
            result (dict): A dictionary containing:
                - paths: ndarray of simulated price paths (simulations x horizon).
                - prob_up: float probability of reaching the up target within the horizon.
                - prob_down: float probability of reaching the down target within the horizon.
                - risk: dict of computed risk metrics (VaR, CVaR, drawdown probabilities, stop-loss breach probability, Kelly fraction, etc.).
                - signal: dict describing the recommended action, confidence, and reasoning.
        """
        print("\n" + "="*70)
        print(f"REGIME RISK PLATFORM v7.0 - {self.ticker}")
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*70)

        self.build_regime_model()
        self.compute_market_beta()
        self.check_macro_context()
        self.run_garch()
        self.compute_historical_validation()

        # RIGOROUS CALIBRATION SUITE
        if run_full_calibration:
            self.bucket_asymmetry_diagnostics()
            self.multi_threshold_calibration()
            self.walk_forward_validation()

        paths = self.simulate()

        # CRITICAL: Verify simulation matches historical stats
        self.verify_simulation_invariants(paths)

        # Analysis
        prob_up = self.analyze_fpt(paths, self.target_up, "up")
        prob_down = self.analyze_fpt(paths, self.target_down, "down")
        risk = self.compute_risk_metrics(paths)
        signal = self.generate_signal(prob_up, prob_down, risk)

        # Print
        self._print_summary(paths, prob_up, prob_down, risk, signal)
        self._plot(paths, prob_up, prob_down, risk, signal)

        return {'paths': paths, 'prob_up': prob_up, 'prob_down': prob_down,
                'risk': risk, 'signal': signal}

    def _print_summary(self, paths, prob_up, prob_down, risk, signal):
        """
        Print a concise textual summary of current state, targets, simulated outcome distribution, risk metrics, and the generated trading signal.
        
        Parameters:
        	paths (ndarray): Simulated price paths with shape (n_simulations, days+1); the summary uses the last column (final prices).
        	prob_up (float): Estimated probability that the price will reach the upward target within the horizon.
        	prob_down (float): Estimated probability that the price will reach the downward target within the horizon.
        	risk (dict): Risk dashboard values including at minimum the keys 'var_95', 'cvar_95', 'prob_dd_20', 'prob_dd_30', 'prob_stop', and 'kelly_fraction'.
        	signal (dict): Signal information with keys 'signal' (str), 'confidence' (numeric percent), and 'reasoning' (iterable of strings) that explain the decision.
        
        """
        final = paths[:, -1]

        print(f"\n{'='*70}")
        print("SUMMARY")
        print("="*70)

        print("\n[STATE]")
        print(f"    Price:   ${self.last_price:.2f}")
        print(f"    Regime:  {self.regime_names.get(self.current_regime, 'Unknown')}")
        print(f"    VIX:     {self.vix_level:.1f}")
        print(f"    Beta:    {self.market_beta:.2f}")

        print("\n[TARGETS] (Sim vs History)")
        print(f"    Up ${self.target_up:.0f}:   Sim {prob_up:.1%} | Hist {self.hist_up_hit:.1%}")
        print(f"    Down ${self.target_down:.0f}: Sim {prob_down:.1%} | Hist {self.hist_down_hit:.1%}")

        print("\n[DISTRIBUTION]")
        print(f"    5th:    ${np.percentile(final, 5):.0f}")
        print(f"    Median: ${np.median(final):.0f}")
        print(f"    95th:   ${np.percentile(final, 95):.0f}")

        print("\n[RISK DASHBOARD]")
        print(f"    VaR(95):         {risk['var_95']*100:+.1f}%")
        print(f"    CVaR(95):        {risk['cvar_95']*100:+.1f}%")
        print(f"    P(DD > 20%):     {risk['prob_dd_20']:.1%}")
        print(f"    P(DD > 30%):     {risk['prob_dd_30']:.1%}")
        print(f"    P(Stop breach):  {risk['prob_stop']:.1%}")
        print(f"    Kelly fraction:  {risk['kelly_fraction']:.0%}")

        print(f"\n{'='*70}")
        print("SIGNAL")
        print("="*70)
        print(f"    >>> {signal['signal']} ({signal['confidence']}%)")
        for r in signal['reasoning']:
            print(f"        - {r}")
        print("="*70)

    def _plot(self, paths, prob_up, prob_down, risk, signal):
        """
        Render a multi-panel diagnostic figure summarizing simulated scenario outputs, risk metrics, calibration, and the generated trading signal.
        
        Parameters:
            paths (ndarray): Simulated price paths with shape (n_simulations, days_ahead).
            prob_up (float): Estimated probability that the up target is hit (0-1).
            prob_down (float): Estimated probability that the down target is hit (0-1).
            risk (dict): Risk dashboard outputs produced by compute_risk_metrics (keys used: 'max_drawdowns',
                'var_95', 'cvar_95', 'prob_dd_20', 'prob_dd_30', 'prob_stop', 'kelly_fraction').
            signal (dict): Signal dictionary produced by generate_signal (keys used: 'signal', 'confidence').
        
        The figure includes:
        - A price cone (median and percentile bands) with target and current price lines and regime/beta in the title.
        - A histogram of path-level maximum drawdowns with -20% and -30% reference lines.
        - A histogram of final simulated prices with median and last-price markers.
        - An exit transition heatmap (off-diagonal exit probabilities used by the simulator).
        - A horizontal bar risk dashboard showing VaR/CVaR and event probabilities.
        - A calibration comparison bar chart (simulated vs historical hit rates for up/down targets).
        - A signal summary box displaying the chosen signal and confidence.
        
        No value is returned; the function displays the plot.
        """
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1, 0.8])

        sig_colors = {'LONG': 'lime', 'SHORT': 'red', 'NEUTRAL': 'yellow', 'CASH': 'gray'}
        title_color = sig_colors.get(signal['signal'], 'white')

        # 1. Price cone
        ax1 = fig.add_subplot(gs[0, :])
        x = np.arange(self.days_ahead)

        p5 = np.percentile(paths, 5, axis=0)
        p25 = np.percentile(paths, 25, axis=0)
        p50 = np.median(paths, axis=0)
        p75 = np.percentile(paths, 75, axis=0)
        p95 = np.percentile(paths, 95, axis=0)

        ax1.fill_between(x, p5, p95, color='cyan', alpha=0.1)
        ax1.fill_between(x, p25, p75, color='cyan', alpha=0.2)
        ax1.plot(x, p50, color='cyan', lw=2.5, label='Median')
        ax1.axhline(self.target_up, color='lime', ls=':', lw=2, label=f'Up ${self.target_up:.0f} ({prob_up:.0%})')
        ax1.axhline(self.target_down, color='red', ls=':', lw=2, label=f'Down ${self.target_down:.0f} ({prob_down:.0%})')
        ax1.axhline(self.last_price, color='white', lw=1, alpha=0.5)

        ax1.set_title(f"{self.ticker} v7.0 | {signal['signal']} | "
                      f"Regime: {self.regime_names.get(self.current_regime, '?')} | β={self.market_beta:.1f}",
                      fontsize=14, color=title_color, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(alpha=0.2)

        # 2. Drawdown distribution
        ax2 = fig.add_subplot(gs[1, 0])
        sns.histplot(risk['max_drawdowns'] * 100, color='red', ax=ax2, bins=50, alpha=0.7)
        ax2.axvline(-20, color='orange', ls='--', label='-20%')
        ax2.axvline(-30, color='red', ls='--', label='-30%')
        ax2.set_title("Max Drawdown Distribution")
        ax2.set_xlabel("Drawdown %")
        ax2.legend()

        # 3. Final price distribution
        ax3 = fig.add_subplot(gs[1, 1])
        final = paths[:, -1]
        sns.histplot(final, color='cyan', ax=ax3, bins=50, alpha=0.7)
        ax3.axvline(self.last_price, color='white', ls='-')
        ax3.axvline(np.median(final), color='yellow', ls='--')
        ax3.set_title("Final Price Distribution")

        # 4. EXIT Transition matrix (what simulator actually uses)
        ax4 = fig.add_subplot(gs[1, 2])
        labels = [self.regime_names.get(i, f"R{i}")[:6] for i in range(self.n_regimes)]
        # FIX: Plot EXIT matrix (diagonal=0, renormalized) to match simulator
        exit_matrix = self.transition_matrix.copy()
        np.fill_diagonal(exit_matrix, 0)
        row_sums = exit_matrix.sum(axis=1, keepdims=True)
        exit_matrix = np.where(row_sums > 0, exit_matrix / row_sums, exit_matrix)
        sns.heatmap(exit_matrix, annot=True, fmt='.2f', cmap='magma',
                    xticklabels=labels, yticklabels=labels, ax=ax4)
        ax4.set_title("Exit Transitions (sim)")

        # 5. Risk metrics bar (with clear labels)
        ax5 = fig.add_subplot(gs[2, 0])
        # FIX: Label clearly - VaR/CVaR are 6mo terminal, DD is path-level
        metrics = ['VaR95 (6mo)', 'CVaR95 (6mo)', 'P(MaxDD>20%)', 'P(MaxDD>30%)', 'P(Stop)']
        values = [abs(risk['var_95'])*100, abs(risk['cvar_95'])*100,
                  risk['prob_dd_20']*100, risk['prob_dd_30']*100, risk['prob_stop']*100]
        colors = ['red' if v > 20 else 'orange' if v > 10 else 'green' for v in values]
        ax5.barh(metrics, values, color=colors)
        ax5.set_title("Risk Dashboard (%)")
        ax5.set_xlim(0, 50)

        # 6. Calibration comparison
        ax6 = fig.add_subplot(gs[2, 1])
        x_labels = ['Up Target', 'Down Target']
        sim = [prob_up * 100, prob_down * 100]
        hist = [self.hist_up_hit * 100, self.hist_down_hit * 100]
        x_pos = np.arange(len(x_labels))
        width = 0.35
        ax6.bar(x_pos - width/2, sim, width, label='Simulation', color='cyan')
        ax6.bar(x_pos + width/2, hist, width, label='Historical', color='orange')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(x_labels)
        ax6.legend()
        ax6.set_title("Sim vs Historical Hit Rates")
        ax6.set_ylabel("%")

        # 7. Signal box
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        ax7.text(0.5, 0.7, signal['signal'], fontsize=32, fontweight='bold',
                color=title_color, ha='center', va='center')
        ax7.text(0.5, 0.4, f"{signal['confidence']}% conf", fontsize=16,
                color='white', ha='center', va='center')
        ax7.text(0.5, 0.15, f"Kelly: {risk['kelly_fraction']:.0%}", fontsize=14,
                color='gray', ha='center', va='center')

        plt.tight_layout()
        plt.show()


# =============================================================================
# EXECUTE
# =============================================================================

if __name__ == "__main__":
    platform = RegimeRiskPlatform(
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
    results = platform.run()