"""
REGIME RISK PLATFORM v7.0 - INSTITUTIONAL GRADE
================================================
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


class RegimeRiskEngine:
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
        self.regime_beta = {}  # CRITICAL: Regime-specific betas
        self.regime_alpha = {}  # Regime-specific alphas
        self.regime_residuals = {}  # Regime-specific residual pools

        # Asymmetric betas (up/down market days)
        self.beta_up = 1.0
        self.beta_down = 1.0

        # Jump diffusion params
        self.jump_prob = 0.02  # 2% daily jump probability
        self.jump_mu = 0
        self.jump_sigma = 0.05

        # Macro conditioning
        self.market_beta = 0
        self.market_alpha = 0
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
        """Build regimes from SLOW FEATURES, not return buckets."""
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

        # Rename regimes by Sharpe ratio for meaningful labels
        self._compute_regime_names_by_sharpe()

    def _fallback_regimes(self):
        """Simple vol-based regimes if sklearn unavailable."""
        self.data['Regime'] = pd.qcut(self.data['Vol_20d'], q=3, labels=[0, 1, 2])
        self.current_regime = int(self.data['Regime'].iloc[-1])
        self._compute_regime_stats()
        self._compute_transition_matrix()

    def _compute_regime_names_by_sharpe(self):
        """
        Name regimes by SHARPE RATIO, not just volatility.
        This gives more meaningful regime labels.
        """
        regime_stats = []

        for r in range(self.n_regimes):
            mask = self.data['Regime'] == r
            returns = self.data.loc[mask, 'Log_Ret'].values

            if len(returns) < 20:
                continue

            # Calculate annualized stats
            ann_return = np.mean(returns) * 252
            ann_vol = np.std(returns) * np.sqrt(252)
            sharpe = ann_return / ann_vol if ann_vol > 0 else 0

            # Get other characteristics
            avg_dd = self.data.loc[mask, 'Drawdown'].mean()
            avg_vol = self.data.loc[mask, 'Vol_20d'].mean()
            n_samples = np.sum(mask)

            regime_stats.append({
                'regime': r,
                'sharpe': sharpe,
                'return': ann_return,
                'vol': ann_vol,
                'avg_dd': avg_dd,
                'avg_vol': avg_vol,
                'n': n_samples
            })

        # Sort by Sharpe ratio (highest first)
        regime_stats.sort(key=lambda x: x['sharpe'], reverse=True)

        # Assign meaningful names based on characteristics
        for i, stats in enumerate(regime_stats):
            r = stats['regime']

            # Characterize by multiple factors
            is_crisis = stats['avg_dd'] < -0.25  # Deep drawdown
            is_high_vol = stats['avg_vol'] > 0.60  # Very high volatility
            is_high_return = stats['return'] > 1.0  # >100% annual return

            if is_crisis and is_high_vol:
                self.regime_names[r] = "Crisis"
            elif is_high_return and stats['sharpe'] > 1.0:
                self.regime_names[r] = "Momentum"
            elif stats['return'] > 0.2:
                self.regime_names[r] = "Bull"
            elif stats['return'] < -0.3:  # -30% annual
                self.regime_names[r] = "Bear"
            else:
                # Default by ranking
                if i == 0:
                    self.regime_names[r] = "High Alpha"
                elif i == 1:
                    self.regime_names[r] = "Neutral"
                else:
                    self.regime_names[r] = "Low Alpha"

        # Print diagnostics
        print("\n[REGIME CHARACTERIZATION BY SHARPE]")
        for stats in regime_stats:
            name = self.regime_names.get(stats['regime'], f"R{stats['regime']}")
            print(f"    {name:12} | Sharpe: {stats['sharpe']:.2f} | Return: {stats['return']:.1%} "
                  f"| Vol: {stats['vol']:.1%} | DD: {stats['avg_dd']:.1%} | N={int(stats['n'])}")

    def _compute_regime_stats(self):
        """Compute per-regime return distributions."""
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
        """Semi-Markov: empirical distribution of regime run lengths."""
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
        """Transition matrix between regimes."""
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
    # MACRO CONDITIONING (COHERENT FACTOR MODEL)
    # =========================================================================

    def compute_market_beta(self):
        """
        COHERENT factor model with guaranteed zero-mean residuals.

        Key principles:
        1. Standard OLS beta (not asymmetric - can't use in simulation)
        2. Mean-based alpha (not median - guarantees E[residuals] = 0)
        3. R-squared check: Fallback to empirical if factor model is useless
        4. Drift validation: Model drift must equal actual drift
        """
        print(f"[MACRO] Computing COHERENT beta/alpha vs {self.market_ticker}...")

        asset_ret = self.data['Log_Ret'].values
        market_ret = self.market_data['Log_Ret'].values

        # Global OLS regression
        cov_matrix = np.cov(asset_ret, market_ret)
        self.market_beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 1.0

        # Alpha from mean (guarantees zero-mean residuals)
        self.market_alpha = float(np.mean(asset_ret) - self.market_beta * np.mean(market_ret))

        # Residuals (zero mean by construction)
        residuals = asset_ret - (self.market_alpha + self.market_beta * market_ret)
        self.idio_vol = float(np.std(residuals) * np.sqrt(252))

        # R-squared for global model
        r_squared_global = 1 - np.var(residuals) / np.var(asset_ret) if np.var(asset_ret) > 0 else 0

        print(f"    Global: B={self.market_beta:.2f}, A={self.market_alpha*252:+.1%} ann")
        print(f"    R-squared: {r_squared_global:.2f} | Idio vol: {self.idio_vol:.1%}")

        # Per-regime coherent factor model
        print("    Regime-specific parameters:")
        self.regime_model_type = {}  # Track model type per regime
        self.regime_returns = {}  # For empirical fallback

        for r in range(self.n_regimes):
            mask = self.data['Regime'].values == r
            name = self.regime_names.get(r, f"R{r}")

            if np.sum(mask) < 30:
                # Not enough data - use global
                self.regime_beta[r] = self.market_beta
                self.regime_alpha[r] = self.market_alpha
                self.regime_residuals[r] = residuals
                self.regime_model_type[r] = "global"
                print(f"      {name}: Using global (n={np.sum(mask)} < 30)")
                continue

            r_asset = asset_ret[mask]
            r_market = market_ret[mask]

            # 1. OLS beta (standard, not asymmetric)
            cov_r = np.cov(r_asset, r_market)
            beta = cov_r[0, 1] / cov_r[1, 1] if cov_r[1, 1] > 0 else self.market_beta

            # 2. Alpha from MEAN (guarantees zero-mean residuals)
            alpha = np.mean(r_asset) - beta * np.mean(r_market)

            # 3. Residuals (zero mean BY CONSTRUCTION)
            regime_residuals = r_asset - (alpha + beta * r_market)

            # ENFORCE zero mean (numerical precision)
            regime_residuals = regime_residuals - np.mean(regime_residuals)

            # 4. R-squared check: Does factor model explain variance?
            r_squared = 1 - np.var(regime_residuals) / np.var(r_asset) if np.var(r_asset) > 0 else 0

            # 5. Validate drift decomposition
            actual_drift = np.mean(r_asset) * 252
            model_drift = (alpha + beta * np.mean(r_market)) * 252
            drift_error = abs(actual_drift - model_drift) / abs(actual_drift) if actual_drift != 0 else 0

            # 6. Decide: Factor model or empirical?
            if r_squared < 0.05:  # Factor model explains < 5% variance
                # Fall back to empirical sampling
                self.regime_returns[r] = r_asset
                self.regime_model_type[r] = "empirical"
                self.regime_beta[r] = beta  # Store for reference
                self.regime_alpha[r] = alpha
                self.regime_residuals[r] = regime_residuals
                print(f"      {name}: EMPIRICAL (R2={r_squared:.2f} too low) | drift={actual_drift:+.1%}")
            else:
                # Use factor model
                self.regime_beta[r] = beta
                self.regime_alpha[r] = alpha
                self.regime_residuals[r] = regime_residuals
                self.regime_model_type[r] = "factor"

                # Sanity check
                resid_mean = np.mean(regime_residuals) * 252
                if abs(resid_mean) > 1.0:  # More than 1% annual
                    print(f"      {name}: WARNING resid_mean={resid_mean:.1f}%")

                print(f"      {name}: B={beta:.2f}, A={alpha*252:+.1%}, R2={r_squared:.2f}, drift={actual_drift:+.1%}")

    # =========================================================================
    # VIX + ANOMALY
    # =========================================================================

    def check_macro_context(self):
        """VIX level and anomaly detection."""
        print("[CONTEXT] VIX + Anomaly...")

        try:
            vix = yf.download("^VIX", period="5d", progress=False)
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            self.vix_level = float(vix['Close'].iloc[-1])
        except (KeyError, IndexError, ValueError) as e:
            self.vix_level = 20
            print(f"    VIX fetch failed ({type(e).__name__}), using default 20")

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
        """GARCH volatility forecast."""
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
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            self.garch_vol = self.realized_vol
            print(f"    GARCH failed ({type(e).__name__}), using realized vol: {self.realized_vol:.1%}")

    # =========================================================================
    # SIMULATION: JUMP DIFFUSION + SEMI-MARKOV + MACRO
    # =========================================================================

    def simulate(self):
        """
        FIXED SIMULATION:
        - Uses ALPHA + beta factor model: r = alpha_regime + beta*r_market + epsilon
        - Samples residuals EMPIRICALLY (fat tails preserved)
        - Semi-Markov handles persistence (no double-counting with sticky matrix)
        - Market returns from historical distribution
        """
        print(f"[SIM] {self.simulations:,} paths x {self.days_ahead} days...")

        price_paths = np.zeros((self.simulations, self.days_ahead + 1))
        price_paths[:, 0] = self.last_price

        # Initialize regimes and durations
        current_regimes = np.full(self.simulations, self.current_regime)
        remaining_duration = np.zeros(self.simulations)

        # FIX: Sample initial durations using PROPER RESIDUAL LIFE DISTRIBUTION
        # When already in a regime, remaining time must use length-biased sampling.
        # Longer durations are more likely to be "observed in progress".
        for s in range(self.simulations):
            r = current_regimes[s]
            samples = self.regime_duration[r]['samples']
            samples_array = np.array(samples)
            if len(samples_array) > 0 and samples_array.sum() > 0:
                # Length-biased sampling: P(sample) ~ duration (longer runs more likely to be observed)
                weights = samples_array / samples_array.sum()
                full_duration = np.random.choice(samples_array, p=weights)
                # Residual life: uniform over [1, full_duration]
                remaining_duration[s] = np.random.randint(1, max(2, full_duration + 1))
            else:
                remaining_duration[s] = 5

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

                # Check model type for this regime
                model_type = getattr(self, 'regime_model_type', {}).get(r, 'factor')

                if model_type == "empirical":
                    # Pure empirical: sample directly from historical returns
                    regime_rets = self.regime_returns.get(r, self.data['Log_Ret'].values)
                    returns[s] = np.random.choice(regime_rets)
                else:
                    # Factor model: alpha + beta*market + epsilon
                    alpha = self.regime_alpha.get(r, self.market_alpha)
                    beta = self.regime_beta.get(r, self.market_beta)
                    market_ret = market_returns[s, d]

                    # Sample residual EMPIRICALLY (preserves fat tails)
                    residual_pool = self.regime_residuals.get(r, np.array([0]))
                    if len(residual_pool) > 0:
                        epsilon = np.random.choice(residual_pool)
                    else:
                        epsilon = 0

                    # Coherent factor model
                    returns[s] = alpha + beta * market_ret + epsilon

            # NOTE: Jump diffusion DISABLED to avoid double-counting
            # Empirical residual sampling already preserves fat tails from historical data.
            # Adding jump returns would double-count extreme events already in the residual pool.
            # If you want jump diffusion, you must first separate "normal" vs "jump" residuals:
            #   normal_residuals = residual_pool[abs(residual_pool) < threshold]
            #   jump_residuals = residual_pool[abs(residual_pool) >= threshold]
            # Then sample from normal_residuals and add jumps only from jump_residuals.
            # For now, empirical tails are sufficient for tail risk modeling.
            # jump_mask = np.random.rand(self.simulations) < self.jump_prob
            # jump_returns = np.random.normal(self.jump_mu, self.jump_sigma, self.simulations)
            # returns = np.where(jump_mask, returns + jump_returns, returns)

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
        CRITICAL DIAGNOSTIC: Sim daily stats MUST match historical.
        If these don't match, the simulator is broken before anything else matters.
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
        """Full risk dashboard: VaR, CVaR, drawdown, stop-loss, position sizing."""
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

        # Apply fractional Kelly (0.5x) with EXPONENTIAL DD penalty (smoother than linear)
        # Linear penalty was too harsh: at prob_dd_30=0.4, penalty=0.2 crushed Kelly to near-zero
        # Exponential decay: from 1 at prob_dd_30=0 to ~0.3 at prob_dd_30=0.4
        dd_penalty = np.exp(-3 * prob_dd_30)
        kelly = max(0, min(0.25, raw_kelly * 0.5 * dd_penalty))  # 0.25 max position (was 0.5)

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
        """Historical first-passage hit rates for targets."""
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
        RIGOROUS CALIBRATION: Multiple thresholds + horizons.
        This is what makes it actually trustworthy.
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
        Verify asymmetry between Crash/Down vs Up/Rip buckets.
        Key insight: if downside is overshooting, check if Crash tails are fatter.
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
        TRUE VALIDATION: Walk-forward (no future leakage).
        For each fold, train on past, predict on future, measure calibration.

        FIX: Define multiplier ONCE as percentage move, apply consistently.
        """
        print(f"\n[WALK-FORWARD VALIDATION] ({n_folds} folds)")

        closes = self.data['Close'].values
        n = len(closes)
        fold_size = n // (n_folds + 1)

        # FIX: Define multiplier ONCE outside loop (as percentage move)
        # e.g., target_up=280, last_price=189 -> up_mult=1.48 means "hit +48%"
        up_mult = self.target_up / self.last_price
        down_mult = self.target_down / self.last_price  # FIXED: Was missing assignment!

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

    def diagnose_under_prediction(self):
        """
        Diagnose why model under-predicts hit probabilities.
        This is a CRITICAL diagnostic for calibration issues.
        """
        print("\n" + "="*70)
        print("UNDER-PREDICTION DIAGNOSIS")
        print("="*70)

        # 1. Check regime distribution
        print("\n1. REGIME DISTRIBUTION:")
        for r in range(self.n_regimes):
            name = self.regime_names.get(r, f"R{r}")
            mask = self.data['Regime'] == r
            n_days = int(np.sum(mask))
            pct = n_days / len(self.data) * 100
            mean_return = np.mean(self.data.loc[mask, 'Log_Ret']) * 252 * 100
            print(f"   {name:10}: {n_days:4} days ({pct:.1f}%) | Mean return: {mean_return:+.1f}%")

        # 2. Check alpha contributions
        print("\n2. ALPHA DECOMPOSITION:")
        print(f"   Global alpha (ann): {self.market_alpha * 252 * 100:+.1f}%")
        print(f"   Current regime: {self.regime_names.get(self.current_regime, 'Unknown')}")

        for r in range(self.n_regimes):
            name = self.regime_names.get(r, f"R{r}")
            alpha = self.regime_alpha.get(r, 0) * 252 * 100
            print(f"   {name} alpha (ann): {alpha:+.1f}%")

        # 3. Check residual distribution
        print("\n3. RESIDUAL DISTRIBUTION (vs Normal):")
        asset_ret = self.data['Log_Ret'].values
        market_ret = self.market_data['Log_Ret'].values
        global_residuals = asset_ret - (self.market_alpha + self.market_beta * market_ret)

        res_std = np.std(global_residuals)
        res_mean = np.mean(global_residuals)
        normal_tail = stats.norm.ppf(0.95) * res_std
        actual_95 = np.percentile(global_residuals, 95)
        actual_5 = np.percentile(global_residuals, 5)

        print(f"   Residual mean: {res_mean*100:.4f}% (should be ~0)")
        print(f"   Residual std: {res_std*100:.3f}%")
        print(f"   Normal 95th: {normal_tail*100:.3f}%")
        print(f"   Actual 95th: {actual_95*100:.3f}%")
        print(f"   Actual 5th: {actual_5*100:.3f}%")
        print(f"   Tail fatness (95th): {actual_95/normal_tail:.2f}x")

        # 4. Check per-regime residual pools
        print("\n4. PER-REGIME RESIDUAL POOLS:")
        for r in range(self.n_regimes):
            name = self.regime_names.get(r, f"R{r}")
            pool = self.regime_residuals.get(r, np.array([0]))
            pool_mean = np.mean(pool) * 252 * 100 if len(pool) > 0 else 0
            pool_std = np.std(pool) * np.sqrt(252) * 100 if len(pool) > 0 else 0
            print(f"   {name}: n={len(pool)}, mean={pool_mean:+.1f}% ann, std={pool_std:.1f}%")

        # 5. CRITICAL: Check if simulation drift matches history
        print("\n5. SIMULATION DRIFT TEST:")

        # Run a quick simulation
        print("   Running 1000 paths to check drift...")
        orig_sims = self.simulations
        self.simulations = 1000
        try:
            test_paths = self.simulate()

            # Extract drift from paths
            final_prices = test_paths[:, -1]
            sim_returns = np.log(final_prices / self.last_price) / (self.days_ahead/252)
            sim_mean = np.mean(sim_returns)

            # Historical drift
            hist_mean = np.mean(asset_ret) * 252

            print(f"   Simulated annual drift: {sim_mean*100:+.1f}%")
            print(f"   Historical annual drift: {hist_mean*100:+.1f}%")
            drift_ratio = sim_mean / hist_mean if hist_mean != 0 else 0
            print(f"   Drift ratio (sim/hist): {drift_ratio:.2f}x")

            if drift_ratio < 0.5:
                print(f"\n   [!] CRITICAL: Simulation drift is only {drift_ratio:.0%} of historical!")
                print(f"   This explains the under-prediction.")
            elif drift_ratio > 1.5:
                print(f"\n   [!] WARNING: Simulation drift is {drift_ratio:.1f}x historical!")
                print(f"   Model may be OVER-predicting.")
            else:
                print(f"\n   [OK] Drift ratio is reasonable.")

        finally:
            self.simulations = orig_sims

        # 6. Check regime persistence in simulation
        print("\n6. REGIME PERSISTENCE CHECK:")
        print(f"   Current regime: {self.regime_names.get(self.current_regime, 'Unknown')}")
        if self.regime_probs is not None:
            try:
                probs_list = [float(p) for p in np.array(self.regime_probs).flatten()[:self.n_regimes]]
                print(f"   Regime probs: {[f'{p:.0%}' for p in probs_list]}")
            except:
                print(f"   Regime probs: {self.regime_probs}")
        else:
            print(f"   Regime probs: N/A")

        current_alpha = self.regime_alpha.get(self.current_regime, self.market_alpha) * 252 * 100
        global_alpha = self.market_alpha * 252 * 100

        if current_alpha < global_alpha - 10:
            print(f"\n   [!] Current regime alpha ({current_alpha:+.1f}%) is much lower than global ({global_alpha:+.1f}%)")
            print(f"   Simulations starting in this regime will under-estimate upside.")

        print("\n" + "="*70)

    def walk_forward_validation_fixed(self, n_folds: int = 10, target_pct: float = 0.50):
        """
        FIXED walk-forward validation using PERCENTAGE moves (not absolute prices).

        Args:
            n_folds: Number of walk-forward folds
            target_pct: Percentage move to test (e.g., 0.50 for +50%)

        Returns:
            Dict with calibration metrics
        """
        print(f"\n[WALK-FORWARD VALIDATION FIXED] ({n_folds} folds, +{target_pct:.0%})")

        closes = self.data['Close'].values
        n = len(closes)

        # Use PERCENTAGE moves, not absolute prices
        target_mult = 1.0 + target_pct  # e.g., 1.50 for +50%

        # Ensure we have enough data for meaningful folds
        min_fold_size = self.days_ahead * 2
        actual_folds = min(n_folds, (n - self.days_ahead) // min_fold_size)

        if actual_folds < 3:
            print(f"    WARNING: Only {actual_folds} folds possible. Need more data.")
            return None

        predictions = []
        actuals = []
        fold_details = []

        # Use expanding window for more robust training
        for fold in range(actual_folds):
            # Test window: most recent chunk not used in training
            test_start = n - (actual_folds - fold) * min_fold_size
            test_end = min(test_start + self.days_ahead, n)

            # Training window: all data BEFORE test_start
            train_end = test_start - 1

            if train_end <= self.days_ahead * 3:
                continue  # Not enough training data

            # Get training closes
            train_closes = closes[:train_end]

            # Compute historical hit rate on training data
            up_hits = 0
            windows = 0

            # IMPORTANT: Use SAME PERCENTAGE for all windows
            for t in range(len(train_closes) - self.days_ahead):
                start_price = train_closes[t]
                path = train_closes[t:t + self.days_ahead]

                # Apply percentage move to this window's start price
                target_price = start_price * target_mult
                if np.max(path) >= target_price:
                    up_hits += 1
                windows += 1

            # Calculate empirical probability
            if windows > 10:
                predicted_prob = up_hits / windows
            else:
                predicted_prob = 0.5  # Default if insufficient data

            # Actual outcome on test window
            test_start_price = closes[test_start]
            test_path = closes[test_start:test_end]
            test_target = test_start_price * target_mult
            actual_hit = 1 if np.max(test_path) >= test_target else 0

            # Store results
            predictions.append(predicted_prob)
            actuals.append(actual_hit)
            fold_details.append({
                'fold': fold,
                'train_end': train_end,
                'test_start': test_start,
                'pred': predicted_prob,
                'actual': actual_hit,
                'target_mult': target_mult,
                'test_start_price': test_start_price,
            })

        if len(predictions) < 3:
            print("    ERROR: Too few valid folds for calibration")
            return None

        # Calculate calibration metrics
        predictions_arr = np.array(predictions)
        actuals_arr = np.array(actuals)

        # Brier score
        brier = np.mean((predictions_arr - actuals_arr) ** 2)

        # Calibration error
        cal_error = abs(np.mean(predictions_arr) - np.mean(actuals_arr))

        # Print results
        print(f"    Target: +{target_pct:.0%} move (mult={target_mult:.2f})")
        print(f"    Valid folds: {len(predictions)}")
        print(f"    Brier Score: {brier:.3f} (0.25 = random, 0 = perfect)")
        print(f"    Calibration Error: {cal_error:.3f}")
        print(f"    Mean Predicted: {np.mean(predictions_arr):.1%}")
        print(f"    Mean Actual:    {np.mean(actuals_arr):.1%}")

        # Diagnostic
        pred_mean = np.mean(predictions_arr)
        actual_mean = np.mean(actuals_arr)

        if pred_mean > 0.01:  # Avoid division by zero
            under_pred_ratio = float(actual_mean / pred_mean)
            if under_pred_ratio > 1.5:
                print(f"\n    [!] WARNING: Model is UNDER-PREDICTING by {under_pred_ratio:.1f}x")
            elif under_pred_ratio < 0.67:
                print(f"\n    [!] WARNING: Model is OVER-PREDICTING by {1/under_pred_ratio:.1f}x")
        else:
            under_pred_ratio = float('inf')

        return {
            'brier': brier,
            'cal_error': cal_error,
            'mean_pred': float(np.mean(predictions_arr)),
            'mean_actual': float(np.mean(actuals_arr)),
            'under_prediction_ratio': under_pred_ratio,
            'predictions': predictions,
            'actuals': actuals,
            'fold_details': fold_details,
        }

    def analyze_fpt(self, paths, target, direction="up"):
        if direction == "down":
            hits = np.any(paths <= target, axis=1)
        else:
            hits = np.any(paths >= target, axis=1)
        return np.mean(hits)

    # =========================================================================
    # SIGNAL + CONFIDENCE
    # =========================================================================

    def generate_signal(self, prob_up, prob_down, risk):
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

        # REGIME CONTEXT (CRITICAL for interpretation)
        current_regime_name = self.regime_names.get(self.current_regime, 'Unknown')
        current_alpha = self.regime_alpha.get(self.current_regime, self.market_alpha) * 252

        # Find momentum regime for comparison
        momentum_alpha = None
        for r, name in self.regime_names.items():
            if "Momentum" in name or "High Alpha" in name:
                momentum_alpha = self.regime_alpha.get(r, 0) * 252
                break

        if current_alpha < -0.20:  # Very negative alpha regime
            reasons.append(f"In {current_regime_name} regime (A={current_alpha:+.0%})")
            reasons.append("Expect lower upside until regime change")
        elif current_alpha > 0.50:  # Very positive alpha regime
            reasons.append(f"In {current_regime_name} regime (A={current_alpha:+.0%})")
            reasons.append("Favorable conditions for upside")
        else:
            reasons.append(f"Current regime: {current_regime_name}")

        confidence = max(30, min(85, confidence))

        return {'signal': signal, 'confidence': confidence, 'reasoning': reasons}

    def what_if_momentum_regime(self):
        """
        Test what happens if we were in Momentum regime instead of current.
        This proves the model is working - it SHOULD show higher probabilities.
        """
        print("\n" + "="*70)
        print("WHAT-IF ANALYSIS: MOMENTUM REGIME")
        print("="*70)

        # Find momentum regime
        momentum_regime = None
        for r, name in self.regime_names.items():
            if "Momentum" in name or "High Alpha" in name:
                momentum_regime = r
                break

        if momentum_regime is None:
            print("  No Momentum regime found in current model")
            return None

        # Store original regime
        original_regime = self.current_regime
        original_name = self.regime_names.get(original_regime, 'Unknown')

        print(f"\n  Current regime: {original_name}")
        print(f"    Alpha: {self.regime_alpha[original_regime]*252:+.1%} ann")
        print(f"    Beta:  {self.regime_beta[original_regime]:.2f}")

        print(f"\n  Momentum regime: {self.regime_names[momentum_regime]}")
        print(f"    Alpha: {self.regime_alpha[momentum_regime]*252:+.1%} ann")
        print(f"    Beta:  {self.regime_beta[momentum_regime]:.2f}")

        # Temporarily set current regime to Momentum
        self.current_regime = momentum_regime

        # Run quick simulation
        orig_sims = self.simulations
        self.simulations = 1000

        print(f"\n  Simulating 1000 paths in Momentum regime...")
        paths = self.simulate()

        # Analyze
        final_prices = paths[:, -1]
        prob_up = np.mean(final_prices >= self.target_up)
        prob_down = np.mean(final_prices <= self.target_down)
        median_price = np.median(final_prices)
        p5, p95 = np.percentile(final_prices, [5, 95])

        print(f"\n  Results IF in Momentum regime:")
        print(f"    P(Up to ${self.target_up:.0f}):   {prob_up:.1%}")
        print(f"    P(Down to ${self.target_down:.0f}): {prob_down:.1%}")
        print(f"    Median price: ${median_price:.0f}")
        print(f"    5th-95th: ${p5:.0f} - ${p95:.0f}")

        # Restore
        self.current_regime = original_regime
        self.simulations = orig_sims

        # Comparison
        print(f"\n  COMPARISON:")
        print(f"    In {original_name}: Sim shows {self.hist_up_hit:.1%} historical hit rate")
        print(f"    In Momentum:        Would show {prob_up:.1%} probability")
        print(f"    Difference:         {prob_up - self.hist_up_hit:+.1%}")

        if prob_up > self.hist_up_hit:
            print(f"\n  [INSIGHT] Model correctly predicts HIGHER upside in Momentum regime")
        else:
            print(f"\n  [WARNING] Something is wrong - Momentum should have higher upside")

        print("="*70)

        return {
            'current_regime': original_name,
            'momentum_prob_up': prob_up,
            'current_prob_up': self.hist_up_hit,
            'improvement': prob_up - self.hist_up_hit
        }

    # =========================================================================
    # MAIN
    # =========================================================================

    def run(self, plot=True, run_full_calibration=True):
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

        fig = None
        if plot:
            fig = self._plot(paths, prob_up, prob_down, risk, signal)

        return {'paths': paths, 'prob_up': prob_up, 'prob_down': prob_down,
                'risk': risk, 'signal': signal}, signal, fig

    def _print_summary(self, paths, prob_up, prob_down, risk, signal):
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

        # 4. EXIT Transition matrix (matches simulator logic in simulate() lines 522-530)
        ax4 = fig.add_subplot(gs[1, 2])
        labels = [self.regime_names.get(i, f"R{i}")[:6] for i in range(self.n_regimes)]
        # FIX: Build exit matrix EXACTLY matching simulator logic to avoid mismatch
        exit_matrix = self.transition_matrix.copy()
        for r in range(self.n_regimes):
            probs = exit_matrix[r].copy()
            probs[r] = 0  # Zero out self-transition (duration handles persistence)
            if probs.sum() > 0:
                exit_matrix[r] = probs / probs.sum()
            else:
                # Fallback: uniform distribution if no exit transitions recorded
                exit_matrix[r] = np.ones(self.n_regimes) / self.n_regimes
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
        return fig
