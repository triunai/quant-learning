import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import gamma, expon, ks_2samp
from sklearn.cluster import KMeans

# Styles
plt.style.use('dark_background')
sns.set_palette("flare")

class SemiMarkovModel:
    def __init__(self, ticker, n_states=5):
        self.ticker = ticker
        self.n_states = n_states
        self.state_map = {}
        self.data = None
        self.duration_data = {}  # Store durations for each state
        self.duration_params = {} # Store fitted parameters for each state
        self.transition_matrix = None # Conditional transition matrix (P_ii = 0)

    def _process_data(self, period="5y"):
        """Fetches data, identifies regimes using Clustering, and extracts durations."""
        print(f"📡 Fetching {period} data for {self.ticker}...")
        df = yf.download(self.ticker, period=period, progress=False, auto_adjust=True)

        # Handle multi-index columns
        if isinstance(df.columns, pd.MultiIndex):
             if 'Close' in df.columns.get_level_values(0):
                 df = df.xs(self.ticker, axis=1, level=1, drop_level=True) if self.ticker in df.columns.get_level_values(1) else df

        col = 'Close' if 'Close' in df.columns else df.columns[0]
        df['Log_Ret'] = np.log(df[col] / df[col].shift(1))
        df = df.dropna()

        # --- 1. Robust Regime Identification (KMeans on Slow Features) ---
        # Features: Rolling Volatility (20d), Rolling Return (60d), Drawdown
        df['Vol_20d'] = df['Log_Ret'].rolling(20).std() * np.sqrt(252)
        df['Ret_60d'] = df[col].pct_change(60)
        df['DD'] = df[col] / df[col].cummax() - 1

        # Drop NaN from rolling windows
        df_clean = df.dropna().copy()

        features = ['Vol_20d', 'Ret_60d', 'DD']
        X = df_clean[features].values

        # Fit KMeans
        try:
            kmeans = KMeans(n_clusters=self.n_states, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            df_clean['Cluster'] = labels
        except Exception as e:
            print(f"⚠️ Clustering failed: {e}. Fallback to qcut.")
            df_clean['Cluster'] = pd.qcut(df_clean['Log_Ret'], q=self.n_states, labels=False, duplicates='drop')

        # --- Sort Clusters by Risk ---
        mapping = self._sort_clusters_by_risk(df_clean)
        df_clean['State_Idx'] = df_clean['Cluster'].map(mapping)

        # Define state map
        base_names = ["Crash", "Bear", "Flat", "Bull", "Rally"]
        if self.n_states == 5:
            self.state_map = {i: name for i, name in enumerate(base_names)}
        else:
            self.state_map = {i: f"Regime {i}" for i in range(self.n_states)}

        # Update main dataframe
        self.data = df_clean

        # --- 2. Duration Extraction ---
        # Identify blocks of consecutive states
        self.data['block'] = (self.data['State_Idx'] != self.data['State_Idx'].shift(1)).cumsum()

        durations = self.data.groupby(['block', 'State_Idx']).size().reset_index(name='duration')

        self.duration_data = {s: [] for s in range(self.n_states)}
        for _, row in durations.iterrows():
            state = int(row['State_Idx'])
            duration = int(row['duration'])
            self.duration_data[state].append(duration)

        # --- 3. Transition Matrix (Conditional on Change) ---
        durations['next_state'] = durations['State_Idx'].shift(-1)
        trans_counts = pd.crosstab(durations['State_Idx'], durations['next_state'])

        # Normalize
        self.transition_matrix = trans_counts.div(trans_counts.sum(axis=1), axis=0)

        # Reindex and clean
        all_states = range(self.n_states)
        self.transition_matrix = self.transition_matrix.reindex(index=all_states, columns=all_states, fill_value=0)

        for i in all_states:
            if self.transition_matrix.loc[i].sum() == 0:
                 probs = np.ones(self.n_states)
                 probs[i] = 0
                 probs = probs / probs.sum()
                 self.transition_matrix.loc[i] = probs

        return self.data

    def _sort_clusters_by_risk(self, df):
        """
        Sorts clusters based on a composite risk score so State 0 is highest risk (Crash).
        Risk Score = Vol * 2 + (-Return) + (-Drawdown * 1.5)
        """
        cluster_stats = []
        for cluster in df['Cluster'].unique():
            subset = df[df['Cluster'] == cluster]
            # High Vol = Risk
            vol = subset['Vol_20d'].mean()
            # Low (Negative) Return = Risk
            ret = subset['Ret_60d'].mean()
            # Low (Negative) Drawdown = Risk (e.g. -0.5 is riskier than -0.1)
            # We want deeper negative numbers to add to risk.
            # So -DD (positive magnitude) adds to risk.
            dd = subset['DD'].mean()

            # Formula: Vol*2 - Ret - DD*1.5
            # Example: Vol=0.2, Ret=-0.1, DD=-0.2 -> 0.4 - (-0.1) - (-0.2)*1.5 = 0.4+0.1+0.3 = 0.8
            risk_score = (vol * 2) - ret - (dd * 1.5)
            cluster_stats.append((cluster, risk_score))

        # Sort by risk score descending (Highest Risk = 0)
        cluster_stats.sort(key=lambda x: x[1], reverse=True)
        return {old: new for new, (old, _) in enumerate(cluster_stats)}

    def fit_distributions(self):
        """Fits Gamma (high vol) and Exponential (low vol) distributions."""
        print("📊 Fitting duration distributions...")
        for state in range(self.n_states):
            durations = self.duration_data[state]
            if not durations:
                self.duration_params[state] = {'dist': 'expon', 'params': (0, 1)}
                continue

            try:
                a, loc, scale = gamma.fit(durations, floc=0)
                self.duration_params[state] = {'dist': 'gamma', 'params': (a, loc, scale)}
            except:
                 self.duration_params[state] = {'dist': 'expon', 'params': (0, np.mean(durations))}

        print("✅ Distributions fitted.")

    def sample_duration(self, state):
        """Samples a fresh duration for the given state."""
        params = self.duration_params[state]
        if params['dist'] == 'gamma':
            a, loc, scale = params['params']
            d = gamma.rvs(a, loc=loc, scale=scale)
        else:
            loc, scale = params['params']
            d = expon.rvs(loc=loc, scale=scale)
        return max(1, int(round(d)))

    def sample_residual_duration(self, state, elapsed_days):
        """Samples remaining duration using Inverse Transform Sampling."""
        params = self.duration_params[state]
        if params['dist'] == 'expon':
            return self.sample_duration(state)

        a, loc, scale = params['params']
        cdf_t = gamma.cdf(elapsed_days, a, loc=loc, scale=scale)

        if cdf_t > 0.999:
             return 1

        u = np.random.uniform(cdf_t, 1.0)
        total_duration = gamma.ppf(u, a, loc=loc, scale=scale)
        remaining = total_duration - elapsed_days
        return max(1, int(round(remaining)))

    def sample_regime_returns(self, state, n_days):
        """
        Samples contiguous historical runs of this state to preserve autocorrelation.
        """
        # Identify runs of this state in historical data
        is_state = (self.data['State_Idx'] == state)
        # Create groups for contiguous regions
        # Shift compare to find boundaries
        groups = (is_state != is_state.shift()).cumsum()

        # Extract runs where state matches
        # Filter first to only get rows of this state
        state_data = self.data[is_state]
        if state_data.empty:
            return np.zeros(n_days)

        # Group filtered data by 'block' (already computed in _process_data)
        # We can use the 'block' column which is unique for each contiguous run
        runs = [group['Log_Ret'].values for _, group in state_data.groupby('block')]

        # Filter for runs long enough? Or stitch?
        # Ideally pick a run that is at least n_days
        valid_runs = [r for r in runs if len(r) >= n_days]

        if valid_runs:
            chosen_run = valid_runs[np.random.randint(len(valid_runs))]
            start_idx = np.random.randint(0, len(chosen_run) - n_days + 1)
            return chosen_run[start_idx : start_idx + n_days]
        else:
            # Fallback: Pick longest run and tile it
            if not runs:
                return np.zeros(n_days)
            longest_run = max(runs, key=len)
            # Tile until we have enough
            tiled = np.tile(longest_run, int(np.ceil(n_days / len(longest_run))) + 1)
            return tiled[:n_days]

    def regime_fatigue_score(self, state, days_in_state):
        """Calculates probability of exit (CDF)."""
        params = self.duration_params.get(state)
        if not params:
            return 0.5
        if params['dist'] == 'gamma':
            a, loc, scale = params['params']
            return gamma.cdf(days_in_state, a, loc=loc, scale=scale)
        else:
            loc, scale = params['params']
            return expon.cdf(days_in_state, loc=loc, scale=scale)

    def get_position_size(self, current_state, days_in_state, base_size=1.0):
        """
        Returns position multiplier based on regime fatigue and type.
        """
        fatigue = self.regime_fatigue_score(current_state, days_in_state)

        # State 0 (Crash) and 1 (Bear) are risky
        if current_state in [0, 1]:
            # In risky regimes, we want to stay defensive until fatigue is VERY high?
            # Or: Fatigue means "likely to end soon".
            # If Crash is likely to end, maybe scale up?
            # User logic: "In risky regimes, fatigue means might get WORSE before better -> multiplier low"
            # Actually user said: "0.3 + (1-fatigue)*0.3" -> If fatigue is 0, mult=0.6. If fatigue is 1, mult=0.3.
            # This implies high fatigue = lower position. "Don't catch a falling knife that's about to turn"?
            # Or maybe "Volatility clusters, so late in regime = peak danger"?
            # Let's stick to user's formula.
            multiplier = 0.3 + (1 - fatigue) * 0.3
        else:
            # Normal regimes (Bull/Rally)
            # High fatigue = likely to end -> reduce size
            multiplier = 0.5 + (1 - fatigue) * 0.5

        return base_size * multiplier

    def validate_model(self, simulated_paths):
        """
        Validates the model by comparing simulated vs empirical statistics.
        Returns a dictionary of validation metrics.
        """
        report = {}

        # 1. Regime Duration KS Test
        # Extract simulated durations
        # This is hard because simulated_paths are prices. We need states.
        # But run_simulation doesn't return states.
        # Ideally, we should refactor run_simulation to return (paths, states).
        # For now, we can skip this or approximate.
        # Let's add a note that full validation requires state history.
        report['duration_ks_test'] = "Requires state history (not returned by run_simulation)"

        # 2. Volatility Autocorrelation (Clustering)
        # Compare ACF of squared returns
        real_rets = self.data['Log_Ret'].dropna().values
        real_vol_proxy = real_rets ** 2

        # Sim returns
        sim_rets = np.diff(np.log(simulated_paths), axis=1)
        sim_vol_proxy = sim_rets ** 2

        # Compute ACF lag 1
        def acf1(x):
            return np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0

        real_acf = acf1(real_vol_proxy)
        sim_acfs = [acf1(path) for path in sim_vol_proxy]
        avg_sim_acf = np.mean(sim_acfs)

        report['real_vol_acf_lag1'] = real_acf
        report['sim_vol_acf_lag1'] = avg_sim_acf
        report['vol_clustering_error'] = abs(real_acf - avg_sim_acf)

        return report

    def run_simulation(self, days=126, simulations=1000):
        """Runs Semi-Markov Monte Carlo simulation."""
        # ... (same as before) ...
        print(f"🎲 Simulating {days} days x {simulations} paths (Semi-Markov)...")

        last_price = self.data['Close'].iloc[-1]

        last_block_idx = self.data['block'].iloc[-1]
        current_state_duration = len(self.data[self.data['block'] == last_block_idx])
        current_state = int(self.data['State_Idx'].iloc[-1])

        sim_paths = np.zeros((simulations, days))

        for sim in range(simulations):
            remaining_duration = self.sample_residual_duration(current_state, current_state_duration)

            curr_s = current_state
            curr_t = 0
            log_rets = []

            first_step_dur = min(remaining_duration, days)
            r = self.sample_regime_returns(curr_s, first_step_dur)
            log_rets.extend(r)
            curr_t += first_step_dur

            if curr_t < days:
                probs = self.transition_matrix.loc[curr_s].values
                probs = probs / probs.sum()
                curr_s = np.random.choice(range(self.n_states), p=probs)

            while curr_t < days:
                dur = self.sample_duration(curr_s)
                step_dur = min(dur, days - curr_t)

                r = self.sample_regime_returns(curr_s, step_dur)
                log_rets.extend(r)
                curr_t += step_dur

                if curr_t >= days:
                    break

                probs = self.transition_matrix.loc[curr_s].values
                probs = probs / probs.sum()
                curr_s = np.random.choice(range(self.n_states), p=probs)

            cum_rets = np.cumsum(log_rets[:days])
            sim_paths[sim, :] = last_price * np.exp(cum_rets)

        return sim_paths

if __name__ == "__main__":
    model = SemiMarkovModel("SPY")
    model._process_data()
    model.fit_distributions()

    print(f"State 0 (Highest Risk) Mapped to: {model.state_map[0]}")

    paths = model.run_simulation(days=100, simulations=100)

    val = model.validate_model(paths)
    print("Validation:", val)

    plt.plot(paths.T, alpha=0.1, color='cyan')
    plt.title("Semi-Markov Paths")
    plt.show()
