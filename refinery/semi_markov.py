import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import gamma, expon
from sklearn.cluster import KMeans

# Styles
plt.style.use('dark_background')
sns.set_palette("flare")

class SemiMarkovModel:
    def __init__(self, ticker, n_states=5):
        self.ticker = ticker
        self.n_states = n_states
        # State map will be dynamically assigned after clustering
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

        # --- Sort Clusters to ensure 0=Crash, 4=Rally ---
        # Sort by Mean 60d Return (Momentum)
        cluster_stats = df_clean.groupby('Cluster')['Ret_60d'].mean().sort_values()

        # Create mapping: Old Label -> New Sorted Label (0..4)
        mapping = {old_label: new_label for new_label, old_label in enumerate(cluster_stats.index)}
        df_clean['State_Idx'] = df_clean['Cluster'].map(mapping)

        # Define state map based on sorted order
        base_names = ["Crash", "Bear", "Flat", "Bull", "Rally"]
        # If n_states != 5, we might need generic names, but assuming 5 for now
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

    def fit_distributions(self):
        """Fits Gamma (high vol) and Exponential (low vol) distributions."""
        print("📊 Fitting duration distributions...")
        for state in range(self.n_states):
            durations = self.duration_data[state]
            if not durations:
                self.duration_params[state] = {'dist': 'expon', 'params': (0, 1)}
                continue

            # Fit Gamma (general case)
            try:
                # floc=0 enforces location at 0 (duration represents length)
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
        """
        Samples remaining duration conditional on already being in state for elapsed_days.
        Uses Inverse Transform Sampling for Gamma: T = F^{-1}(U + F(t)) - t
        """
        params = self.duration_params[state]
        if params['dist'] == 'expon':
            # Memoryless property: Residual life distribution is same as original
            return self.sample_duration(state)

        # Gamma case
        a, loc, scale = params['params']

        # Calculate CDF at current elapsed time (probability we exited before now)
        cdf_t = gamma.cdf(elapsed_days, a, loc=loc, scale=scale)

        # If cdf_t is very close to 1, we are in the extreme tail.
        # Fallback to simple rejection or just return 1 to avoid numerical instability
        if cdf_t > 0.999:
             return 1

        # Sample U ~ Uniform(CDF(t), 1)
        # This represents the cumulative probability of the TOTAL duration
        u = np.random.uniform(cdf_t, 1.0)

        # Invert CDF to get total duration
        total_duration = gamma.ppf(u, a, loc=loc, scale=scale)

        remaining = total_duration - elapsed_days
        return max(1, int(round(remaining)))

    def sample_regime_returns(self, state, n_days):
        """
        Samples a block of returns from the historical data for the given state.
        Preserves autocorrelation/volatility clustering better than random sampling.
        """
        # Get all returns classified as this state
        # Note: These are not necessarily contiguous in time across the whole dataset,
        # but locally they are.
        # Ideally, we pick a specific historical occurrence of this state.

        state_instances = self.data[self.data['State_Idx'] == state]

        if len(state_instances) == 0:
            return np.zeros(n_days)

        # If we just grab all returns, they are disjoint.
        # Better: Pick a random start index in the filtered array?
        # The user audit suggested: "Sample a STARTING point, then take n_days CONSECUTIVE returns"
        # from the filtered array. This implies ignoring the disjoint nature, which is a common approx.
        # Let's implement that for now.

        state_rets = state_instances['Log_Ret'].values
        if len(state_rets) < n_days:
            # Not enough history, wrap around or repeat
            return np.resize(state_rets, n_days)

        start_idx = np.random.randint(0, len(state_rets) - n_days + 1)
        return state_rets[start_idx : start_idx + n_days]

    def regime_fatigue_score(self, state, days_in_state):
        """
        Calculates a 0-1 score indicating how 'tired' the regime is.
        Score = CDF(duration). High score = High probability of exit.
        """
        params = self.duration_params.get(state)
        if not params:
            return 0.5

        if params['dist'] == 'gamma':
            a, loc, scale = params['params']
            return gamma.cdf(days_in_state, a, loc=loc, scale=scale)
        else:
            # Exponential is memoryless, so "fatigue" is constant/undefined?
            # Or we can interpret as CDF too.
            # But technically hazard rate is constant.
            # Let's return CDF for consistency (probability we shouldn't have lasted this long)
            loc, scale = params['params']
            return expon.cdf(days_in_state, loc=loc, scale=scale)

    def run_simulation(self, days=126, simulations=1000):
        """Runs Semi-Markov Monte Carlo simulation with residual duration logic."""
        print(f"🎲 Simulating {days} days x {simulations} paths (Semi-Markov)...")

        last_price = self.data['Close'].iloc[-1]

        # Determine current state context
        last_block_idx = self.data['block'].iloc[-1]
        current_state_duration = len(self.data[self.data['block'] == last_block_idx])
        current_state = int(self.data['State_Idx'].iloc[-1])

        sim_paths = np.zeros((simulations, days))

        for sim in range(simulations):
            # 1. Determine remaining duration in CURRENT state
            remaining_duration = self.sample_residual_duration(current_state, current_state_duration)

            curr_s = current_state
            curr_t = 0
            log_rets = []

            # First block (current state)
            first_step_dur = min(remaining_duration, days)
            r = self.sample_regime_returns(curr_s, first_step_dur)
            log_rets.extend(r)
            curr_t += first_step_dur

            # Transition from current state
            if curr_t < days:
                # Transition logic
                probs = self.transition_matrix.loc[curr_s].values
                probs = probs / probs.sum()
                curr_s = np.random.choice(range(self.n_states), p=probs)

            # Subsequent blocks
            while curr_t < days:
                dur = self.sample_duration(curr_s)
                step_dur = min(dur, days - curr_t)

                r = self.sample_regime_returns(curr_s, step_dur)
                log_rets.extend(r)
                curr_t += step_dur

                if curr_t >= days:
                    break

                # Transition
                probs = self.transition_matrix.loc[curr_s].values
                probs = probs / probs.sum()
                curr_s = np.random.choice(range(self.n_states), p=probs)

            # Convert to price path
            cum_rets = np.cumsum(log_rets[:days])
            sim_paths[sim, :] = last_price * np.exp(cum_rets)

        return sim_paths

if __name__ == "__main__":
    # Quick sanity check
    model = SemiMarkovModel("SPY")
    model._process_data()
    model.fit_distributions()

    # Check fatigue
    print(f"Fatigue for State 0 at 5 days: {model.regime_fatigue_score(0, 5):.2f}")
    print(f"Fatigue for State 0 at 50 days: {model.regime_fatigue_score(0, 50):.2f}")

    paths = model.run_simulation(days=100, simulations=100)
    print(f"Generated {paths.shape} paths.")

    plt.plot(paths.T, alpha=0.1, color='cyan')
    plt.title("Semi-Markov Paths")
    plt.show()
