import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import gamma, expon
from scipy.optimize import curve_fit

# Styles
plt.style.use('dark_background')
sns.set_palette("flare")

class SemiMarkovModel:
    def __init__(self, ticker, n_states=5):
        self.ticker = ticker
        self.n_states = n_states
        self.state_map = {0: "Crash", 1: "Bear", 2: "Flat", 3: "Bull", 4: "Rally"}
        self.data = None
        self.duration_data = {}  # Store durations for each state
        self.duration_params = {} # Store fitted parameters for each state
        self.transition_matrix = None # Conditional transition matrix (P_ii = 0)

    def _process_data(self, period="5y"):
        """Fetches data, identifies regimes, and extracts durations."""
        print(f"📡 Fetching {period} data for {self.ticker}...")
        df = yf.download(self.ticker, period=period, progress=False, auto_adjust=True)
        # Handle multi-index columns if present (common in recent yfinance)
        if isinstance(df.columns, pd.MultiIndex):
             # Assuming 'Close' is at level 0 and ticker is at level 1 or similar structure
             # Actually, auto_adjust=True usually returns single level if one ticker
             # But let's be safe.
             if 'Close' in df.columns.get_level_values(0):
                 df = df.xs(self.ticker, axis=1, level=1, drop_level=True) if self.ticker in df.columns.get_level_values(1) else df

        # Calculate Log Returns
        # Use 'Close' or 'Adj Close' depending on data availability
        col = 'Close' if 'Close' in df.columns else df.columns[0]
        df['Log_Ret'] = np.log(df[col] / df[col].shift(1))
        df = df.dropna()

        # 1. Regime Identification (State 0-4)
        try:
            df['State_Idx'] = pd.qcut(df['Log_Ret'], q=self.n_states, labels=False, duplicates='drop')
        except ValueError:
             # Fallback if too few unique values
             df['State_Idx'] = pd.cut(df['Log_Ret'], bins=self.n_states, labels=False)

        # 2. Duration Extraction
        # Identify blocks of consecutive states
        df['block'] = (df['State_Idx'] != df['State_Idx'].shift(1)).cumsum()

        # Group by block to get duration
        durations = df.groupby(['block', 'State_Idx']).size().reset_index(name='duration')

        # Store durations for each state
        self.duration_data = {s: [] for s in range(self.n_states)}
        for _, row in durations.iterrows():
            state = int(row['State_Idx'])
            duration = int(row['duration'])
            self.duration_data[state].append(duration)

        # 3. Transition Matrix (Conditional on Change)
        # We only care about transitions where state CHANGES
        # Filter df to only rows where block changes
        # actually, easier to use the 'durations' df which represents the sequence of states
        durations['next_state'] = durations['State_Idx'].shift(-1)

        # Calculate transitions from state i to j (where i != j is guaranteed by block logic)
        trans_counts = pd.crosstab(durations['State_Idx'], durations['next_state'])

        # Normalize to get probabilities
        self.transition_matrix = trans_counts.div(trans_counts.sum(axis=1), axis=0)

        # Ensure all states are present
        all_states = range(self.n_states)
        self.transition_matrix = self.transition_matrix.reindex(index=all_states, columns=all_states, fill_value=0)

        # Fill NaNs with equal probability (if a state is never left or never entered)
        # For a sink state (never left), we might force a transition to random state or stay (but this is semi-markov)
        # If row sum is 0 (state never appeared or never transitioned), assign uniform prob to others
        for i in all_states:
            if self.transition_matrix.loc[i].sum() == 0:
                 # Distribute equally among other states
                 probs = np.ones(self.n_states)
                 probs[i] = 0 # Can't go to self
                 probs = probs / probs.sum()
                 self.transition_matrix.loc[i] = probs

        self.data = df
        return df

    def fit_distributions(self):
        """Fits Gamma (high vol) and Exponential (low vol) distributions."""
        print("📊 Fitting duration distributions...")

        # Heuristic:
        # Extreme states (0: Crash, 4: Rally) -> High Vol/Momentum -> Gamma (memory)
        # Middle states (2: Flat) -> Random noise -> Exponential (memoryless-ish)
        # Intermediate (1: Bear, 3: Bull) -> Try Gamma

        for state in range(self.n_states):
            durations = self.duration_data[state]
            if not durations:
                # Fallback if no data
                self.duration_params[state] = {'dist': 'expon', 'params': (0, 1)}
                continue

            # For states 0 (Crash) and 1 (Bear), and maybe 4 (Rally), we expect clustering/memory
            # so we fit Gamma.
            # For State 2 (Flat), maybe Exponential.

            # Let's just fit Gamma for all, as Gamma(1, scale) reduces to Exponential
            # shape (a), loc, scale
            # We enforce loc=0 because duration >= 1
            # Actually, duration is discrete >= 1. Continuous approx is fine.
            # We'll shift data by -1? Or just fit.

            # Gamma fit
            try:
                # fix loc=0 since duration is length of time
                # but durations start at 1.
                # Let's fit to data
                a, loc, scale = gamma.fit(durations, floc=0)
                self.duration_params[state] = {'dist': 'gamma', 'params': (a, loc, scale)}
            except:
                 # Fallback
                 self.duration_params[state] = {'dist': 'expon', 'params': (0, np.mean(durations))}

        print("✅ Distributions fitted.")
        for s, p in self.duration_params.items():
            print(f"   State {s} ({self.state_map[s]}): {p['dist']} {p['params']}")

    def sample_duration(self, state):
        """Samples a duration for the given state."""
        params = self.duration_params[state]
        if params['dist'] == 'gamma':
            a, loc, scale = params['params']
            d = gamma.rvs(a, loc=loc, scale=scale)
        else:
            loc, scale = params['params']
            d = expon.rvs(loc=loc, scale=scale)

        # Duration must be at least 1 integer day
        return max(1, int(round(d)))

    def run_simulation(self, days=126, simulations=1000):
        """Runs Semi-Markov Monte Carlo simulation."""
        print(f"🎲 Simulating {days} days x {simulations} paths (Semi-Markov)...")

        last_price = self.data['Close'].iloc[-1]
        if isinstance(last_price, pd.Series): # Handle if it's a series
             last_price = last_price.item()

        # Initial state logic: We need to know current state and how long we've been in it
        # The prompt says: "Day 30 of low-vol is different from Day 1"
        # So we should calculate current regime duration.

        last_block_idx = self.data['block'].iloc[-1]
        current_state_duration = len(self.data[self.data['block'] == last_block_idx])
        current_state = int(self.data['State_Idx'].iloc[-1])

        # Pre-compute return pools
        return_pools = {
            state: self.data[self.data['State_Idx'] == state]['Log_Ret'].values
            for state in range(self.n_states)
        }

        sim_paths = np.zeros((simulations, days))

        for sim in range(simulations):
            # For each simulation, we start from current state
            # But we need to decide how much LONGER we stay in current state.
            # Conditional duration: P(T > t_current + t | T > t_current)
            # Simplified: Sample full duration D, if D > current_duration, remain = D - current.
            # If D <= current, maybe we transition immediately (1 day)

            # Resample until we get a duration > current_duration?
            # Or just sample a fresh duration?
            # "Memory" implies we use the conditional distribution.

            # Let's try resampling for the first step
            # Limit retries to avoid infinite loop
            remaining_duration = 0
            for _ in range(20):
                d = self.sample_duration(current_state)
                if d > current_state_duration:
                    remaining_duration = d - current_state_duration
                    break
            if remaining_duration == 0:
                remaining_duration = 1 # Force at least 1 day if we assume we are at end

            curr_s = current_state
            curr_t = 0
            log_rets = []

            while curr_t < days:
                # If this is the first step, use calculated remaining_duration
                # Otherwise sample new full duration
                if curr_t == 0:
                    dur = remaining_duration
                else:
                    dur = self.sample_duration(curr_s)

                # Cap duration if it exceeds simulation horizon
                steps = min(dur, days - curr_t)

                # Sample returns for this state
                # Using random choice from historical returns of this state
                if len(return_pools[curr_s]) > 0:
                    r = np.random.choice(return_pools[curr_s], size=steps)
                else:
                    r = np.zeros(steps) # Fallback

                log_rets.extend(r)
                curr_t += steps

                if curr_t >= days:
                    break

                # Transition to NEW state (different from curr_s)
                # Use conditional transition matrix
                probs = self.transition_matrix.loc[curr_s].values
                # Probs should sum to 1, but check for float errors
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
    paths = model.run_simulation(days=100, simulations=100)
    print(f"Generated {paths.shape} paths.")

    plt.plot(paths.T, alpha=0.1, color='cyan')
    plt.title("Semi-Markov Paths")
    plt.show()
