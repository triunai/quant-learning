import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Styles
plt.style.use('dark_background')
sns.set_palette("flare")

class BlendedMarkovModel:
    def __init__(self, ticker, n_states=5):
        self.ticker = ticker
        self.n_states = n_states
        self.state_map = {0: "Crash", 1: "Bear", 2: "Flat", 3: "Bull", 4: "Rally"}
        self.data_10y = None
        self.data_5y = None
        self.matrix_10y = None
        self.matrix_5y = None
        self.matrix_blended = None

    def _process_data(self, period):
        """Internal helper to fetch and bin data robustly."""
        print(f"📡 Fetching {period} data for {self.ticker}...")
        df = yf.download(self.ticker, period=period, progress=False, auto_adjust=True)
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df = df.dropna()

        # 1. ROBUST BINNING: Use numeric labels internally (0-4)
        # This prevents string label sorting errors
        df['State_Idx'] = pd.qcut(df['Log_Ret'], q=self.n_states, labels=False)

        # Calculate transitions
        df['Next_State_Idx'] = df['State_Idx'].shift(-1)

        # 2. MATRIX STABILIZATION: Reindex to ensure 5x5 grid even if states are missing
        matrix = pd.crosstab(df['State_Idx'], df['Next_State_Idx'], normalize='index')
        all_states = range(self.n_states)
        matrix = matrix.reindex(index=all_states, columns=all_states, fill_value=0)

        return df, matrix

    def build_models(self):
        # Build independent chains
        self.data_10y, self.matrix_10y = self._process_data("10y")
        self.data_5y, self.matrix_5y = self._process_data("5y")
        print("✅ Base models built.")

    def blend_matrices(self, weight_5y=0.7):
        """
        The 'Quant' Upgrade: Blends short-term regime (5y) with long-term structure (10y).
        weight_5y: 0.7 means 70% influence from recent data.
        """
        print(f"⚖️ Blending Matrices: {weight_5y*100}% 5Y + {(1-weight_5y)*100}% 10Y")
        self.matrix_blended = (self.matrix_5y * weight_5y) + (self.matrix_10y * (1 - weight_5y))

        # Re-normalize just to be mathematically perfect (floating point errors)
        self.matrix_blended = self.matrix_blended.div(self.matrix_blended.sum(axis=1), axis=0)

    def visualize_blended_matrix(self):
        fig, ax = plt.subplots(figsize=(10, 8))

        # Remap index/columns to names for the plot
        plot_matrix = self.matrix_blended.copy()
        plot_matrix.index = [self.state_map[i] for i in plot_matrix.index]
        plot_matrix.columns = [self.state_map[i] for i in plot_matrix.columns]

        sns.heatmap(plot_matrix, annot=True, fmt=".2f", cmap="magma", ax=ax)
        ax.set_title(f"Blended Transition Matrix ({self.ticker})", fontsize=14, color='white')
        plt.show()

    def run_simulation(self, days=126, simulations=5000):
        """
        VECTORIZED Monte Carlo simulation for speed.
        Uses Blended Matrix for transitions + 5Y Empirical Data for returns.
        """
        last_price = self.data_5y['Close'].iloc[-1].item()
        last_state = int(self.data_5y['State_Idx'].iloc[-1])

        print(f"🎲 Simulating {days} days x {simulations} paths (vectorized)...")

        # Pre-compute: Extract return pools for each state (avoids repeated filtering)
        return_pools = {
            state: self.data_5y[self.data_5y['State_Idx'] == state]['Log_Ret'].values
            for state in range(self.n_states)
        }

        # Pre-compute: Convert transition matrix to numpy for speed
        trans_matrix = self.matrix_blended.values

        # Initialize arrays
        log_returns = np.zeros((simulations, days))
        states = np.full(simulations, last_state, dtype=int)

        for d in range(days):
            # Vectorized state transitions for ALL simulations at once
            new_states = np.array([
                np.random.choice(self.n_states, p=trans_matrix[s])
                for s in states
            ])

            # Vectorized return sampling from pre-computed pools
            returns = np.array([
                np.random.choice(return_pools[s])
                for s in new_states
            ])

            log_returns[:, d] = returns
            states = new_states

        # Vectorized price path calculation using cumulative sum of log returns
        cumulative_log_returns = np.cumsum(log_returns, axis=1)
        sim_paths = last_price * np.exp(cumulative_log_returns)

        return sim_paths

    def get_current_info(self):
        """Returns current price, date, and state for display."""
        last_price = self.data_5y['Close'].iloc[-1].item()
        last_date = self.data_5y.index[-1]
        last_state = int(self.data_5y['State_Idx'].iloc[-1])
        return {
            'price': last_price,
            'date': last_date,
            'state': last_state,
            'state_name': self.state_map[last_state]
        }

# --- EXECUTION ---
TICKER = "NVO"
FORECAST_DAYS = 126  # ~6 months of trading days
SIMULATIONS = 5000

print("="*60)
print("🚀 BLENDED MARKOV STOCK FORECASTER")
print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("="*60)

model = BlendedMarkovModel(TICKER)
model.build_models()
model.blend_matrices(weight_5y=0.7)  # 70% weight to recent behavior

# Get current market context
info = model.get_current_info()
print(f"\n📈 CURRENT MARKET SNAPSHOT ({TICKER})")
print(f"   Last Data Date: {info['date'].strftime('%Y-%m-%d')}")
print(f"   Current Price:  ${info['price']:.2f}")
print(f"   Current State:  {info['state_name']} (Regime {info['state']})")

# 1. Show the "Brain" of the model
model.visualize_blended_matrix()

# 2. Run Forecast
paths = model.run_simulation(days=FORECAST_DAYS, simulations=SIMULATIONS)

# 3. Advanced Analysis
final_prices = paths[:, -1]
current_price = info['price']

# Key percentiles
p5 = np.percentile(final_prices, 5)
p10 = np.percentile(final_prices, 10)
p25 = np.percentile(final_prices, 25)
median_price = np.median(final_prices)
p75 = np.percentile(final_prices, 75)
p90 = np.percentile(final_prices, 90)
p95 = np.percentile(final_prices, 95)

# CVaR (Expected Shortfall)
cvar_5 = np.mean(final_prices[final_prices <= p5])

# Expected returns
median_return = ((median_price - current_price) / current_price) * 100

# Visualization with more context
plt.figure(figsize=(14, 7))
sns.histplot(final_prices, kde=True, color="cyan", bins=50, alpha=0.7)

# Add reference lines
plt.axvline(current_price, color='lime', linestyle='-', linewidth=2,
            label=f'Current: ${current_price:.2f}')
plt.axvline(median_price, color='white', linestyle='--', linewidth=2,
            label=f'Median: ${median_price:.2f} ({median_return:+.1f}%)')
plt.axvline(cvar_5, color='red', linestyle='-', linewidth=2,
            label=f'CVaR 5%: ${cvar_5:.2f}')
plt.axvline(p75, color='gold', linestyle=':', linewidth=1.5,
            label=f'75th Pctl: ${p75:.2f}')

plt.title(f"{TICKER} {FORECAST_DAYS}-Day Forecast Distribution\n"
          f"Analysis: {datetime.now().strftime('%Y-%m-%d')} | Current: ${current_price:.2f}",
          fontsize=14)
plt.xlabel("Forecasted Price (USD)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.show()

# Print comprehensive statistics
print(f"\n{'='*60}")
print(f"📊 FORECAST STATISTICS ({FORECAST_DAYS} trading days ≈ 6 months)")
print(f"{'='*60}")
print("\n🎯 PRICE FORECASTS:")
print(f"   Current Price:    ${current_price:.2f}")
print(f"   Median Forecast:  ${median_price:.2f} ({median_return:+.1f}%)")
print("\n📉 DOWNSIDE SCENARIOS:")
print(f"   CVaR (5%):        ${cvar_5:.2f} ({((cvar_5-current_price)/current_price)*100:+.1f}%)  ← Worst 5% avg")
print(f"   10th Percentile:  ${p10:.2f} ({((p10-current_price)/current_price)*100:+.1f}%)")
print(f"   25th Percentile:  ${p25:.2f} ({((p25-current_price)/current_price)*100:+.1f}%)")
print("\n📈 UPSIDE SCENARIOS:")
print(f"   75th Percentile:  ${p75:.2f} ({((p75-current_price)/current_price)*100:+.1f}%)")
print(f"   90th Percentile:  ${p90:.2f} ({((p90-current_price)/current_price)*100:+.1f}%)")
print(f"   95th Percentile:  ${p95:.2f} ({((p95-current_price)/current_price)*100:+.1f}%)")
print("\n⚡ RISK METRICS:")
print(f"   Volatility (σ):   {np.std(final_prices):.2f}")
print(f"   Range:            ${np.min(final_prices):.2f} - ${np.max(final_prices):.2f}")
print(f"{'='*60}")
