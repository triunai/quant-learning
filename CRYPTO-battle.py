"""
🪙 CRYPTO BATTLE MODEL: Reality vs Theory
==========================================
Applies the Markov vs GBM battle framework to cryptocurrency.
Crypto has the WIDEST "Reality vs Theory" gap due to:
- 24/7 trading (no overnight gaps baked in)
- Extreme sentiment-driven momentum
- Fat tails that make stocks look tame
- Halving cycles and macro regime shifts
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set aesthetic
plt.style.use('dark_background')
sns.set_palette("viridis")

class CryptoBattleModel:
    """
    Battle Model optimized for Crypto assets.
    Key differences from stock model:
    - Uses 365-day annualization (crypto trades 24/7)
    - More aggressive regime detection
    - Includes "Mania" state for crypto-specific behavior
    """
    
    def __init__(self, ticker="BTC-USD", days_ahead=126, simulations=5000, target_price=None):
        self.ticker = ticker
        self.days_ahead = days_ahead
        self.simulations = simulations
        self.target_price = target_price  # If None, will auto-calculate
        
        self.data = None
        self.markov_matrix = None
        self.mu = 0
        self.sigma = 0
        self.last_price = 0
        self.last_date = None
        self.current_state = 0
        
        # 6 regimes for crypto (more extreme than stocks)
        self.n_states = 6
        self.state_map = {
            0: "Capitulation",  # Extreme fear
            1: "Bear", 
            2: "Accumulation",  # Quiet consolidation
            3: "Bull", 
            4: "Rally",
            5: "Mania"  # FOMO peak
        }

    def ingest_data(self):
        print(f"📡 Connecting to {self.ticker} oracle...")
        self.data = yf.download(self.ticker, period="max", auto_adjust=True, progress=False)
        
        # Calculate Log Returns
        self.data['Log_Ret'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.data = self.data.dropna()
        self.last_price = self.data['Close'].iloc[-1].item()
        self.last_date = self.data.index[-1]
        
        # Crypto trades 365 days (use 365 for annualization)
        self.mu = self.data['Log_Ret'].mean() * 365
        self.sigma = self.data['Log_Ret'].std() * np.sqrt(365)
        
        # Auto-set target if not provided (2x current price)
        if self.target_price is None:
            self.target_price = self.last_price * 2
        
        print(f"✅ Loaded {len(self.data)} days of {self.ticker} history")
        print(f"   First date: {self.data.index[0].strftime('%Y-%m-%d')}")
        print(f"   Last date:  {self.last_date.strftime('%Y-%m-%d')}")

    def build_markov_brain(self):
        # Discretize into 6 states
        self.data['State_Idx'] = pd.qcut(self.data['Log_Ret'], q=self.n_states, labels=False)
        self.current_state = int(self.data['State_Idx'].iloc[-1])
        
        # Build Transition Matrix
        self.data['Next_State'] = self.data['State_Idx'].shift(-1)
        matrix = pd.crosstab(self.data['State_Idx'], self.data['Next_State'], normalize='index')
        
        # Stabilize matrix
        self.markov_matrix = matrix.reindex(
            index=range(self.n_states), 
            columns=range(self.n_states), 
            fill_value=0
        )
        print(f"🧠 Crypto Markov brain built: {self.n_states} regimes")

    def simulate_gbm(self):
        """GBM with crypto-adjusted parameters (365-day year)"""
        dt = 1/365  # Crypto trades every day
        Z = np.random.normal(0, 1, (self.simulations, self.days_ahead))
        
        drift_term = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion_term = self.sigma * np.sqrt(dt) * Z
        
        daily_returns = np.exp(drift_term + diffusion_term)
        price_paths = self.last_price * np.cumprod(daily_returns, axis=1)
            
        return price_paths

    def simulate_markov(self):
        """Markov with empirical fat-tail sampling"""
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
        price_paths = self.last_price * np.exp(cumulative_log_returns)
            
        return price_paths

    def analyze_first_passage_time(self, paths):
        hits = np.any(paths >= self.target_price, axis=1)
        raw_times = np.argmax(paths >= self.target_price, axis=1)
        valid_times = raw_times[hits]
        prob_success = np.mean(hits)
        return prob_success, valid_times

    def calculate_regime_metrics(self):
        """Calculate key regime stickiness metrics"""
        matrix = self.markov_matrix.values
        
        # Diagonal values = regime persistence
        persistence = {self.state_map[i]: matrix[i, i] for i in range(self.n_states)}
        
        # Mania stickiness (key for crypto)
        mania_to_mania = matrix[5, 5] if self.n_states > 5 else 0
        cap_to_cap = matrix[0, 0]
        
        return {
            'persistence': persistence,
            'mania_stickiness': mania_to_mania,
            'capitulation_stickiness': cap_to_cap,
            'mania_to_cap_prob': matrix[5, 0] if self.n_states > 5 else 0,  # Crash risk from top
        }

    def run_battle(self):
        print(f"\n{'='*70}")
        print(f"🪙 CRYPTO MODEL BATTLE: {self.ticker}")
        print(f"{'='*70}")
        print(f"📅 Analysis Date:    {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"📊 Last Data Date:   {self.last_date.strftime('%Y-%m-%d')}")
        print(f"💰 Current Price:    ${self.last_price:,.2f}")
        print(f"🎯 Target Price:     ${self.target_price:,.2f} ({((self.target_price/self.last_price)-1)*100:+.1f}%)")
        print(f"📈 Current Regime:   {self.state_map[self.current_state]}")
        print(f"⏱️  Horizon:          {self.days_ahead} days (~{self.days_ahead//30} months)")
        print(f"🎲 Simulations:      {self.simulations:,}")
        print(f"{'='*70}")
        
        print(f"\n📊 GBM Parameters (Annualized @ 365 days):")
        print(f"   Drift (μ):        {self.mu:.2%}")
        print(f"   Volatility (σ):   {self.sigma:.2%}")
        
        # Regime analysis
        metrics = self.calculate_regime_metrics()
        print(f"\n🔥 CRYPTO REGIME ANALYSIS:")
        print(f"   Mania → Mania:       {metrics['mania_stickiness']:.1%} (FOMO persistence)")
        print(f"   Capitulation → Cap:  {metrics['capitulation_stickiness']:.1%} (Fear persistence)")
        print(f"   Mania → Capitulation: {metrics['mania_to_cap_prob']:.1%} (Crash risk from top)")
        
        print(f"\n🎲 Running simulations...")
        paths_gbm = self.simulate_gbm()
        paths_markov = self.simulate_markov()
        
        prob_gbm, times_gbm = self.analyze_first_passage_time(paths_gbm)
        prob_markov, times_markov = self.analyze_first_passage_time(paths_markov)
        
        # Calculate "Reality Premium" - the edge
        reality_premium = prob_markov - prob_gbm
        
        print(f"\n{'='*70}")
        print(f"🏆 BATTLE RESULTS: Probability of hitting ${self.target_price:,.0f}")
        print(f"{'='*70}")
        print(f"   GBM (Theory):     {prob_gbm:.1%}")
        print(f"   Markov (Reality): {prob_markov:.1%}")
        print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"   📈 REALITY PREMIUM: {reality_premium:+.1%}")
        if reality_premium > 0:
            print(f"   → Market is UNDERPRICING upside potential!")
        else:
            print(f"   → Market is OVERPRICING upside potential!")
        
        # Final price distributions
        print(f"\n📊 FINAL PRICE DISTRIBUTIONS (Day {self.days_ahead}):")
        print(f"\n   {'GBM (Theory)':<20} {'Markov (Reality)':<20}")
        print(f"   {'-'*40}")
        print(f"   5th %:   ${np.percentile(paths_gbm[:, -1], 5):>10,.0f}    ${np.percentile(paths_markov[:, -1], 5):>10,.0f}")
        print(f"   25th %:  ${np.percentile(paths_gbm[:, -1], 25):>10,.0f}    ${np.percentile(paths_markov[:, -1], 25):>10,.0f}")
        print(f"   Median:  ${np.median(paths_gbm[:, -1]):>10,.0f}    ${np.median(paths_markov[:, -1]):>10,.0f}")
        print(f"   75th %:  ${np.percentile(paths_gbm[:, -1], 75):>10,.0f}    ${np.percentile(paths_markov[:, -1], 75):>10,.0f}")
        print(f"   95th %:  ${np.percentile(paths_gbm[:, -1], 95):>10,.0f}    ${np.percentile(paths_markov[:, -1], 95):>10,.0f}")
        print(f"   MAX:     ${np.max(paths_gbm[:, -1]):>10,.0f}    ${np.max(paths_markov[:, -1]):>10,.0f}")
        print(f"{'='*70}")
        
        # --- VISUALIZATION ---
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
        
        # 1. Cone Battle
        ax1 = fig.add_subplot(gs[0, :])
        x = np.arange(self.days_ahead)
        
        # Markov
        p50_m = np.median(paths_markov, axis=0)
        p05_m = np.percentile(paths_markov, 5, axis=0)
        p95_m = np.percentile(paths_markov, 95, axis=0)
        ax1.plot(x, p50_m, color='cyan', lw=2.5, label='Markov Median (Reality)')
        ax1.fill_between(x, p05_m, p95_m, color='cyan', alpha=0.15, label='Markov 90% Range')
        
        # GBM
        p50_g = np.median(paths_gbm, axis=0)
        p05_g = np.percentile(paths_gbm, 5, axis=0)
        p95_g = np.percentile(paths_gbm, 95, axis=0)
        ax1.plot(x, p50_g, color='orange', lw=2, ls='--', label='GBM Median (Theory)')
        ax1.fill_between(x, p05_g, p95_g, color='orange', alpha=0.1, label='GBM 90% Range')
        
        ax1.axhline(self.target_price, color='lime', ls=':', lw=2, label=f'Target ${self.target_price:,.0f}')
        ax1.axhline(self.last_price, color='white', ls='-', lw=1, alpha=0.5, label=f'Current ${self.last_price:,.0f}')
        
        ax1.set_title(f"🪙 {self.ticker} Model Battle: {self.days_ahead}-Day Forecast\n"
                      f"Analysis: {datetime.now().strftime('%Y-%m-%d')} | Current: ${self.last_price:,.0f} | Target: ${self.target_price:,.0f}", 
                      fontsize=14)
        ax1.set_xlabel("Days", fontsize=11)
        ax1.set_ylabel("Price (USD)", fontsize=11)
        ax1.legend(fontsize=10, loc='upper left')
        ax1.grid(alpha=0.2)
        ax1.set_yscale('log')  # Log scale for crypto
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 2. First Passage Time
        ax2 = fig.add_subplot(gs[1, 0])
        if len(times_markov) > 10 and len(times_gbm) > 10:
            sns.kdeplot(times_markov, color='cyan', fill=True, ax=ax2, 
                        label=f'Markov (Prob: {prob_markov:.0%})', alpha=0.5)
            sns.kdeplot(times_gbm, color='orange', fill=True, ax=ax2, 
                        label=f'GBM (Prob: {prob_gbm:.0%})', alpha=0.5)
            
            # Add median lines
            if len(times_markov) > 0:
                ax2.axvline(np.median(times_markov), color='cyan', ls='--', 
                           label=f'Markov Median: {np.median(times_markov):.0f} days')
            if len(times_gbm) > 0:
                ax2.axvline(np.median(times_gbm), color='orange', ls='--',
                           label=f'GBM Median: {np.median(times_gbm):.0f} days')
            
            ax2.set_title(f"⏱️ First Passage Time: WHEN do we hit ${self.target_price:,.0f}?", fontsize=12)
            ax2.set_xlabel("Days from Today")
            ax2.set_ylabel("Density")
            ax2.set_xlim(0, self.days_ahead)
            ax2.legend(fontsize=9)
        else:
            hit_text = f"Target ${self.target_price:,.0f} rarely hit in {self.days_ahead} days\n"
            hit_text += f"GBM: {prob_gbm:.1%} | Markov: {prob_markov:.1%}"
            ax2.text(0.5, 0.5, hit_text, ha='center', va='center', 
                    color='yellow', fontsize=14, transform=ax2.transAxes)
            ax2.set_title("First Passage Time Analysis", fontsize=12)
            
        # 3. Heatmap
        ax3 = fig.add_subplot(gs[1, 1])
        plot_matrix = self.markov_matrix.copy()
        plot_matrix.index = [self.state_map[i] for i in plot_matrix.index]
        plot_matrix.columns = [self.state_map[i] for i in plot_matrix.columns]
        
        sns.heatmap(plot_matrix, annot=True, fmt=".2f", cmap="inferno", ax=ax3,
                    cbar_kws={'label': 'Transition Probability'})
        ax3.set_title(f"🧠 {self.ticker} Regime Engine (6-State Markov)", fontsize=12)
        ax3.set_ylabel("Today's State")
        ax3.set_xlabel("Tomorrow's State")
        
        plt.tight_layout()
        plt.show()
        
        return {
            'prob_gbm': prob_gbm,
            'prob_markov': prob_markov,
            'reality_premium': reality_premium,
            'paths_gbm': paths_gbm,
            'paths_markov': paths_markov
        }


# --- EXECUTE ---
if __name__ == "__main__":
    print("="*70)
    print("🪙 CRYPTO BATTLE MODEL: Bitcoin Edition")
    print(f"📅 Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*70)
    
    # BTC Configuration
    battle = CryptoBattleModel(
        ticker="BTC-USD",
        target_price=150000,  # $150K target
        days_ahead=180,       # 6 months
        simulations=5000
    )
    battle.ingest_data()
    battle.build_markov_brain()
    results = battle.run_battle()
    
    print("\n" + "="*70)
    print("🔮 RUNNING ETH COMPARISON...")
    print("="*70)
    
    # ETH Configuration  
    eth_battle = CryptoBattleModel(
        ticker="ETH-USD",
        target_price=5000,   # $5K target
        days_ahead=180,
        simulations=5000
    )
    eth_battle.ingest_data()
    eth_battle.build_markov_brain()
    eth_results = eth_battle.run_battle()
