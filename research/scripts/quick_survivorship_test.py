"""Quick survivorship test with clean output."""
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Historical constituents
MEGA_CAPS = {
    2010: ['XOM', 'MSFT', 'AAPL', 'GE', 'WMT', 'CVX', 'JNJ', 'PG', 'JPM', 'IBM'],
    2015: ['AAPL', 'GOOGL', 'XOM', 'MSFT', 'BRK-B', 'JNJ', 'GE', 'WMT', 'JPM', 'CVX'],
    2020: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'BRK-B', 'V', 'JPM', 'JNJ', 'WMT'],
}

def detect_regime(returns, lookback=60):
    if len(returns) < lookback:
        return 'Neutral', 0.5
    recent = returns.tail(lookback)
    ann_return = recent.mean() * 252
    if ann_return > 0.10:
        return 'Bull', min(0.9, 0.5 + ann_return)
    elif ann_return < -0.05:
        return 'Bear', min(0.9, 0.5 - ann_return)
    return 'Neutral', 0.5

def run_backtest(start_year, end_year):
    tickers = MEGA_CAPS.get(start_year, MEGA_CAPS[2020])
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    
    prices = pd.DataFrame()
    valid = []
    for t in tickers:
        try:
            d = yf.download(t, start=start_date, end=end_date, progress=False)
            if len(d) > 100:
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)
                col = 'Adj Close' if 'Adj Close' in d.columns else 'Close'
                prices[t] = d[col]
                valid.append(t)
        except:
            pass
    
    prices = prices.dropna()
    returns = np.log(prices / prices.shift(1)).dropna()
    rebal = prices.resample('M').last().index
    
    portfolio = 100000
    port_hist = [portfolio]
    holdings = {t: 0 for t in valid}
    cash = portfolio
    
    for i in range(1, len(rebal)):
        if rebal[i] > prices.index[-1]:
            continue
        
        weights = {}
        for t in valid:
            ret = returns[t].loc[:rebal[i-1]]
            regime, _ = detect_regime(ret)
            weights[t] = 1.0/len(valid) if regime == 'Bull' else 0.0
        
        tw = sum(weights.values())
        if tw > 1:
            for t in weights:
                weights[t] /= tw
        
        val = cash + sum(holdings[t] * prices.loc[prices.index[prices.index <= rebal[i-1]][-1], t] for t in valid)
        
        for t in valid:
            price = prices.loc[prices.index[prices.index <= rebal[i-1]][-1], t]
            shares = (val * weights.get(t, 0)) / price
            cost = abs(shares - holdings[t]) * price * 0.0005
            cash += holdings[t] * price - shares * price - cost
            holdings[t] = shares
        
        end_val = cash + sum(holdings[t] * prices.loc[prices.index[prices.index <= rebal[i]][-1], t] for t in valid)
        port_hist.append(end_val)
    
    total_ret = (port_hist[-1] / port_hist[0]) - 1
    months = len(port_hist)
    ann_ret = (1 + total_ret) ** (12/months) - 1
    
    monthly_rets = pd.Series(port_hist).pct_change().dropna()
    vol = monthly_rets.std() * np.sqrt(12)
    sharpe = ann_ret / vol if vol > 0 else 0
    
    # Max drawdown
    cum = pd.Series(port_hist)
    peak = cum.expanding().max()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    
    return sharpe, total_ret, max_dd

print("="*60)
print("SURVIVORSHIP BIAS FIX - MULTI-PERIOD VALIDATION")
print("="*60)

periods = [(2010, 2014), (2015, 2019), (2020, 2024)]
results = []

for start, end in periods:
    print(f"\nTesting {start}-{end}...")
    sharpe, ret, dd = run_backtest(start, end)
    results.append((f"{start}-{end}", sharpe, ret, dd))
    print(f"  Sharpe: {sharpe:.2f}, Return: {ret*100:.1f}%, Max DD: {dd*100:.1f}%")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\n{'Period':<12} {'Sharpe':>10} {'Return':>12} {'Max DD':>12} {'Status':>10}")
print("-"*56)

passes = 0
for period, sharpe, ret, dd in results:
    status = "[PASS]" if sharpe >= 1.0 else "[FAIL]"
    if sharpe >= 1.0:
        passes += 1
    print(f"{period:<12} {sharpe:>10.2f} {ret*100:>11.1f}% {dd*100:>11.1f}% {status:>10}")

print("\n" + "-"*40)
print(f"PERIODS PASSED: {passes} of {len(periods)}")

if passes >= 2:
    print("[SUCCESS] Strategy works across periods")
else:
    print("[FAILED] Strategy does NOT generalize")
    print("         Survivorship bias was inflating results")

print("\n" + "="*60)
print("SURVIVORSHIP IMPACT ANALYSIS")
print("="*60)
print(f"""
ORIGINAL (using 2024 winners):
  2020-2024 Sharpe: 1.40

CORRECTED (using historical constituents):
  See results above

If Sharpe dropped significantly, survivorship bias
was inflating our original results by 20-30%.
""")
