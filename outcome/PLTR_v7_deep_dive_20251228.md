# Deep Dive: What PLTR v7.0 is *actually* saying (2025-12-28)

You’ve built something that’s **internally consistent** (invariants pass, sim ≈ hist hit rates), but the output is also screaming one big truth:

> **PLTR can absolutely rip… but it will also try to shake you out constantly.**
> Your “LONG” is a *directional* call, while the risk dashboard is a *survivability* warning.

---

# 1) 📦 Data Snapshot
**1196 days | $188.71 | Vol: 65.0%**

* You’re using ~5 years of daily data (after rolling-feature dropna).
* **65% annualized vol** is *insane*. Rough translation: **~4.1% typical daily move** (your invariant check confirms that).

**This alone explains why:**
* **P(MaxDD > 30%) ~ 61%**
* **P(Stop -15%) ~ 56%**
Even in “good” environments, this thing whipsaws.

---

# 2) 🧬 Regimes: What “Low Vol” really means
### ✅ Model
**GMM clustering on slow features**: `[Vol_20d, Vol_60d, Ret_20d, Drawdown]`
So regimes are NOT “return buckets”. They’re **market states**.

### Current Regime: Low Vol (prob 67%)
But your diagnostics reveal the trap:
* **Low Vol:** `avg_dd = -18%`, `avg_vol = 55%`
* **Crisis:** `avg_dd = -31%`, `avg_vol = 88%`
* **Normal:** `avg_dd = -2.7%`, `avg_vol = 60%`

⚠️ So “Low Vol” ≠ “healthy calm uptrend”. It’s more like:
✅ **“Quiet bleed / underwater grind”** (low-ish vol, but still sitting in drawdown on average). That’s why your drift in Low Vol is negative.

---

# 3) 📈 Regime Stats (mu, sigma, skew)
* **Low Vol:** `mu = -46%` | `sig = 57%` | `skew = +0.12`
* **Crisis:** `mu = -17%` | `sig = 82%` | `skew = -0.43`
* **Normal:** `mu = +302%` | `sig = 66%` | `skew = +1.83`

### How to interpret this:
* **Crisis skew is negative** → left-tail ugliness. Expected.
* **Normal skew is massively positive** → melt-up regime (lottery-right-tail).
⚠️ The **+302% annualized drift** is a warning sign: it likely means your “Normal” cluster captures a **small set of explosive trend periods**. It’s real (PLTR had insane stretches), but **fragile**.

---

# 4) ⏳ Semi-Markov Durations
* Low Vol: **28d avg**
* Crisis: **33d avg**
* Normal: **13d avg**

**Meaning**: Regimes last **weeks**, not days. Your simulator persistence is driven by these run-lengths.

---

# 5) 🌍 Macro Conditioning: Beta + Alpha + Idio vol
### Beta 1.75
If QQQ moves 1%, PLTR moves ~**1.75%**.

### Alpha (ann) 18.8%
On average, PLTR has had positive “intercept” drift vs QQQ across the whole sample.

### Idiosyncratic vol 51.8%
Stock-specific chaos after removing market factor.

### Regime Alphas:
* Low Vol alpha: **-53%**
* Crisis alpha: **-4%**
* Normal alpha: **+203%**

✅ **Outperformance happens primarily in the “Normal/Melt-up” state**. Low Vol is not your friend. The platform is basically saying: *“I’ll be bullish, but only because the right tail exists.”*

---

# 6) 🧠 Context Layer: VIX + Anomaly
* **VIX 13.6** = calm market.
* Jump prob stays **2%**.
* **Anomaly: No** → no macro red flags.

---

# 7) 📉 GARCH
**GARCH 60% vs realized 65%**
Vol is still high, but not accelerating today. Treat this as **diagnostic only**.

---

# 8) ✅ Historical Hit Rates (First-Passage)
Over a 126-day horizon:
* **Up +48%**: **51.6%**
* **Down -36%**: **20.7%**
The upside target is **more likely than not** historically, but the downside target is **non-trivial** (~1 in 5).

---

# 9) 🧪 Bucket Asymmetry Diagnostics
* **Low Vol:** mild positive skew, but negative drift.
* **Crisis:** negative skew (bad left tail).
* **Normal:** huge positive skew (right-tail rocket fuel).
This is a **convexity profile**: Most regimes are meh/down, but one provides the “lottery payoff.”

---

# 10) 📊 Multi-threshold Calibration Table
This table explains why your sim produces **Up 53% / Down 19%**. It aligns with reality.

---

# 11) 🚨 Walk-forward validation still looks bad (and why)
* Predicted mean: **18%** | Actual mean: **60%** | Brier: **0.317**
**The process changed.** The recent regime is more explosive than the earlier training folds. A 5-fold sample is also tiny/noisy.

---

# 12) ✅ Invariant check passes (this is huge)
Your sim daily returns match hist on mean, std, skew, and kurtosis.
✅ Your simulator isn’t “broken”.

---

# 13) 📉 Why Kelly is 0%
Because your DD penalty is: `max(0, 1 - 2 * prob_dd_30)`.
With `prob_dd_30 = 0.612`, the result is forced to **0**.
✅ This is **sane behavior** for a “DD-aware sizing rule”.

---

# 🧭 The Single Best Interpretation (In one sentence)
**PLTR has strong upside probability and enormous right-tail potential, but the path is so violent that normal risk controls (like a -15% stop) will likely eject you before the payoff.**
