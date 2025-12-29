# 🎓 Workflow: Validating the Regime Engine with AI

This document consolidates validation workflows, moving from manual AI mentorship loops to automated pipelines.

---

## 🏗️ Part 1: The "Crawl" Phase (Manual Human-AI Loop)

Use this phase while rapidly iterating on core mechanics (P0-A, P0-B).

### 1. The "Build-Measure-Learn" Loop
1.  **Run Diagnostics (P0-D):** Generate the "Blame Table."
2.  **Export Evidence:** Screenshot ACF plots, QQ plots, and print stats.
3.  **Consult Oracle:** Paste into AI chat with specific prompts.
4.  **Refine:** Adjust parameters (block_length, etc.).

### 2. Specific Validation Checks

#### Phase A: The "Null Hypothesis" (Mode B - Bootstrap)
*Goal: Prove you can shuffle history correctly.*
*   **Action:** Run 1000 paths for SPY in 'bootstrap' mode.
*   **Check:** Does ACF decay like history? Do tails match?
*   **Prompt:** > "Here is the ACF plot of my simulated returns vs historical returns. Does this look like I successfully preserved volatility clustering? Block length = 20."

#### Phase B: The Crisis Check (Tail Dependence)
*Goal: Prove you catch crashes.*
*   **Action:** Compare `Lower Tail Dep` (Hist vs Sim).
*   **Prompt:** > "My historical Lower Tail Dep is 0.35, Sim is 0.05. I used Coupled Pair Sampling. Why is it decoupling?"

#### Phase C: First-Passage "Race"
*Goal: Verify hitting time logic.*
*   **Action:** Check if $P(\tau_{down} < \tau_{up}) + P(\tau_{up} < \tau_{down}) \le 1.0$.
*   **Prompt:** > "P(Up First) is 30%, P(Down First) is 20%. Is my Absorbing Boundary logic standard?"

---

## 🏃 Part 2: The "Walk" Phase (First Principles Checklist)

Use this physical checklist before every commit.

### 🔍 Layer 1: Code Correctness
*   [ ] Data loaded (checking length).
*   [ ] Log returns ~±5% max daily.
*   [ ] Annual Volatility ~15-60%.
*   [ ] **Billion Dollar Test:**
    *   Flat market -> Asset ~ Alpha + Noise?
    *   Beta=0 -> Asset has no market component?
    *   Alpha=0 -> Asset is Pure Beta * Market?

### 📈 Layer 2: Statistical Soundness
*   [ ] **Back-of-Envelope:** Beta 0.8-1.5? Sharpe < 2.0?
*   [ ] **Eyeball Test:**
    *   Price chart looks "real" (no gaps)?
    *   Returns histogram fat-tailed?
    *   Vol clusters (bursty)?
*   [ ] **Stationary Bootsrap Reality Check:** Does Mode A (Regime) differ meaningfully from simple bootstrap?

### 🏛️ Layer 3: Economic Reality (Macro Sense)
*   [ ] Crisis regimes align with 2008/2020?
*   [ ] Regimes shift around Fed meetings?
*   [ ] Contagion visible (Asset falls harder than Market in crisis)?

---

## 🚀 Part 3: The "Run" Phase (Automated AI Mentor Pipeline)

*Target: Implement `refinery/ai_validator.py` only after Phase 1 is stable.*

### 📋 AI Mentor Architecture

```python
class AIMentorValidator:
    """
    Automates the 'Consult Oracle' step.
    1. Runs platform.
    2. Extracts metrics (Kurtosis, ACF, VaR).
    3. Sends JSON prompt to LLM API.
    4. Returns actionable 'Fix It' list.
    """
    def validate_platform_run(self, results):
        # ... extracts stats ...
        # ... queries OpenRouter ...
        # ... returns Validation Report ...
```

### 🔬 validation Pipeline Steps
1.  **Extract Data:** Simulation Stats, Risk Metrics, Diagnostic Results.
2.  **Generates Prompts:** "Here are the stats for SPY. 1. Is realism plausible? 2. Is risk calibrated? 3. Any diagnostic failures?"
3.  **Output:** A Dashboard Score (0-100) and specific recommendations (e.g., "Increase regime persistence").

### 📊 Validation Dashboard (Concept)
*   **Score Heatmap:** Matrix of Asset (SPY/QQQ/PLTR) vs Score.
*   **Issue Breakdown:** Auto-generated list of "Critical Issues" for each asset.
*   **History:** Chart of validation scores over time.

---

## 💡 Why This Approach?
*   **Manual (Crawl):** You learn the intuition.
*   **Checklist (Walk):** You prevent regression errors.
*   **Automated (Run):** You scale up testing without burning human time.
