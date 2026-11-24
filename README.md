**What it does:**

Takes a public company's ticker as input and produces a stress-tested view of how that company would withstand severe but plausible macroeconomic shocks, summarised as a single resilience score.
Pulls real financial statements (income statement, balance sheet, cash flow) and company profile from yfinance.
Uses an LLM (Gemini) to generate three company-relevant stress scenarios and their transmission mechanisms — for example, "sharp monetary tightening reduces customer capex and equipment orders".
Quantifies each mechanism, propagates the shocks through the financial statements in Python, recalculates ratios, and scores resilience per scenario and across scenarios.
Key design decisions

**Architecture**

Five-stage pipeline: data collection → baseline assessment → scenario generation → shock quantification → propagation → resilience scoring.
All financial arithmetic is deterministic Python. Gemini is confined to narrative generation, classification and (where necessary) proposing assumptions. It never calculates a number that appears in the final output.
Scenario design

Each scenario has exactly one macro driver, chosen from a fixed set (GDP, policy rate, oil price, CPI, USD index, credit spread). One driver per scenario was chosen to avoid overlap and complexity.
Macro shocks are sized at the 95th percentile adverse movement of a 20-year historical distribution (5th percentile for "higher is worse" variables). This means shocks are severity-consistent across companies, so resilience scores are comparable.
Quantification routes

Structural — arithmetic straight from company data (e.g. rate shock × total debt = interest expense change). Only applies to interest rate and credit spread shocks on existing debt.
Empirical — pulls elasticities from academic literature via OpenAlex, with a strict "must-quote" rule so no number is invented.
Assumption — Gemini proposes a value, clearly flagged and unit-aware so its scale matches the shock.
If empirical fails, the pipeline automatically falls back to assumption, with the fallback recorded in the provenance.
Unit handling

Fractional shocks (GDP, oil, CPI, USD) and percentage-point shocks (policy rate, credit spread) require different sensitivity scales by a factor of 100. The pipeline enforces unit-appropriate caps and prompt guidance so this never produces order-of-magnitude errors.
Propagation

Shocks hit primary line items only; downstream effects are derived in Python. Revenue shocks flow through to gross profit, operating income, EBIT, net income, cash flow and ratios.
COGS scales proportionally with revenue; OpEx is held fixed, which captures operating leverage without needing a fixed/variable cost split.
Scoring

Per-ratio score = min of (relative deterioration vs baseline, absolute level vs safe threshold). This penalises both companies that deteriorate sharply and companies that were already fragile.
Four pillars (profitability, liquidity, leverage, cash flow) averaged equally within each scenario.
Cliff rule: if any ratio breaches a critical threshold, the scenario score is capped regardless of other strengths — encoding the fact that solvency events are binary.
Final score = mean across scenarios, with min and max reported so dispersion is visible.
Transparency

Every quantified mechanism carries provenance: for structural, the formula and inputs; for empirical, the paper, quote and sector match; for assumption, the justification and confidence.
Reports distinguish structural / empirical / assumption contributions so the reader knows how much of the score rests on hard data versus judgement.
