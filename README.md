# Jordi Visser Macro Thesis — Risk Portfolio Optimizer

A quantitative portfolio construction and risk optimization toolkit built around the **Jordi Visser macro investment thesis**.

## Core Thesis

- **AI infrastructure buildout is structural** — compute, power, data centers are a multi-year capex supercycle
- **Inflation is re-accelerating** — energy and commodities are the primary hedge (WTI near $98, diesel +49%)
- **Private credit and financials are cracking** — avoid them entirely
- **Recession risk is underpriced** — manage beta and volatility carefully
- **Bitcoin** is a small, non-correlated debasement hedge (keep under 5-8%)

---

## Quickstart

```bash
git clone https://github.com/seankelleyla-maker/jordi-risk-portfolio.git
cd jordi-risk-portfolio
pip install -r optimizer/requirements.txt

# Run all 5 portfolio strategies
python optimizer/jordi_portfolio.py

# Run a single strategy
python optimizer/jordi_portfolio.py --strategy sharpe

# Plot the efficient frontier (requires matplotlib)
python optimizer/jordi_portfolio.py --plot

# Full risk report with stress tests
python analysis/risk_metrics.py

# Correlation and diversification analysis
python analysis/correlation_matrix.py
```

---

## Repository Structure

```
jordi-risk-portfolio/
├── README.md
├── data/
│   └── assets.json              # 14 tickers: returns, vol, beta, sector, thesis role
├── optimizer/
│   ├── jordi_portfolio.py       # Main Markowitz optimizer — 5 strategies
│   ├── config.yaml              # Tune weights and constraints without touching code
│   └── requirements.txt
└── analysis/
    ├── risk_metrics.py          # Sharpe, Sortino, Calmar, VaR, CVaR, stress tests
    └── correlation_matrix.py    # Sector correlation heatmap and diversification report
```

---

## Portfolio Strategies

| Strategy | Description | Best for |
|---|---|---|
| `jordi` | Hand-tuned to Jordi macro thesis | Conviction-based investors |
| `sharpe` | Maximize Sharpe ratio | Best risk-adjusted return |
| `minvar` | Minimize portfolio variance | Capital preservation |
| `maxret` | Maximize return, vol <= 35% | Aggressive growth |
| `parity` | Equal risk contribution per asset | True diversification |

---

## Asset Universe (14 holdings)

| Ticker | Name | Sector | Thesis Role |
|---|---|---|---|
| NVDA | NVIDIA | AI Infrastructure | Core GPU compute |
| MU | Micron Technology | AI Infrastructure | HBM memory chips |
| AMAT | Applied Materials | AI Infrastructure | Semiconductor equipment |
| VRT | Vertiv Holdings | AI Infrastructure | Data center power/cooling |
| SMH | VanEck Semiconductor ETF | AI Infrastructure | Diversified semi exposure |
| DTCR | Global X Data Center ETF | AI Infrastructure | Data center REITs, lower vol |
| CEG | Constellation Energy | Power & Grid | Nuclear baseload for AI data centers |
| VST | Vistra Energy | Power & Grid | Power generation, AI energy demand |
| XLE | Energy Select SPDR ETF | Energy / Oil | Broad inflation hedge |
| XOM | ExxonMobil | Energy / Oil | Supermajor, dividend income |
| COP | ConocoPhillips | Energy / Oil | Low-cost E&P, oil upside |
| CF | CF Industries | Commodities | Nitrogen fertilizer disruption catalyst |
| MOS | Mosaic Company | Commodities | Potash/phosphate, planting season risk |
| IBIT | iShares Bitcoin Trust | Crypto | Non-correlated debasement hedge |

---

## Sector Allocation Bounds (Jordi Constraints)

| Sector | Min | Max | Rationale |
|---|---|---|---|
| AI Infrastructure | 30% | 55% | Core long-term thesis — AI capex supercycle |
| Power & Grid | 8% | 20% | AI energy demand — nuclear baseload critical |
| Energy / Oil | 12% | 25% | Primary inflation re-acceleration hedge |
| Commodities | 6% | 14% | Fertilizer disruption — asymmetric payoff |
| Crypto | 0% | 8% | Small non-correlated debasement hedge only |

---

## Risk Optimization Rules (Jordi Lens)

1. **AI Infrastructure 30-55%** — structural long-term, but cap single names at 15%
2. **Replace individual AI names with ETFs** — SMH/DTCR over NVDA/VRT reduces vol while keeping thematic exposure
3. **Energy at 15-20%** — XLE and COP are the best inflation hedges; XOM adds dividend income
4. **Keep beta between 0.85-1.25** — Jordi warns recession risk is underpriced; high beta hurts near-term
5. **Target Sharpe > 1.0** — if below, trim highest-vol names (VRT, MU) first
6. **Crypto under 8%** — IBIT vol of 72% drags Sharpe ratio at high allocations
7. **CF/MOS are asymmetric** — low weight (6-8%), high event-driven upside if fertilizer thesis hits
8. **True diversifiers** — IBIT (0.10-0.14 corr to tech), CF/MOS (0.16-0.20), Energy (0.22-0.30 vs AI)
9. **No single position > 20%** — idiosyncratic risk cap
10. **Keep 5-10% cash** (not modeled) — dry powder for re-entry if AI names correct 15-20%

---

## Stress Test Scenarios

The `analysis/risk_metrics.py` module runs 5 Jordi-specific stress tests:

| Scenario | Description |
|---|---|
| AI sentiment reversal | DeepSeek-style shock: AI names -25 to -30% |
| Recession confirmed | S&P -20%, risk-off selloff across the board |
| Inflation shock | Oil to $120, rates rise — energy names surge |
| Hormuz closed 6 months | Jordi base case — energy +25-35%, tech -5-12% |
| Soft landing | Fed cuts, AI re-rates — NVDA +25%, energy -5% |

---

## Sample Output

```
Jordi Visser Macro Thesis — Portfolio Optimizer
Assets: 14  |  Risk-free rate: 4.5%

=================================================================
  Jordi Thesis
=================================================================
  Expected Return (ann.)                19.4%
  Volatility (ann.)                     25.4%
  Sharpe Ratio                           0.58
  Portfolio Beta                         1.11
  1-day 95% VaR                         2.64%
  1-day 95% CVaR                        3.31%
  Est. Max Drawdown (1yr)              -21.0%

  AI Infrastructure    47.7%
  Energy / Oil         20.2%
  Power & Grid         15.5%
  Commodities          13.1%
  Crypto                3.5%
```

---

## Disclaimer

This project is for educational and research purposes only. It does not constitute financial advice. Past performance does not predict future results. The expected returns, volatilities, and correlations used are estimates based on analyst consensus and historical data — they are not guaranteed. Always consult a licensed financial advisor before making investment decisions.
