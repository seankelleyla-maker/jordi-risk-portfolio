"""
risk_metrics.py
===============
Standalone risk analysis module for the Jordi Visser portfolio.
Computes VaR, CVaR, max drawdown, beta, Sharpe, Sortino, and
Calmar ratios. Can be run standalone or imported by jordi_portfolio.py.

Usage:
    python analysis/risk_metrics.py
"""

import json
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from optimizer.jordi_portfolio import (
    load_assets, load_config, build_cov,
    jordi_thesis_weights, optimize_max_sharpe,
    port_return, port_vol, port_beta
)

RF_RATE = 0.045

# ── Risk metrics ───────────────────────────────────────────────────────────────

def var_parametric(w, cov, confidence=0.95, horizon_days=1):
    """Parametric (Gaussian) Value at Risk."""
    from scipy.stats import norm
    z = norm.ppf(confidence)
    annual_vol = port_vol(w, cov)
    daily_vol  = annual_vol / np.sqrt(252)
    horizon_vol = daily_vol * np.sqrt(horizon_days)
    return z * horizon_vol

def cvar_parametric(w, cov, confidence=0.95, horizon_days=1):
    """Parametric Conditional VaR (Expected Shortfall)."""
    from scipy.stats import norm
    z = norm.ppf(confidence)
    annual_vol = port_vol(w, cov)
    daily_vol  = annual_vol / np.sqrt(252)
    horizon_vol = daily_vol * np.sqrt(horizon_days)
    return (norm.pdf(z) / (1 - confidence)) * horizon_vol

def sortino_ratio(w, assets, cov, rf=RF_RATE, mar=0.0):
    """
    Sortino ratio: return / downside deviation.
    Better than Sharpe for asymmetric return distributions.
    """
    returns = np.array([a["expected_return"] for a in assets])
    vols    = np.array([a["volatility"] for a in assets])
    r = port_return(w, returns)
    # Downside deviation approximation: vol * sqrt(P(r < MAR))
    # For lognormal, approx 0.5 * total vol
    downside_vol = port_vol(w, cov) * 0.7
    return (r - rf) / downside_vol if downside_vol > 0 else 0

def calmar_ratio(w, assets, cov, rf=RF_RATE):
    """Calmar ratio: annual return / max drawdown."""
    returns = np.array([a["expected_return"] for a in assets])
    r = port_return(w, returns)
    v = port_vol(w, cov)
    max_dd = v * 1.65 * np.sqrt(0.25)  # estimate
    return r / max_dd if max_dd > 0 else 0

def information_ratio(w, assets, cov, benchmark_return=0.12, benchmark_vol=0.16):
    """Information ratio vs a benchmark (default: S&P 500 proxy)."""
    returns = np.array([a["expected_return"] for a in assets])
    r = port_return(w, returns)
    v = port_vol(w, cov)
    tracking_error = np.sqrt(v**2 + benchmark_vol**2 - 2 * 0.7 * v * benchmark_vol)
    return (r - benchmark_return) / tracking_error if tracking_error > 0 else 0

def concentration_herfindahl(w):
    """Herfindahl-Hirschman Index — measure of concentration (0=diverse, 1=concentrated)."""
    return float(np.sum(w**2))

def effective_n(w):
    """Effective number of holdings (1/HHI)."""
    hhi = concentration_herfindahl(w)
    return 1 / hhi if hhi > 0 else 0

def marginal_risk_contribution(w, cov):
    """Marginal contribution of each asset to portfolio vol."""
    sigma = port_vol(w, cov)
    mrc = (cov @ w) / sigma
    return mrc

def risk_contribution(w, cov):
    """Absolute and percentage risk contribution of each asset."""
    mrc = marginal_risk_contribution(w, cov)
    rc_abs = w * mrc
    sigma = port_vol(w, cov)
    rc_pct = rc_abs / sigma
    return rc_abs, rc_pct

def stress_test(w, assets, scenarios):
    """
    Stress test portfolio against named scenarios.
    scenarios: dict of {name: {ticker: shock_pct}}
    Returns estimated portfolio impact for each scenario.
    """
    tickers = [a["ticker"] for a in assets]
    results = {}
    for scenario_name, shocks in scenarios.items():
        impact = sum(w[i] * shocks.get(t, 0) for i, t in enumerate(tickers))
        results[scenario_name] = impact
    return results

# ── Jordi scenario stress tests ────────────────────────────────────────────────

JORDI_SCENARIOS = {
    "AI sentiment reversal (DeepSeek-style -30% AI names)": {
        "NVDA": -0.30, "MU": -0.25, "AMAT": -0.20, "VRT": -0.28,
        "SMH": -0.25, "DTCR": -0.15, "CEG": -0.08, "VST": -0.10,
        "XLE": +0.02, "XOM": +0.02, "COP": +0.02,
        "CF": +0.01, "MOS": +0.01, "IBIT": -0.15
    },
    "Recession confirmed (S&P -20%, risk-off)": {
        "NVDA": -0.40, "MU": -0.35, "AMAT": -0.30, "VRT": -0.38,
        "SMH": -0.32, "DTCR": -0.18, "CEG": -0.12, "VST": -0.15,
        "XLE": -0.18, "XOM": -0.15, "COP": -0.20,
        "CF": -0.15, "MOS": -0.15, "IBIT": -0.35
    },
    "Inflation shock — oil to $120, rates rise": {
        "NVDA": -0.12, "MU": -0.10, "AMAT": -0.08, "VRT": -0.10,
        "SMH": -0.10, "DTCR": -0.05, "CEG": +0.05, "VST": +0.08,
        "XLE": +0.22, "XOM": +0.20, "COP": +0.25,
        "CF": +0.18, "MOS": +0.20, "IBIT": -0.10
    },
    "Hormuz Strait closed 6 months (Jordi base case)": {
        "NVDA": -0.08, "MU": -0.12, "AMAT": -0.05, "VRT": -0.05,
        "SMH": -0.08, "DTCR": -0.03, "CEG": +0.10, "VST": +0.08,
        "XLE": +0.30, "XOM": +0.28, "COP": +0.35,
        "CF": +0.25, "MOS": +0.22, "IBIT": -0.05
    },
    "Soft landing — Fed cuts, AI re-rates": {
        "NVDA": +0.25, "MU": +0.20, "AMAT": +0.15, "VRT": +0.22,
        "SMH": +0.20, "DTCR": +0.15, "CEG": +0.10, "VST": +0.12,
        "XLE": -0.05, "XOM": -0.05, "COP": -0.05,
        "CF": -0.03, "MOS": -0.03, "IBIT": +0.30
    },
}

# ── Reporting ──────────────────────────────────────────────────────────────────

def full_risk_report(w, assets, cov, label="Portfolio"):
    returns = np.array([a["expected_return"] for a in assets])
    betas   = np.array([a["beta"] for a in assets])
    tickers = [a["ticker"] for a in assets]

    print(f"\n{'='*65}")
    print(f"  FULL RISK REPORT: {label}")
    print(f"{'='*65}")

    r     = port_return(w, returns)
    v     = port_vol(w, cov)
    sr    = (r - RF_RATE) / v
    b     = port_beta(w, betas)
    sort  = sortino_ratio(w, assets, cov)
    calm  = calmar_ratio(w, assets, cov)
    ir    = information_ratio(w, assets, cov)
    hhi   = concentration_herfindahl(w)
    eff_n = effective_n(w)

    print(f"\n  Return & Risk")
    print(f"  {'Expected Return (ann.)':<32} {r*100:.1f}%")
    print(f"  {'Volatility (ann.)':<32} {v*100:.1f}%")
    print(f"  {'Beta (vs S&P 500)':<32} {b:.2f}")

    print(f"\n  Risk-Adjusted Performance")
    print(f"  {'Sharpe Ratio (RF=4.5%)':<32} {sr:.2f}")
    print(f"  {'Sortino Ratio':<32} {sort:.2f}")
    print(f"  {'Calmar Ratio':<32} {calm:.2f}")
    print(f"  {'Information Ratio (vs S&P)':<32} {ir:.2f}")

    print(f"\n  Tail Risk (parametric Gaussian)")
    for conf in [0.90, 0.95, 0.99]:
        var_1d  = var_parametric(w, cov, conf, 1)
        cvar_1d = cvar_parametric(w, cov, conf, 1)
        var_10d = var_parametric(w, cov, conf, 10)
        print(f"  {int(conf*100)}% VaR  1d: {var_1d*100:.2f}%   10d: {var_10d*100:.2f}%   CVaR 1d: {cvar_1d*100:.2f}%")

    est_dd = v * 1.65 * np.sqrt(0.25)
    print(f"  {'Est. Max Drawdown (1yr)':<32} {-est_dd*100:.1f}%")

    print(f"\n  Diversification")
    print(f"  {'HHI (concentration)':<32} {hhi:.3f}  (0=diverse, 1=single stock)")
    print(f"  {'Effective N (holdings)':<32} {eff_n:.1f}")

    rc_abs, rc_pct = risk_contribution(w, cov)
    print(f"\n  Risk Contribution by Asset")
    print(f"  {'Ticker':<8} {'Weight':>8} {'Risk Contrib %':>16} {'Bar'}")
    print(f"  {'-'*52}")
    for i in np.argsort(rc_pct)[::-1]:
        if w[i] > 0.005:
            bar = "|" * max(1, int(rc_pct[i] * 80))
            print(f"  {tickers[i]:<8} {w[i]*100:>7.1f}% {rc_pct[i]*100:>14.1f}%  {bar}")

    print(f"\n  Jordi Scenario Stress Tests")
    print(f"  {'Scenario':<50} {'P&L':>8}")
    print(f"  {'-'*60}")
    results = stress_test(w, assets, JORDI_SCENARIOS)
    for name, impact in results.items():
        sign = "+" if impact >= 0 else ""
        print(f"  {name[:49]:<50} {sign}{impact*100:.1f}%")


if __name__ == "__main__":
    assets, rf = load_assets("data/assets.json")
    config = load_config("optimizer/config.yaml")
    cov = build_cov(assets)

    w_jordi  = jordi_thesis_weights(assets, config)
    w_sharpe = optimize_max_sharpe(assets, cov, rf, config)

    full_risk_report(w_jordi,  assets, cov, "Jordi Thesis")
    full_risk_report(w_sharpe, assets, cov, "Max Sharpe")
