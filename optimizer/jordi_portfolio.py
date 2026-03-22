"""
jordi_portfolio.py
==================
Markowitz mean-variance portfolio optimizer aligned to the Jordi Visser
macro investment thesis. Implements five optimization strategies with
macro-aware sector constraints.

Usage:
    python jordi_portfolio.py                    # run all strategies
    python jordi_portfolio.py --strategy sharpe  # single strategy
    python jordi_portfolio.py --plot             # show efficient frontier

Strategies:
    jordi     Hand-tuned Jordi thesis weights
    sharpe    Maximize Sharpe ratio
    minvar    Minimize portfolio variance
    maxret    Maximize return (vol <= 35%)
    parity    Risk parity across sectors
"""

import json
import argparse
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

try:
    import yaml
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

# ── Load asset data ────────────────────────────────────────────────────────────

def load_assets(path="data/assets.json"):
    with open(path) as f:
        data = json.load(f)
    assets = data["assets"]
    rf = data["metadata"]["risk_free_rate"]
    return assets, rf

def load_config(path="optimizer/config.yaml"):
    if not CONFIG_AVAILABLE:
        return None
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return None

# ── Correlation matrix ─────────────────────────────────────────────────────────
# Estimated from historical data and macro factor model
# Order: NVDA, MU, AMAT, VRT, SMH, DTCR, CEG, VST, XLE, XOM, COP, CF, MOS, IBIT

CORR = np.array([
    [1.00, 0.72, 0.65, 0.70, 0.88, 0.60, 0.35, 0.38, 0.28, 0.22, 0.30, 0.20, 0.18, 0.12],
    [0.72, 1.00, 0.60, 0.62, 0.82, 0.55, 0.30, 0.33, 0.25, 0.20, 0.28, 0.18, 0.16, 0.10],
    [0.65, 0.60, 1.00, 0.58, 0.78, 0.52, 0.32, 0.35, 0.26, 0.21, 0.29, 0.19, 0.17, 0.09],
    [0.70, 0.62, 0.58, 1.00, 0.68, 0.52, 0.42, 0.45, 0.30, 0.25, 0.32, 0.22, 0.20, 0.14],
    [0.88, 0.82, 0.78, 0.68, 1.00, 0.65, 0.38, 0.40, 0.30, 0.24, 0.32, 0.22, 0.20, 0.12],
    [0.60, 0.55, 0.52, 0.52, 0.65, 1.00, 0.45, 0.48, 0.32, 0.28, 0.34, 0.24, 0.22, 0.10],
    [0.35, 0.30, 0.32, 0.42, 0.38, 0.45, 1.00, 0.72, 0.38, 0.30, 0.40, 0.28, 0.26, 0.08],
    [0.38, 0.33, 0.35, 0.45, 0.40, 0.48, 0.72, 1.00, 0.40, 0.32, 0.42, 0.30, 0.28, 0.10],
    [0.28, 0.25, 0.26, 0.30, 0.30, 0.32, 0.38, 0.40, 1.00, 0.88, 0.82, 0.42, 0.40, 0.08],
    [0.22, 0.20, 0.21, 0.25, 0.24, 0.28, 0.30, 0.32, 0.88, 1.00, 0.80, 0.38, 0.35, 0.06],
    [0.30, 0.28, 0.29, 0.32, 0.32, 0.34, 0.40, 0.42, 0.82, 0.80, 1.00, 0.44, 0.42, 0.08],
    [0.20, 0.18, 0.19, 0.22, 0.22, 0.24, 0.28, 0.30, 0.42, 0.38, 0.44, 1.00, 0.78, 0.05],
    [0.18, 0.16, 0.17, 0.20, 0.20, 0.22, 0.26, 0.28, 0.40, 0.35, 0.42, 0.78, 1.00, 0.04],
    [0.12, 0.10, 0.09, 0.14, 0.12, 0.10, 0.08, 0.10, 0.08, 0.06, 0.08, 0.05, 0.04, 1.00],
])

# ── Portfolio math ─────────────────────────────────────────────────────────────

def build_cov(assets):
    vols = np.array([a["volatility"] for a in assets])
    return np.outer(vols, vols) * CORR

def port_return(w, returns):
    return float(w @ returns)

def port_vol(w, cov):
    return float(np.sqrt(w @ cov @ w))

def port_sharpe(w, returns, cov, rf):
    r = port_return(w, returns)
    v = port_vol(w, cov)
    return (r - rf) / v

def port_beta(w, betas):
    return float(w @ betas)

def port_metrics(w, assets, cov, rf):
    returns = np.array([a["expected_return"] for a in assets])
    vols    = np.array([a["volatility"] for a in assets])
    betas   = np.array([a["beta"] for a in assets])

    r  = port_return(w, returns)
    v  = port_vol(w, cov)
    sr = (r - rf) / v
    b  = port_beta(w, betas)
    var_95  = v * 1.645 * np.sqrt(1/252)
    cvar_95 = v * 2.063 * np.sqrt(1/252)
    max_dd  = -(v * 1.65 * np.sqrt(0.25))

    return {
        "Expected Return (ann.)":  r,
        "Volatility (ann.)":       v,
        "Sharpe Ratio":            sr,
        "Portfolio Beta":          b,
        "1-day 95% VaR":           var_95,
        "1-day 95% CVaR":          cvar_95,
        "Est. Max Drawdown (1yr)": max_dd,
    }

def sector_weight(w, assets, sector):
    return sum(w[i] for i, a in enumerate(assets) if a["sector"] == sector)

# ── Constraints ────────────────────────────────────────────────────────────────

def build_constraints(assets, config=None):
    sectors = {
        "AI Infrastructure": (0.30, 0.55),
        "Power & Grid":      (0.08, 0.20),
        "Energy / Oil":      (0.12, 0.25),
        "Commodities":       (0.06, 0.14),
        "Crypto":            (0.00, 0.08),
    }
    if config:
        for s, bounds in config.get("sector_bounds", {}).items():
            sectors[s] = (bounds["min"], bounds["max"])

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    for sector, (lo, hi) in sectors.items():
        cons.append({"type": "ineq", "fun": lambda w, s=sector, h=hi: h - sector_weight(w, assets, s)})
        if lo > 0:
            cons.append({"type": "ineq", "fun": lambda w, s=sector, l=lo: sector_weight(w, assets, s) - l})
    return cons

def build_bounds(assets, config=None):
    lo = 0.01
    hi = 0.20
    if config:
        lo = config.get("portfolio", {}).get("min_single_position", 0.01)
        hi = config.get("portfolio", {}).get("max_single_position", 0.20)
    return [(lo, hi)] * len(assets)

# ── Optimizers ─────────────────────────────────────────────────────────────────

def _run(objective, assets, cov, rf, config=None, extra_cons=None):
    n = len(assets)
    cons = build_constraints(assets, config)
    if extra_cons:
        cons += extra_cons
    bounds = build_bounds(assets, config)
    w0 = np.ones(n) / n

    result = minimize(objective, w0, method="SLSQP", bounds=bounds,
                      constraints=cons, options={"ftol": 1e-10, "maxiter": 2000})
    return result.x

def optimize_max_sharpe(assets, cov, rf, config=None):
    returns = np.array([a["expected_return"] for a in assets])
    return _run(lambda w: -port_sharpe(w, returns, cov, rf), assets, cov, rf, config)

def optimize_min_variance(assets, cov, rf, config=None):
    return _run(lambda w: port_vol(w, cov), assets, cov, rf, config)

def optimize_max_return(assets, cov, rf, config=None, max_vol=0.35):
    returns = np.array([a["expected_return"] for a in assets])
    extra = [{"type": "ineq", "fun": lambda w: max_vol - port_vol(w, cov)}]
    return _run(lambda w: -port_return(w, returns), assets, cov, rf, config, extra)

def optimize_risk_parity(assets, cov, rf, config=None):
    def rp_obj(w):
        sigma = port_vol(w, cov)
        mrc = (cov @ w) / sigma
        rc  = w * mrc
        target = sigma / len(assets)
        return float(np.sum((rc - target) ** 2))
    return _run(rp_obj, assets, cov, rf, config)

def jordi_thesis_weights(assets, config=None):
    preset = {
        "NVDA":0.125,"MU":0.062,"AMAT":0.050,"VRT":0.100,"SMH":0.100,"DTCR":0.062,
        "CEG":0.100,"VST":0.062,"XLE":0.087,"XOM":0.062,"COP":0.062,
        "CF":0.075,"MOS":0.062,"IBIT":0.037
    }
    if config:
        preset = config.get("jordi_preset", preset)
    w = np.array([preset.get(a["ticker"], 0.01) for a in assets])
    return w / w.sum()

# ── Efficient frontier ─────────────────────────────────────────────────────────

def efficient_frontier(assets, cov, rf, config=None, n_points=40):
    returns = np.array([a["expected_return"] for a in assets])
    r_min = returns.min() + 0.02
    r_max = returns.max() - 0.05
    targets = np.linspace(r_min, r_max, n_points)
    vols_out, rets_out = [], []

    for target_r in targets:
        cons = build_constraints(assets, config) + [
            {"type": "eq", "fun": lambda w, r=target_r: port_return(w, returns) - r}
        ]
        result = minimize(lambda w: port_vol(w, cov), np.ones(len(assets))/len(assets),
                          method="SLSQP", bounds=build_bounds(assets, config),
                          constraints=cons, options={"ftol":1e-9,"maxiter":1000})
        if result.success and abs(port_return(result.x, returns) - target_r) < 0.005:
            vols_out.append(port_vol(result.x, cov) * 100)
            rets_out.append(port_return(result.x, returns) * 100)

    return vols_out, rets_out

# ── Risk optimization recommendations ─────────────────────────────────────────

def jordi_risk_recommendations(w, assets, cov, rf, config=None):
    m = port_metrics(w, assets, cov, rf)
    sectors = list(set(a["sector"] for a in assets))
    sw = {s: sector_weight(w, assets, s) * 100 for s in sectors}
    recs = []

    ai = sw.get("AI Infrastructure", 0)
    en = sw.get("Energy / Oil", 0)
    com = sw.get("Commodities", 0)
    cry = sw.get("Crypto", 0)
    sr = m["Sharpe Ratio"]
    beta = m["Portfolio Beta"]
    vol = m["Volatility (ann.)"]

    if ai > 50:
        recs.append(("WARN", f"AI concentration {ai:.0f}% is high. Consider capping at 45% and shifting NVDA/VRT -> SMH/DTCR ETFs to reduce idiosyncratic risk while keeping thematic exposure."))
    elif ai < 30:
        recs.append(("WARN", f"AI Infrastructure at {ai:.0f}% is below Jordi's core thesis minimum. The AI capex supercycle ($350B+ in 2026 alone) warrants 30-45% allocation."))
    else:
        recs.append(("OK",   f"AI Infrastructure at {ai:.0f}% is well-balanced. Prefer ETFs (SMH, DTCR) for core, individual names (NVDA, VRT) for satellite positions."))

    if en < 12:
        recs.append(("WARN", f"Energy at {en:.0f}% underweights Jordi's inflation re-acceleration thesis. WTI approaching $98, diesel up 49% — XLE and COP are the primary hedges. Target 15-20%."))
    else:
        recs.append(("OK",   f"Energy at {en:.0f}% is positioned for Jordi's inflation thesis. COP/XOM balance upside with dividend income."))

    if com < 6:
        recs.append(("WARN", f"Commodities at {com:.0f}% underweights fertilizer disruption thesis. CF/MOS asymmetric payoff if Hormuz disruption persists through planting season. Target 8-12%."))
    else:
        recs.append(("OK",   f"Commodities at {com:.0f}% captures the fertilizer catalyst. CF/MOS are low-beta with high event-driven upside."))

    if cry > 8:
        recs.append(("WARN", f"Crypto at {cry:.0f}% exceeds Jordi's recommended ceiling. IBIT vol of 72% drags Sharpe ratio at high weights. Cap at 5-8% for debasement hedge only."))

    if sr < 0.8:
        recs.append(("WARN", f"Sharpe ratio {sr:.2f} is below 0.8. Key levers: (1) Replace NVDA with SMH — same exposure, 14pp lower vol. (2) Add AMLP for income (not modeled). (3) Increase CEG over VST — lower beta, nuclear baseload moat."))
    elif sr < 1.0:
        recs.append(("INFO", f"Sharpe ratio {sr:.2f} is below the 1.0 target. Marginal improvement available by trimming VRT (highest vol in AI space) and adding DTCR."))
    else:
        recs.append(("OK",   f"Sharpe ratio {sr:.2f} exceeds target of 1.0. Risk-adjusted return is strong."))

    if beta > 1.3:
        recs.append(("WARN", f"Beta {beta:.2f} is high — portfolio moves {beta*100:.0f}% of S&P moves. Jordi flags recession risk as underpriced. Consider adding CEG/XOM (beta <0.85) to bring portfolio beta below 1.2."))
    elif beta < 0.8:
        recs.append(("INFO", f"Beta {beta:.2f} is conservative — you may be leaving AI upside on the table. Consider adding SMH or NVDA if conviction in AI buildout is high."))
    else:
        recs.append(("OK",   f"Beta {beta:.2f} is in the target range (0.85-1.25). Balanced market sensitivity given Jordi's macro caution."))

    if vol > 0.38:
        recs.append(("WARN", f"Portfolio vol {vol*100:.1f}% is elevated. Top volatility reducers: replace VRT with DTCR, replace individual energy names with XLE ETF, trim MU in favor of SMH."))
    else:
        recs.append(("OK",   f"Portfolio volatility {vol*100:.1f}% is well-managed for the return target."))

    recs.append(("INFO", "Correlation insight: NVDA/MU/SMH are 0.72-0.88 correlated — true diversifiers are IBIT (0.10-0.14 vs tech), CF/MOS (0.16-0.20 vs AI), and Energy (0.22-0.30 vs AI). Do not zero these out."))
    recs.append(("INFO", "Recession hedge: Jordi flags recession risk as underpriced. Consider 5-10% cash/T-bills (not modeled) as dry powder for re-entry on AI names if the S&P corrects 15-20%."))

    return recs

# ── Printing ───────────────────────────────────────────────────────────────────

def print_portfolio(name, w, assets, cov, rf):
    returns = np.array([a["expected_return"] for a in assets])
    betas   = np.array([a["beta"] for a in assets])
    m = port_metrics(w, assets, cov, rf)

    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"{'='*65}")
    print(f"  {'Metric':<30} {'Value':>12}")
    print(f"  {'-'*44}")
    fmts = {
        "Expected Return (ann.)":  lambda v: f"{v*100:.1f}%",
        "Volatility (ann.)":       lambda v: f"{v*100:.1f}%",
        "Sharpe Ratio":            lambda v: f"{v:.2f}",
        "Portfolio Beta":          lambda v: f"{v:.2f}",
        "1-day 95% VaR":           lambda v: f"{v*100:.2f}%",
        "1-day 95% CVaR":          lambda v: f"{v*100:.2f}%",
        "Est. Max Drawdown (1yr)": lambda v: f"{v*100:.1f}%",
    }
    for k, fmt in fmts.items():
        print(f"  {k:<30} {fmt(m[k]):>12}")

    print(f"\n  {'Ticker':<8} {'Sector':<22} {'Weight':>8}  {'Bar'}")
    print(f"  {'-'*58}")
    for i in np.argsort(w)[::-1]:
        if w[i] > 0.005:
            bar = "|" * max(1, int(w[i] * 60))
            print(f"  {assets[i]['ticker']:<8} {assets[i]['sector']:<22} {w[i]*100:>6.1f}%  {bar}")

    sectors = {}
    for i, a in enumerate(assets):
        sectors[a["sector"]] = sectors.get(a["sector"], 0) + w[i]
    print(f"\n  {'Sector':<22} {'Weight':>8}")
    print(f"  {'-'*32}")
    for s, sw_val in sorted(sectors.items(), key=lambda x: -x[1]):
        print(f"  {s:<22} {sw_val*100:>6.1f}%")

def print_recommendations(w, assets, cov, rf, config=None):
    recs = jordi_risk_recommendations(w, assets, cov, rf, config)
    icons = {"OK": "[+]", "WARN": "[!]", "INFO": "[i]"}
    print(f"\n{'='*65}")
    print("  RISK OPTIMIZATION RECOMMENDATIONS (Jordi lens)")
    print(f"{'='*65}")
    for level, msg in recs:
        icon = icons.get(level, "[?]")
        words = msg.split()
        line, lines = [], []
        for word in words:
            if len(" ".join(line + [word])) > 72:
                lines.append(" ".join(line))
                line = [word]
            else:
                line.append(word)
        if line:
            lines.append(" ".join(line))
        print(f"\n  {icon} {lines[0]}")
        for l in lines[1:]:
            print(f"      {l}")

def compare_all(portfolios, assets, cov, rf):
    print(f"\n{'='*85}")
    print("  PORTFOLIO COMPARISON")
    print(f"{'='*85}")
    names = [p[0] for p in portfolios]
    col = 14
    hdr = f"  {'Metric':<26}" + "".join(f"{n:>{col}}" for n in names)
    print(hdr)
    print("  " + "-" * (26 + col * len(names)))

    def row(label, fn):
        vals = [fn(p[1]) for p in portfolios]
        print(f"  {label:<26}" + "".join(f"{v:>{col}}" for v in vals))

    returns = np.array([a["expected_return"] for a in assets])
    row("Expected Return",  lambda w: f"{port_return(w,returns)*100:.1f}%")
    row("Volatility",       lambda w: f"{port_vol(w,cov)*100:.1f}%")
    row("Sharpe Ratio",     lambda w: f"{port_sharpe(w,returns,cov,rf):.2f}")
    row("Beta",             lambda w: f"{port_beta(w, np.array([a['beta'] for a in assets])):.2f}")
    row("AI Infra %",       lambda w: f"{sector_weight(w,assets,'AI Infrastructure')*100:.0f}%")
    row("Energy %",         lambda w: f"{sector_weight(w,assets,'Energy / Oil')*100:.0f}%")
    row("Power %",          lambda w: f"{sector_weight(w,assets,'Power & Grid')*100:.0f}%")
    row("Commodity %",      lambda w: f"{sector_weight(w,assets,'Commodities')*100:.0f}%")
    row("Crypto %",         lambda w: f"{sector_weight(w,assets,'Crypto')*100:.0f}%")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Jordi Visser Portfolio Optimizer")
    parser.add_argument("--strategy", default="all",
                        choices=["all","jordi","sharpe","minvar","maxret","parity"],
                        help="Which strategy to run (default: all)")
    parser.add_argument("--plot", action="store_true",
                        help="Plot efficient frontier (requires matplotlib)")
    parser.add_argument("--data",   default="data/assets.json")
    parser.add_argument("--config", default="optimizer/config.yaml")
    args = parser.parse_args()

    assets, rf = load_assets(args.data)
    config = load_config(args.config)
    cov = build_cov(assets)

    print("\nJordi Visser Macro Thesis — Portfolio Optimizer")
    print(f"Assets: {len(assets)}  |  Risk-free rate: {rf*100:.1f}%")
    print("Thesis: AI infra buildout + inflation re-acceleration + supply shocks")

    w_jordi  = jordi_thesis_weights(assets, config)
    w_sharpe = optimize_max_sharpe(assets, cov, rf, config)
    w_minvar = optimize_min_variance(assets, cov, rf, config)
    w_maxret = optimize_max_return(assets, cov, rf, config)
    w_parity = optimize_risk_parity(assets, cov, rf, config)

    all_portfolios = [
        ("Jordi Thesis",  w_jordi),
        ("Max Sharpe",    w_sharpe),
        ("Min Variance",  w_minvar),
        ("Max Return",    w_maxret),
        ("Risk Parity",   w_parity),
    ]

    if args.strategy == "all":
        for name, w in all_portfolios:
            print_portfolio(name, w, assets, cov, rf)
        compare_all(all_portfolios, assets, cov, rf)
        print("\n--- Recommendations for Jordi Thesis portfolio ---")
        print_recommendations(w_jordi, assets, cov, rf, config)
    else:
        mapping = {"jordi":w_jordi,"sharpe":w_sharpe,"minvar":w_minvar,"maxret":w_maxret,"parity":w_parity}
        w = mapping[args.strategy]
        label = dict(all_portfolios)[args.strategy] if args.strategy != "jordi" else "Jordi Thesis"
        print_portfolio(label, w, assets, cov, rf)
        print_recommendations(w, assets, cov, rf, config)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            print("\nComputing efficient frontier...")
            fvols, frets = efficient_frontier(assets, cov, rf, config)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(fvols, frets, "b-", lw=2, label="Efficient frontier")

            colors = {"Jordi Thesis":"purple","Max Sharpe":"green","Min Variance":"blue",
                      "Max Return":"orange","Risk Parity":"red"}
            returns = np.array([a["expected_return"] for a in assets])
            for name, w in all_portfolios:
                v = port_vol(w, cov) * 100
                r = port_return(w, returns) * 100
                ax.scatter(v, r, s=120, zorder=5, color=colors.get(name,"gray"),
                           label=f"{name} ({v:.1f}% vol, {r:.1f}% ret)")

            ax.set_xlabel("Annualized Volatility (%)", fontsize=12)
            ax.set_ylabel("Expected Annual Return (%)", fontsize=12)
            ax.set_title("Jordi Thesis Portfolio — Efficient Frontier", fontsize=14)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("efficient_frontier.png", dpi=150, bbox_inches="tight")
            print("Saved: efficient_frontier.png")
            plt.show()
        except ImportError:
            print("matplotlib not installed. Run: pip install matplotlib")

if __name__ == "__main__":
    main()
