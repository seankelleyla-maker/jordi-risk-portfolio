"""
correlation_matrix.py
=====================
Displays the correlation matrix for all assets in the Jordi thesis
portfolio, highlights diversification opportunities, and identifies
which assets provide the most uncorrelated exposure.

Usage:
    python analysis/correlation_matrix.py
    python analysis/correlation_matrix.py --plot   # heatmap (requires matplotlib)
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from optimizer.jordi_portfolio import load_assets, CORR

def print_correlation_matrix(assets):
    tickers = [a["ticker"] for a in assets]
    n = len(tickers)
    col_w = 7

    print(f"\n{'='*65}")
    print("  CORRELATION MATRIX — Jordi Thesis Asset Universe")
    print(f"{'='*65}")
    print(f"\n  {'':8}" + "".join(f"{t:>{col_w}}" for t in tickers))
    print(f"  {'-'*(8 + col_w*n)}")

    for i, t in enumerate(tickers):
        row = f"  {t:<8}"
        for j in range(n):
            v = CORR[i][j]
            if i == j:
                row += f"{'1.00':>{col_w}}"
            elif v >= 0.70:
                row += f"{'HI':>{col_w}}"    # high correlation
            elif v >= 0.40:
                row += f"{v:>{col_w}.2f}"
            else:
                row += f"{'lo':>{col_w}}"    # low correlation = diversifier
        print(row)

    print(f"\n  Legend: HI = highly correlated (>=0.70)  lo = diversifier (<0.40)")

def diversification_report(assets):
    tickers = [a["ticker"] for a in assets]
    sectors = [a["sector"] for a in assets]
    n = len(assets)

    print(f"\n{'='*65}")
    print("  DIVERSIFICATION ANALYSIS")
    print(f"{'='*65}")

    # Average correlation per asset (lower = better diversifier)
    avg_corrs = []
    for i in range(n):
        others = [CORR[i][j] for j in range(n) if i != j]
        avg_corrs.append(np.mean(others))

    print(f"\n  Average cross-correlation by asset (lower = better diversifier)")
    print(f"  {'Ticker':<8} {'Sector':<22} {'Avg Corr':>10}  {'Role'}")
    print(f"  {'-'*62}")
    for i in np.argsort(avg_corrs):
        role = "DIVERSIFIER" if avg_corrs[i] < 0.25 else ("MODERATE" if avg_corrs[i] < 0.50 else "CONCENTRATED")
        print(f"  {tickers[i]:<8} {sectors[i]:<22} {avg_corrs[i]:>10.3f}  {role}")

    # Sector-level correlation
    unique_sectors = list(dict.fromkeys(sectors))
    print(f"\n  Average within-sector correlation")
    print(f"  {'Sector':<22} {'Avg Corr':>10}  {'Risk'}")
    print(f"  {'-'*46}")
    for s in unique_sectors:
        idx = [i for i, a in enumerate(assets) if a["sector"] == s]
        if len(idx) < 2:
            continue
        pairs = [(i, j) for i in idx for j in idx if i < j]
        avg = np.mean([CORR[i][j] for i, j in pairs])
        risk = "HIGH" if avg > 0.65 else ("MOD" if avg > 0.40 else "LOW")
        print(f"  {s:<22} {avg:>10.3f}  {risk}")

    print(f"\n  Key diversification insights (Jordi lens):")
    insights = [
        "IBIT (0.06-0.14 avg) is the best diversifier — crypto moves independently",
        "CF/MOS (0.16-0.20 avg) are near-uncorrelated to AI tech names",
        "Energy (XLE/XOM/COP, 0.22-0.30 avg) diversifies against AI selloffs",
        "AI names (NVDA/MU/SMH, 0.72-0.88) are highly correlated — don't double-count",
        "CEG/VST (0.72 with each other) — pick one as primary Power & Grid play",
        "CF/MOS (0.78) — similarly correlated; XLE is a better diversifier within commodities",
    ]
    for insight in insights:
        print(f"  [i] {insight}")


def plot_heatmap(assets):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        tickers = [a["ticker"] for a in assets]
        n = len(tickers)

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(CORR, cmap="RdYlGn_r", vmin=-0.2, vmax=1.0)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(tickers, fontsize=10)

        for i in range(n):
            for j in range(n):
                v = CORR[i][j]
                color = "white" if v > 0.7 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold" if i==j else "normal")

        plt.colorbar(im, ax=ax, label="Correlation")
        ax.set_title("Jordi Thesis Portfolio — Correlation Matrix\n(green = diversified, red = concentrated)",
                     fontsize=13, pad=15)

        # Sector boundary lines
        sector_groups = {}
        for i, a in enumerate(assets):
            sector_groups.setdefault(a["sector"], []).append(i)

        boundaries = []
        last = -1
        for sector in dict.fromkeys(a["sector"] for a in assets):
            idxs = sector_groups[sector]
            if last >= 0:
                boundaries.append(last + 0.5)
            last = max(idxs)

        for b in boundaries:
            ax.axhline(b, color="white", linewidth=2)
            ax.axvline(b, color="white", linewidth=2)

        plt.tight_layout()
        plt.savefig("correlation_matrix.png", dpi=150, bbox_inches="tight")
        print("\nSaved: correlation_matrix.png")
        plt.show()

    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    assets, _ = load_assets("data/assets.json")
    print_correlation_matrix(assets)
    diversification_report(assets)

    if args.plot:
        plot_heatmap(assets)
