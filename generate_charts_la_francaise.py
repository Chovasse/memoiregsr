"""
Génération de TOUS les graphiques du mémoire — Template La Française
Palette : #201676 #FF3E56 #009BA7 #2E5EAA #4A90D9 #1A7A8A #D4A017 #E8613C #6B7FA3 #4A7C6F
Police : Liberation Sans (proxy DM Sans)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import FuncFormatter
import warnings, os
warnings.filterwarnings("ignore")

# === TEMPLATE LA FRANÇAISE ===
LF = {
    "primary":   "#201676",  # bleu foncé signature
    "red":       "#FF3E56",
    "teal":      "#009BA7",
    "blue":      "#2E5EAA",
    "blue_lt":   "#4A90D9",
    "teal_dk":   "#1A7A8A",
    "gold":      "#D4A017",
    "orange":    "#E8613C",
    "grey":      "#6B7FA3",
    "green":     "#4A7C6F",
}
PALETTE = list(LF.values())

# Style global
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.titleweight": "bold",
    "axes.edgecolor": "#C0C0C0",
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "grid.color": "#E0E0E0",
    "grid.linewidth": 0.5,
})

import pathlib
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_BASE = _SCRIPT_DIR.parent  # = Mémoire/
DATA_DIR = str(_BASE / "data" / "processed")
OUT_DIR  = str(_BASE / "outputs" / "figures")
os.makedirs(OUT_DIR, exist_ok=True)

def lf_style(ax, title="", ylabel="", xlabel=""):
    """Applique le style La Française à un axe."""
    ax.set_title(title, fontweight="bold", color=LF["primary"], pad=12)
    if ylabel: ax.set_ylabel(ylabel, color="#333")
    if xlabel: ax.set_xlabel(xlabel, color="#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C0C0C0")
    ax.spines["bottom"].set_color("#C0C0C0")
    ax.grid(True, alpha=0.4, linewidth=0.5)
    ax.tick_params(colors="#333")

def pct_fmt(x, pos):
    return f"{x:.0%}"

# ===== LOAD DATA =====
print("Chargement des données...")
df = pd.read_pickle(f"{DATA_DIR}/features.pkl")
print(f"  {len(df)} obs, {df.index.min().date()} -> {df.index.max().date()}")

# ========== PARTIE 1 : REVUE DE LITTÉRATURE ==========

def fig_1_1():
    """Figure 1.1 : Prix de l'or"""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(df.index, df["gold"], color=LF["gold"], linewidth=1.0)
    ax.fill_between(df.index, df["gold"], alpha=0.12, color=LF["gold"])
    lf_style(ax, "Figure 1.1 : Cours de l'or (USD/oz), 2000–2026", "USD / once troy")

    events = [
        ("2008-09-15", "Crise financière\n2008", 55),
        ("2011-09-06", "Pic 2011", 55),
        ("2020-03-15", "COVID-19", 55),
        ("2024-10-01", "Rally 2024-25", 55),
    ]
    for ds, lbl, off in events:
        try:
            dt = pd.Timestamp(ds)
            nearest = df.index[df.index.get_indexer([dt], method="nearest")[0]]
            val = df.loc[nearest, "gold"]
            ax.annotate(lbl, xy=(nearest, val), xytext=(0, off),
                       textcoords="offset points", fontsize=8,
                       arrowprops=dict(arrowstyle="->", color=LF["grey"], lw=0.8),
                       ha="center", color=LF["primary"])
        except: pass

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_1_1_gold_price.png", bbox_inches="tight")
    plt.close()
    print("  [OK] 1.1 Gold price")

def fig_1_2():
    """Figure 1.2 : Prix de l'argent"""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(df.index, df["silver"], color=LF["grey"], linewidth=1.0)
    ax.fill_between(df.index, df["silver"], alpha=0.12, color=LF["grey"])
    lf_style(ax, "Figure 1.2 : Cours de l'argent (USD/oz), 2000–2026", "USD / once troy")

    events = [("2011-04-28", "Pic 2011\n~49 $/oz", 40), ("2020-03-15", "COVID-19", 40)]
    for ds, lbl, off in events:
        try:
            dt = pd.Timestamp(ds)
            nearest = df.index[df.index.get_indexer([dt], method="nearest")[0]]
            val = df.loc[nearest, "silver"]
            ax.annotate(lbl, xy=(nearest, val), xytext=(0, off),
                       textcoords="offset points", fontsize=8,
                       arrowprops=dict(arrowstyle="->", color=LF["grey"], lw=0.8),
                       ha="center", color=LF["primary"])
        except: pass

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_1_2_silver_price.png", bbox_inches="tight")
    plt.close()
    print("  [OK] 1.2 Silver price")

def fig_1_3():
    """Figure 1.3 : Demande d'argent (donut)"""
    labels = ["Industriel\n(dont PV)", "Investissement", "Joaillerie", "Argenterie", "Photo"]
    sizes = [50, 25, 17, 5, 3]
    colors = [LF["primary"], LF["teal"], LF["blue"], LF["gold"], LF["grey"]]

    fig, ax = plt.subplots(figsize=(9, 7))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=colors, autopct=lambda p: f"{p:.0f}%",
        startangle=140, pctdistance=0.78,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2.5))
    for t in autotexts:
        t.set_fontsize(11); t.set_fontweight("bold"); t.set_color("white")

    ax.legend(wedges, [f"{l.strip()}  ({s}%)" for l, s in zip(labels, sizes)],
              title="Secteurs de demande", title_fontsize=11,
              loc="center left", bbox_to_anchor=(0.85, 0, 0.5, 1), fontsize=10,
              frameon=True, edgecolor=LF["primary"])
    ax.set_title("Figure 1.3 : Demande mondiale d'argent", fontweight="bold",
                 color=LF["primary"], fontsize=13, pad=10)
    ax.text(0, -0.05, "Source : Silver Institute, 2024", ha="center", fontsize=9,
            style="italic", color=LF["grey"], transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_1_3_silver_demand.png", bbox_inches="tight")
    plt.close()
    print("  [OK] 1.3 Silver demand")

def fig_1_4():
    """Figure 1.4 : Corrélation glissante Or vs S&P 500"""
    gold_ret = df["gold"].pct_change()
    sp_ret = df["spx"].pct_change()
    corr = gold_ret.rolling(252).corr(sp_ret)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(corr.index, corr, 0, where=corr < 0, alpha=0.3, color=LF["red"], label="Négative")
    ax.fill_between(corr.index, corr, 0, where=corr >= 0, alpha=0.2, color=LF["blue"], label="Positive")
    ax.plot(corr.index, corr, color=LF["primary"], linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    lf_style(ax, "Figure 1.4 : Corrélation glissante Or vs S&P 500 (252j)", "Corrélation")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_ylim(-0.6, 0.6)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_1_4_gold_sp500_corr.png", bbox_inches="tight")
    plt.close()
    print("  [OK] 1.4 Correlation")

def fig_1_5():
    """Figure 1.5 : GSR historique avec zones de régime"""
    gsr = df["gsr"].dropna()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhspan(0, 50, alpha=0.06, color=LF["teal"], label="Risk-on (GSR < 50)")
    ax.axhspan(50, 80, alpha=0.04, color=LF["gold"], label="Neutre")
    ax.axhspan(80, 150, alpha=0.06, color=LF["red"], label="Risk-off (GSR > 80)")
    ax.plot(gsr.index, gsr, color=LF["primary"], linewidth=0.9)
    ax.axhline(gsr.mean(), color=LF["gold"], lw=1.2, ls="--", alpha=0.8,
               label=f"Moyenne ({gsr.mean():.0f})")
    lf_style(ax, "Figure 1.5 : Gold/Silver Ratio avec zones de régime", "GSR")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    for ds, lbl, off in [("2020-03-18", "COVID-19\nGSR > 120", 15), ("2011-04-25", "Compression\n2011", -45)]:
        try:
            dt = pd.Timestamp(ds)
            nearest = gsr.index[gsr.index.get_indexer([dt], method="nearest")[0]]
            ax.annotate(lbl, xy=(nearest, gsr.loc[nearest]), xytext=(0, off),
                       textcoords="offset points", fontsize=8,
                       arrowprops=dict(arrowstyle="->", color=LF["grey"], lw=0.8),
                       ha="center", color=LF["primary"])
        except: pass

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_1_5_gsr_regimes.png", bbox_inches="tight")
    plt.close()
    print("  [OK] 1.5 GSR regimes")

def fig_1_6():
    """Figure 1.6 : Distribution du GSR + Q-Q plot"""
    gsr = df["gsr"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    ax1.hist(gsr, bins=60, density=True, alpha=0.6, color=LF["blue"], edgecolor="white")
    # KDE manually (no scipy)
    from numpy import exp
    h = 1.06 * gsr.std() * len(gsr)**(-1/5)  # Silverman bandwidth
    x_kde = np.linspace(gsr.min()-5, gsr.max()+5, 300)
    kde_vals = np.zeros_like(x_kde)
    for xi in gsr.values:
        kde_vals += exp(-0.5 * ((x_kde - xi)/h)**2)
    kde_vals /= (len(gsr) * h * np.sqrt(2*np.pi))
    ax1.plot(x_kde, kde_vals, color=LF["red"], linewidth=2, label="KDE")
    ax1.axvline(gsr.mean(), color=LF["gold"], lw=2, ls="--", label=f"Moyenne = {gsr.mean():.1f}")
    ax1.axvline(gsr.median(), color=LF["primary"], lw=1.5, ls=":", label=f"Médiane = {gsr.median():.1f}")
    lf_style(ax1, "Distribution du GSR", "Densité", "Gold/Silver Ratio")
    ax1.legend(fontsize=9)

    # Q-Q plot manually (no scipy)
    ax2 = axes[1]
    sorted_gsr = np.sort(gsr.values)
    n = len(sorted_gsr)
    theoretical = np.array([np.sqrt(2) * np.math.erfc(2*(1 - (i-0.5)/n)) if False else 0 for i in range(1, n+1)])
    # Simpler: use normal percent point approximation
    probs = (np.arange(1, n+1) - 0.5) / n
    # Approximate normal quantiles using Beasley-Springer-Moro
    theoretical = np.zeros(n)
    for i, p in enumerate(probs):
        # Rational approximation for normal quantile
        if p < 0.5:
            t = np.sqrt(-2 * np.log(p))
            theoretical[i] = -(t - (2.515517 + 0.802853*t + 0.010328*t**2) / (1 + 1.432788*t + 0.189269*t**2 + 0.001308*t**3))
        else:
            t = np.sqrt(-2 * np.log(1-p))
            theoretical[i] = t - (2.515517 + 0.802853*t + 0.010328*t**2) / (1 + 1.432788*t + 0.189269*t**2 + 0.001308*t**3)

    ax2.scatter(theoretical, sorted_gsr, s=4, color=LF["blue"], alpha=0.5)
    # Fit line
    slope = (sorted_gsr[-1] - sorted_gsr[0]) / (theoretical[-1] - theoretical[0])
    intercept = gsr.mean()
    ax2.plot(theoretical, slope * theoretical + intercept, color=LF["red"], lw=2, label="Droite théorique")
    lf_style(ax2, "Q-Q Plot vs Normale", "Quantiles observés", "Quantiles théoriques")
    ax2.legend(fontsize=9)

    fig.suptitle("Figure 1.6 : Distribution et normalité du GSR", fontweight="bold",
                 color=LF["primary"], y=1.02)
    skew, kurt = gsr.skew(), gsr.kurtosis()
    fig.text(0.5, -0.02, f"Skewness = {skew:.2f} | Kurtosis = {kurt:.2f}",
             ha="center", fontsize=9, style="italic", color=LF["grey"])
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_1_6_gsr_distribution.png", bbox_inches="tight")
    plt.close()
    print("  [OK] 1.6 Distribution")

def fig_1_7():
    """Figure 1.7 : GSR vs VIX"""
    data = df[["gsr", "vix"]].dropna()
    fig, ax1 = plt.subplots(figsize=(12, 5.5))
    ax2 = ax1.twinx()
    ln1 = ax1.plot(data.index, data["gsr"], color=LF["primary"], lw=0.9, label="GSR")
    ln2 = ax2.plot(data.index, data["vix"], color=LF["red"], lw=0.6, alpha=0.7, label="VIX")
    ax1.set_ylabel("Gold/Silver Ratio", color=LF["primary"])
    ax2.set_ylabel("VIX", color=LF["red"])
    ax1.tick_params(axis="y", labelcolor=LF["primary"])
    ax2.tick_params(axis="y", labelcolor=LF["red"])
    lines = ln1 + ln2
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=9)
    corr = data["gsr"].corr(data["vix"])
    ax1.set_title(f"Figure 1.7 : GSR vs VIX (corrélation = {corr:.2f})",
                  fontweight="bold", color=LF["primary"])
    ax1.grid(True, alpha=0.3)
    ax1.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_1_7_gsr_vix.png", bbox_inches="tight")
    plt.close()
    print("  [OK] 1.7 GSR vs VIX")

def fig_1_8():
    """Figure 1.8 : Taxonomie ML"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")

    def box(x, y, w, h, text, color, fs=9):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                           facecolor=color, edgecolor="#333", lw=1.2, alpha=0.9)
        ax.add_patch(b)
        tc = "white" if color != "#FFF9E6" else LF["primary"]
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs, fontweight="bold", color=tc)

    ax.text(6, 6.5, "Méthodes de Machine Learning", ha="center", fontsize=14,
            fontweight="bold", color=LF["primary"])
    box(0.3, 4, 3.2, 1.2, "RÉGRESSION\nLINÉAIRE", LF["blue"], 11)
    box(4.4, 4, 3.2, 1.2, "MÉTHODES\nD'ENSEMBLE", LF["orange"], 11)
    box(8.5, 4, 3.2, 1.2, "RÉSEAUX DE\nNEURONES", LF["grey"], 11)
    box(0.3, 2.2, 1.5, 0.8, "OLS", LF["blue_lt"])
    box(2.0, 2.2, 1.5, 0.8, "Ridge /\nLasso", LF["blue_lt"])
    box(4.4, 2.2, 1.5, 0.8, "Random\nForest", LF["teal"])
    box(6.1, 2.2, 1.5, 0.8, "XGBoost", LF["teal"])
    box(8.5, 2.2, 1.5, 0.8, "LSTM", LF["teal_dk"])
    box(10.2, 2.2, 1.5, 0.8, "Deep\nLearning", LF["teal_dk"])
    box(0.3, 0.5, 3.2, 0.8, "Benchmark\n(interprétable)", "#FFF9E6", 9)
    box(4.4, 0.5, 3.2, 0.8, "Modèles principaux\n(perf + stabilité)", "#FFF9E6", 9)
    box(8.5, 0.5, 3.2, 0.8, "Comparaison\n(risque overfitting)", "#FFF9E6", 9)
    for x in [1.9, 6.0, 10.1]:
        ax.annotate("", xy=(x, 3.1), xytext=(x, 3.9), arrowprops=dict(arrowstyle="->", color=LF["grey"], lw=1.5))
        ax.annotate("", xy=(x, 1.4), xytext=(x, 2.1), arrowprops=dict(arrowstyle="->", color=LF["grey"], lw=1))

    fig.suptitle("Figure 1.8 : Taxonomie des méthodes ML", fontweight="bold", color=LF["primary"], y=0.98)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_1_8_ml_taxonomy.png", bbox_inches="tight")
    plt.close()
    print("  [OK] 1.8 Taxonomy")

def fig_1_9():
    """Figure 1.9 : Walk-forward"""
    fig, ax = plt.subplots(figsize=(12, 5))
    n_folds = 5
    years = list(range(2000, 2027))
    for i in range(n_folds):
        y = n_folds - i
        ts, te = i*2, 10+i*2
        vs, ve = te, te+2
        ax.barh(y, te-ts, left=ts, height=0.6, color=LF["blue"], alpha=0.85, edgecolor="white")
        ax.barh(y, ve-vs, left=vs, height=0.6, color=LF["orange"], alpha=0.85, edgecolor="white")
        ax.text(ts+(te-ts)/2, y, "Train", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        ax.text(vs+(ve-vs)/2, y, "Test", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax.barh(0, 5, left=len(years)-6, height=0.6, color=LF["teal"], alpha=0.85, edgecolor="white")
    ax.text(len(years)-3.5, 0, "Test final", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    ax.set_yticks(range(n_folds+1))
    ax.set_yticklabels(["Test final"] + [f"Fold {i+1}" for i in range(n_folds)])
    ax.set_xticks(range(0, len(years), 5))
    ax.set_xticklabels([str(y) for y in years[::5]])
    lf_style(ax, "Figure 1.9 : Walk-forward purged cross-validation", "", "Période")
    ax.legend(handles=[Patch(facecolor=LF["blue"], label="Entraînement"),
                       Patch(facecolor=LF["orange"], label="Validation OOS"),
                       Patch(facecolor=LF["teal"], label="Test final")],
              loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_1_9_walkforward.png", bbox_inches="tight")
    plt.close()
    print("  [OK] 1.9 Walk-forward")

# ========== PARTIE 2 : MÉTHODOLOGIE ==========

def fig_2_1():
    """Figure 2.1 : ACF du GSR"""
    gsr = df["gsr"].dropna().values
    n = len(gsr)
    lags = 60
    mean = gsr.mean()
    denom = np.sum((gsr - mean)**2)
    acf_vals = np.array([np.sum((gsr[:n-k] - mean) * (gsr[k:] - mean)) / denom for k in range(lags+1)])
    ci = 1.96 / np.sqrt(n)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(range(lags+1), acf_vals, color=LF["primary"], alpha=0.8, width=0.8)
    ax.axhline(ci, color=LF["red"], ls="--", lw=0.8, alpha=0.6, label="IC 95%")
    ax.axhline(-ci, color=LF["red"], ls="--", lw=0.8, alpha=0.6)
    ax.axhline(0, color="black", lw=0.5)
    lf_style(ax, "Figure 2.1 : Autocorrélation du GSR", "ACF", "Lag (jours)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_2_1_gsr_acf.png", bbox_inches="tight")
    plt.close()
    print("  [OK] 2.1 ACF")

def fig_2_2():
    """Figure 2.2 : Half-life vs horizons de trading (mismatch)"""
    fig, ax = plt.subplots(figsize=(10, 5))
    horizons = {"TB21\n(21j)": 21, "TB63\n(63j)": 63, "HL médian\nglissant\n(68j)": 68,
                "HL moyen\nglissant\n(93j)": 93, "HL\ncomplet\n(264j)": 264}
    names = list(horizons.keys())
    vals = list(horizons.values())
    colors = [LF["red"], LF["orange"], LF["teal"], LF["blue"], LF["primary"]]
    bars = ax.bar(names, vals, color=colors, alpha=0.85, edgecolor="white", width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+5, f"{v}j", ha="center", fontsize=10, fontweight="bold", color=LF["primary"])
    ax.axhline(264, color=LF["primary"], ls="--", lw=1.5, alpha=0.5, label="Half-life complet (264j)")
    lf_style(ax, "Figure 2.2 : Half-life vs horizons de trading", "Jours")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_2_2_halflife_mismatch.png", bbox_inches="tight")
    plt.close()
    print("  [OK] 2.2 HL mismatch")

def fig_2_3():
    """Figure 2.3 : Rolling half-life"""
    hl_path = f"{DATA_DIR}/rolling_halflife.pkl"
    if not os.path.exists(hl_path):
        print("  [SKIP] 2.3 - rolling_halflife.pkl absent")
        return
    hl = pd.read_pickle(hl_path)
    hl_valid = hl.dropna()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hl_valid.index, hl_valid.values, color=LF["primary"], lw=0.9)
    ax.axhline(hl_valid.median(), color=LF["red"], ls="--", lw=1.5, label=f"Médiane = {hl_valid.median():.0f}j")
    ax.fill_between(hl_valid.index, hl_valid.quantile(0.25), hl_valid.quantile(0.75),
                    alpha=0.1, color=LF["blue"])
    lf_style(ax, "Figure 2.3 : Half-life glissant du GSR (fenêtre 504j)", "Half-life (jours)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_2_3_rolling_halflife.png", bbox_inches="tight")
    plt.close()
    print("  [OK] 2.3 Rolling HL")

# ========== PARTIE 3 : RÉSULTATS ==========

def fig_3_1():
    """Figure 3.1 : Importance des features (top 15)"""
    # Feature importance from labeled data
    labeled_path = f"{DATA_DIR}/labeled.pkl"
    if not os.path.exists(labeled_path):
        print("  [SKIP] 3.1 - labeled.pkl absent")
        return

    # Build quick correlation-based importance
    lab = pd.read_pickle(labeled_path)
    label_cols = [c for c in lab.columns if c.startswith("label")]
    if not label_cols:
        print("  [SKIP] 3.1 - no labels")
        return

    # Use correlations with GSR returns as proxy
    feat_cols = [c for c in lab.columns if c.endswith(("_ret_1d", "_ret_5d", "_ret_21d", "_ratio", "_vol_20d",
                                                        "_drawdown", "_regime", "_momentum", "_change_5d", "_change_21d",
                                                        "_ma_ratio", "_proxy", "_spread", "_premium",
                                                        "_60d"))]
    if len(feat_cols) < 5:
        feat_cols = [c for c in lab.columns if c not in ["gold", "silver", "gsr", "date"] and not c.startswith("label")
                     and lab[c].notna().sum() > len(lab) * 0.5][:20]

    corrs = lab[feat_cols].corrwith(lab["gsr_ret_21d"] if "gsr_ret_21d" in lab.columns else lab["gsr"]).abs().sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(corrs))]
    ax.barh(range(len(corrs)), corrs.values[::-1], color=colors[::-1], alpha=0.85)
    ax.set_yticks(range(len(corrs)))
    ax.set_yticklabels(corrs.index[::-1], fontsize=9)
    lf_style(ax, "Figure 3.1 : Importance des features (corrélation absolue)", "", "|Corrélation|")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_3_1_feature_importance.png", bbox_inches="tight")
    plt.close()
    print("  [OK] 3.1 Feature importance")

def fig_3_5():
    """Figure 3.5 : Comparaison des Sharpe ratios (bar chart La Française style)"""
    # From backtest results
    metrics_path = str(_BASE / "outputs" / "tables" / "backtest_metrics_halflife.csv")
    if not os.path.exists(metrics_path):
        print("  [SKIP] 3.5 - backtest_metrics_halflife.csv absent")
        return

    met = pd.read_csv(metrics_path)
    met = met.sort_values("sharpe", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors_bar = []
    for _, row in met.iterrows():
        if row["sharpe"] >= 0.4: colors_bar.append(LF["teal"])
        elif row["sharpe"] >= 0: colors_bar.append(LF["blue"])
        else: colors_bar.append(LF["red"])

    bars = ax.barh(range(len(met)), met["sharpe"], color=colors_bar, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(met)))
    ax.set_yticklabels(met["name"], fontsize=9)
    ax.axvline(0, color="black", lw=0.5)

    for i, (_, row) in enumerate(met.iterrows()):
        offset = 0.02 if row["sharpe"] >= 0 else -0.02
        ha = "left" if row["sharpe"] >= 0 else "right"
        ax.text(row["sharpe"] + offset, i, f"{row['sharpe']:.3f}", va="center", ha=ha,
                fontsize=9, fontweight="bold", color=LF["primary"])

    lf_style(ax, "Figure 3.5 : Comparaison des ratios de Sharpe", "", "Ratio de Sharpe")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_3_5_sharpe_comparison.png", bbox_inches="tight")
    plt.close()
    print("  [OK] 3.5 Sharpe comparison")

def fig_3_6():
    """Figure 3.6 : Épisodes de régimes extrêmes (GFC, Silver Squeeze, COVID)"""
    gsr = df["gsr"].dropna()
    ma200 = gsr.rolling(200).mean()
    std200 = gsr.rolling(200).std()
    zscore = (gsr - ma200) / std200

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), gridspec_kw={"height_ratios": [2, 1.2, 1.2]})

    # Panel 1: GSR with extreme zones
    ax = axes[0]
    ax.plot(gsr.index, gsr, color=LF["primary"], lw=0.9, label="GSR")
    ax.plot(ma200.index, ma200, color=LF["gold"], lw=1, ls="--", alpha=0.7, label="MA 200j")

    # Highlight extreme episodes
    episodes = [
        ("2008-06-01", "2009-06-01", "GFC 2008", LF["red"]),
        ("2010-10-01", "2011-10-01", "Silver Squeeze", LF["teal"]),
        ("2020-01-01", "2020-12-01", "COVID-19", LF["orange"]),
    ]
    for start, end, lbl, col in episodes:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.12, color=col, label=lbl)

    lf_style(ax, "Figure 3.6 : Épisodes de régimes extrêmes du GSR", "GSR")
    ax.legend(loc="upper left", fontsize=8, ncol=3)

    # Panel 2: Z-score
    ax2 = axes[1]
    ax2.fill_between(zscore.index, zscore, 0, where=zscore > 1.5, alpha=0.3, color=LF["red"])
    ax2.fill_between(zscore.index, zscore, 0, where=zscore < -1.5, alpha=0.3, color=LF["teal"])
    ax2.plot(zscore.index, zscore, color=LF["primary"], lw=0.7)
    ax2.axhline(1.5, color=LF["red"], ls="--", lw=0.8, alpha=0.7)
    ax2.axhline(-1.5, color=LF["teal"], ls="--", lw=0.8, alpha=0.7)
    ax2.axhline(0, color="black", lw=0.3)
    lf_style(ax2, "Z-score du GSR (200j)", "Z-score")

    # Panel 3: Forward returns
    fwd_126 = gsr.pct_change(126).shift(-126)
    ax3 = axes[2]
    ax3.fill_between(fwd_126.index, fwd_126, 0, where=fwd_126 < 0, alpha=0.3, color=LF["teal"])
    ax3.fill_between(fwd_126.index, fwd_126, 0, where=fwd_126 >= 0, alpha=0.3, color=LF["red"])
    ax3.plot(fwd_126.index, fwd_126, color=LF["primary"], lw=0.5, alpha=0.7)
    ax3.axhline(0, color="black", lw=0.3)
    ax3.yaxis.set_major_formatter(FuncFormatter(pct_fmt))
    lf_style(ax3, "Forward return du GSR (126j)", "Rendement")

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_3_6_gsr_regime_episodes.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  [OK] 3.6 Regime episodes")

def fig_3_7():
    """Figure 3.7 : Backtest equity curves (half-life adapted)"""
    pred_path = f"{DATA_DIR}/predictions_halflife.pkl"
    labeled_path = f"{DATA_DIR}/labeled_halflife.pkl"
    if not os.path.exists(pred_path) or not os.path.exists(labeled_path):
        print("  [SKIP] 3.7 - halflife data absent")
        return

    preds = pd.read_pickle(pred_path)
    lab = pd.read_pickle(labeled_path)
    gold, silver = lab["gold"], lab["silver"]
    gold_ret = gold.pct_change().fillna(0)
    silver_ret = silver.pct_change().fillna(0)
    spread_ret = gold_ret - silver_ret

    fig, ax = plt.subplots(figsize=(14, 6))
    strat_colors = {"Brut_ZScore": LF["red"], "XGBoost": LF["primary"],
                    "RandomForest": LF["teal"], "Logistic": LF["blue_lt"]}

    for col in preds.columns:
        sig = preds[col].fillna(0)
        common = sig.index.intersection(spread_ret.index)
        ret = sig.reindex(common).shift(1).fillna(0) * spread_ret.reindex(common)
        cum = (1 + ret).cumprod()
        color = strat_colors.get(col, LF["grey"])
        lw = 2.0 if col == "Brut_ZScore" else 1.5
        ax.plot(cum.index, cum, color=color, lw=lw, label=col)

    # Gold B&H
    cum_gold = (1 + gold_ret).cumprod()
    ax.plot(cum_gold.index, cum_gold, color=LF["gold"], lw=1.5, ls="--", label="Gold B&H")
    ax.axhline(1, color="black", lw=0.5)
    lf_style(ax, "Figure 3.7 : Backtest — Holding Period = Rolling Half-Life", "Valeur cumulée (base 1)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_3_7_backtest_halflife.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  [OK] 3.7 Backtest HL")


# ========== MAIN ==========
if __name__ == "__main__":
    print("=" * 60)
    print("  GRAPHIQUES MÉMOIRE — TEMPLATE LA FRANÇAISE")
    print("=" * 60)

    # Partie 1
    print("\n--- PARTIE 1 : Revue de littérature ---")
    fig_1_1(); fig_1_2(); fig_1_3(); fig_1_4()
    fig_1_5(); fig_1_6(); fig_1_7(); fig_1_8(); fig_1_9()

    # Partie 2
    print("\n--- PARTIE 2 : Méthodologie ---")
    fig_2_1(); fig_2_2(); fig_2_3()

    # Partie 3
    print("\n--- PARTIE 3 : Résultats ---")
    fig_3_1(); fig_3_5(); fig_3_6(); fig_3_7()

    print(f"\n{'='*60}")
    print(f"  {len(os.listdir(OUT_DIR))} figures dans {OUT_DIR}")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith(".png"):
            sz = os.path.getsize(f"{OUT_DIR}/{f}") / 1024
            print(f"  {f} ({sz:.0f} KB)")
    print(f"{'='*60}")
