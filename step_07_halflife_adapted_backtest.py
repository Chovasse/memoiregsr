"""
07_halflife_adapted_backtest.py — Backtest contrarian calibré sur le Half-Life

INNOVATION CLÉ : au lieu d'un horizon de trading fixe (21j ou 63j),
on calibre l'horizon sur le half-life de mean-reversion du GSR.

Logique :
  - Le GSR mean-reverts avec un half-life θ (estimé par Ornstein-Uhlenbeck)
  - Le half-life nous dit COMBIEN DE TEMPS il faut attendre pour que 50% de la déviation se résorbe
  - On utilise donc le half-life comme horizon de holding period
  - On compare trois approches :
    1. Horizon fixe court (TB21 = 21j) — trop court vs half-life
    2. Horizon fixe moyen (TB63 = 63j) — encore trop court
    3. Horizon calibré half-life (rolling ~93-264j selon période)

Rolling half-life : estimé sur fenêtre glissante de 504j (2 ans trading)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

# Config
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_FIG = BASE_DIR / "outputs" / "figures"
OUTPUT_TAB = BASE_DIR / "outputs" / "tables"

for d in [OUTPUT_FIG, OUTPUT_TAB]:
    d.mkdir(parents=True, exist_ok=True)

TRANSACTION_COST_BPS = 10
SLIPPAGE_BPS = 5


def estimate_halflife(series, window=504):
    """
    Estime le half-life de mean-reversion par régression OLS sur le processus OU.
    dX(t) = a + b*X(t-1) + e(t)
    Half-life = -ln(2) / b

    Retourne une Series de rolling half-life.
    """
    halflife_series = pd.Series(index=series.index, dtype=float)
    values = series.values

    for i in range(window, len(values)):
        chunk = values[i-window:i]
        if np.isnan(chunk).sum() > window * 0.1:
            continue

        # Régression : delta_X = a + b * X_lag + epsilon
        y = np.diff(chunk)
        x = chunk[:-1]

        # Filtrer NaN
        mask = ~(np.isnan(y) | np.isnan(x))
        if mask.sum() < 100:
            continue

        y_clean = y[mask]
        x_clean = x[mask]

        # OLS: y = a + b*x
        n = len(x_clean)
        x_mean = x_clean.mean()
        y_mean = y_clean.mean()
        b = np.sum((x_clean - x_mean) * (y_clean - y_mean)) / np.sum((x_clean - x_mean)**2)

        if b < 0:  # Mean-reverting seulement si b < 0
            hl = -np.log(2) / b
            if 20 < hl < 1000:  # Bornes raisonnables
                halflife_series.iloc[i] = hl

    return halflife_series


def contrarian_signal(gsr, lookback=200, zscore_entry=1.5):
    """Signal contrarian basé sur le z-score. Pas de look-ahead."""
    ma = gsr.rolling(lookback).mean()
    std = gsr.rolling(lookback).std()
    zscore = (gsr - ma) / std.replace(0, np.nan)

    signal = pd.Series(0, index=gsr.index, dtype=float)
    signal[zscore > zscore_entry] = -1   # GSR trop haut → short GSR (long silver/short gold)
    signal[zscore < -zscore_entry] = 1   # GSR trop bas → long GSR (long gold/short silver)

    return signal, zscore


def backtest_with_horizon(gsr, gold, silver, signal, horizon_series,
                          tc_bps=10, slippage_bps=5, strategy_name="Strategy"):
    """
    Backtest où la holding period est calibrée sur le half-life.

    Quand un signal est déclenché :
    - On entre en position
    - On maintient la position pendant 'horizon' jours (= half-life adaptatif)
    - On sort quand l'horizon est atteint OU quand le z-score revient à 0
    """
    common_idx = gsr.dropna().index.intersection(gold.dropna().index).intersection(silver.dropna().index)

    gsr = gsr.reindex(common_idx)
    gold = gold.reindex(common_idx)
    silver = silver.reindex(common_idx)
    signal = signal.reindex(common_idx).fillna(0)
    horizon_series = horizon_series.reindex(common_idx).fillna(method='ffill')

    gold_ret = gold.pct_change().fillna(0)
    silver_ret = silver.pct_change().fillna(0)
    spread_ret = gold_ret - silver_ret

    # Position management avec holding period adaptative
    position = pd.Series(0.0, index=common_idx)
    entry_date_idx = None
    current_horizon = 126  # default

    for i in range(1, len(common_idx)):
        if entry_date_idx is not None:
            # On est en position : vérifier si l'horizon est atteint
            days_held = i - entry_date_idx
            hl = horizon_series.iloc[i]
            current_horizon = int(hl) if not np.isnan(hl) else current_horizon

            if days_held >= current_horizon:
                # Sortie : horizon atteint
                position.iloc[i] = 0
                entry_date_idx = None
            else:
                # Maintenir la position
                position.iloc[i] = position.iloc[i-1]
        else:
            # Pas en position : vérifier signal d'entrée
            sig = signal.iloc[i]
            if sig != 0:
                position.iloc[i] = sig
                entry_date_idx = i
                hl = horizon_series.iloc[i]
                current_horizon = int(hl) if not np.isnan(hl) else 126
            else:
                position.iloc[i] = 0

    # Calculer rendements
    tc = (tc_bps + slippage_bps) / 10000
    costs = position.diff().abs().fillna(0) * tc
    strat_ret = position.shift(1).fillna(0) * spread_ret - costs

    return strat_ret, position


def backtest_fixed_horizon(gsr, gold, silver, signal, fixed_horizon,
                           tc_bps=10, slippage_bps=5):
    """Backtest avec horizon de holding fixe (pour comparaison)."""
    horizon_series = pd.Series(fixed_horizon, index=gsr.index)
    ret, pos = backtest_with_horizon(gsr, gold, silver, signal, horizon_series, tc_bps, slippage_bps)
    return ret, pos


def perf_metrics(returns, name="Strategy"):
    if len(returns) == 0 or returns.std() == 0:
        return {"name": name, "ann_return": 0, "sharpe": 0, "sortino": 0,
                "max_drawdown": 0, "calmar": 0, "win_rate": 0, "total_return": 0, "exposure_pct": 0}
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    down = returns[returns < 0]
    down_vol = down.std() * np.sqrt(252) if len(down) > 0 else 0
    sortino = ann_ret / down_vol if down_vol > 0 else 0
    cum = (1 + returns).cumprod()
    peak = cum.expanding().max()
    max_dd = ((cum - peak) / peak).min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    active = (returns != 0).sum()
    total = len(returns)
    win_rate = (returns > 0).sum() / active if active > 0 else 0
    exposure = active / total * 100
    return {
        "name": name, "ann_return": ann_ret, "sharpe": sharpe,
        "sortino": sortino, "max_drawdown": max_dd, "calmar": calmar,
        "win_rate": win_rate, "total_return": cum.iloc[-1] - 1, "exposure_pct": exposure,
    }


def main():
    print("=" * 70)
    print("STEP 7 : BACKTEST CONTRARIAN CALIBRE SUR LE HALF-LIFE")
    print("=" * 70)

    # Charger données
    df = pd.read_csv(DATA_PROCESSED / "master_daily.csv", index_col="date", parse_dates=True)
    gsr = df["gsr"].dropna()
    gold = df["gold"]
    silver = df["silver"]
    spx = df["spx"] if "spx" in df.columns else None

    print(f"\n  Données : {len(gsr)} obs ({gsr.index[0].strftime('%Y-%m')} à {gsr.index[-1].strftime('%Y-%m')})")

    # 1. Estimer le rolling half-life
    print(f"\n--- Estimation du Rolling Half-Life (fenêtre 504j) ---")
    rolling_hl = estimate_halflife(gsr, window=504)
    valid_hl = rolling_hl.dropna()
    print(f"  Observations avec HL valide : {len(valid_hl)}")
    print(f"  Half-life médian : {valid_hl.median():.0f} jours")
    print(f"  Half-life moyen  : {valid_hl.mean():.0f} jours")
    print(f"  Min / Max        : {valid_hl.min():.0f} / {valid_hl.max():.0f} jours")
    print(f"  Q25 / Q75        : {valid_hl.quantile(0.25):.0f} / {valid_hl.quantile(0.75):.0f} jours")

    # 2. Signal contrarian
    print(f"\n--- Signal Contrarian (z-score 200j, seuil ±1.5σ) ---")
    signal, zscore = contrarian_signal(gsr, lookback=200, zscore_entry=1.5)
    n_signals = (signal != 0).sum()
    print(f"  Signaux totaux : {n_signals} ({n_signals/len(signal)*100:.1f}% du temps)")
    print(f"  Shorts (z>1.5) : {(signal == -1).sum()}")
    print(f"  Longs (z<-1.5) : {(signal == 1).sum()}")

    # 3. Backtests comparatifs
    print(f"\n--- Backtests Comparatifs ---")
    results = {}

    # A) Horizon fixe 21j (original TB21)
    ret_21, pos_21 = backtest_fixed_horizon(gsr, gold, silver, signal, 21)
    results["Contrarian_H21"] = perf_metrics(ret_21, "Contrarian H=21j")

    # B) Horizon fixe 63j (original TB63)
    ret_63, pos_63 = backtest_fixed_horizon(gsr, gold, silver, signal, 63)
    results["Contrarian_H63"] = perf_metrics(ret_63, "Contrarian H=63j")

    # C) Horizon fixe 126j (demi half-life full sample)
    ret_126, pos_126 = backtest_fixed_horizon(gsr, gold, silver, signal, 126)
    results["Contrarian_H126"] = perf_metrics(ret_126, "Contrarian H=126j")

    # D) Horizon fixe 264j (full-sample half-life)
    ret_264, pos_264 = backtest_fixed_horizon(gsr, gold, silver, signal, 264)
    results["Contrarian_H264"] = perf_metrics(ret_264, "Contrarian H=264j (HL)")

    # E) Horizon adaptatif (rolling half-life)
    ret_adapt, pos_adapt = backtest_with_horizon(gsr, gold, silver, signal, rolling_hl)
    results["Contrarian_Adaptive"] = perf_metrics(ret_adapt, "Contrarian Adaptatif")

    # F) Benchmarks
    common = gsr.dropna().index.intersection(gold.dropna().index).intersection(silver.dropna().index)
    gold_bh = gold.reindex(common).pct_change().fillna(0)
    silver_bh = silver.reindex(common).pct_change().fillna(0)
    results["Gold_BH"] = perf_metrics(gold_bh, "Gold B&H")
    results["Silver_BH"] = perf_metrics(silver_bh, "Silver B&H")

    if spx is not None:
        spx_ret = spx.reindex(common).pct_change().fillna(0)
        results["SPX_BH"] = perf_metrics(spx_ret, "S&P 500 B&H")

    # Affichage
    print(f"\n{'Name':<25s} {'Sharpe':>8s} {'Return':>8s} {'MaxDD':>8s} {'Sortino':>8s} {'Expo%':>6s}")
    print("-" * 65)
    for k, m in results.items():
        print(f"  {m['name']:<23s} {m['sharpe']:>8.3f} {m['ann_return']*100:>7.1f}% {m['max_drawdown']*100:>7.1f}% {m['sortino']:>8.3f} {m['exposure_pct']:>5.0f}%")

    # 4. Sauvegarder métriques
    metrics_df = pd.DataFrame(results.values())
    metrics_df.to_csv(OUTPUT_TAB / "backtest_halflife_comparison.csv", index=False)
    print(f"\n  Métriques : {OUTPUT_TAB / 'backtest_halflife_comparison.csv'}")

    # 5. Sauvegarder rolling half-life
    hl_df = pd.DataFrame({"date": rolling_hl.index, "rolling_halflife_days": rolling_hl.values})
    hl_df.to_csv(OUTPUT_TAB / "rolling_halflife.csv", index=False)

    # 6. Rendements annuels pour la meilleure stratégie adaptative
    strat_cum = (1 + ret_adapt).cumprod()
    annual = strat_cum.resample("YE").last().pct_change().dropna()
    annual.index = annual.index.year

    annual_df = pd.DataFrame({
        "year": annual.index,
        "contrarian_adaptive_pct": annual.values * 100
    })

    # Ajouter les rendements des autres horizons pour comparaison
    for name, ret_series in [("H21", ret_21), ("H63", ret_63), ("H126", ret_126), ("H264", ret_264)]:
        cum = (1 + ret_series).cumprod()
        ann = cum.resample("YE").last().pct_change().dropna()
        ann.index = ann.index.year
        annual_df[f"contrarian_{name}_pct"] = ann.reindex(annual_df["year"]).values

    annual_df.to_csv(OUTPUT_TAB / "annual_returns_halflife.csv", index=False)

    # 7. Graphiques
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [3, 1, 1]})

    # Panel 1 : Equity curves
    ax = axes[0]
    strategies = {
        "H=21j": ret_21, "H=63j": ret_63, "H=126j": ret_126,
        "H=264j (HL)": ret_264, "Adaptatif (Rolling HL)": ret_adapt
    }
    colors = ["#D3D3D3", "#A9A9A9", "#4472C4", "#2F5496", "#C00000"]

    for (name, ret), color in zip(strategies.items(), colors):
        cum = (1 + ret).cumprod()
        lw = 2.5 if "Adaptatif" in name else (1.8 if "264" in name else 1.0)
        ax.plot(cum.index, cum.values, label=name, color=color, linewidth=lw)

    # Gold benchmark
    cum_gold = (1 + gold_bh).cumprod()
    ax.plot(cum_gold.index, cum_gold.values, label="Gold B&H", color="#FFD700", linewidth=1.5, linestyle="--")

    ax.axhline(1, color="black", linewidth=0.5)
    ax.set_title("Impact de l'Horizon de Holding sur la Performance Contrarian", fontsize=13, fontweight="bold")
    ax.set_ylabel("Valeur cumulée (base 1)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2 : Rolling half-life
    ax2 = axes[1]
    ax2.plot(rolling_hl.index, rolling_hl.values, color="#2F5496", linewidth=1.2)
    ax2.axhline(rolling_hl.median(), color="red", linestyle="--", linewidth=1, label=f"Médiane={rolling_hl.median():.0f}j")
    ax2.axhline(264, color="orange", linestyle=":", linewidth=1, label="Full sample HL=264j")
    ax2.set_ylabel("Half-Life (jours)")
    ax2.set_title("Rolling Half-Life du GSR (fenêtre 504j)", fontsize=11)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, min(rolling_hl.max() * 1.1, 800) if len(valid_hl) > 0 else 500)

    # Panel 3 : Z-score + positions adaptatives
    ax3 = axes[2]
    ax3.plot(zscore.index, zscore.values, color="navy", linewidth=0.6, alpha=0.7)
    ax3.axhline(1.5, color="red", linestyle="--", linewidth=0.8, label="Seuil ±1.5σ")
    ax3.axhline(-1.5, color="green", linestyle="--", linewidth=0.8)
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.fill_between(pos_adapt.index, 0, pos_adapt.values,
                     where=pos_adapt > 0, alpha=0.3, color="green", label="Long GSR")
    ax3.fill_between(pos_adapt.index, 0, pos_adapt.values,
                     where=pos_adapt < 0, alpha=0.3, color="red", label="Short GSR")
    ax3.set_ylabel("Z-Score / Position")
    ax3.set_title("Z-Score et Positions (Stratégie Adaptative)", fontsize=11)
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = OUTPUT_FIG / "fig_10_halflife_backtest.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Figure : {fig_path}")

    # 8. Résumé analytique
    print(f"\n{'=' * 70}")
    print("ANALYSE : IMPACT DU HALF-LIFE SUR LA PERFORMANCE")
    print(f"{'=' * 70}")

    sharpe_21 = results["Contrarian_H21"]["sharpe"]
    sharpe_63 = results["Contrarian_H63"]["sharpe"]
    sharpe_126 = results["Contrarian_H126"]["sharpe"]
    sharpe_264 = results["Contrarian_H264"]["sharpe"]
    sharpe_adapt = results["Contrarian_Adaptive"]["sharpe"]
    sharpe_gold = results["Gold_BH"]["sharpe"]

    print(f"\n  Sharpe par horizon :")
    print(f"    H=21j  (mismatch 12.6x) : {sharpe_21:.3f}")
    print(f"    H=63j  (mismatch 4.2x)  : {sharpe_63:.3f}")
    print(f"    H=126j (mismatch 2.1x)  : {sharpe_126:.3f}")
    print(f"    H=264j (calibré HL)      : {sharpe_264:.3f}")
    print(f"    Adaptatif (rolling HL)   : {sharpe_adapt:.3f}")
    print(f"    Gold B&H (benchmark)     : {sharpe_gold:.3f}")

    best_strat = max(results.items(), key=lambda x: x[1]["sharpe"] if "BH" not in x[0] and "SPX" not in x[0] else -99)
    print(f"\n  Meilleure stratégie : {best_strat[1]['name']} (Sharpe={best_strat[1]['sharpe']:.3f})")

    if sharpe_264 > sharpe_21:
        improvement = (sharpe_264 - sharpe_21)
        print(f"  Gain de calibration : +{improvement:.3f} Sharpe en passant de H21 à H264")

    print(f"\n{'=' * 70}")
    print("STEP 7 TERMINE")
    print(f"{'=' * 70}")

    return results


if __name__ == "__main__":
    main()
