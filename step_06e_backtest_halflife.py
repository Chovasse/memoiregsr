"""
06e_backtest_halflife.py — Backtest contrarian avec horizon calibré Half-Life

Compare :
  - Signal brut z-score (holding = rolling half-life)
  - Signal filtré XGBoost (holding = rolling half-life)
  - Signal filtré RF (holding = rolling half-life)
  - Signal filtré Logistic (holding = rolling half-life)
  - Benchmarks : Gold B&H, Silver B&H, S&P 500

La holding period de chaque trade = rolling half-life au moment de l'entrée.
Cela aligne la durée de position avec la vitesse de mean-reversion estimée.

Sortie :
  - outputs/tables/backtest_metrics_halflife.csv
  - outputs/tables/annual_returns_halflife_ml.csv
  - outputs/figures/fig_11_backtest_halflife_ml.png
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from config import (
    DATA_PROCESSED, OUTPUT_FIG, OUTPUT_TAB,
    TRANSACTION_COST_BPS, SLIPPAGE_BPS
)


def backtest_halflife_strategy(predictions, gold, silver, rolling_hl,
                               tc_bps=10, slippage_bps=5):
    """
    Backtest avec holding period = rolling half-life.

    Quand le signal entre en position, on maintient pendant HL jours.
    """
    common_idx = predictions.dropna().index
    common_idx = common_idx.intersection(gold.dropna().index).intersection(silver.dropna().index)

    pred = predictions.reindex(common_idx).fillna(0)
    gold_c = gold.reindex(common_idx)
    silver_c = silver.reindex(common_idx)
    hl_c = rolling_hl.reindex(common_idx).ffill().fillna(126)

    gold_ret = gold_c.pct_change().fillna(0)
    silver_ret = silver_c.pct_change().fillna(0)
    spread_ret = gold_ret - silver_ret

    # Position management avec holding period = half-life
    position = pd.Series(0.0, index=common_idx)
    entry_idx = None
    cur_horizon = 126

    for i in range(1, len(common_idx)):
        if entry_idx is not None:
            days_held = i - entry_idx
            if days_held >= cur_horizon:
                # Sortie : horizon atteint
                position.iloc[i] = 0
                entry_idx = None
                # Vérifier si nouveau signal
                sig = pred.iloc[i]
                if sig != 0:
                    position.iloc[i] = sig
                    entry_idx = i
                    hl = hl_c.iloc[i]
                    cur_horizon = int(hl) if not np.isnan(hl) else 126
                    cur_horizon = max(21, min(cur_horizon, 504))
            else:
                position.iloc[i] = position.iloc[i-1]
        else:
            sig = pred.iloc[i]
            if sig != 0:
                position.iloc[i] = sig
                entry_idx = i
                hl = hl_c.iloc[i]
                cur_horizon = int(hl) if not np.isnan(hl) else 126
                cur_horizon = max(21, min(cur_horizon, 504))
            else:
                position.iloc[i] = 0

    # Rendements
    tc = (tc_bps + slippage_bps) / 10000
    costs = position.diff().abs().fillna(0) * tc
    strat_ret = position.shift(1).fillna(0) * spread_ret - costs

    return pd.DataFrame({
        "strategy": strat_ret,
        "gold_bh": gold_ret,
        "silver_bh": silver_ret,
        "spread_bh": spread_ret,
        "position": position,
    }, index=common_idx)


def perf_metrics(returns, name="Strategy"):
    if len(returns) == 0 or returns.std() == 0:
        return {"name": name, "ann_return": 0, "sharpe": 0, "sortino": 0,
                "max_drawdown": 0, "calmar": 0, "win_rate": 0,
                "total_return": 0, "exposure_pct": 0}
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
    print("STEP 6e : BACKTEST CONTRARIAN HALFLIFE (SIGNAL BRUT + ML)")
    print("=" * 70)

    # Charger données
    labeled_path = DATA_PROCESSED / "labeled_halflife.pkl"
    pred_path = DATA_PROCESSED / "predictions_halflife.pkl"

    if not labeled_path.exists():
        print("[ERR] labeled_halflife.pkl introuvable. Exécutez step_03d.")
        return
    if not pred_path.exists():
        print("[ERR] predictions_halflife.pkl introuvable. Exécutez step_04d.")
        return

    df = pd.read_pickle(labeled_path)
    preds = pd.read_pickle(pred_path)
    gold, silver = df["gold"], df["silver"]

    # Rolling half-life
    if "rolling_halflife" in df.columns:
        rolling_hl = df["rolling_halflife"]
    else:
        hl_path = DATA_PROCESSED / "rolling_halflife.pkl"
        rolling_hl = pd.read_pickle(hl_path) if hl_path.exists() else pd.Series(126, index=df.index)

    print(f"\n  Données : {len(df)} obs")
    print(f"  Stratégies disponibles : {list(preds.columns)}")
    hl_median = rolling_hl.dropna().median()
    print(f"  Half-life médian : {hl_median:.0f} jours (holding period)")

    # Backtests
    all_metrics = []
    all_equity = {}

    print(f"\n--- Stratégies (holding = rolling half-life) ---")
    for strat_name in preds.columns:
        pred = preds[strat_name].dropna()
        if len(pred) == 0:
            continue

        bt = backtest_halflife_strategy(pred, gold, silver, rolling_hl,
                                        TRANSACTION_COST_BPS, SLIPPAGE_BPS)
        m = perf_metrics(bt["strategy"], strat_name)
        exposure = (bt["position"].abs() > 0).sum() / len(bt) * 100
        m["exposure_pct"] = exposure
        all_metrics.append(m)
        all_equity[strat_name] = bt

        print(f"  {strat_name:25s} : Sharpe={m['sharpe']:.3f}  "
              f"Return={m['ann_return']*100:.1f}%  "
              f"MaxDD={m['max_drawdown']*100:.1f}%  "
              f"Expo={exposure:.0f}%")

    # Benchmarks
    if all_equity:
        first = list(all_equity.values())[0]
        common_idx = first.index

        # Gold B&H
        gold_bh = gold.reindex(common_idx).pct_change().fillna(0)
        m_gold = perf_metrics(gold_bh, "Gold B&H")
        all_metrics.append(m_gold)
        all_equity["Gold B&H"] = pd.DataFrame({"strategy": gold_bh}, index=common_idx)

        # Silver B&H
        silver_bh = silver.reindex(common_idx).pct_change().fillna(0)
        m_silver = perf_metrics(silver_bh, "Silver B&H")
        all_metrics.append(m_silver)

        # S&P 500
        if "spx" in df.columns:
            spx_ret = df["spx"].reindex(common_idx).pct_change().fillna(0)
            m_spx = perf_metrics(spx_ret, "S&P 500")
            all_metrics.append(m_spx)

        print(f"\n  {'Gold B&H':25s} : Sharpe={m_gold['sharpe']:.3f}  "
              f"Return={m_gold['ann_return']*100:.1f}%")
        print(f"  {'Silver B&H':25s} : Sharpe={m_silver['sharpe']:.3f}  "
              f"Return={m_silver['ann_return']*100:.1f}%")

    # Sauvegarder
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.sort_values("sharpe", ascending=False)
    metrics_df.to_csv(OUTPUT_TAB / "backtest_metrics_halflife.csv", index=False)

    print(f"\n  Métriques : {OUTPUT_TAB / 'backtest_metrics_halflife.csv'}")

    # Rendements annuels
    if "Brut_ZScore" in all_equity:
        bt_brut = all_equity["Brut_ZScore"]
        cum = (1 + bt_brut["strategy"]).cumprod()
        annual = cum.resample("YE").last().pct_change().dropna()
        annual.index = annual.index.year
        annual_df = pd.DataFrame({
            "year": annual.index,
            "contrarian_halflife_pct": annual.values * 100
        })
        annual_df.to_csv(OUTPUT_TAB / "annual_returns_halflife_ml.csv", index=False)

    # Figure
    if all_equity:
        fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]})

        ax = axes[0]
        colors = {
            "Brut_ZScore": "#C00000",
            "XGBoost": "#2F5496",
            "RandomForest": "#4472C4",
            "Logistic": "#5B9BD5",
            "SVM": "#7030A0",
        }

        for strat_name, bt in all_equity.items():
            if strat_name in ["Gold B&H"]:
                continue
            cum = (1 + bt["strategy"]).cumprod()
            color = colors.get(strat_name, "#666666")
            lw = 2.0 if strat_name == "Brut_ZScore" else 1.5
            ax.plot(cum.index, cum.values, label=strat_name, color=color, linewidth=lw)

        # Gold benchmark
        cum_gold = (1 + gold_bh).cumprod()
        ax.plot(cum_gold.index, cum_gold.values, label="Gold B&H",
                color="#FFD700", linewidth=1.5, linestyle="--")
        ax.axhline(1, color="black", linewidth=0.5)
        ax.set_title("Backtest Contrarian — Holding Period = Rolling Half-Life",
                     fontsize=13, fontweight="bold")
        ax.set_ylabel("Valeur cumulée (base 1)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Panel 2 : Rolling HL utilisé
        ax2 = axes[1]
        ax2.plot(rolling_hl.index, rolling_hl.values, color="#2F5496", linewidth=0.8)
        ax2.axhline(hl_median, color="red", linestyle="--", linewidth=1,
                    label=f"Médiane={hl_median:.0f}j")
        ax2.set_ylabel("Half-Life (jours)")
        ax2.set_title("Holding Period Adaptative (= Rolling Half-Life)", fontsize=11)
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT_FIG / "fig_11_backtest_halflife_ml.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Figure : {OUTPUT_FIG / 'fig_11_backtest_halflife_ml.png'}")

    # Comparatif final
    print(f"\n{'=' * 70}")
    print("COMPARATIF : HORIZON FIXE (step_06d) vs HALF-LIFE (step_06e)")
    print(f"{'=' * 70}")
    print(f"\n  L'horizon calibré sur le half-life devrait améliorer les résultats")
    print(f"  car il aligne la holding period avec la vitesse de mean-reversion.")
    print(f"\n  Pour comparaison :")
    print(f"    - Step 06d (H=126j fixe) : Brut_ZScore Sharpe = -0.16")
    print(f"    - Step 06e (H=HL adaptatif) : voir résultats ci-dessus")

    print(f"\n{'=' * 70}")
    print("STEP 6e TERMINE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
