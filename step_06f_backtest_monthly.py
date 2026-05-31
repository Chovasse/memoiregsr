"""
06f_backtest_monthly.py — Backtest comparatif : réentraînement mensuel vs semestriel

Compare les résultats du walk-forward mensuel (step_04e) avec ceux
du walk-forward semestriel (step_04d) sur le même backtest engine.

Sortie :
  - outputs/tables/backtest_metrics_monthly_vs_semester.csv
  - outputs/figures/fig_monthly_vs_semester.png
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
    common_idx = predictions.dropna().index
    common_idx = common_idx.intersection(gold.dropna().index).intersection(silver.dropna().index)

    pred = predictions.reindex(common_idx).fillna(0)
    gold_c = gold.reindex(common_idx)
    silver_c = silver.reindex(common_idx)
    hl_c = rolling_hl.reindex(common_idx).ffill().fillna(126)

    gold_ret = gold_c.pct_change().fillna(0)
    silver_ret = silver_c.pct_change().fillna(0)
    spread_ret = gold_ret - silver_ret

    position = pd.Series(0.0, index=common_idx)
    entry_idx = None
    cur_horizon = 126

    for i in range(1, len(common_idx)):
        if entry_idx is not None:
            days_held = i - entry_idx
            if days_held >= cur_horizon:
                position.iloc[i] = 0
                entry_idx = None
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

    tc = (tc_bps + slippage_bps) / 10000
    costs = position.diff().abs().fillna(0) * tc
    strat_ret = position.shift(1).fillna(0) * spread_ret - costs

    return pd.DataFrame({
        "strategy": strat_ret,
        "gold_bh": gold_ret,
        "silver_bh": silver_ret,
        "position": position,
    }, index=common_idx)


def perf_metrics(returns, name="Strategy"):
    if len(returns) == 0 or returns.std() == 0:
        return {"name": name, "ann_return": 0, "sharpe": 0, "sortino": 0,
                "max_drawdown": 0, "calmar": 0, "win_rate": 0, "exposure_pct": 0}
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
    win_rate = (returns > 0).sum() / active if active > 0 else 0
    exposure = active / len(returns) * 100
    return {"name": name, "ann_return": ann_ret, "sharpe": sharpe,
            "sortino": sortino, "max_drawdown": max_dd, "calmar": calmar,
            "win_rate": win_rate, "exposure_pct": exposure}


def main():
    print("=" * 70)
    print("STEP 6f : BACKTEST COMPARATIF — MENSUEL vs SEMESTRIEL")
    print("=" * 70)

    labeled_path = DATA_PROCESSED / "labeled_halflife.pkl"
    monthly_path = DATA_PROCESSED / "predictions_halflife_monthly.pkl"
    semester_path = DATA_PROCESSED / "predictions_halflife.pkl"

    if not labeled_path.exists():
        print("[ERR] labeled_halflife.pkl introuvable.")
        return
    if not monthly_path.exists():
        print("[ERR] predictions_halflife_monthly.pkl introuvable. Exécutez step_04e.")
        return

    df = pd.read_pickle(labeled_path)
    preds_monthly = pd.read_pickle(monthly_path)
    gold, silver = df["gold"], df["silver"]

    if "rolling_halflife" in df.columns:
        rolling_hl = df["rolling_halflife"]
    else:
        hl_path = DATA_PROCESSED / "rolling_halflife.pkl"
        rolling_hl = pd.read_pickle(hl_path) if hl_path.exists() else pd.Series(126, index=df.index)

    # Charger les prédictions semestrielles si disponibles
    has_semester = semester_path.exists()
    if has_semester:
        preds_semester = pd.read_pickle(semester_path)
    else:
        print("  [INFO] Pas de prédictions semestrielles — comparaison impossible")
        preds_semester = None

    all_metrics = []

    # ── Backtest mensuel ──
    print(f"\n--- Stratégies MENSUEL (step=1 mois) ---")
    monthly_equity = {}
    for col in preds_monthly.columns:
        pred = preds_monthly[col].dropna()
        if len(pred) == 0:
            continue
        bt = backtest_halflife_strategy(pred, gold, silver, rolling_hl,
                                        TRANSACTION_COST_BPS, SLIPPAGE_BPS)
        m = perf_metrics(bt["strategy"], f"{col}_monthly")
        m["retrain_freq"] = "mensuel"
        m["model"] = col
        expo = (bt["position"].abs() > 0).sum() / len(bt) * 100
        m["exposure_pct"] = expo
        all_metrics.append(m)
        monthly_equity[col] = bt
        print(f"  {col:20s} : Sharpe={m['sharpe']:.3f}  Return={m['ann_return']*100:.1f}%  MaxDD={m['max_drawdown']*100:.1f}%  Expo={expo:.0f}%")

    # ── Backtest semestriel ──
    semester_equity = {}
    if has_semester:
        print(f"\n--- Stratégies SEMESTRIEL (step=6 mois) ---")
        for col in preds_semester.columns:
            pred = preds_semester[col].dropna()
            if len(pred) == 0:
                continue
            bt = backtest_halflife_strategy(pred, gold, silver, rolling_hl,
                                            TRANSACTION_COST_BPS, SLIPPAGE_BPS)
            m = perf_metrics(bt["strategy"], f"{col}_semester")
            m["retrain_freq"] = "semestriel"
            m["model"] = col
            expo = (bt["position"].abs() > 0).sum() / len(bt) * 100
            m["exposure_pct"] = expo
            all_metrics.append(m)
            semester_equity[col] = bt
            print(f"  {col:20s} : Sharpe={m['sharpe']:.3f}  Return={m['ann_return']*100:.1f}%  MaxDD={m['max_drawdown']*100:.1f}%  Expo={expo:.0f}%")

    # ── Benchmarks ──
    if monthly_equity:
        first = list(monthly_equity.values())[0]
        ci = first.index
        gold_bh = gold.reindex(ci).pct_change().fillna(0)
        silver_bh = silver.reindex(ci).pct_change().fillna(0)
        m_gold = perf_metrics(gold_bh, "Gold B&H")
        m_gold["retrain_freq"] = "benchmark"
        m_gold["model"] = "Gold B&H"
        m_silver = perf_metrics(silver_bh, "Silver B&H")
        m_silver["retrain_freq"] = "benchmark"
        m_silver["model"] = "Silver B&H"
        all_metrics.extend([m_gold, m_silver])
        if "spx" in df.columns:
            spx_ret = df["spx"].reindex(ci).pct_change().fillna(0)
            m_spx = perf_metrics(spx_ret, "S&P 500")
            m_spx["retrain_freq"] = "benchmark"
            m_spx["model"] = "S&P 500"
            all_metrics.append(m_spx)

    # ── Tableau comparatif ──
    print(f"\n{'=' * 70}")
    print("COMPARATIF : MENSUEL vs SEMESTRIEL")
    print(f"{'=' * 70}")
    metrics_df = pd.DataFrame(all_metrics)

    for model in ["Brut_ZScore", "XGBoost", "RandomForest", "Logistic"]:
        row_m = metrics_df[(metrics_df["model"] == model) & (metrics_df["retrain_freq"] == "mensuel")]
        row_s = metrics_df[(metrics_df["model"] == model) & (metrics_df["retrain_freq"] == "semestriel")]
        if len(row_m) > 0 and len(row_s) > 0:
            sm = row_m.iloc[0]["sharpe"]
            ss = row_s.iloc[0]["sharpe"]
            rm = row_m.iloc[0]["ann_return"]
            rs = row_s.iloc[0]["ann_return"]
            print(f"  {model:15s} : Sharpe 6m={ss:.3f} → 1m={sm:.3f} (Δ={sm-ss:+.3f})  "
                  f"Return 6m={rs*100:.1f}% → 1m={rm*100:.1f}%")
        elif len(row_m) > 0:
            sm = row_m.iloc[0]["sharpe"]
            rm = row_m.iloc[0]["ann_return"]
            print(f"  {model:15s} : Mensuel Sharpe={sm:.3f}  Return={rm*100:.1f}%")

    metrics_df.to_csv(OUTPUT_TAB / "backtest_metrics_monthly_vs_semester.csv", index=False)

    # ── Figure comparative ──
    if monthly_equity:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})
        ax = axes[0]

        colors_m = {"Brut_ZScore": "#C00000", "XGBoost": "#2F5496", "RandomForest": "#4472C4", "Logistic": "#5B9BD5"}
        colors_s = {"Brut_ZScore": "#FF9999", "XGBoost": "#95B3D7", "RandomForest": "#B4C7E7", "Logistic": "#C5D9F1"}

        for name, bt in monthly_equity.items():
            cum = (1 + bt["strategy"]).cumprod()
            c = colors_m.get(name, "#333333")
            ax.plot(cum.index, cum.values, label=f"{name} (mensuel)", color=c, linewidth=2.0)

        for name, bt in semester_equity.items():
            cum = (1 + bt["strategy"]).cumprod()
            c = colors_s.get(name, "#999999")
            ax.plot(cum.index, cum.values, label=f"{name} (6 mois)", color=c, linewidth=1.2, linestyle="--")

        # Gold benchmark
        if monthly_equity:
            ci = list(monthly_equity.values())[0].index
            cum_gold = (1 + gold.reindex(ci).pct_change().fillna(0)).cumprod()
            ax.plot(cum_gold.index, cum_gold.values, label="Gold B&H", color="#FFD700", linewidth=1.5, linestyle=":")

        ax.axhline(1, color="black", linewidth=0.5)
        ax.set_title("Réentraînement Mensuel vs Semestriel — Courbes de Performance", fontsize=13, fontweight="bold")
        ax.set_ylabel("Valeur cumulée (base 1)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel 2 : Sharpe rolling 252j
        ax2 = axes[1]
        for name in ["XGBoost", "Brut_ZScore"]:
            if name in monthly_equity:
                roll_sharpe = monthly_equity[name]["strategy"].rolling(252).apply(
                    lambda x: x.mean()/x.std()*np.sqrt(252) if x.std()>0 else 0, raw=True)
                ax2.plot(roll_sharpe.index, roll_sharpe.values, label=f"{name} (mensuel)",
                        color=colors_m.get(name, "#333"), linewidth=1.5)
            if name in semester_equity:
                roll_sharpe = semester_equity[name]["strategy"].rolling(252).apply(
                    lambda x: x.mean()/x.std()*np.sqrt(252) if x.std()>0 else 0, raw=True)
                ax2.plot(roll_sharpe.index, roll_sharpe.values, label=f"{name} (6 mois)",
                        color=colors_s.get(name, "#999"), linewidth=1.0, linestyle="--")

        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.set_ylabel("Sharpe Rolling (252j)")
        ax2.set_title("Sharpe Ratio Glissant — Stabilité Temporelle", fontsize=11)
        ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT_FIG / "fig_monthly_vs_semester.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\n  Figure : {OUTPUT_FIG / 'fig_monthly_vs_semester.png'}")

    print(f"  Métriques : {OUTPUT_TAB / 'backtest_metrics_monthly_vs_semester.csv'}")

    print(f"\n{'=' * 70}")
    print("STEP 6f TERMINÉ")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
