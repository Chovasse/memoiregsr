"""
06b_backtest_grouped.py — Backtest avec les prédictions du modèle groupé
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


def compute_strategy_returns(predictions, gsr, gold, silver, tc_bps=10, slippage_bps=5):
    """
    Rendement du spread : Long Gold / Short Silver quand signal=+1,
    Short Gold / Long Silver quand signal=-1.
    Rendement = position * (gold_ret - silver_ret) net de coûts.
    """
    common_idx = predictions.dropna().index.intersection(gsr.dropna().index)
    common_idx = common_idx.intersection(gold.dropna().index).intersection(silver.dropna().index)
    pred = predictions.loc[common_idx]
    gold_ret = gold.reindex(common_idx).pct_change().fillna(0)
    silver_ret = silver.reindex(common_idx).pct_change().fillna(0)
    spread_ret = gold_ret - silver_ret  # rendement du spread gold/silver
    gsr_ret = gsr.reindex(common_idx).pct_change().fillna(0)
    position = pred.shift(1)  # signal de la veille
    tc = (tc_bps + slippage_bps) / 10000
    costs = position.diff().abs() * tc
    costs = costs.fillna(0)
    strat_ret = position * spread_ret - costs
    strat_ret = strat_ret.fillna(0)
    return pd.DataFrame({
        "strategy": strat_ret,
        "gsr_bh": gsr_ret,
        "gold_bh": gold_ret,
        "silver_bh": silver_ret,
        "spread_bh": spread_ret,
        "position": position,
    }, index=common_idx)


def perf_metrics(returns, name="Strategy"):
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
    win_rate = (returns > 0).sum() / (returns != 0).sum() if (returns != 0).sum() > 0 else 0
    return {
        "name": name, "ann_return": ann_ret, "sharpe": sharpe,
        "sortino": sortino, "max_drawdown": max_dd, "calmar": calmar,
        "win_rate": win_rate, "total_return": cum.iloc[-1] - 1,
    }


def main():
    print("=" * 70)
    print("STEP 6b : BACKTEST (FEATURES GROUPEES)")
    print("=" * 70)

    df = pd.read_pickle(DATA_PROCESSED / "labeled.pkl")
    preds = pd.read_pickle(DATA_PROCESSED / "predictions_grouped.pkl")

    gsr, gold, silver = df["gsr"], df["gold"], df["silver"]

    all_metrics = []
    bt_results = {}

    for model_name in preds.columns:
        pred = preds[model_name].dropna()
        if len(pred) == 0:
            continue
        bt = compute_strategy_returns(pred, gsr, gold, silver, TRANSACTION_COST_BPS, SLIPPAGE_BPS)
        bt_results[model_name] = bt
        m = perf_metrics(bt["strategy"], model_name)
        all_metrics.append(m)
        print(f"  {model_name:15s} : Sharpe={m['sharpe']:.3f}  Return={m['ann_return']*100:.2f}%  MaxDD={m['max_drawdown']*100:.1f}%")

    # Benchmarks
    first = list(bt_results.values())[0]
    for bn, bc in [("Gold B&H", "gold_bh"), ("Silver B&H", "silver_bh"), ("Spread B&H", "spread_bh")]:
        if bc in first.columns:
            all_metrics.append(perf_metrics(first[bc], bn))

    metrics_df = pd.DataFrame(all_metrics).sort_values("sharpe", ascending=False)
    print(f"\n{metrics_df[['name','ann_return','sharpe','sortino','max_drawdown','win_rate']].to_string(index=False)}")
    metrics_df.to_csv(OUTPUT_TAB / "backtest_metrics_grouped.csv", index=False)

    # ── RENDEMENTS ANNUELS ──
    print(f"\n--- Rendements annuels ---")
    best_model = metrics_df.iloc[0]["name"]
    if best_model in bt_results:
        bt_best = bt_results[best_model]
    else:
        bt_best = first
    strat_cum = (1 + bt_best["strategy"]).cumprod()
    annual_returns = strat_cum.resample("YE").last().pct_change().dropna()
    annual_returns.index = annual_returns.index.year
    print(f"  Meilleur modèle : {best_model}")
    print(f"  {'Année':>6s}  {'Rendement':>10s}")
    for yr, ret in annual_returns.items():
        print(f"  {yr:>6d}  {ret*100:>9.2f}%")
    annual_df = pd.DataFrame({"year": annual_returns.index, "return_pct": annual_returns.values * 100})
    annual_df.to_csv(OUTPUT_TAB / "backtest_annual_returns.csv", index=False)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [2, 1]})
    colors = {"XGBoost": "#2F5496", "RandomForest": "#C00000", "Ridge": "#548235",
              "Logistic": "#BF8F00", "SVM": "#7030A0", "Gold B&H": "#FFD700", "Silver B&H": "#C0C0C0"}
    ax = axes[0]
    for mn, bt in bt_results.items():
        ax.plot((1 + bt["strategy"]).cumprod(), label=mn, color=colors.get(mn, "gray"), linewidth=1.5)
    ax.plot((1 + first["gold_bh"]).cumprod(), label="Gold B&H", color="#FFD700", linestyle="--", alpha=0.7)
    ax.plot((1 + first["silver_bh"]).cumprod(), label="Silver B&H", color="#C0C0C0", linestyle="--", alpha=0.7)
    ax.axhline(1, color="black", linewidth=0.5)
    ax.set_title("Performance - Spread Gold/Silver vs Benchmarks", fontsize=14, fontweight="bold")
    ax.set_ylabel("Valeur cumulée (base 1)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Subplot : rendements annuels
    ax2 = axes[1]
    bar_colors = ["#2F5496" if r >= 0 else "#C00000" for r in annual_returns.values]
    ax2.bar(annual_returns.index, annual_returns.values * 100, color=bar_colors, alpha=0.8)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("Année")
    ax2.set_ylabel("Rendement (%)")
    ax2.set_title(f"Rendements annuels ({best_model})", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG / "backtest_equity_grouped.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n{'=' * 70}")
    print("STEP 6b TERMINE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
