"""
06_backtest.py — Backtest de la stratégie de trading basée sur les prédictions ML
Sortie : outputs/tables/backtest_*.csv, outputs/figures/backtest_*.png
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import (
    DATA_PROCESSED, OUTPUT_FIG, OUTPUT_TAB,
    TRANSACTION_COST_BPS, SLIPPAGE_BPS, INITIAL_CAPITAL
)
import warnings
warnings.filterwarnings("ignore")


def compute_strategy_returns(
    predictions: pd.Series,
    gsr: pd.Series,
    gold: pd.Series,
    silver: pd.Series,
    tc_bps: int = 10,
    slippage_bps: int = 5,
) -> pd.DataFrame:
    """
    Stratégie : 
    - Signal +1 (bull GSR) → Long Gold / Short Silver (GSR monte)
    - Signal -1 (bear GSR) → Short Gold / Long Silver (GSR baisse)
    - Signal  0 → Flat (pas de position)
    
    Rendement = variation du GSR × position.
    On applique les coûts de transaction à chaque changement de position.
    """
    # Aligner les séries
    common_idx = predictions.dropna().index.intersection(gsr.dropna().index)
    pred = predictions.loc[common_idx]
    gsr_aligned = gsr.loc[common_idx]
    
    # Rendement journalier du GSR
    gsr_ret = gsr_aligned.pct_change()
    
    # Position : décalée d'un jour (on trade le lendemain du signal)
    position = pred.shift(1)
    
    # Coûts de transaction
    tc = (tc_bps + slippage_bps) / 10000
    position_change = position.diff().abs()
    costs = position_change * tc
    costs = costs.fillna(0)
    
    # Rendement de la stratégie
    strat_ret = position * gsr_ret - costs
    strat_ret = strat_ret.fillna(0)
    
    # Buy & Hold benchmarks
    gold_ret = gold.reindex(common_idx).pct_change().fillna(0)
    silver_ret = silver.reindex(common_idx).pct_change().fillna(0)
    gsr_bh_ret = gsr_ret.fillna(0)
    
    result = pd.DataFrame({
        "strategy": strat_ret,
        "gsr_bh": gsr_bh_ret,
        "gold_bh": gold_ret,
        "silver_bh": silver_ret,
        "position": position,
        "costs": costs,
    }, index=common_idx)
    
    return result


def compute_performance_metrics(returns: pd.Series, name: str = "Strategy") -> dict:
    """Calcule les métriques de performance."""
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    # Sortino (downside deviation)
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0
    
    # Max drawdown
    cum_ret = (1 + returns).cumprod()
    peak = cum_ret.expanding().max()
    drawdown = (cum_ret - peak) / peak
    max_dd = drawdown.min()
    
    # Calmar
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    
    # Win rate
    win_rate = (returns > 0).sum() / (returns != 0).sum() if (returns != 0).sum() > 0 else 0
    
    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    
    return {
        "name": name,
        "ann_return": ann_ret,
        "ann_volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_return": cum_ret.iloc[-1] - 1 if len(cum_ret) > 0 else 0,
        "n_trades": int((returns != 0).sum()),
    }


def plot_equity_curves(bt_results: dict, output_path):
    """Trace les courbes de performance cumulée."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [3, 1, 1]})
    
    colors = {
        "XGBoost": "#2F5496",
        "RandomForest": "#C00000",
        "Ridge": "#548235",
        "Logistic": "#BF8F00",
        "SVM": "#7030A0",
        "Gold B&H": "#FFD700",
        "Silver B&H": "#C0C0C0",
    }
    
    # Panel 1: Equity curves
    ax1 = axes[0]
    for model_name, bt_df in bt_results.items():
        cum_ret = (1 + bt_df["strategy"]).cumprod()
        color = colors.get(model_name, "gray")
        ax1.plot(cum_ret.index, cum_ret.values, label=model_name, color=color, linewidth=1.5)
    
    # Benchmarks (from first model's data)
    first_bt = list(bt_results.values())[0]
    ax1.plot((1 + first_bt["gold_bh"]).cumprod(), label="Gold B&H", 
             color=colors["Gold B&H"], linewidth=1, linestyle="--", alpha=0.7)
    ax1.plot((1 + first_bt["silver_bh"]).cumprod(), label="Silver B&H",
             color=colors["Silver B&H"], linewidth=1, linestyle="--", alpha=0.7)
    
    ax1.set_ylabel("Valeur cumulee (base 1)", fontsize=11)
    ax1.set_title("Performance cumulee des strategies ML vs Benchmarks", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1, color="black", linewidth=0.5, linestyle="-")
    
    # Panel 2: Drawdown du meilleur modèle
    ax2 = axes[1]
    best_model = max(bt_results.keys(), 
                     key=lambda k: (1 + bt_results[k]["strategy"]).cumprod().iloc[-1])
    best_cum = (1 + bt_results[best_model]["strategy"]).cumprod()
    best_peak = best_cum.expanding().max()
    best_dd = (best_cum - best_peak) / best_peak
    ax2.fill_between(best_dd.index, best_dd.values, 0, alpha=0.4, color=colors.get(best_model, "blue"))
    ax2.set_ylabel("Drawdown", fontsize=11)
    ax2.set_title(f"Drawdown - {best_model}", fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Position
    ax3 = axes[2]
    pos = bt_results[best_model]["position"]
    ax3.fill_between(pos.index, pos.values, 0, alpha=0.5, 
                     where=pos > 0, color="green", label="Long GSR")
    ax3.fill_between(pos.index, pos.values, 0, alpha=0.5,
                     where=pos < 0, color="red", label="Short GSR")
    ax3.set_ylabel("Position", fontsize=11)
    ax3.set_title(f"Positions - {best_model}", fontsize=11)
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 60)
    print("STEP 6 : BACKTEST")
    print("=" * 60)

    # Charger données
    df = pd.read_pickle(DATA_PROCESSED / "labeled.pkl")
    predictions = pd.read_pickle(DATA_PROCESSED / "predictions.pkl")

    gsr = df["gsr"]
    gold = df["gold"]
    silver = df["silver"]

    print(f"\n  Predictions disponibles : {list(predictions.columns)}")
    print(f"  Période predictions : {predictions.index[0].date()} -> {predictions.index[-1].date()}")

    # Backtest par modèle
    bt_results = {}
    all_metrics = []

    for model_name in predictions.columns:
        pred = predictions[model_name].dropna()
        if len(pred) == 0:
            continue
        
        print(f"\n--- {model_name} ---")
        bt_df = compute_strategy_returns(
            pred, gsr, gold, silver,
            tc_bps=TRANSACTION_COST_BPS,
            slippage_bps=SLIPPAGE_BPS,
        )
        bt_results[model_name] = bt_df
        
        metrics = compute_performance_metrics(bt_df["strategy"], model_name)
        all_metrics.append(metrics)
        
        print(f"  Return ann.: {metrics['ann_return']*100:.2f}%")
        print(f"  Sharpe     : {metrics['sharpe']:.3f}")
        print(f"  Sortino    : {metrics['sortino']:.3f}")
        print(f"  Max DD     : {metrics['max_drawdown']*100:.2f}%")
        print(f"  Win Rate   : {metrics['win_rate']*100:.1f}%")
        print(f"  Calmar     : {metrics['calmar']:.3f}")

    # Benchmarks
    print(f"\n--- Benchmarks ---")
    first_bt = list(bt_results.values())[0]
    for bench_name, bench_col in [("Gold B&H", "gold_bh"), ("Silver B&H", "silver_bh"), ("GSR B&H", "gsr_bh")]:
        metrics = compute_performance_metrics(first_bt[bench_col], bench_name)
        all_metrics.append(metrics)
        print(f"  {bench_name}: Return={metrics['ann_return']*100:.2f}%, Sharpe={metrics['sharpe']:.3f}, MaxDD={metrics['max_drawdown']*100:.2f}%")

    # Tableau final
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.sort_values("sharpe", ascending=False)
    
    print(f"\n{'=' * 60}")
    print("TABLEAU COMPARATIF FINAL")
    print(f"{'=' * 60}")
    
    display_cols = ["name", "ann_return", "sharpe", "sortino", "max_drawdown", "calmar", "win_rate"]
    print(metrics_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    metrics_df.to_csv(OUTPUT_TAB / "backtest_metrics.csv", index=False)

    # Graphiques
    print("\n  Generating equity curve plot...")
    plot_equity_curves(bt_results, OUTPUT_FIG / "backtest_equity_curves.png")

    # Performance par année
    print("\n--- Performance annuelle (meilleur modele) ---")
    best_model = metrics_df.iloc[0]["name"]
    if best_model in bt_results:
        annual = bt_results[best_model]["strategy"].resample("Y").apply(
            lambda x: (1 + x).prod() - 1
        )
        annual_df = pd.DataFrame({
            "year": annual.index.year,
            "return": annual.values,
        })
        print(annual_df.to_string(index=False))
        annual_df.to_csv(OUTPUT_TAB / "annual_returns.csv", index=False)

    print(f"\n{'=' * 60}")
    print("STEP 6 TERMINE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
