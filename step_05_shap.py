"""
05_shap.py — Analyse SHAP (SHapley Additive exPlanations)
Interprétabilité des modèles XGBoost et Random Forest.
Sortie : outputs/figures/shap_*.png, outputs/tables/shap_*.csv
"""
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

from config import DATA_PROCESSED, OUTPUT_MOD, OUTPUT_FIG, OUTPUT_TAB


def main():
    print("=" * 60)
    print("STEP 5 : ANALYSE SHAP")
    print("=" * 60)

    # Charger données et modèle
    df = pd.read_pickle(DATA_PROCESSED / "labeled.pkl")
    
    with open(OUTPUT_MOD / "feature_cols.json") as f:
        feature_cols = json.load(f)

    xgb_model = joblib.load(OUTPUT_MOD / "XGBoost_last.pkl")
    xgb_scaler = joblib.load(OUTPUT_MOD / "XGBoost_scaler.pkl")

    # Préparer les données (dernier fold de test)
    target_col = "label_tb"
    valid_mask = df[target_col].notna()
    for fc in feature_cols:
        valid_mask &= df[fc].notna()
    df_valid = df.loc[valid_mask]

    # Prendre les 2 dernières années pour SHAP
    cutoff = df_valid.index[-1] - pd.DateOffset(years=2)
    df_shap = df_valid.loc[cutoff:]
    
    X_shap = df_shap[feature_cols]
    X_shap_scaled = xgb_scaler.transform(X_shap)
    X_shap_df = pd.DataFrame(X_shap_scaled, columns=feature_cols, index=X_shap.index)

    print(f"\n  Observations pour SHAP : {len(X_shap_df)}")
    print(f"  Période : {X_shap_df.index[0].date()} -> {X_shap_df.index[-1].date()}")

    # ── SHAP TreeExplainer ──
    print("\n--- Calcul des SHAP values ---")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_shap_df)

    # shap_values est une liste de 3 arrays (classes -1, 0, 1) ou un array 3D
    if isinstance(shap_values, list):
        # Classe 1 (bull) pour l'analyse principale
        shap_bull = shap_values[2]  # index 2 = classe +1 (après remap 0,1,2)
        shap_bear = shap_values[0]  # index 0 = classe -1
    else:
        shap_bull = shap_values[:, :, 2]
        shap_bear = shap_values[:, :, 0]

    # ── 1. Summary Plot (Bee Swarm) ──
    print("\n  Generating SHAP summary plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        shap_bull, X_shap_df,
        max_display=20,
        show=False,
        plot_size=None
    )
    plt.title("SHAP Summary Plot - Classe Bull (+1)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG / "shap_summary_bull.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ── 2. SHAP Bar Plot (importance moyenne) ──
    print("  Generating SHAP bar plot...")
    mean_shap = np.abs(shap_bull).mean(axis=0)
    shap_importance = pd.Series(mean_shap, index=feature_cols).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = 20
    shap_importance.head(top_n).plot.barh(ax=ax, color="#2F5496")
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title(f"Top {top_n} Features par SHAP Importance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG / "shap_bar_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Sauvegarder le ranking
    shap_importance.to_csv(OUTPUT_TAB / "shap_importance.csv")
    print(f"\n  Top 10 SHAP features :")
    for feat, val in shap_importance.head(10).items():
        print(f"    {feat:30s} : {val:.4f}")

    # ── 3. Dependence plots (top 4 features) ──
    print("\n  Generating dependence plots...")
    top_features = shap_importance.head(4).index.tolist()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, feat in enumerate(top_features):
        ax = axes[idx // 2, idx % 2]
        feat_idx = feature_cols.index(feat)
        shap.dependence_plot(
            feat_idx, shap_bull, X_shap_df,
            ax=ax, show=False
        )
        ax.set_title(f"SHAP Dependence: {feat}", fontsize=11)
    plt.suptitle("SHAP Dependence Plots - Top 4 Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG / "shap_dependence_top4.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ── 4. SHAP par régime de marché ──
    print("\n  Generating regime-conditional SHAP...")
    y_shap = df_shap[target_col].astype(int)
    
    regime_shap = {}
    for regime, regime_name in [(-1, "Bear"), (0, "Neutral"), (1, "Bull")]:
        mask = (y_shap == regime).values
        if mask.sum() > 0:
            mean_abs = np.abs(shap_bull[mask]).mean(axis=0)
            regime_shap[regime_name] = pd.Series(mean_abs, index=feature_cols)
    
    if regime_shap:
        regime_df = pd.DataFrame(regime_shap)
        regime_df.to_csv(OUTPUT_TAB / "shap_by_regime.csv")

    print(f"\n{'=' * 60}")
    print("STEP 5 TERMINE")
    print(f"{'=' * 60}")
    print(f"\n  Figures sauvegardees dans : {OUTPUT_FIG}")
    print(f"  Tables sauvegardees dans  : {OUTPUT_TAB}")


if __name__ == "__main__":
    main()
