"""
04b_grouped_models.py — Modèles ML avec features groupées, filtrées,
walk-forward purgé, class_weight balanced, et SHAP temporel.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import xgboost as xgb
import shap
import joblib
import json
import copy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from config import (
    DATA_PROCESSED, OUTPUT_MOD, OUTPUT_TAB, OUTPUT_FIG,
    TRAIN_YEARS, TEST_MONTHS, MIN_TRAIN_OBS,
    XGBOOST_PARAMS, RF_PARAMS, RANDOM_STATE, TB_HORIZON
)

# ══════════════════════════════════════════════════════════════════
FEATURE_GROUPS = {
    "Technique": [
        "rsi_14", "bb_pct", "bb_width", "macd_hist",
        "ma_ratio", "zscore_60", "gsr_roc_14", "atr_14",
    ],
    "Momentum": [
        "gsr_ret_1d", "gsr_ret_5d", "gsr_ret_21d", "gsr_ret_63d",
    ],
    "Macro": [
        "us10y_change_21d", "term_spread", "real_rate_proxy",
        "dxy_ma_ratio", "vix_ma_ratio", "vix_change_5d",
        "m2_yoy", "cpi_momentum",
    ],
    "Cross-Asset": [
        "gold_spx_ratio", "corr_gsr_spx_60d", "corr_gsr_dxy_60d",
        "gold_ret_21d", "silver_ret_21d", "copper_ret_21d",
        "spx_drawdown",
    ],
    "Volatilite": [
        "gsr_vol_20d", "gsr_vol_60d", "vol_ratio_gs",
        "silver_vol_20d", "high_vol_regime",
    ],
}


def get_available_features(df):
    available = {}
    for group, features in FEATURE_GROUPS.items():
        present = [f for f in features if f in df.columns]
        if present:
            available[group] = present
    return available


def filter_correlated(df, features, threshold=0.85):
    if len(features) <= 1:
        return features
    corr = df[features].corr().abs()
    to_drop = set()
    for i in range(len(features)):
        if features[i] in to_drop:
            continue
        for j in range(i + 1, len(features)):
            if features[j] in to_drop:
                continue
            if corr.iloc[i, j] > threshold:
                var_i = df[features[i]].std() / (abs(df[features[i]].mean()) + 1e-10)
                var_j = df[features[j]].std() / (abs(df[features[j]].mean()) + 1e-10)
                drop = features[j] if abs(var_i) >= abs(var_j) else features[i]
                to_drop.add(drop)
    kept = [f for f in features if f not in to_drop]
    if to_drop:
        print(f"      Supprime (corr>{threshold}): {to_drop}")
    return kept


# ══════════════════════════════════════════════════════════════════
# WALK-FORWARD PURGE (López de Prado, AFML Ch.7)
# ══════════════════════════════════════════════════════════════════
def walk_forward_split(df, train_years=5, test_months=6, purge_days=21):
    dates = df.index
    train_start = dates[0]
    t_split = train_start + pd.DateOffset(years=train_years)
    splits = []
    cal_purge = int(purge_days * 1.5)

    while t_split < dates[-1]:
        test_end = min(t_split + pd.DateOffset(months=test_months), dates[-1])
        purge_start = t_split - pd.Timedelta(days=cal_purge)
        train_mask = (dates >= train_start) & (dates < purge_start)
        test_mask = (dates >= t_split) & (dates < test_end)
        purge_mask = (dates >= purge_start) & (dates < t_split)

        if train_mask.sum() >= MIN_TRAIN_OBS and test_mask.sum() > 0:
            splits.append({
                "train_idx": df.index[train_mask],
                "test_idx": df.index[test_mask],
                "n_train": train_mask.sum(),
                "n_purged": purge_mask.sum(),
                "n_test": test_mask.sum(),
                "train_end": purge_start,
                "test_start": t_split,
                "test_end": test_end,
            })
        t_split += pd.DateOffset(months=test_months)
    return splits


def compute_group_shap(shap_values, feature_cols, groups):
    group_importance = {}
    feat_to_grp = {}
    for g, fs in groups.items():
        for f in fs:
            if f in feature_cols:
                feat_to_grp[f] = g
    for group in groups:
        gf = [f for f in feature_cols if feat_to_grp.get(f) == group]
        if not gf:
            continue
        idx = [feature_cols.index(f) for f in gf]
        group_importance[group] = {
            "mean_abs_shap": np.abs(shap_values[:, idx]).mean(),
            "n_features": len(gf),
            "features": gf,
        }
    return group_importance


def main():
    print("=" * 70)
    print("STEP 4b : MODELES ML — FEATURES GROUPEES + WALK-FORWARD PURGE")
    print("=" * 70)

    df = pd.read_pickle(DATA_PROCESSED / "labeled.pkl")
    groups = get_available_features(df)

    print("\n--- Groupes de features ---")
    all_features = []
    for group, feats in groups.items():
        print(f"  {group:15s} : {feats}")
        all_features.extend(feats)

    print("\n--- Filtrage correlations intra-groupe ---")
    filtered_groups = {}
    filtered_features = []
    for group, feats in groups.items():
        print(f"  {group}:")
        kept = filter_correlated(df, feats, threshold=0.85)
        filtered_groups[group] = kept
        filtered_features.extend(kept)
        print(f"    {len(feats)} -> {len(kept)} features")

    n_feat = len(filtered_features)
    print(f"\n  Total features : {n_feat}")

    target_col = "label_tb"
    valid_mask = df[target_col].notna()
    for fc in filtered_features:
        valid_mask &= df[fc].notna()
    df_valid = df.loc[valid_mask].copy()
    X = df_valid[filtered_features]
    y = df_valid[target_col].astype(int)

    print(f"  Observations valides : {len(df_valid)}")
    print(f"  Periode : {df_valid.index[0].date()} -> {df_valid.index[-1].date()}")
    label_dist = {-1: (y==-1).sum(), 0: (y==0).sum(), 1: (y==1).sum()}
    print(f"  Labels : {label_dist}")

    # ── CLASS WEIGHTS (FIX #1) ──
    total = len(y)
    n_classes = len([v for v in label_dist.values() if v > 0])
    class_weights = {}
    for lab, count in label_dist.items():
        if count > 0:
            class_weights[lab] = total / (n_classes * count)
    print(f"  Class weights : { {k: round(v,3) for k,v in class_weights.items()} }")

    # XGBoost scale_pos_weight (pour 3 classes, on utilise sample_weight)
    sample_weights = y.map(class_weights).values

    splits = walk_forward_split(df_valid, TRAIN_YEARS, TEST_MONTHS, purge_days=TB_HORIZON)
    print(f"\n--- Walk-Forward Purge (purge={TB_HORIZON}j) : {len(splits)} folds ---\n")
    print(f"  {'Fold':>4s}  {'Train':>18s}  {'Purge':>5s}  {'Test':>18s}  {'N_tr':>5s}  {'N_te':>5s}")
    for i, s in enumerate(splits):
        print(f"  {i+1:4d}  {str(s['train_idx'][0].date()):>8s}->{str(s['train_end'].date()):>8s}"
              f"  {s['n_purged']:5d}"
              f"  {str(s['test_start'].date()):>8s}->{str(s['test_end'].date()):>8s}"
              f"  {s['n_train']:5d}  {s['n_test']:5d}")

    # ── Modèles avec class_weight balanced (FIX #1) ──
    models_def = {
        "XGBoost": xgb.XGBClassifier(
            **XGBOOST_PARAMS, use_label_encoder=False, eval_metric="mlogloss"
        ),
        "RandomForest": RandomForestClassifier(
            **RF_PARAMS, class_weight="balanced"
        ),
        "Ridge": RidgeClassifier(
            alpha=1.0, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Logistic": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, C=0.1, class_weight="balanced"
        ),
        "SVM": SVC(
            kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE,
            class_weight="balanced"
        ),
    }

    all_results = []
    all_predictions = {}

    # ── FIX #3 : Stocker importance par fold pour voir l'évolution ──
    fold_importances = []  # (fold_idx, date_start, date_end, {feature: importance})
    fold_group_importances = []  # (fold_idx, {group: shap_pct})

    for model_name, model_template in models_def.items():
        print(f"\n{'─'*55}")
        print(f"  {model_name} (class_weight=balanced)")
        print(f"{'─'*55}")

        fold_metrics = []
        model_preds = pd.Series(dtype=float)

        for i, split in enumerate(splits):
            X_train = X.loc[split["train_idx"]]
            y_train = y.loc[split["train_idx"]]
            X_test = X.loc[split["test_idx"]]
            y_test = y.loc[split["test_idx"]]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train)
            X_te_s = scaler.transform(X_test)

            model = copy.deepcopy(model_template)

            if model_name == "XGBoost":
                y_tr = y_train + 1  # remap -1,0,1 -> 0,1,2
                model.set_params(num_class=3)
                # Sample weights pour XGBoost (class_weight equivalent)
                sw_train = y_train.map(class_weights).values
                model.fit(X_tr_s, y_tr, sample_weight=sw_train)
            else:
                model.fit(X_tr_s, y_train)

            y_pred = model.predict(X_te_s)
            if model_name == "XGBoost":
                y_pred = y_pred - 1

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            fold_metrics.append({"accuracy": acc, "f1": f1, "precision": prec, "recall": rec})
            model_preds = pd.concat([model_preds, pd.Series(y_pred, index=split["test_idx"])])

            print(f"    Fold {i+1:2d}: Acc={acc:.3f}  F1={f1:.3f}  Prec={prec:.3f}")

            # ── FIX #3 : SHAP par fold (XGBoost seulement) ──
            if model_name == "XGBoost":
                try:
                    explainer = shap.TreeExplainer(model)
                    sv = explainer.shap_values(pd.DataFrame(X_te_s, columns=filtered_features))
                    shap_bull = sv[2] if isinstance(sv, list) else sv[:, :, 2]
                    mean_abs = np.abs(shap_bull).mean(axis=0)
                    feat_imp_fold = dict(zip(filtered_features, mean_abs))
                    fold_importances.append({
                        "fold": i + 1,
                        "test_start": str(split["test_start"].date()),
                        "test_end": str(split["test_end"].date()),
                        **feat_imp_fold,
                    })
                    # Group importance par fold
                    gi = compute_group_shap(shap_bull, filtered_features, filtered_groups)
                    total_shap = sum(g["mean_abs_shap"] for g in gi.values()) + 1e-10
                    fold_group_importances.append({
                        "fold": i + 1,
                        "test_start": str(split["test_start"].date()),
                        "test_end": str(split["test_end"].date()),
                        **{g: gi[g]["mean_abs_shap"] / total_shap * 100 for g in gi},
                    })
                except Exception as e:
                    print(f"      SHAP fold {i+1} skip: {e}")

                # Sauvegarder dernier modèle
                last_xgb_model = model
                last_xgb_scaler = scaler

        if fold_metrics:
            avg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
            std = {k: np.std([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
            print(f"  >>> MOYENNE : Acc={avg['accuracy']:.3f}+-{std['accuracy']:.3f}  "
                  f"F1={avg['f1']:.3f}+-{std['f1']:.3f}")
            all_results.append({
                "model": model_name, "n_features": n_feat, "n_folds": len(fold_metrics),
                **{f"avg_{k}": v for k, v in avg.items()},
                **{f"std_{k}": v for k, v in std.items()},
            })
            all_predictions[model_name] = model_preds

    # ── Résultats ──
    results_df = pd.DataFrame(all_results).sort_values("avg_f1", ascending=False)
    print(f"\n{'='*70}")
    print("COMPARAISON (walk-forward purge, balanced, features groupees)")
    print(f"{'='*70}")
    print(results_df[["model","avg_accuracy","avg_f1","std_accuracy","std_f1","n_features"]].to_string(index=False))
    results_df.to_csv(OUTPUT_TAB / "model_comparison_grouped.csv", index=False)

    pred_df = pd.DataFrame(all_predictions)
    pred_df.index.name = "date"
    pred_df.to_pickle(DATA_PROCESSED / "predictions_grouped.pkl")

    # ══════════════════════════════════════════════════════════════════
    # FIX #3 : ÉVOLUTION TEMPORELLE DU SHAP
    # ══════════════════════════════════════════════════════════════════
    if fold_importances:
        print(f"\n{'='*70}")
        print("EVOLUTION TEMPORELLE DU SHAP PAR GROUPE")
        print(f"{'='*70}")

        fi_df = pd.DataFrame(fold_importances)
        fi_df.to_csv(OUTPUT_TAB / "shap_evolution_by_fold.csv", index=False)

        gi_df = pd.DataFrame(fold_group_importances)
        gi_df.to_csv(OUTPUT_TAB / "shap_group_evolution.csv", index=False)

        print("\n  Importance groupe par fold (%) :")
        group_cols = [c for c in gi_df.columns if c not in ["fold", "test_start", "test_end"]]
        print(f"  {'Fold':>4s}  {'Periode':>20s}  " + "  ".join(f"{g:>12s}" for g in group_cols))
        for _, row in gi_df.iterrows():
            period = f"{row['test_start']}->{row['test_end']}"
            vals = "  ".join(f"{row[g]:12.1f}" for g in group_cols)
            print(f"  {int(row['fold']):4d}  {period:>20s}  {vals}")

        # ── Plot : évolution des groupes dans le temps ──
        fig, ax = plt.subplots(figsize=(14, 7))
        colors_grp = {"Technique":"#2F5496","Momentum":"#C00000","Macro":"#548235",
                       "Cross-Asset":"#BF8F00","Volatilite":"#7030A0"}
        
        x_labels = [f"F{int(r['fold'])}\n{r['test_start'][:7]}" for _, r in gi_df.iterrows()]
        bottom = np.zeros(len(gi_df))
        
        for group in group_cols:
            vals = gi_df[group].values
            ax.bar(range(len(gi_df)), vals, bottom=bottom, 
                   label=group, color=colors_grp.get(group, "gray"), alpha=0.85)
            bottom += vals
        
        ax.set_xticks(range(len(gi_df)))
        ax.set_xticklabels(x_labels, fontsize=8, rotation=45)
        ax.set_ylabel("Contribution SHAP (%)", fontsize=12)
        ax.set_title("Evolution de l'importance des groupes de features par fold", 
                     fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_FIG / "shap_group_evolution.png", dpi=300, bbox_inches="tight")
        plt.close()

        # ── Plot : top features, leur évolution ──
        feature_cols_only = [c for c in fi_df.columns if c not in ["fold","test_start","test_end"]]
        mean_imp = fi_df[feature_cols_only].mean().sort_values(ascending=False)
        top_8 = mean_imp.head(8).index.tolist()

        fig, ax = plt.subplots(figsize=(14, 7))
        feat_to_grp = {}
        for g, fs in filtered_groups.items():
            for f in fs:
                feat_to_grp[f] = g

        for feat in top_8:
            color = colors_grp.get(feat_to_grp.get(feat, ""), "gray")
            ax.plot(range(len(fi_df)), fi_df[feat].values, marker="o", markersize=4,
                    label=feat, color=color, linewidth=1.5)
        
        ax.set_xticks(range(len(fi_df)))
        ax.set_xticklabels(x_labels, fontsize=8, rotation=45)
        ax.set_ylabel("Mean |SHAP|", fontsize=12)
        ax.set_title("Evolution des top features dans le temps", fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_FIG / "shap_top_features_evolution.png", dpi=300, bbox_inches="tight")
        plt.close()

    # ── SHAP global (dernier fold) ──
    print(f"\n{'='*70}")
    print("SHAP GLOBAL (dernier fold)")
    print(f"{'='*70}")

    if 'last_xgb_model' in dir():
        cutoff = df_valid.index[-1] - pd.DateOffset(years=2)
        X_shap = df_valid.loc[cutoff:, filtered_features]
        X_shap_s = last_xgb_scaler.transform(X_shap)
        X_shap_df = pd.DataFrame(X_shap_s, columns=filtered_features, index=X_shap.index)

        explainer = shap.TreeExplainer(last_xgb_model)
        shap_values = explainer.shap_values(X_shap_df)
        shap_bull = shap_values[2] if isinstance(shap_values, list) else shap_values[:, :, 2]

        mean_shap = np.abs(shap_bull).mean(axis=0)
        feat_imp = pd.Series(mean_shap, index=filtered_features).sort_values(ascending=False)
        feat_imp.to_csv(OUTPUT_TAB / "shap_importance_grouped.csv")

        gi = compute_group_shap(shap_bull, filtered_features, filtered_groups)
        total_shap = sum(g["mean_abs_shap"] for g in gi.values())
        group_summary = []
        print("\n  Importance par GROUPE :")
        for gn in ["Technique","Momentum","Macro","Cross-Asset","Volatilite"]:
            if gn in gi:
                g = gi[gn]
                pct = g["mean_abs_shap"] / total_shap * 100
                top_f = feat_imp.reindex(g["features"]).idxmax()
                print(f"    {gn:15s} : {pct:5.1f}%  (top: {top_f})")
                group_summary.append({"group":gn,"pct_total":pct,"n_features":g["n_features"],"top_feature":top_f})

        pd.DataFrame(group_summary).to_csv(OUTPUT_TAB / "shap_group_importance.csv", index=False)

        # Plots
        colors_grp2 = {"Technique":"#2F5496","Momentum":"#C00000","Macro":"#548235",
                        "Cross-Asset":"#BF8F00","Volatilite":"#7030A0"}
        feat_to_grp2 = {}
        for g, fs in filtered_groups.items():
            for f in fs:
                feat_to_grp2[f] = g

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        gnames = [g["group"] for g in group_summary]
        gvals = [g["pct_total"] for g in group_summary]
        gcols = [colors_grp2.get(g, "gray") for g in gnames]
        bars = axes[0].barh(gnames, gvals, color=gcols)
        axes[0].set_xlabel("Contribution SHAP (%)")
        axes[0].set_title("Importance par groupe", fontsize=14, fontweight="bold")
        axes[0].invert_yaxis()
        for b,v in zip(bars,gvals):
            axes[0].text(b.get_width()+0.5, b.get_y()+b.get_height()/2, f"{v:.1f}%", va="center")
        axes[0].grid(axis="x", alpha=0.3)
        axes[1].pie(gvals, labels=gnames, colors=gcols, autopct="%1.1f%%", pctdistance=0.75, startangle=90)
        axes[1].add_artist(plt.Circle((0,0), 0.5, fc="white"))
        axes[1].set_title("Repartition SHAP", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUTPUT_FIG / "shap_group_importance.png", dpi=300, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(figsize=(12, 10))
        top_n = min(25, len(feat_imp))
        top = feat_imp.head(top_n)
        bcols = [colors_grp2.get(feat_to_grp2.get(f,""),"gray") for f in top.index]
        ax.barh(range(top_n), top.values, color=bcols)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top.index, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP|")
        ax.set_title(f"Top {top_n} Features (colorees par groupe)", fontsize=14, fontweight="bold")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(fc=c,label=g) for g,c in colors_grp2.items()], loc="lower right")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_FIG / "shap_features_by_group.png", dpi=300, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_bull, X_shap_df, max_display=20, show=False, plot_size=None)
        plt.title("SHAP Summary - Bull (+1)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUTPUT_FIG / "shap_summary_grouped.png", dpi=300, bbox_inches="tight")
        plt.close()

        joblib.dump(last_xgb_model, OUTPUT_MOD / "XGBoost_grouped.pkl")
        joblib.dump(last_xgb_scaler, OUTPUT_MOD / "XGBoost_grouped_scaler.pkl")
        with open(OUTPUT_MOD / "feature_groups.json", "w") as f:
            json.dump(filtered_groups, f, indent=2)
        with open(OUTPUT_MOD / "feature_cols_grouped.json", "w") as f:
            json.dump(filtered_features, f)

    print(f"\n{'='*70}")
    print("STEP 4b TERMINE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
