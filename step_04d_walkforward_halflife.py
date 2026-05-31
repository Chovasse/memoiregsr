"""
04d_walkforward_halflife.py — Walk-Forward ML avec horizon calibré Half-Life
VERSION CORRIGEE : exclusion des features qui leakent le label

PROBLEME IDENTIFIE :
  Le label est construit a partir du z-score(200j) > 1.5 sigma.
  Les features gsr_extreme_high, gsr_extreme_low, zscore_60, bb_pct, ma_ratio
  sont des proxies quasi-parfaits de ce z-score -> le modele relit le label.

CORRECTION :
  On exclut TOUTES les features derivees directement du niveau du GSR
  qui encodent implicitement le z-score ou la position relative.
  On ne garde que les features qui apportent de l'information NOUVELLE :
  macro, cross-asset, volatilite, momentum court-terme relatif.

Sortie : data/processed/predictions_halflife.pkl
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from config import (
    DATA_PROCESSED, OUTPUT_TAB, OUTPUT_MOD,
    TEST_MONTHS, MIN_TRAIN_OBS,
    XGBOOST_PARAMS, RF_PARAMS, RANDOM_STATE, N_JOBS
)

TRAIN_YEARS_HL = 3

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score
    import xgboost as xgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("[WARN] sklearn/xgboost non disponibles.")
    print("  pip install scikit-learn xgboost")


# === FEATURES QUI LEAKENT LE LABEL ===
# Le label = f(zscore_200). Toute feature qui encode le niveau relatif
# du GSR par rapport a sa moyenne longue est un proxy du label.
LEAKY_FEATURES = [
    # Proxies directs du z-score 200j (= le label)
    "gsr_extreme_high",   # == label exactement
    "gsr_extreme_low",    # == label exactement
    "zscore_60",          # correle ~0.85 avec zscore_200
    "bb_pct",             # position dans les bandes = z-score court
    "ma_ratio",           # ma50/ma200 = signe du z-score
    "ma_50",              # niveau absolu = encode le z-score
    "ma_200",             # niveau absolu = encode le z-score
    "bb_upper",           # depend du niveau GSR
    "bb_lower",           # depend du niveau GSR
    "bb_width",           # correle avec la volatilite du z-score
    "macd",               # EMA12 - EMA26 = proxy momentum/z-score CT
    "macd_signal",        # derive du MACD
    "macd_hist",          # derive du MACD
    "rsi_14",             # RSI sur le GSR = momentum qui leak le regime
]


def get_clean_feature_columns(df):
    """Features SANS leakage du label."""
    exclude = [
        "gold", "silver", "gsr", "gld", "slv", "spx", "mxwo",
        "xau_spot", "xag_spot",
        "label_halflife", "label_contr_clean", "label_tb21", "label_tb63",
        "label_tb", "label_meanrev",
        "zscore_200", "forward_return_hl", "forward_return_126",
        "horizon_used", "rolling_halflife",
        "date", "m2", "cpi",
    ] + LEAKY_FEATURES

    feat_cols = [c for c in df.columns if c not in exclude and not c.startswith("label")]
    return feat_cols


def walk_forward_halflife(df, feat_cols, label_col="label_halflife",
                          train_years=3, test_months=6):
    if "rolling_halflife" in df.columns:
        median_hl = int(df["rolling_halflife"].dropna().median())
    else:
        median_hl = 67
    purge_days = max(median_hl, 42)
    print(f"  Purge adapte au half-life : {purge_days} jours")

    valid = df.dropna(subset=[label_col] + feat_cols)
    print(f"  Observations valides (toutes classes) : {len(valid)}")

    for v in [-1, 0, 1]:
        n = (valid[label_col] == v).sum()
        print(f"    Classe {v:+d} : {n} ({n/len(valid)*100:.1f}%)")

    X = valid[feat_cols].values
    y = valid[label_col].values.astype(int)
    dates = valid.index

    train_days = train_years * 252
    test_days = test_months * 21

    print(f"  Train={train_days}j, Purge={purge_days}j, Test={test_days}j")
    print(f"  Total necessaire par fold : {train_days + purge_days + test_days}j")
    print(f"  Donnees disponibles : {len(dates)}j")

    folds = []
    start = 0
    while start + train_days + purge_days + test_days <= len(dates):
        train_end = start + train_days
        test_start = train_end + purge_days
        test_end = min(test_start + test_days, len(dates))
        folds.append({
            "train": (start, train_end),
            "test": (test_start, test_end),
            "train_dates": (dates[start], dates[train_end-1]),
            "test_dates": (dates[test_start], dates[min(test_end-1, len(dates)-1)]),
        })
        start += test_days

    print(f"  Nombre de folds : {len(folds)}")

    if len(folds) == 0:
        print("[ERR] Pas assez de donnees pour le walk-forward.")
        return None

    models = {
        "XGBoost": lambda: xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=N_JOBS,
            use_label_encoder=False, eval_metric="mlogloss"
        ),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=20,
            max_features="sqrt", class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=N_JOBS
        ),
        "Logistic": lambda: LogisticRegression(
            max_iter=1000, C=0.1, class_weight="balanced",
            random_state=RANDOM_STATE, multi_class="multinomial"
        ),
    }

    all_predictions = {name: pd.Series(np.nan, index=dates) for name in models}
    fold_metrics = []

    for fold_idx, fold in enumerate(folds):
        tr_start, tr_end = fold["train"]
        te_start, te_end = fold["test"]

        X_train, y_train = X[tr_start:tr_end], y[tr_start:tr_end]
        X_test, y_test = X[te_start:te_end], y[te_start:te_end]
        test_dates_fold = dates[te_start:te_end]

        if len(X_train) < 500 or len(X_test) < 10:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        for name, model_fn in models.items():
            try:
                model = model_fn()

                if name == "XGBoost":
                    y_train_xgb = y_train + 1
                    y_test_xgb = y_test + 1
                    classes, counts = np.unique(y_train_xgb, return_counts=True)
                    total = len(y_train_xgb)
                    weights = np.ones(len(y_train_xgb))
                    for c, cnt in zip(classes, counts):
                        weights[y_train_xgb == c] = total / (len(classes) * cnt)
                    model.fit(X_train_s, y_train_xgb, sample_weight=weights)
                    pred_xgb = model.predict(X_test_s)
                    pred = pred_xgb - 1
                else:
                    model.fit(X_train_s, y_train)
                    pred = model.predict(X_test_s)

                for j, d in enumerate(test_dates_fold):
                    if j < len(pred):
                        all_predictions[name].loc[d] = pred[j]

                acc = accuracy_score(y_test, pred)
                f1 = f1_score(y_test, pred, average="weighted", zero_division=0)
                non_flat_mask = y_test != 0
                acc_signals = accuracy_score(y_test[non_flat_mask], pred[non_flat_mask]) if non_flat_mask.sum() > 0 else 0

                fold_metrics.append({
                    "fold": fold_idx, "model": name,
                    "accuracy": acc, "f1_weighted": f1,
                    "accuracy_signals": acc_signals,
                    "n_train": len(y_train), "n_test": len(y_test),
                    "train_period": str(fold['train_dates'][0].strftime('%Y-%m')) + " -> " + str(fold['train_dates'][1].strftime('%Y-%m')),
                    "test_period": str(fold['test_dates'][0].strftime('%Y-%m')) + " -> " + str(fold['test_dates'][1].strftime('%Y-%m')),
                })

            except Exception as e:
                print(f"    [WARN] Fold {fold_idx} - {name}: {e}")
                continue

        if (fold_idx + 1) % 5 == 0 or fold_idx == len(folds) - 1:
            test_date_str = fold['test_dates'][0].strftime('%Y-%m')
            print(f"    Fold {fold_idx+1}/{len(folds)} termine ({test_date_str})")

    return all_predictions, fold_metrics, dates


def main():
    if not ML_AVAILABLE:
        print("[ERR] Impossible d executer sans sklearn/xgboost.")
        print("  pip install scikit-learn xgboost")
        return

    print("=" * 70)
    print("STEP 4d : WALK-FORWARD ML CALIBRE HALF-LIFE (SANS LEAKAGE)")
    print("=" * 70)

    labeled_path = DATA_PROCESSED / "labeled_halflife.pkl"
    if not labeled_path.exists():
        print("[ERR] labeled_halflife.pkl introuvable. Executez step_03d d abord.")
        return

    df = pd.read_pickle(labeled_path)
    print(f"\n  Donnees : {len(df)} observations")

    # Features PROPRES (sans leakage)
    feat_cols = get_clean_feature_columns(df)
    print(f"  Features candidates (SANS leakage) : {len(feat_cols)}")

    # Afficher les features exclues pour transparence
    print(f"\n  Features EXCLUES (leakage du label) :")
    for f in LEAKY_FEATURES:
        if f in df.columns:
            print(f"    X {f}")

    # Filtrer features avec trop de NaN
    valid_feats = []
    for col in feat_cols:
        if df[col].notna().sum() > len(df) * 0.5:
            valid_feats.append(col)
    feat_cols = valid_feats
    print(f"\n  Features valides finales : {len(feat_cols)}")
    print(f"  Liste : {feat_cols}")

    print(f"\n--- Walk-Forward (train={TRAIN_YEARS_HL}y, test={TEST_MONTHS}m) ---")
    result = walk_forward_halflife(df, feat_cols, "label_halflife",
                                   TRAIN_YEARS_HL, TEST_MONTHS)

    if result is None:
        return

    all_predictions, fold_metrics, dates = result

    pred_df = pd.DataFrame(all_predictions)

    # Signal brut z-score (benchmark)
    if "zscore_200" in df.columns:
        z = df["zscore_200"]
        brut = pd.Series(0, index=df.index, dtype=float)
        brut[z > 1.5] = -1
        brut[z < -1.5] = 1
        pred_df["Brut_ZScore"] = brut.reindex(pred_df.index)

    pred_df.to_pickle(DATA_PROCESSED / "predictions_halflife.pkl")

    metrics_df = pd.DataFrame(fold_metrics)
    print(f"\n--- Metriques moyennes par modele (SANS LEAKAGE) ---")
    summary = metrics_df.groupby("model")[["accuracy", "f1_weighted", "accuracy_signals"]].mean()
    print(summary.to_string())

    metrics_df.to_csv(OUTPUT_TAB / "model_comparison_halflife.csv", index=False)
    summary.to_csv(OUTPUT_TAB / "model_summary_halflife.csv")

    print(f"\n  Predictions : {DATA_PROCESSED / 'predictions_halflife.pkl'}")
    print(f"  Metriques   : {OUTPUT_TAB / 'model_comparison_halflife.csv'}")

    # Statistiques de predictions
    print(f"\n--- Statistiques des predictions OOS ---")
    for name in all_predictions:
        preds = pred_df[name].dropna()
        if len(preds) > 0:
            n_short = (preds == -1).sum()
            n_flat = (preds == 0).sum()
            n_long = (preds == 1).sum()
            print(f"  {name:15s} : {len(preds)} pred, "
                  f"Short={n_short} ({n_short/len(preds)*100:.0f}%), "
                  f"Flat={n_flat} ({n_flat/len(preds)*100:.0f}%), "
                  f"Long={n_long} ({n_long/len(preds)*100:.0f}%)")

    # Diagnostic
    avg_acc = metrics_df["accuracy"].mean()
    if avg_acc > 0.85:
        print(f"\n  [WARN] Accuracy moyenne = {avg_acc:.1%} — possiblement du leakage residuel")
        print(f"  Verifiez les features les plus importantes via SHAP")
    elif avg_acc > 0.5:
        print(f"\n  [OK] Accuracy moyenne = {avg_acc:.1%} — resultats credibles")
    else:
        print(f"\n  [INFO] Accuracy moyenne = {avg_acc:.1%} — modele peu performant")

    print(f"\n{'=' * 70}")
    print("STEP 4d TERMINE — Executez step_06e_backtest_halflife.py")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
