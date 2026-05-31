"""
04e_walkforward_monthly.py — Walk-Forward ML avec réentraînement MENSUEL
VERSION COMPARATIVE : step = 1 mois (21j) vs step = 6 mois (126j)

HYPOTHESE :
  Un réentraînement plus fréquent permet au modèle de mieux s'adapter
  aux changements de régime du GSR, améliorant potentiellement les
  performances de trading.

PARAMETRES :
  - Train : 3 ans (756 jours)
  - Test : 3 mois (63 jours) — assez pour des métriques fiables
  - Step : 1 mois (21 jours) — le modèle est réentraîné chaque mois
  - Purge : adapté au half-life médian (~67 jours)
  - Les folds se chevauchent (sliding window, pas expanding)

CORRECTION LEAKAGE : identique à step_04d (14 features exclues)

Sortie : data/processed/predictions_halflife_monthly.pkl
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from config import (
    DATA_PROCESSED, OUTPUT_TAB, OUTPUT_MOD,
    XGBOOST_PARAMS, RF_PARAMS, RANDOM_STATE, N_JOBS
)

TRAIN_YEARS = 3
TEST_MONTHS = 3   # fenêtre de test = 3 mois (63 obs)
STEP_MONTHS = 1   # réentraînement tous les mois (21 obs d'avancement)

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

LEAKY_FEATURES = [
    "gsr_extreme_high", "gsr_extreme_low", "zscore_60", "bb_pct",
    "ma_ratio", "ma_50", "ma_200", "bb_upper", "bb_lower", "bb_width",
    "macd", "macd_signal", "macd_hist", "rsi_14",
]


def get_clean_feature_columns(df):
    exclude = [
        "gold", "silver", "gsr", "gld", "slv", "spx", "mxwo",
        "xau_spot", "xag_spot",
        "label_halflife", "label_contr_clean", "label_tb21", "label_tb63",
        "label_tb", "label_meanrev",
        "zscore_200", "forward_return_hl", "forward_return_126",
        "horizon_used", "rolling_halflife",
        "date", "m2", "cpi",
    ] + LEAKY_FEATURES
    return [c for c in df.columns if c not in exclude and not c.startswith("label")]


def walk_forward_monthly(df, feat_cols, label_col="label_halflife",
                         train_years=3, test_months=3, step_months=1):
    """Walk-forward avec pas mensuel et fenêtre test de 3 mois."""
    if "rolling_halflife" in df.columns:
        median_hl = int(df["rolling_halflife"].dropna().median())
    else:
        median_hl = 67
    purge_days = max(median_hl, 42)
    print(f"  Purge adapté au half-life : {purge_days} jours")

    valid = df.dropna(subset=[label_col] + feat_cols)
    print(f"  Observations valides : {len(valid)}")

    for v in [-1, 0, 1]:
        n = (valid[label_col] == v).sum()
        print(f"    Classe {v:+d} : {n} ({n/len(valid)*100:.1f}%)")

    X = valid[feat_cols].values
    y = valid[label_col].values.astype(int)
    dates = valid.index

    train_days = train_years * 252
    test_days = test_months * 21
    step_days = step_months * 21

    print(f"  Train={train_days}j, Purge={purge_days}j, Test={test_days}j, Step={step_days}j")
    print(f"  Données disponibles : {len(dates)}j")

    # Construire les folds avec pas glissant
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
        start += step_days  # avance de 1 mois au lieu de test_days

    print(f"  Nombre de folds : {len(folds)}")

    if len(folds) == 0:
        print("[ERR] Pas assez de données pour le walk-forward.")
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

    # Pour gérer les folds chevauchants : on garde la prédiction
    # la plus RECENTE pour chaque date (= modèle le plus à jour)
    all_predictions = {name: {} for name in models}
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
                    classes, counts = np.unique(y_train_xgb, return_counts=True)
                    total = len(y_train_xgb)
                    weights = np.ones(len(y_train_xgb))
                    for c, cnt in zip(classes, counts):
                        weights[y_train_xgb == c] = total / (len(classes) * cnt)
                    model.fit(X_train_s, y_train_xgb, sample_weight=weights)
                    pred = model.predict(X_test_s) - 1
                else:
                    model.fit(X_train_s, y_train)
                    pred = model.predict(X_test_s)

                # Écraser les prédictions existantes avec celles du fold le plus récent
                for j, d in enumerate(test_dates_fold):
                    if j < len(pred):
                        all_predictions[name][d] = pred[j]

                acc = accuracy_score(y_test, pred)
                f1 = f1_score(y_test, pred, average="weighted", zero_division=0)
                non_flat = y_test != 0
                acc_sig = accuracy_score(y_test[non_flat], pred[non_flat]) if non_flat.sum() > 0 else 0

                fold_metrics.append({
                    "fold": fold_idx, "model": name,
                    "accuracy": acc, "f1_weighted": f1,
                    "accuracy_signals": acc_sig,
                    "n_train": len(y_train), "n_test": len(y_test),
                    "train_end": fold['train_dates'][1].strftime('%Y-%m'),
                    "test_period": fold['test_dates'][0].strftime('%Y-%m') + " -> " + fold['test_dates'][1].strftime('%Y-%m'),
                })

            except Exception as e:
                print(f"    [WARN] Fold {fold_idx} - {name}: {e}")

        if (fold_idx + 1) % 10 == 0 or fold_idx == len(folds) - 1:
            print(f"    Fold {fold_idx+1}/{len(folds)} ({fold['test_dates'][0].strftime('%Y-%m')})")

    # Convertir les dicts en Series
    pred_series = {}
    for name, pdict in all_predictions.items():
        s = pd.Series(pdict, dtype=float)
        s = s.sort_index()
        pred_series[name] = s

    return pred_series, fold_metrics, dates


def main():
    if not ML_AVAILABLE:
        print("[ERR] pip install scikit-learn xgboost")
        return

    print("=" * 70)
    print("STEP 4e : WALK-FORWARD MENSUEL (step=1mois, test=3mois)")
    print("=" * 70)

    labeled_path = DATA_PROCESSED / "labeled_halflife.pkl"
    if not labeled_path.exists():
        print("[ERR] labeled_halflife.pkl introuvable. Exécutez step_03d.")
        return

    df = pd.read_pickle(labeled_path)
    print(f"\n  Données : {len(df)} observations")

    feat_cols = get_clean_feature_columns(df)

    valid_feats = [c for c in feat_cols if df[c].notna().sum() > len(df) * 0.5]
    feat_cols = valid_feats
    print(f"  Features (sans leakage) : {len(feat_cols)}")

    print(f"\n  Features EXCLUES (leakage) :")
    for f in LEAKY_FEATURES:
        if f in df.columns:
            print(f"    X {f}")

    print(f"\n--- Walk-Forward Mensuel (train={TRAIN_YEARS}y, test={TEST_MONTHS}m, step={STEP_MONTHS}m) ---")
    result = walk_forward_monthly(df, feat_cols, "label_halflife",
                                   TRAIN_YEARS, TEST_MONTHS, STEP_MONTHS)

    if result is None:
        return

    pred_series, fold_metrics, dates = result

    pred_df = pd.DataFrame(pred_series)

    # Signal brut z-score (benchmark)
    if "zscore_200" in df.columns:
        z = df["zscore_200"]
        brut = pd.Series(0, index=df.index, dtype=float)
        brut[z > 1.5] = -1
        brut[z < -1.5] = 1
        pred_df["Brut_ZScore"] = brut.reindex(pred_df.index)

    pred_df.to_pickle(DATA_PROCESSED / "predictions_halflife_monthly.pkl")

    metrics_df = pd.DataFrame(fold_metrics)
    print(f"\n--- Métriques moyennes (RÉENTRAÎNEMENT MENSUEL) ---")
    summary = metrics_df.groupby("model")[["accuracy", "f1_weighted", "accuracy_signals"]].agg(["mean", "std"])
    print(summary.to_string())

    # Comparaison avec le 6-mois
    print(f"\n--- Comparaison Step 6 mois vs Step 1 mois ---")
    old_metrics_path = OUTPUT_TAB / "model_comparison_halflife.csv"
    if old_metrics_path.exists():
        old = pd.read_csv(old_metrics_path)
        old_summary = old.groupby("model")[["accuracy", "f1_weighted"]].mean()
        new_summary = metrics_df.groupby("model")[["accuracy", "f1_weighted"]].mean()
        for model in new_summary.index:
            if model in old_summary.index:
                old_acc = old_summary.loc[model, "accuracy"]
                new_acc = new_summary.loc[model, "accuracy"]
                diff = new_acc - old_acc
                print(f"  {model:15s} : 6m={old_acc:.3f} -> 1m={new_acc:.3f} (Δ={diff:+.3f})")
    else:
        print("  (pas de résultats 6-mois pour comparaison)")

    metrics_df.to_csv(OUTPUT_TAB / "model_comparison_halflife_monthly.csv", index=False)
    summary_flat = metrics_df.groupby("model")[["accuracy", "f1_weighted", "accuracy_signals"]].mean()
    summary_flat.to_csv(OUTPUT_TAB / "model_summary_halflife_monthly.csv")

    print(f"\n  Predictions : {DATA_PROCESSED / 'predictions_halflife_monthly.pkl'}")
    print(f"  Métriques   : {OUTPUT_TAB / 'model_comparison_halflife_monthly.csv'}")

    # Stats prédictions
    print(f"\n--- Distribution des prédictions OOS ---")
    for name in pred_series:
        preds = pred_df[name].dropna()
        if len(preds) > 0:
            n_short = (preds == -1).sum()
            n_flat = (preds == 0).sum()
            n_long = (preds == 1).sum()
            print(f"  {name:15s} : {len(preds)} pred, "
                  f"Short={n_short} ({n_short/len(preds)*100:.0f}%), "
                  f"Flat={n_flat} ({n_flat/len(preds)*100:.0f}%), "
                  f"Long={n_long} ({n_long/len(preds)*100:.0f}%)")

    print(f"\n{'=' * 70}")
    print("STEP 4e TERMINÉ — Exécutez step_06f_backtest_monthly.py")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
