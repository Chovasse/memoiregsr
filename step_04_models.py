"""
04_models.py — Entraînement et évaluation des modèles ML
Walk-forward validation avec XGBoost, Random Forest, Ridge, SVM, LSTM.
Sortie : outputs/models/, outputs/tables/
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import xgboost as xgb
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

from config import (
    DATA_PROCESSED, OUTPUT_MOD, OUTPUT_TAB, OUTPUT_FIG,
    TRAIN_YEARS, TEST_MONTHS, MIN_TRAIN_OBS,
    XGBOOST_PARAMS, RF_PARAMS, RANDOM_STATE
)


def get_feature_columns(df: pd.DataFrame) -> list:
    """Retourne la liste des colonnes features (exclut prix bruts, labels, dates)."""
    exclude_prefixes = ["label_", "barrier_", "forward_"]
    raw_prices = [
        "gold", "silver", "gld", "slv", "xau_spot", "xag_spot",
        "us10y", "tips10y", "be10y", "fed_funds", "dxy", "vix",
        "spx", "copper", "oil", "mxwo", "m2", "cpi", "gsr",
        "ig_oas", "hy_oas", "cftc_gold", "cftc_silver",
        "gld_tons", "slv_tons", "gvz", "move",
    ]
    feature_cols = []
    for c in df.columns:
        if c in raw_prices:
            continue
        if any(c.startswith(p) for p in exclude_prefixes):
            continue
        feature_cols.append(c)
    return feature_cols


def walk_forward_split(df: pd.DataFrame, train_years: int = 5, test_months: int = 6):
    """
    Générateur de splits walk-forward.
    Train: train_years glissant. Test: test_months suivants.
    """
    dates = df.index
    start = dates[0]
    
    # Premier train set commence au début, dure train_years ans
    train_start = start
    train_end = train_start + pd.DateOffset(years=train_years)
    
    splits = []
    while train_end < dates[-1]:
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > dates[-1]:
            test_end = dates[-1]
        
        train_mask = (dates >= train_start) & (dates < train_end)
        test_mask = (dates >= train_end) & (dates < test_end)
        
        if train_mask.sum() >= MIN_TRAIN_OBS and test_mask.sum() > 0:
            splits.append({
                "train_start": train_start,
                "train_end": train_end,
                "test_start": train_end,
                "test_end": test_end,
                "train_idx": df.index[train_mask],
                "test_idx": df.index[test_mask],
            })
        
        # Avancer le train_start de test_months (expanding window variant)
        train_end = train_end + pd.DateOffset(months=test_months)
    
    return splits


def train_evaluate_fold(X_train, y_train, X_test, y_test, model_name, model):
    """Entraîne un modèle sur un fold et retourne les métriques."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    
    return y_pred, metrics, model, scaler


def main():
    print("=" * 60)
    print("STEP 4 : ENTRAINEMENT DES MODELES")
    print("=" * 60)

    labeled_path = DATA_PROCESSED / "labeled.pkl"
    if not labeled_path.exists():
        print("[ERR] labeled.pkl introuvable. Lancer step_03 d'abord.")
        return

    df = pd.read_pickle(labeled_path)
    
    # Features
    feature_cols = get_feature_columns(df)
    print(f"\n  Features disponibles : {len(feature_cols)}")
    
    # Label cible
    target_col = "label_tb"
    
    # Supprimer les lignes sans label ou avec NaN dans les features
    valid_mask = df[target_col].notna()
    for fc in feature_cols:
        valid_mask &= df[fc].notna()
    
    df_valid = df.loc[valid_mask].copy()
    print(f"  Observations valides : {len(df_valid)} / {len(df)}")
    print(f"  Période : {df_valid.index[0].date()} -> {df_valid.index[-1].date()}")

    X = df_valid[feature_cols]
    y = df_valid[target_col].astype(int)

    # Walk-forward splits
    splits = walk_forward_split(df_valid, TRAIN_YEARS, TEST_MONTHS)
    print(f"\n  Walk-forward : {len(splits)} folds")
    for i, s in enumerate(splits):
        print(f"    Fold {i+1}: train {s['train_start'].date()}->{s['train_end'].date()}, "
              f"test {s['test_start'].date()}->{s['test_end'].date()} "
              f"({len(s['train_idx'])}+{len(s['test_idx'])})")

    # Modèles
    models = {
        "XGBoost": xgb.XGBClassifier(**XGBOOST_PARAMS, use_label_encoder=False, eval_metric="mlogloss"),
        "RandomForest": RandomForestClassifier(**RF_PARAMS),
        "Ridge": RidgeClassifier(alpha=1.0, random_state=RANDOM_STATE),
        "Logistic": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=0.1),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE),
    }

    # Résultats par modèle et par fold
    all_results = []
    all_predictions = {}

    for model_name, model_template in models.items():
        print(f"\n{'─' * 40}")
        print(f"  Modele : {model_name}")
        print(f"{'─' * 40}")
        
        fold_metrics = []
        model_preds = pd.Series(dtype=float, name=model_name)
        
        for i, split in enumerate(splits):
            X_train = X.loc[split["train_idx"]]
            y_train = y.loc[split["train_idx"]]
            X_test = X.loc[split["test_idx"]]
            y_test = y.loc[split["test_idx"]]
            
            # Remap labels pour XGBoost (0, 1, 2 au lieu de -1, 0, 1)
            if model_name == "XGBoost":
                y_train_m = y_train + 1
                y_test_m = y_test + 1
                import copy
                model = copy.deepcopy(model_template)
                model.set_params(num_class=3)
            else:
                y_train_m = y_train
                y_test_m = y_test
                import copy
                model = copy.deepcopy(model_template)
            
            try:
                y_pred, metrics, trained_model, scaler = train_evaluate_fold(
                    X_train, y_train_m, X_test, y_test_m, model_name, model
                )
            except Exception as e:
                print(f"    Fold {i+1} ERREUR : {e}")
                continue
            
            # Remap predictions back
            if model_name == "XGBoost":
                y_pred = y_pred - 1
                y_test_m = y_test_m - 1
            
            # Recalculer métriques avec labels originaux
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            }
            
            fold_metrics.append(metrics)
            pred_series = pd.Series(y_pred, index=split["test_idx"])
            model_preds = pd.concat([model_preds, pred_series])
            
            print(f"    Fold {i+1}: Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")
        
        # Sauvegarder le dernier modèle entraîné
        if fold_metrics:
            joblib.dump(trained_model, OUTPUT_MOD / f"{model_name}_last.pkl")
            joblib.dump(scaler, OUTPUT_MOD / f"{model_name}_scaler.pkl")
        
        # Métriques moyennes
        if fold_metrics:
            avg_metrics = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
            std_metrics = {k: np.std([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
            
            print(f"\n  Moyenne : Acc={avg_metrics['accuracy']:.3f}+-{std_metrics['accuracy']:.3f}, "
                  f"F1={avg_metrics['f1']:.3f}+-{std_metrics['f1']:.3f}")
            
            all_results.append({
                "model": model_name,
                **{f"avg_{k}": v for k, v in avg_metrics.items()},
                **{f"std_{k}": v for k, v in std_metrics.items()},
                "n_folds": len(fold_metrics),
            })
            
            all_predictions[model_name] = model_preds

    # ── Tableau comparatif ──
    print(f"\n{'=' * 60}")
    print("COMPARAISON DES MODELES")
    print(f"{'=' * 60}")
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("avg_f1", ascending=False)
    print(results_df.to_string(index=False))
    results_df.to_csv(OUTPUT_TAB / "model_comparison.csv", index=False)

    # Sauvegarder les prédictions
    pred_df = pd.DataFrame(all_predictions)
    pred_df.index.name = "date"
    pred_df.to_pickle(DATA_PROCESSED / "predictions.pkl")

    # ── Feature importance (XGBoost) ──
    print(f"\n--- Feature Importance (XGBoost, dernier fold) ---")
    try:
        xgb_model = joblib.load(OUTPUT_MOD / "XGBoost_last.pkl")
        importance = pd.Series(
            xgb_model.feature_importances_, 
            index=feature_cols
        ).sort_values(ascending=False)
        
        print("  Top 15 features :")
        for feat, imp in importance.head(15).items():
            print(f"    {feat:30s} : {imp:.4f}")
        
        importance.to_csv(OUTPUT_TAB / "xgb_feature_importance.csv")
    except Exception as e:
        print(f"  [WARN] Impossible de charger XGBoost: {e}")

    # Sauvegarder feature_cols pour réutilisation
    with open(OUTPUT_MOD / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    print(f"\n{'=' * 60}")
    print("STEP 4 TERMINE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
