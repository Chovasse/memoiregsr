"""
03d_labeling_halflife.py — Labels contrariens calibrés sur le Half-Life

INNOVATION : l'horizon du forward return (et donc la durée de validation
du signal) est calibré sur le rolling half-life du GSR, au lieu d'un
horizon fixe arbitraire.

Logique :
  - z > +1.5σ → label -1 (anticipation reversion down du GSR)
  - z < -1.5σ → label +1 (anticipation reversion up du GSR)
  - |z| < 1.5σ → label 0 (flat)

Le forward return est calculé sur un horizon = rolling half-life au temps t.
Cela garantit qu'on évalue le signal sur la durée théorique de mean-reversion.

Sortie : data/processed/labeled_halflife.pkl
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from config import DATA_PROCESSED, TB_ATR_PERIOD


def estimate_rolling_halflife(gsr, window=504):
    """
    Estime le half-life par régression OLS sur le processus Ornstein-Uhlenbeck.
    dX(t) = a + b*X(t-1) + epsilon
    Half-life = -ln(2) / b (si b < 0, ie mean-reverting)

    Fenêtre glissante de 504 jours (2 ans trading).
    """
    hl_series = pd.Series(index=gsr.index, dtype=float)
    values = gsr.values

    for i in range(window, len(values)):
        chunk = values[i-window:i]
        if np.isnan(chunk).sum() > window * 0.1:
            continue

        y = np.diff(chunk)
        x = chunk[:-1]
        mask = ~(np.isnan(y) | np.isnan(x))
        if mask.sum() < 100:
            continue

        y_c, x_c = y[mask], x[mask]
        x_m = x_c.mean()
        denom = np.sum((x_c - x_m)**2)
        if denom == 0:
            continue
        b = np.sum((x_c - x_m) * (y_c - y_c.mean())) / denom

        if b < 0:
            hl = -np.log(2) / b
            if 20 < hl < 1000:
                hl_series.iloc[i] = hl

    return hl_series


def contrarian_labels_halflife(gsr, rolling_hl, lookback=200, zscore_entry=1.5):
    """
    Labels contrariens avec horizon calibré sur le half-life.

    Le forward return est calculé sur horizon = half-life au temps t.
    Si le half-life n'est pas disponible, on utilise la dernière valeur connue.
    """
    ma = gsr.rolling(lookback).mean()
    std = gsr.rolling(lookback).std()
    zscore = (gsr - ma) / std.replace(0, np.nan)

    # Forward-fill le half-life pour avoir une valeur à chaque point
    hl_filled = rolling_hl.ffill().fillna(126)  # default 126j si pas encore estimé

    n = len(gsr)
    labels = np.zeros(n)
    fwd_ret = np.full(n, np.nan)
    horizons_used = np.full(n, np.nan)

    for i in range(lookback, n):
        z = zscore.iloc[i]
        if np.isnan(z):
            labels[i] = np.nan
            continue

        # Label basé sur le z-score
        if z > zscore_entry:
            labels[i] = -1  # GSR trop haut → short
        elif z < -zscore_entry:
            labels[i] = 1   # GSR trop bas → long
        else:
            labels[i] = 0   # flat

        # Forward return calibré sur le half-life
        horizon = int(hl_filled.iloc[i])
        horizon = max(21, min(horizon, 504))  # borné entre 21j et 504j
        horizons_used[i] = horizon

        end_idx = min(i + horizon, n - 1)
        if end_idx > i:
            fwd_ret[i] = (gsr.iloc[end_idx] - gsr.iloc[i]) / gsr.iloc[i]

    # Pas de label pour les dernières observations
    max_horizon = int(hl_filled.iloc[-1]) if not np.isnan(hl_filled.iloc[-1]) else 264
    labels[-max_horizon:] = np.nan

    return pd.DataFrame({
        "label_halflife": labels,
        "zscore_200": zscore.values,
        "forward_return_hl": fwd_ret,
        "horizon_used": horizons_used,
    }, index=gsr.index)


def main():
    print("=" * 70)
    print("STEP 3d : LABELING CONTRARIAN CALIBRE SUR LE HALF-LIFE")
    print("=" * 70)

    features_path = DATA_PROCESSED / "features.pkl"
    if not features_path.exists():
        print("[ERR] features.pkl introuvable.")
        return None

    df = pd.read_pickle(features_path)
    gsr = df["gsr"].dropna()
    print(f"\n  GSR : {len(gsr)} observations")

    # Estimer le rolling half-life
    print(f"\n--- Estimation du Rolling Half-Life (504j) ---")
    rolling_hl = estimate_rolling_halflife(gsr, window=504)
    valid_hl = rolling_hl.dropna()
    print(f"  Observations avec HL valide : {len(valid_hl)}")
    print(f"  Half-life médian : {valid_hl.median():.0f} jours")
    print(f"  Half-life moyen  : {valid_hl.mean():.0f} jours")
    print(f"  Q25 / Q75 : {valid_hl.quantile(0.25):.0f} / {valid_hl.quantile(0.75):.0f}")

    # Labels calibrés
    print(f"\n--- Labels Contrariens (horizon = rolling half-life) ---")
    result = contrarian_labels_halflife(gsr, rolling_hl, lookback=200, zscore_entry=1.5)

    df["label_halflife"] = result["label_halflife"]
    df["zscore_200"] = result["zscore_200"]
    df["forward_return_hl"] = result["forward_return_hl"]
    df["horizon_used"] = result["horizon_used"]
    df["rolling_halflife"] = rolling_hl

    valid = df["label_halflife"].dropna()
    n_total = len(valid)
    n_short = (valid == -1).sum()
    n_long = (valid == 1).sum()
    n_flat = (valid == 0).sum()

    print(f"\n  Distribution :")
    print(f"    Short (-1) : {n_short:5d} ({n_short/n_total*100:.1f}%)")
    print(f"    Flat  ( 0) : {n_flat:5d} ({n_flat/n_total*100:.1f}%)")
    print(f"    Long  (+1) : {n_long:5d} ({n_long/n_total*100:.1f}%)")
    print(f"    Exposition : {(n_short+n_long)/n_total*100:.1f}%")

    avg_horizon = result["horizon_used"].dropna().mean()
    print(f"\n  Horizon moyen utilisé : {avg_horizon:.0f} jours")

    # Sauvegarder
    out_path = DATA_PROCESSED / "labeled_halflife.pkl"
    df.to_pickle(out_path)
    print(f"\n  Sauvegarde : {out_path}")

    # Sauvegarder aussi le rolling half-life séparément
    hl_path = DATA_PROCESSED / "rolling_halflife.pkl"
    rolling_hl.to_pickle(hl_path)

    print(f"\n{'=' * 70}")
    print("STEP 3d TERMINE")
    print(f"{'=' * 70}")

    return df


if __name__ == "__main__":
    main()
