"""
03_labeling.py — Triple-Barrier Labeling (López de Prado)
Crée les labels pour la classification des régimes de marché.
Sortie : data/processed/labeled.pkl
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from config import (
    DATA_PROCESSED, TB_HORIZON, TB_PT_MULT, TB_SL_MULT, TB_ATR_PERIOD
)


def compute_atr(series: pd.Series, period: int = 14) -> pd.Series:
    """ATR simplifié pour une série de prix unique."""
    high = series.rolling(2).max()
    low = series.rolling(2).min()
    tr = high - low
    return tr.rolling(period).mean()


def triple_barrier_label(
    prices: pd.Series,
    horizon: int = 21,
    pt_mult: float = 1.5,
    sl_mult: float = 1.0,
    atr_period: int = 14
) -> pd.DataFrame:
    """
    Triple-barrier labeling.
    
    Pour chaque date t :
    - Barrière haute (take-profit) : prix + pt_mult * ATR
    - Barrière basse (stop-loss) : prix - sl_mult * ATR
    - Barrière temporelle : t + horizon jours
    
    Label :
     1 = take-profit touché en premier (mouvement haussier)
    -1 = stop-loss touché en premier (mouvement baissier)
     0 = ni l'un ni l'autre à l'horizon (range/neutre)
    """
    atr = compute_atr(prices, atr_period)
    n = len(prices)
    
    labels = np.zeros(n)
    touch_dates = np.full(n, np.nan)
    returns = np.zeros(n)
    
    for i in range(n - 1):
        if np.isnan(atr.iloc[i]) or atr.iloc[i] == 0:
            labels[i] = np.nan
            continue
        
        entry = prices.iloc[i]
        upper = entry + pt_mult * atr.iloc[i]
        lower = entry - sl_mult * atr.iloc[i]
        end_idx = min(i + horizon, n - 1)
        
        label = 0
        for j in range(i + 1, end_idx + 1):
            p = prices.iloc[j]
            if p >= upper:
                label = 1
                touch_dates[i] = j - i
                break
            elif p <= lower:
                label = -1
                touch_dates[i] = j - i
                break
        
        if label == 0:
            # Pas de barrière touchée : label basé sur le signe du retour
            final_ret = (prices.iloc[end_idx] - entry) / entry
            returns[i] = final_ret
            # On garde 0 pour les mouvements trop faibles
            if abs(final_ret) > 0.005:  # seuil de 0.5%
                label = 1 if final_ret > 0 else -1
        else:
            returns[i] = (prices.iloc[min(i + int(touch_dates[i]), n-1)] - entry) / entry
        
        labels[i] = label
    
    # Dernières observations : pas assez de futur
    labels[-horizon:] = np.nan
    
    result = pd.DataFrame({
        "label": labels,
        "barrier_days": touch_dates,
        "forward_return": returns,
    }, index=prices.index)
    
    return result


def create_simple_labels(gsr: pd.Series, horizon: int = 21) -> pd.Series:
    """Labels simples basés sur le rendement forward (backup)."""
    fwd_ret = gsr.pct_change(horizon).shift(-horizon)
    
    # Terciles
    q33 = fwd_ret.quantile(0.33)
    q67 = fwd_ret.quantile(0.67)
    
    labels = pd.Series(0, index=gsr.index, name="label_simple")
    labels[fwd_ret > q67] = 1    # hausse
    labels[fwd_ret < q33] = -1   # baisse
    
    return labels


def main():
    print("=" * 60)
    print("STEP 3 : TRIPLE-BARRIER LABELING")
    print("=" * 60)

    features_path = DATA_PROCESSED / "features.pkl"
    if not features_path.exists():
        print("[ERR] features.pkl introuvable. Lancer step_02 d'abord.")
        return None

    df = pd.read_pickle(features_path)
    gsr = df["gsr"].dropna()
    print(f"\n  GSR : {len(gsr)} observations")

    # Triple-barrier labeling
    print(f"\n--- Triple-Barrier (horizon={TB_HORIZON}j, PT={TB_PT_MULT}x, SL={TB_SL_MULT}x ATR) ---")
    tb = triple_barrier_label(
        gsr, 
        horizon=TB_HORIZON,
        pt_mult=TB_PT_MULT, 
        sl_mult=TB_SL_MULT,
        atr_period=TB_ATR_PERIOD
    )
    
    df["label_tb"] = tb["label"]
    df["barrier_days"] = tb["barrier_days"]
    df["forward_return"] = tb["forward_return"]

    # Distribution des labels
    valid = df["label_tb"].dropna()
    print(f"\n  Distribution triple-barrier :")
    for lab in [-1, 0, 1]:
        count = (valid == lab).sum()
        pct = count / len(valid) * 100
        name = {-1: "Bear (-1)", 0: "Neutre (0)", 1: "Bull (+1)"}[lab]
        print(f"    {name:12s} : {count:5d} ({pct:.1f}%)")

    # Labels simples (backup)
    print(f"\n--- Labels simples (terciles, horizon={TB_HORIZON}j) ---")
    df["label_simple"] = create_simple_labels(gsr, TB_HORIZON)
    valid_s = df["label_simple"].dropna()
    for lab in [-1, 0, 1]:
        count = (valid_s == lab).sum()
        pct = count / len(valid_s) * 100
        print(f"    Label {lab:+d} : {count:5d} ({pct:.1f}%)")

    # Label binaire (pour binary classification)
    df["label_binary"] = np.where(df["label_tb"] > 0, 1, 0)
    df.loc[df["label_tb"].isna(), "label_binary"] = np.nan
    print(f"\n  Label binaire : {(df['label_binary']==1).sum()} positifs / {(df['label_binary']==0).sum()} negatifs")

    # Sauvegarde
    out_path = DATA_PROCESSED / "labeled.pkl"
    df.to_pickle(out_path)
    print(f"\n  Sauvegarde : {out_path}")

    print(f"\n{'=' * 60}")
    print("STEP 3 TERMINE")
    print(f"{'=' * 60}")

    return df


if __name__ == "__main__":
    df = main()
