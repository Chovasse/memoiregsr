"""
02_features.py — Feature Engineering
Construit toutes les features pour le modèle ML à partir du master_daily.
Sortie : data/processed/features.pkl
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from config import DATA_PROCESSED, OUTPUT_TAB


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features techniques sur le GSR."""
    gsr = df["gsr"].copy()

    # RSI(14)
    delta = gsr.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Bollinger Bands (20, 2)
    ma20 = gsr.rolling(20).mean()
    std20 = gsr.rolling(20).std()
    df["bb_upper"] = ma20 + 2 * std20
    df["bb_lower"] = ma20 - 2 * std20
    df["bb_pct"] = (gsr - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / ma20

    # MACD(12, 26, 9)
    ema12 = gsr.ewm(span=12, adjust=False).mean()
    ema26 = gsr.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Moving averages
    df["ma_50"] = gsr.rolling(50).mean()
    df["ma_200"] = gsr.rolling(200).mean()
    df["ma_ratio"] = df["ma_50"] / df["ma_200"]  # > 1 = golden cross

    # Z-Score(60)
    ma60 = gsr.rolling(60).mean()
    std60 = gsr.rolling(60).std()
    df["zscore_60"] = (gsr - ma60) / std60.replace(0, np.nan)

    # ATR(14) du GSR
    high = gsr.rolling(2).max()
    low = gsr.rolling(2).min()
    tr = high - low
    df["atr_14"] = tr.rolling(14).mean()

    # Momentum
    df["gsr_ret_1d"] = gsr.pct_change(1)
    df["gsr_ret_5d"] = gsr.pct_change(5)
    df["gsr_ret_21d"] = gsr.pct_change(21)
    df["gsr_ret_63d"] = gsr.pct_change(63)

    # Volatilité réalisée
    log_ret = np.log(gsr / gsr.shift(1))
    df["gsr_vol_20d"] = log_ret.rolling(20).std() * np.sqrt(252)
    df["gsr_vol_60d"] = log_ret.rolling(60).std() * np.sqrt(252)

    # Rate of Change
    df["gsr_roc_14"] = (gsr / gsr.shift(14) - 1) * 100

    return df


def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features macro-économiques."""
    # Rendements et variations
    for col in ["gold", "silver", "spx", "dxy", "oil", "copper"]:
        if col in df.columns:
            df[f"{col}_ret_1d"] = df[col].pct_change(1)
            df[f"{col}_ret_5d"] = df[col].pct_change(5)
            df[f"{col}_ret_21d"] = df[col].pct_change(21)

    # Ratio or/cuivre (indicateur cyclique)
    if "copper" in df.columns:
        df["gold_copper_ratio"] = df["gold"] / df["copper"]

    # Spread taux nominal - réel = breakeven
    if "us10y" in df.columns and "tips10y" in df.columns:
        if "be10y" in df.columns:
            df["real_rate_proxy"] = df["us10y"] - df["be10y"]
        else:
            df["real_rate_proxy"] = df["tips10y"]

    # Variation VIX
    if "vix" in df.columns:
        df["vix_change_5d"] = df["vix"].diff(5)
        df["vix_ma_ratio"] = df["vix"] / df["vix"].rolling(20).mean()

    # Variation DXY
    if "dxy" in df.columns:
        df["dxy_ma_ratio"] = df["dxy"] / df["dxy"].rolling(50).mean()

    # Variation taux
    if "us10y" in df.columns:
        df["us10y_change_21d"] = df["us10y"].diff(21)

    # Terme spread proxy (10Y - Fed Funds)
    if "us10y" in df.columns and "fed_funds" in df.columns:
        df["term_spread"] = df["us10y"] - df["fed_funds"]

    # M2 growth (YoY si mensuel)
    if "m2" in df.columns:
        df["m2_yoy"] = df["m2"].pct_change(252)  # ~12 mois en daily

    # CPI momentum
    if "cpi" in df.columns:
        df["cpi_momentum"] = df["cpi"].pct_change(63)  # ~3 mois

    return df


def add_cross_asset_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features cross-asset et flux."""
    # Gold vs S&P500 ratio
    if "gold" in df.columns and "spx" in df.columns:
        df["gold_spx_ratio"] = df["gold"] / df["spx"]

    # GLD volume (si disponible)
    # ETF premium/discount proxy
    if "gld" in df.columns and "xau_spot" in df.columns:
        df["gld_premium"] = (df["gld"] * 10 - df["xau_spot"]) / df["xau_spot"]  # GLD ~ 1/10 oz

    # SLV premium
    if "slv" in df.columns and "xag_spot" in df.columns:
        df["slv_premium"] = (df["slv"] - df["xag_spot"]) / df["xag_spot"]

    # Corrélation roulante GSR vs SPX (60j)
    if "spx" in df.columns:
        gsr_ret = df["gsr"].pct_change()
        spx_ret = df["spx"].pct_change()
        df["corr_gsr_spx_60d"] = gsr_ret.rolling(60).corr(spx_ret)

    # Corrélation roulante GSR vs DXY
    if "dxy" in df.columns:
        gsr_ret = df["gsr"].pct_change()
        dxy_ret = df["dxy"].pct_change()
        df["corr_gsr_dxy_60d"] = gsr_ret.rolling(60).corr(dxy_ret)

    # Vol réalisée argent (proxy VXSLV)
    if "silver" in df.columns:
        silver_log_ret = np.log(df["silver"] / df["silver"].shift(1))
        df["silver_vol_20d"] = silver_log_ret.rolling(20).std() * np.sqrt(252)

    # Vol ratio gold/silver
    if "gold" in df.columns and "silver" in df.columns:
        gold_log_ret = np.log(df["gold"] / df["gold"].shift(1))
        gold_vol = gold_log_ret.rolling(20).std()
        silver_log_ret = np.log(df["silver"] / df["silver"].shift(1))
        silver_vol = silver_log_ret.rolling(20).std()
        df["vol_ratio_gs"] = gold_vol / silver_vol.replace(0, np.nan)

    return df


def add_regime_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Indicateurs de régime de marché."""
    # SPX drawdown
    if "spx" in df.columns:
        spx_peak = df["spx"].expanding().max()
        df["spx_drawdown"] = (df["spx"] - spx_peak) / spx_peak

    # VIX regime (high vol > 25)
    if "vix" in df.columns:
        df["high_vol_regime"] = (df["vix"] > 25).astype(int)

    # GSR regime (extrêmes)
    gsr = df["gsr"]
    gsr_ma200 = gsr.rolling(200).mean()
    gsr_std200 = gsr.rolling(200).std()
    df["gsr_extreme_high"] = (gsr > gsr_ma200 + 1.5 * gsr_std200).astype(int)
    df["gsr_extreme_low"] = (gsr < gsr_ma200 - 1.5 * gsr_std200).astype(int)

    return df


def main():
    print("=" * 60)
    print("STEP 2 : FEATURE ENGINEERING")
    print("=" * 60)

    # Charger le master
    master_path = DATA_PROCESSED / "master_daily.pkl"
    if not master_path.exists():
        print("[ERR] master_daily.pkl introuvable. Lancer step_01 d'abord.")
        return None

    df = pd.read_pickle(master_path)
    print(f"\n  Master charge : {df.shape[0]} lignes x {df.shape[1]} colonnes")

    n0 = df.shape[1]

    # Features techniques
    print("\n--- Features techniques (GSR) ---")
    df = add_technical_features(df)
    n1 = df.shape[1]
    print(f"  +{n1 - n0} features techniques")

    # Features macro
    print("\n--- Features macro ---")
    df = add_macro_features(df)
    n2 = df.shape[1]
    print(f"  +{n2 - n1} features macro")

    # Features cross-asset
    print("\n--- Features cross-asset ---")
    df = add_cross_asset_features(df)
    n3 = df.shape[1]
    print(f"  +{n3 - n2} features cross-asset")

    # Indicateurs de régime
    print("\n--- Indicateurs de regime ---")
    df = add_regime_indicators(df)
    n4 = df.shape[1]
    print(f"  +{n4 - n3} indicateurs regime")

    # Résumé
    print(f"\n  Total : {n4} colonnes ({n4 - n0} features creees)")

    # Identifier les colonnes features (exclure prix bruts et GSR)
    raw_cols = list(DAILY_FILES_NAMES()) + list(MONTHLY_FILES_NAMES()) + ["gsr"]
    feature_cols = [c for c in df.columns if c not in raw_cols]
    print(f"  Features utilisables : {len(feature_cols)}")

    # Sauvegarder la liste des features
    feature_list = pd.DataFrame({
        "feature": feature_cols,
        "nan_count": [df[c].isna().sum() for c in feature_cols],
        "nan_pct": [df[c].isna().sum() / len(df) * 100 for c in feature_cols],
    })
    feature_list.to_csv(OUTPUT_TAB / "feature_list.csv", index=False)

    # Sauvegarder
    out_path = DATA_PROCESSED / "features.pkl"
    df.to_pickle(out_path)
    print(f"\n  Sauvegarde : {out_path}")

    # Stats descriptives du GSR
    print("\n--- Statistiques descriptives GSR ---")
    gsr = df["gsr"].dropna()
    stats = {
        "Count": len(gsr),
        "Mean": gsr.mean(),
        "Std": gsr.std(),
        "Min": gsr.min(),
        "25%": gsr.quantile(0.25),
        "Median": gsr.median(),
        "75%": gsr.quantile(0.75),
        "Max": gsr.max(),
        "Skew": gsr.skew(),
        "Kurtosis": gsr.kurtosis(),
    }
    for k, v in stats.items():
        print(f"  {k:12s} : {v:.4f}")

    print(f"\n{'=' * 60}")
    print("STEP 2 TERMINE")
    print(f"{'=' * 60}")

    return df


def DAILY_FILES_NAMES():
    from config import DAILY_FILES
    return [v[2] for v in DAILY_FILES.values()]

def MONTHLY_FILES_NAMES():
    from config import MONTHLY_FILES
    return [v[2] for v in MONTHLY_FILES.values()]


if __name__ == "__main__":
    df = main()
