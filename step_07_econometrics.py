"""
07_econometrics.py — Tests économétriques (stationnarité, cointégration)
ADF, KPSS, Johansen, analyse des résidus.
Sortie : outputs/tables/econometrics_*.csv
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from config import DATA_PROCESSED, OUTPUT_TAB


def adf_test(series: pd.Series, name: str) -> dict:
    """Test Augmented Dickey-Fuller."""
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "series": name,
        "test": "ADF",
        "statistic": result[0],
        "p_value": result[1],
        "lags": result[2],
        "critical_1%": result[4]["1%"],
        "critical_5%": result[4]["5%"],
        "critical_10%": result[4]["10%"],
        "stationary": result[1] < 0.05,
    }


def kpss_test(series: pd.Series, name: str) -> dict:
    """Test KPSS."""
    result = kpss(series.dropna(), regression="c", nlags="auto")
    return {
        "series": name,
        "test": "KPSS",
        "statistic": result[0],
        "p_value": result[1],
        "critical_1%": result[3]["1%"],
        "critical_5%": result[3]["5%"],
        "critical_10%": result[3]["10%"],
        "stationary": result[1] > 0.05,  # H0 = stationnaire pour KPSS
    }


def johansen_test(series1: pd.Series, series2: pd.Series, name: str) -> dict:
    """Test de cointégration de Johansen."""
    data = pd.concat([series1, series2], axis=1).dropna()
    result = coint_johansen(data, det_order=0, k_ar_diff=2)
    
    # Trace statistic
    trace_stat = result.lr1
    trace_crit = result.cvt  # critical values at 90%, 95%, 99%
    
    # Max eigenvalue statistic
    max_stat = result.lr2
    max_crit = result.cvm
    
    return {
        "pair": name,
        "trace_r0": trace_stat[0],
        "trace_r0_cv95": trace_crit[0, 1],
        "trace_r0_reject": trace_stat[0] > trace_crit[0, 1],
        "trace_r1": trace_stat[1],
        "trace_r1_cv95": trace_crit[1, 1],
        "trace_r1_reject": trace_stat[1] > trace_crit[1, 1],
        "max_r0": max_stat[0],
        "max_r0_cv95": max_crit[0, 1],
        "max_r0_reject": max_stat[0] > max_crit[0, 1],
        "n_obs": len(data),
    }


def main():
    print("=" * 60)
    print("STEP 7 : TESTS ECONOMETRIQUES")
    print("=" * 60)

    df = pd.read_pickle(DATA_PROCESSED / "master_daily.pkl")
    
    # ── 1. Tests de stationnarité ──
    print("\n--- Tests de stationnarite ---")
    series_to_test = {
        "GSR": df["gsr"],
        "Log(GSR)": np.log(df["gsr"]),
        "Delta(GSR)": df["gsr"].diff(),
        "Gold": df["gold"],
        "Silver": df["silver"],
        "Log(Gold)": np.log(df["gold"]),
        "Log(Silver)": np.log(df["silver"]),
        "Delta(Log(Gold))": np.log(df["gold"]).diff(),
        "Delta(Log(Silver))": np.log(df["silver"]).diff(),
    }
    
    stationarity_results = []
    for name, series in series_to_test.items():
        s = series.dropna()
        if len(s) == 0:
            continue
        adf = adf_test(s, name)
        kpss_r = kpss_test(s, name)
        stationarity_results.append(adf)
        stationarity_results.append(kpss_r)
        
        adf_status = "Stationnaire" if adf["stationary"] else "Non-stationnaire"
        kpss_status = "Stationnaire" if kpss_r["stationary"] else "Non-stationnaire"
        print(f"  {name:25s} | ADF: {adf['statistic']:8.4f} (p={adf['p_value']:.4f}) [{adf_status}] | "
              f"KPSS: {kpss_r['statistic']:8.4f} (p={kpss_r['p_value']:.4f}) [{kpss_status}]")
    
    stat_df = pd.DataFrame(stationarity_results)
    stat_df.to_csv(OUTPUT_TAB / "stationarity_tests.csv", index=False)

    # ── 2. Test de cointégration (Johansen) ──
    print("\n--- Tests de cointegration (Johansen) ---")
    coint_results = []
    
    # Gold-Silver (niveau)
    if "gold" in df.columns and "silver" in df.columns:
        gold = np.log(df["gold"].dropna())
        silver = np.log(df["silver"].dropna())
        
        # Full sample
        j_full = johansen_test(gold, silver, "Log(Gold)-Log(Silver) Full")
        coint_results.append(j_full)
        print(f"\n  Full Sample ({j_full['n_obs']} obs):")
        print(f"    Trace r=0: {j_full['trace_r0']:.2f} (CV95={j_full['trace_r0_cv95']:.2f}) "
              f"{'REJECT' if j_full['trace_r0_reject'] else 'fail to reject'}")
        
        # Sub-periods
        mid = len(gold) // 2
        periods = {
            "2000-2013": (gold.index[0], gold.index[mid]),
            "2013-2026": (gold.index[mid], gold.index[-1]),
        }
        for pname, (start, end) in periods.items():
            g_sub = gold.loc[start:end]
            s_sub = silver.loc[start:end]
            if len(g_sub) > 100:
                j_sub = johansen_test(g_sub, s_sub, f"Log(Gold)-Log(Silver) {pname}")
                coint_results.append(j_sub)
                print(f"\n  {pname} ({j_sub['n_obs']} obs):")
                print(f"    Trace r=0: {j_sub['trace_r0']:.2f} (CV95={j_sub['trace_r0_cv95']:.2f}) "
                      f"{'REJECT' if j_sub['trace_r0_reject'] else 'fail to reject'}")
    
    coint_df = pd.DataFrame(coint_results)
    coint_df.to_csv(OUTPUT_TAB / "cointegration_tests.csv", index=False)

    # ── 3. Statistiques descriptives du GSR ──
    print("\n--- Statistiques descriptives detaillees ---")
    gsr = df["gsr"].dropna()
    
    desc_stats = {
        "N": len(gsr),
        "Mean": gsr.mean(),
        "Std": gsr.std(),
        "Skewness": gsr.skew(),
        "Kurtosis": gsr.kurtosis(),
        "Min": gsr.min(),
        "Q1": gsr.quantile(0.25),
        "Median": gsr.median(),
        "Q3": gsr.quantile(0.75),
        "Max": gsr.max(),
        "JB_stat": stats.jarque_bera(gsr)[0],
        "JB_pvalue": stats.jarque_bera(gsr)[1],
    }
    
    for k, v in desc_stats.items():
        print(f"  {k:15s} : {v:.4f}")
    
    desc_df = pd.DataFrame([desc_stats])
    desc_df.to_csv(OUTPUT_TAB / "gsr_descriptive_stats.csv", index=False)

    # ── 4. Autocorrélation (Ljung-Box) ──
    print("\n--- Test Ljung-Box (autocorrelation) ---")
    gsr_ret = gsr.pct_change().dropna()
    lb_result = acorr_ljungbox(gsr_ret, lags=[5, 10, 20, 40], return_df=True)
    print(lb_result.to_string())
    lb_result.to_csv(OUTPUT_TAB / "ljung_box_test.csv")

    print(f"\n{'=' * 60}")
    print("STEP 7 TERMINE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
