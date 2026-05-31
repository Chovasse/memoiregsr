"""
main.py — Lanceur du pipeline complet
Usage : python3 main.py [mode]

Modes :
  (sans argument) — pipeline principal (momentum 21j)
  meanrev         — pipeline mean-reversion (TB63 + contrarian biaisé)
  clean           — pipeline contrarian PROPRE (sans look-ahead)
  all             — tout (principal + mean-rev + clean)
  
Steps individuels : 1, 2, 3, 3b, 3c, 4, 4b, 4c, 4d, 6, 6b, 6c, 6d, 7
"""
import sys
import time


def run_step(step_id):
    steps = {
        "1":   ("Chargement des donnees",              "step_01_load_data"),
        "2":   ("Feature Engineering",                  "step_02_features"),
        "3":   ("Triple-Barrier Labeling (21j)",        "step_03_labeling"),
        "3b":  ("Labeling Mean-Reversion (63j+contr)",  "step_03b_labeling_meanrev"),
        "3c":  ("Labeling Contrarian CLEAN",            "step_03c_labeling_contrarian_clean"),
        "4":   ("Modeles ML (toutes features)",         "step_04_models"),
        "4b":  ("Modeles ML (features groupees)",       "step_04b_grouped_models"),
        "4c":  ("Modeles Mean-Reversion",               "step_04c_models_meanrev"),
        "4d":  ("Modeles Contrarian CLEAN",             "step_04d_models_contrarian_clean"),
        "6":   ("Backtest (ancien)",                    "step_06_backtest"),
        "6b":  ("Backtest (spread groupe)",             "step_06b_backtest_grouped"),
        "6c":  ("Backtest Mean-Reversion",              "step_06c_backtest_meanrev"),
        "6d":  ("Backtest Contrarian CLEAN",            "step_06d_backtest_contrarian_clean"),
        "7":   ("Tests econometriques",                 "step_07_econometrics"),
    }

    step_id = str(step_id)
    if step_id not in steps:
        print(f"Step '{step_id}' invalide. Disponibles : {list(steps.keys())}")
        return False

    name, module = steps[step_id]
    print(f"\n{'#' * 60}")
    print(f"# STEP {step_id} : {name}")
    print(f"{'#' * 60}\n")

    t0 = time.time()
    mod = __import__(module)
    mod.main()
    print(f"\n  Temps : {time.time() - t0:.1f}s")
    return True


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "all":
            t_total = time.time()
            for step in ["1", "2", "3", "3b", "3c", "4b", "4c", "4d", "6b", "6c", "6d", "7"]:
                if not run_step(step):
                    break
            print(f"\n  PIPELINE COMPLET en {time.time() - t_total:.1f}s")
        elif arg == "meanrev":
            t_total = time.time()
            for step in ["3b", "4c", "6c"]:
                if not run_step(step):
                    break
            print(f"\n  MEAN-REVERSION en {time.time() - t_total:.1f}s")
        elif arg == "clean":
            t_total = time.time()
            for step in ["3c", "4d", "6d"]:
                if not run_step(step):
                    break
            print(f"\n  CONTRARIAN CLEAN en {time.time() - t_total:.1f}s")
        else:
            run_step(arg)
    else:
        print("=" * 60)
        print("  PIPELINE GSR - EXECUTION PRINCIPALE")
        print("=" * 60)

        t_total = time.time()
        for step in ["1", "2", "3", "4b", "6b", "7"]:
            if not run_step(step):
                break

        print(f"\n{'=' * 60}")
        print(f"  PIPELINE TERMINE en {time.time() - t_total:.1f}s")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
