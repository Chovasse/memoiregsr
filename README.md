# memoiregsr
L'ensemble du code Python (pipeline de features, modèles de Machine Learning, backtesting et génération des graphiques) ainsi que les données utilisées sont disponibles sur ce repository GitHub


Pipeline GSR - Mémoire de Recherche Appliquée
Le ratio Gold/Silver comme indicateur prédictif des régimes de marché : une approche Machine Learning
Clément Hovasse - M2 Finance & Marchés, ESDES Business School (2025-2026)
Directeur de mémoire : Sami Ben Jabeur
Description
Ce repository contient le code source du pipeline de Machine Learning développé dans le cadre du mémoire de recherche appliquée. Le pipeline couvre l'ensemble de la chaîne de traitement, depuis le chargement des données brutes jusqu'au backtesting des stratégies de trading.
Structure
pipeline/
    config.py                       Configuration (chemins, paramètres, mapping FactSet)
    main.py                         Lanceur du pipeline complet
    requirements.txt                Dépendances Python
    step_01_load_data.py            Chargement et nettoyage des données FactSet
    step_02_features.py             Feature engineering (33 features, 5 familles)
    step_03_labeling.py             Etiquetage triple-barrier
    step_03d_labeling_halflife.py   Etiquetage calibré sur le half-life glissant
    step_04_models.py               Modèles ML (toutes features)
    step_04b_grouped_models.py      Modèles ML (features groupées)
    step_04d_walkforward_halflife.py  Walk-forward avec horizon half-life
    step_04e_walkforward_monthly.py   Walk-forward avec réentraînement mensuel
    step_05_shap.py                 Analyse SHAP (interprétabilité)
    step_06_backtest.py             Backtest initial
    step_06b_backtest_grouped.py    Backtest avec features groupées
    step_06e_backtest_halflife.py   Backtest horizon half-life
    step_06f_backtest_monthly.py    Backtest réentraînement mensuel
    step_07_econometrics.py         Tests économétriques (ADF, KPSS, Johansen, half-life)
    step_07_halflife_adapted_backtest.py  Backtest adaptatif calibré sur half-life
    generate_charts_la_francaise.py Génération des figures (charte La Française)
generate_charts.py                  Graphiques de la Partie 1 (revue de littérature)
Données
Les données proviennent de la base FactSet (terminal professionnel) et couvrent la période janvier 2000 - mai 2026. L'échantillon comprend 16 séries quotidiennes et 2 séries mensuelles :
Métaux précieux : GC1, SI1, XAUUSD, XAGUSD, GLD, SLV
Macroéconomie : UST 10Y, TIPS 10Y, Breakeven 10Y, Fed Funds, DXY
Risque / Actions : VIX, S&P 500, MSCI World
Commodities cycliques : HG1 (cuivre), CL1 (pétrole WTI)
Séries mensuelles (interpolées) : M2 (masse monétaire), CPI (inflation)
Les données brutes FactSet ne sont pas redistribuables (licence professionnelle). Le fichier data/processed/master_daily.csv contient le dataset nettoyé et agrégé.
Installation et exécution
bashpip install -r requirements.txt
python main.py        # Pipeline complet
python main.py 7      # Step spécifique (ex: tests économétriques)
Modèles utilisés
XGBoost, Random Forest, SVM (RBF), Régression Logistique, Ridge Classifier
Validation : Walk-forward purged cross-validation (30 folds)
Résultats principaux
Les modèles ML ne surperforment pas le hasard pour la prédiction directionnelle du GSR (meilleur accuracy : 52.5%). L'explication réside dans le mismatch entre le half-life de mean-reversion (264 jours) et les horizons de trading testés (21-63 jours). Le GSR conserve une valeur diagnostique comme indicateur de sentiment de marché.
