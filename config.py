# config.py
# Paramètres globaux de l'étude

import pandas as pd
import numpy as np

# Période d'étude
START_YEAR = 1990
END_YEAR = 2024
N_OBS = END_YEAR - START_YEAR + 1  # 35 observations

# Noms exacts des colonnes dans ton CSV
RAW_COLUMNS = {
    "annee": "Année",
    "pibhab": "PIB_Réel_Par_Habitant_USD_PPA_internationnal_constant_2011",
    "iva": "V_A_Manufacturière_%_du_PIB",
    "pop": "Croissance_démographique_%_annuelle",
    "credit": "Crédit_Bancaire_au_secteur_privé_%_du_PIB",
    "dep": "Dépenses_Publiques_%_du_PIB",
    "ide": "Investissement_Direct_Etranger_entrée_nette_%_du_PIB",
    "tcer": "TCER_2010=100",
    "ipc": "Indice_des_Prix_à _la _consommation"
}

# Variables du modèle
DEPENDENT_VAR = "pibhab"
INDEPENDENT_VARS_ALL = ["iva", "pop", "credit", "dep", "ide", "tcer", "ipc"]
# RÉVISION: le jeu initial à 6 régresseurs produisait des VIF > 1000
# (cf. outputs/results/vif_results.txt de la version précédente) car ln_credit
# et ln_ipc sont quasi colinéaires avec ln_iva / la tendance générale (ln_ipc
# corrèle à 0.95 avec ln_pibhab et -0.93 avec ln_pop : effet de tendance commun,
# pas de contenu informatif propre). On les retire pour ne garder que les
# régresseurs dont le VIF reste < 2 une fois combinés (voir outputs/results/vif_results.txt) :
# VA manufacturière (variable d'intérêt), dépenses publiques, IDE, TCER.
INDEPENDENT_VARS_USED = ["iva", "dep", "ide", "tcer"]
INDEPENDENT_VARS_DROPPED = ["credit", "ipc"]
INDEPENDENT_VARS_DROPPED_REASON = (
    "ln_credit et ln_ipc retirés du modèle final: colinéarité sévère avec "
    "ln_iva et avec la tendance de long terme du PIB par habitant (VIF > 100 "
    "dans la spécification initiale à 6 régresseurs). Conservés uniquement "
    "dans l'analyse descriptive et la matrice de corrélation."
)

# Noms complets pour les graphiques et tableaux
VAR_LABELS = {
    "pibhab": "PIB par hab. (PPA 2011, USD)",
    "iva": "VA manufacturière (% PIB)",
    "pop": "Croissance démographique (%)",
    "credit": "Crédit privé (% PIB)",
    "dep": "Dépenses publiques (% PIB)",
    "ide": "IDE nets entrants (% PIB)",
    "tcer": "TCER (2010=100)",
    "ipc": "IPC (2010=100)"
}

# Transformation logarithmique (True/False)
LOG_VARS = {
    "pibhab": True,
    "iva": True,
    "pop": True,
    "credit": True,
    "dep": True,
    "ide": True,
    "tcer": True,
    "ipc": True
}

# Variables qui peuvent avoir des valeurs négatives (pour le décalage)
CAN_BE_NEGATIVE = ["ide"]  # IDE peut être négatif

# Modèle ARDL
# RÉVISION: MAX_LAGS=3 avec sélection AIC donnait ARDL(3,3) = 28 paramètres
# pour 32 observations (4 degrés de liberté résiduels), un R²=0.9997 qui
# signale un surparamétrage plutôt qu'un bon ajustement, et une autocorrélation
# résiduelle forte (LM=31.4, p<0.001, cf. diagnostics_summary.txt). L'AIC est
# connu pour sur-sélectionner les modèles complexes en petit échantillon (n=35);
# on utilise donc le BIC, qui pénalise davantage la complexité, et on limite le
# nombre max de retards pour garder au moins ~15 degrés de liberté résiduels.
MAX_LAGS = 2  # Pour données annuelles, n=35, réduit après diagnostic de surparamétrage
LAG_CRITERION = "bic"  # BIC préféré à AIC en petit échantillon (n=35)
MIN_RESIDUAL_DF = 10  # garde-fou explicite: rejette toute spécification qui descendrait sous ce seuil

# Seuils de significativité
ALPHA = 0.05

# Graphiques
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
FIGURES_DIR = "outputs/figures"
TABLES_DIR = "outputs/tables"
RESULTS_DIR = "outputs/results"

# Random seed
RANDOM_SEED = 42

# Chemins des fichiers
RAW_DATA_PATH = "data/raw/maroc_1990_2024.csv"
PROCESSED_DATA_PATH = "data/processed/maroc_processed.pkl"