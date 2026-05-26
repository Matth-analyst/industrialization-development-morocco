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
INDEPENDENT_VARS_USED = ["iva", "credit", "dep", "ide", "tcer", "ipc"]

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
MAX_LAGS = 3  # Pour données annuelles
LAG_CRITERION = "aic"  # ou "bic"

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