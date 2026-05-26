# src/data_loader.py
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import config
from config import START_YEAR, END_YEAR, RAW_COLUMNS, LOG_VARS, CAN_BE_NEGATIVE

def load_raw_data() -> pd.DataFrame:
    """
    Charge les données brutes depuis le CSV.
    """
    df = pd.read_csv(config.RAW_DATA_PATH)
    
    # Renommer les colonnes avec des noms simples
    rename_dict = {v: k for k, v in RAW_COLUMNS.items()}
    df = df.rename(columns=rename_dict)
    
    # Convertir l'année en index datetime
    df["annee"] = pd.to_datetime(df["annee"].astype(str), format="%Y")
    df = df.set_index("annee")
    
    # Filtrer selon période configurée
    df = df.loc[df.index.year >= START_YEAR]
    df = df.loc[df.index.year <= END_YEAR]
    
    return df

def create_log_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique la transformation logarithmique.
    Gère les valeurs négatives avec un décalage.
    """
    df_log = df.copy()
    
    for var in LOG_VARS.keys():
        if var in df.columns:
            series = df[var]
            
            # Vérifier les valeurs non positives
            if (series <= 0).any() and var in CAN_BE_NEGATIVE:
                # Pour IDE, on peut avoir des valeurs négatives
                # On utilise transformation signe: log(1 + x) pour x négatif ?
                # Solution simple: shift
                min_val = series.min()
                if min_val <= 0:
                    shift = abs(min_val) + 1
                    df_log[f"ln_{var}"] = np.log(series + shift)
                    print(f"ATTENTION: Décalage de {shift} appliqué à {var}")
                else:
                    df_log[f"ln_{var}"] = np.log(series)
            elif (series <= 0).any():
                # Variables qui ne devraient pas être négatives
                shift = abs(series.min()) + 1
                df_log[f"ln_{var}"] = np.log(series + shift)
                print(f"ATTENTION: Décalage de {shift} appliqué à {var} (valeurs non positives détectées)")
            else:
                df_log[f"ln_{var}"] = np.log(series)
        else:
            print(f"ATTENTION: Variable {var} non trouvée dans les données")
    
    return df_log

def create_diff_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée les différences premières des variables en log.
    """
    df_diff = df.copy()
    
    ln_cols = [col for col in df.columns if col.startswith("ln_")]
    for col in ln_cols:
        df_diff[f"d_{col}"] = df[col].diff()
    
    return df_diff

def load_processed_data():
    """
    Fonction principale : charge, nettoie et transforme les données.
    Returns:
        (df_levels, df_log, df_diff)
    """
    raw = load_raw_data()
    
    # Variables brutes sélectionnées
    all_vars = [config.DEPENDENT_VAR] + config.INDEPENDENT_VARS_ALL
    df_levels = raw[all_vars].copy()
    
    # Supprimer les lignes avec des NaN
    initial_len = len(df_levels)
    df_levels = df_levels.dropna()
    if len(df_levels) < initial_len:
        print(f"ATTENTION: {initial_len - len(df_levels)} lignes supprimées (valeurs manquantes)")
    
    # Logs
    df_log = create_log_variables(df_levels)
    
    # Différences
    df_diff = create_diff_variables(df_log)
    
    # Sauvegarde
    processed_path = Path(config.PROCESSED_DATA_PATH)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(processed_path, "wb") as f:
        pd.to_pickle({
            "levels": df_levels,
            "log": df_log,
            "diff": df_diff
        }, f)
    
    print(f"Données sauvegardées dans {processed_path}")
    print(f"Période: {df_levels.index.year.min()} - {df_levels.index.year.max()}")
    print(f"Nombre d'observations: {len(df_levels)}")
    
    return df_levels, df_log, df_diff

if __name__ == "__main__":
    # Test
    levels, log, diff = load_processed_data()
    print("\n=== Aperçu des données transformées ===")
    print(log.head())