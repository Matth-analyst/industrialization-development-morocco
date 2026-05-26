# src/coeff_long_term.py
"""
Calcul des coefficients de long terme à partir du modèle ARDL-ECM.
Formule: β_LT = (Σ coefficients de la variable) / (1 - Σ coefficients des retards de y)
"""

import pandas as pd
import numpy as np
from scipy import stats  # ← AJOUT OBLIGATOIRE
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
import config

def calculate_long_term_coefficients(model_results, p, q, X_cols):
    """
    Calcule les coefficients de long terme à partir du modèle ARDL estimé.
    
    Parameters:
    - model_results: objet results de statsmodels
    - p: nombre de retards de la variable dépendante (y)
    - q: nombre de retards des variables exogènes
    - X_cols: liste des noms des variables exogènes (sans suffixe _L)
    
    Returns:
    - dict: coefficients de long terme
    """
    params = model_results.params
    
    # 1. Calculer la somme des coefficients des retards de y (p)
    y_lag_cols = [f"y_L{i}" for i in range(1, p + 1)]
    sum_y_lags = sum([params.get(col, 0) for col in y_lag_cols])
    
    # Dénominateur pour les coefficients de long terme
    denominator = 1 - sum_y_lags
    
    print("="*80)
    print("CALCUL DES COEFFICIENTS DE LONG TERME")
    print("="*80)
    print(f"Somme des coefficients des retards de y (p={p}): {sum_y_lags:.6f}")
    print(f"Dénominateur (1 - Σy_lags): {denominator:.6f}")
    
    if denominator <= 0:
        print("\n⚠️ Attention: Dénominateur négatif ou nul. Vérifiez la stabilité du modèle.")
    
    # 2. Calculer les coefficients de long terme pour chaque variable exogène
    lt_coefficients = {}
    lt_std_errors = {}
    lt_pvalues = {}
    
    for var in X_cols:
        # Récupérer tous les coefficients de cette variable (contemporain + retards)
        var_cols = [var] + [f"{var}_L{i}" for i in range(1, q + 1)]
        sum_var_coefs = sum([params.get(col, 0) for col in var_cols if col in params])
        
        # Coefficient de long terme
        lt_coef = sum_var_coefs / denominator
        lt_coefficients[var] = lt_coef
        
        # Calcul de l'erreur standard approximative (Delta method simplifié)
        var_cols_present = [col for col in var_cols if col in params]
        var_se = np.sqrt(sum([model_results.bse.get(col, 0)**2 for col in var_cols_present]))
        
        # Approximation grossière de l'erreur standard du coefficient LT
        lt_se = var_se / abs(denominator) if denominator != 0 else np.nan
        lt_std_errors[var] = lt_se
        
        # t-statistique approximative
        if lt_se > 0 and not np.isnan(lt_se):
            t_stat = lt_coef / lt_se
            lt_pvalues[var] = 2 * (1 - stats.t.cdf(abs(t_stat), df=model_results.df_resid))
        else:
            lt_pvalues[var] = np.nan
    
    return lt_coefficients, lt_std_errors, lt_pvalues, denominator

def print_long_term_results(lt_coefficients, lt_pvalues, denominator):
    """
    Affiche les coefficients de long terme de manière lisible.
    """
    print("\n" + "="*80)
    print("COEFFICIENTS DE LONG TERME")
    print("="*80)
    print(f"Formule: β_LT = (Σ coefficients CT de la variable) / (1 - Σ coefficients des retards de y)")
    print(f"Dénominateur: {denominator:.6f}")
    print("\nVariable" + " "*(25) + "Coef LT" + " "*(10) + "p-value" + " "*(10) + "Significativité")
    print("-"*80)
    
    for var, coef in lt_coefficients.items():
        pval = lt_pvalues.get(var, np.nan)
        
        if pval < 0.01:
            sig = "***"
        elif pval < 0.05:
            sig = "**"
        elif pval < 0.10:
            sig = "*"
        else:
            sig = ""
        
        var_label = config.VAR_LABELS.get(var.replace("ln_", ""), var)
        # Tronquer si trop long
        if len(var_label) > 30:
            var_label = var_label[:27] + "..."
        
        pval_display = f"{pval:.4f}" if not np.isnan(pval) else "nan"
        print(f"{var_label:30} {coef:10.6f}     {pval_display:8}       {sig}")

def save_long_term_results(lt_coefficients, lt_pvalues, denominator, output_path):
    """
    Sauvegarde les coefficients de long terme dans un fichier.
    """
    results_df = pd.DataFrame({
        "variable": list(lt_coefficients.keys()),
        "var_label": [config.VAR_LABELS.get(v.replace("ln_", ""), v) for v in lt_coefficients.keys()],
        "long_term_coefficient": list(lt_coefficients.values()),
        "p_value": list(lt_pvalues.values())
    })
    
    results_df["significant_5pct"] = results_df["p_value"] < 0.05
    results_df = results_df.round(6)
    
    # Sauvegarde CSV
    results_df.to_csv(output_path, index=False)
    
    # Sauvegarde TXT
    txt_path = output_path.with_suffix('.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("COEFFICIENTS DE LONG TERME\n")
        f.write("="*80 + "\n\n")
        f.write(f"Dénominateur (1 - Σ retards de y): {denominator:.6f}\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n" + "="*80 + "\n")
        f.write("Significativité: *** p<0.01, ** p<0.05, * p<0.10\n")
    
    print(f"\nRésultats sauvegardés dans: {output_path}")
    print(f"Version texte dans: {txt_path}")

def run_long_term_analysis(best_model, p, q):
    """
    Lance l'analyse des coefficients de long terme.
    """
    # Récupérer les noms des variables exogènes
    X_cols = [f"ln_{var}" for var in config.INDEPENDENT_VARS_USED]
    
    # Calculer les coefficients LT
    lt_coefs, lt_se, lt_pvals, denominator = calculate_long_term_coefficients(
        best_model, p, q, X_cols
    )
    
    # Afficher les résultats
    print_long_term_results(lt_coefs, lt_pvals, denominator)
    
    # Sauvegarder
    output_path = Path(config.RESULTS_DIR) / "long_term_coefficients.csv"
    save_long_term_results(lt_coefs, lt_pvals, denominator, output_path)
    
    return lt_coefs, lt_pvals

if __name__ == "__main__":
    # Test
    print("Module des coefficients de long terme chargé.")