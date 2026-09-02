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
    if abs(denominator) < 0.05:
        print(f"\n⚠️⚠️ Dénominateur quasi nul ({denominator:.4f}): la somme des coefficients "
              "des retards de y est très proche de 1 (racine quasi unitaire dans la dynamique "
              "de court terme). Les coefficients de long terme obtenus par division sont donc "
              "numériquement instables (ils explosent) et NE DOIVENT PAS être interprétés comme "
              "des élasticités de long terme fiables. Ceci est cohérent avec un test des bornes "
              "ne rejetant pas l'absence de cointégration (cf. outputs/results/bounds_test.txt): "
              "la notion même de \"relation de long terme\" n'est pas solidement établie ici.")
    
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
        if abs(denominator) < 0.05:
            f.write(
                "⚠️⚠️ AVERTISSEMENT: dénominateur quasi nul → la somme des coefficients des "
                "retards de y est très proche de 1. Les coefficients de long terme ci-dessous "
                "sont numériquement instables et ne doivent pas être interprétés comme des "
                "élasticités de long terme fiables. Voir outputs/results/bounds_test.txt: le "
                "test des bornes ne rejette pas H0 (pas de relation de cointégration établie), "
                "ce qui est cohérent avec cette instabilité.\n\n"
            )
        f.write(results_df.to_string(index=False))
        f.write("\n\n" + "="*80 + "\n")
        f.write("Significativité: *** p<0.01, ** p<0.05, * p<0.10\n")

        # Discussion explicite des signes économiquement contestables, plutôt que de
        # laisser un coefficient contre-intuitif sans commentaire dans les résultats.
        EXPECTED_SIGNS = {
            "ln_iva": ("+", "l'industrialisation manufacturière est théoriquement associée "
                             "positivement au PIB par habitant (littérature sur la transformation "
                             "structurelle, ex. Rodrik 2016)"),
            "ln_dep": (None, "le signe attendu des dépenses publiques est ambigu a priori "
                              "(effet d'éviction possible vs. investissement public productif)"),
            "ln_ide": ("+", "l'IDE est généralement attendu positif via transfert de capital, "
                             "technologie et emploi"),
            "ln_tcer": (None, "le signe du taux de change réel dépend de la structure "
                               "exportatrice/importatrice du pays"),
        }
        flagged = []
        for var, coef in lt_coefficients.items():
            expected, note = EXPECTED_SIGNS.get(var, (None, ""))
            actual_sign = "+" if coef > 0 else "-"
            if expected is not None and actual_sign != expected:
                flagged.append((var, coef, note))

        if flagged:
            f.write("\n" + "-"*80 + "\n")
            f.write("SIGNES ÉCONOMIQUEMENT CONTRE-INTUITIFS À DISCUTER DANS LE RAPPORT\n")
            f.write("-"*80 + "\n")
            for var, coef, note in flagged:
                label = config.VAR_LABELS.get(var.replace("ln_", ""), var)
                f.write(f"- {label}: coefficient de long terme = {coef:.4f} "
                        f"(signe {'positif' if coef > 0 else 'négatif'} inattendu). {note}.\n"
                        f"  Ne pas laisser ce résultat sans commentaire: discuter les explications "
                        f"possibles (multicolinéarité résiduelle, instabilité d'échantillon, "
                        f"effet de composition, causalité inverse) plutôt que de l'ignorer.\n")
    
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