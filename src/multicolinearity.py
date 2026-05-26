# src/multicollinearity.py
"""
Détection de la multicolinéarité par calcul des VIF (Variance Inflation Factor).
"""

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
import config


def calculate_vif(df_log):
    """
    Calcule les VIF pour toutes les variables explicatives du modèle.
    
    Règle d'interprétation:
    - VIF = 1 : pas de corrélation
    - 1 < VIF < 5 : corrélation modérée (acceptable)
    - 5 ≤ VIF < 10 : corrélation forte (à surveiller)
    - VIF ≥ 10 : multicolinéarité sévère (nécessite une correction)
    
    Returns:
        DataFrame avec les VIF et leur interprétation
    """
    print("\n" + "="*80)
    print("DÉTECTION DE LA MULTICOLINÉARITÉ (VIF)")
    print("="*80)
    
    # Sélectionner les variables explicatives du modèle (en log)
    X_cols = [f"ln_{var}" for var in config.INDEPENDENT_VARS_USED if f"ln_{var}" in df_log.columns]
    
    if len(X_cols) == 0:
        print("Aucune variable explicative trouvée")
        return None
    
    # Extraire les données et supprimer les NaN
    X = df_log[X_cols].dropna()
    
    print(f"Variables analysées: {X_cols}")
    print(f"Nombre d'observations: {len(X)}")
    print(f"Nombre de variables: {len(X_cols)}")
    print("-"*80)
    
    # Calculer les VIF
    vif_data = []
    for i, col in enumerate(X_cols):
        vif = variance_inflation_factor(X.values, i)
        
        # Interprétation
        if vif < 5:
            interpretation = "✅ Corrélation modérée (acceptable)"
        elif vif < 10:
            interpretation = "⚠️ Corrélation forte (à surveiller)"
        else:
            interpretation = "❌ Multicolinéarité sévère (nécessite correction)"
        
        vif_data.append({
            "Variable": col,
            "VIF": round(vif, 2),
            "Interprétation": interpretation
        })
    
    vif_df = pd.DataFrame(vif_data)
    
    # Affichage
    print("\nRÉSULTATS DES VIF:")
    print("-"*80)
    for _, row in vif_df.iterrows():
        print(f"{row['Variable']:20} VIF = {row['VIF']:8.2f}  {row['Interprétation']}")
    
    # Vérification des VIF élevés
    high_vif = vif_df[vif_df["VIF"] >= 10]
    if len(high_vif) > 0:
        print("\n" + "-"*80)
        print("⚠️ ATTENTION: Variables avec VIF ≥ 10 détectées:")
        for _, row in high_vif.iterrows():
            print(f"   - {row['Variable']}: VIF = {row['VIF']:.2f}")
        print("\nSolutions possibles:")
        print("   1. Supprimer la variable la plus corrélée")
        print("   2. Combiner les variables (moyenne, ACP)")
        print("   3. Standardiser les variables")
        print("   4. Accepter et mentionner la limite dans le rapport")
    else:
        print("\n" + "-"*80)
        print("✅ Aucune multicolinéarité sévère détectée (tous les VIF < 10)")
    
    # Sauvegarde
    output_path = Path(config.RESULTS_DIR) / "vif_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vif_df.to_csv(output_path, index=False)
    print(f"\nRésultats sauvegardés dans: {output_path}")
    
    # Sauvegarde texte pour le rapport
    txt_path = Path(config.RESULTS_DIR) / "vif_results.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("DÉTECTION DE LA MULTICOLINÉARITÉ (VIF)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Variables analysées: {X_cols}\n")
        f.write(f"Nombre d'observations: {len(X)}\n")
        f.write(f"Nombre de variables: {len(X_cols)}\n\n")
        f.write("-"*80 + "\n")
        for _, row in vif_df.iterrows():
            f.write(f"{row['Variable']:20} VIF = {row['VIF']:8.2f}  {row['Interprétation']}\n")
        f.write("-"*80 + "\n\n")
        
        if len(high_vif) > 0:
            f.write("⚠️ Variables avec VIF ≥ 10:\n")
            for _, row in high_vif.iterrows():
                f.write(f"   - {row['Variable']}: VIF = {row['VIF']:.2f}\n")
        else:
            f.write("✅ Aucune multicolinéarité sévère.\n")
    
    print(f"Rapport texte sauvegardé dans: {txt_path}")
    
    return vif_df


def correlation_matrix_with_target(df_log):
    """
    Affiche la matrice de corrélation avec la variable dépendante.
    """
    print("\n" + "="*80)
    print("MATRICE DE CORRÉLATION AVEC LA VARIABLE DÉPENDANTE")
    print("="*80)
    
    # Variables à inclure
    y_col = f"ln_{config.DEPENDENT_VAR}"
    X_cols = [f"ln_{var}" for var in config.INDEPENDENT_VARS_USED if f"ln_{var}" in df_log.columns]
    
    all_cols = [y_col] + X_cols
    corr_matrix = df_log[all_cols].corr()
    
    # Afficher les corrélations avec la variable dépendante
    print(f"\nCorrélations avec {y_col}:")
    print("-"*50)
    corr_with_y = corr_matrix[y_col].drop(y_col).sort_values(ascending=False)
    for var, corr in corr_with_y.items():
        if abs(corr) > 0.7:
            print(f"  {var:20} {corr:8.3f}  (⚠️ forte corrélation)")
        else:
            print(f"  {var:20} {corr:8.3f}")
    
    return corr_matrix


def check_condition_number(df_log):
    """
    Calcule le condition number de la matrice des variables explicatives.
    Un condition number > 30 indique une multicolinéarité potentiellement problématique.
    """
    print("\n" + "="*80)
    print("CONDITION NUMBER")
    print("="*80)
    
    # Sélectionner les variables explicatives
    X_cols = [f"ln_{var}" for var in config.INDEPENDENT_VARS_USED if f"ln_{var}" in df_log.columns]
    X = df_log[X_cols].dropna()
    
    # Standardiser les variables (recommandé pour le condition number)
    X_scaled = (X - X.mean()) / X.std()
    
    # Calculer le condition number (ratio de la plus grande à la plus petite valeur singulière)
    from numpy.linalg import svd
    _, s, _ = svd(X_scaled.values)
    condition_number = s[0] / s[-1]
    
    print(f"Condition number: {condition_number:.2f}")
    
    if condition_number < 30:
        print("✅ Condition number faible → multicolinéarité modérée")
    elif condition_number < 100:
        print("⚠️ Condition number modéré → multicolinéarité potentielle")
    else:
        print("❌ Condition number élevé → multicolinéarité sévère")
    
    return condition_number


def run_multicollinearity_check(df_log):
    """
    Lance l'analyse complète de multicolinéarité.
    """
    print("\n" + "="*80)
    print("ANALYSE DE LA MULTICOLINÉARITÉ")
    print("="*80)
    
    # 1. Matrice de corrélation avec la variable dépendante
    corr_matrix = correlation_matrix_with_target(df_log)
    
    # 2. Calcul des VIF
    vif_df = calculate_vif(df_log)
    
    # 3. Condition number
    cond_number = check_condition_number(df_log)
    
    # 4. Recommandation finale
    print("\n" + "="*80)
    print("RECOMMANDATION")
    print("="*80)
    
    if vif_df is not None:
        max_vif = vif_df["VIF"].max()
        if max_vif >= 10:
            print(f"⚠️ VIF maximum = {max_vif:.2f} (≥ 10) → multicolinéarité sévère.")
            print("   Recommandation: supprimer la variable la plus corrélée ou utiliser une ACP.")
        elif max_vif >= 5:
            print(f"⚠️ VIF maximum = {max_vif:.2f} (5 ≤ VIF < 10) → multicolinéarité forte.")
            print("   Recommandation: à surveiller, mais acceptable pour une étude exploratoire.")
        else:
            print(f"✅ VIF maximum = {max_vif:.2f} (< 5) → pas de problème de multicolinéarité majeur.")
    
    return vif_df, cond_number


if __name__ == "__main__":
    from data_loader import load_processed_data
    levels, log, diff = load_processed_data()
    run_multicollinearity_check(log)