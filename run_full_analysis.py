# run_full_analysis.py
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.data_loader import load_processed_data
from src.descriptive_analysis import run_descriptive_analysis
from src.stationarity import run_all_stationarity_tests, get_integration_order, print_integration_summary, save_integration_summary
from src.ardl_model_parsimonious import run_ardl_analysis
from src.coeff_long_terme import run_long_term_analysis
from src.nardl_model import run_nardl_analysis
from src.diagnostics import run_all_diagnostics
from src.autocorrelation_justification import run_autocorrelation_justification
from src.multicolinearity import run_multicollinearity_check

def main():
    print("="*70)
    print("ÉTUDE ÉCONOMÉTRIQUE - INDUSTRIALISATION ET DÉVELOPPEMENT ÉCONOMIQUE AU MAROC")
    print("Période: 1990-2024 (35 observations)")
    print("="*70)
    
    # 1. Chargement
    print("\n[1/4] Chargement des données...")
    df_levels, df_log, df_diff = load_processed_data()
    
    # 2. Analyse descriptive
    print("\n[2/4] Analyse descriptive...")
    stats, corr = run_descriptive_analysis(df_levels, df_log)
    
    # 3. Tests de stationnarité (avec POP exclue)
    print("\n[3/4] Tests de racine unitaire...")
    level_c, level_ct, diff_res = run_all_stationarity_tests(df_log, df_diff)
    integration_order = get_integration_order(level_c, diff_res)
    print_integration_summary(integration_order)
    save_integration_summary(integration_order)
    
    # 4. ARDL
    print("\n[4/4] Estimation ARDL...")
    best_order, best_model = run_ardl_analysis(df_log)
    
    # 5. Coefficients de long terme
    if best_model is not None:
        print("\n[5/5] Calcul des coefficients de long terme...")
        p, q = best_order
        lt_coefs, lt_pvals = run_long_term_analysis(best_model, p, q)
        
    # 6. NARDL (asymétrie)
    print("\n[6/6] Test d'asymétrie (NARDL)...")
    nardl_model, asymmetry_results = run_nardl_analysis(df_log)
    
    # 7. Tests de diagnostic
    if best_model is not None:
        print("\n[7/7] Tests de diagnostic...")
        diagnostics_results = run_all_diagnostics(best_model, df_log)
        
    # 8. Justification de l'autocorrélation (corrigé: best_model au lieu de robust_model)
    if best_model is not None:
        print("\n[8/8] Génération de la justification d'autocorrélation...")
        run_autocorrelation_justification(best_model, df_log)
        
    # Multicolinéarité
    print("\n[8/8] Vérification de la multicolinéarité (VIF)...")
    vif_results, cond_number = run_multicollinearity_check(df_log)
    
    
    print("\n" + "="*70)
    print("Analyse terminée.")
    print("="*70)

if __name__ == "__main__":
    main()