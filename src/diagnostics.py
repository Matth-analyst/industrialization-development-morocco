# src/diagnostics.py
"""
Tests de diagnostic pour le modèle ARDL.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan, het_white
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
import config


def run_autocorrelation_test(model_results):
    """
    Test de Breusch-Godfrey pour l'autocorrélation des résidus.
    H0: pas d'autocorrélation
    """
    print("\n" + "="*80)
    print("1. TEST D'AUTOCORRÉLATION (Breusch-Godfrey)")
    print("="*80)
    
    try:
        bg_result = acorr_breusch_godfrey(model_results, nlags=2)
        
        lm_stat = bg_result[0]
        p_value = bg_result[1]
        
        print(f"Statistique LM: {lm_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("\n❌ Conclusion: Présence d'autocorrélation")
            autocorrelation = True
        else:
            print("\n✅ Conclusion: Pas d'autocorrélation significative")
            autocorrelation = False
        
        return {
            "lm_stat": lm_stat,
            "p_value": p_value,
            "autocorrelation": autocorrelation
        }
    
    except Exception as e:
        print(f"Erreur: {e}")
        return None


def run_heteroscedasticity_tests(model_results, X_design):
    """
    Tests de Breusch-Pagan et White pour l'hétéroscédasticité.
    H0: homoscédasticité
    """
    print("\n" + "="*80)
    print("2. TEST D'HÉTÉROSCÉDASTICITÉ")
    print("="*80)
    
    resid = model_results.resid
    
    # Test de Breusch-Pagan
    print("\n--- Test de Breusch-Pagan ---")
    try:
        bp_result = het_breuschpagan(resid, X_design)
        bp_lm = bp_result[0]
        bp_pvalue = bp_result[1]
        
        print(f"Statistique LM: {bp_lm:.4f}")
        print(f"P-value: {bp_pvalue:.4f}")
        
        if bp_pvalue < 0.05:
            print("❌ Hétéroscédasticité détectée")
            bp_hetero = True
        else:
            print("✅ Homoscédasticité")
            bp_hetero = False
    except Exception as e:
        print(f"Erreur: {e}")
        bp_lm = None
        bp_pvalue = None
        bp_hetero = None
    
    # Test de White
    print("\n--- Test de White ---")
    try:
        white_result = het_white(resid, X_design)
        white_lm = white_result[0]
        white_pvalue = white_result[1]
        
        print(f"Statistique LM: {white_lm:.4f}")
        print(f"P-value: {white_pvalue:.4f}")
        
        if white_pvalue < 0.05:
            print("❌ Hétéroscédasticité détectée")
            white_hetero = True
        else:
            print("✅ Homoscédasticité")
            white_hetero = False
    except Exception as e:
        print(f"Erreur: {e}")
        white_lm = None
        white_pvalue = None
        white_hetero = None
    
    return {
        "bp_lm": bp_lm,
        "bp_pvalue": bp_pvalue,
        "white_lm": white_lm,
        "white_pvalue": white_pvalue,
        "heteroscedasticity": (bp_hetero or white_hetero) if bp_hetero is not None else white_hetero
    }


def run_normality_test(model_results):
    """
    Test de Jarque-Bera pour la normalité des résidus.
    H0: résidus normalement distribués
    """
    print("\n" + "="*80)
    print("3. TEST DE NORMALITÉ (Jarque-Bera)")
    print("="*80)
    
    resid = model_results.resid
    
    n = len(resid)
    skewness = stats.skew(resid)
    kurtosis = stats.kurtosis(resid, fisher=False)
    
    jb_stat = (n / 6) * (skewness**2 + (kurtosis - 3)**2 / 4)
    p_value = 1 - stats.chi2.cdf(jb_stat, df=2)
    
    print(f"Skewness: {skewness:.4f}")
    print(f"Kurtosis: {kurtosis:.4f}")
    print(f"Statistique JB: {jb_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("\n❌ Résidus non normaux")
        normality = False
    else:
        print("\n✅ Résidus normaux")
        normality = True
    
    return {"jb_stat": jb_stat, "p_value": p_value, "normality": normality}


def run_reset_test(model_results, y_true, X_design):
    """
    Test de Ramsey RESET pour la spécification fonctionnelle.
    H0: modèle correctement spécifié
    """
    print("\n" + "="*80)
    print("4. TEST DE RAMSEY RESET")
    print("="*80)
    
    try:
        y_pred = model_results.fittedvalues
        X_augmented = np.column_stack([X_design, y_pred**2, y_pred**3])
        
        model_augmented = sm.OLS(y_true, X_augmented).fit()
        
        rss_r = model_results.ssr
        rss_ur = model_augmented.ssr
        df_r = model_results.df_resid
        df_ur = model_augmented.df_resid
        df_diff = df_r - df_ur
        
        f_stat = ((rss_r - rss_ur) / df_diff) / (rss_ur / df_ur)
        p_value = 1 - stats.f.cdf(f_stat, df_diff, df_ur)
        
        print(f"Statistique F: {f_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("\n❌ Erreur de spécification")
            reset_valid = False
        else:
            print("\n✅ Modèle bien spécifié")
            reset_valid = True
        
        return {"f_stat": f_stat, "p_value": p_value, "reset_valid": reset_valid}
    
    except Exception as e:
        print(f"Erreur: {e}")
        return None


def plot_cusum_tests(model_results, output_path):
    """
    Tests de stabilité CUSUM et CUSUMSQ.
    """
    print("\n" + "="*80)
    print("5. TESTS DE STABILITÉ (CUSUM et CUSUMSQ)")
    print("="*80)
    
    resid = model_results.resid
    n = len(resid)
    
    cumulative_sum = np.cumsum(resid) / np.std(resid)
    cusum_upper = 1.36 * np.sqrt(np.arange(1, n + 1))
    cusum_lower = -1.36 * np.sqrt(np.arange(1, n + 1))
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # CUSUM
    axes[0].plot(cumulative_sum, linewidth=1.5, label='CUSUM')
    axes[0].plot(cusum_upper, 'r--', linewidth=1, label='Borne sup (5%)')
    axes[0].plot(cusum_lower, 'r--', linewidth=1, label='Borne inf (5%)')
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0].set_title('Test CUSUM (stabilité des paramètres)')
    axes[0].set_xlabel('Observation')
    axes[0].set_ylabel('CUSUM')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    cusum_stable = np.all(cumulative_sum <= cusum_upper) and np.all(cumulative_sum >= cusum_lower)
    
    # CUSUMSQ
    cumulative_sum_sq = np.cumsum(resid**2) / np.sum(resid**2)
    cusumsq_upper = 1.36 * np.sqrt(np.arange(1, n + 1)) / np.sqrt(n)
    cusumsq_lower = -1.36 * np.sqrt(np.arange(1, n + 1)) / np.sqrt(n)
    
    axes[1].plot(cumulative_sum_sq, linewidth=1.5, label='CUSUMSQ')
    axes[1].plot(cusumsq_upper, 'r--', linewidth=1, label='Borne sup (5%)')
    axes[1].plot(cusumsq_lower, 'r--', linewidth=1, label='Borne inf (5%)')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_title('Test CUSUMSQ (stabilité de la variance)')
    axes[1].set_xlabel('Observation')
    axes[1].set_ylabel('CUSUMSQ')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    cusumsq_stable = np.all(cumulative_sum_sq <= cusumsq_upper) and np.all(cumulative_sum_sq >= cusumsq_lower)
    
    plt.tight_layout()
    
    fig_path = Path(output_path) / "cusum_tests.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Graphiques sauvegardés: {fig_path}")
    plt.close()
    
    print(f"\nCUSUM stable: {cusum_stable}")
    print(f"CUSUMSQ stable: {cusumsq_stable}")
    
    return {"cusum_stable": cusum_stable, "cusumsq_stable": cusumsq_stable}


def run_all_diagnostics(best_model, df_log):
    """
    Lance tous les tests de diagnostic.
    """
    print("\n" + "="*80)
    print("TESTS DE DIAGNOSTIC DU MODÈLE ARDL")
    print("="*80)
    
    results = {}
    
    # 1. Autocorrélation
    results["autocorrelation"] = run_autocorrelation_test(best_model)
    
    # 2. Hétéroscédasticité
    X_design = best_model.model.exog
    results["heteroscedasticity"] = run_heteroscedasticity_tests(best_model, X_design)
    
    # 3. Normalité
    results["normality"] = run_normality_test(best_model)
    
    # 4. Ramsey RESET
    y_true = best_model.model.endog
    results["reset"] = run_reset_test(best_model, y_true, X_design)
    
    # 5. CUSUM et CUSUMSQ
    results["cusum"] = plot_cusum_tests(best_model, config.RESULTS_DIR)
    
    # Résumé final
    print("\n" + "="*80)
    print("RÉSUMÉ DES TESTS DE DIAGNOSTIC")
    print("="*80)
    
    print("\n1. Autocorrélation (Breusch-Godfrey):")
    if results["autocorrelation"]:
        print(f"   → LM = {results['autocorrelation']['lm_stat']:.4f}, p = {results['autocorrelation']['p_value']:.4f}")
        print(f"   → {'Présence' if results['autocorrelation']['autocorrelation'] else 'Absence'} d'autocorrélation")
    
    print("\n2. Hétéroscédasticité:")
    if results["heteroscedasticity"]:
        bp_lm = results["heteroscedasticity"].get('bp_lm')
        bp_pval = results["heteroscedasticity"].get('bp_pvalue')
        white_lm = results["heteroscedasticity"].get('white_lm')
        white_pval = results["heteroscedasticity"].get('white_pvalue')
        
        if bp_lm is not None:
            print(f"   → Breusch-Pagan: LM = {bp_lm:.4f}, p = {bp_pval:.4f}")
        if white_lm is not None:
            print(f"   → White: LM = {white_lm:.4f}, p = {white_pval:.4f}")
    
    print("\n3. Normalité (Jarque-Bera):")
    if results["normality"]:
        print(f"   → JB = {results['normality']['jb_stat']:.4f}, p = {results['normality']['p_value']:.4f}")
        print(f"   → Résidus {'normaux' if results['normality']['normality'] else 'non normaux'}")
    
    print("\n4. Spécification (Ramsey RESET):")
    if results["reset"]:
        print(f"   → F = {results['reset']['f_stat']:.4f}, p = {results['reset']['p_value']:.4f}")
        print(f"   → Modèle {'bien spécifié' if results['reset']['reset_valid'] else 'mal spécifié'}")
    
    print("\n5. Stabilité (CUSUM/CUSUMSQ):")
    if results["cusum"]:
        print(f"   → CUSUM: {'stable' if results['cusum']['cusum_stable'] else 'instable'}")
        print(f"   → CUSUMSQ: {'stable' if results['cusum']['cusumsq_stable'] else 'instable'}")
    
    # Sauvegarde
    output_path = Path(config.RESULTS_DIR) / "diagnostics_summary.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RÉSUMÉ DES TESTS DE DIAGNOSTIC\n")
        f.write("="*80 + "\n\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\n✅ Résumé sauvegardé: {output_path}")
    
    return results


if __name__ == "__main__":
    from data_loader import load_processed_data
    from ardl_model_parsimonious import run_ardl_analysis
    
    levels, log, diff = load_processed_data()
    best_order, best_model = run_ardl_analysis(log)
    
    if best_model is not None:
        run_all_diagnostics(best_model, log)