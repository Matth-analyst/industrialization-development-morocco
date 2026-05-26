# src/nardl_model.py
"""
Modèle NARDL (Nonlinear ARDL) pour tester l'asymétrie.
Décompose la variable IVA en composantes positive et négative.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
import config

def decompose_positive_negative(series):
    """
    Décompose une série en composantes positive et négative (accumulées).
    
    Returns:
    - pos: accumulation des variations positives
    - neg: accumulation des variations négatives (en valeur absolue)
    """
    diff = series.diff()
    
    pos_diff = diff.where(diff > 0, 0)
    neg_diff = diff.where(diff < 0, 0)
    
    pos = pos_diff.cumsum().fillna(0)
    neg = neg_diff.cumsum().fillna(0)
    
    return pos, neg

def prepare_nardl_data(df_log):
    """
    Prépare les données pour le modèle NARDL.
    """
    # Variable dépendante
    y = df_log["ln_pibhab"].dropna()
    
    # Variables de contrôle (sans IVA qui sera décomposée)
    X_cols = [f"ln_{var}" for var in config.INDEPENDENT_VARS_USED if var != "iva"]
    
    # Décomposer IVA en positif et négatif
    iva_series = df_log["ln_iva"].dropna()
    iva_pos, iva_neg = decompose_positive_negative(iva_series)
    
    # Aligner toutes les séries
    data = pd.DataFrame({
        "y": y,
        "iva_pos": iva_pos,
        "iva_neg": iva_neg
    })
    
    for col in X_cols:
        if col in df_log.columns:
            data[col] = df_log[col]
    
    data = data.dropna()
    
    return data, X_cols

def estimate_nardl(df_log, max_lags=2):
    """
    Estime le modèle NARDL (asymétrique).
    """
    print("\n" + "="*80)
    print("TEST D'ASYMÉTRIE (NARDL)")
    print("="*80)
    
    # Préparer les données
    data, X_cols = prepare_nardl_data(df_log)
    y = data["y"]
    
    print(f"Observations: {len(data)}")
    print(f"Variables: iva_pos, iva_neg, {', '.join(X_cols)}")
    print("-"*80)
    
    best_aic = np.inf
    best_model = None
    best_order = None
    results_list = []
    
    # Tester différentes combinaisons de retards
    for p in range(1, max_lags + 1):
        for q in range(0, max_lags + 1):
            try:
                # Créer les retards de y
                y_lags = pd.DataFrame()
                for lag in range(1, p + 1):
                    y_lags[f"y_L{lag}"] = y.shift(lag)
                
                # Créer les retards pour iva_pos, iva_neg et les contrôles
                X_lagged = pd.DataFrame()
                
                # Retards de iva_pos et iva_neg
                for lag in range(1, q + 1):
                    X_lagged[f"iva_pos_L{lag}"] = data["iva_pos"].shift(lag)
                    X_lagged[f"iva_neg_L{lag}"] = data["iva_neg"].shift(lag)
                
                # Variables contemporaines et retards des contrôles
                for col in X_cols:
                    X_lagged[col] = data[col]  # contemporain
                    for lag in range(1, q + 1):
                        X_lagged[f"{col}_L{lag}"] = data[col].shift(lag)
                
                # Variables contemporaines de iva_pos et iva_neg
                X_lagged["iva_pos"] = data["iva_pos"]
                X_lagged["iva_neg"] = data["iva_neg"]
                
                # Combiner
                all_vars = pd.concat([y_lags, X_lagged], axis=1).dropna()
                y_curr = y.loc[all_vars.index]
                
                # Vérifier le nombre de paramètres
                n_params = 1 + len(all_vars.columns)  # +1 pour constante
                n_obs = len(all_vars)
                
                if n_params >= n_obs - 5:
                    print(f"ARDL(p={p}, q={q}): {n_params} params, {n_obs} obs → ignoré (trop de paramètres)")
                    continue
                
                # Estimation
                X_design = sm.add_constant(all_vars)
                model = sm.OLS(y_curr, X_design).fit()
                
                results_list.append({
                    "p": p,
                    "q": q,
                    "aic": model.aic,
                    "bic": model.bic,
                    "r2_adj": model.rsquared_adj,
                    "n_params": n_params,
                    "n_obs": n_obs
                })
                
                print(f"ARDL(p={p}, q={q}): AIC={model.aic:.2f}, R²adj={model.rsquared_adj:.4f}")
                
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_model = model
                    best_order = (p, q)
                    
            except Exception as e:
                print(f"ARDL(p={p}, q={q}): erreur - {str(e)[:50]}")
                continue
    
    if best_model is None:
        print("\n❌ Aucun modèle NARDL estimé avec succès")
        return None, None, None
    
    print("\n" + "-"*80)
    print(f"✅ Modèle NARDL sélectionné: p={best_order[0]}, q={best_order[1]}")
    print(f"AIC: {best_aic:.4f}")
    print(f"R² ajusté: {best_model.rsquared_adj:.4f}")
    print(f"Observations: {best_model.nobs}, Paramètres: {len(best_model.params)}")
    
    return best_model, best_order, results_list

def wald_asymmetry_test(best_model, q):
    """
    Effectue le test de Wald pour l'asymétrie.
    H0: somme des coefficients de iva_pos = somme des coefficients de iva_neg
    """
    print("\n" + "="*80)
    print("TEST D'ASYMÉTRIE (TEST DE WALD)")
    print("="*80)
    
    params = best_model.params
    
    # Récupérer les coefficients de iva_pos (contemporain + retards)
    pos_cols = ["iva_pos"] + [f"iva_pos_L{i}" for i in range(1, q + 1)]
    neg_cols = ["iva_neg"] + [f"iva_neg_L{i}" for i in range(1, q + 1)]
    
    pos_coefs = [params.get(col, 0) for col in pos_cols if col in params]
    neg_coefs = [params.get(col, 0) for col in neg_cols if col in params]
    
    sum_pos = sum(pos_coefs)
    sum_neg = sum(neg_coefs)
    
    print(f"Somme des coefficients de iva_pos: {sum_pos:.6f}")
    print(f"Somme des coefficients de iva_neg: {sum_neg:.6f}")
    print(f"Différence (pos - neg): {sum_pos - sum_neg:.6f}")
    
    # Test de Wald simplifié
    # H0: sum_pos = sum_neg
    diff = sum_pos - sum_neg
    
    # Récupérer les variances et covariances
    pos_vars = [best_model.bse.get(col, 0)**2 for col in pos_cols if col in params]
    neg_vars = [best_model.bse.get(col, 0)**2 for col in neg_cols if col in params]
    
    # Covariances approximatives (on suppose indépendance pour simplification)
    var_diff = sum(pos_vars) + sum(neg_vars)
    se_diff = np.sqrt(var_diff) if var_diff > 0 else np.nan
    
    if not np.isnan(se_diff) and se_diff > 0:
        t_stat = diff / se_diff
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=best_model.df_resid))
        
        print(f"\nStatistique t: {t_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        print("\n" + "-"*80)
        if p_value < 0.05:
            print("✅ Conclusion: ASYMÉTRIE SIGNIFICATIVE (p < 0.05)")
            print("   → L'effet d'une augmentation de l'industrie diffère de celui d'une diminution.")
            asymmetry_present = True
        else:
            print("❌ Conclusion: PAS D'ASYMÉTRIE SIGNIFICATIVE (p ≥ 0.05)")
            print("   → L'effet est symétrique. Le modèle ARDL linéaire est approprié.")
            asymmetry_present = False
    else:
        print("\n❌ Impossible de calculer le test d'asymétrie")
        asymmetry_present = None
    
    return {
        "sum_pos": sum_pos,
        "sum_neg": sum_neg,
        "diff": diff,
        "t_stat": t_stat if 't_stat' in locals() else None,
        "p_value": p_value if 'p_value' in locals() else None,
        "asymmetry_present": asymmetry_present
    }

def save_nardl_results(best_model, best_order, asymmetry_results, output_path):
    """
    Sauvegarde les résultats du NARDL.
    """
    p, q = best_order
    
    txt_path = output_path / "nardl_results.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MODÈLE NARDL (NONLINEAR ARDL)\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Période: 1990-2024\n")
        f.write(f"Observations: {best_model.nobs}\n")
        f.write(f"Paramètres: {len(best_model.params)}\n")
        f.write(f"Modèle sélectionné: p={p}, q={q}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("RÉSULTATS DE L'ESTIMATION\n")
        f.write("-"*80 + "\n")
        f.write(str(best_model.summary()))
        
        f.write("\n\n" + "-"*80 + "\n")
        f.write("TEST D'ASYMÉTRIE (WALD)\n")
        f.write("-"*80 + "\n")
        f.write(f"Somme des coefficients iva_pos: {asymmetry_results['sum_pos']:.6f}\n")
        f.write(f"Somme des coefficients iva_neg: {asymmetry_results['sum_neg']:.6f}\n")
        f.write(f"Différence: {asymmetry_results['diff']:.6f}\n")
        if asymmetry_results['t_stat']:
            f.write(f"Statistique t: {asymmetry_results['t_stat']:.4f}\n")
            f.write(f"P-value: {asymmetry_results['p_value']:.4f}\n")
        f.write(f"\nAsymétrie significative: {asymmetry_results['asymmetry_present']}\n")
    
    print(f"\nRésultats sauvegardés dans: {txt_path}")

def run_nardl_analysis(df_log):
    """
    Lance l'analyse NARDL complète.
    """
    best_model, best_order, _ = estimate_nardl(df_log, max_lags=2)
    
    if best_model is None:
        print("Erreur: impossible d'estimer le modèle NARDL")
        return None, None
    
    p, q = best_order
    
    # Test d'asymétrie
    asymmetry_results = wald_asymmetry_test(best_model, q)
    
    # Sauvegarde
    output_path = Path(config.RESULTS_DIR)
    save_nardl_results(best_model, best_order, asymmetry_results, output_path)
    
    return best_model, asymmetry_results

if __name__ == "__main__":
    from data_loader import load_processed_data
    levels, log, diff = load_processed_data()
    run_nardl_analysis(log)