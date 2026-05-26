# src/ardl_model.py
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
import config

def select_optimal_lags(df_log: pd.DataFrame):
    """
    Sélectionne les retards optimaux pour le modèle ARDL.
    """
    max_lags = config.MAX_LAGS
    print("="*80)
    print("SÉLECTION DES RETARDS OPTIMAUX (ARDL)")
    print("="*80)
    print(f"Nombre de retards maximum: {max_lags}")
    
    y = df_log["ln_pibhab"].dropna()
    X_cols = [f"ln_{var}" for var in config.INDEPENDENT_VARS_USED if f"ln_{var}" in df_log.columns]
    X = df_log[X_cols]
    
    print(f"Variables explicatives: {len(X_cols)}")
    print("-"*80)
    
    data = pd.concat([y, X], axis=1).dropna()
    y_aligned = data["ln_pibhab"]
    X_aligned = data[X_cols]
    
    best_aic = np.inf
    best_order = None
    best_model = None
    best_X_design = None
    best_y_curr = None
    results_list = []
    
    for p in range(1, max_lags + 1):
        for q in range(0, max_lags + 1):
            n_params = 1 + p + (len(X_cols) * q)
            n_obs_available = len(y_aligned) - max(p, q)
            
            if n_params >= n_obs_available - 5:
                print(f"ARDL(p={p}, q={q}) → {n_params} params, {n_obs_available} obs → ignoré")
                continue
            
            try:
                print(f"ARDL(p={p}, q={q})...", end=" ")
                
                X_lagged = X_aligned.copy()
                for col in X_cols:
                    for lag in range(1, q + 1):
                        X_lagged[f"{col}_L{lag}"] = X_aligned[col].shift(lag)
                
                y_lagged = pd.DataFrame()
                for lag in range(1, p + 1):
                    y_lagged[f"y_L{lag}"] = y_aligned.shift(lag)
                
                all_vars = pd.concat([y_lagged, X_lagged], axis=1).dropna()
                y_curr = y_aligned.loc[all_vars.index]
                X_design = sm.add_constant(all_vars)
                
                model = sm.OLS(y_curr, X_design).fit()
                
                results_list.append({
                    "p": p, "q": q, "aic": model.aic, "bic": model.bic,
                    "r2_adj": model.rsquared_adj, "n_params": len(model.params),
                    "n_obs": len(all_vars)
                })
                
                print(f"AIC={model.aic:.2f}")
                
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = (p, q)
                    best_model = model
                    best_X_design = X_design
                    best_y_curr = y_curr
                    
            except Exception as e:
                print(f"erreur")
                continue
    
    results_df = pd.DataFrame(results_list)
    
    if len(results_df) > 0:
        results_df = results_df.sort_values("aic")
        
        print("\n" + "-"*80)
        print(f"✅ Modèle sélectionné: ARDL(p={best_order[0]}, q={best_order[1]})")
        print(f"Meilleur AIC: {best_aic:.4f}")
        print(f"R² ajusté: {best_model.rsquared_adj:.4f}")
        
        results_path = Path(config.RESULTS_DIR) / "lag_selection.csv"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path, index=False)
        
        return best_order, best_model, best_X_design, best_y_curr
    else:
        print("❌ Aucun modèle trouvé")
        return None, None, None, None

def estimate_robust_model(y, X, maxlags=3):
    """
    Estimation avec erreurs-types robustes HAC (Newey-West).
    """
    try:
        model = sm.OLS(y, X)
        results = model.fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
        return results
    except Exception as e:
        print(f"Erreur HAC: {e}")
        return None

def save_detailed_results(best_order, robust_model, df_log):
    """
    Sauvegarde des résultats détaillés dans un fichier texte.
    """
    p, q = best_order
    
    # Récupérer les noms des variables
    X_cols = [f"ln_{var}" for var in config.INDEPENDENT_VARS_USED
              if f"ln_{var}" in df_log.columns]
    
    output_path = Path(config.RESULTS_DIR) / "ardl_results_hac.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MODÈLE ARDL-ECM AVEC CORRECTION HAC (NEWEY-WEST)\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Date d'exécution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Période d'étude: 1990-2024\n")
        f.write(f"Variable dépendante: PIB par habitant (ln_pibhab)\n\n")
        
        f.write("-"*80 + "\n")
        f.write("SPÉCIFICATION DU MODÈLE\n")
        f.write("-"*80 + "\n")
        f.write(f"Retards de la variable dépendante (p): {p}\n")
        f.write(f"Retards des variables exogènes (q): {q}\n")
        f.write(f"Variables exogènes: {', '.join(X_cols)}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("STATISTIQUES DU MODÈLE\n")
        f.write("-"*80 + "\n")
        f.write(f"Observations: {robust_model.nobs}\n")
        f.write(f"Paramètres estimés: {len(robust_model.params)}\n")
        f.write(f"R²: {robust_model.rsquared:.4f}\n")
        f.write(f"R² ajusté: {robust_model.rsquared_adj:.4f}\n")
        f.write(f"AIC (Critère d'Akaike): {robust_model.aic:.4f}\n")
        f.write(f"BIC (Critère de Schwarz): {robust_model.bic:.4f}\n")
        f.write(f"Log-likelihood: {robust_model.llf:.4f}\n")
        f.write(f"Statistique F: {robust_model.fvalue:.4f}\n")
        f.write(f"P-value (F): {robust_model.f_pvalue:.4e}\n")
        f.write(f"Durbin-Watson: {sm.stats.durbin_watson(robust_model.resid):.4f}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("COEFFICIENTS ESTIMÉS (ERREURS-TYPES ROBUSTES HAC)\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Variable':<30} {'Coefficient':>12} {'Erreur-type':>12} {'t-stat':>10} {'p-value':>10} {'Signif':>8}\n")
        f.write("-"*80 + "\n")
        
        for name, coef, se, tval, pval in zip(robust_model.params.index,
                                               robust_model.params.values,
                                               robust_model.bse.values,
                                               robust_model.tvalues.values,
                                               robust_model.pvalues.values):
            if pval < 0.01:
                sig = "***"
            elif pval < 0.05:
                sig = "**"
            elif pval < 0.10:
                sig = "*"
            else:
                sig = ""
            f.write(f"{name[:30]:<30} {coef:12.6f} {se:12.6f} {tval:10.4f} {pval:10.4f} {sig:>8}\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("NOTE SUR LA SIGNIFICATIVITÉ\n")
        f.write("-"*80 + "\n")
        f.write("*** p < 0.01, ** p < 0.05, * p < 0.10\n")
        f.write("Erreurs-types corrigées par la méthode de Newey-West (HAC) avec 3 retards\n")
    
    print(f"\n✅ Résultats détaillés sauvegardés dans: {output_path}")
    return output_path

def run_ardl_analysis(df_log: pd.DataFrame):
    """
    Lance l'analyse ARDL complète avec correction HAC.
    """
    print("\n" + "="*80)
    print("ANALYSE ARDL AVEC CORRECTION HAC")
    print("="*80)
    
    best_order, best_model_std, X_design, y_curr = select_optimal_lags(df_log)
    
    if best_order is None:
        return None, None
    
    p, q = best_order
    
    print("\n" + "="*80)
    print("ESTIMATION AVEC ERREURS-TYPES ROBUSTES (HAC)")
    print("="*80)
    
    robust_model = estimate_robust_model(y_curr, X_design, maxlags=3)
    
    if robust_model is None:
        return best_order, best_model_std
    
    print(f"\nModèle: ARDL(p={p}, q={q})")
    print(f"Observations: {robust_model.nobs}")
    print(f"R² ajusté: {robust_model.rsquared_adj:.4f}")
    print(f"AIC: {robust_model.aic:.4f}")
    print(f"BIC: {robust_model.bic:.4f}")
    
    print("\nCoefficients significatifs (erreurs-types robustes):")
    print("-"*70)
    significant_found = False
    for name, coef, se, pval in zip(robust_model.params.index,
                                     robust_model.params.values,
                                     robust_model.bse.values,
                                     robust_model.pvalues.values):
        sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.10 else ""))
        if sig:
            significant_found = True
            print(f"  ✅ {name[:25]:25} {coef:10.4f} {se:10.4f} {pval:8.4f} {sig}")
    
    if not significant_found:
        print("  ❌ Aucune variable significative à 10%")
    
    # Sauvegarde détaillée
    save_detailed_results(best_order, robust_model, df_log)
    
    return best_order, robust_model

if __name__ == "__main__":
    from data_loader import load_processed_data
    levels, log, diff = load_processed_data()
    run_ardl_analysis(log)