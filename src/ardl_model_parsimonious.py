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
    
    criterion = getattr(config, "LAG_CRITERION", "bic").lower()
    min_df = getattr(config, "MIN_RESIDUAL_DF", 10)
    best_score = np.inf
    best_order = None
    best_model = None
    best_X_design = None
    best_y_curr = None
    results_list = []
    
    print(f"Critère de sélection: {criterion.upper()} (garde-fou: >= {min_df} degrés de liberté résiduels)")
    
    for p in range(1, max_lags + 1):
        for q in range(0, max_lags + 1):
            n_params = 1 + p + (len(X_cols) * q)
            n_obs_available = len(y_aligned) - max(p, q)
            residual_df = n_obs_available - n_params
            
            if residual_df < min_df:
                print(f"ARDL(p={p}, q={q}) → {n_params} params, {n_obs_available} obs, "
                      f"{residual_df} ddl résiduels < {min_df} → ignoré (surparamétrage)")
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
                
                score = model.bic if criterion == "bic" else model.aic
                print(f"AIC={model.aic:.2f}  BIC={model.bic:.2f}  ddl_resid={len(y_curr) - len(model.params)}")
                
                if score < best_score:
                    best_score = score
                    best_order = (p, q)
                    best_model = model
                    best_X_design = X_design
                    best_y_curr = y_curr
                    
            except Exception as e:
                print(f"erreur")
                continue
    
    results_df = pd.DataFrame(results_list)
    
    if len(results_df) > 0:
        results_df = results_df.sort_values(criterion)
        
        print("\n" + "-"*80)
        print(f"✅ Modèle sélectionné (critère {criterion.upper()}): ARDL(p={best_order[0]}, q={best_order[1]})")
        print(f"Meilleur {criterion.upper()}: {best_score:.4f}")
        print(f"R² ajusté: {best_model.rsquared_adj:.4f}")
        print(f"Degrés de liberté résiduels: {len(best_y_curr) - len(best_model.params)}")
        
        results_path = Path(config.RESULTS_DIR) / "lag_selection.csv"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path, index=False)
        
        return best_order, best_model, best_X_design, best_y_curr
    else:
        print("❌ Aucun modèle trouvé")
        return None, None, None, None

# Valeurs critiques asymptotiques du test des bornes de Pesaran, Shin & Smith (2001),
# Table CI(iii), cas "constante non contrainte, pas de tendance", pour k régresseurs
# de long terme (hors variable dépendante). Ce sont des valeurs asymptotiques (grand
# échantillon) : avec n≈30-35, elles sont une approximation, pas une valeur exacte
# (les tables exactes en petit échantillon, ex. Narayan 2005, seraient préférables
# mais ne sont pas disponibles ici). À interpréter avec prudence pour ce motif.
PSS_BOUNDS_CV = {
    2: {0.10: (3.02, 3.51), 0.05: (3.62, 4.16), 0.01: (5.17, 5.85)},
    3: {0.10: (2.72, 3.77), 0.05: (3.23, 4.35), 0.01: (4.29, 5.61)},
    4: {0.10: (2.45, 3.52), 0.05: (2.86, 4.01), 0.01: (3.74, 5.06)},
    5: {0.10: (2.26, 3.35), 0.05: (2.62, 3.79), 0.01: (3.41, 4.68)},
    6: {0.10: (2.12, 3.23), 0.05: (2.45, 3.61), 0.01: (3.15, 4.43)},
}


def run_bounds_test(df_log: pd.DataFrame, p: int, q: int):
    """
    Test des bornes (Pesaran, Shin & Smith, 2001) pour la cointégration,
    calculé explicitement à partir de la représentation ECM du modèle ARDL(p,q)
    retenu. H0: pas de relation de niveau (tous les coefficients de niveau = 0).

    Ce test était mentionné dans le README de la version précédente du projet
    mais n'était pas implémenté dans le code — cette fonction corrige cela.
    """
    print("\n" + "="*80)
    print("TEST DES BORNES (BOUNDS TEST, PESARAN-SHIN-SMITH 2001)")
    print("="*80)

    y_col = "ln_pibhab"
    X_cols = [f"ln_{var}" for var in config.INDEPENDENT_VARS_USED if f"ln_{var}" in df_log.columns]
    k = len(X_cols)

    data = df_log[[y_col] + X_cols].dropna()

    dy = data[y_col].diff()
    dX = data[X_cols].diff()

    reg = pd.DataFrame(index=data.index)
    reg["dy"] = dy
    # niveaux retardés (t-1) : la partie testée sous H0
    reg["y_L1"] = data[y_col].shift(1)
    for col in X_cols:
        reg[f"{col}_L1"] = data[col].shift(1)
    # termes différenciés à court terme (retards 1..max(p,q)-1)
    for lag in range(1, p):
        reg[f"dy_L{lag}"] = dy.shift(lag)
    for col in X_cols:
        for lag in range(1, q):
            reg[f"d{col}_L{lag}"] = dX[col].shift(lag)

    reg = reg.dropna()
    y_reg = reg["dy"]
    X_reg = sm.add_constant(reg.drop(columns=["dy"]))

    unrestricted = sm.OLS(y_reg, X_reg).fit()

    level_vars = ["y_L1"] + [f"{col}_L1" for col in X_cols]
    hypotheses = " , ".join([f"{v} = 0" for v in level_vars])
    f_test = unrestricted.f_test(hypotheses)

    f_stat = float(f_test.fvalue)
    f_pvalue = float(f_test.pvalue)
    n_obs = int(unrestricted.nobs)

    cv = PSS_BOUNDS_CV.get(k)
    verdict_lines = []
    if cv is not None:
        for alpha, (i0, i1) in cv.items():
            if f_stat > i1:
                verdict = "F au-dessus de la borne I(1) → cointégration supportée"
            elif f_stat < i0:
                verdict = "F en-dessous de la borne I(0) → pas de cointégration"
            else:
                verdict = "F dans la zone d'indétermination (entre I(0) et I(1)) → non concluant"
            verdict_lines.append((alpha, i0, i1, verdict))

    print(f"Variables de niveau testées: {level_vars}")
    print(f"Observations utilisées: {n_obs}")
    print(f"Statistique F: {f_stat:.4f}  (p-value du test F = {f_pvalue:.4f})")
    for alpha, i0, i1, verdict in verdict_lines:
        print(f"  Seuil {int(alpha*100)}%: I(0)={i0}, I(1)={i1} → {verdict}")
    print("⚠️ Valeurs critiques asymptotiques (Pesaran, Shin & Smith 2001, Table CI(iii)); "
          "approximation en petit échantillon (n={}).".format(n_obs))

    results_path = Path(config.RESULTS_DIR) / "bounds_test.txt"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("TEST DES BORNES (BOUNDS TEST) DE PESARAN, SHIN & SMITH (2001)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Modèle: ARDL({p},{q}) — forme ECM\n")
        f.write(f"Variables de niveau testées sous H0 (coefficients nuls): {level_vars}\n")
        f.write(f"Observations utilisées: {n_obs}\n")
        f.write(f"Nombre de régresseurs de long terme (k, hors y): {k}\n\n")
        f.write(f"Statistique F: {f_stat:.4f}\n")
        f.write(f"P-value (test F conjoint, référence asymptotique chi2/F standard): {f_pvalue:.4f}\n\n")
        if cv is not None:
            f.write("Valeurs critiques (Pesaran, Shin & Smith 2001, Table CI(iii), "
                    "constante non contrainte, sans tendance):\n")
            for alpha, i0, i1, verdict in verdict_lines:
                f.write(f"  Seuil {int(alpha*100)}%: borne I(0)={i0}  borne I(1)={i1}  → {verdict}\n")
        else:
            f.write(f"Pas de table de valeurs critiques disponible pour k={k} régresseurs.\n")
        f.write("\n" + "-"*80 + "\n")
        f.write("MISE EN GARDE\n")
        f.write("-"*80 + "\n")
        f.write(
            "Les valeurs critiques utilisées sont asymptotiques (grand échantillon). "
            f"Avec n={n_obs} observations effectives, elles ne sont qu'une approximation: "
            "des tables en petit échantillon (ex. Narayan, 2005) seraient plus appropriées "
            "mais ne sont pas reproduites ici. Ce test doit donc être lu comme un indice "
            "et non comme une preuve formelle de cointégration.\n"
        )

    print(f"\n✅ Résultats sauvegardés dans: {results_path}")
    return f_stat, f_pvalue, verdict_lines


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

    # Test des bornes (cointégration) — calculé sur la spécification retenue
    run_bounds_test(df_log, p, q)
    
    return best_order, robust_model

if __name__ == "__main__":
    from data_loader import load_processed_data
    levels, log, diff = load_processed_data()
    run_ardl_analysis(log)