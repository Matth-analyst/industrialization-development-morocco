# src/stationarity.py
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent.parent))
import config

def adf_test(series, variable_name, maxlag=4, regression='c'):
    """
    Test ADF (Augmented Dickey-Fuller)
    """
    try:
        result = adfuller(series, maxlag=maxlag, regression=regression, autolag='AIC')
    except TypeError:
        result = adfuller(series, maxlag=maxlag, autolag='AIC', regresults=False)
    
    critical_values = {
        "1%": result[4]['1%'] if isinstance(result[4], dict) else result[4][0],
        "5%": result[4]['5%'] if isinstance(result[4], dict) else result[4][1],
        "10%": result[4]['10%'] if isinstance(result[4], dict) else result[4][2]
    }
    
    output = {
        "variable": variable_name,
        "test": "ADF",
        "regression": regression,
        "statistic": result[0],
        "p_value": result[1],
        "critical_values": critical_values,
        "lags": result[2],
        "stationary": result[1] < 0.05
    }
    return output

def kpss_test(series, variable_name, regression='c'):
    """
    Test KPSS (Kwiatkowski-Phillips-Schmidt-Shin)
    """
    result = kpss(series, regression=regression, nlags='auto')
    
    critical_values = {
        "10%": result[3]['10%'],
        "5%": result[3]['5%'],
        "2.5%": result[3]['2.5%'],
        "1%": result[3]['1%']
    }
    
    output = {
        "variable": variable_name,
        "test": "KPSS",
        "regression": regression,
        "statistic": result[0],
        "p_value": result[1],
        "critical_values": critical_values,
        "stationary": result[1] > 0.05
    }
    return output

def save_test_results_to_txt(results_list, filename):
    """
    Sauvegarde les résultats des tests dans un fichier .txt dans results/
    """
    output_path = Path(config.RESULTS_DIR) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"TESTS DE RACINE UNITAIRE\n")
        f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        for res in results_list:
            f.write("-"*60 + "\n")
            f.write(f"Variable: {res['variable']}\n")
            f.write(f"Test: {res['test']}\n")
            f.write(f"Spécification: {res.get('regression', 'N/A')}\n")
            f.write(f"Statistique: {res['statistic']:.6f}\n")
            f.write(f"P-value: {res['p_value']:.6f}\n")
            f.write("\nValeurs critiques:\n")
            for level, cv in res['critical_values'].items():
                f.write(f"  {level}: {cv:.4f}\n")
            f.write(f"\nConclusion: ")
            if res['stationary']:
                if res['test'] == 'KPSS':
                    f.write("Stationnaire (H0 non rejetée)\n")
                else:
                    f.write("Stationnaire (H0 rejetée)\n")
            else:
                if res['test'] == 'KPSS':
                    f.write("Non stationnaire (H0 rejetée)\n")
                else:
                    f.write("Non stationnaire (H0 non rejetée)\n")
            f.write("\n")
    
    print(f"  → Sauvegardé: {output_path}")
    return output_path

def run_all_stationarity_tests(df_log: pd.DataFrame, df_diff: pd.DataFrame):
    """
    Exécute les tests sur toutes les variables.
    """
    ln_cols = [col for col in df_log.columns if col.startswith("ln_")]
    
    print("\n" + "="*80)
    print("TESTS DE RACINE UNITAIRE")
    print("="*80)
    print("H0 ADF: série non stationnaire | H0 KPSS: série stationnaire")
    print("-"*80)
    
    # 1. Tests sur niveaux (constante seule)
    print("\n--- TESTS SUR NIVEAUX (constante) ---")
    level_c_results = []
    for col in ln_cols:
        series = df_log[col].dropna()
        var_name = col.replace("ln_", "")
        
        adf_res = adf_test(series, var_name, regression='c')
        kpss_res = kpss_test(series, var_name, regression='c')
        level_c_results.append(adf_res)
        level_c_results.append(kpss_res)
        
        print(f"\n{var_name.upper()}:")
        print(f"  ADF : stat={adf_res['statistic']:.4f}, p={adf_res['p_value']:.4f} -> {'Stationnaire' if adf_res['stationary'] else 'Non stationnaire'}")
        print(f"  KPSS: stat={kpss_res['statistic']:.4f}, p={kpss_res['p_value']:.4f} -> {'Stationnaire' if kpss_res['stationary'] else 'Non stationnaire'}")
    
    save_test_results_to_txt(level_c_results, "stationarity_level_constant.txt")
    
    # 2. Tests sur niveaux (constante + tendance)
    print("\n--- TESTS SUR NIVEAUX (constante + tendance) ---")
    level_ct_results = []
    for col in ln_cols:
        series = df_log[col].dropna()
        var_name = col.replace("ln_", "")
        
        adf_res = adf_test(series, var_name, regression='ct')
        kpss_res = kpss_test(series, var_name, regression='ct')
        level_ct_results.append(adf_res)
        level_ct_results.append(kpss_res)
        
        print(f"\n{var_name.upper()}:")
        print(f"  ADF : stat={adf_res['statistic']:.4f}, p={adf_res['p_value']:.4f} -> {'Stationnaire' if adf_res['stationary'] else 'Non stationnaire'}")
        print(f"  KPSS: stat={kpss_res['statistic']:.4f}, p={kpss_res['p_value']:.4f} -> {'Stationnaire' if kpss_res['stationary'] else 'Non stationnaire'}")
    
    save_test_results_to_txt(level_ct_results, "stationarity_level_trend.txt")
    
    # 3. Tests sur premières différences
    print("\n--- TESTS SUR PREMIÈRES DIFFÉRENCES (constante) ---")
    diff_results = []
    for col in ln_cols:
        diff_col = f"d_{col}"
        if diff_col in df_diff.columns:
            series = df_diff[diff_col].dropna()
            var_name = f"Δ{col.replace('ln_', '')}"
            
            adf_res = adf_test(series, var_name, regression='c')
            kpss_res = kpss_test(series, var_name, regression='c')
            diff_results.append(adf_res)
            diff_results.append(kpss_res)
            
            print(f"\n{var_name}:")
            print(f"  ADF : stat={adf_res['statistic']:.4f}, p={adf_res['p_value']:.4f} -> {'Stationnaire' if adf_res['stationary'] else 'Non stationnaire'}")
            print(f"  KPSS: stat={kpss_res['statistic']:.4f}, p={kpss_res['p_value']:.4f} -> {'Stationnaire' if kpss_res['stationary'] else 'Non stationnaire'}")
    
    save_test_results_to_txt(diff_results, "stationarity_first_diff.txt")
    
    return level_c_results, level_ct_results, diff_results

def get_integration_order(level_c_results, diff_results) -> dict:
    """
    Détermine l'ordre d'intégration de chaque variable.
    """
    integration_order = {}
    
    level_adf = [r for r in level_c_results if r['test'] == 'ADF']
    diff_adf = [r for r in diff_results if r['test'] == 'ADF']
    
    for adf_level in level_adf:
        var = adf_level['variable']
        adf_diff = next((r for r in diff_adf if r['variable'] == f"Δ{var}"), None)
        
        if adf_level['stationary']:
            integration_order[var] = "I(0)"
        elif adf_diff and adf_diff['stationary']:
            integration_order[var] = "I(1)"
        else:
            integration_order[var] = "I(2) ou plus"
    
    return integration_order

def print_integration_summary(integration_order: dict):
    """
    Affiche le résumé.
    """
    print("\n" + "="*80)
    print("ORDRE D'INTÉGRATION DES VARIABLES")
    print("="*80)
    
    for var, order in integration_order.items():
        print(f"  {var.upper():20} → {order}")
    
    i2_vars = [var for var, order in integration_order.items() if "I(2)" in order]
    
    if i2_vars:
        print(f"\n⚠️ ATTENTION: Variables I(2): {i2_vars}")
    else:
        print("\n✅ Condition ARDL validée: toutes les variables sont I(0) ou I(1)")

def save_integration_summary(integration_order: dict):
    """
    Sauvegarde le résumé dans results/
    """
    output_path = Path(config.RESULTS_DIR) / "integration_order.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ORDRE D'INTÉGRATION DES VARIABLES\n")
        f.write("="*80 + "\n\n")
        
        for var, order in integration_order.items():
            f.write(f"{var.upper():20} → {order}\n")
        
        i2_vars = [var for var, order in integration_order.items() if "I(2)" in order]
        
        if i2_vars:
            f.write(f"\n⚠️ ATTENTION: Variables I(2): {i2_vars}\n")
        else:
            f.write("\n✅ Condition ARDL validée\n")
    
    print(f"\n  → Résumé sauvegardé: {output_path}")

if __name__ == "__main__":
    from data_loader import load_processed_data
    levels, log, diff = load_processed_data()
    
    level_c, level_ct, diff_res = run_all_stationarity_tests(log, diff)
    order = get_integration_order(level_c, diff_res)
    print_integration_summary(order)
    save_integration_summary(order)