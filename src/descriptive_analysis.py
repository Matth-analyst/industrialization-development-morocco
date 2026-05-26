# src/descriptive_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
import config

def descriptive_stats(df_levels: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les statistiques descriptives de base.
    """
    stats = df_levels.describe().T
    stats["variance"] = df_levels.var()
    stats["skewness"] = df_levels.skew()
    stats["kurtosis"] = df_levels.kurtosis()
    
    # Ajouter les noms complets
    stats["variable"] = [config.VAR_LABELS.get(var, var) for var in stats.index]
    
    return stats

def correlation_matrix(df_log: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Calcule et sauvegarde la matrice de corrélation.
    """
    # Sélectionner les colonnes ln_
    ln_cols = [col for col in df_log.columns if col.startswith("ln_")]
    corr = df_log[ln_cols].corr()
    
    if save:
        path = Path(config.TABLES_DIR) / "correlation_matrix.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        corr.to_csv(path)
        print(f"Matrice de corrélation sauvegardée dans {path}")
    
    return corr

def plot_trends(df_levels: pd.DataFrame, save: bool = True):
    """
    Trace les tendances temporelles des variables.
    """
    n_vars = len(df_levels.columns)
    fig, axes = plt.subplots(n_vars, 1, figsize=(10, 3*n_vars), sharex=True)
    
    if n_vars == 1:
        axes = [axes]
    
    for ax, var in zip(axes, df_levels.columns):
        ax.plot(df_levels.index, df_levels[var], linewidth=1.5)
        ax.set_ylabel(config.VAR_LABELS.get(var, var), fontsize=9)
        ax.set_title(f"Évolution de {config.VAR_LABELS.get(var, var)} ({df_levels.index.year.min()}-{df_levels.index.year.max()})", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=0.5)
    
    axes[-1].set_xlabel("Année")
    plt.tight_layout()
    
    if save:
        path = Path(config.FIGURES_DIR) / "trends.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
        print(f"Graphique sauvegardé dans {path}")
    
    plt.show()

def plot_heatmap(df_log: pd.DataFrame, save: bool = True):
    """
    Trace la heatmap des corrélations.
    """
    ln_cols = [col for col in df_log.columns if col.startswith("ln_")]
    corr = df_log[ln_cols].corr()
    
    # Renommer pour lisibilité
    rename = {col: config.VAR_LABELS.get(col.replace("ln_", ""), col) for col in ln_cols}
    corr_renamed = corr.rename(index=rename, columns=rename)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_renamed, annot=True, cmap="RdBu_r", center=0, 
                fmt=".2f", square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Matrice des corrélations (variables en log)")
    
    plt.tight_layout()
    
    if save:
        path = Path(config.FIGURES_DIR) / "correlation_heatmap.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
        print(f"Heatmap sauvegardée dans {path}")
    
    plt.show()

def plot_distributions(df_levels: pd.DataFrame, save: bool = True):
    """
    Trace les distributions (histogrammes + densités).
    """
    n_vars = len(df_levels.columns)
    fig, axes = plt.subplots(n_vars, 1, figsize=(8, 3*n_vars))
    
    if n_vars == 1:
        axes = [axes]
    
    for ax, var in zip(axes, df_levels.columns):
        sns.histplot(df_levels[var], kde=True, ax=ax, color="steelblue", edgecolor="black")
        ax.set_title(config.VAR_LABELS.get(var, var))
        ax.axvline(x=df_levels[var].mean(), color='red', linestyle='--', label=f"Moyenne: {df_levels[var].mean():.2f}")
        ax.axvline(x=df_levels[var].median(), color='green', linestyle='--', label=f"Médiane: {df_levels[var].median():.2f}")
        ax.legend()
    
    plt.tight_layout()
    
    if save:
        path = Path(config.FIGURES_DIR) / "distributions.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
        print(f"Distributions sauvegardées dans {path}")
    
    plt.show()

def run_descriptive_analysis(df_levels: pd.DataFrame, df_log: pd.DataFrame):
    """
    Lance toute l'analyse descriptive.
    """
    print("\n" + "="*60)
    print("ANALYSE DESCRIPTIVE")
    print("="*60)
    
    # 1. Statistiques descriptives
    print("\n--- Statistiques descriptives ---")
    stats = descriptive_stats(df_levels)
    print(stats[["mean", "std", "min", "max", "skewness", "kurtosis"]].round(4))
    
    # Sauvegarde
    stats_path = Path(config.TABLES_DIR) / "descriptive_stats.csv"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(stats_path)
    print(f"\nStatistiques sauvegardées dans {stats_path}")
    
    # 2. Matrice de corrélation
    print("\n--- Matrice de corrélation ---")
    corr = correlation_matrix(df_log, save=True)
    print(corr.round(3))
    
    # 3. Graphiques
    print("\n--- Génération des graphiques ---")
    plot_trends(df_levels, save=True)
    plot_heatmap(df_log, save=True)
    plot_distributions(df_levels, save=True)
    
    return stats, corr

if __name__ == "__main__":
    from data_loader import load_processed_data
    levels, log, diff = load_processed_data()
    run_descriptive_analysis(levels, log)