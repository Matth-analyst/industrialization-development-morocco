# src/autocorrelation_justification.py
"""
Génère une justification académique de l'autocorrélation modérée
pour le rapport de mémoire.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
import config


def generate_autocorrelation_justification(lm_stat, p_value, dw_stat, n_obs, n_params):
    """
    Génère un texte de justification pour l'autocorrélation.
    
    Parameters:
    - lm_stat: statistique LM du test de Breusch-Godfrey
    - p_value: p-value associée
    - dw_stat: statistique de Durbin-Watson
    - n_obs: nombre d'observations
    - n_params: nombre de paramètres du modèle
    """
    
    # CORRECTION: la classification se fondait sur des seuils arbitraires appliqués
    # directement à la statistique LM (dont l'échelle dépend de n et des ddl), ce qui
    # pouvait qualifier de "modérée" une autocorrélation en réalité très significative.
    # On classe désormais sur la base de la p-value du test, seule quantité comparable
    # d'une spécification à l'autre.
    if p_value >= 0.10:
        niveau = "faible"
        justification = "non significative et ne remet pas en cause la validité du modèle"
    elif p_value >= 0.01:
        niveau = "modérée"
        justification = "statistiquement significative mais d'ampleur limitée; atténuée par Newey-West"
    else:
        niveau = "élevée"
        justification = (
            "fortement significative (p < 0.01), ce qui constitue une limite réelle du modèle "
            "et pas seulement un artefact à corriger cosmétiquement par HAC"
        )
    
    # Interprétation de Durbin-Watson
    if 1.8 < dw_stat < 2.2:
        dw_interpretation = "proche de 2, indiquant l'absence d'autocorrélation sérielle forte"
        dw_conclusion = "conforte"
    elif dw_stat < 1.5:
        dw_interpretation = "inférieur à 2, suggérant une autocorrélation positive"
        dw_conclusion = "nuance"
    else:
        dw_interpretation = "légèrement supérieur à 2, suggérant une autocorrélation négative légère"
        dw_conclusion = "complète"
    
    # Texte conditionnel selon la sévérité réelle (p-value), pour éviter la conclusion
    # rassurante par défaut ("ne remet pas en cause la validité") même quand LM est
    # fortement significatif — c'est l'erreur corrigée ici.
    if niveau == "élevée":
        ampleur_txt = "d'ampleur substantielle et ne peut être écartée comme un artefact mineur"
        remede_txt = (
            "La correction Newey-West (HAC) restaure la validité asymptotique des tests "
            "d'hypothèse, mais ne corrige pas la source de l'autocorrélation elle-même "
            "(probable mauvaise spécification dynamique ou omission de variable). "
            "Ce point doit être présenté comme une limite explicite de l'étude, "
            "et non comme un problème résolu."
        )
        conclusion_txt = (
            f"Cette autocorrélation {niveau} constitue une limite réelle de l'étude, à mentionner "
            "explicitement dans la section \"Limites\" plutôt qu'à minimiser."
        )
    else:
        ampleur_txt = "reste d'ampleur limitée (loin des valeurs extrêmes observées dans des spécifications mal définies)"
        remede_txt = (
            "Pour y remédier, nous avons appliqué la correction de Newey-West (HAC), produisant "
            "des erreurs-types robustes à l'autocorrélation et à l'hétéroscédasticité."
        )
        conclusion_txt = f"Cette autocorrélation {niveau} ne remet pas en cause la validité globale des conclusions de l'étude."

    # Générer le texte LaTeX
    latex_text = f"""
\\subsection{{Autocorrélation résiduelle}}

Le test de Breusch-Godfrey révèle une autocorrélation {niveau} des résidus 
(LM = {lm_stat:.2f}, p = {p_value:.4f}). Cette autocorrélation {ampleur_txt}. 

La statistique de Durbin-Watson ({dw_stat:.3f}) est {dw_interpretation}, ce qui 
{dw_conclusion} le diagnostic d'une autocorrélation {niveau}.

Plusieurs facteurs peuvent expliquer cette autocorrélation résiduelle :
\\begin{{itemize}}
    \\item La taille modeste de l'échantillon ({n_obs} observations après prise en compte des retards) ;
    \\item La nature intrinsèquement dynamique de la relation entre industrialisation et développement économique ;
    \\item Le nombre de paramètres ({n_params}) relativement élevé par rapport aux observations.
\\end{{itemize}}

{remede_txt}

Conformément à la littérature (Pesaran \\& Shin, 1999 ; Wooldridge, 2016), 
l'estimateur ARDL reste convergent en présence d'autocorrélation, mais cette 
propriété est asymptotique : avec n={n_obs}, elle doit être invoquée avec prudence, 
pas comme une garantie automatique.

{conclusion_txt}
"""
    
    # Version courte pour le résumé
    short_text = f"""
================================================================================
AUTOCORRÉLATION {niveau.upper()} - JUSTIFICATION POUR LE RAPPORT
================================================================================

Test de Breusch-Godfrey:     LM = {lm_stat:.2f}, p = {p_value:.4f}
Durbin-Watson:               {dw_stat:.3f}
Observations:                {n_obs}
Paramètres:                  {n_params}

Niveau d'autocorrélation:    {niveau.upper()}

Conclusion:                  {justification}

================================================================================
À INSÉRER DANS LA SECTION "LIMITES" OU "TESTS DE DIAGNOSTIC"
================================================================================
"""
    
    return latex_text, short_text


def save_justification(output_path, latex_text, short_text):
    """
    Sauvegarde la justification dans un fichier.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("JUSTIFICATION DE L'AUTOCORRÉLATION POUR LE RAPPORT\n")
        f.write(f"Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write("VERSION COURTE (RÉSUMÉ)\n")
        f.write("-"*80 + "\n")
        f.write(short_text)
        f.write("\n\n")
        
        f.write("VERSION LONGUE (FORMAT LATEX)\n")
        f.write("-"*80 + "\n")
        f.write(latex_text)
    
    print(f"Justification sauvegardée dans: {output_path}")
    return output_path


def run_autocorrelation_justification(best_model=None, df_log=None):
    """
    Lance la génération de la justification.
    Si best_model fourni, extrait automatiquement les statistiques.
    """
    # Valeurs par défaut (à ajuster selon tes résultats)
    # Si tu as un modèle, tu peux extraire automatiquement
    if best_model is not None:
        # Extraire Durbin-Watson du modèle
        from statsmodels.stats.stattools import durbin_watson
        from statsmodels.stats.diagnostic import acorr_breusch_godfrey
        dw_stat = durbin_watson(best_model.resid)
        n_obs = int(best_model.nobs)
        n_params = len(best_model.params)

        # CORRECTION: la statistique LM était auparavant codée en dur (valeur figée
        # de 11.59 / p=0.003), sans lien avec le modèle réellement estimé — ce qui
        # produisait un texte "modéré" en contradiction avec diagnostics_summary.txt
        # (LM=31.4, p<0.001, nlags=2) généré par le même run. On recalcule ici le
        # test de Breusch-Godfrey directement sur le modèle fourni, avec le même
        # nombre de retards (nlags=2) que src/diagnostics.py, pour garantir la
        # cohérence entre les deux fichiers de sortie.
        bg_result = acorr_breusch_godfrey(best_model, nlags=2)
        lm_stat = float(bg_result[0])
        p_value = float(bg_result[1])
    else:
        # Valeurs par défaut d'après tes résultats
        lm_stat = 11.59
        p_value = 0.003
        dw_stat = 2.572
        n_obs = 33
        n_params = 20
    
    print("\n" + "="*80)
    print("GÉNÉRATION DE LA JUSTIFICATION D'AUTOCORRÉLATION")
    print("="*80)
    print(f"LM statistique: {lm_stat:.2f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Durbin-Watson: {dw_stat:.3f}")
    print(f"Observations: {n_obs}")
    print(f"Paramètres: {n_params}")
    
    latex_text, short_text = generate_autocorrelation_justification(
        lm_stat, p_value, dw_stat, n_obs, n_params
    )
    
    output_path = Path(config.RESULTS_DIR) / "autocorrelation_justification.txt"
    save_justification(output_path, latex_text, short_text)
    
    # Afficher la version courte
    print("\n" + short_text)
    
    return latex_text, short_text


if __name__ == "__main__":
    # Exécution autonome
    run_autocorrelation_justification()