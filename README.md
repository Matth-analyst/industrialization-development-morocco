# Économétrie de l'industrialisation et du développement économique au Maroc (1990-2024)

Étude économétrique de la relation entre l'industrialisation et le développement
économique au Maroc sur la période 1990-2024, fondée sur une approche ARDL/NARDL
en série temporelle courte (35 observations annuelles).

## Aperçu du projet

L'étude analyse l'impact de la valeur ajoutée manufacturière, des dépenses
publiques, des investissements directs étrangers et du taux de change réel sur le
PIB par habitant au Maroc. Compte tenu de la taille limitée de l'échantillon
(n=35), l'ensemble de la démarche — sélection des variables, choix des retards,
diagnostics — a été conduit sous contrainte explicite de parcimonie, condition
nécessaire à des résultats interprétables sur une série aussi courte.

Points clés :
- Période : 1990-2024 (35 observations annuelles)
- Variable expliquée : PIB réel par habitant (PPA, USD constants 2011)
- Variables explicatives retenues : VA manufacturière, dépenses publiques, IDE
  nets entrants, taux de change effectif réel
- Modèles : ARDL (Autoregressive Distributed Lag) avec test des bornes pour la
  cointégration, et NARDL (Non-linear ARDL) pour l'asymétrie
- Langage : Python (statsmodels)

## Structure du projet

- `run_full_analysis.py` : script principal, exécute l'ensemble de la chaîne de
  traitement (chargement, tests, estimation, diagnostics).
- `src/` : modules sources pour chaque étape de l'analyse :
  - `data_loader.py` : chargement et prétraitement des données
  - `stationarity.py` : tests de racine unitaire (ADF, PP, KPSS)
  - `multicolinearity.py` : diagnostic de multicolinéarité (VIF, condition number)
  - `ardl_model_parsimonious.py` : sélection du modèle ARDL par BIC sous
    contrainte de degrés de liberté, estimation HAC, test des bornes
  - `coeff_long_terme.py` : calcul et validation des coefficients de long terme
  - `nardl_model.py` : modèle NARDL et test de Wald d'asymétrie
  - `diagnostics.py` : tests de validation (autocorrélation, hétéroscédasticité,
    normalité, RESET, CUSUM)
  - `autocorrelation_justification.py` : synthèse du diagnostic d'autocorrélation
    pour le rapport
  - `robustness_checks.py` : chapitre 3bis, premier tour de tests de robustesse
    (puissance du bounds test en petit échantillon, bootstrap résiduel, test de
    rupture 2005, correction de la formule de Wald du NARDL, causalité de
    Granger bivariée, terme quadratique)
  - `robustness_checks_v2.py` : chapitre 3ter, vérifications de second ordre
    répondant à six objections sur le chapitre 3bis : (1) correction pour tests
    multiples (Bonferroni/Holm/FDR) sur la famille des nouveaux tests, (2)
    puissance simulée du test de Wald d'asymétrie NARDL et VIF individuels
    détaillés, (3) recherche systématique de la date de rupture structurelle
    (test sup-F de type Quandt-Andrews) plutôt qu'une date choisie par
    inspection visuelle, (4) causalité de Granger avec les variables de
    contrôle du modèle principal (DEP, IDE, TCER), (5) test de redondance
    entre le terme quadratique et la décomposition NARDL positive/négative,
    (6) VIF recalculés sur séries stationnaires (différences premières)
- `data/` : données brutes, traitées et externes
- `outputs/` : résultats générés (figures, tables, fichiers texte)
- `rapport_methodologique.md` : rapport détaillé (données, méthodologie,
  résultats, limites)
- `requirements.txt` : dépendances Python

## Installation et utilisation

### Prérequis

Python 3.9+ et les dépendances du projet :

```bash
pip install -r requirements.txt
```

### Exécution

```bash
python run_full_analysis.py
```

Chaque étape écrit ses résultats dans `outputs/results/`, `outputs/tables/` et
`outputs/figures/` ; le détail de chaque test est disponible dans les fichiers
correspondants (voir `rapport_methodologique.md` pour la table de correspondance).

## Méthodologie

1. **Analyse descriptive** : statistiques sommaires, matrice de corrélation.
2. **Tests de stationnarité** (ADF, PP, KPSS) : détermination de l'ordre
   d'intégration de chaque variable, préalable indispensable à la spécification
   ARDL.
3. **Sélection des variables explicatives** : le criblage par VIF exclut du
   modèle explicatif les variables structurellement colinéaires avec la variable
   d'intérêt ou avec la tendance de long terme du PIB (`ln_credit`, `ln_ipc` —
   justification détaillée dans `config.py` et le rapport méthodologique). Les
   quatre variables retenues (VA manufacturière, dépenses publiques, IDE, TCER)
   affichent un VIF maximal de 1,89, largement sous le seuil de vigilance usuel
   (5-10).
4. **Sélection des retards ARDL** : recherche sur grille (p, q) avec sélection
   par critère BIC — préféré à l'AIC dans ce contexte, l'AIC étant connu pour
   favoriser des spécifications trop riches lorsque n est petit — sous une
   contrainte explicite de degrés de liberté résiduels minimum, afin de garantir
   un modèle identifiable de façon fiable.
5. **Estimation ARDL** avec erreurs-types robustes HAC (Newey-West).
6. **Test des bornes** (bounds test, Pesaran, Shin & Smith, 2001) pour statuer
   sur l'existence d'une relation de cointégration, calculé explicitement à
   partir de la représentation ECM du modèle retenu.
7. **Coefficients de long terme** : calculés à partir de la représentation ECM,
   avec un contrôle de stabilité numérique du dénominateur (voir section
   Limites).
8. **Modèle NARDL** : décomposition de la VA manufacturière en composantes
   positive/négative cumulées et test de Wald d'asymétrie de long terme.
9. **Diagnostics** : autocorrélation (Breusch-Godfrey), hétéroscédasticité
   (Breusch-Pagan, White), normalité (Jarque-Bera), spécification (RESET),
   stabilité des paramètres (CUSUM/CUSUMSQ).

## Résultats principaux

- Modèle retenu : ARDL(2,0), 7 paramètres estimés pour 33 observations utiles
  (26 degrés de liberté résiduels).
- Diagnostics : pas d'autocorrélation résiduelle détectée (Breusch-Godfrey,
  p=0,74), pas d'hétéroscédasticité (Breusch-Pagan, p=0,25), spécification non
  rejetée (RESET, p=0,94). La normalité des résidus est rejetée (Jarque-Bera,
  p<0,01) et le CUSUM signale une instabilité partielle des paramètres — deux
  points discutés en section Limites.
- Le test des bornes ne rejette pas H0 : la statistique F (1,76) reste
  sous la borne inférieure I(0) à tous les seuils usuels (10 %, 5 %, 1 %).
  L'existence d'une relation de cointégration entre le PIB par habitant et les
  variables explicatives n'est donc **pas établie** par ces données.
- Les coefficients de long terme dérivés de l'ECM affichent un dénominateur
  quasi nul (racine proche de l'unité dans la dynamique de la variable
  dépendante) : leur interprétation comme élasticités de long terme n'est pas
  fiable, cohérence directe avec l'absence de cointégration détectée par le
  test des bornes.
- Le test de Wald détecte une asymétrie significative de l'effet de la VA
  manufacturière sur le PIB par habitant (différence des sommes de coefficients
  positive/négative = 0,88, t=3,45, p=0,002) : les phases de hausse et de baisse
  de la VA manufacturière n'ont pas un effet symétrique sur le PIB par habitant
  (détail dans `outputs/results/nardl_results.txt`).

Le détail complet des sorties, coefficients et p-values figure dans
`outputs/results/` et dans `rapport_methodologique.md`.

## Vérifications de second ordre (chapitre 3ter)

Une relecture critique du chapitre 3bis a mis au jour plusieurs fragilités dans
les tests de robustesse eux-mêmes. Le script `src/robustness_checks_v2.py`
(résultats dans `outputs/results/v2_*.txt`) documente ces vérifications :

- **Correction pour tests multiples** : sous correction FDR, la causalité
  inversée PIB→IVA détectée au chapitre 3bis (p brut=0,032) devient non
  significative (p=0,064).
- **Puissance du test d'asymétrie NARDL** : ~94-97 %, contrairement au bounds
  test (36 %) — mais colinéarité plus étendue que rapporté (VIF de y_L1=90,
  y_L2=51, en plus de iva_pos/iva_neg).
- **Date de rupture structurelle** : une recherche systématique sur 24 dates
  candidates situe la date optimale en 1997, pas en 2005 (2005 se classe 15e
  sur 24). L'attribution au Plan Émergence n'est pas établie.
- **Granger avec variables de contrôle** : la causalité inversée PIB→IVA
  disparaît complètement (p passe de 0,032 à 0,566) une fois contrôlée pour
  les dépenses publiques, l'IDE et le TCER — cohérent avec un artefact de
  variable omise.
- **Redondance NARDL / terme quadratique** : corrélation de -0,61 entre les
  deux constructions ; probablement un seul phénomène compté deux fois.
- **VIF sur séries stationnaires** : conclusion inchangée (VIF < 2 dans les
  deux cas).

**Conséquence principale** : le résultat présenté au chapitre 3bis comme « la
limite la plus substantielle de l'étude » (la causalité inversée) ne résiste à
aucune des deux vérifications indépendantes menées au chapitre 3ter et doit
être retiré de ce statut. Le sens de causalité entre industrialisation et
niveau de vie reste, à l'issue de ce travail, une question ouverte plutôt
qu'une limite documentée dans un sens précis.

## Limites et portée de l'étude

Avec n=35 observations annuelles, la puissance statistique disponible pour une
spécification ARDL à plusieurs régresseurs reste structurellement limitée. Deux
conséquences en découlent, assumées et documentées plutôt que masquées :

- Le test des bornes ne permet pas de conclure à une cointégration : les
  résultats de long terme doivent être lus comme indicatifs et non comme des
  élasticités estimées avec précision.
- Le rejet de la normalité des résidus et l'instabilité partielle détectée par
  le CUSUM appellent à la prudence sur la stabilité temporelle de la relation
  estimée.

L'étude doit donc être positionnée comme une analyse **exploratoire** de la
dynamique court terme entre industrialisation et développement économique au
Maroc, et non comme une preuve d'une relation de long terme stable. Ce
positionnement, explicite dans le rapport, est ce qui rend les résultats
défendables : chaque diagnostic est rapporté avec sa conclusion réelle, y
compris quand elle est négative.

## Auteur

Projet réalisé dans le cadre d'une étude sur l'économie du Maroc.
