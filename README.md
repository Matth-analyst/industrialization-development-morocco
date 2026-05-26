# Économétrie de l'Industrialisation et du Développement Économique au Maroc (1990-2024)

Ce projet propose une étude économétrique approfondie de la relation entre l'industrialisation et le développement économique au Maroc sur la période allant de 1990 à 2024. L'analyse s'appuie sur des modèles de cointégration avancés pour capter les dynamiques à court et à long terme.

## 📊 Aperçu du Projet

L'objectif principal est d'analyser l'impact de la valeur ajoutée industrielle, du capital humain et de l'ouverture commerciale sur la croissance du Produit Intérieur Brut (PIB) au Maroc. L'étude utilise 35 observations annuelles et intègre des tests de stabilité et de diagnostic rigoureux.

### Points Clés :
- **Période** : 1990 - 2024.
- **Modèles** : ARDL (Autoregressive Distributed Lag) et NARDL (Non-linear ARDL) pour l'analyse des asymétries.
- **Langage** : Python.

## 🛠️ Structure du Projet

Le projet est organisé de la manière suivante :

- `run_full_analysis.py` : Script principal pour exécuter l'ensemble de la chaîne de traitement (chargement, tests, estimation, diagnostics).
- `src/` : Contient les modules sources pour chaque étape de l'analyse :
    - `data_loader.py` : Chargement et prétraitement des données.
    - `stationarity.py` : Tests de racine unitaire (ADF, PP, KPSS).
    - `ardl_model_parsimonious.py` : Estimation du modèle ARDL optimal.
    - `nardl_model.py` : Analyse des effets asymétriques.
    - `diagnostics.py` : Tests de validation (autocorrélation, hétéroscédasticité, normalité, CUSUM).
- `data/` : Répertoires pour les données brutes, traitées et externes.
- `requirements.txt` : Liste des dépendances Python nécessaires.

## 🚀 Installation et Utilisation

### Prérequis

Assurez-vous d'avoir Python 3.9+ installé. Vous pouvez installer les dépendances via pip :

```bash
pip install -r requirements.txt
```

### Exécution

Pour lancer l'analyse complète, exécutez le script à la racine du projet :

```bash
python run_full_analysis.py
```

## 📈 Méthodologie

L'étude suit une approche structurée :
1. **Analyse Descriptive** : Statistiques sommaires et corrélations.
2. **Tests de Stationnarité** : Vérification de l'ordre d'intégration des variables (I(0) ou I(1)).
3. **Estimation ARDL** : Sélection du modèle optimal par le critère AIC/BIC et test des bornes (Bounds Test) pour la cointégration.
4. **Analyse de Long Terme** : Estimation des coefficients d'équilibre.
5. **Modèle NARDL** : Test de l'asymétrie de l'impact de l'industrialisation.
6. **Diagnostics** : Validation statistique de la robustesse des résultats.

## 📝 Auteur

Projet réalisé dans le cadre d'une étude sur l'économie du Maroc.
