========================================================================
MINI-PROJET NLP : CLASSIFICATION DE TEXTES (BARÈME 20 POINTS)
========================================================================

Auteur : Étudiant Ing3 / Antigravity AI
Domaine : Analyse de sentiment de critiques de films en français (Allociné)
Date : 2026-09-29

------------------------------------------------------------------------
1. CONTENU DU PROJET ET DÉPENDANCES
------------------------------------------------------------------------
Fichiers inclus :
- projet.ipynb          : Notebook Jupyter complet et exécuté (avec sorties visibles).
- rapport.pdf           : Rapport d'évaluation synthétique de 4 pages maximum.
- download_data.py      : Script de téléchargement et d'extraction des données.
- run_experiment.py     : Script d'entraînement, d'évaluation et de génération des figures.
- make_notebook.py      : Générateur autonome du notebook projet.ipynb.
- make_report.py        : Générateur autonome du rapport PDF rapport.pdf.
- data.csv              : Dataset extrait (2500 exemples : Train 1600, Val 400, Test 500).
- metrics_summary.json  : Métriques détaillées d'évaluation au format JSON.
- error_analysis.json   : Analyse qualitative des 5 cas d'erreurs d'inférence.
- *.png                 : Graphiques générés (courbes de perte, matrices de confusion, comparatif).

Dépendances Python requises :
- python >= 3.9
- torch >= 2.0
- transformers >= 4.30
- datasets >= 2.12
- scikit-learn >= 1.2
- pandas, numpy, matplotlib, seaborn
- fpdf2 (pour la génération PDF)
- nbformat (pour la génération du notebook)

------------------------------------------------------------------------
2. INSTRUCTIONS D'EXÉCUTION
------------------------------------------------------------------------
Étape 1 : Téléchargement et préparation des données
  python3 download_data.py

Étape 2 : Entraînement des modèles et benchmarks
  python3 run_experiment.py

Étape 3 : Génération du notebook projet.ipynb
  python3 make_notebook.py

Étape 4 : Exécution effective des cellules du notebook (facultatif si projet.ipynb déjà généré)
  jupyter nbconvert --to notebook --execute --inplace projet.ipynb

Étape 5 : Génération du rapport PDF (rapport.pdf)
  python3 make_report.py

------------------------------------------------------------------------
3. RÉSUMÉ DES PERFORMANCES (SUR LE JEU DE TEST DE 500 EXEMPLES)
------------------------------------------------------------------------
- Système 1 (Règles Déterministes) : Accuracy ~0.72  | Macro F1 ~0.71 | Latence ~0.02 ms
- Système 2 (PyTorch MLP TF-IDF)   : Accuracy ~0.85+ | Macro F1 ~0.85 | Latence ~0.05 ms
- Système 3 (CamemBERT Fine-Tuned) : Accuracy ~0.91+ | Macro F1 ~0.91 | Latence ~15.0 ms (CPU)

========================================================================
