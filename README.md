# Implémentation d'un Modèle de Scoring pour "Prêt à Dépenser"

## Contexte

"Prêt à Dépenser" est une société financière spécialisée dans les crédits à la consommation pour des personnes ayant peu ou pas d'historique de prêt. L'entreprise souhaite développer un outil de scoring crédit pour calculer la probabilité qu’un client rembourse son crédit, en s'appuyant sur des données variées (comportementales, institutionnelles, etc.). En réponse à la demande croissante de transparence, un dashboard interactif sera créé pour expliquer les décisions d’octroi de crédit.

## Objectif du Projet

- Construire un modèle de scoring pour prédire la probabilité de défaut de paiement.
- Développer un dashboard interactif pour interpréter les prédictions et améliorer la transparence.
- Mettre en production le modèle de scoring via une API et le dashboard interactif.

## Étapes du Projet

### 1. Modélisation

- **Données** : Utilisation de [données](https://www.kaggle.com/c/home-credit-default-risk/data) variées, nécessitant la jointure de plusieurs tables.
- **Approche** : Sélection et adaptation d'un [kernel Kaggle](https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features/script) pour l'analyse exploratoire et le feature engineering.
- **Modèles Testés** : 
  - DummyClassifier
  - RandomForestClassifier
  - LogisticRegression
  - XGBClassifier
  - LGBMClassifier
- **Optimisation** : Utilisation de GridSearchCV et RandomSearchCV pour optimiser les hyperparamètres, avec suivi via MLFlow.
- **Seuil de Décision** : Ajustement du seuil pour minimiser le coût métier, en tenant compte du déséquilibre entre faux positifs et faux négatifs.

### 2. API

- **Déploiement** : Sérialisation du modèle optimisé et déploiement via une API sur une plateforme Cloud (Heroku).
- **Documentation** : API documentée pour faciliter l'intégration.

### 3. Dashboard

- **Technologie** : Développement avec Streamlit.
- **Fonctionnalités** : 
  - Visualisation du score et interprétation pour chaque client.
  - Comparaison des informations client avec l'ensemble des clients.
- **Accès** : Dashboard déployé sur le Cloud pour un accès facile.

### 4. Analyse de Datadrift

- **Outil** : Utilisation de la bibliothèque evidently pour détecter le data drift entre les données d'entraînement et de production.
- **Rapport** : Création d'un tableau HTML d'analyse du data drift.

### 5. Packages Utilisés

- **Modélisation** : [mlflow_model/requirements.txt](#)
- **API** : [requirements.txt](#)
- **Dashboard** : [Streamlit/requirements.txt](#)

### 6. Tests Unitaires

- **Pipeline de Déploiement** : Tests automatisés via GitHub Actions pour chaque push sur GitHub.
- **Code de Test** : [test_app.py](#)

## Résultats des Livrables

- [API](#)
- [Dashboard](#)

## Livrables

- [Pham-Van_Yann_2_dossier_code_102023](https://github.com/Bruce2Cluny191/Projet7-Implementez_un_modele_de_scoring/tree/main/Pham-Van_Yann_2_dossier_code_102023) : Un dossier contenant le code de la modélisation, le dashboard, et l'API.
- [Pham-Van_Yann_2_dossier_code_102023/data_drift_report.html](https://github.com/Bruce2Cluny191/Projet7-Implementez_un_modele_de_scoring/blob/main/Pham-Van_Yann_2_dossier_code_102023/data_drift_report.html) : Un tableau HTML d’analyse de data drift.
- [Pham-Van_Yann_3_note_methodologique_102023.pdf](https://github.com/Bruce2Cluny191/Projet7-Implementez_un_modele_de_scoring/blob/main/Pham-Van_Yann_3_note_methodologique_102023.pdf) : Une note méthodologique décrivant la démarche de modélisation, le traitement du déséquilibre des classes, et l'analyse du data drift.
- [Pham-Van_Yann_4_presentation_102023.pdf](https://github.com/Bruce2Cluny191/Projet7-Implementez_un_modele_de_scoring/blob/main/Pham-Van_Yann_4_presentation_102023.pdf) : Un support de présentation pour la soutenance.

## Contact

Pour toute question, veuillez me contacter sur [LinkedIn](https://www.linkedin.com/in/chasseur2valeurs/).
