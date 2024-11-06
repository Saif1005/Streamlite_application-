# Classification des Fleurs d'Iris avec plusieurs modèles

## Description

Ce projet utilise l'application Streamlit pour comparer plusieurs modèles de machine learning afin de prédire la catégorie des fleurs d'Iris. Les modèles inclus sont :

- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Régression Logistique

L'application prend en entrée les caractéristiques d'une fleur d'Iris (longueur et largeur du sepale et du pétale) et prédit la catégorie de la fleur à l'aide des différents modèles de classification. Elle affiche également la précision de chaque modèle, et indique le modèle ayant la meilleure performance.

## Fonctionnalités

- Interface simple via Streamlit pour l'entrée des caractéristiques d'une fleur d'Iris.
- Comparaison des performances de quatre modèles de machine learning populaires.
- Affichage de la catégorie prédite et de la précision de chaque modèle.
- Sélection automatique du modèle avec la meilleure précision.

## Prérequis

- Python 3.x
- Bibliothèques Python : `streamlit`, `pandas`, `scikit-learn`, `matplotlib`

Vous pouvez installer ces bibliothèques avec la commande suivante :
```bash
pip install streamlit pandas scikit-learn matplotlib
