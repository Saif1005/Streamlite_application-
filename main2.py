import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.write('''
# Application avancée pour la classification des fleurs d'Iris
Cette application compare plusieurs modèles pour prédire la catégorie des fleurs d'Iris.
''')

st.sidebar.header("Les paramètres d'entrée")

def user_input():
    sepal_length = st.sidebar.slider('Longueur du Sepal', 4.3, 7.9, 5.3)
    sepal_width = st.sidebar.slider('Largeur du Sepal', 2.0, 4.4, 3.3)
    petal_length = st.sidebar.slider('Longueur du Petal', 1.0, 6.9, 2.3)
    petal_width = st.sidebar.slider('Largeur du Petal', 0.1, 2.5, 1.3)
    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    return pd.DataFrame(data, index=[0])

# Récupérer les données d'entrée de l'utilisateur
df = user_input()

st.subheader('Caractéristiques de la fleur sélectionnée')
st.write(df)

# Charger et préparer les données Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Normaliser les données d'entraînement
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Normaliser les données de l'utilisateur
df_scaled = scaler.transform(df)

# Initialiser les modèles
models = {
    'Random Forest': RandomForestClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(),
    'Logistic Regression': LogisticRegression()
}

# Entraîner les modèles et évaluer chaque modèle
performances = {}
for model_name, model in models.items():
    model.fit(X_scaled, y)
    prediction = model.predict(df_scaled)  # Utiliser df_scaled pour les données normalisées
    accuracy = model.score(X_scaled, y)
    performances[model_name] = (prediction, accuracy)

st.subheader("Prédictions et précision des modèles")
for model_name, (prediction, accuracy) in performances.items():
    st.write(f"{model_name} - Catégorie prédite : {iris.target_names[prediction][0]}, Précision : {accuracy:.2f}")

# Utiliser le modèle avec la meilleure précision pour la prédiction finale
best_model_name = max(performances, key=lambda k: performances[k][1])
best_prediction, best_accuracy = performances[best_model_name]

st.subheader(f"Le modèle avec la meilleure précision est : {best_model_name}")
st.write(f"Catégorie prédite : {iris.target_names[best_prediction][0]} avec une précision de {best_accuracy:.2f}")
