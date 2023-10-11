import pandas as pd
import streamlit as st
import requests
import json
import joblib
import shap
import matplotlib.pyplot as plt

def request_prediction(model_uri, client_id):
    """Requête de prédiction envoyée à l'API"""
    # Convertir l'input client au format json
    input = {"client_choice":client_id}
    response = requests.post(url=model_uri, data=json.dumps(input))

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(
                response.status_code, response.text
            )
        )
    return response.json()
def info_client(id_client):
    """Isole la ligne du client voulu"""
    data_client = echantillon_clients.loc[echantillon_clients.index == id_client]
    return data_client

# Récupération et préparation des données
echantillon_clients = pd.read_csv("echantillon_clients.csv", index_col="SK_ID_CURR")
seuil = echantillon_clients.iloc[0]["threshold"]
echantillon_clients = echantillon_clients.drop(columns=["threshold"])
trainset = pd.read_csv("trainset.csv")
trainset_0 = trainset.loc[trainset["TARGET"] == 0].drop(columns=["TARGET"])
trainset_1 = trainset.loc[trainset["TARGET"] == 1].drop(columns=["TARGET"])
shap_values = joblib.load("explainer.joblib")
list_index = echantillon_clients.index.tolist()

# End-point de l'API
URI = "https://scoring-client-6cc83c15008a.herokuapp.com/invocations"
def main():
    """Programme principal pour sélectionner un client puis afficher les informations :
    - jauge d'acceptation/refus
    - interprétabilité locale
    - interprétabilité globale
    - position du client sur la distribution des features selon le statut des clients historiques"""

    # Choix de l'id client par le chargé de relation client
    client_choice = st.sidebar.selectbox(
        "Quel client souhaitez-vous évaluer ?", echantillon_clients.index
    )

    # Chargement des données du client choisi
    data_client = info_client(client_choice)

    # SIDEBAR
    st.sidebar.write("Client :blue[{}]".format(client_choice))
    st.sidebar.write(
        "Âge : :orange[{}] ans".format(int(-data_client["DAYS_BIRTH"] / 365))
    )
    st.sidebar.write(
        "Nombre d'enfant(s) : :orange[{}]".format(
            int(data_client["CNT_CHILDREN"].values[0])
        )
    )
    st.sidebar.write(
        "Revenu total : :orange[{}] $".format(
            int(data_client["AMT_INCOME_TOTAL"].values[0])
        )
    )
    st.sidebar.write(
        "Ancienneté dans l'emploi : :orange[{}] an(s)".format(
            int(-data_client["DAYS_EMPLOYED"] / 365)
        )
    )
    st.sidebar.write(
        "Crédit sollicité : :orange[{}] $".format(
            int(data_client["AMT_CREDIT"].values[0])
        )
    )
    st.sidebar.write(
        "Annuité du prêt : :orange[{}] $".format(
            int(data_client["AMT_ANNUITY"].values[0])
        )
    )
    st.sidebar.write(
        "Prix du bien : :orange[{}] $".format(
            int(data_client["AMT_GOODS_PRICE"].values[0])
        )
    )

    # PAGE PRINCIPALE
    st.title("Dashboard Scoring Crédit")

    # Récupération de la probabilité
    prediction_1 = request_prediction(URI, client_choice)

    if prediction_1 < seuil:
        st.header("Client :blue[{}] : Crédit :green[accepté]".format(client_choice))
        st.subheader(
            "Risque de défaut = :green[{:.1f} %] - Seuil de décision = :orange[{:.1f} %]".format(
                prediction_1 * 100, seuil * 100
            )
        )

    else:
        st.header("Client :blue[{}] : Crédit :red[refusé]".format(client_choice))
        st.subheader(
            "Risque de défaut = :red[{:.1f} %] - Seuil de décision = :orange[{:.1f} %]".format(
                prediction_1 * 100, seuil * 100
            )
        )

    # Positionnement du client sur une jauge
    fig, ax = plt.subplots(figsize=(9, 2))
    plt.yticks([])
    plt.grid(visible=False)
    plt.box(on=False)
    plt.barh(width=100, y=0, color="red")
    plt.barh(width=seuil * 100, y=0, color="green")
    plt.axvline(x=prediction_1 * 100, color="black", linewidth=4, linestyle="--")
    plt.axvline(x=seuil * 100, color="orange", linewidth=4, linestyle="-")
    st.pyplot(fig)

    # Feature Importance locale du client
    if st.checkbox(
        "Visualiser les principales caractéristiques pour le score du client :blue[{}]".format(
            client_choice
        )
    ):
        fig, ax = plt.subplots(figsize=(9, 9))
        shap.plots.waterfall(shap_values[list_index.index(client_choice)], max_display=10)
        st.pyplot(fig)

    # Feature importance globale
    if st.checkbox("Visualiser les caractéristiques les plus importantes au global"):
        fig, ax = plt.subplots(figsize=(9, 9))
        shap.summary_plot(shap_values, max_display=10)
        st.pyplot(fig)

    # Distribution des features selon les classes avec positionnement du client
    if st.checkbox(
        "Situer le client :blue[{}] dans la distribution des caractéristiques selon le comportement des clients".format(
            client_choice
        )
    ):
        caracteristique = st.selectbox(
            "Quelle caractéristique souhaitez-vous observer ?", trainset_0.columns
        )

        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(121)
        ax.hist(trainset_0[caracteristique], color="green", bins=2)
        ax.axvline(
            data_client[caracteristique].values[0],
            color="black",
            linewidth=4,
            linestyle="--",
        )
        ax.set(title="Sans défaut de paiement")
        ax = fig.add_subplot(122)
        ax.hist(trainset_1[caracteristique], color="red", bins=2)
        ax.axvline(
            data_client[caracteristique].values[0],
            color="black",
            linewidth=4,
            linestyle="--",
        )
        ax.set(title="Avec défaut de paiement")
        st.pyplot(fig)

        st.write(
            "Valeur de la caractéristique pour le client :blue[{}] = :orange[{}]".format(
                client_choice, data_client[caracteristique].values[0]
            )
        )

if __name__ == "__main__":
    main()
