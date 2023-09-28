import pandas as pd
import streamlit as st
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap


def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = {"dataframe_records": data}

    response = requests.request(
        method="POST", headers=headers, url=model_uri, json=data_json
    )

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(
                response.status_code, response.text
            )
        )

    return response.json()


def main():
    def info_client(ID):
        """Isole la ligne du client voulu"""
        data_client = echantillon_clients.loc[echantillon_clients.index == ID]
        return data_client

    def chargement_model():
        """Chargement du modèle entraîné sur tous les clients avec target"""
        mod_pickle = open(
            "C:/Users/Yann/Documents/Projet7-Implementez_un_modele_de_scoring/mlflow_model/model.pkl",
            "rb",
        )
        perfect_model = pickle.load(mod_pickle)
        return perfect_model

    echantillon_clients = pd.read_csv(
        "data/echantillon_clients.csv", index_col="SK_ID_CURR"
    )
    trainset = pd.read_csv("data/trainset.csv")
    trainset_0 = trainset.loc[trainset["TARGET"] == 0].drop(columns={"TARGET"})
    trainset_1 = trainset.loc[trainset["TARGET"] == 1].drop(columns={"TARGET"})
    seuil = echantillon_clients.iloc[0]["threshold"]
    echantillon_clients = echantillon_clients.drop(columns={"threshold"})

    MLFLOW_URI = "http://127.0.0.1:5000/invocations"

    client_choice = st.sidebar.selectbox(
        "Quel client souhaitez-vous évaluer ?", echantillon_clients.index
    )

    data_client = info_client(client_choice)

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

    st.title("Dashboard Scoring Crédit")

    data = data_client.to_dict(orient="records")

    pred = request_prediction(MLFLOW_URI, data)
    prediction = pd.DataFrame.from_dict(pred)
    prediction_1 = prediction["predictions"][0][1]

    if prediction_1 < seuil:
        st.header("Client :blue[{}] : Crédit :green[accepté]".format(client_choice))
        st.write(
            "Risque de défaut = :green[{:.1f} %] - Seuil de décision = :orange[{:.1f} %]".format(
                prediction_1 * 100, seuil * 100
            )
        )

    else:
        st.header("Client :blue[{}] : Crédit :red[refusé]".format(client_choice))
        st.write(
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

    # Feature Importance locale du client et globale de l'ensemble des clients
    if st.checkbox(
        "Visualiser les principales caractéristiques pour le score du client :blue[{}]".format(
            client_choice
        )
    ):
        perfect_model = chargement_model()
        f = lambda x: perfect_model.predict_proba(x)[:, 1]
        med = echantillon_clients.mean().values.reshape(
            (1, echantillon_clients.shape[1])
        )
        fig, ax = plt.subplots(figsize=(6, 6))
        explainer = shap.Explainer(f, med)
        shap_values = explainer(data_client, max_evals=1595)
        shap.plots.waterfall(shap_values[0], max_display=10)
        st.pyplot(fig)

    # Distribution des features selon les classes avec positionnement du client
    if st.checkbox(
        "Situer le client :blue[{}] dans la distribution des caractéristiques".format(
            client_choice
        )
    ):
        caracteristique = st.selectbox(
            "Quelle caractéristique souhaitez-vous observer ?", trainset_0.columns
        )

        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(121)
        ax.hist(trainset_0[caracteristique], color="green", bins=20)
        ax.axvline(
            data_client[caracteristique].values[0],
            color="black",
            linewidth=4,
            linestyle="--",
        )
        ax.set(title="Crédits acceptés")
        ax = fig.add_subplot(122)
        ax.hist(trainset_1[caracteristique], color="red", bins=20)
        ax.axvline(
            data_client[caracteristique].values[0],
            color="black",
            linewidth=4,
            linestyle="--",
        )
        ax.set(title="Crédits refusés")
        st.pyplot(fig)

        st.write(
            "Valeur de la caractéristique pour le client :blue[{}] = :orange[{}]".format(
                client_choice, data_client[caracteristique].values[0]
            )
        )


if __name__ == "__main__":
    main()
