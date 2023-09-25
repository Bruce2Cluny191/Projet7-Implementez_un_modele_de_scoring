import pandas as pd
import streamlit as st
import requests


def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = data.iloc[0].to_dict()
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
    echantillon_clients = pd.read_csv("echantillon_clients.csv")

    MLFLOW_URI = "http://127.0.0.1:5000/invocations"

    client_choice = st.sidebar.selectbox(
        "Quel client souhaitez-vous évaluer ?", echantillon_clients.index.tolist()
    )

    st.title("Prédiction de faillite d'un client")

    # occupation_moy = st.number_input('Occupation moyenne de la maison (en nombre d\'habitants)',
    #                                 min_value=0., value=3., step=1.)

    predict_btn = st.button("Prédire")
    if predict_btn:
        data = echantillon_clients[echantillon_clients.index == client_choice]

        pred = request_prediction(MLFLOW_URI, data)[0]

        st.write("La probabilité de faillite du client est de {:.2f}".format(pred))


if __name__ == "__main__":
    main()
