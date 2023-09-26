import pandas as pd
import streamlit as st
import json
import requests


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
    echantillon_clients = pd.read_csv("echantillon_clients.csv")
    seuil = echantillon_clients.iloc[0]["threshold"]
    echantillon_clients = echantillon_clients.drop(columns={"threshold"})

    MLFLOW_URI = "http://127.0.0.1:5000/invocations"

    client_choice = st.sidebar.selectbox(
        "Quel client souhaitez-vous évaluer ?",
        echantillon_clients["SK_ID_CURR"].tolist(),
    )

    st.title("Prédiction de faillite d'un client")

    # occupation_moy = st.number_input('Occupation moyenne de la maison (en nombre d\'habitants)',
    #                                 min_value=0., value=3., step=1.)

    predict_btn = st.button("Prédire")
    if predict_btn:
        data = (
            echantillon_clients.loc[echantillon_clients["SK_ID_CURR"] == client_choice]
            .drop(columns={"SK_ID_CURR"})
            .to_dict(orient="records")
        )

        pred = request_prediction(MLFLOW_URI, data)
        prediction = pd.DataFrame.from_dict(pred)
        prediction_1 = prediction["predictions"][0][1]

        st.write(
            "La probabilité de faillite du client est de {:.2f} %".format(
                prediction_1 * 100
            )
        )


if __name__ == "__main__":
    main()
