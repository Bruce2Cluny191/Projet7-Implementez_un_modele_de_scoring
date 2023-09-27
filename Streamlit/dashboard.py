import pandas as pd
import streamlit as st
import requests
import matplotlib.pyplot as plt


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

    st.title("Dashboard Scoring Crédit")

    data = (
        echantillon_clients.loc[echantillon_clients["SK_ID_CURR"] == client_choice]
        .drop(columns={"SK_ID_CURR"})
        .to_dict(orient="records")
    )

    pred = request_prediction(MLFLOW_URI, data)
    prediction = pd.DataFrame.from_dict(pred)
    prediction_1 = prediction["predictions"][0][1]

    if prediction_1 < seuil:
        st.header("Client {} : Crédit :green[accepté]".format(client_choice))

    else:
        st.header("Client {} : Crédit :red[refusé]".format(client_choice))

    st.write("Risque de défaut = {:.2f} %".format(prediction_1 * 100))

    # Positionnement du client sur une jauge
    fig, ax = plt.subplots(figsize=(9, 2))
    plt.yticks([])
    plt.grid(visible=False)
    plt.box(on=False)
    plt.barh(width=100, y=0, color="red")
    plt.barh(width=seuil * 100, y=0, color="green")
    plt.axvline(x=prediction_1 * 100, color="black", linewidth=4, linestyle="--")
    st.pyplot(fig)


if __name__ == "__main__":
    main()
