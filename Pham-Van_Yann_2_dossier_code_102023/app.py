# 1. Library imports
import pandas as pd
import uvicorn
from fastapi import FastAPI
import pickle

# from pydantic import BaseModel


# Module de prédiction
def predict_proba(data_client):
    return perfect_model.predict_proba(data_client)


# Récupération du modèle
def modele():
    mod_pickle = open(
        "./mlflow_model/model.pkl",
        "rb",
    )
    return pickle.load(mod_pickle)


# Class which describes ID client
# class ID_client(BaseModel):
#     client_choice: int


perfect_model = modele()

echantillon_clients = pd.read_csv(
    "data/echantillon_clients.csv", index_col="SK_ID_CURR"
)
echantillon_clients = echantillon_clients.drop(columns=["threshold"])

app = FastAPI()


@app.post("/invocations")
def operate(ID_client):
    print(ID_client)
    # client_choice = input["client_choice"]
    # proba = predict_proba(
    #     echantillon_clients.loc[echantillon_clients.index == client_choice]
    # )
    return 45


"""

perfect_model = pickle.load(mod_pickle)

# 2.1 Load client_sample & trainset
echantillon_clients = pd.read_csv(
    "data/echantillon_clients.csv", index_col="SK_ID_CURR"
)
trainset = pd.read_csv("data/trainset.csv", index_col="SK_ID_CURR")


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get("/")
def index():
    return {"message": "Hello, World"}


# 4. Class which describes ID client
class ID(BaseModel):
    client_choice: int


# 5. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted flower species with the confidence
@app.post("/invocations")
def predict_proba(id_client: ID):
    id_client = id_client.dict()
    probability = perfect_model.predict_proba(echantillon_clients[id_client])
    return

"""

# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
