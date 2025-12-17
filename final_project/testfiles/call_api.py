import requests
import json
import pickle

from typing import Dict, Any

import pandas as pd
import numpy as np


ENTRY_POINT = "http://0.0.0.0:8000"


def call_model_versions():

    model_versions_request = requests.get(f"{ENTRY_POINT}/model_versions")
    model_versions_json: str = model_versions_request.content.decode("utf-8")
    model_versions: Dict[str, Any] = json.loads(model_versions_json)

    return model_versions


def call_train():

    train_request = requests.get(f"{ENTRY_POINT}/train")
    model_record_json: str = train_request.content.decode("utf-8")
    model_record: Dict[str, Any] = json.loads(model_record_json)

    return model_record


def call_predict(model_id: str, df: pd.DataFrame):

    payload = df.to_dict(orient="split")
    prediction_request = requests.post(f"{ENTRY_POINT}/predict/{model_id}", json=payload)
    prediction_json: str = prediction_request.content.decode("utf-8")
    prediction: Dict[str, Any] = json.loads(prediction_json)

    return prediction


def main():

    record = call_train()
    model_id = record["model_id"]

    with open("unprocessed_dataset.pkl", "rb") as f:
        df: pd.DataFrame = pickle.load(f)

    prediction_unshuffled = call_predict(model_id, df)["model_performance"]

    # Randomize the Churn column
    df["Churn"] = np.random.choice([0, 1], size=len(df))

    prediction_shuffled = call_predict(model_id, df)["model_performance"]

    print(prediction_unshuffled)
    print(prediction_shuffled)


if __name__ == "__main__":
    main()
