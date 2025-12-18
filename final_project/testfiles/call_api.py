import requests
import json
import pickle

from typing import List, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ENTRY_POINT = "http://0.0.0.0:8000"


def call_model_versions():
    """
    Calls the "model_versions" endpoint of the API.

    :return: The decoded endpoint response.
    """

    model_versions_request = requests.get(f"{ENTRY_POINT}/model_versions")
    model_versions_json: str = model_versions_request.content.decode("utf-8")
    model_versions: List[Dict[str, Any]] = json.loads(model_versions_json)

    return model_versions


def call_train():
    """
    Calls the "train" endpoint of the API.

    :return: The decoded endpoint response.
    """

    train_request = requests.get(f"{ENTRY_POINT}/train")
    model_record_json: str = train_request.content.decode("utf-8")
    model_record: Dict[str, Any] = json.loads(model_record_json)

    return model_record


def call_predict(model_id: str, df: pd.DataFrame):
    """
    Calls the "predict" endpoint of the API.

    :param model_id: The ID of the model to be used.
    :param df: The dataframe on which the model should be run.
    :return: The decoded endpoint response.
    """

    payload = df.to_dict(orient="split")
    prediction_request = requests.post(f"{ENTRY_POINT}/predict/{model_id}", json=payload)
    prediction_json: str = prediction_request.content.decode("utf-8")
    prediction: Dict[str, Any] = json.loads(prediction_json)

    return prediction


def call_extend(df: pd.DataFrame):
    """
    Calls the "extend_database" endpoint of the API.

    :param df: The dataframe to be newly added to the database.
    :return: The decoded endpoint response.
    """

    payload = {
        "new_rows": df.to_dict(orient="split"),
        "train_test_flag": "train"
    }
    extension_request = requests.post(f"{ENTRY_POINT}/extend_database", json=payload)
    extension_json: str = extension_request.content.decode("utf-8")
    extension_result: Dict[str, Any] = json.loads(extension_json)

    return extension_result


def create_mock_data(df: pd.DataFrame, size: int) -> pd.DataFrame:
    """
    Creates additional rows we can extend the database with.
    Rows are selected from the given dataframe randomly.
    The Churn column is artificially filled according to the tenure column values.
    Specifically, if the tenure in the selected rows is greater than the median tenure
    in the full dataframe, then a "Yes" is written into the Churn column.
    Otherwise, a "No" is placed.
    This creates an artificial dependency of the Churn column values from the tenure column values.

    :param df: The dataframe from which the random rows will be selected.
    :param size: The number of random rows.
    :return: The new rows with which we can extend the remote database.
    """

    selected_rows = np.random.choice(len(df), size=size, replace=False)
    new_rows = df.iloc[selected_rows].reset_index(drop=True).copy()
    tenure_median = df["tenure"].median()
    new_churn_values = (new_rows["tenure"] > tenure_median).map(lambda x: "Yes" if x else "No")
    new_rows["Churn"] = new_churn_values

    return new_rows


def main():
    """
    Tests the remote random forest training API.
    It calls for model availability, model training, model prediction and database extensions.
    During this, the test performance of the newly trained models should slowly degrade, since
    we "poison" the remote database with these newly added artificial rows.
    Nevertheless, the train performance should be better and better, since the logic behind
    the Churn value generation is simple (it is based on the tenure column), which can be
    easily learned with the random forest algorithm.
    """

    # Check for model existence.
    # If it does not exist, train a new one.
    model_versions = call_model_versions()
    if len(model_versions) == 0:
        record = call_train()
        model_id = record["model_id"]
    else:
        model_id = model_versions[-1]["model_id"]

    with open("unprocessed_dataset.pkl", "rb") as f:
        df: pd.DataFrame = pickle.load(f)

    performances = list()
    fig, ax = plt.subplots(1, 5)
    fig.set_size_inches(20, 5)
    fig.subplots_adjust(wspace=0.5)

    for idx in range(20):

        # Generate artificial new rows
        new_rows = create_mock_data(df, 600)

        # Extend the dataframe and the remote database with the new rows
        df = pd.concat([df, new_rows], axis=0).reset_index(drop=True)
        call_extend(new_rows)

        # Get the performance of the NEW remote model on the extended dataframe
        old_performance = call_predict(model_id, df)["model_performance"]

        # Retrain the model on the extended database
        record = call_train()
        model_id = record["model_id"]

        # Get the performance of the NEW remote model on the extended dataframe
        new_performance = call_predict(model_id, df)["model_performance"]

        # Print performance change on the extended dataframe before and after retraining
        print(f"MCC change: {old_performance['mcc']:.3%} -> {new_performance['mcc']:.3%}")

        # Record the MCC values
        performances.append(new_performance)

        # Plot the performances
        # At first, it should degrade due to the "database poisoning" of the new rows.
        # Then, it should rebound, since the new logic in the new rows is simple to learn,
        # and they start to dominate over the original data.

        for axis in ax:
            axis.cla()
            axis.set_xlabel("Number of retrain turns")

        ax[0].plot([p["mcc"] for p in performances])
        ax[0].set_ylabel("Mathew's Correlation Coefficient")

        ax[1].plot([p["acc"] for p in performances])
        ax[1].set_ylabel("Accuracy")

        ax[2].plot([p["f1"] for p in performances])
        ax[2].set_ylabel("F1-score")

        ax[3].plot([p["precision"] for p in performances])
        ax[3].set_ylabel("Precision")

        ax[4].plot([p["recall"] for p in performances])
        ax[4].set_ylabel("Recall")

        plt.pause(0.1)

    plt.show()


if __name__ == "__main__":
    main()
