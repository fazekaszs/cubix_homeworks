import pickle
import tomllib
import sqlite3
import logging
import json
import warnings

from pathlib import Path
from typing import Dict, Any, Tuple, List
from datetime import datetime

import pandas as pd
import numpy as np

import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier


class RunModelError(Exception):
    pass


class MLModelHandler:


    def __init__(self, config_path: Path, random_seed: int) -> None:

        # Set up logging
        self.logger = logging.getLogger("MLModelHandler")
        self.logger.setLevel(logging.INFO)
        logging_file_handler = logging.FileHandler(
            "MLModelHandler.log", mode="a", encoding="utf-8"
        )
        logging_formatter = logging.Formatter(
            "{asctime} - {levelname} - {message}",
            style="{", datefmt="%Y.%m.%d %H:%M:%S"
        )
        logging_file_handler.setFormatter(logging_formatter)
        self.logger.addHandler(logging_file_handler)

        # Reading the config file
        with open(config_path, "rb") as f:
            self.config_data: Dict[str, Any] = tomllib.load(f)["mlhandler"]

        self.logger.info("Config file successfully read.")

        # Setting up the random number generator for reproducibility
        self.random_seed = random_seed
        self.random_number_generator = np.random.RandomState(seed=self.random_seed)

        # Load the preprocessor config file
        self.preprocessor = self._load_preprocessor_config()

        # Run database initialization
        self.expected_columns = None
        self._initialize_database()


    def _load_preprocessor_config(self) -> Dict[str, Any]:

        preprocessor_file = Path(self.config_data["artifacts_dir"]) / "preprocessing_config.pkl"
        with open(preprocessor_file, "rb") as f:
            preprocessor = pickle.load(f)

        return preprocessor


    def _initialize_database(self) -> None:

        # Check for database existence
        if Path(self.config_data["database"]).exists():
            self.logger.warning("Trying to initialize the database, while it already exists! Skipping...")
            return

        # Load the original dataset from the artifacts directory
        original_dataset_path = Path(self.config_data["artifacts_dir"]) / "preprocessed_dataset.pkl"
        with open(original_dataset_path, "rb") as f:
            original_dataset: pd.DataFrame = pickle.load(f)

        self.logger.info(f"Original dataset loaded with shape: {original_dataset.shape}.")

        # Set the value for the expected columns field
        # Since the Churn column is the target column, we should remove it
        self.expected_columns = original_dataset.columns.drop(["Churn", ]).values

        self.logger.info(f"Column names are: {self.expected_columns}.")

        # Add train/test indicator column
        test_mask = self.random_number_generator.choice(
            len(original_dataset),
            size=int(len(original_dataset) * self.config_data["train_test_split"])
        )
        test_flags = np.zeros(len(original_dataset), dtype=int)
        test_flags[test_mask] = 1
        original_dataset["is_test"] = test_flags

        self.logger.info(f"Added test flag for {len(test_mask)} entries.")

        # Write the DataFrame content to the database
        connection = sqlite3.connect(self.config_data["database"])
        original_dataset.to_sql(name="train", con=connection)
        connection.close()

        self.logger.info(f"Original dataset successfully written to database.")


    def _query_data(self, is_test: bool) -> Tuple[np.ndarray, np.ndarray]:

        # Connect to the database and query
        connection = sqlite3.connect(self.config_data["database"])
        training_data = pd.read_sql(
            f"SELECT * FROM train WHERE is_test = {1 if is_test else 0}",
            connection
        )
        connection.close()

        self.logger.info(f"Queried {len(training_data)} entries from the database ({is_test=}).")

        # Preprocess raw database data for training
        y_train = training_data["Churn"].values
        x_train = training_data.drop(columns=["Churn", "is_test"]).values

        return x_train, y_train


    @staticmethod
    def _evaluate_model(model: RandomForestClassifier, x: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:

        y_prediction = model.predict(x)

        return {
            "mcc": metrics.matthews_corrcoef(y_true, y_prediction),
            "acc": metrics.accuracy_score(y_true, y_prediction),
            "f1": metrics.f1_score(y_true, y_prediction),
            "precision": metrics.precision_score(y_true, y_prediction),
            "recall": metrics.recall_score(y_true, y_prediction),
        }


    def _preprocess_pandas_df(self, model_input: pd.DataFrame):

        # Remove the unnecessary columns
        model_input.drop(columns=self.preprocessor["columns_to_remove"], inplace=True)

        # Encode binary columns
        for binary_column_list, column_encoder in self.preprocessor["binary_columns"]:
            for binary_column_name in binary_column_list:
                model_input[binary_column_name] = column_encoder.transform(model_input[binary_column_name])

        # Encode multiclass columns
        for multiclass_column_list, column_encoder in self.preprocessor["multiclass_columns"]:
            for multiclass_column_name in multiclass_column_list:

                # A warning is thrown due to the OneHotEncoder config...
                # Just ignore it.
                with warnings.catch_warnings(action="ignore"):
                    encoding_matrix = column_encoder.transform(model_input[[multiclass_column_name, ]])

                new_column_names = [f"{multiclass_column_name}:{cat}" for cat in column_encoder.categories_[0]]
                encoding_df = pd.DataFrame(data=encoding_matrix, columns=new_column_names)

                model_input.drop(columns=[multiclass_column_name], inplace=True)
                model_input = pd.concat([model_input, encoding_df], axis=1)

        # Handle the single spaces in the total charges column
        model_input["TotalCharges"] = model_input["TotalCharges"].map(
            lambda x: float(x) if x != " " else float("NaN")
        )

        # Impute missing elements
        for imputer_column_list, column_imputer in self.preprocessor["imputers"]:
            for imputer_column_name in imputer_column_list:
                model_input[imputer_column_name] = column_imputer.transform(model_input[[imputer_column_name, ]])

        # Standardize
        for column_name, transform_name, scaler in self.preprocessor["numeric_scalers"]:

            transformed_column = model_input[[column_name, ]].copy()
            if transform_name == "log1p":
                transformed_column = np.log1p(transformed_column)
            elif transform_name is None:
                pass
            else:
                raise Exception("Invalid transformation name!")

            model_input[column_name] = scaler.transform(transformed_column)


    def train_model(self) -> Dict[str, Any]:

        # Check for model history existence
        history_path = Path(self.config_data["models_dir"]) / "history.json"
        if not history_path.exists():

            self.logger.warning("The model history file does not exist. I will initialize it!")

            with open(history_path, "w") as f:
                json.dump(list(), f)

        # Query the training data for model initialization
        x_train, y_train = self._query_data(is_test=False)

        self.logger.info(f"Preparing for training with x_train {x_train.shape} and y_train {y_train.shape}.")

        # Train the model
        model = RandomForestClassifier(
            n_estimators=self.config_data["rfc_n_estimators"],
            criterion=self.config_data["rfc_criterion"],
            max_depth=self.config_data["rfc_max_depth"],
            min_samples_leaf=self.config_data["rfc_min_samples_leaf"],
            random_state=self.random_seed
        )
        model.fit(x_train, y_train)
        model_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.logger.info(f"Model with ID {model_id} created and trained.")

        # Calculate model performance
        performance_train = self._evaluate_model(model, x_train, y_train)

        # Test the model
        x_test, y_test = self._query_data(is_test=True)
        performance_test = self._evaluate_model(model, x_test, y_test)

        # Create model record
        record = {
            "model_id": model_id,
            "performance_train": performance_train,
            "performance_test": performance_test,
        }

        self.logger.info(f"Performance evaluation for model with ID {model_id} completed.")

        # Save model and model record
        with open(Path(self.config_data["models_dir"]) / f"{model_id}.pkl", "wb") as f:
            pickle.dump(model, f)

        with open(history_path, "r") as f:
            history: List[Any] = json.load(f)
        history.append(record)

        with open(history_path, "w") as f:
            json.dump(history, f, indent=1)

        self.logger.info(f"Model with ID {model_id} saved and added to history.")

        return record


    def list_available_models(self) -> List[Dict[str, Any]]:

        history_path = Path(self.config_data["models_dir"]) / "history.json"

        if not history_path.exists():
            return list()

        with open(history_path, "r") as f:
            history = json.load(f)

        return history


    def run_model(self, model_id: str, model_input: pd.DataFrame):

        # Check for column naming validity
        for column_name in self.expected_columns:
            if column_name not in model_input.columns:
                error_msg = f"The necessary column name {column_name} is not found in the input!"
                self.logger.error(error_msg)
                raise RunModelError(error_msg)

        # Check for history file existence
        history_path = Path(self.config_data["models_dir"]) / "history.json"
        if not history_path.exists():
            error_msg = "History file does not exist!"
            self.logger.error(error_msg)
            raise RunModelError(error_msg)

        # Load history and check for model ID validity
        with open(history_path, "r") as f:
            history = json.load(f)

        available_models = {record["model_id"] for record in history}

        if model_id not in available_models:
            error_msg = "Model name does not exist!"
            self.logger.error(error_msg)
            raise RunModelError(error_msg)

        # Load the selected model file
        model_file = Path(self.config_data["models_dir"]) / f"{model_id}.pkl"
        with open(model_file, "rb") as f:
            model = pickle.load(f)

        # Preprocess the input dataframe
        self._preprocess_pandas_df(model_input)


def main():
    mlh = MLModelHandler(Path("config.toml"), 1994)
    pass


if __name__ == "__main__":
    main()