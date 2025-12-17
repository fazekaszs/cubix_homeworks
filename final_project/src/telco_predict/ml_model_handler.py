import pickle
import tomllib
import sqlite3
import logging
import json
import warnings

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime

import pandas as pd
import numpy as np

import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier


class RunModelError(Exception):
    pass


class MLModelHandler:


    def __init__(self, config_path: Path, random_seed: int) -> None:
        """
        Initializes a new ML model handler object.
        This object handles train and test data operations in the database,
        model training and evaluation, as well as model version control.

        :param config_path: The path to the config file of the handler.
        :param random_seed: Seeding of random operations for reproducability.
        """

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
        self._initialize_database()

        # Set the value for the expected columns field.
        connection = sqlite3.connect(self.config_data["database"])
        self.expected_columns: List[str] = [
            element[1] for element in
            connection.execute(f"PRAGMA table_info({self.config_data['database_table']})").fetchall()
        ]
        connection.close()

        # Since the Churn column is the target column, we should remove it.
        # We also do not need the index and is_test columns.
        self.expected_columns.remove("index")
        self.expected_columns.remove("Churn")
        self.expected_columns.remove("is_test")

        self.logger.info(f"The {len(self.expected_columns)} column names are: {self.expected_columns}.")


    def _load_preprocessor_config(self) -> Dict[str, Any]:
        """
        Loads the raw dataframe preprocessor config file from the artifacts directory.
        This tells the handler how to process specific columns in an input dataframe,
        before adding data to the database or running an ML training/prediction.

        :return: The preprocessor config dictionary.
        """

        preprocessor_file = Path(self.config_data["artifacts_dir"]) / "preprocessing_config.pkl"
        with open(preprocessor_file, "rb") as f:
            preprocessor = pickle.load(f)

        return preprocessor


    def _add_random_test_mask(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This method adds a new column to a dataframe with a title of is_test.
        This column contains zeros and ones, assigned randomly to each row.
        The number of ones is dictated by the train_test_split value in the config file.

        :param df: The target dataframe.
        :return: The same dataframe, but with the is_test column added.
        """

        test_mask = self.random_number_generator.choice(
            len(df),
            size=int(len(df) * self.config_data["train_test_split"]),
            replace=False
        )
        test_flags = np.zeros(len(df), dtype=int)
        test_flags[test_mask] = 1
        df["is_test"] = test_flags

        self.logger.info(f"Added test flag for {len(test_mask)} entries randomly.")

        return df


    def _initialize_database(self) -> None:
        """
        Initializes a new sqlite3 database according to the ML handler's config and
        populates it with data from the preprocessed TELCO dataset found in the
        artifacts directory.
        """

        # Check for database existence
        if Path(self.config_data["database"]).exists():
            self.logger.warning("Trying to initialize the database, while it already exists! Skipping...")
            return

        # Load the original dataset from the artifacts directory
        original_dataset_path = Path(self.config_data["artifacts_dir"]) / "preprocessed_dataset.pkl"
        with open(original_dataset_path, "rb") as f:
            original_dataset: pd.DataFrame = pickle.load(f)

        self.logger.info(f"Original dataset loaded with shape: {original_dataset.shape}.")

        # Add train/test indicator column
        original_dataset = self._add_random_test_mask(original_dataset)

        # Write the DataFrame content to the database
        connection = sqlite3.connect(self.config_data["database"])
        original_dataset.to_sql(name=self.config_data["database_table"], con=connection)
        connection.close()

        self.logger.info(f"Original dataset successfully written to database.")


    def _query_data(self, is_test: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Queries data from the database based on the is_test flag and separates it into a
        model_input and a model_target numpy array.

        :param is_test: Whether to query rows flagged for testing or rows flagged for training.
        :return: The input and output arrays. The latter one is the Churn column.
        """

        # Connect to the database and query
        connection = sqlite3.connect(self.config_data["database"])
        training_data = pd.read_sql(
            f"SELECT * FROM {self.config_data['database_table']} WHERE is_test = {1 if is_test else 0}",
            connection
        )
        connection.close()

        self.logger.info(f"Queried {len(training_data)} entries from the database ({is_test=}).")

        # Preprocess raw database data for training
        model_target = training_data["Churn"].values
        model_input = training_data.drop(columns=["index", "Churn", "is_test"]).values

        return model_input, model_target


    @staticmethod
    def _evaluate_model(
        y_prediction: Tuple[RandomForestClassifier, np.ndarray] | np.ndarray,
        y_true: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluates the performance of a random forest classifier model on a specific input-output pair.

        :param y_prediction: Either a numpy array of predicted values or a (model, input) pair as a tuple.
        :param y_true: The model target output.
        :return: A dictionary containing the model's Matthews correlation coefficient, accuracy, F1-score,
            precision and recall values.
        """

        if type(y_prediction) is tuple:
            y_prediction = y_prediction[0].predict(y_prediction[1])
        else:
            pass

        return {
            "mcc": metrics.matthews_corrcoef(y_true, y_prediction),
            "acc": metrics.accuracy_score(y_true, y_prediction),
            "f1": metrics.f1_score(y_true, y_prediction),
            "precision": metrics.precision_score(y_true, y_prediction),
            "recall": metrics.recall_score(y_true, y_prediction),
        }


    def _preprocess_pandas_df(self, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses a dataframe based on the preprocessor config loaded from the _load_preprocessor_config
        method.
        Removes unnecessary columns, encodes binary and multiclass columns, handles imputer and standardizer
        actions.
        All columns specified in the preprocessor config must be present in the dataframe, except for the
        Churn target column, which is optional (during inference, it is not available, but during training,
        it is necessary).

        :param model_input: The dataframe that must be processed.
        :return: The processed dataframe.
        """

        # Remove the unnecessary columns
        model_input.drop(columns=self.preprocessor["columns_to_remove"], inplace=True)

        # Encode binary columns
        for binary_column_list, column_encoder in self.preprocessor["binary_columns"]:
            for binary_column_name in binary_column_list:

                # Skip the binary Churn column, if it does not exist, like
                # during inference
                if binary_column_name == "Churn" and "Churn" not in model_input.columns:
                    continue

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

        # Filter out only the necessary columns
        if "Churn" in model_input.columns:
            model_input = model_input[self.expected_columns + ["Churn", ]]
            return model_input

        model_input = model_input[self.expected_columns]
        return model_input


    def train_model(self) -> Dict[str, Any]:
        """
        Trains and evaluates a new random forest classifier using data retrieved from the database.
        It also registers the new model in the history JSON file and saves the trained model in a
        pickled format.

        :return: The new record dictionary that was added to the history file.
        """

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
        performance_train = self._evaluate_model((model, x_train), y_train)

        # Test the model
        x_test, y_test = self._query_data(is_test=True)
        performance_test = self._evaluate_model((model, x_test), y_test)

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
        """
        Lists all available models registered in the history file.

        :return: All records from the history JSON file.
        """

        history_path = Path(self.config_data["models_dir"]) / "history.json"

        if not history_path.exists():
            return list()

        with open(history_path, "r") as f:
            history = json.load(f)

        return history


    def run_model(self, model_id: str, model_input: pd.DataFrame) -> Dict[str, Any]:
        """
        After checking for model ID validity, it loads the specified model and runs it
        on the preprocessed model input.
        The model ID must be registered in the history JSON file in order to load it
        successfully.

        :param model_id: The ID of the model to be used.
            This can be obtained by calling the list_available_models method.
        :param model_input: The raw, unprocessed input dataframe to the model.
        :return: The results of the model run.
        """

        self.logger.info(
            f"Preparing for inference with model_id {model_id} "
            f"and with an (unprocessed) dataframe shape of {model_input.shape}."
        )

        # Preprocess the input dataframe
        model_input = self._preprocess_pandas_df(model_input)

        # Check whether a target column is given
        target_values = None
        if "Churn" in model_input.columns:
            target_values = model_input["Churn"].values
            model_input.drop(columns=["Churn", ], inplace=True)

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

        # Initialize the output dictionary
        output = dict()

        # Calculate model prediction
        model_prediction = model.predict(model_input.values)
        output["model_prediction"] = model_prediction.tolist()

        self.logger.info(
            f"Model prediction ran with a predicted churn rate of {np.mean(model_prediction):.3%}."
        )

        # If the model_input also contains a target (Churn) column, calculate performance.
        if target_values is not None:
            model_performance = self._evaluate_model(model_prediction, target_values)
            output["model_performance"] = model_performance

        return output


    def extend_database(self, new_rows: pd.DataFrame, is_test: Optional[bool]) -> None:
        """
        Extends the database with new rows from the specified database.
        The is_test column is populated according to the is_test argument:

        - if it's None, then it is assigned randomly to the new rows,
        - if it's True, then all new rows will be flagged as testing rows,
        - if it's False, then all new rows will be flagged as training rows.

        :param new_rows: The dataframe to be added to the database.
        :param is_test: It specifies how the is_test column is populated.
        :return:
        """

        # Preprocess the input dataframe
        new_rows = self._preprocess_pandas_df(new_rows)

        self.logger.info(
            f"Attempting to extend the database with a dataframe of shape {new_rows.shape}."
        )

        # Add a test indicator column
        if is_test is None:
            new_rows = self._add_random_test_mask(new_rows)
        elif is_test:
            new_rows["is_test"] = np.ones(len(new_rows))
        elif not is_test:
            new_rows["is_test"] = np.zeros(len(new_rows))
        else:
            raise Exception("Unreachable!")

        # Connect to the database and add the new rows
        connection = sqlite3.connect(self.config_data["database"])
        new_rows.to_sql(self.config_data["database_table"], connection, if_exists="append")
        connection.close()

        self.logger.info(
            f"Database successfully extended."
        )


def main():
    mlh = MLModelHandler(Path("config.toml"), 1994)
    pass


if __name__ == "__main__":
    main()