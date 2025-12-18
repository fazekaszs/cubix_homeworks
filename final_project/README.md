# TELCO Customer Churn Prediction

## Introduction

This project is based on the TELCO customer churn data found
[here](https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download).
The goal is to predict whether a customer will need the company's services in
the future or not anymore.

## Notebooks

The EDA, data-preprocessing and model selection steps were performed in Jupyter
notebooks found in the `notebooks` directory.

During EDA, 

- I looked for missing data cells, 
- explored the different datatypes in different columns,
- calculated descriptive statistics, 
- created data visualizations,
- looked for correlations,
- asserted single feature target descriptive power,
- looked for outliers,
- performed dimensionality reduction,
- asserted reduced dimension target descriptive power.

During model selection, I created and saved the table preprocessing pipeline.

During model selection,

- I split the dataset into train, validation and test sets,
- due to the imbalanced nature of the target column, I selected the Matthew's
  correlation coefficient as a metric to be maximized,
- defined Optuna optimization functions for logistic regression, decision
  tree classifiers, random forest classifiers and support vector classifiers,
- ran Optuna for 10'000 steps and saved the study in a persistent database,
- analyzed the study results using different metrics and ML method groupings,
- finalized the hyperparameters and created the final model,
- evaluated the final model with respect to feature importance.

## MLOps Project

The `src/telco_predict` module contains a FastAPI endpoint implementation for
random forest classifier (RFC) training, retraining and evaluation, along with
data handling with options for database extension.
It is a runnable module, which should be first installed with pip.
Installation dependencies can be found in `pyproject.toml` (automatically
detected by pip).
In the source directory, run
```bash
pip install .
python -m telco_predict
```
which will start the server.
When ran elsewhere, the module expects the following files and directories 
to be present in the working directory:

- a `config.toml` file, which contains run parameters (see the example config),
- an `artifacts` directory, containing pickled files for the preprocessed TELCO
  dataset (`processed_dataset.pkl`) and for the table preprocessing 
  configuration (`preprocessing_config.pkl`, both saved from the Jupyter 
  notebooks),
- an empty `models` directory.

The project is structured as follows:

- in `__main__.py`, we have the FastAPI route handling logic,
- in `ml_model_handler.py`, we have the database and model handling logic.

The following endpoints are available:

- `(GET) http://{ip}:{port}/model_versions` lists all available, 
  already trained model records,
- `(GET) http://{ip}:{port}/train` trains a new RFC model, returns its record 
  (model ID, train performance and test performance),
- `(POST) http://{ip}:{port}/predict/{model_id}` runs a prediction using the
  selected model (see the model_versions endpoint), and also expects a
  JSON payload containing the input dataframe elements,
- `(POST) http://{ip}:{port}/extend_database` extends the underlying database
  with new rows from the payload.

During the server run and endpoint calls, the following files and directories
get modified:

- a `customer_data.db` sqlite3 database will be created holding all customer
  data in a processed (i.e., model-ready) manner,
- a `MLModelHandler.log` file will be created, containing log messages from
  the `MLModelHandler` object initialized in `__main__.py`,
- in the `models` directory, a `history.json` file is created, if it did not
  exist before, containing the available models and their performance 
  evaluations (this will be returned during a model_versions endpoint call),
- also in the `models` directory, a pickled file will be created for every
  trained model with the name `{model_id}.pkl`.

In the `testfiles` directory, a short testscript (`call_api.py`) is present
that calls the running server using the unprocessed datafile 
`unprocessed_dataset.pkl`.
It successively extends the database with synthetic data, retrains models and
evaluates their performance.
Doing this, it plots the performance change of the newly trained models.