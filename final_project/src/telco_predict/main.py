# https://fastapi.tiangolo.com/tutorial/response-status-code/
# https://fastapi.tiangolo.com/advanced/response-change-status-code/#use-a-response-parameter
import tomllib

from typing import List, Dict, Any
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Response, status
import uvicorn

from ml_model_handler import MLModelHandler, RunModelError


CONFIG_PATH = Path("config.toml")


def load_app_config() -> Dict[str, Any]:

    with open(CONFIG_PATH, "rb") as f:
        config_data: Dict[str, Any] = tomllib.load(f)["app"]

    return config_data


ml_model_handler = MLModelHandler(CONFIG_PATH, random_seed=1994)
app_config = load_app_config()
app = FastAPI()


@app.get("/model_versions", status_code=200)
async def model_versions() -> List[Dict[str, Any]]:
    return ml_model_handler.list_available_models()


@app.get("/train", status_code=200)
async def train() -> Dict[str, Any]:
    train_record = ml_model_handler.train_model()
    return train_record


@app.post("/predict_test/{model_id}", status_code=200)
async def predict_test(
        model_id: str,
        test_filename: str,
        response: Response
) -> Dict[str, Any]:

    try:
        result = ml_model_handler.run_model(model_id, pd.DataFrame(data=model_input))
    except RunModelError as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": str(e)}

    return {"result": result}


@app.post("/predict/{model_id}", status_code=200)
async def predict_route(
        model_id: str,
        model_input: Dict[str, List[Any]],
        response: Response
) -> Dict[str, Any]:

    try:
        result = ml_model_handler.run_model(model_id, pd.DataFrame(data=model_input))
    except RunModelError as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": str(e)}

    return {"result": result}


@app.post("/add_rows")
async def train_route():
    return {"message": "Unused"}


if __name__ == "__main__":

    uvicorn.run(
        app,
        host=app_config["host"],
        port=app_config["port"]
    )