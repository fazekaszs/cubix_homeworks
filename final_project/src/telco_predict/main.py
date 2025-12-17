# https://fastapi.tiangolo.com/tutorial/response-status-code/
# https://fastapi.tiangolo.com/advanced/response-change-status-code/#use-a-response-parameter
import tomllib

from typing import List, Dict, Any
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Response, status
from pydantic import BaseModel
import uvicorn

from ml_model_handler import MLModelHandler, RunModelError


CONFIG_PATH = Path("config.toml")


class ExtendDatabasePayload(BaseModel):
    new_rows: Dict[str, List[Any]]
    train_test_flag: str


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


@app.post("/predict/{model_id}", status_code=200)
async def predict(
    model_id: str,
    model_input: Dict[str, List[Any]],
    response: Response
) -> Dict[str, Any]:

    df = pd.DataFrame(**model_input)

    try:
        result = ml_model_handler.run_model(model_id, df)
    except RunModelError as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": str(e)}

    return result


@app.post("/extend_database", status_code=200)
async def extend_database(
    payload: ExtendDatabasePayload,
    response: Response
) -> Dict[str, str]:

    df = pd.DataFrame(**payload.new_rows)

    if payload.train_test_flag == "train":
        is_test = False
    elif payload.train_test_flag == "test":
        is_test = True
    elif payload.train_test_flag == "random":
        is_test = None
    else:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": f"Invalid option for train_test_flag ({payload.train_test_flag})!"}

    ml_model_handler.extend_database(df, is_test)

    return {"result": "success"}


if __name__ == "__main__":
    uvicorn.run(app, host=app_config["host"], port=app_config["port"])
