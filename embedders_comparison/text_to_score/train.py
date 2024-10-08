import sys

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sentence_transformers import SentenceTransformer

import numpy as np
import pandas as pd

from tqdm import tqdm

sys.path.append("../..")

from utils.consts import EMBEDDERS

tqdm.pandas()


def get(data):
    bodies = data["Body"].tolist()
    y = data["Score"].values
    return train_test_split(bodies, y, test_size=0.2, random_state=1200)


def estimate_embedder(data: pd.DataFrame, model_name: str) -> float:
    """
    Estimates the performance of a linear regression model using embeddings
    generated by a specified SentenceTransformer model from Hugging Face.

    Args:
        model_name (str): The name of the model to be used for generating embeddings.

    Returns:
        float: The Mean Absolute Error (MAE) of the linear regression model
               on the test set.
    """
    # Download a model from Hugging Face using its name
    selected_model = EMBEDDERS[model_name]
    embedder = SentenceTransformer(selected_model)

    train_bodies, test_bodies, y_train, y_test = get(data)

    X_train = np.array(
        [
            embedder.encode(body)
            for body in tqdm(
                train_bodies, desc=f"Encoding train data with {model_name}"
            )
        ]
    )

    # Create and fit a simple linear regression model
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    X_test = np.array(
        [
            embedder.encode(body)
            for body in tqdm(test_bodies, desc=f"Encoding test data with {model_name}")
        ]
    )

    y_pred = regressor.predict(X_test)
    return round(mean_absolute_error(y_test, y_pred), 4)
