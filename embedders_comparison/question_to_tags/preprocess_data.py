import sys
import os
import pickle

from typing import Any

import pandas as pd

import swifter

sys.path.append("..")  # to make utils importable
import utils.data_worker
import utils.consts


PATH_TO_EMBEDDINGS_TEMPLATE = "../embeddings/{model_name}_{data_name}.obj"


def parse_dataset(filepath: str) -> pd.DataFrame:
    # Parse the XML file
    posts = utils.data_worker.load_dataset(filepath)

    posts["Body"] = posts["Body"].swifter.apply(utils.data_worker.html_to_str)

    # Drop rows where column "Tags" or "Body" has NaN values
    posts = posts.dropna(subset=["Tags", "Body"])

    # important! only real questions, not answers etc.
    questions, answers = utils.data_worker.question_answer_split(posts)

    return questions


def get_dataset_tags(filepath: str) -> list[list[str]]:
    # Parse the XML file
    posts = utils.data_worker.load_dataset(filepath)
    posts = posts.dropna(subset=["Tags"])
    
    # important! only real questions, not answers etc.
    questions, answers = utils.data_worker.question_answer_split(posts)

    tags = utils.data_worker.extract_tags_from_str(questions["Tags"])
    return tags


def create_X_y(filepath: str, embedder_name: str) -> tuple[Any, list[list[str]]]:
    file_path = PATH_TO_EMBEDDINGS_TEMPLATE.format(
        model_name=embedder_name, data_name="bodies"
    )
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            X = pickle.load(file)
    else:
        raise FileExistsError("Dump the embeddings first.")

    y = get_dataset_tags(filepath)
    print(y)

    return X, y
