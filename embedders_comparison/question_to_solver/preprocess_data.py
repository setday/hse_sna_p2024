import sys
import os
import pickle

from typing import Any

import swifter
import pandas as pd
import swifter

from scipy.special import softmax
import numpy as np

sys.path.append("..")  # to make utils importable
import utils.data_worker
import utils.consts


PATH_TO_EMBEDDINGS_TEMPLATE = "./embeddings/{model_name}_{data_name}.obj"


def parse_datasets(filepath: str) -> pd.DataFrame:
    path_to_file = os.path.dirname(filepath)

    if os.path.exists(f"{path_to_file}/dataset_cache/q_an_a.pkl"):
        print("ATTENTION: Found cached result of parse_datasets function! If you have changed the dataset or the preprocessing logic, please remove the cache file from dataset_cache!")

        with open(f"{path_to_file}/dataset_cache/q_an_a.pkl", "rb") as f:
            return pickle.load(f)

    posts = utils.data_worker.load_dataset(filepath)

    print("Preprocessing data...")
    posts["Body"] = posts["Body"].swifter.apply(utils.data_worker.html_to_str)

    # Drop rows where column "Body" has NaN values
    posts = posts.dropna(subset=["Body"])

    # Split data into questions and answers
    # Questions have PostTypeId = 1
    # Answers have PostTypeId = 2
    questions = posts[posts.PostTypeId == "1"]
    answers = posts[posts.PostTypeId == "2"]

    # Drop answers with no parent question or without an owner or without a score
    answers = answers.dropna(subset=["ParentId", "OwnerUserId", "Score"])

    answers["IsAcceptedAnswer"] = answers["Id"].isin(questions["AcceptedAnswerId"])

    # Keep only
    # -> For questions:
    # Id (for indexing)
    # Body (for embedding)
    # -> For answers:
    # Score (for answering probability)
    # IsAcceptedAnswer (for best preferrence matching)
    # ParentId (for question-answer matching)
    # OwnerUserId (for user-based recommendation)
    questions = questions[["Id", "Body"]].astype({"Id": int, "Body": str})
    answers = answers[["Score", "IsAcceptedAnswer", "ParentId", "OwnerUserId"]].astype(
        {"Score": int, "IsAcceptedAnswer": bool, "ParentId": int, "OwnerUserId": int}
    )

    if not os.path.exists(f"{path_to_file}/dataset_cache"):
        os.makedirs(f"{path_to_file}/dataset_cache")
    with open(f"{path_to_file}/dataset_cache/q_an_a.pkl", "wb") as f:
        pickle.dump((questions, answers), f)

    return questions, answers


def create_multytarget_X_y(
    filepath: str, embedder_name: str
) -> tuple[Any, list[list[str]]]:
    file_path = PATH_TO_EMBEDDINGS_TEMPLATE.format(
        model_name=embedder_name, data_name="bodies"
    )
    if not os.path.exists(file_path):
        raise FileExistsError("Dump the embeddings first.")
    with open(file_path, "rb") as file:
        X: pd.DataFrame = pickle.load(file)

    X = X.astype({"ParentId": int})

    questions, answers = parse_datasets(filepath)

    answers = answers.join(X, on="ParentId", rsuffix="Question")

    # Apply softmax to the scores to find the probability of being the best writer for the question
    answers["Probability"] = answers.groupby("ParentId")["Score"].transform(softmax)
    answers.loc[answers.Score < 0, "Probability"] = 0.0
    answers.loc[answers.IsAcceptedAnswer, "Probability"] = 1.0

    answers = answers.dropna()

    Xq = answers["Encoded"].tolist()
    Xa = answers["OwnerUserId"].tolist()

    Xq_np = np.array(Xq)
    Xa_np = np.array([Xa])

    X = np.concatenate((Xq_np, Xa_np.T), axis=1)
    y = answers["Probability"].tolist()

    print(X, y)
    return X, y


def create_besttraget_X_y(
    filepath: str, embedder_name: str
) -> tuple[Any, list[list[str]]]:
    file_path = PATH_TO_EMBEDDINGS_TEMPLATE.format(
        model_name=embedder_name, data_name="bodies"
    )
    if not os.path.exists(file_path):
        raise FileExistsError(f"{file_path} not found. Dump the embeddings first.")
    with open(file_path, "rb") as file:
        X: pd.DataFrame = pickle.load(file)

    X = X.astype({"ParentId": int})

    questions, answers = parse_datasets(filepath)

    answers = answers.join(X, on="ParentId", rsuffix="Question")

    answers.loc[answers.IsAcceptedAnswer, "Score"] = 9999999999
    idx_max = answers.groupby("ParentId")["Score"].idxmax()
    answers = answers.loc[idx_max].dropna()

    X = answers["Encoded"].tolist()
    y = answers["OwnerUserId"].tolist()

    return X, y
