import sys
import json
import os
import pickle
import warnings

from typing import Any

import swifter
import pandas as pd
import swifter
from lxml.etree import XMLParser, parse
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

from scipy.special import softmax
import numpy as np


from utils import metrics

import click

from tqdm import tqdm

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

import lightning as L

from preprocess_data import create_multytarget_X_y, create_besttraget_X_y
from boost_model import SolverPredictor
from linear_model import SolverEvaluatorLightningModule as SolverEvaluator

from dataset import QuestionSolverDataset
PATH_TO_EMBEDDINGS_TEMPLATE = "./embeddings/{model_name}_{data_name}.obj"


def parse_datasets(filepath: str) -> pd.DataFrame:
    path_to_file = os.path.dirname(filepath)

    if os.path.exists(f"{path_to_file}/dataset_cache/q_an_a.pkl"):
        print("ATTENTION: Found cached result of parse_datasets function! If you have changed the dataset or the preprocessing logic, please remove the cache file from dataset_cache!")

        with open(f"{path_to_file}/dataset_cache/q_an_a.pkl", "rb") as f:
            return pickle.load(f)

    posts = load_dataset(filepath)

    print("Preprocessing data...")
    posts["Body"] = posts["Body"].swifter.apply(html_to_str)

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


def predict(model, input_data, target_type="best", embedder_name="MiniLM3"):
    """
    Predict using trained model.

    Args:
        model: Trained model (either SolverPredictor for 'best' target or SolverEvaluator for 'multy' target).
        input_data: Input data for prediction.
        target_type: Either 'best' or 'multy' target type.
        embedder_name: Name of the embedder used for preprocessing input data.

    Returns:
        predictions: Model predictions for the input data.
    """
    if target_type != "multy" or embedder_name != "Albert":
        raise ValueError("Not implemented")

    file_path = PATH_TO_EMBEDDINGS_TEMPLATE.format(
        model_name=embedder_name, data_name="bodies"
    )
    if not os.path.exists(file_path):
        raise FileExistsError("Dump the embeddings first.")
    with open(file_path, "rb") as file:
        X: pd.DataFrame = pickle.load(file)

    X = X.astype({"ParentId": int})

    true_X = X.copy()

    questions, answers = parse_datasets(input_data)

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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Making predictions with multy target model...")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        predictions = model(X_tensor).cpu().numpy()

    finally_normal_pred = {}
    for i in range(len(answers)):
        answer = answers.iloc[i]
        if answer.ParentId not in finally_normal_pred:
            finally_normal_pred[answer.ParentId] = []

        finally_normal_pred[answer.ParentId].append((predictions[i], Xa[i]))

    for question in finally_normal_pred.keys():
        finally_normal_pred[question].sort()

    sys.path.append("..")

    # print(metrics.ILS([[id for prob, id in responders[-5:]] for responders in finally_normal_pred.values()] {
    #     # I do not have embeddingd
    # }))

    return [(question, responders[-1][1]) for question, responders in finally_normal_pred.items()]


@click.command()
@click.option(
    "--target",
    help="use 'best' target to predict the best answer / use 'multy' target to predict the probability of solver to answer the question",
)
@click.option(
    "--embedder", default="MiniLM3", help="the name of the embedder to be used"
)
@click.option("--model_path", required=True, help="path to the trained model file")
@click.option(
    "--input_data", required=True, help="path to the input data for prediction"
)
def main(target, embedder, model_path, input_data):
    print(f"Loading model from {model_path}...")
    model = torch.load(model_path)

    predictions = predict(
        model=model,
        input_data=input_data,
        target_type=target,
        embedder_name=embedder,
    )

    print(*predictions[:5], sep='\n')


if __name__ == "__main__":
    main()
