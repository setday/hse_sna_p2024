import os
import pickle
import warnings

from typing import Any

import pandas as pd
import swifter
from lxml.etree import XMLParser, parse
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

from scipy.special import softmax
import numpy as np


PATH_TO_EMBEDDINGS_TEMPLATE = "../embeddings/{model_name}_{data_name}.obj"

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


def html_to_str(row_html: str) -> str:
    soup = BeautifulSoup(row_html, "html.parser")
    return soup.get_text(separator=" ")


def load_dataset(filepath: str) -> pd.DataFrame:
    print(f"INFO: Loading dataset {filepath}...")

    # Parse the XML file
    p = XMLParser(huge_tree=True)
    tree = parse(filepath, parser=p)
    # Extract elements from the XML tree
    root = tree.getroot()

    print("Extracting data...")
    data = [post.attrib for post in root.findall("row")]

    # Convert to a pandas DataFrame
    posts = pd.DataFrame(data)
    return posts


def parse_datasets(filepath: str) -> pd.DataFrame:
    path_to_file = os.path.dirname(filepath)

    if os.path.exists(f"{path_to_file}/dataset_cache/q_an_a.pkl"):
        print ("ATTENTION: Found cached result of parse_datasets function! If you have changed the dataset or the preprocessing logic, please remove the cache file for dataset_cache!")

        with open(f"{path_to_file}/dataset_cache/q_an_a.pkl", "rb") as f:
            return pickle.load(f)

    # Parse the XML file
    posts = load_dataset(filepath)

    print("Preprocessing data...")
    # Convert the "Body" column from HTML to plain text
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


def create_multytraget_X_y(
    filepath: str, embedder_name: str
) -> tuple[Any, list[list[str]]]:
    file_path = PATH_TO_EMBEDDINGS_TEMPLATE.format(
        model_name=embedder_name, data_name="bodies"
    )
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            X: pd.DataFrame = pickle.load(file)
    else:
        raise FileExistsError("Dump the embeddings first.")

    X = X.astype({"ParentId": int})

    questions, answers = parse_datasets(filepath)
    # with open("tmp_q_a.pkl", "wb") as f:
    #     pickle.dump((questions, answers), f)
    # with open("tmp_q_a.pkl", "rb") as f:
    #     questions, answers = pickle.load(f)

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

    return X, y


def create_besttraget_X_y(
    filepath: str, embedder_name: str
) -> tuple[Any, list[list[str]]]:
    file_path = PATH_TO_EMBEDDINGS_TEMPLATE.format(
        model_name=embedder_name, data_name="bodies"
    )
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            X: pd.DataFrame = pickle.load(file)
    else:
        raise FileExistsError("Dump the embeddings first.")

    X = X.astype({"ParentId": int})

    questions, answers = parse_datasets(filepath)
    # with open("tmp_q_a.pkl", "wb") as f:
    #     pickle.dump((questions, answers), f)
    # with open("tmp_q_a.pkl", "rb") as f:
    # questions, answers = pickle.load(f)

    answers = answers.join(X, on="ParentId", rsuffix="Question")

    # Apply softmax to the scores to find the probability of being the best writer for the question
    answers.loc[answers.IsAcceptedAnswer, "Score"] = 9999999999
    idx_max = answers.groupby("ParentId")["Score"].idxmax()
    answers = answers.loc[idx_max].dropna()

    X = answers["Encoded"].tolist()
    y = answers["OwnerUserId"].tolist()

    return X, y
