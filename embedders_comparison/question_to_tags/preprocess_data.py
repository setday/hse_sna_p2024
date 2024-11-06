import os
import pickle
import warnings

from typing import Any

import pandas as pd
from lxml.etree import XMLParser, parse
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

import swifter


PATH_TO_EMBEDDINGS_TEMPLATE = "../embeddings/{model_name}_{data_name}.obj"

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


def html_to_str(row_html: str) -> str:
    soup = BeautifulSoup(row_html, "html.parser")
    return soup.get_text(separator=" ")


def load_dataset(filepath: str) -> pd.DataFrame:
    # Parse the XML file
    p = XMLParser(huge_tree=True)
    tree = parse(filepath, parser=p)
    # Extract elements from the XML tree
    root = tree.getroot()
    data = []

    for post in root.findall("row"):
        data.append(post.attrib)

    posts = pd.DataFrame(data)
    return posts


def parse_dataset(filepath: str) -> pd.DataFrame:
    # Parse the XML file
    posts = load_dataset(filepath)

    posts["Body"] = posts["Body"].swifter.apply(html_to_str)

    # Drop rows where column "Tags" or "Body" has NaN values
    posts = posts.dropna(subset=["Tags", "Body"])

    posts = posts[
        posts.PostTypeId == "1"
    ]  # important! only real questions, not answers etc.
    return posts


def get_dataset_tags(filepath: str) -> list[list[str]]:
    # Parse the XML file
    posts = load_dataset(filepath)
    posts = posts.dropna(subset=["Tags"])
    posts = posts[
        posts.PostTypeId == "1"
    ]  # important! only real questions, not answers etc.

    tag_series = posts["Tags"]
    tags = [str(str_of_tags).split("|")[1:-1] for str_of_tags in tag_series]
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
