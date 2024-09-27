import pandas as pd
from lxml.etree import XMLParser, parse
import pickle
import os
import warnings
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
warnings.filterwarnings('ignore', category=MarkupResemblesLocatorWarning)


def html_to_str(row_html: str) -> str:
    soup = BeautifulSoup(row_html, 'html.parser')
    return soup.get_text(separator=' ')


def parse_dataset(filepath: str = "../../data/Posts.xml") -> pd.DataFrame:
    # Parse the XML file
    p = XMLParser(huge_tree=True)
    tree = parse(filepath, parser=p)
    # Extract elements from the XML tree
    root = tree.getroot()
    data = []

    for post in root.findall('row'):
        data.append(post.attrib)

    # Convert to a pandas DataFrame
    posts = pd.DataFrame(data)
    posts["Body"] = posts["Body"].apply(html_to_str)

    # Drop rows where column 'Tags' has NaN values
    posts = posts.dropna(subset=['Tags'])
    return posts


def create_X_y(filepath: str, embedder_name: str = "MiniLM3"):
    posts = parse_dataset(filepath=filepath)
    file_path = f"../embeddings/{embedder_name}_body.obj"
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            X = pickle.load(file)
    else:
        raise FileExistsError("Dump the embeddings first.")

    y = posts["Tags"]
    y = [str(str_of_tags).split('|')[1:-1] for str_of_tags in y]

    return X, y
