import pandas as pd
import pickle
import os

def load_dataset(filepath: str = "../data/Posts.xml") -> pd.DataFrame:
    posts = pd.read_xml(filepath, parser="etree")
    print("Data loaded")
    return posts


def create_X_y(posts, embedder_name: str = "MiniLM3"):
    file_path = f"../embeddings/{embedder_name}_body.obj"

    if not os.path.exists(file_path):
        raise FileExistsError("Dump the embeddings first.")

    with open(file_path, 'rb') as file:
        X = pickle.load(file)
    y = [str(str_of_tags).split('|')[1:-1] for str_of_tags in posts["Tags"]]

    return X, y
