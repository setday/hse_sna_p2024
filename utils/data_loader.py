import pandas as pd
import pickle
import os

def load_dataset(filepath: str = "../data/Posts.xml") -> pd.DataFrame:
    posts = pd.read_xml(filepath, parser="etree")
    print("Data loaded")
    return posts[:10000]


def create_X_y(filepath: str, embedder_name: str = "MiniLM3"):
    posts = load_dataset(filepath=filepath)
    file_path = f"../embeddings/{embedder_name}_body.obj"
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            X = pickle.load(file)
    else:
        raise FileExistsError("Dump the embeddings first.")

    y = posts["Tags"]
    y = [str(str_of_tags).split('|')[1:-1] for str_of_tags in y]

    return X, y
