import pandas as pd


def load_dataset(filepath: str = "../data/Posts.xml") -> pd.DataFrame:
    posts = pd.read_xml(filepath, parser="etree")
    print("Data loaded")
    return posts[:10000]
