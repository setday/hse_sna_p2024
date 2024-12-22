import pandas as pd


def load_dataset(filepath: str = "../data/Posts.xml", full: bool = False) -> pd.DataFrame:
    posts = pd.read_xml(filepath, parser="etree")
    print("Data loaded")
    if not full:
        return posts[:10000]
    return posts
