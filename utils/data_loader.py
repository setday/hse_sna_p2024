import os

import pandas as pd

files = [
    # "Badges",
    # "Comments",
    # "PostHistory",
    # "PostLinks",
    "Posts",
    # "Tags",
    # "Users",
    # "Votes",
]


def read_xml(file: str, datapath: str = "data/") -> pd.DataFrame:
    return pd.read_xml(f"{datapath}{file}.xml")


def load_dataset(datapath: str = "data/") -> dict[str, pd.DataFrame]:
    if not os.path.exists(datapath):
        print("Data directory not found. Loading data from XML files.")
        os.system("bash utils/load_data.sh")

    data = {}

    for file in files:
        if os.path.exists(f"{datapath}{file}.xml"):
            data[file] = read_xml(file, datapath)
            print(f"Loaded {file}.xml")
        else:
            print(f"File {file}.xml not found in loaded data.")

    return data
