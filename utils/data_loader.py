import os

# import polars as pl
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

def pl_read_xml(file: str, datapath: str = "data/") -> pd.DataFrame:
    pd_df = pd.read_xml(f"{datapath}{file}.xml")
    # pl_df = pl.DataFrame(pd_df)
    return pd_df

def load_dataset(datapath: str = "data/") -> dict[str, pd.DataFrame]:
    if not os.path.exists(datapath):
        print("Data directory not found. Loading data from XML files.")
        os.system("bash utils/load_data.sh")

    data = {}

    for file in files:
        if not os.path.exists(f"{datapath}{file}.xml"):
            print(f"File {file}.xml not found. Loading data from XML files.")
            os.system(f"bash utils/load_{file}.sh")
        
        data[file] = pl_read_xml(file, datapath)
        print(f"Loaded {file}.xml")

    return data
