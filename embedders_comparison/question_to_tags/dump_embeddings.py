import sys
sys.path.append("../../")

from utils.data_loader import load_dataset
from utils.consts import EMBEDDERS
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm


def dump_embeddings(models: dict[str, str], dataset: pd.DataFrame, column_name: str = "Body") -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    for model_pretty_name, model_official_name in models.items():
        print(model_pretty_name, model_official_name)
        embeddings_file = Path(f"../embeddings/{model_pretty_name}_{column_name.lower()}.obj")
        if embeddings_file.is_file():
            print(f"Embeddings for {model_pretty_name} are already dumped.")
            continue

        embedder = SentenceTransformer(model_official_name).to(device)
        bodies = dataset[column_name].tolist()

        X = []
        for body in tqdm(bodies, desc="Encoding posts"):
            encoded_body = embedder.encode(body, device=device)
            X.append(encoded_body)

        X = np.array(X)
        
        with open(embeddings_file, "wb") as filehandler:
            pickle.dump(X, filehandler)


if __name__ == "__main__":
    print("Loading data...")
    posts = load_dataset(filepath="../../data/Posts.xml") # TODO: use better loader (i.e. preprocessed)
    dump_embeddings(models=EMBEDDERS, dataset=posts)
