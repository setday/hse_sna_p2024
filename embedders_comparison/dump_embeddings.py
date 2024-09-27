from question_to_tags.preprocess_data import parse_dataset
from question_to_tags.embedders import EMBEDDERS
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm


def dump_embeddings(models: dict[str, str], dataset: pd.DataFrame, column_name: str = "Body") -> None:
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    for model_pretty_name, model_official_name in models.items():
        print(model_pretty_name, model_official_name)
        # Check if embeddings are already dumped (using this model)
        embeddings_file = Path(f"embeddings/{model_pretty_name}_{column_name.lower()}.obj")
        if embeddings_file.is_file():  # file exists
            print(f"Embeddings for {model_pretty_name} are already dumped.")
            continue

        # Download a model from Hugging Face using its name and move it to the appropriate device
        embedder = SentenceTransformer(model_official_name).to(device)

        bodies = dataset[column_name].tolist()

        X = []
        for body in tqdm(bodies, desc="Encoding posts"):
            # Encode each 'body' and append it to X
            encoded_body = embedder.encode(body, device=device)  # Move the input to the GPU (if available)
            X.append(encoded_body)

        X = np.array(X)
        with open(embeddings_file, "wb") as filehandler:
            pickle.dump(X, filehandler)


if __name__ == "__main__":
    print("Loading data...")
    posts = parse_dataset(filepath="../data/Posts.xml")
    dump_embeddings(models=EMBEDDERS, dataset=posts, column_name="Body")
