import sys
import pickle
from pathlib import Path

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import chromadb

from question_to_tags.preprocess_data import parse_dataset

sys.path.append("..")

from utils.consts import EMBEDDERS

# PATH_TO_EMBEDDERS_LIST = "embedders_list.json"
PATH_TO_DATASET_POSTS = "../data/Posts.xml"
PATH_TO_EMBEDDINGS = "embeddings/"
PATH_TO_EMBEDDINGS_DB = "embeddings_db/"
PATH_TO_EMBEDDING_FILE_TEMPLETE = "{model_name}_{data_name}"
EMBEDDING_FILE_EXTENSION = ".obj"


def dump_embeddings(
    data_list: list[str],
    data_ids: list[int],
    data_name: str,
    embedder_model: str,
    embedder_pretty_name: str,
    device="cpu",
) -> None:
    # Check if embeddings are already dumped (using this model)
    embeddings_file = Path(
        PATH_TO_EMBEDDINGS
        + PATH_TO_EMBEDDING_FILE_TEMPLETE.format(
            model_name=embedder_pretty_name, data_name=data_name
        )
        + EMBEDDING_FILE_EXTENSION
    )
    if embeddings_file.is_file():  # file exists
        raise FileExistsError("Embeddings are already dumped.")

    # Download a model from Hugging Face using its name and move it to the appropriate device
    embedder = SentenceTransformer(embedder_model).to(device)

    X = []
    for body in tqdm(data_list, desc="Encoding posts"):
        # Encode each 'body' and append it to X
        encoded_body = embedder.encode(
            body, device=device
        )  # Move the input to the GPU (if available)
        X.append(encoded_body)

    X_np = np.array(X)
    with open(embeddings_file, "wb") as filehandler:
        pickle.dump(X_np, filehandler)

    chroma_client = chromadb.PersistentClient(path=PATH_TO_EMBEDDINGS_DB)
    body_collection = chroma_client.create_collection(
        PATH_TO_EMBEDDING_FILE_TEMPLETE.format(
            model_name=embedder_pretty_name, data_name=data_name
        )
    )
    for index, embedding in enumerate(X):
        body_collection.add(
            ids=[data_ids[index]],
            documents=[],
            embeddings=[embedding],
            metadatas=[{"model": embedder_pretty_name, "id": data_ids[index]}],
        )


def try_dump_embeddings(
    data_list: list[str],
    data_ids: list[int],
    data_name: str,
    model_name: str,
    model_pretty_name: str,
    device="cpu",
):
    print(f"Dumping embeddings for {model_pretty_name}...")
    try:
        dump_embeddings(
            data_list=data_list,
            data_ids=data_ids,
            data_name=data_name,
            embedder_model=model_name,
            embedder_pretty_name=model_pretty_name,
            device=device,
        )
        print(f"Embeddings for {model_pretty_name} are dumped.")
    except FileExistsError:
        print(f"Embeddings for {model_pretty_name} are already dumped.")


def get_optimal_device():
    # Check if CUDA is available
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_embedders_list():
    # with open(PATH_TO_EMBEDDERS_LIST, "r") as file:
    #     embedders_list = json.load(file)
    return EMBEDDERS


if __name__ == "__main__":
    print("Loading embedders list...")
    embedders_list = get_embedders_list()

    print("Loading data...")
    posts = parse_dataset(filepath=PATH_TO_DATASET_POSTS)

    print("Choosing device...")
    # Check if CUDA is available
    device = get_optimal_device()
    print(f"device: {device}")

    # Create ebmeddings folder if it doesn't exist
    print("Creating embeddings folder...")
    Path(PATH_TO_EMBEDDINGS).mkdir(parents=True, exist_ok=True)
    Path(PATH_TO_EMBEDDINGS_DB).mkdir(parents=True, exist_ok=True)

    bodies = posts["Body"].tolist()
    data_ids = posts["Id"].tolist()

    for model_pretty_name, model_name in embedders_list.items():
        try_dump_embeddings(
            data_list=bodies,
            data_ids=data_ids,
            data_name="bodies",
            model_name=model_name,
            model_pretty_name=model_pretty_name,
            device=device,
        )
