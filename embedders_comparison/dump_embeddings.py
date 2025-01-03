import sys
import pickle
import json

import click

from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import pandas as pd

from question_to_solver.preprocess_data import parse_datasets

PATH_TO_EMBEDDERS_LIST = "embedders_list.json"
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
    print(embedder_model)
    embedder = SentenceTransformer(embedder_model).to(device)

    data_to_df = []
    for idx, body in tqdm(
        zip(data_ids, data_list), total=len(data_list), desc="Encoding posts"
    ):
        # Encode each 'body' and append it to X
        encoded_body = embedder.encode(
            body, device=device
        )  # Move the input to the GPU (if available)
        data_to_df.append({"ParentId": idx, "Encoded": encoded_body})

    data_df = pd.DataFrame(data_to_df)

    with open(embeddings_file, "wb") as filehandler:
        pickle.dump(data_df, filehandler)

    # X = data_df["Encoded"].tolist()
    # X_np = np.array(X)

    # chroma_client = chromadb.PersistentClient(path=PATH_TO_EMBEDDINGS_DB)
    # body_collection = chroma_client.create_collection(
    #     PATH_TO_EMBEDDING_FILE_TEMPLETE.format(
    #         model_name=embedder_pretty_name, data_name=data_name
    #     )
    # )
    # for index, embedding in enumerate(X):
    #     body_collection.add(
    #         ids=[data_ids[index]],
    #         documents=[],
    #         embeddings=[embedding],
    #         metadatas=[{"model": embedder_pretty_name, "id": data_ids[index]}],
    #     )


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
    with open(PATH_TO_EMBEDDERS_LIST, "r") as file:
        embedders_list = json.load(file)
    return embedders_list


@click.command()
@click.option("--embedder", default=None, help="the name of the embedder to be used (if None, all embedders from embedders_list.json will be used)")
@click.option("--truncate_10k", is_flag=True, help="truncate the dataset to 10'000 posts")
def main(embedder, truncate_10k):
    if truncate_10k:
        print("Attention: Debug truncation is enabled. Only 10'000 posts will be processed!")

    print("Loading embedders list...")
    embedders_list = get_embedders_list()
    if embedder is not None:
        embedders_list = {embedder: embedders_list[embedder]}

    print("Loading data...")
    questions, answers = parse_datasets(filepath=PATH_TO_DATASET_POSTS)

    print("Choosing device...")
    # Check if CUDA is available
    device = get_optimal_device()
    print(f"device: {device}")

    # Create ebmeddings folder if it doesn't exist
    print("Creating embeddings folder...")
    Path(PATH_TO_EMBEDDINGS).mkdir(parents=True, exist_ok=True)
    Path(PATH_TO_EMBEDDINGS_DB).mkdir(parents=True, exist_ok=True)

    bodies = questions["Body"].tolist()
    data_ids = questions["Id"].tolist()

    if truncate_10k:
        bodies = bodies[:10000]
        data_ids = data_ids[:10000]

    for model_pretty_name, model_name in embedders_list.items():
        try_dump_embeddings(
            data_list=bodies,
            data_ids=data_ids,
            data_name="bodies",
            model_name=model_name,
            model_pretty_name=model_pretty_name,
            device=device,
        )


if __name__ == "__main__":
    main()
