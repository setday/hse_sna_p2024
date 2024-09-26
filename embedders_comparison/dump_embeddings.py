import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer
from question_to_tags.preprocess_data import parse_dataset


def dump_embeddings(models: dict[str, str], dataset: pd.DataFrame, column_name: str = "Body") -> None:

    for model_pretty_name, model_official_name in models.items():
        print(model_pretty_name, model_official_name)
        # Check if embeddings are already dumped (using this model)
        embeddings_file = Path(f"embeddings/{model_pretty_name}_body.obj")
        if embeddings_file.is_file():  # file exists
            print(f"Embeddings for {model_pretty_name} are already dumped.")
            continue

        # Download a model from Hugging Face using its name
        embedder = SentenceTransformer(model_official_name)

        bodies = dataset[column_name].tolist()

        X = []
        for body in tqdm(bodies, desc="Encoding posts"):
            # Encode each 'body' and append it to X
            encoded_body = embedder.encode(body)
            X.append(encoded_body)

        X = np.array(X)
        filehandler = open(f"embeddings/{model_pretty_name}_{column_name.lower()}.obj", "wb")
        pickle.dump(X, filehandler)
        filehandler.close()


if __name__ == "__main__":
    models = {
        'Albert': 'paraphrase-albert-small-v2',
        'Roberta': 'all-distilroberta-v1',
        'DistilBert': 'multi-qa-distilbert-cos-v1',
        'MiniLM1': 'all-MiniLM-L6-v2',
        'MiniLM2': 'all-MiniLM-L12-v2',
        'MiniLM3': 'paraphrase-MiniLM-L3-v2'
    }
    print("Loading data...")
    posts = parse_dataset(filepath="../data/Posts.xml")
    dump_embeddings(models=models, dataset=posts, column_name="Body")
