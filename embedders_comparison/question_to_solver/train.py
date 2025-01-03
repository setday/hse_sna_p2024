import sys
import json

import click

from tqdm import tqdm

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

import lightning as L

from preprocess_data import create_multytarget_X_y, create_besttraget_X_y
from boost_model import SolverPredictor
from linear_model import SolverEvaluatorLightningModule as SolverEvaluator

from dataset import QuestionSolverDataset


RANDOM_STATE = 1200

SPLIT_RATIO = 0.8

PATH_TO_DATASET_POSTS = "./../data/Posts.xml"
PATH_TO_EMBEDDERS_LIST = "./embedders_list.json"

tqdm.pandas()


def train_and_evaluate_besttarget_model(
    embedder_name: str, dataset_path: str, truncate_100: bool = False
):
    print("Preparing data for best target...")

    X, y = create_besttraget_X_y(filepath=dataset_path, embedder_name=embedder_name)

    if truncate_100:
        X = X[:100]
        y = y[:100]

    # Shuffle and split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=SPLIT_RATIO,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    model = SolverPredictor()

    # TRAINING
    print("Training...")
    model.fit(X_train, y_train)
    print("Model evaluation...")
    # EVALUATION
    print("===TEST===")
    model.evaluate(X_test, y_test)
    print("===TRAIN===")
    model.evaluate(X_train, y_train)

    return model


def train_and_evaluate_multytarget_model(
    embedder_name: str, dataset_path: str, truncate_100: bool = False
):
    print("Preparing data for multy target...")

    epochs = 1000

    X, y = create_multytarget_X_y(filepath=dataset_path, embedder_name=embedder_name)

    if truncate_100:
        X = X[:100]
        y = y[:100]

    # Shuffle and split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=SPLIT_RATIO,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_dataset = QuestionSolverDataset(X_train, y_train)
    test_dataset = QuestionSolverDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    input_size = X_train.shape[1]  # input dimension

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SolverEvaluator(input_size)

    # TRAINING
    print("Training...")
    trainer = L.Trainer(accelerator="auto", max_epochs=epochs, enable_progress_bar=False, logger=False)
    trainer.fit(model, train_loader, test_loader)
    print("Model evaluation...")
    model = model.model.to(device)
    # EVALUATION
    print("===TEST===")
    model.evaluate(test_loader, device=device)
    print("===TRAIN===")
    model.evaluate(train_loader, device=device)

    return model


def get_embedders_list():
    with open(PATH_TO_EMBEDDERS_LIST, "r") as file:
        embedders_list = json.load(file)
    return embedders_list


@click.command()
@click.option("--target", help="use 'best' target to train the model that predicts the best answer / use 'multy' target to train the model that predicts the probability of solver to answer the question")
@click.option("--embedder", default="MiniLM3", help="the name of the embedder to be used")
@click.option("--truncate_100", is_flag=True, help="truncate the dataset to 100 posts")
@click.option("--save_model", is_flag=True, help="save the model")
def main(target, embedder, truncate_100, save_model):
    if not target or target not in ["best", "multy"]:  # check if the target is valid
        print("Invalid target. Use 'best' or 'multy'")
        sys.exit(1)

    print("Loading embedders list...")
    if embedder is None:
        embedders_list = get_embedders_list().keys()
    else:
        embedders_list = [embedder]

    train_eval_fn = (
        train_and_evaluate_besttarget_model
        if target == "best"
        else train_and_evaluate_multytarget_model
    )

    for embedder in embedders_list:
        print(f"Training model for {embedder}...")

        model = train_eval_fn(
            embedder_name=embedder,
            dataset_path=PATH_TO_DATASET_POSTS,
            truncate_100=truncate_100,
        )

        if save_model:
            torch.save(model, f"{embedder}_{target}_model.pth")


if __name__ == "__main__":
    main()

# Epoch [23/23], Loss: 0.3460

# => Best target <=
# ===TEST===
# 20it [00:00, 1020.66it/s]
# Accuracy: 0.1000
# ===TRAIN===
# 80it [00:00, 1214.21it/s]
# Accuracy: 1.0000

# => Multy target <=
# ===TEST===
# ===TRAIN===
