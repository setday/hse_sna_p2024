import sys

from tqdm import tqdm

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

from preprocess_data import create_multytraget_X_y, create_besttraget_X_y
from boost_model import SolverPredictor
from linear_model import SolverEvaluator

from dataset import QuestionSolverDataset


RANDOM_STATE = 1200

SPLIT_RATIO = 0.8

PATH_TO_DATASET_POSTS = "../../data/Posts.xml"

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


def train_and_evaluate_multytarget_model(
    embedder_name: str, dataset_path: str, truncate_100: bool = False
):
    print("Preparing data for multy target...")

    epochs = 1000

    X, y = create_multytraget_X_y(filepath=dataset_path, embedder_name=embedder_name)

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

    model = SolverEvaluator(input_size).to(device)

    # TRAINING
    print("Training...")
    model.training_loop(train_loader, num_epochs=epochs, device=device, verbose=True)
    print("Model evaluation...")
    # EVALUATION
    print("===TEST===")
    model.evaluate(test_loader, device=device)
    print("===TRAIN===")
    model.evaluate(train_loader, device=device)


if __name__ == "__main__":
    mode = "best_target"
    
    if "best_target" in sys.argv:
        mode = "best_target"
    elif "multy_target" in sys.argv:
        mode = "multy_target"
        
    truncate_100 = "truncate_100" in sys.argv
    if truncate_100:
        print("Attention: Debug truncation is enabled. Only 100 posts will be processed!")

    if mode == "best_target":
        train_and_evaluate_besttarget_model(
            embedder_name="MiniLM3",
            dataset_path=PATH_TO_DATASET_POSTS,
            truncate_100=truncate_100,
        )
    elif mode == "multy_target":
        train_and_evaluate_multytarget_model(
            embedder_name="MiniLM3",
            dataset_path=PATH_TO_DATASET_POSTS,
            truncate_100=truncate_100,
        )

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
