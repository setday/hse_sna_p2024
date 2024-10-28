import sys

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from preprocess_data import create_multytraget_X_y, create_besttraget_X_y
from boost_model import SolverPredictor


RANDOM_STATE = 1200

SPLIT_RATIO = 0.8

PATH_TO_DATASET_POSTS = "../../data/Posts.xml"

tqdm.pandas()


def train_and_evaluate_besttarget_model(
    embedder_name: str, dataset_path: str
):
    print("Preparing data for best target...")

    X, y = create_besttraget_X_y(filepath=dataset_path, embedder_name=embedder_name)

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
    embedder_name: str, dataset_path: str
):
    print("Preparing data for multy target...")

    X, y = create_multytraget_X_y(filepath=dataset_path, embedder_name=embedder_name)

    # Shuffle and split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=SPLIT_RATIO,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    print("Model is not implemented yet.", file=sys.stderr)
    exit(1)
    # model = SolverPredictor()
    model = ...

    # TRAINING
    print("Training...")
    model.fit(X_train, y_train)
    print("Model evaluation...")
    # EVALUATION
    print("===TEST===")
    model.evaluate(X_test, y_test)
    print("===TRAIN===")
    model.evaluate(X_train, y_train)


if __name__ == "__main__":
    mode = "best_target"
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]

    if mode == "best_target":
        train_and_evaluate_besttarget_model(
            embedder_name="MiniLM3",
            dataset_path=PATH_TO_DATASET_POSTS,
        )
    elif mode == "multy_target":
        train_and_evaluate_multytarget_model(
            embedder_name="MiniLM3",
            dataset_path=PATH_TO_DATASET_POSTS,
        )

# Epoch [23/23], Loss: 0.3460

# ===TEST===
# Jaccard Index: 0.2904
# Precision: 0.5540
# Recall: 0.3341
# F1 Score: 0.3871
# ===TRAIN===
# Jaccard Index: 0.3748
# Precision: 0.6714
# Recall: 0.4134
# F1 Score: 0.4808
