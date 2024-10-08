import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from preprocess_data import create_X_y
from dataset import QuestionTagDataset
from embedders_comparison.question_to_tags.linear_model import TagPredictorNN


RANDOM_STATE = 1200

SPLIT_RATIO = 0.8

PATH_TO_DATASET_POSTS = "../../data/Posts.xml"

tqdm.pandas()


def train_and_evaluate_model(
    embedder_name: str, epochs: int, dataset_path: str, model_class: type
):
    print("Preparing data...")

    X, y = create_X_y(filepath=dataset_path, embedder_name=embedder_name)

    # Transorm 'y' to the appropriate format
    mlb = MultiLabelBinarizer()
    y_full_binary = mlb.fit_transform(y)

    # Shuffle and split the data
    X_train, X_test, y_train_binary, y_test_binary = train_test_split(
        X,
        y_full_binary,
        train_size=SPLIT_RATIO,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_dataset = QuestionTagDataset(X_train, y_train_binary)
    test_dataset = QuestionTagDataset(X_test, y_test_binary)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    input_size = X_train.shape[1]  # input dimension
    num_tags = len(np.unique(np.array(y).flatten()))  # output dimension

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    model = model_class(input_size, num_tags).to(device)

    # TRAINING
    print("Training...")
    model.training_loop(train_loader, num_epochs=epochs, device=device, verbose=True)
    print("Model evaluation...")
    # EVALUATION
    print("===TEST===")
    model.evaluate(test_loader, threshold=0.5, device=device)
    print("===TRAIN===")
    model.evaluate(train_loader, threshold=0.5, device=device)


if __name__ == "__main__":
    train_and_evaluate_model(
        embedder_name="MiniLM3",
        epochs=23,
        dataset_path=PATH_TO_DATASET_POSTS,
        model_class=TagPredictorNN,
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
