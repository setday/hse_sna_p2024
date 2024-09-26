from preprocess_data import create_X_y
from dataset import QuestionTagDataset
from model import TagPredictorNN
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
tqdm.pandas()


def train_and_evaluate_model(embedder_name: str = "MiniLM3"):
    print("Preparing data...")

    X, y = create_X_y(filepath="../../data/Posts.xml", embedder_name=embedder_name)
    train_size = int(0.8 * len(y))
    # Shuffle and split the data
    X, y = shuffle(X, y, random_state=1200)
    X_train, X_test = X[:train_size], X[train_size:]

    # Transorm 'y' to the appropriate format
    mlb = MultiLabelBinarizer()
    y_full_binary = mlb.fit_transform(y)
    y_train_binary = y_full_binary[:train_size]
    y_test_binary = y_full_binary[train_size:]

    train_dataset = QuestionTagDataset(X_train, y_train_binary)
    test_dataset = QuestionTagDataset(X_test, y_test_binary)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    input_size = X_train.shape[1]  # input dimension
    num_tags = len(list(set([x for xs in y for x in xs])))  # output dimension

    model = TagPredictorNN(input_size, num_tags)
    # TRAINING
    print("Training...")
    model.training_loop(train_loader, num_epochs=2, verbose=True)
    print("Model evaluation...")
    # EVALUATION
    print("===TEST===")
    model.evaluate(test_loader)
    print("===TRAIN===")
    model.evaluate(train_loader)


if __name__ == "__main__":
    train_and_evaluate_model(embedder_name="MiniLM3")
