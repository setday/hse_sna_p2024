from tqdm import tqdm

from sklearn.metrics import accuracy_score

from catboost import CatBoostClassifier


class SolverPredictor(CatBoostClassifier):
    def __init__(self, max_iter: int = 1000):
        super().__init__(iterations=max_iter, loss_function='MultiClass', random_seed=113, task_type='GPU')

    def evaluate(self, X_test, y_test):
        self.eval()

        all_y_true = []
        all_y_pred = []

        for x, y in tqdm(zip(X_test, y_test)):
            y_pred = self.predict(x)

            all_y_pred.append(y_pred)
            all_y_true.append(y_pred if y_pred in y else y[0])

        accuracy = accuracy_score(all_y_true, all_y_pred)

        print(f"Accuracy: {accuracy:.4f}")
