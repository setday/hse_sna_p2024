import torch
import torch.nn as nn
import torch.optim as optim

import lightning as L

import numpy as np

from sklearn.metrics import r2_score, root_mean_squared_error


class SolverEvaluator(nn.Module):
    def __init__(self, input_size):
        super(SolverEvaluator, self).__init__()
        print(f"Input size: {input_size}")
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x

    def evaluate(self, data_loader, device: str = "cpu"):
        self.eval()

        all_y_true = []
        all_y_pred = []

        with torch.no_grad():
            for inputs, probabilities in data_loader:
                inputs = inputs.to(device)
                outputs = self(inputs).flatten()

                all_y_pred.append(outputs.cpu().numpy())
                all_y_true.append(probabilities.numpy())

        all_y_pred_np = np.concatenate(all_y_pred)
        all_y_true_np = np.concatenate(all_y_true)

        cross_entropy = nn.CrossEntropyLoss()(
            torch.tensor(all_y_pred_np), torch.tensor(all_y_true_np)
        )
        r2 = r2_score(all_y_true_np, all_y_pred_np)
        brier_score = root_mean_squared_error(all_y_true_np, all_y_pred_np)
        l1 = nn.L1Loss()(torch.tensor(all_y_pred_np), torch.tensor(all_y_true_np))

        print(f"Cross entropy: {cross_entropy:.4f}")
        print(f"R2 Score: {r2:.4f}")
        print(f"root MSE: {brier_score:.4f}")
        print(f"L1 Loss: {l1:.4f}")


class SolverEvaluatorLightningModule(L.LightningModule):
    def __init__(self, input_size, lr=1e-3):
        super().__init__()

        self.model = SolverEvaluator(input_size)
        self.loss_fn = nn.BCELoss()

        self.val_loss = []
        self.train_loss = []
        self.valid_accs = []

        self.lr = lr

    def set_lr(self, lr):
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        p = self.model(x).flatten()
        loss = self.loss_fn(p, y)
        self.train_loss.append(loss.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        p = self.model(x).flatten()
        loss = self.loss_fn(p, y)
        self.valid_accs.append(loss.detach())
        return {}

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

    def on_train_epoch_end(self):
        epoch_loss = torch.stack(self.train_loss).mean()
        print(
            f"Epoch {self.trainer.current_epoch},",
            f"train_loss: {epoch_loss.item():.8f}",
        )
        # don't forget to clear the saved losses
        self.train_loss.clear()

    def on_validation_epoch_end(self):
        epoch_accs = torch.tensor(self.valid_accs).float().mean()
        print(
            f"Epoch {self.trainer.current_epoch},",
            f"valid_accs: {epoch_accs.item():.8f}",
        )
        # don't forget to clear the saved accuracies
        self.valid_accs.clear()
