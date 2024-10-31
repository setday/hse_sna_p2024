import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from sklearn.metrics import r2_score


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

    def training_loop(
        self, data_loader, num_epochs=50, device: str = "cpu", verbose: bool = True
    ):
        loss_fn = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            for inputs, probabilities in data_loader:
                inputs, probabilities = (
                    inputs.to(device),
                    probabilities.to(device),
                )  # Move to GPU (if available)
                optimizer.zero_grad()

                # Forward pass
                outputs = self(inputs).flatten()
                loss = loss_fn(outputs, probabilities)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            if verbose and epoch % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/10:.4f}")
            running_loss = 0.0

    def evaluate(self, data_loader, device: str = "cpu"):
        self.eval()

        all_y_true = []
        all_y_pred = []

        with torch.no_grad():
            for inputs, probabilities in data_loader:
                inputs = inputs.to(device)  # Move to GPU (if available)

                # Forward pass
                outputs = self(inputs).flatten()

                # Apply threshold to get binary predictions
                outputs = outputs.cpu().numpy()
                probabilities = probabilities.numpy()

                all_y_pred.append(outputs)
                all_y_true.append(probabilities)

        # Concatenate all batches
        all_y_pred_np = np.concatenate(all_y_pred)
        all_y_true_np = np.concatenate(all_y_true)

        # Compute metrics
        cross_entropy = nn.CrossEntropyLoss()(
            torch.tensor(all_y_pred_np), torch.tensor(all_y_true_np)
        )
        r2 = r2_score(all_y_true_np, all_y_pred_np)
        brier_score = np.mean(np.square(all_y_true_np - all_y_pred_np))

        print(f"Cross entropy: {cross_entropy:.4f}")
        print(f"R2 Score: {r2:.4f}")
        print(f"Brier Score: {brier_score:.4f}")
