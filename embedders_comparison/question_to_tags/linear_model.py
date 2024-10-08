import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score


class TagPredictorNN(nn.Module):
    def __init__(self, input_size, num_tags):
        super(TagPredictorNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, num_tags)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

    def training_loop(
        self, dataloader, num_epochs=50, device: str = "cpu", verbose: bool = True
    ):
        loss_fn = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            for inputs, labels in dataloader:
                inputs, labels = (
                    inputs.to(device),
                    labels.to(device),
                )  # Move to GPU (if available)
                optimizer.zero_grad()

                # Forward pass
                outputs = self(inputs)
                loss = loss_fn(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            if verbose:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/10:.4f}")
            running_loss = 0.0

    def evaluate(self, data_loader, threshold=0.5, device: str = "cpu"):
        self.eval()

        all_y_true = []
        all_y_pred = []

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = (
                    X_batch.float().to(device),
                    y_batch.float().to(device),
                )  # Move to GPU (if available)

                # Forward pass
                y_pred = self(X_batch)

                # Apply threshold to get binary predictions
                y_pred_binary = (y_pred > threshold).float().cpu().numpy()
                y_true_binary = y_batch.cpu().numpy()

                all_y_pred.append(y_pred_binary)
                all_y_true.append(y_true_binary)

        # Concatenate all batches
        all_y_pred_np = np.vstack(all_y_pred)
        all_y_true_np = np.vstack(all_y_true)

        # Compute metrics
        jaccard = jaccard_score(
            all_y_true_np, all_y_pred_np, average="samples", zero_division=0
        )
        precision = precision_score(
            all_y_true_np, all_y_pred_np, average="samples", zero_division=0
        )
        recall = recall_score(
            all_y_true_np, all_y_pred_np, average="samples", zero_division=0
        )
        f1 = f1_score(all_y_true_np, all_y_pred_np, average="samples", zero_division=0)

        print(f"Jaccard Index: {jaccard:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
