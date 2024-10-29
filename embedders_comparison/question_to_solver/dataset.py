import torch
from torch.utils.data import Dataset


class QuestionSolverDataset(Dataset):
    def __init__(self, embeddings, probabilities):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.probabilities = torch.tensor(probabilities, dtype=torch.float32)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.probabilities[idx]
