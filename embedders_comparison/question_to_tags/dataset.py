import torch
from torch.utils.data import Dataset


class QuestionTagDataset(Dataset):
    def __init__(self, embeddings, tags):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.tags = torch.tensor(tags, dtype=torch.float32)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.tags[idx]
