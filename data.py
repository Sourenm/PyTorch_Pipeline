import torch
from torch.utils.data import Dataset

# Dataset
class Data(Dataset):
    """
    A custom PyTorch Dataset for handling features and labels on a specified device.

    Args:
        X (array-like or torch.Tensor): Input features. Will be converted to torch.FloatTensor if not already a tensor.
        y (array-like or torch.Tensor): Target labels. Will be converted to torch.FloatTensor if not already a tensor.
        device (str): Device to store the tensors on. Defaults to 'cuda' if available, otherwise falls back to 'cpu'.

    Attributes:
        X (torch.Tensor): Tensor containing input features on the specified device.
        y (torch.Tensor): Tensor containing target labels on the specified device.
    """

    def __init__(self, X, y, device='cuda'):
        device = torch.device("cuda" if (device.lower() == "cuda" and torch.cuda.is_available()) else "cpu")     
        if not isinstance(X, torch.tensor):
            self.X = torch.tensor(X, dtype=torch.float32).to(device)
        else:
            self.X = X.to(device)
        if not isinstance(y, torch.tensor):
            self.y = torch.tensor(y, dtype=torch.float32).to(device)
        else:
            self.y = y.to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
