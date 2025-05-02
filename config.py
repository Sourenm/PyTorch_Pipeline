from data import Data
from evaluate import evaluate_model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model_factory import build_model
from model_factory.utils import init_weights
from functools import partial
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from prefect import task
from prefect.tasks import task_input_hash
from datetime import timedelta
import pandas as pd
import numpy as np
import os
import json

with open("configs/training/train_config.json") as f:
    TRAIN_CONFIG = json.load(f)

USE_CACHE = TRAIN_CONFIG.get("use_cache_on_tasks", False)
LOG_PRINTS = TRAIN_CONFIG.get("enable_logging", False)

@task(
    name="Get Dataset from Config",
    cache_key_fn=task_input_hash if USE_CACHE else None,
    cache_expiration=timedelta(days=1) if USE_CACHE else None,
    log_prints=LOG_PRINTS
)
def get_dataset_from_config(dataset_config, train=True, device='cuda'):
    dataset_map = {
        "cifar10": datasets.CIFAR10,
        "cifar100": datasets.CIFAR100,
        "mnist": datasets.MNIST,
        "fashionmnist": datasets.FashionMNIST,
        "emnist": datasets.EMNIST,
        "svhn": datasets.SVHN,
        "imagenet": datasets.ImageNet,
        "celeba": datasets.CelebA,
        "stl10": datasets.STL10,
        "caltech101": datasets.Caltech101,
        "caltech256": datasets.Caltech256,
        "eurosat": datasets.EuroSAT,
        "fer2013": datasets.FER2013,
        "gtsrb": datasets.GTSRB,
        "omniglot": datasets.Omniglot,
        "sbdataset": datasets.SBDataset,
        "usps": datasets.USPS
    }

    if dataset_config["type"] == "torchvision":
        name = dataset_config["name"].lower()
        normalize_mean = dataset_config.get("normalize_mean", [0.5])
        normalize_std = dataset_config.get("normalize_std", [0.5])

        if name not in dataset_map:
            raise ValueError(f"Unsupported torchvision dataset: {name}")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std)
        ])
        if name == "svhn":
            return dataset_map[name](root=dataset_config["root"], split='train' if train else 'test', download=True, transform=transform)
        return dataset_map[name](root=dataset_config["root"], train=train, download=True, transform=transform)

    elif dataset_config["type"] == "custom":
        format = dataset_config["format"].lower()
        if format == "csv":
            df = pd.read_csv(dataset_config["path"])
            label_col = dataset_config.get("label_column", "label")
            X = df.drop(columns=[label_col]).values
            y = df[label_col].values
        elif format == "npy":
            X = np.load(dataset_config["X_path"])
            y = np.load(dataset_config["y_path"])
        elif format == "pt":
            data = torch.load(dataset_config["path"])
            if isinstance(data, dict):
                X, y = data["X"], data["y"]
            elif isinstance(data, (list, tuple)) and len(data) == 2:
                X, y = data
            else:
                raise ValueError("Expected a tuple or dict with 'X' and 'y' keys in the .pt file.")
        else:
            raise ValueError(f"Unsupported custom format: {format}")

        return get_data(X, y, device)

    else:
        raise ValueError(f"Unknown dataset type: {dataset_config['type']}")

@task(
    name="Get Data (Tensor Conversion)",
    cache_key_fn=task_input_hash if USE_CACHE else None,
    cache_expiration=timedelta(days=1) if USE_CACHE else None,
    log_prints=LOG_PRINTS
)
def get_data(X, y, device):
    """
    converts non tensor X, y data to tensor data
    """
    return Data(X, y, device)

@task(
    name="Get Dataloader",
    cache_key_fn=task_input_hash if USE_CACHE else None,
    cache_expiration=timedelta(days=1) if USE_CACHE else None,
    log_prints=LOG_PRINTS
)
def get_dataloader(dataset, batch_size=32, shuffle=True):
    """
    build DataLoaders from input using batch_size and shuffle parameters

    Args:
        dataset (tensor): input dataset
        batch_size (int): the batch size
        shuffle (bool): shuffle variable

    Returns:
        torch.utils.data.DataLoader: initialized PyTorch DataLoader
    """
    return DataLoader(dataset, batch_size, shuffle)

@task(
    name="Get Model",
    cache_key_fn=task_input_hash if USE_CACHE else None,
    cache_expiration=timedelta(days=1) if USE_CACHE else None,
    log_prints=LOG_PRINTS
)
def get_model(model_type, layer_specs, device="cuda", method="xavier"):
    """
    Builds and initializes a model given a high-level config.

    Args:
        model_type (str): Type of the model to build (e.g., cnn, rnn, lstm, transformer).
        layer_specs (list): List of dictionaries specifying each layer.
        device (str): "cuda" or "cpu".
        method (str): Weight initialization method (e.g., xavier, kaiming).

    Returns:
        torch.nn.Module: Initialized PyTorch model moved to appropriate device.
    """
    device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")     
    model = build_model(model_type=model_type, layer_specs=layer_specs)

    # Handle device logic
    model = model.to(device)

    # Initialize weights
    model.apply(partial(init_weights, method=method))
    return model

@task(
    name="Evaluate Model",
    log_prints=LOG_PRINTS
)
def get_eval(model, loader, device=None, return_metrics=False, amp=True, crit_mode='loss'):
    """
    Evaluates a PyTorch model on a given dataset loader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader containing the evaluation dataset.
        device (str or torch.device, optional): Device to perform evaluation on ('cuda' or 'cpu').
        return_metrics (bool, optional): Whether to return additional metrics like MSE.
        amp (bool, optional): Whether to use automatic mixed precision (AMP) for inference.
        crit_mode (str, optional): Evaluation mode, either 'loss' or 'accuracy'.

    Returns:
        Depends on `return_metrics` and `crit_mode`. Can be just loss or accuracy, or a tuple with more stats.
    """    
    return evaluate_model(model, loader, device, return_metrics, amp, crit_mode)

def get_loss(loss_name="cross_entropy", **kwargs):
    """
    Returns the appropriate PyTorch loss function given its name.

    Args:
        loss_name (str): The name of the loss function. Options include:
            'mse', 'cross_entropy', 'l1', 'nll', 'bce', 'bce_with_logits',
            'smooth_l1', 'hinge'.
        **kwargs: Additional keyword arguments passed to the loss constructor.

    Returns:
        torch.nn.modules.loss._Loss: The initialized loss function.

    Raises:
        ValueError: If the specified loss_name is not supported.
    """    
    loss_dict = {
        "mse": nn.MSELoss,
        "cross_entropy": nn.CrossEntropyLoss,
        "l1": nn.L1Loss,
        "nll": nn.NLLLoss,
        "bce": nn.BCELoss,
        "bce_with_logits": nn.BCEWithLogitsLoss,
        "smooth_l1": nn.SmoothL1Loss,
        "hinge": nn.HingeEmbeddingLoss,
    }
    if loss_name not in loss_dict:
        raise ValueError(f"Unsupported loss function: {loss_name}")
    return loss_dict[loss_name](**kwargs)

def get_optim(optimizer_name="adam", model_params=None, lr=0.001, **kwargs):
    """
    Returns the appropriate PyTorch optimizer given its name and model parameters.

    Args:
        optimizer_name (str): The name of the optimizer. Options include:
            'adam', 'sgd', 'rmsprop', 'adagrad', 'adamw', 'adamax', 'nadam'.
        model_params (iterable): Parameters of the model to optimize.
        lr (float): Learning rate for the optimizer.
        **kwargs: Additional keyword arguments passed to the optimizer constructor.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.

    Raises:
        ValueError: If the specified optimizer_name is not supported or model_params is None.
    """    
    optimizer_dict = {
        "adam": optim.Adam,
        "sgd": optim.SGD,
        "rmsprop": optim.RMSprop,
        "adagrad": optim.Adagrad,
        "adamw": optim.AdamW,
        "adamax": optim.Adamax,
        "nadam": optim.NAdam,
    }
    if optimizer_name not in optimizer_dict:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    if model_params is None:
        raise ValueError("model_params must be provided to get_optim()")
    return optimizer_dict[optimizer_name](model_params, lr=lr, **kwargs)
