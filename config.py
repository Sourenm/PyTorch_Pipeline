from data import Data
from evaluate import evaluate_model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model_factory import build_model
from model_factory.utils import init_weights
from functools import partial


def get_data(X, y, device):
    """
    converts non tensor X, y data to tensor data
    """
    return Data(X, y, device)

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
