import torch
import torch.nn as nn
from evaluate import evaluate_model
from prefect import task
from config import TRAIN_CONFIG

LOG_PRINTS = TRAIN_CONFIG.get("enable_logging", False)

@task(name="Train One Epoch", log_prints=LOG_PRINTS)
def train_model_one_epoch(model, train_loader, val_loader, criterion, optimizer, device=None, amp=False, crit_mode='loss'):
    """
    Trains the model for one epoch and evaluates it on the validation set.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        criterion (callable): Loss function to compute training and evaluation loss.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        device (str, optional): Device to use for training ('cuda' or 'cpu'). Defaults to auto-detect.
        amp (bool, optional): If True, uses Automatic Mixed Precision (AMP) for faster training on GPUs.
        crit_mode (str, optional): Criterion for tracking performance ('loss' or 'accuracy'). Defaults to 'loss'.

    Returns:
        tuple:
            - acc_train (float): Training performance metric (loss or accuracy).
            - acc_eval (float): Validation performance metric (loss or accuracy).
    """    
    device = torch.device("cuda" if torch.cuda.is_available() and device == "cuda" else "cpu")
    use_amp = amp and device.type == "cuda"

    
    # Initializing
    model.to(device)
    train_losses = []
    eval_losses = []    
    correct_train, total_train = 0, 0
    correct_val, total_val = 0, 0
    acc_train = 0
    acc_eval = 0

    # Training Loop
    model.train()
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                pred = model(xb)
                loss = criterion(pred, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()            
            
            if crit_mode == 'loss':
                total_train += loss.item()
            else:
                _, predicted = torch.max(pred, 1)
                total_train += yb.size(0)
                correct_train += (predicted == yb).sum().item()    

    else:
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()   
            
            if crit_mode == 'loss':
                total_train += loss.item()
            else:
                _, predicted = torch.max(pred, 1)
                total_train += yb.size(0)
                correct_train += (predicted == yb).sum().item()                     
    
    if crit_mode != 'loss':
        acc_train = correct_train / total_train
    else:
        acc_train = total_train / len(train_loader)

    _, __, correct_val, total_val = evaluate_model(model, val_loader, criterion, device, False, amp, crit_mode)       

    if crit_mode != 'loss':
        acc_eval = correct_val / total_val
    else:
        acc_eval = total_val / len(val_loader)


    return acc_train, acc_eval

