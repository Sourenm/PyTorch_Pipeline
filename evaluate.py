import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, loader, criterion, device=None, return_metrics=False, amp=True, crit_mode='loss'):
    """
    Evaluates the model on the provided data loader using the given criterion.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader for the dataset to evaluate on.
        criterion (callable): Loss function to compute evaluation loss.
        device (str, optional): Device to use for evaluation ('cuda' or 'cpu'). Defaults to auto-detect.
        return_metrics (bool, optional): If True, returns additional regression metrics. Defaults to False.
        amp (bool, optional): If True, uses Automatic Mixed Precision (AMP) during evaluation. Defaults to True.
        crit_mode (str, optional): Criterion to evaluate performance ('loss' or 'accuracy'). Defaults to 'loss'.

    Returns:
        tuple:
            - y_pred (np.ndarray): Predicted outputs.
            - y_true (np.ndarray): True target values.
            - correct (int): Number of correct predictions (only if crit_mode='accuracy').
            - total (int or float): Total number of samples or total loss.
            - metrics (dict, optional): Additional regression metrics (only if return_metrics is True and crit_mode='loss').
    """    
    device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    model.eval()
    model.to(device)
    preds, targets = [], []
    correct = 0
    total = 0

    if amp:
        scaler = torch.cuda.amp.GradScaler()
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)

                with torch.cuda.amp.autocast():
                    pred = model(xb)
                    loss = criterion(pred, yb)
                preds.append(pred.cpu())
                targets.append(yb.cpu())                    
                
                if crit_mode == 'loss':
                    total += loss.item()
                else:
                    _, predicted = torch.max(pred, 1)
                    total += yb.size(0)
                    correct += (predicted == yb).sum().item()    

    else:
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                preds.append(pred.cpu())
                targets.append(yb.cpu())
            
            if crit_mode == 'loss':
                total += loss.item()
            else:
                _, predicted = torch.max(pred, 1)
                total += yb.size(0)
                correct += (predicted == yb).sum().item()  

    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(targets).numpy()

    if crit_mode == 'loss' and y_pred.shape[1] == 1:
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print("MSE:", mse)
        print("MAE:", mae)
        print("R²:", r2)

        if return_metrics:
            return y_pred, y_true, correct, total, {"mse": mse, "mae": mae, "r2": r2}
    return y_pred, y_true, correct, total

def plot_predictions_loss(y_true, y_pred):
    """
    Plots a scatter plot of true vs predicted values for regression tasks.

    Args:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Predicted values from the model.

    Returns:
        None
    """    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, color='dodgerblue', alpha=0.6, label='Predictions')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Perfect Prediction')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("True vs Predicted Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_predictions_accuracy(train_accs, test_accs):
    """
    Plots training and testing accuracy across epochs.

    Args:
        train_accs (list or array-like): List of training accuracy values per epoch.
        test_accs (list or array-like): List of testing accuracy values per epoch.

    Returns:
        None
    """    
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(test_accs, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

