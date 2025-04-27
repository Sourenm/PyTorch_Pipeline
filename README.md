# 🔧 Modular Deep Learning Pipeline with PyTorch

This repository provides a modular, extensible deep learning pipeline built with PyTorch. It supports multiple model architectures—including **MLPs**, **CNNs**, **RNNs**, **LSTMs**, and **Transformers**—defined through high-level configuration specs. The framework allows for easy training, evaluation, and future deployment, with a focus on clarity, scalability, and customization.

---

## 🚀 Features

- ✅ High-level model configuration using Python dictionaries
- ✅ Modular support for CNNs, RNNs, LSTMs, MLPs, and Transformers
- ✅ Custom weight initialization (Xavier, Kaiming)
- ✅ Custom loss and optimizer factory modules
- ✅ Training and evaluation pipeline with AMP support
- ✅ Accuracy and regression metric evaluation
- ✅ Matplotlib-based plotting for training curves
- ✅ Includes CIFAR-10 example in `main.py`

---

## 📁 Project Structure
```bash
├── data/ # Dataset storage
├── loss_factory/ # Custom loss implementations
├── model_factory/ # Modular model builders
  ├── cnn.py
  ├── rnn.py
  ├── lstm.py
  ├── transformer.py
  └── utils.py # Utility for validation and config code generation
├── optim_factory/ # Custom optimizers
├── config.py # API for model/loss/optim/dataloader instantiation
├── data.py # Data utility functions
├── evaluate.py # Evaluation and metric utilities
├── train.py # Training loop
└── main.py # Sample usage with CIFAR-10
```
---

## 🧪 Example Usage (CIFAR-10 CNN)

This example demonstrates how to use the CNN architecture on the **CIFAR-10** dataset. You can find this in the `main.py` file.

### Downloading and Preparing the CIFAR-10 Dataset

To get started with the CIFAR-10 dataset, simply run the following in your Python environment. The dataset will be automatically downloaded from the internet if not already available:

```python
import torch
import torchvision
import torchvision.transforms as transforms

# Transforms (normalizing to [-1, 1] for tanh/relu if needed)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(dataset_config["normalize_mean"], dataset_config["normalize_std"])
])

# Load the CIFAR-10 training and test datasets
dataset_name = dataset_config.get("name", "CIFAR10")

if dataset_name == "CIFAR10":
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
else:
    raise ValueError(f"Dataset {dataset_name} not supported yet.")

train_loader = get_dataloader(train_dataset, batch_size=train_config["batch_size"], shuffle=train_config["shuffle"])
test_loader = get_dataloader(test_dataset, batch_size=train_config["batch_size"])
```

### Building and Training the Model

In the main.py, we define read the necessary JSON files for model creation, training specs and dataset. Here is the key section:

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import json
from config import get_model, get_dataloader, get_loss, get_optim
from model_factory.utils import validate_layer_specs, config_to_code
from train import train_model_one_epoch
from evaluate import evaluate_model, plot_predictions_accuracy

# Load JSON configuration files
with open('configs/model/cnn_config.json', 'r') as f:
    model_config = json.load(f)

with open('configs/training/train_config.json', 'r') as f:
    train_config = json.load(f)

with open('configs/data/dataset_config.json', 'r') as f:
    dataset_config = json.load(f)

# --- Building Model ---

def build_and_test(config):
    try:
        device = torch.device("cuda" if (config["device"] == "cuda" and torch.cuda.is_available()) else "cpu")
        validate_layer_specs(config["model_type"], config["layer_specs"])
        model = get_model(config["model_type"], config["layer_specs"], device)
        print("--- Code for reproducibility ---")
        print(config_to_code(config))
        print("\n")
        print("Validated layer spec and model successfully ✅")
        return model
    except Exception as e:
        raise ValueError(f"❌ Error building {config['model_type'].upper()}: {str(e)}\n")

model = build_and_test(model_config)

# --- Preparing Dataset ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(dataset_config["normalize_mean"], dataset_config["normalize_std"])
])

dataset_name = dataset_config.get("name", "CIFAR10")

if dataset_name == "CIFAR10":
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
else:
    raise ValueError(f"Dataset {dataset_name} not supported yet.")

train_loader = get_dataloader(train_dataset, batch_size=train_config["batch_size"], shuffle=train_config["shuffle"])
test_loader = get_dataloader(test_dataset, batch_size=train_config["batch_size"])

# --- Building Loss and Optimizer ---
criterion = get_loss(train_config["loss_function"], reduction='mean')
optimizer = get_optim(
    train_config["optimizer"]["type"],
    model.parameters(),
    lr=train_config["optimizer"]["lr"],
    weight_decay=train_config["optimizer"].get("weight_decay", 0)
)

# --- Training Loop ---
epochs = train_config["epochs"]
train_accs = []
test_accs = []

for epoch in range(epochs):
    train_acc, test_acc = train_model_one_epoch(model, train_loader, test_loader, criterion, optimizer, crit_mode='accuracy')
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    print(f"Epoch {epoch} | Train Acc: {train_accs[-1]*100:.2f}% | Test Acc: {test_accs[-1]*100:.2f}%")

# --- Saving Model ---
torch.save(model.state_dict(), "trained_model.pth")

# --- Plotting Accuracy ---
plot_predictions_accuracy(train_accs, test_accs)
```
This code demonstrates the complete pipeline, including:
- Model Building: A CNN model is created based on the provided cnn_config.
- Data Loading: CIFAR-10 is used with transformations (normalization) applied.
- Training: The model is trained for 5 epochs, with training and test accuracies tracked.
- Evaluation: The final model performance is evaluated and plotted.

## 🔮 Future Work

The following enhancements are planned:

- 🔧 Full JSON-based Configuration:
  - All pipeline elements (model, dataset, optimizer, loss, hyperparameters) will be defined in a single JSON file for easy reproducibility and CLI execution.
  - Progress so far: supporting any form of datasets in `dataset_config.json` is left TODO
- ⚙️ Prefect Orchestration
  - Use Prefect for pipeline management, enabling robust data/compute task orchestration with retry logic, monitoring, and scalability.
- 📈 MLflow Integration
  - Integrate MLflow for experiment tracking, model versioning, and deployment support.

🧠 Requirements

- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- scikit-learn

```python
pip install -r requirements.txt
```

## 🤝 Contributing

Pull requests are welcome! This repository is structured to facilitate modular experimentation and prototyping in deep learning.