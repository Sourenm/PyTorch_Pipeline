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
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Load the CIFAR-10 training and test datasets
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create DataLoader instances for training and testing
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
```

### Building and Training the Model

In the main.py, we define a CNN model and train it on the CIFAR-10 dataset. Here is the key section:

```python
from config import get_model, get_dataloader, get_loss, get_optim
from train import train_model_one_epoch
from evaluate import evaluate_model, plot_predictions_accuracy

# CNN model configuration
cnn_config = {
    "model_type": "cnn",
    "input_shape": (3, 32, 32),
    "layer_specs": [
        {"type": "conv2d", "in_channels": 3, "out_channels": 32, "kernel_size": 3, "padding": 1},
        {"type": "activation", "name": "relu"},
        {"type": "maxpool2d", "kernel_size": 2},

        {"type": "conv2d", "in_channels": 32, "out_channels": 64, "kernel_size": 3, "padding": 1},
        {"type": "activation", "name": "relu"},
        {"type": "maxpool2d", "kernel_size": 2},

        {"type": "conv2d", "in_channels": 64, "out_channels": 128, "kernel_size": 3, "padding": 1},
        {"type": "activation", "name": "relu"},
        {"type": "maxpool2d", "kernel_size": 2},

        {"type": "flatten"},
        {"type": "linear", "in_features": 128 * 4 * 4, "out_features": 256},
        {"type": "activation", "name": "relu"},
        {"type": "linear", "in_features": 256, "out_features": 10}
    ],
    "device": "cuda",
    "loss": "cross_entropy",
    "optimizer": "adam"    

}

# Build the model using the configuration
model = build_and_test(cnn_config)

# Create DataLoader instances (train_loader and test_loader from above)
train_loader = get_dataloader(train_dataset, batch_size=64, shuffle=True)
test_loader = get_dataloader(test_dataset, batch_size=64)

# Loss and optimizer
criterion = get_loss("cross_entropy")
optimizer = get_optim("adam", model.parameters(), lr=0.01)

# Training loop
epochs = 5
train_accs = []
test_accs = []

for epoch in range(epochs):
    train_acc, test_acc = train_model_one_epoch(model, train_loader, test_loader, criterion, optimizer, crit_mode='accuracy')
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    print(f"Epoch {epoch} | Train Acc: {train_accs[-1]*100:.2f}% | Test Acc: {test_accs[-1]*100:.2f}%")

# Save the trained model
torch.save(model.state_dict(), "cnn_cifar10.pth")

# Plotting the accuracy
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