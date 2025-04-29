# 🔧 Modular Deep Learning Pipeline with PyTorch

This repository provides a modular, extensible deep learning pipeline built with PyTorch. It supports multiple model architectures—including **MLPs**, **CNNs**, **RNNs**, **LSTMs**, and **Transformers**—defined through high-level configuration specs. The framework allows for easy training, evaluation, and future deployment, with a focus on clarity, scalability, and customization.

---

## 🚀 Features

- ✅ High-level model configuration using JSON files
- ✅ Modular support for CNNs, RNNs, LSTMs, MLPs, and Transformers
- ✅ Custom weight initialization (Xavier, Kaiming)
- ✅ Custom loss and optimizer factory modules
- ✅ Training and evaluation pipeline with AMP support
- ✅ Accuracy and regression metric evaluation
- ✅ Matplotlib-based plotting for training curves
- ✅ Flexible dataset setup supporting all torchvision datasets

---

## 📁 Project Structure

\`\`\`bash
├── configs/
│   ├── model/
│   │   ├── rnn_config.json
│   │   ├── mlp_config.json
│   │   ├── transformer_config.json
│   │   ├── lstm_config.json
│   │   └── cnn_config.json
│   ├── training/
│   │   └── train_config.json
│   └── data/
│       └── dataset_config.json
├── data/ # Dataset storage
├── loss_factory/
├── model_factory/
├── optim_factory/
├── config.py
├── data.py
├── evaluate.py
├── train.py
└── main.py
\`\`\`

---

## 🧪 Example Usage: CIFAR-10 CNN

You can train a CNN on CIFAR-10 simply by running:

\`\`\`bash
python main.py
\`\`\`

### 🔧 \`dataset_config.json\` Format

The \`dataset_config.json\` supports both **torchvision sample datasets** and **custom datasets**.

#### ✅ For Torchvision Datasets (e.g. CIFAR-10):

\`\`\`json
{
  "type": "torchvision",
  "name": "CIFAR10",
  "train": true,
  "download": true,
  "normalize_mean": [0.5, 0.5, 0.5],
  "normalize_std": [0.5, 0.5, 0.5]
}
\`\`\`

#### ✅ For Custom Datasets (CSV/NPY/Images):

\`\`\`json
{
  "type": "custom",
  "name": "path/to/your/data",
  "format": "csv",
  "label_column": "target",
  "normalize_mean": [0.0],
  "normalize_std": [1.0]
}
\`\`\`

> Only datasets with both \`normalize_mean\` and \`normalize_std\` will be normalized. Omit these keys to skip normalization.

---

## 🏗️ Building and Training the Model

Key steps from \`main.py\`:
\`\`\`python
# Load configs
with open('configs/data/dataset_config.json') as f:
    dataset_config = json.load(f)

# Auto-select torchvision datasets via type field
if dataset_config["type"] == "torchvision":
    dataset_cls = get_torchvision_dataset(dataset_config["name"])
    transform = build_transform(dataset_config)  # builds Normalize only if fields exist
    train_dataset = dataset_cls(root="data", train=True, download=dataset_config.get("download", True), transform=transform)
    test_dataset = dataset_cls(root="data", train=False, download=dataset_config.get("download", True), transform=transform)
else:
    # Handle custom dataset logic here
    X, y = load_custom_dataset(dataset_config)
    train_dataset = Data(X["train"], y["train"], device)
    test_dataset = Data(X["test"], y["test"], device)

# Create loaders
train_loader = get_dataloader(train_dataset, batch_size=train_config["batch_size"], shuffle=train_config["shuffle"])
test_loader = get_dataloader(test_dataset, batch_size=train_config["batch_size"])
\`\`\`

---

## 🔮 Future Work

- ⚙️ Prefect Pipeline Orchestration
- 📈 MLflow Experiment Tracking and Deployment

---

## 📦 Requirements

\`\`\`bash
pip install -r requirements.txt
\`\`\`

**Dependencies** include:
- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- scikit-learn

---

## 🤝 Contributing

Pull requests welcome! The modular layout is ideal for research prototyping and open-source collaboration.
