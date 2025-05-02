
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
- ✅ Seamless Prefect integration for orchestration, caching, and logging

---

## 📁 Project Structure

```bash
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
├── main.py
├── .prefect.yaml
```

---

## 🧪 Example Usage: CIFAR-10 CNN

To run this pipeline:

### ➤ Simple local run (with or without Prefect)
```bash
python main.py
```
This will use Prefect if `"prefect_enabled": true` in `train_config.json`. Otherwise, it runs as a regular Python script.

### ➤ Full Prefect deployment
```bash
prefect deploy
prefect run deployment training-flow/training-flow
```
This registers and runs your flow with Prefect for tracked execution, logging, caching, and scheduling.

---

## 🔧 `train_config.json` Enhancements

The `train_config.json` file now includes flags for fine-tuning and Prefect orchestration:

```json
{
  "batch_size": 64,
  "shuffle": true,
  "loss_function": "cross_entropy",
  "optimizer": {
    "type": "adam",
    "lr": 0.01,
    "weight_decay": 1e-5
  },
  "epochs": 5,
  "resume_from_checkpoint": false,
  "checkpoint_path": "trained_model.pth",
  "freeze_backbone": true,
  "replace_classifier": false,

  // Prefect-specific flags
  "prefect_enabled": true,
  "use_cache_on_tasks": true,
  "enable_logging": true,
  "log_output_path": "logs/training_run.log"
}
```

---

## 🔮 Future Work

- 📈 MLflow Experiment Tracking and Deployment
- 🔁 Dataset versioning and artifact tracking
- 🧠 Model registry and hyperparameter sweeps

---

## 📦 Requirements

```bash
pip install -r requirements.txt
```

**Dependencies** include:
- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- scikit-learn
- prefect

---

## 🤝 Contributing

Pull requests welcome! The modular layout is ideal for research prototyping, scalable production workflows, and open-source collaboration.
