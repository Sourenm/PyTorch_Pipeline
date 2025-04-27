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
# TODO: put a flag to do this or skip it
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(dataset_config["normalize_mean"], dataset_config["normalize_std"])
])

# TODO: accomodate more/any datasets
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
