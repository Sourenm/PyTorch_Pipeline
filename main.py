import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import json
from config import get_model, get_dataloader, get_loss, get_optim, get_dataset_from_config
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
device = torch.device("cuda" if (model_config["device"] == "cuda" and torch.cuda.is_available()) else "cpu")
def build_and_test(config):
    try:        
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

# fine-tuning a previous model if indicated in JSON
if train_config.get("resume_from_checkpoint", False):
    checkpoint_path = train_config["checkpoint_path"]
    print(f"Loading model from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    if train_config.get("freeze_backbone", False):        
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the last layer (child module)
        last_layer = list(model.children())[-1]
        for param in last_layer.parameters():
            param.requires_grad = True
        print("Backbone frozen. Only classifier head will be fine-tuned.")

# changing the number of final classes if indicated in JSON
if model_config["model_type"] == "cnn" and train_config.get("replace_classifier", False):
    num_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_features, train_config["num_classes"])

# --- Preparing Dataset ---
train_dataset = get_dataset_from_config(dataset_config, train=True)
test_dataset = get_dataset_from_config(dataset_config, train=False)

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
