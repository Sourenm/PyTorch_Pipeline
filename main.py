import torch
import torch.nn as nn
from config import get_model, get_dataloader, get_loss, get_optim
from model_factory.utils import validate_layer_specs, config_to_code
from train import train_model_one_epoch
from evaluate import evaluate_model, plot_predictions_accuracy
import torchvision
import torchvision.transforms as transforms

# --- MLP CONFIG ---
mlp_config = {
    "model_type": "mlp",
    "input_shape": (128,),
    "layer_specs": [
        {"type": "linear", "in_features": 128, "out_features": 64},
        {"type": "activation", "name": "relu"},
        {"type": "dropout", "dropout": 0.2},
        {"type": "linear", "in_features": 64, "out_features": 32},
        {"type": "activation", "name": "relu"},
        {"type": "linear", "in_features": 32, "out_features": 10}
    ],
    "device": "cuda"
}

# --- CNN CONFIG ---
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
    "device": "cuda"
}

# --- RNN CONFIG ---
rnn_config = {
    "model_type": "rnn",
    "input_shape": (10, 16),
    "layer_specs": [
        {"type": "rnn", "input_size": 16, "hidden_size": 32, "num_layers": 1, "batch_first": True},
        {"type": "linear", "in_features": 32, "out_features": 10}
    ],
    "device": "cuda"
}

# --- LSTM CONFIG ---
lstm_config = {
    "model_type": "lstm",
    "input_shape": (10, 16),
    "layer_specs": [
        {"type": "lstm", "input_size": 16, "hidden_size": 64, "num_layers": 1, "batch_first": True},
        {"type": "linear", "in_features": 64, "out_features": 10}
    ],
    "device": "cuda"
}

# --- TRANSFORMER CONFIG ---
transformer_config = {
    "model_type": "transformer",
    "input_shape": (10, 32),
    "layer_specs": [
        {"type": "transformer_encoder", "d_model": 32, "num_heads": 4, "num_layers": 2},
        {"type": "linear", "in_features": 32, "out_features": 10}
    ],
    "device": "cuda"
}

def build_and_test(config):
    try:
        device = torch.device("cuda" if (config["device"] == "cuda" and torch.cuda.is_available()) else "cpu") 
        validate_layer_specs(config["model_type"], config["layer_specs"])        
        model = get_model(config["model_type"], config["layer_specs"], config["device"])
        print("--- Code for reproducibility ---")
        print(config_to_code(config))
        print("\n")
        print("Validated layer spec and model successfully ✅")
        return model
    except Exception as e:
        raise ValueError(f"❌ Error building {config['model_type'].upper()}: {str(e)}\n")


# Transforms (normalizing to [-1, 1] for tanh/relu if needed)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# build the model and test the config structure
model = build_and_test(cnn_config)

# build train DataLoader and test DataLoader
train_loader = get_dataloader(torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
test_loader = get_dataloader(torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform), batch_size=64)

# build loss and optimizer functions
criterion = get_loss('cross_entropy', reduction='mean')
optimizer = get_optim("adam", model.parameters(), lr=0.01, weight_decay=1e-5)

# training loop
epochs = 5
train_accs = []
test_accs = []

for epoch in range(epochs):
    train_acc, test_acc = train_model_one_epoch(model, train_loader, test_loader, criterion, optimizer, crit_mode='accuracy')
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    print(f"Epoch {epoch} | Train Acc: {train_accs[-1]*100:.2f}% | Test Acc: {test_accs[-1]*100:.2f}%")

# saving the model
torch.save(model.state_dict(), "cnn_cifar10.pth")

# plotting the accuracy
plot_predictions_accuracy(train_accs, test_accs)