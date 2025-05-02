import torch
import torch.nn as nn
import json
from prefect import flow, get_run_logger

from config import (
    get_eval, get_model, get_dataloader,
    get_loss, get_optim, get_dataset_from_config, TRAIN_CONFIG
)
from model_factory.utils import validate_layer_specs, config_to_code
from train import train_model_one_epoch
from evaluate import plot_predictions_accuracy

# Load JSON configuration files
with open('configs/model/cnn_config.json', 'r') as f:
    model_config = json.load(f)

with open('configs/data/dataset_config.json', 'r') as f:
    dataset_config = json.load(f)

device = torch.device("cuda" if (model_config["device"] == "cuda" and torch.cuda.is_available()) else "cpu")

# --- Building Model ---
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

@flow(name="PyTorch Training Flow")
def training_flow():
    logger = get_run_logger()
    model = build_and_test(model_config)

    # Resume checkpoint if applicable
    if TRAIN_CONFIG.get("resume_from_checkpoint", False):
        checkpoint_path = TRAIN_CONFIG["checkpoint_path"]
        logger.info(f"Loading model from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        if TRAIN_CONFIG.get("freeze_backbone", False):
            for param in model.parameters():
                param.requires_grad = False
            last_layer = list(model.children())[-1]
            for param in last_layer.parameters():
                param.requires_grad = True
            logger.info("Backbone frozen. Only classifier head will be fine-tuned.")

    if model_config["model_type"] == "cnn" and TRAIN_CONFIG.get("replace_classifier", False):
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, TRAIN_CONFIG["num_classes"])

    # Dataset
    train_dataset = get_dataset_from_config(dataset_config, train=True)
    test_dataset = get_dataset_from_config(dataset_config, train=False)

    train_loader = get_dataloader(train_dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=TRAIN_CONFIG["shuffle"])
    test_loader = get_dataloader(test_dataset, batch_size=TRAIN_CONFIG["batch_size"])

    # Loss and optimizer
    criterion = get_loss(TRAIN_CONFIG["loss_function"], reduction='mean')
    optimizer = get_optim(
        TRAIN_CONFIG["optimizer"]["type"],
        model.parameters(),
        lr=TRAIN_CONFIG["optimizer"]["lr"],
        weight_decay=TRAIN_CONFIG["optimizer"].get("weight_decay", 0)
    )

    # Training loop
    train_accs, test_accs = [], []
    for epoch in range(TRAIN_CONFIG["epochs"]):
        train_acc, test_acc = train_model_one_epoch(model, train_loader, test_loader, criterion, optimizer, crit_mode='accuracy')
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        logger.info(f"Epoch {epoch} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")

    # Save and plot
    torch.save(model.state_dict(), "trained_model.pth")
    logger.info("Model saved to trained_model.pth")
    plot_predictions_accuracy(train_accs, test_accs)

if __name__ == "__main__":
    if TRAIN_CONFIG.get("prefect_enabled", False):
        training_flow()
    else:
        training_flow.without_tracking()
