#====================================================
# Project: Racing Line AI
# Authors: Spencer Epp, Samuel Trepac
# Date:    March 23rd - April 28th
#
# Description:
#     A complete training pipeline for the Cerberus model, designed to predict 
#     racing line center points and control features from track layouts. 
#     The pipeline supports multi-stage training with different sequence lengths 
#     and implements weighted hybrid losses for improved model realism.
#
# File Overview:
#     - Define position and control loss functions.
#     - Train Cerberus model progressively on long, medium, and short sequences.
#     - Save best-performing model and scalers.
#     - Evaluate performance and visualize learning curves.
#
# Functions Included:
#     - weighted_mse(): Compute weighted mean squared error.
#     - PositionLoss: Custom position loss class.
#     - ControlLoss: Custom control feature loss class.
#     - evaluate_model(): Evaluate model performance on validation sets.
#     - train_model(): Full training loop with dynamic dataset switching.
#     - run_pipeline(): Load data, initialize model, and run full training workflow.
#====================================================


# === Imports ===
import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm.notebook import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
import torch.serialization
torch.serialization.add_safe_globals([MinMaxScaler])
import matplotlib.pyplot as plt
from CerbModel import Cerberus, save_model, load_model 
from CerbDataClass import RacingLineDataset
from LSTMDataClass import add_contextual_features, is_circular_track, get_centered_sequence


# === Loss Functions ===
"""
    Compute weighted mean squared error between predictions and targets.

    Args:
        preds (torch.Tensor): Predicted outputs.
        targets (torch.Tensor): Ground truth targets.
        w_xyz (tuple): Weights for (X, Y, Z) coordinates.

    Returns:
        torch.Tensor: Weighted mean squared error.
"""
def weighted_mse(preds, targets, w_xyz=(10, 1, 10)):
    loss = ((preds - targets) ** 2)
    x_loss = w_xyz[0] * loss[:, 0]
    y_loss = w_xyz[1] * loss[:, 1]
    z_loss = w_xyz[2] * loss[:, 2]
    return (x_loss + y_loss + z_loss).mean()


"""
    Custom position loss module based on weighted mean squared error.

    Args:
        weight (float): Weighting factor for the loss (unused currently).

    Forward Args:
        predicted (torch.Tensor): Predicted positions.
        target (torch.Tensor): Ground truth positions.

    Returns:
        torch.Tensor: Position loss value.
"""
class PositionLoss(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(PositionLoss, self).__init__()
        self.mse_loss = weighted_mse

    def forward(self, predicted, target):
        loss = self.mse_loss(predicted, target)
        return loss


"""
    Custom control loss module using standard mean squared error.

    Args:
        weight (float): Weighting factor for the loss (currently placeholder).

    Forward Args:
        predicted (torch.Tensor): Predicted control outputs.
        target (torch.Tensor): Ground truth control outputs.

    Returns:
        torch.Tensor: Control loss value.
"""
class ControlLoss(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(ControlLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss()
        
    def forward(self, predicted, target):
        loss = self.mse_loss(predicted, target)
        return loss


# === Evaluation Function ===
"""
    Evaluate model performance on a validation set.

    Args:
        model (nn.Module): Cerberus model instance.
        dataloader (DataLoader): Validation dataloader.
        pos_crit (nn.Module): Position loss function.
        con_crit (nn.Module): Control loss function.
        config (dict): Model and training configuration.

    Returns:
"""
def evaluate_model(model, dataloader, pos_crit, con_crit, config):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, Y_batch, Z_batch in dataloader:
            X_batch, Y_batch, Z_batch = X_batch.to(config["device"]), Y_batch.to(config["device"]), Z_batch.to(config["device"])
            pos, cont = model(X_batch)
            pos_loss = pos_crit(pos, Y_batch)
            con_loss = con_crit(cont, Z_batch)
            
            loss = pos_loss + 0.5 * con_loss
            total_loss += loss.item()
    return total_loss / len(dataloader)


# === Train Cerberus Model ===
"""
    Train Cerberus model with progressive dataset transitions.

    Args:
        model (nn.Module): Cerberus model instance.
        long_dataset (tuple): (train_loader, val_loader) for long sequence data.
        medium_dataset (tuple): (train_loader, val_loader) for medium sequence data.
        short_dataset (tuple): (train_loader, val_loader) for short sequence data.
        config (dict): Training configuration dictionary.
        scalers (tuple): Tuple containing input, position, and control scalers.

    Returns:
        tuple: Lists of training losses and validation losses across epochs.
"""
def train_model(model, long_dataset, medium_dataset, short_dataset, config, scalers):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    pos_crit = PositionLoss()
    con_crit = ControlLoss(weight=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    early_stopping_patience = config["patience"]
    epochs_without_improvement = 0
    best_epoch = 0

    obar = tqdm(range(config["num_epochs"]), desc="Epochs")
    for epoch in obar:
        match epoch:
            case 0:
                train_loader = long_dataset[0]
                val_loader = long_dataset[1]
                
            case val if val == config["long_epochs"]:
                train_loader = medium_dataset[0]
                val_loader = medium_dataset[1]
            case val if val == (config["medium_epochs"] + config["long_epochs"]):
                train_loader = short_dataset[0]
                val_loader = short_dataset[1]
            case _:
                pass
                
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}", leave=False)
        for X_batch, Y_batch, Z_batch in pbar:
            X_batch, Y_batch, Z_batch = X_batch.to(config["device"]), Y_batch.to(config["device"]), Z_batch.to(config["device"])

            
            optimizer.zero_grad()
            p_pos, p_cont = model(X_batch)
            pos_loss, con_loss = pos_crit(p_pos, Y_batch), con_crit(p_cont, Z_batch)
            loss = pos_loss + 0.5 * con_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({"Loss": running_loss / (pbar.n + 1)})

        train_losses.append(running_loss / len(train_loader))
        
        val_loss = evaluate_model(model, val_loader, pos_crit, con_crit, config)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, config, scalers)
            best_epoch = epoch+1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        obar.set_postfix({"Train Loss": running_loss/len(train_loader), "Val Loss": val_loss, "Lr": scheduler.get_last_lr()[0], "best_epoch": best_epoch})

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    return train_losses, val_losses


# === Run Full Pipeline ===
"""
    Execute full Cerberus model training pipeline.

    Args:
        config (dict): Training configuration parameters.
        plot (bool): Whether to plot learning curves after training.

    Returns:
        None
"""
def run_pipeline(config, plot=False):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Preparing dataset...")
    long_dataset = RacingLineDataset(
        config,
        config["long_seq_len"],
    )
    medium_dataset = RacingLineDataset(
        config,
        config["med_seq_len"],
    )
    short_dataset = RacingLineDataset(
        config,
        config["short_seq_len"],
    )

    datasets = [long_dataset, medium_dataset, short_dataset]
    scaler_x = long_dataset.scaler_x
    scaler_y = long_dataset.scaler_y
    scaler_z = long_dataset.scaler_z

    scalers = (scaler_x, scaler_y, scaler_z)
    print("Total sequences loaded:", len(long_dataset))
    
    loaders = []
    
    for dataset in datasets:
        train_len = int(len(dataset) * config["train_split"])
        val_len = len(dataset) - train_len
        train_ds, val_ds = random_split(
            dataset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(config["seed"])
        )
        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
        loaders.append([train_loader, val_loader])

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)

    print(f"Initializing model...")
    model = Cerberus(config=config).to(config["device"])

    print("Training started...")
    train_losses, val_losses = train_model(model, loaders[0], loaders[1], loaders[2], config, scalers)
    print(f"Training complete. Model saved to {config['model_save_path']}")

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Learning Curve")
        plt.legend()
        plt.grid(True)
        plt.show()




# === TESTING CODE (make proper inference file) ===
# === Inference on Testing Track Layouts (from coordinates only, doesnt take image for inference) ===
def print_feature_accuracy(preds, trues, scaler_y, feature_names):
    preds = np.array(preds)
    trues = np.array(trues)

    print(f"\nPer-Feature Accuracy (%):")
    print("-" * 60)
    for i, name in enumerate(feature_names):
        range_train = scaler_y.scale_[i]
        range_test = trues[:, i].max() - trues[:, i].min()

        if range_test == 0:
            print(f"{name:>16}: N/A (zero test range)")
            continue

        mean_error = np.mean(np.abs(preds[:, i] - trues[:, i]))
        acc_train = (1 - (mean_error / range_train)) * 100
        acc_test = (1 - (mean_error / range_test)) * 100

        acc_train = max(0.0, min(100.0, acc_train))
        acc_test = max(0.0, min(100.0, acc_test))

        print(f"{name:>16}: {acc_test:6.2f}% (layout-based)   {acc_train:6.2f}% (train-scale)")

# === Inference On Testing Tracks ===
def run_inference(path):
    print("Loading model and scalers...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config, scaler_x, scaler_y = load_model(path, device)
    model.eval()

    print("Loading unseen layouts from:", config["test_data_path"])
    layout_files = sorted(glob(os.path.join(config["test_data_path"], "*.csv")))
    total_layouts = len(layout_files)
    print(f"Found {total_layouts} layout files.\n")

    for layout_index, layout_path in enumerate(layout_files):
        layout_name = os.path.basename(layout_path)
        print(f"[{layout_index + 1}/{total_layouts}] Predicting layout: {layout_name}")

        df = pd.read_csv(layout_path)
        is_circular = is_circular_track(df, config["output_pos_cols"])
        X = df[config["input_cols"]].values
        Y = df[config["output_pos_cols"]].values
        X_scaled = scaler_x.transform(X)
        n = len(X_scaled)
        preds_real = np.zeros_like(Y)
        trues_real = Y.copy()

        for i in tqdm(range(n), desc=f"[{layout_index + 1}/{total_layouts}]"):
            seq = get_centered_sequence(X_scaled, i, config["seq_len"], is_circular)
            X_tensor = torch.tensor(seq.reshape(1, config["seq_len"], -1), dtype=torch.float32).to(config["device"])

            with torch.no_grad():
                pred_scaled = model(X_tensor).cpu().squeeze().numpy()
                pred_real = scaler_y.inverse_transform(pred_scaled.reshape(1, -1))[0]
            preds_real[i] = pred_real

        # === Plot X/Z comparison ===
        plt.figure(figsize=(12, 6))
        plt.plot(trues_real[:, 0], trues_real[:, 2], label="True", linewidth=2)
        plt.plot(preds_real[:, 0], preds_real[:, 2], label="Predicted", linewidth=2, linestyle="--")
        plt.title(f"X vs Z: {layout_name}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Z Coordinate")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.show()

        # === Accuracy & Spatial Error (X/Z only) ===
        print_feature_accuracy(preds_real, trues_real, scaler_y, config["output_pos_cols"])
        spatial_errors = np.linalg.norm(preds_real[:, [0, 2]] - trues_real[:, [0, 2]], axis=1)
        mean_spatial_error = np.mean(spatial_errors)
        max_spatial_error = np.max(spatial_errors)
        print(f"Mean X/Z spatial error: {mean_spatial_error:.2f}m, Max: {max_spatial_error:.2f}m\n")