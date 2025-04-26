#====================================================
# Project: Racing Line AI
# Authors: Spencer Epp, Samuel Trepac
# Date:    March 23rd - April 28th
#
# Description:
#     Training and evaluation pipeline for the Racing Line AI model.
#     Provides model training, validation, profiling, and full inference
#     on unseen track layouts.
#
# File Overview:
#     This file defines the training and evaluation workflow for the 
#     Racing Line AI project, including loss functions, model training 
#     with profiling, validation error evaluation, input ablation 
#     testing, and inference utilities (reqs game file data).
#
# Functions Included:
#     - weighted_mse(): Weighted mean squared error loss function.
#     - euclidean_loss(): Euclidean distance loss for spatial accuracy.
#     - evaluate_model(): Evaluate model loss over a dataset.
#     - train_model(): Full training loop with early stopping and profiling.
#     - evaluate_on_val_set(): Per-layout validation error calculation.
#     - run_input_ablation(): Evaluate feature importance via input ablation.
#     - run_pipeline(): Full training + validation + testing pipeline.
#     - print_feature_accuracy(): Per-feature accuracy reporting.
#     - run_inference(): Predict racing lines on new layouts and visualize.
#====================================================


# === Imports ===
import os
import gc
import numpy as np
import pandas as pd
from glob import glob
from tqdm.notebook import tqdm
import random
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch.serialization
torch.serialization.add_safe_globals([MinMaxScaler])
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
from Profiler import Profiler, profiled
from LSTMModel import RacingLineLSTMWithAttention, save_model, load_model 
from LSTMDataClass import RacingLineDataset, add_contextual_features, is_circular_track, get_centered_sequence

profiler = Profiler()


# === Weighted Mean Squared Error Loss ===
"""
    Calculate weighted mean squared error (MSE) between predictions and targets.

    Args:
        preds (torch.Tensor): Predicted output tensor.
        targets (torch.Tensor): True output tensor.
        w_xyz (tuple): Weights for X, Y, Z dimensions.

    Returns:
"""
@profiled(profiler, "MSE Loss")
def weighted_mse(preds, targets, w_xyz=(10, 1, 10)):
    loss = ((preds - targets) ** 2)
    x_loss = w_xyz[0] * loss[:, 0]
    y_loss = w_xyz[1] * loss[:, 1]
    z_loss = w_xyz[2] * loss[:, 2]
    return (x_loss + y_loss + z_loss).mean()


# === Euclidean Loss ===
"""
    Calculate the mean Euclidean distance between predictions and targets.

    Args:
        preds (torch.Tensor): Predicted output tensor.
        targets (torch.Tensor): True output tensor.

    Returns:
        torch.Tensor: Mean Euclidean loss value.
"""
@profiled(profiler, "Euclidean Loss")
def euclidean_loss(preds, targets):
    return torch.norm(preds - targets, dim=1).mean()


# === Evaluate Model ===
"""
    Evaluate a model over a given DataLoader and return the average loss.

    Args:
        model (torch.nn.Module): Model to evaluate.
        dataloader (DataLoader): DataLoader for evaluation dataset.
        criterion (function): Loss function to use.
        config (dict): Training configuration.

    Returns:
        float: Average loss over the dataset.
"""
@profiled(profiler, "Evaluate Model")
def evaluate_model(model, dataloader, criterion, config):
    model.eval()
    total_loss = 0
    with torch.inference_mode():
        for X_batch, Y_batch, in dataloader:
            X_batch, Y_batch = X_batch.to(config["device"], non_blocking=True), Y_batch.to(config["device"], non_blocking=True)
            preds = model(X_batch).detach()
            loss = criterion(preds, Y_batch)
            total_loss += loss.item()
    return total_loss / len(dataloader)


# === Train Model ===
"""
    Train the model with profiling, learning rate scheduling, and early stopping.

    Args:
        model (torch.nn.Module): Model to train.
        train_loader (DataLoader): Training set loader.
        val_loader (DataLoader): Validation set loader.
        config (dict): Training configuration dictionary.
        scaler_x (MinMaxScaler, optional): Input scaler.
        scaler_y (MinMaxScaler, optional): Output scaler.

    Returns:
        tuple: (train_losses, val_losses) lists of loss values per epoch.
"""
@profiled(profiler, "Train Model")
def train_model(model, train_loader, val_loader, config, scaler_x=None, scaler_y=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
    criterion = euclidean_loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config["scheduler_patience"])
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    early_stopping_patience = config["patience"]
    epochs_without_improvement = 0
    best_epoch = 0

    obar = tqdm(range(config["num_epochs"]), desc="Epochs")
    for epoch in obar:
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}", leave=False)
        for X_batch, Y_batch, in pbar:

            profiler.start("to_device")
            X_batch, Y_batch = X_batch.to(config["device"], non_blocking=True), Y_batch.to(config["device"], non_blocking=True)
            profiler.stop("to_device")

            profiler.start("forward")
            preds = model(X_batch)
            loss = criterion(preds, Y_batch)
            profiler.stop("forward")

            profiler.start("backward")
            optimizer.zero_grad()
            loss.backward()
            profiler.stop("backward")

            profiler.start("step")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            profiler.stop("step")

            running_loss += loss.item()
            pbar.set_postfix({"Loss": running_loss / (pbar.n + 1)})

        del X_batch, Y_batch, preds, loss
        torch.cuda.empty_cache()
        gc.collect()

        train_losses.append(running_loss / len(train_loader))
        val_loss = evaluate_model(model, val_loader, criterion, config)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            profiler.start("saving model")
            best_val_loss = val_loss
            save_model(model, config, scaler_x, scaler_y, profiler=profiler)
            best_epoch = epoch+1
            epochs_without_improvement = 0
            profiler.stop("saving model")
        else:
            epochs_without_improvement += 1

        obar.set_postfix({"Train Loss": running_loss/len(train_loader), "Val Loss": val_loss, "Lr": scheduler.get_last_lr()[0], "best_epoch": best_epoch})

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    return train_losses, val_losses


# === AMP Enabled Training Function (Probably broken doesnt learn well) ===
# """
#     Train the model using Automatic Mixed Precision (AMP) for faster training on GPUs.
#     Incorporates profiling, gradient scaling, early stopping, and learning rate scheduling.
#     Note: May not converge as well as standard training.

#     Args:
#         model (torch.nn.Module): Model to train.
#         train_loader (DataLoader): Training set loader.
#         val_loader (DataLoader): Validation set loader.
#         config (dict): Training configuration dictionary.
#         scaler_x (MinMaxScaler, optional): Input feature scaler.
#         scaler_y (MinMaxScaler, optional): Output target scaler.

#     Returns:
#         tuple: (train_losses, val_losses) lists of loss values per epoch.
# """
# @profiled(profiler, "Train Model")
# def train_model(model, train_loader, val_loader, config, scaler_x=None, scaler_y=None):
#     optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
#     criterion = euclidean_loss
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config["scheduler_patience"])
    
#     scaler = GradScaler()
#     best_val_loss = float("inf")
#     train_losses = []
#     val_losses = []
#     early_stopping_patience = config["patience"]
#     epochs_without_improvement = 0
#     best_epoch = 0

#     obar = tqdm(range(config["num_epochs"]), desc="Epochs")
#     for epoch in obar:
#         model.train()
#         running_loss = 0.0
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}", leave=False)

#         for X_batch, Y_batch in pbar:
#             profiler.start("to_device")
#             X_batch, Y_batch = X_batch.to(config["device"], non_blocking=True), Y_batch.to(config["device"], non_blocking=True)
#             profiler.stop("to_device")

#             optimizer.zero_grad()

#             profiler.start("forward")
#             with autocast(device_type="cuda"):
#                 preds = model(X_batch)
#                 loss = criterion(preds, Y_batch)
#             profiler.stop("forward")

#             profiler.start("backward")
#             scaler.scale(loss).backward()
#             profiler.stop("backward")

#             profiler.start("step")
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             scaler.step(optimizer)
#             scaler.update()
#             profiler.stop("step")

#             running_loss += loss.item()
#             pbar.set_postfix({"Loss": running_loss / (pbar.n + 1)})

#         train_losses.append(running_loss / len(train_loader))
#         val_loss = evaluate_model(model, val_loader, criterion, config)
#         val_losses.append(val_loss)
#         scheduler.step(val_loss)

#         if val_loss < best_val_loss:
#             profiler.start("saving model")
#             best_val_loss = val_loss
#             save_model(model, config, scaler_x, scaler_y, profiler=profiler)
#             best_epoch = epoch + 1
#             epochs_without_improvement = 0
#             profiler.stop("saving model")
#         else:
#             epochs_without_improvement += 1

#         obar.set_postfix({
#             "Train Loss": running_loss/len(train_loader),
#             "Val Loss": val_loss,
#             "Lr": scheduler.get_last_lr()[0],
#             "best_epoch": best_epoch
#         })

#         if epochs_without_improvement >= early_stopping_patience:
#             print(f"Early stopping triggered after {epoch+1} epochs.")
#             break

#     return train_losses, val_losses


# === Evaluate on Validation Set ===
"""
    Evaluate the trained model on each validation layout individually.

    Args:
        model (torch.nn.Module): Trained model.
        val_files (list): List of validation layout file paths.
        config (dict): Training configuration.
        scaler_x (MinMaxScaler): Input scaler.
        scaler_y (MinMaxScaler): Output scaler.
        profiler (Profiler, optional): Profiler object for timing.

    Returns:
        None
"""
def evaluate_on_val_set(model, val_files, config, scaler_x, scaler_y, profiler=None):
    if profiler: profiler.start("Evaluate On Val Set")

    model.eval()
    layout_errors = []

    for layout_path in val_files:
        layout_name = os.path.basename(layout_path)
        df = pd.read_csv(layout_path)
        df = add_contextual_features(df, profiler=profiler)
        is_circular = is_circular_track(df, config["output_cols"], profiler=profiler)
        X = scaler_x.transform(df[config["input_cols"]].values)
        Y = scaler_y.transform(df[config["output_cols"]].values)
        preds = []

        for i in range(len(X)):
            seq = get_centered_sequence(X, i, config["seq_len"], is_circular, profiler=profiler)
            X_tensor = torch.tensor(seq.reshape(1, config["seq_len"], -1), dtype=torch.float32).to(config["device"])
            with torch.inference_mode():
                pred = model(X_tensor).detach().cpu().squeeze().numpy()
            preds.append(pred)

        preds = np.array(preds)
        errors = np.linalg.norm(preds - Y, axis=1)
        layout_errors.append((layout_name, errors.mean(), errors.max()))

    print("\n=== Per-Layout Validation Error ===")
    for name, mean_err, max_err in sorted(layout_errors, key=lambda x: x[1], reverse=True):
        print(f"{name:40s}  Mean: {mean_err:.4f}  Max: {max_err:.4f}")

    if profiler: profiler.stop("Evaluate On Val Set")


# === Run Input Ablation Test ===
"""
    Perform input ablation to measure feature importance by zeroing inputs.

    Args:
        model (torch.nn.Module): Trained model.
        val_dataset (Dataset): Validation dataset.
        config (dict): Training configuration.

    Returns:
        None
"""
@profiled(profiler, "Input Ablation")
def run_input_ablation(model, val_dataset, config):
    model.eval()
    base_loss = evaluate_model(model, DataLoader(val_dataset, batch_size=config["batch_size"]), euclidean_loss, config)
    print(f"\nBaseline Val Loss: {base_loss:.4f}\n")

    input_cols = config["input_cols"]
    ablation_results = []

    for i, col in enumerate(input_cols):
        all_inputs, all_targets = [], []
        for j in range(len(val_dataset)):
            x, y = val_dataset[j]
            all_inputs.append(x.unsqueeze(0))
            all_targets.append(y.unsqueeze(0))
        all_inputs = torch.cat(all_inputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        ablated_inputs = all_inputs.clone()
        ablated_inputs[:, :, i] = 0.0  # zero out one feature

        ablated_loader = DataLoader(torch.utils.data.TensorDataset(ablated_inputs, all_targets), batch_size=config["batch_size"])
        loss = evaluate_model(model, ablated_loader, euclidean_loss, config)
        impact = loss - base_loss
        ablation_results.append((col, loss, impact))

    print("=== Input Ablation Results ===")
    for col, loss, delta in sorted(ablation_results, key=lambda x: -x[2]):
        print(f"{col:20s}  Loss: {loss:.4f}  ΔLoss: {delta:.4f}")


# === Run Full Training Pipeline ===
"""
    Full end-to-end training and evaluation pipeline, including
    dataset loading, model training, validation, and profiling report.

    Args:
        config (dict): Training configuration dictionary.
                       See ___ for example config.

    Returns:
        None
"""
@profiled(profiler, "Run Inference")
def run_pipeline(config):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("Preparing dataset...")
    profiler.start("Gathering Files and Split")
    all_files = sorted(glob(os.path.join(config["train_data_path"], "*.csv")))
    train_files, val_files = train_test_split(
        all_files,
        test_size=(1 - config["train_split"]),
        random_state=config["seed"]
    )
    profiler.stop("Gathering Files and Split")

    train_dataset = RacingLineDataset(config, file_list=train_files, profiler=profiler)
    val_dataset = RacingLineDataset(config, file_list=val_files, profiler=profiler)
    scaler_x = train_dataset.scaler_x
    scaler_y = train_dataset.scaler_y
    print("Total sequences loaded:", (len(train_dataset)+len(val_dataset)))

    profiler.start("Create DataLoaders")
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True)
    torch.cuda.empty_cache()
    profiler.stop("Create DataLoaders")

    print(f"Initializing model...")
    profiler.start("Init Model")
    model = RacingLineLSTMWithAttention(config, scaler_x, scaler_y).to(config["device"]) 
    profiler.stop("Init Model")

    print("Training started...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, config, scaler_x, scaler_y)
    print(f"Training complete. Model saved to {config['model_save_path']}")

    torch.cuda.empty_cache()
    model, cfg, scaler_x, scaler_y = load_model(config["model_save_path"], config["device"], profiler=profiler)
    evaluate_on_val_set(model, val_files, cfg, scaler_x, scaler_y, profiler=profiler)
    run_input_ablation(model, val_dataset, cfg)

    profiler.report()

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
        df = add_contextual_features(df)
        is_circular = is_circular_track(df, config["output_cols"])
        X = df[config["input_cols"]].values
        Y = df[config["output_cols"]].values
        X_scaled = scaler_x.transform(X)
        n = len(X_scaled)
        preds_real = np.zeros_like(Y)
        trues_real = Y.copy()

        for i in tqdm(range(n), desc=f"[{layout_index + 1}/{total_layouts}]"):
            seq = get_centered_sequence(X_scaled, i, config["seq_len"], is_circular)
            X_tensor = torch.tensor(seq.reshape(1, config["seq_len"], -1), dtype=torch.float32).to(config["device"])

            with torch.inference_mode():
                pred_scaled = model(X_tensor).detach().cpu().squeeze().numpy()
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
        print_feature_accuracy(preds_real, trues_real, scaler_y, config["output_cols"])
        spatial_errors = np.linalg.norm(preds_real[:, [0, 2]] - trues_real[:, [0, 2]], axis=1)
        mean_spatial_error = np.mean(spatial_errors)
        max_spatial_error = np.max(spatial_errors)
        print(f"Mean X/Z spatial error: {mean_spatial_error:.2f}m, Max: {max_spatial_error:.2f}m\n")