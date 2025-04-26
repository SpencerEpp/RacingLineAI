#====================================================
# Project: Racing Line AI
# Authors: Spencer Epp, Samuel Trepac
# Date:    March 23rd - April 28th
#
# Description:
#     Training and evaluation pipeline for the RoboTurbosky CNN model.
#     Provides full model training, validation loss evaluation, 
#     input ablation testing, and result visualization.
#
# File Overview:
#     This file defines the training workflow for the CNN-based Racing Line AI 
#     project, including loss functions, model training with profiling, 
#     validation error reporting, input feature ablation, and learning curve plotting.
#
# Functions Included:
#     - PositionLoss: MSE loss for racing line position prediction.
#     - ControlLoss: MSE loss + penalties for realistic gas and brake outputs.
#     - evaluate_model(): Evaluate average model loss on a dataset.
#     - train_model(): Full training loop with early stopping and profiling.
#     - evaluate_on_val_set(): Calculate average validation error.
#     - run_input_ablation(): Measure feature importance via zeroed input testing.
#     - run_pipeline(): Complete end-to-end training, validation, testing workflow.
#====================================================


# === Imports ===
import gc
import torch as t
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import CNNDataManager
from CNNModel import RoboTurbosky, save_model, load_model
from Profiler import Profiler, profiled

profiler = Profiler()


# === Loss Functions ===
"""
    MSE loss for racing line position prediction.

    Args:
        predicted (torch.Tensor): Predicted positions (batch_size, 2).
        targets (dict): Ground truth dictionary containing 'ai_norm' key.

    Returns:
        torch.Tensor: Mean squared error loss value.
"""
class PositionLoss(t.nn.Module):
    def __init__(self):
        super(PositionLoss, self).__init__()
        self.mse_loss = t.nn.MSELoss()

    @profiled(profiler, "Position Loss")
    def forward(self, predicted, targets):
        return self.mse_loss(predicted, targets["ai_norm"])


"""
    Control output loss combining MSE and penalties for gas/brake values.

    Args:
        predicted (torch.Tensor): Predicted control outputs (batch_size, 3).
        targets (dict): Ground truth dictionary containing 'ai_cont' key.

    Returns:
        torch.Tensor: Combined control loss value.
"""
class ControlLoss(t.nn.Module):
    def __init__(self, weight=1.0):
        super(ControlLoss, self).__init__()
        self.mse_loss = t.nn.MSELoss()
    
    @profiled(profiler, "Control Loss")
    def forward(self, predicted, targets):
        predicted_speed = predicted[:, 0]
        predicted_gas = predicted[:, 1]
        predicted_brake = predicted[:, 2]
        
        loss = (
            self.mse_loss(predicted_speed, targets["ai_cont"][:, 0]) +
            self.mse_loss(predicted_gas, targets["ai_cont"][:, 1]) +
            self.mse_loss(predicted_brake, targets["ai_cont"][:, 2])
        )

        gas_penalty = t.clamp(predicted_gas - 1, min=0) + t.clamp(0 - predicted_gas, min=0)
        brake_penalty = t.clamp(predicted_brake - 1, min=0) + t.clamp(0 - predicted_brake, min=0)

        penalty = 10 * (gas_penalty.pow(2).mean() + brake_penalty.pow(2).mean())

        return loss + penalty


# === Evaluate Model ===
"""
    Evaluate model loss over a validation DataLoader.

    Args:
        model (torch.nn.Module): Model to evaluate.
        dataloader (DataLoader): Validation dataset loader.
        pos_crit (Loss): Position loss criterion.
        cont_crit (Loss): Control loss criterion.

    Returns:
        float: Average validation loss.
"""
@profiled(profiler, "Evaluate Model")
def evaluate_model(model, dataloader, pos_crit, cont_crit):
    model.eval()
    total_loss = 0
    with t.inference_mode():
        for inputs, targets in dataloader:
            inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}
            targets = {k: v.cuda(non_blocking=True) for k, v in targets.items()}
            control_pred, position_pred = model(inputs["patch"].unsqueeze(1), inputs["center"], inputs["metadata"])
            loss = cont_crit(control_pred, targets) + pos_crit(position_pred, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)


# === Train Model ===
"""
    Train the model with profiling, learning curve plotting, and early stopping.

    Args:
        model (torch.nn.Module): Model to train.
        train_loader (DataLoader): Training dataset.
        val_loader (DataLoader): Validation dataset.
        config (dict): Training configuration.

    Returns:
        tuple: Lists of train_losses and val_losses per epoch.
"""
@profiled(profiler, "Train Model")
def train_model(model, train_loader, val_loader, config):
    optimizer = t.optim.Adam(model.parameters(), lr=config["learning_rate"])
    pos_crit = PositionLoss()
    cont_crit = ControlLoss()

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    early_stopping_patience = config.get("patience", 10)
    epochs_without_improvement = 0

    obar = tqdm(range(config["epochs"]), desc="Epochs")
    for epoch in obar:
        model.train()
        running_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=False)
        for inputs, targets in pbar:

            profiler.start("to_device")
            inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}
            targets = {k: v.cuda(non_blocking=True) for k, v in targets.items()}
            profiler.stop("to_device")

            profiler.start("forward")
            control_pred, position_pred = model(inputs["patch"].unsqueeze(1), inputs["center"], inputs["metadata"])
            loss = cont_crit(control_pred, targets) + pos_crit(position_pred, targets)
            profiler.stop("forward")

            profiler.start("backward")
            optimizer.zero_grad()
            loss.backward()
            profiler.stop("backward")

            profiler.start("step")
            optimizer.step()
            profiler.stop("step")

            running_loss += loss.item()
            pbar.set_postfix({"Loss": running_loss / (pbar.n + 1)})

        del inputs, targets, control_pred, position_pred, loss
        t.cuda.empty_cache()
        gc.collect()

        train_losses.append(running_loss / len(train_loader))
        val_loss = evaluate_model(model, val_loader, pos_crit, cont_crit)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            profiler.start("saving model")
            best_val_loss = val_loss
            save_model(model, config)
            epochs_without_improvement = 0
            profiler.start("saving model")
        else:
            epochs_without_improvement += 1

        obar.set_postfix({"Train Loss": train_losses[-1], "Val Loss": val_loss})

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    return train_losses, val_losses


# === Evaluate on Validation Set ===
"""
    Evaluate trained model on validation dataset.

    Args:
        model (torch.nn.Module): Trained model.
        val_loader (DataLoader): Validation dataset.

    Returns:
        None
"""
@profiled(profiler, "Evaluate on Validation Set")
def evaluate_on_val_set(model, val_loader):
    val_loss = evaluate_model(model, val_loader, PositionLoss(), ControlLoss())
    print(f"Validation Loss: {val_loss:.4f}")


# === Run Input Ablation Test ===
"""
    Perform input ablation by zeroing out only *contextual features*,
    leaving center_x and center_z coordinates untouched.

    Args:
        model (torch.nn.Module): Trained CNN model.
        val_loader (DataLoader): Validation dataset loader.

    Returns:
        None
"""
@profiled(profiler, "Input Ablation")
def run_input_ablation(device, model, val_loader):
    model.eval()
    pos_crit = PositionLoss()
    cont_crit = ControlLoss()

    # === Baseline validation loss ===
    base_loss = evaluate_model(model, val_loader, pos_crit, cont_crit)
    print(f"\nBaseline Validation Loss: {base_loss:.4f}\n")

    feature_names = ["distance", "heading_x", "heading_z", "curvature", "track_widths",
                     "track_avg_width", "track_min_width", "track_max_width",
                     "track_total_length", "track_avg_curvature", "track_max_curvature"]

    val_loader_copy = t.utils.data.DataLoader(
        val_loader.dataset,
        batch_size=val_loader.batch_size,
        shuffle=val_loader.shuffle if hasattr(val_loader, 'shuffle') else False,
        num_workers=val_loader.num_workers,
        pin_memory=val_loader.pin_memory,
    )

    ablation_results = []
    for idx, feature_name in tqdm(enumerate(feature_names), total=len(feature_names), desc="Ablation"):
        running_loss = 0.0
        with t.inference_mode():
            for inputs, targets in val_loader_copy:
                inputs = {k: v.clone().to(device) for k, v in inputs.items()}
                targets = {k: v.clone().to(device) for k, v in targets.items()}              
                inputs["metadata"][idx] = 0.0
                control_pred, position_pred = model(inputs["patch"].unsqueeze(1), inputs["center"], inputs["metadata"])
                loss = cont_crit(control_pred, targets) + pos_crit(position_pred, targets)
                running_loss += loss.item()

        ablated_loss = running_loss / len(val_loader)
        delta_loss = ablated_loss - base_loss
        ablation_results.append((feature_name, ablated_loss, delta_loss))

    # === Print Results ===
    print("\n=== Feature Ablation Results ===")
    for feature, loss, delta in sorted(ablation_results, key=lambda x: -x[2]):
        print(f"{feature:25s}  Loss: {loss:.4f}  ΔLoss: {delta:.4f}")


# === Run Full Training Pipeline ===
"""
    Full CNN model training, validation, input ablation, and profiling pipeline.

    Args:
        config (dict): Training configuration.

    Returns:
        None
"""
def run_pipeline(config):
    print("Preparing dataset...")
    profiler.start("Gathering Files and Split")
    train_loader, val_loader = CNNDataManager.load_training_dataset(
        config["dataset_dir"], config["device"], config["split_ratio"], batch_size=config["batch_size"], seed=config["seed"]
    )
    profiler.stop("Gathering Files and Split")
    print("Total sequences loaded:", (len(train_loader)+len(val_loader)))

    print(f"Initializing model...")
    profiler.start("Init Model")
    model = RoboTurbosky(config).to(config["device"])
    profiler.stop("Init Model")

    print("Training started...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, config)
    print(f"Training complete. Model saved to {config['model_save_path']}")

    model, cfg = load_model(config["model_save_path"], device=config["device"])
    evaluate_on_val_set(model, val_loader)
    run_input_ablation(cfg["device"], model, val_loader)

    profiler.report()

    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Learning Curve")
    plt.grid(True)
    plt.legend()
    plt.show()