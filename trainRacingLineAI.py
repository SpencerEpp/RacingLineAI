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


# === Dataset ===
def is_circular_track(df, output_cols, threshold=5.0):
    start = df[output_cols].iloc[0].values
    end = df[output_cols].iloc[-1].values
    dist = np.linalg.norm(start[[0, 2]] - end[[0, 2]])  # Use X and Z
    return dist <= threshold

def get_centered_sequence(X_scaled, center_idx, seq_len, circular):
    half = seq_len // 2
    n = len(X_scaled)

    if circular:
        indices = [(center_idx - half + i) % n for i in range(seq_len)]
        return X_scaled[indices]
    else:
        left = center_idx - half
        right = center_idx + half + 1
        if left < 0:
            delta = X_scaled[1] - X_scaled[0]
            pre = [X_scaled[0] - delta * (i + 1) for i in reversed(range(-left))]
            seq = np.vstack(pre + [*X_scaled[0:right]])
        elif right > n:
            delta = X_scaled[-1] - X_scaled[-2]
            post = [X_scaled[-1] + delta * (i + 1) for i in range(right - n)]
            seq = np.vstack([*X_scaled[left:n]] + post)
        else:
            seq = X_scaled[left:right]
        return seq

class RacingLineDataset(Dataset):
    def __init__(self, config):
        self.inputs, self.targets = [], []
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        all_X, all_Y = [], []
        train_files = sorted(glob(os.path.join(config["train_data_path"], "*.csv")))
        test_files = sorted(glob(os.path.join(config["test_data_path"], "*.csv")))

        # === First pass: Collect data for global fitting ===
        for file in train_files:
            df = pd.read_csv(file)
            X = df[config["input_cols"]].values
            Y = df[config["output_cols"]].values
            all_X.append(X)
            all_Y.append(Y)
        for file in test_files:
            df = pd.read_csv(file)
            X = df[config["input_cols"]].values
            Y = df[config["output_cols"]].values
            all_X.append(X)
            all_Y.append(Y)
        all_X = np.vstack(all_X)
        all_Y = np.vstack(all_Y)
        self.scaler_x.fit(all_X)
        self.scaler_y.fit(all_Y)

        # === Second pass: Normalize and extract sequences ===
        for file in train_files:
            df = pd.read_csv(file)
            is_circular = is_circular_track(df, config["output_cols"])
            X = self.scaler_x.transform(df[config["input_cols"]].values)
            Y = self.scaler_y.transform(df[config["output_cols"]].values)

            for i in range(len(X)):
                self.inputs.append(get_centered_sequence(X, i, config["seq_len"], is_circular))
                self.targets.append(Y[i])

        self.inputs = torch.tensor(np.array(self.inputs), dtype=torch.float32).to(config["device"])
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32).to(config["device"])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
    

# === Model ===
class RacingLineLSTMWithAttention(nn.Module):
    def __init__(self, config, scaler_x=None, scaler_y=None):
        super().__init__()
        self.bidirectional = config["bidirectional"]
        self.hidden_size = config["hidden_size"]
        self.num_directions = 2 if self.bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=config["input_size"],
            hidden_size=self.hidden_size,
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            batch_first=True,
            bidirectional=self.bidirectional
        )

        self.attn = nn.Linear(self.num_directions * self.hidden_size, 1)
        self.dropout = nn.Dropout(config["dropout"])
        self.fc = nn.Linear(self.num_directions * self.hidden_size, config["output_size"])
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_scores = self.attn(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        context = self.dropout(context)
        return self.fc(context)

    def get_attention_weights(self, x):
        lstm_out, _ = self.lstm(x)
        attn_scores = self.attn(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        return attn_weights.squeeze(-1)
    

# === Loss Function ===
def weighted_mse(preds, targets, w_xyz=(10, 1, 10)):
    loss = ((preds - targets) ** 2)
    x_loss = w_xyz[0] * loss[:, 0]
    y_loss = w_xyz[1] * loss[:, 1]
    z_loss = w_xyz[2] * loss[:, 2]
    return (x_loss + y_loss + z_loss).mean()


# === Evaluation Function ===
def evaluate_model(model, dataloader, criterion, config):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, Y_batch, in dataloader:
            X_batch, Y_batch = X_batch.to(config["device"]), Y_batch.to(config["device"])
            preds = model(X_batch)
            loss = criterion(preds, Y_batch)
            total_loss += loss.item()
    return total_loss / len(dataloader)


# === Save and Load Model ===
def save_model(model, config, scaler_x, scaler_y):
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
    }, config["model_save_path"])

def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    scaler_x = checkpoint["scaler_x"]
    scaler_y = checkpoint["scaler_y"]
    model = RacingLineLSTMWithAttention(cfg, scaler_x, scaler_y)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(cfg["device"])
    return model, cfg, scaler_x, scaler_y


# === Training Function ===
def train_model(model, train_loader, val_loader, config, scaler_x, scaler_y):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = weighted_mse
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
            X_batch, Y_batch = X_batch.to(config["device"]), Y_batch.to(config["device"])
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, Y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({"Loss": running_loss / (pbar.n + 1)})

        train_losses.append(running_loss / len(train_loader))
        val_loss = evaluate_model(model, val_loader, criterion, config)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, config, scaler_x, scaler_y)
            best_epoch = epoch+1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        obar.set_postfix({"Train Loss": running_loss/len(train_loader), "Val Loss": val_loss, "Lr": scheduler.get_last_lr()[0], "best_epoch": best_epoch})

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    return train_losses, val_losses


# === Full Pipeline ===
def run_pipeline(config, plot=False):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Preparing dataset...")
    full_dataset = RacingLineDataset(config)
    scaler_x = full_dataset.scaler_x
    scaler_y = full_dataset.scaler_y
    print("Total sequences loaded:", len(full_dataset))

    train_len = int(len(full_dataset) * config["train_split"])
    val_len = len(full_dataset) - train_len
    train_ds, val_ds = random_split(
        full_dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(config["seed"])
    )

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)

    print(f"Initializing model...")
    model = RacingLineLSTMWithAttention(config, scaler_x, scaler_y).to(config["device"]) 

    print("Training started...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, config, scaler_x, scaler_y)
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
        print_feature_accuracy(preds_real, trues_real, scaler_y, config["output_cols"])
        spatial_errors = np.linalg.norm(preds_real[:, [0, 2]] - trues_real[:, [0, 2]], axis=1)
        mean_spatial_error = np.mean(spatial_errors)
        max_spatial_error = np.max(spatial_errors)
        print(f"Mean X/Z spatial error: {mean_spatial_error:.2f}m, Max: {max_spatial_error:.2f}m\n")






# # Test code for sequencing data into sliding windows
# from sklearn.preprocessing import MinMaxScaler
# df = pd.read_csv("./data/testing_layouts/ks_barcelona_layout_gp_Processed_Data.csv")
# scaler_x = MinMaxScaler()
# scaler_y = MinMaxScaler()
# scaler_x.fit(df[config["input_cols"]])
# scaler_y.fit(df[config["output_cols"]])
# config["scaler_x"] = scaler_x
# config["scaler_y"] = scaler_y
# visualize_all_center_points(df, config)

# def visualize_all_center_points(df, config):
#     X = df[config["input_cols"]].values
#     Y = df[config["output_cols"]].values
#     X_scaled = config["scaler_x"].transform(X)
#     is_circular = is_circular_track(df, config["output_cols"])

#     seq_len = config["seq_len"]
#     centers_x, centers_z = [], []

#     for i in range(len(X_scaled)):
#         seq = get_centered_sequence(X_scaled, i, seq_len, is_circular)
#         left_x, left_z = seq[:, 0], seq[:, 2]
#         right_x, right_z = seq[:, 3], seq[:, 5]
#         center_x = (left_x + right_x) / 2
#         center_z = (left_z + right_z) / 2
#         mid = seq_len // 2
#         centers_x.append(center_x[mid])
#         centers_z.append(center_z[mid])

#     plt.figure(figsize=(10, 5))
#     plt.plot(centers_x, centers_z, label="Input Sequence Centers", linewidth=2)
#     plt.title("Sliding Window Center Points (Before Model)")
#     plt.xlabel("X")
#     plt.ylabel("Z")
#     plt.axis("equal")
#     plt.grid(True)
#     plt.legend()
#     plt.show()